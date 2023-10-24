import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset
import open_clip

import os
import random
from PIL import Image
import numpy as np

from .image_features import ImageFeaturizer

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_vgg_features(image, model_name='vgg', layer_index=10):
    # Load pre-trained model
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True).features
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True).features
    else:
        raise ValueError('Invalid model name. Choose "vgg" or "alexnet".')

    # Set model to evaluation mode
    model.eval()

    # Extract features up to the specified layer
    model = torch.nn.Sequential(*list(model.children())[:layer_index+1])

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)

    # Move image to GPU if available
    image = image.to(_DEVICE)
    model = model.to(_DEVICE)

    # Extract features
    with torch.no_grad():
        features = model(image)

    return features


class CLIP_Model:
    def __init__(self, clip_model_name, clip_model_path = None, use_pickscore_encoder = False, device = None):
        self.use_pickscore_encoder = use_pickscore_encoder
        if device is None:
            self.device = _DEVICE
        else:
            self.device = device
        self.precision = 'fp16' if self.device == 'cuda' else 'fp32'
        self.clip_model_name = clip_model_name
        self.clip_model_architecture, self.clip_model_pretrained_dataset_name = self.clip_model_name.split('/', 2)

        print(f"Loading CLIP model {self.clip_model_name}...")
        #self.tokenize = open_clip.get_tokenizer(self.clip_model_architecture)

        if use_pickscore_encoder:
            print("Using PickScore encoder instead of vanilla CLIP!")
            # see https://github.com/yuvalkirstain/PickScore
            from transformers import AutoProcessor, AutoModel
            self.clip_preprocess = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.clip_model      = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(self.device)
            self.img_resolution  = 224
            self.clip_model_architecture = "PickScore_v1"
            
        else:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_model_architecture, 
                pretrained=self.clip_model_pretrained_dataset_name, 
                precision =self.precision,
                device=self.device,
                jit=False,
                cache_dir=clip_model_path
            )
            self.clip_model = self.clip_model.to(self.device).eval()
            self.img_resolution = 336 if '336' in self.clip_model_architecture else 224
            
            # slightly hacky way to get the mean and std of the normalization transform of the loaded clip model:
            self.target_mean, self.target_std = None, None
            for transform in self.clip_preprocess.transforms:
                if isinstance(transform, transforms.Normalize):
                    self.target_mean = transform.mean
                    self.target_std = transform.std
                    break
            
            self.clip_tensor_preprocess = transforms.Compose([
                transforms.Resize(self.img_resolution, antialias=True),
                transforms.CenterCrop(self.img_resolution),
                transforms.Normalize(
                    mean=self.target_mean,
                    std=self.target_std,
                ),
            ])
            
        print(f"CLIP model {self.clip_model_name} with img_resolution {self.img_resolution} loaded!")

    @torch.no_grad()
    def pt_imgs_to_features(self, list_of_tensors: list) -> torch.Tensor:
        if self.use_pickscore_encoder:
            list_of_pil_imgs = [transforms.ToPILImage()(img) for img in list_of_tensors]
            preprocessed_images = self.clip_preprocess(
                images=list_of_pil_imgs,
                return_tensors="pt",
            ).to(_DEVICE)
            image_features = self.clip_model.get_image_features(**preprocessed_images)
        else:
            preprocessed_images = [self.clip_tensor_preprocess(img) for img in list_of_tensors]        
            preprocessed_images = torch.stack(preprocessed_images).to(self.device)

            if self.precision == 'fp16':
                preprocessed_images = preprocessed_images.half()

            image_features = self.clip_model.encode_image(preprocessed_images)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features
    
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, img_resolution, crop_names, device):
        self.image_paths = image_paths
        self.crop_names  = crop_names
        self.device      = device
        self.img_resolution = img_resolution
        self.img_featurizer = ImageFeaturizer()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            pil_img = Image.open(img_path).convert('RGB')
            crops, crop_names = self.extract_crops(pil_img)
            image_features = self.img_featurizer.process(np.array(pil_img))
            return crops, crop_names, img_path, image_features
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.image_paths)-1))
    
    def extract_crops(self, pil_img: Image,
                  sample_fraction_save_to_disk=0.0  # Save a fraction of crops to disk (can be used for debugging)
                  ) -> list:
        """
        Instead of just embedding the entire image once, we extract multiple crops from the image and embed each crop separately.
        This provides more information to the classifier, and eg allows us to detect blurry images based on a small patch
        """
        img = transforms.ToTensor()(pil_img).to(self.device).unsqueeze(0)
        c, h, w = img.shape[1:]
        crops, crop_names = [], []        
        
        if 'centre_crop' in self.crop_names:
            crop_size = min(w, h)
            centre_crop = img[:, :, h // 2 - crop_size // 2:h // 2 + crop_size // 2, w // 2 - crop_size // 2:w // 2 + crop_size // 2]
            centre_crop = F.interpolate(centre_crop, size=self.img_resolution, mode='bilinear', align_corners=False)
            crops.append(centre_crop)
            crop_names.append("centre_crop")
        
        if 'square_padded_crop' in self.crop_names:
            crop_size = max(w, h)
            # Create square padded image:
            square_padded_img = torch.zeros((1, c, crop_size, crop_size), device=self.device)
            # paste the image in the centre of the square padded image:
            start_h = (crop_size - h) // 2
            start_w = (crop_size - w) // 2
            square_padded_img[:, :, start_h:start_h + h, start_w:start_w + w] = img
            # resize to self.img_resolution:
            square_padded_img = F.interpolate(square_padded_img, size=self.img_resolution, mode='bilinear', align_corners=False)
            crops.append(square_padded_img)
            crop_names.append("square_padded_crop")

        if ('subcrop1' in self.crop_names) and ('subcrop2' in self.crop_names):
            # Create two small, square subcrops of different size (to be able to detect blurry images):
            # This specific cropping method is highly experimental and can probably be improved
            subcrop_area_fractions = [0.15, 0.1]
            subcrop_w1 = int((w * h * subcrop_area_fractions[0]) ** 0.5)
            subcrop_h1 = subcrop_w1
            subcrop_w2 = int((w * h * subcrop_area_fractions[1]) ** 0.5)
            subcrop_h2 = subcrop_w2

            if w >= h:  # wide / square img, crop left and right of centre
                subcrop_center_h_1 = h // 2
                subcrop_center_h_2 = subcrop_center_h_1
                subcrop_center_w_1 = w // 4
                subcrop_center_w_2 = w // 4 * 3
            else:  # tall img, crop above and below centre
                subcrop_center_w_1 = w // 2
                subcrop_center_w_2 = subcrop_center_w_1
                subcrop_center_h_1 = h // 4
                subcrop_center_h_2 = h // 4 * 3

            subcrop_1 = img[:, :, subcrop_center_h_1 - subcrop_h1 // 2:subcrop_center_h_1 + subcrop_h1 // 2,
                    subcrop_center_w_1 - subcrop_w1 // 2:subcrop_center_w_1 + subcrop_w1 // 2]
            subcrop_2 = img[:, :, subcrop_center_h_2 - subcrop_h2 // 2:subcrop_center_h_2 + subcrop_h2 // 2,
                    subcrop_center_w_2 - subcrop_w2 // 2:subcrop_center_w_2 + subcrop_w2 // 2]
            
            # resize to self.img_resolution:
            subcrop_1 = F.interpolate(subcrop_1, size=self.img_resolution, mode='bilinear', align_corners=False)
            subcrop_2 = F.interpolate(subcrop_2, size=self.img_resolution, mode='bilinear', align_corners=False)

            crops.append(subcrop_1)
            crops.append(subcrop_2)
            crop_names.append(f"subcrop1_{subcrop_area_fractions[0]}")
            crop_names.append(f"subcrop2_{subcrop_area_fractions[1]}")

        if random.random() < sample_fraction_save_to_disk:
            timestamp = str(int(time.time()*100))
            for crop, crop_name in zip(crops, crop_names):
                crop = crop.squeeze(0).permute(1, 2, 0).cpu().numpy()
                crop = (crop * 255).astype(np.uint8)
                crop = Image.fromarray(crop)
                crop.save(f"./{timestamp}_{crop_name}.jpg")

        # stack the crops into a single tensor:
        crops = torch.cat(crops, dim=0)

        return crops, crop_names


import time
class Timer():
    'convenience class to time code'
    def __init__(self, name, start = False):
        self.name = name
        self.total_time_running = 0.0
        if start:
            self.start()

    def pause(self):
        self.total_time_running += time.time() - self.last_start

    def start(self):
        self.last_start = time.time()

    def status(self):
        print(f'{self.name} accumulated {self.total_time_running:.3f} seconds of runtime')

    def exit(self, *args):
        self.total_time_running += time.time() - self.last_start
        print(f'{self.name} took {self.total_time_running:.3f} seconds')


class AestheticRegressor:
    """
    Aesthetic Regressor to predict the aesthetic score of images.
    """
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.load_model(model_path)
        
        # Load associated CLIP models
        self.clip_models = [CLIP_Model(name, device=self.device) for name in self.model.clip_models]
    
    @torch.no_grad()
    def load_model(self, model_path, verbose=1):
        self.model = torch.load(model_path, map_location = self.device).to(self.device).eval()
        if verbose:
            print("Loaded regression model")
            print(f"Aesthetic Regressor was trained on embeddings from CLIP models:")
            print(self.model.clip_models)
            print(f"Aesthetic Regressor used crops:")
            print(self.model.crop_names)

    @torch.no_grad()
    def predict_score(self, pil_img):
        all_img_features = []
        
        for clip_model in self.clip_models:
            img_dataset = CustomImageDataset([pil_img], clip_model.img_resolution, self.model.crop_names, self.device)
            crops, _ = img_dataset.extract_crops(pil_img)
            features = clip_model.pt_imgs_to_features(crops).unsqueeze(0)  # Add batch dimension
            all_img_features.append(features)
        
        features = torch.stack(all_img_features)
        score = self.model(features.to(self.device).float()).item()

        return score, features
