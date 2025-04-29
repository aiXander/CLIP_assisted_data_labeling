import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset
import open_clip
import os, sys
import random
from PIL import Image
import numpy as np

# Hardcoded hack, TODO: clean this up:
pe_path = "../perception_models"
sys.path.append(pe_path)
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms

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


class CLIP_Encoder:
    def __init__(self, model_name, model_path=None, device=None):
        self.device = device if device else _DEVICE
        self.precision = 'fp16' if self.device == 'cuda' else 'fp32'
        self.model_name = model_name
        self.model_architecture, self.pretrained_dataset = self.model_name.split('/', 2)

        print(f"Loading CLIP model {self.model_name}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_architecture,
            pretrained=self.pretrained_dataset,
            precision=self.precision,
            device=self.device,
            jit=False,
            cache_dir=model_path
        )
        self.model = self.model.to(self.device).eval()
        # Extract resolution from preprocess transforms
        self.img_resolution = 224 # Default
        for t in self.preprocess.transforms:
            if isinstance(t, transforms.Resize):
                # open_clip uses int or tuple for size. If int, it's square.
                size = t.size
                if isinstance(size, int):
                    self.img_resolution = size
                elif isinstance(size, (list, tuple)) and len(size) >= 1:
                     self.img_resolution = size[0] # Assume square if tuple/list
                break


        print(f"CLIP model {self.model_name} with img_resolution {self.img_resolution} loaded on {self.device}!")

    def get_preprocess_transform(self):
        # Return the full preprocessing pipeline from open_clip
        return self.preprocess

    @torch.no_grad()
    def encode_image(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        if self.precision == 'fp16':
            preprocessed_images = preprocessed_images.half()
        image_features = self.model.encode_image(preprocessed_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


class PE_Encoder:
    def __init__(self, model_name, device=None):
        self.device = device if device else _DEVICE
        self.model_name = model_name

        print(f"Loading PE model {self.model_name}...")
        self.model = pe.CLIP.from_config(self.model_name, pretrained=True)
        self.model = self.model.to(self.device).eval()
        self.img_resolution = self.model.image_size
        self.context_length = self.model.context_length # Needed for tokenizer later if text is used

        # Get the appropriate image transform
        # self.preprocess = pe_transforms.get_image_transform(self.img_resolution) # Removed due to potential lambda/pickle issues
        # Define preprocessing directly to ensure pickle compatibility
        self.preprocess = transforms.Compose([
            transforms.Resize(self.img_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.img_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), # Assuming standard CLIP/PE mean
                std=(0.26862954, 0.26130258, 0.27577711),  # Assuming standard CLIP/PE std
            ),
        ])
        # self.tokenizer = pe_transforms.get_text_tokenizer(self.context_length) # If text needed

        print(f"PE model {self.model_name} with img_resolution {self.img_resolution} loaded on {self.device}!")

    def get_preprocess_transform(self):
        # Return the specific PE image transform
        return self.preprocess

    @torch.no_grad()
    @torch.autocast("cuda") # PE example uses autocast
    def encode_image(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        # PE model forward returns tuple (image_features, text_features, logit_scale)
        # We only need image_features for this task. PE might need dummy text.
        # Let's check the signature or assume None works for text if only image features needed.
        # From test.py, it seems PE model expects both image and text.
        # We need a way to get only image features. Let's assume model.encode_image exists, like CLIP.
        # If not, we'll need to adapt. Let's check PE source or documentation.
        # Assuming `encode_image` exists and works like open_clip:
        image_features = self.model.encode_image(preprocessed_images) # Placeholder assumption
        # If encode_image doesn't exist, we'd use:
        # dummy_text = self.tokenizer([""]).to(self.device) # Create dummy text input
        # image_features, _, _ = self.model(preprocessed_images, dummy_text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


class CustomImageDataset(Dataset):
    # Modified __init__ to accept preprocess_transform instead of img_resolution/device
    def __init__(self, image_paths, crop_names, preprocess_transform):
        self.image_paths = image_paths
        self.crop_names  = crop_names
        self.preprocess_transform = preprocess_transform # Store the transform
        self.img_featurizer = ImageFeaturizer()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            pil_img = Image.open(img_path).convert('RGB')
            # extract_crops now returns raw PIL crops or tensors before final preprocessing
            raw_crops, crop_names_list = self.extract_crops(pil_img)
            image_features = self.img_featurizer.process(np.array(pil_img))

            # Apply the specific preprocessing transform to each crop
            processed_crops = torch.stack([self.preprocess_transform(crop) for crop in raw_crops])

            return processed_crops, crop_names_list, img_path, image_features
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            # Return data from a random valid image instead
            random_idx = random.randint(0, len(self.image_paths)-1)
            print(f"Substituting with image index {random_idx}")
            return self.__getitem__(random_idx)

    # Modified extract_crops to return PIL images or basic tensors, preprocessing is done in __getitem__
    def extract_crops(self, pil_img: Image) -> (list, list):
        img_tensor = transforms.ToTensor()(pil_img) # Keep as basic tensor initially
        c, h, w = img_tensor.shape

        raw_crops, crop_names_list = [], []

        # Convert tensor back to PIL for potential transforms that expect PIL
        # Or adjust cropping logic if transforms handle tensors directly
        # Assuming preprocess_transform takes PIL:
        pil_img_for_crop = transforms.ToPILImage()(img_tensor)


        if 'centre_crop' in self.crop_names:
            crop_size = min(pil_img.width, pil_img.height)
            # Use torchvision transforms for cropping PIL images
            centre_crop_transform = transforms.CenterCrop(crop_size)
            centre_crop_pil = centre_crop_transform(pil_img_for_crop)
            raw_crops.append(centre_crop_pil)
            crop_names_list.append("centre_crop")

        if 'square_padded_crop' in self.crop_names:
            crop_size = max(pil_img.width, pil_img.height)
            # Create square padded PIL image
            square_padded_pil = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))
            start_h = (crop_size - pil_img.height) // 2
            start_w = (crop_size - pil_img.width) // 2
            square_padded_pil.paste(pil_img_for_crop, (start_w, start_h))
            raw_crops.append(square_padded_pil)
            crop_names_list.append("square_padded_crop")


        if any('subcrop1' in name for name in self.crop_names) or any('subcrop2' in name for name in self.crop_names):
            subcrop_area_fractions = [0.15, 0.1]
            subcrop_w1 = int((pil_img.width * pil_img.height * subcrop_area_fractions[0]) ** 0.5)
            subcrop_h1 = subcrop_w1
            subcrop_w2 = int((pil_img.width * pil_img.height * subcrop_area_fractions[1]) ** 0.5)
            subcrop_h2 = subcrop_w2

            if pil_img.width >= pil_img.height: # wide / square img
                centers = [(pil_img.width // 4, pil_img.height // 2), (pil_img.width // 4 * 3, pil_img.height // 2)]
            else: # tall img
                centers = [(pil_img.width // 2, pil_img.height // 4), (pil_img.width // 2, pil_img.height // 4 * 3)]

            sizes = [(subcrop_w1, subcrop_h1), (subcrop_w2, subcrop_h2)]
            names = ['subcrop1', 'subcrop2']

            for i, (center_w, center_h) in enumerate(centers):
                 if names[i] in self.crop_names:
                    width, height = sizes[i]
                    left = max(0, center_w - width // 2)
                    top = max(0, center_h - height // 2)
                    right = min(pil_img.width, left + width)
                    bottom = min(pil_img.height, top + height)

                    # Adjust size if crop went out of bounds to maintain aspect ratio (or simply crop)
                    # Using PIL's crop which handles bounds: box is (left, upper, right, lower)
                    subcrop_pil = pil_img_for_crop.crop((left, top, right, bottom))

                    # Ensure the cropped area isn't empty due to rounding/bounds
                    if subcrop_pil.width > 0 and subcrop_pil.height > 0:
                         raw_crops.append(subcrop_pil)
                         crop_names_list.append(names[i])
                    else:
                         print(f"Warning: {names[i]} for image {idx} resulted in zero size.")


        # Return list of PIL images and their names
        return raw_crops, crop_names_list


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
        self.clip_models = [CLIP_Encoder(name, device=self.device) for name in self.model.clip_models]
    
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
            img_dataset = CustomImageDataset([pil_img], clip_model.crop_names, clip_model.get_preprocess_transform())
            crops, _ = img_dataset.extract_crops(pil_img)
            features = clip_model.encode_image(crops).unsqueeze(0)  # Add batch dimension
            all_img_features.append(features)
        
        features = torch.stack(all_img_features).flatten().unsqueeze(0)  # Add batch dimension
        score = self.model(features.to(self.device).float()).item()

        return score, features
