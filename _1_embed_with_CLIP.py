from PIL import Image
import open_clip
import torch, os, time
from tqdm import tqdm
import random
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.nn import functional as F
import numpy as np

from utils.image_features import ImageFeaturizer

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

if 0:
    print("Pretrained clip models available:")
    options = open_clip.list_pretrained()
    for option in options:
        print(option)
    print("-----------------------------")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model = model.to(device)

    # Extract features
    with torch.no_grad():
        features = model(image)

    return features



class CLIP_Model:
    def __init__(self, clip_model_name, clip_model_path = None, use_pickscore_encoder = False):
        self.use_pickscore_encoder = use_pickscore_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                precision='fp16' if self.device == 'cuda' else 'fp32',
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
            ).to("cuda")
            image_features = self.clip_model.get_image_features(**preprocessed_images)
        else:
            preprocessed_images = [self.clip_tensor_preprocess(img) for img in list_of_tensors]        
            preprocessed_images = torch.stack(preprocessed_images).to(self.device)
            image_features = self.clip_model.encode_image(preprocessed_images)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features
    
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, img_resolution, crop_names):
        self.image_paths = image_paths
        self.crop_names  = crop_names
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


        
class CLIP_Feature_Dataset():
    def __init__(self, root_dir, clip_model_name, batch_size, 
                 clip_model_path = None, 
                 force_reencode = False, 
                 shuffle_filenames = True,
                 num_workers = 0,
                 crop_names = ["centre_crop", "square_padded_crop", "subcrop1", "subcrop2"]):
        
        self.root_dir = root_dir
        self.force_reencode = force_reencode
        self.img_extensions = (".png", ".jpg", ".jpeg", ".JPEG", ".JPG", ".PNG")
        self.batch_size = batch_size
        self.crop_names = crop_names

        # Find all images in root_dir:
        print("Searching images..")
        self.img_filepaths = []
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                if name.endswith(self.img_extensions):
                    new_filename = os.path.join(root, name)
                    self.img_filepaths.append(new_filename)
        
        if shuffle_filenames:
            random.shuffle(self.img_filepaths)
        else: # sort filenames:
            self.img_filepaths.sort()

        print(f"---> Found {len(self.img_filepaths)} images in {root_dir}")

        # Get ready for processing:
        self.img_encoder = CLIP_Model(clip_model_name, clip_model_path)
        self.img_dataset = CustomImageDataset(self.img_filepaths, self.img_encoder.img_resolution, crop_names)
        self.dataloader  = DataLoader(self.img_dataset, 
                                        batch_size=batch_size, shuffle=False, 
                                        num_workers=num_workers, prefetch_factor=2)

    def __len__(self):
        return len(self.img_filepaths)

    @torch.no_grad()
    def process(self):
        n_embedded, n_skipped = 0, 0
        print(f"Embedding dataset of {len(self.img_filepaths)} images...")

        for batch_id, batch in enumerate(tqdm(self.dataloader)):
            crops, crop_names_batch, img_paths, img_feature_dict_batch = batch
            batch_size = crops.shape[0]
            base_img_paths     = [os.path.splitext(img_path)[0] for img_path in img_paths]
            feature_save_paths = [base_img_path + ".pt" for base_img_path in base_img_paths]
            crop_names_batch   = [[crop[i] for crop in crop_names_batch] for i in range(batch_size)]

            # collapse all non-img dimensions into a single dimension (to do a batch CLIP-embed):
            crops_stacked = crops.view(-1, *crops.shape[-3:])

            # Find all the already existing .pt files for this batch:
            existing_feature_save_paths = [feature_save_path for feature_save_path in feature_save_paths if os.path.exists(feature_save_path)]
            # Count how many of those files already hold the features for the current CLIP-model:
            already_encoded = 0
            for feature_save_path in existing_feature_save_paths:
                feature_dict = torch.load(feature_save_path)
                if self.img_encoder.clip_model_name in feature_dict.keys():
                    already_encoded += 1

            if self.force_reencode or not already_encoded == batch_size:
                # batch-embed the crops into CLIP:
                features = self.img_encoder.pt_imgs_to_features(crops_stacked)
                # Reshape the features back into [batch_size x n_crops x dim]:
                features = features.view(batch_size, -1, features.shape[-1])

                # save the features as a dict of dicts to disk:
                batch_index = 0
                for feature, feature_save_path, crop_names in zip(features, feature_save_paths, crop_names_batch):
                    feature_dict = {}
                    for img_feature_name in img_feature_dict_batch.keys():
                        feature_dict[img_feature_name] = img_feature_dict_batch[img_feature_name][batch_index]

                    for feature_crop, crop_name in zip(feature, crop_names):
                        feature_dict[crop_name] = feature_crop.unsqueeze(0)

                    # Convert all the tensors in the dict to torch.float32:
                    feature_dict = {k: v.float() for k, v in feature_dict.items()}
                    
                    final_feature_dict = {}
                    if os.path.exists(feature_save_path): # Load the existing feature dict if it exists:
                        final_feature_dict = torch.load(feature_save_path)

                    # nest the current clip_model feature_dict into the final_feature_dict with the CLIP-model name:
                    final_feature_dict[self.img_encoder.clip_model_name] = feature_dict

                    torch.save(final_feature_dict, feature_save_path)
                    batch_index += 1

                n_embedded += batch_size
            else:
                print(f"All images in batch {batch_id} already embedded with {self.img_encoder.clip_model_name}, skipping..")
                n_skipped += batch_size

            if (n_embedded + n_skipped) % 1000 == 0:
                print(f"Skipped {n_skipped} images, embedded {n_embedded} images")

        print("\n--- Feature encoding done! ---\n")
        print(f"Embedded {n_embedded} images ({n_skipped} images were already embedded).")
        print(f"All feature vector dicts were saved to {self.root_dir}")
        print(f"Subcrop names that were saved: {self.crop_names}")
        print("-----------------------------------------------\n\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root directory of the dataset (can contain subdirectories)')
    parser.add_argument('--clip_models_to_use', type=str, nargs='+', default=['ViT-L-14-336/openai'], help='Which (possibly multiple) CLIP models to use for embedding, defaults to ViT-L-14-336/openai')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images to encode at once')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for the dataloader')
    parser.add_argument('--force_reencode', action='store_true', help='Force CLIP re-encoding of all images (default: False)')
    args = parser.parse_args()

    # Which img-crops to embed with CLIP and save to disk, see extract_crops() method:
    crop_names = ['centre_crop', 'square_padded_crop', 'subcrop1', 'subcrop2']
    
    mp.set_start_method('spawn')
    
    print(f"Embedding all imgs with {len(args.clip_models_to_use)} CLIP models: \n--> {args.clip_models_to_use}")

    for clip_model_name in args.clip_models_to_use:
        dataset = CLIP_Feature_Dataset(args.root_dir, clip_model_name, args.batch_size, 
                                    clip_model_path = None, 
                                    force_reencode = args.force_reencode, 
                                    num_workers = args.num_workers,
                                    crop_names = crop_names)
        dataset.process()