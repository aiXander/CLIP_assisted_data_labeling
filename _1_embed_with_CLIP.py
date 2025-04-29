from PIL import Image
# Keep open_clip import for potential type hints or listing models, though not directly used for encoding now
import open_clip 
import torch, os, time
from tqdm import tqdm
import random
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from utils.embedder import CustomImageDataset, CLIP_Encoder, PE_Encoder

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if 0:
    # Consider listing PE models too if available
    print("Pretrained open_clip models available:")
    try:
        options = open_clip.list_pretrained()
        for option in options:
            print(option)
    except Exception as e:
        print(f"Could not list open_clip models: {e}")
    # Add PE model listing here if possible
    # Example: print(pe.CLIP.available_configs()) # Need to import pe
    print("-----------------------------")


# Rename CLIP_Feature_Dataset to Feature_Dataset
class Feature_Dataset():
    # Updated __init__ to handle different model types
    def __init__(self, root_dir, model_name, batch_size,
                 model_path = None,
                 force_reencode = False,
                 shuffle_filenames = True,
                 num_workers = 0,
                 crop_names = ["centre_crop", "square_padded_crop", "subcrop1", "subcrop2"]):

        self.device = _DEVICE
        self.root_dir = root_dir
        self.model_name = model_name # Store the model name
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

        # Instantiate the correct encoder based on model name convention
        # Assuming PE models start with 'PE-' based on test.py
        if model_name.startswith("PE-"):
            self.encoder = PE_Encoder(model_name, device=self.device)
            # PE model path isn't handled by PE_Encoder yet, assuming download from HF
        elif '/' in model_name: # Assume open_clip format like 'ViT-L-14-336/openai'
            self.encoder = CLIP_Encoder(model_name, model_path, device=self.device)
        else:
            raise ValueError(f"Unknown model format: {model_name}. Expected 'PE-...' or 'Arch/Dataset'.")

        # Get the preprocessing transform from the encoder
        preprocess_transform = self.encoder.get_preprocess_transform()

        # Pass the transform to CustomImageDataset
        self.img_dataset = CustomImageDataset(self.img_filepaths, self.crop_names, preprocess_transform)
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers
        }
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = 2 # Default value, adjust if needed

        self.dataloader = DataLoader(self.img_dataset, **dataloader_kwargs)

    def __len__(self):
        return len(self.img_filepaths)

    @torch.no_grad()
    def process(self):
        n_embedded, n_skipped = 0, 0
        print(f"Embedding dataset of {len(self.img_filepaths)} images using {self.model_name}...")

        for batch_id, batch in enumerate(tqdm(self.dataloader)):
            # Batch now contains preprocessed crops directly from CustomImageDataset
            processed_crops, crop_names_batch, img_paths, img_feature_dict_batch = batch
            batch_size = processed_crops.shape[0]
            base_img_paths     = [os.path.splitext(img_path)[0] for img_path in img_paths]
            feature_save_paths = [base_img_path + ".pt" for base_img_path in base_img_paths]
            # Adjust crop_names_batch structure if needed based on how CustomImageDataset returns it
            # Assuming it's now a list of lists [batch_size] x [n_crops]
            # crop_names_batch needs careful handling if not returned per-item by dataloader

            # Collapse batch and crop dimensions for encoding:
            # Input shape: [batch_size, n_crops, C, H, W]
            # Desired shape for encoder: [batch_size * n_crops, C, H, W]
            num_crops = processed_crops.shape[1]
            crops_stacked = processed_crops.view(-1, *processed_crops.shape[2:]) # [B*N_crops, C, H, W]
            crops_stacked = crops_stacked.to(self.device) # Ensure tensor is on the correct device

            # Check existence based on the specific model name
            existing_feature_save_paths = [p for p in feature_save_paths if os.path.exists(p)]
            already_encoded = 0
            for feature_save_path in existing_feature_save_paths:
                try:
                    feature_dict = torch.load(feature_save_path, map_location='cpu') # Load to CPU first
                    if self.model_name in feature_dict.keys(): # Check for the specific model key
                        already_encoded += 1
                except Exception as e:
                    print(f"Warning: Could not load existing feature file {feature_save_path}: {e}")

            if self.force_reencode or not already_encoded == batch_size:
                # Use the encoder instance to embed the stacked, preprocessed crops
                features = self.encoder.encode_image(crops_stacked)
                # Reshape features back to [batch_size, n_crops, dim]
                features = features.view(batch_size, num_crops, features.shape[-1])

                # Save features (logic remains similar, but uses self.model_name as key)
                # Iterate through batch items, including the corresponding crop names list for each image
                for i, (feature_per_image, feature_save_path, img_path, current_crop_names) in enumerate(zip(features, feature_save_paths, img_paths, crop_names_batch)):
                    # Load existing feature dict if it exists, otherwise create new
                    final_feature_dict = {}
                    if os.path.exists(feature_save_path) and not self.force_reencode:
                        try:
                            final_feature_dict = torch.load(feature_save_path, map_location='cpu')
                        except Exception as e:
                            print(f"Warning: Failed to load existing {feature_save_path} for update: {e}")

                    # Extract per-crop features and store them
                    feature_dict_for_model = {}

                    # Load base image features (like HoG, FFT) - needs careful indexing
                    # Assuming img_feature_dict_batch is structured correctly for the batch
                    for img_feature_name in img_feature_dict_batch.keys():
                         # Ensure we are getting the correct item for the current image in the batch (index i)
                         feature_dict_for_model[img_feature_name] = img_feature_dict_batch[img_feature_name][i]

                    # Store features for each crop under its name
                    # current_crop_names is now directly available from the loop
                    for feature_crop, crop_name in zip(feature_per_image, current_crop_names):
                        feature_dict_for_model[crop_name] = feature_crop.unsqueeze(0).cpu() # Store on CPU

                    # Convert all tensors in the dict to float32 for consistency
                    feature_dict_for_model = {k: v.float() if isinstance(v, torch.Tensor) else v
                                              for k, v in feature_dict_for_model.items()}

                    # Add/update the features for the current model in the main dictionary
                    final_feature_dict[self.model_name] = feature_dict_for_model

                    # Save the updated dictionary
                    try:
                        torch.save(final_feature_dict, feature_save_path)
                    except Exception as e:
                        print(f"Error saving features to {feature_save_path}: {e}")

                n_embedded += batch_size
            else:
                n_skipped += batch_size

            if (n_embedded + n_skipped) > 0 and (n_embedded + n_skipped) % 1000 == 0:
                 print(f"Processed {n_embedded + n_skipped} images. Skipped: {n_skipped}, Embedded: {n_embedded}")


        print("\n--- Feature encoding done! ---\n")
        print(f"Embedded {n_embedded} images ({n_skipped} images were already embedded). Features saved with model key '{self.model_name}'.")
        print(f"Feature vector dicts were saved alongside original images in {self.root_dir}")
        print(f"Crop names that were processed: {self.crop_names}")
        print("-----------------------------------------------\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset (can contain subdirectories)')
    # Rename argument from clip_models_to_use to models_to_use
    parser.add_argument('--models_to_use', type=str, nargs='+', default=['ViT-L-14-336/openai'],
                        help='Which CLIP (e.g., ViT-L-14-336/openai) or PE (e.g., PE-Core-B16-224) models to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images to encode at once')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
    parser.add_argument('--force_reencode', action='store_true', help='Force re-encoding of all images for the specified models (default: False)')
    # Add model_path argument if needed for local CLIP models
    parser.add_argument('--model_path', type=str, default=None, help='Path to local directory for downloading/loading models (optional)')
    args = parser.parse_args()

    # Crop names remain the same
    crop_names = ['centre_crop', 'square_padded_crop', 'subcrop1', 'subcrop2']

    mp.set_start_method('spawn')

    print(f"Embedding all imgs with {len(args.models_to_use)} models: \n--> {args.models_to_use}")

    # Loop through the specified models
    for model_name in args.models_to_use:
        print(f"\n--- Processing model: {model_name} ---")
        # Instantiate the renamed Feature_Dataset class
        dataset = Feature_Dataset(args.root_dir, model_name, args.batch_size,
                                model_path = args.model_path, # Pass model_path
                                force_reencode = args.force_reencode,
                                num_workers = args.num_workers,
                                crop_names = crop_names)
        dataset.process()