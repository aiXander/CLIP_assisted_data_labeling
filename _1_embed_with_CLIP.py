from PIL import Image
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

from utils.embedder import CustomImageDataset, CLIP_Model

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if 1:
    print("Pretrained clip models available:")
    options = open_clip.list_pretrained()
    for option in options:
        print(option)
    print("-----------------------------")

        
class CLIP_Feature_Dataset():
    def __init__(self, root_dir, clip_model_name, batch_size, 
                 clip_model_path = None, 
                 force_reencode = False, 
                 shuffle_filenames = True,
                 num_workers = 0,
                 crop_names = ["centre_crop", "square_padded_crop", "subcrop1", "subcrop2"]):

        self.device = _DEVICE
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
        self.img_dataset = CustomImageDataset(self.img_filepaths, self.img_encoder.img_resolution, crop_names, self.device)
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
                n_skipped += batch_size

            if (n_embedded + n_skipped) % 1000 == 0:
                print(f"Skipped {n_skipped} images, embedded {n_embedded} images")

        print("\n--- Feature encoding done! ---\n")
        print(f"Embedded {n_embedded} images ({n_skipped} images were already embedded).")
        print(f"All feature vector dicts were saved to {self.root_dir}")
        print(f"Subcrop names that were saved: {self.crop_names}")
        print("-----------------------------------------------\n\n")

if __name__ == "__main__":

    """
    - Loop over all the images in the root_dir
    - Create multiple, standardized crops for each img
    - Embed them with CLIP (possibly multiple models)
    - Save the embeddings to disk
    """
    
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