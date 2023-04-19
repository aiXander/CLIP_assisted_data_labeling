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

class CLIP_Model:
    def __init__(self, clip_model_name, clip_model_path = None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model_name, clip_model_pretrained_name = clip_model_name.split('/', 2)

        print(f"Loading CLIP model {clip_model_name}...")
        #self.tokenize = open_clip.get_tokenizer(clip_model_name)

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=clip_model_pretrained_name, 
            precision='fp16' if self.device == 'cuda' else 'fp32',
            device=self.device,
            jit=False,
            cache_dir=clip_model_path
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.img_resolution = 336 if '336' in clip_model_name else 224
        
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
        
        print(f"CLIP model {clip_model_name} with img_resolution {self.img_resolution} loaded!")

    def pt_imgs_to_features(self, list_of_tensors: list) -> torch.Tensor:
        preprocessed_images = [self.clip_tensor_preprocess(img) for img in list_of_tensors]        
        preprocessed_images = torch.stack(preprocessed_images).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(preprocessed_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features
    
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, img_resolution, crop_names):
        self.image_paths = image_paths
        self.crop_names  = crop_names
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_resolution = img_resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            pil_img = Image.open(img_path).convert('RGB')
            crops, crop_names = self.extract_crops(pil_img)
            return crops, crop_names, img_path
        except:
            print(f"Error loading image {img_path}")
            return self.__getitem__(random.randint(0, len(self.image_paths)-1))
    
    def extract_crops(self, pil_img: Image,
                  sample_fraction_save_to_disk=0.0  # Save a fraction of crops to disk (can be used for debugging)
                  ) -> list:
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
                 crop_names = ["centre_crop", "square_padded_crop", "subcrop1", "subcrop2"]):
        
        self.root_dir = root_dir
        self.force_reencode = force_reencode
        self.img_extensions = (".png", ".jpg", ".jpeg", ".JPEG", ".JPG", ".PNG")
        self.batch_size = batch_size
        self.crop_names = crop_names

        print("Searching for images..")
        self.img_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(root_dir) for name in files if name.endswith(self.img_extensions)]
        
        if shuffle_filenames:
            random.shuffle(self.img_filepaths)
        else: # sort filenames:
            self.img_filepaths.sort()
        print(f"Found {len(self.img_filepaths)} images in {root_dir}")

        # Get ready for processing:
        self.img_encoder = CLIP_Model(clip_model_name, clip_model_path)
        self.img_dataset = CustomImageDataset(self.img_filepaths, self.img_encoder.img_resolution, crop_names)
        self.dataloader  = DataLoader(self.img_dataset, 
                                        batch_size=batch_size, shuffle=False, 
                                        num_workers=args.num_workers, prefetch_factor=2)

    def __len__(self):
        return len(self.img_filepaths)

    def process(self):
        n_embedded, n_skipped = 0, 0
        print(f"Embedding dataset of {len(self.img_filepaths)} images...")

        for batch in tqdm(self.dataloader):
            crops, crop_names_batch, img_paths = batch
            batch_size = crops.shape[0]

            base_img_paths     = [os.path.splitext(img_path)[0] for img_path in img_paths]
            feature_save_paths = [base_img_path + ".pt" for base_img_path in base_img_paths]
            crop_names_batch   = [[crop[i] for crop in crop_names_batch] for i in range(batch_size)]
            # collapse all non-img dimensions into a single dimension (to do a batch CLIP-embed):
            crops_stacked = crops.view(-1, *crops.shape[-3:])

            if self.force_reencode or not all([os.path.exists(feature_save_path) for feature_save_path in feature_save_paths]):
                # batch-embed the crops into CLIP:
                features = self.img_encoder.pt_imgs_to_features(crops_stacked)
                # Reshape the features back into [batch_size x n_crops x dim]:
                features = features.view(batch_size, -1, features.shape[-1])

                # save the features as a dictionary:
                for feature, feature_save_path, crop_names in zip(features, feature_save_paths, crop_names_batch):
                    feature_dict = {}
                    for feature_crop, crop_name in zip(feature, crop_names):
                        feature_dict[crop_name] = feature_crop.unsqueeze(0)
                    torch.save(feature_dict, feature_save_path)

                n_embedded += batch_size
            else:
                n_skipped += batch_size

            if (n_embedded + n_skipped) % 1000 == 0:
                print(f"Skipped {n_skipped} images, embedded {n_embedded} images")

        print("--- Feature encoding done!")
        print(f"Saved {len(self.img_filepaths)} feature vector dicts to {self.root_dir}")
        print(f"Subcrop names that were saved: {crop_names}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root directory of the dataset')
    parser.add_argument('--clip_model_name', type=str, default = "ViT-L-14-336/openai", help='Name of the open_clip model, see https://github.com/mlfoundations/open_clip/tree/main/src/open_clip/model_configs')
    parser.add_argument('--batch_size', type=int, default=12, help='Number of images to encode at once')
    parser.add_argument('--force_reencode', action='store_true', help='Force CLIP re-encoding of all images (default: False)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for the dataloader')
    args = parser.parse_args()

    crop_names = ['centre_crop', 'square_padded_crop', 'subcrop1', 'subcrop2']
    
    mp.set_start_method('spawn')
    dataset = CLIP_Feature_Dataset(args.root_dir, args.clip_model_name, args.batch_size, 
                                   clip_model_path = None, 
                                   force_reencode = args.force_reencode, 
                                   crop_names = crop_names)
    dataset.process()