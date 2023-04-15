from PIL import Image
import open_clip
import torch, os, time
from tqdm import tqdm
import random
import argparse

class CLIP_Model:
    def __init__(self, clip_model_name, clip_model_path = None, device = "cuda"):
        self.device = device
        clip_model_name, clip_model_pretrained_name = clip_model_name.split('/', 2)

        print(f"Loading CLIP model {clip_model_name}...")

        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=clip_model_pretrained_name, 
            precision='fp16' if device == 'cuda' else 'fp32',
            device=device,
            jit=False,
            cache_dir=clip_model_path
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.img_resolution = 336 if '336' in clip_model_name else 224

        print(f"CLIP model {clip_model_name} with img_resolution {self.img_resolution} loaded!")

    def image_to_features(self, list_of_pil_images: list) -> torch.Tensor:
        images = [self.clip_preprocess(img) for img in list_of_pil_images]
        images = torch.stack(images).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    

class ImageEncoder:
    """
    Helper class to extract CLIP features from images   
    """
    def __init__(self, clip_model_name, clip_model_path = None):
        self.model =  CLIP_Model(clip_model_name, clip_model_path = clip_model_path)

    def encode_image(self, pil_img: Image) -> torch.Tensor:
        crops = self.extract_crops(pil_img)
        features = self.model.image_to_features(crops)
        return features

    def extract_crops(self, pil_img: Image, 
                      sample_fraction_save_to_disk = 0.0, # Save a fraction of crops to disk (can be used for debugging)
                      ) -> list:
        w, h = pil_img.size

        # extract square centre crop:
        centre_crop = pil_img.crop((w//2 - h//2, 0, w//2 + h//2, h))

        # Create square padded image:
        square_padded_img = Image.new("RGB", (max(w,h), max(w,h)), (0, 0, 0))
        square_padded_img.paste(pil_img, (max(0, h-w)//2, max(0, w-h)//2))

        # Create two small, square subcrops of different size (to be able to detect blurry images):
        # This specific cropping method is highly experimental and can probably be improved

        subcrop_area_fractions = [0.15, 0.1]
        subcrop_w1 = int((w * h * subcrop_area_fractions[0]) ** 0.5)
        subcrop_h1 = subcrop_w1
        subcrop_w2 = int((w * h * subcrop_area_fractions[1]) ** 0.5)
        subcrop_h2 = subcrop_w2

        if w >= h: # wide / square img, crop left and right of centre
            subcrop_center_h_1 = h//2
            subcrop_center_h_2 = subcrop_center_h_1
            subcrop_center_w_1 = w//4
            subcrop_center_w_2 = w//4 * 3
        else: # tall img, crop above and below centre
            subcrop_center_w_1 = w//2
            subcrop_center_w_2 = subcrop_center_w_1
            subcrop_center_h_1 = h//4
            subcrop_center_h_2 = h//4 * 3
        
        subcrop_1 = pil_img.crop((subcrop_center_w_1 - subcrop_w1//2, subcrop_center_h_1 - subcrop_h1//2, subcrop_center_w_1 + subcrop_w1//2, subcrop_center_h_1 + subcrop_h1//2))
        subcrop_2 = pil_img.crop((subcrop_center_w_2 - subcrop_w2//2, subcrop_center_h_2 - subcrop_h2//2, subcrop_center_w_2 + subcrop_w2//2, subcrop_center_h_2 + subcrop_h2//2))

        if random.random() < sample_fraction_save_to_disk: # Save all crops to disk:
            img_res = self.model.img_resolution
            centre_crop = centre_crop.resize((img_res,img_res))
            square_padded_img = square_padded_img.resize((img_res,img_res))
            subcrop_1 = subcrop_1.resize((img_res,img_res))
            subcrop_2 = subcrop_2.resize((img_res,img_res))

            timestamp = str(int(time.time()*100))
            centre_crop.save(f"./{timestamp}_centre_crop.jpg")
            square_padded_img.save(f"./{timestamp}_square_padded_img.jpg")
            subcrop_1.save(f"./{timestamp}_subcrop_1.jpg")
            subcrop_2.save(f"./{timestamp}_subcrop_2.jpg")

        return [centre_crop, square_padded_img, subcrop_1, subcrop_2]
    
class CLIP_Feature_Dataset():
    def __init__(self, root_dir, clip_model_name, clip_model_path = None, force_reencode = False, shuffle_filenames = True):
        self.root_dir = root_dir
        self.force_reencode = force_reencode
        self.img_extensions = (".png", ".jpg", ".jpeg", ".JPEG", ".JPG", ".PNG")
        self.img_encoder = ImageEncoder(clip_model_name, clip_model_path = clip_model_path)

        print("Searching for images..")
        self.img_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(root_dir) for name in files if name.endswith(self.img_extensions)]
        
        if shuffle_filenames:
            random.shuffle(self.img_filepaths)
        else: # sort filenames:
            self.img_filepaths.sort()
        print(f"Found {len(self.img_filepaths)} images in {root_dir}")

    def __len__(self):
        return len(self.img_filepaths)

    def process(self):
        for img_filepath in tqdm(self.img_filepaths):
            base_img_path = os.path.splitext(img_filepath)[0]
            feature_save_path = base_img_path + ".pt"

            if not os.path.exists(feature_save_path) or self.force_reencode:
                img = Image.open(img_filepath)
                features = self.img_encoder.encode_image(img)
                torch.save(features, feature_save_path)

        print("--- Feature encoding done!")
        print(f"Saved {len(self.img_filepaths)} feature vectors of shape {str(features.shape)} to {self.root_dir}")


"""

export CUDA_VISIBLE_DEVICES=1
cd /home/xander/Projects/cog/CLIP_active_learning_classifier/CLIP_assisted_data_labeling
python embed_with_CLIP_02.py

"""

if __name__ == "__main__":
    root_dir = "/data/datasets/midjourney2"
    clip_model_name = "ViT-L-14-336/openai"  # "ViT-L-14/openai" #SD 1.x  //  "ViT-H-14/laion2b_s32b_b79k" #SD 2.x
    clip_model_path = "/home/xander/Projects/cog/cache"
    
    dataset = CLIP_Feature_Dataset(root_dir, clip_model_name, clip_model_path = clip_model_path, force_reencode = False)
    dataset.process()