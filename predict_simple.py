import os
import torch
from PIL import Image

from _1_embed_with_CLIP import CustomImageDataset, CLIP_Model

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
            img_dataset = CustomImageDataset([pil_img], clip_model.img_resolution, self.model.crop_names)
            crops, _ = img_dataset.extract_crops(pil_img)
            features = clip_model.pt_imgs_to_features(crops).unsqueeze(0)  # Add batch dimension
            all_img_features.append(features)
        
        features = torch.stack(all_img_features)
        return self.model(features.to(self.device).float()).item()  # Convert tensor to single float value


if __name__ == "__main__":

    input_img_dir = "/home/xander/Projects/cog/xander_eden_stuff/xander/assets/garden"
    model_path    = "/home/xander/Projects/cog/xander_eden_stuff/xander/CLIP_assisted_data_labeling/models/combo_2023-08-02_03:48:00_8.1k_imgs_80_epochs_-1.0000_mse.pth"
    device        = "cpu"

    # Load the scoring model (only do this once in a python session):
    aesthetic_regressor = AestheticRegressor(model_path, device = device)

    # Get all the img_paths:
    list_of_img_paths = [os.path.join(input_img_dir, img_name) for img_name in os.listdir(input_img_dir) if '.jpg' in img_name]

    print("\nPredicting aesthetic scores...")
    for image_path in list_of_img_paths:
        score = aesthetic_regressor.predict_score(Image.open(image_path))
        print(f"Score: {score:.3f} for {os.path.basename(image_path)}")

