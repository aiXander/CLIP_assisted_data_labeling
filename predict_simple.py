import os
import numpy as np
import pickle
import torch
from PIL import Image

from utils.nn_model import SimpleFC
from _1_embed_with_CLIP import CustomImageDataset, CLIP_Model

def load_model(model_path, device):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        model.to(device)
        model.eval()

        # disable gradients:
        for param in model.parameters():
            param.requires_grad = False

        print("Loaded regression model")
        print(f"Aesthetic Regressor was trained on embeddings from CLIP models:")
        print(model.clip_models)
        print(f"Aesthetic Regressor used crops:")
        print(model.crop_names)

    return model

def predict_score(pil_img, aesthetic_regressor, active_clip_models):
    all_img_features = []
    for clip_model in active_clip_models:
        # This can be improved, but is fine for now:
        img_dataset = CustomImageDataset([image_path], clip_model.img_resolution, aesthetic_regressor.crop_names)
        crops, crop_names = img_dataset.extract_crops(pil_img)
        features = clip_model.pt_imgs_to_features(crops)

        # Reshape the features back into [batch_size x n_crops x dim]:
        batch_size = 1
        features = features.view(batch_size, -1, features.shape[-1])

        all_img_features.append(features)

    features = torch.stack(all_img_features)
    score = aesthetic_regressor(features.to(clip_model.device).float()).cpu().numpy().squeeze()
    return score


if __name__ == "__main__":

    input_img_dir = "/home/xander/Projects/cog/xander_eden_stuff/xander/assets/garden"
    model_path    = "/home/xander/Projects/cog/xander_eden_stuff/xander/CLIP_assisted_data_labeling/models/combo_2023-07-31_06:20:57_8.1k_imgs_80_epochs_-1.0000_mse.pkl"
    device        = "cpu"

    # Load the scoring model:
    aesthetic_regressor = load_model(model_path, device)

    # Load the CLIP models corresponding to this pretrained Aesthetic Regressor:
    active_clip_models = []
    for clip_model_name in aesthetic_regressor.clip_models:
        active_clip_models.append(CLIP_Model(clip_model_name, device = device))

    # Get all the img_paths:
    list_of_img_paths = [os.path.join(input_img_dir, img_name) for img_name in os.listdir(input_img_dir) if '.jpg' in img_name]

    print("\nPredicting aesthetic scores...")
    for image_path in list_of_img_paths:
        pil_img = Image.open(image_path)
        score = predict_score(pil_img, aesthetic_regressor, active_clip_models)
        print(f"Score: {score:.3f} for {os.path.basename(image_path)}")

