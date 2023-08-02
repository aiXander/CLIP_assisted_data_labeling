import os
from PIL import Image
from utils.embedder import AestheticRegressor

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

