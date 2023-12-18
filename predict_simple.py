import os
import shutil
from PIL import Image
from utils.embedder import AestheticRegressor
from tqdm import tqdm
import argparse
import torch

def predict_images(img_paths, model_path, device, output_dir = None):

    # Load the scoring model (only do this once in a python session):
    aesthetic_regressor = AestheticRegressor(model_path, device = device)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok = True)

    print("\nPredicting aesthetic scores...")
    for image_path in tqdm(img_paths):
        score, embedding = aesthetic_regressor.predict_score(Image.open(image_path))
        print(f"Score: {score:.3f} for {os.path.basename(image_path)}")

        if output_dir is not None:
            output_path = os.path.join(output_dir, f'{score:.3f}_' + os.path.basename(image_path))
            shutil.copy(image_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # IO args:
    parser.add_argument('--input_img_dir', type=str, help='Root directory of the (optionally multiple) datasets')
    parser.add_argument('--model_path',    type=str, default='models/single_crop_regression_9.4k_imgs_80_epochs.pth', help='Path to the model file (.pth)')
    args = parser.parse_args()

    input_img_dir = args.input_img_dir

    #output_dir = None # dont copy the scored images
    output_dir = input_img_dir + "_aesthetic_scores" # copy the scored images

    # Get all the img_paths:
    img_extensions = [".jpg", ".png", ".jpeg", ".bmp", ".webp"]
    list_of_img_paths = [os.path.join(input_img_dir, img_name) for img_name in os.listdir(input_img_dir) if os.path.splitext(img_name)[1].lower() in img_extensions]
    print(f"Found {len(list_of_img_paths)} images in {input_img_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict_images(list_of_img_paths, args.model_path, device, output_dir)


