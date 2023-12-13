import os
import shutil
from PIL import Image
from utils.embedder import AestheticRegressor
from tqdm import tqdm

if __name__ == "__main__":

    input_img_dir = "/home/rednax/SSD2TB/Fast_Datasets/SD/Labeling/test"
    model_path    = "/home/rednax/SSD2TB/Xander_Tools/CLIP_assisted_data_labeling/models/eden_scorer_2023-12-13_9.4k_imgs_80_epochs.pth"
    #model_path    = "/home/rednax/SSD2TB/Xander_Tools/CLIP_assisted_data_labeling/models/eden_scorer_2023-12-13_9.4k_imgs_80_epochs_1_crop.pth"
    device        = "cpu"

    #output_dir = None # dont copy the scored images
    output_dir = input_img_dir + "_aesthetic_scores_2" # copy the scored images

    # Load the scoring model (only do this once in a python session):
    aesthetic_regressor = AestheticRegressor(model_path, device = device)

    # Get all the img_paths:
    list_of_img_paths = [os.path.join(input_img_dir, img_name) for img_name in os.listdir(input_img_dir) if '.jpg' in img_name]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok = True)

    print("\nPredicting aesthetic scores...")
    for image_path in tqdm(list_of_img_paths):
        score, embedding = aesthetic_regressor.predict_score(Image.open(image_path))
        print(f"Score: {score:.3f} for {os.path.basename(image_path)}")

        if output_dir is not None:
            output_path = os.path.join(output_dir, f'{score:.3f}_' + os.path.basename(image_path))
            shutil.copy(image_path, output_path)



