import os
import shutil
from tqdm import tqdm
import argparse
import pandas as pd

def copy_data(input_dir, min_score, extensions = ['.jpg'], output_suffix = '_subset'):
    '''
    Copy all the files in the root_dir based on predicted label
    
    '''

    output_suffix = f'_>{min_score:.2f}' + output_suffix

    output_folder = os.path.join(input_dir + output_suffix)
    os.makedirs(output_folder, exist_ok=True)

    database_path = os.path.join(os.path.dirname(input_dir), os.path.basename(input_dir) + ".csv")
    database = pd.read_csv(database_path)

    # Get all the rows where the predicted label is above the threshold:
    database = database.loc[database["predicted_label"] >= min_score]
    
    # Loop over the uuids in the database and copy the corresponding files to the output folder:
    counter = [0] * len(extensions)
    for uuid in tqdm(database["uuid"].values):
        for ext in extensions:
            filename = uuid + ext
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_folder, filename)
            shutil.copy(input_path, output_path)
            counter[extensions.index(ext)] += 1

    for ext, count in zip(extensions, counter):
        print(f"Copied {count} files with extension {ext} to {output_folder}")
        
    # count the total number of img files in the output folder:
    img_extensions = ('.jpg', '.jpeg', '.png')
    n_img_files = len([f for f in os.listdir(output_folder) if f.endswith(img_extensions)])
    # append the total number of imgs to the output foldername:
    os.rename(output_folder, output_folder + f"_{n_img_files}_imgs")

'''

cd /home/xander/Projects/cog/CLIP_active_learning_classifier/CLIP_assisted_data_labeling
python 06_create_subset.py --input_dir /data/datasets/midjourney --min_score 0.35

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--min_score', type=float, help='Input directory')
    args = parser.parse_args()

    copy_data(args.input_dir, args.min_score, extensions = ['.jpg', '.txt'])