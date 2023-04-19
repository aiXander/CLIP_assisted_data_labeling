import os
import shutil
from tqdm import tqdm
import argparse
import pandas as pd
from PIL import Image

def copy_data(args, output_suffix = '_subset'):
    '''
    Copy all the files in the root_dir based on predicted label
    '''
    
    # Get all the rows where the predicted label is above the threshold:
    database_path = os.path.join(os.path.dirname(args.input_dir), os.path.basename(args.input_dir) + ".csv")
    database = pd.read_csv(database_path)
    print(f"Loaded database with {len(database)} rows")

    # Define a function to apply the filtering criteria
    def filter_rows(row):
        final_label = row["label"] if pd.notnull(row["label"]) else row["predicted_label"]
        return args.min_score <= final_label <= args.max_score

    # Filter the DataFrame using the function
    database = database[database.apply(filter_rows, axis=1)]
    print(f"Found {len(database)} rows with {args.min_score} < final_label < {args.max_score}")

    output_suffix = f'_{args.min_score:.2f}_to_{args.max_score:.2f}' + output_suffix
    output_folder = os.path.join(args.input_dir + output_suffix)

    if args.test:
        print(f"Running script in TEST MODE: Not actually copying any files to {output_folder}")
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Loop over the uuids in the database and copy the corresponding files to the output folder:
    print(f"Copying files to {output_folder}...")
    counter = [0] * len(args.extensions)
    for uuid in tqdm(database["uuid"].values):
        # get the corresponding img path for this uuid:
        img_path = os.path.join(args.input_dir, uuid + ".jpg")
        with Image.open(img_path) as img:
            width, height = img.size
            aspect_ratio = width / height

        # check if the aspect ratio is within the specified range:
        if aspect_ratio < args.min_aspect_ratio or aspect_ratio > args.max_aspect_ratio:
            continue

        for ext in args.extensions:
            filename = uuid + ext
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(output_folder, filename)
            if not args.test:
                shutil.copy(input_path, output_path)
            counter[args.extensions.index(ext)] += 1

    for ext, count in zip(args.extensions, counter):
        print(f"Copied {count} files with extension {ext} to {output_folder}")
        
    if not args.test:
        # count the total number of img files in the output folder:
        img_extensions = ('.jpg', '.jpeg', '.png')
        n_img_files = len([f for f in os.listdir(output_folder) if f.endswith(img_extensions)])
        # append the total number of imgs to the output foldername:
        os.rename(output_folder, output_folder + f"_{n_img_files}_imgs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--min_score', type=float, help='Input directory')
    parser.add_argument('--max_score', type=float, default=1.0, help='Maximum score to consider')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.txt', '.pt'], help='Extensions to consider')
    parser.add_argument('--min_aspect_ratio', type=float, default=0.25, help='Minimum aspect ratio of imgs to consider')
    parser.add_argument('--max_aspect_ratio', type=float, default=4.00, help='Maximum aspect ratio of imgs to consider')
    parser.add_argument('--test', action='store_true', help='Test mode, wont actually copy anything')
    args = parser.parse_args()

    copy_data(args)