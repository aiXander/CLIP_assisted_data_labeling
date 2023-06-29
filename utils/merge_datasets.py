import os
import pandas as pd
import shutil

"""

cd /home/rednax/SSD2TB/Fast_Datasets/SD/Labeling
python3 merge_datasets.py

merge all datasets in the data_dir (all subfolders and their corresponding .csv files) into two datasets: labeled and unlabeled

"""


# Constants
data_dir   = "/home/rednax/SSD2TB/Fast_Datasets/SD/Labeling/test"
output_dir = "/home/rednax/SSD2TB/Fast_Datasets/SD/Labeling/merged"

labeled_dir   = os.path.join(output_dir, "labeled")
unlabeled_dir = os.path.join(output_dir, "unlabeled")

# Create the output directories if they don't exist
os.makedirs(labeled_dir, exist_ok=True)
os.makedirs(unlabeled_dir, exist_ok=True)

# List of dataframes, one per csv file
dfs = []

# Iterate over the subdirectories
for subdir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir)

    if os.path.isdir(subdir_path):
        # Load the csv file associated with the subdirectory
        csv_path = os.path.join(data_dir, f"{subdir}.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Add a column with the name of the subdirectory:
            df['source_datadir'] = subdir
            dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs)

# Split into labeled and unlabeled
labeled_df   = combined_df[combined_df['label'].notna()]
unlabeled_df = combined_df[combined_df['label'].isna()]

# Save as CSV files
labeled_df.to_csv(os.path.join(output_dir, "labeled.csv"), index=False)
unlabeled_df.to_csv(os.path.join(output_dir, "unlabeled.csv"), index=False)

# Function to move files based on uuids
def move_files(df, source_dir, destination_dir, extensions_to_move = ['.jpg', '.json', '.txt', '.pt', '.pth']):
    moved = 0
    uuids = df['uuid'].values
    source_dirs = df['source_datadir'].values

    for i, uuid in enumerate(uuids):
        for extension in extensions_to_move:
            # Here we assume that the file extension is .jpg, change it as per your dataset
            
            source_file = os.path.join(source_dir, source_dirs[i], f"{uuid}{extension}")

            if os.path.exists(source_file):
                destination_file = os.path.join(destination_dir, f"{uuid}{extension}")
                shutil.move(source_file, destination_file)
                moved += 1

    print(f"Moved {moved} files from {source_dir} to {destination_dir}!")

# Move labeled and unlabeled files to their respective directories
move_files(labeled_df, data_dir, labeled_dir)
move_files(unlabeled_df, data_dir, unlabeled_dir)
