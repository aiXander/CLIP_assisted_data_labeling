import os
import numpy as np
import pandas as pd
import torch, shutil
import pickle, time
import random
from tqdm import tqdm
from nn_model import device, SimpleFC
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir',   type=str, help='Root directory of the dataset')
parser.add_argument('--model_file', type=str, help='Path to the model file (.pkl)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for predicting')
parser.add_argument('--copy_imgs_fraction', type=float, default=0.01, help='Fraction of images to copy to tmp_output directory with prepended prediction score')
args = parser.parse_args()

############################################################################
############################################################################

def find_model(model_name, model_dir = "models"):
    model_files = os.listdir(model_dir)
    for model_file in model_files:
        if model_name in model_file:
            # return the absolute path to the model file:
            return os.path.join(model_dir, model_file)
    return None

args.model_file = find_model(args.model_file)
if args.model_file is None:
    print(f"ERROR: could not find model file {args.model_file}!")
    exit()

output_dir = args.root_dir + '_predicted_scores'
os.makedirs(output_dir, exist_ok=True)


with open(args.model_file, "rb") as file:
    model = pickle.load(file)

def predict(features, paths, uuids, database, row):
    output = model(features).detach().cpu().numpy()

    for i in range(len(output)):
        predicted_score = output[i][0]
        uuid = uuids[i]

        current_timestamp = int(time.time())
        if row is None or len(row) == 0:
            # Create a new entry in the database:
            new_row = {"uuid": uuid, "timestamp": current_timestamp, "predicted_label": predicted_score}
            database = pd.concat([database, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Update the existing entry:
            index_to_update = database.loc[database['uuid'] == uuid]
            
            if len(index_to_update) == 0:
                print(f"ERROR: could not find uuid {uuid} in database!")
                continue
            else:
                index_to_update = index_to_update.index[0]
            # Update the values in the row
            database.loc[index_to_update, 'predicted_label'] = predicted_score
            database.loc[index_to_update, 'timestamp'] = current_timestamp

        if random.random() < args.copy_imgs_fraction:

            # Copy the image to the output directory:
            img_path = paths[i]
            outpath = output_dir + f"/{predicted_score:.3f}_{uuid}.jpg"
            shutil.copy(img_path, outpath)

    return database

label_file = os.path.join(os.path.dirname(args.root_dir), os.path.basename(args.root_dir) + ".csv")
if os.path.exists(label_file):
    database = pd.read_csv(label_file)
    print(f"Loaded existing database: {label_file}.\nDatabase contains {len(database)} entries")
else:
    database = pd.DataFrame(columns=["uuid", "label", "timestamp"])
    print(f"Created new database file at {label_file}")

# add new column 'predicted_label' to the database if not yet present:
if 'predicted_label' not in database.columns:
    database['predicted_label'] = np.nan

# Loop over all *.jpg files in the input_directory that are not yet part of the labeled dataset:
img_files = [f.split('.')[0] for f in os.listdir(args.root_dir) if f.endswith('.jpg')]
print(f"Predicting labels for {len(img_files)} images...")

features, paths, uuids = [], [], []
n_predictions, n_skips = 0,0

for uuid in tqdm(img_files):
    # Get the row in the database that corresponds to this uuid:
    row = database.loc[database["uuid"] == uuid]

    try: #if this uuid is already part of the labeled dataset, just skip
        if row['label'].values[0] is not None and not np.isnan(row['label'].values[0]):
            continue
    except:
        pass

    img_path = os.path.join(args.root_dir, uuid + '.jpg')
    feature_path = os.path.join(args.root_dir, uuid + '.pt')

    if not os.path.exists(feature_path): #if there's no CLIP embbeding, just skip
        n_skips += 1
        continue

    feature_vector = torch.load(feature_path).flatten().to(device).float()
    features.append(feature_vector)
    paths.append(img_path)
    uuids.append(uuid)

    if len(paths) == args.batch_size:
        features = torch.stack(features, dim=0).to(device).float()
        database = predict(features, paths, uuids, database, row)
        n_predictions += args.batch_size
        features, paths, uuids = [], [], []
    if n_predictions % 100 == 0:
        database.to_csv(label_file, index=False)

database.to_csv(label_file, index=False)
print("Done!")
print(f"{n_predictions} of {len(img_files)} images predicted ({n_skips} skipped due to no CLIP embbeding found on disk).")
print(f"Database saved at {label_file}")

