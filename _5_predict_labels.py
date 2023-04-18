import os
import numpy as np
import pandas as pd
import torch, shutil
import pickle, time
import random
from tqdm import tqdm
from nn_model import device, SimpleFC
import argparse

def find_model(model_name, model_dir = "models"):
    model_files = os.listdir(model_dir)
    for model_file in model_files:
        if model_name in model_file:
            # return the absolute path to the model file:
            return os.path.join(model_dir, model_file)
    return None

def predict(features, paths, uuids, database, row, model, output_dir, args):
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
                index_to_update = row.index[0]
                # Update the values in the row
                database.loc[index_to_update, 'predicted_label'] = predicted_score
                database.loc[index_to_update, 'timestamp'] = current_timestamp

            if random.random() < args.copy_imgs_fraction:
                # Copy the image to the output directory:
                img_path = paths[i]
                outpath = output_dir + f"/{predicted_score:.3f}_{uuid}.jpg"
                shutil.copy(img_path, outpath)

        return database


from matplotlib import pyplot as plt
def plot_label_distribution(database, args, max_x = 0.6):
    # Save a plot of the label distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    n, bins, patches = ax.hist(database['predicted_label'].values, bins=100, alpha=0.75, color='blue', edgecolor='black')

    # Customize the plot appearance
    ax.set_title(f'Label Distribution for {os.path.basename(args.root_dir)}', fontsize=18)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(axis='y', alpha=0.75, linestyle='--')

    # Add a text box with mean and standard deviation
    mu = np.mean(database['predicted_label'].values)
    sigma = np.std(database['predicted_label'].values)
    textstr = f'$\mu={mu:.2f}$\n$\sigma={sigma:.2f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # Set a custom y-axis tick format
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))

    # Set the x-axis range
    ax.set_xlim(left=0, right=max_x)

    # Save the plot in the parent dir of args.root_dir
    output_dir = os.path.dirname(args.root_dir)
    plt.savefig(os.path.join(output_dir, f"label_distribution_{os.path.basename(args.root_dir)}.png"))
    plt.close()

def predict_labels(args):

    model_file = find_model(args.model_file)
    if model_file is None:
        print(f"ERROR: could not find model file {model_file}!")
        exit()

    output_dir = args.root_dir + '_predicted_scores'
    os.makedirs(output_dir, exist_ok=True)

    with open(model_file, "rb") as file:
        model = pickle.load(file)

    label_file = os.path.join(os.path.dirname(args.root_dir), os.path.basename(args.root_dir) + ".csv")
    if os.path.exists(label_file):
        database = pd.read_csv(label_file)
        print(f"Loaded existing database: {label_file}.\nDatabase contains {len(database)} entries")
    else:
        database = pd.DataFrame(columns=["uuid", "label", "timestamp", "predicted_label"])
        print(f"Created new database file at {label_file}")

    # add new column 'predicted_label' to the database if not yet present:
    if 'predicted_label' not in database.columns:
        database['predicted_label'] = np.nan

    # Loop over all *.jpg files in the input_directory that are not yet part of the labeled dataset:
    img_files = [f.split('.')[0] for f in os.listdir(args.root_dir) if f.endswith('.jpg')]
    print(f"Predicting labels for {len(img_files)} images...")

    features, paths, uuids = [], [], []
    n_predictions, n_skips, already_labeled = 0,0,0

    for uuid in tqdm(img_files):
        # Get the row in the database that corresponds to this uuid:
        row = database.loc[database["uuid"] == uuid]

        try: #if this uuid is already part of the labeled dataset, just skip
            if row['label'].values[0] is not None and not np.isnan(row['label'].values[0]):
                already_labeled += 1
                continue
        except:
            pass

        img_path = os.path.join(args.root_dir, uuid + '.jpg')
        feature_path = os.path.join(args.root_dir, uuid + '.pt')

        if not os.path.exists(feature_path): #if there's no CLIP embbeding, just skip
            n_skips += 1
            continue

        feature_dict = torch.load(feature_path)
        try:
            img_features = [feature_dict[crop_name] for crop_name in model.crop_names if crop_name in feature_dict.keys()]
            img_features = torch.stack(img_features, dim=0)
        except Exception as e:
            print(f"WARNING: {e} for {uuid}, skipping this sample..")
            continue

        n_features = img_features.shape[0]
        if n_features != len(model.crop_names):
            print(f"WARNING: {n_features} crop features found for {uuid} (expected {len(model.crop_names)}), skipping this sample...")
            continue

        features.append(img_features.flatten().to(device).float())
        paths.append(img_path)
        uuids.append(uuid)

        if len(paths) == args.batch_size:
            features = torch.stack(features, dim=0).to(device).float()
            database = predict(features, paths, uuids, database, row, model, output_dir, args)
            n_predictions += args.batch_size
            features, paths, uuids = [], [], []
        if n_predictions % 100 == 0:
            database.to_csv(label_file, index=False)

    database.to_csv(label_file, index=False)
    plot_label_distribution(database, args)

    print("Done!")
    print(f"{n_predictions} of {len(img_files)} img predicted ({already_labeled} already labeled, {n_skips} skipped due to no CLIP embbeding found on disk).")
    print(f"Database saved at {label_file}")


if __name__ == "__main__":

    """

    TODO add a dataloader class that loads the images from disk on the fly / multi-threaded
    
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',   type=str, help='Root directory of the dataset')
    parser.add_argument('--model_file', type=str, help='Path to the model file (.pkl)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for predicting')
    parser.add_argument('--copy_imgs_fraction', type=float, default=0.01, help='Fraction of images to copy to tmp_output directory with prepended prediction score')
    args = parser.parse_args()

    # recursively apply the model to all subdirectories:
    for root, dirs, files in os.walk(args.root_dir):
        jpg_files = [f for f in files if f.endswith('.jpg')]

        if len(jpg_files) > 0 and "_predicted_scores" not in root:
            args.root_dir = root
            print(f"\n\nPredicting labels for {root}...")
            predict_labels(args)