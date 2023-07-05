import os
import numpy as np
import pandas as pd
import torch, shutil
import pickle, time
import random
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt
import json
from utils.nn_model import device, SimpleFC
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

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

def find_model(model_name, model_dir = "models"):
    if os.path.exists(model_name) and os.path.isfile(model_name) and model_name.endswith(".pkl"):
        return model_name
    model_files = os.listdir(model_dir)
    for model_file in model_files:
        if model_name in model_file:
            # return the absolute path to the model file:
            return os.path.join(model_dir, model_file)
    return None


class CustomDataset(Dataset):
    def __init__(self, uuids, model, args):
        self.uuids = uuids
        self.model = model # prediction model
        self.args = args
        self.feature_shape = None

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        img_path     = os.path.join(self.args.root_dir, uuid + '.jpg')
        feature_path = os.path.join(self.args.root_dir, uuid + '.pt')

        try:
            full_feature_dict = torch.load(feature_path)
            sample_features = []
            for clip_model_name in self.args.clip_models:
                feature_dict = full_feature_dict[clip_model_name]
                clip_features = torch.cat([feature_dict[crop_name] for crop_name in self.model.crop_names if crop_name in feature_dict], dim=0).flatten()
                sample_features.append(clip_features)

            img_features = torch.cat(sample_features, dim=0).flatten()
            self.feature_shape = img_features.shape
        except Exception as e:
            print(f"WARNING: {e} for {uuid}, skipping this sample..")
            return "", "", torch.zeros(self.feature_shape, device=device)

        return uuid, img_path, img_features

@torch.no_grad()
def predict_labels(args):

    model_file = find_model(args.model_file)
    if model_file is None:
        print(f"ERROR: could not find model file {model_file}!")
        exit()

    output_dir = args.root_dir + '_predicted_scores'
    os.makedirs(output_dir, exist_ok=True)

    with open(model_file, "rb") as file:
        model = pickle.load(file)
        model.eval()
        args.clip_models = model.clip_models
        print("Loaded regression model trained on the following CLIP models:")
        print(args.clip_models)

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

    # Get all *.jpg files in the input_directory:
    img_files = [os.path.splitext(f)[0] for f in os.listdir(args.root_dir) if f.endswith('.jpg')]
    dataset = CustomDataset(img_files, model, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, prefetch_factor=2)
    print(f"Predicting labels for {len(dataset)} images...")

    n_predictions = 0

    for uuids, img_paths, features in tqdm(dataloader):
        uuids = list(uuids)
        predicted_labels = model(features.to(device).float()).cpu().numpy().squeeze()

        # filter out samples that have an empty string as uuid:
        remove_indices   = [i for i, uuid in enumerate(uuids) if uuid == ""]
        uuids            = [uuid for i, uuid in enumerate(uuids) if i not in remove_indices]
        img_paths        = [img_path for i, img_path in enumerate(img_paths) if i not in remove_indices]
        predicted_labels = np.delete(predicted_labels, remove_indices)

        # Create a DataFrame for the current batch
        current_timestamp = np.full_like(predicted_labels, int(time.time()))
        batch_data = pd.DataFrame({'uuid': uuids, 'predicted_label': predicted_labels, 'timestamp': current_timestamp})

        # Merge the 'predicted_label' and 'timestamp' columns from batch_data into the database based on the 'uuid' column
        database = database.merge(batch_data[['uuid', 'predicted_label', 'timestamp']], on='uuid', how='outer', suffixes=('', '_new'))

        # Update the 'predicted_label' and 'timestamp' columns in the database with the new values
        database['predicted_label'] = database['predicted_label_new'].where(database['predicted_label_new'].notna(), database['predicted_label'])
        database['timestamp'] = database['timestamp_new'].where(database['timestamp_new'].notna(), database['timestamp'])

        # Drop the temporary columns created during the merge
        database.drop(columns=['predicted_label_new', 'timestamp_new'], inplace=True)
        n_predictions += len(uuids)

        if args.copy_imgs_fraction > 0: # copy a random fraction of the images to the output directory
            indices = np.arange(len(uuids))
            random_indices = indices[np.random.random(len(uuids)) < args.copy_imgs_fraction]
            src_paths = [img_paths[i] for i in random_indices]
            dst_paths = [f"{predicted_labels[i]:.3f}_{uuids[i]}.jpg" for i in random_indices]

            for src, dst in zip(src_paths, dst_paths):
                shutil.copy(src, os.path.join(output_dir, dst))

        if n_predictions % 100 == 0:
            database.to_csv(label_file, index=False)

    database.to_csv(label_file, index=False)
    plot_label_distribution(database, args)

    print("Done!")
    print(f"{n_predictions} of {len(img_files)} img predicted. (the rest was skipped due to errors)")
    print(f"Average predicted label: {database['predicted_label'].mean():.3f}")
    print(f"Database saved at {label_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',   type=str, help='Root directory of the dataset')
    parser.add_argument('--model_file', type=str, help='Path to the model file (.pkl)')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for predicting')
    parser.add_argument('--copy_imgs_fraction', type=float, default=0.01, help='Fraction of images to copy to tmp_output directory with prepended prediction score')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for the dataloader')
    args = parser.parse_args()

    mp.set_start_method('spawn')

    # recursively apply the model to all subdirectories:
    for root, dirs, files in os.walk(args.root_dir):
        jpg_files = [f for f in files if f.endswith('.jpg')]

        if len(jpg_files) > 0 and "_predicted_scores" not in root:
            args.root_dir = root
            print(f"\n\nPredicting labels for {root}...")
            predict_labels(args)