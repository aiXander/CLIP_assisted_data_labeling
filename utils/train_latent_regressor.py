import os
import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
from nn_model import device, SimpleFC, SimpleconvFC
from sklearn.metrics import r2_score

"""
This is an unfinished exerpiment:
given a dataset of StableDiffusion prompt_embeds / aesthetic scores, try to learn a mapping from prompt_embeds to scores
The idea here is to use this regressor to do prompt-augmentation in latent space.

cd /home/rednax/SSD2TB/Xander_Tools/CLIP_assisted_data_labeling/utils
python train_latent_regressor.py --train_data_dir /home/rednax/SSD2TB/Github_repos/cog/eden-sd-pipelines/eden/xander/images/random_c_uc_fin_dataset --train_data_names no_lora eden eden2 --model_name c_uc_regressor --test_fraction 0.4 --dont_save


"""

def train(args):

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    features = []
    labels = []

    # Load all the labeled training data from disk:
    for train_data_name in args.train_data_names:
        n_samples = 0
        skips = 0

        # Load the labels and uuid's from labels.csv
        data = pd.read_csv(os.path.join(args.train_data_dir, train_data_name + '.csv'))
        # Drop all the rows where "label" is NaN:
        #data = data.dropna(subset=["label"])
        # randomly shuffle the data:
        data = data.sample(frac=1).reset_index(drop=True)

        # Load the prompt_embeds from disk (uuid.pth)
        print(f"\nLoading {train_data_name} features from disk...")
        for index, row in tqdm(data.iterrows()):
            try:
                uuid = row["uuid"]
                # load row["label"] is it exists, otherwise load row["predicted_label"]
                label = row["label"] if not np.isnan(row["label"]) else row["predicted_label"]*0.5
                prompt_embeds = torch.load(f"{args.train_data_dir}/{train_data_name}/{uuid}.pth")
                features.append(prompt_embeds.flatten())
                labels.append(label)
                n_samples += 1
            except: # simply skip the sample if something goes wrong
                skips += 1
                continue

        print(f"Loaded {n_samples} samples from {train_data_name}!")
        if skips > 0:
            print(f"(skipped {skips} samples due to loading errors)..")

    features = torch.stack(features, dim=0).to(device).float()
    labels = torch.tensor(labels).to(device).float()

    # Map the labels to 0-1:
    print("Normalizing labels to [0,1]...")
    labels_min, labels_max = labels.min(), labels.max()
    labels = (labels - labels_min) / (labels_max - labels_min)

    print("\n--- All data loaded ---")
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # 2. Create train and test dataloaders
    class RegressionDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    dataset    = RegressionDataset(features, labels)
    train_size = int((1-args.test_fraction) * len(dataset))
    test_size  = len(dataset) - train_size

    print(f"Training on {train_size} samples, testing on {test_size} samples.")

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Create the network
    model = SimpleFC(features.shape[1], args.hidden_sizes, 1, 
                    dropout_prob=args.dropout_prob, 
                    verbose = args.print_network_layout,
                    data_min = labels_min, data_max = labels_max)
    model.train()
    model.to(device)

    # 4. Train the network for n epochs using Adam optimizer and standard regression loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    losses    = [[], []] # train, test losses

    def get_test_loss(model, test_loader, epoch, plot_correlation = 1):
        if len(test_loader) == 0:
            return -1.0, -1
        model.eval()
        test_loss, dummy_test_loss = 0.0, 0.0
        test_preds, test_labels = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                test_loss += loss.item()

                dummy_outputs = torch.ones_like(outputs) * labels.mean()
                dummy_loss = criterion(dummy_outputs.squeeze(), labels)
                dummy_test_loss += dummy_loss.item()

                if plot_correlation:
                    test_preds.append(outputs.cpu().numpy())
                    test_labels.append(labels.cpu().numpy())

        if plot_correlation and epoch % 10 == 0:
            test_preds = np.concatenate(test_preds, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)
            plt.figure(figsize=(8, 8))
            plt.scatter(test_labels, test_preds, alpha=0.1)
            plt.xlabel("True labels")
            plt.ylabel("Predicted labels")
            plt.plot([0, 1], [0, 1], color='r', linestyle='--')
            plt.title(f"Epoch {epoch}, r² = {r2_score(test_labels, test_preds):.3f}")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig("test_set_predictions.png")
            plt.close()

        test_loss /= len(test_loader)
        dummy_test_loss /= len(test_loader)
        model.train()
        return test_loss, dummy_test_loss
    
    def plot_losses(losses, y_axis_percentile_cutoff = 99.75, include_y_zero = 1):
        plt.figure(figsize=(16, 8))
        plt.plot(losses[0], label="Train")
        plt.plot(losses[1], label="Test")
        plt.axhline(y=min(losses[1]), color='r', linestyle='--', label="Best test loss")
        all_losses = losses[0] + losses[1]
        if include_y_zero:
            plt.ylim(0, np.percentile(all_losses, y_axis_percentile_cutoff))
        else:
            plt.ylim(np.min(all_losses), np.percentile(all_losses, y_axis_percentile_cutoff))
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig("losses.png")
        plt.close()

    test_loss, dummy_test_loss = get_test_loss(model, test_loader, -1)
    print(f"\nBefore training, test mse-loss: {test_loss:.4f} (dummy: {dummy_test_loss:.4f})")

    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        test_loss, dummy_test_loss = get_test_loss(model, test_loader, epoch)
        losses[0].append(train_loss)
        losses[1].append(test_loss)
        if epoch % 2 == 0:
            test_str = f", test mse: {test_loss:.4f} (dummy: {dummy_test_loss:.4f})" if test_loss > 0 else ""
            print(f"Epoch {epoch+1} / {args.n_epochs}, train-mse: {train_loss:.4f}{test_str}")
        if epoch % (args.n_epochs // 10) == 0:
            plot_losses(losses)

    # Report:
    if test_loss > 0:
        print(f"---> Best test mse loss: {min(losses[1]):.4f} in epoch {np.argmin(losses[1])+1}")
    plot_losses(losses)

    if not args.dont_save: # Save the model
        model.eval()
        n_train = len(train_dataset) / 1000
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_save_name = f"{args.model_name}_{timestamp}_{n_train:.1f}k_imgs_{args.n_epochs}_epochs_{losses[1][-1]:.4f}_mse"
        os.makedirs("models", exist_ok=True)
        
        with open(f"models/{model_save_name}.pkl", "wb") as file:
            pickle.dump(model, file)

        print("Final model saved as:\n", f"models/{model_save_name}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # IO args:
    parser.add_argument('--train_data_dir', type=str, help='Root directory of the (optionally multiple) datasets')
    parser.add_argument('--train_data_names', type=str, nargs='+', help='Names of the dataset files to train on (space separated)')
    parser.add_argument('--model_name', type=str, default='regressor', help='Name of the model when saved to disk')
    parser.add_argument('--dont_save', action='store_true', help='dont save the model to disk')

    # Training args:
    parser.add_argument('--test_fraction', type=float, default=0.25,   help='Fraction of the training data to use for testing')
    parser.add_argument('--n_epochs',      type=int,   default=80,    help='Number of epochs to train for')
    parser.add_argument('--batch_size',    type=int,   default=32,     help='Batch size for training')
    parser.add_argument('--lr',            type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay',  type=float, default=0.0005, help='Weight decay for the Adam optimizer (default: 0.001)')
    parser.add_argument('--dropout_prob',  type=float, default=0.5,    help='Dropout probability')
    parser.add_argument('--hidden_sizes',  type=int,   nargs='+',      default=[128,128,64], help='Hidden sizes of the FC neural network')

    parser.add_argument('--print_network_layout', action='store_true', help='Print the network layout')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    train(args)