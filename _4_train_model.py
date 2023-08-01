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
from utils.nn_model import device, SimpleFC
from sklearn.metrics import r2_score

def train(args, crop_names, use_img_stat_features):

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    features = []
    labels = []

    if args.clip_models_to_use[0] != "all":
        print(f"\n----> Using clip models: {args.clip_models_to_use}")

    # Load all the labeled training data from disk:
    for train_data_name in args.train_data_names:
        n_samples = 0
        skips = 0

        # Load the labels and uuid's from labels.csv
        data = pd.read_csv(os.path.join(args.train_data_dir, train_data_name + '.csv'))
        # Drop all the rows where "label" is NaN:
        data = data.dropna(subset=["label"])
        # randomly shuffle the data:
        data = data.sample(frac=1).reset_index(drop=True)

        # Load the feature vectors from disk (uuid.pt)
        print(f"\nLoading {train_data_name} features from disk...")

        for index, row in tqdm(data.iterrows()):
            try:
                uuid = row["uuid"]
                label = row["label"]
                full_feature_dict = torch.load(f"{args.train_data_dir}/{train_data_name}/{uuid}.pt")

                if args.clip_models_to_use[0] == "all":
                    args.clip_models_to_use = list(full_feature_dict.keys())
                    print(f"\n----> Using all found clip models: {args.clip_models_to_use}")

                sample_features = []

                for clip_model_name in args.clip_models_to_use:
                    feature_dict = full_feature_dict[clip_model_name]
                    clip_features = torch.cat([feature_dict[crop_name] for crop_name in crop_names if crop_name in feature_dict], dim=0).flatten()
                    missing_crops = set(crop_names) - set(feature_dict.keys())
                    if missing_crops:
                        raise Exception(f"Missing crops {missing_crops} for {uuid}, either re-embed the image, or adjust the crop_names variable for training!")

                    if use_img_stat_features:
                        img_stat_feature_names = [key for key in feature_dict.keys() if key.startswith("img_stat_")]
                        img_stat_features = torch.stack([feature_dict[img_stat_feature_name] for img_stat_feature_name in img_stat_feature_names], dim=0).to(device)
                        all_features = torch.cat([clip_features, img_stat_features], dim=0)
                    else:
                        all_features = clip_features

                    sample_features.append(all_features)

                features.append(torch.cat(sample_features, dim=0))
                labels.append(label)
                n_samples += 1
            except Exception as e: # simply skip the sample if something goes wrong
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
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # 3. Create the regression network:
    model = SimpleFC(features.shape[1], args.hidden_sizes, 1, args.clip_models_to_use,
                    crop_names   = crop_names,
                    dropout_prob = args.dropout_prob, 
                    verbose      = args.print_network_layout)
    model.train()
    model.to(device)

    # 4. Train the network for n epochs using Adam optimizer and standard regression loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    losses    = [[], []] # train, test losses

    def get_test_loss(model, test_loader, epoch, plot_correlation=1):
        if len(test_loader) == 0:
            return -1.0, -1.0
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

        if plot_correlation and epoch % 5 == 0:
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
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_save_name = f"{args.model_name}_{timestamp}_{(len(train_dataset) / 1000):.1f}k_imgs_{args.n_epochs}_epochs_{losses[1][-1]:.4f}_mse"
        os.makedirs("models", exist_ok=True)
        
        with open(f"models/{model_save_name}.pkl", "wb") as file:
            pickle.dump(model, file)

        print("Final model saved to /model dir as:\n", f"{model_save_name}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # IO args:
    parser.add_argument('--train_data_dir', type=str, help='Root directory of the (optionally multiple) datasets')
    parser.add_argument('--train_data_names', type=str, nargs='+', help='Names of the dataset files to train on (space separated)')
    parser.add_argument('--model_name', type=str, default='regressor', help='Name of the model when saved to disk')
    parser.add_argument('--dont_save', action='store_true', help='skip saving the model to disk')

    # Training args:
    parser.add_argument('--clip_models_to_use', metavar='S', type=str, nargs='+', default=['all'], help='Which CLIP model embeddings to use, default: use all found')
    parser.add_argument('--test_fraction', type=float, default=0.25,  help='Fraction of the training data to use for testing')
    parser.add_argument('--n_epochs',      type=int,   default=80,    help='Number of epochs to train for')
    parser.add_argument('--batch_size',    type=int,   default=32,    help='Batch size for training')
    parser.add_argument('--lr',            type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay',  type=float, default=0.0004, help='Weight decay for the Adam optimizer')
    parser.add_argument('--dropout_prob',  type=float, default=0.5,   help='Dropout probability')
    parser.add_argument('--hidden_sizes',  type=int,   nargs='+',     default=[264,128,64], help='Hidden sizes of the FC neural network')

    parser.add_argument('--print_network_layout', action='store_true', help='Print the network layout')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Custom switches to turn on/off certain features:
    crop_names = ['centre_crop', 'square_padded_crop', 'subcrop1_0.15', 'subcrop2_0.1']  # 0.265
    #crop_names = ['centre_crop',  'subcrop2_0.1']      # 0.27
    #crop_names = ['square_padded_crop', 'subcrop2_0.1']  # 0.275
    #crop_names = ['centre_crop']  # 0.285
    #crop_names = ['square_padded_crop'] # 0.29
    #crop_names = ['subcrop1_0.15'] # 0.30
    #crop_names = ['subcrop2_0.1'] # 0.31

    use_img_stat_features = 0

    train(args, crop_names, use_img_stat_features)