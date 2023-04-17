import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
from nn_model import device, SimpleFC

def train(args, crop_names):

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    features = []
    labels = []

    # Load all the labeled training data from disk:
    for train_data_name in args.train_data_names:
        n_samples = 0

        # Load the labels and uuid's from labels.csv
        data = pd.read_csv(os.path.join(args.train_data_dir, train_data_name + '.csv'))
        # Drop all the rows where "label" is NaN:
        data = data.dropna(subset=["label"])
        # randomly shuffle the data:
        data = data.sample(frac=1).reset_index(drop=True)

        # Load the feature vectors from disk (uuid.npy)
        for index, row in data.iterrows():
            try:
                uuid = row["uuid"]
                label = row["label"]
                feature_dict = torch.load(f"{args.train_data_dir}/{train_data_name}/{uuid}.pt")
                img_features = torch.cat([feature_dict[crop_name] for crop_name in crop_names if crop_name in feature_dict], dim=0).flatten()
                missing_crops = set(crop_names) - set(feature_dict.keys())
                if missing_crops:
                    raise Exception(f"Missing crops {missing_crops} for {uuid}, either re-embed the image, or adjust the crop_names variable for training!")

                features.append(img_features)
                labels.append(label)
                n_samples += 1
            except Exception as e:
                continue

        print(f"Loaded {n_samples} samples from {train_data_name}.")

    features = torch.stack(features, dim=0).to(device).float()
    labels = torch.tensor(labels).to(device).float()

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
    input_size = features.shape[1]
    output_size = 1
    model = SimpleFC(input_size, args.hidden_sizes, output_size, 
                    crop_names = crop_names,
                    dropout_prob=args.dropout_prob, 
                    verbose = args.print_network_layout)
    model.train()
    model.to(device)

    # 4. Train the network for n epochs using Adam optimizer and standard regression loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    losses    = []

    def get_test_loss(model, test_loader, epoch):
        if len(test_loader) == 0:
            return 0.0
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}, Test MSE Loss: {test_loss:.4f}")
        return test_loss

    get_test_loss(model, test_loader, -1)

    for epoch in range(args.n_epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        losses.append(get_test_loss(model, test_loader, epoch))

    # Report:
    print(f"Best test mse loss: {min(losses):.4f} in epoch {np.argmin(losses)+1}")
    plt.plot(losses)
    plt.savefig("losses.png")

    if not args.dont_save: # Save the model
        n_train = len(train_dataset)
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_save_name = f"{args.model_name}_{timestamp}_{n_train}_{args.n_epochs}_{losses[-1]:.4f}"
        os.makedirs("models", exist_ok=True)

        #torch.save(model.state_dict(), f"models/{model_save_name}.pt")
        import pickle
        with open(f"models/{model_save_name}.pkl", "wb") as file:
            pickle.dump(model, file)

        print("Model saved as:\n", model_save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # IO args:
    parser.add_argument('--train_data_dir', type=str, help='Root directory of the (optionally multiple) datasets')
    parser.add_argument('--train_data_names', type=str, nargs='+', help='Names of the dataset files to train on (space separated)')
    parser.add_argument('--model_name', type=str, default='regressor', help='Name of the model when saved to disk')
    parser.add_argument('--dont_save', action='store_true', help='Force CLIP re-encoding of all images (default: False)')

    # Training args:
    parser.add_argument('--test_fraction', type=float, default=0.05,  help='Fraction of the training data to use for testing')
    parser.add_argument('--n_epochs',      type=int,   default=30,    help='Number of epochs to train for')
    parser.add_argument('--batch_size',    type=int,   default=128,   help='Batch size for training')
    parser.add_argument('--lr',            type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay',  type=float, default=0.001, help='Weight decay for the Adam optimizer (default: 0.001)')
    parser.add_argument('--dropout_prob',  type=float, default=0.5,   help='Dropout probability')
    parser.add_argument('--hidden_sizes',  type=int,   nargs='+',     default=[264,128,64], help='Hidden sizes of the FC neural network')

    parser.add_argument('--print_network_layout', action='store_true', help='Print the network layout')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    crop_names = ['centre_crop', 'square_padded_crop', 'subcrop1_0.15', 'subcrop2_0.1']
    train(args, crop_names)