import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam

from nn_model import device, SimpleFC

"""

cd /home/xander/Projects/cog/CLIP_active_learning_classifier/CLIP_assisted_data_labeling
python 04_train_model.py



"""

train_data_dir = '/home/xander/Pictures/datasets'
train_data_name = 'mj_filtered_uuid'
batch_size = 4
lr = 0.0001
weight_decay = 0.001
hidden_sizes = [128,32]

test_fraction = 0.05
n_epochs = 10

# Fix all random seeds for reproducibility:
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# 1. Load the labels and uuid's from labels.csv
data = pd.read_csv(os.path.join(train_data_dir, train_data_name + '.csv'))

# Drop all the rows where "label" is NaN:
data = data.dropna(subset=["label"])

# randomly shuffle the data:
data = data.sample(frac=1).reset_index(drop=True)

# 2. Load the feature vectors from disk (uuid.npy)
features = []
labels = []

for index, row in data.iterrows():
    uuid = row["uuid"]
    label = row["label"]
    feature = torch.load(f"{train_data_dir}/{train_data_name}/{uuid}.pt").flatten()
    features.append(feature)
    labels.append(label)

features = torch.stack(features, dim=0).to(device).float()
labels = torch.tensor(labels).to(device).float()

# 3. Create train and test dataloaders
class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = RegressionDataset(features, labels)
train_size = int((1-test_fraction) * len(dataset))
test_size = len(dataset) - train_size

print(f"Training on {train_size} samples, testing on {test_size} samples.")

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. Create the network
input_size = features.shape[1]
output_size = 1
model = SimpleFC(input_size, hidden_sizes, output_size)
model.train()
model.to(device)

# 5. Train the network for n epochs using Adam optimizer and standard regression loss, print test loss
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

losses = []

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
    print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")
    return test_loss

get_test_loss(model, test_loader, -1)

for epoch in range(n_epochs):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    losses.append(get_test_loss(model, test_loader, epoch))

print(f"Best test loss: {min(losses):.4f} in epoch {np.argmin(losses)+1}")

# print a graph of the losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.savefig("losses.png")

# Save the model
n_train = len(train_dataset)
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H:%M:%S")
model_name = f"{timestamp}_{n_train}_{n_epochs}_{losses[-1]:.4f}"
torch.save(model.state_dict(), f"models/{model_name}.pt")

import pickle
# Save the entire neural network model to disk
with open(f"models/{model_name}.pkl", "wb") as file:
    pickle.dump(model, file)