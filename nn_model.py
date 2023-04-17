import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.0, verbose = 0):
        super(SimpleFC, self).__init__()

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # add ReLU after each layer except the last one
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_prob))

        self.layers = nn.ModuleList(layers)

        if verbose > 0:
            # Print the final network layout:
            print(self)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x