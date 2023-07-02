import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, 
                 crop_names = ['centre_crop', 'square_padded_crop', 'subcrop1', 'subcrop2'],
                 use_img_stat_features = False,
                 dropout_prob=0.0, 
                 data_min = None, data_max = None,
                 verbose = 0):
        
        super(SimpleFC, self).__init__()
        self.crop_names = crop_names
        self.use_img_stat_features = use_img_stat_features
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.data_min, self.data_max = data_min, data_max

        # Define the network:
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # add ReLU and Dropout after each layer except the last one
            if i < len(layer_sizes) - 2:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=dropout_prob))

        # Add sigmoid at the end: (assumes that all labels are normalized in the range [0,1])
        layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(layers)

        if verbose > 0: # Print the final network layout:
            print(self)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class SimpleconvFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, 
                 crop_names = ['centre_crop', 'square_padded_crop', 'subcrop1', 'subcrop2'],
                 use_img_stat_features = False,
                 dropout_prob=0.0, 
                 data_min = None, data_max = None,
                 verbose = 0,
                 conv_out_channels = 64, # added new parameter for conv output channels
                 kernel_size = 5): # added new parameter for kernel size
                 
        super(SimpleconvFC, self).__init__()
        self.crop_names = crop_names
        self.use_img_stat_features = use_img_stat_features
        self.data_min, self.data_max = data_min, data_max

        # Define the 1D Convolutional layer:
        input_size = 768*2
        self.conv1 = nn.Conv1d(input_size, conv_out_channels, kernel_size)

        # Adjust input size for FC layers based on conv output:
        layer_sizes = [4672] + hidden_sizes + [output_size]

        # Define the network:
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # add ReLU and Dropout after each layer except the last one
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_prob))

        # Add sigmoid at the end: (assumes that all labels are normalized in the range [0,1])
        layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(layers)

        if verbose > 0: # Print the final network layout:
            print(self)

    def forward(self, x, verbose = 0):
        #x = x.view(x.size(0), 2, 77, 768)

        # Reshape to make 77 the last (temporal dimension), and concatenate the 2 channels (c and uc): into 2*768 features:
        x = x.permute(0, 1, 3, 2).reshape(x.size(0), 2*768, 77)

        if verbose:
            print("Pre conv:")
            print(x.shape)

        x = self.conv1(x)

        if verbose:
            print("Post conv, pre flatten:")
            print(x.shape)

        x = x.view(x.size(0), -1)  # flatten the tensor

        if verbose:
            print("Post conv, post flatten:")
            print(x.shape)

        for layer in self.layers:
            x = layer(x)
        return x