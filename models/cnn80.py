import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
pool_size = 2
kernel_size = 4
num_filters1 = 64  # Increased number of filters
num_filters2 = 128  # Increased number of filters
num_filters3 = 256  # Added an additional convolutional layer
hidden_units1 = 512  # Increased hidden units
hidden_units2 = 256  # Reduced hidden units to balance
dropout_prob = 0.35  # Increased dropout for regularization

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()

        # Define model parameters
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_size[0], num_filters1, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(num_filters2, num_filters3, kernel_size, padding=1)  # Added

        self.pool = nn.MaxPool2d(pool_size, pool_size)

        # Compute the size dynamically
        self._initialize_layers()

    def _initialize_layers(self):
        # Create a dummy input tensor to compute the output size
        channels, x, y = self.input_size
        dummy_input = torch.zeros(1, channels, x, y)
        dummy_output = self._forward_conv(dummy_input)

        # Compute the input size to the fully-connected layer
        num_features = dummy_output.numel()
        self.fc1 = nn.Linear(num_features, hidden_units1)
        self.fc2 = nn.Linear(hidden_units1, hidden_units2)
        self.fc3 = nn.Linear(hidden_units2, self.num_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Added
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)  # Apply dropout
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x
    
    @staticmethod
    def model_name():
        return "cnn80"
