import torch
import torch.nn as nn
import torch.nn.functional as F

POOL_SIZE = 2
KERNEL_SIZE = 3
NUM_FILTERS1 = 12
NUM_FILTERS2 = 24
HIDDEN_UNITS1 = 128
HIDDEN_UNITS2 = 256

class CNN(nn.Module):
    def __init__(self, INPUT_SIZE, NUM_CLASSES):
        super(CNN, self).__init__()

        # Define model parameters
        self.INPUT_SIZE = INPUT_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(INPUT_SIZE[0], NUM_FILTERS1, KERNEL_SIZE, padding=1)
        self.pool = nn.MaxPool2d(POOL_SIZE, POOL_SIZE, padding=1)
        self.conv2 = nn.Conv2d(NUM_FILTERS1, NUM_FILTERS2, KERNEL_SIZE, padding=1)

        # Compute the size dynamically
        self._initialize_layers()

    def _initialize_layers(self):
        # Create a dummy input tensor to compute the output size
        channels, x, y = self.INPUT_SIZE
        dummy_input = torch.zeros(1, channels, x, y)
        dummy_output = self._forward_conv(dummy_input)

        # Compute the input size to the fully-connected layer
        num_features = dummy_output.numel()
        self.fc1 = nn.Linear(num_features, HIDDEN_UNITS1)
        self.fc2 = nn.Linear(HIDDEN_UNITS1, HIDDEN_UNITS2)
        self.fc3 = nn.Linear(HIDDEN_UNITS2, self.NUM_CLASSES)

        self.dropout = nn.Dropout(p=0.1)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x
    
    @staticmethod
    def model_name():
        return "model66"

