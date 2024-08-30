import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
pool_size = 2
kernel_size = 3  # Smaller kernel size for better feature extraction

conv1_out = 64
conv2_out = 128
conv3_out = 256
conv4_out = 512  # Adding an additional convolutional layer
linear1 = 1024  # Increasing linear layer size for more capacity
linear2 = 512
dropout_prob = 0.5  # Increased dropout for regularization

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(input_size[0], conv1_out, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out)

        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out)

        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_out)
        
        self.conv4 = nn.Conv2d(conv3_out, conv4_out, kernel_size, padding=1)  # Additional layer
        self.bn4 = nn.BatchNorm2d(conv4_out)

        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.relu = nn.ReLU()

        self._initialize_layers()

    def _initialize_layers(self):
        channels, x, y = self.input_size
        dummy_input = torch.zeros(1, channels, x, y)
        dummy_output = self._forward_conv(dummy_input)

        num_features = dummy_output.numel()
        self.fc1 = nn.Linear(num_features, linear1)
        self.fc2 = nn.Linear(linear1, linear2)
        self.fc3 = nn.Linear(linear2, self.num_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def _forward_conv(self, x):
        # Convolutional layers with Batch Norm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn4(self.conv4(x)))  # Additional layer
        x = self.pool(x)

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
    
    @staticmethod
    def model_name():
        return "cnnBatchNorm90"

