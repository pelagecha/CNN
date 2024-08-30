import torch
import torch.nn as nn
import torch.nn.functional as F

POOL_SIZE = 2
KERNEL_SIZE = 4
NUM_FILTERS1 = 64
NUM_FILTERS2 = 128
NUM_FILTERS3 = 256
HIDDEN_UNITS1 = 512
HIDDEN_UNITS2 = 256
DROPOUT_PROB = 0.35

class Model(nn.Module):
    def __init__(self, INPUT_SIZE, NUM_CLASSES):
        super(Model, self).__init__()

        self.INPUT_SIZE = INPUT_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(INPUT_SIZE[0], NUM_FILTERS1, KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(NUM_FILTERS1)

        self.conv2 = nn.Conv2d(NUM_FILTERS1, NUM_FILTERS2, KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(NUM_FILTERS2)

        self.conv3 = nn.Conv2d(NUM_FILTERS2, NUM_FILTERS3, KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(NUM_FILTERS3)

        self.pool = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)

        self._initialize_layers()

    def _initialize_layers(self):
        channels, x, y = self.INPUT_SIZE
        dummy_input = torch.zeros(1, channels, x, y)
        dummy_output = self._forward_conv(dummy_input)

        num_features = dummy_output.numel()
        self.fc1 = nn.Linear(num_features, HIDDEN_UNITS1)
        self.fc2 = nn.Linear(HIDDEN_UNITS1, HIDDEN_UNITS2)
        self.fc3 = nn.Linear(HIDDEN_UNITS2, self.NUM_CLASSES)

        self.dropout = nn.Dropout(p=DROPOUT_PROB)

    def _forward_conv(self, x):
        # First convolutional layer with Batch Norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Second convolutional layer with Batch Norm
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Third convolutional layer with Batch Norm
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    @staticmethod
    def model_name():
        return "resnet80"
