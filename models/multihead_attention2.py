import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
pool_size = 2
kernel_size = 3  # Smaller kernel size for better feature extraction

conv1_out = 64
conv2_out = 128
conv3_out = 256
conv4_out = 256  # Adding an additional convolutional layer
linear1 = 256
linear2 = 512
linear3 = 512
linear4 = 256
linear5 = 128
dropout_rate= 0.2  # Increased dropout for regularization
num_heads = 4  # Number of attention heads

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Use BatchNorm2d
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Use BatchNorm2d

        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.match_channels(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # Add the residual connection
        out = self.relu(out)

        return out



class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        self.res_block1 = ResidualBlock(input_size[0], conv1_out, kernel_size, padding=1)
        self.res_block2 = ResidualBlock(conv1_out, conv2_out, kernel_size, padding=1)
        self.res_block3 = ResidualBlock(conv2_out, conv3_out, kernel_size, padding=1)
        self.res_block4 = ResidualBlock(conv3_out, conv4_out, kernel_size, padding=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.flatten = nn.Flatten(start_dim=1)  # Flatten the output of global average pooling

        self.multihead_attention = nn.MultiheadAttention(embed_dim=conv4_out, num_heads=num_heads)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.relu = nn.ReLU()

        self._initialize_layers()

    def _initialize_layers(self):
        channels, x, y = self.input_size
        dummy_input = torch.zeros(1, channels, x, y)
        dummy_output = self._forward_conv(dummy_input)

        num_features = dummy_output.numel()
        self.fc1 = nn.Linear(num_features, linear1, bias=False)
        self.fc2 = nn.Linear(linear1,      linear2, bias=False)
        self.fc3 = nn.Linear(linear2,      linear3, bias=False)
        self.fc4 = nn.Linear(linear3,      linear4, bias=False)
        self.fc5 = nn.Linear(linear4,      linear5, bias=False)
        self.fc6 = nn.Linear(linear5,      self.num_classes)

        self.dropout = nn.Dropout(p=dropout_rate)

    def _forward_conv(self, x):
        x = self.pool(self.res_block1(x))
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))
        x = self.pool(self.res_block4(x))

        # Apply global average pooling
        x = self.global_avg_pool(x)
        x = self.flatten(x)  # Shape: (batch_size, channels)

        # Prepare input for multi-head attention
        x = x.unsqueeze(0)  # Add a sequence dimension
        x, _ = self.multihead_attention(x, x, x)
        x = x.squeeze(0)  # Remove the sequence dimension

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc4(x), negative_slope=0.01)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc5(x), negative_slope=0.01)
        x = self.dropout(x)

        x = self.fc6(x)
        return x

    
    @staticmethod
    def model_name():
        return "multihead_attention2"
