import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
pool_size = 2
kernel_size = 3
conv1_out = 64
conv2_out = 128
conv3_out = 256
conv4_out = 512
linear1 = 1024
linear2 = 512
dropout_rate = 0.1
num_heads = 8
reduction_ratio = 16  # SE block reduction ratio
dilated_kernel_size = 3  # Dilated convolution kernel size

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        se = self.pool(x)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.match_channels(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE block
        out += identity  # Add the residual connection
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Replace the convolutional layers with advanced residual blocks
        self.res_block1 = ResidualBlock(input_size[0], conv1_out, kernel_size, padding=1)
        self.res_block2 = ResidualBlock(conv1_out, conv2_out, kernel_size, padding=1)
        self.res_block3 = ResidualBlock(conv2_out, conv3_out, kernel_size, padding=1)
        self.res_block4 = ResidualBlock(conv3_out, conv4_out, kernel_size, padding=1)

        # Multi-head attention mechanism
        self.flatten = nn.Flatten(start_dim=2)  # Flatten the spatial dimensions for attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=conv4_out, num_heads=num_heads)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer

        self._initialize_layers()

    def _initialize_layers(self):
        channels, x, y = self.input_size
        dummy_input = torch.zeros(1, channels, x, y)
        dummy_output = self._forward_conv(dummy_input)

        num_features = dummy_output.numel()
        self.fc1 = nn.Linear(num_features, linear1)
        self.fc2 = nn.Linear(linear1, linear2)
        self.fc3 = nn.Linear(linear2, self.num_classes)

        self.dropout = nn.Dropout(p=dropout_rate)

    def _forward_conv(self, x):
        # Forward pass through the residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor

        # Apply multi-head attention
        b, c = x.size()
        x = x.unsqueeze(1)  # Add sequence dimension
        x, _ = self.multihead_attention(x, x, x)
        x = x.squeeze(1)  # Remove sequence dimension

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

    @staticmethod
    def model_name():
        return "seblock"
