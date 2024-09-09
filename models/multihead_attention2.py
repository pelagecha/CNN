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
dropout_rate= 0.5  # Increased dropout for regularization
num_heads = 12  # Number of attention heads

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) # changed to False
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False) # changed to False
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If the input and output channels differ, we use a 1x1 convolution to match them
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

        # Update the number of residual blocks
        self.res_block1 = ResidualBlock(input_size[0], conv1_out, kernel_size, padding=1)
        self.res_block2 = ResidualBlock(conv1_out, conv2_out, kernel_size, padding=1)
        self.res_block3 = ResidualBlock(conv2_out, conv3_out, kernel_size, padding=1)
        self.res_block4 = ResidualBlock(conv3_out, conv4_out, kernel_size, padding=1)
        self.res_block5 = ResidualBlock(conv4_out, conv4_out, kernel_size, padding=1)  # New block
        self.res_block6 = ResidualBlock(conv4_out, conv4_out, kernel_size, padding=1)  # New block

        # Multi-head attention mechanism
        self.flatten = nn.Flatten(start_dim=2)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=conv4_out, num_heads=num_heads)

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

        self.dropout = nn.Dropout(p=dropout_rate)

    def _forward_conv(self, x):
        # Forward pass through the residual blocks
        x = self.pool(self.res_block1(x))
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))
        x = self.pool(self.res_block4(x))
        x = self.pool(self.res_block5(x))  # New block
        x = self.pool(self.res_block6(x))  # New block

        # Flatten the spatial dimensions (Height and Width) before multi-head attention
        b, c, h, w = x.size()
        x = self.flatten(x)  # Shape: (batch_size, channels, height * width)
        x = x.permute(2, 0, 1)  # Permute for multi-head attention (sequence_length, batch_size, embed_dim)

        # Apply multi-head attention
        x, _ = self.multihead_attention(x, x, x)

        x = x.permute(1, 2, 0)  # Revert the permute operation
        x = x.view(b, c, h, w)  # Reshape to original dimensions with attended features

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        
        # Flatten the tensor for fully connected layers
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

    
    @staticmethod
    def model_name():
        return "multihead_attention2"
