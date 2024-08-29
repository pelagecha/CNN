import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1(nn.Module):
    def __init__(self):
        # convolutional layers
        super(Conv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5) # input channels (in this case 3 bc rgb), out_channels, kernel size nxn
        self.pool = nn.MaxPool2d(2, 2)  # kernel size, stride
        self.conv2 = nn.Conv2d(12, 24, 5)

        # fully-connected layers
        self.fc1 = nn.Linear(24*5*5, 120) # input like that because that's the shape of the output of conv2
        self.fc2 = nn.Linear(120, 84) # input size, output size
        self.fc3 = nn.Linear(84, 10) # output=10 because 10 image classes

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24*5*5) # flattens the 2-d conv layer

        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = self.fc3(x)
        # softmax already applied with "criterion"
        return x