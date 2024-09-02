import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.multihead_attention import Model
import helpers
import json

dataset_name = "CIFAR10"                                      # Dataset to use ("CIFAR10" or "MNIST")

with open('settings.json', 'r') as f: dataset_settings = json.load(f)
settings = dataset_settings[dataset_name]                     # Settings for the selected dataset

device = helpers.select_processor()                           # Select compatible device
model = Model(input_size=settings["input_size"], 
            num_classes=settings["num_classes"]).to(device)   # Initialize model with dataset-specific settings
device = helpers.select_processor()
model_path, _ = helpers.model_dirs(model.model_name())
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
model.eval()  # Set the model to evaluation mode

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

# Define class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Evaluate the model on the test set
model.to(device)
test_accuracy = helpers.eval(model, test_loader, device)
print(f'Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%')

# Function to visualize some test images and predictions
def denormalize(tensor, mean, std):
    """
    Denormalize an image tensor by reversing the normalization process.
    Args:
        tensor (torch.Tensor): The normalized image tensor.
        mean (tuple): The mean values used during normalization.
        std (tuple): The standard deviation values used during normalization.
    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean

def visualize_predictions(loader, model, classes):
    model.eval()
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Convert images to numpy for plotting
    images = images.cpu()
    predicted = predicted.cpu()
    labels = labels.cpu()

    # Denormalize images
    images = denormalize(images, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    images = torch.clamp(images, 0, 1)  # Ensure values are within [0, 1]

    # Plot the images with predicted labels
    fig = plt.figure(figsize=(10, 5))
    for idx in range(len(images)):
        ax = fig.add_subplot(1, len(images), idx+1, xticks=[], yticks=[])
        
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(images[idx])
        
        # Check for invalid values in the image tensor
        npimg = np.array(img)
        if np.isnan(npimg).any() or np.isinf(npimg).any():
            print(f"Warning: Image {idx} contains NaN or Inf values.")
            continue

        # Display the image
        ax.imshow(npimg)
        ax.set_title(f'{classes[predicted[idx]]}\n({classes[labels[idx]]})')
    
    plt.show()


# Example usage (ensure you have a DataLoader 'test_loader' and model loaded)
visualize_predictions(test_loader, model, classes)
