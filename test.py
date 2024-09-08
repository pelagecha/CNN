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

dataset_name = "CANCER"  # Dataset to use ("CIFAR10" or "CIFAR100")
batch_size = 256                                              # Number of samples per batch

# Load dataset settings
with open('settings.json', 'r') as f: 
    dataset_settings = json.load(f)
settings = dataset_settings[dataset_name]  # Settings for the selected dataset

device = helpers.select_processor()  # Select compatible device
model = Model(input_size=settings["input_size"], 
               num_classes=settings["num_classes"]).to(device)  # Initialize model with dataset-specific settings

# Load the model
model_path, _ = helpers.model_dirs(model.model_name(), dataset_name)
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose(helpers.transform_init(dataset_name))

train_loader, test_loader, train_dataset, test_dataset = helpers.get_loaders(dataset_name=dataset_name, transform=transform, batch_size=batch_size)


# Evaluate the model on the test set
test_accuracy = helpers.eval(model, test_loader, device)
print(f'Accuracy of the network on the {len(train_dataset)} test images: {test_accuracy:.2f}%')


# # Load CIFAR-100 dataset
# if dataset_name == "CIFAR10":
#     dataset_class = torchvision.datasets.CIFAR10
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# elif dataset_name == "CIFAR100":
#     dataset_class = torchvision.datasets.CIFAR100
#     classes = (
#         'apple', 'aquarium_fish', 'bed', 'bee', 'beetle', 'bottle', 'bowl', 'bramble', 'bus', 'cabinet',
#         'can', 'cap', 'car', 'chair', 'clock', 'computer', 'couch', 'crab', 'cup', 'dolphin',
#         'elephant', 'emu', 'fan', 'fig', 'fire_engine', 'flamingo', 'flashlight', 'forest', 'fork', 'four-poster',
#         'fountain', 'garbage_truck', 'giraffe', 'goose', 'grand_piano', 'hammer', 'harp', 'hat', 'headphones', 'helicopter',
#         'ice_cream', 'jacket', 'jigsaw_puzzle', 'kangaroo', 'ketchup', 'keyboard', 'lamp', 'laptop', 'lemon', 'lion',
#         'microphone', 'microwave', 'mushroom', 'nail', 'net', 'orange', 'ostrich', 'owl', 'panda', 'parrot',
#         'piano', 'pizza', 'playground', 'purse', 'rabbit', 'raccoon', 'refrigerator', 'remote_control', 'rocket', 'rug',
#         'sailboat', 'saxophone', 'scissors', 'skateboard', 'skis', 'snowboard', 'soccer_ball', 'spider', 'sponge',
#         'squirrel', 'starfish', 'steering_wheel', 'stove', 'submarine', 'suitcase', 'table', 'tank', 'telephone', 'television',
#         'tiger', 'toaster', 'train', 'trumpet', 'tulip', 'umbrella', 'van', 'vase', 'watermelon', 'wheelchair',
#         'willow_tree', 'zebra'
#     )



# Function to visualize some test images and predictions
# def denormalize(tensor, mean, std):
#     mean = torch.tensor(mean).view(1, 3, 1, 1)
#     std = torch.tensor(std).view(1, 3, 1, 1)
#     return tensor * std + mean

# def visualize_predictions(loader, model, classes):
    # model.eval()
    # dataiter = iter(loader)
    # images, labels = next(dataiter)
    # images, labels = images.to(device), labels.to(device)
    # outputs = model(images)
    # _, predicted = torch.max(outputs, 1)

    # # Convert images to numpy for plotting
    # images = images.cpu()
    # predicted = predicted.cpu()
    # labels = labels.cpu()

    # # Denormalize images
    # images = denormalize(images, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # images = torch.clamp(images, 0, 1)  # Ensure values are within [0, 1]

    # # Plot the images with predicted labels
    # fig = plt.figure(figsize=(15, 15))
    # for idx in range(len(images)):
    #     ax = fig.add_subplot(1, len(images), idx + 1, xticks=[], yticks=[])

    #     # Convert tensor to PIL Image
    #     img = transforms.ToPILImage()(images[idx])

    #     # Check for invalid values in the image tensor
    #     npimg = np.array(img)
    #     if np.isnan(npimg).any() or np.isinf(npimg).any():
    #         print(f"Warning: Image {idx} contains NaN or Inf values.")
    #         continue

    #     # Display the image
    #     ax.imshow(npimg)
    #     ax.set_title(f'{classes[predicted[idx]]}\n({classes[labels[idx]]})')

    # plt.show()

# Example usage (ensure you have a DataLoader 'test_loader' and model loaded)
# visualize_predictions(test_loader, model, classes)
