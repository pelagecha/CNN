import os # OS-related tasks
import json # JSON file handling
import warnings # Handling warnings

import torch                                            
import torch.nn as nn                                   
import torch.nn.functional as F                         
import torch.utils                                      
import torch.utils.data                                 
from torch.utils.data import DataLoader                 

import torchvision                                      
import torchvision.transforms as transforms            
import matplotlib.pyplot as plt                         
import numpy as np                                      
from tqdm import tqdm # Progress bar for loops
import helpers                                          

from models.resnet80 import Model # select the model to use


# -------------------------------------------- Main Setup -----------------------------------------------------
dataset_name = "CIFAR10"                                      # Dataset to use ("CIFAR10" or "MNIST")

with open('settings.json', 'r') as f: dataset_settings = json.load(f)
settings = dataset_settings[dataset_name]                     # Settings for the selected dataset

device = helpers.select_processor()                           # Select compatible device
model = Model(INPUT_SIZE=settings["input_size"], 
            NUM_CLASSES=settings["num_classes"]).to(device)   # Initialize model with dataset-specific settings

# Hyperparameters
batch_size = 256                                              # Number of samples per batch
lr = 0.001                                                    # Learning rate for the optimizer
num_epochs = 5                                                # Total number of epochs for training

# Loss Function
criterion = nn.CrossEntropyLoss()                             # Loss function for multi-class classification tasks

# Optimizer
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)      # AdamW optimizer with weight decay for regularization

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 
                                             step_size=10,    # Frequency (in epochs) to update the learning rate
                                             gamma=0.75)      # Factor by which the learning rate is reduced

# Model Paths
model_path, accuracy_path = helpers.model_dirs(model.model_name())  # Paths for saving model and accuracy
train_losses = [] # stuff gor graphs
# -------------------------------------------------------------------------------------------------------------



# -------------------------------------------- Dataset Setup ----------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
transform = transforms.Compose(helpers.transform_init(dataset_name))
if dataset_name == "MNIST":
    train_dataset = torchvision.datasets.mnist(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.mnist(root='./data', train=False, download=True, transform=transform)
elif dataset_name == "CIFAR10":
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check if model exists and load it
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded existing model from {model_path}")
else:
    print("No existing model found. Starting from scratch.")
# -------------------------------------------------------------------------------------------------------------





# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()  # Set model to training mode
    
    # Initialize the progress bar for the epoch
    with tqdm(total=len(train_loader), desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='step') as pbar:
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimiser.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            optimiser.step()

            # Accumulate loss
            running_loss += loss.item() * images.size(0)

            # Compute average loss for the current batch
            average_loss = running_loss / ((i + 1) * train_loader.batch_size)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'Avg Loss': f'{average_loss:.4f}'})

    # Compute average loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    # Step the scheduler after each epoch
    scheduler.step()



# Evaluate on the test dataset
test_accuracy = helpers.eval(model, test_loader, device)

# Print summary of the epoch
print(f'\nTraining finished.')
print(f'  Average Loss: {epoch_loss:.4f}')
print(f'  Accuracy: {test_accuracy:.2f}%\n')

# Save model and accuracy only if the new accuracy is higher
helpers.save(model.state_dict(), model_path, test_accuracy, accuracy_path)

print("Finished Training")

# Plotting the loss curve
helpers.show_loss(train_losses)
