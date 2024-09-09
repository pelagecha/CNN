import os # OS-related tasks
import json # JSON file handling
import warnings # Handling warnings

import torch                                            
import torch.nn as nn                                   
import torch.nn.functional as F                         
import torch.utils                                      
import torch.utils.data                                 
from torch.utils.data import DataLoader                 
                              
import torchvision.transforms as transforms            
import matplotlib.pyplot as plt                         
import numpy as np                                      
from tqdm import tqdm 
import time
import helpers                                          

from models.multihead_attention2 import Model # select the model to use


# -------------------------------------------- Main Setup -----------------------------------------------------
dataset_name = "CANCER"                                       # Dataset to use ("CIFAR10", "MNIST" etc.)
device = helpers.select_processor()                           # Select compatible device
retrain = False                                               # Select whether to start learning from scratch (False)
with open('settings.json', 'r') as f: dataset_settings = json.load(f)
settings = dataset_settings[dataset_name]                     # Settings for the selected dataset

model = Model(input_size=settings["input_size"], 
            num_classes=settings["num_classes"]).to(device)   # Initialize model with dataset-specific settings

# Hyperparameters
batch_size = 256                                              # Number of samples per batch
lr = 0.001                                                    # Learning rate for the optimizer
num_epochs = 20                                              # Total number of epochs for training

# Loss Function
criterion = nn.CrossEntropyLoss()                             # Loss function for multi-class classification tasks

# Optimizer
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)       # AdamW optimizer with weight decay for regularization

# Learning Rate Scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 
#                                              step_size=5,     # Frequency (in epochs) to update the learning rate
#                                              gamma=0.1)       # Factor by which the learning rate is reduced
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.25, patience=3, threshold=0.025, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# Model Paths
model_path, accuracy_path = helpers.model_dirs(model.model_name(), dataset_name)  # Paths for saving model and accuracy
train_losses = [] # stuff gor graphs
train_accuracies = []
# -------------------------------------------------------------------------------------------------------------



# -------------------------------------------- Dataset Setup ----------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
transform = transforms.Compose(helpers.transform_init(dataset_name))
train_loader, test_loader, train_dataset, test_dataset = helpers.get_loaders(dataset_name=dataset_name, transform=transform, batch_size=batch_size)


# Check if a saved model exists and load it
if retrain and os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # train_losses = checkpoint['train_losses']
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimiser.step()                                                  # Optimization step
            running_loss += loss.item() * images.size(0)                      # Accumulate loss
            average_loss = running_loss / ((i + 1) * train_loader.batch_size) # Compute average loss for the current batch

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'Avg Loss': f'{average_loss:.4f}'})
            current_lr = scheduler.get_last_lr()[0]  # Get the first element if you have a single parameter group
            # pbar.set_postfix({'LR': f'{current_lr:.4f}'})
            pbar.set_postfix({'LR': f'{scheduler.get_last_lr()[0]:.4f}', 'Loss': f'{loss:.4f}'})

    # Compute average loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    # Step the scheduler after each epoch
    scheduler.step(epoch_loss)
    test_accuracy = helpers.eval(model, test_loader, device)
    train_accuracies.append(test_accuracy)
# ------------------------------------------------------------------------------------------------------
# plt.plot(train_accuracies)
# plt.show()

# Evaluate on the test dataset
test_accuracy = helpers.eval(model, test_loader, device)

# Print summary of the epoch
print(f'\nTraining finished.')
print(f'  Average Loss: {epoch_loss:.4f}')
print(f'  Accuracy: {test_accuracy:.2f}%\n')

# Save model and accuracy only if the new accuracy is higher
helpers.save(model.state_dict(), model_path, test_accuracy, accuracy_path)
# helpers.save(model.state_dict(), optimiser.state_dict(), scheduler.state_dict(), train_losses, model_path, test_accuracy, accuracy_path)

print("Finished Training")
with open("losses.txt", "a+") as f:
    f.write("=" * 90 + "\n")
    f.write(f"--{model.model_name()}-- at {test_accuracy:.2f}% accuracy and a {epoch_loss:.4f} loss\n{train_losses}\n")
# Plotting the loss curve
helpers.show_loss(train_losses, model.model_name(), dataset_name)
time.sleep(1)
helpers.show_loss(train_accuracies, model.model_name(), dataset_name)
