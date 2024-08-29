import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import helpers
from models.model66 import CNN # select the model to use
from tqdm import tqdm

device = helpers.select_processor() # selects the device that is compatible with your system
model = CNN().to(device)          # model that's used

# hyperparams
num_epochs=1
batch_size=128
lr=0.001


# Model Paths
model_path, accuracy_path = helpers.model_dirs(model.model_name())

# Lists to store loss and accuracy
train_losses = []
test_accuracies = []

transform = transforms.Compose([ # addded random cropping
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.7)
n_total_steps = len(train_loader)


# Check if model exists and load it
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded existing model from {model_path}")
else:
    print("No existing model found. Starting from scratch.")

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

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # Compute average loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    # Step the scheduler after each epoch
    scheduler.step()



# Evaluate on the test dataset
model.eval()
n_correct = 0
n_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

test_accuracy = 100 * n_correct / n_samples
test_accuracies.append(test_accuracy)

# Print summary of the epoch
print(f'\nTraining finished.')
print(f'  Average Loss: {epoch_loss:.4f}')
print(f'  Accuracy: {test_accuracy:.2f}%\n')

# Save model and accuracy only if the new accuracy is higher
helpers.save(model.state_dict(), model_path, test_accuracy, accuracy_path)

print("Finished Training")

# Plotting the loss curve
helpers.show_loss(train_losses)
