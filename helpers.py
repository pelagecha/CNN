import torch
import os
import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader    
import torchvision
from datetime import datetime
from tqdm import tqdm 
import math

def select_processor():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device('cuda')
    else:
        print("Using CPU")
        return torch.device('cpu')


def model_dirs(model_name, dataset_name): # manage the paths for trained models
    model_dir = f'./compiled/{model_name}_{dataset_name}/'  # directory for the model
    model_path = os.path.join(model_dir, f"{model_name}.pt")  # path to the model file
    accuracy_path = os.path.join(model_dir, f"{model_name}_accuracy.pt")  # path to the accuracy file
    os.makedirs(model_dir, exist_ok=True)

    return model_path, accuracy_path


def save(model_state_dict, model_path, test_accuracy, accuracy_path):
    """
    Saves the model state and accuracy. Only saves if the new accuracy is better.
    
    Args:
        model_state_dict (dict): State dictionary of the model.
        model_path (str): Path to save the model.
        test_accuracy (float): Accuracy of the model on the test set.
        accuracy_path (str): Path to save the accuracy.
    """
    try:
        # Try to load previous accuracy
        prev_accuracy = torch.load(accuracy_path)
        if test_accuracy > prev_accuracy:
            print("New accuracy is better. Saving model...")
            torch.save(model_state_dict, model_path)
            torch.save(test_accuracy, accuracy_path)
        else:
            print("New accuracy is worse. No changes made.")
    except FileNotFoundError:
        # No previous accuracy file found, save the new model and accuracy
        print("No previous accuracy file found. Saving new model...")
        torch.save(model_state_dict, model_path)
        torch.save(test_accuracy, accuracy_path)


def transform_init(dataset_name):
    with open('settings.json', 'r') as f: 
        dataset_settings = json.load(f)

    settings = dataset_settings[dataset_name] # settings for the dataset that's used, like image dimensions etc
    transform_settings = settings["transform"]
    input_size = settings["input_size"]
    transform_list = []
    
    for augmentation in transform_settings["augmentations"]:
        if augmentation == "RandomHorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip())
        elif augmentation == "RandomCrop":
            transform_list.append(transforms.RandomCrop(input_size[1], padding=4))
        elif augmentation == "ToTensor":
            transform_list.append(transforms.ToTensor())

    # Add normalization as the last transform
    transform_list.append(transforms.Normalize(
        mean=transform_settings["normalize_mean"],
        std=transform_settings["normalize_std"]
    ))
    return transform_list

def eval(model, test_loader, device):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model: The model to evaluate.
    - test_loader: DataLoader for the test dataset.
    - device: The device (CPU or GPU) where the model and data should be located.

    Returns:
    - test_accuracy: The accuracy of the model on the test dataset.
    """
    model.eval()                        # Set the model to evaluation mode
    n_correct = 0
    n_samples = 0

    with torch.no_grad():              # Disable gradient calculations
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * n_correct / n_samples

    return test_accuracy

def show_loss(train_losses, model_name, dataset_name, save_dir='./graphs'):
    """
    Plots the training loss curve and saves the plot to a file.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{dataset_name}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    
    # Plot the training losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    
    # Save the plot
    plt.savefig(filepath)
    plt.close()  # Close the plot to free up memory

    print(f"Loss plot saved to {filepath}")



def show_accuracy(test_accuracies):
    """
    Plots the test accuracy curve.

    Args:
        test_accuracies (list): List of test accuracies recorded over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    plt.show()

def get_loaders(dataset_name, transform, batch_size):
    if dataset_name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "ImageNet":
        train_dataset = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=transform)
    elif dataset_name == "STL10":
        train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
    elif dataset_name == "SVHN":
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset



def optimal_lr(model, train_loader, dataset_name, train_dataset, scheduler, device, criterion, optimiser):
    # --------------------- finding optimal lr --------------------------
    num_lr_steps = 1000
    start_lr = 1e-4
    end_lr = 1.0
    lrs = torch.linspace(start_lr, end_lr, num_lr_steps)  # Linear range of learning rates
    lri = []  # Learning rates used
    lossi = []  # Losses recorded

    num_batches = len(train_loader)
    num_epochs = math.ceil(num_lr_steps / num_batches)
    print(f"Going to run {num_epochs} epochs to find the perfect learning rate!")

    lr_ind = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set model to training mode

        # Initialize the progress bar for the epoch
        with tqdm(total=num_batches, desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='step') as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                if lr_ind >= num_lr_steps:
                    break
                
                # Update learning rate for this iteration
                lr = lrs[lr_ind]
                for param_group in optimiser.param_groups:
                    param_group['lr'] = lr

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimiser.step()  # Optimization step

                # Record the learning rate and the loss
                lri.append(lrs[lr_ind].item())
                lossi.append(loss.item())

                running_loss += loss.item() * images.size(0)  # Accumulate loss
                average_loss = running_loss / ((i + 1) * train_loader.batch_size)  # Compute average loss for the current batch

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'LR': f'{lr:.4e}', 'Loss': f'{average_loss:.4f}'})

                lr_ind += 1

    # Plotting the results
    save_dir = './graphs'
    plt.plot(lri, lossi)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create filename with timestamp
    filename = f"{model.__class__.__name__}_{dataset_name}_loss.png"
    filepath = os.path.join(save_dir, filename)

    # Save the plot
    plt.savefig(filepath)
    plt.close()  # Close the plot to free up memory

    # Finding the optimal learning rate
    min_so_far = [start_lr, float("inf")]  # lr, loss
    sliding_window_width = 20
    
    for i in range(len(lossi) - sliding_window_width):
        window = lossi[i:i + sliding_window_width]
        avg = sum(window) / sliding_window_width
        if avg < min_so_far[1]:
            min_so_far = [lri[i + sliding_window_width // 2], avg]

    optimal_lr_value = min_so_far[0]
    print(f"Found optimal learning rate value: {optimal_lr_value}")
    return optimal_lr_value