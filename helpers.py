import torch
import os
import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms

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


def model_dirs(model_name): # manage the paths for trained models
    model_dir = f'./compiled/{model_name}/'  # directory for the model
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

def show_loss(train_losses):
    """
    Plots the training loss curve.

    Args:
        train_losses (list): List of training losses recorded over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    plt.show()

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

