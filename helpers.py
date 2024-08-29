import torch
import os
import matplotlib.pyplot as plt


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
    model_dir = f'./trained/{model_name}/'  # directory for the model
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
