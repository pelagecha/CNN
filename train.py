import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('mps')

# hyperparams
num_epochs=10
batch_size=64
lr=0.001

# Lists to store loss and accuracy
train_losses = []
test_accuracies = []

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




from models.model80 import CNN # model to use
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
n_total_steps = len(train_loader)

# training
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Accumulate loss
        running_loss += loss.item() * images.size(0)

        # Print progress
        if (i + 1) % (n_total_steps // 10) == 0:  # Print every 10% of the way through
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {running_loss / ((i + 1) * batch_size):.4f}")

    epoch_loss = running_loss / len(train_dataset)  # Use dataset size instead of loader size
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}] finished. Average Loss: {epoch_loss:.3f}")

print("Finished Training")
PATH = f'./{model.model_name()}'
torch.save(model.state_dict(), PATH)



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        # Update class-wise statistics
        for i in range(labels.size(0)):  # Iterate over the batch
            label = labels[i].item()  # Convert tensor to Python integer
            pred = predicted[i].item()  # Convert tensor to Python integer
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    # Compute overall accuracy
    acc = 100 * n_correct / n_samples
    test_accuracies.append(acc)
    print(f"Accuracy of the network: {acc}%")

    # Compute per-class accuracy
    for i in range(10):
        if n_class_samples[i] > 0:
            acc = 100 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of {classes[i]}: {acc}%")
        else:
            print(f"Accuracy of {classes[i]}: No samples found")



# Plotting the loss curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# Plotting the accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.show()

