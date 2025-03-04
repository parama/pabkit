"""
Simple example of using the Process-Aware Benchmarking (PAB) toolkit.

This script demonstrates how to track a model's learning trajectory
using PAB during training on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import pab
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pab import ProcessAwareBenchmark
from pab.visualization import plot_learning_trajectory, plot_class_progression, plot_pab_summary
from pab.utils import compute_class_accuracies, evaluate_adversarial_robustness


# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 20
batch_size = 128
learning_rate = 0.001

# Load and preprocess CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# Class names for CIFAR-10
class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    
def main():
    # Create model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    
    # Initialize Process-Aware Benchmarking
    pab = ProcessAwareBenchmark(
        checkpoint_dir='./results/checkpoints',
        save_frequency=5,
        track_representations=True
    )
    
    # Training loop
    total_steps = len(train_loader)
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_steps}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average training metrics
        train_accuracy = train_correct / train_total
        train_loss = train_loss / total_steps
        
        # Evaluate model
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        # Calculate average validation metrics
        val_accuracy = val_correct / val_total
        val_loss = val_loss / len(test_loader)
        
        print(f'Epoch [{epoch}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Compute per-class accuracies
        class_accuracies = compute_class_accuracies(
            model, test_loader, num_classes=10, device=device
        )
        
        # Evaluate adversarial robustness (every 5 epochs to save time)
        adversarial_acc = None
        if epoch % 5 == 0:
            adv_metrics = evaluate_adversarial_robustness(
                model, test_loader, device=device
            )
            adversarial_acc = adv_metrics['adversarial_accuracy']
        
        # Track metrics with PAB
        pab.track_epoch(
            model=model,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_accuracy,
            val_acc=val_accuracy,
            class_accuracies=class_accuracies,
            adversarial_acc=adversarial_acc
        )
    
    # Evaluate the learning trajectory
    eval_results = pab.evaluate_trajectory()
    print("\nPAB Evaluation Results:")
    print(eval_results)
    
    # Print PAB summary
    summary = pab.summarize()
    print("\nPAB Summary:")
    print(summary)
    
    # Save summary to file
    with open('./results/pab_summary.txt', 'w') as f:
        f.write(summary)
    
    # Visualize learning trajectory
    plot_learning_trajectory(
        train_losses=pab.metrics['train_loss'],
        val_losses=pab.metrics['val_loss'],
        train_accs=pab.metrics['train_acc'],
        val_accs=pab.metrics['val_acc'],
        save_path='./results/learning_trajectory.png'
    )
    
    # Visualize class progression
    plot_class_progression(
        class_accuracies=pab.metrics['class_accuracy'],
        class_names=class_names,
        save_path='./results/class_progression.png'
    )
    
    # Create comprehensive PAB summary visualization
    plot_pab_summary(
        metrics=pab.metrics,
        save_path='./results/pab_summary.png'
    )
    
    print("\nTraining complete! Results saved to './results' directory.")


if __name__ == '__main__':
    main()
