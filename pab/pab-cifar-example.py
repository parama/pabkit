"""
Demo of Process-Aware Benchmarking (PAB) toolkit on CIFAR-10 using ResNet18.

This example demonstrates how to use PAB to track and analyze model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import argparse
import os

# Import PAB toolkit
from pab import PABTracker, track_training, pgd_attack

def parse_args():
    parser = argparse.ArgumentParser(description='Process-Aware Benchmarking (PAB) Demo')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--save-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--model-name', type=str, default='resnet18_cifar10', help='Model name')
    parser.add_argument('--evaluate-adversarial', action='store_true', help='Evaluate adversarial robustness')
    parser.add_argument('--extract-repr', action='store_true', help='Extract model representations')
    parser.add_argument('--checkpoint-freq', type=int, default=5, help='Checkpoint frequency in epochs')
    return parser.parse_args()

def get_cifar10_loaders(batch_size=128):
    """Prepare CIFAR-10 dataloaders"""
    # Data preprocessing and augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Create datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return trainloader, testloader

def create_model():
    """Create a ResNet18 model for CIFAR-10"""
    model = resnet18(pretrained=False)
    
    # Modify first conv layer to handle CIFAR-10's 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove first maxpool layer as CIFAR-10 images are too small
    model.maxpool = nn.Identity()
    
    # Modify final fully connected layer for CIFAR-10's 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Add method to get representations (for PAB)
    def get_representation(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Return the features before the final FC layer
        return x
    
    # Bind the method to the model instance
    model.get_representation = get_representation.__get__(model)
    
    return model

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get dataloaders
    trainloader, testloader = get_cifar10_loaders(args.batch_size)
    
    # Create model
    model = create_model()
    
    # Setup optimizer and criterion
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Configure adversarial evaluation if requested
    attack_fn = None
    attack_params = None
    if args.evaluate_adversarial:
        attack_fn = pgd_attack
        attack_params = {
            'epsilon': 8/255,  # CIFAR-10 common epsilon
            'alpha': 2/255,    # Step size
            'iterations': 10,
            'norm': 'Linf'
        }
    
    # Train with PAB tracking
    model, tracker = track_training(
        model=model,
        train_loader=trainloader,
        val_loader=testloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epochs,
        device=device,
        num_classes=10,  # CIFAR-10 has 10 classes
        attack_fn=attack_fn,
        attack_params=attack_params,
        model_name=args.model_name,
        save_dir=args.save_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_model=True,
        extract_repr=args.extract_repr,
        scheduler=scheduler
    )
    
    print(f"Training completed. Results saved to {args.save_dir}/{args.model_name}")
    
    # Generate final report
    report = tracker.generate_report()
    print("PAB Report Summary:")
    for metric, value in report.get('pab_metrics', {}).items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
