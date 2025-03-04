"""
ImageNet Case Study for Process-Aware Benchmarking (PAB).

This script demonstrates PAB evaluation on ImageNet with three model architectures:
- ResNet-50
- EfficientNet
- Vision Transformer (ViT)

It includes:
- Learning trajectory tracking
- Class-wise learning progression analysis
- Adversarial robustness evaluation
- Visualization of results
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import sys

# Add parent directory to path for importing PAB
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pab import ProcessAwareBenchmark
from pab.visualization import (
    plot_learning_trajectory, 
    plot_class_progression, 
    plot_robustness_curve, 
    plot_pab_summary
)
from pab.utils import (
    compute_class_accuracies, 
    evaluate_adversarial_robustness, 
    extract_feature_representations,
    export_metrics_to_json
)

# Parse arguments
parser = argparse.ArgumentParser(description='ImageNet PAB Case Study')
parser.add_argument('--data_dir', type=str, default='./data/imagenet', 
                   help='Path to ImageNet dataset')
parser.add_argument('--output_dir', type=str, default='./results/imagenet', 
                   help='Directory to save results')
parser.add_argument('--model_type', type=str, default='resnet50', 
                   choices=['resnet50', 'efficientnet', 'vit'],
                   help='Model architecture to evaluate')
parser.add_argument('--batch_size', type=int, default=128, 
                   help='Batch size for training')
parser.add_argument('--epochs', type=int, default=90, 
                   help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.1, 
                   help='Initial learning rate')
parser.add_argument('--checkpoint_frequency', type=int, default=5, 
                   help='Save checkpoint every N epochs')
parser.add_argument('--eval_frequency', type=int, default=1, 
                   help='Evaluate model every N epochs')
parser.add_argument('--adv_eval_frequency', type=int, default=10, 
                   help='Evaluate adversarial robustness every N epochs')
parser.add_argument('--debug', action='store_true', 
                   help='Use small subset of data for debugging')
parser.add_argument('--load_checkpoint', type=str, default=None, 
                   help='Path to load checkpoint from')
parser.add_argument('--num_classes', type=int, default=1000, 
                   help='Number of classes in dataset')
parser.add_argument('--distributed', action='store_true', 
                   help='Use distributed training')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
checkpoint_dir = os.path.join(args.output_dir, 'checkpoints', args.model_type)
os.makedirs(checkpoint_dir, exist_ok=True)

# Save configuration
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageNet dataset
def load_dataset():
    if args.debug:
        # Use small subset for debugging
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=train_transform
        )
        val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_dir, 'val'),
            transform=test_transform
        )
        
        # Create small subsets
        train_indices = np.random.choice(len(train_dataset), 1000, replace=False)
        val_indices = np.random.choice(len(val_dataset), 500, replace=False)
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    else:
        # Use full dataset
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=train_transform
        )
        val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_dir, 'val'),
            transform=test_transform
        )
    
    return train_dataset, val_dataset

# Initialize data loaders
train_dataset, val_dataset = load_dataset()
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Load model
def get_model():
    if args.model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)
        feature_layer_name = 'layer4'  # Last residual block
    elif args.model_type == 'efficientnet':
        model = torchvision.models.efficientnet_b0(pretrained=False, num_classes=args.num_classes)
        feature_layer_name = 'features.8'  # A late convolutional block
    elif args.model_type == 'vit':
        model = torchvision.models.vit_b_16(pretrained=False, num_classes=args.num_classes)
        feature_layer_name = 'encoder.layers.11'  # Last transformer block
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model, feature_layer_name

model, feature_layer_name = get_model()
model = model.to(device)

if args.distributed:
    model = torch.nn.DataParallel(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), 
    lr=args.learning_rate, 
    momentum=0.9, 
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=args.epochs
)

# Initialize Process-Aware Benchmarking
pab = ProcessAwareBenchmark(
    checkpoint_dir=checkpoint_dir,
    save_frequency=args.checkpoint_frequency,
    track_representations=True
)

# Load checkpoint if specified
start_epoch = 1
if args.load_checkpoint:
    checkpoint = torch.load(args.load_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    pab.metrics = checkpoint['pab_metrics']
    print(f"Loaded checkpoint from epoch {start_epoch-1}")

# Training function
def train(epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return train_loss / len(train_loader), correct / total

# Evaluation function
def evaluate():
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Evaluation')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': val_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    return val_loss / len(val_loader), correct / total

# Extract class names for ImageNet
def get_class_names():
    # Load class names if available
    try:
        with open(os.path.join(args.data_dir, 'imagenet_classes.txt'), 'r') as f:
            class_names = {i: line.strip() for i, line in enumerate(f)}
    except FileNotFoundError:
        # Use class indices if names not available
        class_names = {i: f"Class {i}" for i in range(args.num_classes)}
    
    return class_names

class_names = get_class_names()

# Main training loop
for epoch in range(start_epoch, args.epochs + 1):
    # Train
    train_loss, train_acc = train(epoch)
    
    # Evaluate
    if epoch % args.eval_frequency == 0:
        val_loss, val_acc = evaluate()
        
        # Compute per-class accuracies (sample a subset for efficiency)
        if args.debug or epoch % (args.eval_frequency * 5) == 0:
            class_accuracies = compute_class_accuracies(
                model, val_loader, num_classes=args.num_classes, device=device
            )
        else:
            class_accuracies = None
        
        # Evaluate adversarial robustness
        adversarial_acc = None
        if epoch % args.adv_eval_frequency == 0:
            print("Evaluating adversarial robustness...")
            # Create a smaller loader for adversarial evaluation (for efficiency)
            adv_indices = np.random.choice(len(val_dataset), 
                                           min(1000, len(val_dataset)), 
                                           replace=False)
            adv_dataset = Subset(val_dataset, adv_indices)
            adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, 
                                    shuffle=False, num_workers=2)
            
            adv_metrics = evaluate_adversarial_robustness(
                model, adv_loader, device=device, epsilon=0.03
            )
            adversarial_acc = adv_metrics['adversarial_accuracy']
            print(f"Adversarial accuracy: {adversarial_acc:.4f}")
        
        # Extract feature representations (for rule evolution metric)
        feature_extractor = None
        if epoch % args.checkpoint_frequency == 0:
            # Create a smaller loader for feature extraction
            indices = np.random.choice(len(val_dataset), 
                                       min(500, len(val_dataset)), 
                                       replace=False)
            subset_dataset = Subset(val_dataset, indices)
            subset_loader = DataLoader(subset_dataset, batch_size=args.batch_size, 
                                      shuffle=False, num_workers=2)
            
            # Extract features
            features = extract_feature_representations(
                model, subset_loader, layer_name=feature_layer_name, 
                device=device, num_samples=100
            )
            feature_extractor = lambda m: features
        
        # Track metrics with PAB
        pab.track_epoch(
            model=model,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            class_accuracies=class_accuracies,
            adversarial_acc=adversarial_acc,
            feature_extractor=feature_extractor
        )
        
        # Save training state
        if epoch % args.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"checkpoint_epoch_{epoch:04d}"
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'pab_metrics': pab.metrics
            }, f"{checkpoint_path}.pt")
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Update learning rate
    scheduler.step()

# Evaluate final model
print("Evaluating final model...")
val_loss, val_acc = evaluate()

# Perform final adversarial robustness evaluation
print("Evaluating adversarial robustness...")
adv_metrics = evaluate_adversarial_robustness(
    model, val_loader, device=device, epsilon=0.03
)

# Final PAB tracking
pab.track_epoch(
    model=model,
    epoch=args.epochs,
    train_loss=train_loss,
    val_loss=val_loss,
    train_acc=train_acc,
    val_acc=val_acc,
    adversarial_acc=adv_metrics['adversarial_accuracy']
)

# Evaluate the learning trajectory
print("Evaluating learning trajectory...")
eval_results = pab.evaluate_trajectory()
print("PAB Evaluation Results:")
print(json.dumps(eval_results, indent=2))

# Generate PAB summary
summary = pab.summarize()
print("\nPAB Summary:")
print(summary)

# Save summary to file
with open(os.path.join(args.output_dir, f'{args.model_type}_pab_summary.txt'), 'w') as f:
    f.write(summary)

# Export metrics to JSON
export_metrics_to_json(
    pab.metrics, 
    os.path.join(args.output_dir, f'{args.model_type}_metrics.json')
)

# Create visualizations
print("Creating visualizations...")

# Learning trajectory plot
plot_learning_trajectory(
    train_losses=pab.metrics['train_loss'],
    val_losses=pab.metrics['val_loss'],
    train_accs=pab.metrics['train_acc'],
    val_accs=pab.metrics['val_acc'],
    title=f"{args.model_type} Learning Trajectory on ImageNet",
    save_path=os.path.join(args.output_dir, f'{args.model_type}_learning_trajectory.png')
)

# Class progression plot
if pab.metrics['class_accuracy']:
    plot_class_progression(
        class_accuracies=pab.metrics['class_accuracy'],
        class_names=class_names,
        num_classes_to_show=10,
        save_path=os.path.join(args.output_dir, f'{args.model_type}_class_progression.png')
    )

# Robustness curve
if pab.metrics['adversarial_robustness']:
    plot_robustness_curve(
        clean_accuracies=pab.metrics['val_acc'],
        adversarial_accuracies=pab.metrics['adversarial_robustness'],
        save_path=os.path.join(args.output_dir, f'{args.model_type}_robustness_curve.png')
    )

# Comprehensive PAB summary plot
plot_pab_summary(
    metrics=pab.metrics,
    save_path=os.path.join(args.output_dir, f'{args.model_type}_pab_summary.png')
)

print(f"\nImageNet PAB case study for {args.model_type} completed!")
print(f"Results saved to {args.output_dir}")
