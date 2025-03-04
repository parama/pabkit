"""
Representation Analysis Using Process-Aware Benchmarking (PAB).

This script analyzes the evolution of model representations during training
to understand how feature spaces develop and how models learn structured abstractions.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from tqdm import tqdm
import sys

# Add parent directory to path for importing PAB
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pab.utils import extract_feature_representations
from pab.tracking import CheckpointManager

# Parse arguments
parser = argparse.ArgumentParser(description='PAB Representation Analysis')
parser.add_argument('--checkpoint_dir', type=str, required=True,
                   help='Directory containing model checkpoints')
parser.add_argument('--data_dir', type=str, default='./data',
                   help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10',
                   choices=['cifar10', 'cifar100', 'imagenet'],
                   help='Dataset to use for analysis')
parser.add_argument('--output_dir', type=str, default='./results/representation',
                   help='Directory to save analysis results')
parser.add_argument('--batch_size', type=int, default=64,
                   help='Batch size for feature extraction')
parser.add_argument('--num_samples', type=int, default=1000,
                   help='Number of samples to use for analysis')
parser.add_argument('--layer_name', type=str, default=None,
                   help='Name of layer to extract features from')
parser.add_argument('--model_type', type=str, default='resnet',
                   choices=['resnet', 'efficientnet', 'vit'],
                   help='Model architecture type')
parser.add_argument('--every_n', type=int, default=5,
                   help='Analyze every N checkpoints')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
def load_dataset():
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform
        )
        num_classes = 100
    elif args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_dir, 'imagenet', 'val'),
            transform=transform
        )
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create a subset for analysis
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    
    # Create data loader
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return loader, dataset.classes if hasattr(dataset, 'classes') else None, num_classes

# Load dataset
data_loader, class_names, num_classes = load_dataset()
print(f"Loaded {args.dataset} dataset with {len(data_loader.dataset)} samples")

# Get model architecture
def get_model(checkpoint=None):
    if args.model_type == 'resnet':
        model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
        if args.layer_name is None:
            args.layer_name = 'layer4'
    elif args.model_type == 'efficientnet':
        model = torchvision.models.efficientnet_b0(pretrained=False, num_classes=num_classes)
        if args.layer_name is None:
            args.layer_name = 'features.8'
    elif args.model_type == 'vit':
        model = torchvision.models.vit_b_16(pretrained=False, num_classes=num_classes)
        if args.layer_name is None:
            args.layer_name = 'encoder.layers.11'
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load checkpoint if provided
    if checkpoint is not None:
        model.load_state_dict(checkpoint)
    
    return model

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(args.checkpoint_dir)
checkpoints = checkpoint_manager.list_checkpoints()

if not checkpoints:
    print(f"No checkpoints found in {args.checkpoint_dir}")
    sys.exit(1)

print(f"Found {len(checkpoints)} checkpoints")

# Analyze feature representations across checkpoints
representations = {}
labels = []

# Get target labels from a batch
for inputs, batch_labels in data_loader:
    labels.extend(batch_labels.numpy())
    if len(labels) >= args.num_samples:
        labels = labels[:args.num_samples]
        break

# Extract features from each checkpoint
for checkpoint_name in tqdm(checkpoints[::args.every_n], desc="Extracting features"):
    model_state, _ = checkpoint_manager.load_checkpoint(checkpoint_name)
    
    # Extract epoch number from checkpoint name
    try:
        epoch = int(checkpoint_name.split('_')[-1])
    except ValueError:
        epoch = int(checkpoints.index(checkpoint_name) * args.every_n)
    
    # Load model with checkpoint
    model = get_model(model_state)
    model = model.to(device)
    model.eval()
    
    # Extract features
    features = extract_feature_representations(
        model, data_loader, layer_name=args.layer_name, 
        device=device, num_samples=args.num_samples
    )
    
    representations[epoch] = features

# Calculate representation similarity matrix
epochs = sorted(representations.keys())
similarity_matrix = np.zeros((len(epochs), len(epochs)))

for i, epoch1 in enumerate(epochs):
    for j, epoch2 in enumerate(epochs):
        # Flatten representations
        repr1 = representations[epoch1].reshape(representations[epoch1].shape[0], -1)
        repr2 = representations[epoch2].reshape(representations[epoch2].shape[0], -1)
        
        # Compute cosine similarity between averaged representations
        repr1_mean = repr1.mean(axis=0)
        repr2_mean = repr2.mean(axis=0)
        
        norm1 = np.linalg.norm(repr1_mean)
        norm2 = np.linalg.norm(repr2_mean)
        
        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(repr1_mean, repr2_mean) / (norm1 * norm2)
        else:
            similarity = 0
            
        similarity_matrix[i, j] = similarity

# Calculate representation divergence for each epoch transition
divergences = []
for i in range(1, len(epochs)):
    repr1 = representations[epochs[i-1]].reshape(representations[epochs[i-1]].shape[0], -1)
    repr2 = representations[epochs[i]].reshape(representations[epochs[i]].shape[0], -1)
    
    # Normalize representations
    repr1 = repr1 / (np.linalg.norm(repr1, axis=1)[:, np.newaxis] + 1e-8)
    repr2 = repr2 / (np.linalg.norm(repr2, axis=1)[:, np.newaxis] + 1e-8)
    
    # Compute average L2 distance
    divergence = np.mean(np.sqrt(np.sum((repr1 - repr2) ** 2, axis=1)))
    divergences.append((epochs[i], divergence))

# Create visualizations

# 1. Representation similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.xlabel('Epoch')
plt.ylabel('Epoch')
plt.title('Representation Similarity Matrix')
plt.xticks(np.arange(len(epochs)), epochs)
plt.yticks(np.arange(len(epochs)), epochs)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'similarity_matrix.png'), dpi=300)

# 2. Representation divergence over training
plt.figure(figsize=(10, 6))
plt.plot([d[0] for d in divergences], [d[1] for d in divergences], 'o-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Representation Divergence')
plt.title('Feature Representation Evolution')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'representation_divergence.png'), dpi=300)

# 3. PCA visualization of representation evolution
plt.figure(figsize=(15, 10))

# Select a few key epochs to visualize
key_epochs = epochs[::max(1, len(epochs)//5)]
if epochs[-1] not in key_epochs:
    key_epochs.append(epochs[-1])

# Get unique labels
unique_labels = np.unique(labels)
if len(unique_labels) > 10:
    # If too many classes, just show the first 10
    classes_to_show = unique_labels[:10]
else:
    classes_to_show = unique_labels

# Colors for different classes
colors = plt.cm.tab10(np.linspace(0, 1, len(classes_to_show)))

for i, epoch in enumerate(key_epochs):
    plt.subplot(2, (len(key_epochs) + 1) // 2, i + 1)
    
    # Get representations for this epoch
    repr_data = representations[epoch].reshape(representations[epoch].shape[0], -1)
    
    # Apply PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(repr_data)
    
    # Plot each class with a different color
    for j, cls in enumerate(classes_to_show):
        mask = np.array(labels) == cls
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                   color=colors[j], alpha=0.7, s=30, label=class_names[cls] if class_names else f"Class {cls}")
    
    if i == 0:
        plt.legend(loc='best')
    
    plt.title(f'Epoch {epoch}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, linestyle='--', alpha=0.3)

plt.suptitle('Evolution of Feature Representations (PCA)', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(os.path.join(args.output_dir, 'pca_evolution.png'), dpi=300)

# 4. t-SNE visualization of final representation
plt.figure(figsize=(12, 10))

final_epoch = epochs[-1]
repr_data = representations[final_epoch].reshape(representations[final_epoch].shape[0], -1)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
reduced_data = tsne.fit_transform(repr_data)

# Plot each class with a different color
for j, cls in enumerate(classes_to_show):
    mask = np.array(labels) == cls
    plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
               color=colors[j], alpha=0.7, s=50, label=class_names[cls] if class_names else f"Class {cls}")

plt.legend(loc='best')
plt.title(f't-SNE Visualization of Final Representations (Epoch {final_epoch})')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'tsne_final.png'), dpi=300)

# Save numerical results
results = {
    'similarity_matrix': similarity_matrix.tolist(),
    'epochs': epochs,
    'divergences': divergences,
    'final_epoch': final_epoch
}

with open(os.path.join(args.output_dir, 'representation_analysis.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"Representation analysis completed. Results saved to {args.output_dir}")
