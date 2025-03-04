"""
ImageNet dataset utilities for Process-Aware Benchmarking (PAB).

This module provides utilities for working with ImageNet datasets
in PAB experiments and analyses.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset

# ImageNet mean and std values for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default transformations for ImageNet
DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

DEFAULT_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_imagenet(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
    subset_size: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Load ImageNet dataset with optional subsampling.
    
    Args:
        data_dir: Directory containing ImageNet dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        train_transform: Transformations for training data
        test_transform: Transformations for test data
        subset_size: Number of samples to use (if None, use full dataset)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use default transforms if not provided
    if train_transform is None:
        train_transform = DEFAULT_TRAIN_TRANSFORM
    if test_transform is None:
        test_transform = DEFAULT_TEST_TRANSFORM
    
    # Load training data
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )
    
    # Load validation data
    val_dir = os.path.join(data_dir, 'val')
    val_dataset = torchvision.datasets.ImageFolder(
        val_dir,
        transform=test_transform
    )
    
    # Create subset if specified
    if subset_size is not None:
        if subset_size > len(train_dataset):
            print(f"Warning: Requested subset size {subset_size} is larger than dataset size {len(train_dataset)}.")
            subset_size = len(train_dataset)
        
        # Create subset for training data
        train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        
        # Create subset for validation data
        val_subset_size = min(subset_size // 5, len(val_dataset))
        val_indices = np.random.choice(len(val_dataset), val_subset_size, replace=False)
        val_dataset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Load class names if available
    class_names = {}
    class_names_path = os.path.join(data_dir, 'imagenet_classes.txt')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            for i, line in enumerate(f):
                class_names[i] = line.strip()
    else:
        # Create basic class names from folder names
        class_to_idx = (
            train_dataset.class_to_idx 
            if not isinstance(train_dataset, Subset) 
            else train_dataset.dataset.class_to_idx
        )
        for cls_name, idx in class_to_idx.items():
            class_names[idx] = cls_name
    
    return train_loader, val_loader, class_names

def get_imagenet_classes() -> List[str]:
    """
    Get list of ImageNet class names.
    
    Returns:
        List of class names
    """
    try:
        # Try to load from torchvision
        import json
        import pkg_resources
        
        # Load mapping from class index to class name
        with open(pkg_resources.resource_filename('torchvision', 'data/imagenet_classes.txt')) as f:
            class_names = [line.strip() for line in f.readlines()]
        
        return class_names
    except (ImportError, FileNotFoundError):
        # Fallback to returning class indices as strings
        return [f"Class {i}" for i in range(1000)]

class ImageNetSubset(Dataset):
    """ImageNet subset with specific classes."""
    
    def __init__(
        self,
        data_dir: str,
        classes: List[int],
        transform: Optional[transforms.Compose] = None,
        split: str = 'val'
    ):
        """
        Initialize ImageNet subset.
        
        Args:
            data_dir: Directory containing ImageNet dataset
            classes: List of class indices to include
            transform: Transformations to apply
            split: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.classes = set(classes)
        self.transform = transform or DEFAULT_TEST_TRANSFORM
        self.split = split
        
        # Load full dataset
        dataset_dir = os.path.join(data_dir, split)
        self.dataset = torchvision.datasets.ImageFolder(
            dataset_dir,
            transform=self.transform
        )
        
        # Filter samples by class
        self.samples = []
        for i, (path, target) in enumerate(self.dataset.samples):
            if target in self.classes:
                self.samples.append((path, target, i))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target, original_idx = self.samples[idx]
        
        # Load image using dataset's loader
        image = self.dataset.loader(path)
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target

def create_challenging_subset(
    data_dir: str,
    batch_size: int = 256,
    transform: Optional[transforms.Compose] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, List[int]]:
    """
    Create a challenging subset of ImageNet with fine-grained categories.
    
    Args:
        data_dir: Directory containing ImageNet dataset
        batch_size: Batch size for DataLoader
        transform: Transformations to apply
        num_workers: Number of workers for DataLoader
        
    Returns:
        Tuple of (data_loader, class_indices)
    """
    # Define challenging classes (e.g., different dog breeds)
    challenging_classes = [
        151, 152, 153, 154, 155,  # Different dog breeds
        156, 157, 158, 159, 160,
        161, 162, 163, 164, 165,
        166, 167, 168, 219, 220,  # Different cat breeds
        221, 222, 223, 249, 250,  # Different bird species
        251, 252, 253, 254, 255
    ]
    
    # Create subset
    dataset = ImageNetSubset(
        data_dir=data_dir,
        classes=challenging_classes,
        transform=transform,
        split='val'
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, challenging_classes
