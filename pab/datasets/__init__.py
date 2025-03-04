"""
Dataset utilities for Process-Aware Benchmarking (PAB).

This package provides datasets and utilities for PAB experiments and analyses.
"""

from typing import Optional, Tuple
from torch.utils.data import DataLoader

from .imagenet import (
    load_imagenet, 
    get_imagenet_classes, 
    ImageNetSubset, 
    create_challenging_subset,
    IMAGENET_MEAN,
    IMAGENET_STD
)

from .dataset_utils import load_cifar10, load_cifar100

def load_dataset(
    name: str,
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name ('cifar10', 'cifar100', 'imagenet', etc.)
        data_dir: Directory to store dataset
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loaders
        subset_size: Number of samples to use (if None, use full dataset)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    loaders = get_dataset_loaders(
        dataset_name=name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=subset_size
    )
    
    # Return just the loaders (first two elements) from get_dataset_loaders
    return loaders[0], loaders[1]

def get_dataset_loaders(
    dataset_name, 
    data_dir, 
    batch_size=128, 
    num_workers=4, 
    subset_size=None
):
    """
    Get data loaders for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cifar10', 'imagenet')
        data_dir: Directory containing the dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        subset_size: Number of samples to use (if None, use full dataset)
        
    Returns:
        Tuple of (train_loader, test_loader, class_names)
    """
    if dataset_name.lower() == 'imagenet':
        return load_imagenet(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size
        )
    
    elif dataset_name.lower() == 'cifar10':
        return load_cifar10(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size
        )
    
    elif dataset_name.lower() == 'cifar100':
        return load_cifar100(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")