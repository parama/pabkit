"""
Dataset utilities for Process-Aware Benchmarking (PAB).
"""

import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader

def load_dataset(
    name: str,
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name ('cifar10', 'cifar100', 'imagenet', etc.)
        data_dir: Directory to store dataset
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if name == 'cifar10':
        return load_cifar10(data_dir, batch_size, num_workers)
    elif name == 'cifar100':
        return load_cifar100(data_dir, batch_size, num_workers)
    elif name == 'imagenet':
        return load_imagenet(data_dir, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def load_cifar10(
    data_dir: str,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset."""
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
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader

def load_cifar100(
    data_dir: str,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-100 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader

def load_imagenet(
    data_dir: str,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """Load ImageNet dataset."""
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Transforms for training and validation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        f"{data_dir}/train", transform=transform_train
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        f"{data_dir}/val", transform=transform_val
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
