"""
Dataset utilities for Process-Aware Benchmarking (PAB).

This package provides datasets and utilities for PAB experiments and analyses.
"""

from .imagenet import (
    load_imagenet, 
    get_imagenet_classes, 
    ImageNetSubset, 
    create_challenging_subset,
    IMAGENET_MEAN,
    IMAGENET_STD
)

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
    import torch
    import torchvision
    import torchvision.transforms as transforms
    
    if dataset_name.lower() == 'imagenet':
        return load_imagenet(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size
        )
    
    elif dataset_name.lower() == 'cifar10':
        # CIFAR-10 transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        
        # Create subset if specified
        if subset_size is not None:
            import numpy as np
            if subset_size > len(train_dataset):
                print(f"Warning: Requested subset size {subset_size} is larger than dataset size {len(train_dataset)}.")
                subset_size = len(train_dataset)
            
            # Create subset for training data
            train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            
            # Create subset for test data
            test_subset_size = min(subset_size // 5, len(test_dataset))
            test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # CIFAR-10 class names
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
        
        return train_loader, test_loader, class_names
    
    elif dataset_name.lower() == 'cifar100':
        # CIFAR-100 transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-100 dataset
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        
        # Create subset if specified
        if subset_size is not None:
            import numpy as np
            if subset_size > len(train_dataset):
                print(f"Warning: Requested subset size {subset_size} is larger than dataset size {len(train_dataset)}.")
                subset_size = len(train_dataset)
            
            # Create subset for training data
            train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            
            # Create subset for test data
            test_subset_size = min(subset_size // 5, len(test_dataset))
            test_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # CIFAR-100 classes (use indices as we don't want to include all 100 class names here)
        class_names = {i: f"Class {i}" for i in range(100)}
        
        # Try to load actual class names
        try:
            class_names.update({
                i: name for i, name in enumerate(train_dataset.classes)
            }) if hasattr(train_dataset, 'classes') else None
        except:
            pass
        
        return train_loader, test_loader, class_names
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
