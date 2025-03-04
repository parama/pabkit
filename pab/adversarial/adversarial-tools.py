"""
Adversarial attack and evaluation tools for Process-Aware Benchmarking (PAB).

This module provides utilities for evaluating model robustness against adversarial attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional, Callable

def fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 0.03,
    device: torch.device = None
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: Model to attack
        inputs: Input images (B, C, H, W)
        targets: True labels
        epsilon: Perturbation magnitude
        device: Device to run on
        
    Returns:
        Adversarial examples
    """
    if device is None:
        device = next(model.parameters()).device
    
    inputs = inputs.clone().detach().to(device)
    targets = targets.clone().detach().to(device)
    inputs.requires_grad = True
    
    # Forward pass
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    
    # Calculate gradients
    model.zero_grad()
    loss.backward()
    
    # Create adversarial examples
    data_grad = inputs.grad.data
    sign_data_grad = data_grad.sign()
    
    # Add perturbation
    perturbed_inputs = inputs + epsilon * sign_data_grad
    
    # Clamp to ensure valid pixel range [0, 1]
    perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    
    return perturbed_inputs

def pgd_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_steps: int = 10,
    random_start: bool = True,
    device: torch.device = None
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: Model to attack
        inputs: Input images (B, C, H, W)
        targets: True labels
        epsilon: Perturbation magnitude
        alpha: Step size
        num_steps: Number of steps
        random_start: Use random initialization
        device: Device to run on
        
    Returns:
        Adversarial examples
    """
    if device is None:
        device = next(model.parameters()).device
    
    inputs = inputs.clone().detach().to(device)
    targets = targets.clone().detach().to(device)
    
    # Random initialization
    if random_start:
        delta = torch.rand_like(inputs, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(inputs, requires_grad=True)
    
    # PGD attack
    for _ in range(num_steps):
        outputs = model(inputs + delta)
        loss = F.cross_entropy(outputs, targets)
        
        # Calculate gradients
        model.zero_grad()
        loss.backward()
        
        # Update delta
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    
    # Create adversarial examples
    perturbed_inputs = (inputs + delta).detach()
    
    # Clamp to ensure valid pixel range [0, 1]
    perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    
    return perturbed_inputs

def evaluate_adversarial_robustness(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    attack_type: str = 'fgsm',
    epsilon: float = 0.03,
    device: torch.device = None,
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model robustness against adversarial attacks.
    
    Args:
        model: Model to evaluate
        loader: DataLoader containing test samples
        attack_type: 'fgsm' or 'pgd'
        epsilon: Perturbation magnitude
        device: Device to run on
        num_batches: Number of batches to evaluate (if None, use all)
        
    Returns:
        Dictionary of robustness metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Evaluate on clean data
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(targets).sum().item()
            
            # Generate adversarial examples
            if attack_type.lower() == 'fgsm':
                perturbed_inputs = fgsm_attack(model, inputs, targets, epsilon, device)
            elif attack_type.lower() == 'pgd':
                perturbed_inputs = pgd_attack(model, inputs, targets, epsilon, device=device)
            else:
                raise ValueError(f"Unsupported attack type: {attack_type}")
            
            # Evaluate on adversarial data
            with torch.no_grad():
                outputs = model(perturbed_inputs)
                _, predicted = outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
    
    clean_accuracy = clean_correct / total
    adv_accuracy = adv_correct / total
    
    return {
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'robustness_ratio': adv_accuracy / clean_accuracy,
        'accuracy_drop': clean_accuracy - adv_accuracy,
        'attack_type': attack_type,
        'epsilon': epsilon,
        'total_samples': total
    }

def get_vulnerable_classes(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    attack_type: str = 'fgsm',
    epsilon: float = 0.03,
    device: torch.device = None,
    threshold: float = 0.5  # Minimum accuracy drop to consider a class vulnerable
) -> Dict[int, float]:
    """
    Identify classes that are particularly vulnerable to adversarial attacks.
    
    Args:
        model: Model to evaluate
        loader: DataLoader containing test samples
        attack_type: 'fgsm' or 'pgd'
        epsilon: Perturbation magnitude
        device: Device to run on
        threshold: Minimum accuracy drop to consider a class vulnerable
        
    Returns:
        Dictionary mapping class IDs to vulnerability scores (accuracy drop)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    class_clean_correct = {}
    class_adv_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Evaluate on clean data
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                
                if label not in class_total:
                    class_total[label] = 0
                    class_clean_correct[label] = 0
                    class_adv_correct[label] = 0
                
                class_total[label] += 1
                class_clean_correct[label] += predicted[i].eq(targets[i]).item()
            
            # Generate adversarial examples
            if attack_type.lower() == 'fgsm':
                perturbed_inputs = fgsm_attack(model, inputs, targets, epsilon, device)
            elif attack_type.lower() == 'pgd':
                perturbed_inputs = pgd_attack(model, inputs, targets, epsilon, device=device)
            else:
                raise ValueError(f"Unsupported attack type: {attack_type}")
            
            # Evaluate on adversarial data
            with torch.no_grad():
                outputs = model(perturbed_inputs)
                _, predicted = outputs.max(1)
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_adv_correct[label] += predicted[i].eq(targets[i]).item()
    
    # Calculate vulnerability scores
    vulnerable_classes = {}
    
    for label in class_total:
        if class_total[label] > 0:
            clean_acc = class_clean_correct[label] / class_total[label]
            adv_acc = class_adv_correct[label] / class_total[label]
            
            accuracy_drop = clean_acc - adv_acc
            
            if accuracy_drop >= threshold:
                vulnerable_classes[label] = accuracy_drop
    
    return vulnerable_classes
