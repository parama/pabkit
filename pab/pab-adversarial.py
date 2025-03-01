"""
Adversarial attack implementations for Process-Aware Benchmarking.

This module provides implementations of common adversarial attacks
for evaluating model robustness over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, Tuple, Any

def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    norm: str = 'Linf',
    targeted: bool = False,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: The model to attack
        images: Clean images
        labels: True labels (for untargeted) or target labels (for targeted)
        epsilon: Attack strength parameter
        norm: Type of norm to use ('Linf' or 'L2')
        targeted: Whether to perform a targeted attack
        device: Device to run the attack on
        
    Returns:
        Adversarial examples
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    
    # Calculate loss
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Create adversarial examples
    with torch.no_grad():
        if targeted:
            # For targeted attacks, we want to minimize loss, so we move against the gradient
            sign_factor = -1
        else:
            # For untargeted attacks, we want to maximize loss, so we move with the gradient
            sign_factor = 1
            
        if norm == 'Linf':
            perturbed_images = images + sign_factor * epsilon * images.grad.sign()
        elif norm == 'L2':
            gradients = images.grad
            # Normalize gradients
            grad_norms = torch.norm(gradients.view(gradients.shape[0], -1), p=2, dim=1)
            normalized_gradients = gradients / (grad_norms.view(-1, 1, 1, 1) + 1e-10)
            perturbed_images = images + sign_factor * epsilon * normalized_gradients
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        
        # Ensure valid pixel range [0, 1]
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    iterations: int = 10,
    norm: str = 'Linf',
    targeted: bool = False,
    random_start: bool = True,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: The model to attack
        images: Clean images
        labels: True labels (for untargeted) or target labels (for targeted)
        epsilon: Maximum perturbation
        alpha: Step size
        iterations: Number of iterations
        norm: Type of norm to use ('Linf' or 'L2')
        targeted: Whether to perform a targeted attack
        random_start: Whether to start with random perturbation
        device: Device to run the attack on
        
    Returns:
        Adversarial examples
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Initialize with random perturbation if specified
    if random_start:
        if norm == 'Linf':
            adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        elif norm == 'L2':
            delta = torch.empty_like(images).normal_()
            # Normalize and scale
            delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
            delta = delta / (delta_norms.view(-1, 1, 1, 1) + 1e-10) * epsilon
            adv_images = images + delta
        else:
            raise ValueError(f"Unsupported norm: {norm}")
            
        adv_images = torch.clamp(adv_images, 0, 1)
    else:
        adv_images = images.clone()
    
    for _ in range(iterations):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        cost = loss_fn(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        cost.backward()
        
        # Update adversarial images
        with torch.no_grad():
            if targeted:
                # For targeted attacks, we want to minimize loss, so we move against the gradient
                sign_factor = -1
            else:
                # For untargeted attacks, we want to maximize loss, so we move with the gradient
                sign_factor = 1
                
            if norm == 'Linf':
                adv_images = adv_images + sign_factor * alpha * adv_images.grad.sign()
                # Project back to epsilon ball
                delta = torch.clamp(adv_images - images, -epsilon, epsilon)
                adv_images = images + delta
                
            elif norm == 'L2':
                gradients = adv_images.grad
                # Normalize gradients
                grad_norms = torch.norm(gradients.view(gradients.shape[0], -1), p=2, dim=1)
                normalized_gradients = gradients / (grad_norms.view(-1, 1, 1, 1) + 1e-10)
                
                # Update images
                adv_images = adv_images + sign_factor * alpha * normalized_gradients
                
                # Calculate norm of perturbations
                delta = adv_images - images
                delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
                
                # Project back to epsilon ball
                factor = torch.min(torch.ones_like(delta_norms), epsilon / (delta_norms + 1e-10))
                delta = delta * factor.view(-1, 1, 1, 1)
                adv_images = images + delta
            
            # Ensure valid pixel range [0, 1]
            adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images

def cw_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool = False,
    c: float = 1.0,
    kappa: float = 0,
    iterations: int = 100,
    lr: float = 0.01,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Carlini & Wagner (C&W) L2 attack.
    
    Args:
        model: The model to attack
        images: Clean images
        labels: True labels (for untargeted) or target labels (for targeted)
        targeted: Whether to perform a targeted attack
        c: Weighting for the loss term
        kappa: Confidence parameter
        iterations: Number of iterations
        lr: Learning rate
        device: Device to run the attack on
        
    Returns:
        Adversarial examples
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Change of variable: tanh(w) to ensure valid pixel range
    w = torch.zeros_like(images).to(device)
    w.requires_grad = True
    
    # Setup optimizer
    optimizer = torch.optim.Adam([w], lr=lr)
    
    # Original images
    orig_images = images.clone()
    
    def f(x):
        """Helper function for computing the objective."""
        outputs = model(x)
        one_hot = torch.zeros_like(outputs).scatter_(1, labels.unsqueeze(1), 1)
        
        # Targeted: minimize target class, maximize others
        # Untargeted: maximize target class, minimize others
        if targeted:
            target_logits = torch.sum(one_hot * outputs, dim=1)
            other_logits = torch.max((1 - one_hot) * outputs - one_hot * 10000, dim=1)[0]
            return torch.clamp(other_logits - target_logits + kappa, min=0)
        else:
            target_logits = torch.sum(one_hot * outputs, dim=1)
            other_logits = torch.max((1 - one_hot) * outputs, dim=1)[0]
            return torch.clamp(target_logits - other_logits + kappa, min=0)
    
    # Start attack iterations
    for _ in range(iterations):
        # Convert w to pixel space using tanh
        adv_images = 0.5 * (torch.tanh(w) + 1)
        
        # Compute loss
        l2_dist = torch.sum((adv_images - orig_images) ** 2, dim=(1, 2, 3))
        f_loss = f(adv_images)
        loss = c * f_loss + l2_dist
        
        # Update
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    
    # Final adversarial images
    with torch.no_grad():
        adv_images = 0.5 * (torch.tanh(w) + 1)
    
    return adv_images

def evaluate_adversarial_robustness(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    attack_fn: Callable,
    attack_params: dict = None,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate model's robustness against adversarial attacks.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        attack_fn: Function to generate adversarial examples
        attack_params: Parameters for the attack function
        device: Device to run evaluation on
        
    Returns:
        Dictionary with metrics on adversarial robustness
    """
    model.eval()
    
    # Default attack parameters
    if attack_params is None:
        attack_params = {}
    
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        adv_inputs = attack_fn(model, inputs, targets, **attack_params)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return {
        'accuracy': accuracy,
        'samples': total,
        'attack': attack_fn.__name__,
        'attack_params': attack_params
    }
