"""
Utility functions for Process-Aware Benchmarking (PAB).
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

logger = logging.getLogger(__name__)

def extract_feature_representations(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    layer_name: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_samples: int = 100
) -> np.ndarray:
    """
    Extract feature representations from a specific layer of the model.
    
    Args:
        model: PyTorch model
        loader: DataLoader containing samples
        layer_name: Name of the layer to extract features from (if None, uses the penultimate layer)
        device: Device to run the model on
        num_samples: Number of samples to use
        
    Returns:
        Numpy array of feature representations
    """
    model = model.to(device)
    model.eval()
    
    # Storage for activations
    activations = []
    
    # Register hook to extract activations
    if layer_name is not None:
        # Find the layer by name
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            logger.warning(f"Layer {layer_name} not found. Using penultimate layer.")
            # Fall back to penultimate layer
            layer_name = None
    
    if layer_name is None:
        # Use the penultimate layer (assume the last layer is the classifier)
        # For common architectures like ResNet, this would be the avg pool layer
        if hasattr(model, 'avgpool'):
            target_layer = model.avgpool
        elif hasattr(model, 'features'):
            # For VGG-like architectures
            target_layer = model.features[-1]
        else:
            # Try to find the layer before the classifier
            found = False
            for name, module in model.named_children():
                if name == 'classifier' or name == 'fc':
                    found = True
                    break
                target_layer = module
            
            if not found:
                logger.warning("Could not determine penultimate layer. Using model output.")
                # Just return the model output
                with torch.no_grad():
                    all_features = []
                    count = 0
                    
                    for inputs, _ in loader:
                        if count >= num_samples:
                            break
                        
                        inputs = inputs.to(device)
                        features = model(inputs)
                        all_features.append(features.cpu().numpy())
                        
                        count += inputs.size(0)
                    
                    return np.vstack(all_features)
    
    # Register forward hook
    activations = []
    
    def hook_fn(module, input, output):
        # Store the output of the layer
        activations.append(output.detach().cpu())
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    # Extract features
    with torch.no_grad():
        count = 0
        for inputs, _ in loader:
            if count >= num_samples:
                break
            
            inputs = inputs.to(device)
            _ = model(inputs)
            
            count += inputs.size(0)
    
    # Remove the hook
    hook.remove()
    
    # Process the activations
    if activations:
        # Concatenate activations
        all_activations = torch.cat(activations, dim=0)
        
        # Flatten if needed
        if len(all_activations.shape) > 2:
            all_activations = all_activations.view(all_activations.size(0), -1)
        
        return all_activations.numpy()
    else:
        logger.error("No activations captured.")
        return np.array([])


def generate_adversarial_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epsilon: float = 0.03,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate adversarial examples using Fast Gradient Sign Method (FGSM).
    
    Args:
        model: PyTorch model
        loader: DataLoader containing samples
        epsilon: Perturbation magnitude
        device: Device to run the model on
        
    Returns:
        Tuple of (adversarial_examples, true_labels)
    """
    model = model.to(device)
    model.eval()
    
    # Storage for adversarial examples
    adv_examples = []
    true_labels = []
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        
        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Collect gradients
        data_grad = inputs.grad.data
        
        # Create adversarial examples
        perturbed_inputs = inputs + epsilon * data_grad.sign()
        
        # Clamp to ensure valid pixel range [0, 1]
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        # Store examples
        adv_examples.append(perturbed_inputs.detach().cpu())
        true_labels.append(targets.cpu())
    
    return torch.cat(adv_examples), torch.cat(true_labels)


def evaluate_adversarial_robustness(
    model: torch.nn.Module,
    clean_loader: torch.utils.data.DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    epsilon: float = 0.03
) -> Dict[str, float]:
    """
    Evaluate adversarial robustness of a model.
    
    Args:
        model: PyTorch model
        clean_loader: DataLoader containing clean samples
        device: Device to run the model on
        epsilon: Perturbation magnitude
        
    Returns:
        Dictionary of robustness metrics
    """
    model = model.to(device)
    model.eval()
    
    # Evaluate on clean data
    clean_correct = 0
    clean_total = 0
    
    with torch.no_grad():
        for inputs, targets in clean_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(targets).sum().item()
            clean_total += targets.size(0)
    
    clean_accuracy = clean_correct / clean_total
    
    # Generate adversarial examples and evaluate
    adv_examples, true_labels = generate_adversarial_examples(
        model, clean_loader, epsilon, device
    )
    
    adv_correct = 0
    adv_total = 0
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(adv_examples), 64):  # Process in batches
            batch_inputs = adv_examples[i:i+64].to(device)
            batch_targets = true_labels[i:i+64].to(device)
            
            outputs = model(batch_inputs)
            _, predicted = outputs.max(1)
            adv_correct += predicted.eq(batch_targets).sum().item()
            adv_total += batch_targets.size(0)
    
    adv_accuracy = adv_correct / adv_total
    
    return {
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'robustness_ratio': adv_accuracy / clean_accuracy,
        'accuracy_drop': clean_accuracy - adv_accuracy
    }


def compute_class_accuracies(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[int, float]:
    """
    Compute per-class accuracies.
    
    Args:
        model: PyTorch model
        loader: DataLoader containing samples
        num_classes: Number of classes
        device: Device to run the model on
        
    Returns:
        Dictionary mapping class IDs to accuracies
    """
    model = model.to(device)
    model.eval()
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            correct = predicted.eq(targets)
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    # Compute accuracies
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = class_correct[i] / class_total[i]
    
    return class_accuracies


def get_feature_extractor(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    layer_name: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Callable:
    """
    Create a function that extracts features from a model.
    
    Args:
        model: PyTorch model
        loader: DataLoader to use for feature extraction
        layer_name: Name of the layer to extract features from
        device: Device to run the model on
        
    Returns:
        Function that takes a model and returns feature representations
    """
    def extractor(model: torch.nn.Module) -> np.ndarray:
        return extract_feature_representations(
            model, loader, layer_name, device
        )
    
    return extractor


def calculate_consistency(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    transforms: List[Callable],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """
    Calculate prediction consistency under different transformations.
    
    Args:
        model: PyTorch model
        loader: DataLoader containing samples
        transforms: List of transformation functions to apply
        device: Device to run the model on
        
    Returns:
        Consistency score (higher is better)
    """
    model = model.to(device)
    model.eval()
    
    consistent = 0
    total = 0
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            
            # Get original predictions
            outputs = model(inputs)
            _, original_preds = outputs.max(1)
            
            # Check if predictions are consistent across transformations
            for transform in transforms:
                transformed_inputs = transform(inputs)
                outputs = model(transformed_inputs)
                _, transformed_preds = outputs.max(1)
                
                # Check consistency
                consistent += (original_preds == transformed_preds).sum().item()
                total += original_preds.size(0)
    
    return consistent / total if total > 0 else 0.0


def export_metrics_to_json(metrics: Dict[str, Any], path: str):
    """
    Export PAB metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics to export
        path: Path to save the JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy arrays and tensors to lists
    cleaned_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            cleaned_metrics[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            cleaned_metrics[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, dict):
            # Handle nested dictionaries
            cleaned_metrics[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    cleaned_metrics[key][subkey] = subvalue.tolist()
                elif isinstance(subvalue, torch.Tensor):
                    cleaned_metrics[key][subkey] = subvalue.detach().cpu().numpy().tolist()
                elif isinstance(subvalue, list) and subvalue and isinstance(subvalue[0], (np.ndarray, torch.Tensor)):
                    cleaned_metrics[key][subkey] = [
                        item.tolist() if isinstance(item, np.ndarray) 
                        else item.detach().cpu().numpy().tolist() if isinstance(item, torch.Tensor)
                        else item
                        for item in subvalue
                    ]
                else:
                    cleaned_metrics[key][subkey] = subvalue
        elif isinstance(value, list) and value and isinstance(value[0], (np.ndarray, torch.Tensor)):
            cleaned_metrics[key] = [
                item.tolist() if isinstance(item, np.ndarray) 
                else item.detach().cpu().numpy().tolist() if isinstance(item, torch.Tensor)
                else item
                for item in value
            ]
        else:
            cleaned_metrics[key] = value
    
    # Write to file
    with open(path, 'w') as f:
        json.dump(cleaned_metrics, f, indent=2)
    
    logger.info(f"Metrics exported to {path}")
