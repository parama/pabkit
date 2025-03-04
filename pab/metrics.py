"""
Process-Aware Benchmarking (PAB) metrics.

This module implements various metrics for evaluating learning trajectories
in the PAB framework.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any

def learning_stability(previous_loss: float, current_loss: float) -> float:
    """
    Calculate learning trajectory stability.
    
    A robust model should demonstrate a gradual and structured learning process.
    This metric measures the smoothness of the loss trajectory.
    
    Args:
        previous_loss: Loss at previous epoch
        current_loss: Loss at current epoch
        
    Returns:
        Stability metric value (lower is more stable)
    """
    # Smaller values indicate a more stable learning trajectory
    return abs(previous_loss - current_loss) / (previous_loss + 1e-8)


def generalization_efficiency(train_loss: float, val_loss: float) -> float:
    """
    Calculate instantaneous generalization efficiency.
    
    This measures how well the model generalizes at the current point in training.
    
    Args:
        train_loss: Training loss
        val_loss: Validation loss
        
    Returns:
        Generalization efficiency (lower is better)
    """
    # Smaller gap indicates better generalization
    return val_loss - train_loss


def rule_evolution(
    previous_repr: Any,
    current_repr: Any
) -> float:
    """
    Measure rule formation divergence between two feature representations.
    
    Learning should involve the refinement of structured abstractions
    rather than abrupt shifts in representation.
    
    Args:
        previous_repr: Previous feature representation
        current_repr: Current feature representation
        
    Returns:
        Rule evolution metric (lower means smoother evolution)
    """
    # Convert to numpy if tensors
    if isinstance(previous_repr, torch.Tensor):
        previous_repr = previous_repr.detach().cpu().numpy()
    if isinstance(current_repr, torch.Tensor):
        current_repr = current_repr.detach().cpu().numpy()
    
    # Flatten if necessary
    if len(previous_repr.shape) > 1:
        previous_repr = previous_repr.reshape(-1)
    if len(current_repr.shape) > 1:
        current_repr = current_repr.reshape(-1)
    
    # Normalize to unit length
    prev_norm = np.linalg.norm(previous_repr)
    curr_norm = np.linalg.norm(current_repr)
    
    if prev_norm > 0:
        previous_repr = previous_repr / prev_norm
    if curr_norm > 0:
        current_repr = current_repr / curr_norm
    
    # Calculate L2 distance between normalized representations
    return np.linalg.norm(previous_repr - current_repr)


def class_wise_progression(
    class_accuracies: Dict[int, List[float]],
    threshold: float = 0.8
) -> Tuple[List[int], List[int], List[int]]:
    """
    Analyze class-wise learning progression over time.
    
    Identifies which classes are learned early, which are learned late,
    and which show unstable learning.
    
    Args:
        class_accuracies: Dictionary mapping class IDs to lists of accuracies over time
        threshold: Accuracy threshold for considering a class "learned"
        
    Returns:
        Tuple of (early_classes, late_classes, unstable_classes)
    """
    early_classes = []
    late_classes = []
    unstable_classes = []
    
    num_epochs = max(len(accs) for accs in class_accuracies.values())
    early_cutoff = num_epochs // 3
    late_cutoff = (2 * num_epochs) // 3
    
    for class_id, accuracies in class_accuracies.items():
        # Ensure we have accuracies for all epochs
        if len(accuracies) < num_epochs:
            # Pad with zeros
            accuracies = accuracies + [0] * (num_epochs - len(accuracies))
        
        # Find first epoch where accuracy exceeds threshold
        try:
            first_learned = next(i for i, acc in enumerate(accuracies) if acc >= threshold)
        except StopIteration:
            # Never reaches threshold
            late_classes.append(class_id)
            continue
        
        # Check for stability - does accuracy drop below threshold after learning?
        unstable = any(acc < threshold for acc in accuracies[first_learned:])
        
        if unstable:
            unstable_classes.append(class_id)
        
        # Categorize by when class was first learned
        if first_learned < early_cutoff:
            early_classes.append(class_id)
        elif first_learned >= late_cutoff:
            late_classes.append(class_id)
    
    return early_classes, late_classes, unstable_classes


def robustness_evolution(
    clean_accuracies: List[float],
    adversarial_accuracies: List[float]
) -> Dict[str, Any]:
    """
    Analyze how robustness evolves over training.
    
    Args:
        clean_accuracies: List of accuracies on clean test data
        adversarial_accuracies: List of accuracies on adversarially perturbed data
        
    Returns:
        Dictionary of robustness metrics
    """
    if len(clean_accuracies) != len(adversarial_accuracies):
        raise ValueError("Clean and adversarial accuracy lists must have the same length")
    
    # Convert to numpy arrays
    clean = np.array(clean_accuracies)
    adv = np.array(adversarial_accuracies)
    
    # Calculate robustness ratio (adversarial acc / clean acc)
    robustness_ratio = adv / (clean + 1e-8)
    
    # Find peak robustness
    peak_idx = np.argmax(adv)
    peak_epoch = peak_idx + 1  # 1-indexed epochs
    peak_value = adv[peak_idx]
    
    # Robustness at the end of training
    final_value = adv[-1]
    
    # Degradation from peak (if any)
    degradation = (peak_value - final_value) / (peak_value + 1e-8)
    
    return {
        'peak_epoch': peak_epoch,
        'peak_value': peak_value,
        'final_value': final_value,
        'degradation': degradation,
        'robustness_ratio': robustness_ratio.tolist()
    }


def representation_similarity(repr1: Any, repr2: Any) -> float:
    """
    Calculate similarity between two model representations.
    
    This can be used to compare representations between epochs
    or between different models.
    
    Args:
        repr1: First representation
        repr2: Second representation
        
    Returns:
        Similarity score (higher means more similar)
    """
    # Convert to numpy if tensors
    if isinstance(repr1, torch.Tensor):
        repr1 = repr1.detach().cpu().numpy()
    if isinstance(repr2, torch.Tensor):
        repr2 = repr2.detach().cpu().numpy()
    
    # Flatten if necessary
    if len(repr1.shape) > 1:
        repr1 = repr1.reshape(-1)
    if len(repr2.shape) > 1:
        repr2 = repr2.reshape(-1)
    
    # Normalize to unit length
    repr1_norm = np.linalg.norm(repr1)
    repr2_norm = np.linalg.norm(repr2)
    
    if repr1_norm > 0:
        repr1 = repr1 / repr1_norm
    if repr2_norm > 0:
        repr2 = repr2 / repr2_norm
    
    # Calculate cosine similarity
    return np.dot(repr1, repr2)


def learning_curve_predictability(loss_trajectory: List[float]) -> float:
    """
    Calculate learning curve predictability.
    
    In human learning, expertise develops through structured exposure
    and refinement. This metric assesses whether a model follows a
    similarly structured trajectory.
    
    Args:
        loss_trajectory: List of loss values over training
        
    Returns:
        Predictability score (lower is more predictable/human-like)
    """
    if len(loss_trajectory) < 3:
        return 0.0
    
    # Calculate differences between consecutive losses
    diffs = [loss_trajectory[i] - loss_trajectory[i-1] for i in range(1, len(loss_trajectory))]
    
    # Calculate variance of differences (lower variance = more predictable)
    return np.var(diffs)


def overfitting_risk(train_losses: List[float], val_losses: List[float]) -> Dict[str, Any]:
    """
    Assess the risk of overfitting based on training and validation loss trajectories.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        
    Returns:
        Dictionary with overfitting risk metrics
    """
    if len(train_losses) != len(val_losses):
        raise ValueError("Training and validation loss lists must have the same length")
    
    # Ensure we have lists of floats
    train_losses = list(map(float, train_losses))
    val_losses = list(map(float, val_losses))
    
    # Calculate generalization gap at each epoch
    gaps = [val - train for val, train in zip(val_losses, val_losses)]
    
    # Calculate trend in gap (is it increasing?)
    gap_trend = np.polyfit(np.arange(len(gaps)), gaps, 1)[0]
    
    # Find minimum validation loss and corresponding epoch
    min_val_idx = np.argmin(val_losses)
    min_val_epoch = min_val_idx + 1  # 1-indexed epochs
    
    # Calculate how much validation loss increased from minimum
    if min_val_idx < len(val_losses) - 1:
        val_degradation = (val_losses[-1] - val_losses[min_val_idx]) / val_losses[min_val_idx]
    else:
        val_degradation = 0.0
    
    return {
        'gap_trend': gap_trend,
        'early_stopping_epoch': min_val_epoch,
        'validation_degradation': val_degradation,
        'overfitting_detected': gap_trend > 0 and val_degradation > 0.05
    }
