"""
Visualization tools for Process-Aware Benchmarking (PAB).

This module provides functions for visualizing various aspects of
model learning trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from matplotlib.figure import Figure


def plot_learning_trajectory(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Learning Trajectory",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot learning trajectory showing loss and accuracy curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: Optional list of training accuracies
        val_accs: Optional list of validation accuracies
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    epochs = range(1, len(train_losses) + 1)
    
    # Create plots based on what data is available
    if train_accs is not None and val_accs is not None:
        # Both loss and accuracy
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        # Just loss
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = None
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracies if available
    if ax2 is not None and train_accs is not None and val_accs is not None:
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_class_progression(
    class_accuracies: Dict[int, List[float]],
    class_names: Optional[Dict[int, str]] = None,
    num_classes_to_show: int = 5,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot class-wise learning progression.
    
    Args:
        class_accuracies: Dictionary mapping class IDs to lists of accuracies
        class_names: Optional dictionary mapping class IDs to class names
        num_classes_to_show: Number of interesting classes to show
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    # Find most interesting classes to show:
    # - Early learning classes (fastest to converge)
    # - Late learning classes (slowest to converge)
    
    # First, calculate convergence time for each class
    convergence_times = {}
    num_epochs = max(len(accs) for accs in class_accuracies.values())
    threshold = 0.7  # Accuracy threshold for "learned"
    
    for class_id, accuracies in class_accuracies.items():
        # Pad accuracies if needed
        if len(accuracies) < num_epochs:
            accuracies = accuracies + [accuracies[-1]] * (num_epochs - len(accuracies))
        
        # Find first epoch where accuracy exceeds threshold
        try:
            conv_time = next(i for i, acc in enumerate(accuracies) if acc >= threshold)
        except StopIteration:
            # Never converges
            conv_time = num_epochs
        
        convergence_times[class_id] = conv_time
    
    # Sort classes by convergence time
    sorted_classes = sorted(convergence_times.keys(), key=lambda c: convergence_times[c])
    
    # Select early and late classes
    early_classes = sorted_classes[:min(num_classes_to_show, len(sorted_classes)//2)]
    late_classes = sorted_classes[-min(num_classes_to_show, len(sorted_classes)//2):]
    
    classes_to_show = early_classes + late_classes
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, num_epochs + 1)
    
    # Plot early classes with solid lines
    for class_id in early_classes:
        accs = class_accuracies[class_id]
        # Pad if needed
        if len(accs) < num_epochs:
            accs = accs + [accs[-1]] * (num_epochs - len(accs))
        
        label = f"Class {class_id}"
        if class_names and class_id in class_names:
            label = class_names[class_id]
            
        ax.plot(epochs, accs, '-', linewidth=2, label=f"{label} (Early)")
    
    # Plot late classes with dashed lines
    for class_id in late_classes:
        accs = class_accuracies[class_id]
        # Pad if needed
        if len(accs) < num_epochs:
            accs = accs + [accs[-1]] * (num_epochs - len(accs))
        
        label = f"Class {class_id}"
        if class_names and class_id in class_names:
            label = class_names[class_id]
            
        ax.plot(epochs, accs, '--', linewidth=2, label=f"{label} (Late)")
    
    # Add threshold line
    ax.axhline(y=threshold, color='r', linestyle=':', label=f"Learning Threshold ({threshold})")
    
    ax.set_title('Class-wise Learning Progression')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_robustness_curve(
    clean_accuracies: List[float],
    adversarial_accuracies: List[float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot adversarial robustness evolution over training.
    
    Args:
        clean_accuracies: List of accuracies on clean test data
        adversarial_accuracies: List of accuracies on adversarially perturbed data
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    epochs = range(1, len(clean_accuracies) + 1)
    
    # Plot clean and adversarial accuracies
    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color=color1)
    line1 = ax1.plot(epochs, clean_accuracies, color=color1, label='Clean Accuracy')
    line2 = ax1.plot(epochs, adversarial_accuracies, 'tab:orange', label='Adversarial Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for robustness ratio
    color2 = 'tab:green'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Robustness Ratio', color=color2)
    
    # Calculate robustness ratio (adversarial acc / clean acc)
    robustness_ratio = [adv / (clean + 1e-8) for adv, clean in zip(adversarial_accuracies, clean_accuracies)]
    line3 = ax2.plot(epochs, robustness_ratio, color=color2, linestyle='--', label='Robustness Ratio')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Find peak robustness
    peak_idx = np.argmax(adversarial_accuracies)
    ax1.axvline(x=peak_idx + 1, color='r', linestyle=':', label=f'Peak Robustness (Epoch {peak_idx + 1})')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    
    ax1.set_title('Adversarial Robustness Evolution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_generalization_gap(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot generalization gap over training.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: Optional list of training accuracies
        val_accs: Optional list of validation accuracies
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(train_losses) + 1)
    
    # Loss gap
    loss_gap = [val - train for val, train in zip(val_losses, train_losses)]
    ax1.plot(epochs, loss_gap, 'b-', label='Val Loss - Train Loss')
    ax1.set_title('Loss Generalization Gap')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss Gap')
    
    # Find minimum validation loss
    min_val_idx = np.argmin(val_losses)
    ax1.axvline(x=min_val_idx + 1, color='r', linestyle=':', 
                label=f'Min Val Loss (Epoch {min_val_idx + 1})')
    
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy gap (if available)
    if train_accs is not None and val_accs is not None:
        acc_gap = [train - val for train, val in zip(train_accs, val_accs)]
        ax2.plot(epochs, acc_gap, 'g-', label='Train Acc - Val Acc')
        ax2.set_title('Accuracy Generalization Gap')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy Gap')
        
        # Find maximum validation accuracy
        max_val_idx = np.argmax(val_accs)
        ax2.axvline(x=max_val_idx + 1, color='r', linestyle=':', 
                    label=f'Max Val Acc (Epoch {max_val_idx + 1})')
        
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        ax2.set_visible(False)
    
    plt.suptitle('Generalization Analysis')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pab_summary(
    metrics: Dict[str, Any],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a comprehensive PAB summary visualization.
    
    Args:
        metrics: Dictionary containing all PAB metrics
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Extract required metrics
    train_losses = metrics.get('train_loss', [])
    val_losses = metrics.get('val_loss', [])
    train_accs = metrics.get('train_acc', [])
    val_accs = metrics.get('val_acc', [])
    stability = metrics.get('stability', [])
    gen_efficiency = metrics.get('gen_efficiency', [])
    class_accuracy = metrics.get('class_accuracy', {})
    adv_robustness = metrics.get('adversarial_robustness', [])
    
    if not train_losses or not val_losses:
        return fig
    
    epochs = range(1, len(train_losses) + 1)
    
    # 2x2 grid of subplots
    ax1 = fig.add_subplot(2, 2, 1)  # Learning curves
    ax2 = fig.add_subplot(2, 2, 2)  # Stability
    ax3 = fig.add_subplot(2, 2, 3)  # Class progression
    ax4 = fig.add_subplot(2, 2, 4)  # Robustness or gen gap
    
    # 1. Learning curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    
    if train_accs and val_accs:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, train_accs, 'b--', alpha=0.7, label='Train Acc')
        ax1_twin.plot(epochs, val_accs, 'r--', alpha=0.7, label='Val Acc')
        ax1_twin.set_ylabel('Accuracy')
    
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Stability
    if stability:
        ax2.plot(epochs[1:], stability, 'g-', label='Learning Stability')
        ax2.set_title('Learning Stability')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Stability (lower is better)')
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        ax2.set_visible(False)
    
    # 3. Class progression (select a few interesting classes)
    if class_accuracy:
        num_epochs = max(len(accs) for accs in class_accuracy.values())
        
        # Find early and late classes
        convergence_times = {}
        for class_id, accs in class_accuracy.items():
            # Find first epoch where accuracy exceeds 0.7
            try:
                conv_time = next(i for i, acc in enumerate(accs) if acc >= 0.7)
            except StopIteration:
                conv_time = num_epochs
            convergence_times[class_id] = conv_time
        
        # Get 3 fastest and 3 slowest classes
        sorted_classes = sorted(convergence_times.keys(), key=lambda c: convergence_times[c])
        classes_to_show = sorted_classes[:3] + sorted_classes[-3:]
        
        for class_id in classes_to_show:
            accs = class_accuracy[class_id]
            is_early = class_id in sorted_classes[:3]
            style = '-' if is_early else '--'
            label = f"Class {class_id} ({'Early' if is_early else 'Late'})"
            
            # Pad if needed
            if len(accs) < num_epochs:
                accs = accs + [accs[-1]] * (num_epochs - len(accs))
                
            ax3.plot(range(1, len(accs) + 1), accs, style, label=label)
        
        ax3.set_title('Class-wise Learning Progression')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        ax3.legend(loc='lower right')
        ax3.grid(True, linestyle='--', alpha=0.7)
    else:
        ax3.set_visible(False)
    
    # 4. Robustness or gen gap
    if adv_robustness:
        # Robustness curve
        ax4.plot(epochs, val_accs, 'b-', label='Clean Accuracy')
        ax4.plot(epochs, adv_robustness, 'r-', label='Adversarial Accuracy')
        
        # Find peak robustness
        peak_idx = np.argmax(adv_robustness)
        ax4.axvline(x=peak_idx + 1, color='g', linestyle=':', 
                    label=f'Peak Robustness (Epoch {peak_idx + 1})')
        
        ax4.set_title('Adversarial Robustness')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy')
        ax4.legend(loc='lower right')
        ax4.grid(True, linestyle='--', alpha=0.7)
    elif train_losses and val_losses:
        # Generalization gap
        gen_gap = [val - train for val, train in zip(val_losses, train_losses)]
        ax4.plot(epochs, gen_gap, 'g-', label='Generalization Gap')
        ax4.set_title('Generalization Gap')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Val Loss - Train Loss')
        ax4.grid(True, linestyle='--', alpha=0.7)
    else:
        ax4.set_visible(False)
    
    plt.suptitle('Process-Aware Benchmarking (PAB) Summary', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_learning_trajectories(
    model_metrics: Dict[str, Dict[str, Any]],
    metric_name: str = 'val_acc',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare learning trajectories of multiple models.
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
        metric_name: Name of the metric to compare ('val_acc', 'train_loss', etc.)
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define line styles and colors for different models
    line_styles = ['-', '--', '-.', ':']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        if metric_name in metrics and metrics[metric_name]:
            values = metrics[metric_name]
            epochs = range(1, len(values) + 1)
            
            style_idx = i % len(line_styles)
            color_idx = i % len(colors)
            
            ax.plot(epochs, values, 
                    linestyle=line_styles[style_idx],
                    color=colors[color_idx], 
                    linewidth=2,
                    label=model_name)
    
    # Set plot details
    metric_display_name = metric_name.replace('_', ' ').title()
    ax.set_title(f'Comparing {metric_display_name} Across Models')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_display_name)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig