"""
Core functionality for Process-Aware Benchmarking (PAB).

This module provides the main interfaces for evaluating models
using the PAB framework.
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable

from .metrics import (
    learning_stability,
    generalization_efficiency,
    rule_evolution
)
from .tracking import CheckpointManager

logger = logging.getLogger(__name__)

class ProcessAwareBenchmark:
    """
    Main class for Process-Aware Benchmarking (PAB) evaluation.
    
    This class provides a comprehensive interface for tracking and analyzing
    a model's learning trajectory over time.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = './pab_checkpoints',
        save_frequency: int = 5,
        track_representations: bool = True
    ):
        """
        Initialize a Process-Aware Benchmark instance.
        
        Args:
            checkpoint_dir: Directory to store model checkpoints
            save_frequency: How often to save model checkpoints (in epochs)
            track_representations: Whether to track feature representations
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.track_representations = track_representations
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'stability': [],
            'gen_efficiency': [],
            'rule_evolution': [],
            'class_accuracy': {},
            'adversarial_robustness': []
        }
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def track_epoch(
        self,
        model,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        class_accuracies: Optional[Dict[int, float]] = None,
        adversarial_acc: Optional[float] = None,
        feature_extractor: Optional[Callable] = None
    ):
        """
        Track a single training epoch.
        
        Args:
            model: The model being evaluated
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            train_acc: Training accuracy for this epoch
            val_acc: Validation accuracy for this epoch
            class_accuracies: Per-class accuracy dictionary {class_id: accuracy}
            adversarial_acc: Accuracy on adversarially perturbed data (if available)
            feature_extractor: Function to extract feature representations from model
        """
        # Record basic metrics
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        
        # Calculate PAB-specific metrics
        if len(self.metrics['train_loss']) > 1:
            # Stability metric
            stability = learning_stability(
                self.metrics['val_loss'][-2],
                self.metrics['val_loss'][-1]
            )
            self.metrics['stability'].append(stability)
            
            # Generalization efficiency
            gen_eff = generalization_efficiency(
                self.metrics['train_loss'][-1],
                self.metrics['val_loss'][-1]
            )
            self.metrics['gen_efficiency'].append(gen_eff)
            
            # Rule evolution (if we're tracking representations)
            if self.track_representations and feature_extractor is not None:
                # Extract current feature representation
                curr_repr = feature_extractor(model)
                
                # If we have a previous representation, calculate rule evolution
                if hasattr(self, 'prev_repr'):
                    rule_evol = rule_evolution(self.prev_repr, curr_repr)
                    self.metrics['rule_evolution'].append(rule_evol)
                
                # Save current representation as previous for next epoch
                self.prev_repr = curr_repr
        
        # Record class-wise accuracies
        if class_accuracies:
            for class_id, acc in class_accuracies.items():
                if class_id not in self.metrics['class_accuracy']:
                    self.metrics['class_accuracy'][class_id] = []
                self.metrics['class_accuracy'][class_id].append(acc)
        
        # Record adversarial robustness
        if adversarial_acc is not None:
            self.metrics['adversarial_robustness'].append(adversarial_acc)
        
        # Save checkpoint if needed
        if epoch % self.save_frequency == 0:
            self.checkpoint_manager.save_checkpoint(model, epoch, self.metrics)
    
    def evaluate_trajectory(self) -> Dict[str, Dict]:
        """
        Evaluate the overall learning trajectory.
        
        Returns:
            Dictionary of PAB evaluation metrics
        """
        if len(self.metrics['train_loss']) < 2:
            logger.warning("Not enough epochs tracked to evaluate trajectory")
            return {}
        
        results = {}
        
        # Overall learning stability
        results['overall_stability'] = {
            'mean': np.mean(self.metrics['stability']),
            'std': np.std(self.metrics['stability']),
            'max': np.max(self.metrics['stability']) if self.metrics['stability'] else None,
            'min': np.min(self.metrics['stability']) if self.metrics['stability'] else None
        }
        
        # Generalization dynamics
        train_losses = np.array(self.metrics['train_loss'])
        val_losses = np.array(self.metrics['val_loss'])
        gen_gap = val_losses - train_losses
        
        results['generalization'] = {
            'final_gap': gen_gap[-1] if len(gen_gap) > 0 else None,
            'max_gap': np.max(gen_gap) if len(gen_gap) > 0 else None,
            'gap_trend': 'increasing' if gen_gap[-1] > gen_gap[0] else 'decreasing',
            'early_stopping_epoch': np.argmin(val_losses) + 1
        }
        
        # Class-wise learning patterns
        if self.metrics['class_accuracy']:
            early_classes = []
            late_classes = []
            
            for class_id, acc_list in self.metrics['class_accuracy'].items():
                if len(acc_list) < len(self.metrics['train_loss']):
                    # Pad with zeros if necessary
                    acc_list = acc_list + [0] * (len(self.metrics['train_loss']) - len(acc_list))
                
                # Find epoch where accuracy exceeds 80%
                conv_epoch = next((i for i, acc in enumerate(acc_list) if acc >= 0.8), len(acc_list))
                
                # Early learners converge in first third of training
                if conv_epoch < len(acc_list) // 3:
                    early_classes.append(class_id)
                # Late learners converge in last third of training
                elif conv_epoch > 2 * len(acc_list) // 3:
                    late_classes.append(class_id)
            
            results['class_patterns'] = {
                'early_classes': early_classes,
                'late_classes': late_classes,
                'num_early': len(early_classes),
                'num_late': len(late_classes)
            }
        
        # Adversarial robustness trend
        if self.metrics['adversarial_robustness']:
            rob = np.array(self.metrics['adversarial_robustness'])
            rob_peak_epoch = np.argmax(rob) + 1
            
            results['robustness'] = {
                'peak_epoch': rob_peak_epoch,
                'peak_value': np.max(rob),
                'final_value': rob[-1],
                'degradation': (np.max(rob) - rob[-1]) / np.max(rob) if np.max(rob) > 0 else 0
            }
        
        return results
    
    def summarize(self) -> str:
        """
        Generate a human-readable summary of the PAB evaluation.
        
        Returns:
            String summary of the PAB results
        """
        eval_results = self.evaluate_trajectory()
        if not eval_results:
            return "Not enough data to generate PAB summary."
        
        lines = ["Process-Aware Benchmarking (PAB) Summary", "="*50, ""]
        
        # Training overview
        lines.append(f"Training duration: {len(self.metrics['train_loss'])} epochs")
        lines.append(f"Final validation accuracy: {self.metrics['val_acc'][-1]:.4f}")
        lines.append("")
        
        # Learning stability
        stability = eval_results.get('overall_stability', {})
        if stability:
            lines.append("Learning Stability:")
            lines.append(f"  Mean: {stability.get('mean', 0):.4f} ± {stability.get('std', 0):.4f}")
            lines.append(f"  Max instability: {stability.get('max', 0):.4f}")
            lines.append("")
        
        # Generalization
        gen = eval_results.get('generalization', {})
        if gen:
            lines.append("Generalization:")
            lines.append(f"  Final train-validation gap: {gen.get('final_gap', 0):.4f}")
            lines.append(f"  Gap trend: {gen.get('gap_trend', 'unknown')}")
            lines.append(f"  Optimal early stopping: epoch {gen.get('early_stopping_epoch', 0)}")
            lines.append("")
        
        # Class patterns
        class_pat = eval_results.get('class_patterns', {})
        if class_pat:
            lines.append("Class-wise Learning Patterns:")
            lines.append(f"  Early-learning classes: {len(class_pat.get('early_classes', []))} classes")
            lines.append(f"  Late-learning classes: {len(class_pat.get('late_classes', []))} classes")
            lines.append("")
        
        # Robustness
        rob = eval_results.get('robustness', {})
        if rob:
            lines.append("Adversarial Robustness:")
            lines.append(f"  Peak robustness: {rob.get('peak_value', 0):.4f} at epoch {rob.get('peak_epoch', 0)}")
            lines.append(f"  Final robustness: {rob.get('final_value', 0):.4f}")
            degradation = rob.get('degradation', 0) * 100
            lines.append(f"  Robustness degradation: {degradation:.2f}%")
            lines.append("")
        
        # PAB recommendations
        lines.append("PAB Recommendations:")
        
        if gen and gen.get('gap_trend') == 'increasing':
            lines.append("  • Model shows signs of overfitting, consider early stopping or regularization")
        
        if rob and rob.get('degradation', 0) > 0.1:
            lines.append("  • Adversarial robustness peaks before final epoch, suggesting robustness-accuracy tradeoff")
        
        if stability and stability.get('std', 0) > 0.1:
            lines.append("  • Training exhibits instability, consider more stable optimization strategy")
        
        return "\n".join(lines)


def track_learning_curve(
    model,
    dataset,
    epochs: int = 100,
    batch_size: int = 256,
    optimizer = None,
    criterion = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir: str = './pab_checkpoints',
    save_frequency: int = 5,
    adversarial_test: bool = False
) -> ProcessAwareBenchmark:
    """
    Track the learning curve of a model during training.

    This is a convenience function that handles training the model
    while tracking PAB metrics.
    
    Args:
        model: The model to train and evaluate
        dataset: Dataset to use for training and validation
        epochs: Number of epochs to train
        batch_size: Batch size for training
        optimizer: Optimizer to use (if None, uses Adam)
        criterion: Loss function (if None, uses CrossEntropyLoss)
        device: Device to use for training ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        save_frequency: How often to save checkpoints (in epochs)
        adversarial_test: Whether to evaluate adversarial robustness
        
    Returns:
        ProcessAwareBenchmark instance with tracked metrics
    """
    # Implementation depends on your specific needs and would typically
    # include dataset splitting, training loop, validation, etc.
    # This is a placeholder for the full implementation

    # Initialize PAB
    pab = ProcessAwareBenchmark(
        checkpoint_dir=checkpoint_dir,
        save_frequency=save_frequency
    )
    
    # For now, we'll just return the PAB instance
    return pab


def evaluate_trajectory(checkpoint_dir: str) -> Dict:
    """
    Evaluate a model trajectory from saved checkpoints.
    
    Args:
        checkpoint_dir: Directory containing saved checkpoints
        
    Returns:
        Dictionary of PAB evaluation metrics
    """
    # Load checkpoints
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return {}
    
    # Load metrics from the last checkpoint
    _, metrics = checkpoint_manager.load_checkpoint(checkpoints[-1])
    
    # Create PAB instance and populate with loaded metrics
    pab = ProcessAwareBenchmark(checkpoint_dir=checkpoint_dir)
    pab.metrics = metrics
    
    # Evaluate trajectory
    return pab.evaluate_trajectory()


def compare_models(
    model_dirs: List[str],
    names: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compare multiple models using PAB metrics.
    
    Args:
        model_dirs: List of directories containing model checkpoints
        names: Optional list of model names (for better reporting)
        
    Returns:
        Dictionary of comparative PAB metrics
    """
    if names is None:
        names = [f"Model_{i}" for i in range(len(model_dirs))]
    
    if len(names) != len(model_dirs):
        raise ValueError("Number of names must match number of model directories")
    
    results = {}
    
    for name, model_dir in zip(names, model_dirs):
        eval_results = evaluate_trajectory(model_dir)
        if eval_results:
            results[name] = eval_results
    
    return results
