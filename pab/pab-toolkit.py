"""
Process-Aware Benchmarking (PAB) Toolkit

A Python library for evaluating machine learning models based on their learning trajectories
rather than solely on final performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable, Optional, Union, Any
import torch
from torch.utils.data import DataLoader
import time
import logging
import os
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PABTracker:
    """
    Main class for tracking model training and evaluation through the Process-Aware Benchmarking approach.
    
    This class implements the core functionality described in the PAB papers, including:
    - Learning trajectory stability
    - Generalization efficiency
    - Rule evolution metrics
    - Class-wise learning progression
    """
    
    def __init__(self, 
                 model_name: str,
                 save_dir: str = './pab_results',
                 checkpoint_freq: int = 5):
        """
        Initialize the PAB tracker.
        
        Args:
            model_name: Name of the model being tracked
            save_dir: Directory to save results
            checkpoint_freq: Frequency (in epochs) for detailed checkpointing and analysis
        """
        self.model_name = model_name
        self.save_dir = os.path.join(save_dir, model_name)
        self.checkpoint_freq = checkpoint_freq
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.epochs_tracked = []
        self.train_metrics = {}
        self.val_metrics = {}
        self.adv_metrics = {}  # For adversarial robustness
        self.representation_shifts = []  # For tracking feature representation shifts
        self.class_wise_metrics = {}  # For tracking class-wise learning progression
        
        logger.info(f"Initialized PAB tracker for model: {model_name}")
        logger.info(f"Results will be saved to: {self.save_dir}")

    def track_epoch(self, 
                   epoch: int, 
                   model: Any,
                   train_metrics: Dict[str, float],
                   val_metrics: Dict[str, float],
                   class_metrics: Optional[Dict[str, Dict[str, float]]] = None,
                   adv_metrics: Optional[Dict[str, float]] = None,
                   representation: Optional[np.ndarray] = None):
        """
        Track model performance and characteristics for a single epoch.
        
        Args:
            epoch: Current epoch number
            model: The model being tracked (for optional feature analysis)
            train_metrics: Dictionary of training metrics (e.g., {'loss': 0.5, 'accuracy': 0.8})
            val_metrics: Dictionary of validation metrics
            class_metrics: Optional dictionary of class-wise metrics
            adv_metrics: Optional dictionary of adversarial metrics
            representation: Optional representation vector/embedding for tracking representation shift
        """
        self.epochs_tracked.append(epoch)
        
        # Store metrics
        for key, value in train_metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)
            
        for key, value in val_metrics.items():
            if key not in self.val_metrics:
                self.val_metrics[key] = []
            self.val_metrics[key].append(value)
        
        if adv_metrics:
            for key, value in adv_metrics.items():
                if key not in self.adv_metrics:
                    self.adv_metrics[key] = []
                self.adv_metrics[key].append(value)
        
        if class_metrics:
            for class_name, metrics in class_metrics.items():
                if class_name not in self.class_wise_metrics:
                    self.class_wise_metrics[class_name] = {}
                for key, value in metrics.items():
                    if key not in self.class_wise_metrics[class_name]:
                        self.class_wise_metrics[class_name][key] = []
                    self.class_wise_metrics[class_name][key].append(value)
        
        # Track representation shifts if provided
        if representation is not None:
            self.representation_shifts.append(representation)
            
        # Every checkpoint_freq epochs, compute additional PAB metrics
        if epoch % self.checkpoint_freq == 0 or epoch == 1:
            logger.info(f"Computing PAB metrics for epoch {epoch}")
            self._compute_pab_metrics(epoch)
            
            # Save current state
            self._save_checkpoint(epoch)
    
    def _compute_pab_metrics(self, epoch: int) -> Dict[str, float]:
        """
        Compute Process-Aware Benchmarking metrics based on tracked data.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of PAB metrics
        """
        metrics = {}
        
        # Only compute if we have at least 2 epochs worth of data
        if len(self.epochs_tracked) < 2:
            return metrics
        
        # 1. Learning Trajectory Stability
        if 'loss' in self.train_metrics:
            # Based on equation 2 in the paper
            loss_diffs = np.abs(np.diff(self.train_metrics['loss']))
            metrics['learning_stability'] = np.mean(loss_diffs)
        
        # 2. Generalization Efficiency (train-val gap)
        if 'loss' in self.train_metrics and 'loss' in self.val_metrics:
            # Based on equation 3 in the paper
            train_loss = self.train_metrics['loss'][-1]
            val_loss = self.val_metrics['loss'][-1]
            metrics['generalization_gap'] = val_loss - train_loss
        
        # 3. Rule Evolution (representation shift)
        if len(self.representation_shifts) >= 2:
            # Based on equation 4 in the paper
            latest = self.representation_shifts[-1]
            previous = self.representation_shifts[-2]
            if isinstance(latest, np.ndarray) and isinstance(previous, np.ndarray):
                metrics['representation_shift'] = np.linalg.norm(latest - previous)
        
        # 4. Learning Curve Predictability
        if 'accuracy' in self.val_metrics and len(self.val_metrics['accuracy']) >= 2:
            # Based on equation 5 in the paper
            acc_diffs = np.diff(self.val_metrics['accuracy'])
            metrics['learning_predictability'] = np.mean(acc_diffs**2)
        
        logger.info(f"PAB metrics for epoch {epoch}: {metrics}")
        return metrics
    
    def _save_checkpoint(self, epoch: int):
        """Save the current state of PAB tracking"""
        checkpoint_data = {
            'model_name': self.model_name,
            'epoch': epoch,
            'epochs_tracked': self.epochs_tracked,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'class_wise_metrics': self.class_wise_metrics
        }
        
        # Add adversarial metrics if available
        if self.adv_metrics:
            checkpoint_data['adv_metrics'] = self.adv_metrics
        
        # Don't save the representation shifts as they might be large
        filename = os.path.join(self.save_dir, f'pab_checkpoint_epoch_{epoch}.json')
        with open(filename, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        
        logger.info(f"Saved PAB checkpoint to {filename}")
    
    def generate_report(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive PAB report with metrics and visualizations.
        
        Args:
            save_plots: Whether to save plots or just generate data
            
        Returns:
            Dictionary with PAB metrics and analysis
        """
        report = {
            'model_name': self.model_name,
            'total_epochs': max(self.epochs_tracked) if self.epochs_tracked else 0,
            'pab_metrics': {}
        }
        
        # 1. Overall trajectory analysis
        if self.train_metrics and self.val_metrics:
            if 'accuracy' in self.train_metrics and 'accuracy' in self.val_metrics:
                train_acc = self.train_metrics['accuracy']
                val_acc = self.val_metrics['accuracy']
                
                # Compute final metrics
                report['pab_metrics']['final_train_accuracy'] = train_acc[-1]
                report['pab_metrics']['final_val_accuracy'] = val_acc[-1]
                report['pab_metrics']['train_val_gap'] = train_acc[-1] - val_acc[-1]
                
                # Compute learning velocity (rate of improvement)
                if len(val_acc) > 5:
                    early_rate = (val_acc[4] - val_acc[0]) / 5
                    late_rate = (val_acc[-1] - val_acc[-6]) / 5
                    report['pab_metrics']['early_learning_rate'] = early_rate
                    report['pab_metrics']['late_learning_rate'] = late_rate
                    report['pab_metrics']['learning_decay'] = late_rate / early_rate if early_rate != 0 else float('inf')
                
                # Find epoch of peak validation performance
                peak_val_epoch = np.argmax(val_acc) + 1  # +1 because epochs are usually 1-indexed
                report['pab_metrics']['peak_val_epoch'] = peak_val_epoch
                report['pab_metrics']['peak_val_accuracy'] = max(val_acc)
                
                # Detect overfitting
                if peak_val_epoch < len(self.epochs_tracked):
                    report['pab_metrics']['overfitting_detected'] = peak_val_epoch < max(self.epochs_tracked)
                    
                # Plot learning curves
                if save_plots:
                    self._plot_learning_curves(report['pab_metrics']['peak_val_epoch'])
        
        # 2. Class-wise learning progression
        if self.class_wise_metrics:
            report['class_analysis'] = self._analyze_class_wise_learning()
            
            if save_plots:
                self._plot_class_wise_progression()
                
        # 3. Adversarial robustness analysis
        if self.adv_metrics:
            report['adversarial_analysis'] = self._analyze_adversarial_robustness()
            
            if save_plots:
                self._plot_adversarial_robustness()
        
        # Save the report
        report_path = os.path.join(self.save_dir, 'pab_final_report.json')
        with open(report_path, 'w') as f:
            # Filter out numpy arrays or other non-serializable objects
            clean_report = {k: v for k, v in report.items() if k != 'class_analysis_details'}
            json.dump(clean_report, f, indent=4)
        
        logger.info(f"Generated PAB report saved to {report_path}")
        return report
    
    def _analyze_class_wise_learning(self) -> Dict[str, Any]:
        """Analyze class-wise learning patterns"""
        analysis = {
            'early_learners': [],
            'late_learners': [],
            'unstable_classes': []
        }
        
        if not self.class_wise_metrics:
            return analysis
            
        # For each class, analyze learning progression
        for class_name, metrics in self.class_wise_metrics.items():
            if 'accuracy' not in metrics or len(metrics['accuracy']) < 3:
                continue
                
            acc = metrics['accuracy']
            
            # Early learners: classes that reach 80% of their final accuracy in the first third of training
            final_acc = acc[-1]
            early_thresh = 0.8 * final_acc
            early_epochs = [i for i, a in enumerate(acc[:len(acc)//3]) if a >= early_thresh]
            if early_epochs:
                analysis['early_learners'].append((class_name, self.epochs_tracked[early_epochs[0]]))
            
            # Late learners: classes that only reach 80% of their final accuracy in the last third of training
            late_epochs = [i for i, a in enumerate(acc[:2*len(acc)//3]) if a >= early_thresh]
            if not late_epochs and final_acc > 0.5:  # Only consider classes with reasonable final accuracy
                analysis['late_learners'].append(class_name)
            
            # Unstable classes: classes with high variance in accuracy
            if len(acc) > 5:
                acc_diffs = np.abs(np.diff(acc))
                if np.mean(acc_diffs) > 0.1:  # More than 10% average change between epochs
                    analysis['unstable_classes'].append(class_name)
        
        # Sort early learners by epoch
        analysis['early_learners'] = sorted(analysis['early_learners'], key=lambda x: x[1])
        
        return analysis
    
    def _analyze_adversarial_robustness(self) -> Dict[str, Any]:
        """Analyze adversarial robustness over time"""
        analysis = {}
        
        if not self.adv_metrics or 'accuracy' not in self.adv_metrics:
            return analysis
            
        adv_acc = self.adv_metrics['accuracy']
        
        # Find peak adversarial robustness
        peak_adv_epoch = np.argmax(adv_acc) + 1  # +1 because epochs are usually 1-indexed
        analysis['peak_adversarial_epoch'] = peak_adv_epoch
        analysis['peak_adversarial_accuracy'] = max(adv_acc)
        
        # Check if adversarial robustness degrades after peak
        if peak_adv_epoch < max(self.epochs_tracked):
            final_adv_acc = adv_acc[-1]
            analysis['robustness_degradation'] = max(adv_acc) - final_adv_acc
            analysis['degradation_detected'] = analysis['robustness_degradation'] > 0.05  # 5% threshold
        
        # Compare with validation accuracy
        if 'accuracy' in self.val_metrics:
            val_acc = self.val_metrics['accuracy']
            
            # Correlation between standard and adversarial accuracy
            analysis['adv_corr'] = np.corrcoef(val_acc, adv_acc)[0, 1] if len(val_acc) == len(adv_acc) else None
            
            # Vulnerability gap: standard vs adversarial
            analysis['vulnerability_gap'] = [v - a for v, a in zip(val_acc, adv_acc)]
            analysis['final_vulnerability_gap'] = val_acc[-1] - adv_acc[-1]
        
        return analysis
    
    def _plot_learning_curves(self, peak_epoch: Optional[int] = None):
        """Generate learning curve plots with PAB insights"""
        plt.figure(figsize=(12, 8))
        
        # Plot standard learning curves
        x = self.epochs_tracked
        
        if 'accuracy' in self.train_metrics:
            plt.plot(x, self.train_metrics['accuracy'], 'b-', label='Train Accuracy')
        if 'accuracy' in self.val_metrics:
            plt.plot(x, self.val_metrics['accuracy'], 'g-', label='Validation Accuracy')
        if 'accuracy' in self.adv_metrics:
            plt.plot(x, self.adv_metrics['accuracy'], 'r-', label='Adversarial Accuracy')
            
        # Mark peak validation performance
        if peak_epoch and peak_epoch in self.epochs_tracked:
            idx = self.epochs_tracked.index(peak_epoch)
            if 'accuracy' in self.val_metrics and idx < len(self.val_metrics['accuracy']):
                plt.axvline(x=peak_epoch, linestyle='--', color='k', alpha=0.5)
                plt.text(peak_epoch + 0.5, 0.5, f'Peak Validation: Epoch {peak_epoch}', 
                         rotation=90, verticalalignment='center')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Process-Aware Benchmarking: {self.model_name} Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, 'learning_curves.png'))
        plt.close()
    
    def _plot_class_wise_progression(self):
        """Generate plots showing class-wise learning progression"""
        if not self.class_wise_metrics:
            return
            
        # Select a subset of classes for clarity (e.g., early learners and late learners)
        analysis = self._analyze_class_wise_learning()
        plot_classes = []
        
        # Include some early learners
        for class_name, _ in analysis['early_learners'][:3]:
            plot_classes.append(class_name)
            
        # Include some late learners
        for class_name in analysis['late_learners'][:3]:
            plot_classes.append(class_name)
            
        # Include some unstable classes
        for class_name in analysis['unstable_classes'][:3]:
            if class_name not in plot_classes:
                plot_classes.append(class_name)
        
        # Plot class-wise accuracies
        plt.figure(figsize=(12, 8))
        
        for class_name in plot_classes:
            if 'accuracy' in self.class_wise_metrics[class_name]:
                plt.plot(self.epochs_tracked, 
                         self.class_wise_metrics[class_name]['accuracy'], 
                         label=f'Class: {class_name}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Class Accuracy')
        plt.title(f'Class-wise Learning Progression: {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, 'class_wise_progression.png'))
        plt.close()
    
    def _plot_adversarial_robustness(self):
        """Generate plots showing adversarial robustness over time"""
        if not self.adv_metrics or 'accuracy' not in self.adv_metrics:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot adversarial accuracy
        plt.plot(self.epochs_tracked, self.adv_metrics['accuracy'], 'r-', label='Adversarial Accuracy')
        
        # Plot standard accuracy for comparison
        if 'accuracy' in self.val_metrics:
            plt.plot(self.epochs_tracked, self.val_metrics['accuracy'], 'g-', label='Standard Accuracy')
        
        # Plot vulnerability gap
        if 'accuracy' in self.val_metrics:
            gap = [v - a for v, a in zip(self.val_metrics['accuracy'], self.adv_metrics['accuracy'])]
            plt.plot(self.epochs_tracked, gap, 'b--', label='Vulnerability Gap')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy / Gap')
        plt.title(f'Adversarial Robustness Over Time: {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, 'adversarial_robustness.png'))
        plt.close()


# Helper functions to extract model representations
def extract_representation(model, dataloader, device='cuda'):
    """
    Extract model's internal representation on a batch of data.
    For simplicity, we average the representations across samples.
    
    Args:
        model: The model to analyze
        dataloader: DataLoader to provide samples
        device: Device to run extraction on
        
    Returns:
        Numpy array of the average representation
    """
    if not hasattr(model, 'get_representation'):
        logger.warning("Model doesn't have a get_representation method, skipping representation extraction")
        return None
    
    model.eval()
    representations = []
    
    with torch.no_grad():
        # Only use a small batch for efficiency
        for i, (inputs, _) in enumerate(dataloader):
            if i >= 1:  # Just use one batch
                break
                
            inputs = inputs.to(device)
            representation = model.get_representation(inputs)
            
            if isinstance(representation, torch.Tensor):
                representations.append(representation.cpu().numpy())
    
    if representations:
        # Average across batches
        avg_representation = np.mean(np.concatenate(representations, axis=0), axis=0)
        return avg_representation
    return None


def get_class_metrics(model, dataloader, num_classes, device='cuda'):
    """
    Compute per-class metrics on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        num_classes: Number of classes
        device: Device to run evaluation on
        
    Returns:
        Dictionary of per-class metrics
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Compute per-class accuracy
    class_metrics = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            class_metrics[str(i)] = {'accuracy': accuracy, 'samples': class_total[i]}
    
    return class_metrics


def evaluate_adversarial_robustness(model, dataloader, attack_fn, device='cuda'):
    """
    Evaluate model's robustness against adversarial examples.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        attack_fn: Function to generate adversarial examples
        device: Device to run evaluation on
        
    Returns:
        Dictionary of adversarial metrics
    """
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        adv_inputs = attack_fn(model, inputs, targets)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return {'accuracy': accuracy, 'samples': total}


def track_training(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    num_epochs,
    device='cuda',
    num_classes=None,
    adv_attack_fn=None,
    adv_val_loader=None,
    model_name='model',
    save_dir='./pab_results',
    checkpoint_freq=5,
    extract_repr=False
):
    """
    End-to-end training function with PAB tracking.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to run training on
        num_classes: Number of classes (for class-wise tracking)
        adv_attack_fn: Function to generate adversarial examples
        adv_val_loader: DataLoader for adversarial evaluation
        model_name: Name for the model
        save_dir: Directory to save results
        checkpoint_freq: Frequency for detailed checkpointing
        extract_repr: Whether to extract model representations
        
    Returns:
        Trained model and PAB tracker
    """
    # Initialize PAB tracker
    tracker = PABTracker(model_name=model_name, save_dir=save_dir, checkpoint_freq=checkpoint_freq)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training phase
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': train_accuracy
        }
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Epoch {epoch} Validation'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': val_accuracy
        }
        
        # Class-wise metrics (if requested and num_classes provided)
        class_metrics = None
        if num_classes is not None:
            class_metrics = get_class_metrics(model, val_loader, num_classes, device)
        
        # Adversarial robustness (if requested)
        adv_metrics = None
        if adv_attack_fn is not None and adv_val_loader is not None:
            adv_metrics = evaluate_adversarial_robustness(model, adv_val_loader, adv_attack_fn, device)
        
        # Feature representation extraction (if requested)
        representation = None
        if extract_repr:
            representation = extract_representation(model, val_loader, device)
        
        # Track the epoch with PAB
        tracker.track_epoch(
            epoch=epoch,
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            class_metrics=class_metrics,
            adv_metrics=adv_metrics,
            representation=representation
        )
        
        # Print status
        logger.info(f'Epoch {epoch}: Train Loss: {train_metrics["loss"]:.4f}, '
                   f'Train Acc: {train_metrics["accuracy"]:.4f}, '
                   f'Val Loss: {val_metrics["loss"]:.4f}, '
                   f'Val Acc: {val_metrics["accuracy"]:.4f}')
    
    # Generate final PAB report
    tracker.generate_report()
    
    return model, tracker


# Example of a simple PGD adversarial attack function
def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, iters=10, device='cuda'):
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: The model to attack
        images: Clean images
        labels: True labels
        eps: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        
    Returns:
        Adversarial examples
    """
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    loss = torch.nn.CrossEntropyLoss()
    
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1)
    
    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()
        
        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    
    return adv_images
