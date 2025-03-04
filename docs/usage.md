# Process-Aware Benchmarking (PAB) Usage Guide

This document provides detailed instructions for using the Process-Aware Benchmarking (PAB) toolkit in various scenarios.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Tracking Learning Curves](#tracking-learning-curves)
4. [Class-wise Learning Analysis](#class-wise-learning-analysis)
5. [Adversarial Robustness Tracking](#adversarial-robustness-tracking)
6. [Feature Representation Analysis](#feature-representation-analysis)
7. [Model Comparison](#model-comparison)
8. [Using the Command-Line Interface](#using-the-command-line-interface)
9. [Advanced Configuration](#advanced-configuration)
10. [Troubleshooting](#troubleshooting)

## Installation

You can install the PAB toolkit using pip:

```bash
pip install pabkit
```

Or install from source:

```bash
git clone https://github.com/yourusername/pabkit.git
cd pabkit
pip install -e .
```

## Basic Usage

Here's a simple example of how to integrate PAB into your training loop:

```python
from pab import ProcessAwareBenchmark

# Initialize PAB
pab = ProcessAwareBenchmark(
    checkpoint_dir='./checkpoints',
    save_frequency=5,
    track_representations=True
)

# In your training loop
for epoch in range(1, num_epochs+1):
    # Train for one epoch
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    
    # Evaluate
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    # Track metrics with PAB
    pab.track_epoch(
        model=model,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc
    )
    
    # Print progress
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# Evaluate the learning trajectory
eval_results = pab.evaluate_trajectory()
print(pab.summarize())
```

## Tracking Learning Curves

PAB can track and visualize learning curves to help you understand how your model's performance evolves during training:

```python
from pab.visualization import plot_learning_trajectory

# After training
plot_learning_trajectory(
    train_losses=pab.metrics['train_loss'],
    val_losses=pab.metrics['val_loss'],
    train_accs=pab.metrics['train_acc'],
    val_accs=pab.metrics['val_acc'],
    save_path='learning_curves.png'
)
```

## Class-wise Learning Analysis

PAB can track how quickly different classes are learned:

```python
from pab.utils import compute_class_accuracies
from pab.visualization import plot_class_progression

# During evaluation in the training loop
class_accuracies = compute_class_accuracies(
    model, val_loader, num_classes=num_classes, device=device
)

# Track in PAB
pab.track_epoch(
    model=model,
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    train_acc=train_acc,
    val_acc=val_acc,
    class_accuracies=class_accuracies
)

# After training, visualize class progression
plot_class_progression(
    class_accuracies=pab.metrics['class_accuracy'],
    class_names=class_names,  # Optional dict mapping class IDs to names
    save_path='class_progression.png'
)
```

## Adversarial Robustness Tracking

PAB can track how your model's robustness to adversarial examples evolves during training:

```python
from pab.utils import evaluate_adversarial_robustness
from pab.visualization import plot_robustness_curve

# Periodically during training
adv_metrics = evaluate_adversarial_robustness(
    model, val_loader, device=device, epsilon=0.03
)

# Track in PAB
pab.track_epoch(
    model=model,
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    train_acc=train_acc,
    val_acc=val_acc,
    adversarial_acc=adv_metrics['adversarial_accuracy']
)

# After training, visualize robustness evolution
plot_robustness_curve(
    clean_accuracies=pab.metrics['val_acc'],
    adversarial_accuracies=pab.metrics['adversarial_robustness'],
    save_path='robustness_evolution.png'
)
```

## Feature Representation Analysis

PAB allows you to track how your model's feature representations evolve during training:

```python
from pab.utils import extract_feature_representations

# During training
features = extract_feature_representations(
    model, val_loader, layer_name='layer4', device=device
)

# Define a function that returns these features
feature_extractor = lambda m: features

# Track in PAB
pab.track_epoch(
    model=model,
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    train_acc=train_acc,
    val_acc=val_acc,
    feature_extractor=feature_extractor
)
```

## Model Comparison

PAB makes it easy to compare multiple models:

```python
from pab import compare_models
from pab.visualization import compare_learning_trajectories

# Compare models
comparison_results = compare_models(
    model_dirs=['./checkpoints/model1', './checkpoints/model2'],
    names=['ResNet50', 'EfficientNet']
)

# Visualize comparison
compare_learning_trajectories(
    model_metrics={
        'ResNet50': model1_metrics,
        'EfficientNet': model2_metrics
    },
    metric_name='val_acc',
    save_path='model_comparison.png'
)
```

## Using the Command-Line Interface

PAB provides a command-line interface for common tasks:

```bash
# Analyze checkpoints from a trained model
pab-cli analyze --checkpoint_dir ./checkpoints --output_dir ./results

# Compare multiple models
pab-cli compare --model_dirs ./checkpoints/model1 ./checkpoints/model2 --model_names ResNet50 EfficientNet

# Visualize metrics
pab-cli visualize --metrics_file ./results/pab_metrics.json --type learning_curve

# Generate a comprehensive report
pab-cli report --checkpoint_dir ./checkpoints --model_name ResNet50
```

## Advanced Configuration

PAB can be customized for specific use cases:

```python
# Custom checkpointing
from pab.tracking import CheckpointManager

checkpoint_manager = CheckpointManager('./checkpoints')

# Save a checkpoint
checkpoint_manager.save_checkpoint(
    model, 
    epoch=10, 
    metrics={'train_loss': 0.5, 'val_acc': 0.95}
)

# Prune checkpoints to save space
checkpoint_manager.prune_checkpoints(
    keep_last_n=5,  # Keep the 5 most recent checkpoints
    keep_every_n=10  # Keep every 10th checkpoint for older epochs
)
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'pab'

Make sure you've installed the package:

```bash
pip install pabkit
```

#### "No checkpoints found" Error

Ensure your checkpoint directory contains valid checkpoints:

```python
from pab.tracking import CheckpointManager

# List available checkpoints
checkpoint_manager = CheckpointManager('./checkpoints')
checkpoints = checkpoint_manager.list_checkpoints()
print(checkpoints)
```

#### Out of Memory Errors

When working with large models, you might encounter memory issues. Try:

```python
# Use sparse checkpointing
pab = ProcessAwareBenchmark(
    checkpoint_dir='./checkpoints',
    save_frequency=10  # Save less frequently
)

# Use smaller batches for feature extraction
from pab.utils import extract_feature_representations

features = extract_feature_representations(
    model, val_loader, 
    num_samples=100  # Use fewer samples
)
```

For more help, please refer to the API documentation or open an issue on GitHub.
