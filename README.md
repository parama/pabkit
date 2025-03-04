# Process-Aware Benchmarking (PAB) Toolkit

A Python toolkit for evaluating machine learning models based on their learning trajectories rather than solely on final performance metrics.

## Overview

Traditional machine learning benchmarks focus on static evaluation, judging models by their final accuracy or other correctness-based metrics. However, this approach overlooks critical aspects of the learning process such as:

- How models refine their knowledge over time
- When and how generalization emerges
- Whether models truly learn structured representations or simply memorize training data
- How robustness evolves during training

Process-Aware Benchmarking (PAB) addresses these limitations by tracking the entire learning trajectory, providing deeper insights into model behavior and generalization capabilities.

## Installation

```bash
pip install pabkit
```

Or install from source:

```bash
git clone https://github.com/yourusername/pabkit.git
cd pabkit
pip install -e .
```

## Quick Start

Here's a simple example of using PAB to track a model's learning trajectory:

```python
from pab import ProcessAwareBenchmark, track_learning_curve
import torch
import torchvision

# Load a model and dataset
model = torchvision.models.resnet18(pretrained=False)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
                                            transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, 
                                           transform=torchvision.transforms.ToTensor())

# Track the learning trajectory
pab = track_learning_curve(
    model=model,
    dataset=(train_dataset, test_dataset),
    epochs=100,
    batch_size=128
)

# Evaluate the trajectory
results = pab.evaluate_trajectory()
print(pab.summarize())
```

## Key Features

### Learning Trajectory Analysis

PAB tracks how models evolve during training, capturing metrics like:

- Loss and accuracy curves
- Generalization gap over time
- Class-wise learning progression
- Feature representation shifts

### Robustness Evaluation

Track how model robustness changes during training:

- Adversarial robustness over epochs
- Consistency under transformations
- Stability of decision boundaries

### Visualization Tools

Visualize learning dynamics with built-in plotting functions:

- Learning curves
- Class progression
- Robustness evolution
- Generalization gap

### Checkpoint Management

Efficiently manage model checkpoints across training:

- Save checkpoints at regular intervals
- Load and compare checkpoints
- Prune checkpoints to save disk space

## Core Components

### ProcessAwareBenchmark

The main class for tracking and analyzing learning trajectories:

```python
from pab import ProcessAwareBenchmark

pab = ProcessAwareBenchmark(
    checkpoint_dir='./checkpoints',
    save_frequency=5,
    track_representations=True
)

# Track each epoch
for epoch in range(1, epochs+1):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    # Track metrics
    pab.track_epoch(
        model=model,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc
    )

# Evaluate the learning trajectory
results = pab.evaluate_trajectory()
```

### Visualization

Create insightful visualizations of learning trajectories:

```python
from pab.visualization import plot_learning_trajectory, plot_class_progression

# Plot learning curves
fig = plot_learning_trajectory(
    train_losses=pab.metrics['train_loss'],
    val_losses=pab.metrics['val_loss'],
    train_accs=pab.metrics['train_acc'],
    val_accs=pab.metrics['val_acc'],
    save_path='learning_curves.png'
)

# Plot class-wise learning progression
fig = plot_class_progression(
    class_accuracies=pab.metrics['class_accuracy'],
    save_path='class_progression.png'
)
```

## Advanced Usage

### Adversarial Robustness Tracking

```python
from pab.utils import evaluate_adversarial_robustness

# Track adversarial robustness
adversarial_metrics = evaluate_adversarial_robustness(
    model=model,
    clean_loader=test_loader,
    epsilon=0.03
)

# Add to PAB metrics
pab.metrics['adversarial_robustness'].append(adversarial_metrics['adversarial_accuracy'])
```

### Feature Representation Analysis

```python
from pab.utils import get_feature_extractor

# Create feature extractor
feature_extractor = get_feature_extractor(
    model=model,
    loader=test_loader,
    layer_name='layer4'  # For ResNet
)

# Track representation evolution
representations = feature_extractor(model)
```

### Comparing Models

```python
from pab import compare_models

# Compare models using PAB metrics
results = compare_models(
    model_dirs=['./checkpoints/model1', './checkpoints/model2'],
    names=['ResNet18', 'EfficientNet']
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use the PAB toolkit in your research, please cite:

```
@article{pab2025,
  title={Process-Aware Benchmarking: A Novel, Theoretically Grounded Approach to Benchmarking for Modern Machine Learning},
  author={Pal, Parama},
  journal={SIGKDD 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
