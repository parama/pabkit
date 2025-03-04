# Process-Aware Benchmarking (PAB) Toolkit

A Python toolkit for evaluating machine learning models based on their learning trajectories rather than solely on final performance metrics.

## Overview

Traditional machine learning benchmarks focus on static evaluation, judging models by their final accuracy or other correctness-based metrics. However, this approach overlooks critical aspects of the learning process such as:

- How models refine their knowledge over time
- When and how generalization emerges
- Whether models truly learn structured representations or simply memorize training data
- How robustness evolves during training

Process-Aware Benchmarking (PAB) addresses these limitations by tracking the entire learning trajectory, providing deeper insights into model behavior and generalization capabilities.

## File Organization

```
pabkit/
├── README.md                     # Project overview and usage instructions
├── setup.py                      # Package installation configuration
├── requirements.txt              # Package dependencies
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Distribution manifest
│
├── pab/                          # Main package directory
│   ├── __init__.py               # Package initialization and imports
│   ├── core.py                   # Core PAB functionality and classes
│   ├── metrics.py                # Metrics calculations for trajectory analysis
│   ├── visualization.py          # Plotting and visualization utilities
│   ├── utils.py                  # Helper functions and utilities
│   ├── cli.py                    # Command-line interface
│   │
│   ├── tracking/                 # Model checkpoint tracking
│   │   ├── __init__.py
│   │   └── checkpoint_manager.py
│   │
│   ├── adversarial/              # Adversarial attack utilities
│   │   └── __init__.py
│   │
│   ├── datasets/                 # Dataset utilities
│   │   ├── __init__.py
│   │   └── imagenet.py
│   │
│   └── config/                   # Configuration management
│       ├── __init__.py
│       └── default_config.py
│
├── bin/                          # Command-line scripts
│   └── pab-cli                   # CLI entry point
│
├── examples/                     # Example scripts
│   ├── __init__.py
│   ├── simple_example.py         # Basic usage with CIFAR-10
│   ├── imagenet_case_study.py    # ImageNet case study from the paper
│   ├── model_comparison.py       # Comparing multiple models
│   ├── representation_analysis.py # Feature representation analysis
│   ├── comparative_analysis.py   # In-depth comparative analysis
│   └── pab_tutorial.ipynb        # Jupyter notebook tutorial
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_metrics.py
│   └── test_tracking.py
│
└── docs/                         # Documentation
    ├── usage.md                  # Detailed usage instructions
    ├── mathematical_formalism.md # Mathematical foundations of PAB
    └── api_reference.md          # API reference documentation
```

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

## Command-Line Interface

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

## Examples

Explore the `examples/` directory for detailed examples:

- `simple_example.py`: Basic usage with CIFAR-10
- `imagenet_case_study.py`: ImageNet case study from the paper
- `model_comparison.py`: Comparing multiple models
- `representation_analysis.py`: Feature representation analysis
- `pab_tutorial.ipynb`: Jupyter notebook tutorial

## Documentation

Refer to the `docs/` directory for detailed documentation:

- `usage.md`: Detailed usage instructions
- `mathematical_formalism.md`: Mathematical foundations of PAB
- `api_reference.md`: API reference documentation

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
