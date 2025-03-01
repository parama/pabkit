# Process-Aware Benchmarking (PAB) Toolkit

A Python toolkit for evaluating machine learning models based on their learning trajectories rather than solely on final performance metrics.

## Overview

Process-Aware Benchmarking (PAB) is a theoretically grounded framework that shifts evaluation from static correctness to process-aware analysis. This toolkit implements the core concepts described in the PAB papers, allowing you to:

- Track learning trajectories throughout training
- Monitor class-wise learning progression
- Analyze adversarial robustness over time
- Detect memorization vs. generalization patterns
- Generate comprehensive reports with PAB metrics

## Installation

```bash
# From PyPI
pip install pab-toolkit

# From source
git clone https://github.com/pab-team/pab-toolkit.git
cd pab-toolkit
pip install -e .
```

## Quick Start

```python
from pab import PABTracker, track_training
import torch
import torchvision
from torchvision.models import resnet18

# Prepare data and model
trainloader, testloader = get_dataloaders()
model = resnet18(pretrained=False, num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Train with PAB tracking
model, tracker = track_training(
    model=model,
    train_loader=trainloader,
    val_loader=testloader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=100,
    device='cuda',
    num_classes=10,
    model_name='resnet18_cifar10',
    save_dir='./results'
)

# Generate final report
report = tracker.generate_report()
```

## Core Features

### 1. Learning Trajectory Tracking

PAB monitors how a model's performance evolves over time, not just its final accuracy. It tracks:

- Learning stability (smoothness of the loss trajectory)
- Generalization efficiency (train-val gap trends)
- Representation shifts (changes in feature space)

### 2. Class-wise Learning Progression

Different classes are learned at different rates and with varying difficulty. PAB monitors:

- Early learners (classes that converge quickly)
- Late learners (classes that take longer to learn)
- Unstable classes (classes with erratic learning patterns)

### 3. Adversarial Robustness Evolution

Robustness is not static. PAB tracks how adversarial vulnerability changes during training:

- Peak robustness epochs
- Robustness degradation detection
- Vulnerability gap analysis

### 4. Memorization vs. Generalization

PAB helps distinguish models that truly generalize from those that memorize, by tracking:

- Train-validation gap trends
- Overfitting signals
- Learning phase identification

## Examples

See the `examples` directory for complete usage examples:

- `examples/cifar10_demo.py`: Basic CIFAR-10 classification with PAB tracking
- `examples/imagenet_demo.py`: ImageNet classification with PAB

## Visualization

PAB generates plots to visualize learning dynamics:

- Learning curves with PAB insights
- Class-wise progression visualization
- Adversarial robustness trends

## Citation

If you use PAB in your research, please cite the original papers:

```bibtex
@inproceedings{
  author = {Anonymous},
  title = {Process Makes Perfect - Transforming How We Benchmark Machine Learning Models},
  booktitle = {International Conference on Machine Learning},
  year = {2025}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
