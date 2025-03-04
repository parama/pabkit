"""
Comparative analysis of multiple models using Process-Aware Benchmarking (PAB).

This script provides an in-depth comparison of different model architectures
based on their learning trajectories, generalization patterns, and robustness.
"""

import os
import sys
import argparse
import json
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Any

# Add parent directory to path for importing PAB
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pab import ProcessAwareBenchmark, compare_models
from pab.visualization import (
    compare_learning_trajectories,
    plot_pab_summary
)
from pab.tracking import CheckpointManager
from pab.utils import compute_class_accuracies, representation_similarity

# Parse arguments
parser = argparse.ArgumentParser(description='Comparative PAB Analysis')
parser.add_argument('--model_dirs', nargs='+', required=True,
                   help='List of directories containing model checkpoints')
parser.add_argument('--model_names', nargs='+', 
                   help='List of model names (for better reporting)')
parser.add_argument('--dataset', type=str, default='imagenet',
                   choices=['cifar10', 'cifar100', 'imagenet'],
                   help='Dataset used for training')
parser.add_argument('--output_dir', type=str, default='./results/comparative',
                   help='Directory to save comparison results')
parser.add_argument('--metrics', nargs='+', default=['val_acc', 'train_loss'],
                   help='Metrics to compare (e.g., val_acc, train_loss)')
parser.add_argument('--load_checkpoint', action='store_true',
                   help='Load models from checkpoints for detailed analysis')
parser.add_argument('--epochs', nargs='+', type=int,
                   help='Specific epochs to analyze for each model')
parser.add_argument('--detailed', action='store_true',
                   help='Perform detailed analysis (slower)')
parser.add_argument('--report_format', type=str, default='md',
                   choices=['md', 'txt', 'json'],
                   help='Format for comparison report')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Validate arguments
if args.model_names is None:
    args.model_names = [f"Model_{i}" for i in range(len(args.model_dirs))]

if len(args.model_names) != len(args.model_dirs):
    print("Warning: Number of model names does not match number of model directories.")
    args.model_names = [f"Model_{i}" for i in range(len(args.model_dirs))]

# Load metrics for each model
model_metrics = {}
checkpoint_managers = {}

for i, (model_dir, model_name) in enumerate(zip(args.model_dirs, args.model_names)):
    # Create checkpoint manager
    checkpoint_managers[model_name] = CheckpointManager(model_dir)
    checkpoints = checkpoint_managers[model_name].list_checkpoints()
    
    if not checkpoints:
        print(f"Warning: No checkpoints found for {model_name} in {model_dir}")
        continue
    
    # Load metrics from the last checkpoint
    _, metrics = checkpoint_managers[model_name].load_checkpoint(checkpoints[-1])
    model_metrics[model_name] = metrics
    
    print(f"Loaded metrics for {model_name}: {len(metrics.get('train_loss', []))} epochs")

# Compare models using PAB
print("Comparing models using PAB...")
comparison_results = compare_models(
    model_dirs=args.model_dirs,
    names=args.model_names
)

# Save comparison results
with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
    json.dump(comparison_results, f, indent=2)

# Create visualizations for each metric
for metric in args.metrics:
    if all(metric in metrics for metrics in model_metrics.values()):
        print(f"Creating comparison plot for {metric}...")
        fig = compare_learning_trajectories(
            model_metrics=model_metrics,
            metric_name=metric,
            figsize=(12, 8),
            save_path=os.path.join(args.output_dir, f'comparison_{metric}.png')
        )
    else:
        print(f"Warning: Metric '{metric}' not available for all models.")

# Create generalization gap comparison
if all('train_loss' in metrics and 'val_loss' in metrics for metrics in model_metrics.values()):
    print("Creating generalization gap comparison...")
    plt.figure(figsize=(12, 8))
    
    for i, (name, metrics) in enumerate(model_metrics.items()):
        train_losses = metrics['train_loss']
        val_losses = metrics['val_loss']
        epochs = range(1, len(train_losses) + 1)
        
        # Calculate generalization gap
        gen_gap = [val - train for val, train in zip(val_losses, train_losses)]
        
        plt.plot(epochs, gen_gap, linewidth=2, label=name)
    
    plt.xlabel('Epochs')
    plt.ylabel('Generalization Gap (Val Loss - Train Loss)')
    plt.title('Generalization Gap Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(args.output_dir, 'generalization_gap_comparison.png'), dpi=300)

# Create stability comparison
if all('stability' in metrics for metrics in model_metrics.values()):
    print("Creating stability comparison...")
    plt.figure(figsize=(12, 8))
    
    for i, (name, metrics) in enumerate(model_metrics.items()):
        stability = metrics['stability']
        epochs = range(2, len(stability) + 2)  # Stability starts from epoch 2
        
        plt.plot(epochs, stability, linewidth=2, label=name)
    
    plt.xlabel('Epochs')
    plt.ylabel('Learning Stability')
    plt.title('Learning Stability Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(args.output_dir, 'stability_comparison.png'), dpi=300)

# Create robustness comparison
if all('adversarial_robustness' in metrics and len(metrics['adversarial_robustness']) > 0 for metrics in model_metrics.values()):
    print("Creating robustness comparison...")
    plt.figure(figsize=(12, 8))
    
    for i, (name, metrics) in enumerate(model_metrics.items()):
        # Get epochs where adversarial robustness was measured
        adv_epochs = []
        for j in range(len(metrics['val_acc'])):
            if j < len(metrics['adversarial_robustness']) and metrics['adversarial_robustness'][j] is not None:
                adv_epochs.append(j + 1)
        
        # Get adversarial robustness values (filtering None values)
        adv_robustness = [metrics['adversarial_robustness'][j-1] for j in adv_epochs]
        
        plt.plot(adv_epochs, adv_robustness, 'o-', linewidth=2, label=name)
    
    plt.xlabel('Epochs')
    plt.ylabel('Adversarial Accuracy')
    plt.title('Adversarial Robustness Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(args.output_dir, 'robustness_comparison.png'), dpi=300)

# Create class learning comparison (early vs. late learning classes)
if args.detailed and all('class_accuracy' in metrics and metrics['class_accuracy'] for metrics in model_metrics.values()):
    print("Creating class learning comparison...")
    
    # Get total number of classes
    num_classes = max(len(metrics['class_accuracy']) for metrics in model_metrics.values())
    
    # Create scatter plot of class-wise learning times
    plt.figure(figsize=(14, 10))
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_metrics)))
    
    for i, (name, metrics) in enumerate(model_metrics.items()):
        class_learning_times = {}
        
        for class_id, accuracies in metrics['class_accuracy'].items():
            # Find first epoch where accuracy exceeds 0.7
            try:
                learning_time = next(j for j, acc in enumerate(accuracies) if acc >= 0.7) + 1
            except (StopIteration, IndexError):
                learning_time = len(accuracies) + 1  # Never reached threshold
            
            class_learning_times[class_id] = learning_time
        
        # Sort classes by learning time
        sorted_classes = sorted(class_learning_times.items(), key=lambda x: x[1])
        
        # Plot class learning times
        plt.scatter(
            [item[0] for item in sorted_classes],
            [item[1] for item in sorted_classes],
            alpha=0.7,
            s=50,
            color=colors[i],
            label=name
        )
    
    plt.xlabel('Class ID')
    plt.ylabel('Epoch When Class Reached 70% Accuracy')
    plt.title('Class Learning Time Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(args.output_dir, 'class_learning_comparison.png'), dpi=300)

# Create a comparative summary plot
print("Creating comparative summary plot...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Final accuracy comparison
ax1 = axes[0, 0]
model_names = []
val_accs = []
train_accs = []

for name, metrics in model_metrics.items():
    if 'val_acc' in metrics and metrics['val_acc']:
        model_names.append(name)
        val_accs.append(metrics['val_acc'][-1])
        train_accs.append(metrics['train_acc'][-1] if 'train_acc' in metrics and metrics['train_acc'] else 0)

x = np.arange(len(model_names))
width = 0.35

ax1.bar(x - width/2, train_accs, width, label='Train Accuracy')
ax1.bar(x + width/2, val_accs, width, label='Validation Accuracy')

ax1.set_ylabel('Accuracy')
ax1.set_title('Final Accuracy Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.legend()

# 2. Generalization gap comparison
ax2 = axes[0, 1]
gen_gaps = []

for name, metrics in model_metrics.items():
    if 'train_loss' in metrics and 'val_loss' in metrics:
        final_gen_gap = metrics['val_loss'][-1] - metrics['train_loss'][-1]
        gen_gaps.append(final_gen_gap)
    else:
        gen_gaps.append(0)

ax2.bar(model_names, gen_gaps, color='orange')
ax2.set_ylabel('Generalization Gap')
ax2.set_title('Final Generalization Gap')
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names)

# 3. Stability comparison
ax3 = axes[1, 0]
stability_means = []
stability_stds = []

for name, metrics in model_metrics.items():
    if 'stability' in metrics and metrics['stability']:
        stability_means.append(np.mean(metrics['stability']))
        stability_stds.append(np.std(metrics['stability']))
    else:
        stability_means.append(0)
        stability_stds.append(0)

ax3.bar(model_names, stability_means, yerr=stability_stds, color='green')
ax3.set_ylabel('Learning Stability')
ax3.set_title('Average Learning Stability')
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels(model_names)

# 4. Robustness comparison
ax4 = axes[1, 1]
robustness = []

for name, metrics in model_metrics.items():
    if 'adversarial_robustness' in metrics and metrics['adversarial_robustness']:
        # Filter out None values
        adv_robustness = [r for r in metrics['adversarial_robustness'] if r is not None]
        if adv_robustness:
            robustness.append(adv_robustness[-1])
        else:
            robustness.append(0)
    else:
        robustness.append(0)

ax4.bar(model_names, robustness, color='red')
ax4.set_ylabel('Adversarial Accuracy')
ax4.set_title('Final Adversarial Robustness')
ax4.set_xticks(range(len(model_names)))
ax4.set_xticklabels(model_names)

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'comparative_summary.png'), dpi=300)

# Generate comparison report
print("Generating comparison report...")

if args.report_format == 'md':
    report_path = os.path.join(args.output_dir, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write("# Process-Aware Benchmarking (PAB) Model Comparison\n\n")
        
        # Basic information
        f.write("## Comparison Overview\n\n")
        f.write(f"- Dataset: {args.dataset}\n")
        f.write(f"- Models compared: {', '.join(args.model_names)}\n")
        f.write(f"- Number of epochs: {', '.join(str(len(metrics.get('train_loss', []))) for metrics in model_metrics.values())}\n\n")
        
        # Model comparison table
        f.write("## Performance Summary\n\n")
        f.write("| Model | Final Accuracy | Generalization Gap | Learning Stability | Adversarial Robustness |\n")
        f.write("|-------|----------------|---------------------|-------------------|------------------------|\n")
        
        for i, name in enumerate(model_names):
            f.write(f"| {name} | {val_accs[i]:.4f} | {gen_gaps[i]:.4f} | {stability_means[i]:.4f} ± {stability_stds[i]:.4f} | {robustness[i]:.4f} |\n")
        
        f.write("\n")
        
        # Detailed analysis for each model
        f.write("## Detailed Analysis\n\n")
        
        for name, result in comparison_results.items():
            f.write(f"### {name}\n\n")
            
            # Learning stability
            f.write("#### Learning Stability\n\n")
            stability = result.get('overall_stability', {})
            f.write(f"- Mean stability: {stability.get('mean', 0):.4f}\n")
            f.write(f"- Stability standard deviation: {stability.get('std', 0):.4f}\n")
            f.write(f"- Maximum instability: {stability.get('max', 0):.4f}\n\n")
            
            # Generalization
            f.write("#### Generalization\n\n")
            gen = result.get('generalization', {})
            f.write(f"- Final generalization gap: {gen.get('final_gap', 0):.4f}\n")
            f.write(f"- Gap trend: {gen.get('gap_trend', 'unknown')}\n")
            f.write(f"- Optimal early stopping: epoch {gen.get('early_stopping_epoch', 0)}\n\n")
            
            # Class learning patterns
            f.write("#### Class Learning Patterns\n\n")
            class_pat = result.get('class_patterns', {})
            f.write(f"- Early learning classes: {class_pat.get('num_early', 0)} classes\n")
            f.write(f"- Late learning classes: {class_pat.get('num_late', 0)} classes\n\n")
            
            # Robustness
            f.write("#### Adversarial Robustness\n\n")
            rob = result.get('robustness', {})
            f.write(f"- Peak robustness: {rob.get('peak_value', 0):.4f} at epoch {rob.get('peak_epoch', 0)}\n")
            f.write(f"- Final robustness: {rob.get('final_value', 0):.4f}\n")
            f.write(f"- Robustness degradation: {rob.get('degradation', 0)*100:.2f}%\n\n")
        
        # Recommendations
        f.write("## PAB Recommendations\n\n")
        
        best_model = model_names[np.argmax(val_accs)]
        most_stable = model_names[np.argmin(stability_means)]
        best_gen = model_names[np.argmin(gen_gaps)]
        most_robust = model_names[np.argmax(robustness)]
        
        f.write(f"- **Best overall accuracy**: {best_model} achieves the highest validation accuracy.\n")
        f.write(f"- **Most stable learning**: {most_stable} shows the most stable learning progression.\n")
        f.write(f"- **Best generalization**: {best_gen} has the smallest generalization gap.\n")
        f.write(f"- **Most robust**: {most_robust} demonstrates the highest adversarial robustness.\n\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("### Comparative Summary\n\n")
        f.write("![Comparative Summary](comparative_summary.png)\n\n")
        
        for metric in args.metrics:
            if all(metric in metrics for metrics in model_metrics.values()):
                f.write(f"### {metric.replace('_', ' ').title()} Comparison\n\n")
                f.write(f"![{metric.replace('_', ' ').title()} Comparison](comparison_{metric}.png)\n\n")
        
        f.write("### Generalization Gap Comparison\n\n")
        f.write("![Generalization Gap Comparison](generalization_gap_comparison.png)\n\n")
        
        f.write("### Learning Stability Comparison\n\n")
        f.write("![Learning Stability Comparison](stability_comparison.png)\n\n")
        
        if all('adversarial_robustness' in metrics and len(metrics['adversarial_robustness']) > 0 for metrics in model_metrics.values()):
            f.write("### Adversarial Robustness Comparison\n\n")
            f.write("![Adversarial Robustness Comparison](robustness_comparison.png)\n\n")
        
        if args.detailed and all('class_accuracy' in metrics and metrics['class_accuracy'] for metrics in model_metrics.values()):
            f.write("### Class Learning Comparison\n\n")
            f.write("![Class Learning Comparison](class_learning_comparison.png)\n\n")

elif args.report_format == 'txt':
    report_path = os.path.join(args.output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("Process-Aware Benchmarking (PAB) Model Comparison\n")
        f.write("="*60 + "\n\n")
        
        # Basic information
        f.write("Comparison Overview:\n")
        f.write("-"*30 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Models compared: {', '.join(args.model_names)}\n")
        f.write(f"Number of epochs: {', '.join(str(len(metrics.get('train_loss', []))) for metrics in model_metrics.values())}\n\n")
        
        # Model comparison summary
        f.write("Performance Summary:\n")
        f.write("-"*30 + "\n")
        
        header = f"{'Model':<15} {'Final Acc':<12} {'Gen Gap':<12} {'Stability':<20} {'Adv Robustness':<15}\n"
        f.write(header)
        f.write("-"*70 + "\n")
        
        for i, name in enumerate(model_names):
            line = f"{name:<15} {val_accs[i]:<12.4f} {gen_gaps[i]:<12.4f} {stability_means[i]:<8.4f} ± {stability_stds[i]:<8.4f} {robustness[i]:<15.4f}\n"
            f.write(line)
        
        f.write("\n")
        
        # Detailed analysis for each model
        f.write("Detailed Analysis:\n")
        f.write("-"*30 + "\n\n")
        
        for name, result in comparison_results.items():
            f.write(f"{name}:\n")
            f.write("-"*len(name) + "\n")
            
            # Learning stability
            f.write("Learning Stability:\n")
            stability = result.get('overall_stability', {})
            f.write(f"  Mean stability: {stability.get('mean', 0):.4f}\n")
            f.write(f"  Stability standard deviation: {stability.get('std', 0):.4f}\n")
            f.write(f"  Maximum instability: {stability.get('max', 0):.4f}\n\n")
            
            # Generalization
            f.write("Generalization:\n")
            gen = result.get('generalization', {})
            f.write(f"  Final generalization gap: {gen.get('final_gap', 0):.4f}\n")
            f.write(f"  Gap trend: {gen.get('gap_trend', 'unknown')}\n")
            f.write(f"  Optimal early stopping: epoch {gen.get('early_stopping_epoch', 0)}\n\n")
            
            # Class learning patterns
            f.write("Class Learning Patterns:\n")
            class_pat = result.get('class_patterns', {})
            f.write(f"  Early learning classes: {class_pat.get('num_early', 0)} classes\n")
            f.write(f"  Late learning classes: {class_pat.get('num_late', 0)} classes\n\n")
            
            # Robustness
            f.write("Adversarial Robustness:\n")
            rob = result.get('robustness', {})
            f.write(f"  Peak robustness: {rob.get('peak_value', 0):.4f} at epoch {rob.get('peak_epoch', 0)}\n")
            f.write(f"  Final robustness: {rob.get('final_value', 0):.4f}\n")
            f.write(f"  Robustness degradation: {rob.get('degradation', 0)*100:.2f}%\n\n")
        
        # Recommendations
        f.write("PAB Recommendations:\n")
        f.write("-"*30 + "\n")
        
        best_model = model_names[np.argmax(val_accs)]
        most_stable = model_names[np.argmin(stability_means)]
        best_gen = model_names[np.argmin(gen_gaps)]
        most_robust = model_names[np.argmax(robustness)]
        
        f.write(f"* Best overall accuracy: {best_model} achieves the highest validation accuracy.\n")
        f.write(f"* Most stable learning: {most_stable} shows the most stable learning progression.\n")
        f.write(f"* Best generalization: {best_gen} has the smallest generalization gap.\n")
        f.write(f"* Most robust: {most_robust} demonstrates the highest adversarial robustness.\n\n")
        
        f.write("Visualizations saved to output directory.\n")

elif args.report_format == 'json':
    # Create a comprehensive JSON report
    report = {
        'overview': {
            'dataset': args.dataset,
            'models': args.model_names,
            'epochs': [len(metrics.get('train_loss', [])) for metrics in model_metrics.values()]
        },
        'summary': {
            'final_accuracy': {name: acc for name, acc in zip(model_names, val_accs)},
            'generalization_gap': {name: gap for name, gap in zip(model_names, gen_gaps)},
            'learning_stability': {
                name: {'mean': mean, 'std': std} 
                for name, mean, std in zip(model_names, stability_means, stability_stds)
            },
            'adversarial_robustness': {name: rob for name, rob in zip(model_names, robustness)}
        },
        'detailed_analysis': comparison_results,
        'recommendations': {
            'best_accuracy': model_names[np.argmax(val_accs)],
            'most_stable': model_names[np.argmin(stability_means)],
            'best_generalization': model_names[np.argmin(gen_gaps)],
            'most_robust': model_names[np.argmax(robustness)]
        },
        'visualizations': [
            'comparative_summary.png',
            *[f'comparison_{metric}.png' for metric in args.metrics],
            'generalization_gap_comparison.png',
            'stability_comparison.png',
            *(['robustness_comparison.png'] if all('adversarial_robustness' in metrics and len(metrics['adversarial_robustness']) > 0 for metrics in model_metrics.values()) else []),
            *(['class_learning_comparison.png'] if args.detailed and all('class_accuracy' in metrics and metrics['class_accuracy'] for metrics in model_metrics.values()) else [])
        ]
    }
    
    report_path = os.path.join(args.output_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

print(f"Comparison completed! Results saved to {args.output_dir}")
print(f"Report saved to {report_path}")
