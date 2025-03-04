"""
Model Comparison Using Process-Aware Benchmarking (PAB).

This script demonstrates how to compare multiple models using the PAB framework.
It loads checkpoints from different models and compares their learning trajectories.
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for importing PAB
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pab import compare_models
from pab.visualization import compare_learning_trajectories

# Parse arguments
parser = argparse.ArgumentParser(description='PAB Model Comparison')
parser.add_argument('--model_dirs', nargs='+', required=True,
                   help='List of directories containing model checkpoints')
parser.add_argument('--model_names', nargs='+', 
                   help='List of model names (for better reporting)')
parser.add_argument('--output_dir', type=str, default='./results/comparison',
                   help='Directory to save comparison results')
parser.add_argument('--metrics', nargs='+', default=['val_acc', 'train_loss'],
                   help='Metrics to compare (e.g., val_acc, train_loss)')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Set model names if not provided
if args.model_names is None:
    args.model_names = [f"Model_{i}" for i in range(len(args.model_dirs))]

if len(args.model_names) != len(args.model_dirs):
    print("Warning: Number of model names does not match number of model directories.")
    args.model_names = [f"Model_{i}" for i in range(len(args.model_dirs))]

# Compare models using PAB
print("Comparing models...")
comparison_results = compare_models(
    model_dirs=args.model_dirs,
    names=args.model_names
)

# Save comparison results to JSON
with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
    json.dump(comparison_results, f, indent=2)

# Extract metrics for visualization
model_metrics = {}
for name, result in comparison_results.items():
    metrics_path = None
    
    # Look for metrics file in model directory
    for model_dir, model_name in zip(args.model_dirs, args.model_names):
        if model_name == name:
            metrics_path = os.path.join(model_dir, f"{model_name}_metrics.json")
            if not os.path.exists(metrics_path):
                metrics_path = os.path.join(model_dir, "metrics.json")
            break
    
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        model_metrics[name] = metrics
    else:
        print(f"Warning: Could not find metrics for {name}. Using evaluation results only.")
        model_metrics[name] = {
            'gen_efficiency': result.get('generalization', {}).get('final_gap', 0),
            'stability': result.get('overall_stability', {}).get('mean', 0),
        }

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

# Create a comparative summary plot
plt.figure(figsize=(12, 8))

# Set up variables for plotting
metrics_to_plot = [
    ('generalization', 'final_gap', 'Generalization Gap'),
    ('overall_stability', 'mean', 'Learning Stability'),
    ('robustness', 'degradation', 'Robustness Degradation'),
    ('class_patterns', 'num_early', 'Early Learning Classes')
]

num_metrics = len(metrics_to_plot)
bar_width = 0.8 / len(comparison_results)
colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_results)))
index = np.arange(num_metrics)

for i, (name, result) in enumerate(comparison_results.items()):
    values = []
    for section, key, _ in metrics_to_plot:
        if section in result and key in result[section]:
            values.append(result[section][key])
        else:
            values.append(0)
    
    plt.bar(index + i * bar_width, values, bar_width, 
            label=name, color=colors[i], alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Comparative Model Analysis Using PAB Metrics')
plt.xticks(index + bar_width * (len(comparison_results) - 1) / 2, 
           [label for _, _, label in metrics_to_plot])
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(args.output_dir, 'comparative_summary.png'), dpi=300)

print(f"Creating PAB comparison report...")
with open(os.path.join(args.output_dir, 'comparison_report.txt'), 'w') as f:
    f.write("Process-Aware Benchmarking (PAB) Model Comparison\n")
    f.write("="*50 + "\n\n")
    
    for name, result in comparison_results.items():
        f.write(f"Model: {name}\n")
        f.write("-"*30 + "\n")
        
        # Learning stability
        stability = result.get('overall_stability', {})
        f.write("Learning Stability:\n")
        for k, v in stability.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")
        
        # Generalization
        gen = result.get('generalization', {})
        f.write("Generalization:\n")
        for k, v in gen.items():
            if isinstance(v, (int, float)):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        # Class patterns
        class_pat = result.get('class_patterns', {})
        f.write("Class Learning Patterns:\n")
        for k, v in class_pat.items():
            if k not in ['early_classes', 'late_classes']:
                f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        # Robustness
        rob = result.get('robustness', {})
        f.write("Adversarial Robustness:\n")
        for k, v in rob.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")
        
        # Recommendations
        f.write("PAB Recommendations:\n")
        if gen.get('gap_trend') == 'increasing':
            f.write("  • Model shows signs of overfitting\n")
        if rob.get('degradation', 0) > 0.1:
            f.write("  • Robustness degrades during training\n")
        if stability.get('std', 0) > 0.1:
            f.write("  • Training exhibits instability\n")
        f.write("\n\n")

print(f"Comparison completed! Results saved to {args.output_dir}")
