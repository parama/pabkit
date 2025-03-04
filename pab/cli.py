"""
Command-line interface for the Process-Aware Benchmarking (PAB) toolkit.

This module provides command-line tools for working with PAB:
- Analyzing existing models
- Visualizing learning trajectories
- Comparing multiple models
- Generating PAB reports
"""

import os
import sys
import argparse
import json
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any

from .core import ProcessAwareBenchmark, evaluate_trajectory
from .visualization import (
    plot_learning_trajectory,
    plot_class_progression,
    plot_robustness_curve,
    plot_generalization_gap,
    plot_pab_summary,
    compare_learning_trajectories
)
from .tracking import CheckpointManager
from .utils import export_metrics_to_json

def analyze_command(args):
    """Analyze checkpoints from a trained model using PAB."""
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory '{args.checkpoint_dir}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate trajectory
    results = evaluate_trajectory(args.checkpoint_dir)
    
    # Save results
    with open(os.path.join(args.output_dir, 'pab_evaluation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create PAB instance
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    # Load metrics from the last checkpoint
    _, metrics = checkpoint_manager.load_checkpoint(checkpoints[-1])
    
    # Create PAB instance and populate with loaded metrics
    pab = ProcessAwareBenchmark(checkpoint_dir=args.checkpoint_dir)
    pab.metrics = metrics
    
    # Generate summary
    summary = pab.summarize()
    print("\nPAB Summary:")
    print(summary)
    
    # Save summary to file
    with open(os.path.join(args.output_dir, 'pab_summary.txt'), 'w') as f:
        f.write(summary)
    
    # Create visualizations
    if 'train_loss' in metrics and 'val_loss' in metrics:
        plot_learning_trajectory(
            train_losses=metrics['train_loss'],
            val_losses=metrics['val_loss'],
            train_accs=metrics.get('train_acc', None),
            val_accs=metrics.get('val_acc', None),
            save_path=os.path.join(args.output_dir, 'learning_trajectory.png')
        )
    
    if 'class_accuracy' in metrics and metrics['class_accuracy']:
        plot_class_progression(
            class_accuracies=metrics['class_accuracy'],
            save_path=os.path.join(args.output_dir, 'class_progression.png')
        )
    
    if 'adversarial_robustness' in metrics and metrics['adversarial_robustness'] and 'val_acc' in metrics:
        plot_robustness_curve(
            clean_accuracies=metrics['val_acc'],
            adversarial_accuracies=metrics['adversarial_robustness'],
            save_path=os.path.join(args.output_dir, 'robustness_curve.png')
        )
    
    if 'train_loss' in metrics and 'val_loss' in metrics:
        plot_generalization_gap(
            train_losses=metrics['train_loss'],
            val_losses=metrics['val_loss'],
            train_accs=metrics.get('train_acc', None),
            val_accs=metrics.get('val_acc', None),
            save_path=os.path.join(args.output_dir, 'generalization_gap.png')
        )
    
    # Comprehensive summary plot
    plot_pab_summary(
        metrics=metrics,
        save_path=os.path.join(args.output_dir, 'pab_summary.png')
    )
    
    # Export metrics to JSON
    export_metrics_to_json(
        metrics,
        os.path.join(args.output_dir, 'pab_metrics.json')
    )
    
    print(f"Analysis completed. Results saved to {args.output_dir}")

def compare_command(args):
    """Compare multiple models using PAB."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify model directories
    for model_dir in args.model_dirs:
        if not os.path.exists(model_dir):
            print(f"Error: Model directory '{model_dir}' does not exist.")
            return
    
    # Set model names if not provided
    if not args.model_names:
        args.model_names = [f"Model_{i}" for i in range(len(args.model_dirs))]
    
    if len(args.model_names) != len(args.model_dirs):
        print("Warning: Number of model names does not match number of model directories.")
        args.model_names = [f"Model_{i}" for i in range(len(args.model_dirs))]
    
    # Import compare_models function here to avoid circular imports
    from .core import compare_models
    
    # Compare models
    comparison_results = compare_models(
        model_dirs=args.model_dirs,
        names=args.model_names
    )
    
    # Save comparison results
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Load metrics for each model
    model_metrics = {}
    for i, (model_dir, model_name) in enumerate(zip(args.model_dirs, args.model_names)):
        # Try to load metrics from model directory
        metrics_path = os.path.join(model_dir, 'pab_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metrics[model_name] = json.load(f)
        else:
            # If metrics file doesn't exist, try loading from checkpoint
            checkpoint_manager = CheckpointManager(model_dir)
            checkpoints = checkpoint_manager.list_checkpoints()
            
            if checkpoints:
                # Load metrics from the last checkpoint
                _, metrics = checkpoint_manager.load_checkpoint(checkpoints[-1])
                model_metrics[model_name] = metrics
            else:
                print(f"Warning: No metrics found for model '{model_name}'")
    
    # Create comparative visualizations
    for metric in args.metrics:
        # Check if metric exists in all models
        if all(metric in metrics for metrics in model_metrics.values()):
            compare_learning_trajectories(
                model_metrics=model_metrics,
                metric_name=metric,
                save_path=os.path.join(args.output_dir, f'comparison_{metric}.png')
            )
    
    # Generate comparative report
    with open(os.path.join(args.output_dir, 'comparison_report.txt'), 'w') as f:
        f.write("Process-Aware Benchmarking (PAB) Comparative Analysis\n")
        f.write("="*60 + "\n\n")
        
        for name, result in comparison_results.items():
            f.write(f"Model: {name}\n")
            f.write("-"*30 + "\n")
            
            # Write evaluation results
            for section, values in result.items():
                f.write(f"{section.replace('_', ' ').title()}:\n")
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {key}: {value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {values}\n")
                f.write("\n")
            
            f.write("\n")
    
    print(f"Comparison completed. Results saved to {args.output_dir}")

def visualize_command(args):
    """Visualize PAB metrics from a JSON file."""
    # Verify metrics file exists
    if not os.path.exists(args.metrics_file):
        print(f"Error: Metrics file '{args.metrics_file}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    with open(args.metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create visualizations based on specified type
    if args.type == 'learning_curve':
        if 'train_loss' in metrics and 'val_loss' in metrics:
            plot_learning_trajectory(
                train_losses=metrics['train_loss'],
                val_losses=metrics['val_loss'],
                train_accs=metrics.get('train_acc', None),
                val_accs=metrics.get('val_acc', None),
                save_path=os.path.join(args.output_dir, 'learning_trajectory.png')
            )
        else:
            print("Error: Required metrics not found for learning curve visualization.")
    
    elif args.type == 'class_progression':
        if 'class_accuracy' in metrics and metrics['class_accuracy']:
            plot_class_progression(
                class_accuracies=metrics['class_accuracy'],
                save_path=os.path.join(args.output_dir, 'class_progression.png')
            )
        else:
            print("Error: Required metrics not found for class progression visualization.")
    
    elif args.type == 'robustness':
        if 'adversarial_robustness' in metrics and metrics['adversarial_robustness'] and 'val_acc' in metrics:
            plot_robustness_curve(
                clean_accuracies=metrics['val_acc'],
                adversarial_accuracies=metrics['adversarial_robustness'],
                save_path=os.path.join(args.output_dir, 'robustness_curve.png')
            )
        else:
            print("Error: Required metrics not found for robustness visualization.")
    
    elif args.type == 'generalization':
        if 'train_loss' in metrics and 'val_loss' in metrics:
            plot_generalization_gap(
                train_losses=metrics['train_loss'],
                val_losses=metrics['val_loss'],
                train_accs=metrics.get('train_acc', None),
                val_accs=metrics.get('val_acc', None),
                save_path=os.path.join(args.output_dir, 'generalization_gap.png')
            )
        else:
            print("Error: Required metrics not found for generalization visualization.")
    
    elif args.type == 'summary':
        plot_pab_summary(
            metrics=metrics,
            save_path=os.path.join(args.output_dir, 'pab_summary.png')
        )
    
    else:
        print(f"Error: Unknown visualization type '{args.type}'")
        return
    
    print(f"Visualization completed. Results saved to {args.output_dir}")

def report_command(args):
    """Generate a comprehensive PAB report."""
    # Verify checkpoints directory exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory '{args.checkpoint_dir}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate trajectory
    results = evaluate_trajectory(args.checkpoint_dir)
    
    # Create PAB instance
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    # Load metrics from the last checkpoint
    _, metrics = checkpoint_manager.load_checkpoint(checkpoints[-1])
    
    # Create PAB instance and populate with loaded metrics
    pab = ProcessAwareBenchmark(checkpoint_dir=args.checkpoint_dir)
    pab.metrics = metrics
    
    # Generate summary
    summary = pab.summarize()
    
    # Create report file
    with open(os.path.join(args.output_dir, 'pab_report.md'), 'w') as f:
        f.write("# Process-Aware Benchmarking (PAB) Report\n\n")
        
        # Model information
        f.write("## Model Information\n\n")
        f.write(f"- Model name: {args.model_name}\n")
        f.write(f"- Checkpoint directory: {args.checkpoint_dir}\n")
        f.write(f"- Number of checkpoints: {len(checkpoints)}\n")
        f.write(f"- Training duration: {len(metrics.get('train_loss', [])) if 'train_loss' in metrics else 'Unknown'} epochs\n\n")
        
        # PAB Summary
        f.write("## PAB Summary\n\n")
        f.write("```\n")
        f.write(summary)
        f.write("\n```\n\n")
        
        # Learning Trajectory
        f.write("## Learning Trajectory\n\n")
        f.write("![Learning Trajectory](learning_trajectory.png)\n\n")
        
        # Generalization Analysis
        f.write("## Generalization Analysis\n\n")
        if 'generalization' in results:
            gen = results['generalization']
            f.write("### Key Findings\n\n")
            f.write(f"- Final generalization gap: {gen.get('final_gap', 'Unknown')}\n")
            f.write(f"- Generalization trend: {gen.get('gap_trend', 'Unknown')}\n")
            f.write(f"- Optimal early stopping epoch: {gen.get('early_stopping_epoch', 'Unknown')}\n\n")
        
        f.write("![Generalization Gap](generalization_gap.png)\n\n")
        
        # Class-wise Learning
        f.write("## Class-wise Learning Progression\n\n")
        if 'class_patterns' in results:
            patterns = results['class_patterns']
            f.write("### Key Findings\n\n")
            f.write(f"- Early learning classes: {patterns.get('num_early', 0)} classes\n")
            f.write(f"- Late learning classes: {patterns.get('num_late', 0)} classes\n\n")
        
        f.write("![Class Progression](class_progression.png)\n\n")
        
        # Adversarial Robustness
        f.write("## Adversarial Robustness\n\n")
        if 'robustness' in results:
            rob = results['robustness']
            f.write("### Key Findings\n\n")
            f.write(f"- Peak robustness: {rob.get('peak_value', 0):.4f} at epoch {rob.get('peak_epoch', 0)}\n")
            f.write(f"- Final robustness: {rob.get('final_value', 0):.4f}\n")
            f.write(f"- Robustness degradation: {rob.get('degradation', 0)*100:.2f}%\n\n")
        
        if 'adversarial_robustness' in metrics and metrics['adversarial_robustness']:
            f.write("![Robustness Curve](robustness_curve.png)\n\n")
        
        # Recommendations
        f.write("## PAB Recommendations\n\n")
        
        if 'generalization' in results and results['generalization'].get('gap_trend') == 'increasing':
            f.write("- **Overfitting detected**: The model shows signs of overfitting. Consider early stopping or regularization.\n")
        
        if 'robustness' in results and results['robustness'].get('degradation', 0) > 0.1:
            f.write("- **Robustness degradation**: Adversarial robustness peaks before the final epoch, suggesting a robustness-accuracy tradeoff.\n")
        
        if 'overall_stability' in results and results['overall_stability'].get('std', 0) > 0.1:
            f.write("- **Training instability**: The training exhibits fluctuations. Consider a more stable optimization strategy.\n")
    
    # Create visualizations
    if 'train_loss' in metrics and 'val_loss' in metrics:
        plot_learning_trajectory(
            train_losses=metrics['train_loss'],
            val_losses=metrics['val_loss'],
            train_accs=metrics.get('train_acc', None),
            val_accs=metrics.get('val_acc', None),
            save_path=os.path.join(args.output_dir, 'learning_trajectory.png')
        )
    
    if 'class_accuracy' in metrics and metrics['class_accuracy']:
        plot_class_progression(
            class_accuracies=metrics['class_accuracy'],
            save_path=os.path.join(args.output_dir, 'class_progression.png')
        )
    
    if 'adversarial_robustness' in metrics and metrics['adversarial_robustness'] and 'val_acc' in metrics:
        plot_robustness_curve(
            clean_accuracies=metrics['val_acc'],
            adversarial_accuracies=metrics['adversarial_robustness'],
            save_path=os.path.join(args.output_dir, 'robustness_curve.png')
        )
    
    if 'train_loss' in metrics and 'val_loss' in metrics:
        plot_generalization_gap(
            train_losses=metrics['train_loss'],
            val_losses=metrics['val_loss'],
            train_accs=metrics.get('train_acc', None),
            val_accs=metrics.get('val_acc', None),
            save_path=os.path.join(args.output_dir, 'generalization_gap.png')
        )
    
    # Comprehensive summary plot
    plot_pab_summary(
        metrics=metrics,
        save_path=os.path.join(args.output_dir, 'pab_summary.png')
    )
    
    # Save metrics
    export_metrics_to_json(
        metrics,
        os.path.join(args.output_dir, 'pab_metrics.json')
    )
    
    print(f"Report generated. Results saved to {args.output_dir}")

def main():
    """Main entry point for the PAB command-line interface."""
    parser = argparse.ArgumentParser(description='Process-Aware Benchmarking (PAB) Toolkit')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model checkpoints')
    analyze_parser.add_argument('--checkpoint_dir', type=str, required=True,
                               help='Directory containing model checkpoints')
    analyze_parser.add_argument('--output_dir', type=str, default='./pab_results',
                               help='Directory to save analysis results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--model_dirs', nargs='+', required=True,
                               help='List of directories containing model checkpoints')
    compare_parser.add_argument('--model_names', nargs='+',
                               help='List of model names (for better reporting)')
    compare_parser.add_argument('--output_dir', type=str, default='./pab_comparison',
                               help='Directory to save comparison results')
    compare_parser.add_argument('--metrics', nargs='+', default=['val_acc', 'train_loss'],
                               help='Metrics to compare (e.g., val_acc, train_loss)')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize PAB metrics')
    visualize_parser.add_argument('--metrics_file', type=str, required=True,
                                 help='JSON file containing PAB metrics')
    visualize_parser.add_argument('--output_dir', type=str, default='./pab_visualizations',
                                 help='Directory to save visualizations')
    visualize_parser.add_argument('--type', type=str, default='summary',
                                 choices=['learning_curve', 'class_progression', 
                                          'robustness', 'generalization', 'summary'],
                                 help='Type of visualization to create')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate PAB report')
    report_parser.add_argument('--checkpoint_dir', type=str, required=True,
                              help='Directory containing model checkpoints')
    report_parser.add_argument('--output_dir', type=str, default='./pab_report',
                              help='Directory to save report')
    report_parser.add_argument('--model_name', type=str, default='Model',
                              help='Name of the model for the report')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Run the selected command
    if args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'compare':
        compare_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    elif args.command == 'report':
        report_command(args)

if __name__ == '__main__':
    main()
