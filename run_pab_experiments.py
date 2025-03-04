#!/usr/bin/env python
"""
Main script for running Process-Aware Benchmarking (PAB) experiments.

Usage:
    python run_pab_experiments.py --config config.yaml
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
import logging
from pab.config import load_config
from pab.datasets import load_dataset
from pab.core import ProcessAwareBenchmark, track_learning_curve
from pab.utils import export_metrics_to_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PAB experiments")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_model(config):
    """Create model based on configuration."""
    architecture = config['model']['architecture']
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    
    if architecture == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'efficientnet':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'vit':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vit_b_16', pretrained=pretrained)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

def create_optimizer(config, model):
    """Create optimizer based on configuration."""
    optimizer_name = config['training']['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config['training']['momentum']
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer

def create_scheduler(config, optimizer):
    """Create learning rate scheduler based on configuration."""
    scheduler_name = config['training']['lr_scheduler']
    epochs = config['training']['epochs']
    
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        step_size = config['training']['step_size']
        gamma = config['training']['gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'multistep':
        milestones = config['training']['milestones']
        gamma = config['training']['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        scheduler = None
    
    return scheduler

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_loader, val_loader = load_dataset(
        name=config['data']['dataset'],
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    logger.info(f"Loaded {config['data']['dataset']} dataset")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    logger.info(f"Created {config['model']['architecture']} model")
    
    # Create optimizer
    optimizer = create_optimizer(config, model)
    
    # Create scheduler
    scheduler = create_scheduler(config, optimizer)
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['pab']['checkpoint_dir'], exist_ok=True)
    
    # Create PAB instance
    pab = ProcessAwareBenchmark(
        checkpoint_dir=config['pab']['checkpoint_dir'],
        save_frequency=config['pab']['save_frequency'],
        track_representations=config['pab']['track_representations']
    )
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(1, config['training']['epochs'] + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        
        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Track per-class accuracies
        class_accs = None
        if epoch % config['pab']['save_frequency'] == 0:
            # Compute per-class accuracies periodically to save computation
            from pab.utils import compute_class_accuracies
            class_accs = compute_class_accuracies(
                model, val_loader, config['model']['num_classes'], device
            )
        
        # Track adversarial robustness if requested
        adv_acc = None
        if config['pab']['adversarial_test'] and epoch % config['pab']['save_frequency'] == 0:
            from pab.utils import evaluate_adversarial_robustness
            adv_metrics = evaluate_adversarial_robustness(
                model, val_loader, device, epsilon=config['pab']['epsilon']
            )
            adv_acc = adv_metrics['adversarial_accuracy']
        
        # Track epoch in PAB
        pab.track_epoch(
            model=model,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            class_accuracies=class_accs,
            adversarial_acc=adv_acc
        )
        
        # Print progress
        logger.info(f"Epoch {epoch}/{config['training']['epochs']} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save final model if requested
    if config['output']['save_model']:
        final_model_path = os.path.join(config['output']['results_dir'], 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
    
    # Save metrics if requested
    if config['output']['save_metrics']:
        metrics_path = os.path.join(config['output']['results_dir'], 'pab_metrics.json')
        export_metrics_to_json(pab.metrics, metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate visualizations if requested
    if config['output']['plot_metrics']:
        from pab.visualization import (
            plot_learning_trajectory,
            plot_class_progression,
            plot_robustness_curve,
            plot_generalization_gap,
            plot_pab_summary
        )
        
        # Learning trajectory plot
        if 'train_loss' in pab.metrics and 'val_loss' in pab.metrics:
            plot_learning_trajectory(
                train_losses=pab.metrics['train_loss'],
                val_losses=pab.metrics['val_loss'],
                train_accs=pab.metrics.get('train_acc'),
                val_accs=pab.metrics.get('val_acc'),
                save_path=os.path.join(config['output']['results_dir'], 'learning_trajectory.png')
            )
        
        # Class progression plot
        if 'class_accuracy' in pab.metrics and pab.metrics['class_accuracy']:
            plot_class_progression(
                class_accuracies=pab.metrics['class_accuracy'],
                save_path=os.path.join(config['output']['results_dir'], 'class_progression.png')
            )
        
        # Robustness plot
        if 'adversarial_robustness' in pab.metrics and pab.metrics['adversarial_robustness']:
            plot_robustness_curve(
                clean_accuracies=pab.metrics['val_acc'],
                adversarial_accuracies=pab.metrics['adversarial_robustness'],
                save_path=os.path.join(config['output']['results_dir'], 'robustness_curve.png')
            )
        
        # Generalization gap plot
        if 'train_loss' in pab.metrics and 'val_loss' in pab.metrics:
            plot_generalization_gap(
                train_losses=pab.metrics['train_loss'],
                val_losses=pab.metrics['val_loss'],
                train_accs=pab.metrics.get('train_acc'),
                val_accs=pab.metrics.get('val_acc'),
                save_path=os.path.join(config['output']['results_dir'], 'generalization_gap.png')
            )
        
        # Summary plot
        plot_pab_summary(
            metrics=pab.metrics,
            save_path=os.path.join(config['output']['results_dir'], 'pab_summary.png')
        )
        
        logger.info(f"Plots saved to {config['output']['results_dir']}")
    
    # Evaluate trajectory
    results = pab.evaluate_trajectory()
    
    # Print summary
    summary = pab.summarize()
    print("\nPAB Summary:")
    print(summary)
    
    # Save summary to file
    summary_path = os.path.join(config['output']['results_dir'], 'pab_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info("Experiment completed successfully")

if __name__ == '__main__':
    main()
