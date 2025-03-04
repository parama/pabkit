"""
Default configuration parameters for the Process-Aware Benchmarking (PAB) toolkit.

This module defines default parameters for PAB components to ensure consistent behavior
across different experiments and analyses.
"""

# Core PAB parameters
PAB_DEFAULTS = {
    'checkpoint_dir': './pab_checkpoints',  # Default directory for storing checkpoints
    'save_frequency': 5,                    # Save checkpoints every N epochs
    'track_representations': True,          # Whether to track feature representations
    'metrics_file': 'pab_metrics.json',     # Default metrics file name
    'max_checkpoints': 100,                 # Maximum number of checkpoints to keep
}

# Visualization parameters
VISUALIZATION_DEFAULTS = {
    'figsize_learning_curve': (12, 5),      # Figure size for learning curves (width, height)
    'figsize_class_progression': (10, 6),   # Figure size for class progression plots
    'figsize_robustness': (10, 6),          # Figure size for robustness curves
    'figsize_summary': (15, 10),            # Figure size for summary plots
    'dpi': 300,                             # DPI for saved figures
    'cmap': 'viridis',                      # Default colormap
    'line_width': 2,                        # Line width for plots
    'grid_alpha': 0.7,                      # Grid transparency
    'save_format': 'png',                   # Default format for saved figures
}

# Adversarial evaluation parameters
ADVERSARIAL_DEFAULTS = {
    'epsilon': 0.03,                        # Default perturbation magnitude for FGSM
    'alpha': 0.01,                          # Step size for PGD attacks
    'num_steps': 10,                        # Number of steps for PGD attacks
    'random_start': True,                   # Use random initialization for PGD
}

# Feature representation parameters
REPRESENTATION_DEFAULTS = {
    'pca_components': 2,                    # Number of PCA components
    'tsne_perplexity': 30,                  # t-SNE perplexity parameter
    'num_samples': 1000,                    # Number of samples for representation analysis
}

# Class progression parameters
CLASS_PROGRESSION_DEFAULTS = {
    'early_cutoff': 0.333,                  # Threshold for early-learning classes (fraction of training)
    'late_cutoff': 0.667,                   # Threshold for late-learning classes (fraction of training)
    'accuracy_threshold': 0.7,              # Accuracy threshold for considering a class "learned"
}

# Evaluation thresholds
EVALUATION_THRESHOLDS = {
    'stability_high': 0.1,                  # Threshold for high stability
    'stability_low': 0.05,                  # Threshold for low stability
    'gen_gap_high': 0.2,                    # Threshold for high generalization gap
    'gen_gap_low': 0.1,                     # Threshold for low generalization gap
    'robustness_degradation': 0.1,          # Threshold for concerning robustness degradation
}

# Model layers of interest for different architectures
MODEL_LAYERS = {
    'resnet18': 'layer4',                   # Last residual block in ResNet-18
    'resnet50': 'layer4',                   # Last residual block in ResNet-50
    'efficientnet': 'features.8',           # Late-stage feature block in EfficientNet
    'vit': 'encoder.layers.11',             # Last transformer block in ViT
    'densenet': 'features.denseblock4',     # Last dense block in DenseNet
    'mobilenet': 'features.18',             # Late convolutional block in MobileNet
}

# Training hyperparameters for standard benchmarks
TRAINING_HYPERPARAMS = {
    'cifar10': {
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 200,
    },
    'cifar100': {
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 200,
    },
    'imagenet': {
        'batch_size': 256,
        'learning_rate': 0.1,
        'weight_decay': 1e-4,
        'epochs': 90,
    }
}

# Data transformations for standard datasets
DATA_TRANSFORMS = {
    'cifar10': {
        'train': [
            ('RandomCrop', 32, {'padding': 4}),
            ('RandomHorizontalFlip', None, {}),
            ('ToTensor', None, {}),
            ('Normalize', (0.5, 0.5, 0.5), {'std': (0.5, 0.5, 0.5)})
        ],
        'test': [
            ('ToTensor', None, {}),
            ('Normalize', (0.5, 0.5, 0.5), {'std': (0.5, 0.5, 0.5)})
        ]
    },
    'imagenet': {
        'train': [
            ('RandomResizedCrop', 224, {}),
            ('RandomHorizontalFlip', None, {}),
            ('ToTensor', None, {}),
            ('Normalize', (0.485, 0.456, 0.406), {'std': (0.229, 0.224, 0.225)})
        ],
        'test': [
            ('Resize', 256, {}),
            ('CenterCrop', 224, {}),
            ('ToTensor', None, {}),
            ('Normalize', (0.485, 0.456, 0.406), {'std': (0.229, 0.224, 0.225)})
        ]
    }
}
