"""
Default configuration for the PAB toolkit.
"""

DEFAULT_CONFIG = {
    'experiment': {
        'name': 'pab_experiment',
        'description': 'Process-Aware Benchmarking experiment',
        'seed': 42,
        'device': 'cuda' # or 'cpu'
    },
    'data': {
        'dataset': 'cifar10', # one of 'cifar10', 'cifar100', 'imagenet', 'custom'
        'data_dir': './data',
        'train_split': 0.8,
        'batch_size': 128,
        'num_workers': 4
    },
    'model': {
        'architecture': 'resnet18', # or 'efficientnet', 'vit', etc.
        'pretrained': False,
        'num_classes': 10
    },
    'training': {
        'epochs': 100,
        'optimizer': 'adam', # or 'sgd'
        'learning_rate': 0.001,
        'momentum': 0.9, # for SGD
        'weight_decay': 1e-4,
        'lr_scheduler': 'cosine', # or 'step', 'multistep', etc.
        'step_size': 30, # for StepLR
        'gamma': 0.1, # for StepLR/MultiStepLR
        'milestones': [30, 60, 90] # for MultiStepLR
    },
    'pab': {
        'checkpoint_dir': './pab_checkpoints',
        'save_frequency': 5,
        'track_representations': True,
        'adversarial_test': False,
        'epsilon': 0.03 # for adversarial testing
    },
    'output': {
        'results_dir': './pab_results',
        'save_model': True,
        'save_metrics': True,
        'plot_metrics': True
    }
}
