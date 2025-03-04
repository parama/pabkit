"""
Process-Aware Benchmarking (PAB) Toolkit

A framework for evaluating machine learning models based on their learning trajectories
rather than solely on final performance metrics.

Core functionality:
- Track learning curves and progression over epochs
- Analyze generalization vs. memorization dynamics
- Assess model robustness over time
- Evaluate representation shifts during training
"""

__version__ = "0.1.0"

from .core import (
    ProcessAwareBenchmark,
    track_learning_curve,
    evaluate_trajectory,
    compare_models
)

from .metrics import (
    learning_stability,
    generalization_efficiency,
    rule_evolution,
    class_wise_progression,
    robustness_evolution
)

from .visualization import (
    plot_learning_trajectory,
    plot_class_progression,
    plot_robustness_curve,
    plot_generalization_gap
)

from .tracking import (
    Checkpoint,
    CheckpointManager
)

__all__ = [
    # Core
    'ProcessAwareBenchmark',
    'track_learning_curve',
    'evaluate_trajectory',
    'compare_models',
    
    # Metrics
    'learning_stability',
    'generalization_efficiency',
    'rule_evolution',
    'class_wise_progression',
    'robustness_evolution',
    
    # Visualization
    'plot_learning_trajectory',
    'plot_class_progression',
    'plot_robustness_curve',
    'plot_generalization_gap',
    
    # Tracking
    'Checkpoint',
    'CheckpointManager'
]
