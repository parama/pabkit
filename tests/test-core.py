"""
Unit tests for the PAB core module.
"""

import os
import shutil
import unittest
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pab import ProcessAwareBenchmark, evaluate_trajectory, compare_models
from pab.tracking import CheckpointManager

class SimpleMLP(nn.Module):
    """Simple MLP for testing"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    def forward(self, x):
        return self.layers(x)

class TestProcessAwareBenchmark(unittest.TestCase):
    """Test cases for ProcessAwareBenchmark class"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create a simple model
        self.model = SimpleMLP()
        
        # Initialize PAB
        self.pab = ProcessAwareBenchmark(
            checkpoint_dir=self.checkpoint_dir,
            save_frequency=1
        )
    
    def tearDown(self):
        """Clean up test environment after each test method"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test PAB initialization"""
        self.assertEqual(self.pab.checkpoint_dir, self.checkpoint_dir)
        self.assertEqual(self.pab.save_frequency, 1)
        self.assertTrue(self.pab.track_representations)
        
        # Check metrics initialization
        self.assertIn('train_loss', self.pab.metrics)
        self.assertIn('val_loss', self.pab.metrics)
        self.assertIn('train_acc', self.pab.metrics)
        self.assertIn('val_acc', self.pab.metrics)
        self.assertIn('stability', self.pab.metrics)
        self.assertIn('gen_efficiency', self.pab.metrics)
        self.assertIn('rule_evolution', self.pab.metrics)
        self.assertIn('class_accuracy', self.pab.metrics)
        self.assertIn('adversarial_robustness', self.pab.metrics)
    
    def test_track_epoch(self):
        """Test tracking an epoch"""
        # Track a few epochs
        for epoch in range(1, 4):
            self.pab.track_epoch(
                model=self.model,
                epoch=epoch,
                train_loss=1.0 / epoch,
                val_loss=1.2 / epoch,
                train_acc=0.5 + 0.1 * epoch,
                val_acc=0.45 + 0.1 * epoch
            )
        
        # Check metrics
        self.assertEqual(len(self.pab.metrics['train_loss']), 3)
        self.assertEqual(len(self.pab.metrics['val_loss']), 3)
        self.assertEqual(len(self.pab.metrics['train_acc']), 3)
        self.assertEqual(len(self.pab.metrics['val_acc']), 3)
        
        # Check stability and gen_efficiency (should have 2 entries since they need 2 epochs)
        self.assertEqual(len(self.pab.metrics['stability']), 2)
        self.assertEqual(len(self.pab.metrics['gen_efficiency']), 2)
        
        # Check checkpoint creation
        checkpoint_files = os.listdir(self.checkpoint_dir)
        self.assertTrue(any("checkpoint_epoch_0001" in f for f in checkpoint_files))
        self.assertTrue(any("checkpoint_epoch_0002" in f for f in checkpoint_files))
        self.assertTrue(any("checkpoint_epoch_0003" in f for f in checkpoint_files))
    
    def test_evaluate_trajectory(self):
        """Test trajectory evaluation"""
        # Track a few epochs
        for epoch in range(1, 6):
            self.pab.track_epoch(
                model=self.model,
                epoch=epoch,
                train_loss=1.0 / epoch,
                val_loss=1.2 / epoch,
                train_acc=0.5 + 0.1 * epoch,
                val_acc=0.45 + 0.1 * epoch
            )
        
        # Evaluate trajectory
        results = self.pab.evaluate_trajectory()
        
        # Check result structure
        self.assertIn('overall_stability', results)
        self.assertIn('generalization', results)
        
        # Check generalization metrics
        self.assertIn('final_gap', results['generalization'])
        self.assertIn('gap_trend', results['generalization'])
        self.assertIn('early_stopping_epoch', results['generalization'])
    
    def test_summarize(self):
        """Test summary generation"""
        # Track a few epochs
        for epoch in range(1, 6):
            self.pab.track_epoch(
                model=self.model,
                epoch=epoch,
                train_loss=1.0 / epoch,
                val_loss=1.2 / epoch,
                train_acc=0.5 + 0.1 * epoch,
                val_acc=0.45 + 0.1 * epoch
            )
        
        # Generate summary
        summary = self.pab.summarize()
        
        # Check summary content
        self.assertIsInstance(summary, str)
        self.assertIn("Process-Aware Benchmarking (PAB) Summary", summary)
        self.assertIn("Training duration:", summary)
        self.assertIn("Learning Stability:", summary)
        self.assertIn("Generalization:", summary)
    
    def test_class_wise_tracking(self):
        """Test class-wise accuracy tracking"""
        # Create some class accuracies
        class_accuracies = {
            0: 0.8,
            1: 0.6,
            2: 0.7
        }
        
        # Track epoch with class accuracies
        self.pab.track_epoch(
            model=self.model,
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_acc=0.7,
            val_acc=0.65,
            class_accuracies=class_accuracies
        )
        
        # Check class accuracy tracking
        self.assertIn(0, self.pab.metrics['class_accuracy'])
        self.assertIn(1, self.pab.metrics['class_accuracy'])
        self.assertIn(2, self.pab.metrics['class_accuracy'])
        self.assertEqual(self.pab.metrics['class_accuracy'][0][0], 0.8)
        self.assertEqual(self.pab.metrics['class_accuracy'][1][0], 0.6)
        self.assertEqual(self.pab.metrics['class_accuracy'][2][0], 0.7)
    
    def test_adversarial_tracking(self):
        """Test adversarial robustness tracking"""
        # Track epoch with adversarial accuracy
        self.pab.track_epoch(
            model=self.model,
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_acc=0.7,
            val_acc=0.65,
            adversarial_acc=0.4
        )
        
        # Check adversarial robustness tracking
        self.assertEqual(len(self.pab.metrics['adversarial_robustness']), 1)
        self.assertEqual(self.pab.metrics['adversarial_robustness'][0], 0.4)

class TestTrajectoryEvaluation(unittest.TestCase):
    """Test cases for trajectory evaluation functions"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir1 = os.path.join(self.temp_dir, 'model1')
        self.checkpoint_dir2 = os.path.join(self.temp_dir, 'model2')
        os.makedirs(self.checkpoint_dir1, exist_ok=True)
        os.makedirs(self.checkpoint_dir2, exist_ok=True)
        
        # Create simple models
        self.model1 = SimpleMLP()
        self.model2 = SimpleMLP()
        
        # Create checkpoint managers
        self.cm1 = CheckpointManager(self.checkpoint_dir1)
        self.cm2 = CheckpointManager(self.checkpoint_dir2)
        
        # Create metrics
        self.metrics1 = {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
            'train_acc': [0.5, 0.6, 0.7, 0.8, 0.85],
            'val_acc': [0.45, 0.55, 0.65, 0.75, 0.8],
            'stability': [0.1, 0.05, 0.04, 0.03],
            'gen_efficiency': [0.1, 0.1, 0.1, 0.1],
        }
        
        self.metrics2 = {
            'train_loss': [0.9, 0.7, 0.5, 0.4, 0.3],
            'val_loss': [1.0, 0.8, 0.6, 0.5, 0.45],
            'train_acc': [0.55, 0.65, 0.75, 0.85, 0.9],
            'val_acc': [0.5, 0.6, 0.7, 0.8, 0.85],
            'stability': [0.08, 0.06, 0.05, 0.04],
            'gen_efficiency': [0.1, 0.1, 0.1, 0.15],
        }
        
        # Save checkpoints
        for epoch in range(1, 6):
            self.cm1.save_checkpoint(self.model1, epoch, self.metrics1)
            self.cm2.save_checkpoint(self.model2, epoch, self.metrics2)
    
    def tearDown(self):
        """Clean up test environment after each test method"""
        shutil.rmtree(self.temp_dir)
    
    def test_evaluate_trajectory(self):
        """Test evaluate_trajectory function"""
        results1 = evaluate_trajectory(self.checkpoint_dir1)
        self.assertIn('overall_stability', results1)
        self.assertIn('generalization', results1)
        
        # Test with second model
        results2 = evaluate_trajectory(self.checkpoint_dir2)
        self.assertIn('overall_stability', results2)
        self.assertIn('generalization', results2)
    
    def test_compare_models(self):
        """Test compare_models function"""
        comparison = compare_models(
            model_dirs=[self.checkpoint_dir1, self.checkpoint_dir2],
            names=['Model1', 'Model2']
        )
        
        # Check comparison results
        self.assertIn('Model1', comparison)
        self.assertIn('Model2', comparison)
        
        # Check metrics in comparison
        self.assertIn('overall_stability', comparison['Model1'])
        self.assertIn('generalization', comparison['Model1'])
        self.assertIn('overall_stability', comparison['Model2'])
        self.assertIn('generalization', comparison['Model2'])

if __name__ == '__main__':
    unittest.main()
