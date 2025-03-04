"""
Unit tests for the PAB metrics module.
"""

import unittest
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pab.metrics import (
    learning_stability,
    generalization_efficiency,
    rule_evolution,
    class_wise_progression,
    robustness_evolution,
    representation_similarity,
    learning_curve_predictability,
    overfitting_risk
)

class TestMetricsFunctions(unittest.TestCase):
    """Test cases for PAB metrics functions"""
    
    def test_learning_stability(self):
        """Test learning stability metric"""
        # Stable learning (small change)
        stability = learning_stability(1.0, 0.9)
        self.assertLess(stability, 0.15)
        
        # Unstable learning (large change)
        stability = learning_stability(1.0, 0.5)
        self.assertGreater(stability, 0.4)
        
        # No change
        stability = learning_stability(1.0, 1.0)
        self.assertEqual(stability, 0.0)
    
    def test_generalization_efficiency(self):
        """Test generalization efficiency metric"""
        # Good generalization (small gap)
        gen_eff = generalization_efficiency(0.5, 0.6)
        self.assertEqual(gen_eff, 0.1)
        
        # Poor generalization (large gap)
        gen_eff = generalization_efficiency(0.5, 1.0)
        self.assertEqual(gen_eff, 0.5)
        
        # Perfect generalization (no gap)
        gen_eff = generalization_efficiency(0.5, 0.5)
        self.assertEqual(gen_eff, 0.0)
    
    def test_rule_evolution(self):
        """Test rule evolution metric"""
        # Create feature representations as numpy arrays
        prev_repr = np.array([0.1, 0.2, 0.3, 0.4])
        curr_repr = np.array([0.2, 0.3, 0.4, 0.5])
        
        # Calculate rule evolution
        evol = rule_evolution(prev_repr, curr_repr)
        self.assertGreater(evol, 0.0)
        self.assertLess(evol, 1.0)
        
        # Test with torch tensors
        prev_repr_torch = torch.tensor([0.1, 0.2, 0.3, 0.4])
        curr_repr_torch = torch.tensor([0.2, 0.3, 0.4, 0.5])
        
        evol_torch = rule_evolution(prev_repr_torch, curr_repr_torch)
        self.assertGreater(evol_torch, 0.0)
        self.assertLess(evol_torch, 1.0)
        
        # Test with identical representations
        evol_identical = rule_evolution(prev_repr, prev_repr)
        self.assertAlmostEqual(evol_identical, 0.0, places=6)
    
    def test_class_wise_progression(self):
        """Test class-wise progression analysis"""
        # Create class accuracies
        class_accuracies = {
            0: [0.2, 0.5, 0.8, 0.9],  # Early class
            1: [0.1, 0.2, 0.3, 0.5],  # Late class
            2: [0.7, 0.8, 0.5, 0.9],  # Unstable class
        }
        
        # Analyze class progression
        early, late, unstable = class_wise_progression(class_accuracies)
        
        # Check classification
        self.assertIn(0, early)
        self.assertIn(1, late)
        self.assertIn(2, unstable)
    
    def test_robustness_evolution(self):
        """Test robustness evolution analysis"""
        # Create accuracy lists
        clean_accs = [0.5, 0.7, 0.8, 0.9, 0.92]
        adv_accs = [0.3, 0.5, 0.6, 0.55, 0.5]  # Peak at epoch 3, then degrades
        
        # Analyze robustness evolution
        rob_metrics = robustness_evolution(clean_accs, adv_accs)
        
        # Check metrics
        self.assertEqual(rob_metrics['peak_epoch'], 3)  # 1-indexed
        self.assertEqual(rob_metrics['peak_value'], 0.6)
        self.assertEqual(rob_metrics['final_value'], 0.5)
        
        # Check degradation (from peak to final)
        expected_degradation = (0.6 - 0.5) / 0.6
        self.assertAlmostEqual(rob_metrics['degradation'], expected_degradation)
    
    def test_representation_similarity(self):
        """Test representation similarity metric"""
        # Create feature representations
        repr1 = np.array([0.1, 0.2, 0.3, 0.4])
        repr2 = np.array([0.2, 0.3, 0.4, 0.5])
        
        # Calculate similarity
        sim = representation_similarity(repr1, repr2)
        
        # Check similarity (should be high for similar vectors)
        self.assertGreater(sim, 0.9)
        
        # Test with very different vectors
        repr3 = np.array([-0.2, -0.3, -0.4, -0.5])
        sim2 = representation_similarity(repr1, repr3)
        
        # Should be negative for opposite vectors
        self.assertLess(sim2, 0.0)
    
    def test_learning_curve_predictability(self):
        """Test learning curve predictability metric"""
        # Smooth learning curve
        smooth_curve = [1.0, 0.9, 0.8, 0.7, 0.6]
        smooth_pred = learning_curve_predictability(smooth_curve)
        
        # Erratic learning curve
        erratic_curve = [1.0, 0.5, 0.9, 0.3, 0.7]
        erratic_pred = learning_curve_predictability(erratic_curve)
        
        # Erratic curve should have higher unpredictability (variance)
        self.assertGreater(erratic_pred, smooth_pred)
    
    def test_overfitting_risk(self):
        """Test overfitting risk assessment"""
        # No overfitting case
        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        val_losses = [1.1, 0.9, 0.7, 0.6, 0.5]
        
        risk_metrics = overfitting_risk(train_losses, val_losses)
        
        # Check metrics
        self.assertIn('gap_trend', risk_metrics)
        self.assertIn('early_stopping_epoch', risk_metrics)
        self.assertIn('validation_degradation', risk_metrics)
        self.assertIn('overfitting_detected', risk_metrics)
        
        # Validation degradation should be zero (val loss keeps decreasing)
        self.assertEqual(risk_metrics['validation_degradation'], 0.0)
        
        # Overfitting case
        train_losses2 = [1.0, 0.8, 0.6, 0.4, 0.2]
        val_losses2 = [1.1, 0.9, 0.7, 0.8, 0.9]  # Validation loss starts increasing after epoch 3
        
        risk_metrics2 = overfitting_risk(train_losses2, val_losses2)
        
        # Should detect overfitting
        self.assertTrue(risk_metrics2['overfitting_detected'])
        self.assertEqual(risk_metrics2['early_stopping_epoch'], 3)  # 1-indexed
        
        # Validation degradation should be positive
        self.assertGreater(risk_metrics2['validation_degradation'], 0.2)

if __name__ == '__main__':
    unittest.main()
