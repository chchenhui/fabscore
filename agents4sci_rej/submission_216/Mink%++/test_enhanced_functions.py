#!/usr/bin/env python3
"""
Test script to verify the enhanced functions work correctly.
Tests weighted harmonic mean and sequence coherence penalty in isolation.
"""
import torch
import numpy as np
from improved_proposed_method import compute_weighted_harmonic_mean, compute_sequence_coherence_penalty

def test_weighted_harmonic_mean():
    """Test the weighted harmonic mean function"""
    print("Testing weighted harmonic mean...")
    
    # Test case 1: Normal values
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 2.0, 3.0]
    result = compute_weighted_harmonic_mean(values, weights)
    print(f"Normal case - Values: {values}, Weights: {weights}, Result: {result}")
    
    # Test case 2: With outliers (should be more robust than arithmetic mean)
    values_with_outlier = [1.0, 2.0, 100.0]  # Large outlier
    weights_normal = [1.0, 1.0, 1.0]
    result_hm = compute_weighted_harmonic_mean(values_with_outlier, weights_normal)
    result_am = np.average(values_with_outlier, weights=weights_normal)
    print(f"Outlier test - Values: {values_with_outlier}")
    print(f"  Harmonic mean: {result_hm}")
    print(f"  Arithmetic mean: {result_am}")
    print(f"  Harmonic mean is more robust: {result_hm < result_am}")
    
    # Test case 3: Edge cases
    empty_values = []
    empty_weights = []
    result_empty = compute_weighted_harmonic_mean(empty_values, empty_weights)
    print(f"Empty case result: {result_empty}")
    
    print("Weighted harmonic mean tests completed.\n")

def test_sequence_coherence_penalty():
    """Test the sequence coherence penalty function"""
    print("Testing sequence coherence penalty...")
    
    # Create mock probability distributions
    vocab_size = 1000
    seq_length = 10
    
    # Test case 1: Smooth probability sequence (low penalty expected)
    smooth_probs = torch.softmax(torch.randn(seq_length, vocab_size) * 0.5, dim=-1)  # Low variance
    mock_input_ids = torch.randint(0, vocab_size, (seq_length,))
    
    penalty_smooth = compute_sequence_coherence_penalty(smooth_probs, mock_input_ids)
    print(f"Smooth sequence penalty: {penalty_smooth}")
    
    # Test case 2: Erratic probability sequence (high penalty expected)
    erratic_probs = torch.softmax(torch.randn(seq_length, vocab_size) * 5.0, dim=-1)  # High variance
    penalty_erratic = compute_sequence_coherence_penalty(erratic_probs, mock_input_ids)
    print(f"Erratic sequence penalty: {penalty_erratic}")
    
    # Test case 3: Edge case - short sequence
    short_probs = torch.softmax(torch.randn(1, vocab_size), dim=-1)
    short_input_ids = torch.randint(0, vocab_size, (1,))
    penalty_short = compute_sequence_coherence_penalty(short_probs, short_input_ids)
    print(f"Short sequence penalty: {penalty_short}")
    
    print(f"Erratic sequence has higher penalty: {penalty_erratic > penalty_smooth}")
    print("Sequence coherence penalty tests completed.\n")

def test_integration():
    """Test that the functions work together"""
    print("Testing integration...")
    
    # Mock layer features
    layer_features_1 = {
        'entropy': 5.2,
        'gini': 0.3,
        'top5': 0.4,
        'top10': 0.6,
        'eff_vocab': 0.2
    }
    
    layer_features_2 = {
        'entropy': 4.8,
        'gini': 0.4,
        'top5': 0.5,
        'top10': 0.7,
        'eff_vocab': 0.18
    }
    
    layer_features_list = [layer_features_1, layer_features_2]
    layer_weights = [0.4, 0.6]
    
    # Test weighted harmonic mean aggregation
    print("Testing harmonic mean aggregation for each feature:")
    for feature_name in layer_features_1.keys():
        values = [layer_features_1[feature_name], layer_features_2[feature_name]]
        hm_result = compute_weighted_harmonic_mean(values, layer_weights)
        am_result = np.average(values, weights=layer_weights)
        print(f"  {feature_name}: HM={hm_result:.4f}, AM={am_result:.4f}")
    
    print("Integration tests completed.\n")

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED FUNCTIONS TEST SUITE")
    print("=" * 60)
    
    test_weighted_harmonic_mean()
    test_sequence_coherence_penalty()
    test_integration()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)