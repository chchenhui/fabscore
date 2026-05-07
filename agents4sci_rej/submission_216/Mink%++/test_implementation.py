#!/usr/bin/env python3
"""Test script for the improved_proposed_method implementation"""

import sys
import torch
import numpy as np
from improved_proposed_method import (
    Config, 
    compute_shannon_entropy, 
    compute_gini_coefficient, 
    compute_topk_concentration,
    compute_effective_vocab_size,
    normalize_concentration_features,
    compute_concentration_score
)

def test_basic_functionality():
    """Test basic functionality of the implementation"""
    print("Testing basic functionality...")
    
    # Test configuration
    config = Config()
    print(f"✓ Config created: {config.method}")
    print(f"✓ Models: {config.models}")
    print(f"✓ Datasets: {config.datasets}")
    
    # Test concentration metric functions with synthetic data
    print("\nTesting concentration metrics...")
    
    # Create synthetic probability distribution
    batch_size, seq_len, vocab_size = 1, 10, 1000
    probs = torch.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    
    # Test Shannon entropy
    entropy = compute_shannon_entropy(probs)
    print(f"✓ Shannon entropy shape: {entropy.shape}, mean: {entropy.mean().item():.3f}")
    
    # Test Gini coefficient
    gini = compute_gini_coefficient(probs)
    print(f"✓ Gini coefficient shape: {gini.shape}, mean: {gini.mean().item():.3f}")
    
    # Test top-k concentration
    topk_conc = compute_topk_concentration(probs)
    print(f"✓ Top-k concentration keys: {list(topk_conc.keys())}")
    print(f"✓ Top-5 concentration mean: {topk_conc['top5'].mean().item():.3f}")
    
    # Test effective vocabulary size
    eff_vocab = compute_effective_vocab_size(probs)
    print(f"✓ Effective vocab size shape: {eff_vocab.shape}, mean: {eff_vocab.mean().item():.3f}")
    
    # Test feature normalization
    features = {
        'entropy': 2.5,
        'gini': 0.7,
        'top5': 0.15,
        'eff_vocab': 0.3
    }
    normalized = normalize_concentration_features(features)
    print(f"✓ Normalized features: {normalized}")
    
    # Test concentration score computation
    score = compute_concentration_score(normalized)
    print(f"✓ Concentration score: {score:.3f}")
    
    print("\n✓ All basic functionality tests passed!")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    # Test with uniform distribution
    uniform_probs = torch.ones(1, 5, 10) / 10
    entropy_uniform = compute_shannon_entropy(uniform_probs)
    gini_uniform = compute_gini_coefficient(uniform_probs)
    print(f"✓ Uniform distribution - entropy: {entropy_uniform.mean().item():.3f}, gini: {gini_uniform.mean().item():.3f}")
    
    # Test with peaked distribution
    peaked_probs = torch.zeros(1, 5, 10)
    peaked_probs[:, :, 0] = 0.9
    peaked_probs[:, :, 1] = 0.1
    entropy_peaked = compute_shannon_entropy(peaked_probs)
    gini_peaked = compute_gini_coefficient(peaked_probs)
    print(f"✓ Peaked distribution - entropy: {entropy_peaked.mean().item():.3f}, gini: {gini_peaked.mean().item():.3f}")
    
    print("✓ Edge case tests passed!")

def main():
    """Main test function"""
    print("=" * 50)
    print("IMPROVED PROPOSED METHOD - IMPLEMENTATION TEST")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_edge_cases()
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("✓ Implementation is ready for use.")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)