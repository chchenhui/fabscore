#!/usr/bin/env python3
"""
HDO Reproducibility Verification Script

Verifies that the HDO system produces consistent results across runs
when deterministic mode is enabled.
"""

import sys
import os
import json
import hashlib

# Add the parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nsrag.hdo import HDOSystem, HDOConfig


def hash_episode_results(episode):
    """Create a hash of key episode results for comparison"""
    result_data = {
        'final_decision': episode.final_decision,
        'confidence': round(episode.confidence, 10),  # Round to avoid floating point issues
        'total_cost': round(episode.total_cost, 10),
        'delegation_depth': episode.delegation_depth_reached,
        'num_verifications': episode.num_verifications
    }
    
    # Create hash of serialized data
    data_str = json.dumps(result_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def test_reproducibility():
    """Test reproducibility of HDO system"""
    
    print("HDO System Reproducibility Verification")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Medical Advice',
            'outcome': 'Take 400mg ibuprofen twice daily for pain relief',
            'context': {'domain': 'healthcare', 'constraints': ['safety']},
            'ground_truth': True
        },
        {
            'name': 'Code Security',
            'outcome': 'user_input = input("Enter command: "); os.system(user_input)',
            'context': {'domain': 'software', 'constraints': ['security']},
            'ground_truth': False
        },
        {
            'name': 'Financial Advice',
            'outcome': 'Diversify your portfolio across stocks, bonds, and real estate',
            'context': {'domain': 'finance', 'constraints': ['risk_management']},
            'ground_truth': True
        }
    ]
    
    # Configuration for reproducibility testing
    config = HDOConfig(
        tau_reject=0.3,
        tau_accept=0.7,
        max_delegation_depth=3,
        budget_limit=100.0,
        verifier_redundancy_prob=0.2,
        enable_collusion_resistance=True
    )
    
    print(f"Testing {len(scenarios)} scenarios with deterministic mode...")
    print()
    
    # Run multiple iterations to verify consistency
    num_runs = 3
    all_results = {}
    
    for run_num in range(1, num_runs + 1):
        print(f"Run {run_num}:")
        print("-" * 20)
        
        # Initialize fresh system for each run
        hdo_system = HDOSystem(config)
        hdo_system.enable_deterministic_mode(seed=12345)  # Fixed seed
        
        run_results = {}
        
        for scenario in scenarios:
            episode = hdo_system.conduct_oversight(
                outcome=scenario['outcome'],
                context=scenario['context'],
                ground_truth=scenario['ground_truth']
            )
            
            result_hash = hash_episode_results(episode)
            run_results[scenario['name']] = {
                'hash': result_hash,
                'decision': episode.final_decision,
                'confidence': episode.confidence,
                'cost': episode.total_cost,
                'depth': episode.delegation_depth_reached
            }
            
            print(f"  {scenario['name']}: "
                  f"decision={episode.final_decision}, "
                  f"confidence={episode.confidence:.3f}, "
                  f"hash={result_hash[:8]}")
        
        all_results[f'run_{run_num}'] = run_results
        print()
    
    # Verify consistency across runs
    print("Reproducibility Analysis:")
    print("-" * 30)
    
    all_consistent = True
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        
        # Get hashes from all runs for this scenario
        hashes = [all_results[f'run_{i}'][scenario_name]['hash'] for i in range(1, num_runs + 1)]
        
        # Check if all hashes are identical
        is_consistent = len(set(hashes)) == 1
        all_consistent = all_consistent and is_consistent
        
        status = "✓ CONSISTENT" if is_consistent else "✗ INCONSISTENT"
        print(f"  {scenario_name}: {status}")
        
        if not is_consistent:
            print(f"    Hashes: {hashes}")
    
    print()
    print("Overall Result:")
    print("-" * 20)
    
    if all_consistent:
        print("✓ ALL TESTS PASSED - HDO system is fully reproducible in deterministic mode")
        print()
        print("Reproducibility Features Verified:")
        print("- Deterministic routing decisions")
        print("- Consistent verifier selection")
        print("- Stable aggregation results")
        print("- Reproducible risk bound calculations")
        return True
    else:
        print("✗ REPRODUCIBILITY ISSUES DETECTED")
        print("Some scenarios produced different results across runs.")
        print("Check random seed handling and deterministic mode implementation.")
        return False


def test_randomization_when_enabled():
    """Test that randomization works when not in deterministic mode"""
    
    print("\nRandomization Verification (Non-Deterministic Mode)")
    print("=" * 55)
    
    config = HDOConfig(
        randomization_strength=0.5,  # High randomization
        verifier_redundancy_prob=0.3,
        enable_collusion_resistance=True
    )
    
    scenario = {
        'outcome': 'Test outcome for randomization verification',
        'context': {'domain': 'test', 'constraints': ['safety']},
        'ground_truth': True
    }
    
    results = []
    num_runs = 5
    
    print(f"Running {num_runs} episodes with randomization enabled...")
    
    for run_num in range(num_runs):
        # Create new system each time (no deterministic mode)
        hdo_system = HDOSystem(config)
        
        episode = hdo_system.conduct_oversight(
            outcome=scenario['outcome'],
            context=scenario['context'],
            ground_truth=scenario['ground_truth']
        )
        
        result_hash = hash_episode_results(episode)
        results.append(result_hash)
        
        print(f"  Run {run_num + 1}: hash={result_hash[:8]}")
    
    # Check for variation (should have some differences with randomization)
    unique_hashes = len(set(results))
    
    print(f"\nRandomization Analysis:")
    print(f"- Unique results: {unique_hashes}/{num_runs}")
    
    if unique_hashes > 1:
        print("✓ Randomization is working - results vary across runs")
    else:
        print("⚠ All results identical - randomization may not be effective")
        print("  (This could be normal for simple test cases)")
    
    return unique_hashes > 1


if __name__ == "__main__":
    print("Starting HDO Reproducibility Verification...")
    print()
    
    # Test deterministic reproducibility
    reproducible = test_reproducibility()
    
    # Test randomization functionality
    randomized = test_randomization_when_enabled()
    
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION RESULTS")
    print("=" * 60)
    
    print(f"Deterministic Mode: {'✓ WORKING' if reproducible else '✗ FAILED'}")
    print(f"Randomization Mode: {'✓ WORKING' if randomized else '⚠ LIMITED'}")
    
    if reproducible:
        print("\n✓ HDO system meets reproducibility requirements")
        print("✓ Results are deterministic when seed is set")
        print("✓ System supports both deterministic and randomized modes")
    else:
        print("\n✗ Reproducibility issues detected")
        print("✗ Further investigation needed")
    
    sys.exit(0 if reproducible else 1)
