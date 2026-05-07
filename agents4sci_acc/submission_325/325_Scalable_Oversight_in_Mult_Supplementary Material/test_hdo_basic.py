#!/usr/bin/env python3
"""
Basic HDO System Test

Tests core HDO functionality without heavy dependencies.
"""

import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_hdo():
    """Test basic HDO system functionality"""
    
    print("Testing HDO System Components...")
    print("-" * 40)
    
    # Test imports
    try:
        from nsrag.hdo import (
            HDOSystem, HDOConfig, HDOMode,
            DebateTree, ClaimType,
            NLIVerifier, CodeVerifier, RuleVerifier, RetrievalVerifier,
            AggregationMethod, RoutingStrategy
        )
        print("✓ All HDO modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test HDO configuration
    try:
        config = HDOConfig(
            tau_reject=0.2,
            tau_accept=0.8,
            max_delegation_depth=3,
            budget_limit=100.0
        )
        print("✓ HDO configuration created successfully")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Test HDO system initialization
    try:
        hdo_system = HDOSystem(config)
        print(f"✓ HDO system initialized with {len(hdo_system.verifiers)} verifiers")
    except Exception as e:
        print(f"✗ System initialization error: {e}")
        return False
    
    # Test debate tree creation
    try:
        tree = DebateTree("Is this outcome aligned?", {"test": "context"})
        print(f"✓ Debate tree created with root node: {tree.root_id}")
    except Exception as e:
        print(f"✗ Debate tree error: {e}")
        return False
    
    # Test verifier creation
    try:
        nli_verifier = NLIVerifier("test_nli")
        code_verifier = CodeVerifier("test_code")
        rule_verifier = RuleVerifier("test_rule")
        retrieval_verifier = RetrievalVerifier("test_retrieval")
        
        print("✓ All verifier types created successfully")
        print(f"  - NLI verifier supports: {[ct.value for ct in nli_verifier.supported_claim_types]}")
        print(f"  - Code verifier supports: {[ct.value for ct in code_verifier.supported_claim_types]}")
        print(f"  - Rule verifier supports: {[ct.value for ct in rule_verifier.supported_claim_types]}")
        print(f"  - Retrieval verifier supports: {[ct.value for ct in retrieval_verifier.supported_claim_types]}")
    except Exception as e:
        print(f"✗ Verifier creation error: {e}")
        return False
    
    # Test simple oversight episode
    try:
        print("\nTesting simple oversight episode...")
        
        episode = hdo_system.conduct_oversight(
            outcome="This is a test outcome that should be verified for alignment",
            context={
                'domain': 'test',
                'constraints': ['safety', 'accuracy'],
                'goals': ['helpfulness', 'harmlessness']
            },
            ground_truth=True
        )
        
        print(f"✓ Oversight episode completed:")
        print(f"  - Episode ID: {episode.episode_id}")
        print(f"  - Final decision: {episode.final_decision}")
        print(f"  - Confidence: {episode.confidence:.3f}")
        print(f"  - Total cost: ${episode.total_cost:.2f}")
        print(f"  - Delegation depth: {episode.delegation_depth_reached}")
        print(f"  - Status: {episode.status}")
        
        if episode.risk_bound:
            print(f"  - Risk bound: {episode.risk_bound.combined_bound:.4f}")
            print(f"  - Delegation benefit: {episode.risk_bound.delegation_benefit:.4f}")
        
        if episode.debate_tree:
            stats = episode.debate_tree.get_stats()
            print(f"  - Tree nodes: {stats['total_nodes']}")
            print(f"  - Leaf nodes: {stats['leaf_nodes']}")
        
    except Exception as e:
        print(f"✗ Oversight episode error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test system statistics
    try:
        stats = hdo_system.get_system_statistics()
        print(f"\n✓ System statistics retrieved:")
        if 'episode_statistics' in stats:
            ep_stats = stats['episode_statistics']
            print(f"  - Total episodes: {ep_stats.get('total_episodes', 0)}")
            print(f"  - Completed episodes: {ep_stats.get('completed_episodes', 0)}")
            print(f"  - Average cost: ${ep_stats.get('average_cost', 0):.2f}")
    except Exception as e:
        print(f"✗ Statistics error: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("✓ ALL TESTS PASSED - HDO System is working correctly!")
    print("=" * 40)
    
    return True


if __name__ == "__main__":
    success = test_basic_hdo()
    sys.exit(0 if success else 1)
