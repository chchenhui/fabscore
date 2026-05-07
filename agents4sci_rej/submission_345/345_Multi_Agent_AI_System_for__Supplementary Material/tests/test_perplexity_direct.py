#!/usr/bin/env python3
"""
Test Perplexity provider directly
"""

import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

def test_perplexity_direct():
    """Test Perplexity provider directly"""
    
    print("=== PERPLEXITY DIRECT TEST ===")
    
    try:
        from model_router import get_router, TaskType
        
        router = get_router()
        print(f"Available providers: {list(router.providers.keys())}")
        print(f"Perplexity enabled: {router.config['enable_perplexity_review']}")
        
        # Check what provider gets selected for OBJECTIVE_REVIEW
        provider = router.route_task(TaskType.OBJECTIVE_REVIEW)
        print(f"OBJECTIVE_REVIEW routes to: {provider}")
        
        # Try to force enable Perplexity temporarily and test
        if 'perplexity' in router.providers:
            print("\nTesting Perplexity provider directly...")
            
            # Temporarily enable perplexity
            original_setting = router.config['enable_perplexity_review']
            router.config['enable_perplexity_review'] = True
            
            try:
                response = router.generate(
                    prompt="Review this pharmaceutical data for accuracy: Tezspire is a biologic for severe asthma with good efficacy.",
                    task_type=TaskType.OBJECTIVE_REVIEW,
                    system_prompt="You are an objective pharmaceutical reviewer.",
                    max_tokens=200,
                    temperature=0.7
                )
                print(f"SUCCESS: Perplexity response: {len(response.content)} chars, model: {response.model_used}")
                
            except Exception as e:
                print(f"ERROR: Perplexity test failed: {e}")
            
            finally:
                # Restore original setting
                router.config['enable_perplexity_review'] = original_setting
        else:
            print("ERROR: Perplexity provider not available")
        
        print("\nAll provider status:")
        for provider_name in ['openai', 'deepseek', 'perplexity', 'anthropic']:
            status = "✓" if provider_name in router.providers else "✗"
            print(f"  {provider_name}: {status}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_perplexity_direct()