#!/usr/bin/env python3
"""
Test all 4 main providers individually
"""

import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

def test_all_providers():
    """Test each provider individually"""
    
    from model_router import get_router, TaskType
    
    router = get_router()
    print("=== TESTING ALL PROVIDERS ===")
    
    tests = [
        ("GPT-5", TaskType.COMPLEX_REASONING, "What is Tezspire?"),
        ("DeepSeek", TaskType.BULK_PARSING, "Parse this drug data"),
        ("Perplexity", TaskType.OBJECTIVE_REVIEW, "Review this analysis"),
        ("Claude", TaskType.LONG_CONTEXT, "Analyze this document")
    ]
    
    results = []
    
    for provider_name, task_type, prompt in tests:
        try:
            print(f"\nTesting {provider_name} ({task_type.value})...")
            response = router.generate(prompt, task_type, max_tokens=50)
            print(f"SUCCESS: {response.model_used} - {len(response.content)} chars")
            results.append((provider_name, True, response.model_used))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((provider_name, False, str(e)))
    
    print("\n=== SUMMARY ===")
    for provider_name, success, details in results:
        status = "PASS" if success else "FAIL"
        print(f"{provider_name}: {status} - {details}")
    
    all_working = all(result[1] for result in results)
    print(f"\nAll providers working: {'YES' if all_working else 'NO'}")
    
    return all_working

if __name__ == "__main__":
    test_all_providers()