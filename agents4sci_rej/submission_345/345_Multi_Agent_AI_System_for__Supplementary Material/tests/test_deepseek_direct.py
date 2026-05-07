#!/usr/bin/env python3
"""
Direct DeepSeek API test to isolate the hanging issue
"""

import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

def test_deepseek_direct():
    """Test DeepSeek API directly"""
    
    print("=== DIRECT DEEPSEEK TEST ===")
    
    try:
        from model_router import get_router, TaskType
        
        print("[TEST] Initializing router...")
        router = get_router()
        print("[TEST] Router initialized")
        
        # Test 1: Simple classification (this worked before)
        print("\n[TEST] Test 1: Simple classification...")
        response1 = router.generate(
            prompt="What is Tezspire?",
            task_type=TaskType.CLASSIFICATION,
            system_prompt="You are a pharmaceutical expert.",
            max_tokens=100
        )
        print(f"[TEST] Test 1 SUCCESS: {len(response1.content)} chars, model: {response1.model_used}")
        
        # Test 2: Bulk parsing (this is where it hangs)
        print("\n[TEST] Test 2: Bulk parsing...")
        print("[TEST] About to call router.generate for bulk parsing...")
        
        response2 = router.generate(
            prompt="Collect pharmaceutical data for Tezspire including FDA approval, financials, and clinical data.",
            task_type=TaskType.BULK_PARSING,
            system_prompt="You are a pharmaceutical data collection specialist.",
            max_tokens=500
        )
        print(f"[TEST] Test 2 SUCCESS: {len(response2.content)} chars, model: {response2.model_used}")
        
        print("\n[SUCCESS] Both DeepSeek calls completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing DeepSeek API directly...")
    result = test_deepseek_direct()
    print(f"Final result: {'SUCCESS' if result else 'FAILED'}")