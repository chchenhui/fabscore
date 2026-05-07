#!/usr/bin/env python3
"""
Minimal Phase 2 test to isolate the hanging issue
"""

import asyncio
import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

async def test_minimal_orchestrator():
    """Test just the orchestrator initialization and first step"""
    
    print("=== MINIMAL PHASE 2 TEST ===")
    
    try:
        print("[STEP 1] Importing orchestrator...")
        from gpt5_orchestrator import GPT5Orchestrator
        print("[STEP 1] Import successful")
        
        print("[STEP 2] Initializing orchestrator...")
        orchestrator = GPT5Orchestrator()
        print("[STEP 2] Orchestrator initialized")
        
        print("[STEP 3] Testing query parsing only...")
        query = "What is Tezspire?"
        print(f"[STEP 3] Query: {query}")
        
        print("[STEP 3] About to call _parse_query...")
        result = await orchestrator._parse_query(query)
        print(f"[STEP 3] Query parsing result: {result}")
        
        print("[SUCCESS] Minimal test completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Starting minimal Phase 2 test...")
    try:
        result = asyncio.run(test_minimal_orchestrator())
        print(f"Final result: {'SUCCESS' if result else 'FAILED'}")
    except Exception as e:
        print(f"MAIN ERROR: {e}")

if __name__ == "__main__":
    main()