#!/usr/bin/env python3
"""
Simple test runner for the marketplace framework tests.
Demonstrates comprehensive testing without mocks.
"""
import subprocess
import sys
from pathlib import Path

def run_test_suite():
    """Run the complete test suite."""
    print("🧪 Running Marketplace Framework Test Suite")
    print("=" * 60)
    
    # Set up environment
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src"))
    
    test_files = [
        ("Entity Tests", "tests/test_entities.py"),
        ("Ranking Algorithm Tests", "tests/test_ranking_algorithm.py"),
        ("Reputation System Tests", "tests/test_reputation_simple.py"),
        ("Reflection Adjustments Tests", "tests/test_reflection_adjustments.py"),
        ("Validation Tests", "tests/test_validation_and_assumptions.py"),
        ("Decision Logic Tests", "tests/test_decision_logic_simple.py"),
        ("Consistency Tests", "tests/test_prompt_code_consistency.py")
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_file in test_files:
        print(f"\n📋 Running {test_name}...")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode == 0:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                print("Error output:")
                print(result.stdout[-500:])  # Last 500 chars
                failed += 1
                
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passing! Framework is ready for research.")
    else:
        print("⚠️  Some tests failed. Check output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
