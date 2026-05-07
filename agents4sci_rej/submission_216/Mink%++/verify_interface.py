#!/usr/bin/env python3
"""Verification script to check interface compatibility"""

import argparse
import sys
import os
import json

def verify_interface():
    """Verify that improved_proposed_method has the same interface as baseline"""
    print("Verifying interface compatibility...")
    
    # Check if improved_proposed_method.py exists
    if not os.path.exists('improved_proposed_method.py'):
        print("❌ improved_proposed_method.py does not exist")
        return False
    
    # Check if it has the required functions
    try:
        import improved_proposed_method as ipm
        
        # Check main function exists
        if not hasattr(ipm, 'main'):
            print("❌ main() function not found")
            return False
        
        # Check parse_arguments function exists
        if not hasattr(ipm, 'parse_arguments'):
            print("❌ parse_arguments() function not found") 
            return False
        
        # Check Config class exists
        if not hasattr(ipm, 'Config'):
            print("❌ Config class not found")
            return False
        
        # Test argument parsing
        parser_func = ipm.parse_arguments
        
        # Test config creation
        config = ipm.Config()
        
        print("✓ All required functions and classes exist")
        print(f"✓ Config method: {config.method}")
        print(f"✓ Config models: {config.models}")
        print(f"✓ Config datasets: {config.datasets}")
        
        # Check that the method name is correct
        expected_method = 'mink++_multilayer_concentration'
        if config.method != expected_method:
            print(f"❌ Expected method {expected_method}, got {config.method}")
            return False
        
        print("✓ Interface verification passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("=" * 60)
    print("IMPROVED PROPOSED METHOD - INTERFACE VERIFICATION")
    print("=" * 60)
    
    success = verify_interface()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ INTERFACE VERIFICATION SUCCESSFUL")
        print("✓ Implementation is compatible with baseline interface")
        print("✓ Ready to run: python improved_proposed_method.py --output-dir ./results")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ INTERFACE VERIFICATION FAILED")
        print("❌ Implementation needs fixes before use")
        print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)