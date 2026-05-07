#!/usr/bin/env python3
"""
Test AutoSurvey setup - verify all dependencies are working
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing AutoSurvey dependencies...")
    print("-" * 50)
    
    imports = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("tiktoken", "TikToken"),
        ("tinydb", "TinyDB"),
        ("h5py", "H5PY"),
    ]
    
    success = True
    for module_name, display_name in imports:
        try:
            __import__(module_name)
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"✗ {display_name} failed: {e}")
            success = False
    
    print("-" * 50)
    
    # Test local AutoSurvey modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    local_imports = [
        ("src.model", "AutoSurvey Model"),
        ("src.database", "AutoSurvey Database"),
        ("src.agents.outline_writer", "Outline Writer"),
        ("src.agents.writer", "Writer"),
        ("src.agents.judge", "Judge"),
    ]
    
    for module_name, display_name in local_imports:
        try:
            __import__(module_name)
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"✗ {display_name} failed: {e}")
            success = False
    
    print("-" * 50)
    
    # Check PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available, will use CPU")
    except:
        print("ℹ Could not check CUDA availability")
    
    print("-" * 50)
    
    if success:
        print("✓ All dependencies imported successfully!")
        print("\nYou can now run AutoSurvey with:")
        print("  python main.py --topic 'your topic' --api_key 'your_key'")
        print("\nOr with local API:")
        print("  python run_local_survey.py")
    else:
        print("✗ Some dependencies failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)