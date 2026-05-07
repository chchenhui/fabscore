#!/usr/bin/env python
"""
Test script to verify cNMF installation
"""

print("Testing cNMF environment...")
print("-" * 50)

# Test core imports
try:
    import cnmf
    print(f"✓ cNMF imported successfully")
    if hasattr(cnmf, '__version__'):
        print(f"  Version: {cnmf.__version__}")
except ImportError as e:
    print(f"✗ Failed to import cNMF: {e}")
    exit(1)

# Test key dependencies
dependencies = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'), 
    ('scipy', 'scipy'),
    ('scikit-learn', 'sklearn'),
    ('scanpy', 'scanpy'),
    ('anndata', 'anndata'),
    ('matplotlib', 'matplotlib'),
    ('harmonypy', 'harmonypy'),
    ('scikit-misc', 'skmisc')
]

print("\nChecking dependencies:")
for display_name, import_name in dependencies:
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✓ {display_name}: {version}")
    except ImportError:
        print(f"✗ {display_name}: Not installed")

# Test basic cNMF functionality
print("\n" + "-" * 50)
print("Testing basic cNMF functionality...")
try:
    from cnmf import cNMF
    print("✓ Successfully imported cNMF class")
    
    # Test that we can access key methods
    methods = ['prepare', 'factorize', 'combine', 'k_selection_plot', 'consensus']
    print("\nVerifying cNMF methods exist:")
    for method in methods:
        if hasattr(cNMF, method):
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} not found")
            
except Exception as e:
    print(f"✗ Error testing cNMF functionality: {e}")

print("\n" + "=" * 50)
print("cNMF environment test complete!")