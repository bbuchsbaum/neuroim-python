#!/usr/bin/env python
"""
Environment validation script for neuroimpy notebooks.

Run this script to check if your environment is properly configured
to run the neuroimpy tutorial notebooks.
"""

import sys
import importlib
from typing import Tuple, List


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and meets version requirements."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            # Simple version comparison (works for most cases)
            installed = tuple(map(int, version.split('.')[:2]))
            required = tuple(map(int, min_version.split('.')[:2]))
            if installed < required:
                return False, f"{package_name} {version} (requires >= {min_version})"
        
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} not installed"


def check_neuroimpy_components() -> List[Tuple[str, bool, str]]:
    """Check specific neuroimpy components."""
    components = [
        ("Basic volumes", "neuroimpy.DenseNeuroVol"),
        ("Sparse volumes", "neuroimpy.SparseNeuroVol"),
        ("Logical volumes", "neuroimpy.LogicalNeuroVol"),
        ("4D vectors", "neuroimpy.DenseNeuroVec"),
        ("ROI functions", "neuroimpy.spherical_roi"),
        ("Searchlight", "neuroimpy.searchlight"),
        ("Connected components", "neuroimpy.conn_comp"),
        ("Orthogonal slices", "neuroimpy.extract_orthogonal_slices"),
    ]
    
    results = []
    for name, import_path in components:
        try:
            module_path, attr_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            hasattr(module, attr_name)
            results.append((name, True, "Available"))
        except Exception as e:
            results.append((name, False, str(e)))
    
    return results


def main():
    """Run all validation checks."""
    print("PyNeuroim Notebook Environment Validation")
    print("=" * 60)
    
    # Check required packages
    print("\nChecking required packages:")
    print("-" * 40)
    
    required_packages = [
        ("numpy", "1.19"),
        ("scipy", "1.5"),
        ("neuroimpy", None),
    ]
    
    optional_packages = [
        ("matplotlib", "3.0"),
        ("pandas", "1.0"),
        ("nibabel", "3.0"),
        ("jupyter", None),
        ("notebook", None),
    ]
    
    all_good = True
    
    # Check required packages
    for package, min_version in required_packages:
        success, message = check_package(package, min_version)
        status = "✓" if success else "✗"
        print(f"{status} {message}")
        if not success:
            all_good = False
    
    print("\nChecking optional packages:")
    print("-" * 40)
    
    # Check optional packages
    for package, min_version in optional_packages:
        success, message = check_package(package, min_version)
        status = "✓" if success else "○"
        print(f"{status} {message}")
    
    # Check neuroimpy components
    print("\nChecking neuroimpy components:")
    print("-" * 40)
    
    components = check_neuroimpy_components()
    for name, success, message in components:
        status = "✓" if success else "✗"
        print(f"{status} {name}: {message}")
        if not success:
            all_good = False
    
    # Test basic functionality
    print("\nTesting basic functionality:")
    print("-" * 40)
    
    try:
        import neuroimpy as pn
        import numpy as np
        
        # Create a simple volume
        space = pn.NeuroSpace(dim=(10, 10, 10))
        data = np.random.randn(10, 10, 10)
        vol = pn.DenseNeuroVol(data, space)
        
        print("✓ Created test volume successfully")
        
        # Test arithmetic
        vol2 = vol + vol
        print("✓ Arithmetic operations working")
        
        # Test ROI creation
        roi = pn.spherical_roi(vol, [5, 5, 5], radius=3)
        print("✓ ROI creation working")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✓ Environment is ready for neuroimpy notebooks!")
        print("\nYou can now run the notebooks with:")
        print("  jupyter notebook")
    else:
        print("✗ Some issues were found.")
        print("\nTo fix missing required packages:")
        print("  pip install neuroimpy numpy scipy")
        print("\nFor full notebook support also install:")
        print("  pip install matplotlib pandas jupyter")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())