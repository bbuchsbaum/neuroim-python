#!/usr/bin/env python
"""Test that all notebooks execute without errors."""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import warnings

# This module is a CLI-style utility script and should not be collected by pytest.
__test__ = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_notebook(notebook_path):
    """Execute a notebook and return success status."""
    print(f"\nTesting: {os.path.basename(notebook_path)}")
    print("-" * 50)
    
    try:
        # Read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create preprocessor with timeout, using current Python
        ep = ExecutePreprocessor(timeout=600)
        
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        print(f"✓ {os.path.basename(notebook_path)} executed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ {os.path.basename(notebook_path)} failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        
        # Try to find which cell failed
        if hasattr(e, 'traceback'):
            print("\nTraceback:")
            print(e.traceback)
        
        return False

def main():
    """Test all notebooks in the notebooks directory."""
    notebooks_dir = "/Users/bbuchsbaum/code/neuroimpy/docs/source/tutorials/notebooks"
    
    # Get all .ipynb files except index
    notebooks = [
        os.path.join(notebooks_dir, f) 
        for f in os.listdir(notebooks_dir) 
        if f.endswith('.ipynb') and not f.startswith('00_')
    ]
    
    print(f"Found {len(notebooks)} notebooks to test")
    
    results = {}
    for notebook in sorted(notebooks):
        results[notebook] = test_notebook(notebook)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for notebook, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {os.path.basename(notebook)}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
