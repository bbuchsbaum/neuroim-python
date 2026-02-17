#!/usr/bin/env python
"""
Automated notebook execution test for CI/CD integration.

This script programmatically executes all tutorial notebooks and verifies
they run without errors. It can be integrated into continuous integration
workflows to ensure notebooks remain functional as the library evolves.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# This module is a CLI-style utility script and should not be collected by pytest.
__test__ = False

# Try to import nbformat and nbconvert
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    NOTEBOOK_LIBS_AVAILABLE = True
except ImportError:
    NOTEBOOK_LIBS_AVAILABLE = False
    print("Warning: nbformat and nbconvert not installed. Install with:")
    print("  pip install nbformat nbconvert")


def find_notebooks(directory: Path) -> List[Path]:
    """Find all Jupyter notebooks in the directory."""
    return list(directory.glob("*.ipynb"))


def execute_notebook(notebook_path: Path, timeout: int = 600) -> Tuple[bool, str]:
    """
    Execute a Jupyter notebook and return success status and any error message.
    
    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file
    timeout : int
        Maximum time in seconds to wait for notebook execution
        
    Returns
    -------
    success : bool
        True if notebook executed successfully
    error_msg : str
        Error message if execution failed, empty string otherwise
    """
    if not NOTEBOOK_LIBS_AVAILABLE:
        return False, "nbformat and nbconvert libraries not available"
    
    try:
        # Read notebook
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Create preprocessor
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name='python3',
            allow_errors=False
        )
        
        # Execute notebook
        ep.preprocess(notebook, {'metadata': {'path': notebook_path.parent}})
        
        return True, ""
        
    except Exception as e:
        # Extract detailed error information
        error_msg = f"Error executing {notebook_path.name}:\n"
        error_msg += f"{type(e).__name__}: {str(e)}\n"
        error_msg += traceback.format_exc()
        
        return False, error_msg


def test_notebook_imports(notebook_path: Path) -> Tuple[bool, str]:
    """
    Quick test to check if notebook imports work correctly.
    
    This is a lighter-weight test that just checks imports without
    executing the entire notebook.
    """
    try:
        with open(notebook_path, 'r') as f:
            notebook_data = json.load(f)
        
        # Extract import statements from code cells
        imports = []
        for cell in notebook_data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                lines = source.split('\n')
                for line in lines:
                    if line.strip().startswith(('import ', 'from ')):
                        imports.append(line.strip())
        
        # Try to execute imports
        for import_stmt in imports:
            try:
                exec(import_stmt)
            except ImportError as e:
                return False, f"Import failed: {import_stmt}\n{str(e)}"
            except Exception as e:
                # Ignore other exceptions (might be due to missing context)
                pass
        
        return True, ""
        
    except Exception as e:
        return False, f"Error reading notebook: {str(e)}"


def run_all_tests(notebook_dir: Path, full_execution: bool = True) -> Dict[str, bool]:
    """
    Run tests on all notebooks in the directory.
    
    Parameters
    ----------
    notebook_dir : Path
        Directory containing notebooks
    full_execution : bool
        If True, execute entire notebooks. If False, only test imports.
        
    Returns
    -------
    results : dict
        Dictionary mapping notebook names to success status
    """
    results = {}
    notebooks = find_notebooks(notebook_dir)
    
    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        return results
    
    print(f"Found {len(notebooks)} notebooks to test")
    print("=" * 60)
    
    for notebook_path in sorted(notebooks):
        notebook_name = notebook_path.name
        print(f"\nTesting {notebook_name}...")
        
        if full_execution and NOTEBOOK_LIBS_AVAILABLE:
            success, error_msg = execute_notebook(notebook_path)
            test_type = "Full execution"
        else:
            success, error_msg = test_notebook_imports(notebook_path)
            test_type = "Import test"
        
        results[notebook_name] = success
        
        if success:
            print(f"✓ {test_type} passed")
        else:
            print(f"✗ {test_type} failed")
            if error_msg:
                print(f"  {error_msg}")
    
    return results


def main():
    """Main entry point for the test script."""
    # Get notebook directory
    script_dir = Path(__file__).parent
    notebook_dir = script_dir
    
    # Parse command line arguments
    full_execution = "--full" in sys.argv
    
    if not full_execution and not NOTEBOOK_LIBS_AVAILABLE:
        print("\nRunning import tests only (nbformat/nbconvert not installed)")
    elif full_execution and not NOTEBOOK_LIBS_AVAILABLE:
        print("\nError: Full execution requested but nbformat/nbconvert not installed")
        print("Install with: pip install nbformat nbconvert")
        sys.exit(1)
    
    # Run tests
    print(f"\nTesting notebooks in: {notebook_dir}")
    results = run_all_tests(notebook_dir, full_execution)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    failed = len(results) - passed
    
    for notebook, success in sorted(results.items()):
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {notebook}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
