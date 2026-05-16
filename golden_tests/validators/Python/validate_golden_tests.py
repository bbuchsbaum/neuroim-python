#!/usr/bin/env python3
"""
Golden Test Validator for neuroim

This module provides tools to validate golden tests that ensure semantic equivalence
across different language implementations (R, Python, Rust) of neuroimaging software.
"""

import xml.etree.ElementTree as ET
import numpy as np
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import traceback
from dataclasses import dataclass
from enum import Enum


class CheckType(Enum):
    """Types of numeric checks supported."""
    EXACT_VALUE = "exact_value"
    RANGE = "range"
    RELATIVE = "relative"
    STATISTICAL = "statistical"
    DIMENSION = "dimension"


@dataclass
class NumericCheck:
    """Represents a numeric check from the golden test XML."""
    check_type: CheckType
    name: str
    location: str
    expected: Any
    tolerance: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class TestResult:
    """Result of a single numeric check."""
    check_name: str
    passed: bool
    actual_value: Any
    expected_value: Any
    message: str


class GoldenTestValidator:
    """Validates golden tests for neuroim."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.namespace = {'gt': 'http://golden-tests.org/schema'}
    
    def parse_golden_test(self, xml_path: str) -> Dict[str, Any]:
        """Parse a golden test XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract metadata
        metadata = {
            'id': root.find('.//gt:id', self.namespace).text,
            'version': root.find('.//gt:version', self.namespace).text,
            'description': root.find('.//gt:description', self.namespace).text,
            'tags': [tag.text for tag in root.findall('.//gt:tag', self.namespace)]
        }
        
        # Extract semantic description
        semantic = {
            'purpose': root.find('.//gt:purpose', self.namespace).text.strip(),
            'algorithm': root.find('.//gt:algorithm', self.namespace).text.strip()
        }
        
        # Extract inputs
        inputs = self._parse_inputs(root)
        
        # Extract expected outputs
        checks = self._parse_numeric_checks(root)
        
        # Extract Python implementation if exists.
        # sync_golden_tests.py emits the <implementations> subtree without the
        # golden-tests namespace, so fall back to a namespace-agnostic lookup.
        python_impl = root.find('.//gt:Python', self.namespace)
        if python_impl is None:
            python_impl = root.find('.//Python')
        python_code = python_impl.text if python_impl is not None else None
        
        return {
            'metadata': metadata,
            'semantic': semantic,
            'inputs': inputs,
            'checks': checks,
            'python_code': python_code
        }
    
    def _parse_inputs(self, root: ET.Element) -> Dict[str, Any]:
        """Parse input parameters and data from XML."""
        inputs = {}
        
        # Parse parameters
        params = {}
        for param in root.findall('.//gt:parameter', self.namespace):
            name = param.find('gt:name', self.namespace).text
            value = param.find('gt:value', self.namespace).text
            params[name] = self._parse_value(value)
        inputs['parameters'] = params
        
        # Parse vectors
        vectors = {}
        for vector in root.findall('.//gt:vector', self.namespace):
            name = vector.find('gt:name', self.namespace).text
            values = vector.find('gt:values', self.namespace).text
            vectors[name] = self._parse_value(values)
        inputs['vectors'] = vectors
        
        # Parse matrices
        matrices = {}
        for matrix in root.findall('.//gt:matrix', self.namespace):
            name = matrix.find('gt:name', self.namespace).text
            
            # Handle different ways of specifying dimensions
            dims_elem = matrix.find('gt:dimensions', self.namespace)
            if dims_elem is not None:
                # Format: <dimensions>8 3</dimensions>
                dims = [int(x) for x in dims_elem.text.split()]
            else:
                # Format: <rows>8</rows><cols>3</cols>
                rows_elem = matrix.find('gt:rows', self.namespace)
                cols_elem = matrix.find('gt:cols', self.namespace)
                if rows_elem is not None and cols_elem is not None:
                    dims = [int(rows_elem.text), int(cols_elem.text)]
                else:
                    continue
            
            values = matrix.find('gt:values', self.namespace).text
            mat_values = self._parse_value(values)
            
            # Handle multi-line matrix values
            if isinstance(mat_values, str):
                # Clean up multi-line values
                clean_values = ' '.join(values.strip().split())
                mat_values = self._parse_value(clean_values)
            
            matrices[name] = np.array(mat_values).reshape(dims)
        inputs['matrices'] = matrices
        
        # Parse arrays (3D+ data)
        arrays = {}
        for array in root.findall('.//gt:array', self.namespace):
            name = array.find('gt:name', self.namespace).text
            dims_elem = array.find('gt:dimensions', self.namespace)
            if dims_elem is not None:
                dims = [int(x) for x in dims_elem.text.split()]
            else:
                continue
            
            values = array.find('gt:values', self.namespace).text
            arr_values = self._parse_value(values)
            
            # Handle multi-line array values
            if isinstance(arr_values, str):
                clean_values = ' '.join(values.strip().split())
                arr_values = self._parse_value(clean_values)
            
            arrays[name] = np.array(arr_values).reshape(dims)
        inputs['arrays'] = arrays
        
        return inputs
    
    def _parse_numeric_checks(self, root: ET.Element) -> List[NumericCheck]:
        """Parse numeric checks from XML."""
        checks = []
        
        for check in root.findall('.//gt:check', self.namespace):
            check_type = CheckType(check.find('gt:type', self.namespace).text)
            name = check.find('gt:name', self.namespace).text
            location = check.find('gt:location', self.namespace).text
            
            # Expected value is optional for range checks
            expected_elem = check.find('gt:expected', self.namespace)
            expected = self._parse_value(expected_elem.text) if expected_elem is not None else None
            
            # Parse tolerance if present
            tolerance_elem = check.find('gt:tolerance', self.namespace)
            tolerance = float(tolerance_elem.text) if tolerance_elem is not None else None
            
            # Parse range values if present
            min_elem = check.find('gt:min', self.namespace)
            max_elem = check.find('gt:max', self.namespace)
            min_value = float(min_elem.text) if min_elem is not None else None
            max_value = float(max_elem.text) if max_elem is not None else None
            
            checks.append(NumericCheck(
                check_type=check_type,
                name=name,
                location=location,
                expected=expected,
                tolerance=tolerance,
                min_value=min_value,
                max_value=max_value
            ))
        
        return checks
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string into appropriate Python type."""
        # Try to parse as numeric array
        try:
            values = [float(x) for x in value_str.split()]
            return values if len(values) > 1 else values[0]
        except ValueError:
            # Return as string if not numeric
            return value_str
    
    def validate_test(self, xml_path: str) -> Tuple[bool, List[TestResult]]:
        """Validate a single golden test."""
        test_data = self.parse_golden_test(xml_path)
        
        if test_data['python_code'] is None:
            return False, [TestResult(
                check_name="implementation",
                passed=False,
                actual_value=None,
                expected_value="Python implementation",
                message="No Python implementation found in test"
            )]
        
        # Execute Python code and capture results
        try:
            results = self._execute_python_code(test_data)
            return self._validate_results(results, test_data['checks'])
        except Exception as e:
            return False, [TestResult(
                check_name="execution",
                passed=False,
                actual_value=str(e),
                expected_value="Successful execution",
                message=f"Error executing Python code: {str(e)}"
            )]
    
    def _execute_python_code(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Python implementation and return results."""
        # Create execution environment with inputs
        exec_globals = {
            'np': np,
            '__builtins__': __builtins__,
        }
        
        # Add input parameters to namespace
        for name, value in test_data['inputs']['parameters'].items():
            exec_globals[name] = value
        
        for name, value in test_data['inputs'].get('vectors', {}).items():
            exec_globals[name] = value
            
        for name, value in test_data['inputs'].get('matrices', {}).items():
            exec_globals[name] = value
            
        for name, value in test_data['inputs'].get('arrays', {}).items():
            exec_globals[name] = value
        
        # Execute the Python code
        exec(test_data['python_code'], exec_globals)
        
        # Extract variables that match check locations
        results = {}
        for check in test_data['checks']:
            location = check.location
            try:
                # Try to evaluate the location as a Python expression
                # This handles function calls, attribute access, indexing, etc.
                result = eval(location, exec_globals)
                results[location] = result
            except Exception as e:
                # If direct evaluation fails, try some common transformations
                try:
                    # Handle R-style array indexing: bounds_result[,1] -> bounds_result[:,0]
                    if '[,' in location:
                        # R uses 1-based column indexing, Python uses 0-based
                        py_location = location.replace('[,', '[:,')
                        # Convert column indices from 1-based to 0-based
                        import re
                        py_location = re.sub(r',(\d+)\]', lambda m: f',{int(m.group(1))-1}]', py_location)
                        # Special handling for 4D arrays with [,,,n] pattern  
                        # sum(vec4d[,,,1]) -> sum(vec4d[:,:,:,0])
                        if py_location.count(':,') == 2 and py_location.endswith(']'):
                            # This is likely a 4D array access like vec4d[:,:,:,0]
                            # Try to evaluate it directly
                            result = eval(py_location, exec_globals)
                        else:
                            result = eval(py_location, exec_globals)
                        results[location] = result
                    elif '[' in location and ',' not in location:
                        # Handle 1D indexing: vol[1] -> vol[0]
                        import re
                        match = re.match(r'(\w+)\[(\d+)\]', location)
                        if match:
                            var_name, idx = match.groups()
                            py_location = f"{var_name}[{int(idx)-1}]"
                            result = eval(py_location, exec_globals)
                            results[location] = result
                    else:
                        raise e
                except:
                    if self.verbose:
                        print(f"Warning: Could not extract {location}: {e}")
        
        return results
    
    def _validate_results(self, results: Dict[str, Any], checks: List[NumericCheck]) -> Tuple[bool, List[TestResult]]:
        """Validate results against expected checks."""
        test_results = []
        all_passed = True
        
        for check in checks:
            if check.location not in results:
                test_results.append(TestResult(
                    check_name=check.name,
                    passed=False,
                    actual_value=None,
                    expected_value=check.expected,
                    message=f"Result '{check.location}' not found in execution output"
                ))
                all_passed = False
                continue
            
            actual = results[check.location]
            passed, message = self._perform_check(actual, check)
            
            test_results.append(TestResult(
                check_name=check.name,
                passed=passed,
                actual_value=actual,
                expected_value=check.expected,
                message=message
            ))
            
            if not passed:
                all_passed = False
        
        return all_passed, test_results
    
    def _perform_check(self, actual: Any, check: NumericCheck) -> Tuple[bool, str]:
        """Perform a single numeric check."""
        if check.check_type == CheckType.DIMENSION:
            return self._check_dimension(actual, check.expected)
        elif check.check_type == CheckType.EXACT_VALUE:
            return self._check_exact_value(actual, check.expected, check.tolerance)
        elif check.check_type == CheckType.RANGE:
            return self._check_range(actual, check.min_value, check.max_value)
        elif check.check_type == CheckType.RELATIVE:
            return self._check_relative(actual, check.expected, check.tolerance)
        elif check.check_type == CheckType.STATISTICAL:
            return self._check_statistical(actual, check)
        else:
            return False, f"Unknown check type: {check.check_type}"
    
    def _check_dimension(self, actual: Any, expected: Any) -> Tuple[bool, str]:
        """Check dimensions match."""
        # For dimension checks, we're comparing dimension values
        # If actual is a numpy array, we want to compare its shape to expected dimensions
        # If actual is already a list/tuple of dimensions, compare directly
        
        # Extract dimensions from numpy arrays.
        # For 1D vectors that already represent dimensions (e.g., dim(space) -> [4,4,4]),
        # compare values directly rather than the vector's shape ([3]).
        if isinstance(actual, np.ndarray):
            if actual.ndim == 0:
                actual_dims = [actual.item()]
            elif actual.ndim == 1:
                actual_dims = list(actual)
            else:
                actual_dims = list(actual.shape)
        elif hasattr(actual, 'shape'):
            actual_dims = list(actual.shape)
        elif hasattr(actual, '__len__') and not isinstance(actual, str):
            # It's already a list/tuple of dimensions
            actual_dims = list(actual)
        else:
            # Single value, wrap in list
            actual_dims = [actual]
        
        # Convert expected to list if it isn't already
        if hasattr(expected, '__len__') and not isinstance(expected, str):
            expected_dims = list(expected)
        else:
            expected_dims = [expected]
        
        # Convert to arrays for comparison
        actual_arr = np.array(actual_dims)
        expected_arr = np.array(expected_dims)
        
        # Check if the values match
        if np.allclose(actual_arr, expected_arr):
            return True, "Dimensions match"
        else:
            return False, f"Expected dimensions {expected_arr.tolist()}, got {actual_arr.tolist()}"
    
    def _check_exact_value(self, actual: Any, expected: Any, tolerance: Optional[float] = None) -> Tuple[bool, str]:
        """Check exact values with optional tolerance."""
        if tolerance is None:
            tolerance = 0
        
        # Convert to numpy arrays for easier comparison
        actual_arr = np.atleast_1d(actual)
        expected_arr = np.atleast_1d(expected)
        
        if actual_arr.shape != expected_arr.shape:
            return False, f"Shape mismatch: {actual_arr.shape} vs {expected_arr.shape}"
        
        if np.allclose(actual_arr, expected_arr, atol=tolerance, rtol=0):
            return True, "Values match within tolerance"
        else:
            max_diff = np.max(np.abs(actual_arr - expected_arr))
            return False, f"Maximum difference {max_diff} exceeds tolerance {tolerance}"
    
    def _check_range(self, actual: Any, min_val: float, max_val: float) -> Tuple[bool, str]:
        """Check values are within range."""
        actual_arr = np.atleast_1d(actual)
        
        if np.all((actual_arr >= min_val) & (actual_arr <= max_val)):
            return True, f"All values within range [{min_val}, {max_val}]"
        else:
            out_of_range = np.sum((actual_arr < min_val) | (actual_arr > max_val))
            return False, f"{out_of_range} values outside range [{min_val}, {max_val}]"
    
    def _check_relative(self, actual: Any, expected: Any, tolerance: float) -> Tuple[bool, str]:
        """Check relative difference."""
        actual_arr = np.atleast_1d(actual)
        expected_arr = np.atleast_1d(expected)
        
        if np.allclose(actual_arr, expected_arr, atol=0, rtol=tolerance):
            return True, f"Values match within {tolerance*100}% relative tolerance"
        else:
            max_rel_diff = np.max(np.abs((actual_arr - expected_arr) / expected_arr))
            return False, f"Maximum relative difference {max_rel_diff:.2%} exceeds {tolerance:.2%}"
    
    def _check_statistical(self, actual: Any, check: NumericCheck) -> Tuple[bool, str]:
        """Check statistical properties."""
        # This would need more implementation based on specific statistical checks
        return False, "Statistical checks not yet implemented"


def validate_all_golden_tests(specs_dir: str, verbose: bool = False) -> Dict[str, Tuple[bool, List[TestResult]]]:
    """Validate all golden tests in a directory."""
    validator = GoldenTestValidator(verbose=verbose)
    results = {}
    
    specs_path = Path(specs_dir)
    for xml_file in specs_path.rglob("*.xml"):
        if verbose:
            print(f"\nValidating: {xml_file}")
        
        passed, test_results = validator.validate_test(str(xml_file))
        results[str(xml_file)] = (passed, test_results)
        
        if verbose:
            print(f"  Result: {'PASSED' if passed else 'FAILED'}")
            for result in test_results:
                if not result.passed:
                    print(f"    ✗ {result.check_name}: {result.message}")
    
    return results


def print_summary(results: Dict[str, Tuple[bool, List[TestResult]]]) -> None:
    """Print summary of test results."""
    total_tests = len(results)
    passed_tests = sum(1 for passed, _ in results.values() if passed)
    
    print(f"\nGolden Test Summary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    
    if passed_tests < total_tests:
        print("\nFailed tests:")
        for test_path, (passed, test_results) in results.items():
            if not passed:
                print(f"\n  {Path(test_path).name}:")
                for result in test_results:
                    if not result.passed:
                        print(f"    - {result.check_name}: {result.message}")


if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate golden tests for neuroim")
    parser.add_argument("path", help="Path to test XML file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Single file validation
        validator = GoldenTestValidator(verbose=args.verbose)
        passed, results = validator.validate_test(args.path)
        
        print(f"Test: {Path(args.path).name}")
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
        
        if not passed or args.verbose:
            for result in results:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.check_name}: {result.message}")
                if args.verbose and not result.passed:
                    print(f"    Expected: {result.expected_value}")
                    print(f"    Actual: {result.actual_value}")
    else:
        # Directory validation
        results = validate_all_golden_tests(args.path, verbose=args.verbose)
        print_summary(results)
