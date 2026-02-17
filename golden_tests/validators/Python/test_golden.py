#!/usr/bin/env python3
"""
Pytest integration for golden tests.

This module provides pytest fixtures and test generation for running golden tests
as part of the standard test suite.
"""

import pytest
from pathlib import Path
import sys
import os

# Add the validator module to path
sys.path.insert(0, os.path.dirname(__file__))
from validate_golden_tests import GoldenTestValidator, TestResult


# Find the golden tests directory
GOLDEN_TESTS_DIR = Path(__file__).parent.parent.parent / "specs"


def get_all_golden_tests():
    """Collect all golden test XML files."""
    if not GOLDEN_TESTS_DIR.exists():
        return []
    
    tests = []
    for xml_file in GOLDEN_TESTS_DIR.rglob("*.xml"):
        # Create test ID from relative path
        relative_path = xml_file.relative_to(GOLDEN_TESTS_DIR)
        test_id = str(relative_path).replace(os.sep, "::").replace(".xml", "")
        tests.append((test_id, str(xml_file)))
    
    return tests


class TestGolden:
    """Test class for golden tests."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return GoldenTestValidator(verbose=True)
    
    @pytest.mark.parametrize("test_id,xml_path", get_all_golden_tests())
    def test_golden_test(self, test_id, xml_path, validator):
        """Run a single golden test."""
        # Parse test to check if Python implementation exists
        test_data = validator.parse_golden_test(xml_path)
        
        if test_data['python_code'] is None:
            pytest.skip(f"No Python implementation for {test_id}")
        
        # Run the test
        passed, results = validator.validate_test(xml_path)
        
        # Format detailed message for failures
        if not passed:
            failure_messages = []
            for result in results:
                if not result.passed:
                    msg = f"\n{result.check_name}:\n"
                    msg += f"  {result.message}\n"
                    msg += f"  Expected: {result.expected_value}\n"
                    msg += f"  Actual: {result.actual_value}"
                    failure_messages.append(msg)
            
            pytest.fail(
                f"Golden test '{test_id}' failed:\n" + 
                "\n".join(failure_messages)
            )
    
    def test_validator_parsing(self, validator):
        """Test that the validator can parse XML files correctly."""
        # Find any test file
        test_files = list(GOLDEN_TESTS_DIR.rglob("*.xml"))
        if not test_files:
            pytest.skip("No golden test files found")
        
        # Try to parse the first test file
        test_file = test_files[0]
        test_data = validator.parse_golden_test(str(test_file))
        
        # Check required fields
        assert 'metadata' in test_data
        assert 'semantic' in test_data
        assert 'inputs' in test_data
        assert 'checks' in test_data
        
        # Check metadata fields
        assert 'id' in test_data['metadata']
        assert 'version' in test_data['metadata']
        assert 'description' in test_data['metadata']
    
    def test_check_exact_value(self, validator):
        """Test exact value checking logic."""
        from validate_golden_tests import NumericCheck, CheckType
        
        check = NumericCheck(
            check_type=CheckType.EXACT_VALUE,
            name="test",
            location="test",
            expected=[1.0, 2.0, 3.0],
            tolerance=1e-10
        )
        
        # Test passing case
        passed, msg = validator._check_exact_value([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1e-10)
        assert passed
        
        # Test failing case
        passed, msg = validator._check_exact_value([1.0, 2.0, 3.1], [1.0, 2.0, 3.0], 1e-10)
        assert not passed
    
    def test_check_dimension(self, validator):
        """Test dimension checking logic."""
        import numpy as np
        
        # Test with numpy array
        arr = np.zeros((4, 4, 4))
        passed, msg = validator._check_dimension(arr, [4, 4, 4])
        assert passed
        
        # Test with list
        passed, msg = validator._check_dimension([4, 4, 4], [4, 4, 4])
        assert passed
        
        # Test failing case
        passed, msg = validator._check_dimension(arr, [3, 3, 3])
        assert not passed


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "golden: mark test as a golden test for cross-language validation"
    )


if __name__ == "__main__":
    # Allow running as script for debugging
    import sys
    pytest.main([__file__] + sys.argv[1:])