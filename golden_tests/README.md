# Golden Tests for neuroimpy

This directory contains golden tests for the neuroimpy package, designed to ensure semantic equivalence across different language implementations (R, Python, Rust).

## Overview

Golden tests focus on validating numeric outputs rather than implementation details. Each test specifies:
- Input data and parameters
- Expected numeric outputs with tolerances
- Semantic description of the algorithm
- Language-specific implementations

## Directory Structure

```
golden_tests/
├── specs/                      # Test specifications in XML
│   └── core/
│       ├── spatial_reference/  # Spatial coordinate system tests
│       ├── volume_operations/  # 3D volume operations
│       ├── vector_operations/  # 4D time series operations
│       ├── spatial_algorithms/ # Algorithms like connected components
│       └── io_operations/      # File I/O tests
├── schema/                     # XML schema definition
└── validators/                 # Language-specific validators
    └── Python/                 # Python test validator and pytest integration
```

## Running Golden Tests

### Using the Validator Directly

```python
from golden_tests.validators.Python.validate_golden_tests import GoldenTestValidator

# Validate a single test
validator = GoldenTestValidator(verbose=True)
passed, results = validator.validate_test("golden_tests/specs/core/spatial_reference/neurospace_construction.xml")

# Validate all tests
from golden_tests.validators.Python.validate_golden_tests import validate_all_golden_tests
results = validate_all_golden_tests("golden_tests/specs")
```

### Using pytest

```bash
# Run all golden tests
pytest golden_tests/validators/Python/test_golden.py

# Run with verbose output
pytest golden_tests/validators/Python/test_golden.py -v

# Run a specific test
pytest golden_tests/validators/Python/test_golden.py::TestGolden::test_golden_test[core::spatial_reference::neurospace_construction]
```

### Command Line Interface

```bash
# Validate a single test
python golden_tests/validators/Python/validate_golden_tests.py golden_tests/specs/core/spatial_reference/neurospace_construction.xml

# Validate all tests in a directory
python golden_tests/validators/Python/validate_golden_tests.py golden_tests/specs -v
```

## Current Test Status

| Test ID | R | Python | Description |
|---------|---|--------|-------------|
| neurospace_construction | ✅ | ⏳ | 3D space creation and coordinate transformations |
| neurovol_construction | ✅ | ⏳ | 3D volume creation from numeric arrays |
| neurovol_arithmetic | ✅ | ⏳ | Element-wise operations between volumes |
| neurovol_indexing | ✅ | ⏳ | Various indexing patterns |
| sparse_neurovol | ✅ | ⏳ | Sparse volume operations |
| neurovec_construction | ✅ | ⏳ | 4D data and time series extraction |
| connected_components | ✅ | ⏳ | Clustering algorithm on binary volumes |
| roi_operations | ✅ | ⏳ | Region of interest operations |
| basic_io_cycle | ✅ | ⏳ | Data persistence and metadata |
| concatenation | ✅ | ⏳ | Combining 3D volumes into 4D |

Legend: ✅ Complete, ⏳ Pending

## Adding Python Implementations

To add Python implementations to the golden tests:

1. Open the XML test file (e.g., `specs/core/spatial_reference/neurospace_construction.xml`)
2. Look for the `<implementations>` section
3. Add your Python code within `<Python><![CDATA[...]]></Python>` tags
4. Update the `<propagation_status>` to mark Python as complete
5. Run the test to validate your implementation

Example:
```xml
<implementations>
  <R><![CDATA[
    # R implementation
  ]]></R>
  <Python><![CDATA[
import numpy as np
from neuroimpy import NeuroSpace

# Create NeuroSpace
space = NeuroSpace(dim=(4, 4, 4), 
                   spacing=(2, 2, 2), 
                   origin=(-3, -3, -3))

# Test coordinate transformations
voxel_in = [1, 2, 3]
world_coords = space.grid_to_coord(voxel_in)

# ... rest of implementation
  ]]></Python>
</implementations>
```

## Test Development Guidelines

1. **Start Simple**: Begin with basic tests before complex scenarios
2. **Use Small Data**: Prefer 4x4x4 volumes for easier debugging
3. **Document Intent**: Include clear semantic descriptions
4. **Set Appropriate Tolerances**:
   - Exact values (integers): tolerance = 0
   - Deterministic floats: tolerance = 1e-10
   - Statistical operations: tolerance = 1e-6
   - Iterative algorithms: adjust as needed

## Debugging Failed Tests

1. Run with verbose mode to see detailed output
2. Check that all required variables are defined in the implementation
3. Verify array dimensions and data types match expectations
4. Use the validator's parse output to inspect test structure

## Integration with CI/CD

Add to your test suite:
```python
# In your main test configuration
pytest.main([
    "tests/",  # Your regular tests
    "golden_tests/validators/Python/test_golden.py"  # Golden tests
])
```

## Synchronizing with Source Repository

Use the `sync_golden_tests.py` script to keep tests synchronized with the R implementation:

```bash
# Check for updates (dry run)
python golden_tests/sync_golden_tests.py \
    --source ~/code/neuroim2/golden_tests \
    --target ./golden_tests \
    --language Python \
    --dry-run

# Apply updates
python golden_tests/sync_golden_tests.py \
    --source ~/code/neuroim2/golden_tests \
    --target ./golden_tests \
    --language Python
```

See [SYNC_GUIDE.md](SYNC_GUIDE.md) for detailed usage instructions.

## Contributing

When implementing golden tests:
1. Ensure your Python code is self-contained (imports included)
2. Match the semantic description exactly
3. Use the same variable names as specified in the test
4. Test locally before submitting
5. Update the propagation status when complete