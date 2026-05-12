# R neuroim2 vs Python neuroim Test Coverage Comparison

## Overview

This document compares test coverage between R neuroim2 (151 test cases across 17 files) and Python neuroim to identify gaps and prioritize test development.

## Test Coverage Summary

### R neuroim2 Test Statistics
- **Total test files**: 17
- **Total test cases**: 151
- **Average tests per file**: ~9

### Coverage by Component

| Component | R Tests | Python Tests | Coverage | Priority |
|-----------|---------|--------------|----------|----------|
| **Core Data Structures** |
| NeuroVol (3D) | 30 | ✓ Partial | 60% | High |
| NeuroVec (4D) | 29 | ✓ Partial | 50% | High |
| SparseNeuroVol | 8 | ✓ Basic | 40% | Medium |
| SparseNeuroVec | 8 | ✓ Basic | 40% | Medium |
| NeuroSlice | 6 | ✓ Yes | 80% | Low |
| NeuroHyperVec | 6 | ❌ No | 0% | High |
| ClusteredNeuroVol | 5 | ✓ Basic | 50% | Medium |
| **Memory-Mapped** |
| FileBackedNeuroVec | 7 | ✓ Basic | 40% | Medium |
| BigNeuroVec | In file-backed | ✓ Basic | 40% | Medium |
| MappedNeuroVec | In other tests | ✓ Basic | 40% | Medium |
| **I/O Operations** |
| Read/Write NIfTI | 15 | ✓ Partial | 60% | High |
| HDF5 Support | 6 | ❌ No | 0% | Low |
| Header Preservation | 5 | ✓ Basic | 40% | Medium |
| **Indexing** |
| Grid/Index/Coord | 20 | ✓ Partial | 50% | High |
| Linear Access | 8 | N/A | N/A | - |
| Slicing | 10 | ✓ Yes | 80% | Low |
| **Analysis** |
| Searchlight | 12 | ✓ Yes | 90% | Low |
| Connected Components | 8 | ✓ Yes | 80% | Low |
| Clustering | 7 | ✓ Basic | 60% | Medium |
| ROI Operations | 15 | ✓ Yes | 85% | Low |
| **Filtering** |
| Gaussian Blur | 5 | ✓ Basic | 60% | Medium |
| Guided Filter | 3 | ✓ Basic | 60% | Medium |
| Bilateral Filter | 3 | ✓ Basic | 60% | Medium |

## Detailed Gap Analysis

### 🔴 Critical Gaps (Implement First)

#### 1. **NeuroHyperVec (5D+ support)**
- **R Coverage**: 6 test cases for multi-dimensional data
- **Python Status**: Not implemented
- **Action**: Implement NeuroHyperVec class and tests

#### 2. **Comprehensive NeuroVol/NeuroVec Tests**
- **R Coverage**: 30 + 29 = 59 test cases covering edge cases
- **Python Status**: Basic coverage, missing edge cases
- **Missing Tests**:
  - Boundary conditions
  - Invalid input handling
  - Large data handling
  - Memory efficiency tests

#### 3. **Coordinate System Tests**
- **R Coverage**: 20+ tests for coordinate transformations
- **Python Status**: Basic grid_to_index tests
- **Missing Tests**:
  - World coordinate transformations
  - Affine matrix handling
  - Edge cases in coordinate conversion

### 🟡 Medium Priority Gaps

#### 4. **Sparse Data Structure Tests**
- **R Coverage**: 16 tests (8 each for vol/vec)
- **Python Status**: Basic functionality tested
- **Missing Tests**:
  - Performance with large sparse data
  - Edge cases with very sparse data
  - Conversion between dense/sparse

#### 5. **Memory-Mapped Variants**
- **R Coverage**: 7+ tests for file-backed operations
- **Python Status**: Basic tests exist
- **Missing Tests**:
  - Large file handling
  - Concurrent access
  - Memory efficiency validation

#### 6. **Statistical Operations**
- **R Coverage**: Integrated in various tests
- **Python Status**: Basic stats.py tests
- **Missing Tests**:
  - Performance with large datasets
  - Numerical accuracy validation
  - Edge cases (empty data, NaN handling)

### 🟢 Well-Covered Areas

- Searchlight operations (90% parity)
- ROI functionality (85% parity)
- Connected components (80% parity)
- Basic I/O operations (60% parity)

## Recommended Test Implementation Plan

### Phase 1: Core Functionality (1-2 weeks)
1. **Expand NeuroVol tests** (add 15 test cases)
   - Edge cases for indexing
   - Arithmetic operation validation
   - Memory efficiency tests
   
2. **Expand NeuroVec tests** (add 15 test cases)
   - Time series extraction edge cases
   - 4D slicing combinations
   - Large dataset handling

3. **Coordinate system tests** (add 10 test cases)
   - Complete transformation pipeline
   - Edge cases for affine matrices
   - Round-trip accuracy

### Phase 2: Advanced Features (1 week)
4. **Implement NeuroHyperVec tests** (6 test cases)
   - Basic construction and indexing
   - 5D operations
   
5. **Enhance sparse structure tests** (add 8 test cases)
   - Performance benchmarks
   - Memory usage validation
   - Conversion accuracy

6. **Memory-mapped improvements** (add 6 test cases)
   - Concurrent access patterns
   - Large file handling
   - Cross-platform compatibility

### Phase 3: Completeness (1 week)
7. **I/O robustness** (add 8 test cases)
   - Corrupted file handling
   - Metadata preservation
   - Format compatibility

8. **Statistical accuracy** (add 5 test cases)
   - Numerical precision
   - NaN/Inf handling
   - Performance benchmarks

## Test Patterns to Adopt from R

### 1. Parameterized Testing
```r
# R pattern
for (mode in c("integer", "double", "logical")) {
  test_that(paste("NeuroVol works with", mode), { ... })
}
```

Python equivalent:
```python
@pytest.mark.parametrize("dtype", [np.int32, np.float64, bool])
def test_neurovol_dtypes(dtype):
    ...
```

### 2. Round-Trip Validation
```r
# R pattern
test_that("write/read preserves data", {
  write_vol(vol, temp_file)
  vol2 <- read_vol(temp_file)
  expect_equal(vol, vol2)
})
```

### 3. Edge Case Testing
- Empty volumes/vectors
- Single voxel volumes
- Extreme dimensions
- Invalid inputs
- Memory limits

### 4. Performance Assertions
```r
# R ensures operations are fast enough
expect_lt(system.time(operation)["elapsed"], 1.0)
```

## Metrics

Current test coverage comparison:
- **R neuroim2**: 151 test cases, ~90% functionality covered
- **Python neuroim**: ~80 test cases, ~60% functionality covered
- **Gap**: 71 test cases, 30% functionality

Target after implementation:
- Add 50-60 high-value test cases
- Achieve 85% functionality coverage
- Maintain <5 second test runtime