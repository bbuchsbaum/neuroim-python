# Systematic Translation Strategy: neuroim2 (R) → neuroim (Python)

## 1. Translation Principles

### A. Syntax Mapping
- **R S4 classes** → **Python classes with `@dataclass` or standard classes**
- **R generic functions** → **Python methods or properties**
- **R `[` indexing** → **Python `__getitem__` with numpy-style indexing**
- **R vectors (1-indexed)** → **Python arrays (0-indexed)**
- **R lists** → **Python dicts or lists**
- **R factors** → **Python enums or categorical arrays**

### B. Naming Conventions
- **R snake_case** → **Python snake_case** (maintain consistency)
- **R dots in names** (e.g., `as.mask`) → **Python underscores** (e.g., `as_mask`)
- **R camelCase classes** → **Python CamelCase classes**

### C. Method Translation Pattern
```r
# R generic function
series <- function(x, i, j, k, ...) UseMethod("series")
series.NeuroVec <- function(x, i, j, k, ...) { ... }
```
→
```python
# Python method
class NeuroVec:
    def series(self, i, j, k, **kwargs):
        ...
```

## 2. Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
1. **NeuroSpace** and coordinate systems
2. **Axis classes** (NamedAxis, AxisSet hierarchy)
3. **Basic transformations** (coord_to_grid, etc.)
4. **MetaInfo** classes

### Phase 2: 3D Volumes (Weeks 3-4)
1. **NeuroVol** abstract base class
2. **DenseNeuroVol** implementation
3. **SparseNeuroVol** implementation
4. **LogicalNeuroVol** for masks
5. Basic I/O for NIfTI files

### Phase 3: 4D Vectors (Weeks 5-6)
1. **NeuroVec** abstract base class
2. **DenseNeuroVec** implementation
3. **SparseNeuroVec** implementation
4. Memory-mapped variants

### Phase 4: ROI System (Week 7)
1. **ROI** base class
2. **ROICoords**, **ROIVol**, **ROIVec**
3. ROI window classes

### Phase 5: Advanced Operations (Weeks 8-9)
1. **Searchlight** iterators
2. **Spatial filters** (gaussian, bilateral, etc.)
3. **Connected components**
4. **Resampling** operations

### Phase 6: File I/O & Formats (Week 10)
1. Complete NIfTI support
2. AFNI format support
3. Binary readers/writers
4. Header utilities

## 3. Testing Strategy

### A. Direct Translation Tests
For each R function, create a parallel Python test:
```r
# R test
test_that("coord_to_grid works", {
  space <- NeuroSpace(c(10,10,10), c(1,1,1))
  expect_equal(coord_to_grid(space, c(5,5,5)), c(6,6,6))
})
```
→
```python
# Python test
def test_coord_to_grid():
    space = NeuroSpace((10,10,10), (1,1,1))
    assert np.array_equal(coord_to_grid(space, [5,5,5]), [5,5,5])  # Note: 0-indexed
```

### B. Compatibility Tests
- Load R-created files in Python
- Save Python files readable by R
- Compare numerical outputs

## 4. Documentation Strategy

### A. API Documentation
```python
def coord_to_grid(space: NeuroSpace, coords: np.ndarray) -> np.ndarray:
    """Convert real-world coordinates to grid indices.
    
    Direct translation of R's coord_to_grid function.
    
    Parameters
    ----------
    space : NeuroSpace
        The spatial reference system
    coords : array-like
        Real-world coordinates (x, y, z)
        
    Returns
    -------
    np.ndarray
        Grid indices (i, j, k) - Note: 0-indexed unlike R
        
    See Also
    --------
    grid_to_coord : Inverse transformation
    
    R Equivalent
    ------------
    neuroim2::coord_to_grid
    """
```

### B. Migration Guide
Create side-by-side examples showing R → Python translations

## 5. Quality Assurance

### A. Feature Parity Checklist
Track implementation status in `FEATURE_PARITY.md`

### B. Performance Benchmarks
Compare R vs Python performance for key operations

### C. Validation Suite
Cross-validate outputs between R and Python implementations

## 6. Python-Specific Enhancements

### A. Pythonic Interfaces
While maintaining close syntax, add Pythonic alternatives:
```python
# R-like syntax (for compatibility)
vol.series(i=5, j=10, k=15)

# Pythonic alternative
vol[5, 10, 15, :]  # Using __getitem__
```

### B. Context Managers
```python
# Python enhancement
with read_vec("data.nii", mode="mmap") as vec:
    # Automatic cleanup
```

### C. Iterator Protocol
```python
# Pythonic iteration
for roi in searchlight(vol, radius=3):
    process(roi)
```

## 7. Development Workflow

### A. Branch Strategy
- `main`: Stable releases
- `develop`: Integration branch
- `feature/phase-X`: Phase-specific development
- `compat/r-tests`: R compatibility testing

### B. CI/CD Pipeline
1. Run Python tests
2. Run R-Python compatibility tests
3. Check feature parity
4. Performance benchmarks

### C. Release Milestones
- v0.1: Core infrastructure (Phase 1)
- v0.2: Basic 3D operations (Phase 2)
- v0.3: 4D support (Phase 3)
- v0.4: ROI system (Phase 4)
- v0.5: Advanced operations (Phase 5)
- v1.0: Full feature parity

## 8. R-Python Bridge (Future)

Consider creating `rpy2` bindings for:
- Direct R object conversion
- Shared memory operations
- Seamless interoperability

## 9. Key Challenges & Solutions

### A. 1-indexing vs 0-indexing
- Clearly document differences
- Provide conversion utilities
- Add warnings for common mistakes

### B. S4 Method Dispatch
- Use Python's `functools.singledispatch` for similar behavior
- Or use class methods/properties

### C. Memory Management
- R's copy-on-write → Python's explicit copying
- Use views where possible
- Document memory behavior

### D. Missing Values
- R's NA → Python's np.nan or masked arrays
- Consistent handling across operations

## 10. Success Metrics

1. **Functional**: All R functions have Python equivalents
2. **Syntactic**: Code translation requires minimal changes
3. **Semantic**: Same inputs produce same outputs
4. **Performance**: Python ≥ R performance for most operations
5. **Usability**: Clear documentation and migration path