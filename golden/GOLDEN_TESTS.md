# Golden Tests: AI Reference

## Purpose
Golden tests ensure semantic equivalence across language implementations (R, Python, Rust) of the same software by validating numeric outputs rather than implementation details.

## Core Concepts

### What Golden Tests Are
- Language-agnostic test specifications in XML format
- Focus on WHAT code should do (semantic behavior), not HOW
- Validate using numeric outputs with tolerances
- Each language implements same semantics differently

### Key Principles
1. **Numeric Focus**: All validations based on matrix dimensions, values, statistical properties
2. **Semantic Descriptions**: Purpose + mathematical algorithm in each test
3. **Progressive Enhancement**: Spec → R → Python → Rust implementations
4. **Language Agnosticism**: Describe behavior, not implementation

## XML Test Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<golden_test xmlns="http://golden-tests.org/schema">
  <metadata>
    <id>unique_test_id</id>
    <version>1.0</version>
    <description>Brief description</description>
    <tags><tag>category</tag></tags>
  </metadata>
  
  <semantic_description>
    <purpose>What functionality is tested</purpose>
    <algorithm>Step-by-step mathematical description</algorithm>
  </semantic_description>
  
  <inputs>
    <!-- Structured test data -->
  </inputs>
  
  <expected_outputs>
    <numeric_checks>
      <check>
        <type>exact_value|range|relative|statistical</type>
        <location>where to check</location>
        <expected>value</expected>
        <tolerance>acceptable deviation</tolerance>
      </check>
    </numeric_checks>
  </expected_outputs>
  
  <implementations>
    <R><![CDATA[# R code]]></R>
    <Python><![CDATA[# Python code]]></Python>
    <Rust><![CDATA[# Rust code]]></Rust>
  </implementations>
  
  <propagation_status>
    <implementation lang="R" status="complete" date="2024-01-15"/>
    <implementation lang="Python" status="pending"/>
  </propagation_status>
</golden_test>
```

## Directory Structure

```
golden_tests/
├── specs/
│   ├── core/
│   │   ├── event_model/
│   │   │   ├── basic_hrf.xml          # Start here
│   │   │   ├── multiple_conditions.xml
│   │   │   └── continuous_regressors.xml
│   │   ├── baseline_model/
│   │   └── hrf_bases/
│   ├── integration/
│   └── edge_cases/
├── schema/
│   └── golden_test.xsd
└── validators/
    ├── R/validate_specs.R
    ├── Python/validate_specs.py
    └── Rust/validate_specs.rs
```

## Validation Types

1. **Dimensional**: Matrix/array dimensions, shape consistency
2. **Value Checks**:
   - Exact: For integers/categorical
   - Approximate: Floating-point with tolerance
   - Range: Within bounds
   - Relative: Percentage-based
3. **Statistical**: Sum, mean, std dev, min/max, percentiles
4. **Structural**: Column/row names, ordering, sparsity

## Workflow for New Tests

1. **Create XML spec** in appropriate directory
2. **Write semantic description** (purpose + algorithm)
3. **Implement in R** and validate outputs
4. **Document propagation status**
5. **Other languages** see failing tests and implement
6. **Update XML** with their implementations

## Workflow for New Language Implementation

1. **Get test specs** (submodule/copy from R repo)
2. **Create validator** in your test suite:
   ```python
   class GoldenTestValidator:
       def parse_golden_test(xml_path)
       def perform_numeric_check(matrix, check)
       def validate_test(xml_path)
   ```
3. **Start with basic_hrf.xml** - simplest test
4. **Implement required functionality** based on semantic descriptions
5. **Add your code to XML** implementations section
6. **Submit PR** to share implementation

## Test Sharing Methods

1. **Git Submodule** (recommended):
   ```bash
   git submodule add https://github.com/user/fmridesign.git fmridesign-r
   ln -s fmridesign-r/golden_tests golden_tests
   ```

2. **Separate Repository**: Dedicated golden-tests repo

3. **Package Distribution**: Include in package data

## Best Practices

### Adding Tests
- One behavior per test
- Start simple, minimal inputs
- Document WHY test exists
- Set appropriate tolerances (tighter for deterministic, looser for iterative)

### Implementing in New Language
- Match semantics, not syntax
- Focus on numeric equivalence
- Document any deviations in `<implementation_notes>`
- Use idiomatic code for your language

### Handling Differences
```xml
<implementation_notes>
  <note lang="Python">
    Uses scipy.linalg, tolerance 1e-6 for eigenvalues
  </note>
</implementation_notes>
```

## Current Test Status

| Test ID | R | Python | Rust |
|---------|---|--------|------|
| event_model_basic_hrf | ✅ | ⏳ | ⏳ |
| event_model_multiple_conditions | ✅ | ⏳ | ⏳ |
| baseline_model_polynomial_drift | ✅ | ⏳ | ⏳ |

Legend: ✅ Complete, ⏳ Pending, 🚧 In Progress

## Key Files to Read for Context

1. **Test examples**: Look at `specs/core/event_model/basic_hrf.xml`
2. **Schema**: Review `schema/golden_test.xsd` for XML structure
3. **Validators**: Check language-specific validators for implementation patterns

## Critical Points for AI Understanding

1. **Tests define behavior, not implementation** - focus on mathematical equivalence
2. **All validation is numeric** - no string comparisons, UI testing, or performance metrics
3. **Each language maintains own validator** - not in golden_tests directory
4. **Progressive workflow** - R implements first, others follow semantic spec
5. **Tolerances matter** - document why specific values chosen
6. **Cross-language collaboration** - implementations shared via XML updates

## Common Pitfalls to Avoid

- Don't modify existing specs when adding new language
- Don't test implementation details (data structures, variable names)
- Don't assume specific libraries available
- Don't use language-specific features in semantic descriptions
- Don't forget to update propagation_status

## Lessons Learned and Best Practices

### 1. Write Idiomatic Code, Not Syntax Emulation
- **Principle**: Each language should use its natural patterns to achieve semantic equivalence
- **Why it matters**: Trying to emulate R syntax in Python (or vice versa) leads to unnatural, hard-to-maintain code
- **Best practice**:
  - Focus on producing the same numeric results, not mimicking syntax
  - Use language-appropriate data structures and patterns
  - Example: Use numpy arrays naturally in Python rather than trying to make them behave like R matrices

### 2. Handle Cross-Language Indexing Differences
- **Challenge**: R uses 1-based indexing, Python/Rust use 0-based
- **Solution patterns**:
  - Create wrapper classes for test compatibility when needed
  - Convert indices at the boundary between test framework and implementation
  - Document index conversions clearly
- **Example**: For R's `vol[1,1,1]` use Python's `vol[0,0,0]` internally, but provide a wrapper for tests

### 3. Make Attributes Callable for Test Framework
- **Challenge**: Test expressions like `dim(obj)` expect functions, but Python often uses attributes
- **Solution**: Create helper functions that wrap attribute access
- **Example**:
  ```python
  def dim(obj):
      return obj.dim if hasattr(obj, 'dim') else obj.shape
  ```

### 4. Handle Complex Test Expressions
- **Challenge**: R-style expressions like `sum(vec4d[,,,1])` don't translate directly
- **Solutions**:
  1. Compute values directly and store in variables the test can find
  2. Extend validators to handle language-specific patterns
  3. Use wrapper classes that understand special slicing syntax
- **Best practice**: Start simple - compute the expected values directly rather than over-engineering

### 5. Memory Layout Matters
- **Principle**: R uses column-major (Fortran) order, Python defaults to row-major (C) order
- **Why it matters**: Array reshaping and flattening produce different results
- **Best practice**: Use `order='F'` in numpy operations when matching R behavior
- **Example**: `np.arange(1, 28).reshape((3, 3, 3), order='F')`

### 6. Validator Limitations and Workarounds
- **Common validator issues**:
  - R-style array slicing syntax (e.g., `[,1]` or `[,,,1]`)
  - Function calls vs attribute access
  - Complex expressions in test locations
- **Workarounds**:
  - Store intermediate results in named variables
  - Extend validators to handle common patterns
  - Use wrapper classes for complex indexing

### 7. Test Development Workflow (Updated)
1. **Understand the R implementation**: Read and run the R code first
2. **Identify semantic goals**: What numeric results must match?
3. **Write idiomatic implementation**: Use natural patterns for your language
4. **Handle impedance mismatches**: Add wrappers/helpers for test compatibility
5. **Iterate on failures**: Each failure teaches something about differences
6. **Document solutions**: Comment why certain patterns were needed

### 8. Common Cross-Language Pitfalls
- **Don't assume function names match**: `concat` in R might be `concatenate` or a method in Python
- **Check import availability**: Not all functions may be exported (`from package import *` might miss some)
- **Verify method vs function**: R's `as.logical(roi)` might be Python's `roi.as_logical()`
- **Test sparse representations**: Different languages handle sparse data differently

### 9. Debugging Cross-Language Tests
- **When tests fail mysteriously**:
  1. Check if the function/method exists and is imported
  2. Verify the exact shape and type of data structures
  3. Print intermediate values to understand transformations
  4. Compare memory layouts (row vs column major)
  5. Check if wrapper classes are preserving necessary attributes

### 10. Architecture Patterns for Test Compatibility
- **Wrapper classes**: Bridge between language-specific implementations and test expectations
- **Helper functions**: Make attributes callable, handle special operations
- **Strategic computation**: Pre-compute complex expressions that validators struggle with
- **Clear separation**: Keep test compatibility code separate from core implementation

### Previous Lessons (Still Valid)

### 11. Always Execute Code Before Writing Tests
- **Principle**: Never assume API behavior - verify through execution
- **Why it matters**: Function signatures, return types, and data structures often differ from expectations
- **Best practice**: 
  - Run code interactively before writing test specifications
  - Verify actual outputs match your mental model
  - Document any surprising behaviors in test comments

### 12. Be Aware of Function Polymorphism
- **Principle**: Many functions have multiple signatures with different behaviors
- **Why it matters**: The same function name may process arguments differently based on type or count
- **Best practice**:
  - Test all relevant function signatures
  - Read documentation for overloaded methods
  - Example: A function might treat `func(vec)` vs `func(x, y, z)` completely differently

### 13. Handle Object Systems Appropriately
- **Principle**: Different languages use different object models (S3/S4/R6 in R, classes in Python, structs in Rust)
- **Why it matters**: Direct field access, type coercion, and method calls vary by system
- **Best practice**:
  - Use appropriate accessor methods rather than direct field access
  - Test type conversions explicitly
  - Don't assume automatic coercion will work

### 14. XML Encoding Requirements
- **Principle**: XML has reserved characters that must be escaped
- **Common escapes**:
  - `<` → `&lt;`
  - `>` → `&gt;`
  - `&` → `&amp;`
  - `"` → `&quot;`
  - `'` → `&apos;`
- **Best practice**: Always escape comparison operators and special characters in test expressions

### 15. Verify Data Structure Internals
- **Principle**: Don't assume field names or structure without verification
- **Why it matters**: Internal representations often differ from external documentation
- **Best practice**:
  - Inspect objects programmatically (e.g., `str()` in R, `dir()` in Python)
  - Check field names exactly as they appear
  - Verify nested structure assumptions

This consolidated reference provides everything needed to understand and work with golden tests efficiently.