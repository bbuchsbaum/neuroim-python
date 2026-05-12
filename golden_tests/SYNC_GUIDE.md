# Golden Test Sync Guide

## Overview

The `sync_golden_tests.py` script synchronizes golden test specifications between a source repository (typically the R implementation) and a target repository (Python, Rust, etc.) while preserving language-specific implementations.

## Quick Start

```bash
# Basic sync from neuroim2 (R) to neuroim (Python)
python golden_tests/sync_golden_tests.py \
    --source ~/code/neuroim2/golden_tests \
    --target ./golden_tests \
    --language Python

# Dry run to preview changes
python golden_tests/sync_golden_tests.py \
    --source ~/code/neuroim2/golden_tests \
    --target ./golden_tests \
    --language Python \
    --dry-run

# Verbose output for details
python golden_tests/sync_golden_tests.py \
    --source ~/code/neuroim2/golden_tests \
    --target ./golden_tests \
    --language Python \
    --verbose
```

## How It Works

1. **Compares Test Specifications**: The script compares the XML test specifications between source and target, ignoring implementation sections
2. **Identifies Changes**:
   - New tests added in source
   - Tests modified in source
   - Tests deleted from source
3. **Preserves Implementations**: When syncing, it preserves existing language implementations in the target
4. **Updates Metadata**: Automatically updates propagation status when implementations exist

## Command Line Options

- `--source, -s`: Path to source repository golden tests directory (required)
- `--target, -t`: Path to target repository golden tests directory (required)
- `--language, -l`: Target language implementations to preserve, e.g., Python, Rust, Julia (required)
- `--dry-run, -n`: Preview changes without applying them
- `--verbose, -v`: Show detailed output

## Output Interpretation

```
Golden Test Sync Report
======================
Source: ~/code/neuroim2/golden_tests
Target: ./golden_tests
Language: Python

New tests found: 2
  - core/spatial_algorithms/watershed.xml
  - edge_cases/extreme_dimensions.xml

Modified tests: 1
  - core/volume_operations/neurovol_arithmetic.xml

Implementation Status:
  Total tests: 12
  [✓] Implemented: 10
  [✗] Need implementation: 2
```

## Workflow

### Regular Sync (Recommended)

1. Before starting work, sync to check for updates:
   ```bash
   python golden_tests/sync_golden_tests.py \
       --source ~/code/neuroim2/golden_tests \
       --target ./golden_tests \
       --language Python \
       --dry-run
   ```

2. If changes are found, review them and apply:
   ```bash
   python golden_tests/sync_golden_tests.py \
       --source ~/code/neuroim2/golden_tests \
       --target ./golden_tests \
       --language Python
   ```

3. Implement any new tests that were added

4. Run validation to ensure tests still pass:
   ```bash
   python golden_tests/validators/Python/validate_golden_tests.py golden_tests/specs
   ```

### After R Package Updates

When the R package is updated:

1. Pull latest changes in the R repository
2. Run sync to see what changed
3. Review modified tests carefully - spec changes might require implementation updates
4. Test thoroughly after syncing

## Safety Features

- **Dry Run Mode**: Always preview changes before applying
- **Implementation Preservation**: Never overwrites existing implementations
- **Clear Reporting**: Shows exactly what will change
- **XML Validation**: Ensures valid XML structure is maintained

## Troubleshooting

### "Source directory does not exist"
- Check the path to the source repository
- Ensure you have the R repository cloned locally

### "No changes detected" but you expect changes
- The script only compares test specifications, not implementations
- Check if the changes are only in implementation sections

### Modified tests after sync
- This is normal when the R implementation updates test specs
- Review changes to ensure your implementation still matches the spec

## Using with Other Languages

The script is language-agnostic. For Rust:

```bash
python sync_golden_tests.py \
    --source ~/code/neuroim2/golden_tests \
    --target ~/code/neuroim-rust/golden_tests \
    --language Rust
```

For Julia:

```bash
python sync_golden_tests.py \
    --source ~/code/neuroim2/golden_tests \
    --target ~/code/NeuroIM.jl/golden_tests \
    --language Julia
```

## Best Practices

1. **Run regularly**: Check for updates before starting new work
2. **Use dry-run first**: Always preview changes
3. **Review modifications**: Test spec changes might affect your implementation
4. **Commit after sync**: Keep sync changes in separate commits for clarity
5. **Document sync**: Note in commit messages when tests were synced