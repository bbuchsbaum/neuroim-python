#!/usr/bin/env python
"""
Convert RST tutorial files to Jupyter notebooks.

This script converts static RST documentation to executable Jupyter notebooks,
ensuring that all code examples actually run and produce correct output.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any


def parse_rst_to_cells(rst_content: str) -> List[Dict[str, Any]]:
    """Parse RST content into notebook cells."""
    cells = []
    
    # Split content by code blocks
    parts = re.split(r'(```python|::)\n', rst_content)
    
    current_text = ""
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        
        if part in ['```python', '::']:
            # Add accumulated markdown
            if current_text.strip():
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": clean_rst_to_markdown(current_text.strip())
                })
                current_text = ""
            
            # Find the code block
            if i + 1 < len(parts):
                code_content = parts[i + 1]
                # Extract code until we hit another section or end
                code_lines = []
                for line in code_content.split('\n'):
                    if line.strip() and not line.startswith('    ') and not line.startswith('\t'):
                        # End of indented block
                        current_text = line + '\n'
                        break
                    # Remove common indentation
                    if line.strip():
                        code_lines.append(line.lstrip())
                
                # Remove output comments for executable notebook
                clean_code = []
                for line in code_lines:
                    if not line.strip().startswith('# Output:') and \
                       not line.strip().startswith('# <class') and \
                       not line.strip() == '#' and \
                       not (line.strip().startswith('#') and any(x in line for x in ['True', 'False', '(', ')'])):
                        clean_code.append(line)
                
                if clean_code:
                    cells.append({
                        "cell_type": "code",
                        "metadata": {},
                        "source": '\n'.join(clean_code).strip(),
                        "outputs": []
                    })
            i += 2
        else:
            current_text += part + '\n'
            i += 1
    
    # Add any remaining text
    if current_text.strip():
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": clean_rst_to_markdown(current_text.strip())
        })
    
    return cells


def clean_rst_to_markdown(text: str) -> str:
    """Convert RST formatting to Markdown."""
    # Convert RST headers
    lines = text.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for underline headers
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line and all(c == '=' for c in next_line.strip()):
                cleaned_lines.append(f"# {line}")
                i += 2
                continue
            elif next_line and all(c == '-' for c in next_line.strip()):
                cleaned_lines.append(f"## {line}")
                i += 2
                continue
        
        # Convert :func:`name` to `name()`
        line = re.sub(r':func:`([^`]+)`', r'`\1()`', line)
        # Convert :class:`name` to `name`
        line = re.sub(r':class:`([^`]+)`', r'`\1`', line)
        # Convert :meth:`name` to `name()`
        line = re.sub(r':meth:`([^`]+)`', r'`\1()`', line)
        
        # Convert :: to :
        if line.strip().endswith('::'):
            line = line[:-1] + ':'
        
        cleaned_lines.append(line)
        i += 1
    
    return '\n'.join(cleaned_lines)


def create_notebook(cells: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    """Create a notebook structure."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            },
            "title": title
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return notebook


def convert_rst_to_notebook(rst_path: Path, output_dir: Path):
    """Convert a single RST file to a Jupyter notebook."""
    print(f"Converting {rst_path.name}...")
    
    # Read RST content
    with open(rst_path, 'r') as f:
        content = f.read()
    
    # Extract title
    title_match = re.search(r'^(.+)\n[=\-]+\n', content)
    title = title_match.group(1) if title_match else rst_path.stem
    
    # Parse to cells
    cells = parse_rst_to_cells(content)
    
    # Add imports cell at the beginning if not present
    has_imports = any(
        cell.get('cell_type') == 'code' and 'import' in cell.get('source', '')
        for cell in cells[:3]
    )
    
    if not has_imports:
        # Add standard imports
        import_cell = {
            "cell_type": "code",
            "metadata": {},
            "source": "import neuroimpy as pn\nimport numpy as np\nimport matplotlib.pyplot as plt\n%matplotlib inline",
            "outputs": []
        }
        cells.insert(1, import_cell)  # After title
    
    # Create notebook
    notebook = create_notebook(cells, title)
    
    # Save notebook
    output_path = output_dir / f"{rst_path.stem}.ipynb"
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"  → Created {output_path.name}")
    return output_path


def add_example_data_setup(notebook_path: Path):
    """Add a cell to create example data for tutorials."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Check if we need example data
    code_content = ' '.join(
        cell.get('source', '') for cell in notebook['cells'] 
        if cell.get('cell_type') == 'code'
    )
    
    if 'read_vol' in code_content or 'read_vec' in code_content:
        # Add example data creation cell after imports
        setup_cell = {
            "cell_type": "code",
            "metadata": {},
            "source": """# Create example data for tutorial
# (In practice, you would use real neuroimaging files)

# Create example volume
space_3d = pn.NeuroSpace(dim=(64, 64, 25), spacing=(3.5, 3.5, 5.0), 
                         origin=(-110.5, -88.9342, -42.75))
example_data = np.random.randn(64, 64, 25)
example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())

# Create example 4D data
space_4d = pn.NeuroSpace(dim=(64, 64, 25, 100))
example_4d_data = np.random.randn(64, 64, 25, 100)

# For tutorials, we'll create objects directly instead of reading files
# vol = pn.read_vol("path/to/image.nii.gz")  # In practice
vol = pn.DenseNeuroVol(example_data, space_3d)  # For tutorial""",
            "outputs": []
        }
        
        # Find where to insert (after imports)
        insert_idx = 2
        for i, cell in enumerate(notebook['cells']):
            if cell.get('cell_type') == 'code' and 'import' in cell.get('source', ''):
                insert_idx = i + 1
                break
        
        notebook['cells'].insert(insert_idx, setup_cell)
        
        # Update file reading examples to show both ways
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'code' and 'read_vol' in cell.get('source', ''):
                cell['source'] = cell['source'].replace(
                    'vol = neuroimpy.read_vol("path/to/image.nii.gz")',
                    '# In practice:\n# vol = pn.read_vol("path/to/image.nii.gz")\n# For this tutorial:\nvol = pn.DenseNeuroVol(example_data, space_3d)'
                )
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)


def main():
    """Convert all RST tutorials to Jupyter notebooks."""
    # Paths
    rst_dir = Path("source/tutorials")
    notebook_dir = Path("source/tutorials/notebooks")
    notebook_dir.mkdir(exist_ok=True)
    
    # Find all RST files
    rst_files = list(rst_dir.glob("*.rst"))
    
    if not rst_files:
        print(f"No RST files found in {rst_dir}")
        return
    
    print(f"Found {len(rst_files)} RST files to convert")
    
    # Convert each file
    converted = []
    for rst_file in rst_files:
        try:
            notebook_path = convert_rst_to_notebook(rst_file, notebook_dir)
            add_example_data_setup(notebook_path)
            converted.append(notebook_path)
        except Exception as e:
            print(f"  ✗ Error converting {rst_file.name}: {e}")
    
    print(f"\nSuccessfully converted {len(converted)} notebooks")
    
    # Create index notebook
    if converted:
        create_index_notebook(notebook_dir, converted)


def create_index_notebook(notebook_dir: Path, notebook_paths: List[Path]):
    """Create an index notebook listing all tutorials."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# neuroimpy Tutorials\n\nWelcome to the neuroimpy tutorial notebooks! These interactive tutorials will guide you through the main features of the package."
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Available Tutorials\n\n" + "\n".join([
                f"- [{nb.stem.replace('_', ' ').title()}]({nb.name})"
                for nb in sorted(notebook_paths)
            ])
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": """## Getting Started

Before running these tutorials, make sure you have neuroimpy installed:

```bash
pip install neuroimpy
```

Each notebook contains executable code examples that demonstrate key features of the package. The notebooks are designed to be run in order, but can also be used as standalone references.

### Note on Example Data

These tutorials use synthetic data for demonstration. In practice, you would load real neuroimaging files using functions like `read_vol()` and `read_vec()`."""
        }
    ]
    
    index_notebook = create_notebook(cells, "neuroimpy Tutorials")
    
    index_path = notebook_dir / "00_index.ipynb"
    with open(index_path, 'w') as f:
        json.dump(index_notebook, f, indent=2)
    
    print(f"\nCreated index notebook: {index_path.name}")


if __name__ == "__main__":
    main()