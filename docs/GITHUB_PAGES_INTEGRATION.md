# GitHub Pages Integration Guide

This guide explains how the neuroimpy documentation and notebooks are integrated with GitHub Pages.

## Overview

The documentation system includes:
- Sphinx-generated API documentation
- Tutorial notebooks (Jupyter)
- Markdown documentation files
- Automated deployment to GitHub Pages

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation index
│   ├── tutorials/
│   │   ├── notebooks.rst    # Notebook section index
│   │   └── notebooks/       # Jupyter notebooks
│   │       ├── README.md
│   │       ├── image_volumes.ipynb
│   │       ├── neuro_vectors.ipynb
│   │       ├── regions_of_interest.ipynb
│   │       └── pipelines.ipynb
│   └── api/                 # API documentation
└── build/                   # Generated HTML (not in repo)
```

## How It Works

### 1. Documentation Build Process

When code is pushed to the main branch:

1. GitHub Actions triggers the `docs.yml` workflow
2. Sphinx builds the documentation including:
   - API docs from docstrings
   - RST tutorial files
   - Jupyter notebooks (via nbsphinx)
   - Markdown files (via myst-parser)
3. HTML output is generated in `docs/build/html/`

### 2. Notebook Integration

Notebooks are integrated through:
- **nbsphinx**: Renders notebooks as HTML pages
- **myst-parser**: Allows Markdown files in documentation
- Notebooks appear in the documentation navigation

### 3. GitHub Pages Deployment

The workflow automatically:
1. Builds documentation on push to main
2. Uploads HTML as artifacts
3. Deploys to GitHub Pages using `peaceiris/actions-gh-pages`
4. Documentation appears at: `https://[username].github.io/neuroimpy/`

## Configuration Details

### Sphinx Configuration (conf.py)

```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',           # For Jupyter notebooks
    'myst_parser',        # For Markdown files
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
```

### GitHub Actions Workflow (docs.yml)

Key features:
- Builds on push to main branch
- Installs documentation dependencies
- Runs Sphinx to generate HTML
- Deploys to gh-pages branch

## Viewing Documentation

### Local Preview

```bash
cd docs
make clean
make html
# Open docs/build/html/index.html in browser
```

### Online (GitHub Pages)

Once deployed, documentation is available at:
- Main site: `https://[username].github.io/neuroimpy/`
- Notebooks: `https://[username].github.io/neuroimpy/tutorials/notebooks.html`

## Notebook Features in Documentation

### Rendered Output
- Code cells show syntax-highlighted code
- Output cells display results
- Markdown cells are rendered as documentation

### Navigation
- Notebooks appear in the sidebar navigation
- Each notebook is a separate page
- Cross-references work between notebooks and API docs

### Search
- Notebook content is indexed for search
- Code and text are searchable

## Maintenance

### Adding New Notebooks

1. Create notebook in `docs/source/tutorials/notebooks/`
2. Update `notebooks.rst` to include it
3. Test locally with `make html`
4. Commit and push - GitHub Actions handles the rest

### Updating Documentation

1. Edit RST/MD files or notebooks
2. Test locally
3. Push to main branch
4. GitHub Actions rebuilds and deploys automatically

### Testing Notebooks

Before deployment:
```bash
cd docs/source/tutorials/notebooks
python validate_environment.py
python test_notebooks_simple.py
```

## Troubleshooting

### Build Failures

Check GitHub Actions logs for:
- Missing dependencies
- Notebook execution errors
- Sphinx warnings

### Notebook Issues

- Ensure notebooks have valid JSON structure
- All cells must have execution_count
- Use standard imports: `import neuroimpy as pn`

### Local Testing

Test the full build locally:
```bash
cd docs
pip install -r requirements-docs.txt  # If exists
make clean
make html
```

## Requirements

Create `docs/requirements-docs.txt`:
```
sphinx>=4.0
sphinx-rtd-theme
sphinx-autodoc-typehints
nbsphinx
myst-parser
jupyter
pandoc
```

## Benefits

1. **Single Source**: Notebooks serve as both tutorials and documentation
2. **Always Current**: Documentation updates automatically
3. **Interactive**: Users can download and run notebooks
4. **Searchable**: Full-text search includes notebook content
5. **Version Control**: Everything is tracked in Git

## Next Steps

1. Enable GitHub Pages in repository settings
2. Set source to gh-pages branch
3. Optionally configure custom domain
4. Monitor Actions tab for build status

The documentation will automatically update whenever changes are pushed to the main branch!