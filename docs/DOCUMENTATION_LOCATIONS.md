# Documentation Locations

## Where Documents Appear

### 1. GitHub Pages (Public Documentation)
Once GitHub Pages is enabled and the workflow runs:
- **Main URL**: `https://[username].github.io/neuroimpy/`
- **Notebooks**: `https://[username].github.io/neuroimpy/tutorials/notebooks.html`
- **Individual notebooks**:
  - `https://[username].github.io/neuroimpy/tutorials/notebooks/image_volumes.html`
  - `https://[username].github.io/neuroimpy/tutorials/notebooks/neuro_vectors.html`
  - `https://[username].github.io/neuroimpy/tutorials/notebooks/regions_of_interest.html`
  - `https://[username].github.io/neuroimpy/tutorials/notebooks/pipelines.html`

### 2. Local Build
When you build documentation locally:
```bash
cd docs
make html
```
- **Location**: `docs/build/html/index.html`
- **Notebooks**: `docs/build/html/tutorials/notebooks.html`

### 3. Repository Structure
Source files are located at:
- **Notebooks**: `docs/source/tutorials/notebooks/*.ipynb`
- **Documentation**: `docs/source/tutorials/notebooks/*.md`
- **Configuration**: `docs/source/conf.py`

## GitHub Pages Setup

### Enable GitHub Pages:
1. Go to Settings → Pages in your GitHub repository
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`
4. Save

### The workflow will:
1. Build documentation on every push to `main`
2. Deploy to the `gh-pages` branch
3. Serve at `https://[username].github.io/neuroimpy/`

## Integration Features

### Automatic Features:
- ✅ Notebooks rendered as HTML with syntax highlighting
- ✅ Executable code cells shown with outputs
- ✅ Full-text search across all documentation
- ✅ Navigation sidebar includes notebooks
- ✅ Markdown files (README.md) are included
- ✅ Cross-references between notebooks and API docs

### Documentation Updates:
- Push to main branch → Automatic rebuild
- New notebooks → Add to `notebooks.rst`
- Changes visible in ~2-5 minutes

## Testing Before Deployment

1. **Test notebooks work**:
   ```bash
   cd docs/source/tutorials/notebooks
   python validate_environment.py
   python test_notebooks_simple.py
   ```

2. **Test documentation builds**:
   ```bash
   cd docs
   make clean
   make html
   # Check for warnings/errors
   ```

3. **Preview locally**:
   ```bash
   python -m http.server 8000 --directory docs/build/html
   # Open http://localhost:8000
   ```

## Current Status

✅ Notebooks are ready and tested
✅ Sphinx configuration updated for notebooks
✅ GitHub Actions workflow configured
✅ All dependencies specified
✅ Integration guide created

🔄 Pending: Enable GitHub Pages in repository settings