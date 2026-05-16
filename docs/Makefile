# Documentation makefile. Quarto is the primary site; Sphinx targets remain
# available for the legacy docs/source tree during migration.

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
QUARTO_PYTHON ?= $(shell command -v python)
export QUARTO_PYTHON

# quartodoc config lives in _quartodoc.yml (single source of truth, merged
# into the project via `metadata-files:` at quarto render time). The
# quartodoc CLI does NOT honor `metadata-files:`, so it must be pointed at
# the file explicitly or it fails with "No `quartodoc:` section found".
QUARTODOC_CONFIG = _quartodoc.yml

.PHONY: help quarto quarto-preview html lint

lint:
	cd .. && python tools/docs_lint.py

quarto: lint
	quartodoc build --config $(QUARTODOC_CONFIG) && quarto render

quarto-preview:
	quartodoc build --config $(QUARTODOC_CONFIG) && quarto preview --no-browser

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
