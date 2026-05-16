.PHONY: docs docs-preview verify-evidence verify-install

QUARTO_PYTHON ?= $(shell command -v python)
export QUARTO_PYTHON

docs:
	cd docs && quartodoc build && quarto render

docs-preview:
	cd docs && quartodoc build && quarto preview --no-browser

verify-install:
	$(QUARTO_PYTHON) tools/verify_install.py

verify-evidence: verify-install
	PYTHONPATH=src:tests:. $(QUARTO_PYTHON) tools/verify_evidence.py
