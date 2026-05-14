.PHONY: docs docs-preview verify-evidence

QUARTO_PYTHON ?= $(shell command -v python)
export QUARTO_PYTHON

docs:
	cd docs && quartodoc build && quarto render

docs-preview:
	cd docs && quartodoc build && quarto preview --no-browser

verify-evidence:
	PYTHONPATH=src:tests:. $(QUARTO_PYTHON) tools/verify_evidence.py
