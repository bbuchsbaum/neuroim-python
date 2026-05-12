.PHONY: docs docs-preview

QUARTO_PYTHON ?= $(shell command -v python)
export QUARTO_PYTHON

docs:
	cd docs && quartodoc build && quarto render

docs-preview:
	cd docs && quartodoc build && quarto preview --no-browser
