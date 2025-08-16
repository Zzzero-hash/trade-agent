# Sphinx Makefile
# Copyright (C) 2001-2002 Python Software Foundation
# Distributed under the terms of the Python Software Foundation License.

# Variables
SPHINX := sphinx-build
BUILDDIR := _build
ALLSPHINXOPTS := -W
SPHINXOPTS :=

# Target: html
html:
	$(SPHINX) -q -n $(ALLSPHINXOPTS) \
		$(BUILDDIR)/html \
		.

# Target: clean
clean:
	-rm -rf $(BUILDDIR)/html

# Target: help
help:
	@echo "Usage:"
	@echo "  make html  - Build HTML documentation"
	@echo "  make clean - Clean built documentation"
	@echo "  make help  - Show this help message"

.PHONY: html clean help
