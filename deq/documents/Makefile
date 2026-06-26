# Makefile for Sphinx documentation
# for the deq package
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean genapi htmlfast html tutorial tutorial-highlight-only tutorial-clean

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _auto
	rm -rf api/generated
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# generate api docs
genapi:
	sphinx-apidoc  -o  _auto  -d 10 ../deq
	mkdir -p api/generated/
	cp _auto/* api/generated/

# faster build by skipping execution of all notebooks
htmlfast: genapi
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D nbsphinx_execute=never

html: genapi
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

tutorial: tutorial/scripts/node_modules
	python tutorial/scripts/run_generators.py
	python tutorial/scripts/highlight_deq.py

tutorial-highlight-only: tutorial/scripts/node_modules
	python tutorial/scripts/highlight_deq.py

tutorial-clean:
	python tutorial/scripts/clean_examples.py

# Install JS dependencies (shiki) required by highlight_deq.py.
# Re-runs whenever package.json or package-lock.json changes.
tutorial/scripts/node_modules: tutorial/scripts/package.json tutorial/scripts/package-lock.json
	cd tutorial/scripts && npm install
	@touch $@
