# Minimal makefile for Sphinx documentation
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

.PHONY: help Makefile


deploy_docs:
	aws s3 sync _build/html s3://docs.kxysolutions.com/ --acl public-read --metadata-directive REPLACE --cache-control max-age=86400 --profile $(AWSCLIPROFILE)


refresh_web:
	aws cloudfront create-invalidation --distribution-id E1YRSKXSRFPX1L --paths $(PATHS) --profile $(AWSCLIPROFILE)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)




