

# Update the s3 bucket of the docs website
deploy_docs:
	aws s3 sync docs/_build/html s3://docs.kxysolutions.com/ --acl public-read --metadata-directive REPLACE --cache-control max-age=86400 --profile kxy

# Invalidate certain cached files in the cloudfront distribution
refresh_web:
	aws cloudfront create-invalidation --distribution-id E1YRSKXSRFPX1L --paths $(PATHS) --profile kxy

# Cut a PyPi release
pypi_release:
	python setup.py sdist bdist_wheel 
	twine check dist/* 
	twine upload --skip-existing dist/*

install:
	pip install .

# Route any other make target to Sphinx
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)# You can set these variables from the command line, and also
