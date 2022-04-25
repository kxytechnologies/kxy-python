VERSION = 1.4.10

# Update the s3 bucket of the docs website
deploy_docs:
	aws s3 sync docs/_build/html s3://www.kxy.ai/reference/ --acl public-read --metadata-directive REPLACE --cache-control max-age=86400 --profile kxy

# Invalidate certain cached files in the cloudfront distribution
refresh_web:
	aws cloudfront create-invalidation --distribution-id EJZS9SM07YXKX --paths $(PATHS) --profile kxy

# Cut a PyPi release
pypi_release:
	python setup.py sdist bdist_wheel 
	twine check dist/* 
	twine upload --skip-existing dist/*

install:
	pip install .


docker_release:
	docker build -t kxytechnologies/kxy:latest ./docker/kxy/
	docker login --username drylnks && docker push kxytechnologies/kxy:latest


docker_release_github:
	docker build -t ghcr.io/kxytechnologies/kxy-python:latest ./docker/kxy/
	# echo $(CR_PAT) | docker login ghcr.io -u USERNAME --password-stdin && docker push ghcr.io/kxytechnologies/kxy-python:latest
	docker push ghcr.io/kxytechnologies/kxy-python:latest
	docker build -t ghcr.io/kxytechnologies/kxy-python:$(VERSION) ./docker/kxy/
	# echo $(CR_PAT) | docker login ghcr.io -u USERNAME --password-stdin && docker push ghcr.io/kxytechnologies/kxy-python:$(VERSION)
	docker push ghcr.io/kxytechnologies/kxy-python:$(VERSION)


one_shot_release:
	make clean
	make html
	make deploy_docs
	make refresh_web PATHS=/reference/*
	make docker_release
	

update_docs:
	make clean
	make html
	make deploy_docs
	make refresh_web PATHS=/reference/*


github_release:
	gh release create v$(VERSION) -F CHANGELOG.md


package_release:
	make pypi_release
	make github_release
	timeout 5
	make docker_release_github
	make docker_release


osr:
	make one_shot_release


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
