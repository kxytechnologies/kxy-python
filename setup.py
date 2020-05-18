# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:54:00 2020

@author: ylkomsamo
"""

import sys
sys.path.append('.')
from setuptools import setup, find_packages
version = "0.0.16"

setup(name="kxy",
	version=version,
	zip_safe=False,
	license="AGPLv3",
    author="Dr. Yves-Laurent Kom Samo",
    author_email="github@kxy.ai",
	url="https://docs.kxysolutions.com",
	description = "Python API to the AutoML KXY platform", 
	long_description="The KXY platform is the first and only AutoML platform for model-free" \
		" pre-learning and post-learning. More information at https://docs.kxysolutions.com.",
    project_urls={
        "Documentation": "https://docs.kxysolutions.com",
        "Source Code": "https://github.com/kxytechnologies/kxy-python/"},
	download_url = "https://github.com/kxytechnologies/kxy-python/archive/v%s.tar.gz" % version,
	keywords = ["AutoML", "Pre-Learning", "Post-Learning", "Model-Free ML"],
	packages=find_packages(exclude=["tests"]),
	install_requires=["numpy>=1.13.1", "scipy>=1.4.1", "pandas>=0.23.0", "requests==2.22.0", \
        "statsmodels"],
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3 :: Only",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]
)
