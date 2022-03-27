# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:54:00 2020

@author: ylkomsamo
"""

import sys
sys.path.append('.')
from setuptools import setup, find_packages

with open('README.md') as f:
	long_description = f.read()
	
version = "1.4.3"
setup(name="kxy",
	version=version,
	zip_safe=False,
	license="GPLv3",
	author="Dr. Yves-Laurent Kom Samo",
	author_email="github@kxy.ai",
	url="https://www.kxy.ai",
	description = "A Powerful Serverless Pre-Learning and Post-Learning Analysis Toolkit",
	long_description=long_description,
	long_description_content_type='text/markdown',  # This is important!
	project_urls={
		"Documentation": "https://www.kxy.ai/reference",
		"Source Code": "https://github.com/kxytechnologies/kxy-python/"},
	download_url = "https://github.com/kxytechnologies/kxy-python/archive/v%s.tar.gz" % version,
	keywords = ["Lean ML", "AutoML", "Pre-Learning", "Post-Learning", "Model-Free ML"],
	packages=find_packages(exclude=["tests"]),
	install_requires=["numpy>=1.13.1", "scipy>=1.4.1", "pandas>=0.23.0", "requests>=2.22.0", "pandarallel", "halo", "ipywidgets", "scikit-learn"],
	classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Programming Language :: Python :: 3 :: Only",
		"Development Status :: 5 - Production/Stable",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Mathematics"
	],
    scripts=['bin/kxy']
)
