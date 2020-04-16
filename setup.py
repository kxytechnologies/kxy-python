# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:54:00 2020

@author: ylkomsamo
"""

import sys
sys.path.append('.')
from setuptools import setup, find_packages

setup(name='kxy',
      version='0.0',
      zip_safe=False,
      license='AGPLv3',
      download_url='https://github.com/kxytechnologies/kxy-python/archive/v0.0.1.tar.gz',
      keywords = ['AutoML', 'Pre-Learning', 'Post-Learning', 'Model-Free ML'],
      packages=find_packages(exclude=['tests']),
      install_requires=['numpy>=1.13.1',
                        'pandas>=0.19.2'])
