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
      packages=find_packages(exclude=['tests']),
      install_requires=['numpy>=1.13.1',
                        'pandas>=0.19.2'])
