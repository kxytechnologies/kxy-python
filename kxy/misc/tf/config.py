#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global default training configs
"""
# LEARNING PARAMETERS
LR = 0.005
EPOCHS = 20

# ADAM PARAMETERS
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-04
AMSGRAD = False
BATCH_SIZE = 500


def set_default_parameter(name, value):
	'''
	Utility function to change parameters above at runtime.
	'''
	import logging
	globals()[name.upper()] = value
	return

def get_default_parameter(name):
	return eval(name.upper())