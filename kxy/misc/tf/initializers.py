#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Tensorflow initializers.
"""
import logging

from tensorflow.keras.initializers import GlorotUniform

LOCAL_SEED = None
INITIALIZER_COUNT = 0

def frozen_glorot_uniform():
	'''
	Deterministic GlorotUniform initializer.
	'''
	if LOCAL_SEED is not None:
		initializer =  GlorotUniform(LOCAL_SEED+INITIALIZER_COUNT)
		globals()['INITIALIZER_COUNT'] = INITIALIZER_COUNT + 1
		return initializer
	else:
		return GlorotUniform()

def set_initializers_seed(seed):
	globals()['LOCAL_SEED'] = seed