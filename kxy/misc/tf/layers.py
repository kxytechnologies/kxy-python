#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom tensorflow layers.
"""
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)
from tensorflow.keras.layers import Layer


class InitializableDense(Layer):
	''' 
	'''
	def __init__(self, units, initial_w=None, initial_b=None, bias=False):
		'''
		initial_w should be None or a 2D numpy array.
		initial_b should be None or a 1D numpy array.
		'''
		super(InitializableDense, self).__init__()
		self.units = units
		self.with_bias = bias
		self.w_initializer = 'zeros' if initial_w is None else tf.constant_initializer(initial_w)

		if self.with_bias:
			self.b_initializer = 'zeros' if initial_b is None else tf.constant_initializer(initial_b)


	def build(self, input_shape):
		''' '''
		self.w = self.add_weight(shape=(input_shape[-1], self.units), \
			initializer=self.w_initializer, trainable=True, name='quad_w')

		if self.with_bias:
			self.b = self.add_weight(shape=(self.units,), \
				initializer=self.b_initializer, trainable=True, name='quad_b')


	def call(self, inputs):
		''' '''
		return tf.matmul(inputs, self.w)+self.b if self.with_bias else tf.matmul(inputs, self.w)
