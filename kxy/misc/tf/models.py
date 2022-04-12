#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Tensorflow models.
"""
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)

from tensorflow.keras import Model
from tensorflow.keras.backend import clip
from tensorflow.keras.layers import Dense, Lambda, concatenate, Dot
from tensorflow.keras.constraints import UnitNorm

from .initializers import frozen_glorot_uniform
from .layers import InitializableDense


class CopulaModel(Model):
	"""
	Maximum-entropy copula under (possibly sparse) Spearman rank correlation constraints.
	"""
	def __init__(self, d, subsets=[]):
		super(CopulaModel, self).__init__()
		self.d = d
		if subsets == []:
			subsets = [[_ for _ in range(d)]]

		self.subsets = subsets
		self.n_subsets = len(self.subsets)
		self.p_samples = Lambda(lambda x: x[:,:,0])
		self.q_samples = Lambda(lambda x: x[:,:,1])

		self.fx_non_mon_layer_1s = [Dense(3, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform()) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_2s = [Dense(5, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform()) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_3s = [Dense(3, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform()) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_4s = [Dense(1) for _ in range(self.n_subsets)]

		eff_ds = [len(subset)+1 for subset in self.subsets]
		self.spears = [InitializableDense(eff_d) for eff_d in eff_ds]
		self.dots = [Dot(1) for _ in range(self.n_subsets)]

		# Mixing layers
		self.mixing_layer1 = Dense(5, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.mixing_layer2 = Dense(5, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.mixing_layer3 = Dense(1, kernel_initializer=frozen_glorot_uniform())



	def subset_statistics(self, u, i):
		'''
		Statistics function for the i-th subset of variables.
		'''
		n = tf.shape(u)[0]
		res = tf.zeros(shape=[n, 1], dtype=tf.float64)
		ui = tf.gather(u, self.subsets[i], axis=1)

		# Constraints beyond quadratic
		fui = self.fx_non_mon_layer_1s[i](ui)
		fui = self.fx_non_mon_layer_2s[i](fui)
		fui = self.fx_non_mon_layer_3s[i](fui)
		fui = self.fx_non_mon_layer_4s[i](fui)
		ui = concatenate([ui, fui], axis=1)
	
		# Spearman terms
		spearman_term = self.spears[i](ui)
		spearman_term = self.dots[i]([spearman_term, ui])
		res = tf.add(res, spearman_term)
		return res


	def statistics(self, u):
		'''
		Statistics function.
		''' 
		if self.n_subsets > 1:
			ts = [self.subset_statistics(u, i) for i in range(self.n_subsets)]
			t = concatenate(ts, axis=1)
			t = self.mixing_layer1(t)
			t = self.mixing_layer2(t)
			t = self.mixing_layer3(t)
		else:
			t = self.subset_statistics(u, 0)
		return t


	def call(self, inputs):
		'''
		'''
		p_samples = self.p_samples(inputs)
		t_p = self.statistics(p_samples)

		q_samples = self.q_samples(inputs)
		t_q = self.statistics(q_samples)
		
		t = concatenate([t_p, t_q], axis=1)
		t = clip(t, -100., 100.)
		return t


	def copula(self, inputs):
		'''
		'''
		u = tf.constant(inputs)
		c = math_ops.exp(self.statistics(u))
		return c.numpy()/c.numpy().mean()




class PFSOneShotModel(Model):
	"""
	Model for Principal Feature Selection.
	"""
	def __init__(self, x_ixs, y_ixs, ox_ixs=[], oy_ixs=[], p=1):
		super(PFSOneShotModel, self).__init__()
		self.p_samples = Lambda(lambda x: x[:,:,0])
		self.q_samples = Lambda(lambda x: x[:,:,1])

		# Feature direction
		self.w_constraint = UnitNorm(axis=0)
		self.w_layer = Dense(p, use_bias=False, kernel_constraint=self.w_constraint, kernel_initializer=frozen_glorot_uniform())

		# Z network
		n_outer_z = 3
		n_inner_z = 5
		self.z_layer_1 = Dense(n_outer_z, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.z_layer_2 = Dense(n_inner_z, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.z_layer_3 = Dense(n_outer_z, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())

		# Y network
		n_outer_y = 3
		n_inner_y = 5
		self.y_layer_1 = Dense(n_outer_y, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.y_layer_2 = Dense(n_inner_y, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())
		self.y_layer_3 = Dense(n_outer_y, activation=tf.nn.silu, kernel_initializer=frozen_glorot_uniform())

		# Outer quadratic constraints
		n_q = 1+n_outer_y+len(oy_ixs)+len(y_ixs)+n_outer_z+len(ox_ixs)
		self.quad_load = InitializableDense(n_q)
		self.quad_dot = Dot(1)

		# Indices of x and y.
		self.x_ixs = x_ixs
		self.y_ixs = y_ixs
		self.ox_ixs = ox_ixs
		self.oy_ixs = oy_ixs


	def fx(self, u):
		''' '''
		x = tf.gather(u, self.x_ixs, axis=1)

		# z = w^Tx
		z = self.w_layer(x)

		# f(z)
		fx = self.z_layer_1(z)
		fx = self.z_layer_2(fx)
		fx = self.z_layer_3(fx)

		if self.ox_ixs:
			old_fx = tf.gather(u, self.ox_ixs, axis=1)
			fx = concatenate([fx, old_fx], axis=1)

		return fx


	def gy(self, u):
		''' '''
		y = tf.gather(u, self.y_ixs, axis=1)
		gy = self.y_layer_1(y)
		gy = self.y_layer_2(gy)
		gy = self.y_layer_3(gy)

		if self.oy_ixs:
			old_gy = tf.gather(u, self.oy_ixs, axis=1)
			gy = concatenate([gy, old_gy], axis=1)

		return gy


	def statistics(self, u):
		'''
		'''
		n = tf.shape(u)[0]
		y = tf.gather(u, self.y_ixs, axis=1)
		res = tf.zeros(shape=[n, 1], dtype=tf.float64)

		fx = self.fx(u)
		gy = self.gy(u)

		# q = [1, f(z), y, g(y)]H[1, f(z), y, g(y)]^T
		# Note: the intercept is critical here, do not remove it!
		fxgy = concatenate([tf.ones(shape=[n, 1], dtype=tf.float64), fx, y, gy], axis=1)
		q = self.quad_load(fxgy)
		q = self.quad_dot([q, fxgy])

		res = tf.add(res, q)

		return res


	def call(self, inputs):
		'''
		'''
		p_samples = self.p_samples(inputs)
		t_p = self.statistics(p_samples)

		q_samples = self.q_samples(inputs)
		t_q = self.statistics(q_samples)
		
		t = concatenate([t_p, t_q], axis=1)
		t = clip(t, -200., 400.)
		return t



class PFSModel(PFSOneShotModel):
	"""
	Model for Principal Feature Selection.
	"""
	def __init__(self, x_ixs, y_ixs, ox_ixs=[], oy_ixs=[]):
		super(PFSModel, self).__init__(x_ixs, y_ixs, ox_ixs=ox_ixs, oy_ixs=oy_ixs, p=1)


