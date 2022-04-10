#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Tensorflow generators.
"""
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)
from tensorflow.keras.utils import Sequence

LOCAL_SEED = None

def set_generators_seed(seed):
	globals()['LOCAL_SEED'] = seed


rankdata = lambda x: 1.+np.argsort(np.argsort(x, axis=0), axis=0)
class CopulaBatchGenerator(Sequence):
	''' 
	Random batch generator of maximum-entropy copula learning.
	'''
	def __init__(self, z, batch_size=1000, steps_per_epoch=100):
		self.batch_size = batch_size
		self.d = z.shape[1]
		self.n = z.shape[0]
		self.z = z
		self.steps_per_epoch = steps_per_epoch
		self.emp_u = rankdata(self.z)/(self.n + 1.)
		self.emp_u[np.isnan(self.z)] = 0.5
		self.rnd_gen = np.random.default_rng(LOCAL_SEED)

		if self.n < 200*self.d:
			dn = 200*self.d - self.n
			selected_rows = self.rnd_gen.choice(self.n, dn, replace=True)
			emp_u = self.emp_u[selected_rows, :].copy()
			scale = 1./(100.*self.n)
			emp_u += (scale*self.rnd_gen.uniform(size=emp_u.shape) - 0.5*scale)
			self.emp_u = np.concatenate([self.emp_u, emp_u], axis=0)
			self.n = self.emp_u.shape[0]

		self.batch_selector = self.rnd_gen.choice(self.n, self.batch_size*self.steps_per_epoch, replace=True)
		self.batch_selector = self.batch_selector.reshape((self.steps_per_epoch, self.batch_size))


	def getitem_ndarray(self, idx):
		''' '''
		i = idx % self.steps_per_epoch
		selected_rows = self.batch_selector[i]
		emp_u_ = self.emp_u[selected_rows, :]
		z_p = emp_u_.copy()
		z_q = self.rnd_gen.uniform(size=emp_u_.shape)

		z = np.empty((self.batch_size, self.d, 2))
		z[:, :, 0] = z_p
		z[:, :, 1] = z_q
		batch_x = z
		batch_y = np.ones((self.batch_size, 2))  # Not used  
		return batch_x, batch_y


	def __getitem__(self, idx):
		''' '''
		batch_x, batch_y = self.getitem_ndarray(idx)
		return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)


	def __len__(self):
		return self.steps_per_epoch



class PFSBatchGenerator(Sequence):
	'''
	Random batch generator.
	'''
	def __init__(self, x, y, ox=None, oy=None, batch_size=1000, steps_per_epoch=100, n_shuffle=5):
		self.rnd_gen = np.random.default_rng(LOCAL_SEED)
		assert x.shape[0] == y.shape[0]
		self.batch_size = batch_size
		self.n_shuffle = n_shuffle
		self.n = x.shape[0]

		x = x if len(x.shape) > 1 else x[:, None]
		y = y if len(y.shape) > 1 else y[:, None]
		ox = ox if ox is None or len(ox.shape) > 1 else ox[:, None]
		oy = oy if oy is None or len(oy.shape) > 1 else oy[:, None]

		self.x = x
		self.y = y
		self.ox = ox
		self.oy = oy
		self.z = np.concatenate([self.x, self.y, self.ox, self.oy], axis=1) if not self.ox is None else np.concatenate([self.x, self.y], axis=1)
		self.d = self.z.shape[1]

		self.steps_per_epoch = steps_per_epoch
		replace = False if self.n > self.batch_size*self.steps_per_epoch else True
		self.batch_selector = self.rnd_gen.choice(self.n, self.batch_size*self.steps_per_epoch, replace=replace)
		self.batch_selector = self.batch_selector.reshape((self.steps_per_epoch, self.batch_size))
		
		
	def getitem_ndarray(self, idx):
		''' '''
		i = idx % self.steps_per_epoch
		selected_rows = self.batch_selector[i]
		x_ = self.x[selected_rows, :]
		y_ = self.y[selected_rows, :]
		z_ = self.z[selected_rows, :]
		if not self.ox is None:
			ox_ = self.ox[selected_rows, :]
			oy_ = self.oy[selected_rows, :]

		z_p = None
		z_q = None
		for _ in range(self.n_shuffle):
			z_p = z_.copy() if z_p is None else np.concatenate([z_p, z_.copy()], axis=0)
			y_q = y_.copy()
			randomize = np.arange(y_q.shape[0])
			self.rnd_gen.shuffle(randomize)
			y_q = y_q[randomize]
			if not self.ox is None:
				oy_q = oy_.copy()
				oy_q = oy_q[randomize]
			z_q_ = np.concatenate([x_, y_q.copy(), ox_, oy_q], axis=1) if not self.ox is None else np.concatenate([x_, y_q.copy()], axis=1)
			z_q = z_q_.copy() if z_q is None else np.concatenate([z_q, z_q_.copy()], axis=0)

		z = np.empty((self.batch_size*self.n_shuffle, self.d, 2))
		z[:, :, 0] = z_p
		z[:, :, 1] = z_q
		batch_x = z
		batch_y = np.ones((self.batch_size*self.n_shuffle, 2))  # Not used   
		return batch_x, batch_y


	def __getitem__(self, idx):
		''' '''
		batch_x, batch_y = self.getitem_ndarray(idx)
		return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)
	
	def __len__(self):
		return self.steps_per_epoch




