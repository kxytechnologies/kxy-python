#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tensorflow learners.
"""
import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

from .generators import CopulaBatchGenerator, PFSBatchGenerator, set_generators_seed
from .initializers import set_initializers_seed
from .models import CopulaModel, PFSModel, PFSOneShotModel
from .losses import MINDLoss, ApproximateMINDLoss, RectifiedMINDLoss
from .config import get_default_parameter

def set_seed(seed):
	set_generators_seed(seed)
	set_initializers_seed(seed)


class CopulaLearner(object):
	'''
	Maximum-entropy learner.
	'''
	def __init__(self, d, beta_1=None, beta_2=None, epsilon=None, amsgrad=None, \
			name='Adam', lr=None, subsets=[]):
		self.d = d
		self.model = CopulaModel(self.d, subsets=subsets)
		beta_1 = get_default_parameter('beta_1') if beta_1 is None else beta_1
		beta_2 = get_default_parameter('beta_2') if beta_2 is None else beta_2
		lr = get_default_parameter('lr') if lr is None else lr
		amsgrad = get_default_parameter('amsgrad') if amsgrad is None else amsgrad
		epsilon = get_default_parameter('epsilon') if epsilon is None else epsilon
		logging.info('Using the Adam optimizer with learning parameters: ' \
			'lr: %.4f, beta_1: %.4f, beta_2: %.4f, epsilon: %.8f, amsgrad: %s' % \
			(lr, beta_1, beta_2, epsilon, amsgrad))
		self.opt = Adam(beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, \
			name=name, lr=lr)
		self.loss = MINDLoss()
		self.model.compile(optimizer=self.opt, loss=self.loss)
		self.copula_entropy = None


	def fit(self, z, batch_size=10000, steps_per_epoch=1000, epochs=None):
		''' '''
		z_gen = CopulaBatchGenerator(z, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
		epochs = get_default_parameter('epochs') if epochs is None else epochs
		self.model.fit(z_gen, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, \
			callbacks=[EarlyStopping(patience=3, monitor='loss'), TerminateOnNaN()])
		self.copula_entropy = self.model.evaluate(z_gen)




class PFSLearner(object):
	'''
	Principal Feature Learner.
	'''
	def __init__(self, dx, dy=1, dox=0, doy=0, beta_1=None, beta_2=None, epsilon=None, amsgrad=None, \
			lr=None, name='Adam', expand_y=True):
		self.expand_y = expand_y
		self.dx = dx
		self.dy = dy
		self.dox = dox
		self.doy = doy
		x_ixs = [_ for _ in range(dx)]
		y_ixs = [dx+_ for _ in range(dy)]
		ox_ixs = [dx+dy+_ for _ in range(dox)]
		oy_ixs = [dx+dy+dox+_ for _ in range(doy)]

		self.model = PFSModel(x_ixs, y_ixs, ox_ixs=ox_ixs, oy_ixs=oy_ixs, expand_y=expand_y)
		beta_1 = get_default_parameter('beta_1') if beta_1 is None else beta_1
		beta_2 = get_default_parameter('beta_2') if beta_2 is None else beta_2
		lr = get_default_parameter('lr') if lr is None else lr
		amsgrad = get_default_parameter('amsgrad') if amsgrad is None else amsgrad
		epsilon = get_default_parameter('epsilon') if epsilon is None else epsilon
		logging.info('Using the Adam optimizer with learning parameters: ' \
			'lr: %.4f, beta_1: %.4f, beta_2: %.4f, epsilon: %.8f, amsgrad: %s' % \
			(lr, beta_1, beta_2, epsilon, amsgrad))
		self.opt = Adam(beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, \
			name=name, lr=lr)
		self.loss = RectifiedMINDLoss() # MINDLoss()
		self.model.compile(optimizer=self.opt, loss=self.loss)
		self.mutual_information = None
		self.feature_direction = None
		self.statistics = None


	def fit(self, x, y, ox=None, oy=None, batch_size=None, n_shuffle=5, epochs=None, mi_eps=0.00001):
		''' '''
		n = x.shape[0]
		batch_size = get_default_parameter('batch_size') if batch_size is None else batch_size
		steps_per_epoch = n//batch_size
		steps_per_epoch = min(max(steps_per_epoch, 100), 1000)

		z_gen = PFSBatchGenerator(x, y, ox=ox, oy=oy, batch_size=batch_size, \
			steps_per_epoch=steps_per_epoch, n_shuffle=n_shuffle)
		epochs = get_default_parameter('epochs') if epochs is None else epochs
		self.model.fit(z_gen, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, \
			callbacks=[EarlyStopping(patience=3, monitor='loss'), TerminateOnNaN()])
		self.mutual_information = -self.model.evaluate(z_gen)
		w = self.model.w_layer.get_weights()[0]

		if self.mutual_information < mi_eps:
			# Retrain to avoid MI collapse to 0.
			batch_size = 2*batch_size
			z_gen = PFSBatchGenerator(x, y, ox=ox, oy=oy, batch_size=batch_size, \
				steps_per_epoch=steps_per_epoch, n_shuffle=n_shuffle)
			self.model.fit(z_gen, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, \
				callbacks=[EarlyStopping(patience=3, monitor='loss'), TerminateOnNaN()])
			self.mutual_information = -self.model.evaluate(z_gen)
			w = self.model.w_layer.get_weights()[0]

		# The feature direction should be normal
		w = w.flatten()
		w = w/np.sqrt(np.dot(w, w))

		# The principal feature should point in the same direction as the target (i.e. <y, w^Tx> = cov(y, w^Tx) > 0)
		corr_sign = np.sign(np.corrcoef(y, np.dot(x, w))[0, 1])
		w = corr_sign*w

		self.feature_direction = w
		self.fx = self.model.fx(tf.constant(z_gen.z)).numpy()
		if self.expand_y:
			self.gy = self.model.gy(tf.constant(z_gen.z)).numpy()


	def learned_constraints_x(self, x):
		'''
		'''
		n = x.shape[0]
		y = np.zeros((n, self.dy))
		ox = np.zeros((n, self.dox)) if self.dox>0 else None
		oy = np.zeros((n, self.doy)) if self.doy>0 else None
		z_gen = PFSBatchGenerator(x, y, ox=ox, oy=oy, n_shuffle=1)
		fx = self.model.fx(tf.constant(z_gen.z)).numpy()

		return fx


	def learned_constraints_y(self, y):
		'''
		'''
		n = x.shape[0]
		x = np.zeros((n, self.dx))
		ox = np.zeros((n, self.dox)) if self.dox>0 else None
		oy = np.zeros((n, self.doy)) if self.doy>0 else None
		z_gen = PFSBatchGenerator(x, y, ox=ox, oy=oy, n_shuffle=1)
		gy = self.model.gy(tf.constant(z_gen.z)).numpy()

		return gy




class PFSOneShotLearner(object):
	'''
	Principal Feature Learner learning multiple principal features simultaneously.
	'''
	def __init__(self, dx, dy=1, beta_1=None, beta_2=None, epsilon=None, amsgrad=None, \
			lr=None, name='Adam', p=1, expand_y=True):
		self.expand_y = expand_y
		x_ixs = [_ for _ in range(dx)]
		y_ixs = [dx+_ for _ in range(dy)]

		self.model = PFSOneShotModel(x_ixs, y_ixs, p=p, expand_y=expand_y)
		beta_1 = get_default_parameter('beta_1') if beta_1 is None else beta_1
		beta_2 = get_default_parameter('beta_2') if beta_2 is None else beta_2
		lr = get_default_parameter('lr') if lr is None else lr
		amsgrad = get_default_parameter('amsgrad') if amsgrad is None else amsgrad
		epsilon = get_default_parameter('epsilon') if epsilon is None else epsilon
		logging.info('Using the Adam optimizer with learning parameters: ' \
			'lr: %.4f, beta_1: %.4f, beta_2: %.4f, epsilon: %.8f, amsgrad: %s' % \
			(lr, beta_1, beta_2, epsilon, amsgrad))
		self.opt = Adam(beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, \
			name=name, lr=lr)
		self.loss = RectifiedMINDLoss() # MINDLoss()
		self.model.compile(optimizer=self.opt, loss=self.loss)
		self.mutual_information = None
		self.feature_direction = None
		self.statistics = None


	def fit(self, x, y, batch_size=None, n_shuffle=5, epochs=None, mi_eps=0.00001):
		''' '''
		n = x.shape[0]
		batch_size = get_default_parameter('batch_size') if batch_size is None else batch_size
		steps_per_epoch = n//batch_size
		steps_per_epoch = min(max(steps_per_epoch, 100), 1000)

		z_gen = PFSBatchGenerator(x, y, batch_size=batch_size, \
			steps_per_epoch=steps_per_epoch, n_shuffle=n_shuffle)
		epochs = get_default_parameter('epochs') if epochs is None else epochs
		self.model.fit(z_gen, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
		self.mutual_information = -self.model.evaluate(z_gen)
		w = self.model.w_layer.get_weights()[0]

		if self.mutual_information < mi_eps:
			# Retrain to avoid MI collapse to 0.
			batch_size = 2*batch_size
			z_gen = PFSBatchGenerator(x, y, batch_size=batch_size, \
				steps_per_epoch=steps_per_epoch, n_shuffle=n_shuffle)
			self.model.fit(z_gen, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
			self.mutual_information = -self.model.evaluate(z_gen)
			w = self.model.w_layer.get_weights()[0]

		# The feature direction should be normal
		# This should already have been taken care of downstream, but just in case.
		w = w/np.linalg.norm(w, axis=0)

		# Each principal feature should point in the same direction as the target (i.e. <y, w^Tx> = cov(y, w^Tx) > 0)
		for j in range(w.shape[1]):
			corr_sign = np.sign(np.corrcoef(y.flatten(), np.dot(x, w[:, j]))[0, 1])
			w[:, j] = corr_sign*w[:, j]

		w = w.T

		self.feature_directions = w





