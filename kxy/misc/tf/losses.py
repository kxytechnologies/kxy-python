#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Tensorflow losses.
"""
from multiprocessing import Pool, cpu_count
import numpy as np

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import Loss

from .ops import rectified_exp, d_rectified_exp


class MINDLoss(Loss):  
	'''
	MIND loss function: :math:`-E_P(T(x, y)^T\theta) + \log E_Q(e^{T(x, y)^T\theta})`.
	'''
	def call(self, y_true, y_pred):
		''' '''
		p_samples = y_pred[:, 0]
		q_samples = y_pred[:, 1]
		mi = -tf.reduce_mean(p_samples) + math_ops.log(tf.reduce_mean(math_ops.exp(q_samples)))
		return mi


class ApproximateMINDLoss(Loss):
	'''
	MIND loss function with a gentler version of the exponential: :math:`-E_P(r_exp(T(x, y)^T\theta)) + \log E_Q(dr_exp(T(x, y)^T\theta)`. :math:`r_exp(t) = exp(t)` if :math:`t<0` and :math:`r_exp(t) = 1+x+(1/2)x^2+(1/6)x^2`.
	'''
	def call(self, y_true, y_pred):
		''' '''
		p_samples = y_pred[:, 0]
		q_samples = y_pred[:, 1]
		mi = -tf.reduce_mean(p_samples) + math_ops.log(tf.reduce_mean(rectified_exp(q_samples)))
		return mi


class RectifiedMINDLoss(Loss):
	'''
	Rectified-MIND loss function: :math:`-E_P(\log dr_exp((T(x, y)^T\theta)) + \log E_Q(dr_exp(T(x, y)^T\theta)`. :math:`r_exp(t) = exp(t)` if :math:`t<0` and :math:`r_exp(t) = 1+x+(1/2)x^2+(1/6)x^2`.
	'''
	def call(self, y_true, y_pred):
		''' '''
		p_samples = y_pred[:, 0]
		q_samples = y_pred[:, 1]
		mi = -tf.reduce_mean(math_ops.log(d_rectified_exp(p_samples))) + math_ops.log(tf.reduce_mean(d_rectified_exp(q_samples)))
		return mi



