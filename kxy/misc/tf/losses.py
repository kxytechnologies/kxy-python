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


class MINDLoss(Loss):  
	'''
	Loss function.
	'''
	def call(self, y_true, y_pred):
		''' '''
		p_samples = y_pred[:, 0]
		q_samples = y_pred[:, 1]
		mi = -tf.reduce_mean(p_samples) + math_ops.log(tf.reduce_mean(math_ops.exp(q_samples)))
		return mi
