#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom math operations.
"""
from multiprocessing import Pool, cpu_count
import numpy as np

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)
from tensorflow.python.ops import math_ops

def rectified_exp(t):
	'''
	:math:`r_exp(t) = exp(t)` if :math:`t<0` and :math:`r_exp(t) = 1+x+(1/2)x^2+(1/6)x^3`.
	'''
	exp = math_ops.exp(t)
	approx_exp = 1.+t+(1./2.)*tf.math.pow(t, 2.)+(1./6.)*tf.math.pow(t, 3.)
	condition = tf.greater(t, 0.0)
	r_exp = tf.where(condition, x=approx_exp, y=exp)
	return r_exp


def d_rectified_exp(t):
	'''
	:math:`dr_exp(t) = exp(t)` if :math:`t<0` and :math:`dr_exp(t) = 1+x+(1/2)x^2`.
	'''
	dexp = math_ops.exp(t)
	approx_dexp = 1.+t+(1./2.)*tf.math.pow(t, 2.)
	condition = tf.greater(t, 0.0)
	dr_exp = tf.where(condition, x=approx_dexp, y=dexp)
	return dr_exp