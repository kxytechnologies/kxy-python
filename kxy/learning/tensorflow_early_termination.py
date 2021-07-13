#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tensorflow v2 code to terminate training of a deep learning regressor or classifier when the running loss is much 
lower than a threshold, typically the theoretical-best.
"""
import logging
import numpy as np
from tensorflow.keras.callbacks import Callback


class TerminateIfOverfittedTF(Callback):
	''' 
	Tensorflow callback that terminates training at the end of a batch when the running loss is smaller than the theoretical best, which is strong indication that the model will end up overfitting.

	Parameters
	----------
	loss_key : str
		Which loss to base early-termination on. Example values are: :code:`'loss'`, :code:`'classification_error'`, and any other registered loss metrics.
	theoretical_best : float
		The theoretical-smallest loss achievable without overfiting, obtained using :code:`df.kxy.data_valuation`



	.. seealso::

		:ref:`kxy.pre_learning.achievable_performance.data_valuation <data-valuation>`.


	'''
	def __init__(self, theoretical_best, loss_key):
		super(EarlyTermination, self).__init__()
		self._supports_tf_logs = True
		self.theoretical_best = theoretical_best
		self.loss_key = loss_key

	def on_batch_end(self, batch, logs=None):
		''' '''
		logs = logs or {}
		if 'accuracy' in logs:
			logs['classification_error'] = 1.-logs['accuracy']

		loss = logs.get(self.loss_key, -np.inf)
		if loss < self.theoretical_best:
			logging.warning('Loss %s (%.4f) is much smaller than the theoretical best %.4f' % (self.loss_key, loss, self.theoretical_best))
			self.model.stop_training = True
