#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from time import time
import numpy as np
from tqdm import tqdm

class RFE(object):
	"""
	Implementation of the Recursive Feature Elimination (RFE) feature selection algorithm.

	Reference: 
	"""
	def __init__(self, learner_func, path=None):
		"""
		Constructor.

		Parameters
		----------
		learner_func : func | callable
			Function or callable that expects one optional argument :code:`n_vars` and returns an instance of a superviser learner (regressor or classifier) following the scikit-learn convention, and expecting :code:`n_vars` features.
			
			Specifically, the learner should have a :code:`fit(x_train, y_train)` method. The learner should also have a :code:`feature_importances_` property or attribute, which is an array or a list containing feature importances once the model has been trained.

			There should be as many importance scores in :code:`feature_importances_` as columns in :code:`x_train`.

		"""
		self.selected_variables = []
		self.learner_func = learner_func
		self.path = path


	def fit(self, x_df, y_df, n_vars, max_duration=None):
		"""
		Performs a run of the Recursive Feature Elimination (RFE) feature selection algorithm.

		Starting with all features, we recursively train a learner, calculate all feature importance scores, remove the least important feature, and repeat until we are left with :code:`n_vars` features.

		Parameters
		----------
		x_df : pd.DataFrame
			A dataframe containing all features.
		y_df : pd.DataFrame
			A dataframe containing the target.
		n_vars : int
			The number of features to keep.
		max_duration : float | None (default)
			If not None, then feature elimination will stop after this many seconds.

		Attributes
		----------
		selected_variables : list
			The list of the :code:`n_vars` features we kept.


		Returns
		-------
		m : sklearn-like model (an instance returned by :code:`learner_func`)
			An instance returned by :code:`learner_func` trained with the :code:`n_vars` features we kept.

		"""
		columns = [_ for _ in x_df.columns]
		y = y_df.values

		# Fit the model
		x = x_df[columns].values
		current_n_vars = len(columns)
		start_time = time()
		m = self.learner_func(n_vars=current_n_vars)
		m.fit(x, y)
		importances = [_ for _ in m.feature_importances_]

		n_rounds = max(current_n_vars-n_vars, 0)
		for _ in tqdm(range(n_rounds)):
			duration = time()-start_time
			if max_duration and duration > max_duration:
				logging.warning('We have exceeded the configured maximum duration %.2fs: exiting...' % max_duration)
				break

			# Remove the least important variable
			importances = [_ for _ in m.feature_importances_]
			least_important_ix = np.argmin(np.abs(importances))
			importances.pop(least_important_ix)
			least_important_feature = columns[least_important_ix]
			logging.info('Deleting feature %s' % least_important_feature)
			columns.remove(least_important_feature)
			current_n_vars = len(columns)

			# Re-fit the model
			x = x_df[columns].values
			m = self.learner_func(n_vars=current_n_vars, path=self.path)
			m.fit(x, y)

		self.selected_variables = [col for _,  col in sorted(zip(importances, columns), reverse=True)]

		return m


