#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from time import time
import numpy as np

class NaiveLearner(object):
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


	def fit(self, x_df, y_df):
		"""
		Fit the model without feature selection.

		Parameters
		----------
		x_df : pd.DataFrame
			A dataframe containing all features.
		y_df : pd.DataFrame
			A dataframe containing the target.

		Attributes
		----------
		selected_variables : list
			The list of features.

		Returns
		-------
		m : sklearn-like model (an instance returned by :code:`learner_func`)
			An instance returned by :code:`learner_func` trained with all features.

		"""
		columns = [_ for _ in x_df.columns]
		y = y_df.values
		x = x_df[columns].values
		n_vars = len(columns)
		m = self.learner_func(n_vars=n_vars, path=self.path)
		m.fit(x, y)
		self.selected_variables = columns

		return m


