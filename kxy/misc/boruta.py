#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from time import time
import numpy as np
import pandas as pd
from scipy.stats import binom

from tqdm import tqdm

class Boruta(object):
	"""
	Implementation of the Boruta feature selection algorithm.

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


	def fit(self, x_df, y_df, n_evaluations=20, pval=0.95, max_duration=None):
		"""
		Performs a run of the Boruta feature selection algorithm. 

		Specifically, :code:`n_evaluations` times we randomly shuffle all feature values to define so-called 'shadow features', which we add to the original features.

		The random shuffling destroys any association between shadow features and the target. Thus, none of them should be considered important.

		We train the learner with both original and shadow features, and calculate the highest feature importance score among shadow features. 

		We consider an original feature a 'hit' if its feature importance score is higher than the highest feature importance score of shadow features, otherwise it is a miss.

		For each original feature, we end up with the number of hits in the random :code:`n_evaluations` runs, and we use the following statistical test to conclude if the original feature is relevant.

		The null hypothesis :math:`H_0` is that we do not know if an original feature is relevant for the problem at hand.

		Under :math:`H_0`, the probability that a feature will be a hit in one of the random trials above is :math:`p=0.5`, and its number of hits after :code:`n_evaluations` runs follows a binomial distribution with probability :math:`0.5` and total number of trials :math:`n_evaluations`.

		We reject :math:`H_0` and accept the alternative hypothesis :math:`H_1` that a feature is relevant when the number of hits observed is higher than the :code:`pval` quantile under :math:`H_0`.

		We reject :math:`H_0` and accept the alternative hypothesis :math:`H_2` that a feature is *not* relevant when the number of hits observed is lower than the :code:`1-pval` quantile under :math:`H_0`.

		A warning is raised when we were not able to reject :math:`H_0` for some features.

		Features that were determined to be relevant are used to train the final learner, which this method returns. 

		When no feature was determined to be relevant, this method returns :code:`None`.


		Parameters
		----------
		x_df : pd.DataFrame
			A dataframe containing all features.
		y_df : pd.DataFrame
			A dataframe containing the target.
		n_evaluations : int (default 20)
			Number of trials in the Boruta algorithm.
		pval : float (between 0 and 1, default 0.95)
			Quantile level above which features are considered relevant.
		max_duration : float | None (default)
			If not None, then feature elimination will stop after this many seconds.


		Attributes
		----------
		selected_variables : list
			The list of features deemed relevant, sorted in decreasing number of hits.
		ambiguous_variables : list
			The list of features that we could not conclude are either relevant or irrelevant.
		hits : dict
			Dictionary containing the number of hits for each feature.


		Returns
		-------
		m : sklearn-like model (an instance returned by :code:`learner_func`)
			An instance returned by :code:`learner_func` trained with features that were deemed relevant.

		"""
		columns = [_ for _ in x_df.columns]
		y = y_df.values

		hits = {col: 0 for col in columns}
		start_time = time()
		for trial in tqdm(range(n_evaluations)):
			# Construct shaddow features
			train_x_df = pd.DataFrame(x_df.sample(frac=1).values, \
				index=x_df.index, columns=['shaddow_%s' % col for col in columns])

			# Add to real features
			train_x_df = pd.concat([x_df, train_x_df], axis=1)
			all_columns = [_ for _ in train_x_df.columns]

			# Fit the model
			x = train_x_df[all_columns].values
			current_n_vars = len(all_columns)
			m = self.learner_func(n_vars=current_n_vars)
			m.fit(x, y)

			# Increase the hit count
			importances = [_ for _ in m.feature_importances_]
			assert len(importances) == len(all_columns), \
				'The number of importance scores should be %d, received %d' % (len(all_columns), len(importances))
			hit_threshold = np.max([np.abs(importances[i]) \
				for i in range(current_n_vars) if all_columns[i].startswith('shaddow_')])

			for i in range(current_n_vars):
				if not all_columns[i].startswith('shaddow_') and np.abs(importances[i]) > hit_threshold:
					hits[all_columns[i]] = hits[all_columns[i]] + 1

			duration = time()-start_time
			if max_duration and duration > max_duration:
				logging.warning('We have exceeded the configured maximum duration %.2fs: exiting after %d trials...' % (max_duration, trial+1))
				n_evaluations = trial+1
				break


		# Testing whether the number of hits is statistically significant to conclude the feature is important.
		# H0: We can't tell if a feature is important or not. Said differently, there is 50% chance that a feature will be a 'hit' in a round.
		# H1: A feature is important.
		# H2: A feature is unimportant.

		# Under H0, the number of hits after n_evaluations trials should be in [hm, hM] with probability pval, where hm and hM are binomial quantiles.
		# We reject H0 and accept H1 when a feature's number of hits exceeds hM.
		# We reject H0 and accept H2 when a feature's number of hits is lower than hm. 
		cdfs = binom.cdf([hits[col] for col in columns], n_evaluations, 0.5)
		selected_variables = [columns[i] for i in range(len(columns)) if cdfs[i] > pval]
		associated_hits = [hits[col] for col in selected_variables]

		self.selected_variables = [col for _,  col in sorted(zip(associated_hits, selected_variables), reverse=True)]
		self.ambiguous_variables = [columns[i] for i in range(len(columns)) if cdfs[i] <= pval and cdfs[i] >= 1.-pval]

		if self.ambiguous_variables:
			logging.warning('Features %s have not been determined important or unimportant. Consider increasing n_evaluations.' % \
				str(self.ambiguous_variables))

		self.hits = hits
		if self.selected_variables == []:
			logging.warning('No variable/feature was deemed important by the model.')
			return None

		# Keep only important features and retrain the model.
		x = x_df[self.selected_variables].values
		n_vars = len(self.selected_variables)
		m = self.learner_func(n_vars=n_vars, path=self.path)
		m.fit(x, y)

		return m


