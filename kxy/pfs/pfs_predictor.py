#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pkl

from .pfs_selector import PFS, PCA


class PFSPredictor(object):
	def _predict(self, obj):
		assert hasattr(self, 'models'), 'The model should be first fitted'
		assert hasattr(self, 'feature_directions'), 'The model should first be fitted'
		assert hasattr(self, 'x_columns'), 'The model should first be fitted'
		assert self.feature_directions.shape[0] > 0, 'There should be at least one feature selected'

		z = np.dot(obj[self.x_columns].values, self.feature_directions.T)
		y = self.models[0].predict(z)
		predictions = pd.DataFrame(index=obj.index)
		predictions[self.target_column] = y

		return predictions


	def predict(self, obj, memory_bound=False):
		"""
		Make predictions using the fitted model.


		Parameters
		----------
		obj : pandas.DataFrame
			A dataframe containing test explanatory variables/features about which we want to make predictions.
		memory_bound : bool (Default False)
			Whether we should try to save memory.


		Returns
		-------
		result : pandas.DataFrame
			A dataframe with the same index as :code:`obj`, and with one column whose name is the :code:`target_column` used for training.
		"""
		if memory_bound:
			n = obj.shape[0]
			max_n = 1000000
			res = pd.DataFrame(index=obj.index)
			res[self.target_column] = np.nan
			i = 0
			while i < n:
				res.iloc[i:i+max_n] = self._predict(obj.iloc[i:i+max_n])
				i += max_n
			return res

		else:
			return self._predict(obj)


	def save(self, path):
		"""
		Cache the predictor to disk.
		"""
		meta_path = path + '-meta-' + self.__class__.__name__
		meta = {'target_column': self.target_column, 'feature_directions': self.feature_directions, 'x_columns': self.x_columns}
		with open(meta_path, 'wb') as f:
			pkl.dump(meta, f)
		self.models[0].save(path + '-' + self.__class__.__name__)
		

	@classmethod
	def load(cls, path, learner_func):
		"""
		Load the predictor from disk.
		"""
		meta_path = path + '-meta-' + cls.__name__
		with open(meta_path, 'rb') as f:
			meta = pkl.load(f)
		target_column = meta['target_column']
		feature_directions = meta['feature_directions']
		x_columns = meta['x_columns']

		n_vars = feature_directions.shape[0]
		model = learner_func(n_vars=n_vars, path=path + '-' + cls.__name__, safe=False)
		
		predictor = cls()
		predictor.models = [model]
		predictor.feature_directions = feature_directions
		predictor.target_column = target_column
		predictor.x_columns = x_columns

		return predictor


	def get_feature_selector(self):
		"""
		"""
		return PFS()

	@property
	def p(self):
		return self.feature_directions.shape[0]


	def fit(self, obj, target_column, learner_func, max_duration=None, path=None, p=None):
		"""
		Fits a supervised learner enriched with feature selection using the Principal Feature Selection (PFS) algorithm.


		Parameters
		----------
		obj : pandas.DataFrame
			A dataframe containing training explanatory variables/features as well as the target.
		target_column : str
			The name of the column in :code:`obj` containing targets.
		learner_func : func | callable
			Function or callable that expects one optional argument :code:`n_vars` and returns an instance of a superviser learner (regressor or classifier) following the scikit-learn convention, and expecting :code:`n_vars` features. Specifically, the learner should have a :code:`fit(x_train, y_train)` method. The learner should also have a :code:`feature_importances_` property or attribute, which is an array or a list containing feature importances once the model has been trained. There should be as many importance scores in :code:`feature_importances_` as columns in :code:`fit(x_train, y_train)`.
		max_duration : float | None (default)
			If not None, then feature elimination will stop after this many seconds.
		p : int | None (default)
			The number of principal features to learn when using one-shot PFS.


		Attributes
		----------
		feature_directions : np.array
			The matrix whose rows are the directions in which to project the original features to get principal features.
		target_column : str
			The name of the column used as target.
		models : list
			An array whose first entry is the fitted model.
		x_columns : list
			The list of columns used for PFS sorted alphabetically.


		Returns
		-------
		results : dict
			A dictionary containing, among other things, feature directions.

		"""
		self.target_column = target_column
		self.x_columns = sorted([_ for _ in obj.columns if _ != target_column])

		x = obj[self.x_columns].values
		y = obj[[target_column]].values

		# Construct principal features
		principal_feature_selector = self.get_feature_selector()
		self.feature_directions = principal_feature_selector.fit(x, y, max_duration=max_duration, p=p)
		z = np.dot(x, self.feature_directions.T) # Principal features

		# Train the learner
		n_vars = self.feature_directions.shape[0]
		m = learner_func(n_vars=n_vars)
		m.fit(z, y)
		self.models = [m]
		if path:
			self.save(path)

		results = {'Feature Directions': self.feature_directions}
		return results


class PCAPredictor(PFSPredictor):
	def __init__(self, energy_loss_frac=0.05):
		self.energy_loss_frac = energy_loss_frac

	def get_feature_selector(self):
		return PCA(energy_loss_frac=self.energy_loss_frac)

