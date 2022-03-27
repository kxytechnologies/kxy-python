#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pkl

from .boruta import Boruta
from .naive import NaiveLearner
from .rfe import RFE


class BasePredictor(object):
	def _predict(self, obj):
		assert hasattr(self, 'models'), 'The model should be first fitted'
		assert hasattr(self, 'selected_variables'), 'The model should be first fitted'
		assert len(self.selected_variables) > 0, 'There should be at least one feature selected'

		x = obj[self.selected_variables].values
		y = self.models[0].predict(x)
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
		meta = {'target_column': self.target_column, 'selected_variables': self.selected_variables}
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
		selected_variables = meta['selected_variables']

		n_vars = len(selected_variables)
		model = learner_func(n_vars=n_vars, path=path + '-' + cls.__name__, safe=False)
		
		predictor = cls()
		predictor.models = [model]
		predictor.selected_variables = selected_variables
		predictor.target_column = target_column

		return predictor




class RFEPredictor(BasePredictor):
	def fit(self, obj, target_column, learner_func, n_features, max_duration=None, path=None):
		"""
		Fits a supervised learner enriched with feature selection using the Recursive Feature Elimination algorithm.


		Parameters
		----------
		obj : pandas.DataFrame
			A dataframe containing training explanatory variables/features as well as the target.
		target_column : str
			The name of the column in :code:`obj` containing targets.
		learner_func : func | callable
			Function or callable that expects one optional argument :code:`n_vars` and returns an instance of a superviser learner (regressor or classifier) following the scikit-learn convention, and expecting :code:`n_vars` features. Specifically, the learner should have a :code:`fit(x_train, y_train)` method. The learner should also have a :code:`feature_importances_` property or attribute, which is an array or a list containing feature importances once the model has been trained. There should be as many importance scores in :code:`feature_importances_` as columns in :code:`fit(x_train, y_train)`.
		n_features : int
			The number of features to keep.
		max_duration : float | None (default)
			If not None, then feature elimination will stop after this many seconds.


		Attributes
		----------
		selected_variables : list
			The list of the :code:`n_features` features we kept.
		target_column : str
			The name of the column used as target.
		models : list
			An array whose first entry is the fitted model.


		Returns
		-------
		results : dict
			A dictionary containing, among other things, selected variables.

		"""
		self.target_column = target_column
		x_columns = [_ for _ in obj.columns if _ != target_column]
		x_df = obj[x_columns]
		y_df = obj[[target_column]]

		derived_path = None if path is None else path + '-' + self.__class__.__name__
		feature_selector = RFE(learner_func, path=derived_path)
		m = feature_selector.fit(x_df, y_df, n_features, max_duration=max_duration)
		self.models = [m]
		self.selected_variables = feature_selector.selected_variables
		if path:
			self.save(path)

		results = {'Selected Variables': self.selected_variables}
		return results



class BorutaPredictor(BasePredictor):
	def fit(self, obj, target_column, learner_func, n_evaluations=20, pval=0.95, max_duration=None, path=None):
		"""
		Fits a supervised learner enriched with feature selection using the Boruta algorithm.


		Parameters
		----------
		obj : pandas.DataFrame
			A dataframe containing training explanatory variables/features as well as the target.
		target_column : str
			The name of the column in :code:`obj` containing targets.
		learner_func : func | callable
			Function or callable that expects one optional argument :code:`n_vars` and returns an instance of a superviser learner (regressor or classifier) following the scikit-learn convention, and expecting :code:`n_vars` features. Specifically, the learner should have a :code:`fit(x_train, y_train)` method. The learner should also have a :code:`feature_importances_` property or attribute, which is an array or a list containing feature importances once the model has been trained. There should be as many importance scores in :code:`feature_importances_` as columns in :code:`fit(x_train, y_train)`.
		n_evaluations : int (Default 20)
			The number of trials to run in the Boruta algorithm.
		pval : float (Default 0.95)
			The quantile level above which to consider a feature relevant in the Boruta algorithm.
		max_duration : float | None (default)
			If not None, then feature elimination will stop after this many seconds.

		Attributes
		----------
		selected_variables : list
			The list of the :code:`n_features` features we kept.
		target_column : str
			The name of the column used as target.
		models : list
			An array whose first entry is the fitted model.


		Returns
		-------
		results : dict
			A dictionary containing, among other things, selected variables.

		"""
		self.target_column = target_column
		x_columns = [_ for _ in obj.columns if _ != target_column]
		x_df = obj[x_columns]
		y_df = obj[[target_column]]

		derived_path = None if path is None else path + '-' + self.__class__.__name__
		feature_selector = Boruta(learner_func, path=derived_path)
		m = feature_selector.fit(x_df, y_df, n_evaluations=n_evaluations, pval=pval, max_duration=max_duration)
		self.models = [m]
		self.selected_variables = feature_selector.selected_variables
		if path:
			self.save(path)

		results = {'Selected Variables': self.selected_variables}
		return results



class NaivePredictor(BasePredictor):
	def fit(self, obj, target_column, learner_func, path=None):
		"""
		Fits a supervised learner without feature selection.


		Parameters
		----------
		obj : pandas.DataFrame
			A dataframe containing training explanatory variables/features as well as the target.
		target_column : str
			The name of the column in :code:`obj` containing targets.
		learner_func : func | callable
			Function or callable that expects one optional argument :code:`n_vars` and returns an instance of a superviser learner (regressor or classifier) following the scikit-learn convention, and expecting :code:`n_vars` features. Specifically, the learner should have a :code:`fit(x_train, y_train)` method. The learner should also have a :code:`feature_importances_` property or attribute, which is an array or a list containing feature importances once the model has been trained. There should be as many importance scores in :code:`feature_importances_` as columns in :code:`fit(x_train, y_train)`.

		Attributes
		----------
		selected_variables : list
			The list of the :code:`n_features` features we kept.
		target_column : str
			The name of the column used as target.
		models : list
			An array whose first entry is the fitted model.


		Returns
		-------
		results : dict
			A dictionary containing, among other things, selected variables.

		"""
		self.target_column = target_column
		x_columns = [_ for _ in obj.columns if _ != target_column]
		x_df = obj[x_columns]
		y_df = obj[[target_column]]

		derived_path = None if path is None else path + '-' + self.__class__.__name__
		feature_selector = NaiveLearner(learner_func, path=derived_path)
		m = feature_selector.fit(x_df, y_df)
		self.models = [m]
		self.selected_variables = feature_selector.selected_variables
		if path:
			self.save(path)

		results = {'Selected Variables': self.selected_variables}
		return results








