#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from .base_accessor import BaseAccessor
from .features_utils import rmspe_score, neg_rmspe_score
from .pre_learning_accessor import PreLearningAccessor

@pd.api.extensions.register_dataframe_accessor("kxy_learning")
class LearningAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics for automatically training predictive models.

	This class defines the :code:`kxy_learning` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_learning.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""
	def fit(self, target_column, learner_cls, problem_type=None, snr='auto', train_frac=0.8, random_state=0, \
			force_redo=False, max_n_features=None, min_n_features=None, start_n_features=1, anonymize=False, \
			benchmark_feature=None, missing_value_imputation=False, score='auto', n_down_perf_before_stop=1):
		"""
		Train a lean boosted supervised learner, bringing in variables one at a time, in decreasing order of importance (as per :code:`df.kxy.variable_selection`), until doing so no longer improves validation performance or another stopping criterion is met.

		Specifically, training proceeds as follows. First, KXY's model-free variable selection is run (i.e. :code:`df.kxy.variable_selection`). 

		Then we train a model (instance of :code:`learner_cls`) using the :code:`start_n_features` most important feature/variable to predict the target (defined by :code:`target_column`).

		Then we consider using the second most important variable to fix the mistakes made by the previously trained model.
	
		If doing so improves performance on the validation set, we keep going until either performance no longer improves on the validation set :code:`n_down_perf_before_stop` consecutive times, or we've selected :code:`max_n_features` features.



		Parameters
		----------
		target_column : str
			The name of the column containing true labels.
		learner_cls : str
			The class base learners should be instances of. They should define a :code:`fit(x, y)` method and a :code:`predict(x)` method.
		problem_type : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.
		snr : 'auto' | 'low' | 'high'
			Set to :code:`low` if the problem is difficult (i.e. has a low signal-to-noise ratio) or the number of rows is small relative to the number of columns. Only used for model-free variable selection.
		train_frac : float
			The fraction of rows used for training and validation.
		random_state : int
			The seed to use for random training/validation/testing split.
		force_redo : bool
			Fitted models are saved. Set this parameters to :code:`True` to ignore saved models and refit.
		min_n_features : int | None
			Boosting will not stop until at least this many features/explanatory variables are selected.
		max_n_features : int | None
			Boosting will stop as soon as this many features/explanatory variables are selected.
		start_n_features : int
			The number of most important features boosting will start with.
		anonymize : bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost).
		benchmark_feature : str | None
			When not None, 'benchmark' performance metrics using this column as predictor will be reported in the output dictionary.
		missing_value_imputation : bool
			When set to True, replace missing values with medians.
		n_down_perf_before_stop : int
			Number of consecutive down performances to observe before boosting stops.


		Returns
		-------
		result : dict
			Dictionary containing selected variables, as well as training, validation and testing performance.
		"""
		obj = self._obj
		assert inspect.isclass(learner_cls), 'learner_cls should be a class'
		assert target_column in obj.columns, 'The target column should be a valid column'
		if problem_type is None:
			problem_type = 'classification' if self.is_discrete(target_column) else 'regression'
		assert problem_type in ('classification', 'regression')
		self.problem_type = problem_type

		for col in obj.columns:
			assert not self.is_categorical(col), 'All columns should be numeric'

		x_columns = [_ for _ in obj.columns if _ != target_column]
		if self.problem_type == 'classification':
			labels = set(list(obj[target_column].values.astype(int)))
			binary_labels = {0, 1}
			assert labels.issubset(binary_labels), 'Classification labels should either be 0 or 1'

		if benchmark_feature:
			assert benchmark_feature in obj.columns, 'The benchmark feature should be a valid column'
		self.benchmark_feature = benchmark_feature
		if score == 'auto':
			score = 'r2_score' if problem_type == 'regression' else 'accuracy_score'
		score_func = eval(score)

		if getattr(self, 'models', None) is None or force_redo:
			# 0. Train/Validation split
			self.target_column = target_column
			# Reserve 1-train_frac for testing, and train_frac for training and validation
			self.test_df = obj.sample(frac=1.-train_frac, random_state=random_state)
			self.train_val_df = obj.drop(self.test_df.index)
			# Reserve train_frac*train_frac for training [...]
			self.train_df = self.train_val_df.sample(frac=train_frac, random_state=random_state)
			# [...] and train_frac*(1-train_frac) for validation.
			self.val_df = self.train_val_df.drop(self.train_df.index)

			if missing_value_imputation:
				# Basic missing value imputation
				self.train_df.fillna(self.train_df.median(), inplace=True)
				self.val_df.fillna(self.train_df.median(), inplace=True)
				self.test_df.fillna(self.train_df.median(), inplace=True)

			# 1. Model-free variable selection
			vs_accessor = PreLearningAccessor(obj)
			if obj.memory_usage(index=False).sum()/(1024.0*1024.0*1024.0) > 1.:
				# The dataframe is too big to be sent as such: we need to normalize and reduce the precision before uploading the file.
				anonymize=True
			self.variable_selection_results = vs_accessor.variable_selection(self.target_column, problem_type=self.problem_type, \
				snr=snr, anonymize=anonymize)
			self.variables = [_ for _ in self.variable_selection_results['Variable'].values if _.lower() != 'no variable']
			n_variables = len(self.variables)
			if max_n_features:
				n_variables = min(n_variables, max_n_features)

			# 2. Sequentially add variables in decreasing order of importance, each time with the aim to predict previous residuals.
			y_train_df = self.train_df[[self.target_column]]
			y_train = y_train_df.values.copy()
			target_train = y_train.copy()
			y_train_pred = None
			y_val_df = self.val_df[[self.target_column]]
			y_val = y_val_df.values.copy()
			target_val = y_val.copy()
			y_val_pred = None

			models = []
			if problem_type == 'regression':
				initial_pred = np.ones_like(y_val)*y_val.mean()
				previous_score = score_func(y_val, initial_pred)
			else:
				score_label = 'Running Achievable Accuracy'
				previous_score = float(self.variable_selection_results[score_label].loc[0])

			self.start_n_features = start_n_features
			n_down_perf = 0
			for i in range(self.start_n_features, n_variables+1):
				vs = self.variables[:i]
				x_train = self.train_df[vs].values.copy()
				x_val = self.val_df[vs].values.copy()

				# Create the new model
				m = learner_cls()

				# Fit the new model
				m.fit(x_train, target_train)

				# Error predictions
				target_train_pred = m.predict(x_train)
				target_train_pred = target_train_pred if len(target_train_pred.shape) > 1 else target_train_pred[:, None]
				target_val_pred = m.predict(x_val)
				target_val_pred = target_val_pred if len(target_val_pred.shape) > 1 else target_val_pred[:, None]

				# Target predictions updates
				if self.problem_type == 'regression':
					y_train_pred = target_train_pred.copy() if y_train_pred is None else y_train_pred+target_train_pred
					y_val_pred = target_val_pred.copy() if y_val_pred is None else y_val_pred+target_val_pred
				else:
					y_train_pred = target_train_pred.copy() if y_train_pred is None else np.abs(y_train_pred-target_train_pred)
					y_val_pred = target_val_pred.copy() if y_val_pred is None else np.abs(y_val_pred-target_val_pred) 

				# New validation score
				val_score = score_func(y_val, y_val_pred)
				temp_models = []
				if val_score > previous_score or (min_n_features and i<=min_n_features):
					n_down_perf = 0
					models += temp_models
					temp_models = []
					logging.info('Variable #%d (%s) increased validation performance from %.3f to %.3f' % (i, self.variables[i-1], previous_score, val_score))
					previous_score = val_score
					if self.problem_type == 'regression':
						target_train = target_train-target_train_pred
						target_val = target_val-target_val_pred
					else:
						target_train = np.logical_not(target_train == target_train_pred).astype(int)
						target_val = np.logical_not(target_val == target_train_pred).astype(int)
					models += [m]

				else:
					n_down_perf += 1
					logging.info('Validation performance did not increase for the %d-th consecutive time. Old: %.3f, New: %.3f, Variable: %s' % (n_down_perf, previous_score, val_score, self.variables[i-1]))
					if n_down_perf >= n_down_perf_before_stop:
						# Only stop after a certain number of consecutive down performance
						logging.info('Stopping training as validation performance did not increase %d consecutive times.' % n_down_perf_before_stop)
						break
					else:
						if self.problem_type == 'regression':
							target_train = target_train-target_train_pred
							target_val = target_val-target_val_pred
						else:
							target_train = np.logical_not(target_train == target_train_pred).astype(int)
							target_val = np.logical_not(target_val == target_train_pred).astype(int)
						temp_models += [m]

				if max_n_features and (i==max_n_features):
					logging.info('Stopping training as the maximum number of variables (%d) has been reached' % max_n_features)
					break				

			self.models = models

			# Compute training/validation/testing performances
			x_train_df = self.train_df[x_columns]
			y_train_df = self.train_df[[target_column]]
			train_predictions = self.predict(x_train_df)
			self.train_score = score_func(y_train_df.values, train_predictions.values)
			if self.problem_type == 'regression':
				self.train_rmse = mean_squared_error(y_train_df.values, train_predictions.values, squared=False)
				self.train_rmspe = rmspe_score(y_train_df.values, train_predictions.values)
				self.train_r2 = r2_score(y_train_df.values.flatten(), train_predictions.values.flatten())

				if self.benchmark_feature:
					train_benchmark = self.train_df[self.benchmark_feature]
					self.train_benchmark_score = score_func(y_train_df.values, train_benchmark.values)
					self.train_benchmark_rmse = mean_squared_error(y_train_df.values, train_benchmark.values, squared=False)
					self.train_benchmark_rmspe = rmspe_score(y_train_df.values, train_benchmark.values)

			x_val_df = self.val_df[x_columns]
			y_val_df = self.val_df[[target_column]]
			val_predictions = self.predict(x_val_df)
			self.val_score = score_func(y_train_df.values, train_predictions.values)
			if self.problem_type == 'regression':
				self.val_rmse = mean_squared_error(y_val_df.values, val_predictions.values, squared=False)
				self.val_rmspe = rmspe_score(y_val_df.values, val_predictions.values)
				self.val_r2 = r2_score(y_val_df.values.flatten(), val_predictions.values.flatten())

				if self.benchmark_feature:
					val_benchmark = self.val_df[self.benchmark_feature]
					self.val_benchmark_score = score_func(y_val_df.values, val_benchmark.values)
					self.val_benchmark_rmse = mean_squared_error(y_val_df.values, val_benchmark.values, squared=False)
					self.val_benchmark_rmspe = rmspe_score(y_val_df.values, val_benchmark.values)
				
			x_test_df = self.test_df[x_columns]
			y_test_df = self.test_df[[target_column]]
			test_predictions = self.predict(x_test_df)
			self.test_score = score_func(y_test_df.values, test_predictions.values)
			if self.problem_type == 'regression':
				self.test_rmse = mean_squared_error(y_test_df.values, test_predictions.values, squared=False)
				self.test_rmspe = rmspe_score(y_test_df.values, test_predictions.values)
				self.test_r2 = r2_score(y_test_df.values.flatten(), test_predictions.values.flatten())

				if self.benchmark_feature:
					test_benchmark = self.test_df[self.benchmark_feature]
					self.test_benchmark_score = score_func(y_test_df.values, test_benchmark.values)
					self.test_benchmark_rmse = mean_squared_error(y_test_df.values, test_benchmark.values, squared=False)
					self.test_benchmark_rmspe = rmspe_score(y_test_df.values, test_benchmark.values)

			self.selected_variables = self.variables[:self.start_n_features+len(self.models)-1]

		results = {'Selected Variables': self.selected_variables}
		if self.problem_type == 'regression':
			results['Training R-Squared'] = '%.3f' % self.train_score
			results['Validation R-Squared'] = '%.3f' % self.val_score
			results['Testing R-Squared'] = '%.3f' % self.test_score
			results['Training RMSE'] = '%.5f' % self.train_rmse
			results['Validation RMSE'] = '%.5f' % self.val_rmse
			results['Testing RMSE'] = '%.5f' % self.test_rmse
			results['Training RMSPE'] = '%.5f' % self.train_rmspe
			results['Validation RMSPE'] = '%.5f' % self.val_rmspe
			results['Testing RMSPE'] = '%.5f' % self.test_rmspe
			if self.benchmark_feature:
				results['Benchmark Training R-Squared'] = '%.3f' % self.train_benchmark_score
				results['Benchmark Validation R-Squared'] = '%.3f' % self.val_benchmark_score
				results['Benchmark Testing R-Squared'] = '%.3f' % self.test_benchmark_score
				results['Benchmark Training RMSE'] = '%.5f' % self.train_benchmark_rmse
				results['Benchmark Validation RMSE'] = '%.5f' % self.val_benchmark_rmse
				results['Benchmark Testing RMSE'] = '%.5f' % self.test_benchmark_rmse
				results['Benchmark Training RMSPE'] = '%.5f' % self.train_benchmark_rmspe
				results['Benchmark Validation RMSPE'] = '%.5f' % self.val_benchmark_rmspe
				results['Benchmark Testing RMSPE'] = '%.5f' % self.test_benchmark_rmspe



		if self.problem_type == 'classification':
			results['Training Accuracy'] = '%.3f' % self.train_score
			results['Validation Accuracy'] = '%.3f' % self.val_score
			results['Testing Accuracy'] = '%.3f' % self.test_score

		return results
		


	def predict(self, obj):
		"""
		Make predictions using the fitted model.


		Parameters
		----------
		obj : pandas.DataFrame
			A dataframe containing test explanatory variables about which we want to make predictions.


		Returns
		-------
		result : pandas.DataFrame
			A dataframe with the same index as :code:`obj`, and with one column whose name is the :code:`target_column` used for training.
		"""
		assert hasattr(self, 'models'), 'The model should be first fitted'
		for col in obj.columns:
			assert not obj.kxy.is_categorical(col), 'All columns should be numeric'

		data = obj[self.variables]
		y_pred = None
		for i in range(self.start_n_features, self.start_n_features+len(self.models)):
			vs = self.variables[:i]
			x = data[vs].copy()
			y_error_pred = self.models[i-self.start_n_features].predict(x)

			if self.problem_type == 'regression':
				y_pred = y_error_pred.copy() if y_pred is None else y_pred+y_error_pred
			else:
				y_pred = y_error_pred.copy() if y_pred is None else np.abs(y_pred-y_error_pred) 

		predictions = pd.DataFrame(y_pred, columns=[self.target_column], index=data.index)

		return predictions





