#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from .base_accessor import BaseAccessor
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
			benchmark_feature=None):
		"""
		Train a lean boosted supervised learner, bringing in variables one at a time, in decreasing order of importance (as per :code:`df.kxy.variable_selection`), until doing so no longer improves validation performance or another stopping criterion is met.

		Specifically, training proceeds as follows. First, KXY's model-free variable selection is run (i.e. :code:`df.kxy.variable_selection`). 

		Then we train a model (instance of :code:`learner_cls`) using the :code:`start_n_features` most important feature/variable to predict the target (defined by :code:`target_column`).

		Then we consider using the second most important variable to fix the mistakes made by the previously trained model.
	
		If doing so improves performance on the validation set, we keep going until either performance no longer improves on the validation set, or we've selected :code:`max_n_features` features.



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
			score_label = 'Running Achievable R-Squared' if self.problem_type == 'regression' else 'Running Achievable Accuracy'
			previous_score = float(self.variable_selection_results[score_label].loc[0])
			self.start_n_features = start_n_features
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
				if self.problem_type == 'regression':
					val_score = r2_score(y_val, y_val_pred)
				else:
					val_score = accuracy_score(y_val, y_val_pred)

				if val_score > previous_score or (min_n_features and i<=min_n_features):
					logging.info('The %d-th variable (%s) increased validation performance from %.3f to %.3f' % (i, self.variables[i-1], previous_score, val_score))
					previous_score = val_score
					if self.problem_type == 'regression':
						target_train = target_train-target_train_pred
						target_val = target_val-target_val_pred
					else:
						target_train = np.logical_not(target_train == target_train_pred).astype(int)
						target_val = np.logical_not(target_val == target_train_pred).astype(int)
					models += [m]
				else:
					logging.info('Stopping training as variable %s does not increase validation performance (old: %.3f, new: %.3f)' % (self.variables[i-1], previous_score, val_score))
					break

				if max_n_features and (i==max_n_features):
					logging.info('Stopping training as the maximum number of variables (%d) has been reached' % max_n_features)
					break				

			self.models = models

			# Compute training/validation/testing performances
			x_train_df = self.train_df[x_columns]
			y_train_df = self.train_df[target_column]
			train_predictions = self.predict(x_train_df)
			if self.problem_type == 'regression':
				self.train_score = r2_score(y_train_df.values, train_predictions.values)
				self.train_rmse = mean_squared_error(y_train_df.values, train_predictions.values, squared=False)
				self.train_rmspe = mean_squared_error(np.ones_like(y_train_df.values.flatten()), train_predictions.values.flatten()/y_train_df.values.flatten(), squared=False)

				if self.benchmark_feature:
					train_benchmark = self.train_df[self.benchmark_feature]
					self.train_benchmark_score = r2_score(y_train_df.values, train_benchmark.values)
					self.train_benchmark_rmse = mean_squared_error(y_train_df.values, train_benchmark.values, squared=False)
					self.train_benchmark_rmspe = mean_squared_error(np.ones_like(y_train_df.values.flatten()), \
						train_benchmark.values.flatten()/y_train_df.values.flatten(), squared=False)

			else:
				self.train_score = accuracy_score(y_train_df.values, train_predictions.values)

			x_val_df = self.val_df[x_columns]
			y_val_df = self.val_df[target_column]
			val_predictions = self.predict(x_val_df)
			if self.problem_type == 'regression':
				self.val_score = r2_score(y_val_df.values, val_predictions.values)
				self.val_rmse = mean_squared_error(y_val_df.values, val_predictions.values, squared=False)
				self.val_rmspe = mean_squared_error(np.ones_like(y_val_df.values.flatten()), val_predictions.values.flatten()/y_val_df.values.flatten(), squared=False)

				if self.benchmark_feature:
					val_benchmark = self.val_df[self.benchmark_feature]
					self.val_benchmark_score = r2_score(y_val_df.values, val_benchmark.values)
					self.val_benchmark_rmse = mean_squared_error(y_val_df.values, val_benchmark.values, squared=False)
					self.val_benchmark_rmspe = mean_squared_error(np.ones_like(y_val_df.values.flatten()), \
						val_benchmark.values.flatten()/y_val_df.values.flatten(), squared=False)

			else:
				self.val_score = accuracy_score(y_val_df.values, val_predictions.values)

			x_test_df = self.test_df[x_columns]
			y_test_df = self.test_df[target_column]
			test_predictions = self.predict(x_test_df)
			if self.problem_type == 'regression':
				self.test_score = r2_score(y_test_df.values, test_predictions.values)
				self.test_rmse = mean_squared_error(y_test_df.values, test_predictions.values, squared=False)
				self.test_rmspe = mean_squared_error(np.ones_like(y_test_df.values.flatten()), test_predictions.values.flatten()/y_test_df.values.flatten(), squared=False)

				if self.benchmark_feature:
					test_benchmark = self.test_df[self.benchmark_feature]
					self.test_benchmark_score = r2_score(y_test_df.values, test_benchmark.values)
					self.test_benchmark_rmse = mean_squared_error(y_test_df.values, test_benchmark.values, squared=False)
					self.test_benchmark_rmspe = mean_squared_error(np.ones_like(y_test_df.values.flatten()), \
						test_benchmark.values.flatten()/y_test_df.values.flatten(), squared=False)

			else:
				self.test_score = accuracy_score(y_test_df.values, test_predictions.values)

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





