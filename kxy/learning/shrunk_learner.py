#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import gc
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

try:
	get_ipython().__class__.__name__
	from halo import HaloNotebook as Halo
except:
	from halo import Halo

from ..pandas_extension.features_utils import rmspe_score, neg_rmspe_score, neg_mae_score, neg_rmse_score
from ..pandas_extension.pre_learning_accessor import PreLearningAccessor



class BaselineClassifier(object):
	"""
	Classifier following the scikit-learn API and always predicting the most frequent class in-sample.
	"""
	def fit(self, x, y):
		values, counts = np.unique(y, return_counts=True)
		ind = np.argmax(counts)
		self.training_mode = values[ind]

	def predict(self, x):
		try:
			n = x.shape[0]
			return np.array([self.training_mode]*n)[:, None]

		except AttributeError:
			logging.error('The model should be fitted first')



class BaselineRegressor(object):
	"""
	Regressor following the scikit-learn API and always predicting the in-sample mean.
	"""
	def __init__(self, baseline='mean'):
		assert baseline in ['mean', 'median']
		self._baseline = np.nanmean if baseline == 'mean' else np.nanmedian


	def fit(self, x, y):
		self.training_baseline = self._baseline(y)

	def predict(self, x):
		try:
			n = x.shape[0]
			return np.ones((n, 1))*self.training_baseline

		except AttributeError:
			logging.error('The model should be fitted first')



class ShrunkLearner(object):
	"""
	Wrapper to seamlessly add effective variable selection to any supervised learner. 
	"""
	def _non_additive_fit(self, obj, target_column, learner_func, problem_type=None, snr='auto', train_frac=0.8, random_state=0, \
			force_redo=False, max_n_features=None, min_n_features=None, start_n_features=None, anonymize=False, \
			benchmark_feature=None, missing_value_imputation=False, score='auto', n_down_perf_before_stop=3, \
			regression_baseline='mean', regression_error_type='additive', return_scores=False, start_n_features_perf_frac=0.9):
		# A base learner here does not fix mistakes made by another.
		assert inspect.isfunction(learner_func), 'learner_func should be a class'
		assert target_column in obj.columns, 'The target column should be a valid column'
		if problem_type is None:
			problem_type = 'classification' if obj.kxy.is_discrete(target_column) else 'regression'
		assert problem_type in ('classification', 'regression')
		self.problem_type = problem_type
		self.additive_learning = False
		assert regression_error_type in ('additive', 'multiplicative')
		self.regression_error_type = regression_error_type

		for col in obj.columns:
			assert not obj.kxy.is_categorical(col), 'All columns should be numeric'

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
		self.val_scores = []

		if getattr(self, 'models', None) is None or force_redo:
			# 0. Train/Validation split
			self.target_column = target_column
			if return_scores:
				# Reserve 1-train_frac for testing, and train_frac for training and validation
				self.test_df = obj.sample(frac=1.-train_frac, random_state=random_state)
				self.train_val_df = obj.drop(self.test_df.index)
			else:
				self.train_val_df = obj
			# Reserve train_frac*train_frac for training [...]
			self.train_df = self.train_val_df.sample(frac=train_frac, random_state=random_state)
			# [...] and train_frac*(1-train_frac) for validation.
			self.val_df = self.train_val_df.drop(self.train_df.index)

			if missing_value_imputation:
				# Basic missing value imputation
				self.train_df.fillna(self.train_df.median(), inplace=True)
				self.val_df.fillna(self.train_df.median(), inplace=True)
				if return_scores:
					self.test_df.fillna(self.train_df.median(), inplace=True)

			# 1. Model-free variable selection
			vs_accessor = PreLearningAccessor(obj)
			self.variable_selection_results = vs_accessor.variable_selection(self.target_column, problem_type=self.problem_type, \
				snr=snr, anonymize=anonymize)

			self.variables = [_ for _ in self.variable_selection_results['Variable'].values if _.lower() != 'no variable']
			n_variables = len(self.variables)
			if max_n_features:
				n_variables = min(n_variables, max_n_features)

			if start_n_features is None:
				perfs = [_ for _ in self.variable_selection_results['Running Achievable R-Squared'].astype(float)]
				max_perf = np.max(perfs)
				perf_threshold = start_n_features_perf_frac*max_perf
				start_n_features = n_variables-len([_ for _ in perfs if _ > perf_threshold])+1

			# 2. Sequentially add variables in decreasing order of importance.
			# 2.1 Baseline performance
			y_train = self.train_df[[self.target_column]].values
			y_val = self.val_df[[self.target_column]].values
			if return_scores:
				y_test = self.test_df[[self.target_column]].values
			x_train = self.train_df[self.variables[:1]].values
			x_val = self.val_df[self.variables[:1]].values

			base_m = BaselineRegressor(baseline=regression_baseline) if problem_type == 'regression' else BaselineClassifier()
			base_m.fit(x_train, y_train)
			y_val_pred = base_m.predict(x_val)
			previous_score = score_func(y_val, y_val_pred)

			spinner = Halo(text='Lean Boosting:', spinner='dots')
			spinner.start()
			logging.info('Baseline score (%s): %.4f' % (score, previous_score))
			spinner.text = 'Lean Boosting -- Baseline %s: %.4f' % (score, previous_score)

			self.start_n_features = min(start_n_features, n_variables)
			self.models = []
			self.max_var_ixs = []
			n_down_perf = 0
			for i in range(self.start_n_features, n_variables+1):
				gc.collect()
				vs = self.variables[:i]
				x_train = self.train_df[vs].values
				x_val = self.val_df[vs].values
				n_vars = x_train.shape[1] if len(x_train.shape) > 1 else 1

				# Create the new model
				m = learner_func(n_vars=n_vars)

				# Fit the new model
				m.fit(x_train, y_train)

				# New validation score
				y_val_pred = m.predict(x_val)
				val_score = score_func(y_val, y_val_pred)

				if val_score > previous_score or (min_n_features and i<=min_n_features):
					n_down_perf = 0
					logging.info('Variable #%d (%s) increased validation performance from %.4f to %.4f' % (i, self.variables[i-1], previous_score, val_score))
					spinner.text = 'Lean Boosting -- %d Variables, Validation %s: %.4f' % (i, score, val_score)
					previous_score = val_score
					self.models = [m]
					self.max_var_ixs = [i]
					self.val_scores = self.val_scores + [(i, val_score)]

				else:
					n_down_perf += 1
					logging.info('Validation performance did not increase for the %d-th consecutive time. Old: %.4f, New: %.4f, Variable: %s' % (n_down_perf, previous_score, val_score, self.variables[i-1]))
					if n_down_perf >= n_down_perf_before_stop:
						# Only stop after a certain number of consecutive down performance
						logging.info('Stopping training as validation performance did not increase %d consecutive times.' % n_down_perf_before_stop)
						spinner.succeed()
						break

				if max_n_features and (i==max_n_features):
					logging.info('Stopping training as the maximum number of variables (%d) has been reached' % max_n_features)
					spinner.succeed()
					break		

			self.selected_variables = self.variables[:self.max_var_ixs[-1]] if self.models else []
			if self.models == []:
				self.models = [base_m]
				self.max_var_ixs = [1]
				self.val_scores = [(0, previous_score)]

			results = {'Selected Variables': self.selected_variables}
			if return_scores:
				# Inputs
				x_train = self.train_df[self.selected_variables].values
				x_val = self.val_df[self.selected_variables].values
				x_test = self.test_df[self.selected_variables].values

				# Predictions
				self.y_train_pred = self.models[0].predict(x_train)
				self.y_val_pred = self.models[0].predict(x_val)
				self.y_test_pred = self.models[0].predict(x_test)

				# Scores
				self.train_score = score_func(y_train, self.y_train_pred)
				self.val_score = score_func(y_val, self.y_val_pred)
				self.test_score = score_func(y_test, self.y_test_pred)

				results['Training Score'] = '%.5f' % self.train_score
				results['Validation Score'] = '%.5f' % self.val_score
				results['Testing Score'] = '%.5f' % self.test_score

				if self.problem_type == 'regression':
					results['Training R-Squared'] = '%.3f' % r2_score(y_train.flatten(), self.y_train_pred.flatten())
					results['Validation R-Squared'] = '%.3f' % r2_score(y_val.flatten(), self.y_val_pred.flatten())
					results['Testing R-Squared'] = '%.3f' % r2_score(y_test.flatten(), self.y_test_pred.flatten())

					results['Training RMSE'] = '%.5f' % mean_squared_error(y_train.flatten(), self.y_train_pred.flatten(), squared=False)
					results['Validation RMSE'] = '%.5f' % mean_squared_error(y_val.flatten(), self.y_val_pred.flatten(), squared=False)
					results['Testing RMSE'] = '%.5f' % mean_squared_error(y_test.flatten(), self.y_test_pred.flatten(), squared=False)

				if self.problem_type == 'classification':
					results['Training Accuracy'] = '%.3f' % self.train_score
					results['Validation Accuracy'] =  '%.3f' % self.val_score
					results['Testing Accuracy'] = '%.3f' % self.test_score

			return results




	def _additive_fit(self, obj, target_column, learner_func, problem_type=None, snr='auto', train_frac=0.8, random_state=0, \
			force_redo=False, max_n_features=None, min_n_features=None, start_n_features=None, anonymize=False, \
			benchmark_feature=None, missing_value_imputation=False, score='auto', n_down_perf_before_stop=3, \
			regression_baseline='mean', regression_error_type='additive', return_scores=False, start_n_features_perf_frac=0.9):
		# A base learner here is fitted to the residuals of the best model so far.
		assert inspect.isfunction(learner_func), 'learner_func should be a class'
		assert target_column in obj.columns, 'The target column should be a valid column'
		if problem_type is None:
			problem_type = 'classification' if obj.kxy.is_discrete(target_column) else 'regression'
		assert problem_type in ('classification', 'regression')
		self.problem_type = problem_type
		self.additive_learning = True
		assert regression_error_type in ('additive', 'multiplicative')
		self.regression_error_type = regression_error_type

		for col in obj.columns:
			assert not obj.kxy.is_categorical(col), 'All columns should be numeric'

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
		self.val_scores = []

		if getattr(self, 'models', None) is None or force_redo:
			# 0. Train/Validation split
			self.target_column = target_column
			if return_scores:
				# Reserve 1-train_frac for testing, and train_frac for training and validation
				self.test_df = obj.sample(frac=1.-train_frac, random_state=random_state)
				self.train_val_df = obj.drop(self.test_df.index)
			else:
				self.train_val_df = obj
			# Reserve train_frac*train_frac for training [...]
			self.train_df = self.train_val_df.sample(frac=train_frac, random_state=random_state)
			# [...] and train_frac*(1-train_frac) for validation.
			self.val_df = self.train_val_df.drop(self.train_df.index)

			if missing_value_imputation:
				# Basic missing value imputation
				self.train_df.fillna(self.train_df.median(), inplace=True)
				self.val_df.fillna(self.train_df.median(), inplace=True)
				if return_scores:
					self.test_df.fillna(self.train_df.median(), inplace=True)

			# 1. Model-free variable selection
			vs_accessor = PreLearningAccessor(obj)
			self.variable_selection_results = vs_accessor.variable_selection(self.target_column, problem_type=self.problem_type, \
				snr=snr, anonymize=anonymize)

			self.variables = [_ for _ in self.variable_selection_results['Variable'].values if _.lower() != 'no variable']
			n_variables = len(self.variables)
			if max_n_features:
				n_variables = min(n_variables, max_n_features)

			if start_n_features is None:
				perfs = [_ for _ in self.variable_selection_results['Running Achievable R-Squared'].astype(float)]
				max_perf = np.max(perfs)
				perf_threshold = start_n_features_perf_frac*max_perf
				start_n_features = n_variables-len([_ for _ in perfs if _ > perf_threshold])+1

			# 2. Sequentially add variables in decreasing order of importance.
			# 2.1 Baseline performance
			y_train = self.train_df[[self.target_column]].values
			y_val = self.val_df[[self.target_column]].values
			if return_scores:
				y_test = self.test_df[[self.target_column]].values
			x_train = self.train_df[self.variables[:1]].values
			x_val = self.val_df[self.variables[:1]].values

			base_m = BaselineRegressor(baseline=regression_baseline) if problem_type == 'regression' else BaselineClassifier()
			base_m.fit(x_train, y_train)
			y_val_pred = base_m.predict(x_val)
			previous_score = score_func(y_val, y_val_pred)

			spinner = Halo(text='Lean Boosting:', spinner='dots')
			spinner.start()
			logging.info('Baseline score (%s): %.4f' % (score, previous_score))
			spinner.text = 'Lean Boosting -- Baseline %s: %.4f' % (score, previous_score)

			self.start_n_features = min(start_n_features, n_variables)
			n_down_perf = 0
			target_train = y_train.copy()
			y_val_pred = None
			self.models = []
			self.max_var_ixs = []
			for i in range(self.start_n_features, n_variables+1):
				gc.collect()
				vs = self.variables[:i]
				x_train = self.train_df[vs].values
				x_val = self.val_df[vs].values
				n_vars = x_train.shape[1] if len(x_train.shape) > 1 else 1

				# Create the new model
				m = learner_func(n_vars=n_vars)

				# Fit the new model
				m.fit(x_train, target_train)

				# New validation score
				target_val_pred = m.predict(x_val)
				target_val_pred = target_val_pred if len(target_val_pred.shape) > 1 else target_val_pred[:, None]
				target_train_pred = m.predict(x_train)
				target_train_pred = target_train_pred if len(target_train_pred.shape) > 1 else target_train_pred[:, None]

				if y_val_pred is None:
					y_val_pred = target_val_pred
				else:
					if self.problem_type == 'regression':
						y_val_pred = y_val_pred+target_val_pred if self.regression_error_type == 'additive' else \
							y_val_pred*target_val_pred

					if self.problem_type == 'classification':
						y_val_pred = np.abs(y_val_pred-target_val_pred)

				val_score = score_func(y_val, y_val_pred)
				if val_score > previous_score or (min_n_features and i<=min_n_features):
					n_down_perf = 0
					logging.info('Variable #%d (%s) increased validation performance from %.4f to %.4f' % (i, self.variables[i-1], previous_score, val_score))
					spinner.text = 'Lean Boosting -- %d Variables, Validation %s: %.4f' % (i, score, val_score)
					previous_score = val_score

					if self.problem_type == 'regression':
						target_train = target_train-target_train_pred if self.regression_error_type == 'additive' else target_train/target_train_pred

					if self.problem_type == 'classification':
						target_train = np.logical_not(target_train == target_train_pred).astype(int)

					self.models = self.models+[m]
					self.max_var_ixs = self.max_var_ixs+[i]
					self.val_scores = self.val_scores + [(i, val_score)]

				else:
					n_down_perf += 1
					logging.info('Validation performance did not increase for the %d-th consecutive time. Old: %.4f, New: %.4f, Variable: %s' % (n_down_perf, previous_score, val_score, self.variables[i-1]))
					if n_down_perf >= n_down_perf_before_stop:
						# Only stop after a certain number of consecutive down performance
						logging.info('Stopping training as validation performance did not increase %d consecutive times.' % n_down_perf_before_stop)
						spinner.succeed()
						break

				if max_n_features and (i==max_n_features):
					logging.info('Stopping training as the maximum number of variables (%d) has been reached' % max_n_features)
					spinner.succeed()
					break	

			self.selected_variables = self.variables[:self.max_var_ixs[-1]] if self.models else []
			if self.models == []:
				self.models = [base_m]
				self.max_var_ixs = [1]
				self.val_scores = [(0, previous_score)]

			results = {'Selected Variables': self.selected_variables}
			if return_scores:
				# Inputs
				x_train = self.train_df[self.selected_variables].values
				x_val = self.val_df[self.selected_variables].values
				x_test = self.test_df[self.selected_variables].values

				# Predictions
				self.y_train_pred = self.predict(self.train_df)
				self.y_train_pred = self.y_train_pred.values.flatten()

				self.y_val_pred = self.predict(self.val_df)
				self.y_val_pred = self.y_val_pred.values.flatten()

				self.y_test_pred = self.predict(self.test_df)
				self.y_test_pred = self.y_test_pred.values.flatten()

				# Scores
				self.train_score = score_func(y_train, self.y_train_pred)
				self.val_score = score_func(y_val, self.y_val_pred)
				self.test_score = score_func(y_test, self.y_test_pred)

				results['Training Score'] = '%.5f' % self.train_score
				results['Validation Score'] = '%.5f' % self.val_score
				results['Testing Score'] = '%.5f' % self.test_score

				if self.problem_type == 'regression':
					results['Training R-Squared'] = '%.3f' % r2_score(y_train.flatten(), self.y_train_pred.flatten())
					results['Validation R-Squared'] = '%.3f' % r2_score(y_val.flatten(), self.y_val_pred.flatten())
					results['Testing R-Squared'] = '%.3f' % r2_score(y_test.flatten(), self.y_test_pred.flatten())

					results['Training RMSE'] = '%.5f' % mean_squared_error(y_train.flatten(), self.y_train_pred.flatten(), squared=False)
					results['Validation RMSE'] = '%.5f' % mean_squared_error(y_val.flatten(), self.y_val_pred.flatten(), squared=False)
					results['Testing RMSE'] = '%.5f' % mean_squared_error(y_test.flatten(), self.y_test_pred.flatten(), squared=False)

				if self.problem_type == 'classification':
					results['Training Accuracy'] = '%.3f' % self.train_score
					results['Validation Accuracy'] =  '%.3f' % self.val_score
					results['Testing Accuracy'] = '%.3f' % self.test_score

			return results



	def fit(self, obj, target_column, learner_func, problem_type=None, snr='auto', train_frac=0.8, random_state=0, \
			force_redo=False, max_n_features=None, min_n_features=None, start_n_features=None, anonymize=False, \
			benchmark_feature=None, missing_value_imputation=False, score='auto', n_down_perf_before_stop=3, \
			regression_baseline='mean', additive_learning=False, regression_error_type='additive', return_scores=False, \
			start_n_features_perf_frac=0.9):
		"""
		Train a lean boosted supervised learner, bringing in variables one at a time, in decreasing order of importance (as per :code:`obj.kxy.variable_selection`), until doing so no longer improves validation performance or another stopping criterion is met.

		Specifically, training proceeds as follows. First, KXY's model-free variable selection is run (i.e. :code:`obj.kxy.variable_selection`). 

		Then we train a model (instance returned by :code:`learner_func`) using the :code:`start_n_features` most important feature/variable to predict the target (defined by :code:`target_column`).

		Next we consider adding one variable at a time to fix the mistakes made by the previously trained model when :code:`additive_learning` is :code:`True`.
	
		If doing so improves performance on the validation set, we keep going until either performance no longer improves on the validation set :code:`n_down_perf_before_stop` consecutive times, or we've selected :code:`max_n_features` features.

		When :code:`start_n_features` is :code:`None` the initial set of variables is the smallest set of variables with which we may achieve :code:`start_n_features_perf_frac` of the performance we could achieve using all variables (as per :code:`obj.kxy.variable_selection`).

		When :code:`additive_learning` is set to :code:`False`, after adding a new variable, we will train the new model on the original problem, rather than trying to improve residuals.


		Parameters
		----------
		obj : pd.DataFrame
			The pandas dataframe containing training data.
		target_column : str
			The name of the column containing true labels.
		learner_func : function
			A function returning an instance of a base learner. They should define a :code:`fit(x, y)` method and a :code:`predict(x)` method.
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
		regression_baseline : str (:code:`mean` | :code:`median`)
			Whether to use the unconditional mean or median as the best predictor in the absence of explanatory variables.
			Choosing the mean corresponds to minimizing the L2 norm, whereas choosing the median corresponds to minimizing the L1 norm.
		additive_learning : bool
			When a new variable is added, whether errors/residuals should be fixed or a new model should be learned from scratch.
		regression_error_type : str ('additive' | 'multiplicative')
			For regression problems with additive learning, this determines whether the final model should be additive (pruning tries to reduce regressor residuals) or multiplicative (i.e. pruning tries to bring the ratio between true and predicted labels as closed to 1 as possible).
		start_n_features_perf_frac : float (between 0 and 1)
			When :code:`start_n_features` is not specified, it is set to the number of variables required to achieve a fraction :code:`start_n_features_perf_frac` of the maximum performance achievable (as per :code:`obj.kxy.variable_selection`).
		return_scores : bool (Default False)
			Whether to return training, validation and testing performance after lean boosting.




		Returns
		-------
		result : dict
			Dictionary containing selected variables, as well as training, validation and testing performance.
		"""

		if additive_learning:
			return self._additive_fit(obj, target_column, learner_func, problem_type=problem_type, snr=snr, train_frac=train_frac, random_state=random_state, \
				force_redo=force_redo, max_n_features=max_n_features, min_n_features=min_n_features, start_n_features=start_n_features, anonymize=anonymize, \
				benchmark_feature=benchmark_feature, missing_value_imputation=missing_value_imputation, score=score, n_down_perf_before_stop=n_down_perf_before_stop, \
				regression_baseline=regression_baseline, regression_error_type=regression_error_type, return_scores=return_scores, start_n_features_perf_frac=start_n_features_perf_frac)

		else:
			return self._non_additive_fit(obj, target_column, learner_func, problem_type=problem_type, snr=snr, train_frac=train_frac, random_state=random_state, \
				force_redo=force_redo, max_n_features=max_n_features, min_n_features=min_n_features, start_n_features=start_n_features, anonymize=anonymize, \
				benchmark_feature=benchmark_feature, missing_value_imputation=missing_value_imputation, score=score, n_down_perf_before_stop=n_down_perf_before_stop, \
				regression_baseline=regression_baseline, regression_error_type=regression_error_type, return_scores=return_scores, start_n_features_perf_frac=start_n_features_perf_frac)



	def _additive_predict(self, obj):
		# Global predict in an additive setting (when base learners are fitted to residuals, not the original problem).
		assert hasattr(self, 'models'), 'The model should be first fitted'
		for col in obj.columns:
			assert not obj.kxy.is_categorical(col), 'All columns should be numeric'

		data = obj[self.variables]
		y_pred = None
		for i in range(len(self.models)):
			vs = self.variables[:self.max_var_ixs[i]]
			x = data[vs].values
			y_error_pred = self.models[i].predict(x)
			y_error_pred = y_error_pred if len(y_error_pred.shape) > 1 else y_error_pred[:, None]
			if y_pred is None:
				y_pred = y_error_pred
			else:
				if self.problem_type == 'regression':
					y_pred = y_pred+y_error_pred if self.regression_error_type == 'additive' else \
						y_pred*y_error_pred

				if self.problem_type == 'classification':
					y_pred = np.abs(y_pred-y_error_pred)

		y_pred = y_pred if len(y_pred.shape) > 1 else y_pred[:, None]
		predictions = pd.DataFrame(y_pred, columns=[self.target_column], index=data.index)

		return predictions


	def _non_additive_predict(self, obj):
		# Global predict in a non-additive setting (when base learners are fitted to the original problem, no residuals).
		assert hasattr(self, 'models'), 'The model should be first fitted'
		for col in obj.columns:
			assert not obj.kxy.is_categorical(col), 'All columns should be numeric'

		data = obj[self.variables]
		x = data[self.selected_variables].values
		y_pred = self.models[0].predict(x)
		y_pred = y_pred if len(y_pred.shape) > 1 else y_pred[:, None]
		predictions = pd.DataFrame(y_pred, columns=[self.target_column], index=data.index)

		return predictions


	def _predict(self, obj):
		assert hasattr(self, 'models'), 'The model should be first fitted'
		if self.additive_learning:
			return self._additive_predict(obj)
		else:
			return self._non_additive_predict(obj)


	def predict(self, obj, memory_bound=False):
		"""
		Make predictions using the fitted model.


		Parameters
		----------
		obj : pandas.DataFrame
			A dataframe containing test explanatory variables about which we want to make predictions.
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
