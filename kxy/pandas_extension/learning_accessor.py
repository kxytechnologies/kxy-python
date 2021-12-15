#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from .base_accessor import BaseAccessor
from ..learning.shrunk_learner import ShrunkLearner



@pd.api.extensions.register_dataframe_accessor("kxy_learning")
class LearningAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics for automatically training predictive models.

	This class defines the :code:`kxy_learning` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_learning.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""


	def fit(self, target_column, learner_func, problem_type=None, snr='auto', train_frac=0.8, random_state=0, \
			max_n_features=None, min_n_features=None, start_n_features=None, anonymize=False, \
			benchmark_feature=None, missing_value_imputation=False, score='auto', n_down_perf_before_stop=3, \
			regression_baseline='mean', additive_learning=False, regression_error_type='additive', return_scores=False, \
			start_n_features_perf_frac=0.9):
		"""
		Train a lean boosted supervised learner, bringing in variables one at a time, in decreasing order of importance (as per :code:`df.kxy.variable_selection`), until doing so no longer improves validation performance or another stopping criterion is met.

		Specifically, training proceeds as follows. First, KXY's model-free variable selection is run (i.e. :code:`df.kxy.variable_selection`). 

		Then we train a model (instance returned by :code:`learner_func`) using the :code:`start_n_features` most important feature/variable to predict the target (defined by :code:`target_column`).

		Next we consider adding one variable at a time to fix the mistakes made by the previously trained model when :code:`additive_learning` is :code:`True`.
	
		If doing so improves performance on the validation set, we keep going until either performance no longer improves on the validation set :code:`n_down_perf_before_stop` consecutive times, or we've selected :code:`max_n_features` features.

		When :code:`start_n_features` is :code:`None` the initial set of variables is the smallest set of variables with which we may achieve :code:`start_n_features_perf_frac` of the performance we could achieve using all variables (as per :code:`df.kxy.variable_selection`).

		When :code:`additive_learning` is set to :code:`False`, after adding a new variable, we will train the new model on the original problem, rather than trying to improve residuals.


		Parameters
		----------
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
			When :code:`start_n_features` is not specified, it is set to the number of variables required to achieve a fraction :code:`start_n_features_perf_frac` of the maximum performance achievable (as per :code:`df.kxy.variable_selection`).
		return_scores : bool (Default False)
			Whether to return training, validation and testing performance after lean boosting.




		Returns
		-------
		result : dict
			Dictionary containing selected variables, as well as training, validation and testing performance, and the trained model.
		"""

		predictor = ShrunkLearner()
		res = predictor.fit(self._obj, target_column, learner_func, problem_type=problem_type, snr=snr, train_frac=train_frac, random_state=random_state, \
				max_n_features=max_n_features, min_n_features=min_n_features, start_n_features=start_n_features, anonymize=anonymize, \
				benchmark_feature=benchmark_feature, missing_value_imputation=missing_value_imputation, score=score, n_down_perf_before_stop=n_down_perf_before_stop, \
				regression_baseline=regression_baseline, regression_error_type=regression_error_type, return_scores=return_scores, start_n_features_perf_frac=start_n_features_perf_frac)
		self.predictor = predictor
		res['predictor'] = predictor

		return res


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
		assert hasattr(self, 'predictor'), 'The model should be first fitted'
		return self.predictor.predict(obj, memory_bound=memory_bound)





