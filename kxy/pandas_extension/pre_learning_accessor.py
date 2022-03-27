#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from kxy.pre_learning import data_valuation as dv
from kxy.pre_learning import variable_selection as vs

from .base_accessor import BaseAccessor

@pd.api.extensions.register_dataframe_accessor("kxy_pre_learning")
class PreLearningAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics for **post-learning** in supervised learning problems.

	This class defines the :code:`kxy_pre_learning` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_pre_learning.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""
	def data_valuation(self, target_column, problem_type=None, anonymize=None, snr='auto', include_mutual_information=False, file_name=None):
		"""
		Estimate the highest performance metrics achievable when predicting the :code:`target_column` using all other columns.

		When :code:`problem_type=None`, the nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`target_column` is categorical.


		Parameters
		----------
		target_column : str
			The name of the column containing true labels.
		problem_type : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.
		anonymize : None | bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost). When set to None (the default), your data will be anonymized when it is too big.
		include_mutual_information : bool
			Whether to include the mutual information between target and explanatory variables in the result.


		Returns
		-------
		achievable_performance : pandas.Dataframe
			The result is a pandas.Dataframe with columns (where applicable):

			* :code:`'Achievable Accuracy'`: The highest classification accuracy that can be achieved by a model using provided inputs to predict the label.
			* :code:`'Achievable R^2'`: The highest :math:`R^2` that can be achieved by a model using provided inputs to predict the label.
			* :code:`'Achievable RMSE'`: The lowest Root Mean Square Error that can be achieved by a model using provided inputs to predict the label.		
			* :code:`'Achievable Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a model using provided inputs to predict the label.



		.. admonition:: Theoretical Foundation

			Section :ref:`1 - Achievable Performance`.


		.. seealso::

			:ref:`kxy.pre_learning.achievable_performance.data_valuation <data-valuation>`
			
		"""
		assert target_column in self._obj.columns, 'The target_column should be a column'
		if problem_type is None:
			problem_type = 'classification' if self.is_discrete(target_column) else 'regression'
		self.check_problem_type(problem_type, target_column)

		_obj = self.anonymize(columns_to_exclude=[target_column]) if anonymize or (anonymize is None and self.is_too_large) else self._obj

		return dv(_obj, target_column, problem_type, snr=snr, include_mutual_information=include_mutual_information, \
			file_name=file_name)


	def variable_selection(self, target_column, problem_type=None, anonymize=None, snr='auto', file_name=None):
		"""
		Runs the model-free variable selection analysis.

		When :code:`problem_type=None`, the nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`target_column` is categorical.


		Parameters
		----------
		target_column : str
			The name of the column containing true labels.
		problem_type : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.
		anonymize : None | bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost). When set to None (the default), your data will be anonymized when it is too big.

		Returns
		-------
		result : pandas.DataFrame
			The result is a pandas.DataFrame with columns (where applicable):

			* :code:`'Selection Order'`: The order in which the associated variable was selected, starting at 1 for the most important variable.
			* :code:`'Variable'`: The column name corresponding to the input variable.
			* :code:`'Running Achievable R^2'`: The highest :math:`R^2` that can be achieved by a classification model using all variables selected so far, including this one.
			* :code:`'Running Achievable Accuracy'`: The highest classification accuracy that can be achieved by a classification model using all variables selected so far, including this one.
			* :code:`'Running Achievable RMSE'`: The highest classification accuracy that can be achieved by a classification model using all variables selected so far, including this one.


		.. admonition:: Theoretical Foundation

			Section :ref:`2 - Variable Selection Analysis`.

		.. seealso::

			:ref:`kxy.pre_learning.variable_selection.variable_selection <variable-selection>`
		"""
		assert target_column in self._obj.columns, 'The target_column should be a column'
		if problem_type is None:
			problem_type = 'classification' if self.is_discrete(target_column) else 'regression'
		self.check_problem_type(problem_type, target_column)
		
		_obj = self.anonymize(columns_to_exclude=[target_column]) if anonymize or (anonymize is None and self.is_too_large) else self._obj

		return vs(_obj, target_column, problem_type, snr=snr, file_name=file_name)





