#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from kxy.api import upload_data
from kxy.post_learning import data_driven_improvability as ddi
from kxy.post_learning import model_driven_improvability as mdi
from kxy.post_learning import model_explanation as me

from .base_accessor import BaseAccessor

@pd.api.extensions.register_dataframe_accessor("kxy_post_learning")
class PostLearningAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics for **post-learning** in supervised learning problems.

	This class defines the :code:`kxy_post_learning` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_post_learning.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""

	def data_driven_improvability(self, target_column, new_variables, problem_type=None, anonymize=False, snr='auto'):
		"""
		Estimate the potential performance boost that a set of new explanatory variables can bring about.


		Parameters
		----------
		target_column : str
			The name of the column containing true labels.
		new_variables : list
			The names of the columns to use as new explanatory variables.
		problem_type : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from whether or not :code:`target_column` is categorical.
		anonymize : bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost).



		Returns
		-------
		result : pandas.Dataframe
			The result is a pandas.Dataframe with columns (where applicable):

			* :code:`'Accuracy Boost'`: The classification accuracy boost that the new explanatory variables can bring about.
			* :code:`'R-Squared Boost'`: The :math:`R^2` boost that the new explanatory variables can bring about.
			* :code:`'RMSE Reduction'`: The reduction in Root Mean Square Error that the new explanatory variables can bring about.
			* :code:`'Log-Likelihood Per Sample Boost'`: The boost in log-likelihood per sample that the new explanatory variables can bring about.


		.. admonition:: Theoretical Foundation

			Section :ref:`3 - Model Improvability`.


		.. seealso::

			:ref:`kxy.post_learning.improvability.data_driven_improvability <data-driven-improvability>`

		"""
		assert target_column in self._obj.columns, 'The target_column should be a column'
		if problem_type is None:
			problem_type = 'classification' if self.is_discrete(target_column) else 'regression'

		_obj = self.anonymize(columns_to_exclude=[target_column]) if anonymize or self.is_too_large else self._obj

		return ddi(_obj, target_column, new_variables, problem_type, snr=snr)


	def model_driven_improvability(self, target_column, prediction_column, problem_type=None, anonymize=False, snr='auto'):
		"""
		Estimate the extent to which a trained supervised learner may be improved in a model-driven fashion (i.e. without resorting to additional explanatory variables).


		Parameters
		----------
		target_column : str
			The name of the column containing true labels.
		prediction_column : str
			The name of the column containing model predictions.
		problem_type : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from whether or not :code:`target_column` is categorical.
		anonymize : bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost).


		Returns
		-------
		result : pandas.Dataframe
			The result is a pandas.Dataframe with columns (where applicable):

			* :code:`'Lost Accuracy'`: The amount of classification accuracy that was irreversibly lost when training the supervised learner.
			* :code:`'Lost R-Squared'`: The amount of :math:`R^2` that was irreversibly lost when training the supervised learner.
			* :code:`'Lost RMSE'`: The amount of Root Mean Square Error that was irreversibly lost when training the supervised learner.		
			* :code:`'Lost Log-Likelihood Per Sample'`: The amount of true log-likelihood per sample that was irreversibly lost when training the supervised learner.

			* :code:`'Residual R-Squared'`: For regression problems, this is the highest :math:`R^2` that may be achieved when using explanatory variables to predict regression residuals.
			* :code:`'Residual RMSE'`: For regression problems, this is the lowest Root Mean Square Error that may be achieved when using explanatory variables to predict regression residuals.
			* :code:`'Residual Log-Likelihood Per Sample'`: For regression problems, this is the highest log-likelihood per sample that may be achieved when using explanatory variables to predict regression residuals.


		.. admonition:: Theoretical Foundation

			Section :ref:`3 - Model Improvability`.


		.. seealso::

			:ref:`kxy.post_learning.improvability.model_driven_improvability <model-driven-improvability>`

		"""
		assert target_column in self._obj.columns, 'The target_column should be a column'
		assert prediction_column in self._obj.columns, 'The prediction_column should be a column'
		if problem_type is None:
			problem_type = 'classification' if self.is_discrete(target_column) else 'regression'

		_obj = self.anonymize(columns_to_exclude=[target_column, prediction_column]) if anonymize or self.is_too_large else self._obj

		return mdi(_obj, target_column, prediction_column, problem_type, snr=snr)


	def model_explanation(self, prediction_column, problem_type=None, anonymize=False, snr='auto'):
		"""
		Analyzes the variables that a model relies on the most in a brute-force fashion.
		
		The first variable is the variable the model relies on the most. The second variable is the variable that complements the first variable the most in explaining model decisions etc.

		Running performances should be understood as the performance achievable when trying to guess model predictions using variables with selection order smaller or equal to that of the row.

		When :code:`problem_type=None`, the nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`target_column` is categorical.


		Parameters
		----------
		prediction_column : str
			The name of the column containing model predictions.
		problem_type : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.
		anonymize : bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost).

		Returns
		-------
		result : pandas.DataFrame
			The result is a pandas.Dataframe with columns (where applicable):

			* :code:`'Selection Order'`: The order in which the associated variable was selected, starting at 1 for the most important variable.
			* :code:`'Variable'`: The column name corresponding to the input variable.
			* :code:`'Running Achievable R^2'`: The highest :math:`R^2` that can be achieved by a classification model using all variables selected so far, including this one.
			* :code:`'Running Achievable Accuracy'`: The highest classification accuracy that can be achieved by a classification model using all variables selected so far, including this one.
			* :code:`'Running Achievable RMSE'`: The highest classification accuracy that can be achieved by a classification model using all variables selected so far, including this one.


		.. admonition:: Theoretical Foundation

			Section :ref:`2 - Variable Selection Analysis`.


		.. seealso::

			:ref:`kxy.post_learning.model_explanation.model_explanation <variable-selection>`
			
		"""
		assert prediction_column in self._obj.columns, 'The prediction_column should be a column'
		if problem_type is None:
			problem_type = 'classification' if self.is_discrete(prediction_column) else 'regression'

		_obj = self.anonymize(columns_to_exclude=[prediction_column]) if anonymize or self.is_too_large else self._obj

		return me(_obj, prediction_column, problem_type, snr=snr)


