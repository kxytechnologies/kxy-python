#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from kxy.api.core import auto_predictability, prepare_data_for_mutual_info_analysis

from kxy.classification import classification_model_improvability_analysis, \
	classification_model_explanation_analysis, classification_bias

from kxy.regression import regression_model_improvability_analysis, \
	regression_model_explanation_analysis, regression_bias

from .base_accessor import BaseAccessor


@pd.api.extensions.register_dataframe_accessor("kxy_post_learning")
class PostLearningAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics for **post-learning** in supervised learning problems.

	This class defines the :code:`kxy_post_learning` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_post_learning.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""

	def model_improvability_analysis(self, label_column, model_prediction_column, input_columns=(), 
			space='dual', categorical_encoding='two-split', problem=None):
		"""
		Runs the model improvability analysis on a trained supervised learning model.

		The nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`label_column` is categorical.


		Parameters
		----------
		label_column : str
			The name of the column containing true labels.
		model_prediction_column : str
			The name of the column containing labels predicted by the model.
		input_columns : set
			List of columns to use as inputs. When an empty set/list is provided, all columns but :code:`model_prediction_column` and :code:`label_column` are used as inputs.
		space : str, 'primal' | 'dual'
			The space in which the maximum entropy problem is solved. 
			When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
			When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
		categorical_encoding : str, 'one-hot' | 'two-split' (default)
			The encoding method to use to represent categorical variables. 
			See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.
		problem : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.


		Returns
		-------
		res : pandas.Styler
			res.data is a pandas.Dataframe with columns (where applicable):

				* :code:`'Leftover R^2'`: The amount by which the trained model's :math:`R^2` can still be increased without resorting to additional inputs, simply through better modeling.
				* :code:`'Leftover Log-Likelihood Per Sample'`: The amount by which the trained model's true log-likelihood per sample can still be increased without resorting to additional inputs, simply through better modeling.
				* :code:`'Leftover Accuracy'`: The amount by which the trained model's classification accuracy can still be increased without resorting to additional inputs, simply through better modeling.


		.. admonition:: Theoretical Foundation

			Section :ref:`3 - Model Improvability`.

		.. seealso::

			* :ref:`kxy.regression.regression_model_improvability_analysis <regression-model-improvability-analysis>`
			* :ref:`kxy.classification.classification_model_improvability_analysis <classification-model-improvability-analysis>`
		"""
		if problem is None:
			problem = 'classification' if self.is_discrete(label_column) else 'regression'

		columns = [col for col in self._obj.columns if col != label_column and col != model_prediction_column] \
			if len(input_columns) == 0 else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		y = self._obj[label_column].values
		y_p = self._obj[model_prediction_column].values
		x_c = self._obj[continuous_columns].values.astype(float) if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values.astype(str) if len(discrete_columns) > 0 else None

		res = regression_model_improvability_analysis(x_c, y_p, y, x_d=x_d, space=space, \
				categorical_encoding=categorical_encoding) if problem == 'regression' \
			else classification_model_improvability_analysis(x_c, y_p, y, x_d=x_d, space=space, \
				categorical_encoding=categorical_encoding)
		res = res.style.hide_index()

		return res



	def model_explanation_analysis(self, model_prediction_column, input_columns=(), space='dual', categorical_encoding='two-split', problem=None):
		"""
		Runs the model explanation analysis on a trained supervised learning model.

		The nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`model_prediction_column` is categorical.

		Parameters
		----------
		model_prediction_column : str
			The name of the column containing predicted labels.
		input_columns : set
			List of columns to use as inputs. When an empty set/list is provided, all columns but :code:`model_prediction_column` are used as inputs.
		space : str, 'primal' | 'dual'
			The space in which the maximum entropy problem is solved. 
			When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
			When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
		categorical_encoding : str, 'one-hot' | 'two-split' (default)
			The encoding method to use to represent categorical variables. 
			See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.
		problem : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.


		Returns
		-------
		res : pandas.Styler
			res.data is a pandas.Dataframe with columns:

				* :code:`'Variable'`: The column name corresponding to the input variable.
				* :code:`'Selection Order'`: The order in which the associated variable was selected, starting at 1 for the most important variable.
				* :code:`'Univariate Explained R^2'`: The :math:`R^2` between predicted labels and this variable.
				* :code:`'Running Explained R^2'`: The :math:`R^2` between predicted labels and all variables selected so far, including this one.
				* :code:`'Marginal Explained R^2'`: The increase in :math:`R^2` between predicted labels and all variables selected so far that is due to adding this variable in the selection scheme.



		.. admonition:: Theoretical Foundation

			Section :ref:`a) Model Explanation`.

		.. seealso::

			* :ref:`kxy.regression.regression_model_explanation_analysis <regression-model-explanation-analysis>`
			* :ref:`kxy.classification.classification_model_explanation_analysis <classification-model-explanation-analysis>`
		"""
		if problem is None:
			problem = 'classification' if self.is_discrete(model_prediction_column) else 'regression'

		columns = [col for col in self._obj.columns if col != model_prediction_column] if len(input_columns) == 0\
			else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		f_x = self._obj[model_prediction_column].values
		x_c = self._obj[continuous_columns].values.astype(float) if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values.astype(str) if len(discrete_columns) > 0 else None

		res = regression_model_explanation_analysis(x_c, f_x, x_d=x_d, space=space, categorical_encoding=categorical_encoding) if problem == 'regression' \
			else classification_model_explanation_analysis(x_c, f_x, x_d=x_d, space=space, categorical_encoding=categorical_encoding)

		variable_columns = continuous_columns + discrete_columns
		res['Variable'] = res['Variable'].map({i: variable_columns[i] for i in range(len(variable_columns))})
		res.set_index(['Variable'], inplace=True)

		try:
			import seaborn as sns
			cm = sns.light_palette("green", as_cmap=True)
			res = res.style.background_gradient(cmap=cm)
		except:
			res = res.style.background_gradient()
			

		return res



	def bias(self, bias_source_column, model_prediction_column, linear_scale=True, categorical_encoding='two-split', problem=None):
		"""
		Quantifies the bias in a supervised learning model as the mutual information between a possible cause and model predictions.

		The nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`model_prediction_column` is categorical.


		Parameters
		----------
		bias_source_column : str
			The name of the column containing values of the bias factor (e.g. age, gender, etc.)
		model_prediction_column : str
			The name of the column containing predicted labels associated to the values of :code:`bias_source_column`.
		linear_scale : bool
			Whether the bias should be returned in the linear/correlation scale or in the mutual information scale (in nats).
		categorical_encoding : str, 'one-hot' | 'two-split' (default)
			The encoding method to use to represent categorical variables. 
			See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.
		problem : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.

		Returns
		-------
		 : float
			The mutual information :math:`m` or :math:`1-e^{-2m}` if :code:`linear_scale=True`.


		.. admonition:: Theoretical Foundation

			Section :ref:`b) Quantifying Bias in Models`.

		.. seealso::

			* :ref:`kxy.regression.regression_bias <regression-bias>`
			* :ref:`kxy.classification.classification_bias <classification-bias>`
		"""
		if problem is None:
			problem = 'classification' if self.is_discrete(model_prediction_column) else 'regression'

		f_x = self._obj[model_prediction_column].values
		z = self._obj[bias_source_column].values

		return regression_bias(f_x, z, linear_scale=linear_scale, categorical_encoding=categorical_encoding) if problem == 'regression' \
			else classification_bias(f_x, z, linear_scale=linear_scale)
