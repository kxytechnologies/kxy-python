#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extension of the pandas DataFrame class to allow data scientists to tap into 
the power of the KXY API within the comfort of their favorite data structure.

All methods defined in the :code:`KXYAccessor` class are accessible from any 
DataFrame instance under :code:`df.kxy.*`, so long as the :code:`kxy` python 
package is imported alongside :code:`pandas`.

See https://pandas.pydata.org/pandas-docs/stable/development/extending.html 
"""

from functools import lru_cache, wraps
import logging
import os
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from kxy.api.core import least_total_correlation, spearman_corr, scalar_continuous_entropy, \
	pearson_corr
from kxy.api import solve_copula_async, mutual_information_analysis
from kxy.classification import classification_feasibility, classification_suboptimality, \
	classification_input_incremental_importance
from kxy.finance import information_adjusted_beta, information_adjusted_correlation
from kxy.regression import regression_feasibility, regression_suboptimality, regression_additive_suboptimality, \
	regression_input_incremental_importance



@pd.api.extensions.register_dataframe_accessor("kxy")
class KXYAccessor(object):
	"""
	Extension of the pandas.DataFrame class with various analytics for **pre-learning** and **post-learning**,
	in supervised learning problems.
	"""
	def __init__(self, pandas_obj):
		self._validate(pandas_obj)
		self._obj = pandas_obj

	@staticmethod
	def _validate(obj):
		api_key = os.getenv('KXY_API_KEY')
		assert api_key is not None, 'The KXY_API_KEY environment variable should be set with a valid API key'

		
	def regression_feasibility(self, label_column, input_columns=()):
		"""
		Quantifies how feasible a regression problem is by computing the amount of uncertainty
		about the label that can be reduced by knowing the inputs.

		.. math::
			FE(x; y) &= h(y)-h\\left(y \\vert x \\right) \\

									  &= I(y, x)

									  &= h\\left(u_x\\right)-h\\left(u_{x,y}\\right)

		Copula entropies are estimated by solving a maximum-entropy copula problem under concordance-like
		constraints on x and on (y, x) jointly. 


		.. seealso::

			:ref:`kxy.regression.pre_learning.regression-feasibility <regression-feasibility>`.


		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		input_columns : set, optional
			The set of columns to as inputs. When input_columns is the empty set,
			all columns except for label_column are used as inputs.

		Returns
		-------
		f : float
			The feasibility score in :math:`[0, \\infty]`. The larger the better.

		Raises
		------
		AssertionError
			If label_column is in input_columns.
		"""
		input_columns = list(set(self._obj.columns)-set([label_column])) if input_columns == () \
			else list(input_columns)
		assert label_column not in input_columns, "The output cannot be a input"

		input_importance = self.individual_input_importance(label_column, input_columns=input_columns,\
			problem='regression')
		input_sorted_by_importance = input_importance['input'].values

		return regression_feasibility(self._obj[input_sorted_by_importance].values, self._obj[label_column].values)




	def adjust_quantized_values(self):
		"""
		Add a negligible random gitter to columns that are continuous but quantized in an attempt 
		to make observations unique, and so as to avoid theoretical incongruities.
		"""
		previously_adjusted = getattr(self._obj, 'previously_adjusted', False)
		if not previously_adjusted:
			random_state = np.random.RandomState(0)
			for col in self._obj.columns:
				if self.is_discrete(col):
					continue

				x = self._obj[col].values
				sx = np.unique(x)
				if len(sx) != self._obj.shape[0]:
					dx = min(np.min(sx[1:]-sx[:-1])/10000.0, 0.00001)
					eps = random_state.rand(*x.shape)*dx
					x += eps

		setattr(self._obj, 'previously_adjusted', True)




	def classification_feasibility(self, label_column, discrete_input_columns=(), \
			continuous_input_columns=()):
		"""
		Quantifies how feasible a classification problem is by computing the amount of uncertainty
		about the label that can be reduced by knowing the inputs, in a model-free fashion.

		.. math::
			FE(x; y) &= h(y)-h\\left(y \\vert x \\right) \\

									 &= I(y, x)

		.. seealso::

			:ref:`kxy.classification.pre_learning.classification_feasibility <classification-feasibility>`.

		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		discrete_input_columns : set, optional
			The set of columns, if any, to use as inputs that are discrete.
		continuous_input_columns : set, optional
			The set of columns to use as inputs that are continuous.

		Returns
		-------
		f : float
			The feasibility score in :math:`[0, \\infty]`. The larger the better.
		"""
		self.adjust_quantized_values()
		assert label_column not in discrete_input_columns, "The output cannot be a input"
		assert label_column not in continuous_input_columns, "The output cannot be a input"

		continuous_input_columns = [col for col in self._obj.columns if col != label_column and not self.is_discrete(col)] \
			if continuous_input_columns == () else list(continuous_input_columns)
		assert len(continuous_input_columns) > 0, "Continuous inputs are required"

		x_d = self._obj[discrete_input_columns].values if len(discrete_input_columns) > 0 else None
		x_c = self._obj[continuous_input_columns].values
		y = self._obj[label_column].values

		return classification_feasibility(x_c, y, x_d=x_d)



	def classification_suboptimality(self, prediction_column, label_column, discrete_input_columns=(), \
			continuous_input_columns=()):
		"""
		Quantifies the extent to which a (multinomial) classifier can be improved without requiring additional inputs.

		.. note::

			The conditional entropy :math:`h \\left( y \\vert x \\right)` represents the amount of information 
			about :math:`y` that cannot be explained by :math:`x`. If we denote :math:`f(x)` the label 
			predicted by our classifier, :math:`h \\left( y \\vert f(x) \\right)` represents the amount
			of information about :math:`y` that the classifier is not able to explain using :math:`x`.

			A natural metric for how suboptimal a particular classifier is can therefore be defined as the 
			difference between the amount of information about :math:`y` that cannot be explained by 
			:math:`f(x)` and the amount of information about :math:`y` that cannot be explained by :math:`x`

			.. math::

				\\text{SO}(f; x) &= h \\left( y \\vert f(x) \\right) - h \\left( y \\vert x \\right) \\

				:&= I\\left(y, x \\right) - I\\left(y, f(x) \\right) \\

				 &\\geq 0.

			This classification suboptimality metric is 0 if and only if :math:`f(x)` fully captures any information about :math:`y`
			that is contained in :math:`x`. When 

			.. math::

				\\text{SO}(f; x) > 0 

			on the other hand, there exists a classification model using :math:`x` as inputs that can better predict :math:`y`. The larger 
			:math:`\\text{SO}(f; x)`, the more the classification model is suboptimal and can be improved.


		Parameters
		----------
		prediction_column : str
			The column containing predicted labels.
		label_column : str
			The column containing true labels.
		continuous_input_columns : set
			The set of columns containing continuous inputs.
		discrete_input_columns : set
			The set of columns containing discrete input, if any.


		Returns
		-------
		d : float
			The classifier's suboptimality measure.


		.. seealso::

			:ref:`kxy.classification.post_learning.classification_suboptimality <classification-suboptimality>`
		"""
		assert label_column not in discrete_input_columns, "The output cannot be a input"
		assert label_column not in continuous_input_columns, "The output cannot be a input"
		self.adjust_quantized_values()
		continuous_input_columns = [col for col in self._obj.columns if col not in (label_column, prediction_column) \
			and not self.is_discrete(col)] if continuous_input_columns == () else list(continuous_input_columns)
		assert len(continuous_input_columns) > 0, "Continuous inputs are required"

		x_d = self._obj[discrete_input_columns].values if len(discrete_input_columns) > 0 else None
		x_c = self._obj[continuous_input_columns].values
		y = self._obj[label_column].values
		yp = self._obj[prediction_column].values

		return classification_suboptimality(yp, y, x_c, x_d=x_d)



	def individual_input_importance(self, label_column, input_columns=(), problem=None, space='dual'):
		"""
		.. _dataframe-input-importance:
		Calculates the importance of each input in the input set at solving the supervised
		learning problem where the label is defined by the label_column.

		
		.. note::

			Input importance is defined as the mutual information between the input column 
			and the label column. The supervised learning problem can either be specified as :code:`'classification'` or 
			:code:`'regression'` using the :code:`problem` argument, or inferred from the type of, and the number 
			of distinct values in the :code:`label_column`.


		.. seealso::

			* :ref:`kxy.classification.pre_learning.classification_feasibility <classification-feasibility>`
			* :ref:`kxy.classification.pre_learning.regression_feasibility <regression-feasibility>`


		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		input_columns : set, optional
			The set of columns to as inputs. When input_columns is the empty set,
			all columns except for label_column are used as inputs.
		problem : str or None (default), optional
			The type of supervised learning problem. One of None (default), 'classification'
			or 'regression'. When problem is None, the supervised learning problem is inferred
			based on whether labels are numeric and the percentage of distinct labels.

		Returns
		-------
		importance : pd.DataFrame
			A dataframe with an input column, an individual importance column, and a normalized individual importance column.

		Raises
		------
		AssertionError
			If problem is neither :code:`None` nor :code:`'classification'` nor :code:`'regression'`, or if 
			:code:`label_column` is in :code:`input_columns`.
		"""
		input_columns = list(set(self._obj.columns)-set([label_column])) if input_columns == () \
			else list(input_columns)
		assert label_column not in input_columns, "The output cannot be a input"
		assert problem is None or problem in ('classification', 'regression'), \
			"The problem should be either None, 'classification' or 'regression'"

		if problem is None:
			problem = 'classification' if self.is_discrete(label_column) else 'regression'


		importance = {}
		with ThreadPoolExecutor(max_workers=10) as p:
			args = [(col, label_column, problem, space) for col in input_columns]
			for imp in p.map(self.__individual_input_importance, args):
				importance.update(imp)

		total_importance = np.sum([importance[col] for col in importance.keys() if importance[col]])
		scale = 1./total_importance if total_importance > 0. else 0.0

		importance_df = pd.DataFrame({
			'input': [k for k, v in sorted(importance.items(), key=lambda item: -item[1])], \
			'individual_importance': [v for k, v in sorted(importance.items(), key=lambda item: -item[1])], \
			'normalized_individual_importance': [v*scale for k, v in sorted(importance.items(), key=lambda item: -item[1])]})

		importance_df['cum_normalized_individual_importance'] = importance_df['normalized_individual_importance'].cumsum()

		return importance_df


	def __individual_input_importance(self, args):
		col, label_column, problem, space = args
		if problem == 'classification':
			self.adjust_quantized_values()
			return {col: classification_feasibility(\
							None, self._obj[label_column].values, x_d=self._obj[col].values, space=space) if self.is_discrete(col) else \
							   classification_feasibility(self._obj[col].values, self._obj[label_column].values, x_d=None, space=space)}

		else:
			return {col: regression_feasibility(\
				self._obj[col].values, self._obj[label_column].values, space=space) if not self.is_categorical(col) else \
				classification_feasibility(self._obj[label_column].values, self._obj[col].values, x_d=None, space=space)}



	def is_discrete(self, column):
		"""
		Determine whether the input column contains discrete observations.
		"""
		ret = (not np.can_cast(self._obj[column].values, float))
		ret = ret or len(list(set(self._obj[column].values))) < 0.5*self._obj.shape[0]

		return ret


	def is_categorical(self, column):
		"""
		Determine whether the input column is not numeric.
		"""
		return not np.can_cast(self._obj[column].values, float)


	def corr(self, columns=(), method='information-adjusted', min_periods=1, p=0):
		"""
		Calculates the auto-correlation matrix of all columns or the input subset.


		Parameters
		----------
		columns : set, optional
			The set of columns to use. If not provided, all columns are used.
		method : str, optional
			Which method to use to calculate the auto-correlation matrix. Supported
			values are 'information-adjusted' (the default) and all 'method' values of pandas.DataFrame.corr.
		p : int, optional
			The number of lags to use when generating Spearman rank auto-correlation to use 
			as empirical evidence in the maximum-entropy problem. The default value is 0, which 
			corresponds to assuming rows are i.i.d. This is also the only supported value for now.
		min_periods : int, optional
			Only used when method is not 'information-adjusted'. 
			See the documentation of pandas.DataFrame.corr.

		Returns
		-------
		c : pd.DataFrame
			The auto-correlation matrix.


		.. seealso::

			:ref:`kxy.finance.risk_analysis.information_adjusted_correlation <information-adjusted-correlation>`
		"""
		columns = self._obj.columns if columns == () else list(columns)

		if method == 'information-adjusted':
			c = information_adjusted_correlation(self._obj[columns].values, self._obj[columns].values)
			return pd.DataFrame(c, columns=columns, index=columns)
		else:
			return pd.DataFrame.corr(self._obj[columns], method=method, min_periods=min_periods)


	def beta(self, column_y, column_x, method='information-adjusted'):
		"""
		Calculates the information-adjusted beta of a portfolio or asset (whose returns are provided in 
		column_y) with respect to the market (whose returns are provided in column_x).

		.. note::

			The information-adjusted beta coefficient generalizes the traditional (CAPM/OLS/Pearson) beta 
			in that, unlike CAPM beta that only captures linear cross-sectional dependence, the 
			information-adjusted beta captures cross-sectional and temporal dependence, linear and nonlinear.

			The IA-beta is 0 if and only if the portfolio or asset exhibit no dependence with the market, linear
			or nonlinear, cross-sectional or temporal.


		Parameters
		----------
		colummn_y : str
			The name of the column to use for portfolio/asset returns.
		colummn_x : str
			The name of the column to use for market returns.
		method : str, optional
			Either 'information-adjusted' for information-adjusted beta or 'pearson' for the traditional OLS/pearson beta coefficient.

		Returns
		-------
		c : float
			The beta coefficient.

		Raises
		------
		AssertionError
			If the method is neither 'information-adjusted' nor 'pearson'.


		.. seealso::

			:ref:`kxy.finance.factor_analysis.information_adjusted_beta <information-adjusted-beta>`
		"""
		assert method in ('information-adjusted', 'pearson'), "Allowed methods are 'information-adjusted' and 'pearson'."

		if method == 'information-adjusted':
			return information_adjusted_beta(self._obj[column_y].values, self._obj[column_x].values)

		c = np.corrcoef(self._obj[column_y].values, self._obj[column_x].values)[0, 1]
		return c*np.sqrt(self._obj[column_y].values.var()/self._obj[column_x].values.var())


	def total_correlation(self, columns=()):
		"""
		Calculates the total correlation between all columns or the input subset.


		Parameters
		----------
		columns : set, optional
			The set of columns to use. If not provided, all columns are used.

		Returns
		-------
		c : float
			The total correlation.


		.. seealso::

			:ref:`kxy.api.core.mutual_information.least_total_correlation <least-total-correlation>`
		"""
		columns = self._obj.columns if columns == () else list(columns)
		return least_total_correlation(self._obj[columns].values)


	def regression_suboptimality(self, prediction_column, label_column, input_columns=()):
		"""
		Quantifies the extent to which a calibrated regression model can be improved without requiring
		additional inputs.


		.. note::

			The aim of a regression model is to find the function :math:`x \\to f(x) \in \\mathbb{R}` 
			to be used as our predictor for :math:`y` given :math:`x` so that :math:`f(x)` fully captures all
			the information about :math:`y` that is contained in :math:`x`.  For instance, this will be 
			the case when the true generative model takes the form

			.. math::

				y = f(x) + \\epsilon

			with :math:`\\epsilon` statistically independent from :math:`y`, in which case :math:`h \\left( y \\vert x \\right) = h(\\epsilon)`.

			More generally, the conditional entropy :math:`h \\left( y \\vert x \\right)` represents 
			the amount of information about :math:`y` that cannot be explained by :math:`x`, while 
			:math:`h \\left( y \\vert f(x) \\right)` represents the amount of information 
			about :math:`y` that cannot be explained by the regression model 

			.. math::

				y = f(x) + \\epsilon.

			A natural metric for how suboptimal a particular regression model is can therefore be defined as
			the difference between what the amount of information about :math:`y` that cannot be explained by 
			:math:`f(x)` and the amount of information about :math:`y` that cannot be explained by :math:`x`


			.. math::

				\\text{SO}(f; x) &= h \\left( y \\vert f(x) \\right) - h \\left( y \\vert x \\right) \\

				:&= I\\left(y, x \\right) - I\\left(y, f(x) \\right) \\

				 &\\geq 0.

			This regression suboptimality metric is 0 if and only if :math:`f(x)` fully captures any information about :math:`y`
			that is contained in :math:`x`. When 

			.. math::

				\\text{SO}(f; x) > 0 

			on the other hand, there exists a regression model using :math:`x` as inputs that can better predict :math:`y`. The larger 
			:math:`\\text{SO}(f; x)`, the more the regression model is suboptimal and can be improved.


		Parameters
		----------
		prediction_column : str
			The name of the column containing regression predictions.
		label_column : str
			The name of the column containing regression labels.
		input_columns: set
			The set of columns to use as inputs. When not specified, all columns 
			are used except. for the prediction and label columns.

		Returns
		-------
		s : float
			The suboptimality score, defined as the mutual information between residuals and labels.
			The higher the value, the more the model can be improved.


		.. seealso::

			:ref:`kxy.regression.post_learning.regression_suboptimality <regression-suboptimality>`
		"""
		self.adjust_quantized_values()
		assert not self.is_categorical(prediction_column), "The prediction column should not be categorical"
		assert not self.is_categorical(label_column), "The label column should not be categorical"

		input_columns = [_ for _ in self._obj.columns if _ not in (prediction_column, label_column) ] \
			if input_columns == () else list(input_columns)

		# Direct estimation of SO
		result_1 = regression_suboptimality(self._obj[prediction_column].values, self._obj[label_column].values, \
			self._obj[input_columns].values)

		# SO = ASO + h(y) - h(e)
		result_2 = self.regression_additive_suboptimality(prediction_column, label_column, \
			input_columns=input_columns)
		e = (self._obj[prediction_column]-self._obj[label_column]).values
		result_2 += scalar_continuous_entropy(self._obj[label_column].values) - scalar_continuous_entropy(e)

		return max(result_1, result_2)




	def regression_additive_suboptimality(self, prediction_column, label_column, input_columns=()):
		"""
		Quantifies the extent to which a regression model can be improved without requiring additional inputs, by evaluating 
		how informative its residuals still are about the inputs.

		.. note::

			Additive regression models aim a

			t breaking down a label :math:`y` as the sum of a component that solely depend on 
			inputs :math:`f(x)` and a residual component that is statistically independent from inputs :math:`\\epsilon`

			.. math::

				y = f(x) + \\epsilon.

			In an ideal scenario, the regreession residual :math:`\\epsilon` would indeed be stastically independent from the inputs
			:math:`x`. In pratice however, this might not be the case, for instance when the space of candidate functions used by
			the regression model isn't flexible enough (e.g. linear regression or basis functions regression), or the optimization
			has not converged to the global optimum. 

			Any departure from statistical independence between residuals :math:`\\epsilon` and inputs :math:`x` is an indication that what
			:math:`x` can reveal about :math:`y` is not fully captured by :math:`f(x)`, which implies that the regression model can be improved.

			Thus, we define the additive suboptimality of a regression model as the mutual information between its residuals and its inputs

			.. math::

				\\text{ASO}(f; x) := I\\left( y-f(x), x \\right)


		Parameters
		----------
		prediction_column : str
			The name of the column containing regression predictions.
		label_column : str
			The name of the column containing regression labels.
		input_columns: set
			The set of columns to use as inputs. When not specified, all columns 
			are used except. for the prediction and label columns.


		Returns
		-------
		d : float
			The regression's additive suboptimality measure.


		.. seealso::

			:ref:`kxy.regression.post_learning.regression_additive_suboptimality <regression-additive-suboptimality>`
		"""
		assert not self.is_categorical(prediction_column), "The prediction column should not be categorical"
		assert not self.is_categorical(label_column), "The label column should not be categorical"

		input_columns = [_ for _ in self._obj.columns if _ not in (prediction_column, label_column) ] \
			if input_columns == () else list(input_columns)

		e = (self._obj[prediction_column]-self._obj[label_column]).values
		x = self._obj[input_columns].values

		return regression_additive_suboptimality(e, x)




	def _pre_solve(self):
		"""
		Pre-emptively solve the max-entropy problems remotely, and in the background.
		"""
		data = np.hstack((self._obj.values, np.abs(self._obj.values-self._obj.values.mean(axis=0))))
		corr = spearman_corr(data)
		return solve_copula_async(corr)
		
	

	def regression_input_incremental_importance(self, label_column, input_columns=(), space='dual', greedy=True):
		"""
		Quantifies how important each input is at solving a regression problem,
		taking into possible information redundancy between inputs.

		
		.. note::

			The incremental importance of input :math:`x` for predicting label :math:`y` once we already
			know inputs :math:`z` is defined as the conditional mutual information :math:`I(y; x|z)`.

			When greedy=True, we first select the column with the highest mutual information with the label. Then we 
			sequentially select, among all remaining input columns, the one with the highest conditional mutual
			information with the label, conditional on all previously selected inputs, until there is no
			input left to select.

			When greedy=False, inputs are selected in decreasing order of their mutual information with the output.

		.. important::

			This function only supports continuous inputs.


		.. seealso::

			* :ref:`kxy.data.dataframe.DataFrame.individual_input_importance <dataframe-input-importance>`
			* :ref:`kxy.regression.pre_learning.regression_input_incremental_importance <regression-input-incremental-importance>`



		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		input_columns : set, optional
			The set of columns to as inputs. When input_columns is the empty set,
			all columns except for label_column are used as inputs.
		problem : str or None (default), optional
			The type of supervised learning problem. One of None (default), 'classification'
			or 'regression'. When problem is None, the supervised learning problem is inferred
			based on whether labels are numeric and the percentage of distinct labels.

		Returns
		-------
		importance : pd.DataFrame
			A dataframe with an input column, an incremental importance column, a normalized incremental importance column, 
			and a column with the order in which variables were selected.

		Raises
		------
		AssertionError
			If :code:`label_column` is in :code:`input_columns`.
		"""
		input_columns = list(set(self._obj.columns)-set([label_column])) if input_columns == () \
			else list(input_columns)
		continuous_inputs = [col for col in input_columns if not self.is_categorical(col)]

		data = np.hstack((self._obj[label_column].values[:, None], self._obj[continuous_inputs].values, \
			np.abs(self._obj[continuous_inputs].values-np.nanmean(self._obj[continuous_inputs].values, axis=0))))
		corr = pearson_corr(data) if space == 'primal' else spearman_corr(data)
		mi_analysis = mutual_information_analysis(corr, 0, space=space, greedy=greedy)
		columns = [label_column] + continuous_inputs + continuous_inputs

		if mi_analysis is None:
			return {}

		remaining_columns = continuous_inputs

		res = {}
		order = {}
		idx = 1
		for i in range(1, 1+2*len(continuous_inputs)):
			column_id = mi_analysis['selection_order'][str(i)]
			column = columns[column_id]
			if column in remaining_columns:
				res[column] = mi_analysis['conditional_mutual_informations'][str(i)]
				order[column] = idx
				idx += 1
				remaining_columns.remove(column)

		# Normalize and format as a dataframe.
		total_importance = np.sum([res[col] for col in res.keys() if res[col]])
		scale = 1./total_importance if total_importance > 0. else 0.0

		importance_df = pd.DataFrame({
			'input': [k for k, v in sorted(order.items(), key=lambda item: item[1])], \
			'selection_order': [v for k, v in sorted(order.items(), key=lambda item: item[1])], \
			'incremental_importance': [res[k] for k, v in sorted(order.items(), key=lambda item: item[1])], \
			'normalized_incremental_importance': [res[k]*scale for k, v in sorted(order.items(), key=lambda item: item[1])]})
		importance_df['cum_normalized_incremental_importance'] = importance_df['normalized_incremental_importance'].cumsum()

		return importance_df



	def classification_input_incremental_importance(self, label_column, input_columns=(), space='dual'):
		"""
		Quantifies how important each input is at solving a classification problem,
		taking into possible information redundancy between inputs.

		
		.. note::

			The incremental importance of input :math:`x` for predicting label :math:`y` once we already
			know inputs :math:`z` is defined as the conditional mutual information :math:`I(y; x|z)`.

			We first select the column with the highest mutual information with the label. Then we 
			sequentially select, among all remaining input columns, the one with the highest conditional mutual
			information with the label, conditional on all previously selected inputs, until there is no
			input left to select.


		.. seealso::

			* :ref:`kxy.data.dataframe.DataFrame.individual_input_importance <dataframe-input-importance>`
			* :ref:`kxy.classification.pre_learning.classification_input_incremental_importance <classification-input-incremental-importance>`



		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		input_columns : set, optional
			The set of columns to as inputs. When input_columns is the empty set,
			all columns except for label_column are used as inputs.


		Returns
		-------
		importance : DataFrame
			A dataframe with an input column, an incremental importance column, a normalized incremental importance column, 
			and a column with the order in which variables were selected.
		Raises
		------
		AssertionError
			If :code:`label_column` is in :code:`input_columns`.
		"""
		self.adjust_quantized_values()
		input_columns = list(set(self._obj.columns)-set([label_column])) if input_columns == () \
			else list(input_columns)

		res = {}
		order = {} # Order in which inputs where selected
		categorical_conditions = []
		continuous_conditions = []

		while len(input_columns) > 0:
			importance = {}
			with ThreadPoolExecutor(max_workers=10) as p:
				args = [(col, label_column, continuous_conditions, categorical_conditions, space) for col in input_columns]
				for imp in p.map(self.__classification_input_incremental_importance, args):
					importance.update(imp)

			for key, value in sorted(importance.items(), key=lambda x: -x[1]):
				res[key] = value
				input_columns.remove(key)
				if self.is_discrete(key):
					categorical_conditions += [key]
				else:
					continuous_conditions += [key]
				order[key] = len(categorical_conditions) + len(continuous_conditions)
				logging.info('Selected column %s as the %s most important input, with incremental importance %.6f' % \
					(key, '1st' if order[key] == 1 else '2nd' if order[key] == 2 else '3rd' \
						if order[key] == 3 else '%dth' % order[key], res[key]))
				break
		
		# Step 3: Normalize and format as a dataframe.
		total_importance = np.sum([res[col] for col in res.keys() if res[col]])
		scale = 1./total_importance if total_importance > 0. else 0.0

		importance_df = pd.DataFrame({
			'input': [k for k, v in sorted(order.items(), key=lambda item: item[1])], \
			'selection_order': [v for k, v in sorted(order.items(), key=lambda item: item[1])], \
			'incremental_importance': [res[k] for k, v in sorted(order.items(), key=lambda item: item[1])], \
			'normalized_incremental_importance': [res[k]*scale for k, v in sorted(order.items(), key=lambda item: item[1])]})
		importance_df['cum_normalized_incremental_importance'] = importance_df['normalized_incremental_importance'].cumsum()

		return importance_df



	def __classification_input_incremental_importance(self, args):
		col, label_column, continuous_conditions, categorical_conditions, space = args

		if len(continuous_conditions) == 0 and len(categorical_conditions) == 0:
			return {col: classification_feasibility(\
						None, self._obj[label_column].values, x_d=self._obj[col].values, space=space) if self.is_discrete(col) else \
						   classification_feasibility(self._obj[col].values, self._obj[label_column].values, x_d=None, space=space)}

		return {col: \
				classification_input_incremental_importance(\
					None, self._obj[label_column].values, None if len(continuous_conditions) == 0 \
					else self._obj[continuous_conditions].values, x_d=self._obj[col].values, \
					z_d=None if len(categorical_conditions) == 0 else self._obj[categorical_conditions].values, \
					space=space) if self.is_discrete(col) else \
				classification_input_incremental_importance(\
					self._obj[col].values, self._obj[label_column].values, None if len(continuous_conditions) == 0 \
					else self._obj[continuous_conditions].values, x_d=None, \
					z_d=None if len(categorical_conditions) == 0 else self._obj[categorical_conditions].values, \
					space=space)}



	def incremental_input_importance(self, label_column, input_columns=(), space='dual', greedy=True):
		"""
		Returns :code:`DataFrame.classification_input_incremental_importance` or 
		:code:`DataFrame.regression_input_incremental_importance` depending on whether the label 
		is categorical or continuous
		"""
		problem = 'classification' if self.is_discrete(label_column) else 'regression'

		if problem == 'classification':
			self.adjust_quantized_values()
			return self.classification_input_incremental_importance(label_column, input_columns=input_columns, \
				space=space)

		else:
			return self.regression_input_incremental_importance(label_column, input_columns=input_columns, \
				space=space, greedy=greedy)



	def __hash__(self):
		return hash(self._obj.to_string())



