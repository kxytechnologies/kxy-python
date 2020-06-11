#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
We define a custom :code:`kxy` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_ below, 
namely the class :code:`KXYAccessor`, that extends the pandas DataFrame class with all our analyses, thereby allowing data scientists to tap into 
the power of the :code:`kxy` toolkit within the comfort of their favorite data structure.

All methods defined in the :code:`KXYAccessor` class are accessible from any DataFrame instance as :code:`df.kxy.<method_name>`, so long as the :code:`kxy` python 
package is imported alongside :code:`pandas`. 
"""

from functools import lru_cache, wraps
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import seaborn as sns

from kxy.asset_management import information_adjusted_beta, information_adjusted_correlation, \
	robust_pearson_corr

from kxy.api.core import spearman_corr, pearson_corr, auto_predictability

from kxy.classification import classification_achievable_performance_analysis, \
	classification_variable_selection_analysis, classification_model_improvability_analysis, \
	classification_model_explanation_analysis, classification_bias

from kxy.regression import regression_achievable_performance_analysis, regression_variable_selection_analysis, \
	regression_model_improvability_analysis, regression_model_explanation_analysis, regression_bias



@pd.api.extensions.register_dataframe_accessor("kxy")
class KXYAccessor(object):
	"""
	Extension of the pandas.DataFrame class with various analytics for **pre-learning** and **post-learning**,
	in supervised learning problems.
	"""
	def __init__(self, pandas_obj):
		self._obj = pandas_obj


	def is_discrete(self, column):
		"""
		Determine whether the input column contains discrete observations.
		"""
		ret = (not np.can_cast(self._obj[column].values, float))
		ret = ret or len(list(set(self._obj[column].values))) < 0.5*self._obj.shape[0]

		return ret


	def is_categorical(self, column):
		"""
		Determine whether the input column contains categorical observations.
		"""
		ret = (not np.can_cast(self._obj[column].values, float))

		return ret


	def corr(self, columns=(), method='information-adjusted', min_periods=1, p=0, p_ic='hqic'):
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
			The number of auto-correlation lags to use as empirical evidence in the maximum-entropy problem. 
			The default value is 0, which corresponds to assuming rows are i.i.d. Values other than 0 are only
			supported in the robust-pearson method. When p is None, it is inferred from the sample.
		min_periods : int, optional
			Only used when method is not 'information-adjusted'. 
			See the documentation of pandas.DataFrame.corr.
		p_ic : str
			The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`. Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion), 'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). Same as the 'ic' parameter of :code:`statsmodels.tsa.api.VAR`.



		Returns
		-------
		c : pandas.DataFrame
			The auto-correlation matrix.


		.. seealso::

			:ref:`kxy.finance.risk_analysis.information_adjusted_correlation <information-adjusted-correlation>`
			:ref:`kxy.finance.risk_analysis.robust_pearson_corr <robust-pearson-corr>`
		"""
		columns = self._obj.columns if columns == () else list(columns)

		if method == 'information-adjusted':
			c = information_adjusted_correlation(self._obj[columns].values, y=None)
			return pd.DataFrame(c, columns=columns, index=columns)

		if method == 'robust-pearson':
			c = robust_pearson_corr(self._obj[columns].values, y=None, p=p, p_ic=p_ic)
			return pd.DataFrame(c, columns=columns, index=columns)

		if method == 'spearman':
			c = spearman_corr(self._obj[columns].values)
			return pd.DataFrame(c, columns=columns, index=columns)

		if method == 'pearson':
			c = pearson_corr(self._obj[columns].values)
			return pd.DataFrame(c, columns=columns, index=columns)

		else:
			return pd.DataFrame.corr(self._obj[columns], method=method, min_periods=min_periods)



	def beta(self, market_returns_column, asset_returns_columns=(), risk_free_column=None,\
			method='information-adjusted', p=0, p_ic='hqic'):
		"""
		Calculates the beta of a portfolio/asset (whose returns are provided in column_y) 
		with respect to the market (whose returns are provided in market_returns_column) using a variety
		of estimation methods including the standard OLS/Pearson methods and information theoretical 
		alternatives aiming at accounting for nonlinearities and memory in asset returns.


		Parameters
		----------
		asset_returns_columns : str or list of str
			The name(s) of the column(s) to use for portfolio/asset returns.
		market_returns_column : str
			The name of the column to use for market returns.
		method : str, optional
			One of 'information-adjusted', 'robust-pearson', 'spearman',  or 'pearson'. This is the method to use
			to estimate the correlation between portfolio/asset returns and market returns.
		p : int, optional
			The number of auto-correlation lags to use as empirical evidence in the maximum-entropy problem. 
			The default value is 0, which corresponds to assuming rows are i.i.d. Values other than 0 are only
			supported in the robust-pearson method. When p is None, it is inferred from the sample.
		p_ic : str
			The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`.
			Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion),
			'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). Same as the 'ic' parameter of 
			:code:`statsmodels.tsa.api.VAR`.


		Returns
		-------
		c : pandas.DataFrame
			The beta coefficient(s).


		.. seealso::

			:ref:`kxy.finance.factor_analysis.information_adjusted_beta <information-adjusted-beta>`
		"""
		asset_returns_columns = [_ for _ in self._obj.columns if _ != market_returns_column] if asset_returns_columns == () \
			else [asset_returns_columns] if type(asset_returns_columns) == str else list(asset_returns_columns)
		columns = [market_returns_column] + asset_returns_columns

		c = self.corr(method=method, columns=columns, p=p, p_ic=p_ic).values[0, 1:]
		betas = c * np.sqrt(np.nanvar(self._obj[asset_returns_columns].values, axis=0)/\
			np.nanvar(self._obj[market_returns_column].values))

		if type(asset_returns_columns) == str:
			res = pd.DataFrame({asset_returns_columns: betas}).T.rename(columns={0: 'beta'})
		else:
			res = pd.DataFrame({asset_returns_columns[i]: [betas[i]] \
				for i in range(len(asset_returns_columns))}).T.rename(columns={0: 'beta'})

		return res



	def achievable_performance_analysis(self, label_column, input_columns=(), space='dual'):
		"""
		Runs the achievable performance analysis on a trained supervised learning model.

		The nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`label_column` is categorical.


		Parameters
		----------
		label_column : str
			The name of the column containing true labels.
		input_columns : set
			List of columns to use as inputs. When an empty set/list is provided, all columns but :code:`label_column` are used as inputs.
		space : str, 'primal' | 'dual'
			The space in which the maximum entropy problem is solved. 
			When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
			When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.



		Returns
		-------
		res : pandas.Styler
			res.data is a pandas.Dataframe with columns (where applicable):

			* :code:`'Achievable R^2'`: The highest :math:`R^2` that can be achieved by a model using provided inputs to predict the label.
			* :code:`'Achievable Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a model using provided inputs to predict the label.
			* :code:`'Achievable Accuracy'`: The highest classification accuracy that can be achieved by a model using provided inputs to predict the label.



		.. admonition:: Theoretical Foundation

			Section :ref:`1 - Achievable Performance`.

		.. seealso::

			* :ref:`kxy.regression.regression_achievable_performance_analysis <regression-achievable-performance-analysis>`
			* :ref:`kxy.classification.classification_achievable_performance_analysis <classification-achievable-performance-analysis>`
		"""
		problem = 'classification' if self.is_discrete(label_column) else 'regression'
		columns = [col for col in self._obj.columns if col != label_column] if len(input_columns) == 0\
			else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		y = self._obj[label_column].values
		x_c = self._obj[continuous_columns].values if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values if len(discrete_columns) > 0 else None

		res = regression_achievable_performance_analysis(x_c, y, x_d=x_d, space=space) if problem == 'regression' \
			else classification_achievable_performance_analysis(x_c, y, x_d=x_d, space=space)

		return res


	def variable_selection_analysis(self, label_column, input_columns=(), space='dual'):
		"""
		Runs the model variable selection analysis on a trained supervised learning model.

		The nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`label_column` is categorical.


		Parameters
		----------
		label_column : str
			The name of the column containing true labels.
		input_columns : set
			List of columns to use as inputs. When an empty set/list is provided, all columns but :code:`label_column` are used as inputs.
		space : str, 'primal' | 'dual'
			The space in which the maximum entropy problem is solved. 
			When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
			When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


		Returns
		-------
		res : pandas.Styler
			res.data is a pandas.Dataframe with columns (where applicable):

				* :code:`'Variable'`: The column name corresponding to the input variable.
				* :code:`'Selection Order'`: The order in which the associated variable was selected, starting at 1 for the most important variable.
				* :code:`'Univariate Achievable R^2'`: The highest :math:`R^2` that can be achieved by a classification model solely using this variable.
				* :code:`'Maximum Marginal R^2 Increase'`: The highest amount by which the :math:`R^2` can be increased as a result of adding this variable in the variable selection scheme.
				* :code:`'Running Achievable R^2'`: The highest :math:`R^2` that can be achieved by a classification model using all variables selected so far, including this one.
				* :code:`'Univariate Achievable Accuracy'`: The highest classification accuracy that can be achieved by a classification model solely using this variable.
				* :code:`'Maximum Marginal Accuracy Increase'`: The highest amount by which the classification accuracy can be increased as a result of adding this variable in the variable selection scheme.
				* :code:`'Running Achievable Accuracy'`: The highest classification accuracy that can be achieved by a classification model using all variables selected so far, including this one.
				* :code:`'Conditional Mutual Information'`: The mutual information between this variable and the label, conditional on all variables previously selected.
				* :code:`'Running Mutual Information'`: The mutual information between all variables selected so far, including this one, and the label.
				* :code:`'Univariate Achievable True Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a classification model solely using this variable.
				* :code:`'Maximum Marginal True Log-Likelihood Per Sample Increase'`: The highest amount by which the true log-likelihood per sample can be increased as a result of adding this variable in the variable selection scheme.
				* :code:`'Running Achievable True Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a classification model using all variables selected so far, including this one.
				* :code:`'Maximum Marginal True Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase as a result of adding this variable.
				* :code:`'Running Maximum Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase (over the log-likelihood of the naive strategy consisting of predicting the mode of :math:`y`) as a result of using all variables selected so far, including this one.


		.. admonition:: Theoretical Foundation

			Section :ref:`2 - Variable Selection Analysis`.


		.. seealso::

			* :ref:`kxy.regression.regression_variable_selection_analysis <regression-variable-selection-analysis>`
			* :ref:`kxy.classification.classification_variable_selection_analysis <classification-variable-selection-analysis>`
		"""
		problem = 'classification' if self.is_discrete(label_column) else 'regression'
		columns = [col for col in self._obj.columns if col != label_column] if len(input_columns) == 0\
			else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		y = self._obj[label_column].values
		x_c = self._obj[continuous_columns].values.astype(float) if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values.astype(str) if len(discrete_columns) > 0 else None

		res = regression_variable_selection_analysis(x_c, y, x_d=x_d, space=space) if problem == 'regression' \
			else classification_variable_selection_analysis(x_c, y, x_d=x_d, space=space)

		variable_columns = continuous_columns + discrete_columns
		res['Variable'] = res['Variable'].map({i: variable_columns[i] for i in range(len(variable_columns))})
		res.set_index(['Variable'], inplace=True)

		cm = sns.light_palette("green", as_cmap=True)
		res = res.style.background_gradient(cmap=cm)

		return res


	def model_improvability_analysis(self, label_column, model_prediction_column, input_columns=(), 
			space='dual'):
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
		problem = 'classification' if self.is_discrete(label_column) else 'regression'
		columns = [col for col in self._obj.columns if col != label_column and col != model_prediction_column] \
			if len(input_columns) == 0 else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		y = self._obj[label_column].values
		y_p = self._obj[model_prediction_column].values
		x_c = self._obj[continuous_columns].values.astype(float) if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values.astype(str) if len(discrete_columns) > 0 else None

		res = regression_model_improvability_analysis(x_c, y_p, y, x_d=x_d, space=space) if problem == 'regression' \
			else classification_model_improvability_analysis(x_c, y_p, y, x_d=x_d, space=space)

		return res



	def model_explanation_analysis(self, model_prediction_column, input_columns=(), space='dual'):
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
		problem = 'classification' if self.is_discrete(model_prediction_column) else 'regression'
		columns = [col for col in self._obj.columns if col != model_prediction_column] if len(input_columns) == 0\
			else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		f_x = self._obj[model_prediction_column].values
		x_c = self._obj[continuous_columns].values.astype(float) if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values.astype(str) if len(discrete_columns) > 0 else None

		res = regression_model_explanation_analysis(x_c, f_x, x_d=x_d, space=space) if problem == 'regression' \
			else classification_model_explanation_analysis(x_c, f_x, x_d=x_d, space=space)

		variable_columns = continuous_columns + discrete_columns
		res['Variable'] = res['Variable'].map({i: variable_columns[i] for i in range(len(variable_columns))})
		res.set_index(['Variable'], inplace=True)

		cm = sns.light_palette("green", as_cmap=True)
		res = res.style.background_gradient(cmap=cm)

		return res



	def bias(self, bias_source_column, model_prediction_column, linear_scale=True):
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
		problem = 'classification' if self.is_discrete(model_prediction_column) else 'regression'

		f_x = self._obj[model_prediction_column].values
		z = self._obj[bias_source_column].values

		return regression_bias(f_x, z, linear_scale=linear_scale) if problem == 'regression' \
			else classification_bias(f_x, z, linear_scale=linear_scale)



	def auto_predictability(self, columns=(), space='primal', robust=True, p=None):
		"""
		Estimates the measure of auto-predictability of the time series corresponding to the input columns.

		.. math::

			\\mathbb{PR}\\left(\\{x_t \\} \\right)		:&= h\\left(x_* \\right) - h\\left( \\{ x_t \\} \\right) \\

														 &= h\\left(u_{x_*}\\right) - h\\left( \\{ u_{x_t} \\} \\right).


		Parameters
		----------
		columns : list
			The input columns.
		space : str, 'primal' | 'dual'
			The space in which the maximum entropy problem is solved. 
			When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
			When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
		p : int, optional
			The number of auto-correlation lags to use as empirical evidence in the maximum-entropy problem. 
			When p is None (the default), it is inferred from the sample.
		p_ic : str
			The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`. Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion), 'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). Same as the 'ic' parameter of :code:`statsmodels.tsa.api.VAR`.


		Returns
		-------
		 : float
			The measure of auto-predictability in nats/period.


		.. note::
			The time series is assumed to have a fixed period, and the index is assumed to be sorted.


		.. seealso::

			* :ref:`kxy.api.core.entropy_rate.auto_predictability <auto-predictability>`
		"""
		if columns == ():
			columns = self._obj.columns
		aps = [(col, auto_predictability(self._obj[col].values, space=space, robust=robust, p=p)) for col in columns]
		aps = sorted(aps, key=lambda x: x[1])
		res = pd.DataFrame({_[0]: [_[1]] for _ in aps})

		return res.T.rename(columns={0: 'PR'})




	def dataset_valuation(self, label_column, existing_input_columns, new_input_columns, space="dual"):
		"""
		Quantifies the additional performance that can be brought about by adding a set of new variables, as the difference between achievable performance with and without the new dataset.


		Parameters
		----------
		label_column : str
			The name of the column containing the true labels of the supervised learning problem.
		existing_input_columns : set
			List of columns corresponding to existing inputs/variables.
		new_input_columns : set
			List of columns corresponding to the new dataset.
		space : str, 'primal' | 'dual'
			The space in which the maximum entropy problem is solved. 
			When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
			When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


		Returns
		-------
		res : pandas.Styler
			res.data is a Dataframe with columns (where applicable):

			* :code:`'Increased Achievable R^2'`: The highest :math:`R^2` increase that can result from adding the new dataset.
			* :code:`'Increased Achievable Log-Likelihood Per Sample'`: The highest increase in the true log-likelihood per sample that can result from adding the new dataset.
			* :code:`'Increased Achievable Accuracy'`: The highest increase in the classification accuracy that can result from adding the new dataset.


		.. admonition:: Theoretical Foundation

			Section :ref:`1 - Achievable Performance`.

		.. seealso::

			* :ref:`kxy.regression.regression_achievable_performance_analysis <regression-achievable-performance-analysis>`
			* :ref:`kxy.classification.classification_achievable_performance_analysis <classification-achievable-performance-analysis>`
		"""
		if existing_input_columns is None or len(existing_input_columns) == 0:
			return self.achievable_performance_analysis(label_column, input_columns=new_input_columns)

		if new_input_columns is None or len(new_input_columns) == 0:
			problem = 'classification' if self.is_discrete(label_column) else 'regression'
			if problem == 'classification':
				probas = [1.*(self._obj[self._obj[label_column]==cat].shape[0])/self._obj.shape[0] for cat in list(set(list(self._obj[label_column].values)))]
				return pd.DataFrame({\
					'Achievable R^2': [0.0], \
					'Achievable Log-Likelihood Per Sample': [0.0], \
					'Achievable Accuracy': [np.max(probas)]})

			else:
				return pd.DataFrame({\
					'Achievable R^2': [0.0], \
					'Achievable Log-Likelihood Per Sample': [0.0]})	

		all_inputs = set(list(existing_input_columns)+list(new_input_columns))
		new_perf = self.achievable_performance_analysis(label_column, input_columns=all_inputs, space=space)
		old_perf = self.achievable_performance_analysis(label_column, input_columns=existing_input_columns, space=space)
		imp_perf = new_perf-old_perf
		imp_perf.rename(columns={col: col.replace('Achievable', 'Increased Achievable') for col in imp_perf.columns}, inplace=True)

		cm = sns.light_palette("green", as_cmap=True)
		imp_perf = imp_perf.style.background_gradient(cmap=cm)

		return imp_perf





	def __hash__(self):
		return hash(self._obj.to_string())



