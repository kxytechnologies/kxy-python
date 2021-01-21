#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from kxy.api.core import auto_predictability

from kxy.classification import classification_achievable_performance_analysis, \
	classification_variable_selection_analysis

from kxy.regression import regression_achievable_performance_analysis, \
	regression_variable_selection_analysis

from .base_accessor import BaseAccessor


@pd.api.extensions.register_dataframe_accessor("kxy_pre_learning")
class PreLearningAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics **pre-learning** in supervised learning problems.

	This class defines the :code:`kxy_pre_learning` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_pre_learning.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""
	def achievable_performance_analysis(self, label_column, input_columns=(), space='dual', categorical_encoding='two-split', problem=None):
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
		categorical_encoding : str, 'one-hot' | 'two-split' (default)
			The encoding method to use to represent categorical variables. 
			See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.
		problem : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.



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
		if problem is None:
			problem = 'classification' if self.is_discrete(label_column) else 'regression'
		columns = [col for col in self._obj.columns if col != label_column] if len(input_columns) == 0\
			else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		y = self._obj[label_column].values
		x_c = self._obj[continuous_columns].values if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values if len(discrete_columns) > 0 else None

		res = regression_achievable_performance_analysis(x_c, y, x_d=x_d, space=space, categorical_encoding=categorical_encoding) \
			if problem == 'regression' else classification_achievable_performance_analysis(x_c, y, x_d=x_d, space=space, \
				categorical_encoding=categorical_encoding)
		res = res.style.hide_index()

		return res


	def variable_selection_analysis(self, label_column, input_columns=(), space='dual', categorical_encoding='two-split', problem=None):
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
		categorical_encoding : str, 'one-hot' | 'two-split' (default)
			The encoding method to use to represent categorical variables. 
			See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.
		problem : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.

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
				* :code:`'Conditional Mutual Information (nats)'`: The mutual information between this variable and the label, conditional on all variables previously selected.
				* :code:`'Running Mutual Information (nats)'`: The mutual information between all variables selected so far, including this one, and the label.
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
		if problem is None:
			problem = 'classification' if self.is_discrete(label_column) else 'regression'

		columns = [col for col in self._obj.columns if col != label_column] if len(input_columns) == 0\
			else input_columns
		discrete_columns = [col for col in columns if self.is_categorical(col)]
		continuous_columns =  [col for col in columns if not self.is_categorical(col)]

		y = self._obj[label_column].values
		x_c = self._obj[continuous_columns].values.astype(float) if len(continuous_columns) > 0 else None
		x_d = self._obj[discrete_columns].values.astype(str) if len(discrete_columns) > 0 else None

		res = regression_variable_selection_analysis(x_c, y, x_d=x_d, space=space, categorical_encoding=categorical_encoding) \
			if problem == 'regression' else classification_variable_selection_analysis(x_c, y, x_d=x_d, space=space, \
				categorical_encoding=categorical_encoding)

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




	def dataset_valuation(self, label_column, existing_input_columns, new_input_columns, space="dual", categorical_encoding="two-split", problem=None):
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
		categorical_encoding : str, 'one-hot' | 'two-split' (default)
			The encoding method to use to represent categorical variables. 
			See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.
		problem : None | 'classification' | 'regression'
			The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.

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
			return self.achievable_performance_analysis(label_column, input_columns=new_input_columns, \
				categorical_encoding=categorical_encoding)

		if new_input_columns is None or len(new_input_columns) == 0:
			if problem is None:
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
		new_perf = self.achievable_performance_analysis(label_column, input_columns=all_inputs, space=space, \
			categorical_encoding=categorical_encoding)
		old_perf = self.achievable_performance_analysis(label_column, input_columns=existing_input_columns, space=space, \
			categorical_encoding=categorical_encoding)
		imp_perf = new_perf-old_perf
		imp_perf.rename(columns={col: col.replace('Achievable', 'Increased Achievable') for col in imp_perf.columns}, inplace=True)

		try:
			import seaborn as sns
			cm = sns.light_palette("green", as_cmap=True)
			imp_perf = imp_perf.style.background_gradient(cmap=cm)
		except:
			imp_perf = imp_perf.style.background_gradient()		

		return imp_perf
