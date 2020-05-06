#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extension of pandas DataFrame class to allow data scientists to tap into 
the power of the KXY API within the comfort of their favorite data structure.
"""

from functools import lru_cache, wraps
from threading import Thread

import numpy as np
import pandas as pd

from kxy.api.core import least_total_correlation, spearman_corr
from kxy.api import solve_copula_async
from kxy.classification import classification_feasibility, classification_suboptimality
from kxy.finance import information_adjusted_beta, information_adjusted_correlation
from kxy.regression import regression_feasibility, regression_suboptimality, regression_additive_suboptimality

from .decorators import decorate_methods, decorate_all_methods


def cast_to_kxy_dataframe(method):
	"""
	Decorator casting the return of a method to kxy.DataFrame when it returns pd.DataFrame.
	"""
	@wraps(method)
	def wrapper(*args, **kwargs):
		res = method(*args, **kwargs)
		if isinstance(res, pd.DataFrame):
			return DataFrame(data=res.values, columns=res.columns, index=res.index)
		return res

	return wrapper


@decorate_all_methods(cast_to_kxy_dataframe)
class _iLocIndexer(pd.core.indexing._iLocIndexer):
	pass

@decorate_all_methods(cast_to_kxy_dataframe)
class _LocIndexer(pd.core.indexing._LocIndexer):
	pass

@decorate_all_methods(cast_to_kxy_dataframe)
class _iAtIndexer(pd.core.indexing._iAtIndexer):
	pass

@decorate_all_methods(cast_to_kxy_dataframe)
class _AtIndexer(pd.core.indexing._AtIndexer):
	pass

@decorate_methods(cast_to_kxy_dataframe, include=[\
	'__add__', '__sub__', '__mul__', '__div__', '__floordiv__', '__truediv__', '__mod__', '__pow__', \
	'__iadd__', '__isub__', '__imul__', '__idiv__', '__ifloordiv__', '__imod__', '__ipow__', \
	'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow', 'sample', \
	'radd', 'rsub', 'rmul', 'rdiv', 'rtruediv', 'rfloordiv', 'rmod', 'rpow', \
	'__eq__', '__ne__', '__gt__', '__lt__', '__le__', '__ge__', '__neg__', '__pos__',  '__invert__', \
	'eq', 'ne', 'gt', 'lt', 'le', 'ge', 'apply', 'applymap', 'abs', 'clip', 'count', \
	'cov', 'cummin', 'cummax', 'cumprod', 'cumsum', 'round', 'sum', 'std', 'var', 'cov', \
	'nunique', 'rename', 'rename_axis', 'reset_index', 'drop_level', 'pivot', 'T', 'rank', \
	'transpose', 'fillna', 'dropna', 'pct_change', 'swapaxes', 'swaplevel', 'where', 'tail', \
	'head', 'shift', 'asfreq', 'dot', 'transform', 'astype', 'copy'])
class DataFrame(pd.DataFrame):
	"""
	Extension of the pandas.DataFrame class with various analytics for **pre-learning** and **post-learning**,
	in supervised learning problems.

	.. important::

		A wide range of operators and methods inherited from pandas.DataFrame have been overriden to return a kxy.DataFrame
		instead of a pandas.DataFrame.
	"""
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

			:ref:`kxy.regression.pre_leanring.regression-feasibility <regression-feasibility>`.


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
		input_columns = list(set(self.columns)-set([label_column])) if input_columns == () \
			else list(input_columns)
		assert label_column not in input_columns, "The output cannot be a input"

		input_importance = self.input_importance(label_column, input_columns=input_columns,\
			problem='regression')
		input_sorted_by_importance = input_importance['input'].values

		return regression_feasibility(self[input_sorted_by_importance].values, self[label_column].values)


	def classification_feasibility(self, label_column, discrete_input_columns=(), \
			continuous_input_columns=()):
		"""
		Quantifies how feasible a classification problem is by computing the amount of uncertainty
		about the label that can be reduced by knowing the inputs, in a model-free fashion.

		.. math::
			FE(x; y) &= h(y)-h\\left(y \\vert x \\right) \\

									 &= I(y, x)

		.. seealso::

			:ref:`kxy.classification.pre_leanring.classification_feasibility <classification-feasibility>`.

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
		assert label_column not in discrete_input_columns, "The output cannot be a input"
		assert label_column not in continuous_input_columns, "The output cannot be a input"

		continuous_input_columns = [col for col in self.columns if col != label_column and not self.is_discrete(col)] \
			if continuous_input_columns == () else list(continuous_input_columns)
		assert len(continuous_input_columns) > 0, "Continuous inputs are required"

		x_d = self[discrete_input_columns].values if len(discrete_input_columns) > 0 else None
		x_c = self[continuous_input_columns].values
		y = self[label_column].values

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

		continuous_input_columns = [col for col in self.columns if col not in (label_column, prediction_column) \
			and not self.is_discrete(col)] if continuous_input_columns == () else list(continuous_input_columns)
		assert len(continuous_input_columns) > 0, "Continuous inputs are required"

		x_d = self[discrete_input_columns].values if len(discrete_input_columns) > 0 else None
		x_c = self[continuous_input_columns].values
		y = self[label_column].values
		yp = self[prediction_column].values

		return classification_suboptimality(yp, y, x_c, x_d=x_d)



	def input_importance(self, label_column, input_columns=(), problem=None):
		"""
		Calculates the importance of each input in the input set at solving the supervised
		learning problem where the label is defined by the label_column.

		
		.. note::

			Input importance is defined as the mutual information between the input column 
			and the label column. The supervised learning problem can either be specified as :code:`'classification'` or 
			:code:`'regression'` using the :code:`problem` argument, or inferred from the type of, and the number 
			of distinct values in the :code:`label_column`.


		.. seealso::

			* :ref:`kxy.classification.pre_leanring.classification_feasibility <classification-feasibility>`
			* :ref:`kxy.classification.pre_leanring.regression_feasibility <regression-feasibility>`


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
		importance : DataFrame
			A dataframe with as input column, an importance column, and a normalized importance column.

		Raises
		------
		AssertionError
			If problem is neither :code:`None` nor :code:`'classification'` nor :code:`'regression'`, or if 
			:code:`label_column` is in :code:`input_columns`.
		"""
		input_columns = list(set(self.columns)-set([label_column])) if input_columns == () \
			else list(input_columns)
		assert label_column not in input_columns, "The output cannot be a input"
		assert problem is None or problem in ('classification', 'regression'), \
			"The problem should be either None, 'classification' or 'regression'"

		if problem is None:
			problem = 'classification' if self.is_discrete(label_column) else 'regression'

		if problem == 'classification':
			importance = {col: classification_feasibility(\
							None, self[label_column].values, x_d=self[col].values) if self.is_discrete(col) else \
							   classification_feasibility(self[col].values, self[label_column].values, x_d=None) \
							for col in input_columns}

		else:
			importance = {col: regression_feasibility(\
				self[col].values, self[label_column].values) if not self.is_categorical(col) else \
				classification_feasibility(self[label_column].values, self[col].values, x_d=None) \
				for col in input_columns}

		total_importance = np.sum([importance[col] for col in importance.keys() if importance[col]])
		scale = 1./total_importance if total_importance > 0. else 0.0

		importance_df = DataFrame({
			'input': [k for k, v in sorted(importance.items(), key=lambda item: -item[1])], \
			'importance': [v for k, v in sorted(importance.items(), key=lambda item: -item[1])], \
			'normalized_importance': [v*scale for k, v in sorted(importance.items(), key=lambda item: -item[1])]})

		return importance_df


	def is_discrete(self, column):
		"""
		Determine whether the input column contains discrete observations.
		"""
		ret = (not np.can_cast(self[column].values, float))
		ret = ret or len(list(set(self[column].values))) < 0.5*self.shape[0]

		return ret


	def is_categorical(self, column):
		"""
		Determinee whether the input column is not numeric.
		"""
		return not np.can_cast(self[column].values, float)


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
		c : DataFrame
			The auto-correlation matrix.


		.. seealso::

			:ref:`kxy.finance.risk_analysis.information_adjusted_correlation <information-adjusted-correlation>`
		"""
		columns = self.columns if columns == () else list(columns)

		if method == 'information-adjusted':
			c = information_adjusted_correlation(self[columns].values, self[columns].values)
			return pd.DataFrame(c, columns=columns, index=columns)
		else:
			return pd.DataFrame.corr(self[columns], method=method, min_periods=min_periods)


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
			return information_adjusted_beta(self[column_y].values, self[column_x].values)

		c = np.corrcoef(self[column_y].values, self[column_x].values)[0, 1]
		return c*np.sqrt(self[column_y].values.var()/self[column_x].values.var())


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
		columns = self.columns if columns == () else list(columns)
		return least_total_correlation(self[columns].values)


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
		assert not self.is_categorical(prediction_column), "The prediction column should not be categorical"
		assert not self.is_categorical(label_column), "The label column should not be categorical"

		input_columns = [_ for _ in self.columns if _ not in (prediction_column, label_column) ] \
			if input_columns == () else list(input_columns)

		return regression_suboptimality(self[prediction_column].values, self[label_column].values, \
			self[input_columns].values)




	def regression_additive_suboptimality(self, prediction_column, label_column, input_columns=()):
		"""
		Quantifies the extent to which a regression model can be improved without requiring additional inputs, by evaluating 
		how informative its residuals still are about the inputs.

		.. note::

			Additive regression models aim at breaking down a label :math:`y` as the sum of a component that solely depend on 
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

		input_columns = [_ for _ in self.columns if _ not in (prediction_column, label_column) ] \
			if input_columns == () else list(input_columns)

		e = (self[prediction_column]-self[label_column]).values
		x = self[input_columns].values

		return regression_additive_suboptimality(e, x)


	def __hash__(self):
		return hash(self.to_string())


	@property
	def iloc(self):
		"""
		Ensures the inherited iloc operations return a DataFrame instead of a pandas.DataFrame.
		"""
		return _iLocIndexer("iloc", self)


	@property
	def loc(self):
		"""
		Ensures the inherited loc operations return a DataFrame instead of a pandas.DataFrame.
		"""
		return _LocIndexer("loc", self)


	@property
	def at(self):
		"""
		Ensures the inherited at operations return a DataFrame instead of a pandas.DataFrame.
		"""
		return _AtIndexer("at", self)


	@property
	def iat(self):
		"""
		Ensures the inherited iat operations return a DataFrame instead of a pandas.DataFrame.
		"""
		return _iAtIndexer("iat", self)


	def _pre_solve(self):
		"""
		Pre-emptively solve the max-entropy problems remotely, and in the background.
		"""
		data = np.hstack((self.values, np.abs(self.values-self.values.mean(axis=0))))
		corr = spearman_corr(data)
		return solve_copula_async(corr)
		
	



@cast_to_kxy_dataframe
def read_csv(*args, **kwargs):
	"""
	.. _read-csv:
	Same as pandas.read_csv, but returns a kxy.DataFrame object.
	"""
	return pd.read_csv(*args, **kwargs)


@cast_to_kxy_dataframe
def read_excel(*args, **kwargs):
	"""
	.. _read-excel:
	Same as pandas.read_excel, but returns a kxy.DataFrame object.
	"""
	return pd.read_excel(*args, **kwargs)


@cast_to_kxy_dataframe
def read_html(*args, **kwargs):
	"""
	.. _read-html:
	Same as pandas.read_html, but returns a kxy.DataFrame object.
	"""
	return pd.read_html(*args, **kwargs)


@cast_to_kxy_dataframe
def read_table(*args, **kwargs):
	"""
	.. _read-table:
	Same as pandas.read_table, but returns a kxy.DataFrame object.
	"""
	return pd.read_table(*args, **kwargs)


@cast_to_kxy_dataframe
def read_sql(*args, **kwargs):
	"""
	.. _read-sql:
	Same as pandas.read_sql, but returns a kxy.DataFrame object.
	"""
	return pd.read_sql(*args, **kwargs)

