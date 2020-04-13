#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file extends pandas' dataframe object to allow data scientists to tap into 
the power of the KXY API within the comfort of their favorite data structure.
"""

from functools import lru_cache, wraps

import numpy as np
import pandas as pd

from kxy.api.core import least_total_correlation
from kxy.classification import classification_feasibility
from kxy.finance import information_adjusted_beta, information_adjusted_correlation
from kxy.regression import regression_feasibility, regression_suboptimality

from .decorators import decorate_methods


def cast_to_kxy_dataframe(method):
	"""
	Cast the return of a method to kxy.DataFrame if it returns pd.DataFrame.
	"""
	@wraps(method)
	def wrapper(*args, **kwargs):
		res = method(*args, **kwargs)
		if isinstance(res, pd.DataFrame):
			return DataFrame(data=res.values, columns=res.columns, index=res.index)
		return res

	return wrapper



@decorate_methods(cast_to_kxy_dataframe, include=[\
	'__add__', '__sub__', '__mul__', '__div__', '__floordiv__', '__truediv__', '__mod__', '__pow__', \
	'__iadd__', '__isub__', '__imul__', '__idiv__', '__ifloordiv__', '__imod__', '__ipow__', \
	'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow', \
	'radd', 'rsub', 'rmul', 'rdiv', 'rtruediv', 'rfloordiv', 'rmod', 'rpow', \
	'__eq__', '__ne__', '__gt__', '__lt__', '__le__', '__ge__', '__neg__', '__pos__',  '__invert__', \
	'eq', 'ne', 'gt', 'lt', 'le', 'ge', 'apply', 'applymap', 'abs', 'clip', 'count', \
	'cov', 'cummin', 'cummax', 'cumprod', 'cumsum', 'round', 'sum', 'std', 'var', 'cov', \
	'nunique', 'rename', 'rename_axis', 'reset_index', 'drop_level', 'pivot', 'T', 'rank', \
	'transpose', 'fillna', 'dropna', 'pct_change', 'swapaxes', 'swaplevel', 'where', 'tail', \
	'head', 'shift', 'asfreq', 'dot', 'transform', 'astype', 'copy'])
class DataFrame(pd.DataFrame):
	"""
	Extension of pandas' DataFrame class with various analytics for pre-learning and post-learning,
	in supervised learning problems.
	"""
	def regression_feasibility(self, label_column, features_columns=()):
		"""
		Quantifies how feasible a regression problem is by computing the amount of uncertainty
		about the label that can be reduced by knowing the features, in a model-free fashion.

		.. math::
			\text{feasibility}(y; x) &= h(y)-h\left(y \vert x \right) \\
									 &= I(y, x)
									 &= h\left(u_x\right)-h\left(u_{x,y}\right)

		Copula entropies are estimated by solving a maximum-entropy copula problem under concordance-like
		constraints on x and on (y, x) jointly. See also :ref:`regression-feasibility`.

		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		features_columns : set, optional
			The set of columns to as features. When features_columns is the empty set,
			all columns except for label_column are used as features.

		Returns
		-------
		f : float
			The feasibility score in :math:`[0, \infty]`. The larger the better.

		Raises
		------
		AssertionError
			If label_column is in features_columns.
		"""
		features_columns = list(set(self.columns)-set(label_column)) if features_columns == () \
			else list(features_columns)
		assert label_column not in features_columns, "The output cannot be a feature"

		return regression_feasibility(self[features_columns].values, self[label_column].values)


	def classification_feasibility(self, output_column, discrete_features_columns=(), \
			continuous_features_columns=()):
		"""
		Quantifies how feasible a classification problem is by computing the amount of uncertainty
		about the label that can be reduced by knowing the features, in a model-free fashion.

		.. math::
			\text{feasibility}(y; x) &= h(y)-h\left(y \vert x \right) \\
									 &= I(y, x)

		See :ref:`classification-feasibility` for more details.

		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		discrete_features_columns : set, optional
			The set of columns, if any, to use as features that are discrete.
		continuous_features_columns : set, optional
			The set of columns to use as features that are continuous.

		Returns
		-------
		f : float
			The feasibility score in :math:`[0, \infty]`. The larger the better.
		"""
		assert output_column not in discrete_features_columns, "The output cannot be a feature"
		assert output_column not in continuous_features_columns, "The output cannot be a feature"

		continuous_features_columns = [col for col in self.columns if col != output_column and not self.is_discrete(col)] \
			if continuous_features_columns == () else list(continuous_features_columns)
		assert len(continuous_features_columns) > 0, "Continuous features are required"

		x_d = self[discrete_features_columns].values if len(discrete_features_columns) > 0 else sNone
		x_c = self[continuous_features_columns].values
		y = self[output_column].values

		return classification_feasibility(x_c, y, x_d=x_d)


	@lru_cache(maxsize=32)
	def features_importance(self, label_column, features_columns=(), problem=None):
		"""
		Calculates the importance of each feature in the input set at solving the supervised
		learning problem where the label is defined by the label_column.

		Feature importance is defined as the mutual information between the feature column 
		and the label column.

		The supervised learning problem can either be specified as 'classification' or 
		'regression' using the problem colummn, or inferred from the type of, and the number 
		of distinct values in the label_column.

		See also :ref:`classification-feasibility` and :ref:`classification-feasibility`.

		Parameters
		----------
		label_column : str
			The name of the column to use as label.
		features_columns : set, optional
			The set of columns to as features. When features_columns is the empty set,
			all columns except for label_column are used as features.
		problem : str or None (default), optional
			The type of supervised learning problem. One of None (default), 'classification'
			or 'regression'. When problem is None, the supervised learning problem is inferred
			based on whether labels are numeric and the percentage of distinct labels.

		Returns
		-------
		importance : dict
			A dictionary whose keys are feature names and values the corresponding importances.

		Raises
		------
		AssertionError
			If problem is neither None nor 'classification' nor 'regression', or if 
			label_column is in features_columns.
		"""
		features_columns = list(set(self.columns)-set(label_column)) if features_columns == () \
			else list(features_columns)
		assert label_column not in features_columns, "The output cannot be a feature"
		assert problem is None or problem in ('classification', 'regression'), \
			"The problem should be either None, 'classification' or 'regression'"

		if problem is None:
			problem = 'classification' if self.is_discrete(label_column) else 'regression'

		if problem == 'classification':
			importance = {col: classification_feasibility(\
							None, self[label_column].values, x_d=self[col].values) if self.is_discrete(col) else \
							   classification_feasibility(self[col].values, self[label_column].values, x_d=None) \
							for col in features_columns}

		else:
			importance = {col: regression_feasibility(\
				self[col].values, self[label_column].values) if not self.is_discrete(col) else \
				classification_feasibility(self[label_column].values, self[col].values, x_d=None) \
				for col in features_columns}

		return importance


	def is_discrete(self, column):
		"""
		Determine whether the input column contains discrete observations.
		"""
		ret = (not np.can_cast(self[column].values, float))
		ret = ret or len(list(set(self[column].values))) < 0.5*self.shape[0]

		return ret


	@lru_cache(maxsize=32)
	def corr(self, columns=(), method='information-adjusted', min_periods=1, p=0):
		"""
		Calculates the auto-correlation matrix of all columns or the input subset.

		See also :ref:`information-adjusted-correlation`.

		Parameters
		----------
		columns : set, optional
			The set of columns to use. If not provided, all columns are used.
		method : str, optional
			Which method to use to calculate the auto-correlation matrix. Supported
			values are 'information-adjusted' (the default) (see :ref:`information-adjusted-correlation`)
			and all 'method' values of pandas.DataFrame.corr.
		p : int, optional
			The number of lags to use when generating Spearman rank auto-correlation to use 
			as empirical evidence in the maximum-entropy problem. The default value is 0, which 
			corresponds to assuming rows are i.i.d. This is also the only supported value for now.
			See :ref:`information-adjusted-correlation`.
		min_periods : int, optional
			Only used when method is not 'information-adjusted'. 
			See the documentation of pandas.DataFrame.corr.

		Returns
		-------
		c : DataFrame
			The auto-correlation matrix.
		"""
		columns = self.columns if columns == () else list(columns)

		if method == 'information-adjusted':
			c = information_adjusted_correlation(self[columns].values, self[columns].values)
			return pd.DataFrame(c, columns=columns, index=columns)
		else:
			return pd.DataFrame.corr(self[columns], method=method, min_periods=min_periods)


	@lru_cache(maxsize=32)
	def beta(self, column_y, column_x, method='information-adjusted'):
		"""
		Calculates the information-adjusted beta of a portfolio or asset (whose returns are provided in 
		column_y) with respect to the market (whose returns are provided in column_x).

		The information-adjusted beta coefficient generalizes the traditional (CAPM/OLS/Pearson) beta 
		in that, unlike CAPM beta that only captures linear cross-sectional dependence, the 
		information-adjusted beta captures cross-sectional and temporal dependence, linear and nonlinear.

		The IA-beta is 0 if and only if the portfolio or asset exhibit no dependence with the market, linear
		or nonlinear, cross-sectional or temporal.

		See also :ref:`information-adjusted-beta`.

		Parameters
		----------
		colummn_y : str
			The name of the column to use for portfolio/asset returns.
		colummn_x : str
			The name of the column to use for market returns.
		method : str, optional
			Either 'information-adjusted' for information-adjusted beta (see :ref:`information-adjusted-beta`),
			or 'pearson' for the traditional OLS/pearson beta coefficient.

		Returns
		-------
		c : float
			The beta coefficient.

		Raises
		------
		AssertionError
			If the method is neither 'information-adjusted' nor 'pearson'.
		"""
		assert method in ('information-adjusted', 'pearson'), "Allowed methods are 'information-adjusted' and 'pearson'."

		if method == 'information-adjusted':
			return information_adjusted_beta(self[column_y].values, self[column_x].values)

		c = np.corrcoef(self[column_y].values, self[column_x].values)[0, 1]
		return c*np.sqrt(self[column_y].values.var()/self[column_x].values.var())


	@lru_cache(maxsize=32)
	def total_correlation(self, columns=()):
		"""
		Calculates the total correlation between all columns or the input subset.

		See also :ref:`least-total-correlation`.

		Parameters
		----------
		columns : set, optional
			The set of columns to use. If not provided, all columns are used.

		Returns
		-------
		c : float
			The total correlation.
		"""
		columns = self.columns if columns == () else list(columns)
		return least_total_correlation(self[columns].values)


	@lru_cache(maxsize=32)
	def regression_suboptimality(self, residual_column, label_column):
		"""
		Quantifies to what extend a calibrated regression model can be improved by evaluating 
		the mutual information between the regression residuals and the label.

		A large mutual information betweeen residuals and label is evidence that the calibrated 
		model can still be improved on, using the same features. As usual, the mutual information
		is estimated in a model-free fashion. 

		This function should typically be used in the post-learning stage of the modelling cycle
		to determine whether resources would yield a higher ROI if allocated to improving the model
		or to aquiring new datasets or features.

		See also :ref:`model-suboptimality`.

		Parameters
		----------
		residual_column : str
			The name of the column containing regression residuals.
		label_column : str
			The name of the column containing regression labels.

		Returns
		-------
		s : float
			The suboptimality score, defined as the mutual information between residuals and labels.
			The higher the value, the more the model can be improved.
		"""
		assert not self.is_discrete(residual_column), "The residual column should not be discrete"
		assert not self.is_discrete(label_column), "The label column should not be discrete"		

		return regression_suboptimality(self[residual_column], self[label_column])



	def __hash__(self):
		return hash(self.to_string())



@cast_to_kxy_dataframe
def read_csv(*args, **kwargs):
	"""
	Same as pandas.read_csv, but returns a kxy.DataFrame.
	"""
	return pd.read_csv(*args, **kwargs)


@cast_to_kxy_dataframe
def read_excel(*args, **kwargs):
	"""
	Same as pandas.read_excel, but returns a kxy.DataFrame.
	"""
	return pd.read_excel(*args, **kwargs)


@cast_to_kxy_dataframe
def read_html(*args, **kwargs):
	"""
	Same as pandas.read_html, but returns a kxy.DataFrame.
	"""
	return pd.read_html(*args, **kwargs)


@cast_to_kxy_dataframe
def read_table(*args, **kwargs):
	"""
	Same as pandas.read_table, but returns a kxy.DataFrame.
	"""
	return pd.read_table(*args, **kwargs)


@cast_to_kxy_dataframe
def read_sql(*args, **kwargs):
	"""
	Same as pandas.read_sql, but returns a kxy.DataFrame.
	"""
	return pd.read_sql(*args, **kwargs)

