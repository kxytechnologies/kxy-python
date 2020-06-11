#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from kxy.api.core import least_continuous_mutual_information, least_mixed_mutual_information

from .pre_learning import regression_variable_selection_analysis, \
	regression_achievable_performance_analysis



def regression_model_improvability_analysis(x_c, y_p, y, x_d=None, space='dual'):
	"""
	.. _regression-model-improvability-analysis:
	Runs the model improvability analysis on a trained regression model.

	Parameters
	----------
	x_c : (n,d) np.array
		Continuous inputs.
	y_p : (n,) np.array
		Labels predicted by the model and corresponding to inputs x_c and x_d.
	y : (n,) np.array
		True labels.
	x_d : (n, d) np.array or None (default), optional
		Discrete inputs.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


	Returns
	-------
	a : pandas.DataFrame

		Dataframe with columns:

		* :code:`'Leftover R^2'`: The amount by which the trained model's :math:`R^2` can still be improved without resorting to additional inputs, simply through better modeling.
		* :code:`'Leftover Log-Likelihood Per Sample'`: The amount by which the trained model's true log-likelihood per sample can still be increased without resorting to additional inputs, simply through better modeling.


	.. admonition:: Theoretical Foundation

		Section :ref:`3 - Model Improvability`.
	"""
	achievable_perf = regression_achievable_performance_analysis(x_c, y, x_d=x_d, space=space)
	achieved_perf   = regression_achievable_performance_analysis(y_p, y, x_d=None, space=space)
	
	improvable_perf = achievable_perf-achieved_perf
	improvable_perf.rename(columns={col: col.replace('Achievable', 'Lost') for col in improvable_perf.columns}, inplace=True)
	improvable_perf = np.maximum(improvable_perf, 0.0)

	residual_perf = regression_achievable_performance_analysis(x_c, y-y_p, x_d=x_d, space=space)
	residual_perf.rename(columns={col: col.replace('Achievable', 'Residual') for col in residual_perf.columns}, inplace=True)
	residual_perf = np.maximum(residual_perf, 0.0)

	return pd.concat([improvable_perf, residual_perf], axis=1)



def regression_model_explanation_analysis(x_c, f_x, x_d=None, space='dual'):
	"""
	.. _regression-model-explanation-analysis:
	Runs the model explanation analysis on a trained regression model.

	Parameters
	----------
	x_c : (n,d) np.array
		Continuous inputs.
	f_x : (n,) np.array
		Labels predicted by the model and corresponding to inputs x_c and x_d.
	x_d : (n, d) np.array or None (default), optional
		Discrete inputs.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


	Returns
	-------
	a : pandas.DataFrame

		Dataframe with columns:

		* :code:`'Variable'`: The variable index starting from 0 at the leftmost column of :code:`x_c` and ending at the rightmost column of :code:`x_d`.
		* :code:`'Selection Order'`: The order in which the associated variable was selected, starting at 1 for the most important variable.
		* :code:`'Univariate Explained R^2'`: The :math:`R^2` between predicted labels and this variable.
		* :code:`'Running Explained R^2'`: The :math:`R^2` between predicted labels and all variables selected so far, including this one.
		* :code:`'Marginal Explained R^2'`: The increase in :math:`R^2` between predicted labels and all variables selected so far that is due to adding this variable in the selection scheme.

	.. admonition:: Theoretical Foundation

		Section :ref:`a) Model Explanation`.
	"""
	res = regression_variable_selection_analysis(x_c, f_x, x_d=x_d, space=space)
	res = res[['Variable', 'Selection Order', 'Univariate Achievable R^2', 'Maximum Marginal R^2 Increase', \
		'Running Achievable R^2']]
	res.rename(columns={'Univariate Achievable R^2': 'Univariate Explained R^2', \
		'Maximum Marginal R^2 Increase': 'Marginal Explained R^2', \
		'Running Achievable R^2': 'Running Explained R^2'}, inplace=True)

	return res



def regression_bias(f_x, z, linear_scale=True, space='dual'):
	"""
	.. _regression-bias:
	Quantifies the bias in a regression model as the mutual information between a category variable and model predictions.


	Parameters
	----------
	f_x : (n,) np.array
		The model decisions.
	z : (n,) np.array
		The associated variable through which the bias could arise.



	Returns
	-------
	b : float
		The mutual information :math:`m` or :math:`1-e^{-2m}` if :code:`linear_scale=True`.


	.. admonition:: Theoretical Foundation

		Section :ref:`b) Quantifying Bias in Models`.
	"""
	mi = least_mixed_mutual_information(f_x, z, space=space, non_monotonic_extension=True)
	if linear_scale:
		mi = 1.-np.exp(-2.*mi)

	return mi

