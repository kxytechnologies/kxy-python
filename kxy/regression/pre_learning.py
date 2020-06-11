#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


from kxy.api import mutual_information_analysis
from kxy.api.core import least_continuous_mutual_information, least_mixed_conditional_mutual_information, \
	scalar_continuous_entropy, spearman_corr, pearson_corr


def regression_achievable_performance_analysis(x_c, y, x_d=None, space='dual'):
	"""
	.. _regression-achievable-performance-analysis:
	Quantifies the :math:`R^2` that can be achieved when trying to predict :math:`y` with :math:`x`.

	.. math::

		\\bar{R}^2 = 1 - e^{-2I(y; f(x))}


	Parameters
	----------
	x_c : (n,d) np.array
		Continuous inputs.
	y : (n,) np.array
		Labels.
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

		* :code:`'Achievable R^2'`: The highest :math:`R^2` that can be achieved by a regression model using provided inputs.
		* :code:`'Achievable Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a regression model using provided inputs.
		

	.. admonition:: Theoretical Foundation

		Section :ref:`1 - Achievable Performance`.
	"""
	assert len(y.shape) == 1 or y.shape[1] == 1, 'y should be a one dimensional numpy array'

	if x_d is None:
		mi = least_continuous_mutual_information(x_c, y, space=space)

	else:
		y_ = y[:, None] if len(y.shape) == 1 else y
		x_d_ = np.array(['*_*'.join(list(r)) for r in x_d]) if len(y.shape) > 1 else x_d

		# I(y; x_d)
		mi = least_mixed_conditional_mutual_information(y, x_d_, space=space, non_monotonic_extension=False)
		categories = list(set(list(x_d_)))
		n = x_d_.shape[0]
		probas = np.array([1.*len(x_d_[x_d_==cat])/n for cat in categories])

		# I(y; x_c|x_d)
		mi += np.sum([probas[i]*least_continuous_mutual_information(\
			x_c[x_d_==categories[i]], y[x_d_==categories[i]], space=space) for i in range(len(categories))\
			if probas[i] > 0.0])

	hy = scalar_continuous_entropy(y, space=space)

	return pd.DataFrame({\
		'Achievable R^2': [1.-np.exp(-2.*mi)], \
		'Achievable True Log-Likelihood Per Sample': [hy-mi]})



def regression_variable_selection_analysis(x_c, y, x_d=None, space='dual'):
	"""
	.. _regression-variable-selection-analysis:
	Runs the variable selection analysis for a regression problem.


	Parameters
	----------
	x_c : (n,d) np.array
		Continuous inputs.
	y : (n,) np.array
		Labels.
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
		* :code:`'Univariate Achievable R^2'`: The highest :math:`R^2` that can be achieved by a regression model solely using this variable.
		* :code:`'Maximum Marginal R^2 Increase'`: The highest amount by which the :math:`R^2` can be increased as a result of adding this variable in the variable selection scheme.
		* :code:`'Running Achievable R^2'`: The highest :math:`R^2` that can be achieved by a regression model using all variables selected so far, including this one.
		* :code:`'Conditional Mutual Information'`: The mutual information between this variable and the label, conditional on all variables previously selected.
		* :code:`'Running Mutual Information'`: The mutual information between all variables selected so far, including this one, and the label.
		* :code:`'Maximum Marginal True Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase as a result of adding this variable.
		* :code:`'Running Maximum Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase (over the log-likelihood of the naive strategy consisting of predicting the mode of :math:`y`) as a result of using all variables selected so far, including this one.


	.. admonition:: Theoretical Foundation

		Section :ref:`2 - Variable Selection Analysis`.
	"""
	assert x_d is None, 'Variable selection for regression does not yet support discrete variables'

	d = x_c.shape[1]
	data = np.hstack((y[:, None] , x_c, np.abs(x_c-np.nanmean(x_c, axis=0))))
	corr = pearson_corr(data) if space == 'primal' else spearman_corr(data)
	mi_analysis = mutual_information_analysis(corr, 0, space=space, greedy=True)
	columns = [0] + [_ for _ in range(d)] + [_ for _ in range(d)]

	if mi_analysis is None:
		return None

	remaining_columns = [_ for _ in range(d)]

	cmis = {}
	rsqs = {}
	mis = {}
	order = {}
	idx = 1
	for i in range(1, 1+2*d):
		column_id = mi_analysis['selection_order'][str(i)]
		column = columns[column_id]
		if column in remaining_columns:
			mis[column] = least_continuous_mutual_information(x_c[:, [column]], y, space=space)
			if idx == 1:
				cmis[column] = mis[column]
			else:
				cmis[column] = mi_analysis['conditional_mutual_informations'][str(i)]
			rsqs[column] = 1.-np.exp(-2.*mis[column])
			order[column] = idx
			idx += 1
			remaining_columns.remove(column)

	rsq_inc = {}
	run_mi = 0
	run_rsqs = {}
	for v, o in sorted(order.items(), key=lambda item: item[1]):
		run_mi += cmis[v]
		run_rsqs[o] = 1.-np.exp(-2.*run_mi)

		if rsq_inc == {}:
			rsq_inc[o] = 1.-np.exp(-2.*cmis[v])
			
		else:
			rsq_inc[o] = run_rsqs[o]-run_rsqs[o-1]


	n = y.shape[0]
	importance_df = pd.DataFrame({
		'Variable': [v for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Selection Order': [o for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Achievable R^2 analysis
		'Univariate Achievable R^2': [rsqs[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal R^2 Increase': [rsq_inc[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable R^2': [run_rsqs[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Conditional mutual information analysis
		'Univariate Mutual Information': [mis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Conditional Mutual Information': [cmis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# The largest likelihood by which the likelihood can be increased as a result of adding this variable
		'Univariate Maximum True Log-Likelihood Increase Per Sample': [mis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal True Log-Likelihood Increase Per Sample': [cmis[v] for v, o in sorted(order.items(), key=lambda item: item[1])]})

	importance_df['Running Mutual Information'] = importance_df['Conditional Mutual Information'].cumsum()
	# The largest value by which the lof likelihood can be increased by using the variables selected so far.
	importance_df['Running Maximum Log-Likelihood Increase Per Sample'] = importance_df['Maximum Marginal True Log-Likelihood Increase Per Sample'].cumsum()

	return importance_df


