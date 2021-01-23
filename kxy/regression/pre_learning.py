#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


from kxy.api import mutual_information_analysis
from kxy.api.core import scalar_continuous_entropy, prepare_data_for_mutual_info_analysis


def regression_achievable_performance_analysis(x_c, y, x_d=None, space='dual', categorical_encoding='two-split'):
	"""
	.. _regression-achievable-performance-analysis:
	Quantifies the :math:`R^2` that can be achieved when trying to predict :math:`y` with :math:`x`.

	.. math::

		\\bar{R}^2 = 1 - e^{-2I(y; f(x))}


	Parameters
	----------
	x_c : (n,) or (n, d) np.array
		Continuous inputs.
	x_d : (n,) or (n, d) np.array or None (default), optional
		Discrete inputs.
	y : (n,) np.array
		Labels.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
	categorical_encoding : str, 'one-hot' | 'two-split' (default)
		The encoding method to use to represent categorical variables. 
		See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.

	Returns
	-------
	a : pandas.DataFrame

		Dataframe with columns:

		* :code:`'Achievable R^2'`: The highest :math:`R^2` that can be achieved by a regression model using provided inputs.
		* :code:`'Achievable Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a regression model using provided inputs.
		* :code:`'Achievable RMSE'`: The lowest RMSE that can be achieved by a regression model using provided inputs.

	.. admonition:: Theoretical Foundation

		Section :ref:`1 - Achievable Performance`.
	"""
	data = prepare_data_for_mutual_info_analysis(x_c, x_d, y, None, space=space, \
		non_monotonic_extension=True, categorical_encoding=categorical_encoding)
	output_indices = data['output_indices']
	corr = data['corr']
	batch_indices = data['batch_indices']

	mi_analysis = mutual_information_analysis(corr, output_indices, space=space, batch_indices=batch_indices)
	mi = mi_analysis['mutual_information']
	hy = scalar_continuous_entropy(y, space=space)
	std_y = np.sqrt(np.var(y))

	return pd.DataFrame({\
		'Achievable R^2': [1.-np.exp(-2.*mi)], \
		'Achievable Log-Likelihood Per Sample': [mi-hy],
		'Achievable RMSE': std_y*np.exp(-mi)})



def regression_variable_selection_analysis(x_c, y, x_d=None, space='dual', categorical_encoding='two-split'):
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
	categorical_encoding : str, 'one-hot' | 'two-split' (default)
		The encoding method to use to represent categorical variables. 
		See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.

	Returns
	-------
	a : pandas.DataFrame

		Dataframe with columns:

		* :code:`'Variable'`: The variable index starting from 0 at the leftmost column of :code:`x_c` and ending at the rightmost column of :code:`x_d`.
		* :code:`'Selection Order'`: The order in which the associated variable was selected, starting at 1 for the most important variable.
		* :code:`'Univariate Achievable R^2'`: The highest :math:`R^2` that can be achieved by a regression model solely using this variable.
		* :code:`'Maximum Marginal R^2 Increase'`: The highest amount by which the :math:`R^2` can be increased as a result of adding this variable in the variable selection scheme.
		* :code:`'Running Achievable R^2'`: The highest :math:`R^2` that can be achieved by a regression model using all variables selected so far, including this one.
		* :code:`'Conditional Mutual Information (nats)'`: The mutual information between this variable and the label, conditional on all variables previously selected.
		* :code:`'Running Mutual Information (nats)'`: The mutual information between all variables selected so far, including this one, and the label.
		* :code:`'Maximum Marginal True Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase as a result of adding this variable.
		* :code:`'Running Maximum Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase (over the log-likelihood of the naive strategy consisting of predicting the mode of :math:`y`) as a result of using all variables selected so far, including this one.


	.. admonition:: Theoretical Foundation

		Section :ref:`2 - Variable Selection Analysis`.
	"""
	data = prepare_data_for_mutual_info_analysis(x_c, x_d, y, None, space=space, \
		non_monotonic_extension=True, categorical_encoding=categorical_encoding)
	output_indices = data['output_indices']
	corr = data['corr']
	batch_indices = data['batch_indices']

	mi_analysis = mutual_information_analysis(corr, output_indices, space=space, batch_indices=batch_indices)

	d = len(batch_indices)
	batches = [_ for _ in range(d)]
	remaining_columns = [_ for _ in range(d)]

	if mi_analysis is None:
		return None

	std_y = np.sqrt(np.var(y))
	cmis = {}
	rsqs = {}
	rmses = {}
	mis = {}
	order = {}
	idx = 1
	for i in range(1, d+1):
		column_id = mi_analysis['selection_order'][str(i)]
		column = batches[column_id]
		if column in remaining_columns:
			mis[column] = mi_analysis['individual_mutual_informations'][str(i)]
			if idx == 1:
				cmis[column] = mis[column]
			else:
				cmis[column] = mi_analysis['conditional_mutual_informations'][str(i)]
			rsqs[column] = 1.-np.exp(-2.*mis[column])
			rmses[column] = std_y*np.exp(-mis[column])
			order[column] = idx
			idx += 1
			remaining_columns.remove(column)

	rsq_inc = {}
	rmse_dec = {}
	run_mi = 0
	run_rsqs = {}
	run_rmses = {}
	for v, o in sorted(order.items(), key=lambda item: item[1]):
		run_mi += cmis[v]
		run_rsqs[o] = 1.-np.exp(-2.*run_mi)
		run_rmses[o] = std_y*np.exp(-run_mi)

		if rsq_inc == {}:
			rsq_inc[o] = 1.-np.exp(-2.*cmis[v])
			rmse_dec[o] = std_y-std_y*np.exp(-cmis[v])
			
		else:
			rsq_inc[o] = run_rsqs[o]-run_rsqs[o-1]
			rmse_dec[o] = run_rmses[o-1]-run_rmses[o]

	n = y.shape[0]
	max_r_2 = np.max([_ for _ in run_rsqs.values()])
	importance_df = pd.DataFrame({
		'Variable': [v for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Selection Order': [o for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Achievable R^2 analysis
		'Univariate Achievable R^2': [rsqs[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal R^2 Increase': [rsq_inc[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable R^2': [run_rsqs[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable R^2 (%)': [100.*run_rsqs[o]/max_r_2 for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Achievable RMSE analysis
		'Univariate Achievable RMSE': [rmses[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal RMSE Decrease': [rmse_dec[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable RMSE': [run_rmses[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable RMSE/STD (%)': [100.*run_rmses[o]/std_y for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Conditional mutual information analysis
		'Univariate Mutual Information (nats)': [mis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Conditional Mutual Information (nats)': [cmis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# The largest likelihood by which the likelihood can be increased as a result of adding this variable
		'Univariate Maximum True Log-Likelihood Increase Per Sample': [mis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal True Log-Likelihood Increase Per Sample': [cmis[v] for v, o in sorted(order.items(), key=lambda item: item[1])]})

	importance_df['Running Mutual Information (nats)'] = importance_df['Conditional Mutual Information (nats)'].cumsum()
	max_mi = importance_df['Running Mutual Information (nats)'].max()
	importance_df['Running Mutual Information (%)'] = 100.*importance_df['Running Mutual Information (nats)']/max_mi

	# The largest value by which the lof likelihood can be increased by using the variables selected so far.
	importance_df['Running Maximum Log-Likelihood Increase Per Sample'] = importance_df['Maximum Marginal True Log-Likelihood Increase Per Sample'].cumsum()

	return importance_df


