#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from kxy.api import mutual_information_analysis, discrete_mutual_information
from kxy.api.core import discrete_entropy, hqi, prepare_data_for_mutual_info_analysis, \
	one_hot_encoding, two_split_encoding, scalar_continuous_entropy

def classification_achievable_performance_analysis(x_c, y, x_d=None, space='dual', \
		categorical_encoding='two-split'):
	"""
	.. _classification-achievable-performance-analysis:
	Quantifies the performance that can be achieved when trying to predict :math:`y` with :math:`x`.

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

		* :code:`'Achievable R^2'`: The highest :math:`R^2` that can be achieved by a classification model using provided inputs to predict the label.
		* :code:`'Achievable Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a classification model using provided inputs to predict the label.
		* :code:`'Achievable Accuracy'`: The highest classification accuracy that can be achieved by a classification model using provided inputs to predict the label.


	.. admonition:: Theoretical Foundation

		Section :ref:`1 - Achievable Performance`.
	"""
	data = prepare_data_for_mutual_info_analysis(x_c, x_d, None, y, space=space, \
		non_monotonic_extension=True, categorical_encoding=categorical_encoding)
	output_indices = data['output_indices']
	corr = data['corr']
	batch_indices = data['batch_indices']

	mi_analysis = mutual_information_analysis(corr, output_indices, space=space, batch_indices=batch_indices)
	mi = mi_analysis['mutual_information']
	huy = mi_analysis['output_copula_entropy']

	q = len(list(set(y)))
	e = one_hot_encoding(y) if categorical_encoding == 'one-hot' else two_split_encoding(y)
	hy = np.sum([scalar_continuous_entropy(e[:, i]) for i in range(e.shape[1])]) + huy

	return pd.DataFrame({\
		'Achievable R^2': [1.-np.exp(-2.*mi)], \
		'Achievable Log-Likelihood Per Sample': [-max(hy-mi, 0.0)], \
		'Achievable Accuracy': [hqi(max(hy-mi, 0.0), q)]})



def classification_variable_selection_analysis(x_c, y, x_d=None, space='dual', categorical_encoding='two-split'):
	"""
	.. _classification-variable-selection-analysis:
	Runs the variable selection analysis for a classification problem.

	Parameters
	----------
	x_c : (n, d) np.array
		Continuous inputs.
	y : (n,) np.array
		Labels.
	x_d : (n, q) np.array or None (default), optional
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
	"""
	data = prepare_data_for_mutual_info_analysis(x_c, x_d, None, y, space=space, \
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

	cmis = {}
	rsqs = {}
	accs = {}
	mis = {}
	log_liks = {}
	order = {}
	idx = 1
	huy = mi_analysis['output_copula_entropy']
	e = one_hot_encoding(y) if categorical_encoding == 'one-hot' else two_split_encoding(y)
	hy = discrete_entropy(e[:, 0]) if e.shape[1] == 1 else \
		np.sum([scalar_continuous_entropy(e[:, i]) for i in range(e.shape[1])]) + huy
	q = len(list(set(y)))
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
			accs[column] = hqi(max(hy-mis[column], 0.0), q)
			log_liks[column] = -max(hy-mis[column], 0.0)
			order[column] = idx
			idx += 1
			remaining_columns.remove(column)

	rsq_inc = {}
	acc_inc = {}
	log_lik_inc = {}
	run_mi = 0
	run_rsqs = {}
	run_accs = {}
	run_log_liks = {}
	for v, o in sorted(order.items(), key=lambda item: item[1]):
		run_mi += cmis[v]
		run_rsqs[o] = 1.-np.exp(-2.*run_mi)
		run_accs[o] = hqi(max(hy-run_mi, 0.0), q)
		run_log_liks[o] = -max(hy-run_mi, 0.0)

		if rsq_inc == {}:
			rsq_inc[o] = 1.-np.exp(-2.*cmis[v])
			acc_inc[o] = run_accs[o]-hqi(hy, q)
			log_lik_inc[o] = run_mi
			
		else:
			rsq_inc[o] = run_rsqs[o]-run_rsqs[o-1]
			acc_inc[o] = run_accs[o]-run_accs[o-1]
			log_lik_inc[o] = run_log_liks[o]-run_log_liks[o-1]

	max_r_2 = np.max([_ for _ in run_rsqs.values()])
	importance_df = pd.DataFrame({
		'Variable': [v for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Selection Order': [o for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Achievable R^2 analysis
		'Univariate Achievable R^2': [rsqs[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal R^2 Increase': [rsq_inc[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable R^2': [run_rsqs[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable R^2 (%)': [100.*run_rsqs[o]/max_r_2 for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Accuracy Analysis
		'Univariate Achievable Accuracy': [accs[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal Accuracy Increase': [acc_inc[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable Accuracy': [run_accs[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# Conditional mutual information analysis
		'Univariate Mutual Information (nats)': [mis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Conditional Mutual Information (nats)': [cmis[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \

		# The largest likelihood by which the likelihood can be increased as a result of adding this variable
		'Univariate Achievable True Log-Likelihood Per Sample': [log_liks[v] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Maximum Marginal True Log-Likelihood Per Sample Increase': [log_lik_inc[o] for v, o in sorted(order.items(), key=lambda item: item[1])], \
		'Running Achievable True Log-Likelihood Per Sample': [run_log_liks[o] for v, o in sorted(order.items(), key=lambda item: item[1])]})

	importance_df['Running Mutual Information (nats)'] = importance_df['Conditional Mutual Information (nats)'].cumsum()
	max_mi = importance_df['Running Mutual Information (nats)'].max()
	importance_df['Running Mutual Information (%)'] = 100.*importance_df['Running Mutual Information (nats)']/max_mi

	return importance_df


