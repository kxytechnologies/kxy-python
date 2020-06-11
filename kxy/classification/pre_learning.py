#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from kxy.api.core import least_mixed_mutual_information, discrete_mutual_information, \
	discrete_entropy, hqi, least_mixed_conditional_mutual_information


def classification_achievable_performance_analysis(x_c, y, x_d=None, space='dual'):
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
	assert x_d is not None or x_c is not None, "x_c and x_d cannot be both None."

	if x_c is None:
		mi = discrete_mutual_information(x_d, y)

	else:
		mi = least_mixed_mutual_information(x_c, y, x_d=x_d, space=space, non_monotonic_extension=True)

	hy = discrete_entropy(y)
	q = len(list(set(y)))
	n = y.shape[0]

	return pd.DataFrame({\
		'Achievable R^2': [1.-np.exp(-2.*mi)], \
		'Achievable Log-Likelihood Per Sample': [-max(hy-mi, 0.0)], \
		'Achievable Accuracy': [hqi(max(hy-mi, 0.0), q)]})



def classification_variable_selection_analysis(x_c, y, x_d=None, space='dual'):
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
		* :code:`'Conditional Mutual Information'`: The mutual information between this variable and the label, conditional on all variables previously selected.
		* :code:`'Running Mutual Information'`: The mutual information between all variables selected so far, including this one, and the label.
		* :code:`'Univariate Achievable True Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a classification model solely using this variable.
		* :code:`'Maximum Marginal True Log-Likelihood Per Sample Increase'`: The highest amount by which the true log-likelihood per sample can be increased as a result of adding this variable in the variable selection scheme.
		* :code:`'Running Achievable True Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a classification model using all variables selected so far, including this one.
		* :code:`'Maximum Marginal True Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase as a result of adding this variable.
		* :code:`'Running Maximum Log-Likelihood Increase Per Sample'`: The highest amount by which the true log-likelihood per sample can increase (over the log-likelihood of the naive strategy consisting of predicting the mode of :math:`y`) as a result of using all variables selected so far, including this one.


	.. admonition:: Theoretical Foundation

		Section :ref:`2 - Variable Selection Analysis`.
	"""
	x_c_ = x_c[:, None] if len(x_c.shape) == 1 else x_c
	x_d_ = None if x_d is None else x_d[:, None] if len(x_d.shape) == 1 else x_d
	n_inputs = x_c_.shape[1] if x_d is None else x_c_.shape[1] + x_d_.shape[1]
	inputs = [_ for _ in range(n_inputs)]
	categorical_inputs = [_ for _ in range(x_c_.shape[1], n_inputs)]

	final_cmis = {}
	final_mis = {}
	final_univariate_rsq = {}
	final_univariate_acc = {}
	final_univariate_llik = {}
	order = {}
	categorical_conditions = []
	continuous_conditions = []
	hy = discrete_entropy(y)
	q = len(list(set(y)))
	n = y.shape[0]

	def conditional_mutual_information(args):
		# Calculate the conditional mutual information
		i_, cont_cs, cat_cs = args
		_cat_cs = [_-x_c_.shape[1] for _ in cat_cs]
		z_c = None if len(cont_cs) == 0 else x_c_[:, list(cont_cs)]
		z_d = None if len(cat_cs) == 0 else x_d_[:, list(_cat_cs)]
		_x_c = None if i_ in categorical_inputs else x_c_[:, [i_]]
		_x_d = None if x_d_ is None else x_d_[:, i_-x_c_.shape[1]] if i_ in categorical_inputs else None
		is_cat = i_ in categorical_inputs

		mi_ = discrete_mutual_information(_x_d, y) if _x_c is None else \
			least_mixed_mutual_information(_x_c, y, space=space, non_monotonic_extension=True)

		mi_ = min(mi_, hy) # I(y, x) = h(y)-h(y|x) <= h(y) when y is discrete.

		if len(cont_cs) == 0 and len(cat_cs) == 0:
			# (discrete, continuous) | None, or (discrete, discrete) | None
			cmi_ = mi_

		# elif len(cont_cs) == 0 and is_cat:
		# 	# (discrete, discrete | discrete)
		# 	flat_c = np.array(['*_*'.join(list(_)) for _ in z_d])
		# 	flat_all = np.array([flat_c[i] + '*_*' + _x_d[i] for i in range(flat_c.shape[0])])
		# 	cmi_ = discrete_mutual_information(y, flat_all)-discrete_mutual_information(y, flat_c)

		# elif len(cont_cs) == 0 and not is_cat:
		# 	# (discrete, continuous | discrete)
		# 	cmi_ = least_mixed_conditional_mutual_information(_x_c, y, None, z_d=z_d, \
		# 		space=space, non_monotonic_extension=True)


		# elif len(cat_cs) == 0 and is_cat:
		# 	# (discrete, discrete | continuous)
		# 	# I(y; d|c) = I(y; d,c)-I(y; c) 
		# 	#           = I(y;c|d) + I(y; d)-I(y; c)
		# 	cmi_ = least_mixed_conditional_mutual_information(_x_c, y, None, z_d=z_d, \
		# 		space=space, non_monotonic_extension=True)
		# 	cmi_ += discrete_mutual_information(_x_d, y)
		# 	cmi_ -= least_mixed_mutual_information(_x_c, y, space=space, \
		# 		non_monotonic_extension=True)
		# 	cmi_ = max(cmi_, 0.0)


		# elif len(cat_cs) == 0 and not is_cat:
		# 	# (discrete, continuous | continuous)
		# 	cmi_ = least_mixed_conditional_mutual_information(_x_c, y, z_c, \
		# 		space=space, non_monotonic_extension=True)

		# elif is_cat:
		# 	# (discrete, discrete | discrete & continuous)
		# 	# I(y; d|z_c, z_d) = I(y; d, z_c, z_d)-I(y; z_c, z_d) 
		# 	#                  = I(y; z_c| d, z_d) + I(y; d, z_d) - I(y; z_c | z_d) - I(y; z_d)
		# 	flat_c = np.array(['*_*'.join(list(_)) for _ in z_d])
		# 	flat_all = np.array([flat_c[i] + '*_*' + _x_d[i] for i in range(flat_c.shape[0])])
		# 	cmi_ = least_mixed_conditional_mutual_information(z_c, y, None, z_d=flat_all, \
		# 		space=space, non_monotonic_extension=True)
		# 	cmi_ += discrete_mutual_information(y, flat_all)
		# 	cmi_ -= least_mixed_conditional_mutual_information(z_c, y, None, z_d=flat_c, \
		# 		space=space, non_monotonic_extension=True)
		# 	cmi_ -= discrete_mutual_information(y, flat_c)
		# 	cmi_ = max(cmi_, 0.0)

		else:
			# (discrete, continuous | discrete & continuous)
			cmi_ = least_mixed_conditional_mutual_information(_x_c, y, z_c, x_d=_x_d, z_d=z_d, \
				space=space, non_monotonic_extension=True)
			cmi_ = min(cmi_, hy)
			

		return {i_: [cmi_, mi_]}


	while len(inputs) > 0:
		cmis = {}
		with ThreadPoolExecutor(max_workers=10) as p:
			args = [(i, continuous_conditions, categorical_conditions) for i in inputs]
			for cmi in p.map(conditional_mutual_information, args):
				cmis.update(cmi)

		for key, value in sorted(cmis.items(), key=lambda x: -x[1][0]):
			final_cmis[key] = value[0]
			final_mis[key] = value[1]
			final_univariate_rsq[key] = 1.-np.exp(-2.*final_mis[key])
			final_univariate_acc[key] = hqi(max(hy-final_mis[key], 0.0), q)
			final_univariate_llik[key] = min(final_mis[key]-hy, 0.0)

			inputs.remove(key)
			if key in categorical_inputs:
				categorical_conditions += [key]
			else:
				continuous_conditions += [key]
			order[key] = len(categorical_conditions) + len(continuous_conditions)
			break


	final_rsq_inc = {}
	final_run_rsqs = {}

	final_acc_inc = {}
	final_run_acc = {}

	final_llik_inc = {}
	final_run_llik = {}

	run_mi = 0
	for v, o in sorted(order.items(), key=lambda item: item[1]):
		run_mi += final_cmis[v]
		final_run_rsqs[o] = 1.-np.exp(-2.*run_mi)
		final_run_acc[o] = hqi(max(hy-run_mi, 0.0), q)
		final_run_llik[o] = -max(hy-run_mi, 0.0)

		if final_rsq_inc == {}:
			final_rsq_inc[o] = final_run_rsqs[o]
			final_acc_inc[o] = final_run_acc[o]
			final_llik_inc[o] = final_run_llik[o]
			
		else:
			final_rsq_inc[o] = final_run_rsqs[o]-final_run_rsqs[o-1]
			final_acc_inc[o] = final_run_acc[o]-final_run_acc[o-1]
			final_llik_inc[o] = final_run_llik[o]-final_run_llik[o-1]


	sorted_order = sorted(order.items(), key=lambda item: item[1])
	importance_df = pd.DataFrame({
		'Variable': [v for v, o in sorted_order], \
		'Selection Order': [o for v, o in sorted_order], \

		# Achievable R^2 analysis
		'Univariate Achievable R^2': [final_univariate_rsq[v] for v, o in sorted_order], \
		'Maximum Marginal R^2 Increase': [final_rsq_inc[o] for v, o in sorted_order], \
		'Running Achievable R^2': [final_run_rsqs[o] for v, o in sorted_order], \

		# Achievable accuracy analysis
		'Univariate Achievable Accuracy': [final_univariate_acc[v] for v, o in sorted_order], \
		'Maximum Marginal Accuracy Increase': [final_acc_inc[o] for v, o in sorted_order], \
		'Running Achievable Accuracy': [final_run_acc[o] for v, o in sorted_order], \

		# Achievable log-likelihood analysis
		'Univariate Achievable True Log-Likelihood Per Sample': [final_univariate_llik[v] for v, o in sorted_order], \
		'Maximum Marginal True Log-Likelihood Per Sample Increase': [final_llik_inc[o] for v, o in sorted_order], \
		'Running Achievable True Log-Likelihood Per Sample': [final_run_llik[o] for v, o in sorted_order], \

		# Conditional mutual information analysis
		'Univariate Mutual Information': [final_mis[v] for v, o in sorted_order], \
		'Conditional Mutual Information': [final_cmis[v] for v, o in sorted_order]})
	importance_df['Running Mutual Information'] = importance_df['Conditional Mutual Information'].cumsum()


	return importance_df


