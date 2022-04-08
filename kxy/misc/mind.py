#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow Implementation of MIND ([1]) under Spearman rank correlation constraints.

[1] Kom Samo, Y. (2021). Inductive Mutual Information Estimation: A Convex Maximum-Entropy Copula Approach . <i>Proceedings of The 24th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 130:2242-2250 Available from https://proceedings.mlr.press/v130/kom-samo21a.html.
"""
import numpy as np

from kxy.misc.tf import CopulaLearner

def copula_entropy(z, subsets=[]):
	'''
	Estimate the entropy of the copula distribution of a d-dimensional random vector using MIND ([1]) with Spearman rank correlation constraints.


	Parameters
	----------
	z : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.


	Returns
	-------
	ent : float
		The estimated copula entropy.
	'''
	if len(z.shape)==1 or z.shape[1]==1:
		return 0.0

	d = z.shape[1]
	cl = CopulaLearner(d, subsets=subsets)
	cl.fit(z)
	ent = min(cl.copula_entropy, 0.0)

	return ent



def mutual_information(y, x):
	'''
	Estimate the mutual information between two random vectors using MIND ([1]) with Spearman rank correlation constraints.


	Parameters
	----------
	y : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.
	x : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.


	Returns
	-------
	mi : float
		The estimated mutual information.
	'''
	y = y[:, None] if len(y.shape)==1 else y
	x = x[:, None] if len(x.shape)==1 else x
	z = np.concatenate([y, x], axis=1)
	huy = copula_entropy(y)
	hux = copula_entropy(x)
	huz = copula_entropy(z)
	mi = max(huy+hux-huz, 0.0)

	return mi


def run_d_dimensional_gaussian_experiment(d, rho, n=1000):
	'''
	'''
	# Cholesky decomposition of corr = np.array([[1., rho], [rho, 1.]])
	L = np.array([[1., 0.], [rho, np.sqrt(1.-rho*rho)]])
	y = np.empty((n, d))
	x = np.empty((n, d))
	for i in range(d):
		u = np.random.randn(n, 2)
		z = np.dot(L, u.T).T
		y[:, i] = z[:, 0].copy()
		x[:, i] = z[:, 1].copy()

	estimated_mi = mutual_information(y, x)
	true_mi = -d*0.5*np.log(1.-rho*rho)

	return estimated_mi, true_mi



if __name__ == '__main__':
	rho = 0.95
	d = 20
	estimated_mi, true_mi = run_d_dimensional_gaussian_experiment(d, rho)
	print('%dd Gaussian Mutual Information: Estimated %.4f, True (theoretical) %.4f' % (\
		d, estimated_mi, true_mi))



