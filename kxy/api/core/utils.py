#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
from numpy.dual import svd

def spearman_corr(x):
	"""
	Calculate the Spearman rank correlation matrix.

	.. math::

		\\text{corr}[i, j] = \\frac{12}{n^2-1} \\left[ \\left( \\frac{1}{n} \\sum_{k=1}^n R_{ki} R_{kj} \\right) - \\frac{(n+1)^2}{4}\\right]

	Parameters
	----------
	x : (n, d) np.array
		Input data representing n i.i.d. draws from the d-dimensional 
		random variable, whose Spearman rank correlation matrix this 
		function calculates.


	Returns
	-------
	corr : np.array
		The Spearman rank correlation matrix.
	"""
	# R[i,j] is the rank of x[i, j] among x[1, j] ... x[n, j]
	n, m = x.shape
	R = 1+x.argsort(axis=0).argsort(axis=0)
	corr = np.cov(R.T, bias=True)/((n*n-1.)/12.)
	corr = np.round(corr, 3)

	return corr



def pearson_corr(x):
	"""
	Calculate the Pearson correlation matrix, ignoring nan.

	Parameters
	----------
	x : (n, d) np.array
		Input data representing n i.i.d. draws from the d-dimensional 
		random variable, whose Pearson correlation matrix this 
		function calculates.


	Returns
	-------
	corr : np.array
		The Pearson correlation matrix.
	"""
	mask = np.all(~np.isnan(x), axis=1)
	nx = x[mask]

	nx = nx - np.mean(nx, axis=0)
	nx /= np.std(nx, axis=0)

	n = nx.shape[0]
	c = np.dot(nx.T, nx)/(n-1)

	return c




def avg_pairwise_spearman_corr(x):
	"""
	Calculates the sample average pairwise Spearman rank correlation 
	between columns of the 2-dimensional input array `x` of shape (n, d) as

	.. math::

		\\frac{12}{n^2-1} \\left[ \\frac{2}{d(d-1)} \\sum_{j<j^\\prime} \\left[ \\left( \\frac{1}{n} \\sum_{i=1}^n R_{ij} R_{ij^\\prime} \\right) - \\frac{(n+1)^2}{4}\\right] \\right]

	where :math:`R_{ij}` is the rank of :math:`x_{ij}` among :math:`x_{1j} \\dots x_{nj}`. 

	See Eq. (3.3.3) in [1]_.


	Parameters
	----------
	x : (n, d) np.array
		Input data representing n i.i.d. draws from the d-dimensional 
		random variable, whose average pairwise Spearman rank correlation
		this function calculates.


	Returns
	-------
	rho : float
		The average pairwise Spearman rank correlation.



	.. rubric:: References

	.. [1] Joe, H. Journal of multivariate analysis 35 (1), 12-30, 1990.
	"""
	corr = spearman_corr(x)
	rho = (np.sum(corr)-m)/(m*(m-1.))

	return rho



def pre_conditioner(x):
	"""
	Given n i.i.d. samples from a random variable x, compute a square matrix A such that 
	coordinates of :math:`z = Ax` have similar pairwise Spearman rank correlation.

	.. note::

		The current implementation computes A as :math:`A = U^T` where :math:`C = UDU^T` 
		is the SVD decomposition of the covariance matrix of :math:`x`.

		It is worth recalling that if :math:`x` is mean 0 and has covariance matrix :math:`C`, then 
		:math:`z := U^T x` is also mean 0 and has covariance matrix :math:`D`.

		The Spearman rank correlation of :math:`z` might not be the identity matrix, but its off diagonal terms
		should have the same magnitude (fairly small).


	Parameters
	----------
	x : (n, d) np.array
		Input data representing n i.i.d. draws from the d-dimensional 
		random variable.


	Returns
	-------
	A : (d, d) np.array
		The pre-conditioning matrix.
	ld: float
		:math:`\\log |\\text{det} A|`
	"""
	cov = np.cov(x.T)
	u, s, v = svd(cov)

	return v, 0.0


