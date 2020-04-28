#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
from numpy.dual import svd

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
	# R[i,j] is the rank of x[i, j] among x[1, j] ... x[n, j]
	n, m = x.shape
	R = 1+x.argsort(axis=0).argsort(axis=0)
	spearman_corr = np.cov(R.T, bias=True)/((n*n-1.)/12.)
	rho = (np.sum(spearman_corr)-m)/(m*(m-1.))

	return rho



def pre_conditioner(x):
	"""
	Given n i.i.d. samples from a random variable x, compute a square matrix A such that 
	coordinates of :math:`z = Ax` have similar pairwise Spearman rank correlation.

	.. note::

		The current implementation computes A as :math:`A = D^{-\\frac{1}{2}} U^T` where :math:`C = UDU^T` 
		is the SVD decomposition of the covariance matrix of :math:`x`.

		It is worth recalling that if :math:`x` is mean 0 and has covariance matrix :math:`C`, then 
		:math:`z := D^{-\\frac{1}{2}} U^T x` is also mean 0 and has covariance matrix the identity matrix.

		The Spearman rank correlation of :math:`z` might not be the identity matrix, but its off diagonal terms
		will have the same magnitude (fairly small).


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
	cov = np.cov(x)
	u, s, v = svd(cov)
	eps = 0.0
	oc = np.max(s)/np.min(s)
	if oc > 1.0:
		nc = np.min([oc, 1e3])
		eps = np.min(s)*(oc-nc)/(nc-1.0)
	
	a = np.dot(np.diag(1.0/(np.sqrt(np.absolute(s) + eps))), u.T)
	det_a = np.prod(1.0/(np.sqrt(np.absolute(s) + eps)))

	return a, np.log(np.abs(det_a))


