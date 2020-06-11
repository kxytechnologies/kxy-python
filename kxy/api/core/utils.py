#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
from numpy.dual import svd

def spearman_corr(x):
	"""
	Calculate the Spearman rank correlation matrix, ignoring nans.

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
	n, m = x.shape
	mask = np.isnan(x).copy()
	valid_mask = np.logical_not(mask).astype(int)

	# R[i,j] is the rank of x[i, j] among x[1, j] ... x[n, j]
	R = (1+x.argsort(axis=0).argsort(axis=0)).astype(float)
	np.copyto(R, np.nan, where=mask)
	o = np.ones((m, m))
	non_nan_ns = np.einsum('ij, ik->jk', valid_mask, valid_mask) # Number of non-nan pairs
	mean_R = np.nanmean(R, axis=0)
	demeaned_R = R - mean_R
	standard_R = demeaned_R/np.nanstd(demeaned_R, axis=0)
	np.copyto(standard_R, 0.0, where=mask)
	corr = np.divide(np.einsum('ij, ik->jk', standard_R, standard_R), non_nan_ns)
	corr = np.round(corr, 3)

	return corr



def pearson_corr(x):
	"""
	Calculate the Pearson correlation matrix, ignoring nans.

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
	n, m = x.shape
	mask = np.isnan(x).copy()
	valid_mask = np.logical_not(mask).astype(int)

	np.copyto(x, np.nan, where=mask)
	o = np.ones((m, m))
	non_nan_ns = np.einsum('ij, ik->jk', valid_mask, valid_mask) # Number of non-nan pairs
	mean_x = np.nanmean(x, axis=0)
	demeaned_x = x - mean_x
	standard_x = demeaned_x/np.nanstd(demeaned_x, axis=0)
	np.copyto(standard_x, 0.0, where=mask)
	corr = np.divide(np.einsum('ij, ik->jk', standard_x, standard_x), non_nan_ns)
	corr = np.round(corr, 3)

	return corr




def robust_log_det(c):
	"""
	Computes the logarithm of the determinant of a positive definite matrix in a fashion that is more robust to ill-conditioning than taking the logarithm of np.linalg.det.

	.. note::
		Specifically, we compute the SVD of c, and return the sum of the log of eigenvalues. np.linalg.det on the other hand computes the Cholesky decomposition of c, which is more likely to fail than its SVD, and takes the product of its diagonal elements, which could be subject to underflow error when diagonal elements are small.

	Parameters
	----------
	c: (d, d) np.array 
		Square input matrix for computing log-determinant.

	Returns
	-------
	d : float
		Log-determinant of the input matrix.
	"""
	u, s, v = svd(c)
	
	return np.sum(np.log(s))


def hqi(h, q):
	"""
	Computes :math:`\\bar{h}_q^{-1}(x)` where

	.. math::

		\\bar{h}_q(a) = -a \\log a -(1-a) \\log \\left(\\frac{1-a}{q-1}\\right), ~~~~ a \\geq \\frac{1}{q}.
	"""
	if h==0:
		return 1.0

	a = np.round(np.arange(1./q, 1.0, 0.001), 3)
	h_ = -a*np.log(a)-(1.-a)*np.log((1.-a)/(q-1.))
	idx = np.abs(h_-h).argmin()

	return a[idx]


