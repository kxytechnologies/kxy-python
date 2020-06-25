#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
from scipy.stats.mstats import rankdata
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
	R = rankdata(x, axis=0)
	np.copyto(R, np.nan, where=mask)
	mean_R = np.nanmean(R, axis=0)
	demeaned_R = R - mean_R
	istd_R = 1./np.nanstd(demeaned_R, axis=0)
	np.copyto(istd_R, 0.0, where=np.isinf(istd_R))
	np.copyto(istd_R, 0.0, where=np.isnan(istd_R))
	standard_R = demeaned_R*istd_R
	np.copyto(standard_R, 0.0, where=mask)
	non_nan_ns = np.einsum('ij, ik->jk', valid_mask, valid_mask) # Number of non-nan pairs
	corr = np.divide(np.einsum('ij, ik->jk', standard_R, standard_R), non_nan_ns)
	np.fill_diagonal(corr, 1.)
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
	istd_x = 1./np.nanstd(demeaned_x, axis=0)
	np.copyto(istd_x, 0.0, where=np.isinf(istd_x))
	np.copyto(istd_x, 0.0, where=np.isnan(istd_x))
	standard_x = demeaned_x*istd_x
	np.copyto(standard_x, 0.0, where=mask)
	corr = np.divide(np.einsum('ij, ik->jk', standard_x, standard_x), non_nan_ns)
	np.fill_diagonal(corr, 1.)
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


def one_hot_encoding(x):
	"""
	.. _one-hot-encoding:
	Computes the `one-hot-encoding <https://en.wikipedia.org/wiki/One-hot>`_ representation of the input array.

	The representation used is the one where all distincts inputs are first converted to string, then sorted using Python's :code:`sorted` method. 

	The :math:`i`-th element in this sort has its :math:`i`-th bit from the right set to 1 and all others to 0.


	Parameters
	----------
	x: (n,) or (n, d) np.array
		Array of n inputs (typically strings) to encode and that take q distinct values.


	Returns
	-------
	 : (n, q) np.array
		Binary array representing the one-hot encoding representation of the inputs.
	"""
	x_ = np.array([str(_) for _ in x]).astype(str)
	n = len(x_)
	cats = set(x_)
	cats = list(sorted(cats))
	q = len(cats)
	res = np.zeros((n, q)).astype(int)

	for i in range(q):
		res[x_ == cats[i], q-1-i] = 1

	if len(res.shape) == 1:
		res = res[:, None]

	return res


def two_split_encoding(x):
	"""
	.. _two-split-encoding:
	Also known as binary encoding, the two-split encoding method turns categorical data taking :math:`q` distinct values into the ordinal data :math:`1, \\dots, q`, and then generates the binary representation of the ordinal data. 

	This encoding is more economical than one-hot encoding. Unlike the one-hot encoding methods which requires as many columns as the number of distinct inputs, two-split encoding only requires :math:`\\log_2 \\left\\lceil q \\right\\rceil` columns.

	Every bit in the encoding splits the set of all :math:`q` possible categories in 2 subsets of equal size (when q is a power of 2), and the value of the bit determines which subset the category of interest belongs to. Each bit generates a different partitioning of the set of all distinct categories.

	The ordinal value assigned to a categorical value is the order of its string representation among all :math:`q` distinct categorical values in the array.

	In other words, while a bit in one-hot encoding determines whether the category is equal to a specific values, a bit in the two-split encoding determines whether the category belong in one of two subsets of equal size.

	Parameters
	----------
	x: (n,) or (n, d) np.array
		Array of n inputs (typically strings) to encode and that take q distinct values.


	Returns
	-------
	 : (n, q) np.array
		Binary array representing the two-split encoding representation of the inputs.
	"""
	x_ = np.array([str(_) for _ in x]).astype(str)
	n = len(x_)
	cats = set(x_)
	cats = list(sorted(cats))
	q = len(cats)
	h = int(np.ceil(np.log2(q)))
	cats_map = {cats[i]: i for i in range(q)}
	res = np.array([[int(_) for _ in list(bin(cats_map[x_[i]])[2:].zfill(h))] for i in range(n)])

	if len(res.shape) == 1:
		res = res[:, None]

	return res



def prepare_data_for_mutual_info_analysis(x_c, x_d, y_c, y_d, non_monotonic_extension=True, \
		categorical_encoding='two-split', space='dual'):
	"""
	"""
	from kxy.api.core import scalar_continuous_entropy
	assert y_c is None or np.can_cast(y_c, float), 'y_c should be an array of numeric values'
	assert y_c is None or len(y_c.shape) == 1 or y_c.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'
	assert y_d is None or len(y_d.shape) == 1 or y_d.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'
	assert x_c is None or np.can_cast(x_c, float), 'x_c should represent draws from a continuous random variable.'
	assert x_c is not None or x_d is not None, 'Both x_c and z_c cannot be None'
	assert y_c is not None or y_d is not None, 'Both y_c and y_d cannot be None'
	assert y_c is None or y_d is None, 'Both y_c and y_d cannot be not None'

	res = {}
	to_stack = []
	# Ouput
	if y_d is None:
		# Regression
		y_ = y_c[:, None] if len(y_c.shape) == 1 else y_c
		to_stack += [y_.copy()]
		output_indices = [0]
		n_outputs = y_.shape[1]
		# if non_monotonic_extension and space=='dual':
		# 	to_stack += [np.abs(y_-np.nanmean(y_, axis=0))]
		# 	output_indices += [1]
		# 	n_outputs += y_.shape[1]

	if y_c is None:
		# Classification: threat the label like its encoded version
		y_ = one_hot_encoding(y_d) if categorical_encoding == 'one-hot' else two_split_encoding(y_d)
		to_stack += [y_.copy()]
		output_indices = [_ for _ in range(y_.shape[1])]
		n_outputs = y_.shape[1]
		# if non_monotonic_extension and space=='dual':
		# 	to_stack += [np.abs(y_-np.nanmean(y_, axis=0))]
		# 	output_indices += [_ for _ in range(y_.shape[1], 2*y_.shape[1])]
		# 	n_outputs += y_.shape[1]

	# Inputs
	n_inputs = 0
	n_features = 0

	x_c_ = None if x_c is None else x_c[:, None] if len(x_c.shape) == 1 else x_c
	x_d_ = None if x_d is None else x_d[:, None] if len(x_d.shape) == 1 else x_d
	batch_indices = []

	if x_c is not None:
		d = x_c_.shape[1]
		n_inputs += d
		to_stack += [x_c_]
		n_features += d
		if non_monotonic_extension and space=='dual':
			to_stack += [np.abs(x_c_-np.nanmean(x_c_, axis=0))]
			n_features += d
			batch_indices += [[i, i+d] for i in range(n_outputs, d+n_outputs)]

		else:
			batch_indices += [[i] for i in range(n_outputs, d+n_outputs)]

	z = np.hstack(to_stack)
	if x_d is not None:
		n_inputs += x_d_.shape[1]
		for j in range(x_d_.shape[1]):
			n_vars = z.shape[1]
			e = one_hot_encoding(x_d_[:, j]) if categorical_encoding == 'one-hot' else two_split_encoding(x_d_[:, j])
			dd = e.shape[1]
			# Threat the encoding of categorical variables as continuous
			z = e.copy() if z is None else np.hstack((z, e.copy()))
			if non_monotonic_extension and space=='dual':
				z = np.hstack((z, np.abs(e-np.nanmean(e, axis=0))))
				dd *= 2 
			n_features += dd
			batch_indices += [[_ for _ in range(n_vars, n_vars+dd)]]

	z = z.astype(float)

	# Constraints
	corr = pearson_corr(z) if space == 'primal' else spearman_corr(z)

	# Parameter validation
	assert len(batch_indices) == n_inputs
	assert corr.shape == (n_outputs+n_features, n_outputs+n_features)
	assert not np.isnan(corr).any()

	# Putting everything together
	res['corr'] = corr
	res['batch_indices'] = batch_indices
	res['output_indices'] = output_indices

	return res





