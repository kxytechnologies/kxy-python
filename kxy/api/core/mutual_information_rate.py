#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .entropy_rate import gaussian_var_copula_entropy_rate


def least_continuous_mutual_information_rate(x, y, space='primal', robust=False, p=None):
	"""
	Estimate the maximum entropy mutual information rate between two scalar or vector valued time series.

	When space='primal' the maximum entropy problem is formulated in the original space, using as constraints the 
	Pearson autocovariance function. The solution is that (x, y) is a Gaussian Vector-Autoregressive process.

	When space='dual' the maximum entropy problem is formulated in the copula-uniform dual space, using as constraints
	the Spearman rank autocorrelation function.

	.. warning::
		Maximum entropy optimization in the copula-uniform dual space is not yet supported for time series. 


	Parameters
	----------
	x : (T, d) np.array
		Sample path from the first time series.
	y : (T, q) np.array
		Sample path from the second time series.
	space : str
		One of :code:`'primal'` and :code:`'dual'` to choose where the maximum entropy optimization problem should be formulated.
	robust : bool
		If True and :code:`space='primal'`, then the autocovariance function is estimated by computing the Spearman rank autocorrelation
		function, inferring the equivalent autocorrelation function assuming Gaussianity, and then scaling back with the sample variances.
	p : int or None
		The number of autocorrelation lags to use for the maximum entropy problem. If set to :code:`None` (the default) and if :code:`space` is
		:code:`primal`, then it is inferred by fitting a VAR model on the joint time series using the Hannan-Quinn information criterion.



	Returns
	-------
	i : float
		The mutual information rate between x and y, in nats.

	Raises
	------
	AssertionError
		If :code:`space` is not :code:`'primal'`.
	"""
	assert space == "primal", "Maximum-entropy optimization in the dual space is not yet implemented for time series"

	x_ = x if len(x.shape) > 1 else x[:, None]
	y_ = y if len(y.shape) > 1 else y[:, None]
	z_ = np.hstack((x_, y_))

	huxy, q = gaussian_var_copula_entropy_rate(z_, p=p, robust=robust)
	hux, _ = gaussian_var_copula_entropy_rate(x_, p=q, robust=robust)
	huy, _ = gaussian_var_copula_entropy_rate(y_, p=q, robust=robust)

	return max(hux+huy-huxy, 0.0)


