#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .entropy_rate import gaussian_var_copula_entropy_rate


def least_continuous_mutual_information_rate(x, y, space='primal', robust=False, p=None, p_ic='hqic'):
	"""
	Estimate the maximum entropy mutual information rate between two scalar or vector valued time series.


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
	p_ic : str
		The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`. 
		Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion), 'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). 
		Same as the 'ic' parameter of :code:`statsmodels.tsa.api.VAR`.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson autocovariance constraints, leading to the Gaussian VAR.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank autocorrelation constraints.


	.. warning::
		Maximum entropy optimization in the copula-uniform dual space is not yet supported for time series. 
		

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

	huxy, q = gaussian_var_copula_entropy_rate(z_, p=p, robust=robust, p_ic=p_ic)
	hux, _ = gaussian_var_copula_entropy_rate(x_, p=q, robust=robust, p_ic=p_ic)
	huy, _ = gaussian_var_copula_entropy_rate(y_, p=q, robust=robust, p_ic=p_ic)

	return max(hux+huy-huxy, 0.0)


