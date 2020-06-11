#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings('ignore', module='statsmodels.tsa')
from statsmodels.tsa.api import VAR, AR

from kxy.api import robust_pearson_corr_from_spearman
from .utils import robust_log_det
from .entropy import least_structured_copula_entropy



def pearson_acovf(sample, max_order=10, robust=False):
	"""
	Estimates the sample Pearson autocovariance function of a scalar-valued or vector-valued discrete-time
	stationary ergodic stochastic process :math:`\\{z_t\\}` from a single sample of size :math:`T`, 
	:math:`(\\hat{z}_1, \\dots, \\hat{z}_T)`.

	.. math::
		C(h) := \\frac{1}{T} \\sum_{t=1+h}^T (\\hat{z}_t - \\bar{z})(\\hat{z}_{t-h} - \\bar{z})^T
		
	with :math:`\\bar{z} := \\frac{1}{T} \\sum_{t=1}^T \\hat{z}_t`.


	Parameters
	----------
	sample: (T, d) np.array 
		Array of T sample observations of a d-dimensional process.
	max_order: int
		Maximum number of lags to compute for the autocovariance function.
	robust: bool
		If True, the Pearson autocovariance function is estimated by first estimating a Spearman rank correlation,
		and then inferring the equivalent Pearson autocovariance function, under the Gaussian assumption.

	Returns
	-------
	acf : (max_order, d, d) np.array
		Sample autocovariance function up to order max_order.
	"""
	x = sample.copy()
	T = x.shape[0]
	one_d = False
	if len(x.shape) < 2:
		x = x[:, None]
		one_d = True

	mask = np.isnan(x).copy()
	if robust:
		# Use ranks as we are estimating Spearman's rank autocorrelation first,
		# before mapping it back to Pearson's autocorrelation.
		original_stds = np.nanstd(x, axis=0)
		x = (1+x.argsort(axis=0).argsort(axis=0)).astype(float)
		np.copyto(x, np.nan, where=mask)


	non_nan_t = T - mask.astype(float).sum(axis=0)
	o = np.ones((x.shape[1], x.shape[1]))
	Ts = np.minimum(o*non_nan_t, (o.T*non_nan_t).T)
	mean = np.nanmean(x, axis=0)
	demeaned_x = x - mean
	np.copyto(demeaned_x, 0.0, where=mask)

	acf = [np.divide(np.einsum('ij, ik->jk', demeaned_x, demeaned_x), Ts)]
	acf += [np.divide(np.einsum('ij, ik->jk', demeaned_x[h:, :], demeaned_x[:-h, :]), Ts) for h in range(1, max_order)]
	acf = np.array(acf)

	if one_d:
		acf = acf.flatten()
		if robust:
			c = acf/acf[0]
			# c = robust_pearson_corr_from_spearman(c)
			acf = c*original_stds[0]

	else:
		if robust:
			istds = 1./np.sqrt(np.diag(acf[0]))
			for i in range(acf.shape[0]):
				c = ((acf[i]*istds).T*istds).T
				# c = robust_pearson_corr_from_spearman(c)
				acf[i] = ((c.copy()*original_stds).T*original_stds).T
			
	return acf



def estimate_pearson_autocovariance(sample, p, robust=False):
	"""
	.. _estimate-pearson-autocovariance:
	Estimates the sample autocovariance function of a vector-value process :math:`\\{x_t\\}` up to lag p (starting from 0). 


	Parameters
	----------
	sample: (T, d) np.array 
		Array of T sample observations of a d-dimensional process.
	p : int
		Number of lags to compute for the autocovariance function.
	robust: bool
		If True, the Pearson autocovariance function is estimated by first estimating a Spearman rank correlation,
		and then inferring the equivalent Pearson autocovariance function, under the Gaussian assumption.

	Returns
	-------
	ac : (dp, dp) np.array
		Sample autocovariance matrix whose ij block of size pxp is the covariance between :math:`x_{t+i}` and :math:`x_{t+j}`.
	"""
	T = sample.shape[0]
	d = 1 if len(sample.shape) < 2 else sample.shape[1]

	sample_acov = pearson_acovf(sample, max_order=p, robust=robust)

	# Toeplitz sample autocovariance
	ac = np.zeros((d*p, d*p))
	for j in range(0, d*p, d):
		for i in range(j, d*p, d):
			h = (i - j) // d
			ac[i:i + d, j:j + d] = sample_acov[h].copy()
			ac[j:j + d, i:i + d] = sample_acov[h].T.copy()

	return ac



def gaussian_var_entropy_rate(sample, p, robust=False):
	"""
	Estimates the entropy rate of a stationary Gaussian VAR(p) or AR(p), namely

	.. math::
		h\\left( \\{x_t\\} \\right) = \\frac{1}{2} \\log \\left( \\frac{|K_p|}{|K_{p-1}|} \\right) + \\frac{d}{2} \\log \\left( 2 \\pi e\\right)

	where  :math:`|K_p|` is the determinant of the lag-p autocovariance matrix corresponding to this process, from a sample
	path of size T.


	Parameters
	----------
	sample: (T, d) np.array 
		Array of T sample observations of a d-dimensional process.
	p : int
		Number of lags to compute for the autocovariance function.
	robust: bool
		If True, the Pearson autocovariance function is estimated by first estimating a Spearman rank correlation,
		and then inferring the equivalent Pearson autocovariance function, under the Gaussian assumption.


	.. seealso::
		* :ref:`estimate_pearson_autocovariance <estimate-pearson-autocovariance>`
		


	Returns
	-------
	h : float
		The entropy rate of the process.
	"""
	d = 1 if len(sample.shape) < 2 else sample.shape[1]
	gamma_p = estimate_pearson_autocovariance(sample, p+1, robust=robust)

	if p > 0:
		h = 0.5 * (robust_log_det(2. * np.pi * np.e * gamma_p[:, :]) -\
			robust_log_det(2. * np.pi * np.e * gamma_p[:-d, :-d]))
	else:
		h = 0.5 * robust_log_det(2. * np.pi * np.e * gamma_p[:d, :d])

	return h



def gaussian_var_copula_entropy_rate(sample, p=None, robust=False, p_ic='hqic'):
	"""
	Estimates the entropy rate of the copula-uniform dual representation of a stationary Gaussian VAR(p) (or AR(p)) process from a sample path.

	We recall that the copula-uniform representation of a :math:`\\mathbb{R}^d`-valued process :math:`\\{x_t\\} := \\{(x_{1t}, \\dots, x_{dt}) \\}`
	is, by definition, the process :math:`\\{ u_t \\} := \\{ \\left( F_{1t}\\left(x_{1t}\\right), \\dots, F_{dt}\\left(x_{dt}\\right) \\right) \\}` 
	where :math:`F_{it}` is the cummulative density function of :math:`x_{it}`.

	It can be shown that 

	.. math::
		h\\left( \\{ x_t \\}\\right) = h\\left( \\{ u_t \\}\\right) + \\sum_{i=1}^d h\\left( x_{i*}\\right) 

	where :math:`h\\left(x_{i*}\\right)` is the entropy of the i-th coordinate process at any time.



	Parameters
	----------
	sample: (T, d) np.array 
		Array of T sample observations of a :math:`d`-dimensional process.
	p : int or None
		Number of lags to compute for the autocovariance function. If :code:`p=None` (the default), it is inferred by fitting a VAR model on the sample, using as information criterion :code:`p_ic`.
	robust: bool
		If True, the Pearson autocovariance function is estimated by first estimating a Spearman rank correlation, and then inferring the equivalent Pearson autocovariance function, under the Gaussian assumption.
	p_ic : str
		The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`. 
		Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion), 'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). 
		Same as the 'ic' parameter of :code:`statsmodels.tsa.api.VAR`.
	Returns
	-------
	h : float
		The entropy rate of the copula-uniform dual representation of the input process.
	p : int
		Order of the VAR(p).
	"""
	_sample = sample[~np.isnan(sample).any(axis=1)] if len(sample.shape) > 1 else sample[~np.isnan(sample)]
	if p == None:
		# Fit an AR and use the fitted p.
		max_lag = int(round(12*(_sample.shape[0]/100.)**(1/4.)))
		if len(_sample.shape) == 1 or _sample.shape[1] == 1:
			m = AR(_sample)
			p = m.fit(ic=p_ic).k_ar
		else:
			m = VAR(_sample)
			p = m.fit(ic=p_ic).k_ar

	x = _sample if len(_sample.shape) > 1 else _sample[:, None]
	res = -np.sum(0.5*np.log(2.*np.pi*np.e*np.var(x, axis=0)))
	res += gaussian_var_entropy_rate(x, p, robust=robust)

	return res, p


def auto_predictability(sample, p=None, robust=False, p_ic='hqic', space='primal'):
	"""
	.. _auto-predictability:
	Estimates the measure of auto-predictability of a (vector-value) time series:

	.. math::

		\\mathbb{PR}\\left(\\{x_t \\} \\right)		:&= h\\left(x_* \\right) - h\\left( \\{ x_t \\} \\right) \\

													 &= h\\left(u_{x_*}\\right) - h\\left( \\{ u_{x_t} \\} \\right).


	Parameters
	----------
	sample: (T, d) np.array 
		Array of T sample observations of a :math:`d`-dimensional process.
	p : int or None
		Number of lags to compute for the autocovariance function. If :code:`p=None` (the default), it is inferred by fitting a VAR model on the sample, using as information criterion :code:`p_ic`.
	robust: bool
		If True, the Pearson autocovariance function is estimated by first estimating a Spearman rank correlation, and then inferring the equivalent Pearson autocovariance function, under the Gaussian assumption.
	p_ic : str
		The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`. 
		Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion), 'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). 
		Same as the 'ic' parameter of :code:`statsmodels.tsa.api.VAR`.

	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


	.. warning::

		This function only supports primal inference for now.

	"""
	assert space == 'primal'
	h_u = least_structured_copula_entropy(sample, space=space)
	h_u_r = gaussian_var_copula_entropy_rate(sample, p=p, robust=robust, p_ic=p_ic)[0]

	return max(h_u - h_u_r, 0.0)



