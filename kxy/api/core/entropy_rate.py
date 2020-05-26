#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings('ignore', module='statsmodels.tsa')
from statsmodels.tsa.api import VAR, AR

from kxy.api import robust_pearson_corr_from_spearman
from .utils import robust_log_det



def pearson_acovf(sample, max_order=10, robust=False):
	"""
	Estimate the sample Pearson autocovariance function of a scalar-valued or vector-valued discrete-time
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

	if robust:
		# Use ranks as we are estimating Spearman's rank autocorrelation first,
		# before mapping it back to Pearson's autocorrelation.
		original_stds = np.std(x, axis=0)
		x = 1+x.argsort(axis=0).argsort(axis=0)

	mean = np.mean(x, axis=0)
	demeaned_x = x - mean

	acf = [np.einsum('ij, ik->jk', demeaned_x, demeaned_x) / T]
	acf += [np.einsum('ij, ik->jk', demeaned_x[h:, :], demeaned_x[:-h, :]) / T for h in range(1, max_order)]
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
	Estimates the sample autocovariance function of a vector-value process :math:`\\{x_t\\}` up to lag p (starting from 0). 


	Parameters
	----------
	sample: (T, d) np.array 
		Array of T sample observations of a d-dimensional process.
	p : int
		Number of lags to compute for the autocovariance function.


	Returns
	-------
	ac : (dp, dp)
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



def gaussian_var_copula_entropy_rate(sample, p=None, robust=False):
	"""
	Estimate the entropy rate of the copula-uniform dual representation of a stationary Gaussian VAR(p) (or AR(p)) process from a sample path.

	We recall that the copula-uniform representation of a :math:`\\mathbb{R}^d`-valued process :math:`\\{x_t\\} := \\{(x_{1t}, \\dots, x_{dt}) \\}`
	is, by definition, the process :math:`\\{ u_t \\} := \\{ \\left( F_{1t}\\left(x_{1t}\\right), \\dots, F_{dt}\\left(x_{dt}\\right) \\right) \\}` 
	where :math:`F_{it}` is the cummulative density function of :math:`x_{it}`.

	It can be shown that 

	.. math::
		h\\left( \\{ x_t \\}\\right) = h\\left( \\{ u_t \\}\\right) + \\sum_{i=1}^d h\\left( x_{i*}\\right) 

	where :math:`\\left(x_{i*}\\right)` is the entropy of the i-th coordinate process at any time.



	Parameters
	----------
	sample: (T, d) np.array 
		Array of T sample observations of a d-dimensional process.
	p : int or None
		Number of lags to compute for the autocovariance function. If p=None (the default), it is inferred by fitting a VAR model on the sample, using the Hannan-Quinn information criterion.
	robust: bool
		If True, the Pearson autocovariance function is estimated by first estimating a Spearman rank correlation, and then inferring the equivalent Pearson autocovariance function, under the Gaussian assumption.


	Returns
	-------
	h : float
		The entropy rate of the copula-uniform dual representation of the input process.
	p : int
		Order of the VAR(p)
	"""
	if p == None:
		# Fit an AR and use the fitted p.
		max_lag = int(round(12*(sample.shape[0]/100.)**(1/4.)))
		if len(sample.shape) == 1 or sample.shape[1] == 1:
			m = AR(sample)
			p = m.fit(ic='hqic').k_ar
		else:
			m = VAR(sample)
			p = m.fit(ic='hqic').k_ar

	x = sample if len(sample.shape) > 1 else x[:, None]
	res = -np.sum(0.5*np.log(2.*np.pi*np.e*np.var(x, axis=0)))
	res += gaussian_var_entropy_rate(x, p, robust=robust)

	return res, p


