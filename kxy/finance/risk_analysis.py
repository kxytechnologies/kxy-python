#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor
import numpy as np

import warnings
warnings.filterwarnings('ignore', module='statsmodels.tsa')
from statsmodels.tsa.api import VAR

from kxy.api.core import least_continuous_mutual_information, spearman_corr, \
	estimate_pearson_autocovariance, robust_log_det
from kxy.api import information_adjusted_correlation_from_spearman, \
	robust_pearson_corr_from_spearman


def information_adjusted_correlation(x, y=None, p=0):
	"""
	.. _information-adjusted-correlation:
	Calculates the information-adjusted correlation matrix between two arrays.

	.. note::

		Pearson's correlation coefficient quantifies the linear association between two random variables.
		Indeed, two random variables can be statistically dependent despite being decorrelated. Such dependence
		will typically materialize during tail events, the worst timing from a risk management perspective.


		The mutual information rate between two scalar time series provides an alternative that fully captures 
		linear and nonlinear, cross-sectional and temporal dependence:

		.. math::
			I\\left(\\left\{x_t\\right\}, \\left\{y_t\\right\}\\right) = h\\left( \\left\{x_t\\right\} \\right) +  h\\left( \\left\{y_t\\right\} \\right) - h\\left(\\left\{x_t, y_t\\right\}  \\right)

		where :math:`h\\left( \\left\{ x_t \\right\} \\right)` is the entropy rate of the process :math:`\\left\{ x_t \\right\}`. 

		Specifically, the mutual information rate is 0 if and only if the two processes are statistically independent, and in particular exhibit no
		cross-sectional or temporal dependence, linear or nonlinear.


		When :math:`\\left\{ x_t, y_t \\right\}` is Gaussian, stationary and memoryless, for instance when :math:`\\left(x_i, y_i \\right)`
		are assumed i.i.d Gaussian, the mutual entropy rate reads

		.. math::
			I\\left(\\left\{x_t\\right\}, \\left\{y_t\\right\}\\right) = -\\frac{1}{2} \log \\left(1- \\text{Corr}\\left(x_t, y_t\\right)^2 \\right).


		We generalize this formula and define the **information-adjusted correlation** as the quantity :math:`\\text{IACorr}\\left( \\left\{x_t\\right\}, \\left\{y_t\\right\} \\right)`
		so that the mutual information rate always reads

		.. math::
			I\\left(\\left\{x_t\\right\}, \\left\{y_t\\right\}\\right) = -\\frac{1}{2} \log \\left(1- \\text{IACorr}\\left( \\left\{x_t\\right\}, \\left\{y_t\\right\} \\right)^2 \\right),

		whether or not the time series are jointly Gaussian and memoryless.

		.. math::
			\\text{IACorr}\\left(\\left\{x_t\\right\}, \\left\{y_t\\right\}\\right) := \\text{sign}\\left( \\text{Corr}\\left(x_., y_.\\right) \\right)\\sqrt{1-e^{-2 I\\left(\\left\{x_t\\right\}, \\left\{y_t\\right\}\\right)}}

		where :math:`\\text{sign}(x)=1` if and only if :math:`x \geq 0` and :math:`-1` otherwise. Note that the information-adjusted correlation is 0
		if and only if the two time series are statistically independent, and in particular exhibit no cross-sectional or temporal dependencee.



	Parameters
	----------
	x : (n,) or (n, d) np.array
		n i.i.d. draws from a scalar or vector random variable.
	y : (n,) or (n, q) np.array
		n i.i.d. draws from a scalar or vector random variable jointly sampled with x.
	p : int
		The number of lags to use when generating Spearman rank auto-correlation to use 
		as empirical evidence in the maximum-entropy problem. The default value is 0, which 
		corresponds to assuming rows are i.i.d. This is also the only supported value for now.

	Returns
	-------
	c : np.array
		The information-adjusted correlation matrix between the two random variables. 

	Raises
	------
	AssertionError
		If p is different from 0. Higher values will be supported later.
	"""
	assert p==0, 'Only p=0 is supported for now'

	x_ = x[:, None] if len(x.shape) == 1 else x
	y_ = x_ if y is None else y[:, None] if len(y.shape) == 1 else y

	if p == 0:
		if (x_.shape == y_.shape) and np.allclose(x_, y_):
			corr = spearman_corr(x_)
			ia_corr = information_adjusted_correlation_from_spearman(corr)

		else:
			z = np.hstack([x_, y_])
			corr = spearman_corr(z)
			nx = x_.shape[1]
			ia_corr = information_adjusted_correlation_from_spearman(corr[:nx, nx:])
				
		return ia_corr




def robust_pearson_corr(x, y=None, p=0):
	"""
	.. _robust-pearson-corr:
	Computes a robust estimator of the Pearson correlation matrix 
	between :math:`x` and :math:`y` (or :math:`x` is :math:`y` is None) as the Pearson correlation matrix
	that is equivalent to the sample Spearman correlation matrix, assuming :math:`(x, y)` is jointly
	Gaussian.


	Parameters
	----------
	x : (n,) or (n, d) np.array
		n i.i.d. draws from a scalar or vector random variable.
	y : (n,) or (n, q) np.array
		n i.i.d. draws from a scalar or vector random variable jointly sampled with x.
	p : int
		The number of lags to use when generating Spearman rank auto-correlation. 
		The default value is 0, which corresponds to assuming rows are i.i.d. 


	Returns
	-------
	c : np.array
		The robust Pearson correlation matrix between the two random variables. 
	"""
	x_ = x[:, None] if len(x.shape) == 1 else x
	y_ = x_ if y is None else y[:, None] if len(y.shape) == 1 else y
	is_auto_corr = (x_.shape == y_.shape) and np.allclose(x_, y_)
	z_ = x_ if is_auto_corr else np.hstack([x_, y_])

	if p == 0:
		if is_auto_corr:
			corr = spearman_corr(x_)
			rb_corr = robust_pearson_corr_from_spearman(corr)

		else:
			corr = spearman_corr(z_)
			nx = x_.shape[1]
			rb_corr = robust_pearson_corr_from_spearman(corr[:nx, nx:])

		return rb_corr



	else:
		if (x_.shape == y_.shape) and np.allclose(x_, y_):
			corr = spearman_corr(x_)

		else:
			nx = x_.shape[1]
			corr = spearman_corr(z_)[:nx, nx:]

		rb_corr = np.zeros_like(corr)
		if p is None:
			m = VAR(z_)
			try:
				p = m.fit(ic='hqic').k_ar
			except:
				p = 1

		gamma_p = estimate_pearson_autocovariance(z_, p+1, robust=True)
		d = x_.shape[1]
		q = y_.shape[1]
		n = d if is_auto_corr else d+q

		def thread_mi_f(args):
			'''
			'''
			i, j = args
			if np.allclose(x_[:, i], y_[:, j]):
				return np.inf

			x_idx = list(i + np.arange(0, p+1).astype(int)*n)
			s = 0 if is_auto_corr else d
			y_idx = list(s + j + np.arange(0, p+1).astype(int)*n)
			xy_idx = sorted(x_idx + y_idx)

			gij_p = gamma_p[xy_idx].T[xy_idx].T
			huij = 0.5 * (robust_log_det(2. * np.pi * np.e * gij_p[:, :]) -\
						robust_log_det(2. * np.pi * np.e * gij_p[:-2, :-2]))

			gi_p = gamma_p[x_idx].T[x_idx].T
			hui = 0.5 * (robust_log_det(2. * np.pi * np.e * gi_p[:, :]) -\
						robust_log_det(2. * np.pi * np.e * gi_p[:-1, :-1]))

			gj_p = gamma_p[y_idx].T[y_idx].T
			huj = 0.5 * (robust_log_det(2. * np.pi * np.e * gj_p[:, :]) -\
						robust_log_det(2. * np.pi * np.e * gj_p[:-1, :-1]))

			return max(hui+huj-huij, 0.0)


		with ThreadPoolExecutor(max_workers=min(d*q, 100)) as pool:
			all_args = [(i, j) for j in range(q) for i in range(d)]
			results = [_ for _ in pool.map(thread_mi_f, all_args)]

		mis = np.zeros_like(corr)
		for k in range(len(all_args)):
			i, j = all_args[k]
			mis[i, j] = results[k]

		rb_corr = np.sqrt(1.-np.exp(-2.*mis))*np.sign(corr)

		return rb_corr

