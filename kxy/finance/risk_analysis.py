#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from kxy.api.core import least_continuous_mutual_information


def information_adjusted_correlation(x, y, p=0):
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

		where :math:`h\\left( \\left\{ x_t \\right\} \\right)` is the entropy rate of the process :math:`\\left\{ x_t \\right\}`. Specifically,
		the mutual information rate is 0 if and only if the two processes are statistically independent, and in particular exhibit no
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


		Checkout `this blog post <https://medium.com/kxytechnologies/https-medium-com-pit-ai-technologies-the-black-swans-in-your-market-neutral-portfolios-part-1-e17fc18a42a7>`_ 
		for a discussion on the limits of Pearson correlation and beta, and `this other blog post <https://medium.com/kxytechnologies/https-medium-com-pit-ai-technologies-the-black-swans-in-your-market-neutral-portfolios-part-2-b5e8d691b214>`_
		for the introduction of information-adjusted correlation and beta as remedies.

	Parameters
	----------
	x : (n,) or (n, p) np.array
		n i.i.d. draws from a scalar or vector random variable.
	y : (n,) or (n, p) np.array
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
	y_ = y[:, None] if len(x.shape) == 1 else y
	squared = (x_.shape == y_.shape) and np.allclose(x_, y_)

	c = np.empty((x_.shape[1], y_.shape[1]))

	def f(args):
		i, j = args
		if (i == j) and np.allclose(x_[:, i], y_[:, j]):
			return ((i, j), 1.0)

		mi = least_continuous_mutual_information(x_[:, i], y_[:, j], non_monotonic_extension=False)
		p_corr = np.corrcoef(x_[:, i], y_[:, j])[0, 1]
		s = 1. if p_corr >= 0.0 else -1.
		res = s * np.sqrt(1.-np.exp(-2.*mi))

		return ((i, j), res)

	with ThreadPoolExecutor(max_workers=10) as p:
		args = [(i, j) for i in range(c.shape[0]) for j in range(c.shape[1])]
		for res in p.map(f, args):
			c[res[0][0], res[0][1]] = res[1]
			
	return c

