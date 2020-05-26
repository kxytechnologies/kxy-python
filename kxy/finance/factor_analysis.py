#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .risk_analysis import information_adjusted_correlation


def information_adjusted_beta(r, r_m):
	"""
	.. _information-adjusted-beta:
	Calculates the informmation-adjusted beta.


	.. note::

		The standard beta coefficient of an asset or a portfolio is defined by the CAPM model

		.. math::

			r = \\alpha + r_f + \\beta (r_m-r_f) + \\epsilon

		where :math:`r_f` is a deterministic risk-free rate, :math:`r_m` represents market returns, and
		:math:`\\epsilon` an idiosyncratic noise term. It follows that

		.. math::
			\\beta = \\text{Corr}\\left(r, r_m\\right) \\sqrt{\\frac{\mathbb{V}\\text{ar}\\left(r\\right)}{\mathbb{V}\\text{ar}\\left(r_m\\right)}},

		where :math:`\\text{Corr}` is Pearson's correlation coefficient. 

		The information-adjusted correlation generalizes the foregoing equations and reads 

		.. math::
			\\text{IA}\\beta = \\text{IACorr}\\left(r, r_m\\right) \\sqrt{\\frac{\mathbb{V}\\text{ar}\\left(r\\right)}{\mathbb{V}\\text{ar}\left(r_m\\right)}}.

		While Pearson's correlation (and therefore beta) only captures linear relationships between portfolio returns and market returns, the information-adjusted correlation fully captures nonlinear and temporal dependencies between portfolio returns and market returns.


	.. seealso::

		:ref:`kxy.finance.risk_analysis.information_adjusted_correlation <information-adjusted-correlation>`


	Parameters
	----------
	r : (n,) or (n,d) np.array
		The array of asset(s) or portfolio(s) returns.
	r_m : (n,) np.array
		The array of market returns.


	Returns
	-------
	c : float
		The information-adjusted of the asset or portfolio.

	Raises
	------
	AssertionError
		If any returns array is not one-dimensional.
	"""
	assert len(r_m.shape)==1 or r_m.shape[1]==1, 'r_m should be one-dimensional'

	return information_adjusted_correlation(r_m, r).flatten() * np.sqrt(r.var(axis=0)/r_m.var())

