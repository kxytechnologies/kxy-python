#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from kxy.api.core import least_continuous_mutual_information, least_continuous_conditional_mutual_information


def regression_difficulty(x, y):
	"""
	Estimates how difficult a regression problem is, using as 
	metric the inverse of the mutual information between features and output, :math:`1/I\\left(x, y\\right)`.

	The lower the mutual information (the higher its inverse), the more difficult
	it will be to predict :math:`y` using :math:`x`. The worst case scenario occurs 
	when :math:`y` and :math:`x` are statistically independent, in which case the difficulty
	is infinite. 

	The best case scenario occurs when :math:`y` is fully determined by :math:`x`,
	in which case the mutual information can be as high as :math:`+\\infty`, and the difficulty
	(its inverse) as low as 0.

	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the features generating distribution.
	y : (n,) np.array
		n i.i.d. draws from the (continuous) laels generating distribution, sampled
		jointly with x.

	Returns
	-------
	d : float (non-negative)
		The regression difficulty.


	.. seealso:: 

		:ref:`kxy.api.core.mutual_information.least_continuous_mutual_information <least-continuous-mutual-information>`
	"""
	return 1./least_continuous_mutual_information(x, y)



def regression_feasibility(x, y):
	"""
	.. _regression-feasibility:
	Estimates how feasible a regression problem is, using as 
	metric the mutual information between features and output, :math:`I\\left(x, y\\right)`.

	The lower the mutual information, the more difficult it will be to predict :math:`y` using :math:`x`. 
	The worst case scenario occurs when :math:`y` and :math:`x` are statistically independent, 
	in which case the feasibility is 0. 

	The best case scenario occurs when :math:`y` is fully determined by :math:`x`,
	in which case the mutual information can be as high as :math:`+\\infty`.

	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the features generating distribution.
	y : (n,) np.array
		n i.i.d. draws from the (continuous) laels generating distribution, sampled
		jointly with x.

	Returns
	-------
	d : float (non-negative)
		The regression feasibility.


	.. seealso:: 

		:ref:`kxy.api.core.mutual_information.least_continuous_mutual_information <least-continuous-mutual-information>`
	"""
	return least_continuous_mutual_information(x, y)



def regression_input_incremental_importance(x, y, z):
	"""
	.. _regression-input-incremental-importance:
	Quantifies the value of adding inputss :math:`x` to inputs :math:`z` for forecasting :math:`y` as the conditional 
	mutual information :math:`I(y; x|z)`.


	Parameters
	----------
	x : (n,d) np.array
		Candidate additional inputs.
	y : (n,) np.array
		Regression output.
	z : (n, d) np.array
		Existing inputs.


	Returns
	-------
	i : float
		The incremental importance.


	.. seealso:: 

		:ref:`kxy.regression.pre_learning.regression_feasibility <regression-feasibility>`
	"""
	assert len(y.shape) == 1 or y.shape[1] == 1, 'y should be a one dimensional numpy array'

	x_ = np.reshape(x, (len(x), 1)) if len(x.shape) == 1 else x.copy()
	z_ = np.reshape(z, (len(z), 1)) if len(z.shape) == 1 else z.copy()

	cmi = least_continuous_conditional_mutual_information(\
		np.hstack((x_, np.abs(x_-x_.mean(axis=0)))), y, \
		np.hstack((z_, np.abs(z_-z_.mean(axis=0)))))

	return cmi



