#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kxy.api.core import least_continuous_mutual_information


def regression_difficulty(x, y):
	"""
	Estimates how difficult a regression problem is, using as 
	metric the inverse of the mutual information between features and output.

	.. math::
		\frac{1}{I\left(x, y\right)}

	The lower the mutual information (the higher its inverse), the more difficult
	it will be to predict :math:`y` using :math:`x`. The worst case scenario occurs 
	when :math:`y` and :math:`x` are statistically independent, in which case the difficulty
	is infinite. 

	The best case scenario occurs when :math:`y` is fully determined by :math:`x`,
	in which case the mutual information can be as high as :math:`+\infty`, and the difficulty
	(its inverse) as low as 0.

	See also :ref:`least-continuous-mutual-information`.

	Parameters
	----------
	x : (n, d) array_like
		n i.i.d. draws from the features generating distribution.
	y : (n) array_like
		n i.i.d. draws from the (continuous) laels generating distribution, sampled
		jointly with x.

	Returns
	-------
	d : float (non-negative)
		The regression difficulty.
	"""
	return 1./least_continuous_mutual_information(x, y)



def regression_feasibility(x, y):
	"""
	.. _regression-feasibility:
	Estimates how feasible a regression problem is, using as 
	metric the mutual information between features and output.

	.. math::
		I\left(x, y\right)

	The lower the mutual information, the more difficult it will be to predict :math:`y` using :math:`x`. 
	The worst case scenario occurs when :math:`y` and :math:`x` are statistically independent, 
	in which case the feasibility is 0. 

	The best case scenario occurs when :math:`y` is fully determined by :math:`x`,
	in which case the mutual information can be as high as :math:`+\infty`.

	See also :ref:`least-continuous-mutual-information`.

	Parameters
	----------
	x : (n, d) array_like
		n i.i.d. draws from the features generating distribution.
	y : (n) array_like
		n i.i.d. draws from the (continuous) laels generating distribution, sampled
		jointly with x.

	Returns
	-------
	d : float (non-negative)
		The regression feasibility.
	"""
	return least_continuous_mutual_information(x, y)


