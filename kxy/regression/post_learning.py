#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kxy.api.core import least_continuous_mutual_information


def regression_suboptimality(residuals, y):
	"""
	.. _regression-suboptimality:
	Evaluate how optimal a model is by calculating the mutual information
	between the model residuals and the regression target.

	.. math::
		I\left(\epsilon, y\right)

	A regression model is optimal when its residuals are statistically independent
	from the output y, or equivalently, when :math:`I\left(\epsilon, y\right)=0`.

	Mutual information between residuals and output indicate that the regression 
	model can be improved. The higher the mutual information, the more the model
	can be improved.

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
	d : float
		The regression difficulty.
	"""
	return least_continuous_mutual_information(residuals, y)

