#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kxy.api.core import least_mixed_mutual_information, discrete_mutual_information


def classification_difficulty(x_c, y, x_d=None):
	"""
	Estimates how difficult a classification problem is, using as 
	metric the inverse of the mutual information between features and output.

	.. math::
		\\frac{1}{I\\left({x_c, x_d}, y\\right)}

	Parameters
	----------
	x_c : (n, d) array_like
		n i.i.d. draws from the continuous data generating distribution.
	x_d : (n,) array_like or None (default)
		n i.i.d. draws from the discrete data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n,) array_like
		n i.i.d. draws from the (discrete) labels generating distribution, sampled jointly with x.

	Returns
	-------
	d : float
		The classification difficulty.


	.. seealso::  

		:ref:`kxy.api.core.mutual_information.least_mixed_mutual_information <least-mixed-mutual-information>`.
	"""
	return 1./least_mixed_mutual_information(x_c, y, x_d=x_d)



def classification_feasibility(x_c, y, x_d=None):
	"""
	.. _classification-feasibility:
	Estimates how feasible a classification problem is, using as 
	metric the mutual information between features and output.

	.. math::
		I\\left({x_c, x_d}, y\\right)


	Parameters
	----------
	x_c : (n, d) array_like or None
		n i.i.d. draws from the continuous data generating distribution, or None if there no continuous
		features. x_c and x_d cannot be both None.
	x_d : (n,) array_like or None (default)
		n i.i.d. draws from the discrete data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n,) array_like
		n i.i.d. draws from the (discrete) labels generating distribution, sampled jointly with x_c and/or x_d.


	Raises
	------
	AssertionError
		If x_d and x_c are both None.

	Returns
	-------
	d : float
		The classification feasibility.


	.. seealso:: 

		* :ref:`kxy.api.core.mutual_information.least_mixed_mutual_information <least-mixed-mutual-information>` 
		* :ref:`kxy.api.core.mutual_information.discrete_mutual_information <discrete-mutual-information>`
	"""
	assert x_d is not None or x_c is not None, "x_c and x_d cannot be both None."

	if x_c is None:
		return discrete_mutual_information(x_d, y)

	return least_mixed_mutual_information(x_c, y, x_d=x_d)


