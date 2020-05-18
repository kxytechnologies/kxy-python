#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


from kxy.api.core import least_mixed_mutual_information, discrete_mutual_information, \
	least_mixed_conditional_mutual_information


def classification_difficulty(x_c, y, x_d=None, space='dual'):
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
	return 1./least_mixed_mutual_information(x_c, y, x_d=x_d, space=space, non_monotonic_extension=False)



def classification_feasibility(x_c, y, x_d=None, space='dual'):
	"""
	.. _classification-feasibility:
	Estimates how feasible a classification problem is, using as 
	metric the mutual information between features and output.

	.. math::
		I\\left({x_c, x_d}, y\\right)


	Parameters
	----------
	x_c : (n, d) np.array or None
		n i.i.d. draws from the continuous data generating distribution, or None if there no continuous
		features. x_c and x_d cannot be both None.
	x_d : (n,) np.array or None (default)
		n i.i.d. draws from the discrete data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n,) np.array
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

	return least_mixed_mutual_information(x_c, y, x_d=x_d, space=space, non_monotonic_extension=True)




def classification_input_incremental_importance(x_c, y, z_c, x_d=None, z_d=None, space='dual'):
	"""
	.. _classification-input-incremental-importance:
	Quantifies the value of adding input :math:`x=(x_c, x_d)` to inputs :math:`z=(z_c, z_d)` for forecasting :math:`y` as the conditional 
	mutual information :math:`I(y; x|z)`.


	Parameters
	----------
	x_c : (n, d) np.array
		n i.i.d. draws from the generating distribution of candidate continuous inputs.
	z_c : (n, d) np.array
		n i.i.d. draws from the generating distribution of existing continuous conditions.
	x_d : (n, d) np.array or None (default), optional
		n i.i.d. draws from the generating distribution of candidate categorical inputs.
	z_d : (n, d) np.array or None (default), optional
		n i.i.d. draws from the generating distribution of existing categorical conditions.
	y : (n,) np.array
		n i.i.d. draws from the (categorical) labels generating distribution, sampled
		jointly with x.


	Returns
	-------
	i : float
		The incremental importance.


	.. seealso:: 

		:ref:`kxy.classification.pre_learning.classification_feasibility <classification-feasibility>`
	"""
	assert len(y.shape) == 1 or y.shape[1] == 1, 'y should be a one dimensional numpy array'

	x_ = np.reshape(x_c, (len(x_c), 1)) if len(x_c.shape) == 1 else x_c.copy()
	z_ = np.reshape(z_c, (len(z_c), 1)) if len(z_c.shape) == 1 else z_c.copy()
	cmi = least_mixed_conditional_mutual_information(x_, y, z_, x_d=x_d, z_d=z_d, space=space, non_monotonic_extension=True)

	return cmi




