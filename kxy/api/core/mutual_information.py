#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import requests

from kxy.api import APIClient, solve_copula_sync

from .utils import spearman_corr, pearson_corr
from .entropy import least_structured_copula_entropy, least_structured_continuous_entropy, \
	least_structured_mixed_entropy, discrete_entropy


def least_total_correlation(x, space='dual'):
	""" 
	.. _least-total-correlation:
	Estimates the smallest total correlation of a continuous d-dimensional random vector,
	that is consistent with the average pairwise Spearman rank correlation estimated 
	from the input sample.

	.. math::
		TC(x) := \sum_{i=1}^d h(x_i)-h(x) = -h(u).

	This is the negative of :ref:`kxy.api.core.entropy.least_structured_copula_entropy <least-structured-copula-entropy>`.

	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the data generating distribution.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.

	Returns
	-------
	tc : float
		The smallest total correlation consistent with observed empirical evidence, in nats.

	Raises
	------
	AssertionError
		If x is not a two dimensional array.
	"""
	assert len(x.shape) == 2
	return -least_structured_copula_entropy(x, space=space)



def least_continuous_mutual_information(x, y, space='dual', non_monotonic_extension=True):
	"""
	.. _least-continuous-mutual-information:
	Estimates the mutual information between a :math:`d`-dimensional random vector :math:`x` of inputs
	and a continuous scalar random variable :math:`y` (or label), assuming the least amount of 
	structure in :math:`x` and :math:`(x, y)`, other than what is necessary to be consistent with some 
	observed properties.

	.. note::

		.. math::
			I(x, y) &= h(x) + h(y) - h(x, y) \\

			        &= h\\left(u_x\\right) - h\\left(u_{x,y}\\right)
		
		where :math:`u_x` (resp. :math:`u_{x,y}`) is the copula-uniform representation of :math:`x`
		(resp. :math:`(x, y)`).

		We use as model for :math:`u_{x, y}` the least structured copula that is consistent with 
		the Spearman rank correlation matrix of the joint vector :math:`(x, y)`. 

	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the features generating distribution.
	y : (n,) np.array
		n i.i.d. draws from the (continuous) labels generating distribution, sampled
		jointly with x.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.

	Returns
	-------
	i : float
		The mutual information between x and y, **in nats**.

	Raises
	------
	AssertionError
		If y is not a one dimensional array or x or y are not numeric.
	"""
	assert np.can_cast(x, float) and np.can_cast(y, float), 'x and y should represent draws from continuous random variables.'
	assert len(y.shape) == 1 or y.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'

	x_ = np.reshape(x, (len(x), 1)) if len(x.shape) == 1 else x.copy()
	y_ = np.reshape(y, (len(y), 1)) if len(y.shape) == 1 else y.copy()

	corr = pearson_corr(np.hstack([y_, x_])) if space == 'primal' or not non_monotonic_extension else \
		spearman_corr(np.hstack([y_, x_, np.abs(x_-x_.mean(axis=0))]))
	mi = solve_copula_sync(corr, mode='mutual_information_v_output', output_index=0, solve_async=False, \
		space=space)

	return mi



def least_continuous_conditional_mutual_information(x, y, z, space='dual', non_monotonic_extension=True):
	"""
	.. _least-continuous-conditional-mutual-information:
	Estimates the conditional mutual information between a :math:`d`-dimensional random vector :math:`x` of inputs
	and a continuous scalar random variable :math:`y` (or label), conditional on a third continuous random 
	variable :math:`z`, assuming the least amount of structure in :math:`(x, y, z)`, other than what is 
	necessary to be consistent with some observed properties.

	.. note::

		.. math::
			I(x; y|z) &= h(x|z) + h(y|z) - h(x, y|z) \\

			        &= h\\left(u_x | u_z \\right) - h\\left(u_{x,y} | u_z \\right)

			        &= I(y; x, z) - I(y; z)

		
		where :math:`u_x` (resp. :math:`u_{x,y}`, :math:`u_z`) is the copula-uniform representation of :math:`x`
		(resp. :math:`(x, y)`, :math:`z`).

		We use as model for :math:`u_{x, y, z}` the least structured copula that is consistent with 
		maximum entropy constraints in the chosen space. 

	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the inputs generating distribution.
	y : (n,) np.array
		n i.i.d. draws from the (continuous) labels generating distribution, sampled
		jointly with x.
	z : (n,) np.array
		n i.i.d. draws from the (continuous) labels generating distribution, sampled
		jointly with z.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.

	Returns
	-------
	i : float
		The mutual information between x and y, **in nats**.

	Raises
	------
	AssertionError
		If y is not a one dimensional array or x, y and z are not all numeric.
	"""
	assert np.can_cast(x, float) and np.can_cast(y,  float) and np.can_cast(z, float), \
		'x, y and z should represent draws from continuous random variables.'
	assert len(y.shape) == 1 or y.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'

	x_ = np.reshape(x, (len(x), 1)) if len(x.shape) == 1 else x.copy()
	y_ = np.reshape(y, (len(y), 1)) if len(y.shape) == 1 else y.copy()
	z_ = np.reshape(z, (len(z), 1)) if len(z.shape) == 1 else z.copy()

	mi_y_xz = least_continuous_mutual_information(np.hstack([x_, z_]), y_, space=space, \
		non_monotonic_extension=non_monotonic_extension)
	mi_y_z = least_continuous_mutual_information(z_, y_, space=space, \
		non_monotonic_extension=non_monotonic_extension)

	return max(mi_y_xz-mi_y_z, 0.0)




def least_mixed_mutual_information(x_c, y, x_d=None, space='dual', non_monotonic_extension=True):
	"""
	.. _least-mixed-mutual-information:
	Estimates the mutual inforrmation between some features (a d-dimensional random vector
	:math:`x_c` and possibly a discrete feature variable :math:`x_d` and a discrete output/label 
	random variable.	

	.. note::

		.. math::
			I({x_c, x_d}; y) &= h(y) + h(x_c, x_d) - h(x_c, x_d, y) \\

							 &= h(x_c, x_d) - h\\left(x_c, x_d \\vert y\\right) \\

							 &= h(x_c, x_d) - E\\left[h\\left(x_c, x_d \\vert y=.\\right)\\right] \\

							:&= h(x_c, x_d) - \sum_{j=1}^q \mathbb{P}(y=j)  h\\left(x_c, x_d \\vert y=j \\right)

		
		When there are discrete features:

		* :math:`h(x_c, x_d)` is estimated using :ref:`kxy.api.core.entropy.least_structured_mixed_entropy <least-structured-mixed-entropy>`,
		* :math:`\mathbb{P}(y=j)` is estimated using frequencies, 
		* :math:`h\\left(x_c, x_d | y=j\\right)` is estimated using :ref:`kxy.api.core.entropy.least_structured_mixed_entropy <least-structured-mixed-entropy>`.

		When there are no discrete features, :math:`x_d` can simply be removed from the equations above, and :ref:`kxy.api.core.entropy.least_structured_continuous_entropy <least-structured-continuous-entropy>` is used for estimation instead of :ref:`kxy.api.core.entropy.least_structured_mixed_entropy <least-structured-mixed-entropy>`.

	Parameters
	----------
	x_c : (n, d) np.array
		n i.i.d. draws from the continuous data generating distribution.
	x_d : (n,) np.array or None (default)
		n i.i.d. draws from the discrete data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n,) np.array
		n i.i.d. draws from the (discrete) labels generating distribution, sampled jointly with x.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.

	Returns
	-------
	i : float
		The mutual information between features and discrete labels, **in nats**.

	Raises
	------
	AssertionError
		If y is not a one-dimensional array, or x_c is not an array of numbers.
	"""
	assert np.can_cast(x_c, float), 'x_c should represent draws from a continuous random variable.'
	assert len(y.shape) == 1 or y.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'

	x_c_ = np.reshape(x_c, (len(x_c), 1)) if len(x_c.shape) == 1 else x_c.copy()
	x_c_r = np.hstack((x_c_, np.abs(x_c_-x_c_.mean(axis=0)))) if non_monotonic_extension else x_c_

	categories = list(set(list(y)))
	n = y.shape[0]
	probas = np.array([1.*len(y[y==cat])/n for cat in categories])

	h = least_structured_continuous_entropy(x_c_r, space=space) if x_d is None else least_structured_mixed_entropy(x_c_r, x_d, space=space)
	wh = np.sum([probas[i] * least_structured_continuous_entropy(x_c_r[y==categories[i]], space=space) for i in range(len(categories))]) if x_d is None \
		else np.sum([probas[i] * least_structured_mixed_entropy(x_c_r[y==categories[i]], x_d[y==categories[i]], space=space) for i in range(len(categories))])

	return max(h-wh, 0.0)




def least_mixed_conditional_mutual_information(x_c, y, z_c, x_d=None, z_d=None, space='dual', non_monotonic_extension=True):
	"""
	.. _least-mixed-conditional-mutual-information:
	Estimates the conditional mutual information between a dimensional random vector :math:`x` of inputs
	and a discrete scalar random variable :math:`y` (or label), conditional on a third random 
	variable :math:`z`, assuming the least amount of structure in :math:`(x, y, z)`, other than what is 
	necessary to be consistent with some observed properties.

	.. math::
		I(y; x_c, x_d | z_c, z_d) = I(y; x_c, x_d, z_c, z_d) - I(y; z_c, z_d)

		
	Parameters
	----------
	x_c : (n, d) np.array
		n i.i.d. draws from the generating distribution of continuous inputs.
	z_c : (n, d) np.array
		n i.i.d. draws from the generating distribution of continuous conditions.
	x_d : (n, d) np.array or None (default), optional
		n i.i.d. draws from the generating distribution of categorical inputs.
	z_d : (n, d) np.array or None (default), optional
		n i.i.d. draws from the generating distribution of categorical conditions.
	y : (n,) np.array
		n i.i.d. draws from the (categorical) labels generating distribution, sampled
		jointly with x.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


	Returns
	-------
	i : float
		The mutual information between (x_c, x_d) and y, conditional on (z_c, z_d), **in nats**.

	Raises
	------
	AssertionError
		If y is not a one dimensional array or x and z are not all numeric.
	"""
	assert x_c is not None or z_c is not None, 'Both x_c and z_c cannot be None'
	assert x_c is not None or x_d is not None, 'Both x_c and x_d cannot be None'
	assert z_c is not None or z_d is not None, 'Both z_c and z_d cannot be None'
	if z_c is None and z_d is not None and x_c is not None:
		assert np.can_cast(x_c, float), 'x_c should be continuous'
		# I(y; x_c | z_d, x_d) = \sum_i p_i I(y; x_c | x_d, z_d = i)
		ds = z_d if x_d is None else np.hstack((x_d, z_d))
		ds = np.array(['*_*'.join(list(_)) for _ in ds])
		ds_cats, ds_counts = np.unique(ds)
		ds_probas = ds_counts/ds.shape[0]
		cmi = np.sum([ds_probas[i]*least_mixed_mutual_information(\
						x_c[ds==ds_cats[i]], y[ds==ds_cats[i]].flatten(), \
						x_d=None, space=space, non_monotonic_extension=non_monotonic_extension) \
						for i in range(ds_cats.shape[0])])

		return max(cmi, 0.0)


	if x_c is None and x_d is not None and z_c is not None:
		assert np.can_cast(z_c, float), 'z_c should be continuous'
		# I(y; x_d | z_c, z_d) = I(y; x_d, z_c, z_d) - I(y; z_c, z_d)
		ds = x_d if z_d is None else np.hstack((x_d, z_d))
		ds = np.array(['*_*'.join(list(_)) for _ in ds])
		cmi = least_mixed_mutual_information(\
						z_c, y.flatten(), \
						x_d=ds, space=space, \
						non_monotonic_extension=non_monotonic_extension)

		cmi -= least_mixed_mutual_information(\
						z_c, y.flatten(), \
						x_d=np.array(['*_*'.join(list(_)) for _ in x_d]), space=space, \
						non_monotonic_extension=non_monotonic_extension)

		return max(cmi, 0.0)

	# I(y; x_c, x_d | z_c, z_d) = I(y; x_c, x_d, z_c, z_d) - I(y; z_c, z_d)
	assert np.can_cast(x_c, float) and np.can_cast(z_c, float), 'z_c should be continuous'
	ds = None if x_d is None and z_d is None else x_d if z_d is None else z_d if x_d is None else np.hstack((x_d, z_d))
	ds = None if ds is None else np.array(['*_*'.join(list(_)) for _ in ds])

	x_c_ = x_c[:, None] if len(x_c.shape) == 1 else x_c
	z_c_ = z_c[:, None] if len(z_c.shape) == 1 else z_c
	cs = np.hstack((x_c_, z_c_))

	cmi = least_mixed_mutual_information(\
					cs, y.flatten(), \
					x_d=ds, space=space, \
					non_monotonic_extension=non_monotonic_extension)

	cmi -= least_mixed_mutual_information(\
					z_c, y.flatten(), \
					x_d=z_d, space=space, \
					non_monotonic_extension=non_monotonic_extension)

	return max(cmi, 0.0)




def discrete_mutual_information(x, y):
	"""
	.. _discrete-mutual-information:
	Estimates the (Shannon) mutual information between two discrete random variables from
	i.i.d. samples, using the plug-in estimator of Shannon entropy.

	Parameters
	----------
	x : (n,) np.array or None (default)
		n i.i.d. draws from a discrete distribution.
	y : (n,) np.array
		n i.i.d. draws from another discrete distribution, sampled jointly with x.

	Returns
	-------
	i : float
		The mutual information between x and y, **in nats**.

	Raises
	------
	AssertionError
		If y or x is not a one-dimensional array, or if x and y have different shapes.
	"""
	assert len(y.shape) == 1 or y.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'
	assert len(x.shape) == 1 or x.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'
	assert x.shape[0] == y.shape[0], 'Both arrays should have the same dimension.'

	hx = discrete_entropy(x)
	hy = discrete_entropy(y)
	hxy = discrete_entropy(np.array([str(x[i]) + '*_*' + str(y[i]) for i in range(x.shape[0])]))

	return max(0., hx+hy-hxy)


