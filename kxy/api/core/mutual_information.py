#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import requests

from kxy.api import APIClient, mutual_information_analysis


from .utils import prepare_data_for_mutual_info_analysis
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


def least_mutual_information(x_c, x_d, y_c, y_d, space='dual', non_monotonic_extension=True, \
		categorical_encoding='two-split'):
	"""
	"""
	res = prepare_data_for_mutual_info_analysis(x_c, x_d, y_c, y_d, space=space, \
		non_monotonic_extension=non_monotonic_extension, categorical_encoding=categorical_encoding)
	output_indices = res['output_indices']
	corr = res['corr']
	batch_indices = res['batch_indices']

	mi_ana = mutual_information_analysis(corr, output_indices, space=space, batch_indices=batch_indices)

	if mi_ana is None:
		return None

	return mi_ana['mutual_information']


def least_continuous_mutual_information(x_c, y, x_d=None, space='dual', non_monotonic_extension=True, \
		categorical_encoding='two-split'):
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
	x_c : (n,) or (n, d) np.array
		n i.i.d. draws from the continuous inputs data generating distribution.
	x_d : (n, q) np.array or None (default)
		n i.i.d. draws from the discrete inputs data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n,) np.array
		n i.i.d. draws from the (continuous) labels generating distribution, sampled
		jointly with x.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
	categorical_encoding : str, 'one-hot' | 'two-split' (default)
		The encoding method to use to represent categorical variables. 
		See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.

	Returns
	-------
	i : float
		The mutual information between x and y, **in nats**.

	Raises
	------
	AssertionError
		When input parameters are invalid.
	"""
	return least_mutual_information(x_c, x_d, y, None, space=space, non_monotonic_extension=non_monotonic_extension, \
		categorical_encoding=categorical_encoding)



def least_continuous_conditional_mutual_information(x_c, y, z_c, x_d=None, z_d=None, space='dual',\
	non_monotonic_extension=True, categorical_encoding='two-split'):
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
	x_c : (n, d) np.array
		n i.i.d. draws from the continuous inputs data generating distribution.
	x_d : (n, q) np.array or None (default)
		n i.i.d. draws from the discrete inputs data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n,) np.array
		n i.i.d. draws from the (continuous) labels generating distribution, sampled
		jointly with x.
	z_c : (n, d) np.array
		n i.i.d. draws from the generating distribution of continuous conditions.
	z_d : (n, d) np.array or None (default), optional
		n i.i.d. draws from the generating distribution of categorical conditions.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
	categorical_encoding : str, 'one-hot' | 'two-split' (default)
		The encoding method to use to represent categorical variables. 
		See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.

	Returns
	-------
	i : float
		The mutual information between x and y, **in nats**.

	Raises
	------
	AssertionError
		If y is not a one dimensional array or x, y and z are not all numeric.
	"""
	assert np.can_cast(y, float), 'y should be an array of numeric values'
	assert len(y.shape) == 1 or y.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'
	assert x_c is None or np.can_cast(x_c, float), 'x_c should represent draws from a continuous random variable.'
	assert z_c is None or np.can_cast(z_c, float), 'x_c should represent draws from a continuous random variable.'
	assert x_c is not None or x_d is not None, 'Both x_c and z_c cannot be None'
	assert z_c is not None or z_d is not None, 'Both z_c and z_d cannot be None'

	x_c_ = None if x_c is None else x_c[:, None] if len(x_c.shape) == 1 else x_c
	z_c_ = None if z_c is None else z_c[:, None] if len(z_c.shape) == 1 else z_c
	x_d_ = None if x_d is None else x_d[:, None] if len(x_d.shape) == 1 else x_d
	z_d_ = None if z_d is None else z_d[:, None] if len(z_d.shape) == 1 else z_d

	# I(y; x_c, x_d | z_c, z_d) = I(y; x_c, x_d, z_c, z_d) - I(y; z_c, z_d)
	j_c = None if (x_c is None and z_c is None) else x_c_ if z_c is None else z_c_ if x_c is None else np.hstack((x_c_, z_c_)) # (x_c, z_c)
	j_d = None if (x_d is None and z_d is None) else x_d_ if z_d is None else z_d_ if x_d is None else np.hstack((x_d_, z_d_)) # (x_d, z_d)

	cmi = least_continuous_mutual_information(j_c, y, x_d=j_d, space=space, \
		non_monotonic_extension=non_monotonic_extension, categorical_encoding=categorical_encoding)
	cmi -= least_continuous_mutual_information(z_c, y, x_d=z_d, space=space, \
		non_monotonic_extension=non_monotonic_extension, categorical_encoding=categorical_encoding)

	return max(cmi, 0.0)




def least_mixed_mutual_information(x_c, y, x_d=None, space='dual', non_monotonic_extension=True, \
		categorical_encoding='two-split'):
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
		n i.i.d. draws from the continuous inputs data generating distribution.
	x_d : (n, q) np.array or None (default)
		n i.i.d. draws from the discrete inputs data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n,) np.array
		n i.i.d. draws from the (discrete) labels generating distribution, sampled jointly with x.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
	categorical_encoding : str, 'one-hot' | 'two-split' (default)
		The encoding method to use to represent categorical variables. 
		See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.


	Returns
	-------
	i : float
		The mutual information between features and discrete labels, **in nats**.

	Raises
	------
	AssertionError
		If y is not a one-dimensional array, or x_c is not an array of numbers.
	"""
	return least_mutual_information(x_c, x_d, None, y, space=space, non_monotonic_extension=non_monotonic_extension, \
		categorical_encoding=categorical_encoding)




def least_mixed_conditional_mutual_information(x_c, y, z_c, x_d=None, z_d=None, space='dual', \
	non_monotonic_extension=True, categorical_encoding='two-split'):
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
	x_d : (n, d) np.array or None (default), optional
		n i.i.d. draws from the generating distribution of categorical inputs.
	z_c : (n, d) np.array
		n i.i.d. draws from the generating distribution of continuous conditions.
	z_d : (n, d) np.array or None (default), optional
		n i.i.d. draws from the generating distribution of categorical conditions.
	y : (n,) np.array
		n i.i.d. draws from the (categorical) labels generating distribution, sampled
		jointly with x.
	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
	categorical_encoding : str, 'one-hot' | 'two-split' (default)
		The encoding method to use to represent categorical variables. 
		See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.

	Returns
	-------
	i : float
		The mutual information between (x_c, x_d) and y, conditional on (z_c, z_d), **in nats**.

	Raises
	------
	AssertionError
		If y is not a one dimensional array or x and z are not all numeric.
	"""
	assert x_c is not None or x_d is not None, 'Both x_c and z_c cannot be None'
	assert z_c is not None or z_d is not None, 'Both z_c and z_d cannot be None'
	assert x_c is None or np.can_cast(x_c, float), 'x_c should represent draws from a continuous random variable.'
	assert z_c is None or np.can_cast(z_c, float), 'x_c should represent draws from a continuous random variable.'

	x_c_ = None if x_c is None else x_c[:, None] if len(x_c.shape) == 1 else x_c
	z_c_ = None if z_c is None else z_c[:, None] if len(z_c.shape) == 1 else z_c
	x_d_ = None if x_d is None else x_d[:, None] if len(x_d.shape) == 1 else x_d
	z_d_ = None if z_d is None else z_d[:, None] if len(z_d.shape) == 1 else z_d

	# I(y; x_c, x_d | z_c, z_d) = I(y; x_c, x_d, z_c, z_d) - I(y; z_c, z_d)
	j_c = None if (x_c is None and z_c is None) else x_c_ if z_c is None else z_c_ if x_c is None else np.hstack((x_c_, z_c_)) # (x_c, z_c)
	j_d = None if (x_d is None and z_d is None) else x_d_ if z_d is None else z_d_ if x_d is None else np.hstack((x_d_, z_d_)) # (x_d, z_d)

	cmi = least_mixed_mutual_information(j_c, y.flatten(), x_d=j_d, space=space, \
		non_monotonic_extension=non_monotonic_extension, categorical_encoding=categorical_encoding)
	cmi -= least_mixed_mutual_information(z_c, y.flatten(), x_d=z_d, space=space, \
		non_monotonic_extension=non_monotonic_extension, categorical_encoding=categorical_encoding)

	return max(cmi, 0.0)



def discrete_mutual_information(x, y):
	"""
	.. _discrete-mutual-information:
	Estimates the (Shannon) mutual information between two discrete random variables from
	i.i.d. samples, using the plug-in estimator of Shannon entropy.

	Parameters
	----------
	x : (n,) or (n, d) np.array or None (default)
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
	assert x.shape[0] == y.shape[0], 'Both arrays should have the same dimension.'

	x_ = x.flatten() if (len(x.shape) == 1) or (x.shape[1] == 1) else np.array(['*_*'.join([str(_) for _ in x[i]]) for i in len(x)])

	hx = discrete_entropy(x_)
	hy = discrete_entropy(y)
	hxy = discrete_entropy(np.array([str(x_[i]) + '*_*' + str(y[i]) for i in range(x.shape[0])]))

	return max(0., hx+hy-hxy)


