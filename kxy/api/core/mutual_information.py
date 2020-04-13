#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file leverages the KXY API to evaluate mutual information and related measure derived
from solving mmaximum entropy copula problems under concordance measures. All optimization 
problems are solved by the KXY API and users require an API key  that should be set in the 
environment variable KXY_API_KEY.
"""
import numpy as np

from .utils import avg_pairwise_spearman_corr
from .entropy import least_structured_copula_entropy, least_structured_continuous_entropy, \
	least_structured_mixed_entropy, max_ent_copula_entropy, discrete_entropy


def least_total_correlation(x):
	""" 
	.. _least-total-correlation:
	Estimates the smallest total correlation of a continuous d-dimensional random vector,
	that is consistent with the average pairwise Spearman rank correlation estimated 
	from the input sample.

	.. math::
		TC(x) :&= \sum_{i=1}^d h(x_i)-h(x) \\
		       &= -h(u).

	This is the negative of :ref:`least-structured-copula-entropy`.

	Parameters
	----------
	x : (n, d) array_like
		n i.i.d. draws from the data generating distribution.

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
	return -least_structured_copula_entropy(x)



def least_continuous_mutual_information(x,  y):
	"""
	.. _least-continuous-mutual-information:
	Estimates the mutual inforrmation between a d-dimensional random vector :math:`x` of features
	and a continuous scalar random variable :math:`y` (or label), assuming the least amount of 
	structure in :math:`x` and :math:`(x, y)`, other than what is necessary to be consistent with some 
	observed concordance measures. 

	.. math::
		I(x, y) &= h(x) + h(y) - h(x, y)
				&= h\left(u_x\right) - h\left(u_{x,y}\right)
	
	where :math:`u_x` (resp. :math:`u_{x,y}`) is the copula-uniform representation of :math:`x`
	(resp. :math:`(x, y)`).

	We use as model for :math:`u_x` the least structured copula that is consistent with with the
	average pairwise Spearman rank correlation between coordinates of :math:`x`. 

	We use as model for :math:`u_{x, y}` the least structured copula that is consistent with both 
	the average pairwise Spearman rank correlation between coordinates of :math:`x`, and the 
	average pairwise Spearman rank correlation between coordinates of :math:`(x, y)`.

	Parameters
	----------
	x : (n, d) array_like
		n i.i.d. draws from the features generating distribution.
	y : (n) array_like
		n i.i.d. draws from the (continuous) laels generating distribution, sampled
		jointly with x.

	Returns
	-------
	i : float
		The mutual information between x and y, in nats.

	Raises
	------
	AssertionError
		If x is not a two dimensional array.


	"""
	assert np.can_cast(x, float) and np.can_cast(y,  float), 'x and y should represent draws from continuous random variables.'
	assert len(y.shape) == 1 or y.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'

	x_ = np.reshape(x, (len(x), 1)) if len(x.shape) == 1 else x.copy()
	y_ = np.reshape(y, (len(y), 1)) if len(y.shape) == 1 else y.copy()

	if len(x.shape) == 1 or x.shape[1] == 1:
		return least_total_correlation(np.hstack([x_, y_]))

	rho_features = avg_pairwise_spearman_corr(x)
	rho_full = avg_pairwise_spearman_corr(np.hstack([x, y_]))
	d = x.shape[1]

	hux = max_ent_copula_entropy(method='average-pairwise-spearman-rho', \
		rho=rho_features, d=d)

	huxy = max_ent_copula_entropy(method='average-pairwise-spearman-rho-1vd', \
		rho_full=rho_full, rho_features=rho_features, d=d)

	return max(hux-huxy, 0.0)



def least_mixed_mutual_information(x_c, y, x_d=None):
	"""
	.. _least-mixed-mutual-information:
	Estimates the mutual inforrmation between some features (a d-dimensional random vector
	:math:`x_c` and possibly a discrete feature variable :math:`x_d`) and a discrete output/label 
	random variable.	

	.. math::
		I({x_c, x_d}, y) &= h(y) + h(x_c, x_d) - h(x_c, x_d, y) \\
						  &= h(x_c, x_d) - h\left(x_c, x_d | y\right) \\
						  &= h(x_c, x_d) - E\left[h\left(x_c, x_d | y=.\right)\right] \\
						  :&= h(x_c, x_d) - \sum_{j=1}^q \mathbb{P}(y=j) * h\left(x_c, x_d | y=j\right)

	
	Where there are discrete features, :math:`h(x_c, x_d)` is estimated using :ref:`least-structured-mixed-entropy`,
	:math:`\mathbb{P}(y=j)` is estimated using frequencies, :math:`h\left(x_c, x_d | y=j\right)` is estimated 
	using :ref:`least-structured-mixed-entropy`.

	When there are no discrete features, :math:`x_d` can simply be removed from the equations above, and 
	:ref:`least-structured-continuous-entropy` is used for estimation instead of :ref:`least-structured-mixed-entropy`.

	Parameters
	----------
	x_c : (n, d) array_like
		n i.i.d. draws from the continuous data generating distribution.
	x_d : (n) array_like or None (default)
		n i.i.d. draws from the discrete data generating distribution, jointly sampled with x_c, or None
		if there are no discrete features.
	y : (n) array_like
		n i.i.d. draws from the (discrete) labels generating distribution, sampled jointly with x.

	Returns
	-------
	i : float
		The mutual information between features and discrete labels, in nats.

	Raises
	------
	AssertionError
		If y is not a one-dimensional array, or x_c is not an array of numbers.
	"""
	assert np.can_cast(x_c, float), 'x_c should represent draws from a continuous random variable.'
	assert len(y.shape) == 1 or y.shape[1] == 1, 'Only one-dimensional outputs are supported for now.'

	categories = list(set(list(y)))
	n = y.shape[0]
	probas = np.array([1.*len(y[y==cat])/n for cat in categories])

	h = least_structured_continuous_entropy(x_c) if x_d is None else least_structured_mixed_entropy(x_c, x_d)
	h -= np.sum([probas[i] * least_structured_continuous_entropy(x_c[y==categories[i]]) for i in range(len(categories))]) if x_d is None \
		else np.sum([probas[i] * least_structured_mixed_entropy(x_c[y==categories[i]], x_d[y==categories[i]]) for i in range(len(categories))])

	return max(h, 0.0)



def discrete_mutual_information(x, y):
	"""
	Estimates the (Shannon) mutual information between two discrete random variables from
	i.i.d. samples.

	Parameters
	----------
	x : (n) array_like or None (default)
		n i.i.d. draws from a discrete distribution.
	y : (n) array_like
		n i.i.d. draws from another discrete distribution, sampled jointly with x.

	Returns
	-------
	i : float
		The mutual information between x and y, in nats.

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
	hxy = discrete_entropy([str(x[i]) + '*_*' + str(y[i]) for i in range(x.shape[0])])

	return max(0., hx+hy-hxy)







