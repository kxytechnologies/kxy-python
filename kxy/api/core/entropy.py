#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import requests

import numpy as np
import scipy.special as spe
import statsmodels.api as sm

from kxy.api import APIClient, solve_copula_sync
from .utils import spearman_corr, pearson_corr


def _scalar_continuous_entropy(x, space='dual', method='gaussian-kde'):
	if space == 'primal' or method == 'gaussian':
		return 0.5*np.log(2.*np.pi*np.e*np.var(x))

	if method == '1-spacing':
		sorted_x = np.unique(x)
		n = sorted_x.shape[0]
		ent = np.sum(np.log(n*(sorted_x[1:]-sorted_x[:-1])))/n - spe.digamma(1)
		return ent

	if method == 'gaussian-kde':
		kde = sm.nonparametric.KDEUnivariate(x)
		kde.fit(kernel='gau')
		return kde.entropy



def scalar_continuous_entropy(x, space='dual', method='gaussian-kde'):
	"""
	.. _scalar-continuous-entropy:
	Estimates the (differential) entropy of a continuous scalar random variable.

	Multiple methods are supported:

	* :code:`'gaussian'` for Gaussian moment matching: :math:`h(x) = \\frac{1}{2} \\log\\left(2 \\pi e \\sigma^2 \\right)`.
	* :code:`'1-spacing'` for the standard 1-spacing estimator (see [1]_ and [2]_):

	.. math::
		h(x) \\approx - \\gamma(1) + \\frac{1}{n-1} \\sum_{i=1}^{n-1} \\log \\left[ n \\left(x_{(i+1)} - x_{(i)} \\right) \\right],

	where :math:`x_{(i)}` is the i-th smallest sample, and :math:`\\gamma` is `the digamma function. <https://en.wikipedia.org/wiki/Digamma_function>`_
	
	* :code:`'gaussian-kde'` (the default) for Gaussian kernel density estimation.

	.. math::
		h(x) \\approx \\frac{1}{n} \\sum_{i=1}^n \\log\\left( \\hat{p}\\left(x_i\\right) \\right)

	where :math:`\\hat{p}` is the Gaussian kernel density estimator of the true pdf using :code:`statsmodels.api.nonparametric.KDEUnivariate`.



	
	Parameters
	----------
	x : (n,) np.array
		i.i.d. samples from the distribution of interest.

	Returns
	-------
	h : float
		The (differential) entropy of the continuous scalar random variable of interest.

	Raises
	------
	AssertionError
		If the input has the wrong shape or the method is not supported.
		

	.. rubric:: References

	.. [1] Kozachenko, L. F., and Nikolai N. Leonenko. "Sample estimate of the entropy of a random vector." 
		Problemy Peredachi Informatsii 23.2 (1987): 9-16.

	.. [2] Beirlant, J., Dudewicz, E.J., Györfi, L., van der Meulen, E.C. "Nonparametric entropy estimation: an overview." 
		International Journal of Mathematical and Statistical Sciences. 6 (1): 17–40. (1997) ISSN 1055-7490. 
	"""
	assert len(x.shape) == 1 or x.shape[1] == 1, 'x should be a one dimensional numpy array'
	assert method in ('gaussian', '1-spacing', 'gaussian-kde')

	# Check if the distribution is a mixture of discrete and continuous
	categories, counts  = np.unique(x.flatten(), return_counts=True)
	n = x.shape[0]
	int_proba = 0.01
	exp_n_sample = n*int_proba
	std_n_sample = np.sqrt(n*(int_proba-int_proba**2))
	threshold = exp_n_sample + 5*std_n_sample

	discrete_part = categories[counts>threshold]
	continuous_part = categories[counts<=threshold]

	if len(discrete_part) == 0:
		return _scalar_continuous_entropy(x, space=space, method=method)


	if len(continuous_part) == 0:
		return discrete_entropy(discrete_part)


	ent_disc_part = discrete_entropy(discrete_part)
	ent_cont_part = _scalar_continuous_entropy(continuous_part, space=space, method=method)
	proba_discrete = 1.*len(discrete_part)/n
	proba_continuous = 1.-proba_discrete 
	ent_disc_cont = -proba_discrete*np.log(proba_discrete)-proba_continuous*np.log(proba_continuous)

	return ent_disc_part + ent_cont_part - ent_disc_cont



def discrete_entropy(x):
	"""
	.. _discrete-entropy:
	Estimates the (Shannon) entropy of a discrete random variable taking up to q distinct values 
	given n i.i.d samples,

	.. math::
		h(x) = - \\sum_{i=1}^q p_i \\log p_i,

	using the plug-in estimator.


	Parameters
	----------
	x : (n,) np.array
		i.i.d. samples from the distribution of interest.

	Returns
	-------
	h : float
		The (differential) entropy of the discrete random variable of interest.

	Raises
	------
	AssertionError
		If the input has the wrong shape.

	"""
	assert len(x.shape) == 1, 'Only flat numpy arrays are supported'
	categories = list(set(list(x)))
	n = len(x)
	probas = np.array([1.*len(x[x==cat])/n for cat in categories])

	return -np.dot(np.log(probas), probas)


def least_structured_copula_entropy(x, space='dual'):
	"""
	.. _least-structured-copula-entropy:
	Estimates the entropy of the maximum-entropy copula in the chosen space.

	.. note::
	
		This also corresponds to least amount of total correlation that is evidenced by our maximum-entropy constraints.

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
	h : float
		The (differential) entropy of the least structured copula consistent with maximum-entropy constraints.
	"""
	if len(x.shape) == 1 or x.shape[1] == 1:
		# By convention, the copula-dual representation of a 1d random variable is the uniform[0, 1].
		return 0.0

	corr = pearson_corr(x) if space == 'primal' else spearman_corr(x)
	h = solve_copula_sync(corr, mode='copula_entropy', solve_async=False, space=space)

	return h


def least_structured_continuous_entropy(x, space='dual'):
	""" 
	.. _least-structured-continuous-entropy:
	Estimates the entropy of a continuous :math:`d`-dimensional random variable under the least structured assumption for its copula. 

	When :math:`d>1`,

	.. math::
		h(x) = h(u) + \\sum_{i=1}^d h(x_i).

	* :math:`h(u)` is estimated using :ref:`least_structured_copula_entropy <least-structured-copula-entropy>`.
	* :math:`h(x_i)` are estimated using :ref:`scalar_continuous_entropy <scalar-continuous-entropy>`. 


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
	h : float
		The (differential) entropy of the data generating distribution, assuming its copula is maximum-entropy in the chosen space. By convention, when :math:`d=1`, this function is the same as :ref:`scalar_continuous_entropy <scalar-continuous-entropy>`, and returns 0 when :math:`n=1`.
	"""
	if x.shape[0] == 1:
		# By convention, the entropy of a single sample is 0.
		return 0.0

	if len(x.shape) == 1 or x.shape[1] == 1:
		return scalar_continuous_entropy(x, space=space)

	ch = least_structured_copula_entropy(x, space=space)
	ih = np.sum([scalar_continuous_entropy(x[:, i], space=space) for i in range(x.shape[1])])

	return ih+ch




def least_structured_mixed_entropy(x_c, x_d, space='dual'):
	"""
	.. _least-structured-mixed-entropy:
	Estimates the joint entropy :math:`h(x_c, x_d)`, where :math:`x_c` is continuous random vector and :math:`x_d` is a discrete random vector.

	.. note::

		We use the identities

		.. math::
			h(x, y) &= h(y) + h(x|y) \\

			        &= h(y) + E \\left[ h(x \\vert y=.) \\right]

		that are true when :math:`x` and :math:`y` are either both continuous or both discrete
		to extend the definition of the joint entropy to the case where one is continuous and
		the other discrete.

		Specifically,

		.. math::
			h(x_c, x_d) = h(x_d) + \sum_{j=1}^q \mathbb{P}(x_d=j) h\\left(x_c \\vert x_d=j \\right).

		* :math:`h(x_d)` is estimated using :ref:`discrete_entropy <discrete-entropy>`.
		* :math:`\mathbb{P}(x_d=j)` is estimated using frequencies.
		* :math:`h\\left(x_c \\vert x_d=j \\right)` is estimated using :ref:`least_structured_continuous_entropy <least-structured-continuous-entropy>`.


	Parameters
	----------
	x_c : (n, d) np.array
		n i.i.d. draws from the continuous data generating distribution.

	x_d : (n,) np.array
		n i.i.d. draws from the discrete data generating distribution, jointly sampled with x_c.

	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.

	Returns
	-------
	h : float
		The entropy of the least structured distribution as evidenced by average the maximum entropy constraints.
	"""
	labels = x_d.copy() if len(x_d.shape) == 1 else np.array([','.join([str(_) for _ in row]) for row in x_d])
	categories = list(set(labels))

	n = labels.shape[0]
	probas = np.array([1.*len(labels[labels==cat])/n for cat in categories])
	h = -np.dot(probas, np.log(probas))
	h += np.sum([probas[i] * least_structured_continuous_entropy(x_c[labels==categories[i]], space=space) for i in range(len(categories))])

	return h



