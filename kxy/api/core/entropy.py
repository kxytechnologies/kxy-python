#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file leverages the KXY API to solve various maximum entropy copula problems
under concordance measures. All optimization problems are solved by the KXY API
and users require an API key that should be set in the environment variable
KXY_API_KEY.
"""
import logging
import requests

import numpy as np
import scipy.special as spe

from kxy.api import APIClient
from .utils import avg_pairwise_spearman_corr, pre_conditioner

def max_ent_copula_entropy(method='average-pairwise-spearman-rho', d=None, rho=None,  \
		rho_features=None, rho_full=None):
	"""
	.. _max-ent-copula-entropy:
	Solves a constrained least-informative copula problem and returns the differential 
	entropy of the least informative copula satisfying the input constraints.

	.. important::

		The optimimzation problem is solved on the KXY infrastructure, not locally, and an 
		API key is required to do so.


	.. note::

		Let :math:`x \in \mathbb{R}^d` be a continuous random variable with probability density
		function :math:`p`, copula density :math:`c` and copula-uniform representation :math:`u := (F_1(x_i), \dots , F_d(x_d))` where 
		:math:`F_i` is the cummulative density function of coordinate variable :math:`x_i`. The differential entropy of :math:`x`, defined as 

		.. math::
			h(x) := -\int_{\mathbb{R}^d} p(x) \log p(x) dx

		can be broken down as 

		.. math::
			h(x) = h(u) + \sum_{i=1}^d h(x_i).

		:math:`h(u)=h(c)` is also the entropy of the copula of :math:`x` . Note that :math:`\sum_{i=1}^d h(x_i)-h(x)=-h(u)`
		is the total correlation of :math:`x` and quantifies information redundancy between 
		coordinates of :math:`x`.

		Given a functional :math:`g` of :math:`c` that can estimated from samples of math:`x` ,
		this function solves the following optimization problem over the set of all copulas :math:`\mathcal{C}`

		.. math::
			&\max_{c \in \mathcal{C}} h(c), \\

			&\\text{subject to } g(c) = \\alpha


		Functionals that are currently supported are population versions of the average pairwise 
		Spearman's rank correlation among a subset of coordinates of x.

		.. math::
			g(c) = 12 \\left[ \\frac{2}{d(d-1)} \\sum_{j<j^\\prime} E \\left( u_j u_{j^\\prime} \\right) \\right]-3

		For sample estimates based on math:`x`, see Eq. (3.3.3) in :cite:`JoeH90`.


	Parameters
	----------
	method : str (default: 'average-pairwise-spearman-rho')
		The type of constraint to use for the optimization problem. It should be either
		'average-pairwise-spearman-rho' or 'average-pairwise-spearman-rho-1vd'.

		When method='average-pairwise-spearman-rho', the functional :math:`g` used maps 
		a copula to its population average pairwise Spearman's rank correlation.

		When method='average-pairwise-spearman-rho-1vd', the functional :math:`g` used maps 
		a copula to its population average pairwise Spearman's rank correlation and the 
		population average pairwise Spearman's rank correlation of all its coordinates except one.
	d : int
		Dimension of the random variable. Defaulted to None, but should always be set.
	rho : float
		Average pairwise Spearman's rank correlation accross the full random vector. Defaulted to None,
		but should be set if method='average-pairwise-spearman-rho'.
	rho_full : float
		Average pairwise Spearman's rank correlation accross the full random vector. Defaulted to None,
		but should be set if method='average-pairwise-spearman-rho-1vd'.
	rho_features : float
		Average pairwise Spearman's rank correlation accross (d-1) coordinates. Defaulted to None,
		but should be set if method='average-pairwise-spearman-rho-1vd'.	

	Returns
	-------
	h : float
		The entropy of the least informative copula consistent with the input constraints.

	Raises
	------
	AssertionError
		If any required parameter is missing.

	ValueError
		If the optimization problem is not feasible.

	Exception
		If the KXY API is not able to return an answer quickly enough. This typically means
		you should try again later.


	.. rubric:: References

	.. [JoeH90] Joe, H. Journal of multivariate analysis 35 (1), 12-30, 1990.
	"""
	assert method in ('average-pairwise-spearman-rho', 'average-pairwise-spearman-rho-1vd'), \
		'Method not supported.'

	api_response = None
	if method == 'average-pairwise-spearman-rho':
		assert d is not None, 'The dimension d should be provided.'
		assert rho is not None, 'The average pairwise Spearman rho should be provided'

		api_response = APIClient.route(\
			path='/core/dependence/copula/maximum-entropy/entropy/rv/full', method='GET', \
			d=d, rho=rho)
		logging.info('rho: %.4f, d:%d, api_response: %s' % (rho, d, api_response.json()))

	if method == 'average-pairwise-spearman-rho-1vd':
		assert d is not None, 'The d dimension of the feature space should be provided'
		assert rho_features is not None, \
			'The average pairwise Spearman rank correlation between features should be provided'
		assert rho_full is not None, \
			'The average pairwise Spearman rank correlation including features and the ouput should be provided.'

		api_response = APIClient.route(\
			path='/core/dependence/copula/maximum-entropy/entropy/rv/1vd', method='GET', \
			d=d, rho_features=rho_features, rho_full=rho_full)
		logging.info('rho_features: %.4f, rho_full: %.4f, d: %d, api_response: %s' % (\
			rho_features, rho_full, d, api_response.json()))
	

	if api_response.status_code == requests.codes.ok:
		return api_response.json()['entropy']

	if api_response.status_code == 403:
		raise ValueError(api_response.json().get('reason', api_response.json().get('message')))

	if api_response.status_code == 404:
		raise Exception(api_response.json().get('reason', api_response.json().get('message')))

	return None


def scalar_continuous_entropy(x):
	"""
	.. _scalar-continuous-entropy:
	Estimates the (differential) entropy of a continuous scalar random variable using the standard 1-spacing estimator (:cite:`KozN87`, :cite:`BerD97`):

	.. math::
		h(x) \\approx - \gamma(1) + \\frac{1}{n-1} \\sum_{i=1}^{n-1} \log \\left[ n \\left(x_{(i+1)} - x_{(i)} \\right) \\right],

	where :math:`x_{(i)}` is the i-th smallest entry in :math:`(x_1, \dots, x_n)`, and :math:`\\gamma` is
	the digamma function. 

	
	.. note::

		A Gaussian noise with negligible standard deviation (1/10000-th of the input standard deviation) is added to the input
		for robustness.

	
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
		If the input has the wrong shape.
		

	.. rubric:: References

	.. [KozN87] Kozachenko, L. F., and Nikolai N. Leonenko. "Sample estimate of the entropy of a random vector." 
		Problemy Peredachi Informatsii 23.2 (1987): 9-16.

	.. [BerD97] Beirlant, J., Dudewicz, E.J., Györfi, L., van der Meulen, E.C. "Nonparametric entropy estimation: an overview." 
		International Journal of Mathematical and Statistical Sciences. 6 (1): 17–40. (1997) ISSN 1055-7490. 
	"""
	assert len(x.shape) == 1 or x.shape[1] == 1, 'x should be a one dimensional numpy array'

	noise_std = x.std()/10000.
	x += noise_std * np.random.randn(*x.shape)
	sorted_x = np.unique(x)
	n = sorted_x.shape[0]
	ent = np.sum(np.log(n*(sorted_x[1:]-sorted_x[:-1])))/n - spe.digamma(1)

	return ent


def discrete_entropy(x):
	"""
	.. _discrete-entropy:
	Estimates the (Shannon) entropy of a discrete random variable taking up to q distinct values 
	given n i.i.d samples.

	.. math::
		h(x) = - \sum_{i=1}^q p_i \log p_i

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


def least_structured_copula_entropy(x):
	"""
	.. _least-structured-copula-entropy:
	Estimates the entropy of the least informative copula model whose average pairwise
	Spearman rank correlation is the same as the sample estimate from the input
	array.

	.. note::
	
		This also corresponds to least amount of total correlation that is evidenced
		by the sample average pairwise Spearman's rank correlation.

	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the data generating distribution.

	Returns
	-------
	h : float
		The (differential) entropy of the least structured copula 
		consistent with observed average pairwise Spearman's rank correlation. See also :ref:`max_ent_copula_entropy <max-ent-copula-entropy>`.
	"""
	if len(x.shape) == 1 or x.shape[1] == 1:
		# By convention, the copula-dual representation 
		# of a 1d random variable is the uniform[0, 1].
		return 0.0

	d = x.shape[1]
	rho = avg_pairwise_spearman_corr(x)
	h = max_ent_copula_entropy(method='average-pairwise-spearman-rho', d=d, rho=rho)

	return h


def least_structured_continuous_entropy(x):
	""" 
	.. _least-structured-continuous-entropy:
	Estimates the entropy of a continuous d-dimensional random variable under the 
	least structured assumption for its copula. When :math:`d>1`,

	.. math::
		h(x) = h(u) + \sum_{i=1}^d h(x_i),

	and :math:`h(u)` is estimated using :ref:`least_structured_copula_entropy <least-structured-copula-entropy>` and 
	:math:`h(x_i)` are estimated using :ref:`scalar_continuous_entropy <scalar-continuous-entropy>`. 


	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the data generating distribution.

	Returns
	-------
	h : float
		The (differential) entropy of the data generating distribution, assuming 
		its copula is the least structured copula consistent with the observed 
		average pairwise Spearman's rank correlation. 

		By convention, when :math:`d=1`, this function is the same as :ref:`scalar_continuous_entropy. <scalar-continuous-entropy>`, 
		and returns 0 when :math:`n=1`.
	"""
	if x.shape[0] == 1:
		# By convention, the entropy of a single sample is 0.
		return 0.0

	if len(x.shape) == 1 or x.shape[1] == 1:
		return scalar_continuous_entropy(x)

	x = x - x.mean(axis=0)
	a, log_abs_deta = pre_conditioner(x.T)
	z = np.dot(a, x.T).T

	ch = least_structured_copula_entropy(z)
	ih = np.sum([scalar_continuous_entropy(z[:, i]) for i in range(z.shape[1])])

	return ih+ch-log_abs_det_a


def least_structured_mixed_entropy(x_c, x_d):
	"""
	.. _least-structured-mixed-entropy:
	Estimates the joint entropy :math:`h(x_c, x_d)`, where :math:`x_c` is continuous
	random vector and :math:`x_d` is a discrete random vector.

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

		:math:`h(x_d)` is estimated using :ref:`discrete_entropy <discrete-entropy>`, :math:`\mathbb{P}(x_d=j)` is estimated using frequencies, and
		:math:`h\\left(x_c \\vert x_d=j \\right)` is estimated using :ref:`least_structured_continuous_entropy <least-structured-continuous-entropy>`.


	Parameters
	----------
	x_c : (n, d) np.array
		n i.i.d. draws from the continuous data generating distribution.

	x_d : (n,) np.array
		n i.i.d. draws from the discrete data generating distribution, jointly sampled with x_c.

	Returns
	-------
	h : float
		The entropy of the least structured distribution as evidenced by average pairwise Spearman
		rank correlations.
	"""
	labels = x_d.copy() if len(x_d.shape) == 1 else np.array([','.join([str(_) for _ in row]) for row in x_d])
	categories = list(set(labels))

	n = labels.shape[0]
	probas = np.array([1.*len(labels[labels==cat])/n for cat in categories])
	h = -np.dot(probas, np.log(probas))
	h += np.sum([probas[i] * least_structured_continuous_entropy(x_c[labels==categories[i]]) for i in range(len(categories))])

	return h



