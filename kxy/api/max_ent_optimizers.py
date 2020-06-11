#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import requests
from time import sleep, time

import numpy as np

from .client import APIClient



def solve_copula_async(corr):
	"""
	.. _solve-copula-async:
	Solve the maximum-entropy copula problem under Spearman rank correlation matrix constraints asynchronously.

	.. note:: 

		The solution to the optimization problem is not returned. This function should be used to pre-compute the solution to the optimization problem for later use.


	.. seealso::

		:ref:`kxy.api.optimizers.solve_sync <solve-copula-sync>`.


	Parameters
	----------
	corrr : np.array
		The Spearman correlation matrix.


	Returns
	-------
	success : bool
		Whether the request went through successfully.
	"""
	try:
		c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
		logging.debug('Launching a max-ent solver in the background with corr=%s' % c)
		api_response = APIClient.route(path='/core/dependence/copula/maximum-entropy/entropy/rv/pre-compute', \
			method='POST', corr=c)

		if api_response.status_code == requests.codes.ok:
			return True

		logging.info(api_response.json())
		return False

	except:
		return False





def solve_copula_sync(corr, mode=None, output_index=None, solve_async=True, space='dual'):
	"""
	.. _solve-copula-sync:
	Solve the maximum-entropy copula problem under Spearman rank correlation matrix constraints synchronously.

	.. note:: 

		This function blocks until the optimization problem is solved, and returrns the requested quantity.

	.. seealso::

		:ref:`kxy.api.optimizers.solve_sync <solve-copula-async>`.


	Parameters
	----------
	corrr : np.array
		The Spearman correlation matrix.

	mode : str
		One of :code:`'copula_entropy'`, and :code:`'mutual_information_v_output'`.

		When mode is :code:`'copula_entropy'` the function returns the entropy of the copula of the random
		vector whose Spearman correlation matrix is the input :code:`corr`.

		When mode is :code:`'mutual_information_v_output'` the function returns the mutual information between 
		a continuous output variable :math:`y` (specificied by :code:`output_index`) and continuous input 
		variables :math:`x` (specificied by the other variables): :math:`I(x, y)`. We recall that the mutual information 
		between two continuous variables is the mutual information of their copula-uniform dual representations. The copula 
		of :math:`(x, y)` is learned as the maximum-entropy copula under the constraint that the Spearman correlation matrix 
		is the input :code:`corr`.

	output_index : int
		The index of the column that should be used as output variable when mode is :code:`'mutual_information_v_output'`.



	Returns
	-------
	r : float
		The requested result, namely the copula entropy, the mutual information or the conditional
		mutual information, depending on the value of the mode.
	"""
	assert mode in ('copula_entropy', 'mutual_information_v_output')

	if mode in ['mutual_information_v_output', 'conditional_mutual_information']:
		assert output_index is not None, 'The output index should be provided'

	if solve_async:
		solve_copula_async(corr)

	if mode == 'mutual_information_v_output':
		res = mutual_information_analysis(corr, output_index, space=space)
		if res is None:
			return None

		return res['mutual_information']

	if mode == 'copula_entropy':
		res = copula_entropy_analysis(corr, space=space)
		if res is None:
			return None
		return res['copula_entropy']


	return None



def mutual_information_analysis(corr, output_index, space='dual', greedy=True):
	'''
	Analyzes the dependency between :math:`d`-dimensional continuous random vector :math:`x=\\left(x_1, \\dots, x_d \\right)` and
	a continuous random scalar :math:`y` whose joint correlation matrix is :code:`corr`, the column :code:`output_index` of which 
	represents the variable :math:`y` and the others the variable :math:`x`.


	Recall that, for any permutation :math:`\\pi_1, \\dots, \\pi_d` of :math:`1, \\dots, d`, by the tower law,

	.. math::

		I\\left(y; x_1, \\dots, x_d\\right) = I\\left(y; x_{\\pi_1}\\right) + \\sum_{i=2}^d I\\left(y; x_{\\pi_i} \\vert x_{\\pi_{i-1}}, \\dots, x_{\\pi_1} \\right).


	This function estimates the mutual information :math:`I(y; x)` by learning the following permutation. 

	When greedy is True:

	* :math:`x_{\\pi_1}` is the input with the largest maximum entropy mutual information with :math:`y` under Spearman rank correlation constraints.

	* :math:`x_{\\pi_i}` for :math:`i>1` is the input with the largest maximum entropy conditional mutual information :math:`I\\left(y; * \\vert x_{\\pi_{i-1}}, \\dots, x_{\\pi_1}\\right)`. Note that by the time :math:`\\pi_i` is selected, :math:`I\\left(y; x_{\\pi_{i-1}}, \\dots, x_{\\pi_1}\\right)` is already known, so that the maximum entropy conditional mutual information is simply derived from the maximum entropy copula distribution of :math:`I\\left(y; x_{\\pi_i}, \\dots, x_{\\pi_1}\\right)`.

	This function returns the learned permutation of inputs, the associated conditional mutual informations (a.k.a, the incremental input importance scores), as well as the mutual information :math:`I\\left(y; x_1, \\dots, x_d\\right)`.

	When greedy is False, :math:`\\pi_i` is the input with the i-th largest mutual information with the output.


	Parameters
	----------
	corrr : np.array
		The Spearman correlation matrix.

	output_index: int
		The index of the column to use as output.


	Returns
	-------
	res : dict
		Dictionary with keys :code:`mutual_information`, :code:`selection_order`, and :code:`conditional_mutual_informations`.
	'''
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
	opt_launched = False
	max_retry = 60
	first_try = True
	retry_count = 0
	request_id = ''
	while (first_try or api_response.status_code == requests.codes.retry) and retry_count < max_retry:
		first_try = False

		if request_id == '':
			query_start_time = time()
			# First attempt
			logging.debug('Querying mutual information analysis with corr=%s and output_index=%d' % (c, output_index))
			api_response = APIClient.route(path='/rv/mutual-information-analysis', method='POST',\
				corr=c, output_index=output_index, request_id=request_id, timestamp=int(time()), \
				space=space, greedy=int(greedy))
			query_duration = time()-query_start_time

		else:
			query_start_time = time()
			# Subsequent attempt: refer to the initial request
			logging.debug('Querying mutual information analysis for request_id=%s' % request_id)
			api_response = APIClient.route(path='/rv/mutual-information-analysis', method='POST',\
				request_id=request_id, timestamp=int(time()), space=space)
			query_duration = time()-query_start_time

		retry_count += 1
		if api_response.status_code == requests.codes.retry:
			request_id = api_response.json()['request_id']
			sleep(.1 if query_duration > 10. else 10.)


	if api_response.status_code == requests.codes.ok:
		return api_response.json()

	else:
		logging.warning(api_response.json())

	return None



def copula_entropy_analysis(corr, space='dual'):
	'''
	Analyzes the entropy of the copula of a :math:`d`-dimensional continuous random vector :math:`x=\\left(x_1, \\dots, x_d \\right)`, with copula-uniform representation :math:`u=\\left(u_1, \\dots, u_d \\right)`.

	Recall that, for any permutation :math:`(1), \\dots, (d)` of :math:`1, \\dots, d`, by the tower law,

	.. math::

		h\\left(u_1, \\dots, u_d\\right) = \\sum_{i=2}^d h\\left( u_{(i)} \\vert u_{(i-1)}, \\dots, u_{(1)} \\right).


	This function estimates the copula entropy  :math:`h(u)` by learning the following permutation:

	* :math:`x_{(1)}` and :math:`x_{(2)}` are chosen to be the two random variables with smallest copula entropy (or equivalently, the highest mutual information).

	* :math:`x_{(i)}` for :math:`i>1` is the input with the smallest conditional copula entropy :math:`h\\left(* \\vert u_{(i-1)}, \\dots, u_{(1)} \\right)` (or equivalently, the highest mutual information :math:`I\\left(*; x_{(1)}, \\dots, x_{(i-1)}\\right)`). Note that by the time :math:`(i)` is selected, :math:`h\\left(u_{(i-1)}, \\dots, u_{(1)}\\right)` is already known, so that the maximum entropy conditional entropy is simply derived from the maximum entropy copula distribution of :math:`\\left(x_{(i)}, \\dots, x_{(1)}\\right)`.

	This function returns the learned permutation of inputs, the association conditional entropies, as well as the copula entropy :math:`h(u)`.


	Parameters
	----------
	corrr : np.array
		The Spearman correlation matrix.

	output_index: int
		The index of the column to use as output.


	Returns
	-------
	res : dict
		Dictionary with keys :code:`copula_entropy`, :code:`selection_order`, and :code:`conditional_copula_entropies`.
	'''
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
	opt_launched = False
	max_retry = 60
	first_try = True
	retry_count = 0
	request_id = ''
	while (first_try or api_response.status_code == requests.codes.retry) and retry_count < max_retry:
		first_try = False

		if request_id == '':
			query_start_time = time()
			# First attempt
			logging.debug('Querying copula entropy analysis with corr=%s' % (c))
			api_response = APIClient.route(path='/rv/copula-entropy-analysis', method='POST', corr=c, \
				request_id=request_id, timestamp=int(time()), space=space)
			query_duration = time()-query_start_time

		else:
			query_start_time = time()
			# Subsequent attempt: refer to the initial request
			logging.debug('Querying copula entropy analysis for request_id=%s' % request_id)
			api_response = APIClient.route(path='/rv/copula-entropy-analysis', method='POST', \
				request_id=request_id, timestamp=int(time()), space=space)
			query_duration = time()-query_start_time

		retry_count += 1
		if api_response.status_code == requests.codes.retry:
			request_id = api_response.json()['request_id']
			sleep(.1 if query_duration > 10. else 10.)


	if api_response.status_code == requests.codes.ok:
		return api_response.json()

	else:
		logging.warning(api_response.json())

	return None




def information_adjusted_correlation_from_spearman(corr, space='dual'):
	'''
	Determine the Pearson correlation matrix that should be used as plug-in replacement
	for the standard Pearson correlation estimator, in situations where Gaussianity has
	to be assumed for simplicity, but the data-generating distribution exhibits non-Gaussian
	traits (e.g. heavier or lighter tails).

	We proceed as follows, for every pairwise Spearman rank correlation, we estmate the smallest
	mutual information that is consistent with the pairwise Spearman rank correlation. This is also 
	the mutual information of the model which, among all models that have the same Spearman rank 
	correlation, is the most uncertain about everything else. 

	This mutual information captures the essence of the dependence between the two variables without
	assuming Gaussianity. To go back to the Gaussian scale, we determine the Pearson correlation which, 
	under the Gaussian assumption would yield the same mutual information as estimated.


	Parameters
	----------
	corr : np.array
		The Spearman correlation matrix.


	Returns
	-------
	res : np.array
		The array of equivalent Pearson correlation coefficients.
	'''
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
	opt_launched = False
	max_retry = 60
	first_try = True
	retry_count = 0
	request_id = ''
	while (first_try or api_response.status_code == requests.codes.retry) and retry_count < max_retry:
		first_try = False

		if request_id == '':
			query_start_time = time()
			# First attempt
			logging.debug('Querying information adjusted correlation with corr=%s' % (c))
			api_response = APIClient.route(path='/rv/information-adjusted-correlation', method='POST', corr=c, \
				request_id=request_id, timestamp=int(time()), space=space)
			query_duration = time()-query_start_time

		else:
			query_start_time = time()
			# Subsequent attempt: refer to the initial request
			logging.debug('Querying information adjusted correlation for request_id=%s' % request_id)
			api_response = APIClient.route(path='/rv/information-adjusted-correlation', method='POST', \
				request_id=request_id, timestamp=int(time()), space=space)
			query_duration = time()-query_start_time

		retry_count += 1
		if api_response.status_code == requests.codes.retry:
			request_id = api_response.json()['request_id']
			sleep(.1 if query_duration > 10. else 10.)


	if api_response.status_code == requests.codes.ok:
		return np.array(api_response.json()['ia_corr']).astype(float)

	else:
		logging.warning(api_response.json())

	return None



def robust_pearson_corr_from_spearman(corr):
	'''
	Return the Pearson correlation matrix that is equivalent to the input 
	Spearman rank correlation matrix, assuming inputs are jointly Gaussian.


	Parameters
	----------
	corr : np.array
		The Spearman correlation matrix.


	Returns
	-------
	res : np.array
		The array of equivalent Pearson correlation coefficients.
	'''
	return information_adjusted_correlation_from_spearman(corr, space='primal')



