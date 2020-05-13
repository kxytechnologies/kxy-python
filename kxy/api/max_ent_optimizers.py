#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import requests
from time import sleep, time

from .client import APIClient



def solve_copula_async(corr):
	"""
	.. _solve-copula-async:
	Solve the maximum-entropy copula problem under Spearman rank correlation matrix constraints asynchronously.

	.. note:: 

		The solution to the optimization problem is not returned. This function should be used to pre-compute the solution 
		to the optimization problem for later use.


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





def solve_copula_sync(corr, mode=None, output_index=None, solve_async=True):
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
			if mode == 'copula_entropy':
				logging.debug('Querying the max-ent solution with corr=%s and mode=%s' % (c, mode))
				api_response = APIClient.route(path='/rv/copula-entropy-analysis', method='POST', corr=c, \
					request_id=request_id, timestamp=int(time()))

			if mode == 'mutual_information_v_output':
				logging.debug('Querying the max-ent solution with corr=%s, mode=%s and output_index=%d' % (c, mode, output_index))
				api_response = APIClient.route(path='/rv/mutual-information-analysis', method='POST',\
					corr=c, output_index=output_index, request_id=request_id, timestamp=int(time()))
			query_duration = time()-query_start_time

		else:
			query_start_time = time()
			# Subsequent attempt: refer to the initial request
			logging.debug('Querying the max-ent solution for request_id=%s' % request_id)
			if mode == 'copula_entropy':
				api_response = APIClient.route(path='/rv/copula-entropy-analysis', method='POST', \
					request_id=request_id, timestamp=int(time()))

			if mode == 'mutual_information_v_output':
				api_response = APIClient.route(path='/rv/mutual-information-analysis', method='POST',\
					request_id=request_id, timestamp=int(time()))
			query_duration = time()-query_start_time

		retry_count += 1
		if api_response.status_code == requests.codes.retry:
			request_id = api_response.json()['request_id']
			sleep(.1 if query_duration > 10. else 10.)


	if api_response.status_code == requests.codes.ok:
		if mode == 'copula_entropy':
			h = api_response.json()['copula_entropy']
			if h is not None:
				return float(h)

		if mode == 'mutual_information_v_output':
			mi = api_response.json()['mutual_information']
			if not mi is None:
				return float(mi)

	else:
		logging.warning(api_response.json())

	return None


