#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import requests
from time import sleep

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
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
	logging.debug('Launching a max-ent solver in the background with corr=%s' % c)
	api_response = APIClient.route(path='/core/dependence/copula/maximum-entropy/entropy/rv/pre-compute', \
		method='GET', corr=c)

	if api_response.status_code == requests.codes.ok:
		return True

	logging.info(api_response.json())
	return False




def solve_copula_sync(corr, mode=None, output_index=None, input_indices=None, condition_indices=None):
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
		One of :code:`'copula_entropy'`, :code:`'mutual_information_v_output'`, and :code:`'conditional_mutual_information'`.

		When mode is :code:`'copula_entropy'` the function returns the entropy of the copula of the random
		vector whose Spearman correlation matrix is the input :code:`corr`.

		When mode is :code:`'mutual_information_v_output'` the function returns the mutual information between 
		a continuous output variable :math:`y` (specificied by :code:`output_index`) and continuous input 
		variables :math:`x` (specificied by the other variables): :math:`I(x, y)`. We recall that the mutual information 
		between two continuous variables is the mutual information of their copula-uniform dual representations. The copula 
		of :math:`(x, y)` is learned as the maximum-entropy copula under the constraint that the Spearman correlation matrix 
		is the input :code:`corr`.

		When mode is :code:`'conditional_mutual_information'` the function returns the conditional mutual information
		between output :math:`y` (specified by column :code:`output_index`) and inputs :math:`x` 
		(specified by columns :code:`input_indices`) conditional on :math:`z` (specified by columns :code:`condition_indices`):
		:math:`I(x, y\\vert z)`. We recall that the conditional mutual information of two random variables given a third, 
		is the conditional mutual information between the associated copula-uniform dual representations. The copula of 
		:math:`(x, y, z)` is learned as the maximum-entropy copula under the constraint that its Spearman correlation matrix 
		is given by the input :code:`corr`.

	output_index : int
		The index of the column that should be used as output variable when mode is either :code:`'mutual_information_v_output'` or
		:code:`'conditional_mutual_information'`.

	input_indices : list
		The indices of the columns that should be used as input variables when mode is :code:`'conditional_mutual_information'`.

	condition_indices : list
		The indices of the columns that should be used as condition variables when mode is :code:`'conditional_mutual_information'`.


	Returns
	-------
	r : float
		The requested result, namely the copula entropy, the mutual information or the conditional
		mutual information, depending on the value of the mode.
	"""
	assert mode in ('copula_entropy', 'mutual_information_v_output', 'conditional_mutual_information')

	if mode in ['mutual_information_v_output', 'conditional_mutual_information']:
		assert output_index is not None, 'The output index should be provided'

	if mode == 'conditional_mutual_information':
		assert input_indices is not None, 'Input indices should be provided for conditional mutual information.'
		assert condition_indices is not None, 'Condition indices should be provided for conditional mutual information.'

	solve_copula_async(corr)
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
	opt_launched = False
	max_retry = 60
	first_try = True
	retry_count = 0
	while (first_try or api_response.status_code == requests.codes.not_found) and retry_count < max_retry:
		logging.debug('Querying the max-ent solution with corr=%s and mode=%s' % (c, mode))
		first_try = False
		if mode == 'copula_entropy':
			api_response = APIClient.route(\
				path='/core/dependence/copula/maximum-entropy/entropy/rv/all', method='GET', \
				corr=c, mode='copula_entropy')

		if mode == 'mutual_information_v_output':
			api_response = APIClient.route(\
				path='/core/dependence/copula/maximum-entropy/entropy/rv/all', method='GET', \
				corr=c, mode='mutual_information_v_output', output_index=output_index)

		if mode == 'conditional_mutual_information':
			input_indices = json.dumps(input_indices)
			condition_indices = json.dumps(condition_indices)
			api_response = APIClient.route(\
				path='/core/dependence/copula/maximum-entropy/entropy/rv/all', method='GET', \
				corr=c, mode='conditional_mutual_information', output_index=output_index, \
				input_indices=input_indices, condition_indices=condition_indices)

		retry_count += 1
		if api_response.status_code == requests.codes.not_found:
			sleep(10.)

	logging.debug(api_response.json())
	if api_response.status_code == requests.codes.ok:
		if mode == 'copula_entropy':
			h = api_response.json()['entropy']
			if h is not None:
				return float(h)

		if mode == 'mutual_information_v_output':
			mi = api_response.json()['mutual_information']
			if not mi is None:
				return float(mi)

		if mode == 'conditional_mutual_information':
			mi = api_response.json()['conditional_mutual_information']
			if not mi is None:
				return float(mi)

	return None


