#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import requests
from time import sleep, time

import numpy as np

from .client import APIClient


def mutual_information_analysis(corr, output_indices, space='dual', batch_indices=[]):
	'''
	Analyzes the dependency between :math:`d`-dimensional continuous random vector :math:`x=\\left(x_1, \\dots, x_d \\right)` and
	one or more continuous random scalar :math:`y`.


	Recall that, for any permutation :math:`\\pi_1, \\dots, \\pi_d` of :math:`1, \\dots, d`, by the tower law,

	.. math::

		I\\left(y; x_1, \\dots, x_d\\right) = I\\left(y; x_{\\pi_1}\\right) + \\sum_{i=2}^d I\\left(y; x_{\\pi_i} \\vert x_{\\pi_{i-1}}, \\dots, x_{\\pi_1} \\right).


	This function estimates the mutual information :math:`I(y; x)` by learning the following permutation. 

	* :math:`x_{\\pi_1}` is the input with the largest maximum entropy mutual information with :math:`y` under Spearman rank correlation constraints.

	* :math:`x_{\\pi_i}` for :math:`i>1` is the input with the largest maximum entropy conditional mutual information :math:`I\\left(y; * \\vert x_{\\pi_{i-1}}, \\dots, x_{\\pi_1}\\right)`. Note that by the time :math:`\\pi_i` is selected, :math:`I\\left(y; x_{\\pi_{i-1}}, \\dots, x_{\\pi_1}\\right)` is already known, so that the maximum entropy conditional mutual information is simply derived from the maximum entropy copula distribution of :math:`I\\left(y; x_{\\pi_i}, \\dots, x_{\\pi_1}\\right)`.

	This function returns the learned permutation of inputs, the associated conditional mutual informations (a.k.a, the incremental input importance scores), as well as the mutual information :math:`I\\left(y; x_1, \\dots, x_d\\right)`.


	Parameters
	----------
	corr : (d, d) np.array
		The Spearman correlation matrix.

	output_indices: list of int 
		The index (indices) of the column to use as output.

	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.



	Returns
	-------
	res : dict
		Dictionary with keys :code:`mutual_information`, :code:`selection_order`, and :code:`conditional_mutual_informations`.
	'''
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
	bi = json.dumps(batch_indices)
	oi = json.dumps(output_indices)

	max_retry = 60
	first_try = True
	retry_count = 0
	request_id = ''
	while (first_try or api_response.status_code == requests.codes.retry) and retry_count < max_retry:
		first_try = False

		if request_id == '':
			query_start_time = time()
			# First attempt
			logging.debug('Querying mutual information analysis with corr=%s and output_indices=%s' % (c, output_indices))
			api_response = APIClient.route(path='/rv/mutual-information-analysis', method='POST',\
				corr=c, output_indices=oi, request_id=request_id, timestamp=int(time()), \
				space=space, batch_indices=bi)

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
	corr : np.array
		The Spearman correlation matrix.

	output_index: int
		The index of the column to use as output.

	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


	Returns
	-------
	res : dict
		Dictionary with keys :code:`copula_entropy`, :code:`selection_order`, and :code:`conditional_copula_entropies`.
	'''
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
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
		logging.warning(api_response)
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

	space : str, 'primal' | 'dual'
		The space in which the maximum entropy problem is solved. 
		When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
		When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.


	Returns
	-------
	res : np.array
		The array of equivalent Pearson correlation coefficients.
	'''
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
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




def predict_copula_uniform(u_x_dict_list, corr, output_indices, space='dual', batch_indices=[], \
		problem_type='regression'):
	'''
	'''
	c = json.dumps([['%.3f' % corr[i, j]  for j in range(corr.shape[1])] for i in range(corr.shape[0])])
	ux = json.dumps(u_x_dict_list)
	bi = json.dumps(batch_indices)
	oi = json.dumps(output_indices)

	max_retry = 60
	first_try = True
	retry_count = 0
	request_id = ''

	while (first_try or api_response.status_code == requests.codes.retry) and retry_count < max_retry:
		first_try = False

		if request_id == '':
			query_start_time = time()
			# First attempt
			logging.debug('Requesting predictions endpoint.')
			api_response = APIClient.route(path='/rv/predictions', method='POST', request_type='auth',
			 timestamp=int(time()))
			request_id   = api_response.json()['request_id']
			response = api_response.json()['prediction_endpoint']
			prediction_endpoint = response['url']
			additional_fields = response['fields']

			logging.debug('Submitting prediction job.')
			params = {'corr': c, 'output_indices': oi, 'request_id': request_id, 'timestamp': int(time()), \
				'space': space, 'batch_indices': bi, 'u_x_dict_list': ux, 'problem_type': problem_type
			}

			key = 'rv_predict_full/' + request_id + '.json'
			os.makedirs('rv_predict_full', exist_ok=True)
			with open(key, 'w') as f:
				json.dump(params, f)

			with open(key, 'rb') as f:
				files = {'file': (key, f)}
				response = requests.post(prediction_endpoint, data=response['fields'], \
					files=files)

			logging.debug('Done submitting prediction job.')
			query_duration = time()-query_start_time

		else:
			query_start_time = time()
			# Subsequent attempt: refer to the initial request
			logging.debug('Requesting predictions.')
			api_response = APIClient.route(path='/rv/predictions', method='POST',\
				request_id=request_id, timestamp=int(time()))
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


