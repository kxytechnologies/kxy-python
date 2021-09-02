#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimation of the highest performance achievable in a supervised learning problem.
E.g. :math:`R^2`, RMSE, classification accuracy, true log-likelihood per observation.
"""
import logging
logging.basicConfig(level=logging.INFO)
import requests
import sys
from time import time, sleep

import numpy as np
import pandas as pd

from kxy.api import APIClient, upload_data, approx_opt_remaining_time

# Cache old job ids to avoid being charged twice for the same job.
VALUATION_JOB_IDS = {}

def data_valuation(data_df, target_column, problem_type, snr='auto'):
	"""
	.. _data-valuation:
	Estimate the highest performance metrics achievable when predicting the :code:`target_column` using all other columns.

	When :code:`problem_type=None`, the nature of the supervised learning problem (i.e. regression or classification) is inferred from whether or not :code:`target_column` is categorical.


	Parameters
	----------
	data_df : pandas.DataFrame
		The pandas DataFrame containing the data.
	target_column : str
		The name of the column containing true labels.
	problem_type : None | 'classification' | 'regression'
		The type of supervised learning problem. When None, it is inferred from the column type and the number of distinct values.



	Returns
	-------
	achievable_performance : pandas.Dataframe
		The result is a pandas.Dataframe with columns (where applicable):

		* :code:`'Achievable Accuracy'`: The highest classification accuracy that can be achieved by a model using provided inputs to predict the label.
		* :code:`'Achievable R-Squared'`: The highest :math:`R^2` that can be achieved by a model using provided inputs to predict the label.
		* :code:`'Achievable RMSE'`: The lowest Root Mean Square Error that can be achieved by a model using provided inputs to predict the label.		
		* :code:`'Achievable Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a model using provided inputs to predict the label.


	.. admonition:: Theoretical Foundation

		Section :ref:`1 - Achievable Performance`.
	"""
	assert target_column in data_df.columns, 'The label column should be a column of the dataframe.'
	assert problem_type.lower() in ['classification', 'regression']
	if problem_type.lower() == 'regression':
		assert np.can_cast(data_df[target_column], float), 'The target column should be numeric'

	k = 0
	kp = 0
	max_k = 100
	sys.stdout.write('\r')
	sys.stdout.write("[{:{}}] {:d}% ETA: {}".format("="*k+">", max_k, k, approx_opt_remaining_time(k)))
	sys.stdout.flush()

	file_identifier = upload_data(data_df)
	if file_identifier:
		job_id = VALUATION_JOB_IDS.get((file_identifier, target_column, problem_type), None)

		if job_id:
			api_response = APIClient.route(
				path='/wk/data-valuation', method='POST', 
				file_identifier=file_identifier, target_column=target_column, \
				problem_type=problem_type, \
				timestamp=int(time()), job_id=job_id, \
				snr=snr)
		else:
			api_response = APIClient.route(
				path='/wk/data-valuation', method='POST', \
				file_identifier=file_identifier, target_column=target_column, \
				problem_type=problem_type, timestamp=int(time()), \
				snr=snr)

		initial_time = time()
		while api_response.status_code == requests.codes.ok and k < max_k:
			if kp%2 != 0:
				sleep(2 if kp<5 else 10 if k < max_k-4 else 300)
				kp += 1
				k = kp//2
				sys.stdout.write('\r')
				sys.stdout.write("[{:{}}] {:d}%".format("="*k+">", max_k, k))
				sys.stdout.flush()
			else:
				try:
					sys.stdout.write('\r')
					response = api_response.json()
					if 'job_id' in response:
						job_id = response['job_id']
						VALUATION_JOB_IDS[(file_identifier, target_column, problem_type)] = job_id
						sleep(2 if kp<5 else 10 if k < max_k-4 else 300)
						kp += 1
						k = kp//2
						sys.stdout.write("[{:{}}] {:d}% ETA: {}".format("="*k+">", max_k, k, approx_opt_remaining_time(k)))
						sys.stdout.flush()
						# Note: it is important to pass the job_id to avoid being charged twice for the same work.
						api_response = APIClient.route(
							path='/wk/data-valuation', method='POST', 
							file_identifier=file_identifier, target_column=target_column, \
							problem_type=problem_type, \
							timestamp=int(time()), job_id=job_id, \
							snr=snr)
					else:
						duration = int(time()-initial_time)
						duration = str(duration) + 's' if duration < 60 else str(duration//60) + 'min'
						sys.stdout.write("[{:{}}] {:d}% ETA: {} Duration: {}".format("="*max_k, max_k, max_k, approx_opt_remaining_time(max_k), duration))
						sys.stdout.write('\n')
						sys.stdout.flush()
						result = {}
						if 'r-squared' in response:
							result['Achievable R-Squared'] = [response['r-squared']]

						if 'log-likelihood' in response:
							result['Achievable Log-Likelihood Per Sample'] = [response['log-likelihood']]

						if 'rmse' in response and problem_type.lower() == 'regression':
							result['Achievable RMSE'] = [response['rmse']]

						if 'accuracy' in response and problem_type.lower() == 'classification':
							result['Achievable Accuracy'] = [response['accuracy']]

						result = pd.DataFrame.from_dict(result)

						return result

				except:
					logging.exception('\nData valuation failed. Last HTTP code: %s' % api_response.status_code)
					return None


		if api_response.status_code != requests.codes.ok:
			try:
				response = api_response.json()
				if 'message' in response:
					logging.error('\n%s' % response['message'])
			except:
				logging.error('\nData valuation failed. Last HTTP code: %s' % api_response.status_code)

	return None


