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

try:
	get_ipython().__class__.__name__
	from halo import HaloNotebook as Halo
except:
	from halo import Halo

from kxy.api import APIClient, upload_data

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
	max_k = 100

	file_name = upload_data(data_df)
	spinner = Halo(text='Waiting for results from the backend.', spinner='dots')
	spinner.start()

	if file_name:
		job_id = VALUATION_JOB_IDS.get((file_name, target_column, problem_type, snr), None)

		if job_id:
			api_response = APIClient.route(
				path='/wk/data-valuation', method='POST', 
				file_name=file_name, target_column=target_column, \
				problem_type=problem_type, \
				timestamp=int(time()), job_id=job_id, \
				snr=snr)
		else:
			api_response = APIClient.route(
				path='/wk/data-valuation', method='POST', \
				file_name=file_name, target_column=target_column, \
				problem_type=problem_type, timestamp=int(time()), \
				snr=snr)

		initial_time = time()
		while api_response.status_code == requests.codes.ok and k < max_k:
			try:
				response = api_response.json()
				if 'eta' in response:
					progress_text = '%s%% Completed.' % response['progress_pct'] if 'progress_pct' in response else ''
					spinner.text = 'Waiting for results from the backend. ETA: %s. %s' % (response['eta'], progress_text)

				if ('job_id' in response) and ('r-squared' not in response):
					job_id = response['job_id']
					VALUATION_JOB_IDS[(file_name, target_column, problem_type, snr)] = job_id
					k += 1
					sleep(15.)

					# Note: it is important to pass the job_id to avoid being charged twice for the same work.
					api_response = APIClient.route(
						path='/wk/data-valuation', method='POST', 
						file_name=file_name, target_column=target_column, \
						problem_type=problem_type, \
						timestamp=int(time()), job_id=job_id, \
						snr=snr)

					try:
						response = api_response.json()
						if 'eta' in response:
							progress_text = '%s%% Completed.' % response['progress_pct'] if 'progress_pct' in response else ''
							spinner.text = 'Waiting for results from the backend. ETA: %s. %s' % (response['eta'], progress_text)
					except:
						pass

				if ('job_id' not in response) or ('r-squared' in response):
					duration = int(time()-initial_time)
					duration = str(duration) + 's' if duration < 60 else str(duration//60) + 'min'

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

					spinner.text = 'Received results from the backend after %s.' % duration
					spinner.succeed()

					return result

			except:
				logging.exception('\nData valuation failed. Last HTTP code: %s' % api_response.status_code)
				spinner.text = 'The backend encountered an unexpected error we are looking into. Please try again later.'
				spinner.fail()
				return None


		if api_response.status_code != requests.codes.ok:
			spinner.text = 'The backend is taking longer than expected. Try again later.'
			spinner.fail()
			try:
				response = api_response.json()
				if 'message' in response:
					logging.error('\n%s' % response['message'])
			except:
				logging.error('\nData valuation failed. Last HTTP code: %s' % api_response.status_code)

	return None


