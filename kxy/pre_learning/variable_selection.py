#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimation of the top-:math:`k` most valuable variables in a supervised learning problem for every possible :math:`k`, and 
the corresponding achievable performances.
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
VARIABLE_SELECTION_JOB_IDS = {}

def variable_selection(data_df, target_column, problem_type, snr='auto'):
	"""
	.. _variable-selection:
	Runs the model-free variable selection analysis.

	The first variable is the variable that explains the label the most, when used in isolation. The second variable is the variable that complements the first variable the most for predicting the label etc.

	Running performances should be understood as the performance achievable when trying to predict the label using variables with selection order smaller or equal to that of the row.

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
	result : pandas.DataFrame
		The result is a pandas.Dataframe with columns (where applicable):

		* :code:`'Selection Order'`: The order in which the associated variable was selected, starting at 1 for the most important variable.
		* :code:`'Variable'`: The column name corresponding to the input variable.
		* :code:`'Running Achievable R-Squared'`: The highest :math:`R^2` that can be achieved by a classification model using all variables selected so far, including this one.
		* :code:`'Running Achievable Accuracy'`: The highest classification accuracy that can be achieved by a classification model using all variables selected so far, including this one.
		* :code:`'Running Achievable RMSE'`: The highest classification accuracy that can be achieved by a classification model using all variables selected so far, including this one.


	.. admonition:: Theoretical Foundation

		Section :ref:`2 - Variable Selection Analysis`.

	"""
	assert target_column in data_df.columns, 'The label column should be a column of the dataframe.'
	assert problem_type.lower() in ['classification', 'regression']
	if problem_type.lower() == 'regression':
		assert np.can_cast(data_df[target_column], float), 'The target column should be numeric'

	file_name = upload_data(data_df)
	spinner = Halo(text='Waiting for results from the backend.', spinner='dots')
	spinner.start()

	k = 0
	max_k = 100

	if file_name:
		job_id = VARIABLE_SELECTION_JOB_IDS.get((file_name, target_column, problem_type, snr), None)
		if job_id:
			api_response = APIClient.route(
				path='/wk/variable-selection', method='POST', \
				file_name=file_name, target_column=target_column, \
				problem_type=problem_type, timestamp=int(time()), job_id=job_id, \
				snr=snr)
		else:
			api_response = APIClient.route(
				path='/wk/variable-selection', method='POST', \
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
					
				if ('job_id' in response) and ('selection_order' not in response):
					job_id = response['job_id']
					VARIABLE_SELECTION_JOB_IDS[(file_name, target_column, problem_type, snr)] = job_id
					k += 1
					sleep(15.)

					# Note: it is important to pass the job_id to avoid being charged twice for the work.
					api_response = APIClient.route(
						path='/wk/variable-selection', method='POST', \
						file_name=file_name, target_column=target_column, \
						problem_type=problem_type, timestamp=int(time()), job_id=job_id, \
						snr=snr)

					try:
						response = api_response.json()
						if 'eta' in response:
							progress_text = '%s%% Completed.' % response['progress_pct'] if 'progress_pct' in response else ''
							spinner.text = 'Waiting for results from the backend. ETA: %s. %s' % (response['eta'], progress_text)
					except:
						pass

				if ('job_id' not in response) or ('selection_order' in response):
					duration = int(time()-initial_time)
					duration = str(duration) + 's' if duration < 60 else str(duration//60) + 'min'

					result = {}
					if 'selection_order' in response:
						result['Selection Order'] = response['selection_order']				

					if 'variable' in response:
						result['Variable'] = response['variable']	

					if 'r-squared' in response:
						result['Running Achievable R-Squared'] = response['r-squared']

					if 'log-likelihood' in response:
						result['Running Achievable Log-Likelihood Per Sample'] = response['log-likelihood']

					if 'rmse' in response and problem_type.lower() == 'regression':
						result['Running Achievable RMSE'] = response['rmse']

					if 'accuracy' in response and problem_type.lower() == 'classification':
						result['Running Achievable Accuracy'] = response['accuracy']

					result = pd.DataFrame.from_dict(result)

					if 'selection_order' in response:
						result.set_index('Selection Order', inplace=True)

					spinner.text = 'Received results from the backend after %s.' % duration
					spinner.succeed()
					return result


			except:
				logging.exception('\nVariable selection failed. Last HTTP code: %s, Content: %s' % (api_response.status_code, api_response.content))
				spinner.text = 'The backend encountered an unexpected error we are looking into. Please try again later.'
				spinner.fail()
				return None

		if api_response.status_code != requests.codes.ok:
			spinner.text = 'The backend is taking longer than expected. Please try again later.'
			spinner.fail()
			try:
				response = api_response.json()
				if 'message' in response:
					logging.error('\n%s' % response['message'])
			except:
				logging.error('\nVariable selection failed. Last HTTP code: %s, Content: %s' % (api_response.status_code, api_response.content))

	return None
