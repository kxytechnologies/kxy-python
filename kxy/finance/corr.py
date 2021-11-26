#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
IACORR_JOB_IDS = {}

def information_adjusted_correlation(data_df, market_column, asset_column):
	"""
	Estimate the information-adjusted correlation between an asset return :math:`r` and the market return :math:`r_m`: :math:`\\text{IA-Corr}\\left(r, r_m \\right) := \\text{sgn}\\left(\\text{Corr}\\left(r, r_m \\right) \\right) \\left[1 - e^{-2I(r, r_m)} \\right]`, where :math:`\\text{sgn}\\left(\\text{Corr}\\left(r, r_m \\right) \\right)` the sign of the Pearson correlation coefficient.

	Unlike Pearson's correlation coefficient, which is 0 if and only if asset return and market return are **decorrelated** (i.e. they exhibit no linear relation), information-adjusted correlation is 0 if and only if market and asset returns are **statistically independent** (i.e. the exhibit no relation, linear or nonlinear).


	Parameters
	----------
	data_df : pandas.DataFrame
		The pandas DataFrame containing the data.
	market_column : str
		The name of the column containing market returns.
	asset_column : str
		The name of the column containing asset returns.


	Returns
	-------
	result : float
		The information-adjusted correlation.

	"""
	assert market_column in data_df.columns, 'The market column should be a column of the dataframe.'
	assert asset_column in data_df.columns, 'The asset column should be a column of the dataframe.'
	assert np.can_cast(data_df[market_column], float), 'The market return column should be numeric'
	assert np.can_cast(data_df[asset_column], float), 'The asset return column should be numeric'

	k = 0
	kp = 0
	max_k = 100
	spinner = Halo(text='Waiting for results from the backend.', spinner='dots')
	spinner.start()

	df = data_df[[market_column, asset_column]]
	file_name = upload_data(df)
	if file_name:
		job_id = IACORR_JOB_IDS.get(file_name, None)

		if job_id:
			api_response = APIClient.route(
				path='/wk/ia-corr', method='POST', 
				file_name=file_name, market_column=market_column, \
				asset_column=asset_column, \
				timestamp=int(time()), job_id=job_id)
		else:
			api_response = APIClient.route(
				path='/wk/ia-corr', method='POST', \
				file_name=file_name, market_column=market_column, \
				asset_column=asset_column, \
				timestamp=int(time()))

		initial_time = time()
		while api_response.status_code == requests.codes.ok and k < max_k:
			if kp%2 != 0:
				sleep(2 if kp<5 else 5 if k < max_k-4 else 300)
				kp += 4
				k = kp//2
			else:
				try:
					response = api_response.json()
					if 'job_id' in response:
						job_id = response['job_id']
						IACORR_JOB_IDS[file_name] = job_id
						sleep(2 if kp<5 else 5 if k < max_k-4 else 300)
						kp += 4
						k = kp//2

						# Note: it is important to pass the job_id to avoid being charged twice for the same work.
						api_response = APIClient.route(
							path='/wk/ia-corr', method='POST', 
							file_name=file_name, market_column=market_column, \
							asset_column=asset_column, \
							timestamp=int(time()), job_id=job_id)

						try:
							response = api_response.json()
							if 'eta' in response:
								progress_text = '%s%% Completed.' % response['progress_pct'] if 'progress_pct' in response else ''
								spinner.text = 'Waiting for results from the backend. ETA: %s. %s' % (response['eta'], progress_text)
						except:
							pass

					if 'job_id' not in response:
						duration = int(time()-initial_time)
						duration = str(duration) + 's' if duration < 60 else str(duration//60) + 'min'
						spinner.text = 'Received results from the backend in %s' % duration
						spinner.succeed()

						if 'ia-corr' in response:
							return response['ia-corr']
						else:
							return np.nan

				except:
					spinner.text = 'The backend encountered an unexpected error we are looking into. Please try again later.'
					spinner.fail()
					logging.exception('\nInformation-adjusted correlation failed. Last HTTP code: %s' % api_response.status_code)
					return None


		if api_response.status_code != requests.codes.ok:
			spinner.text = 'The backend is taking longer than expected. Please try again later.'
			spinner.fail()
			try:
				response = api_response.json()
				if 'message' in response:
					logging.error('\n%s' % response['message'])
			except:
				logging.error('\nInformation-adjusted correlation failed. Last HTTP code: %s' % api_response.status_code)

	return None



