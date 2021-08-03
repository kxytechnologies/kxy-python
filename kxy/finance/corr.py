#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO)
import requests
import sys
from time import time, sleep

import numpy as np
import pandas as pd

from kxy.api import APIClient, upload_data, approx_beta_remaining_time

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
	sys.stdout.write('\r')
	sys.stdout.write("[{:{}}] {:d}% ETA: {}".format("="*k+">", max_k, k, approx_beta_remaining_time(k)))
	sys.stdout.flush()

	df = data_df[[market_column, asset_column]]
	file_identifier = upload_data(df)
	if file_identifier:
		job_id = IACORR_JOB_IDS.get(file_identifier, None)

		if job_id:
			api_response = APIClient.route(
				path='/wk/ia-corr', method='POST', 
				file_identifier=file_identifier, market_column=market_column, \
				asset_column=asset_column, \
				timestamp=int(time()), job_id=job_id)
		else:
			api_response = APIClient.route(
				path='/wk/ia-corr', method='POST', \
				file_identifier=file_identifier, market_column=market_column, \
				asset_column=asset_column, \
				timestamp=int(time()))

		initial_time = time()
		while api_response.status_code == requests.codes.ok and k < max_k:
			if kp%2 != 0:
				sleep(2 if kp<5 else 5 if k < max_k-4 else 300)
				kp += 4
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
						IACORR_JOB_IDS[file_identifier] = job_id
						sleep(2 if kp<5 else 5 if k < max_k-4 else 300)
						kp += 4
						k = kp//2
						sys.stdout.write("[{:{}}] {:d}% ETA: {}".format("="*k+">", max_k, k, approx_beta_remaining_time(k)))
						sys.stdout.flush()
						# Note: it is important to pass the job_id to avoid being charged twice for the same work.
						api_response = APIClient.route(
							path='/wk/ia-corr', method='POST', 
							file_identifier=file_identifier, market_column=market_column, \
							asset_column=asset_column, \
							timestamp=int(time()), job_id=job_id)
					else:
						duration = int(time()-initial_time)
						duration = str(duration) + 's' if duration < 60 else str(duration//60) + 'min'
						sys.stdout.write("[{:{}}] {:d}% ETA: {} Duration: {}".format("="*max_k, max_k, max_k, approx_beta_remaining_time(max_k), duration))
						sys.stdout.write('\n')
						sys.stdout.flush()

						if 'ia-corr' in response:
							return response['ia-corr']
						else:
							return np.nan

				except:
					logging.exception('\nInformation-adjusted correlation failed. Last HTTP code: %s' % api_response.status_code)
					return None


		if api_response.status_code != requests.codes.ok:
			try:
				response = api_response.json()
				if 'message' in response:
					logging.error('\n%s' % response['message'])
			except:
				logging.error('\nInformation-adjusted correlation failed. Last HTTP code: %s' % api_response.status_code)

	return None



