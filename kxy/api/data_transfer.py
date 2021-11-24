#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
To run our analyzes, the KXY backend needs your data. The methods below are the only methods involved in sharing your data with us. The :code:`kxy` package only uploads your data `if` and `when` needed.
"""
import hashlib
import logging
logging.basicConfig(level=logging.INFO)
from time import time
import requests

import pandas as pd
import numpy as np
try:
	get_ipython().__class__.__name__
	from halo import HaloNotebook as Halo
except:
	from halo import Halo

from .client import APIClient


UPLOADED_FILES = {}

def generate_upload_url(file_name):
	"""
	Requests a pre-signed URL to upload a dataset.

	Parameters
	----------
	file_name: str
		A string that uniquely identifies the content of the file.

	Returns
	-------
	d : dict or None
		The dictionary containing the pre-signed url.
	"""
	api_response = APIClient.route(
			path='/wk/generate-signed-upload-url', method='POST',\
			file_name=file_name, timestamp=int(time()))

	if api_response.status_code == requests.codes.ok:
		api_response = api_response.json()
		if 'presigned_url' in api_response:
			presigned_url = api_response['presigned_url']
			return presigned_url

		elif api_response.get('file_already_exists', False):
			logging.debug('This file was previously uploaded.')
			return {}

		else:
			return None

	else:
		api_response = api_response.json()
		if 'message' in api_response:
			logging.warning("\n%s" % api_response['message'])
		return None


def upload_data(df):
	"""
	Updloads a dataframe to kxy servers.

	Parameters
	----------
	df: pd.DataFrame
		The dataframe to upload.

	Returns
	-------
	d : bool
		Whether the upload was successful.
	"""
	logging.debug('')
	logging.debug('Hashing the data to form the file name')
	content = pd.util.hash_pandas_object(df).to_string()
	data_identifier = hashlib.sha256(content.encode()).hexdigest()
	columns = str(sorted([col for col in df.columns]))
	columns_identifier = hashlib.sha256(columns.encode()).hexdigest()
	identifier = hashlib.sha256((data_identifier+columns_identifier).encode()).hexdigest()
	memory_usage = df.memory_usage(index=False).sum()/(1024.0*1024.0*1024.0)
	file_name = identifier + '.parquet.gzip' if memory_usage > 1.5 else identifier + '.parquet' if memory_usage > 0.5 else identifier + '.csv'
	logging.debug('Done hashing the data')

	if UPLOADED_FILES.get(identifier, False):
		logging.debug('The file with identifier %s was previously uplooaded' % identifier)
		return file_name

	logging.debug('Requesting a signed upload URL')
	presigned_url = generate_upload_url(file_name)

	if presigned_url is None:
		logging.warning('Failed to retrieve the signed upload URL')
		return None
	else:
		logging.debug('Signed upload URL retrieved')

	if presigned_url == {}:
		logging.debug('This file was previously uploaded')
		UPLOADED_FILES[identifier] = True
		return file_name


	logging.debug('Preparing data for upload')
	spinner = Halo(text='Preparing data upload', spinner='dots')
	spinner.start()
	if file_name.endswith('.parquet.gzip'):
		# Truncate floats with excessive precision to save space.
		df.columns = df.columns.astype(str)
		_bytes = df.to_parquet(index=False, compression='gzip')
	elif file_name.endswith('.parquet'):
		# Truncate floats with excessive precision to save space.
		df.columns = df.columns.astype(str)
		_bytes = df.to_parquet(index=False)
	else:
		_bytes = df.to_csv(index=False)
	spinner.succeed()

	files = {'file': _bytes}
	url = presigned_url['url']
	data = presigned_url['fields']
	logging.debug('Done preparing the data to upload')	
	logging.debug('Uploading the data')
	spinner.start('Uploading data')
	upload_response = requests.post(url, data=data, files=files)
	spinner.succeed()

	if upload_response.status_code in [requests.codes.ok, requests.codes.created, requests.codes.accepted, requests.codes.no_content]:
		logging.debug('Data successfully uploaded')
		UPLOADED_FILES[identifier] = True
		return file_name
	else:
		logging.warning('Failed to upload the file. Received status code %s.' % (upload_response.status_code))

	return None


