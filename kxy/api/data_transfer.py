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

from .client import APIClient


UPLOADED_FILES = {}

def generate_upload_url(identifier):
	"""
	Requests a pre-signed URL to upload a dataset.

	Parameters
	----------
	identifier: str
		A string that uniquely identifies the content of the file.

	Returns
	-------
	d : dict or None
		The dictionary containing the pre-signed url.
	"""
	api_response = APIClient.route(
			path='/wk/generate-signed-upload-url', method='POST',\
			file_identifier=identifier, timestamp=int(time()))

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
	identifier = hashlib.sha256(df.to_string().encode()).hexdigest()

	if UPLOADED_FILES.get(identifier, False):
		logging.debug('The file with identifier %s was previously uplooaded' % identifier)
		return identifier

	presigned_url = generate_upload_url(identifier)

	if presigned_url is None:
		return None

	if presigned_url == {}:
		UPLOADED_FILES[identifier] = True
		return identifier

	file_name = identifier + '.csv'
	files = {'file': (file_name, df.to_csv(index=False))}
	url = presigned_url['url']
	data = presigned_url['fields']
	upload_response = requests.post(url, data=data, files=files)

	if upload_response.status_code in [requests.codes.ok, requests.codes.created, requests.codes.accepted, requests.codes.no_content]:
		UPLOADED_FILES[identifier] = True
		return identifier
	else:
		logging.warning('Failed to upload the file. Received status code %s.' % (upload_response.status_code))

	return None


