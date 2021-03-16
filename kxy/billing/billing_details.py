#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Everything billing.
"""
import logging
logging.basicConfig(level=logging.INFO)
import requests
from time import time

from kxy.api import APIClient


def get_upcoming_invoice():
	"""
	Retrieves all items that will show up in your next invoice.

	Returns
	-------
	d : dict
		The dictionary containing all items that will appear in your next invoice.
		E.g. :code:`{'Type of charge': {'total_usd': ..., 'quantity': ..., 'description': ..., 'billing_period_start_timestamp': ..., 'billing_period_end_timestamp': ...}, ... }`
	"""
	api_response = APIClient.route(
			path='/wk/billing/upcoming-invoice', method='POST',\
			timestamp=int(time()))
	try:
		if api_response.status_code == requests.codes.ok:
			api_response = api_response.json()
			return api_response
		else:
			return {}
	except:
		logging.exception('Failed to retrieve your upcoming invoice.')
		return {}
