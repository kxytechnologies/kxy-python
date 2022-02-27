#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================
kxy.api.decorators
==================
"""

from functools import wraps
import json
import logging
import os
import requests

TRIAL_API_KEY = 'SZiRisvhzC7KBgROZG5dE1VQIlE8Jk4DbQ1YZdZ0'

def get_api_key():
	"""
	Retrieves the store API key, or None if none was provided.
	"""
	home = os.path.expanduser("~")
	path = os.path.join(home, '.kxy')
	file_name = os.path.join(path, 'config')
	try:
		with open(file_name, 'r') as f:
			config = json.load(f)
			existing_key = config.get('KXY_API_KEY', TRIAL_API_KEY)
			return existing_key
	except:
		return os.environ.get('KXY_API_KEY', TRIAL_API_KEY)

	return None



def has_api_key():
	"""
	Returns whether or not an API key was provided as a result of running :code:`kxy configure`.
	"""
	return get_api_key() is not None



def requires_api_key(method):
	"""
	Decorator used to make functions and methods calls fail
	when they require an API key and the user did not provide on 
	by running :code:`kxy configure`. The decorated function or method 
	is otherwise not affected.

	Raises
	------
	AssertionError
		If an API key was not previously recorded.
	"""
	@wraps(method)
	def wrapper(*args, **kw):		
		assert has_api_key(), "An API key should be provided. Please run 'kxy configure'"
		return method(*args, **kw)

	return wrapper



def log_backend_warnings(method):
	"""
	Decorator used to make requests hitting the backend log backend warnings.
	"""
	@wraps(method)
	def wrapper(*args, **kw):
		response = method(*args, **kw)
		try:
			if response.status_code == requests.codes.ok:
				response_json = response.json()
				if 'warning' in response_json:
					logging.warning('%s' % response_json['warning'])
		except:
			pass
		return response

	return wrapper





