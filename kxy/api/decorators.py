#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================
kxy.api.decorators
==================
"""

from functools import wraps
import json
import os

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
			existing_key = config.get('KXY_API_KEY', None)
			return existing_key
	except:
		return None

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