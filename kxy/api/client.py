#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python client for the KXY RESTful API.
"""

from functools import lru_cache
import os
import requests

from .decorators import requires_api_key, get_api_key


class APIClient(object):
	"""
	Python client for the RESTful KXY API. All API methods require an API key. The API key must be set by running :code:`kxy configure` from the terminal.
	"""
	@staticmethod
	def stage():
		"""
		Defines the deployment stage of the RESTful API the client should talk to.

		Returns
		-------
		v : str
			The API stage to use.
		"""
		return 'v0'

	@staticmethod
	def url(path):
		"""
		Turns a relative path into a full API endpoint url.

		Parameters
		----------
		path: str
			The relative path of the API resource.

		Returns
		-------
		u : str
			The full URL of the API resource.
		"""
		path = path.strip('/')

		return 'https://api.kxysolutions.com/%s/' % APIClient.stage() + path


	@staticmethod
	@requires_api_key
	def get(path, **params):
		"""
		.. important:: This method requires a valid API key.

		Issues a GET request to the API resource identified by the input path.

		Parameters
		----------
		path: str
			The relative path of the API resource.
		params: dict, optional
			The query parameters of the GET request. Any keyword argument is 
			automatically interpreted as a request parameter, its name is used
			as the parameter name, and its value as the parameter value.

		Returns
		-------
		response: requests.Response
			The response of the API. The request HTTP status code can be accessed
			through `response.status_code`. To check if the request was succesful,
			inspect `response.ok`. When the API returned data, they can be accessed
			through `response.json()`. Supported status codes are:

			200: 
				The request was successful and the API returned some data accessible through
				`response.json()`.
			403: 
				The request failed because some parameter are either invalid or missing.
				Check `response.json()['reason']` for more information.
			404:
				The request failed because the API couldn't yet solve the problem of interest.
				You should typically try again another time. Check `response.json()['reason']`
				for more information.
		"""
		url = APIClient.url(path)
		api_key = get_api_key()
		response = requests.get(url, params=params, headers={'x-api-key': api_key, \
			'content-type': 'application/json'})

		return response


	@staticmethod
	@requires_api_key
	def post(path, **params):
		"""
		.. important:: This method requires a valid API key.

		Issues a POST request to the API resource identified by the input path.

		Parameters
		----------
		path: str
			The relative path of the API resource.
		params: dict, optional
			The data to be submitted to the API as part of the POST request, as 
			a JSON. Any keyword argument is automatically interpreted as a 
			key of the JSON data that will be submitted to the API, 
			and its value the associated value in the JSON.

		Returns
		-------
		response: requests.Response
			The response of the API. The request HTTP status code can be accessed
			through `response.status_code`. To check if the request was succesful,
			inspect `response.ok`. When the API returned data, they can be accessed
			through `response.json()`.

			Supported status codes are:

			200: 
				The request was successful and the API returned some data accessible through
				`response.json()`.
			403: 
				The request failed because some parameter are either invalid or missing.
				Check `response.json()['reason']` for more information.
			404:
				The request failed because the API couldn't yet solve the problem of interest.
				You should typically try again another time. Check `response.json()['reason']`
				for more information.
		"""
		url = APIClient.url(path)
		api_key = get_api_key()
		response = requests.post(url, json=params, headers={'x-api-key': api_key, \
			'content-type': 'application/json'})

		return response


	@staticmethod
	@lru_cache(maxsize=32)
	def route(path=None, method=None, **params):
		"""
		.. important:: This method requires a valid API key.

		Generic method to issue a GET or a POST request to the API resource identified
		by the input path.

		Parameters
		----------
		path: str
			The relative path of the API resource.

		method: str
			The REST method. Should be either `'GET'` or `'POST'`.

		params: dict, optional
			The data to be submitted to the API as a JSON for POST requests, or
			query parameters in the case of GET requests.

		Returns
		-------
		response: requests.Response
			The response of the API. The request HTTP status code can be accessed
			through `response.status_code`. To check if the request was succesful,
			inspect `response.ok`. When the API returned data, they can be accessed
			through `response.json()`.

			Supported status codes are:

			200: 
				The request was successful and the API returned some data accessible through
				`response.json()`.
			403: 
				The request failed because some parameter are either invalid or missing.
				Check `response.json()['reason']` for more information.
			404:
				The request failed because the API couldn't yet solve the problem of interest.
				You should typically try again another time. Check `response.json()['reason']`
				for more information.

		Raises
		------
		ValueError
			If path is None or method is neither 'GET', nor 'POST'.
		"""
		if path is None or method is None or \
				method.upper() not in ('GET', 'POST'):
			return None

		if method.upper() == 'GET':
			return APIClient.get(path, **params)

		if method.upper() == 'POST':
			return APIClient.post(path, **params)




