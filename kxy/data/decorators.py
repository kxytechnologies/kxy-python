#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect

def decorate_methods(decorator, include=[]):
	"""
	Systematically decorates all methods of a class, except for the ones 
	listed in exclude.
	"""
	def decorate(cls):
		method_names = [_[0] for _ in inspect.getmembers(cls, inspect.isroutine) if _[0] in include]
		for attr in method_names:
			setattr(cls, attr, decorator(getattr(cls, attr)))
		return cls
	return decorate



def decorate_all_methods(decorator):
	"""
	Systematically decorates all methods of a class, except for the ones 
	listed in exclude.
	"""
	def decorate(cls):
		method_names = [_[0] for _ in inspect.getmembers(cls, inspect.isroutine)]
		for attr in method_names:
			setattr(cls, attr, decorator(getattr(cls, attr)))
		return cls
	return decorate