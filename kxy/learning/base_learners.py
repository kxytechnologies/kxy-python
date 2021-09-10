#!/usr/bin/env python
# -*- coding: utf-8 -*-

MODELS_TAKING_FLAT_OUTPUTS = [\
	'sklearn.ensemble.RandomForestRegressor', \
	'sklearn.neural_network.MLPRegressor', 
]

def get_sklearn_learner(class_name, *args, **kwargs):
	'''
	Generate base learner class as a subclass of an sklearn learner class, but one whose hyper-parameters are frozen to user-specified values.

	The class name should be full (e.g. sklearn.ensemble.RandomForestRegressor).
	'''
	import sklearn.ensemble
	import sklearn.gaussian_process
	import sklearn.linear_model
	import sklearn.kernel_ridge
	import sklearn.neural_network
	import sklearn.svm
	import sklearn.tree

	BaseLearner = eval(class_name)
	class Learner(BaseLearner):
		def __init__(self,):
			super(Learner, self).__init__(*args, **kwargs)

		def fit(self, *args, **kwargs):
			args_ = list(args)
			args_[1] = args_[1].flatten() if class_name in MODELS_TAKING_FLAT_OUTPUTS else args_[1]
			return super(Learner, self).fit(*args_, **kwargs)


	return Learner

def get_xgboost_learner(class_name, *args, **kwargs):
	'''
	Generate base learner class as a subclass of an xgboost learner class, but one whose hyper-parameters are frozen to user-specified values.

	The class name should be full (e.g. xgboost.XGBRegressor).
	'''
	import xgboost
	BaseLearner = eval(class_name)
	class Learner(BaseLearner):
		def __init__(self,):
			super(Learner, self).__init__(*args, **kwargs)

	return Learner

# TODO: Add support for tensorflow learners and learners in other frameworks.