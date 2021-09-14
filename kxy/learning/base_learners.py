#!/usr/bin/env python
# -*- coding: utf-8 -*-

MODELS_TAKING_FLAT_OUTPUTS = [\
	'sklearn.ensemble.RandomForestRegressor', \
	'sklearn.neural_network.MLPRegressor', \
	'lightgbm.LGBMClassifier', \
	'lightgbm.LGBMRegressor'
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

def get_lightgbm_learner(class_name, boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, \
		subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, \
		subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, \
		silent='warn', importance_type='split', **kwargs):
	'''
	Generate base learner class as a subclass of an lightgbm learner class, but one whose hyper-parameters are frozen to user-specified values.

	The class name should be full (e.g. lightgbm.LGBMRegressor).
	'''
	import lightgbm
	BaseLearner = eval(class_name)

	class Learner(BaseLearner):
		def __init__(self, boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, \
				n_estimators=n_estimators, subsample_for_bin=subsample_for_bin, objective=objective, class_weight=class_weight, \
				min_split_gain=min_split_gain, min_child_weight=min_child_weight, min_child_samples=min_child_samples, subsample=subsample, \
				subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, \
				random_state=random_state, n_jobs=n_jobs, silent=silent, importance_type=importance_type, **kwargs):
			super(Learner, self).__init__(boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, \
				n_estimators=n_estimators, subsample_for_bin=subsample_for_bin, objective=objective, class_weight=class_weight, \
				min_split_gain=min_split_gain, min_child_weight=min_child_weight, min_child_samples=min_child_samples, subsample=subsample, \
				subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, \
				random_state=random_state, n_jobs=n_jobs, silent=silent, importance_type=importance_type, **kwargs)

		def fit(self, *args, **kwargs):
			args_ = list(args)
			args_[1] = args_[1].flatten() if class_name in MODELS_TAKING_FLAT_OUTPUTS else args_[1]
			return super(Learner, self).fit(*args_, **kwargs)

	return Learner




# TODO: Add support for tensorflow learners and learners in other frameworks.