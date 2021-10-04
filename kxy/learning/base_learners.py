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
		def __init__(self):
			super(Learner, self).__init__(*args, **kwargs)

		def fit(self, *args, **kwargs):
			args_ = list(args)
			args_[1] = args_[1].flatten() if class_name in MODELS_TAKING_FLAT_OUTPUTS else args_[1]
			return super(Learner, self).fit(*args_, **kwargs)

	def create_instance(n_vars=None):
		return Learner()

	return create_instance



def get_xgboost_learner(class_name, *args, **kwargs):
	'''
	Generate a base learner class as a subclass of an xgboost learner class, but one whose hyper-parameters are frozen to user-specified values.

	The class name should be full (e.g. xgboost.XGBRegressor).
	'''
	import xgboost
	BaseLearner = eval(class_name)
	class Learner(BaseLearner):
		def __init__(self):
			super(Learner, self).__init__(*args, **kwargs)

	def create_instance(n_vars=None):
		return Learner()

	return create_instance



def get_lightgbm_learner(class_name, boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, \
		subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, \
		subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, \
		silent='warn', importance_type='split', **kwargs):
	'''
	Generate a base learner class as a subclass of an lightgbm learner class, but one whose hyper-parameters are frozen to user-specified values.

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

	def create_instance(n_vars=None):
		return Learner()

	return create_instance



def get_tensorflow_dense_learner(class_name, layers, loss, optimizer='adam', sk_params={}, fit_kwargs={}, predict_kwargs={}):
	'''
	Generate a base learner class as a subclass of a tensorflow learner class, but one whose hyper-parameters are frozen to user-specified values.

	This function requires both Tensorflow to be installed.

	Parameters
	----------
	class_name : str ('tf.python.keras.wrappers.scikit_learn.KerasRegressor' | 'tf.python.keras.wrappers.scikit_learn.KerasClassifier')
		Determines whether the learner is a regressor or a classifier.
	layers : list
		List of tuples of the form (n_neurons, activation)
	loss : str
		The tensorflow loss (e.g. 'mean_absolute_error')
	optimizer : str
		Which tensorflow optimizer to use to fit the model.
	sk_params : dict
		E.g.: :code:`{'epochs': 10, 'batch_size': 100}`. See https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasRegressor and https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier
	fit_kwargs : dict
		Arguments to be used to fit the model. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit for legal arguments.
	predict_kwargs : dict
		Arguments to be used to make predictions. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict for legal arguments.

	'''
	import tensorflow as tf
	from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
	from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
	BaseLearner = eval(class_name)

	class Learner(BaseLearner):
		def __init__(self, n_vars):
			assert len(layers) > 0, 'There should be at least one layer'

			def build_fn():
				model = tf.keras.Sequential()
				model.add(tf.keras.layers.Dense(layers[0][0], input_dim=n_vars, activation=layers[0][1]))
				for i in range(1, len(layers)):
					model.add(tf.keras.layers.Dense(layers[i][0], activation=layers[i][1]))
				model.compile(loss=loss, optimizer=optimizer)
				return model

			super(Learner, self).__init__(build_fn=build_fn, **sk_params)

		def fit(self, x, y):
			return super(Learner, self).fit(x, y, **fit_kwargs)

		def predict(self, x):
			return super(Learner, self).predict(x, **predict_kwargs)

	def create_instance(n_vars=None):
		return Learner(n_vars)

	return create_instance



def get_pytorch_dense_learner(class_name, layers, optimizer='Adam', sk_params={}, fit_kwargs={}, predict_kwargs={}):
	'''
	Generate a base learner class as a subclass of a pytorch learner class, but one whose hyper-parameters are frozen to user-specified values.

	The class name should either :code:`skorch.NeuralNetRegressor`, :code:`skorch.NeuralNetClassifier`, :code:`NeuralNetRegressor` or :code:`NeuralNetClassifier`.

	This function requires both PyTorch and Skorch to be installed.


	Parameters
	----------
	class_name : str ('skorch.NeuralNetRegressor' | 'skorch.NeuralNetClassifier')
		Determines whether the learner is a regressor or a classifier.
	layers : list
		List of tuples of the form (n_neurons, activation)
	optimizer : str
		Name of the class in :code:`torch.optim` to use as optimizer to train the model.
	sk_params : dict
		E.g.: :code:`{'epochs': 10, 'batch_size': 100}`. See https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasRegressor and https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier
	fit_kwargs : dict
		Arguments to be used to fit the model. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit for legal arguments.
	predict_kwargs : dict
		Arguments to be used to make predictions. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict for legal arguments.

	'''
	import numpy as np
	import torch
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	from torch import nn
	import torch.nn.functional as F
	import skorch
	class_name = 'skorch.%s' % class_name if 'skorch.' not in class_name else class_name
	BaseLearner = eval(class_name)
	optimizer = eval('torch.optim.%s' % optimizer)

	class Learner(BaseLearner):
		def __init__(self, n_vars):
			assert len(layers) > 0, 'There should be at least one layer'

			class Module(nn.Module):
				def __init__(self):
					super(Module, self).__init__()
					self.linear_transformations = nn.ModuleList()
					self.activations = []
					previous_dim = n_vars
					for _l in layers:
						self.linear_transformations += [nn.Linear(previous_dim, _l[0])]
						previous_dim = _l[0]
						self.activations  += [None if _l[1] is None else eval('F.%s' % _l[1], {'F': F})]

				def forward(self, X, **kwargs):
					for i in range(len(self.activations)):
						if self.activations[i] is None:
							X = self.linear_transformations[i](X)
						else:
							X = self.activations[i](self.linear_transformations[i](X))
					return X

			super(Learner, self).__init__(Module, optimizer=optimizer, device=device, **sk_params)


		def fit(self, x, y):
			try:
				return super(Learner, self).fit(x, y, **fit_kwargs)
			except:
				return super(Learner, self).fit(x.astype(np.float32), y.astype(np.float32), **fit_kwargs)

		def predict(self, x):
			try:
				return super(Learner, self).predict(x, **predict_kwargs)
			except:
				return super(Learner, self).predict(x.astype(np.float32), **predict_kwargs)


	def create_instance(n_vars=None):
		return Learner(n_vars)

	return create_instance




















