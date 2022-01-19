#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle as pkl
import numpy as np

MODELS_TAKING_FLAT_OUTPUTS = [\
	'sklearn.ensemble.RandomForestRegressor', \
	'sklearn.neural_network.MLPRegressor', \
	'sklearn.linear_model.LassoCV',
	'lightgbm.LGBMClassifier', \
	'lightgbm.LGBMRegressor'
]

def get_sklearn_learner(class_name, *args, fit_intercept=True, fit_kwargs={}, predict_kwargs={}, **kwargs):
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
	import sklearn.neighbors

	global Learner
	if class_name == 'sklearn.linear_model.LassoCV':
		class Learner(sklearn.linear_model.LassoCV):
			eps = kwargs.get('eps', 0.001)
			n_alphas = kwargs.get('n_alphas', 100)
			alphas = kwargs.get('alphas', None)
			fit_intercept = kwargs.get('fit_intercept', True)
			precompute = kwargs.get('precompute', 'auto')
			max_iter = kwargs.get('max_iter', 1000)
			tol = kwargs.get('tol', 0.0001)
			copy_X = kwargs.get('copy_X', True)
			cv = kwargs.get('cv', None)
			verbose = kwargs.get('verbose', False)
			n_jobs = kwargs.get('n_jobs', None)
			positive = kwargs.get('positive', False)
			random_state = kwargs.get('random_state', None)
			selection = kwargs.get('selection', 'cyclic')
			normalize = kwargs.get('normalize', 'deprecated')

			def __init__(self, eps=eps, n_alphas=n_alphas, alphas=alphas, \
					fit_intercept=fit_intercept, precompute=precompute, \
					max_iter=max_iter, tol=tol, copy_X=copy_X, cv=cv, verbose=verbose, \
					n_jobs=n_jobs, positive=positive, random_state=random_state, \
					selection=selection, normalize=normalize):
				super(Learner, self).__init__(eps=eps, n_alphas=n_alphas, alphas=alphas, \
					fit_intercept=fit_intercept, precompute=precompute, \
					max_iter=max_iter, tol=tol, copy_X=copy_X, cv=cv, verbose=verbose, \
					n_jobs=n_jobs, positive=positive, random_state=random_state, \
					selection=selection, normalize=normalize)

			def fit(self, x, y):
				y_ = y.flatten() if class_name in MODELS_TAKING_FLAT_OUTPUTS else y
				return super(Learner, self).fit(x, y_, **fit_kwargs)

			def predict(self, x):
				return super(Learner, self).predict(x, **predict_kwargs)

			@property
			def feature_importances_(self):
				try:
					coef = np.abs(self.coef_)
					coef = np.array(coef)
					if len(coef.shape) > 1:
						return np.mean(coef, axis=0)
					else:
						return coef
				except:
					return []

			def __getstate__(self):
				try:
					state = super(Learner, self).__getstate__()
				except AttributeError:
					state = super(Learner, self).__dict__.copy()
				state['feature_importances_'] = self.feature_importances_
				return state

			def save(self, path):
				with open(path, 'wb') as f:
					pkl.dump(self, f)

			def __setstate__(self, state):
				self.__dict__ = state
				self.predict = lambda x: super(Learner, self).predict(x, **predict_kwargs)

			@classmethod
			def load(cls, path):
				with open(path, 'rb') as f:
					model = pkl.load(f)
				return model


	elif class_name == 'sklearn.linear_model.LinearRegression':
		fit_intercept = kwargs.get('fit_intercept', True)
		copy_X = kwargs.get('copy_X', True)
		n_jobs = kwargs.get('n_jobs', None)
		normalize = kwargs.get('normalize', 'deprecated')

		class Learner(sklearn.linear_model.LinearRegression):
			def __init__(self, fit_intercept=fit_intercept,\
					copy_X=copy_X, n_jobs=n_jobs, normalize=normalize):
				super(Learner, self).__init__(fit_intercept=fit_intercept,\
					copy_X=copy_X, n_jobs=n_jobs, normalize=normalize)

			def fit(self, x, y):
				y_ = y.flatten() if class_name in MODELS_TAKING_FLAT_OUTPUTS else y
				return super(Learner, self).fit(x, y_, **fit_kwargs)

			def predict(self, x):
				return super(Learner, self).predict(x, **predict_kwargs)

			@property
			def feature_importances_(self):
				try:
					coef = np.abs(self.coef_)
					coef = np.array(coef)
					if len(coef.shape) > 1:
						return np.mean(coef, axis=0)
					else:
						return coef
				except:
					return []

			def __getstate__(self):
				try:
					state = super(Learner, self).__getstate__()
				except AttributeError:
					state = super(Learner, self).__dict__.copy()
				state['feature_importances_'] = self.feature_importances_
				return state

			def save(self, path):
				with open(path, 'wb') as f:
					pkl.dump(self, f)

			def __setstate__(self, state):
				self.__dict__ = state
				self.predict = lambda x: super(Learner, self).predict(x, **predict_kwargs)

			@classmethod
			def load(cls, path):
				with open(path, 'rb') as f:
					model = pkl.load(f)
				return model

	else:
		BaseLearner = eval(class_name)
		class Learner(BaseLearner):
			def __init__(self):
				super(Learner, self).__init__(*args, **kwargs)

			def fit(self, x, y):
				y_ = y.flatten() if class_name in MODELS_TAKING_FLAT_OUTPUTS else y
				return super(Learner, self).fit(x, y_, **fit_kwargs)

			def predict(self, x):
				return super(Learner, self).predict(x, **predict_kwargs)

			@property
			def feature_importances_(self):
				try:
					return super(Learner, self).feature_importances_
				except:
					try:
						coef = np.abs(self.coef_)
						coef = np.array(coef)
						if len(coef.shape) > 1:
							return np.mean(coef, axis=0)
						else:
							return coef
					except:
						return []

			def __getstate__(self):
				try:
					state = super(Learner, self).__getstate__()
				except AttributeError:
					state = super(Learner, self).__dict__.copy()
				state['feature_importances_'] = self.feature_importances_
				return state

			def save(self, path):
				with open(path, 'wb') as f:
					pkl.dump(self, f)

			def __setstate__(self, state):
				self.__dict__ = state
				self.predict = lambda x: super(Learner, self).predict(x, **predict_kwargs)

			@classmethod
			def load(cls, path):
				with open(path, 'rb') as f:
					model = pkl.load(f)
				return model


	def create_instance(n_vars=None, path=None):
		if path is None:
			return Learner()
		else:
			return Learner.load(path)

	return create_instance



def get_xgboost_learner(class_name, *args, fit_kwargs={}, predict_kwargs={}, **kwargs):
	'''
	Generate a base learner class as a subclass of an xgboost learner class, but one whose hyper-parameters are frozen to user-specified values.

	The class name should be full (e.g. xgboost.XGBRegressor).
	'''
	import xgboost
	global Learner

	BaseLearner = eval(class_name)
	class Learner(BaseLearner):
		def __init__(self):
			super(Learner, self).__init__(*args, **kwargs)

		def fit(self, x, y):
			return super(Learner, self).fit(x, y, **fit_kwargs)

		def predict(self, x):
			return super(Learner, self).predict(x, **predict_kwargs)

		def __getstate__(self):
			try:
				state = super(Learner, self).__getstate__()
			except AttributeError:
				state = super(Learner, self).__dict__.copy()
			return state

		def save(self, path):
			with open(path, 'wb') as f:
				pkl.dump(self, f)

		def __setstate__(self, state):
			self.__dict__ = state
			self.predict = lambda x: super(Learner, self).predict(x, **predict_kwargs)

		@classmethod
		def load(cls, path):
			with open(path, 'rb') as f:
				model = pkl.load(f)
			return model


	def create_instance(n_vars=None, path=None):
		if path is None:
			return Learner()
		else:
			return Learner.load(path)

	return create_instance



def get_lightgbm_learner_sklearn_api(class_name, boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, \
		subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, \
		subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, \
		silent='warn', importance_type='split', fit_kwargs={}, predict_kwargs={}, **kwargs):
	'''
	Generate a base learner class as a subclass of an lightgbm learner class (using the sklearn api), but one whose hyper-parameters are frozen to user-specified values.

	The class name should be full (e.g. lightgbm.LGBMRegressor).
	'''
	import lightgbm
	global Learner
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

		def fit(self, x, y):
			y_ = y.flatten() if class_name in MODELS_TAKING_FLAT_OUTPUTS else y
			return super(Learner, self).fit(x, y_, **fit_kwargs)

		def predict(self, x):
			return super(Learner, self).predict(x, **predict_kwargs)	

		def __getstate__(self):
			try:
				state = super(Learner, self).__getstate__()
			except AttributeError:
				state = super(Learner, self).__dict__.copy()
			return state

		def save(self, path):
			with open(path, 'wb') as f:
				pkl.dump(self, f)

		def __setstate__(self, state):
			self.__dict__ = state
			self.predict = lambda x: super(Learner, self).predict(x, **predict_kwargs)

		@classmethod
		def load(cls, path):
			with open(path, 'rb') as f:
				model = pkl.load(f)
			return model


	def create_instance(n_vars=None, path=None):
		if path is None:
			return Learner()
		else:
			return Learner.load(path)

	return create_instance



def get_lightgbm_learner_learning_api(params, num_boost_round=100, fobj=None, feval=None, init_model=None, feature_name='auto', \
		categorical_feature='auto', early_stopping_rounds=None, verbose_eval='warn', learning_rates=None, \
		keep_training_booster=False, callbacks=None, split_random_seed=None, weight_func=None, verbose=-1, \
		importance_type='split'):
	'''
	Generate a base learner class as a subclass of an lightgbm learner class (using the regular training api), but one whose hyper-parameters are frozen to user-specified values.

	All parameters are the same as :code:`lightgbm.train` parameters with the same name.
	'''
	import lightgbm
	from sklearn.model_selection import train_test_split
	global Learner

	class Learner(object):
		def __init__(self,):
			if 'verbose' not in params:
				params['verbose'] = verbose
			self.params = params
			self.num_boost_round = num_boost_round
			self.fobj = fobj
			self.feval = feval
			self.init_model = init_model
			self.feature_name = feature_name
			self.categorical_feature = categorical_feature
			self.early_stopping_rounds = early_stopping_rounds
			self.evals_result = {}
			self.verbose_eval = verbose_eval
			self.learning_rates = learning_rates
			self.keep_training_booster = keep_training_booster
			self.callbacks = callbacks
			self.weight_func = weight_func
			self.importance_type = importance_type

		def fit(self, x, y):
			x_train, x_val, y_train,  y_val = train_test_split(x, y, test_size=0.2, random_state=split_random_seed)
			y_train = y_train.ravel()
			y_val = y_val.ravel()
			if self.weight_func:
				train_weights = self.weight_func(x_train, y_train)
				val_weights   = self.weight_func(x_val, y_val)
				train_dataset = lightgbm.Dataset(x_train, y_train, weight=train_weights)
				val_dataset = lightgbm.Dataset(x_val, y_val, weight=val_weights)
			else:
				train_dataset = lightgbm.Dataset(x_train, y_train)
				val_dataset = lightgbm.Dataset(x_val, y_val)
				
			self._model = lightgbm.train(\
				self.params, train_dataset, num_boost_round=self.num_boost_round, \
				valid_sets=[train_dataset, val_dataset], valid_names=['training', 'validation'], fobj=self.fobj, \
				feval=self.feval, init_model=self.init_model, feature_name=self.feature_name, \
				categorical_feature=self.categorical_feature, early_stopping_rounds=self.early_stopping_rounds, \
				evals_result=self.evals_result, verbose_eval=self.verbose_eval, learning_rates=self.learning_rates, \
				keep_training_booster=self.keep_training_booster, callbacks=self.callbacks)

		def predict(self, x):
			'''
			'''
			assert hasattr(self, '_model'), 'The model has not been fitted yet'
			y_pred = self._model.predict(x)

			if self.params.get('objective', '') == 'binary':
				y_pred = (y_pred > 0.5).astype(int)

			if self.params.get('objective', '') == 'multiclass':
				y_pred = np.argmax(y_pred, axis=1).astype(int)

			return y_pred

		@property
		def feature_importances_(self):
			try:
				return self._model.feature_importance(importance_type=self.importance_type)
			except:
				return []

		def save(self, path):
			with open(path, 'wb') as f:
				pkl.dump(self._model, f)

		@classmethod
		def load(cls, path):
			with open(path, 'rb') as f:
				_model = pkl.load(f)
				learner = Learner()
				learner._model = _model
			return learner


	def create_instance(n_vars=None, path=None):
		if path is None:
			return Learner()
		else:
			return Learner.load(path)



	return create_instance



def get_tensorflow_dense_learner(class_name, layers, loss, optimizer='adam', fit_kwargs={}, predict_kwargs={}, **kwargs):
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
	kwargs : dict
		Other named parameters to use in the constructor of the base class.
	fit_kwargs : dict
		Arguments to be used to fit the model. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit for legal arguments.
	predict_kwargs : dict
		Arguments to be used to make predictions. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict for legal arguments.
	kwargs : dict
		Other named parameters to use in the constructor of the base class. E.g. :code:`epochs`, :code:`batch_size`.
		See https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasRegressor and https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier

	'''
	import tensorflow as tf
	from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
	from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
	from tensorflow.keras.models import load_model
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

			super(Learner, self).__init__(build_fn=build_fn, **kwargs)

		def fit(self, x, y):
			return super(Learner, self).fit(x, y, **fit_kwargs)

		def predict(self, x):
			return super(Learner, self).predict(x, **predict_kwargs)

		def save(self, path):
			return self.model.save(path)

		@classmethod
		def load(cls, path, n_vars):
			model = load_model(path)
			n_vars = 1 if n_vars is None else n_vars
			learner = Learner(n_vars)
			learner.model = model
			return learner


	def create_instance(n_vars=None, path=None):
		if path is None:
			return Learner(n_vars)
		else:
			return Learner.load(path, n_vars)


	return create_instance



def get_pytorch_dense_learner(class_name, layers, optimizer='Adam', fit_kwargs={}, predict_kwargs={}, **kwargs):
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
	kwargs : dict
		Other named parameters to use in the constructor of the base class.
	fit_kwargs : dict
		Arguments to be used to fit the model.
	predict_kwargs : dict
		Arguments to be used to make predictions.

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

			super(Learner, self).__init__(Module, optimizer=optimizer, device=device, **kwargs)


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

		def save(self, path):
			return self.save_params(f_params=path)

		@classmethod
		def load(cls, path, n_vars):
			learner = Learner(n_vars)
			learner.initialize()
			learner.load_params(f_params=path)
			return learner


	def create_instance(n_vars=None, path=None):
		if path is None:
			return Learner(n_vars)
		else:
			return Learner.load(path, n_vars)


	return create_instance




















