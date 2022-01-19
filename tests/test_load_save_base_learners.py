import numpy as np

from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote, BankMarketing
from kxy.learning import get_xgboost_learner, get_tensorflow_dense_learner, get_pytorch_dense_learner, \
	get_lightgbm_learner_sklearn_api, get_lightgbm_learner_learning_api, get_sklearn_learner


def test_lean_boosted_sklearn_regressor():
	for clz in ['sklearn.neighbors.KNeighborsRegressor', 'sklearn.linear_model.LassoCV', 'sklearn.linear_model.LinearRegression']:
		# Regression
		sklearn_regressor_cls = get_sklearn_learner(clz)
		dataset = Abalone()
		target_column = dataset.y_column
		df = dataset.df

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

		# Model building
		results = features_df.kxy.fit(target_column, sklearn_regressor_cls, \
			problem_type='regression', additive_learning=True, return_scores=True, \
			n_down_perf_before_stop=1)
		model = results['predictor'].models[0]
		feature_columns = results['Selected Variables']
		x = features_df[feature_columns].values
		predictions = model.predict(x)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		model.save(path)

		loaded_model = sklearn_regressor_cls(path=path)
		loaded_predictions = loaded_model.predict(x)

		assert np.allclose(predictions, loaded_predictions)


def test_lean_boosted_sklearn_classifier():
	for clz in ['sklearn.neighbors.KNeighborsClassifier', 'sklearn.ensemble.RandomForestClassifier', 'sklearn.ensemble.AdaBoostClassifier']:
		# Classification
		sklearn_classifier_cls = get_sklearn_learner(clz)

		dataset = BankNote()
		target_column = dataset.y_column
		df = dataset.df

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

		# Model building
		results = features_df.kxy.fit(target_column, sklearn_classifier_cls, \
			problem_type='classification', additive_learning=True, return_scores=True, \
			n_down_perf_before_stop=1)
		model = results['predictor'].models[0]
		feature_columns = results['Selected Variables']
		x = features_df[feature_columns].values
		predictions = model.predict(x)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		model.save(path)

		loaded_model = sklearn_classifier_cls(path=path)
		loaded_predictions = loaded_model.predict(x)

		assert np.allclose(predictions, loaded_predictions)


def test_lean_boosted_xgboost_regressor():
	for clz in ['xgboost.XGBRegressor']:
		# Regression
		xgboost_regressor_cls = get_xgboost_learner(clz)
		dataset = Abalone()
		target_column = dataset.y_column
		df = dataset.df

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

		# Model building
		results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
			problem_type='regression', additive_learning=True, return_scores=True, \
			n_down_perf_before_stop=1)
		model = results['predictor'].models[0]
		feature_columns = results['Selected Variables']
		x = features_df[feature_columns].values
		predictions = model.predict(x)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		model.save(path)

		loaded_model = xgboost_regressor_cls(path=path)
		loaded_predictions = loaded_model.predict(x)

		assert np.allclose(predictions, loaded_predictions)


def test_lean_boosted_xgboost_classifier():
	for clz in ['xgboost.XGBClassifier']:
		# Classification
		xgboost_classifier_cls = get_xgboost_learner(clz)

		dataset = BankNote()
		target_column = dataset.y_column
		df = dataset.df

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

		# Model building
		results = features_df.kxy.fit(target_column, xgboost_classifier_cls, \
			problem_type='classification', additive_learning=True, return_scores=True, \
			n_down_perf_before_stop=1)
		model = results['predictor'].models[0]
		feature_columns = results['Selected Variables']
		x = features_df[feature_columns].values
		predictions = model.predict(x)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		model.save(path)

		loaded_model = xgboost_classifier_cls(path=path)
		loaded_predictions = loaded_model.predict(x)

		assert np.allclose(predictions, loaded_predictions)



def test_lean_boosted_lightgbm_sklearn_regressor():
	# Regression
	clz = 'lightgbm.LGBMRegressor'
	lightgbm_regressor_cls = get_lightgbm_learner_sklearn_api(clz)
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, lightgbm_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True, \
		n_down_perf_before_stop=1)
	model = results['predictor'].models[0]
	feature_columns = results['Selected Variables']
	x = features_df[feature_columns].values
	predictions = model.predict(x)
	path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
	model.save(path)

	loaded_model = lightgbm_regressor_cls(path=path)
	loaded_predictions = loaded_model.predict(x)

	assert np.allclose(predictions, loaded_predictions)



def test_lean_boosted_lightgbm_learning_regressor():
	# Regression
	params = params = {
		'objective': 'rmse',  
		'boosting_type': 'gbdt',
		'num_leaves': 100,
		'n_jobs': -1,
		'learning_rate': 0.1,
		'feature_fraction': 0.8,
		'bagging_fraction': 0.8,
		'verbose': -1,
	}
	lightgbm_regressor_cls = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
		early_stopping_rounds=50, verbose_eval=50, split_random_seed=42)
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, lightgbm_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True, \
		n_down_perf_before_stop=1)
	model = results['predictor'].models[0]
	feature_columns = results['Selected Variables']
	x = features_df[feature_columns].values
	predictions = model.predict(x)
	path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, 'lightbm-learning-api-regressor')
	model.save(path)

	loaded_model = lightgbm_regressor_cls(path=path)
	loaded_predictions = loaded_model.predict(x)

	assert np.allclose(predictions, loaded_predictions)



def test_lean_boosted_tensorflow_regressor():
	import tensorflow as tf
	tf.random.set_seed(0)
	# Regression
	layers = [(10, 'relu'), (5, 'relu'), (1, 'linear')]
	loss = 'mean_absolute_error'
	optimizer = 'adam'
	clz = 'KerasRegressor'
	tf_regressor_cls = get_tensorflow_dense_learner(clz, layers, loss, optimizer=optimizer, \
		epochs=10, batch_size=100)
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, tf_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True, \
		n_down_perf_before_stop=1)
	model = results['predictor'].models[0]
	feature_columns = results['Selected Variables']
	x = features_df[feature_columns].values
	predictions = model.predict(x)
	path = '../kxy/misc/cache/%s-%s.h5' % (dataset.name, clz)
	model.save(path)

	n_vars = x.shape[1]
	loaded_model = tf_regressor_cls(n_vars=n_vars, path=path)
	loaded_predictions = loaded_model.predict(x)

	assert np.allclose(predictions, loaded_predictions)



def test_lean_boosted_pytorch_regressor():
	import torch
	torch.manual_seed(0)
	# Regression
	layers = [(10, 'relu'), (5, 'relu'), (1, None)]
	clz = 'skorch.NeuralNetRegressor'
	pt_regressor_cls = get_pytorch_dense_learner(clz, layers, max_epochs=10, batch_size=100)
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, pt_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True, \
		n_down_perf_before_stop=1)
	model = results['predictor'].models[0]
	feature_columns = results['Selected Variables']
	x = features_df[feature_columns].values
	predictions = model.predict(x)
	path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
	model.save(path)

	n_vars = x.shape[1]
	loaded_model = pt_regressor_cls(n_vars=n_vars, path=path)
	loaded_predictions = loaded_model.predict(x)

	assert np.allclose(predictions, loaded_predictions)






	