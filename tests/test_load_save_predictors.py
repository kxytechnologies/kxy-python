import numpy as np

from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote, BankMarketing
from kxy.learning import get_xgboost_learner, get_tensorflow_dense_learner, get_pytorch_dense_learner, \
	get_lightgbm_learner_sklearn_api, get_lightgbm_learner_learning_api, get_sklearn_learner
from kxy.misc.predictors import RFEPredictor, BorutaPredictor
from kxy.learning.leanml_predictor import LeanMLPredictor


def test_rfe_predictor_sklearn():
	for clz in ['sklearn.linear_model.LassoCV', 'sklearn.linear_model.LinearRegression']:
		# Regression
		sklearn_regressor_cls = get_sklearn_learner(clz)
		dataset = Abalone()
		target_column = dataset.y_column
		df = dataset.df

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

		# Model building
		results = features_df.kxy.fit(target_column, sklearn_regressor_cls, \
			problem_type='regression', feature_selection_method='rfe', rfe_n_features=5)
		predictor = results['predictor']
		predictions = predictor.predict(features_df)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		predictor.save(path)

		loaded_predictor = RFEPredictor.load(path, sklearn_regressor_cls)
		loaded_predictions = loaded_predictor.predict(features_df)

		assert np.allclose(predictions, loaded_predictions)


def test_boruta_predictor_sklearn():
	for clz in ['sklearn.linear_model.LassoCV', 'sklearn.linear_model.LinearRegression']:
		# Regression
		sklearn_regressor_cls = get_sklearn_learner(clz)
		dataset = Abalone()
		target_column = dataset.y_column
		df = dataset.df

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

		# Model building
		results = features_df.kxy.fit(target_column, sklearn_regressor_cls, \
			problem_type='regression', feature_selection_method='boruta', rfe_n_features=5)
		predictor = results['predictor']
		predictions = predictor.predict(features_df)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		predictor.save(path)

		loaded_predictor = BorutaPredictor.load(path, sklearn_regressor_cls)
		loaded_predictions = loaded_predictor.predict(features_df)

		assert np.allclose(predictions, loaded_predictions)


def test_rfe_predictor_xgboost():
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
			problem_type='regression', feature_selection_method='rfe', rfe_n_features=5)
		predictor = results['predictor']
		predictions = predictor.predict(features_df)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		predictor.save(path)

		loaded_predictor = RFEPredictor.load(path, xgboost_regressor_cls)
		loaded_predictions = loaded_predictor.predict(features_df)

		assert np.allclose(predictions, loaded_predictions)


def test_boruta_predictor_xgboost():
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
			problem_type='regression', feature_selection_method='boruta', rfe_n_features=5)
		predictor = results['predictor']
		predictions = predictor.predict(features_df)
		path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, clz)
		predictor.save(path)

		loaded_predictor = BorutaPredictor.load(path, xgboost_regressor_cls)
		loaded_predictions = loaded_predictor.predict(features_df)

		assert np.allclose(predictions, loaded_predictions)



def test_rfe_predictor_lightgbm():
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
		problem_type='regression', feature_selection_method='rfe', rfe_n_features=5)
	predictor = results['predictor']
	predictions = predictor.predict(features_df)
	path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, 'lightbm-learning-api-regressor')
	predictor.save(path)

	loaded_predictor = RFEPredictor.load(path, lightgbm_regressor_cls)
	loaded_predictions = loaded_predictor.predict(features_df)

	assert np.allclose(predictions, loaded_predictions)


def test_boruta_predictor_lightgbm():
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
		problem_type='regression', feature_selection_method='boruta', rfe_n_features=5)
	predictor = results['predictor']
	predictions = predictor.predict(features_df)
	path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, 'lightbm-learning-api-regressor')
	predictor.save(path)

	loaded_predictor = BorutaPredictor.load(path, lightgbm_regressor_cls)
	loaded_predictions = loaded_predictor.predict(features_df)

	assert np.allclose(predictions, loaded_predictions)



def test_leanml_predictor_lightgbm():
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
		problem_type='regression')
	feature_columns = results['Selected Variables']
	predictor = results['predictor']
	predictions = predictor.predict(features_df[feature_columns])
	path = '../kxy/misc/cache/%s-%s.sav' % (dataset.name, 'lightbm-learning-api-regressor')
	predictor.save(path)

	loaded_predictor = LeanMLPredictor.load(path, lightgbm_regressor_cls)
	loaded_predictions = loaded_predictor.predict(features_df[feature_columns])

	assert np.allclose(predictions, loaded_predictions)


