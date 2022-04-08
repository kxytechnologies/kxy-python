import numpy as np

import kxy
from kxy.learning import get_sklearn_learner, get_lightgbm_learner_learning_api, get_xgboost_learner
from kxy.pfs import PCAPredictor, PCA
from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote, BankMarketing


def test_shape():
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])
	y = features_df[target_column].values
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x = features_df[x_columns].values

	# Principal features construction
	feature_directions = PCA().fit(x, y)
	assert feature_directions.shape[1] == x.shape[1]

	predictor = PCAPredictor()
	learner_func = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', random_state=0)
	results = predictor.fit(features_df, target_column, learner_func)
	feature_directions = results['Feature Directions']
	assert feature_directions.shape[1] == x.shape[1]


def test_orthonormality():
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])
	y = features_df[target_column].values
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x = features_df[x_columns].values

	# Principal features construction
	feature_directions = PCA().fit(x, y)
	n_directions = feature_directions.shape[0]
	for i in range(n_directions):
		assert np.allclose(np.dot(feature_directions[i, :], feature_directions[i, :]), 1.)
		for j in range(n_directions):
			if j != i:
				assert np.abs(np.dot(feature_directions[i, :], feature_directions[j, :])) < 1e-7

	predictor = PCAPredictor()
	learner_func = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', random_state=0)
	results = predictor.fit(features_df, target_column, learner_func)
	feature_directions = results['Feature Directions']
	n_directions = feature_directions.shape[0]
	for i in range(n_directions):
		assert np.allclose(np.dot(feature_directions[i, :], feature_directions[i, :]), 1.)





def test_pca_feature_selection():
	# Regression
	xgboost_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
		problem_type='regression', feature_selection_method='pfs')
	assert results['Feature Directions'].shape[1] == features_df.shape[1]-1
	predictor = results['predictor']
	predictions = predictor.predict(features_df)
	assert len(predictions.columns) == 1
	assert target_column in predictions.columns
	assert set(features_df.index).difference(set(predictions.index)) == set()
	assert set(predictions.index).difference(set(features_df.index)) == set()


def test_save_pca():
	# Regression
	xgboost_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	path = 'Abalone'
	results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
		problem_type='regression', feature_selection_method='pca', \
		path=path)
	loaded_predictor = PCAPredictor().load(path, xgboost_regressor_cls)
	feature_directions = loaded_predictor.feature_directions
	assert feature_directions.shape[1] == features_df.shape[1]-1
	predictions = loaded_predictor.predict(features_df)
	assert len(predictions.columns) == 1
	assert target_column in predictions.columns
	assert set(features_df.index).difference(set(predictions.index)) == set()
	assert set(predictions.index).difference(set(features_df.index)) == set()



