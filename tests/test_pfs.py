import numpy as np

import kxy
from kxy.learning import get_sklearn_learner, get_lightgbm_learner_learning_api, get_xgboost_learner
from kxy.pfs import PFSPredictor, PFS, PCA
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
	feature_directions = PFS().fit(x, y)
	assert feature_directions.shape[1] == x.shape[1]

	predictor = PFSPredictor()
	learner_func = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', random_state=0)
	results = predictor.fit(features_df, target_column, learner_func)
	feature_directions = results['Feature Directions']
	assert feature_directions.shape[1] == x.shape[1]


def test_norm():
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])
	y = features_df[target_column].values
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x = features_df[x_columns].values

	# Principal features construction
	feature_directions = PFS().fit(x, y)
	n_directions = feature_directions.shape[0]
	for i in range(n_directions):
		assert np.allclose(np.dot(feature_directions[i, :], feature_directions[i, :]), 1.)

	predictor = PFSPredictor()
	learner_func = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', random_state=0)
	results = predictor.fit(features_df, target_column, learner_func)
	feature_directions = results['Feature Directions']
	n_directions = feature_directions.shape[0]
	for i in range(n_directions):
		assert np.allclose(np.dot(feature_directions[i, :], feature_directions[i, :]), 1.)


def test_pfs_feature_selection():
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


def test_save_pfs():
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
		problem_type='regression', feature_selection_method='pfs', \
		path=path)
	loaded_predictor = PFSPredictor().load(path, xgboost_regressor_cls)
	feature_directions = loaded_predictor.feature_directions
	assert feature_directions.shape[1] == features_df.shape[1]-1
	predictions = loaded_predictor.predict(features_df)
	assert len(predictions.columns) == 1
	assert target_column in predictions.columns
	assert set(features_df.index).difference(set(predictions.index)) == set()
	assert set(predictions.index).difference(set(features_df.index)) == set()


def test_pfs_accuracy():
	# Generate the data
	d = 100
	w = np.ones(d)/d
	x = np.random.randn(10000, d)
	xTw = np.dot(x, w)
	y = xTw + 2.*xTw**2 + 0.5*xTw**3

	# Run PFS
	selector = PFS()
	selector.fit(x, y)

	# Learned principal directions
	F = selector.feature_directions

	# Learned principal features
	z = np.dot(x, F.T)

	# Accuracy
	true_f_1 = w/np.linalg.norm(w)
	learned_f_1 = F[0, :]
	e = np.linalg.norm(true_f_1-learned_f_1)

	assert e <= 0.10



