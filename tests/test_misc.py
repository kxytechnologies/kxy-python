from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote, BankMarketing
from kxy.learning import get_xgboost_learner, get_tensorflow_dense_learner, get_pytorch_dense_learner, \
	get_lightgbm_learner_sklearn_api, get_lightgbm_learner_learning_api, get_sklearn_learner



def test_boruta():
	# Regression
	sklearn_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, sklearn_regressor_cls, \
		problem_type='regression', feature_selection_method='boruta', boruta_n_evaluations=100)
	assert results['Selected Variables'] == ['Shucked weight', 'Shell weight', 'Sex_I', \
		'Shucked weight.ABS(* - Q25(*))', 'Whole weight']


def test_rfe():
	# Regression
	sklearn_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, sklearn_regressor_cls, \
		problem_type='regression', feature_selection_method='rfe', rfe_n_features=10)
	assert results['Selected Variables'] == ['Shell weight', 'Sex_I', 'Shucked weight.ABS(* - Q25(*))', \
		'Whole weight.ABS(* - Q25(*))', 'Shucked weight.ABS(* - MEDIAN(*))', 'Shucked weight', \
		'Shucked weight.ABS(* - Q75(*))', 'Shucked weight.ABS(* - MEAN(*))', 'Diameter.ABS(* - Q25(*))', \
		'Diameter.ABS(* - Q75(*))']