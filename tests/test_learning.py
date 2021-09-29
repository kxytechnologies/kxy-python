from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote
from kxy.learning import get_xgboost_learner


def test_lean_boosted_xgboost_regressor():
	# Regression
	xgboost_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
		problem_type='regression')
	assert results['Testing R-Squared'] == '0.440'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight']


def test_lean_boosted_xgboost_classifier():
	# Binary classification
	xgboost_classifier_cls = get_xgboost_learner('xgboost.XGBClassifier', use_label_encoder=False, 
		eval_metric='logloss', learning_rate=0.1, max_depth=10)
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, xgboost_classifier_cls, \
		problem_type='classification')

	assert results['Testing Accuracy'] == '0.854'
	assert results['Selected Variables'] == ['Variance', '|Skewness - Q25(Skewness)|']


def test_single_learner():
	# Regression
	xgboost_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
		problem_type='regression', start_n_features=2, min_n_features=2, max_n_features=2)
	assert results['Testing R-Squared'] == '0.493'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight']
	assert len(features_df.kxy.models) == 1



def test_n_down_perf_before_stop():
	# Regression
	xgboost_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
		problem_type='regression', n_down_perf_before_stop=3)
	assert results['Testing R-Squared'] == '0.440'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight']

