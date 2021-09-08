from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote
from kxy.learning import get_sklearn_learner


def test_lean_boosted_xgboost_regressor():
	# Regression
	xgboost_regressor_cls = get_sklearn_learner('xgboost.XGBRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
		problem_type='regression')
	assert results['Testing R-Squared'] == '0.440'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight']


def test_lean_boosted_xgboost_classifier():
	# Binary classification
	xgboost_classifier_cls = get_sklearn_learner('xgboost.XGBClassifier', use_label_encoder=False, 
		eval_metric='logloss', learning_rate=0.1, max_depth=10)
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, xgboost_classifier_cls, \
		problem_type='classification')

	assert results['Testing Accuracy'] == '0.861'
	assert results['Selected Variables'] == ['Variance', '|Skewness - Q25(Skewness)|']
