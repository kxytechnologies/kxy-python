import numpy as np
np.random.seed(0)

import kxy
from kxy.misc import Boruta
from kxy.learning import get_sklearn_learner, get_lightgbm_learner_learning_api, get_xgboost_learner

from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote, BankMarketing


def test_lasso_cv():
	regressor_cls = get_sklearn_learner('sklearn.linear_model.LassoCV')
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)

	fs = Boruta(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	# Assertions
	assert len(fs.selected_variables) == 15
	assert fs.selected_variables == ['Shucked weight.ABS(* - Q75(*))', 'Shucked weight', 'Shell weight', \
		'Length.ABS(* - Q25(*))', 'Height.ABS(* - Q75(*))', 'Height', 'Whole weight', 'Viscera weight.ABS(* - Q75(*))', \
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight', 'Diameter', 'Whole weight.ABS(* - Q25(*))', \
		'Shucked weight.ABS(* - Q25(*))', 'Whole weight.ABS(* - Q75(*))', 'Diameter.ABS(* - Q75(*))']


def test_lasso():
	regressor_cls = get_sklearn_learner('sklearn.linear_model.Lasso')
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)

	fs = Boruta(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	# Assertions
	assert len(fs.selected_variables) == 0
	assert fs.selected_variables == []



def test_linear_regression():
	regressor_cls = get_sklearn_learner('sklearn.linear_model.LinearRegression', random_state=0)
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)
	
	fs = Boruta(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	# Assertions
	assert len(fs.selected_variables) == 3
	assert fs.selected_variables == ['Sex_M', 'Sex_I', 'Sex_F']



def test_lightgbm_regression():
	lgbm_params = {
		'objective': 'rmse',  
		'boosting_type': 'gbdt',
		'n_jobs': -1,
		'learning_rate': 0.1,
		'verbose': -1,
	}
	regressor_cls = get_lightgbm_learner_learning_api(lgbm_params, num_boost_round=2000, \
		early_stopping_rounds=5, split_random_seed=0)
	
	fs = Boruta(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	assert len(fs.selected_variables) == 13
	assert fs.selected_variables == ['Shucked weight.ABS(* - Q75(*))', 'Shucked weight.ABS(* - MEDIAN(*))', \
		'Shucked weight.ABS(* - MEAN(*))', 'Shucked weight', 'Shell weight.ABS(* - MEAN(*))', 'Shell weight', \
		'Shucked weight.ABS(* - Q25(*))', 'Sex_I', 'Diameter', 'Whole weight', 'Shell weight.ABS(* - Q75(*))', \
		'Whole weight.ABS(* - MEAN(*))', 'Height']



def test_random_forest_regression():
	regressor_cls = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', random_state=0)
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)
	
	fs = Boruta(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	# Assertions
	assert len(fs.selected_variables) == 11
	assert fs.selected_variables == ['Shucked weight.ABS(* - Q75(*))', 'Shucked weight.ABS(* - Q25(*))', 'Shucked weight.ABS(* - MEDIAN(*))', \
		'Shucked weight.ABS(* - MEAN(*))', 'Shucked weight', 'Shell weight.ABS(* - Q75(*))', 'Shell weight.ABS(* - Q25(*))', \
		'Shell weight.ABS(* - MEDIAN(*))', 'Shell weight.ABS(* - MEAN(*))', 'Shell weight', 'Sex_I']



def test_xgboost_regression():
	regressor_cls = get_xgboost_learner('xgboost.XGBRegressor', random_state=0)
	
	fs = Boruta(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	# Assertions
	assert len(fs.selected_variables) == 5
	assert fs.selected_variables == ['Shucked weight', 'Shell weight', 'Sex_I', 'Shucked weight.ABS(* - Q25(*))', 'Whole weight']



def test_xgboost_classifier():
	# Binary classification
	classifier_cls = get_xgboost_learner('xgboost.XGBClassifier', use_label_encoder=False, 
		eval_metric='logloss', learning_rate=0.1, max_depth=10)
	fs = Boruta(classifier_cls)

	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	# Assertions
	assert len(fs.selected_variables) == 8
	assert fs.selected_variables == ['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Variance.ABS(* - MEAN(*))', \
		'Skewness.ABS(* - MEDIAN(*))', 'Skewness.ABS(* - MEAN(*))', 'Kurtosis.ABS(* - MEDIAN(*))']



def test_lightgbm_classifier():
	# Classification
	params = params = {
		'objective': 'binary',
		'metric': ['auc', 'binary_logloss'],
		'boosting_type': 'gbdt',
	}
	classifier_cls = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
		early_stopping_rounds=5, verbose_eval=50, split_random_seed=42)
	fs = Boruta(classifier_cls)
	
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	m = fs.fit(x_df, y_df)

	# Assertions
	assert len(fs.selected_variables) == 8
	print(fs.selected_variables)
	assert fs.selected_variables == ['Variance.ABS(* - MEAN(*))', 'Variance', 'Skewness.ABS(* - MEAN(*))', 'Skewness', \
		'Kurtosis', 'Entropy', 'Variance.ABS(* - Q25(*))', 'Kurtosis.ABS(* - Q75(*))']




