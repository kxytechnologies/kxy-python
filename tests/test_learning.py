from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote, BankMarketing
from kxy.learning import get_xgboost_learner, get_tensorflow_dense_learner, get_pytorch_dense_learner, \
	get_lightgbm_learner_sklearn_api, get_lightgbm_learner_learning_api, get_sklearn_learner


def test_lean_boosted_sklearn_regressor():
	# Regression
	sklearn_regressor_cls = get_sklearn_learner('sklearn.neighbors.KNeighborsRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, sklearn_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True)
	assert results['Testing R-Squared'] == '0.513'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', \
		'Shell weight.ABS(* - Q25(*))', 'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', \
		'Height', 'Length', 'Diameter', 'Sex_I', 'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', \
		'Viscera weight.ABS(* - Q75(*))', 'Viscera weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F']


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
		problem_type='regression', additive_learning=True, return_scores=True)
	assert results['Testing R-Squared'] == '0.496'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', 'Shell weight.ABS(* - Q25(*))', \
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Height', 'Length', 'Diameter', 'Sex_I', \
		'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))', \
		'Viscera weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F']


def test_lean_boosted_lightgbm_regressor():
	# Regression
	lightgbm_regressor_cls = get_lightgbm_learner_sklearn_api('lightgbm.LGBMRegressor')
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, lightgbm_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True)
	assert results['Testing R-Squared'] == '0.548'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', 'Shell weight.ABS(* - Q25(*))', \
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Height', 'Length', 'Diameter', 'Sex_I', \
		'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))', \
		'Viscera weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F']


def test_lean_boosted_lightgbm_learning_api_regressor():
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
		problem_type='regression', additive_learning=True, return_scores=True)
	assert results['Testing R-Squared'] == '0.554'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', 'Shell weight.ABS(* - Q25(*))',\
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Height', 'Length', 'Diameter', 'Sex_I',\
		'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))',\
		'Viscera weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F']


def test_lean_boosted_tensorflow_regressor():
	import tensorflow as tf
	tf.random.set_seed(0)
	# Regression
	layers = [(10, 'relu'), (5, 'relu'), (1, 'linear')]
	loss = 'mean_absolute_error'
	optimizer = 'adam'
	tf_regressor_cls = get_tensorflow_dense_learner('KerasRegressor', layers, loss, optimizer=optimizer, \
		epochs=10, batch_size=100)
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column], fill_na=True)

	# Model building
	results = features_df.kxy.fit(target_column, tf_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True)
	assert results['Testing R-Squared'] == '0.351'
	print(results['Selected Variables'])
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', 'Shell weight.ABS(* - Q25(*))',\
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Height', 'Length', 'Diameter', 'Sex_I',\
		'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))',\
		'Viscera weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F', 'Shucked weight.ABS(* - Q75(*))',\
		'Shucked weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q75(*))']


def test_lean_boosted_pytorch_regressor():
	import torch
	torch.manual_seed(0)
	# Regression
	layers = [(10, 'relu'), (5, 'relu'), (1, None)]
	pt_regressor_cls = get_pytorch_dense_learner('skorch.NeuralNetRegressor', layers, max_epochs=10, batch_size=100)
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column], fill_na=True)

	# Model building
	results = features_df.kxy.fit(target_column, pt_regressor_cls, \
		problem_type='regression', additive_learning=True, return_scores=True)
	assert results['Testing R-Squared'] == '0.562'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', 'Shell weight.ABS(* - Q25(*))',\
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Height', 'Length', 'Diameter', 'Sex_I',\
		'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))', 'Viscera weight.ABS(* - Q25(*))',\
		'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F', 'Shucked weight.ABS(* - Q75(*))', 'Shucked weight.ABS(* - Q25(*))',\
		'Diameter.ABS(* - Q75(*))', 'Length.ABS(* - Q75(*))', 'Length.ABS(* - Q25(*))', 'Height.ABS(* - MEDIAN(*))',\
		'Height.ABS(* - MEAN(*))', 'Length.ABS(* - MEDIAN(*))', 'Whole weight.ABS(* - MEDIAN(*))', 'Shell weight.ABS(* - MEDIAN(*))',\
		'Length.ABS(* - MEAN(*))', 'Shucked weight.ABS(* - MEAN(*))', 'Shell weight.ABS(* - MEAN(*))', 'Whole weight.ABS(* - Q25(*))',\
		'Whole weight.ABS(* - Q75(*))', 'Shell weight.ABS(* - Q75(*))', 'Viscera weight', 'Height.ABS(* - Q75(*))', 'Height.ABS(* - Q25(*))',\
		'Whole weight.ABS(* - MEAN(*))', 'Diameter.ABS(* - MEAN(*))']


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
		problem_type='classification', additive_learning=True, return_scores=True)

	assert results['Testing Accuracy'] == '0.974'
	assert results['Selected Variables'] == ['Variance', 'Skewness.ABS(* - Q25(*))', 'Kurtosis']


def test_lean_boosted_lightgbm_classifier():
	# Classification
	lightgbm_classifier_cls = get_lightgbm_learner_sklearn_api('lightgbm.LGBMClassifier')
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Model building
	results = features_df.kxy.fit(target_column, lightgbm_classifier_cls, \
		problem_type='classification', additive_learning=True, return_scores=True)

	assert results['Testing Accuracy'] == '0.989'
	assert results['Selected Variables'] == ['Variance', 'Skewness.ABS(* - Q25(*))', 'Kurtosis']


def test_lean_boosted_lightgbm_learning_api_classifier():
	# Classification
	params = params = {
		'objective': 'binary',
		'metric': ['auc', 'binary_logloss'],
		'boosting_type': 'gbdt',
	}
	lightgbm_classifier_cls = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
		early_stopping_rounds=5, verbose_eval=50, split_random_seed=42)
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])
	features_df[target_column] = features_df[target_column].astype(int)

	# Model building
	results = features_df.kxy.fit(target_column, lightgbm_classifier_cls, \
		problem_type='classification', additive_learning=True, return_scores=True)

	assert results['Testing Accuracy'] == '0.938'
	assert results['Selected Variables'] == ['Variance', 'Skewness.ABS(* - Q25(*))', 'Kurtosis']



def test_lean_boosted_tensorflow_classifier():
	import tensorflow as tf
	tf.random.set_seed(0)
	# Regression
	layers = [(10, 'relu'), (5, 'relu'), (1, 'sigmoid')]
	loss = 'binary_crossentropy'
	optimizer = 'adam'
	tf_classifier_cls = get_tensorflow_dense_learner('KerasClassifier', layers, loss, optimizer=optimizer, \
		epochs=100, batch_size=100)
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column], fill_na=True)
	# features_df.drop('y_no', axis=1, inplace=True)
	# target_column = 'y_yes'

	# Model building
	results = features_df.kxy.fit(target_column, tf_classifier_cls, problem_type='classification', \
		additive_learning=True, return_scores=True)
	assert results['Testing Accuracy'] == '0.985'
	assert results['Selected Variables'] == ['Variance', 'Skewness.ABS(* - Q25(*))', 'Kurtosis']


def test_lean_boosted_pytorch_classifier():
	from torch import nn
	import torch
	torch.manual_seed(0)
	# Regression
	layers = [(10, 'relu'), (5, 'relu'), (1, 'sigmoid')]
	pt_classifier_cls = get_pytorch_dense_learner('skorch.NeuralNetClassifier', layers, \
		max_epochs=100, batch_size=100, criterion=nn.BCELoss)
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column], fill_na=True)

	# Model building
	results = features_df.kxy.fit(target_column, pt_classifier_cls, problem_type='classification', \
		additive_learning=True, return_scores=True)
	assert results['Testing Accuracy'] == '0.573'
	print(results['Selected Variables'])
	assert results['Selected Variables'] == []


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
		problem_type='regression', start_n_features=2, min_n_features=2, max_n_features=2, \
		additive_learning=True, return_scores=True)
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
		problem_type='regression', n_down_perf_before_stop=3, \
		additive_learning=True, return_scores=True)
	assert results['Testing R-Squared'] == '0.496'
	print(results['Selected Variables'])
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', 'Shell weight.ABS(* - Q25(*))',\
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Height', 'Length', 'Diameter',\
		'Sex_I', 'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))',\
		'Viscera weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F']


def test_non_additive_lean_boosted_regressor():
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
		problem_type='regression', additive_learning=False, return_scores=True)
	assert results['Testing R-Squared'] == '0.554'
	assert results['Selected Variables'] == ['Shell weight', 'Shucked weight', 'Whole weight', 'Shell weight.ABS(* - Q25(*))',\
		'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Height', 'Length', 'Diameter', 'Sex_I',\
		'Shucked weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))',\
		'Viscera weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_M', 'Sex_F']


def test_non_additive_lean_boosted_classifier():
	# Classification
	params = {
		'objective': 'binary',
		'metric': ['auc', 'binary_logloss'],
		'boosting_type': 'gbdt',
	}
	lightgbm_classifier_cls = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
		early_stopping_rounds=5, verbose_eval=50, split_random_seed=42)
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])
	features_df[target_column] = features_df[target_column].astype(int)

	# Model building
	results = features_df.kxy.fit(target_column, lightgbm_classifier_cls, \
		problem_type='classification', additive_learning=False, return_scores=True)

	assert results['Testing Accuracy'] == '0.964'
	assert results['Selected Variables'] == ['Variance', 'Skewness.ABS(* - Q25(*))', 'Kurtosis', 'Skewness', 'Entropy']



