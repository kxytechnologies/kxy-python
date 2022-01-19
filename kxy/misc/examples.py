#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split

from kxy.learning import get_xgboost_learner, get_lightgbm_learner_learning_api, get_sklearn_learner
from kxy.misc.predictors import BorutaPredictor, RFEPredictor
from kxy.learning.shrunk_learner import ShrunkLearner as LeanMLPredictor

from kxy_datasets.regressions import all_regression_datasets
from kxy_datasets.classifications import all_classification_datasets

def regression_benchmark(learner_func, model_name):
	# LeanML vs Boruta vs RFE
	for dataset_cls in all_regression_datasets:
		dataset = dataset_cls()
		target_column = dataset.y_column
		dataset_name = dataset.name
		df = dataset.df
		print(dataset_name)

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', \
			exclude=[target_column])
		train_features_df, test_features_df = train_test_split(features_df, test_size=0.2, \
			random_state=0)
		path = './cache/%s-%s-regression-benchmark.sav' % (model_name, dataset_name)

		for feature_selection_method in ['leanml', 'rfe', 'boruta']:
			clz = LeanMLPredictor if feature_selection_method == 'leanml' else RFEPredictor \
				if feature_selection_method == 'rfe' else BorutaPredictor

			print(feature_selection_method)
			# Model building
			if feature_selection_method == 'leanml':
				try:
					predictor = clz.load(path, learner_func)
				except:
					results = train_features_df.kxy.fit(target_column, learner_func, \
						problem_type='regression', feature_selection_method=feature_selection_method)
					predictor = results['predictor']
					predictor.save(path)
				rfe_n_features = len(predictor.selected_variables)
			else:
				try:
					predictor = clz.load(path, learner_func)
				except:
					results = train_features_df.kxy.fit(target_column, learner_func, \
						problem_type='regression', feature_selection_method=feature_selection_method, \
						rfe_n_features=rfe_n_features)
					predictor = results['predictor']
					predictor.save(path)

			# TODO: predict


def lightgbm_regression_benchmark():
	# LeanML vs Boruta vs RFE
	params = {
		'objective': 'rmse',  
		'boosting_type': 'gbdt',
		'num_leaves': 100,
		'n_jobs': -1,
		'learning_rate': 0.1,
		'verbose': -1,
	}
	lightgbm_regressor_cls = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
		early_stopping_rounds=50, verbose_eval=50)
	regression_benchmark(lightgbm_regressor_cls, 'lightgbm')


def xgboost_regression_benchmark():
	# LeanML vs Boruta vs RFE
	xgboost_regressor_cls = get_xgboost_learner('xgboost.XGBRegressor')
	regression_benchmark(xgboost_regressor_cls, 'xgboost')


def random_forest_regression_benchmark():
	# LeanML vs Boruta vs RFE
	rf_regressor_cls = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', \
                        min_samples_split=200, max_samples=20000, n_estimators=100)
	regression_benchmark(rf_regressor_cls, 'rf')



def classification_benchmark(learner_func, model_name):
	# LeanML vs Boruta vs RFE
	for dataset_cls in all_classification_datasets:
		dataset = dataset_cls()
		target_column = dataset.y_column
		dataset_name = dataset.name
		df = dataset.df
		print(dataset_name)

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', \
			exclude=[target_column])
		train_features_df, test_features_df = train_test_split(features_df, test_size=0.2, \
			random_state=0)
		target_columns = [_ for _ in features_df.columns if target_column in _]
		feature_columns = [_ for _ in features_df.columns if not target_column in _]
		target_column = target_columns[0]
		columns = [target_column] + feature_columns
		features_df = features_df[columns]
		path = './cache/%s-%s-classification-benchmark.sav' % (model_name, dataset_name)

		for feature_selection_method in ['leanml', 'rfe', 'boruta']:
			clz = LeanMLPredictor if feature_selection_method == 'leanml' else RFEPredictor \
				if feature_selection_method == 'rfe' else BorutaPredictor
			print(feature_selection_method)
			# Model building
			if feature_selection_method == 'leanml':
				try:
					predictor = clz.load(path, learner_func)
				except:
					results = train_features_df.kxy.fit(target_column, learner_func, \
						problem_type='classification', feature_selection_method=feature_selection_method)
					predictor = results['predictor']
					predictor.save(path)
				rfe_n_features = len(predictor.selected_variables)
			else:
				try:
					predictor = clz.load(path, learner_func)
				except:
					results = train_features_df.kxy.fit(target_column, learner_func, \
						problem_type='classification', feature_selection_method=feature_selection_method, \
						rfe_n_features=rfe_n_features)
					predictor = results['predictor']
					predictor.save(path)

			# TODO: predict


def lightgbm_classification_benchmark():
	# LeanML vs Boruta vs RFE
	params = {
		'objective': 'binary',
		'metric': ['auc', 'binary_logloss'],
	}

	lightgbm_classifier_cls = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
		early_stopping_rounds=50, verbose_eval=50)
	classification_benchmark(lightgbm_classifier_cls, 'lightgbm')


def xgboost_classification_benchmark():
	# LeanML vs Boruta vs RFE
	xgboost_classifier_cls = get_xgboost_learner('xgboost.XGBClassifier')
	classification_benchmark(xgboost_classifier_cls, 'xgboost')


def random_forest_classification_benchmark():
	# LeanML vs Boruta vs RFE
	rf_classifier_cls = get_sklearn_learner('sklearn.ensemble.RandomForestClassifier', \
                        min_samples_split=200, max_samples=20000, n_estimators=100)
	classification_benchmark(rf_classifier_cls, 'rf')

if __name__ == '__main__':
	lightgbm_regression_benchmark()



