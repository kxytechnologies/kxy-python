#!/usr/bin/env python
# -*- coding: utf-8 -*-
from time import time
import json
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, roc_auc_score

from kxy.learning import get_xgboost_learner, get_lightgbm_learner_learning_api, get_sklearn_learner
from kxy.misc.predictors import BorutaPredictor, RFEPredictor
from kxy.learning.shrunk_learner import ShrunkLearner as LeanMLPredictor

from kxy_datasets.regressions import all_regression_datasets
from kxy_datasets.classifications import all_classification_datasets, \
	LetterRecognition, MagicGamma, SensorLessDrive, Shuttle, SkinSegmentation, \
	HeartAttack, HeartDisease, Titanic, WaterQuality

def regression_benchmark(learner_func, model_name):
	try:
		with open('./cache/regression_benchmark_perfs.json', 'r') as f:
			perfs = json.load(f)

		with open('./cache/regression_benchmark_durations.json', 'r') as f:
			durations = json.load(f)
	except:
		perfs = {}
		durations = {}

	# LeanML vs Boruta vs RFE
	for dataset_cls in all_regression_datasets:
		dataset = dataset_cls()
		target_column = dataset.y_column
		dataset_name = dataset.name
		df = dataset.df
		perfs[dataset_name]= {}
		durations[dataset_name]= {}

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', \
			exclude=[target_column])
		train_features_df, test_features_df = train_test_split(features_df, test_size=0.2, \
			random_state=0)
		test_labels_df = test_features_df[[target_column]]
		test_features_df.drop(target_column, axis=1, inplace=True)
		path = './cache/%s-%s-regression-benchmark.sav' % (model_name, dataset_name)

		logging.warning('%s %d Features' % (dataset_name, train_features_df.shape[1]-1))
		for feature_selection_method in ['leanml', 'rfe', 'boruta']:
			duration = time()
			clz = LeanMLPredictor if feature_selection_method == 'leanml' else RFEPredictor \
				if feature_selection_method == 'rfe' else BorutaPredictor
			logging.warning(feature_selection_method)

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

			# Evaluation
			test_predictions_df = predictor.predict(test_features_df)
			duration = time()-duration
			perf = r2_score(\
				test_labels_df[target_column].values, \
				test_predictions_df[target_column].values)
			perfs[dataset_name][feature_selection_method]=perf
			durations[dataset_name][feature_selection_method]=duration

			with open('./cache/regression_benchmark_perfs.json', 'w') as f:
				json.dump(perfs, f)

			with open('./cache/regression_benchmark_durations.json', 'w') as f:
				json.dump(durations, f)

			logging.warning('%s, %s, %s --- R-Squared: %.2f, Duration: %.2fs' % (\
				model_name, dataset_name, feature_selection_method, perf, duration))
			



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
	try:
		with open('./cache/classification_benchmark_perfs.json', 'r') as f:
			perfs = json.load(f)

		with open('./cache/classification_benchmark_durations.json', 'r') as f:
			durations = json.load(f)
	except:
		perfs = {}
		durations = {}

	# LeanML vs Boruta vs RFE

	for dataset_cls in [HeartAttack, HeartDisease, Titanic, WaterQuality]: #all_classification_datasets:
		print(dataset_cls.__name__)
		dataset = dataset_cls()
		target_column = dataset.y_column
		dataset_name = dataset.name
		perfs[dataset_name]= {}
		durations[dataset_name]= {}
		df = dataset.df

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', \
			exclude=[target_column])

		if target_column in features_df:
			target_df = pd.get_dummies(df[target_column], prefix=str(target_column))
			features_df = features_df.drop(target_column, axis=1)
			features_df = pd.concat([features_df, target_df], axis=1)

		train_features_df, test_features_df = train_test_split(features_df, test_size=0.2, \
			random_state=0)
		target_columns = [_ for _ in features_df.columns if str(_).startswith(str(target_column))]
		target_column = target_columns[0]
		test_labels_df = test_features_df.loc[:, [target_column]]
		train_labels_df = train_features_df.loc[:, [target_column]]

		for col in target_columns:
			if col != target_column:
				test_features_df = test_features_df.drop(col, axis=1)
				train_features_df = train_features_df.drop(col, axis=1)
			else:
				test_features_df = test_features_df.drop(col, axis=1)

		path = './cache/%s-%s-classification-benchmark.sav' % (model_name, dataset_name)
		logging.warning('%s %d Features, Target: %s' % (dataset_name, train_features_df.shape[1]-1, target_column))
		for feature_selection_method in ['leanml']:
			duration = time()
			clz = LeanMLPredictor if feature_selection_method == 'leanml' else RFEPredictor \
				if feature_selection_method == 'rfe' else BorutaPredictor
			logging.warning(feature_selection_method)
			# Model building
			if feature_selection_method == 'leanml':
				try:
					predictor = clz.load(path, learner_func)
				except:
					results = train_features_df.kxy.fit(target_column, learner_func, \
						problem_type='classification', feature_selection_method=feature_selection_method)
					predictor = results['predictor']
					predictor.save(path)
					duration = time()-duration
					durations[dataset_name][feature_selection_method]=duration
					with open('./cache/classification_benchmark_durations.json', 'w') as f:
						json.dump(durations, f)
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
					duration = time()-duration
					durations[dataset_name][feature_selection_method]=duration
					with open('./cache/classification_benchmark_durations.json', 'w') as f:
						json.dump(durations, f)

			# Evaluation
			test_predictions_df = predictor.predict(test_features_df)
			perf = roc_auc_score(\
				test_labels_df[target_column].values, \
				test_predictions_df[target_column].values, \
				multi_class='ovr'
			)
			perfs[dataset_name][feature_selection_method]=perf

			with open('./cache/classification_benchmark_perfs.json', 'w') as f:
				json.dump(perfs, f)

			duration = durations[dataset_name].get(feature_selection_method, 0.)
			logging.warning('%s, %s, %s --- AUC: %.2f, Duration: %.2fs' % (\
				model_name, dataset_name, feature_selection_method, perf, duration))


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
	# lightgbm_regression_benchmark()
	lightgbm_classification_benchmark()



