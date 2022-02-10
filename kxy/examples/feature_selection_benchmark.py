#!/usr/bin/env python
# -*- coding: utf-8 -*-
from time import time
import json
import logging
import pandas as pd
import numpy as np
from time import sleep

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, roc_auc_score

from kxy.learning import get_xgboost_learner, get_lightgbm_learner_learning_api, get_sklearn_learner
from kxy.misc.predictors import BorutaPredictor, RFEPredictor, NaivePredictor
from kxy.learning.shrunk_learner import ShrunkLearner as LeanMLPredictor

from kxy_datasets.regressions import all_regression_datasets
from kxy_datasets.classifications import all_classification_datasets

def regression_benchmark(learner_func, model_name):
	try:
		with open('./cache/%s_regression_benchmark_perfs.json' % model_name, 'r') as f:
			perfs = json.load(f)

		with open('./cache/%s_regression_benchmark_durations.json' % model_name, 'r') as f:
			durations = json.load(f)
	except:
		perfs = {}
		durations = {}

	try:
		with open('./cache/%s_regression_benchmark_n_features.json' % model_name, 'r') as f:
			n_features = json.load(f)
	except:
		n_features = {}

	# LeanML vs Boruta vs RFE
	for dataset_cls in all_regression_datasets:
		print(dataset_cls.__name__)
		dataset = dataset_cls()
		target_column = dataset.y_column
		dataset_name = dataset.name
		df = dataset.df
		perfs[dataset_name] = perfs.get(dataset_name, {})
		durations[dataset_name]= durations.get(dataset_name, {})
		n_features[dataset_name] = n_features.get(dataset_name, {})

		# Features generation
		features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', \
			exclude=[target_column])
		train_features_df, test_features_df = train_test_split(features_df, test_size=0.2, \
			random_state=0)
		test_labels_df = test_features_df.loc[:, [target_column]]
		test_features_df = test_features_df.drop(target_column, axis=1)
		path = './cache/%s-%s-regression-benchmark.sav' % (model_name, dataset_name)

		logging.warning('%s %d Features, Target: %s' % (dataset_name, train_features_df.shape[1]-1, target_column))
		for feature_selection_method in ['leanml', 'none', 'rfe', 'boruta']:
			duration = time()
			clz = LeanMLPredictor if feature_selection_method == 'leanml' else RFEPredictor \
				if feature_selection_method == 'rfe' else BorutaPredictor if feature_selection_method == 'boruta' \
				else NaivePredictor
			logging.warning(feature_selection_method)

			# Model building
			if feature_selection_method == 'leanml':
				try:
					predictor = clz.load(path, learner_func)
				except:
					try:
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='regression', feature_selection_method=feature_selection_method)
					except:
						train_features_df = train_features_df.dropna(axis=0)
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='regression', feature_selection_method=feature_selection_method)
					predictor = results['predictor']
					predictor.save(path)
					duration = time()-duration
					durations[dataset_name][feature_selection_method]=duration
					with open('./cache/%s_regression_benchmark_durations.json' % model_name, 'w') as f:
						json.dump(durations, f)
				rfe_n_features = len(predictor.selected_variables)
			else:
				try:
					predictor = clz.load(path, learner_func)
				except:
					try:
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='regression', feature_selection_method=feature_selection_method, \
							rfe_n_features=rfe_n_features, max_duration=3600.0)
					except:
						train_features_df = train_features_df.dropna(axis=0)
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='regression', feature_selection_method=feature_selection_method, \
							rfe_n_features=rfe_n_features, max_duration=3600.0)
					predictor = results['predictor']
					try:
						predictor.save(path)
					except:
						pass
					duration = time()-duration
					durations[dataset_name][feature_selection_method]=duration
					with open('./cache/%s_regression_benchmark_durations.json' % model_name, 'w') as f:
						json.dump(durations, f)

			n_features[dataset_name][feature_selection_method] = len(predictor.selected_variables)

			# Evaluation
			try:
				try:
					test_predictions_df = predictor.predict(test_features_df)
				except:
					nan_features = test_features_df.isna().any(axis=1)
					test_features_df = test_features_df.loc[np.logical_not(nan_features), :]
					test_predictions_df = predictor.predict(test_features_df)
					test_labels_df = test_labels_df.loc[np.logical_not(nan_features), :]

				perf = r2_score(\
					test_labels_df[target_column].values, \
					test_predictions_df[target_column].values)
			except:
				logging.exception('Somthing bad happened')
				perf = 0.0
			perfs[dataset_name][feature_selection_method]=perf
			
			with open('./cache/%s_regression_benchmark_perfs.json' % model_name, 'w') as f:
				json.dump(perfs, f)

			with open('./cache/%s_regression_benchmark_n_features.json' % model_name, 'w') as f:
				json.dump(n_features, f)

			duration = durations[dataset_name].get(feature_selection_method, 0.)
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
						min_samples_split=0.01, max_samples=0.5, n_estimators=100)
	regression_benchmark(rf_regressor_cls, 'rf')



def classification_benchmark(learner_func, model_name):
	try:
		with open('./cache/%s_classification_benchmark_perfs.json' % model_name, 'r') as f:
			perfs = json.load(f)

		with open('./cache/%s_classification_benchmark_durations.json' % model_name, 'r') as f:
			durations = json.load(f)
	except:
		perfs = {}
		durations = {}

	try:
		with open('./cache/%s_classification_benchmark_n_features.json' % model_name, 'r') as f:
			n_features = json.load(f)
	except:
		n_features = {}

	# LeanML vs Boruta vs RFE
	for dataset_cls in all_classification_datasets:
		print(dataset_cls.__name__)
		dataset = dataset_cls()
		target_column = dataset.y_column
		dataset_name = dataset.name
		perfs[dataset_name] = perfs.get(dataset_name, {})
		durations[dataset_name]= durations.get(dataset_name, {})
		n_features[dataset_name] = n_features.get(dataset_name, {})
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
		for feature_selection_method in ['leanml', 'none', 'rfe', 'boruta']:
			duration = time()
			clz = LeanMLPredictor if feature_selection_method == 'leanml' else RFEPredictor \
				if feature_selection_method == 'rfe' else BorutaPredictor if feature_selection_method == 'boruta' \
				else NaivePredictor
			logging.warning(feature_selection_method)
			# Model building
			if feature_selection_method == 'leanml':
				try:
					predictor = clz.load(path, learner_func)
				except:
					try:
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='classification', feature_selection_method=feature_selection_method)
					except:
						train_features_df = train_features_df.dropna(axis=0)
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='classification', feature_selection_method=feature_selection_method)
					predictor = results['predictor']
					predictor.save(path)
					duration = time()-duration
					durations[dataset_name][feature_selection_method]=duration
					with open('./cache/%s_classification_benchmark_durations.json' % model_name, 'w') as f:
						json.dump(durations, f)
				rfe_n_features = len(predictor.selected_variables)
			else:
				try:
					predictor = clz.load(path, learner_func)
				except:
					try:
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='classification', feature_selection_method=feature_selection_method, \
							rfe_n_features=rfe_n_features, max_duration=3600.)
					except:
						train_features_df = train_features_df.dropna(axis=0)
						results = train_features_df.kxy.fit(target_column, learner_func, \
							problem_type='classification', feature_selection_method=feature_selection_method, \
							rfe_n_features=rfe_n_features, max_duration=3600.)

					predictor = results['predictor']
					try:
						predictor.save(path)
					except:
						pass
					duration = time()-duration
					durations[dataset_name][feature_selection_method]=duration
					with open('./cache/%s_classification_benchmark_durations.json' % model_name, 'w') as f:
						json.dump(durations, f)

			n_features[dataset_name][feature_selection_method] = len(predictor.selected_variables)
			# Evaluation
			try:
				try:
					test_predictions_df = predictor.predict(test_features_df)
				except:
					nan_features = test_features_df.isna().any(axis=1)
					test_features_df = test_features_df.loc[np.logical_not(nan_features), :]
					test_predictions_df = predictor.predict(test_features_df)
					test_labels_df = test_labels_df.loc[np.logical_not(nan_features), :]
				perf = roc_auc_score(\
					test_labels_df[target_column].values, \
					test_predictions_df[target_column].values, \
					multi_class='ovr'
				)
			except:
				logging.exception('Somthing bad happened')
				perf = 0.5
			perfs[dataset_name][feature_selection_method]=perf

			with open('./cache/%s_classification_benchmark_perfs.json' % model_name, 'w') as f:
				json.dump(perfs, f)

			with open('./cache/%s_classification_benchmark_n_features.json' % model_name, 'w') as f:
				json.dump(n_features, f)

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
						min_samples_split=0.01, max_samples=0.5, n_estimators=100)
	classification_benchmark(rf_classifier_cls, 'rf')


def summary(model_name='lightgbm'):
	dataset_names = []
	sources = []
	ds = []
	ns = []
	problem_types = []

	problem_type = None
	for l in [all_classification_datasets, all_regression_datasets]:
		problem_type = 'classification' if problem_type is None else 'regression'
		try:
			with open('./cache/%s_%s_benchmark_n_features.json' % (model_name, problem_type), 'r') as f:
				n_features = json.load(f)
		except:
			logging.exception('Boom')
			n_features = {}

		for dataset_cls in l:
			dataset = dataset_cls()
			dataset_name = dataset_cls.__name__
			print(dataset_name)
			dataset_names += [dataset_name]
			ns += [dataset.df.shape[0]]
			ds += [n_features[dataset.name]['none']] 
			sources += ['UCI' if 'UCI' in dataset.name else 'Kaggle']
			problem_types += [problem_type]

	df = pd.DataFrame(data={'Dataset': dataset_names, 'Number of Features': ds, 'Number of Rows': ns, 'Problem Type': problem_types, 'Source': sources})
	df = df.sort_values(by=['Number of Features', 'Number of Rows', 'Problem Type'])
	df.reset_index(drop=True, inplace=True)
	print(df.to_markdown())
	
	return df


def results(pretty_model_name='LightGBM', model_name='lightgbm'):
	print()
	print('============================')
	print(model_name)
	print('============================')
	print()
	leanml_comp = []
	boruta_comp = []
	diff_comp = []
	ns_leanml = []
	ns_boruta = []
	ns_none = []
	perfs_leanml = []
	perfs_boruta = []
	perfs_rfe = []
	perfs_none = []
	durations_leanml = []
	durations_boruta = []
	durations_rfe = []
	durations_none = []

	for problem_type in ['classification', 'regression']:
		try:
			with open('./cache/%s_%s_benchmark_n_features.json' % (model_name, problem_type), 'r') as f:
				n_features = json.load(f)

			with open('./cache/%s_%s_benchmark_durations.json' % (model_name, problem_type), 'r') as f:
				durations = json.load(f)

			with open('./cache/%s_%s_benchmark_perfs.json' % (model_name, problem_type), 'r') as f:
				perfs = json.load(f)
		except:
			logging.exception('Boom')
			n_features = {}
			durations = {}
			perfs = {}

		for k, d in n_features.items():
			n_boruta = d['boruta']
			n_leanml = d['leanml']
			n = d['none']
			leanml_comp += [1.-1.*n_leanml/n]
			boruta_comp += [1.-1.*n_boruta/n]
			diff_comp += [1.*(n_boruta-n_leanml)/n]
			ns_leanml += [n_leanml]
			ns_boruta += [n_boruta]
			ns_none += [n]

			perfs_leanml += [perfs[k]['leanml']]
			perfs_boruta += [perfs[k]['boruta']]
			perfs_rfe += [perfs[k]['rfe']]
			perfs_none += [perfs[k]['none']]

			durations_leanml += [durations[k]['leanml']/n]
			durations_boruta += [durations[k]['boruta']/n]
			durations_rfe += [durations[k]['rfe']/n]
			durations_none += [durations[k]['none']/n]

			print()
			print('%s All: %d, LeanML: %d, Boruta: %d' % (k, n, n_leanml, n_boruta))
			print('%s LeanML Compression Rate: %.3f' % (k, 1.-1.*n_leanml/n))
			print('%s Boruta Compression Rate: %.3f' % (k, 1.-1.*n_boruta/n))

	print()
	print('LeanML Compression Rate: %.2f' % np.mean(leanml_comp))
	print('Boruta Compression Rate: %.2f' % np.mean(boruta_comp))
	print('Difference in Compression Rate: %.2f' % np.mean(diff_comp))

	print()
	print('LeanML Duration: %.2f' % np.mean(durations_leanml))
	print('Boruta Duration: %.2f' % np.mean(durations_boruta))
	print('RFE Duration: %.2f' % np.mean(durations_rfe))
	print('No Feature Selection Duration: %.2f' % np.mean(durations_none))

	print()
	print('LeanML Performance: %.2f' % np.mean(perfs_leanml))
	print('Boruta Performance: %.2f' % np.mean(perfs_boruta))
	print('RFE Performance: %.2f' % np.mean(np.maximum(perfs_rfe, 0.0)))
	print('No Feature Selection Perf: %.2f' % np.mean(perfs_none))

	df = pd.DataFrame(data={'LeanML': leanml_comp, 'Boruta': boruta_comp})

	import matplotlib
	from matplotlib import pyplot as plt
	params = {
		'axes.titlesize':'25',
		'axes.labelsize':'22',
	}
	matplotlib.rcParams.update(params)

	plt.figure(figsize=(12,12))
	fig = df.boxplot(fontsize=20, whis=3, figsize=(12, 12))
	plt.ylabel('Compression Rate')
	plt.savefig('./plot_%s_compression_boxplot.png' % model_name)
	plt.close()


	plt.figure(figsize=(12,12))
	fig = df.plot(kind='hist',
			alpha=0.7,
			bins=7,
			rot=45,
			grid=True,
			figsize=(12,12),
			fontsize=20)
	plt.xlabel('Compression Rate')
	plt.ylabel('Number of Datasets')
	plt.savefig('./plot_%s_compression_hist.png' % model_name)
	plt.close()


	plt.figure(figsize=(12,12))
	df = pd.DataFrame(data={'LeanML': ns_leanml, 'Boruta': ns_boruta})
	fig = df.plot(kind='hist',
			alpha=0.7,
			bins=7,
			rot=45,
			grid=True,
			figsize=(12,12),
			fontsize=20)
	plt.xlabel('Number of Selected Features')
	plt.ylabel('Number of Datasets')
	plt.savefig('./plot_%s_size_hist.png' % model_name)
	plt.close()


	plt.figure(figsize=(12,12))
	df = pd.DataFrame(data={'Boruta': [ns_boruta[i]-ns_leanml[i] for i in range(len(ns_leanml))]})
	fig = df.plot(kind='hist',
			alpha=0.7,
			bins=7,
			rot=45,
			grid=True,
			figsize=(12,12),
			fontsize=20)
	plt.xlabel('Number of Excessive Features')
	plt.ylabel('Number of Datasets')
	plt.savefig('./plot_%s_add_size_hist.png' % model_name)
	plt.close()


	plt.figure(figsize=(12,12))
	df = pd.DataFrame(data={'LeanML': perfs_leanml, 'Boruta': perfs_boruta, 'RFE': perfs_rfe, \
		'No Feature Selection': perfs_none})
	df = df.clip(lower=0.0)
	fig = df.boxplot(fontsize=20, whis=3, figsize=(12, 12))
	plt.ylabel('Performance')
	plt.savefig('./plot_%s_perf_boxplot.png' % model_name)
	plt.close()


	plt.figure(figsize=(12,12))
	fig = df.plot(kind='hist',
			alpha=0.7,
			bins=7,
			rot=45,
			grid=True,
			figsize=(12,12),
			fontsize=20)
	plt.xlabel('Performance')
	plt.ylabel('Number of Datasets')
	plt.savefig('./plot_%s_perf_hist.png' % model_name)
	plt.close()


	plt.figure(figsize=(12,12))
	df = pd.DataFrame(data={'LeanML': durations_leanml, \
		'Boruta': durations_boruta, \
		'RFE': durations_rfe
	})
	fig = df.plot(kind='hist',
			alpha=0.7,
			bins=15,
			rot=45,
			grid=True,
			figsize=(12,12),
			fontsize=20)
	plt.xlabel('Training Duration\n(Seconds Per Candidate Feature)')
	plt.ylabel('Number of Datasets')
	plt.savefig('./plot_%s_duration_hist.png' % model_name)
	plt.close()


	plt.figure(figsize=(12,12))
	fig = df.boxplot(fontsize=20, whis=20, figsize=(12, 12))
	plt.ylabel('Training Duration\n(Seconds Per Candidate Feature)')
	plt.savefig('./plot_%s_duration_boxplot.png' % model_name)
	plt.close()


	compression_rates = {
		'Model': '%s' % 'LightGBM' if model_name == 'lightgbm' else 'Random Forest' if model_name == 'rf' else 'XGBoost' if model_name == 'xgboost' else '',
		'No Feature Selection': '0.00 \u00b1 0.00', 
		'LeanML': '%.2f \u00b1 %.2f' % (np.mean(leanml_comp), np.std(leanml_comp)/np.sqrt(len(leanml_comp))),
		'Boruta': '%.2f \u00b1 %.2f' % (np.mean(boruta_comp), np.std(boruta_comp)/np.sqrt(len(boruta_comp))),
		'RFE': '-'
	}
	durations =  {
		'Model': '%s' % 'LightGBM' if model_name == 'lightgbm' else 'Random Forest' if model_name == 'rf' else 'XGBoost' if model_name == 'xgboost' else '',
		'No Feature Selection': '%.2f \u00b1 %.2f' % (np.mean(durations_none), np.std(durations_none)/np.sqrt(len(durations_none))),
		'LeanML': '%.2f \u00b1 %.2f' % (np.mean(durations_leanml), np.std(durations_leanml)/np.sqrt(len(durations_leanml))),
		'Boruta': '%.2f \u00b1 %.2f' % (np.mean(durations_boruta), np.std(durations_boruta)/np.sqrt(len(durations_boruta))),
		'RFE': '%.2f \u00b1 %.2f' % (np.mean(durations_rfe), np.std(durations_rfe)/np.sqrt(len(durations_rfe)))
	}

	perfs_none = np.maximum(perfs_none, 0.0)
	perfs_leanml = np.maximum(perfs_leanml, 0.0)
	perfs_boruta = np.maximum(perfs_boruta, 0.0)
	perfs_rfe = np.maximum(perfs_rfe, 0.0)
	performances = {
		'Model': '%s' % 'LightGBM' if model_name == 'lightgbm' else 'Random Forest' if model_name == 'rf' else 'XGBoost' if model_name == 'xgboost' else '',
		'No Feature Selection': '%.2f \u00b1 %.2f' % (np.mean(perfs_none), np.std(perfs_none)/np.sqrt(len(perfs_none))),
		'LeanML': '%.2f \u00b1 %.2f' % (np.mean(perfs_leanml), np.std(perfs_leanml)/np.sqrt(len(perfs_leanml))),
		'Boruta': '%.2f \u00b1 %.2f' % (np.mean(perfs_boruta), np.std(perfs_boruta)/np.sqrt(len(perfs_boruta))),
		'RFE': '%.2f \u00b1 %.2f' % (np.mean(perfs_rfe), np.std(perfs_rfe)/np.sqrt(len(perfs_rfe)))
	}

	boruta_n_exc_feat = [ns_boruta[i]-ns_leanml[i] for i in range(len(ns_leanml))]
	none_n_exc_feat = [ns_none[i]-ns_leanml[i] for i in range(len(ns_none))]	
	n_excessive_features = {
		'Model': '%s' % 'LightGBM' if model_name == 'lightgbm' else 'Random Forest' if model_name == 'rf' else 'XGBoost' if model_name == 'xgboost' else '',
		'No Feature Selection': '%d \u00b1 %d' % (np.mean(none_n_exc_feat), np.std(none_n_exc_feat)/np.sqrt(len(none_n_exc_feat))),
		'LeanML': '0 \u00b1 0',
		'Boruta': '%d \u00b1 %d' % (np.mean(boruta_n_exc_feat), np.std(boruta_n_exc_feat)/np.sqrt(len(boruta_n_exc_feat))),
		'RFE': '-'
	}

	print(compression_rates)
	print(durations)
	print(performances)
	print(n_excessive_features)

	comps = [0.00, np.mean(leanml_comp), np.mean(boruta_comp), np.mean(leanml_comp)]
	perfs = [np.mean(perfs_none), np.mean(perfs_leanml), np.mean(perfs_boruta), np.mean(perfs_rfe)]
	durs  = [np.mean(durations_none), np.mean(durations_leanml), np.mean(durations_boruta), np.mean(durations_rfe)]
	labels = ['No Feature Selection', 'LeanML', 'Boruta', 'RFE']

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
	axes[0].scatter(perfs, comps, color='red')
	axes[0].annotate(
	    labels[0], 
	    (perfs[0], comps[0]), 
	    xytext=(perfs[0]-0.2, comps[0]+0.07), 
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)
	axes[0].annotate(
	    labels[1], 
	    (perfs[1], comps[1]), 
	    xytext=(perfs[1]-0.1, comps[1]-0.03),
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)
	axes[0].annotate(
	    labels[2], 
	    (perfs[2], comps[2]), 
	    xytext=(perfs[2]-0.05, comps[2]-0.07),
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)
	axes[0].annotate(
	    labels[3], 
	    (perfs[3], comps[3]), 
	    xytext=(perfs[3]+0.02, comps[3]-0.07),
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)

	# axes[0].set_title(pretty_model_name, fontsize=16)
	axes[0].set_xlabel('Performance', fontsize=15)
	axes[0].set_ylabel('Compression Rate\n(Fraction of Deleted Features)', fontsize=15)
	# plt.savefig('./plot_%s_performance_frontiere.png' % model_name)
	# plt.close()


	# fig, ax = plt.subplots(figsize=(12,12))
	axes[1].scatter(durs, comps, color='red')
	axes[1].annotate(
	    labels[0], 
	    (durs[0], comps[0]), 
	    xytext=(durs[0]-0.05, comps[0]+0.07), 
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)
	axes[1].annotate(
	    labels[1], 
	    (durs[1], comps[1]), 
	    xytext=(durs[1]-0.02, comps[1]-0.07),
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)
	axes[1].annotate(
	    labels[2], 
	    (durs[2], comps[2]), 
	    xytext=(durs[2]-0.3, comps[2]-0.07),
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)
	axes[1].annotate(
	    labels[3], 
	    (durs[3], comps[3]), 
	    xytext=(durs[3]-0.05, comps[3]-0.07),
	    arrowprops=dict(
	        arrowstyle="simple",
	        connectionstyle="arc3,rad=-0.2",
	        facecolor='black'),
	    fontsize=12
	)

	fig.suptitle(pretty_model_name, fontsize=16)
	axes[1].set_xlabel('Duration (s)/#Candidate Features', fontsize=15)
	axes[1].set_ylabel('Compression Rate\n(Fraction of Deleted Features)', fontsize=15)
	fig.tight_layout()
	plt.savefig('./plot_%s_frontiere.png' % model_name)
	plt.close()



	return compression_rates, performances, durations, n_excessive_features




if __name__ == '__main__':
	# logging.warning('LightGBM\n\n')
	# lightgbm_regression_benchmark()
	# lightgbm_classification_benchmark()

	# logging.warning('XGBoost\n\n')
	# xgboost_regression_benchmark()
	# xgboost_classification_benchmark()

	# logging.warning('Random Forest\n\n')
	# random_forest_regression_benchmark()
	# random_forest_classification_benchmark()

	# summary()

	lgbm_compression_rates, lgbm_performances, lgbm_durations, lgbm_n_excessive_features = results(
		pretty_model_name='LightGBM', model_name='lightgbm')
	rf_compression_rates, rf_performances, rf_durations, rf_n_excessive_features = results(
		pretty_model_name='Random Forest', model_name='rf')
	xgb_compression_rates, xgb_performances, xgb_durations, xgb_n_excessive_features = results(
		pretty_model_name='XGBoost', model_name='xgboost')

	compression_rates_df = pd.DataFrame([lgbm_compression_rates, rf_compression_rates, xgb_compression_rates])
	compression_rates_df.set_index('Model', inplace=True)
	print()
	print('Compression Rate')
	print(compression_rates_df.to_html())

	print()
	print('Performance')
	performances_df = pd.DataFrame([lgbm_performances, rf_performances, xgb_performances])
	performances_df.set_index('Model', inplace=True)
	print(performances_df.to_html())

	print()
	print('Duration/Feature')
	durations_df = pd.DataFrame([lgbm_durations, rf_durations, xgb_durations])
	durations_df.set_index('Model', inplace=True)
	print(durations_df.to_html())

	print()
	print('#Excessive Features')
	n_excessive_df = pd.DataFrame([lgbm_n_excessive_features, rf_n_excessive_features, xgb_n_excessive_features])
	n_excessive_df.set_index('Model', inplace=True)
	print(n_excessive_df.to_html())





