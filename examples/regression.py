#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
if os.environ.get('KXY_API_KEY', None) is None:
	os.environ['KXY_API_KEY'] = 'YOUR API KEY GOES HERE'
import logging

import numpy as np
import pandas as pd
import kxy

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s - %(process)d - %(levelname)s - %(message)s', level=logging.INFO)
	# Regression: 
	df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data', \
		sep='[ ]{1,2}', names=['Longitudinal Position', 'Prismatic Coeefficient', 'Length-Displacement', \
		'Beam-Draught Ratio', 'Length-Beam Ratio', 'Froude Number', 'Residuary Resistance'], engine='python')
	df.rename(columns={col: col.title() for col in df.columns}, inplace=True)
	logging.info('\n %s' % df)

	"""
	PRE-LEARNING
	"""
	label_column = 'Residuary Resistance'
	# Pre-Learning: How feasible or solvable is this problem? Are inputs any useful?
	logging.info('Feasibility: %.4f, Entropy: %.4f' % (\
		df.kxy.regression_feasibility(label_column), kxy.scalar_continuous_entropy(df[label_column].values)))
	
	# Pre-Learning: How useful is each input individually?
	importance_df = df.kxy.individual_input_importance(label_column, problem='regression')
	logging.info('')
	logging.info('')
	logging.info('Individual Input Importance (Dual):\n %s' % importance_df.round(4))
	importance_df.plot.bar(x='input', y='individual_importance', rot=90)

	# Incremental importance
	logging.info('')
	logging.info('')
	logging.info('Information-Adjusted Correlation Matrix:\n %s' % df.kxy.corr().round(4))
	logging.info('')
	logging.info('')
	logging.info('Pearson Correlation Matrix:\n %s' % df.kxy.corr(method='pearson').round(4))
	logging.info('')
	logging.info('')
	importance_df = df.kxy.incremental_input_importance(label_column, greedy=True)
	logging.info('Incremental Input Importance (Dual):\n %s' % importance_df.round(4))
	logging.info('')
	logging.info('')
	importance_df = df.kxy.incremental_input_importance(label_column, space='primal', greedy=True)
	logging.info('Incremental Input Importance (Primal):\n %s ' % importance_df.round(4))




	"""
	LEARNING (BASIC)
	"""
	from sklearn.linear_model import LinearRegression
	# Training
	train_size = 200
	train_df = df.iloc[:train_size]
	x_train = train_df[['Froude Number']].values
	y_train = train_df[label_column].values
	model = LinearRegression().fit(x_train, y_train)

	# Testing
	test_df = df.iloc[train_size:]
	x_test = test_df[['Froude Number']].values
	y_test = test_df[label_column].values

	# Out-of-sample predictions
	predictions = model.predict(x_test)
	test_df = test_df.assign(prediction=predictions)

	# Out-of-sample accuracy (R^2)
	logging.info('Out-Of-Sample R^2: %.2f' % (model.score(x_test, y_test)))

	"""
	POST-LEARNING
	"""
	# How suboptimal is this linear regression model?
	logging.info('Additive Suboptimality: %.4f' % test_df.kxy.regression_additive_suboptimality('prediction', label_column))
	logging.info('Suboptimality: %.4f' % test_df.kxy.regression_suboptimality('prediction', label_column))

