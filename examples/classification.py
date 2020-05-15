#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
if os.environ.get('KXY_API_KEY', None) is None:
	os.environ['KXY_API_KEY'] = 'YOUR API KEY GOES HERE'
import pylab as plt
import pandas as pd

import kxy

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s - %(process)d - %(levelname)s - %(message)s', level=logging.INFO)
	# Classification: predicting whether a given banknote is authentic 
	# given a number of measures taken from a photograph.
	df = kxy.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt', \
		names=['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Is Fake']).sort_index()

	"""
	PRE-LEARNING
	"""
	# Pre-Learning: How feasible or solvable is this problem? Are inputs any useful?
	logging.info('Overall Classification Feasibility: %.2f nats' % df.classification_feasibility('Is Fake'))
	logging.info('Entropy: %.6f' % kxy.discrete_entropy(df['Is Fake'].values))

	# Pre-Learning: How useful is each input individually?
	logging.info('Individual Input Importance')
	importance_df_1 = df.individual_input_importance('Is Fake')
	logging.info('\n %s' % importance_df_1.round(2))
	importance_df_1 = importance_df_1.set_index(['input'])


	# Incremental importance
	logging.info('')
	logging.info('')
	logging.info('Incremental Input Importance (Dual)')
	importance_df_2 = df.incremental_input_importance('Is Fake')
	logging.info('\n %s' % importance_df_2.round(2))

	logging.info('')
	logging.info('')
	logging.info('Incremental Input Importance (Primal)')
	importance_df_3 = df.incremental_input_importance('Is Fake', space='primal')
	logging.info('\n %s' % importance_df_3.round(2))
	importance_df_2 = importance_df_2.set_index(['input'])

	importance_df = pd.concat([importance_df_1, importance_df_2], axis=1)
	importance_df.reset_index(inplace=True)
	importance_df.rename(columns={'individual_importance': 'Individual Importance', \
		'incremental_importance': 'Incremental Importance', 'index': 'Input', 'selection_order': 'Selection Order', \
		'input': 'Input'}, inplace=True)
	importance_df = importance_df[['Input', 'Individual Importance', 'Incremental Importance', 'Selection Order']].sort_values(by=['Selection Order'], ascending=True)
	ax = importance_df[['Input', 'Individual Importance', 'Incremental Importance']].plot.bar(x='Input', rot=0)
	ax.set_ylabel('Importance (nats)')
	plt.savefig('/Users/yl/Dropbox/KXY Technologies, Inc./GitHubCodeBase/kxy-python/docs/images/bn_importance.png', dpi=500)

	# ax = importance_df.plot.bar(x='input', y='incremental_importance', rot=0)
	# ax.set_ylabel('Input Incremental Importance (nats)')
	# plt.savefig('/Users/yl/Dropbox/KXY Technologies, Inc./GitHubCodeBase/kxy-python/docs/images/bn_incremental_importance.png', dpi=500)


	"""
	LEARNING
	"""
	from sklearn.linear_model import LogisticRegression
	# Training
	train_df = df.iloc[:1000]
	x_train = train_df[['Variance', 'Skewness', 'Kurtosis']].values 
	y_train = train_df['Is Fake'].values
	classifier = LogisticRegression(random_state=0).fit(x_train, y_train)

	# Testing
	test_df = df.iloc[1000:]
	x_test =  test_df[['Variance', 'Skewness', 'Kurtosis']].values
	y_test = test_df['Is Fake'].values

	# Out-of-sample predictions
	predictions = classifier.predict(x_test)
	test_df['prediction'] = predictions

	# Out-of-sample accuracy in %
	logging.info('Out-of-Sample Accuraccy: %.2f%%' % (100. * classifier.score(x_test, y_test)))


	"""
	POST-LEARNING
	"""
	# How suboptimal is this logistic regression model?
	logging.info('Suboptimality: %.6f' %test_df.classification_suboptimality('prediction', 'Is Fake', discrete_input_columns=(), \
				continuous_input_columns=()))
	# How does it compare to the best case scenario
	logging.info('Training Classification Feasibility: %.2f nats' % train_df.classification_feasibility('Is Fake'))

