#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
if os.environ.get('KXY_API_KEY', None) is None:
	os.environ['KXY_API_KEY'] = 'YOUR API KEY GOES HERE'
import pylab as plt
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
	print('Overall Classification Feasibility: %.2f nats' % df.classification_feasibility('Is Fake'))

	# Pre-Learning: How useful is each input individually?
	importance_df = df.input_importance('Is Fake')
	print('Feature Importance')
	print(importance_df.round(2))
	ax = importance_df.plot.bar(x='input', y='importance', rot=0)
	ax.set_ylabel('Feature Importance (nats)')
	plt.savefig('/Users/yl/Dropbox/KXY Technologies, Inc./GitHubCodeBase/kxy-python/docs/images/bn_importance.png', dpi=500)


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
	print('Out-of-Sample Accuraccy: %.2f%%' % (100. * classifier.score(x_test, y_test)))


	"""
	POST-LEARNING
	"""
	# How suboptimal is this logistic regression model?
	print('Suboptimality: %.2f' %test_df.classification_suboptimality('prediction', 'Is Fake', discrete_input_columns=(), \
				continuous_input_columns=()))
	# How does it compare to the best case scenario
	print('Training Classification Feasibility: %.2f nats' % train_df.classification_feasibility('Is Fake'))

