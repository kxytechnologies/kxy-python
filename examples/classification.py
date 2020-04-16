#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['KXY_API_KEY'] = 'YOUR API KEY GOES HERE' #'uD7ncgzjqs3ktJnar1QNI9rL8K7wpu1H2DejCDZ2'
import kxy

if __name__ == '__main__':
	# Classification: predicting whether a given banknote is authentic 
	# given a number of measures taken from a photograph.
	df = kxy.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt', \
		names=['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Is Fake'])

	"""
	PRE-LEARNING
	"""
	# Pre-Learning: How feasible or solvable is this problem? Are features any useful?
	print(df.classification_feasibility('Is Fake'))
	
	# Pre-Learning: How useful is each feature individually?
	importance_df = df.features_importance('Is Fake')
	print(importance_df)
	importance_df.plot.bar(x='feature', y='importance', rot=90)


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
	print('%.2f%%' % (100. * classifier.score(x_test, y_test)))


	"""
	POST-LEARNING
	"""
	# How suboptimal is this logistic regression model?
	print(test_df.classification_suboptimality('prediction', 'Is Fake', discrete_features_columns=(), \
				continuous_features_columns=()))
	# How does it compare to the best case scenario
	print(train_df.classification_feasibility('Is Fake'))

