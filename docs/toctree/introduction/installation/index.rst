
===============
Getting Started
===============

The **KXY platform** is the `only AutoML platform` dedicated to bringing about **business value** by addressing two questions 
that are fundamental to the data science life cycle in a modeel-free, fast, rigorous and and cost-effective manner:

* **Pre-Learning**: How valuable is a dataset for improving decision making, and what datasets should be used in models?
* **Post-Learning**: Which action would yield a higher ROI, trying to improvee your production model using the same datasets, or investing in acquiring additional and complementary datasets?


Installation
------------

The KXY python package can be installed from PyPi by running

.. code-block:: bash
	
	pip install kxy

The package only supports python 3.


Authentication
--------------

All heavy-duty operations, including but not limited to solving any maximum-entropy optimization problem, are run on the KXY infrastructure and require an API key. The API key should be set in the environment variable ``KXY_API_KEY``. 

This can be done on MacOS or Linux distributions by running 

.. code-block:: bash
	
	export KXY_API_KEY='YOUR API KEY GOES HERE'

or from within your python program by running 

.. code-block:: python

	>>> import os
	>>> os.environ['KXY_API_KEY'] = 'YOUR API KEY GOES HERE'


Examples
--------


Loading Data
^^^^^^^^^^^^

.. code-block:: python

	>>> import kxy
	>>> # Bank note UCI dataset (binary classification)
	>>> url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/'\
	... 	'data_banknote_authentication.txt'
	>>> df = kxy.read_csv(url, names=['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Is Fake'])

.. seealso::

	:ref:`kxy.read_excel <read-excel>`, :ref:`kxy.read_html <read-html>`, :ref:`kxy.read_sql <read-sql>`, and :ref:`kxy.read_table <read-table>`.


Classification
^^^^^^^^^^^^^^

Pre-Learning: Feasibility
"""""""""""""""""""""""""
To what extent can we predict whether the bank note is fake from an image using provided 
features (irrespective of the classification model)?

.. code-block:: python

	>>> df.classification_feasibility('Is Fake')
	0.855179

The higher the number is relative to the entropy of the responsee, the better. A value of 0
means no model can successfully solve this classification problem using provided features, no
matter how complex, no matter how deep. In such as case, resources should be allocated to 
looking for more relevant dataset to use as features, rather than increasing model complexity.

.. code-block:: python

	>>> kxy.discrete_entropy(df['Is Fake'].values)
	0.686998


Pre-Learning: Feature Importance
""""""""""""""""""""""""""""""""
Once we know the problem is feasible using features at hand, the next question before we jump
into modelling is what are the features that are the most useful for solving this problem. Once
more, this qustion is asked and answered independently from any classification model (hence the expression **pre-learning**),
and reduces time wasted improving models fitted on irrelevant features.


.. code-block:: python

	>>> importance_df = df.features_importance('Is Fake')
	>>> importance_df
	    feature  importance
	0  Variance    0.385260
	1  Skewness    0.226708
	2  Kurtosis    0.119311
	3  Entropy     0.000000
	>>> importance_df.plot.bar(x='feature', y='importance', rot=0)
	

.. figure:: ../../../images/bn_importance.png
	:width: 500px
	:align: center
	:height: 300px
	:alt: Importance bar plot
	:figclass: align-center


Learning
""""""""
Let's train a linear classifier on our dataset.

.. code-block:: python

	>>> from sklearn.linear_model import LogisticRegression
	>>> # Training
	>>> train_df = df.iloc[:1000]
	>>> x_train = train_df[['Variance', 'Skewness', 'Kurtosis']].values 
	>>> y_train = train_df['Is Fake'].values
	>>> classifier = LogisticRegression(random_state=0).fit(x_train, y_train)

	>>> # Testing
	>>> test_df = df.iloc[1000:]
	>>> x_test =  test_df[['Variance', 'Skewness', 'Kurtosis']].values
	>>> y_test = test_df['Is Fake'].values

	>>> # Out-of-sample predictions
	>>> predictions = classifier.predict(x_test)
	>>> test_df['prediction'] = predictions

	>>> # Out-of-sample accuracy in %
	>>> '%.2f%%' % (100. * classifier.score(x_test, y_test))
	'98.92%'


Post-Learning: Suboptimality
""""""""""""""""""""""""""""
As we train models, we are faced with the following dilemma on resource allocation: are we better off investing time and money 
trying to improve our existing model with a complex model, or have we already gotten the most out of existing features, and we 
should instead invest in acquiring complementary and/or more informative features?

Back to our bank note example, given how high an out-of-sample accuracy we got, it might seem like a silly question, but can we do better using the same features and a nonlinear model?

.. code-block:: python

	>>> test_df.classification_suboptimality('prediction', 'Is Fake', \
	... 	discrete_features_columns=(), continuous_features_columns=())
	0.012520
	>>> train_df.classification_feasibility('Is Fake')
	0.557628

As it turns out, a simple logistic regression allows us to extract 98% of the intrinsic value there is in using the 3 features above to determmine whether a bank note is fake. Thus, using a nonlinear model might not yield the highest ROI. 

That a nonlinear model would not perform materially better than a linear model is consistent with the visualization below, where it can be seen that a curve would not necessarily do a much better job at separating geniune (green) from fake (red) notes than a straight line.


.. code-block:: python

	>>> import pylab as plt
	>>> fig, ax = plt.subplots(2, 2, figsize=(15, 10))
	>>> df.plot.scatter(ax=ax[0, 0], x='Variance', y='Skewness', c=df['Is Fake'].\
	...		apply(lambda x: 'r' if x == 1. else 'g'))
	>>> df.plot.scatter(ax=ax[0, 1], x='Variance', y='Kurtosis', c=df['Is Fake'].\
	...		apply(lambda x: 'r' if x == 1. else 'g'))
	>>> df.plot.scatter(ax=ax[1, 0], x='Skewness', y='Kurtosis', c=df['Is Fake'].\
	...		apply(lambda x: 'r' if x == 1. else 'g'))
	>>> df.plot.scatter(ax=ax[1, 1], x='Kurtosis', y='Entropy',  c=df['Is Fake'].\
	...		apply(lambda x: 'r' if x == 1. else 'g'))


.. figure:: ../../../images/bn_separability.png
	:width: 900px
	:align: center
	:height: 500px
	:alt: Importance bar plot
	:figclass: align-center



Regression
^^^^^^^^^^

.. code-block:: python

	>>> # Regression: 
	>>> df = kxy.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data', \
		sep='[ ]{1,2}', names=['Longitudinal Position', 'Prismatic Coeefficient', 'Length-Displacement', \
		'Beam-Draught Ratio', 'Length-Beam Ratio', 'Froude Number', 'Residuary Resistance'])
	>>> df.rename(columns={col: col.title() for col in df.columns}, inplace=True)
	>>> print(df)
    Longitudinal Position  Prismatic Coeefficient  Length-Displacement  Beam-Draught Ratio  Length-Beam Ratio  Froude Number  Residuary Resistance
	0                     -2.3                   0.568                 4.78                3.99               3.17          0.125                  0.11
	1                     -2.3                   0.568                 4.78                3.99               3.17          0.150                  0.27
	2                     -2.3                   0.568                 4.78                3.99               3.17          0.175                  0.47
	3                     -2.3                   0.568                 4.78                3.99               3.17          0.200                  0.78
	4                     -2.3                   0.568                 4.78                3.99               3.17          0.225                  1.18
	..                     ...                     ...                  ...                 ...                ...            ...                   ...
	303                   -2.3                   0.600                 4.34                4.23               2.73          0.350                  8.47
	304                   -2.3                   0.600                 4.34                4.23               2.73          0.375                 12.27
	305                   -2.3                   0.600                 4.34                4.23               2.73          0.400                 19.59
	306                   -2.3                   0.600                 4.34                4.23               2.73          0.425                 30.48
	307                   -2.3                   0.600                 4.34                4.23               2.73          0.450                 46.66

	[308 rows x 7 columns]


Pre-Learning
""""""""""""

.. code-block:: python

	>>> label_column = 'Residuary Resistance'
	>>> # Pre-Learning: How feasible or solvable is this problem? Are features any useful?
	>>> print('Feasibility: %.4f, Entropy: %.4f' % (\
	... 	df.regression_feasibility(label_column), kxy.scalar_continuous_entropy(df[label_column].values)))
	Feasibility: 0.0180, Entropy: 2.8780

	>>> # Pre-Learning: How useful is each feature individually?
	>>> importance_df = df.features_importance(label_column, problem='regression')
	>>> print(importance_df)
	                  feature  importance
	0           Froude Number      1.7618
	1   Longitudinal Position      0.0004
	2     Length-Displacement      0.0004
	3      Beam-Draught Ratio      0.0002
	4       Length-Beam Ratio          -0
	5  Prismatic Coeefficient          -0


Post-Learning
"""""""""""""

.. code-block:: python

	>>> # Learning (Basic Linear Regression)
	>>> from sklearn.linear_model import LinearRegression
	>>> # Training
	>>> train_size = 200
	>>> train_df = df.iloc[:train_size]
	>>> x_train = train_df[['Froude Number']].values
	>>> y_train = train_df[label_column].values
	>>> model = LinearRegression().fit(x_train, y_train)

	>>> # Testing
	>>> test_df = df.iloc[train_size:]
	>>> x_test = test_df[['Froude Number']].values
	>>> y_test = test_df[label_column].values

	>>> # Out-of-sample predictions
	>>> predictions = model.predict(x_test)
	>>> test_df['Prediction'] = predictions

	>>> # Out-of-sample accuracy (R^2)
	>>> print('Out-Of-Sample R^2: %.2f' % (model.score(x_test, y_test)))
	Out-Of-Sample R^2: 0.65

	>>> # How suboptimal is this linear regression model?
	>>> # Can we do better with a nonlinear model, without new features?
	>>> print('Additive Suboptimality: %.4f' % \
	...		test_df.regression_additive_suboptimality('Prediction', label_column))
	Additive Suboptimality: 0.0000


