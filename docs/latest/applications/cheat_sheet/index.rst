.. meta::
	:description: Description of KXY's main functions, and how to access them in Python.
	:keywords:  KXY Tutorials, KXY Cheatsheet.
	:http-equiv=content-language: en



----------
Cheatsheet
----------

Imports
-------

.. code-block:: python

   import pandas as pd
   import kxy

From now on, :code:`df` refers to a Pandas dataframe object and :code:`y_column` is the column of :code:`df` to be used as target. All columns in :code:`df` but :code:`y_column` are treated as explanatory variables. :code:`problem_type` is a variable taking value :code:`'regression'` for regression problems and :code:`'classification'` for classification problems.

Data Valuation
--------------

.. code-block:: python

   df.kxy.data_valuation(y_column, problem_type=problem_type)


By default, your data is transmitted to our backend in clear. To anonymize your data before performing data valuation, simply set :code:`anonymize=True`.

.. code-block:: python

   df.kxy.data_valuation(y_column, problem_type=problem_type, anonymize=True) # Data valuation using anonymized data.



Automatic (Model-Free) Feature Selection
----------------------------------------

.. code-block:: python

   df.kxy.variable_selection(y_column, problem_type=problem_type)

By default, your data is transmitted to our backend in clear. To anonymize your data before performing automatic feature selection, simply set :code:`anonymize=True`.

.. code-block:: python

   df.kxy.variable_selection(y_column, problem_type=problem_type, anonymize=True) # Variable selection using anonymized data.



Model Compression
-----------------
Here's how to wrap feature selection around LightGBM in Python.

.. code-block:: python

   from kxy.learning import get_lightgbm_learner_learning_api

   params = {
      'objective': 'rmse',  
      'boosting_type': 'gbdt',
      'num_leaves': 100,
      'n_jobs': -1,
      'learning_rate': 0.1,
      'verbose': -1,
   }
   learner_func = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
      early_stopping_rounds=50, verbose_eval=50, feature_selection_method='leanml')
   results = df.kxy.fit(y_column, learner_func, problem_type=problem_type)

   # The trained model
   predictor = results['predictor']

   # Feature columns selected
   selected_variables = predictor.selected_variables

   # To make predictions out of a dataframe of test data.
   predictions = predictor.predict(test_df)

Parameters of :code:`get_lightgbm_learner_learning_api` should be the same as those of :code:`lightgbm.train`. See the `LightGBM documentation<https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html>`_.


Wrapping feature selection around another model in Python is identical except for :code:`learner_func`. Here's how to create :code:`learner_func` for other models.

For XGBoost:

.. code-block:: python

   from kxy.learning import get_xgboost_learner
   # Use 'xgboost.XGBClassifier' for classification problems.
   xgboost_learner_func = get_xgboost_learner('xgboost.XGBRegressor')


Parameters of :code:`get_xgboost_learner` should be those you'd pass to instantiate :code:`xgboost.XGBRegressor` or :code:`xgboost.XGBClassifier`. See the `XGBoost documentation<https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn>`_.


For Scikit-Learn models:

.. code-block:: python

   from kxy.learning import get_sklearn_learner
   # Replace 'sklearn.ensemble.RandomForestRegressor' with the import path of the sklearn model you want to use. 
   rf_learner_func = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', \
                  min_samples_split=0.01, max_samples=0.5, n_estimators=100)
   df.kxy.fit(y_column, rf_learner_func, problem_type=problem_type)


Parameters of :code:`get_sklearn_learner` should be those you'd pass to instantiate the scikit-learn model.



Model-Driven Improvability
--------------------------
For the model-driven improvability analysis, predictions made by the production model should be contained in a column of the :code:`df`. The variable :code:`prediction_column` refers to said column. All columns in :code:`df` but :code:`y_column` and :code:`prediction_column` are considered to be the explanatory variables/features used to train the production model.


.. code-block:: python

   anonymize = False # Set to True to anonymize your data before model-driven improvability
   df.kxy.model_driven_improvability(y_column, prediction_column, problem_type=problem_type, anonymize=anonymize)



Data-Driven Improvability
-------------------------
For the data-driven improvability analysis, the list of columns representing new features/explanatory variables to consider (:code:`new_variables`) should be provided. All columns in :code:`df` that are neither :code:`y_column` nor contained in :code:`new_variables` are assumed to be the explanatory variables/features used to trained the production model.


.. code-block:: python

   anonymize = False # Set to True to anonymize your data before model-driven improvability
   df.kxy.data_driven_improvability(y_column, new_variables, problem_type=problem_type, anonymize=anonymize)



