.. meta::
	:description: Examples and tutorials illustrating how the KXY AutoML platform works, and what can be done with it.
	:keywords:  KXY Tutorials, KXY Examples.
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



-----------------------
Examples (Kaggle & UCI)
-----------------------

* :ref:`APS Failure (UCI, Classification, n=76000, d=170, 2 classes)`
* :ref:`Abalone (UCI, Regression, n=4177, d=8)`
* :ref:`Adult (UCI, Classification, n=48843, d=14, 3 classes)`
* :ref:`Air Foil (UCI, Regression, n=1503, d=5)`
* :ref:`Air Quality (UCI, Regression, n=8991, d=14)`
* :ref:`Avila (UCI, Classification, n=20867, d=10, 12 classes)`
* :ref:`Bank Marketing (UCI, Classification, n=41188, d=20, 2 classes)`
* :ref:`Bank Note (UCI, Classification, n=1372, d=4, 2 classes)`
* :ref:`Bike Sharing (UCI, Regression, n=17379, d=18)`
* :ref:`Blog Feedback (UCI, Regression, n=60021, d=280)`
* :ref:`CT Slices (UCI, Regression, n=53500, d=385)`
* :ref:`Card Default (UCI, Classification, n=30000, d=23, 2 classes)`
* :ref:`Concrete (UCI, Regression, n=1030, d=8)`
* :ref:`Diabetic Retinopathy (UCI, Classification, n=1151, d=19, 2 classes)`
* :ref:`EEG Eye State (UCI, Classification, n=14980, d=14, 2 classes)`
* :ref:`Energy Efficiency (UCI, Regression, n=768, d=8)`
* :ref:`Facebook Comments (UCI, Regression, n=209074, d=53)`
* :ref:`Heart Attack (Kaggle, Classification, n=303, d=13, 2 classes)`
* :ref:`Heart Disease (Kaggle, Classification, n=303, d=13, 2 classes)`
* :ref:`House Prices Advanced (Kaggle, Regression, n=1460, d=79)`
* :ref:`Landsat (UCI, Classification, n=6435, d=36, 6 classes)`
* :ref:`Letter Recognition (UCI, Classification, n=20000, d=16, 26 classes)`
* :ref:`Magic Gamma (UCI, Classification, n=19020, d=10, 2 classes)`
* :ref:`Naval Propulsion (UCI, Regression, n=11934, d=16)`
* :ref:`Online News (UCI, Regression, n=39644, d=58)`
* :ref:`Parkinson (UCI, Regression, n=5875, d=20)`
* :ref:`Power Plant (UCI, Regression, n=9568, d=4)`
* :ref:`Real Estate (UCI, Regression, n=414, d=6)`
* :ref:`Sensor Less Drive (UCI, Classification, n=58509, d=48, 11 classes)`
* :ref:`Shuttle (UCI, Classification, n=58000, d=9, 7 classes)`
* :ref:`Skin Segmentation (UCI, Classification, n=245057, d=3, 2 classes)`
* :ref:`Social Media Buzz (UCI, Regression, n=583250, d=77)`
* :ref:`Superconductivity (UCI, Regression, n=21263, d=81)`
* :ref:`Titanic (Kaggle, Classification, n=891, d=11, 2 classes)`
* :ref:`Water Quality (Kaggle, Classification, n=3276, d=9, 2 classes)`
* :ref:`White Wine Quality (UCI, Regression, n=4898, d=11)`
* :ref:`Yacht Hydrodynamics (UCI, Regression, n=308, d=6)`
* :ref:`Year Prediction MSD (UCI, Regression, n=515345, d=90)`



------------
Case Studies
------------

* :ref:`Evaluating KXY's Data Valuation Function (Classification)`
* :ref:`Evaluating KXY's Data Valuation Function (Regression)`
* :ref:`Automatically Pruning Redundant Features With KXY`
* :ref:`Detecting Features That Are Only Useful In Conjunction With Others`
* :ref:`Better Solving Heavily Unbalanced Classification Problems With KXY`



Classification
--------------

* :ref:`Toy Visual Classification Example`
* :ref:`Classification Problem With Some Useless Variables`
* :ref:`Complex Classification Example`


Regression
----------
* :ref:`Toy 1D Regression Examples`
* :ref:`Toy Multivariate Regression Examples`
* :ref:`Regression Problem With Some Useless Variables`
* :ref:`Complex Regression Example`






