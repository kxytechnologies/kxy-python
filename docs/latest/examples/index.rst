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




------------
Case Studies
------------


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

