.. meta::
	:description: How we use your data.
	:keywords:  Pandas Dataframe, Lean ML, KXY.
	:http-equiv=content-language: en

=========
Your Data
=========

How We Use Your Data
--------------------

.. automodule:: kxy.api.data_transfer
    :members:


Anonymizing Your Data
---------------------
Fortunately, our analyses are invariant by various transformations that can completely anonymize your data. 

You may simply run :code:`df_anonymized = df.kxy.anonymize()` on any dataframe :code:`df` to anonymize it, and work with :code:`df_anonymized` instead :code:`df`. 

Check out the function below for more information on how we anonymize your data.

.. automethod:: kxy.pandas_extension.base_accessor.BaseAccessor.anonymize


