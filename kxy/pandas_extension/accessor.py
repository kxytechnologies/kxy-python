#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
We define a custom :code:`kxy` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_ below, 
namely the class :code:`Accessor`, that extends the pandas DataFrame class with all our analyses, thereby allowing data scientists to tap into 
the power of the :code:`kxy` toolkit within the comfort of their favorite data structure.

All methods defined in the :code:`Accessor` class are accessible from any DataFrame instance as :code:`df.kxy.<method_name>`, so long as the :code:`kxy` python 
package is imported alongside :code:`pandas`. 
"""


import pandas as pd

from .features_accessor import FeaturesAccessor
from .finance_accessor import FinanceAccessor
from .learning_accessor import LearningAccessor
from .post_learning_accessor import PostLearningAccessor
from .pre_learning_accessor import PreLearningAccessor


@pd.api.extensions.register_dataframe_accessor("kxy")
class Accessor(PreLearningAccessor, LearningAccessor, PostLearningAccessor, FinanceAccessor, FeaturesAccessor):
	"""
	Extension of the pandas.DataFrame class with the full capabilities of the :code:`kxy` platform.
	"""
	pass