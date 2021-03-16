#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from .base_accessor import BaseAccessor

@pd.api.extensions.register_dataframe_accessor("kxy_finance")
class FinanceAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics finance specific analytics.

	This class defines the :code:`kxy_finance` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_finance.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""
	pass