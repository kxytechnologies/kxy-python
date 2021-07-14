#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from kxy.finance import information_adjusted_correlation as ia_corr

from .base_accessor import BaseAccessor

@pd.api.extensions.register_dataframe_accessor("kxy_finance")
class FinanceAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various finance-specific analytics.

	This class defines the :code:`kxy_finance` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_finance.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""
	def information_adjusted_beta(self, market_column, asset_column, anonymize=False):
		"""
		Estimate the information-adjusted beta of an asset return :math:`r` relative to the market return :math:`r_m`: :math:`\\text{IA-}\\beta := \\text{IA-Corr}\\left(r, r_m \\right) \\sqrt{\\frac{\\text{Var}(r)}{\\text{Var}(r_m)}}`,
		where :math:`\\text{IA-Corr}\\left(r, r_m \\right) := \\text{sgn}\\left(\\text{Corr}\\left(r, r_m \\right) \\right) \\left[1 - e^{-2I(r, r_m)} \\right]` denotes the information-adjusted correlation coefficient, with :math:`\\text{sgn}\\left(\\text{Corr}\\left(r, r_m \\right) \\right)` the sign of the Pearson correlation coefficient.

		Unlike the traditional beta coefficient, namely :math:`\\beta := \\text{Corr}\\left(r, r_m \\right) \\sqrt{\\frac{\\text{Var}(r)}{\\text{Var}(r_m)}}`, that only captures linear relations between market and asset returns, and that is 0 if and only if the two are **decorrelated**, :math:`\\text{IA-}\\beta` captures any relationship between asset return and market return, linear or nonlinear, and is 0 if and only if the two variables are **statistically independent**. 

		Parameters
		----------
		market_column : str
			The name of the column containing market returns.
		asset_column : str
			The name of the column containing asset returns.
		anonymize : bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost).


		Returns
		-------
		result : float
			The information-adjusted beta coefficient.

		"""
		assert market_column in self._obj.columns, 'The market column should be a column'
		assert asset_column in self._obj.columns, 'The asset column should be a column'

		m_std = np.nanstd(self._obj[market_column].values)
		a_std = np.nanstd(self._obj[asset_column].values)

		return self.information_adjusted_correlation(market_column, asset_column, anonymize=anonymize)*a_std/m_std



	def information_adjusted_correlation(self, market_column, asset_column, anonymize=False):
		"""
		Estimate the information-adjusted correlation between an asset return :math:`r` and the market return :math:`r_m`: :math:`\\text{IA-Corr}\\left(r, r_m \\right) := \\text{sgn}\\left(\\text{Corr}\\left(r, r_m \\right) \\right) \\left[1 - e^{-2I(r, r_m)} \\right]`, where :math:`\\text{sgn}\\left(\\text{Corr}\\left(r, r_m \\right) \\right)` is the sign of the Pearson correlation coefficient.

		Unlike Pearson's correlation coefficient, which is 0 if and only if asset return and market return are **decorrelated** (i.e. they exhibit no linear relation), information-adjusted correlation is 0 if and only if market and asset returns are **statistically independent** (i.e. the exhibit no relation, linear or nonlinear).


		Parameters
		----------
		market_column : str
			The name of the column containing market returns.
		asset_column : str
			The name of the column containing asset returns.
		anonymize : bool
			When set to true, your explanatory variables will never be shared with KXY (at no performance cost).


		Returns
		-------
		result : float
			The information-adjusted correlation.

		"""
		assert market_column in self._obj.columns, 'The market column should be a column'
		assert asset_column in self._obj.columns, 'The asset column should be a column'

		_obj = self.anonymize(columns_to_exclude=[]) if anonymize else self._obj

		return ia_corr(_obj, market_column, asset_column)







