#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .base_accessor import BaseAccessor


@pd.api.extensions.register_dataframe_accessor("kxy_finance")
class FinanceAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various analytics for finance and asset management problems.

	This class defines the :code:`kxy_finance` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_finance.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""

	def beta(self, market_returns_column, asset_returns_columns=(), risk_free_column=None,\
			method='information-adjusted', p=0, p_ic='hqic'):
		"""
		Calculates the beta of a portfolio/asset (whose returns are provided in column_y) 
		with respect to the market (whose returns are provided in market_returns_column) using a variety
		of estimation methods including the standard OLS/Pearson methods and information theoretical 
		alternatives aiming at accounting for nonlinearities and memory in asset returns.


		Parameters
		----------
		asset_returns_columns : str or list of str
			The name(s) of the column(s) to use for portfolio/asset returns.
		market_returns_column : str
			The name of the column to use for market returns.
		method : str, optional
			One of 'information-adjusted', 'robust-pearson', 'spearman',  or 'pearson'. This is the method to use
			to estimate the correlation between portfolio/asset returns and market returns.
		p : int, optional
			The number of auto-correlation lags to use as empirical evidence in the maximum-entropy problem. 
			The default value is 0, which corresponds to assuming rows are i.i.d. Values other than 0 are only
			supported in the robust-pearson method. When p is None, it is inferred from the sample.
		p_ic : str
			The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`.
			Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion),
			'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). Same as the 'ic' parameter of 
			:code:`statsmodels.tsa.api.VAR`.


		Returns
		-------
		c : pandas.DataFrame
			The beta coefficient(s).


		.. seealso::

			:ref:`kxy.finance.factor_analysis.information_adjusted_beta <information-adjusted-beta>`
		"""
		asset_returns_columns = [_ for _ in self._obj.columns if _ != market_returns_column] if asset_returns_columns == () \
			else [asset_returns_columns] if type(asset_returns_columns) == str else list(asset_returns_columns)
		columns = [market_returns_column] + asset_returns_columns

		c = self.corr(method=method, columns=columns, p=p, p_ic=p_ic).values[0, 1:]
		betas = c * np.sqrt(np.nanvar(self._obj[asset_returns_columns].values, axis=0)/\
			np.nanvar(self._obj[market_returns_column].values))

		if type(asset_returns_columns) == str:
			res = pd.DataFrame({asset_returns_columns: betas}).T.rename(columns={0: 'beta'})
		else:
			res = pd.DataFrame({asset_returns_columns[i]: [betas[i]] \
				for i in range(len(asset_returns_columns))}).T.rename(columns={0: 'beta'})

		return res


