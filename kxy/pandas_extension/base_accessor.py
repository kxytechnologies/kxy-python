#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from kxy.api import mutual_information_analysis
from kxy.api.core import prepare_data_for_mutual_info_analysis, spearman_corr, pearson_corr

from kxy.asset_management import information_adjusted_correlation, robust_pearson_corr


class BaseAccessor(object):
	"""
	Base class inheritated by our customs accessors.
	"""
	def __init__(self, pandas_obj):
		self._obj = pandas_obj


	def corr(self, columns=(), method='information-adjusted', min_periods=1, p=0, p_ic='hqic'):
		"""
		Calculates the auto-correlation matrix of all columns or the input subset.


		Parameters
		----------
		columns : set, optional
			The set of columns to use. If not provided, all columns are used.
		method : str, optional
			Which method to use to calculate the auto-correlation matrix. Supported
			values are 'information-adjusted' (the default) and all 'method' values of pandas.DataFrame.corr.
		p : int, optional
			The number of auto-correlation lags to use as empirical evidence in the maximum-entropy problem. 
			The default value is 0, which corresponds to assuming rows are i.i.d. Values other than 0 are only
			supported in the robust-pearson method. When p is None, it is inferred from the sample.
		min_periods : int, optional
			Only used when method is not 'information-adjusted'. 
			See the documentation of pandas.DataFrame.corr.
		p_ic : str
			The criterion used to learn the optimal value of :code:`p` (by fitting a VAR(p) model) when :code:`p=None`. Should be one of 'hqic' (Hannan-Quinn Information Criterion), 'aic' (Akaike Information Criterion), 'bic' (Bayes Information Criterion) and 't-stat' (based on last lag). Same as the 'ic' parameter of :code:`statsmodels.tsa.api.VAR`.



		Returns
		-------
		c : pandas.DataFrame
			The auto-correlation matrix.


		.. seealso::

			:ref:`kxy.finance.risk_analysis.information_adjusted_correlation <information-adjusted-correlation>`
			:ref:`kxy.finance.risk_analysis.robust_pearson_corr <robust-pearson-corr>`
		"""
		columns = self._obj.columns if columns == () else list(columns)

		if method == 'information-adjusted':
			c = information_adjusted_correlation(self._obj[columns].values, y=None)
			return pd.DataFrame(c, columns=columns, index=columns)

		if method == 'robust-pearson':
			c = robust_pearson_corr(self._obj[columns].values, y=None, p=p, p_ic=p_ic)
			return pd.DataFrame(c, columns=columns, index=columns)

		if method == 'spearman':
			c = spearman_corr(self._obj[columns].values)
			return pd.DataFrame(c, columns=columns, index=columns)

		if method == 'pearson':
			c = pearson_corr(self._obj[columns].values)
			return pd.DataFrame(c, columns=columns, index=columns)

		else:
			return pd.DataFrame.corr(self._obj[columns], method=method, min_periods=min_periods)


	def is_discrete(self, column):
		"""
		Determine whether the input column contains discrete observations.
		"""
		ret = (not np.can_cast(self._obj[column].values, float))
		ret = ret or len(list(set(self._obj[column].values))) < 0.5*self._obj.shape[0]

		return ret


	def is_categorical(self, column):
		"""
		Determine whether the input column contains categorical observations.
		"""
		ret = (not np.can_cast(self._obj[column].values, float))

		return ret


	def _mutual_information_analysis(self, label_column, input_columns=(), space='dual', \
		categorical_encoding="two-split", non_monotonic_extension=True):
		"""
		"""
		columns = input_columns if len(input_columns) > 0 else [_ for _ in self._obj.columns if _ != label_column]
		cont_columns =  [col for col in columns if not self.is_categorical(col)]
		cat_columns = [col for col in columns if self.is_categorical(col)]

		x_c = self._obj[cont_columns].values if len(cont_columns) > 0 else None
		x_d = self._obj[cat_columns].values if len(cat_columns) > 0 else None
		y_c = None if self.is_categorical(label_column) else self._obj[label_column].values
		y_d = None if not self.is_categorical(label_column) else self._obj[label_column].values

		res = prepare_data_for_mutual_info_analysis(x_c, x_d, y_c, y_d, space=space, \
			non_monotonic_extension=non_monotonic_extension, categorical_encoding=categorical_encoding)
		output_indices = res['output_indices']
		corr = res['corr']
		batch_indices = res['batch_indices']
		mi_ana = mutual_information_analysis(corr, output_indices, space=space, batch_indices=batch_indices)

		return mi_ana, res


	def describe(self,):
		for col in sorted(self._obj.columns):
			print('       ')
			print('---------' + '-'.join(['' for c in col]))
			print('Column: %s' % col)
			print('---------' + '-'.join(['' for c in col]))
			if self._obj.kxy.is_categorical(col):
				print('Type:      Categorical')
				labels, counts = np.unique(self._obj[col].values.astype(str), return_counts=True)
				labels_with_counts = [(labels[i], 100.*counts[i]/self._obj.shape[0]) \
									  for i in range(len(labels))]
				labels_with_counts = sorted(labels_with_counts, key=lambda x: -x[1])
				tot = 0.0
				for label, freq in labels_with_counts:
					print('Frequency: %s%%, Label: %s' % (('%.2f' % freq).rjust(5, ' '), label))
					tot += freq
					if tot > 90. and tot < 100.:
						print('Other Labels: %.2f%%' % (100.-tot))
						break
			else:
				print('Type:   Continuous')
				print('Max:    %.4f' % self._obj[col].max())
				print('Mean:   %.4f' % self._obj[col].mean())
				print('Median: %.4f' % self._obj[col].median())
				print('Min:    %.4f' % self._obj[col].min())  


	def __hash__(self):
		return hash(self._obj.to_string())


