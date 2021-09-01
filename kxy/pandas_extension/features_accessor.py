#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .base_accessor import BaseAccessor
from .features_utils import nanskew, nankurtosis, q25, q75, mode, modefreq, nextmode, nextmodefreq, lastmode, lastmodefreq, \
	nanmin, nanmax, nanmean, nanstd, nanmedian, nanmaxmmin


@pd.api.extensions.register_dataframe_accessor("kxy_features")
class FeaturesAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various feature engineering functionalities.

	This class defines the :code:`kxy_features` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_features.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""
	def ordinally_encode(self, target_column=None, method='one_hot'):
		"""
		Encode categorical (non-numeric) data.

		Parameters
		----------
		target_column : str
			The name of the column containing labels. When this column is categorical, each label is replaced by a distinct integer.
		method : 'one_hot' (default) | 'binary'
			Whether to use one-hot encoding or binary encoding to encode categorical variables.


		Returns
		-------
		result : pandas.DataFrame
			The ordinarily encoded dataframe.
		"""
		assert target_column is None or target_column in self._obj.columns, 'The target_column should be a column'
		assert method.lower() in ['one_hot', 'binary'], 'The encoding method should be either one_hot or binary'

		cat_columns = [col for col in self._obj.columns if col is not target_column and self.is_categorical(col)]
		
		if cat_columns:
			num_columns = [col for col in self._obj.columns if col is not target_column and col not in cat_columns]
			if method.lower() == 'one_hot':
				# One-hot encode categorical explanatory variables
				cat_x_encoded = pd.get_dummies(self._obj[cat_columns], prefix=cat_columns)

			if method.lower() == 'binary':
				# Binary encode categorical explanatory variables
				cat_x_encoded = pd.concat([self._binary_encode(col) for col in cat_columns], axis=1)

			if target_column:
				df_encoded = pd.concat([self._obj[[target_column]+num_columns], cat_x_encoded], axis=1)
			else:
				df_encoded = pd.concat([self._obj[num_columns], cat_x_encoded], axis=1)

			# Ordinarily encode the target if needed (e.g. for classification with non-numeric classes)
			if target_column and self.is_categorical(target_column):
				all_classes = sorted(list(set(list(self._obj[target_column]))))
				classes_map = {all_classes[i]: i for i in range(len(all_classes))}
				df_encoded[target_column] = df_encoded[target_column].apply(lambda c: classes_map[c])

		else:
			df_encoded = self._obj

		return df_encoded.astype(float)


	def _binary_encode(self, column):
		'''
		Binary-encode a specific column.
		'''
		assert self.is_categorical(column), 'The column should be categorical'
		x = self._obj[column].values
		x_ = np.array([str(_) for _ in x]).astype(str)
		n = len(x_)
		cats = set(x_)
		cats = list(sorted(cats))
		q = len(cats)
		h = int(np.ceil(np.log2(q)))
		cats_map = {cats[i]: i for i in range(q)}
		res = np.array([[int(_) for _ in list(bin(cats_map[x_[i]])[2:].zfill(h))] for i in range(n)])

		if len(res.shape) == 1:
			res = res[:, None]

		columns = ['%s_bit_%d' % (column, i) for i in range(h)]
		res = pd.DataFrame(res, columns=columns)

		return res


	def entity_features(self, entity, exclude=[], name='*'):
		"""
		Group rows corresponding to the same entity and apply aggregation functions.

		For each ordinal column, we apply the following aggregation functions to rows corresponding to the same entity: mean, standard deviation, median, skewness, kurtosis, 25th and 75th percentiles, minimum, maximum, and the difference betwween maximum and minimum.

		For each non-ordinal column, we apply the following aggregation functions to rows corresponding to the same entity: mode and its frequency, second most frequent label and its frequency, least frequent label and its frequency.


		Parameters
		----------
		entity : str
			The column mapping rows to entities.
		exclude : list
			A list of columns to exclude from feature transformations.


		Returns
		-------
		result : pandas.DataFrame
			The dataframe of features.
		"""
		assert entity in self._obj.columns, 'The entity %s should be valid column' % entity
		entity_grp = self._obj.groupby(entity)

		cat_columns = [col for col in self._obj.columns if self.is_categorical(col) and col != entity and col not in exclude]
		ord_columns = [col for col in self._obj.columns if not self.is_categorical(col) and col != entity and col not in exclude]
		columns = cat_columns + ord_columns

		dfs = []

		# Aggregation of categorical variables
		if cat_columns:
			# Most frequent value and it's frequency of occurence
			cat_agg = {'MODE(%s)' % col: (col, mode) for col in cat_columns}
			cat_agg.update({'MODEFREQ(%s)' % col: (col, modefreq) for col in cat_columns})

			# Second most frequent value and it's frequency of occurence
			cat_agg.update({'NEXTMODE(%s)' % col: (col, nextmode) for col in cat_columns})
			cat_agg.update({'NEXTMODEFREQ(%s)' % col: (col, nextmodefreq) for col in cat_columns})

			# Least frequent value and it's frequency of occurence
			cat_agg.update({'LASTMODE(%s)' % col: (col, lastmode) for col in cat_columns})
			cat_agg.update({'LASTMODEFREQ(%s)' % col: (col, lastmodefreq) for col in cat_columns})

			cat_df = entity_grp.agg(**cat_agg)
			dfs += [cat_df]

		# Aggregation of ordinal variables
		if ord_columns:
			ord_agg = {'MEAN(%s)' % col: (col, nanmean) for col in ord_columns}
			ord_agg.update({'STD(%s)' % col: (col, nanstd) for col in ord_columns})
			ord_agg.update({'MEDIAN(%s)' % col: (col, nanmedian) for col in ord_columns})
			ord_agg.update({'SKEW(%s)' % col: (col, nanskew) for col in ord_columns})
			ord_agg.update({'KURT(%s)' % col: (col, nankurtosis) for col in ord_columns})
			ord_agg.update({'Q25(%s)' % col: (col, q25) for col in ord_columns})
			ord_agg.update({'Q75(%s)' % col: (col, q75) for col in ord_columns})
			ord_agg.update({'MIN(%s)' % col: (col, nanmin) for col in ord_columns})
			ord_agg.update({'MAX(%s)' % col: (col, nanmax) for col in ord_columns})
			ord_agg.update({'MAX-MIN(%s)' % col: (col, nanmaxmmin) for col in ord_columns})
			ord_df = entity_grp.agg(**ord_agg)
			dfs += [ord_df]
		
		# Number of rows per entity
		count_agg = {'COUNT(%s)' % name: (columns[0], 'count')}
		count_df = entity_grp.agg(**count_agg)
		dfs += [count_df]

		df = pd.concat(dfs, axis=1)

		return df


	def deviation_features(self, exclude=[], means=None, quantiles=None, return_baselines=False):
		"""
		Extend the dataframe with deviations of ordinal columns from row-wise aggregtes such as mean, median, 25th and 75th percentiles.


		Parameters
		----------
		exclude : list
			A list of columns to exclude from feature transformations.
		means : pandas.DataFrame | None
			Which values, if any, to use as means.
		quantiles : pandas.DataFrame | None
			Which values, if any, to use as 25th, 50th, 75th percentiles.
		return_baselines : bool
			Whether to return which baselines have been used.


		Returns
		-------
		result : pandas.DataFrame
			The original dataframe extended with computed features.
		"""
		ord_columns = [col for col in self._obj.columns if not self.is_categorical(col) and col not in exclude]

		if means is None:
			means = self._obj.mean(axis=0, skipna=True)

		if quantiles is None:
			quantiles = self._obj.quantile(q=[0.25, 0.5, 0.75])

		df = self._obj.copy()
		if ord_columns:
			for col in ord_columns:
				df['| %s - MEAN(%s)|' % (col, col)] = np.abs(df[col]-means.loc[col])
				df['| %s - MEDIAN(%s)|' % (col, col)] = np.abs(df[col]-quantiles.loc[0.5][col])
				df['| %s - Q25(%s)|' % (col, col)] = np.abs(df[col]-quantiles.loc[0.25][col])
				df['| %s - Q75(%s)|' % (col, col)] = np.abs(df[col]-quantiles.loc[0.75][col])

		if return_baselines:
			return df, means, quantiles
		else:
			return df


	def temporal_features(self, max_lag=10, exclude=[], index=None):
		"""
		Extend the dataframe with some rolling statistics (e.g. rolling average, rolling min, rolling max, rolling max-rolling min, etc.) for all lags from 2 to the configured maximum lag.


		Parameters
		----------
		exclude : list
			A list of columns to exclude from feature transformations.
		max_lag : int
			The largest lag to consider.
		index : str | None (default)
			The column, if any, to set as index and sort before computing rolling statistics.


		Returns
		-------
		result : pandas.DataFrame
			The original dataframe extended with computed temporal features.
		"""
		ord_columns = [col for col in self._obj.columns if not self.is_categorical(col) and col not in exclude]
		df = self._obj.copy()
		if index:
			df = df.set_index(index)
		df = df.sort_index()
		dfs = [df.copy()]

		for lag in range(2, max_lag+2):
			rol_grp = df.rolling(window=lag, min_periods=1)
			lagged_mean_df = rol_grp.aggregate(nanmean).rename(columns={col: 'MEAN(%s, %d)' % (col, lag) for col in ord_columns})
			dfs += [lagged_mean_df.copy()]
			lagged_min_df = rol_grp.aggregate(nanmin).rename(columns={col: 'MIN(%s, %d)' % (col, lag) for col in ord_columns})
			dfs += [lagged_min_df.copy()]
			lagged_max_df = rol_grp.aggregate(nanmax).rename(columns={col: 'MAX(%s, %d)' % (col, lag) for col in ord_columns})
			dfs += [lagged_max_df.copy()]
			lagged_maxmmin_df = rol_grp.aggregate(nanmaxmmin).rename(columns={col: 'MAX-MIN(%s, %d)' % (col, lag) for col in ord_columns})
			dfs += [lagged_maxmmin_df.copy()]

		df = pd.concat(dfs, axis=1)

		return df


	def generate_features(self, entity=None, encoding_method='one_hot', index=None, max_lag=None, exclude=[], \
			means=None, quantiles=None, return_baselines=False, name='*'):
		"""
		Generate a wide range of candidate features to search from.

		We first compute entity features if needed. Then we extend the resulting dataframe with deviations of ordinal columns from row-wise aggregtes such as mean, median, 25th and 75th percentiles. Finally, we ordinally-encode the resulting dataframe and apply temporal transformations if required.


		Parameters
		----------
		entity : str
			The column mapping rows to entities.
		encoding_method : 'one_hot' (default) | 'binary'
			The encoding method to use for categorical variables.
		exclude : list
			A list of columns to exclude from feature transformations.
		max_lag : int | None
			The largest lag, if any, to consider for temporal features. Set to None to avoid temporal features.
		index : str | None (default)
			The column, if any, to set as index and sort before computing temporal features.
		means : pandas.DataFrame | None
			Which values, if any, to use as means for deviation features.
		quantiles : pandas.DataFrame | None
			Which values, if any, to use as 25th, 50th, 75th percentiles for deviation features.
		return_baselines : bool
			Whether to return which baselines have been used for deviation features.


		Returns
		-------
		result : pandas.DataFrame
			The original dataframe extended with computed temporal features.
		"""
		accessor = self
		if entity:
			# Entity features
			df = accessor.entity_features(entity, name=name)
			accessor = FeaturesAccessor(df)

		# Deviation features
		res = accessor.deviation_features(exclude=exclude, means=means, quantiles=quantiles, return_baselines=return_baselines)
		df = res[0] if return_baselines else res
		accessor = FeaturesAccessor(df)

		# Ordinally encode
		df = accessor.ordinally_encode(method=encoding_method)
		accessor = FeaturesAccessor(df)

		if max_lag:
			# Temporal/trend features
			df = accessor.temporal_features(exclude=exclude, max_lag=max_lag, index=index)

		if return_baselines:
			return df, res[1], res[2]
		else:
			return df



