#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from .base_accessor import BaseAccessor
from .features_utils import nanskew, nankurtosis, q25, q75, mode, modefreq, nextmode, nextmodefreq, lastmode, lastmodefreq, \
	nanmin, nanmax, nanmean, nanstd, nanmedian, nanmaxmmin, nansum, nanskewabs, nankurtosisabs, q25abs, q75abs,  \
	nanminabs, nanmaxabs, nanmeanabs, nanstdabs, nanmedianabs, nanmaxmminabs, nansumabs, n_unique


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
		"""
		Binary-encode a specific column.
		"""
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


	def entity_features(self, entity, exclude=[], entity_name='*', filter_target=None, filter_target_gt=None, filter_target_lt=None, \
			include_filter_target=False):
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
		filter_target : str | None
			When specified, this is a column based on which we need to restrict the dataframe before generating features.
		filter_target_gt : str | None
			When specified, only rows with :code:`filter_target` greater than :code:`filter_target_gt` will be considered for feature generation.
		filter_target_lt : str | None
			When specified, only rows with :code:`filter_target` smaller than :code:`filter_target_gt` will be considered for feature generation.
		include_filter_target : bool
			Whether to use :code:`filter_target` for features generation.


		Returns
		-------
		result : pandas.DataFrame
			The dataframe of features.
		"""
		assert entity in self._obj.columns, 'The entity %s should be valid column' % entity
		_columns = self._obj.columns
		if filter_target and not include_filter_target:
			_columns = [_ for _ in _columns if _ != filter_target]

		cat_columns = list(set([col for col in _columns if self.is_categorical(col) and col != entity and col not in exclude]))
		ord_columns = list(set([col for col in _columns if not self.is_categorical(col) and col != entity and col not in exclude]))
		if ord_columns:
			mix_sgn_columns = [col for col in ord_columns if self._obj[col].lt(0).any() and self._obj[col].gt(0).any()]

		columns = cat_columns + ord_columns

		dfs = []
		agg = {}
		# Aggregation of categorical variables
		if cat_columns:
			# Most frequent value and it's frequency of occurence
			agg.update({'%s__%s.MODE()' % (entity, col): (col, mode) for col in cat_columns})
			agg.update({'%s__%s.MODEFREQ()' % (entity, col): (col, modefreq) for col in cat_columns})

			# Second most frequent value and it's frequency of occurence
			agg.update({'%s__%s.NEXTMODE()' % (entity, col): (col, nextmode) for col in cat_columns})
			agg.update({'%s__%s.NEXTMODEFREQ()' % (entity, col): (col, nextmodefreq) for col in cat_columns})

			# Least frequent value and it's frequency of occurence
			agg.update({'%s__%s.LASTMODE()' % (entity, col): (col, lastmode) for col in cat_columns})
			agg.update({'%s__%s.LASTMODEFREQ()' % (entity, col): (col, lastmodefreq) for col in cat_columns})

			# Number of unique values
			agg.update({'%s__%s.NUM_UNIQUE()' % (entity, col): (col, n_unique) for col in cat_columns})


		# Aggregation of ordinal variables
		if ord_columns:
			agg.update({'%s__%s.SUM()' % (entity, col): (col, nansum) for col in ord_columns})
			agg.update({'%s__%s.MEAN()' % (entity, col): (col, nanmean) for col in ord_columns})
			agg.update({'%s__%s.STD()' % (entity, col): (col, nanstd) for col in ord_columns})
			agg.update({'%s__%s.MEDIAN()' % (entity, col): (col, nanmedian) for col in ord_columns})
			agg.update({'%s__%s.SKEW()' % (entity, col): (col, nanskew) for col in ord_columns})
			agg.update({'%s__%s.KURT()' % (entity, col): (col, nankurtosis) for col in ord_columns})
			agg.update({'%s__%s.Q25()' % (entity, col): (col, q25) for col in ord_columns})
			agg.update({'%s__%s.Q75()' % (entity, col): (col, q75) for col in ord_columns})
			agg.update({'%s__%s.MIN()' % (entity, col): (col, nanmin) for col in ord_columns})
			agg.update({'%s__%s.MAX()' % (entity, col): (col, nanmax) for col in ord_columns})
			agg.update({'%s__%s.MAX-MIN()' % (entity, col): (col, nanmaxmmin) for col in ord_columns})

			if mix_sgn_columns:
				agg.update({'%s__%s.ABS().SUM()' % (entity, col): (col, nansumabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().MEAN()' % (entity, col): (col, nanmeanabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().STD()' % (entity, col): (col, nanstdabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().MEDIAN()' % (entity, col): (col, nanmedianabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().SKEW()' % (entity, col): (col, nanskewabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().KURT()' % (entity, col): (col, nankurtosisabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().Q25()' % (entity, col): (col, q25abs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().Q75()' % (entity, col): (col, q75abs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().MIN()' % (entity, col): (col, nanminabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().MAX()' % (entity, col): (col, nanmaxabs) for col in mix_sgn_columns})
				agg.update({'%s__%s.ABS().MAX-MIN()' % (entity, col): (col, nanmaxmminabs) for col in mix_sgn_columns})
		
		# Number of rows per entity
		agg.update({'COUNT(%s)' % entity_name: (columns[0], 'count')})

		# Filter if necessary
		obj = self._obj.copy()
		if filter_target:
			assert filter_target in obj.columns, 'The filter column should be a valid column'
			if filter_target_gt:
				obj = obj[obj[filter_target] > filter_target_gt]

			if filter_target_lt:
				obj = obj[obj[filter_target] < filter_target_lt]

		# Features
		entity_grp = obj.groupby(entity)
		df = entity_grp.agg(**agg)

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
				df['%s.ABS(* - MEAN(*))' % col] = np.abs(df[col]-means.loc[col])
				df['%s.ABS(* - MEDIAN(*))' % col] = np.abs(df[col]-quantiles.loc[0.5][col])
				df['%s.ABS(* - Q25(*))' % col] = np.abs(df[col]-quantiles.loc[0.25][col])
				df['%s.ABS(* - Q75(*))' % col] = np.abs(df[col]-quantiles.loc[0.75][col])

		if return_baselines:
			return df, means, quantiles
		else:
			return df


	def temporal_features(self, max_lag=10, exclude=[], index=None, groupby=None, sort_by=None):
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
		groupby : str | None
			If provided, we will use this column to perform a groupby before temporal aggregation.
		sort_by : str | list | None
			Columns, if any, we need to sort the dataframe by, prior to rolling.


		Returns
		-------
		result : pandas.DataFrame
			The original dataframe extended with computed temporal features.
		"""
		if index:
			exclude += [index]

		if groupby:
			exclude += [groupby]

		ord_columns = [col for col in self._obj.columns if not self.is_categorical(col) and col not in exclude]
		df = self._obj.copy()
		if index and (df.index.name != index):
			df = df.set_index(index)

		if sort_by:
			df = df.sort_values(by=sort_by)

		dfs = [df.copy()]

		if groupby:
			pandarallel.initialize()
			def apply_func(s):
				''' '''
				lag = max_lag+1
				col_map = {}
				col_map.update({'%s_nanmean' % col: '%s.GROUPBY(%s).LAST(%d).MEAN()' % (col, groupby, lag) for col in ord_columns})
				col_map.update({'%s_nanmin' % col: '%s.GROUPBY(%s).LAST(%d).MIN()' % (col, groupby, lag) for col in ord_columns})
				col_map.update({'%s_nanmax' % col: '%s.GROUPBY(%s).LAST(%d).MAX()' % (col, groupby, lag) for col in ord_columns})
				col_map.update({'%s_nanmaxmmin' % col: '%s.GROUPBY(%s).LAST(%d).MAX-MIN()' % (col, groupby, lag) for col in ord_columns})
				cols = [_ for _ in col_map.values()]
				res = s.rolling(lag, min_periods=1).aggregate([nanmean, nanmin, nanmax, nanmaxmmin])
				res.columns = res.columns.map('_'.join).to_series().map(col_map)
				return res[cols]

			feat_df = df.groupby(groupby, sort=False).parallel_apply(apply_func)
			feat_df.reset_index(inplace=True)
			if df.index.name:
				feat_df.set_index(df.index.name, inplace=True)
			else:
				feat_df.set_index('index', inplace=True)
				feat_df.index.name = df.index.name
			dfs += [feat_df]
			df = pd.concat(dfs, axis=1)

		else:
			lag = max_lag+1
			col_map = {}
			col_map.update({'%s_nanmean' % col: '%s.LAST(%d).MEAN()' % (col, lag) for col in ord_columns})
			col_map.update({'%s_nanmin' % col: '%s.LAST(%d).MIN()' % (col, lag) for col in ord_columns})
			col_map.update({'%s_nanmax' % col: '%s.LAST(%d).MAX()' % (col, lag) for col in ord_columns})
			col_map.update({'%s_nanmaxmmin' % col: '%s.LAST(%d).MAX-MIN()' % (col, lag) for col in ord_columns})
			cols = [_ for _ in col_map.values()]
			feat_df = df.rolling(lag, min_periods=1).aggregate([nanmean, nanmin, nanmax, nanmaxmmin])
			feat_df.columns = feat_df.columns.map('_'.join).to_series().map(col_map)
			dfs += [feat_df[cols]]
			df = pd.concat(dfs, axis=1)

		return df


	def process_time_columns(self, columns):
		"""
		Extract features from timestamp columns such as: Month, Day, Day of Week, Hour, AM/PM.


		Parameters
		----------
		columns : list
			The list of columns that should be interprated as UTC epoch timestamps.


		Returns
		-------
		result : pandas.DataFrame
			The features dataframe (does not include the original dataframe)
		"""
		res = pd.DataFrame(index=self._obj.index)
		for col in columns:
			times = pd.to_datetime(self._obj[col], unit="s", infer_datetime_format=True)
			res['%s.HOUR()' % col] = times.dt.hour
			res['%s.DAYOFWEEK()' % col] = times.dt.dayofweek
			res['%s.DAY()' % col] = times.dt.day
			res['%s.MONTH()' % col] = times.dt.month
			res['%s.YEAR()' % col] = times.dt.year

		return res


	def generate_features(self, entity=None, encoding_method='one_hot', index=None, max_lag=None, exclude=[], \
			means=None, quantiles=None, return_baselines=False, entity_name='*', filter_target=None, \
			filter_target_gt=None, filter_target_lt=None, include_filter_target=False, fill_na=False, \
			temporal_groupby=None, temporal_sort_by=None, time_columns=None):
		"""
		Generate a wide range of candidate features to search from.

		We first compute entity features if needed. 

		Then we extend the resulting dataframe with deviations of ordinal columns from row-wise aggregtes such as mean, median, 25th and 75th percentiles. 

		Finally, we ordinally-encode the resulting dataframe and apply temporal transformations if required.


		Parameters
		----------
		entity : str
			The column mapping rows to entities.
		filter_target : str | None
			When specified, this is a column based on which we need to restrict the dataframe before generating entity features.
		filter_target_gt : str | None
			When specified, only rows with :code:`filter_target` greater than :code:`filter_target_gt` will be considered for entity feature generation.
		filter_target_lt : str | None
			When specified, only rows with :code:`filter_target` smaller than :code:`filter_target_gt` will be considered for entity feature generation.
		include_filter_target : bool
			Whether to use :code:`filter_target` for features generation.
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
		temporal_groupby : str | None
			If provided, we will use this column to perform a groupby before temporal aggregation.
		temporal_sort_by : str | None
			If provided, we will use this column to sort the dataframe before rolling when computing temporal features.
		time_columns : list | None
			The list of columns that correspond to times and from which we should extract features such as hour, day of week etc.


		Returns
		-------
		result : pandas.DataFrame
			The original dataframe extended with computed temporal features.
		"""
		accessor = self
		if entity:
			# Entity features
			df = accessor.entity_features(entity, entity_name=entity_name, filter_target=filter_target, \
				filter_target_gt=filter_target_gt, filter_target_lt=filter_target_lt, \
				include_filter_target=include_filter_target)
			accessor = FeaturesAccessor(df)

		if index:
			exclude += [index]
			
		if temporal_groupby:
			exclude += [temporal_groupby]

		if time_columns:
			# Extract information from time columns such as hour, day of week etc.
			df = accessor.process_time_columns(time_columns)
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
			df = accessor.temporal_features(exclude=exclude, max_lag=max_lag, index=index, groupby=temporal_groupby, sort_by=temporal_sort_by)

		if fill_na:
			df = df.fillna(df.median(skipna=True))

		if return_baselines:
			return df, res[1], res[2]
		else:
			return df



