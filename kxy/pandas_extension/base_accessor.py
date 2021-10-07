#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import numpy as np
from scipy.stats import norm
import pandas as pd


class BaseAccessor(object):
	"""
	Base class inheritated by our customs accessors.
	"""
	def __init__(self, pandas_obj):
		self._obj = pandas_obj


	def is_discrete(self, column):
		"""
		Determine whether the input column contains discrete (i.e as opposed to continuous) observations.
		"""
		if self.is_categorical(column):
			return True

		n = self._obj.shape[0]
		values, counts = np.unique(self._obj[column].values, return_counts=True)
		unique_n = len(values)

		if unique_n < 0.05*n:
			return True

		counts = np.array(list(sorted(counts)))
		if np.sum(counts[-10:]) > 0.8*n:
			return True

		return False


	def is_categorical(self, column):
		"""
		Determine whether the input column contains categorical (i.e. non-ordinal) observations.
		"""
		ret = (not np.can_cast(self._obj[column].values, float))

		return ret


	@property
	def is_too_large(self):
		return self._obj.memory_usage(index=False).sum()/(1024.0*1024.0*1024.0) > 1.5


	def describe(self,):
		for col in sorted(self._obj.columns):
			print('         ')
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
				m   = self._obj[col].min()
				M   = self._obj[col].max()
				mn  = self._obj[col].mean()
				q50 = self._obj[col].median()
				q25 = self._obj[col].quantile(0.25)
				q75 = self._obj[col].quantile(0.75)

				print('Type:   Continuous')
				print('Max:    %s' % ('%.1f' % M if M < 10. else '{:,}'.format(int(M))))
				print('p75:    %s' % ('%.1f' % q75 if q75 < 10. else '{:,}'.format(int(q75))))
				print('Mean:   %s' % ('%.1f' % mn if mn < 10. else '{:,}'.format(int(mn))))
				print('Median: %s' % ('%.1f' % q50 if q50 < 10. else '{:,}'.format(int(q50))))
				print('p25:    %s' % ('%.1f' % q25 if q25 < 10. else '{:,}'.format(int(q25))))
				print('Min:    %s' % ('%.1f' % m if m < 10. else '{:,}'.format(int(m))))


	def anonymize(self, columns_to_exclude=[]):
		"""
		Anonymize the dataframe in a manner that leaves all pre-learning and post-learning analyses (including data valuation, variable selection, model-driven improvability, data-driven improvability and model explanation) invariant.

		Any transformation on continuous variables that preserves ranks will not change our pre-learning and post-learning analyses. The same holds for any 1-to-1 transformation on categorical variables.

		This implementation replaces ordinal values (i.e. any column that can be cast as a float) with their within-column Gaussian score. For each non-ordinal column, we form the set of all possible values, we assign a unique integer index to each value in the set, and we systematically replace said value appearing in the dataframe by the hexadecimal code of its associated integer index. 

		For regression problems, accurate estimation of RMSE related metrics require the target column (and the prediction column for post-learning analyses) not to be anonymized.


		Parameters
		----------
		columns_to_exclude: list (optional)
			List of columns not to anonymize (e.g. target and prediction columns for regression problems).


		Returns
		-------
		result : pandas.DataFrame
			The result is a pandas.Dataframe with columns (where applicable):
		"""
		df = self._obj.copy()
		for col in df.columns:
			if col in columns_to_exclude:
				continue

			if df.kxy.is_categorical(col):
				unique_values = list(sorted(set(list(df[col].values))))
				mapping = {unique_values[i]: "0x{:03x}".format(i) for i in range(len(unique_values))}
				df[col] = df[col].apply(lambda x: mapping.get(x))
			else:
				# Note: Any monotonic transformation applied to any continuous column would work.
				# The gaussian scoring below makes no assumption on marginals whatsoever. 
				x = df[col].values
				x = x - np.nanmean(x)
				s = np.nanstd(x)
				if s > 0.0:
					x = x/s
					x = norm.cdf(x)
				df[col] = np.around(x.copy(), 3)


		return df



	def __hash__(self):
		return hashlib.sha256(self._obj.to_string().encode()).hexdigest()




