#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .base_accessor import BaseAccessor

@pd.api.extensions.register_dataframe_accessor("kxy_features")
class FeaturesAccessor(BaseAccessor):
	"""
	Extension of the pandas.DataFrame class with various feature engineering functionalities.

	This class defines the :code:`kxy_features` `pandas accessor <https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

	All its methods defined are accessible from any DataFrame instance as :code:`df.kxy_features.<method_name>`, so long as the :code:`kxy` python package is imported alongside :code:`pandas`. 
	"""
	def ordinally_encode(self, target_column=None, method='one_hot'):
		'''
		Encode categorical (non-numeric) data.
		'''
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






