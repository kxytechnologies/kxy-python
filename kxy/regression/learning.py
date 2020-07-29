
import numpy as np

from kxy.api import mutual_information_analysis, predict_copula_uniform
from kxy.api.core import prepare_data_for_mutual_info_analysis, prepare_test_data_for_prediction

from .pre_learning import regression_achievable_performance_analysis


class MaxEntRegressor(object):
	

	def fit(self, x_c, y, x_d=None, space="dual", categorical_encoding="two-split"):
		'''
		'''
		data = prepare_data_for_mutual_info_analysis(x_c, x_d, y, None, space=space, \
			non_monotonic_extension=True, categorical_encoding=categorical_encoding)
		output_indices = data['output_indices']
		corr = data['corr']
		batch_indices = data['batch_indices']
		self.corr_train = corr
		self.train_x_c = x_c
		self.train_x_d = x_d
		self.train_y_c = y
		self.space = space
		self.categorical_encoding = categorical_encoding
		mi_analysis = mutual_information_analysis(corr, output_indices, space=space, batch_indices=batch_indices)

		return self.achievable_performance


	@property	
	def achievable_performance(self):
		'''
		'''
		assert getattr(self, 'train_x_c', None) is not None or getattr(self, 'train_x_d', None) is not None,\
			'The model has not been fitted yet.'

		return regression_achievable_performance_analysis(self.train_x_c, self.train_y_c, x_d=self.train_x_d, \
			space=self.space)


	def predict(self, x_c, x_d=None):
		'''
		'''
		test_x_c = x_c
		test_x_d = x_d

		res = prepare_test_data_for_prediction(test_x_c, test_x_d, self.train_x_c, self.train_x_d, \
			self.train_y_c, None, non_monotonic_extension=True, categorical_encoding=self.categorical_encoding, \
			space=self.space)

		corr = res['corr']
		u_x_dict_list = res['u_x_dict_list']
		output_indices = res['output_indices']
		batch_indices  = res['batch_indices']

		predictions = predict_copula_uniform(u_x_dict_list, corr, output_indices, space=self.space, \
			batch_indices=batch_indices)

		posterior_mean_u_y = np.array(predictions['posterior_means']).astype(float)
		np.copyto(posterior_mean_u_y, 0.5, where=np.isnan(posterior_mean_u_y))
		posterior_std_u_y  = np.array(predictions['posterior_stds']).astype(float)
		np.copyto(posterior_std_u_y, np.sqrt(1./12.), where=np.isnan(posterior_std_u_y))

		posterior_mean_y   = np.percentile(self.train_y_c, 100.0*posterior_mean_u_y.flatten(), axis=0)

		return posterior_mean_u_y, posterior_std_u_y, posterior_mean_y



