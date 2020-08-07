
import numpy as np

from kxy.api import mutual_information_analysis, predict_copula_uniform
from kxy.api.core import prepare_data_for_mutual_info_analysis, prepare_test_data_for_prediction

from .pre_learning import regression_achievable_performance_analysis


class MaxEntRegressor(object):
	'''
	Regression model based on the maximum-entropy principle.
	'''

	def fit(self, x_c, y, x_d=None, space="dual", categorical_encoding="two-split"):
		"""
		Solves the maximum-entropy problem on the API, in the background. 

		The resulting joint copula-uniform distribution :math:`(u_y, u_x)` is the cornerstone of inference.


		Parameters
		----------
		x_c : (n,) or (n, d) np.array
			Continuous inputs.
		x_d : (n,) or (n, d) np.array or None (default), optional
			Discrete inputs.
		y : (n,) np.array
			Labels.
		space : str, 'primal' | 'dual'
			The space in which the maximum entropy problem is solved. 
			When :code:`space='primal'`, the maximum entropy problem is solved in the original observation space, under Pearson covariance constraints, leading to the Gaussian copula.
			When :code:`space='dual'`, the maximum entropy problem is solved in the copula-uniform dual space, under Spearman rank correlation constraints.
		categorical_encoding : str, 'one-hot' | 'two-split' (default)
			The encoding method to use to represent categorical variables. 
			See :ref:`kxy.api.core.utils.one_hot_encoding <one-hot-encoding>` and :ref:`kxy.api.core.utils.two_split_encoding <two-split-encoding>`.


		Returns
		-------
		a : pandas.DataFrame

			Dataframe with columns:

			* :code:`'Achievable R^2'`: The highest :math:`R^2` that can be achieved by a regression model using provided inputs.
			* :code:`'Achievable Log-Likelihood Per Sample'`: The highest true log-likelihood per sample that can be achieved by a regression model using provided inputs.
		"""
		data = prepare_data_for_mutual_info_analysis(x_c, x_d, y, None, space=space, \
			non_monotonic_extension=True, categorical_encoding=categorical_encoding)
		output_indices = data['output_indices']
		corr = data['corr']
		batch_indices = data['batch_indices']
		self.corr_train = corr
		self.train_x_c = x_c.astype(float)
		self.train_x_d = x_d.astype(str) if x_d is not None else None
		self.train_y_c = y.astype(float).flatten()

		self.space = space
		self.categorical_encoding = categorical_encoding
		mi_analysis = mutual_information_analysis(corr, output_indices, space=space, batch_indices=batch_indices)

		return self.achievable_performance


	@property	
	def achievable_performance(self):
		"""
		Dataframe containing the highest performance that can be achieved. Requires the model to be fitted first.

		.. seealso::

			:ref:`kxy.regression.pre_learning.regression_achievable_performance_analysis <regression-achievable-performance-analysis>`

		"""
		assert getattr(self, 'train_x_c', None) is not None or getattr(self, 'train_x_d', None) is not None,\
			'The model has not been fitted yet.'

		return regression_achievable_performance_analysis(self.train_x_c, self.train_y_c, x_d=self.train_x_d, \
			space=self.space)


	def predict(self, x_c, x_d=None):
		"""
		Calculates the posterior mean and posterior standard deviation of the copula-uniform representation of the output

		.. math::
			E(u_y|x= *) \\text{ and } \\sqrt{Var \\left( u_y \\vert x= *  \\right)}

		under the maximum-entropy distribution for the copula-uniform representations :math:`(u_y, u_x)`, and infer predicted outputs.

		Missing inputs are handled gracefully, and the posterior distribution is based on provided inputs.


		Parameters
		----------
		x_c : (n,d) np.array
			Test continuous inputs. Missing inputs, if any, should be represented as np.nan or None.
		x_d : (n, d) np.array or None (default), optional
			Test discrete inputs. Missing inputs, if any, should be represented as np.nan or None.


		Returns
		-------
		 : dict

			Dictionary with keys:

				* :code:`posterior_mean_u_y`: The ndarray of the posterior mean of outputs encodings.
				* :code:`posterior_std_u_y`: The ndarray of the posterior std of outputs encodings.
				* :code:`predicted_labels`: The ndarray of predicted outputs corresponding to `posterior_mean_u_y`.
		"""
		test_x_c = x_c
		test_x_d = x_d

		res = prepare_test_data_for_prediction(test_x_c, test_x_d, self.train_x_c, self.train_x_d, \
			self.train_y_c, None, non_monotonic_extension=True, categorical_encoding=self.categorical_encoding, \
			space=self.space)

		corr = res['corr']
		u_x_dict_list  = res['u_x_dict_list']
		output_indices = res['output_indices']
		batch_indices  = res['batch_indices']
		problem_type   = res['problem_type']

		predictions = predict_copula_uniform(u_x_dict_list, corr, output_indices, space=self.space, \
			batch_indices=batch_indices, problem_type=problem_type)

		posterior_mean_u_y = np.array(predictions['posterior_means']).astype(float)
		np.copyto(posterior_mean_u_y, 0.5, where=np.isnan(posterior_mean_u_y))
		posterior_std_u_y  = np.array(predictions['posterior_stds']).astype(float)
		np.copyto(posterior_std_u_y, np.sqrt(1./12.), where=np.isnan(posterior_std_u_y))

		predicted_labels   = np.percentile(self.train_y_c, 100.0*posterior_mean_u_y.flatten(), axis=0)

		return {'posterior_mean_u_y': posterior_mean_u_y, 'posterior_std_u_y': posterior_std_u_y, 'predicted_labels': predicted_labels}






