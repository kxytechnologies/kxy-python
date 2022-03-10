#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow Implementation of MIND ([1]) under Spearman rank correlation constraints.

[1] Kom Samo, Y. (2021). Inductive Mutual Information Estimation: A Convex Maximum-Entropy Copula Approach . <i>Proceedings of The 24th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 130:2242-2250 Available from https://proceedings.mlr.press/v130/kom-samo21a.html.
"""
import logging
logging.basicConfig(level=logging.INFO)

from multiprocessing import Pool, cpu_count
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)

from tensorflow.keras import Model
from tensorflow.keras.backend import pow as tf_pow
from tensorflow.keras.backend import cast, clip
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import Dense, Lambda, concatenate, Dot, Dropout, Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.python.ops import math_ops

rankdata = lambda x: 1.+np.argsort(np.argsort(x, axis=0), axis=0)



class CopulaBatchGenerator(Sequence):
	''' 
	Random batch generator.
	'''
	def __init__(self, z, batch_size=1000, steps_per_epoch=100):
		self.batch_size = batch_size
		self.d = z.shape[1]
		self.n = z.shape[0]
		self.z = z
		self.steps_per_epoch = steps_per_epoch
		self.emp_u = rankdata(self.z)/(self.n + 1.)
		self.emp_u[np.isnan(self.z)] = 0.5

		if self.n < 200*self.d:
			dn = 200*self.d - self.n
			selected_rows = np.random.choice(self.n, dn, replace=True)
			emp_u = self.emp_u[selected_rows, :].copy()
			scale = 1./(100.*self.n)
			emp_u += (scale*np.random.rand(*emp_u.shape) - 0.5*scale)
			self.emp_u = np.concatenate([self.emp_u, emp_u], axis=0)
			self.n = self.emp_u.shape[0]

		self.batch_selector = np.random.choice(self.n, self.batch_size*self.steps_per_epoch, replace=True)
		self.batch_selector = self.batch_selector.reshape((self.steps_per_epoch, self.batch_size))


	def getitem_ndarray(self, idx):
		''' '''
		i = idx % self.steps_per_epoch
		selected_rows = self.batch_selector[i]
		emp_u_ = self.emp_u[selected_rows, :]
		z_p = emp_u_.copy()
		z_q = np.random.rand(*emp_u_.shape)

		z = np.empty((self.batch_size, self.d, 2))
		z[:, :, 0] = z_p
		z[:, :, 1] = z_q
		batch_x = z
		batch_y = np.ones((self.batch_size, 2))  # Not used  
		return batch_x, batch_y


	def __getitem__(self, idx):
		''' '''
		batch_x, batch_y = self.getitem_ndarray(idx)
		return tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)


	def __len__(self):
		return self.steps_per_epoch

		

class InitializableDense(Layer):
	''' 
	'''
	def __init__(self, units, initial_w=None, initial_b=None, bias=False):
		'''
		initial_w should be None or a 2D numpy array.
		initial_b should be None or a 1D numpy array.
		'''
		super(InitializableDense, self).__init__()
		self.units = units
		self.with_bias = bias
		self.w_initializer = 'zeros' if initial_w is None else tf.constant_initializer(initial_w)

		if self.with_bias:
			self.b_initializer = 'zeros' if initial_b is None else tf.constant_initializer(initial_b)


	def build(self, input_shape):
		''' '''
		self.w = self.add_weight(shape=(input_shape[-1], self.units), \
			initializer=self.w_initializer, trainable=True, name='quad_w')

		if self.with_bias:
			self.b = self.add_weight(shape=(self.units,), \
				initializer=self.b_initializer, trainable=True, name='quad_b')


	def call(self, inputs):
		''' '''
		return tf.matmul(inputs, self.w)+self.b if self.with_bias else tf.matmul(inputs, self.w)




class CopulaModel(Model):
	'''
	Maximum-entropy copula under (possibly sparse) Spearman rank correlation constraints.
	'''
	def __init__(self, d, subsets=[]):
		super(CopulaModel, self).__init__()
		self.d = d
		if subsets == []:
			subsets = [[_ for _ in range(d)]]

		self.subsets = subsets
		self.n_subsets = len(self.subsets)
		self.p_samples = Lambda(lambda x: x[:,:,0])
		self.q_samples = Lambda(lambda x: x[:,:,1])

		self.fx_non_mon_layer_1s = [Dense(3, activation=tf.nn.relu) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_2s = [Dense(5, activation=tf.nn.relu) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_3s = [Dense(3, activation=tf.nn.relu) for _ in range(self.n_subsets)]
		self.fx_non_mon_layer_4s = [Dense(1) for _ in range(self.n_subsets)]

		eff_ds = [len(subset)+1 for subset in self.subsets]
		self.spears = [InitializableDense(eff_d) for eff_d in eff_ds]
		self.dots = [Dot(1) for _ in range(self.n_subsets)]

		# Mixing layers
		self.mixing_layer1 = Dense(5, activation=tf.nn.relu)
		self.mixing_layer2 = Dense(5, activation=tf.nn.relu)
		self.mixing_layer3 = Dense(1)


	def subset_statistics(self, u, i):
		'''
		Statistics function for the i-th subset of variables.
		''' 
		n = tf.shape(u)[0]
		res = tf.zeros(shape=[n, 1], dtype=tf.float64)
		ui = tf.gather(u, self.subsets[i], axis=1)

		# Constraints beyond quadratic
		fui = self.fx_non_mon_layer_1s[i](ui)
		fui = self.fx_non_mon_layer_2s[i](fui)
		fui = self.fx_non_mon_layer_3s[i](fui)
		fui = self.fx_non_mon_layer_4s[i](fui)
		ui = concatenate([ui, fui], axis=1)
	
		# Spearman terms
		spearman_term = self.spears[i](ui)
		spearman_term = self.dots[i]([spearman_term, ui])
		res = tf.add(res, spearman_term)
			
		return res


	def statistics(self, u):
		'''
		Statistics function.
		''' 
		if self.n_subsets > 1:
			ts = [self.subset_statistics(u, i) for i in range(self.n_subsets)]
			t = concatenate(ts, axis=1)
			t = self.mixing_layer1(t)
			t = self.mixing_layer2(t)
			t = self.mixing_layer3(t)
		else:
			t = self.subset_statistics(u, 0)

		return t


	def call(self, inputs):
		''' '''        
		p_samples = self.p_samples(inputs)
		t_p = self.statistics(p_samples)

		q_samples = self.q_samples(inputs)
		t_q = self.statistics(q_samples)
		
		t = concatenate([t_p, t_q], axis=1)
		t = clip(t, -100., 100.)
		
		return t


	def copula(self, inputs):
		''' '''
		u = tf.constant(inputs)
		c = math_ops.exp(self.statistics(u))
		return c.numpy()/c.numpy().mean()




class MINDLoss(Loss):  
	'''
	Loss function.
	'''
	def call(self, y_true, y_pred):
		''' '''
		p_samples = y_pred[:, 0]
		q_samples = y_pred[:, 1]
		mi = -tf.reduce_mean(p_samples) + math_ops.log(tf.reduce_mean(math_ops.exp(q_samples)))
		return mi



class CopulaLearner(object):
	'''
	Maximum-entropy learner.
	'''
	def __init__(self, d, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, \
			name='Adam', lr=0.01, subsets=[]):
		self.d = d
		self.model = CopulaModel(self.d, subsets=subsets)
		self.opt = Adam(beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, \
			name=name, lr=lr)
		self.loss = MINDLoss()
		self.model.compile(optimizer=self.opt, loss=self.loss)
		self.copula_entropy = None


	def fit(self, z, batch_size=10000, steps_per_epoch=1000):
		''' '''
		epochs = 20
		z_gen = CopulaBatchGenerator(z, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
		self.model.fit(z_gen, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, \
			callbacks=[EarlyStopping(patience=3, monitor='loss'), TerminateOnNaN()])
		self.copula_entropy = self.model.evaluate(z_gen)




def copula_entropy(z, subsets=[]):
	'''
	Estimate the entropy of the copula distribution of a d-dimensional random vector using MIND ([1]) with Spearman rank correlation constraints.


	Parameters
	----------
	z : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.


	Returns
	-------
	ent : float
		The estimated copula entropy.
	'''
	if len(z.shape)==1 or z.shape[1]==1:
		return 0.0

	d = z.shape[1]
	cl = CopulaLearner(d, subsets=subsets)
	cl.fit(z)
	ent = min(cl.copula_entropy, 0.0)

	return ent



def mutual_information(y, x):
	'''
	Estimate the mutual information between two random vectors using MIND ([1]) with Spearman rank correlation constraints.


	Parameters
	----------
	y : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.
	x : np.array
		Vector whose rows are samples from the d-dimensional random vector and columns its coordinates.


	Returns
	-------
	mi : float
		The estimated mutual information.
	'''
	y = y[:, None] if len(y.shape)==1 else y
	x = x[:, None] if len(x.shape)==1 else x
	z = np.concatenate([y, x], axis=1)
	huy = copula_entropy(y)
	hux = copula_entropy(x)
	huz = copula_entropy(z)
	mi = max(huy+hux-huz, 0.0)

	return mi


def run_d_dimensional_gaussian_experiment(d, rho, n=1000):
	'''
	'''
	# Cholesky decomposition of corr = np.array([[1., rho], [rho, 1.]])
	L = np.array([[1., 0.], [rho, np.sqrt(1.-rho*rho)]])
	y = np.empty((n, d))
	x = np.empty((n, d))
	for i in range(d):
		u = np.random.randn(n, 2)
		z = np.dot(L, u.T).T
		y[:, i] = z[:, 0].copy()
		x[:, i] = z[:, 1].copy()

	estimated_mi = mutual_information(y, x)
	true_mi = -d*0.5*np.log(1.-rho*rho)

	return estimated_mi, true_mi



if __name__ == '__main__':
	rho = 0.95
	d = 20
	estimated_mi, true_mi = run_d_dimensional_gaussian_experiment(d, rho)
	print('%dd Gaussian Mutual Information: Estimated %.4f, True (theoretical) %.4f' % (\
		d, estimated_mi, true_mi))



