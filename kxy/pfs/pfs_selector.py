#!/usr/bin/env python
# -*- coding: utf-8 -*-
from time import time
import logging
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

from kxy.misc.tf import PFSLearner, PFSOneShotLearner




def learn_principal_direction(y, x, ox=None, oy=None, epochs=20):
	"""
	Learn the i-th principal feature when using :math:`x` to predict :math:`y`.

	Parameters
	----------
	x : np.array
		2D array of shape :math:`(n, d)` containing original features.
	y : np.array
		Array of shape :math:`(n)` or :math:`(n, 1)` containing targets.

	Returns
	-------
	w : np.array
		The first principal direction.
	mi: float
		The mutual information :math:`I(y; w_i^Tx, \\dots, w_1^Tx)`.
	"""
	dx = 1 if len(x.shape) == 1 else x.shape[1]
	dy = 1 if len(y.shape) == 1 else y.shape[1]
	dox = 0 if ox is None else 1 if len(ox.shape) == 1 else ox.shape[1]
	doy = 0 if oy is None else 1 if len(oy.shape) == 1 else oy.shape[1]

	learner = PFSLearner(dx, dy=dy, dox=dox, doy=doy)
	learner.fit(x, y, ox=ox, oy=oy, epochs=epochs)

	mi = learner.mutual_information
	w = learner.feature_direction
	ox = learner.fx
	oy = learner.gy

	return w, mi, ox, oy



def learn_principal_directions_one_shot(y, x, p, epochs=20):
	"""
	Jointly learn p principal features.

	Parameters
	----------
	x : np.array
		2D array of shape :math:`(n, d)` containing original features.
	y : np.array
		Array of shape :math:`(n)` or :math:`(n, 1)` containing targets.
	p : int
		The number of principal features to learn.

	Returns
	-------
	w : np.array
		The matrix whose rows are the p principal directions.
	"""
	dx = 1 if len(x.shape) == 1 else x.shape[1]
	learner = PFSOneShotLearner(dx, p=p)
	learner.fit(x, y, epochs=epochs)
	w = learner.feature_directions
	mi = learner.mutual_information

	return w, mi




class PFS(object):
	"""
	Principal Feature Selection.
	"""
	def fit(self, x, y, p=None, mi_tolerance=0.0001, max_duration=None, epochs=20):
		"""
		Perform Principal Feature Selection using :math:`x` to predict :math:`y`.

		Specifically, we are looking for a :math:`p x d` matrix :math:`W` whose :math:`p` rows are learned sequentially such that :math:`z := Wx` is a great feature vector for predicting :math:`y`.

		Each row of :math:`W` is normal: :math:`||w_i||=1`, and the corresponding principal feature, namely :math:`w_i^Tx`, points in the same direction as :math:`y` (i.e. :math:`Cov(y, w_i^Tx) > 0`).

		The first row :math:`w_1` is learned so as to maximize the mutual information :math:`I(y; x^Tw_1)`.

		The second row :math:`w_2` is learned so as to maximize the conditional mutual information :math:`I(y; x^Tw_2 | x^Tw_1)`.	

		More generally, the :math:`(i+1)`-th row :math:`w_{i+1}` is learned so as to maximize the conditional mutual information :math:`I(y; x^Tw_{i+1} | [x^Tw_1, ..., x^Tw_i])`.


		Parameters
		----------
		x : np.array
			2D array of shape :math:`(n, d)` containing original features.
		y : np.array
			Array of shape :math:`(n)` or :math:`(n, 1)` containing targets.
		p : int | None (default)
			The number of features to select. When :code:`None` (the default) we stop when the estimated mutual information smaller than the mutual information tolerance parameter, or when we have exceeded the maximum duration. A value of :code:`p` that is not :code:`None` triggers one-shot PFS.
		mi_tolerance: float
			The smallest estimated mutual information required to keep looking for new feature directions.
		max_duration : float | None (default)
			The maximum amount of time (in second) to allocate to PFS.


		Returns
		-------
		W : np.array
			2D array whose rows are directions to use to compute principal features: :math:`z = Wx`.
		"""
		if max_duration:
			start_time = time()

		rows = []
		d = 1 if len(x.shape) == 1 else x.shape[1]
		if p is None:
			t = y.flatten().copy()
			old_mi = 0.0
			ox = None
			oy = None
			for i in range(d):
				w, mi, ox, oy = learn_principal_direction(t, x, ox=ox, oy=oy, epochs=epochs)

				if mi-old_mi < mi_tolerance:
					logging.info('The mutual information %.4f after %d round has not increase by more than %.4f: stopping.' % (
						mi, i+1, mi_tolerance))
					break
				else:
					logging.info('The mutual information has increased from %.4f to %.4f after %d rounds.' % (old_mi, mi, i+1))
					rows += [w.copy()]

				if max_duration:
					if time()-start_time > max_duration:
						logging.info('PFS has exceeded the configured maximum duration: exiting.')
						break

				old_mi = mi

			if rows == []:
				logging.warning('The only principal feature selected is not informative about the target: I(y; w^Tx)=%.4f' % mi)
				rows += [w.copy()]

			self.feature_directions = np.array(rows)
			self.mutual_information = old_mi
		else:
			# Learn all p principal features jointly.
			feature_directions, mi = learn_principal_directions_one_shot(y, x, p, epochs=epochs)
			self.feature_directions = feature_directions
			self.mutual_information = mi

		return self.feature_directions



class PCA(object):
	"""
	Principal Component Analysis.
	"""
	def __init__(self, energy_loss_frac=0.05):
		self.energy_loss_frac = energy_loss_frac


	def fit(self, x, _, max_duration=None, p=None):
		"""
		"""
		cov_x = np.cov(x.T) # Columns in x should represent variables and rows observations.
		u, d, v = np.linalg.svd(cov_x)
		cum_energy = np.cumsum(d)
		energy = cum_energy[-1]
		p = len([_ for _ in cum_energy if _ <= (1.-self.energy_loss_frac)*energy])

		self.feature_directions = u[:, :p].T

		return self.feature_directions




