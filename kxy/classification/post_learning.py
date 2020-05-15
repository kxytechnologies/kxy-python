#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kxy.api.core import least_mixed_mutual_information, discrete_mutual_information

def classification_suboptimality(yp, y, x_c, x_d=None, space='dual'):
	"""
	.. _classification-suboptimality:
	Quantifies the extent to which a (multinomial) classifier can be improved without requiring additional features.

	.. note::

		The conditional entropy :math:`h \\left( y \\vert x \\right)` represents the amount of information 
		about :math:`y` that cannot be explained by :math:`x`. If we denote :math:`f(x)` the label 
		predicted by our classifier, :math:`h \\left( y \\vert f(x) \\right)` represents the amount
		of information about :math:`y` that the classifier is not able to explain using :math:`x`.

		A natural metric for how suboptimal a particular classifier is can therefore be defined as the 
		difference between the amount of information about :math:`y` that cannot be explained by 
		:math:`f(x)` and the amount of information about :math:`y` that cannot be explained by :math:`x`

		.. math::

			\\text{SO}(f; x) &= h \\left( y \\vert f(x) \\right) - h \\left( y \\vert x \\right) \\

										 :&= I\\left(y, x \\right) - I\\left(y, f(x) \\right) \\

										  &\\geq 0.

		This classification suboptimality metric is 0 if and only if :math:`f(x)` fully captures any information about :math:`y`
		that is contained in :math:`x`. When 

		.. math::

			\\text{SO}(f; x) > 0 

		on the other hand, there exists a classification model using :math:`x` as features that can better predict :math:`y`. The larger 
		:math:`\\text{SO}(f; x)`, the more the classification model is suboptimal and can be improved.


	Parameters
	----------
	x_c : (n, d) np.array
		n i.i.d. draws from the continuous features generating distribution.
	x_d : (n,) np.array
		n i.i.d. draws from the discrete features generating distribution, if any, sampled jointly with x_c.
	y : (n,) np.array
		n i.i.d. draws from the (discrete) labels generating distribution, sampled
		jointly with x_c and x_d.
	yp : (n,) np.array
		Discrete predictions of y from our classifier.


	Returns
	-------
	d : float
		The classifier's suboptimality measure.


	.. seealso:: 

		:ref:`kxy.api.core.mutual_information.least_mixed_mutual_information <least-mixed-mutual-information>`
	"""
	return least_mixed_mutual_information(x_c, y, x_d=x_d, space=space)-discrete_mutual_information(y, yp)


