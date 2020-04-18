#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kxy.api.core import least_continuous_mutual_information


def regression_suboptimality(yp, y, x):
	"""
	.. _regression-suboptimality:
	Quantifies the extent to which a regression model can be improved without requiring adding features.

	.. note::

		The aim of a regression model is to find the function :math:`x \\to f(x) \in \\mathbb{R}` 
		to be used as our predictor for :math:`y` given :math:`x` so that :math:`f(x)` fully captures all
		the information about :math:`y` that is contained in :math:`x`.  For instance, this will be 
		the case when the true generative model takes the form

		.. math::

			y = f(x) + \\epsilon

		with :math:`\\epsilon` statistically independent from :math:`y`, in which case :math:`h \\left( y \\vert x \\right) = h(\\epsilon)`.

		More generally, the conditional entropy :math:`h \\left( y \\vert x \\right)` represents 
		the amount of information about :math:`y` that cannot be explained by :math:`x`, while 
		:math:`h \\left( y \\vert \\tilde{f}(x) \\right)` represents the amount of information 
		about :math:`y` that cannot be explained by the regression model 

		.. math::

			y = \\tilde{f}(x) + \\epsilon.

		A natural metric for how suboptimal a particular regression model is can therefore be defined as
		the difference between the amount of information about :math:`y` that cannot be explained by 
		:math:`\\tilde{f}(x)` and the amount of information about :math:`y` that cannot be explained by :math:`x`


		.. math::

			\\text{subopt}(\\tilde{f}; x) &= h \\left( y \\vert \\tilde{f}(x) \\right) - h \\left( y \\vert x \\right) \\

			:&= I\\left(y, x \\right) - I\\left(y, \\tilde{f}(x) \\right) \\

			&\geq 0.

		This regression suboptimality metric is 0 if and only if :math:`\\tilde{f}(x)` fully captures any information about :math:`y`
		that is contained in :math:`x`. When 

		.. math::

			\\text{subopt}(\\tilde{f}; x) > 0 

		on the other hand, there exists a regression model using :math:`x` as features that can better predict :math:`y`. The larger 
		:math:`\\text{subopt}(\\tilde{f}; x)`, the more the regression model is suboptimal and can be improved.


	Parameters
	----------
	x : (n, d) np.array
		n i.i.d. draws from the features generating distribution.
	y : (n,) np.array
		n i.i.d. draws from the (continuous) labels generating distribution, sampled
		jointly with x.
	yp : (n,) np.array
		Predictions of y.

	Returns
	-------
	d : float
		The regression's suboptimality measure.


	.. seealso:: 

		:ref:`kxy.api.core.mutual_information.least_continuous_mutual_information <least-continuous-mutual-information>`
	"""
	return least_continuous_mutual_information(x, y)-least_continuous_mutual_information(yp, y)




def regression_additive_suboptimality(e, x):
	"""
	.. _regression-additive-suboptimality:
	Quantifies the extent to which a regression model can be improved without requiring additional features, by evaluating 
	how informative its residuals still are about the features.

	.. note::

		Additive regression models aim at breaking down a label :math:`y` as the sum of a component that solely depend on 
		features :math:`f(x)` and a residual component that is statistically independent from features :math:`\\epsilon`

		.. math::

			y = f(x) + \\epsilon.

		In an ideal scenario, the regreession residual :math:`\\epsilon` would indeed be stastically independent from the features
		:math:`x`. In pratice however, this might not be the case, for instance when the space of candidate functions used by
		the regression model isn't flexible enough (e.g. linear regression or basis functions regression), or the optimization
		has not converged to the global optimum. 

		Any departure from statistical independence between residuals :math:`\\epsilon` and features :math:`x` is an indication that what
		:math:`x` can reveal about :math:`y` is not fully captured by :math:`f(x)`, which implies that the regression model can be improved.

		Thus, we define the additive suboptimality of a regression model as the mutual information between its residuals and its features

		.. math::

			\\text{add-subopt}(\\tilde{f}; x) := I\\left( y-\\tilde{f}(x), x \\right)


	Parameters
	----------
	e : (n,) np.array
		Regression residuals.
	x : (n,) np.array
		Regression features.


	Returns
	-------
	d : float
		The regression's additive suboptimality measure.


	.. seealso:: 

		:ref:`kxy.api.core.mutual_information.least_continuous_mutual_information <least-continuous-mutual-information>`
	"""
	return least_continuous_mutual_information(x, e)

