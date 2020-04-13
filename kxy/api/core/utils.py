#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 


def avg_pairwise_spearman_corr(x):
	"""
	Calculate the sample average pairwise Spearman rank correlation 
	between columns of the 2-dimensional input array `x` of shape (n, d).

	.. math::
		\frac{12}{n^2-1} \left[ \frac{2}{d(d-1)} \sum_{j<j^\prime} \frac{1}{n} \sum_{i=1}^n R_{ij} R_{ij^\prime} - \frac{(n+1)^2}{4}\right]

	where :math:`R_{ij}` is the rank of :math:`x_{ij}` among :math:`x_{1j} \dots x_{nj}`. See Eq. (3.3.3) in :cite:`JoeH90`.

    Parameters
    ----------
    x : (n, d) array_like
        Input data representing n i.i.d. draws from the d-dimensional 
        random variable, whose average pairwise Spearman rank correlation
        this function calculates.

	.. rubric:: References

	.. [joeh90] Joe, H. Journal of multivariate analysis 35 (1), 12-30, 1990.
	"""
	# R[i,j] is the rank of x[i, j] among x[1, j] ... x[n, j]
	n, m = x.shape
	R = 1+x.argsort(axis=0).argsort(axis=0)
	rho = np.sum([[np.dot(R[:, j], R[:, jp])/n for jp in range(1, m) for j in range(jp)]])/(0.5*m*(m-1)) - (n+1.)**2/4
	rho /= ((n*n-1.)/12.)

	return rho