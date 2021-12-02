#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import kurtosis, skew

def nanskew(a, axis=0, bias=True):
	''' '''
	return skew(a, axis=axis, bias=bias, nan_policy='omit')

def nankurtosis(a, axis=0, fisher=True, bias=True):
	''' '''
	return kurtosis(a, axis=axis, bias=bias, nan_policy='omit')

def nanmin(a, axis=None, out=None):
	''' '''
	try:
		return np.nanmin(a, axis=axis, out=out)
	except:
		return np.nan

def nanmax(a, axis=None, out=None):
	''' '''
	try:
		return np.nanmax(a, axis=axis, out=out)
	except:
		return np.nan

def nanmaxmmin(a, axis=None, out=None):
	''' '''
	return nanmax(a, axis=axis, out=out)-nanmin(a, axis=axis, out=out)

def nanmean(a, axis=None, out=None):
	''' '''
	try:
		return np.nanmean(a, axis=axis, out=out)
	except:
		return np.nan

def nansum(a, axis=None, out=None):
	''' '''
	try:
		return np.nansum(a, axis=axis, out=out)
	except:
		return np.nan


def nanstd(a, axis=None, dtype=None, out=None):
	''' '''
	try:
		return np.nanstd(a, axis=axis, out=out)
	except:
		return np.nan

def nanmedian(a, axis=None, out=None, overwrite_input=False):
	''' '''
	try:
		return np.nanmedian(a, axis=axis, out=out, overwrite_input=overwrite_input)
	except:
		return np.nan

def q25(x):
	''' '''
	return x.quantile(0.25)

def q75(x):
	''' '''
	return x.quantile(0.75)

def nanskewabs(a, axis=0, bias=True):
	''' '''
	return skew(np.abs(a), axis=axis, bias=bias, nan_policy='omit')

def nankurtosisabs(a, axis=0, fisher=True, bias=True):
	''' '''
	return kurtosis(np.abs(a), axis=axis, bias=bias, nan_policy='omit')

def nanminabs(a, axis=None, out=None):
	''' '''
	try:
		return np.nanmin(np.abs(a), axis=axis, out=out)
	except:
		return np.nan

def nanmaxabs(a, axis=None, out=None):
	''' '''
	try:
		return np.nanmax(np.abs(a), axis=axis, out=out)
	except:
		return np.nan

def nanmaxmminabs(a, axis=None, out=None):
	''' '''
	return nanmax(np.abs(a), axis=axis, out=out)-nanmin(a, axis=axis, out=out)

def nanmeanabs(a, axis=None, out=None):
	''' '''
	try:
		return np.nanmean(np.abs(a), axis=axis, out=out)
	except:
		return np.nan

def nansumabs(a, axis=None, out=None):
	''' '''
	try:
		return np.nansum(np.abs(a), axis=axis, out=out)
	except:
		return np.nan

def nanstdabs(a, axis=None, dtype=None, out=None):
	''' '''
	try:
		return np.nanstd(np.abs(a), axis=axis, out=out)
	except:
		return np.nan

def nanmedianabs(a, axis=None, out=None, overwrite_input=False):
	''' '''
	try:
		return np.nanmedian(np.abs(a), axis=axis, out=out, overwrite_input=overwrite_input)
	except:
		return np.nan

def q25abs(x):
	''' '''
	return np.abs(x).quantile(0.25)

def q75abs(x):
	''' '''
	return np.abs(x).quantile(0.75)

def n_unique(x):
	''' '''
	vc = x.value_counts(normalize=True, sort=True, ascending=False)
	return len(vc.index)

def mode(x):
	''' '''
	vc = x.value_counts(normalize=True, sort=True, ascending=False)
	return vc.index[0] if len(vc.index) > 0 else np.nan

def modefreq(x):
	''' '''
	vc = x.value_counts(normalize=True, sort=True, ascending=False)
	return vc.values[0] if len(vc.index) > 0 else np.nan

def lastmode(x):
	''' '''
	vc = x.value_counts(normalize=True, sort=True, ascending=False)
	return vc.index[-1] if len(vc.index) > 0 else np.nan

def lastmodefreq(x):
	''' '''
	vc = x.value_counts(normalize=True, sort=True, ascending=False)
	return vc.values[-1] if len(vc.index) > 0 else np.nan

def nextmode(x):
	''' '''
	vc = x.value_counts(normalize=True, sort=True, ascending=False)
	return vc.index[1] if len(vc.index) > 1 else vc.index[0] if len(vc.index) > 0 else np.nan

def nextmodefreq(x):
	''' '''
	vc = x.value_counts(normalize=True, sort=True, ascending=False)
	return vc.values[1] if len(vc.values) > 1 else vc.values[0] if len(vc.index) > 0 else np.nan

def rmspe_score(y_true, y_pred):
	''' '''
	return np.sqrt(np.nanmean(np.square((y_true.flatten() - y_pred.flatten()) / y_true.flatten())))

def neg_rmspe_score(y_true, y_pred):
	''' '''
	return -rmspe_score(y_true, y_pred)


def neg_mae_score(y_true, y_pred):
	''' '''
	return -np.nanmean(np.abs(y_true-y_pred))


def neg_rmse_score(y_true, y_pred):
	''' '''
	return -np.sqrt(np.nanmean((y_true-y_pred)**2))


