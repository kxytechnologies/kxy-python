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
