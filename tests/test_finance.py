import numpy as np
import pandas as pd
import kxy

def test_ia_corr_anon():
	x = np.random.randn(10000, 2)
	df = pd.DataFrame(x, columns=['market_column', 'asset_column'])
	iab_anon = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=True)
	iab = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=False)
	assert iab_anon == iab, 'Anonymized and non-anonymized results should be identical'


def test_ia_corr_nan():
	x = np.random.randn(10000, 2)
	x[100:200, 0] = np.nan
	x[200:300, 1] = np.nan
	df = pd.DataFrame(x, columns=['market_column', 'asset_column'])
	iab_anon = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=True)
	assert not np.isnan(iab_anon)
	iab = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=False)
	assert not np.isnan(iab)
	assert iab == iab, 'Anonymized and non-anonymized results should be identical'




