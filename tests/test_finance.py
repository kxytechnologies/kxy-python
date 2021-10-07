import numpy as np
import pandas as pd
import kxy

def test_ia_corr_anon():
	x = np.random.randn(10000, 2)
	df = pd.DataFrame(x, columns=['market_column', 'asset_column'])
	iab_anon = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=True)
	iab = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=False)
	assert np.allclose(iab, iab_anon, atol=1e-03), 'Anonymized and non-anonymized results should be identical (%.4f vs %.4f)' % (iab, iab_anon)


def test_ia_corr_nan():
	x = np.random.randn(10000, 2)
	x[100:200, 0] = np.nan
	x[200:300, 1] = np.nan
	df = pd.DataFrame(x, columns=['market_column', 'asset_column'])
	iab_anon = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=True)
	assert not np.isnan(iab_anon)
	iab = df.kxy.information_adjusted_beta('market_column','asset_column', anonymize=False)
	assert not np.isnan(iab)
	assert np.allclose(iab, iab_anon, atol=1e-03), 'Anonymized and non-anonymized results should be identical (%.4f vs %.4f)' % (iab, iab_anon)

