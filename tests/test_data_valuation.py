from kxy_datasets.regressions import Abalone


def test_include_mi():
	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df
	results = df.kxy.data_valuation(target_column, problem_type='regression', \
		include_mutual_information=True)
	assert 'Mutual Information' in results