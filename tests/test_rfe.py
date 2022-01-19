import kxy
from kxy.misc import RFE
from kxy.learning import get_sklearn_learner, get_lightgbm_learner_learning_api, get_xgboost_learner

from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote, BankMarketing

def test_lasso_cv():
	regressor_cls = get_sklearn_learner('sklearn.linear_model.LassoCV')
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)

	fs = RFE(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Shucked weight', 'Shell weight', 'Length.ABS(* - Q25(*))', 'Height', 'Shucked weight.ABS(* - Q75(*))', \
		'Diameter', 'Height.ABS(* - Q75(*))', 'Viscera weight.ABS(* - MEDIAN(*))', 'Whole weight.ABS(* - Q25(*))', 'Viscera weight.ABS(* - Q75(*))', \
		'Viscera weight', 'Shucked weight.ABS(* - Q25(*))', 'Whole weight', 'Diameter.ABS(* - Q75(*))', 'Whole weight.ABS(* - Q75(*))', \
		'Diameter.ABS(* - MEAN(*))', 'Shell weight.ABS(* - MEDIAN(*))', 'Length.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - Q25(*))', 'Sex_I', \
		'Shell weight.ABS(* - Q25(*))', 'Length', 'Sex_M', 'Whole weight.ABS(* - MEDIAN(*))', 'Whole weight.ABS(* - MEAN(*))', 'Viscera weight.ABS(* - Q25(*))', \
		'Viscera weight.ABS(* - MEAN(*))', 'Shucked weight.ABS(* - MEDIAN(*))', 'Shucked weight.ABS(* - MEAN(*))', 'Shell weight.ABS(* - Q75(*))', \
		'Shell weight.ABS(* - MEAN(*))', 'Sex_F', 'Height.ABS(* - Q25(*))']


def test_lasso():
	regressor_cls = get_sklearn_learner('sklearn.linear_model.Lasso')
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)

	fs = RFE(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Whole weight.ABS(* - Q75(*))', 'Whole weight.ABS(* - Q25(*))', 'Whole weight.ABS(* - MEDIAN(*))', \
		'Whole weight.ABS(* - MEAN(*))', 'Viscera weight.ABS(* - Q75(*))', 'Viscera weight.ABS(* - Q25(*))', 'Viscera weight.ABS(* - MEDIAN(*))', \
		'Viscera weight.ABS(* - MEAN(*))', 'Viscera weight', 'Shucked weight.ABS(* - Q75(*))', 'Shucked weight.ABS(* - Q25(*))', \
		'Shucked weight.ABS(* - MEDIAN(*))', 'Shucked weight.ABS(* - MEAN(*))', 'Shell weight.ABS(* - Q75(*))', 'Shell weight.ABS(* - Q25(*))', \
		'Shell weight.ABS(* - MEDIAN(*))', 'Shell weight.ABS(* - MEAN(*))', 'Shell weight', 'Sex_M', 'Sex_I', 'Sex_F', 'Length.ABS(* - Q75(*))', \
		'Length.ABS(* - Q25(*))', 'Length.ABS(* - MEDIAN(*))', 'Length.ABS(* - MEAN(*))', 'Height.ABS(* - Q75(*))', 'Height.ABS(* - Q25(*))', \
		'Height.ABS(* - MEDIAN(*))', 'Height.ABS(* - MEAN(*))', 'Diameter.ABS(* - Q75(*))', 'Diameter.ABS(* - Q25(*))', 'Diameter.ABS(* - MEDIAN(*))', \
		'Diameter.ABS(* - MEAN(*))']



def test_linear_regression():
	regressor_cls = get_sklearn_learner('sklearn.linear_model.LinearRegression')
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)
	
	fs = RFE(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Height.ABS(* - MEDIAN(*))', 'Height.ABS(* - MEAN(*))', 'Viscera weight.ABS(* - MEDIAN(*))', \
		'Shell weight', 'Diameter.ABS(* - MEAN(*))', 'Shell weight.ABS(* - MEDIAN(*))', 'Shucked weight', 'Viscera weight.ABS(* - MEAN(*))', \
		'Shell weight.ABS(* - MEAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Diameter', 'Height', 'Length.ABS(* - Q25(*))', \
		'Shucked weight.ABS(* - Q75(*))', 'Height.ABS(* - Q75(*))', 'Viscera weight', 'Shell weight.ABS(* - Q25(*))', 'Diameter.ABS(* - Q75(*))', \
		'Shucked weight.ABS(* - MEDIAN(*))', 'Whole weight.ABS(* - Q25(*))', 'Length', 'Shucked weight.ABS(* - MEAN(*))', 'Viscera weight.ABS(* - Q75(*))', \
		'Shucked weight.ABS(* - Q25(*))', 'Whole weight', 'Whole weight.ABS(* - MEAN(*))', 'Viscera weight.ABS(* - Q25(*))', 'Whole weight.ABS(* - Q75(*))', \
		'Whole weight.ABS(* - MEDIAN(*))', 'Length.ABS(* - MEDIAN(*))', 'Length.ABS(* - Q75(*))', 'Length.ABS(* - MEAN(*))', 'Shell weight.ABS(* - Q75(*))']



def test_lightgbm_regression():
	lgbm_params = {
		'objective': 'rmse',  
		'boosting_type': 'gbdt',
		'n_jobs': -1,
		'learning_rate': 0.1,
		'verbose': -1,
	}
	regressor_cls = get_lightgbm_learner_learning_api(lgbm_params, num_boost_round=2000, \
		early_stopping_rounds=5, split_random_seed=0)
	
	fs = RFE(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Shell weight', 'Shucked weight', 'Shucked weight.ABS(* - MEAN(*))', 'Whole weight.ABS(* - MEAN(*))', \
		'Viscera weight', 'Shucked weight.ABS(* - MEDIAN(*))', 'Shell weight.ABS(* - MEAN(*))', 'Whole weight', 'Shucked weight.ABS(* - Q75(*))', \
		'Height', 'Diameter', 'Sex_I', 'Shell weight.ABS(* - Q75(*))', 'Whole weight.ABS(* - Q25(*))', 'Shucked weight.ABS(* - Q25(*))', \
		'Length', 'Viscera weight.ABS(* - Q75(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Diameter.ABS(* - MEAN(*))', 'Viscera weight.ABS(* - MEDIAN(*))', \
		'Whole weight.ABS(* - MEDIAN(*))', 'Height.ABS(* - Q75(*))', 'Diameter.ABS(* - Q75(*))', 'Sex_F', 'Whole weight.ABS(* - Q75(*))', \
		'Viscera weight.ABS(* - Q25(*))', 'Shell weight.ABS(* - Q25(*))', 'Length.ABS(* - MEDIAN(*))', 'Height.ABS(* - MEAN(*))', \
		'Shell weight.ABS(* - MEDIAN(*))', 'Diameter.ABS(* - Q25(*))', 'Length.ABS(* - MEAN(*))', 'Height.ABS(* - MEDIAN(*))']

	n_vars = 5
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Shell weight', 'Shucked weight', 'Whole weight', 'Viscera weight', 'Shucked weight.ABS(* - MEAN(*))']



def test_random_forest_regression():
	regressor_cls = get_sklearn_learner('sklearn.ensemble.RandomForestRegressor', random_state=0)
	from warnings import simplefilter
	from sklearn.exceptions import ConvergenceWarning
	simplefilter("ignore", category=ConvergenceWarning)
	
	fs = RFE(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Shell weight', 'Shucked weight.ABS(* - Q25(*))', 'Shell weight.ABS(* - Q25(*))', 'Shell weight.ABS(* - MEDIAN(*))', \
		'Shucked weight', 'Shell weight.ABS(* - Q75(*))', 'Shell weight.ABS(* - MEAN(*))', 'Shucked weight.ABS(* - MEDIAN(*))', 'Shucked weight.ABS(* - MEAN(*))', \
		'Shucked weight.ABS(* - Q75(*))', 'Viscera weight.ABS(* - Q75(*))', 'Sex_I', 'Whole weight.ABS(* - Q75(*))', 'Whole weight.ABS(* - Q25(*))', \
		'Whole weight.ABS(* - MEAN(*))', 'Viscera weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - MEAN(*))', 'Diameter.ABS(* - MEAN(*))', \
		'Whole weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q25(*))', 'Whole weight', 'Height.ABS(* - Q75(*))', 'Diameter.ABS(* - Q75(*))', \
		'Viscera weight', 'Length.ABS(* - MEAN(*))', 'Length.ABS(* - Q75(*))', 'Diameter.ABS(* - Q25(*))', 'Length.ABS(* - Q25(*))', 'Length.ABS(* - MEDIAN(*))', \
		'Height.ABS(* - MEAN(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Height', 'Height.ABS(* - Q25(*))']



def test_xgboost_regression():
	regressor_cls = get_xgboost_learner('xgboost.XGBRegressor', random_state=0)
	
	fs = RFE(regressor_cls)

	dataset = Abalone()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Shell weight', 'Sex_I', 'Shucked weight.ABS(* - Q25(*))', 'Shucked weight', 'Shucked weight.ABS(* - MEDIAN(*))', \
		'Shucked weight.ABS(* - MEAN(*))', 'Diameter.ABS(* - Q75(*))', 'Height.ABS(* - Q75(*))', 'Diameter.ABS(* - MEAN(*))', 'Diameter.ABS(* - Q25(*))', \
		'Whole weight.ABS(* - MEDIAN(*))', 'Viscera weight.ABS(* - Q75(*))', 'Sex_M', 'Height.ABS(* - MEAN(*))', 'Shucked weight.ABS(* - Q75(*))', \
		'Viscera weight.ABS(* - MEAN(*))', 'Height.ABS(* - Q25(*))', 'Whole weight.ABS(* - MEAN(*))', 'Shell weight.ABS(* - Q25(*))', 'Whole weight.ABS(* - Q25(*))', \
		'Length.ABS(* - MEAN(*))', 'Length.ABS(* - Q75(*))', 'Whole weight.ABS(* - Q75(*))', 'Diameter.ABS(* - MEDIAN(*))', 'Shell weight.ABS(* - Q75(*))', \
		'Shell weight.ABS(* - MEAN(*))', 'Shell weight.ABS(* - MEDIAN(*))', 'Length.ABS(* - MEDIAN(*))', 'Sex_F', 'Viscera weight', 'Whole weight', \
		'Length.ABS(* - Q25(*))', 'Viscera weight.ABS(* - MEDIAN(*))']



def test_xgboost_classifier():
	# Binary classification
	classifier_cls = get_xgboost_learner('xgboost.XGBClassifier', use_label_encoder=False, 
		eval_metric='logloss', learning_rate=0.1, max_depth=10)
	fs = RFE(classifier_cls)

	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Skewness.ABS(* - MEDIAN(*))', \
		'Variance.ABS(* - MEAN(*))', 'Skewness.ABS(* - MEAN(*))', 'Kurtosis.ABS(* - MEDIAN(*))', 'Kurtosis.ABS(* - Q25(*))', \
		'Entropy.ABS(* - MEDIAN(*))', 'Skewness.ABS(* - Q25(*))', 'Entropy.ABS(* - MEAN(*))', 'Variance.ABS(* - Q25(*))', \
		'Kurtosis.ABS(* - MEAN(*))', 'Kurtosis.ABS(* - Q75(*))']



def test_lightgbm_classifier():
	# Classification
	params = params = {
		'objective': 'binary',
		'metric': ['auc', 'binary_logloss'],
		'boosting_type': 'gbdt',
	}
	classifier_cls = get_lightgbm_learner_learning_api(params, num_boost_round=10000, \
		early_stopping_rounds=5, verbose_eval=50, split_random_seed=42)
	fs = RFE(classifier_cls)
	
	dataset = BankNote()
	target_column = dataset.y_column
	df = dataset.df

	# Features generation
	features_df = df.kxy.generate_features(entity=None, max_lag=None, entity_name='*', exclude=[target_column])

	# Feature selection
	x_columns = [_ for _ in features_df.columns if _ != target_column]
	x_df = features_df[x_columns]
	y_df = features_df[[target_column]]
	n_vars = max(x_df.shape[1]-5, 1)
	m = fs.fit(x_df, y_df, n_vars)

	# Assertions
	assert len(fs.selected_variables) == n_vars
	assert fs.selected_variables == ['Variance', 'Kurtosis', 'Skewness.ABS(* - MEAN(*))', 'Skewness', 'Variance.ABS(* - MEAN(*))', \
		'Entropy', 'Variance.ABS(* - Q25(*))', 'Kurtosis.ABS(* - MEDIAN(*))', 'Kurtosis.ABS(* - Q75(*))', 'Skewness.ABS(* - MEDIAN(*))', \
		'Kurtosis.ABS(* - Q25(*))', 'Kurtosis.ABS(* - MEAN(*))', 'Variance.ABS(* - MEDIAN(*))', 'Entropy.ABS(* - MEDIAN(*))', \
		'Entropy.ABS(* - Q25(*))']


