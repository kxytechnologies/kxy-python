from kxy_datasets.regressions import Abalone
from kxy_datasets.classifications import BankNote
from kxy.learning import get_sklearn_learner


# # Regression
# xgboost_regressor_cls = get_sklearn_learner('xgboost.XGBRegressor')
# dataset = Abalone()
# target_column = dataset.y_column
# df = dataset.df

# # Features generation
# features_df = df.kxy.generate_features(entity=None, max_lag=None, name='*', exclude=[target_column])

# # Model building
# results = features_df.kxy.fit(target_column, xgboost_regressor_cls, \
# 	problem_type='regression')
# print(results)


# Binary classification
xgboost_classifier_cls = get_sklearn_learner(
	'xgboost.XGBClassifier', use_label_encoder=False, 
	eval_metric='logloss', learning_rate=0.1,
	max_depth=10, scale_pos_weight=1.3)
dataset = BankNote()
target_column = dataset.y_column
df = dataset.df

# Features generation
features_df = df.kxy.generate_features(entity=None, max_lag=None, name='*', exclude=[target_column])

# Model building
results = features_df.kxy.fit(target_column, xgboost_classifier_cls, \
	problem_type='classification')
print(results)