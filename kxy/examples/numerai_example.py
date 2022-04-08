from sklearn.metrics import r2_score
import kxy
import pandas as pd
from kxy.learning import get_lightgbm_learner_sklearn_api

########
# Data #
########
## Uncomemnt to download Numerai data
# from numerapi import NumerAPI
# napi = NumerAPI()
# current_round = napi.get_current_round(tournament=8)
# napi.download_dataset("numerai_training_data_int8.parquet", "numerai_training_data_int8.parquet")

df = pd.read_parquet('numerai_training_data_int8.parquet')
target_column, problem_type = 'target', 'regression'
feature_columns = [_ for _ in df.columns if _.startswith('feature_')]
columns = feature_columns + [target_column]
df = df[columns]


####################
# Train/Test Split #
####################
random_seed = 2
test_df = df.sample(frac=0.8, random_state=random_seed)
train_df = df.drop(test_df.index)
train_features = train_df[feature_columns]
train_labels = train_df[[target_column]]
test_features = test_df[feature_columns]
test_labels = test_df[[target_column]]


##########################
# With Feature Selection #
##########################
# LightGBM model factory
lightgbm_regressor_learner_cls = get_lightgbm_learner_sklearn_api('lightgbm.LGBMRegressor', \
    n_jobs=-1, colsample_bytree=0.1, learning_rate=0.01, n_estimators=2000, max_depth=5)

# Lean boosting fit
results = train_df.kxy.fit(target_column, lightgbm_regressor_learner_cls, \
    problem_type=problem_type, feature_selection_method='pfs', pfs_p=100, \
    data_identifier='numerai_training_data_int8_train_seed_%d.parquet.gzip' % random_seed)

predictor = results['predictor']
p = predictor.feature_directions.shape[0]
print('Number of features: %d' % p)

# selected_features = predictor.selected_variables
# print('Selected Variables')
# print(selected_features)

# Training/Testing Predictions
train_predictions = predictor.predict(train_features)
test_predictions = predictor.predict(test_features)

# Training/Testing Performance
train_r2 = r2_score(train_labels, train_predictions)
test_r2 = r2_score(test_labels, test_predictions)

print('Compressed LightGBM: Training R^2: %.4f, Testing R^2: %.4f' % (train_r2, test_r2))


#################################
# Fit Without Feature Selection #
#################################
results = train_df.kxy.fit(target_column, lightgbm_regressor_learner_cls, \
    problem_type=problem_type, feature_selection_method=None)
naive_predictor = results['predictor']

# Training/Testing Predictions
naive_train_predictions = naive_predictor.predict(train_features)
naive_test_predictions = naive_predictor.predict(test_features)

# Training/Testing Performance
naive_train_r2 = r2_score(train_labels, naive_train_predictions)
naive_test_r2 = r2_score(test_labels, naive_test_predictions)

print('Naive LightGBM: Training R^2: %.4f, Testing R^2: %.4f' % (naive_train_r2, naive_test_r2))


