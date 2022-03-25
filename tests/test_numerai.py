import logging
import kxy
import pandas as pd
df = pd.read_parquet('numerai_training_data_int8.parquet')
columns = [_ for _ in df.columns if _ == 'target' or _.startswith('feature_')]
df = df[columns]

# logging.debug('Selecting variables')

# results = df.kxy.variable_selection('target', problem_type='regression', \
#   file_name='numerai_training_data_int8_p2.parquet.gzip', anonymize=False)
# print(results)



import kxy
import pandas as pd
from kxy.learning import get_lightgbm_learner_sklearn_api
target_column, problem_type = 'target', 'regression'
lightgbm_regressor_learner_cls = get_lightgbm_learner_sklearn_api('lightgbm.LGBMRegressor', \
    n_jobs=-1, colsample_bytree=0.1, learning_rate=0.01, n_estimators=2000, max_depth=5)

print(df.head())
results = df.kxy.fit(target_column, lightgbm_regressor_learner_cls, \
    problem_type=problem_type, feature_selection_method='leanml', \
    data_identifier='numerai_training_data_int8_p1.parquet.gzip')

predictor = results['predictor']
selected_features = predictor.selected_variables

print(selected_features)
