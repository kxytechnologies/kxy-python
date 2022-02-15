# 0. As a one-off, run 'pip install kxy', then 'kxy configure'
# This import is necessary to get all df.kxy.* methods
import kxy

# 1. Load your data
# pip install kxy_datasets
from kxy_datasets.classifications import BankMarketing
dataset = BankMarketing()
target_column = dataset.y_column
df = dataset.df

# 2. Generate candidate features
features_df = df.kxy.generate_features(entity=None, max_lag=None,\
    entity_name='*', exclude=[target_column])
features_df = features_df.drop('y_yes', axis=1)
target_column = 'y_no'

# 3. Training/Testing split
# pip install scikit-learn
from sklearn.model_selection import train_test_split
train_features_df, test_features_df = train_test_split(features_df, \
	test_size=0.2, random_state=0)
test_labels_df = test_features_df.loc[:, [target_column]]
test_features_df = test_features_df.drop(target_column, axis=1)

# 4. Create a LightGBM learner function.

# A learner function is a function that expects up to two optional 
# variables: n_vars and path. When called it returns an instance of 
# 'predictive model' expecting n_vars features. The path parameter, 
# when provided, allows the learner function to load a saved model 
# from disk. 

# A 'predictive model' here is any class with a fit(self, x, y) method 
# and predict(self, x) method. To use the path argument of the learner 
# function, the class should also define a save(self, path) method to 
# save a model to disk, and a load(cls, path) class method to load a 
# saved model from disk. 

# See kxy.learning.base_learners for helper functions that allow you 
# create learner functions that return instances of popular predictive 
# models (e.g. lightgbm, xgboost, sklearn, tensorflow, pytorch models 
# etc.).

from kxy.learning import get_lightgbm_learner_learning_api
params = {
	'objective': 'binary',
	'metric': ['auc', 'binary_logloss'],
}
lightgbm_learner_func = get_lightgbm_learner_learning_api(params, \
	num_boost_round=10000, early_stopping_rounds=50, verbose_eval=50, \
	split_random_seed=0)
    
# 5. Fit a LightGBM classifier wrapped around LeanML feature selection
results = train_features_df.kxy.fit(target_column, \
	lightgbm_learner_func, problem_type='classification', \
	feature_selection_method='leanml')
predictor = results['predictor']

# 6. Make predictions from a dataframe of test features
test_predictions_df = predictor.predict(test_features_df)

# 7. Compute out-of-sample accuracy and AUC
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(
    test_labels_df[target_column].values, \
    test_predictions_df[target_column].values, \
)
auc = roc_auc_score( \
    test_labels_df[target_column].values, \
    test_predictions_df[target_column].values, \
    multi_class='ovr'
)

print('LeanML -- Testing Accuracy: %.2f, AUC: %.2f' % (accuracy, auc))
selected_features = predictor.selected_variables
print('LeanML -- Selected Variables:')
import pprint as pp
pp.pprint(selected_features)

# 8. (Optional) Save the trained model.
path = './lightgbm_uci_bank_marketing.sav'
predictor.save(path)

# 9. (Optional) Load the saved model.
from kxy.learning.leanml_predictor import LeanMLPredictor
loaded_predictor = LeanMLPredictor.load(path, lightgbm_learner_func)



# 10.a Fit a LightGBM classifier wrapped around RFE feature selection
n_leanml_features = len(selected_features)
rfe_results = train_features_df.kxy.fit(target_column, \
	lightgbm_learner_func, problem_type='classification', \
	feature_selection_method='rfe', rfe_n_features=n_leanml_features)
rfe_predictor = rfe_results['predictor']

# 10.b Fit a LightGBM classifier wrapped around Boruta feature 
# selection.
boruta_results = train_features_df.kxy.fit(target_column, \
	lightgbm_learner_func, problem_type='classification', \
	feature_selection_method='boruta', boruta_n_evaluations= 20, \
    boruta_pval=0.95)
boruta_predictor = boruta_results['predictor']

# 10.c Fit a LightGBM classifier wrapped around Boruta feature 
# selection.
none_results = train_features_df.kxy.fit(target_column, \
	lightgbm_learner_func, problem_type='classification', \
	feature_selection_method=None)
none_predictor = none_results['predictor']

# 11. Make predictions from a dataframe of test features
rfe_test_predictions_df = rfe_predictor.predict(test_features_df)
boruta_test_predictions_df = boruta_predictor.predict(test_features_df)
none_test_predictions_df = none_predictor.predict(test_features_df)

# 12. Compute out-of-sample accuracy and AUC
rfe_accuracy = accuracy_score(
    test_labels_df[target_column].values, \
    rfe_test_predictions_df[target_column].values, \
)
rfe_auc = roc_auc_score( \
    test_labels_df[target_column].values, \
    rfe_test_predictions_df[target_column].values, \
    multi_class='ovr'
)

boruta_accuracy = accuracy_score(
    test_labels_df[target_column].values, \
    boruta_test_predictions_df[target_column].values, \
)
boruta_auc = roc_auc_score( \
    test_labels_df[target_column].values, \
    boruta_test_predictions_df[target_column].values, \
    multi_class='ovr'
)

none_accuracy = accuracy_score(
    test_labels_df[target_column].values, \
    none_test_predictions_df[target_column].values, \
)
none_auc = roc_auc_score( \
    test_labels_df[target_column].values, \
    none_test_predictions_df[target_column].values, \
    multi_class='ovr'
)

print('RFE -- Accuracy: %.2f, AUC: %.2f' % (rfe_accuracy, rfe_auc))
rfe_selected_features = rfe_predictor.selected_variables
print('RFE -- Selected Variables:')
pp.pprint(rfe_selected_features)
print()

print('Boruta -- Accuracy: %.2f, AUC: %.2f' % (boruta_accuracy, \
	boruta_auc))
boruta_selected_features = boruta_predictor.selected_variables
print('Boruta -- Selected Variables:')
pp.pprint(boruta_selected_features)
print()

print('No Feature Selection -- Accuracy: %.2f, AUC: %.2f' % (none_accuracy, \
	none_auc))
all_features = none_predictor.selected_variables
print('No Feature Selection -- Selected Variables:')
pp.pprint(all_features)


