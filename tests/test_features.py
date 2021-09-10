

from kxy_datasets.kaggle_classifications import Titanic


dataset = Titanic()
# df = dataset.df.kxy.entity_features('Age')
# print(df.shape)
# print(df)

# df = dataset.df.kxy.deviation_features()
# print(df.shape)
# print(df)

# df = dataset.df.kxy.temporal_features()
# print(df.shape)
# print(df)

res = dataset.df.kxy.generate_features(entity='Age', exclude=['Survived'], encoding_method='one_hot', 
	index=None, return_baselines=True, entity_name='passengers')
print(res[0].shape)
print(res[0])
print(res[1])
print(res[2])