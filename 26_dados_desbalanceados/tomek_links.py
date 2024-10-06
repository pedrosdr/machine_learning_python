import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import TomekLinks
import seaborn as sns

df = pd.read_csv('../census.csv')

x = df.iloc[:,1:15].to_numpy()
y = df.iloc[:,15].to_numpy()

categorical_cols = [1,3,5,6,7,8,9,13]
pipex = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
], remainder='passthrough')
x = pipex.fit_transform(x).toarray()

tl = TomekLinks(sampling_strategy='all')
xunder, yunder = tl.fit_resample(x, y)

# training with raw data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

res1 = []
for i in range(15):
    rf1 = RandomForestClassifier(
        n_estimators=100,
        criterion='entropy',
        min_samples_leaf=1,
        min_samples_split=5
    )
    rf1.fit(xtrain, ytrain)
    res1.append(accuracy_score(ytest, rf1.predict(xtest)))
    print(i)
res1 = np.array(res1)

# training with undersampled data
xtrain, xtest, ytrain, ytest = train_test_split(xunder, yunder, test_size=0.15)

res2 = []
for i in range(15):
    rf2 = RandomForestClassifier(
        n_estimators=100,
        criterion='entropy',
        min_samples_leaf=1,
        min_samples_split=5
    )
    rf2.fit(xtrain, ytrain)
    res2.append(accuracy_score(ytest, rf2.predict(xtest)))
    print(i)
res2 = np.array(res2)

# Comparing results
res1 = res1.reshape(-1,1)
res1 = np.concatenate([res1, np.zeros([15, 1])], axis=1)

res2 = res2.reshape(-1,1)
res2 = np.concatenate([res2, np.ones([15, 1])], axis=1)

res = np.concatenate([res2, res1], axis=0)

sns.boxenplot(y=res[:,0], x=res[:,1], hue=res[:,1])
