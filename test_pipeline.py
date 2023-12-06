from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
import pandas as pd
import numpy as np

df = pd.read_csv('iris.csv')
df_x = df.iloc[:,0:4]
df_y = df.iloc[:,4]

class PandasConverter():
    def __init__(self):
        pass
    
    def transform(self, X, y=None):
        return X.to_numpy()
    
    def fit(self, X, y=None):
        return self
    
pipe = Pipeline([
    ('converter', PandasConverter()),
    ('scaler', StandardScaler())
])

pipe.fit(df_x)
x = pipe.transform(df_x)

lenc = LabelEncoder()
y = lenc.fit_transform(df_y.to_numpy())

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

# tree = DecisionTreeClassifier(criterion='entropy')
# tree.fit(xtrain, ytrain)
# predictions = tree.predict(xtest)

svm = SVC(kernel = 'poly', random_state=1)
svm.fit(xtrain, ytrain)
predictions = svm.predict(xtest)

accuracy_score(ytest, predictions)
print(confusion_matrix(ytest, predictions))
