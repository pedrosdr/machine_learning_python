import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('census.csv')
sns.countplot(data=df, x='income', hue='income')

x = df.iloc[:,1:15].to_numpy()
y = df.iloc[:,15].to_numpy()

categorical_cols = [1,3,5,6,7,8,9,13]
encoders = []
for i in categorical_cols:
    encoder = LabelEncoder()
    x[:,i] = encoder.fit_transform(x[:,i])
    encoders.append(encoder)
    
    
# Subamostragem
