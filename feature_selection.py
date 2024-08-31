import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, LabelEncoder,
                                   MinMaxScaler)
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

base = pd.read_csv('census.csv')

colunas = base.columns[1:-1]

x = base.iloc[:,1:15]
y = base.iloc[:,15:16]


# Numeric columns
numeric_cols = ['age', 'final.weight', 'education.num', 'capital.gain',
                'capital.loos', 'hour.per.week', 'education']

# Categorical columns
categorical_cols = ['workclass', 'marital.status', 'occupation',
                    'relationship', 'race', 'sex', 'native.country']

x.loc[:,categorical_cols]

# Ordinal columns
x['education'].unique()
x['education'] = x['education'].map({
    ' Preschool': 1,   # Educação infantil
    ' 1st-4th': 1,     # Ensino fundamental (anos iniciais)
    ' 5th-6th': 1,     # Ensino fundamental (anos intermediários)
    ' 7th-8th': 1,     # Ensino fundamental (anos finais)
    ' 9th': 2,         # Ensino médio (1º ano)
    ' 10th': 2,        # Ensino médio (2º ano)
    ' 11th': 2,        # Ensino médio (3º ano)
    ' 12th': 2,        # Ensino médio completo
    ' HS-grad': 3,     # Ensino médio completo ("High School Graduate")
    ' Some-college': 4,# Alguma faculdade ("Some college")
    ' Assoc-voc': 4,   # Diploma de associado vocacional
    ' Assoc-acdm': 4,  # Diploma de associado acadêmico
    ' Bachelors': 4,   # Bacharelado
    ' Masters': 5,     # Mestrado
    ' Prof-school': 5, # Escola profissionalizante (equivalente a especialização)
    ' Doctorate': 6    # Doutorado
})

pipe_numerical = Pipeline(steps=[
    ['imputer', SimpleImputer(strategy='median')],
    ['scaler', MinMaxScaler()]
])

pipe_categorical = Pipeline(steps=[
    ['imputer', SimpleImputer(strategy='most_frequent')],
    ['onehot', OneHotEncoder(handle_unknown='ignore')]
])

# transformer = ColumnTransformer(transformers=[
#     ['numerical_columns', pipe_numerical, numeric_cols],
#     ['categorical_columns', pipe_categorical, categorical_cols]
# ])

transformer = ColumnTransformer(transformers=[
    ['numerical_columns', pipe_numerical, numeric_cols]
])

x = transformer.fit_transform(x)

# y preprocessing
encodery = LabelEncoder()
y = encodery.fit_transform(y.to_numpy().flatten()).reshape(-1,1)
pipey = Pipeline(steps=[
    ['imputer', SimpleImputer(strategy='most_frequent')]
])

y = pipey.fit_transform(y)

xtrain, ytrain, xtest, ytest = train_test_split(x, y)


# Variance Threshold
sns.heatmap(np.var(x, axis=0).reshape(-1,1))

vt = VarianceThreshold(threshold=0.02)
vt.fit_transform(x).shape
vt.get_feature_names_out()


# ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(x, y.reshape(-1))
etc.feature_importances_.argsort()
pd.Series(numeric_cols)[(-etc.feature_importances_).argsort()]
