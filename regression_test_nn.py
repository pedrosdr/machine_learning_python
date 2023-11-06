from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

df = pd.DataFrame()
df['x1'] = [2, 3, 4, 5, 6]
df['x2'] = df['x1'] ** 2
df['x3'] = df['x1'] ** 3
df['y'] = 3 + df['x1'] * 5 + df['x2'] * 2 + df['x3'] * 0.5

xtrain = df.loc[:,['x1', 'x2', 'x3']].to_numpy()
ytrain = df.loc[:,['y']].to_numpy().T[0]

xtest = np.array([[7, 9, 4]])
ytest = np.array([58])

x = np.concatenate((xtrain, xtest), axis = 0)
y = np.concatenate((ytrain, ytest), axis = 0)
params = {
    'max_iter': [1000],
    'verbose': [False],
    'solver': ['adam'],
    'activation': ['identity', 'relu'],
    'hidden_layer_sizes': [(2,2), (3,3)],
    'tol': [0.1, 0.01],
    'learning_rate_init': [0.1, 0.01]
}

 
gs = GridSearchCV(MLPRegressor(), params)
gs.fit(x, y)
gs.best_params_

reg = gs.best_estimator_

prediction = gs.predict(np.array([[40, 2, 20]]))
