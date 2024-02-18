from preprocessing import preprocess_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from scikeras.wrappers import KerasRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Decision Tree
tree = DecisionTreeRegressor(max_depth=5)

# SVR
svr = SVR()

# RN
def getModel():
    rn = Sequential()
    rn.add(Dense(8, activation='relu', input_shape=(16,)))
    rn.add(Dropout(0.2))
    
    rn.add(Dense(8, activation='relu'))
    rn.add(Dropout(0.2))
    
    rn.add(Dense(8, activation='relu'))
    rn.add(Dropout(0.2))

    rn.add(Dense(1, activation='linear'))
    
    zin = Input(shape=(16,))
    zout = rn(zin)
    
    model = Model(zin, zout)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return model

rn = KerasRegressor(getModel, batch_size=100, epochs=200)


###################### Cross Validation #######################################
x = pd.read_csv('house_prices_x.csv').to_numpy()
y = pd.read_csv('house_prices_y.csv').to_numpy()

# Tree
results_tree = []
for i in range(10):
    kf = KFold(n_splits=5, shuffle=True)
    results = cross_val_score(tree, x, y, scoring='neg_mean_squared_error', cv=kf)
    results_tree.append(results.mean())
    
# SVR
results_svr = []
for i in range(10):
    kf = KFold(n_splits=5, shuffle=True)
    results = cross_val_score(svr, x, y, scoring='neg_mean_squared_error', cv=kf)
    results_svr.append(results.mean())

results_rn = []
for i in range(10):
    kf = KFold(n_splits=5, shuffle=True)
    results = cross_val_score(rn, x, y, scoring='neg_mean_squared_error', cv=kf)
    results_rn.append(results.mean())
    
# Salvando a base de dados
df_tree = np.c_[results_tree, ['tree' for x in range(10)]]
df_svr = np.c_[results_svr, ['svr' for x in range(10)]]
df_rn = np.c_[results_rn, ['rn' for x in range(10)]]

df_cv = np.concatenate((df_tree, df_svr, df_rn))
df_cv = pd.DataFrame(df_cv, columns=['MSE', 'GROUP'])
df_cv['MSE'] = df_cv['MSE'].astype(np.float64)

df_cv.to_csv('cv_results.csv', index=False)



