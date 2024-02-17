import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import pyplot as plt

base = pd.read_csv('house_prices.csv').drop(columns=['id', 'date']).dropna()

# Preprocessing
q1 = base.iloc[:,0].quantile(0.01)
q2 = base.iloc[:,0].quantile(0.99)
base = base[(base.iloc[:,0] > q1) & (base.iloc[:,0] < q2)]
    

x = base.iloc[:,1:].to_numpy()
y = base.iloc[:,0].to_numpy()


scalerx = StandardScaler()
x = scalerx.fit_transform(x)

scalery = StandardScaler()
y = scalery.fit_transform(y.reshape(-1,1)).reshape(-1,)

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

# Model
model = SVR(kernel='rbf')
model.fit(xtrain, ytrain)

# Predictions
ynew = model.predict(xtest)

ynew2 = scalery.inverse_transform(ynew.reshape(-1,1)).reshape(-1,)
ytest2 = scalery.inverse_transform(ytest.reshape(-1,1)).reshape(-1,)

print('R2:', r2_score(ytest2, ynew2))
print('Min Absolute Error', mean_absolute_error(ytest2, ynew2))

plt.plot([x for x in range(len(ytest2))], ytest2)
plt.plot([x for x in range(len(ynew2))], ynew2)