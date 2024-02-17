import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import pyplot as plt

base = pd.read_csv('plano_saude.csv')

x = base['idade'].to_numpy().reshape(-1,1)
y = base['custo'].to_numpy()

scalerx = StandardScaler()
x = scalerx.fit_transform(x)

scalery = StandardScaler()
y = scalery.fit_transform(y.reshape(-1,1)).reshape(-1,)

# Modelo
model = SVR(kernel='rbf', C=100)
model.fit(x, y)

ynew = model.predict(x)
ynew2 = scalery.inverse_transform(ynew.reshape(-1,1)).reshape(-1,)
y2 = scalery.inverse_transform(y.reshape(-1,1)).reshape(-1,)

print(r2_score(y2, ynew2))

# Plots
plt.plot(x, y2)
plt.plot(x, ynew2)
