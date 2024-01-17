import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import pyplot as plt

base = pd.read_csv('plano_saude.csv')

x = base['idade'].to_numpy().reshape(-1,1)
y = base['custo'].to_numpy()

# Model
model = RandomForestRegressor()
model.fit(x, y)

ynew = model.predict(x)

print(r2_score(y, ynew))

# Plots
plt.plot(x, y)
plt.plot(x, ynew)
