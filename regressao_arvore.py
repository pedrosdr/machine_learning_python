from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as plx
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = 'browser'

# df = pd.read_csv('house_prices.csv')
# df = df.iloc[:,2:]

# x = df.iloc[:,1:].to_numpy().astype(np.float64)
# y = df.iloc[:,0].to_numpy().astype(np.float64)

# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# tree = DecisionTreeRegressor(max_depth=6)
# tree.fit(xtrain, ytrain)
# predictions = tree.predict(xtest)

# print(mean_absolute_error(ytest, predictions))
# print(r2_score(ytest, predictions))

# Teste para randoms



x = 2*np.random.randn(100, 1)
y = 2 * np.random.randn(100, 1) + 0.5 * x**2 + 0.3 * x**3

xp = PolynomialFeatures(degree=3).fit_transform(x)

tree = DecisionTreeRegressor(max_depth=2)
tree.fit(xp, y)
y_new = tree.predict(xp)

print(r2_score(y, y_new))

fig = go.Figure()
fig.add_trace(go.Scatter(x = x[:,0], y = y[:,0], mode='markers'))
fig.add_trace(go.Scatter(x=x[:,0], y=y_new, mode='markers'))
fig.show()

