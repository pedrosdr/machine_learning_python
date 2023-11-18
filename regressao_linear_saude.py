import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('plano_saude.csv')
x = df.loc[:,'idade'].to_numpy().reshape((10, 1))
y = df.loc[:,'custo'].to_numpy().reshape((10, 1))

# plt.scatter(x, y)
# np.corrcoef(x, y)



# # MSE
# x_b = np.c_[np.ones((10,1)), x]
# theta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y

# x_new = np.array([[20], [60]])
# x_new_b = np.c_[np.ones((2,1)), x_new]
# y_new = x_new_b @ theta

# plt.scatter(x, y)
# plt.plot(x_new, y_new)



# Gradient Descent
scx = StandardScaler()
scx.fit(x)
xg = scx.transform(x)

scy = StandardScaler()
scy.fit(y)
yg = scy.transform(y)

x_b = np.c_[np.ones((10,1)), xg]
lr = 0.01
n_iter = 1000
m = 10
theta = np.random.randn(2, 1)

for i in range(n_iter):
    gradients = (2/m) * x_b.T @ (x_b @ theta - yg)
    theta -= lr * gradients
    
x_new = np.array([[20], [60]])
x_new = scx.transform(x_new)
x_new_b = np.c_[np.ones((2,1)), x_new]
y_new = x_new_b @ theta
y_new = scy.inverse_transform(y_new)
x_new = scx.inverse_transform(x_new)

plt.scatter(x, y)
plt.plot(x_new, y_new)



# # Sklearn
# lr = LinearRegression()
# lr.fit(x, y)
# previsoes = lr.predict(x)

# plt.scatter(x, y)
# plt.plot(x, previsoes)
