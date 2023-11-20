import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris.csv').to_numpy()
# df[:,4] = [0 if x == 'Iris-setosa' else 1 for x in df[:,4]]
x = df[:,0:4]
y = df[:,4]

lr = LogisticRegression()
lr.fit(x, y)
y_new = lr.predict(x)

print(accuracy_score(y, y_new))


# Gradient Descent
df = pd.read_csv('iris.csv').to_numpy()
df[:,4] = [0 if x == 'Iris-setosa' else 1 for x in df[:,4]]
x = df[:,0:4].astype(np.float64)
y = df[:,4].reshape((y.shape[0], 1)).astype(np.float64)

def logit(y_):
    return 1 / (1 + np.exp(-y_))

n_iter = 1000
lr = 0.01
theta = np.random.randn(5, 1)
x_b = np.c_[np.ones((150, 1)), x]

for i in range(n_iter):
    gradients = (2/150) * x_b.T @ (logit(x_b @ theta) - y)
    theta -= lr * gradients

y_new = [0 if x < 0.5 else 1 for x in logit(x_b @ theta)]

print(accuracy_score(y, y_new))

