import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

with open('risco_credito.pkl', 'rb') as f:
    x, y = pickle.load(f)

x = np.delete(x, np.where(y == 'moderado'), axis=0)
y = np.delete(y, np.where(y == 'moderado'), axis=0)

lr = LogisticRegression(random_state=1)
lr.fit(x, y)

print(lr.intercept_)
print(lr.coef_)

previsoes = lr.predict([[0, 0, 1, 2],[2, 0, 0, 0]])
