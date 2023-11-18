import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('house_prices.csv')
df = df.drop(labels = ['id', 'date'], axis=1)

# sns.heatmap(df.corr())

x = df.iloc[:,1:-2].to_numpy()
x = np.c_[x, x**2]
y = df.iloc[:,0].to_numpy()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
ytrain = ytrain.reshape((ytrain.shape[0]), 1)
ytest = ytest.reshape((ytest.shape[0]), 1)

lr = LinearRegression()
lr. fit(xtest, ytest)
predictions = lr.predict(xtest)

print(r2_score(ytest, predictions))
print(mean_absolute_error(ytest, predictions))

plt.scatter(xtest[:,3], ytest)
plt.scatter(xtest[:,3], predictions)
