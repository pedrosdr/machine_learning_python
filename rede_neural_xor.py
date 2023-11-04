import pickle
import plotly.express as plx
import plotly.io as pio
import pandas as pd
from sklearn.cluster import KMeans

pio.renderers.default = 'browser'

with open('credit.pkl', 'rb') as f:
    xtrain, ytrain, xtest, ytest = pickle.load(f)

df = pd.DataFrame(xtrain)
df[3] = ytrain

km = KMeans(n_clusters=3)
km.fit(xtrain, ytrain)
df['cluster'] = km.predict(xtrain)


# plx.scatter(df, x = 1, y = 2).show()
plx.scatter_3d(df, x = 0, y = 1, z = 2, color='cluster').show()
