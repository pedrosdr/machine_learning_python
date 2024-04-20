import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pio.renderers.default = "browser"

base = pd.read_csv('credit_clients.csv', sep=';')
base['BILL_TOTAL'] = base.iloc[:,12:18].sum(axis=1)

x = base.to_numpy()[:,[1,2,3,4,5,25]]
scaler = StandardScaler()
x = scaler.fit_transform(x)

wcss = [KMeans(n_clusters=i).fit(x).inertia_/x.shape[0] for i in list(range(1, 11))]
plt.plot(list(range(1,11)), wcss)

kmeans = KMeans(4)
kmeans.fit(x)

pca = PCA(2)
x_ = pca.fit_transform(x)

fig = px.scatter(x=x[:,0], y=x[:,5], color=[str(i) for i in kmeans.labels_])
fig

fig = px.scatter(x=x_[:,0], y=x_[:,1], color=[str(i) for i in kmeans.labels_])
fig
    
    
