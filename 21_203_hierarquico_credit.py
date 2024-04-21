import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

pio.renderers.default = "browser"

base = pd.read_csv('credit_clients.csv', sep=';')
base['BILL_TOTAL'] = base.iloc[:,12:18].sum(axis=1)

x = base.to_numpy()[:500,[1,2,3,4,5,25]]
scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(2)
x_ = pca.fit_transform(x)

dendrograma = dendrogram(linkage(x, method='ward'))

hc = AgglomerativeClustering(3, linkage='ward', metric='euclidean')
classes = hc.fit_predict(x)

fig = px.scatter(x=x_[:,0], y=x_[:,1], color=classes)
fig
    