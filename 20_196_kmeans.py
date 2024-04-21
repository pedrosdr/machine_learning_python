import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

pio.renderers.default = "browser"

base_salario = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                        [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                        [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

kmeans = KMeans(3)
kmeans.fit(base_salario)
res = np.column_stack((base_salario, kmeans.predict(base_salario)))
fig1 = px.scatter(x=res[:,0], y=res[:,1], color=res[:,2].astype(str))
fig2 = px.scatter(x=kmeans.cluster_centers_[:,0], y=kmeans.cluster_centers_[:,1], size=[12, 12, 12])
fig = go.Figure(data=fig1.data + fig2.data)


xr, yr = make_blobs(200, centers=5)
px.scatter(x=xr[:,0], y=xr[:,1], color=yr.astype(str))

kmeans = KMeans(5)
kmeans.fit(xr)
px.scatter(x=xr[:,0], y=xr[:,1], color=kmeans.labels_.astype(str))
