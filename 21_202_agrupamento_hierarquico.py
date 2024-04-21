import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

pio.renderers.default = "browser"

base_salario = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                        [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                        [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

dendrograma = dendrogram(linkage(base_salario, method='ward'))

hc = AgglomerativeClustering(2, linkage='ward', metric='euclidean')
classes = hc.fit_predict(base_salario)

fig = px.scatter(x=base_salario[:,0], y=base_salario[:,1], color = classes)
fig
