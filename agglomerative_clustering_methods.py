import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
pio.renderers.default = 'browser'


###############################################################################
####                              WARD                                     ####
###############################################################################

c1 = np.array([10 + 2*np.random.randn(20),
               10 + 2*np.random.randn(20),
               np.zeros(20)]).T

c2 = np.array([30 + 3*np.random.randn(20),
               30 + 3*np.random.randn(20),
               np.ones(20)]).T

c3 = np.array([20 + 1.3*np.random.randn(20),  
               10 + 1.3*np.random.randn(20),
               2*np.ones(20)]).T

df = np.concatenate((c1, c2, c3))

c = pd.DataFrame(data=df).groupby(2).mean().reset_index().iloc[:,[1,2,0]].to_numpy()

fig1 = px.scatter(x=df[:,0], y=df[:,1], color=df[:,2])
fig2 = px.scatter(x=c[:,0], y=c[:,1], size=[1 for x in range(len(c))])
go.Figure(data=fig1.data + fig2.data)

def wcss(df:np.ndarray):
    c = df.mean(axis=0)
    t = pd.DataFrame(data=df)
    t = t.apply(lambda x: x-c, axis=1).apply(np.square, axis=1)
    t = t.apply(np.sum, axis=1).apply(np.sqrt, axis=1).mean()
    
    plt.scatter(df[:,0], df[:,1], color='red')
    plt.scatter(c[0], c[1], color='blue')
    plt.show()
    plt.close()
    
    return t
    
wcss_0, wcss_1, wcss_2 = [wcss(df[df[:,2]==i][:,:2]) for i in range(3)]
wcss_01 = wcss(df[np.isin(df[:,2], [0,1])])
wcss_02 = wcss(df[np.isin(df[:,2], [0,2])])
wcss_12 = wcss(df[np.isin(df[:,2], [1,2])])

delta_01 = wcss_01 - wcss_0 - wcss_1
delta_02 = wcss_02 - wcss_0 - wcss_2
delta_12 = wcss_12 - wcss_1 - wcss_2

dendrogram(linkage(df[:,:2], method='ward', metric='euclidean'))
