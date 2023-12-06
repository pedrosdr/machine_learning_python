import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import plotly.express as plx
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import math

pio.renderers.default = 'browser'

df = pd.read_csv('plano_saude.csv')
x = df['idade'].to_numpy().reshape((10, 1))
y = df['custo'].to_numpy().reshape((10, 1))

pf = PolynomialFeatures(3)
pf.fit(x)
x_p = pf.transform(x)

lm = Ridge(1)
lm.fit(x_p, y)
y_new = lm.predict(x_p)

# plot = plx.scatter(x = x.reshape((10,)), y= y.reshape((10,)))
# plot.add_trace(go.Scatter(x= x.reshape((10,)), y = y_new.reshape((10,)), mode='lines'))
# plot.show()

print(math.sqrt(mean_squared_error(y, y_new)))





