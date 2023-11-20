import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge


m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x + 0.4 * x**2 + 2 + np.random.randn(m, 1)



# # Elastic Net -> Lasso
# en1 = ElasticNet(alpha=0.0001, l1_ratio=1)
# en1.fit(np.c_[x, x**2, x**3], y)
# y_new = en1.predict(np.c_[x, x**2, x**3])

# plt.scatter(x, y)
# plt.scatter(x, y_new)



# # Elastic Net -> Ridge
# en2 = ElasticNet(alpha=0.0001, l1_ratio=0)
# en2.fit(np.c_[x, x**2, x**3], y)
# y_new = en2.predict(np.c_[x, x**2, x**3])

# plt.scatter(x, y)
# plt.scatter(x, y_new)



# Elastic Net -> Mixed
en2 = ElasticNet(alpha=0.001, l1_ratio=0.5)
en2.fit(np.c_[x, x**2, x**3], y)
y_new = en2.predict(np.c_[x, x**2, x**3])

plt.scatter(x, y)
plt.scatter(x, y_new)



# Ridge
rdg = Ridge(alpha=1)
rdg.fit(x, y)
y_new = rdg.predict(x)

plt.scatter(x, y)
plt.scatter(x, y_new)