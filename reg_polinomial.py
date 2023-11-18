import numpy as np
import matplotlib.pyplot as plt


m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)

# # MSE
# x_b = np.c_[x**2, x, np.ones((m, 1))]

# tb = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y

# y_new = x_b @ tb

# plt.scatter(x, y)
# plt.scatter(x, y_new)


# # Gradient Descent
# x_b = np.c_[x**2, x, np.ones((m, 1))]
# lr = 0.01
# n_iter = 50
# theta = np.random.randn(3, 1)

# for i in range(n_iter):
#     gradients = (2/m) * x_b.T @ (x_b @ theta - y)
#     theta -= lr * gradients

# y_new = x_b @ theta

# plt.scatter(x, y)
# plt.scatter(x, y_new)


# Gradient Descent with Momentum
x_b = np.c_[x**2, x, np.ones((m, 1))]
lr = 0.01
n_iter = 100
theta = np.random.randn(3, 1)
momentum = 0.9
beta = 0.9

for i in range(n_iter):
    gradients = (2/m) * x_b.T @ (x_b @ theta - y)
    momentum = beta * momentum - lr * gradients
    theta += momentum

y_new = x_b @ theta

plt.scatter(x, y)
plt.scatter(x, y_new)

    
