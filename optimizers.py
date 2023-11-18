import numpy as np
import matplotlib.pyplot as plt


m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x + 2 + np.random.randn(m, 1)

# # MSE
# x_b = np.c_[x, np.ones((m, 1))]

# tb = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y

# y_new = x_b @ tb

# plt.scatter(x, y)
# plt.scatter(x, y_new)


# # Gradient Descent
# x_b = np.c_[x, np.ones((m, 1))]
# lr = 0.01
# n_iter = 50
# theta = np.random.randn(2, 1)

# for i in range(n_iter):
#     gradients = (2/m) * x_b.T @ (x_b @ theta - y)
#     theta -= lr * gradients

# y_new = x_b @ theta

# plt.scatter(x, y)
# plt.scatter(x, y_new)


# # Gradient Descent with Momentum
# x_b = np.c_[x, np.ones((m, 1))]
# lr = 0.01
# n_iter = 100
# theta = np.random.randn(2, 1)
# momentum = 0.9
# beta = 0.9

# for i in range(n_iter):
#     gradients = (2/m) * x_b.T @ (x_b @ theta - y)
#     momentum = beta * momentum - lr * gradients
#     theta += momentum

# y_new = x_b @ theta

# plt.scatter(x, y)
# plt.scatter(x, y_new)


# # Gradiente Acelerado de Nesterov
# x_b = np.c_[x, np.ones((m, 1))]
# lr = 0.01
# n_iter = 30
# theta = np.random.randn(2, 1)
# momentum = 0.9
# beta = 0.9

# for i in range(n_iter):
#     gradients = (2/m) * x_b.T @ (x_b @ (theta + beta * momentum) - y)
#     momentum = beta * momentum - lr * gradients
#     theta += momentum

# y_new = x_b @ theta

# plt.scatter(x, y)
# plt.scatter(x, y_new)


# # AdaGrad
# x_b = np.c_[x, np.ones((m, 1))]
# lr = 0.1
# n_iter = 100
# theta = np.random.randn(2, 1)
# s = 0.9
# e = 1 * 10**(-10)

# for i in range(n_iter):
#     gradients = (2/m) * x_b.T @ (x_b @ theta - y)
#     s = s + gradients ** 2
#     theta -= np.divide(lr * gradients, np.sqrt(s + e))

# y_new = x_b @ theta

# plt.scatter(x, y)
# plt.scatter(x, y_new)



# RMSProp
x_b = np.c_[x, np.ones((m, 1))]
lr = 0.01
n_iter = 50
theta = np.random.randn(2, 1)
s = 0.9
beta = 0.9
e = 1 * 10**(-10)

for i in range(n_iter):
    gradients = (2/m) * x_b.T @ (x_b @ theta - y)
    s = beta * s + (1-beta) * gradients * gradients
    theta -= np.divide(lr * gradients, np.sqrt(s + e))

y_new = x_b @ theta

plt.scatter(x, y)
plt.scatter(x, y_new)



# # Adam
# x_b = np.c_[x, np.ones((m, 1))]
# lr = 0.01
# n_iter = 1000
# theta = np.random.randn(2, 1)
# momentum = 0
# beta1 = 0.9
# beta2 = 0.999
# s = 0
# e = 1 * 10 ** (-5)

# for i in range(n_iter):
#     gradients = (2/m) * x_b.T @ (x_b @ theta - y)
    
#     momentum = momentum * beta1 - (1 - momentum) * gradients
#     s = s * beta2 - (1 - beta2) * gradients * gradients
    
#     momentum = momentum / (1-beta1)
#     s = s / (1-beta2)
    
#     theta += np.divide(lr * momentum, np.sqrt(np.abs(s) + e))
    
# y_new = x_b @ theta

# plt.scatter(x, y)
# plt.scatter(x, y_new)
