import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(42)

# Generate data from a cubic function y = 3*x^3 + noise
n_train = 30
n_test = 100
x_train = np.random.uniform(-3, 3, size=n_train)
x_test = np.random.uniform(-3, 3, size=n_test)


def true_func(x):
    return 3 * x**3


noise_scale = 20.0
y_train = true_func(x_train) + np.random.normal(0, noise_scale, size=n_train)
y_test = true_func(x_test) + np.random.normal(0, noise_scale, size=n_test)

# Fit polynomial models of degree 1..10 using ordinary least squares
degrees = list(range(1, 11))
train_mse = []
test_mse = []
models = {}

for d in degrees:
    # design matrix with columns [1, x, x^2, ..., x^d]
    X_train = np.vander(x_train, N=d + 1, increasing=True)
    X_test = np.vander(x_test, N=d + 1, increasing=True)
    # solve for coefficients via least squares
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    models[d] = coef
    y_train_pred = X_train.dot(coef)
    y_test_pred = X_test.dot(coef)
    train_mse.append(np.mean((y_train - y_train_pred) ** 2))
    test_mse.append(np.mean((y_test - y_test_pred) ** 2))

# Plot data and fitted curves for degrees 1, 3, 10
x_grid = np.linspace(-3.5, 3.5, 400)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color="black", label="train data")
plt.scatter(x_test, y_test, color="tab:gray", alpha=0.5, label="test data")
for d in [1, 3, 10]:
    coef = models[d]
    X_grid = np.vander(x_grid, N=d + 1, increasing=True)
    y_grid = X_grid.dot(coef)
    plt.plot(x_grid, y_grid, label=f"degree {d}")
# plot true function
plt.plot(x_grid, true_func(x_grid), "k--", label="true cubic")
plt.legend()
plt.title("Polynomial fits (degrees 1, 3, 10)")
plt.xlabel("x")
plt.ylabel("y")

# Plot train/test MSE vs degree
plt.subplot(1, 2, 2)
plt.plot(degrees, train_mse, "o-", label="train MSE")
plt.plot(degrees, test_mse, "o-", label="test MSE")
plt.xticks(degrees)
plt.xlabel("polynomial degree")
plt.ylabel("MSE")
plt.title("Train and test MSE vs model complexity")
plt.legend()
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    pass