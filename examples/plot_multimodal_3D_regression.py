"""
=====================
Multimodal Regression
=====================

In multimodal regression we do not try to fit a function f(x) = y but a
probability distribution p(y|x) with more than one peak in the probability
density function.

The dataset that we use to illustrate multimodal regression by Gaussian
mixture regression is from Section 5 of

C. M. Bishop, "Mixture Density Networks", 1994,
https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf

On the left side of the figure you see the training data and the fitted
GMM indicated by ellipses corresponding to its components. On the right
side you see the predicted probability density p(y|x=0.5). There are
three peaks that correspond to three different valid predictions. Each
peak is represented by at least one of the Gaussians of the GMM.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses


random_state = np.random.RandomState(3)
x = np.arange(0, 20, 0.1)
y = np.sin(x)
z = y * np.sin(x)

XY_train = np.column_stack((x, y, z))
print(f'XY_train: {XY_train.shape}')
gmm = GMM(n_components=30, random_state=random_state)
gmm.from_samples(XY_train)

plt.figure(figsize=(10, 5))

ax = plt.subplot(121)
ax.set_title("Dataset and GMM")
ax.scatter(x, y, s=1)
colors = ["r", "g", "b", "orange"]
# plot_error_ellipses(ax, gmm, colors=colors)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax = plt.subplot(122)
ax.set_title("Conditional Distribution")
Y = np.linspace(np.min(y), np.max(y), 100)
Z = np.linspace(np.min(z), np.max(z), 100)
test_index = 10
X_test = x[test_index]
Y_GT = y[test_index]
Z_GT = z[test_index]
print(f"test data: {X_test}, {Y_GT}, {Z_GT}")

X_test_array = np.array([X_test])
Y_pred = gmm.predict([0], X_test_array[:, None])
print(f'Y_pred: {Y_pred}')

target_data = []
for i, y in enumerate(Y):
    for i, z in enumerate(Z):
        target_data.append(np.array([y, z]))

target_data = np.array(target_data)
conditional_gmm = gmm.condition([0], [X_test])
p_of_Y = conditional_gmm.to_probability_density(target_data)
highest_p = np.argmax(p_of_Y)
print(f'Y_pred from cond: {target_data[highest_p]}')


plt.tight_layout()
plt.show()
