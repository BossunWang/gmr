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

from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from gmr import GMM, plot_error_ellipses


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


random_state = np.random.RandomState(3)
x = np.arange(0, 20, 0.1)
y = np.sin(x)
z = y * np.sin(x)

XY_train = np.column_stack((x, y, z))
print(f'XY_train: {XY_train.shape}')
gmm = GMM(n_components=50, random_state=random_state)
gmm.from_samples(XY_train)

YZ_pred, weights = gmm.predict([0], x[:, None])
print(f'Y_pred: {YZ_pred.shape}')
class_label = np.argmax(weights, axis=-1)
print(f'class: {class_label}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
# ax = plt.subplot(121)
ax.set_title("Dataset and GMM")
ax.scatter(x, y, z, s=3, label="GT")
ax.scatter(x, YZ_pred[:, 0], YZ_pred[:, 1], s=3, label="pred")
# ax.plot_surface(x, y, z[:, None], zorder=-11, cmap=cm.twilight)
ax.legend()
plt.show()

Y = np.linspace(np.min(y), np.max(y), 100)
Z = np.linspace(np.min(z), np.max(z), 100)

target_data = []
for i, y in enumerate(Y):
    for j, z in enumerate(Z):
        target_data.append(np.array([y, z]))

target_data = np.array(target_data)
conditional_gmm = gmm.condition([0], x[0])
p_of_Y = conditional_gmm.to_probability_density(target_data)
print(p_of_Y.shape)
highest_p = np.argmax(p_of_Y)

# softmax
p_of_Y = np.exp(p_of_Y - logsumexp(p_of_Y))

print(f'Y_pred probability: {np.max(p_of_Y)}')
print(f'Y_pred from cond: {target_data[highest_p]}')