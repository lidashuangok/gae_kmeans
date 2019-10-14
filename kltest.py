import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import norm

#import seaborn as sns


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# def kl_divergence_1(p, q):
#     return np. sum(np.where(p != 0, p * np.log(p / q), 0))
#     tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
#                                  tf.square(tf.exp(model.z_log_std)), 1))
x = np.arange(-10, 10, 0.001)
p = norm.pdf(x, 0, 2)
q = norm.pdf(x, 2, 2)

print(x)

plt.title('KL(P||Q) = %1.3f' % kl_divergence(p, q))
plt.plot(x, p)
plt.plot(x, q, c='red')
plt.show()
