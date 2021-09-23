import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

x = tf.range(-5.0, 5.0, 0.5)
xx = np.arange(-5, 5, 0.5)
n_sample = len(x)

noise = tf.random.normal((1, n_sample), 0.0, 1.0)
noise2 = np.random.normal(0, 1, n_sample)
Y = 5 * x - 6 + noise
yy = 5 * xx - 6 + noise2
plt.plot(xx, yy, 'ro')

ones = tf.ones((1, n_sample))
X = tf.concat((ones, [x]), 0)
theta = tf.Variable([[10.0], [-5.0]])
alpha = 0.01
eps = 0.0001

opt = tf.keras.optimizers.SGD(learning_rate = 0.1)
while True:
    with tf.GradientTape() as tape:
        y_hat = tf.multiply(X, theta)
        loss = tf.reduce_sum(tf.pow(y_hat-Y, 2)) / (2 * n_sample)
    grads = tape.gradient(loss, [theta])

    opt.apply_gradients(zip(grads, [theta]))

    x_vis = np.array([-5.0, 5.0])
    y_vis = theta[1][0] * x_vis + theta[0][0]
    plt.plot(x_vis, y_vis)
    plt.pause(0.05)
    if abs(grads[0][0][0].numpy()) < eps and abs(grads[0][1][0].numpy()) < eps:
        break
print('Theta toi tu: {}Gia tri nho nhat Loss{}'.format(theta.numpy(), loss.numpy()))
plt.show()