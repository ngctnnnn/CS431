import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


S1= np.array([[8],[0]])
S2= np.array([[2],[5]])

n_sample = 10
red_points = S1 + np.random.normal(0, 1.5, size=(2,n_sample))
blue_points = S2 + np.random.normal(0, 1.5, size=(2,n_sample))

X_1 = tf.convert_to_tensor(red_points, dtype=tf.float32)
X_2 = tf.convert_to_tensor(blue_points, dtype=tf.float32)

x0, y0 = X_1 
x1, y1 = X_2

X = tf.concat((X_1, X_2), 1)

weights = tf.Variable([[-1.0], [2.0]])

alpha = 1e-3
lr = 0.1
const = tf.constant(1.0, dtype=tf.float32)
y = []
for i in range(n_sample):
    tmp = np.random.randint(0, 2)
    y.append(tmp*const)

Y = np.array(y)
Y = tf.convert_to_tensor(y, dtype=tf.float32)

@tf.function
def sigmoid(Z):
    return 1/(1 + tf.math.exp(-Z))

opt = tf.keras.optimizers.SGD(learning_rate = 0.1)

eps = 1e-4


y = tf.transpose(y)
print(y)

@tf.function
def loss_function(theta, X, Y):
    return tf.math.log(sigmoid(tf.transpose(theta) * X)) + (1.0 - Y)*tf.math.log(1.0 - sigmoid(tf.transpose(theta) * X))

while True:
    with tf.GradientTape() as tape:
        loss_value = sigmoid(tf.matmul(weights, X))
        # loss_value = loss_function(weights, X, Y)
    grads = tape.gradient(loss_value, [weights])
    
    opt.apply_gradients(zip(grads, [weights]))
    
    
    
    x_vis = np.array([0, 2])
    y_vis = weights[1][0] * x_vis + weights[0][0]
    plt.plot(x_vis, y_vis)
    plt.pause(0.05)
    if abs(grads[0][0].numpy()) < eps and abs(grads[0][1].numpy()) < eps:
        break

print(weights.numpy(), loss_value.numpy())

plt.scatter(red_points, blue_points)
plt.plot(y)
plt.show()