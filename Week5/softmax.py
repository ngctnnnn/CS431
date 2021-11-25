import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import sparse

means = [[2, 2], [8, 2], [2, 7], [8, 7]]
cov = [[1, 0], [0, 1]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N) 
X1 = np.random.multivariate_normal(means[1], cov, N) 
X2 = np.random.multivariate_normal(means[2], cov, N) 
X3 = np.random.multivariate_normal(means[3], cov, N) 

X = np.concatenate((X0, X1, X2, X3), axis=0).T
X = np.concatenate((np.ones((1, 4 * N)), X), axis = 0)
C = 4

label = np.asarray([0]*N + [1]*N + [2]*N + [3]*N).T

def convert_labels(y, C = C):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y 

Y = convert_labels(label, C)

theta = tf.Variable(np.array([[1., 1., 1., 1.],
                              [2., 2., 2., 2.],
                              [3., 3., 3., 3.]]))
eps = 1e-4
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
cnt = 0
while True:
    with tf.GradientTape() as tape:
        f = tf.linalg.matmul(tf.transpose(theta), X)
        Y_ = tf.nn.softmax(f, axis=0) 
        loss = -tf.reduce_mean(Y * tf.math.log(Y_))
        print("loss :", loss.numpy())
    grads = tape.gradient(loss, [theta]) 
    print("gradient :", grads[0][0][1].numpy())
    opt.apply_gradients(zip(grads, [theta]))
    
    if abs(grads[0][0][1].numpy()) < eps and cnt > 0:
        break
    cnt = 1

f = tf.linalg.matmul(tf.transpose(theta), [[1.0, 2.0], [2.0, 3.0], [4.0, 1.0]])
Y_ = tf.nn.softmax(f, axis=0) 

full = [] 
a = [] 
b = [] 
c = []
for i in range(0, 11): 
    for j in range(0, 11): 
        a.append(1.0)
        b.append(i * 1.0) 
        c.append(j * 1.0) 

full.append(a) 
full.append(b)
full.append(c)
f = tf.linalg.matmul(tf.transpose(theta), full)
Y_ = tf.nn.softmax(f, axis=0) 

xx1, yy1 = [], []
xx2, yy2 = [], []
xx3, yy3 = [], []
xx4, yy4 = [], []

for i in range(0, len(Y_[0])): 
    cls = [Y_[0][i], Y_[1][i], Y_[2][i], Y_[3][i]] 
    k = tf.math.argmax(cls)

    if k == 0: 
        xx1.append(b[i])
        yy1.append(c[i]) 
    if k == 1: 
        xx2.append(b[i]) 
        yy2.append(c[i])
    if k == 2: 
        xx3.append(b[i]) 
        yy3.append(c[i])
    if k == 3: 
        xx4.append(b[i]) 
        yy4.append(c[i])
plt.plot(X0[:, 0], X0[:, 1], 'bo')
plt.plot(X1[:, 0], X1[:, 1], 'go')
plt.plot(X2[:, 0], X2[:, 1], 'ro')
plt.plot(X3[:, 0], X3[:, 1], 'mo') 
plt.plot(xx1, yy1, 'bo')
plt.plot(xx2, yy2, 'go')
plt.plot(xx3, yy3, 'ro')
plt.plot(xx4, yy4, 'mo')
plt.show()