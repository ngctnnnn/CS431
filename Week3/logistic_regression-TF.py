import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

S1= np.array([8, 0])
S2= np.array([2, 5])

n_sample = 10
red_points = S1 + np.random.normal(0, 1.5, size=(n_sample, 2))
blue_points = S2 + np.random.normal(0, 1.5, size=(n_sample, 2))
                                                                                 
eps = 0.01

plt.plot(red_points[:, 0], red_points[:, 1], 'ro')
plt.plot(blue_points[:, 0], blue_points[:, 1], 'bo')

X_data = np.concatenate([red_points, blue_points])

ones = np.ones(n_sample)
zeros = np.zeros(n_sample)

y_data = np.concatenate([ones, zeros])


def get_model():
    inputs = keras.Input(shape=(2, ))
    outputs = layers.Dense(1, activation='sigmoid')(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.optimizers.SGD(0.1)
model = get_model()

while True:
    with tf.GradientTape() as tape:
        X_data_ = tf.cast(X_data, tf.float32)
        y_data_ = tf.cast(y_data.reshape(-1,1), tf.float32)

        pred = model(X_data_)
        loss = keras.losses.binary_crossentropy(y_data_, pred)
        grads = tape.gradient(loss, model.trainable_variables)
 
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    x_vis = X_data
    y_vis = model.trainable_variables[0][0].numpy() * x_vis + model.trainable_variables[0][1]
    
    plt.plot(x_vis, y_vis)
    plt.pause(0.01)
    
    if abs(grads[0][0].numpy()) < eps and abs(grads[0][1].numpy()) < eps:
        break

print('Theta toi tu: ', model.trainable_variables)