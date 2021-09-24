import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.utils import shuffle

S1 = np.array([8, 0])
S2 = np.array([2, 5])

n_sample = 10
red_points = S1 + np.random.normal(0, 1.5, size=(n_sample, 2))
blue_points = S2 + np.random.normal(0, 1.5, size=(n_sample, 2))

@tf.function
def sigmoid(Z):
    return 1/(1 + tf.math.exp(-Z))
                                                                                 
plt.plot(red_points[:, 0], red_points[:, 1], 'ro')
plt.plot(blue_points[:, 0], blue_points[:, 1], 'bo')

X_data = np.concatenate([red_points, blue_points])

ones = np.ones(n_sample)
zeros = np.zeros(n_sample)

y_data = np.concatenate([ones, zeros])
y_data = shuffle(y_data)

def get_model():
    inputs = keras.Input(shape=(2, ))
    outputs = layers.Dense(1, activation='sigmoid')(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

opt = tf.optimizers.SGD(0.1)
model = get_model()
eps = 1e-3

while True:
    with tf.GradientTape() as tape:
        X_data = tf.cast(X_data, tf.float32)
        Y_data = tf.cast(y_data.reshape(-1,1), tf.float32)

        prediction = model(X_data)
        
        '''Loss function'''
        loss = keras.losses.binary_crossentropy(Y_data, prediction)
        grads = tape.gradient(loss, model.trainable_variables)
         
    opt.apply_gradients(zip(grads, model.trainable_variables))
    y = model.trainable_variables[0][0][0].numpy() * X_data + model.trainable_variables[0][1][0]
    
    plt.plot(X_data, y)
    plt.pause(0.05)
    
    if abs(grads[0][0][0].numpy()) < eps and abs(grads[0][1][0].numpy()) < eps:
        break

print('Theta toi tu: ', model.trainable_variables)