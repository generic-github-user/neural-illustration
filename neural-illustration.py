import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt
import time

R = 20
timesteps = 1

resolution = (R, R)
sample = np.zeros(resolution)
sample[0:R//3, 0:R//3] = 1
# neural illustration via dynamic programming

actions = np.zeros((1, (timesteps * 3 + 1)))
rewards = np.zeros((1, 1))

actor = tf.keras.models.Sequential([
    # layers.InputLayer(input_shape=(1)),
    # layers.Dense(1, batch_input_shape=(1,)),
    layers.Dense(1),
    layers.RepeatVector(timesteps),
    layers.Dense(10, activation='tanh'),
    layers.LSTM(10, stateful=False, return_sequences=True),
    layers.Dense(10, activation='tanh'),
    # layers.Dense(5, activation='tanh'),
    # layers.BatchNormalization(),
    layers.Dense(3, activation='tanh'),
    layers.Rescaling(10),
    # layers.LayerNormalization()
    layers.Flatten()
])

critic = tf.keras.models.Sequential([
    # layers.Dense(4),
    layers.BatchNormalization(),
    layers.Dense(16, activation='tanh'),
    # layers.Dense(10, activation='tanh'),
    layers.Dense(10, activation='tanh'),
    layers.Dense(1, activation='relu')
])
