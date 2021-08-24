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

actor_optimizer = optimizers.Adam(learning_rate=0.0003)
critic_optimizer = optimizers.Adam(learning_rate=0.0004)
actor_history = []
critic_history = []

def generate(action_buffer=actions, reward_buffer=rewards, log=False, noise=1):
    # actor.reset_states()
    canvas = np.zeros(resolution)
    grid = np.stack(np.mgrid[-10:10:complex(imag=R), -10:10:complex(imag=R)])
    inputs = np.ones((1, 1))
    outputs = actor(inputs).numpy()
    # for timestep in outputs[0]:
        # print(timestep)
    action_noise = np.random.normal(0, 2, [timesteps * 3])
    # action_noise[::3] = np.random.normal(0, 0.5, [5])
    action_noise *= noise
    outputs[0] += action_noise
    if log:
        print(outputs)
    for timestep in np.split((outputs[0]), timesteps):
        x, y, r = timestep
        r = np.abs(r)
        brush = np.linalg.norm(grid - np.array([x, y])[..., None, None], ord=2, axis=0)# < 2
        # canvas += brush
        # canvas = np.where(brush, 1, canvas)
        canvas += (1 / (brush * 0.2)) ** 0.01
        # canvas += (10 - brush) ** 2
    state_vector = np.concatenate([inputs, outputs], axis=1)
    # print(state_vector.shape)
    canvas -= canvas.min()
    canvas /= canvas.max()
    action_buffer = np.append(action_buffer, state_vector, axis=0)
    reward_buffer = np.append(reward_buffer, np.mean(np.abs(canvas - sample) ** 1)[None, ..., None], axis=0)
    if log:
        print(action_buffer)
        print(reward_buffer)
    return action_buffer, reward_buffer, canvas
