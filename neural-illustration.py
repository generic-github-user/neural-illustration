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
