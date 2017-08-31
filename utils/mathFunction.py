# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x, w=-1.0, b=1.0):
    return 1.0 / (1 + np.exp(w * x + b))


def deltaSigmoid(x):
    return np.exp(-1.0 * x) / np.power(1 + np.exp(-1.0 * x), 2)


def sin(x):
    return np.sin(x)


