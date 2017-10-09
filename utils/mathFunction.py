# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x, w=-1.0, b=1.0):
    return 1.0 / (1 + np.exp(w * x + b))


# sigmoid 导数
def deltaSigmoid(x):
    return np.exp(-1.0 * x) / np.power(1 + np.exp(-1.0 * x), 2)


def entropy(x):
    return -1 * (x * np.log(x) + (1 - x) * np.log(1 - x))


def sin(x):
    return np.sin(x)


# sigmoid积分
def SSigmoid(x):
    return x + np.log(1 + np.exp(-1 * x))


def disLogSampler(x):
    range_max = sum(x)
    return (np.log(x + 2) - np.log(x + 1)) / np.log(range_max + 1)
