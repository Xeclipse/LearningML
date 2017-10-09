# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np
import mathFunction as mf


def painFunction(f, x):
    plt.plot(x, f(x))
    plt.show()


x = np.linspace(0, 30, 30)
# painFunction(mf.SSigmoid, x)
plt.scatter(x,mf.disLogSampler(x))
plt.show()