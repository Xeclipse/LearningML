# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np
import mathFunction as mf


def painFunction(f, x):
    plt.plot(x, f(x))
    plt.show()




x = np.linspace(-10, 10, 500)
painFunction(mf.deltaSigmoid, x)
