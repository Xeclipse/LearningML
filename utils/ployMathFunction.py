# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np
import mathFunction as mf


def painFunction(f, x):
    plt.plot(x, f(x))
    plt.show()


x = np.linspace(-10, 10, 300)
y= mf.sigmoid(x)
plt.plot(x, y)
plt.show()
#plt.scatter(x,mf.sigmoid(x))
#plt.show()