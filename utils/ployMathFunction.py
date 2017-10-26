# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np
import mathFunction as mf


def painFunction(f, x):
    plt.plot(x, f(x))
    plt.show()


x=np.linspace(0,1,1000)
y=1-np.sqrt(0.001/x)

plt.plot(x, y)
plt.show()
#plt.scatter(x,mf.sigmoid(x))
#plt.show()