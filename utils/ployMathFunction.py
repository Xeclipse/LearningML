# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np
import mathFunction as mf


def painFunction(f, x):
    plt.plot(x, f(x))
    plt.show()

ratio = 0.9
addOneDay=100
x = range(50)
y= []
for i,v in enumerate(x):
    if(i==0):
        y.append(addOneDay)
    else:
        y.append(np.sqrt(y[i - 1] * ratio)+addOneDay)
plt.plot(x, y)
plt.show()
#plt.scatter(x,mf.sigmoid(x))
#plt.show()