# coding:utf-8


import nltk
import jieba
import pandas
from collections import Counter
import math
import numpy as np
import tensorflow as tf

# def StirlingApproximation(n):
#     return np.sqrt(2 * np.pi) * np.power(n, n + 0.5) / np.exp(n)
#
# print StirlingApproximation(12)
# from utils import mathFunction
from SpamFilter import preprocess
from utils import textProcess

# preprocess.rawText()

a = 10
b = 20
a = a ^ b
b = a ^ b
a = a ^ b

print a,b
