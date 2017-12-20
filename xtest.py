# coding:utf-8


import nltk
import jieba
import pandas
from collections import Counter
import math
import numpy as np


def StirlingApproximation(n):
    return np.sqrt(2 * np.pi) * np.power(n, n + 0.5) / np.exp(n)
