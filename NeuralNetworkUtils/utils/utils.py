#coding : utf-8
import math


def oneHotALabel(label, maxLabelId, onValue=1.0, offValue = 0.0):
    ret =[0] * (maxLabelId+1)
    if type(label) is list:
        for i in label:
            ret[int(i)] = onValue
    else:
        ret[int(label)] = onValue
    return ret


def oneHotLabels(labels, maxLabelId = None, onValue=1.0, offValue = 0.0):
    if maxLabelId is None:
        try:
            maxLabelId = int(max([max(k) for k in labels]))
        except:
            maxLabelId = int(max(labels))
    ret = []
    for i in labels:
        ret.append(oneHotALabel(i, maxLabelId, onValue, offValue))
    return ret


def split2Batches(batchSize, X, Y):
    batchX = []
    batchY = []
    batchNum = int(math.ceil(1.0 * X.__len__() / batchSize))
    for i in range(batchNum):
        start = batchSize * i
        end = min([batchSize * (i + 1), len(X)])
        batchX.append(X[start:end])
        batchY.append(Y[start:end])
    return batchX, batchY, batchNum
