# coding:utf-8

from utils import textProcess
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import numpy as np


def rawText():
    f = open("./Data/search-filter-no-spam.log")
    output = open("./Data/search-filter-no-spam.format", 'w')
    data = []
    count = 0
    while 1:
        if count % 10000 == 0: print count
        try:
            l = f.readline()
            if l is None or l == "":
                break
            try:
                sen = l.split('spamdata=')[1]
                # data.append(sen)
                output.write(sen)
            except:
                pass
        except:
            pass
        count += 1
    output.close()
    f.close()


def generateTrainData():
    labels = []
    f = open("/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/search-filter-no-spam.format")
    sentences = [sen.strip() for sen in f.readlines() if sen!="\n"]
    f.close()
    labels.extend([[0,1]]*sentences.__len__())
    dic = {}
    dic, indexedData = textProcess.indexText(sentences, dic)
    indexedData = [textProcess.padding(l, 50) for l in indexedData]
    f = open("/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/search-filter-spam.format")
    sentences2 = [sen.strip() for sen in f.readlines() if sen!="\n"]
    labels.extend([[1,0]]*sentences2.__len__())
    f.close()
    dic, indexedData2 = textProcess.indexText(sentences2, dic)
    indexedData.extend(indexedData2)
    indexedData= [textProcess.padding(s,50) for s in indexedData]

    X_sparse = coo_matrix(indexedData)
    indexedData, X_sparse, labels = shuffle(indexedData, X_sparse, labels, random_state=0)

    return indexedData, labels, dic

