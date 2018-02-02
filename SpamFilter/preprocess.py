# coding:utf-8
import re
from pypinyin import lazy_pinyin

from utils import textProcess
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import numpy as np
import jieba


def getfeature(str):
    feature = ""
    # 分字
    #Todo: 分词, 获取拼音, 字形码
    result = jieba.cut(str.strip())
    for r in result:
        feature = feature + r + " "
        list = lazy_pinyin(r, errors='ignore')
        feature = feature + "".join(list) + " "
    # 获取数字
    nums = re.findall(r'\d+', str)
    if len(nums) > 0:
        feature = feature + "nums"
    return feature


def rawText():
    f = open("./SpamFilter/Data/search-filter-spam.log")
    output = open("./SpamFilter/Data/search-filter-spam.format", 'w')
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
                if sen is not None and sen != "/n":
                    output.write(sen)
            except:
                print l
        except:
            pass
        count += 1
    output.close()
    f.close()


def shuffleData(indexedData, labels):
    X_sparse = coo_matrix(indexedData)
    indexedData, X_sparse, labels = shuffle(indexedData, X_sparse, labels, random_state=0)
    return indexedData, labels


def generateTrainData(file, dic, senDim):
    idsens, labels, dic = generateData(file, senDim, dic, True)
    return idsens, labels, dic

def generateTestData(file, dic, senDim):
    idsens, labels, dic = generateData(file, senDim, dic, False)
    return idsens, labels, dic

# print getfeature("今天真的好开心啊，^_^，666666")

def generateData(file, paddingSenDim=50, dic={}, changeDic = True):
    with open(file) as f:
        indexSentences = []
        labels =[]
        line = ""
        while 1:
            line = f.readline()
            if not line:
                break
            try:
                sen = line.split('spamdata=')[1].strip()
            except:
                print 'no sen:', sen
            try:
                sen = sen.decode('utf-8')
            except:
                print 'utf-8 decode error: ', sen
            try:
                indexedSen, dic = textProcess.indexSentence(sen, dic, addDict=changeDic)
                indexedSen = textProcess.padding(indexedSen, paddingSenDim)
                indexSentences.append(indexedSen)
                if line.find('isSapm=false')>=0:
                    labels.append([0, 1])
                else:
                    labels.append([1, 0])
            except:
                print 'error in building index, sen: ', line
        f.close()
        return indexSentences, labels, dic

#
# dic = {}
# idsens, labels, dic = generateData(file='/Users/nali/PycharmProjects/LearningML/SpamFilter/Data/search-filter-spam.log',
#                            senDim=40, dic=dic)
# id2Word = textProcess.reverseDic(dic)
# for l in idsens:
#     print l
#     print textProcess.id2String(l, id2Word)
# indexedData, labels, dic, reverseDic = generateTrainData(20)
# print np.sum(labels,0)