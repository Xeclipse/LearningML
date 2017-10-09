# -*- coding:utf-8 -*-
import pandas

import csv
import jieba
#jieba.load_userdict('wordDict.txt')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


# 读取训练集
def readtrain():
    df = pandas.read_csv("./TrainData/Train.csv", "utf-8", engine='python', header=0, delimiter=',')

    content_train = df["describe"]  # 第一列为文本内容，并去除列名
    opinion_train = df['class']  # 第二列为类别，并去除列名

    print('训练集有 %s 条句子'%(len(content_train)))
    train = [content_train, opinion_train]
    return train


# 将utf8的列表转换成unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode('utf8'))
    return a


# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        if i is np.nan:
            continue
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c


# corpus = ["我 来到 北京 清华大学", "他 来到 了 网易 杭研 大厦", "小明 硕士 毕业 与 中国 科学院"]
train = readtrain()
content = segmentWord(train[0])
opinion = train[1]

# 划分
train_content = content[:5243]
test_content = content[5243:]
train_opinion = opinion[:5243].tolist()
test_opinion = opinion[5243:].tolist()


# 计算权重
vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  # 先转换成词频矩阵，再计算TFIDF值
print(tfidf.shape)


# 单独预测
'''
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
# 分类器
clf = MultinomialNB().fit(tfidf, opinion)
docs = ["在 标准 状态 下 途观 的 行李厢 容积 仅 为 400 L", "新 买 的 锋驭 怎么 没有 随 车 灭火器"]
new_tfidf = tfidftransformer.transform(vectorizer.transform(docs))
predicted = clf.predict(new_tfidf)
print predicted
'''


# 训练和预测一体
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear'))])
text_clf = text_clf.fit(train_content, train_opinion)
predicted = text_clf.predict(test_content)
print('SVC',np.mean(predicted == test_opinion))
#print test_opinion
print(set(predicted))
#print metrics.confusion_matrix(test_opinion,predicted) # 混淆矩阵

result = pd.DataFrame(columns=['content', 'predicted'])
result['content'] = test_content
result['predicted'] = predicted.tolist()
#print result.head()
result.to_csv("test.csv", encoding="utf-8")



