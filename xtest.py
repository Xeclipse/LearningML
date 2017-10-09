# coding:utf-8


import nltk
import jieba
import pandas
from collections import Counter

df = pandas.read_csv("./TrainData/Train.csv", "utf-8", engine='python', header=0, delimiter=',')

content_train = df["describe"]  # 第一列为文本内容，并去除列名
opinion_train = df['class']  # 第二列为类别，并去除列名

print('训练集有 %s 条句子' % (len(content_train)))
train = [content_train, opinion_train]
# c=Counter()
# for l in train[1]:
#     try:
#         words = jieba.cut(l)
#         c.update(words)
#         #print " ".join(words)
#     except:
#         print l
# print c
# sc=sorted(c.items(),key=lambda x:x[1],reverse=True)
# f= open("labels.txt",'w')
# for i in sc:
#     f.write(i[0].encode('utf-8')+':'+str(i[1]).encode('utf-8')+'\n')
f = open("reflect.txt", 'w')
for i in train[0]:
    f.write(i.decode("gbk", "ignore").encode('utf-8'))
    f.write('\n')
