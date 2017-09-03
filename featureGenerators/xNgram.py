#coding:utf-8
import nltk

#retrun a generator
def xNGram(sequernce, n):
    return nltk.ngrams(sequernce,n)


def xTfidf():
    pass


seq=u"12345"
for i in nltk.bigrams(seq):
    print i