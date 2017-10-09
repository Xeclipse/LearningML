#coding:utf-8
import nltk

#retrun a generator
def xNGram(sequernce, n):
    return nltk.ngrams(sequernce,n)


a="i want you"
for i in nltk.ngrams(a,2):
    print i