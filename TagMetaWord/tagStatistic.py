# coding:utf-8

import re
import nltk
import jieba
import utils.textProcess as tp


# 找到频繁出现的人工标签，它们可能是元数据
def tagStatistic(file="./Data/album_info.csv", outfile="./Data/tags.txt"):
    with open(file) as f:
        line = ""
        text = []
        count = 0
        while 1:
            count += 1
            line = f.readline()
            if not line: break
            reline = re.match("\".*\",", line)
            if reline:
                tags = reline.group().replace("\"", "").split(',')
                text.extend(tags)
            if count % 100000 == 0:
                print count / 100000,
        f.close()
        print 'finish loading'
        freq = nltk.FreqDist(text)
        src = []

        for i in freq.keys():
            if (freq[i] > 20 and len(i) <= 10 and len(i) > 1):
                src.append((i, freq[i]))
        src = sorted(src, key=lambda x: x[1], reverse=True)
        tp.saveItems(src, file="./Data/metaTags.item")


# 统计文档中的词频
def docStatistic(docsFile):
    sens = []
    with open(docsFile) as f:
        line = ""
        count = 0
        while 1:
            count += 1
            line = f.readline()
            if not line: break
            try:
                words = jieba.cut(line.strip(), HMM=True)
                if words is not None:
                    sens.extend(words)
            except:
                print line
            if count % 10000 == 0:
                print count / 10000,
        f.close()
    print 'finish reading file'
    print 'start statistic'
    freq = nltk.FreqDist(sens)
    print 'end statistic'
    return freq


# 将文档中的词频按指定格式保存到文件中
def wordCount():
    docFile = "./Data/album_info.csv"
    freq = docStatistic(docsFile=docFile)
    sortItems = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    with open("./Data/wordcount.count", 'w') as fw:
        for i, v in sortItems:
            try:
                fw.write(i.encode("utf-8") + '\t' + str(v) + '\n')
            except:
                print i
        fw.close()


# 判断一个Item是否是合法的Item，这里的Item是指（词，词频）这样格式的元组
def isValidateItem(item):
    if (len(item[0].decode('utf-8')) > 1 and int(item[1]) > 5):
        return True
    return False


# 过滤词频文件，将合法的词和其词频保存到新的文件里，以.dic结尾，表明这是一个dict类型的实例
def readAndFilterDic():
    ret = {}
    with open("./Data/wordcount.count") as f:
        for l in f.readlines():
            try:
                item = l.strip().split('\t')
                if isValidateItem(item):
                    ret[item[0].decode("utf-8")] = int(item[1])
            except:
                print l
        f.close()
    return ret


# words = readAndFilterDic()
# tp.saveDict(words, "./Data/wordcount.dic")
# print words.__len__()


# 将词频文件里的词做成一个词-index的字典，便于之后进行操作，节约存储空间
def indexWordFromWordCountDic():
    wordCountDic = tp.loadDict("./Data/wordcount.dic")
    ret = {}
    count = 2
    for i in wordCountDic.keys():
        try:
            ret[i] = count
            count += 1
        except:
            print i
    return ret


# indexedWordDic = indexWordFromWordCountDic()
# tp.saveDict(indexedWordDic, "./Data/wordIndex.dic")
# print indexedWordDic.__len__()

def processSen(tagList, sen, relatedTable):
    sen = list(set(sen))
    for tag in tagList:
        try:
            subDic = relatedTable[tag]
        except:
            subDic = {}
            relatedTable[tag] = subDic
        for k in sen:
            try:
                subDic[k] += 1
            except:
                subDic[k] = 1


def keepUsefulWithDict(items, dic):
    ret = []
    for i in items:
        if i in dic:
            ret.append(i)
    return ret


def genRelatedWord(metaDic, wordDic, docFile):
    retTable = {}
    with open(docFile) as f:
        count = 0
        while 1:
            count += 1
            line = f.readline()
            if not line: break
            reline = re.match("\".*\",", line)
            if reline:
                try:
                    tags = reline.group()
                    otherSen = line.replace(tags, "")
                    words = jieba.cut(otherSen.strip(), HMM=True)
                    words = keepUsefulWithDict(words, wordDic)
                    tags = tags.replace("\"", "").split(',')
                    tags = keepUsefulWithDict(tags, metaDic)
                    if tags is None or len(tags) == 0 or words is None or len(words) == 0:
                        continue
                    processSen(tags, words, retTable)
                except:
                    print "error in analysis:", line
            if count % 10000 == 0:
                print count / 10000,
        f.close()
        print 'finish loading'
    return retTable


# print "load vocabulary..."
# wordIndex = tp.loadDict("./Data/wordIndex.dic")
# print "load metaId..."
# metaCount = tp.items2Dic(tp.loadItems("./Data/metaTags.item"))
# print "gen related map"
# relatedTabel = genRelatedWord(metaCount, wordIndex, "./Data/album_info.csv")
# print "saving"
# tp.saveDict(relatedTabel, "./Data/relatedWordMap.dic")
# print "finish"

relatedMap = tp.loadDict("./Data/relatedWordMap.dic")
while 1:
    meta = raw_input("meta:")
    dic = relatedMap[meta]
    diclen = len(dic)
    for i in dic:
        if 1.0 * dic[i] / diclen >= 0.01:
            print i, dic[i]
