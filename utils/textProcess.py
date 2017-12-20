# coding:utf-8
import pickle


# sentences=[
#     [word, word,...]
#     [word, word, ...]
# ]
def indexText(sentences, dic):
    indexedText = []
    for sen in sentences:
        indexedSen, dic = indexSentence(sen, dic)
        indexedText.append(indexedSen)
    return indexedText, dic


# sentence = [ word, word, word ]
# 0: padding
# 1: unknown
def indexSentence(sentence, dic, addDict=True, unknownIndex=1):
    count = dic.__len__() + 2
    indexSen = []
    try:
        for word in sentence:
            try:
                if word not in dic:
                    if addDict:
                        indexSen.append(count)
                        dic[word] = count
                        count += 1
                    else:
                        if unknownIndex > 0:
                            indexSen.append(unknownIndex)
                else:
                    indexSen.append(dic[word])
            except:
                print word
    except:
        print "sentence is not in right format"
    return indexSen, dic


# padding with 0
def padding(sentence, len):
    if sentence.__len__() == len:
        return sentence[:]
    elif sentence.__len__() < len:
        tmp = sentence[:]
        tmp.extend([0] * (len - sentence.__len__()))
        return tmp
    elif sentence.__len__() > len:
        return sentence[:len]


def reverseDic(dic):
    ret = {}
    try:
        for k, v in dic.items():
            ret[v] = k
    except:
        print 'reverse dict error'
    return ret


def id2String(data, dic):
    str = ""
    for i in data:
        if i == 0: break
        if i in dic:
            str += dic[i]
        else:
            str += '$UNC$'
    return str


def saveDict(dic, file):
    with open(file, 'w') as f:
        pickle.dump(dic, f)


def loadDict(file):
    with open(file) as f:
        return pickle.load(f)


def saveItems(items, file, splitTag='\t'):
    with open(file, 'w') as fw:
        for i, v in items:
            try:
                fw.write(i+ splitTag + str(v) + '\n')
            except:
                print i
        fw.close()


def loadItems(file, splitTag='\t'):
    items = []
    with open(file) as f:
        for l in f.readlines():
            ele = l.strip().split(splitTag)
            items.append((ele[0], ele[1]))
        f.close()
    return items

def items2Dic(items):
    ret={}
    for i,v in items:
        ret[i]=v
    return ret