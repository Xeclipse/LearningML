# coding:utf-8
import pickle


# sentences=[
#     [word, word,...]
#     [word, word, ...]
# ]
def indexText(sentences, dic, addDict=True):
    indexedText = []
    for sen in sentences:
        indexedSen, dic = indexSentence(sen, dic, addDict)
        indexedText.append(indexedSen)
    return indexedText, dic


# sentence = [ word, word, word ]
# 0: padding
# 1: unknown
# if unkonwn index is less than 1, than the unknon symbol would NOT be added into index sentence
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


def genRelationFromSentence(tagList, sentence, model=None):
    if model is None:
        model = {}
    for i in tagList:
        if i not in model:
            subdic = {}
            model[i] = subdic
        else:
            subdic = model[i]
        for w in sentence:
            try:
                subdic[w] += 1
            except:
                subdic[w] = 1
    return model


def sortDicByKeyAndReindex(dic, startIndex = 0):
    its = dic.items()
    its = sorted(its, key=lambda x: x[0])
    count = startIndex
    sortedDic = {}
    for i in its:
        sortedDic[i[0]] = count
        count += 1
    return sortedDic

#实现一个类似于bucket的padding操作, PS:会改变传入的实参
#maxPaddingLen表示最长的padding距离
def batchPadding(all, maxPaddingLen =100):
    for i in range(len(all)):
        maxBatchLen = min([max([len(k) for k in all[i]]), maxPaddingLen])
        for k in range(len(all[i])):
            all[i][k] = padding(all[i][k], maxBatchLen)


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
            try:
                maxList = [max(k) for k in labels if len(k)>0]
            except Exception as e:
                print e.message
            maxLabelId = max( maxList )
        except:
            maxLabelId = max(labels)
    ret = []
    for i in labels:
        ret.append(oneHotALabel(i, maxLabelId, onValue, offValue))
    return ret