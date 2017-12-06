# coding:utf-8



# sentences=[
#     [word, word,...]
#     [word, word, ...]
# ]
def indexText(sentences, dic):
    indexedText = []
    count = 1
    for sen in sentences:
        indexSen = []
        for word in sen:
            try:
                if word not in dic:
                    dic[word] = count
                    count += 1
                indexSen.append(dic[word])
            except:
                print word
        indexedText.append(indexSen)
    return dic, indexedText


def padding(sentence, len):
    if sentence.__len__() == len:
        return sentence[:]
    elif sentence.__len__() < len:
        tmp = sentence[:]
        tmp.extend([0] * (len - sentence.__len__()))
        return tmp
    elif sentence.__len__() > len:
        return sentence[:len]


