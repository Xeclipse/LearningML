# -*- coding:utf-8 -*-
import re
import pypinyin


def addSpace(str):
    ret = ""
    str = str.decode("utf-8")
    for w in str:
        ret += w + " "
    return ret.encode("utf-8")


def translateAndPinyinAndFormat():
    splitedSentence = []
    consective = []
    strInfo = re.compile(r"(,|\.|\?|！|？|，|》|《|。| 。|、|：|/|~|…|（|）|&|\*)+")
    digitReplace = re.compile(r"[a-zA-Z0-9]+")
    spaceReplace = re.compile(r"\s+")

    f = open("../TrainData/reflect.txt")
    wf = open("../ResultData/formatSen.txt", 'w')
    for l in f.readlines():
        l = strInfo.sub(" ", l.strip())
        l = digitReplace.sub(" ", l)
        l = spaceReplace.sub(" ", l)
        s = l.split(" ")
        for i in s:
            if i != " " and i != "\n" and i.__len__() > 5:
                consective.append(i)
                splitedSentence(addSpace(i))
                wf.write(addSpace(i) + "\n")
    wf.close()
    f.close()

    pwf = open("../ResultData/pinyinSen.txt", 'w')
    for i in consective:
        pinyin = pypinyin.lazy_pinyin(i.decode("utf-8"))
        pwf.write(" ".join(pinyin).encode("utf-8") + "\n")
    pwf.close()
    return consective, splitedSentence


def indexFile(file, outfile, dicFile):
    f = open(file)
    wf = open(outfile, 'w')
    dic = {}
    count = 0
    for i in f.readlines():
        out = []
        eles = i.decode("utf-8").strip().split(' ')
        for k in eles:
            index = -1
            if k==' ': continue
            if dic.has_key(k):
                index = dic[k]
            else:
                dic[k] = count
                index = count
                count += 1
            out.append(str(index))
        wf.write(" ".join(out)+"\n")
    wf.close()


    dicWriter = open(dicFile, 'w')
    for it in dic.items():
        dicWriter.write(it[0].encode("utf-8"))
        dicWriter.write("\t")
        dicWriter.write(str(it[1])+"\n")
    dicWriter.close()

# indexFile("../ResultData/formatSen.txt", "../ResultData/indexFormatSen.txt", "../ResultData/dicFormatSen.txt")
indexFile("../ResultData/pinyinSen.txt", "../ResultData/indexPinyinSen.txt", "../ResultData/dicPinyinSen.txt")