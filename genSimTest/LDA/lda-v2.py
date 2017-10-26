import gensim.models as gm
import gensim.corpora as gc
from gensim.models import LdaModel
import time
import matplotlib.pyplot as plt


def main():
    # # read file1
    # file = open("/Users/nali/PycharmProjects/LearningML/TrainData/user-played-album-3-day.csv")
    # line = file.readline()
    # userPlayDic = {}
    # train = []
    # ids = []
    # count = 0
    # print "step1: reading file"
    # line = file.readline()
    # while line :
    #     ele = line.strip().split(',')
    #     if ele.__len__() < 2:
    #         continue
    #     id = ele[0]
    #     user = ele[1]
    #     if not userPlayDic.has_key(user):
    #         userPlayDic[user] = []
    #     userPlayDic[user].append(id)
    #     line = file.readline()
    #     count += 1
    #     if count % 100000 == 0:
    #         print count
    # file.close()
    # for i, v in userPlayDic.items():
    #     train.append(v)
    # print train.__len__()
    # # preprocess
    # print "step2: build dict"
    # dictionary = gc.Dictionary(train)
    # print dictionary.values().__len__()
    # corpus = [dictionary.doc2bow(text) for text in train]
    # print "step3: LDA"
    #ç

    # t1 = time.time()
    # lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100,iterations=50)
    # lda.save('./ModelSave/userPlayLda-2.lda')
    # t2 = time.time()
    # print t2 - t1, "seconds"

    lda = LdaModel.load('./ModelSave/userPlayLda.lda')
    print "step:4 post-process"
    print lda.get_term_topics(word_id=2887,minimum_probability=0.0000000001)
    print lda.get_term_topics(word_id=7902,minimum_probability=0.0000000001)
    # id=raw_input("input an topic id:")
    # while id!="exit":
    #     ans = lda.show_topic(topicid=int(id),topn=15)
    #     ans = [i[0] for i in ans]
    #     for a in ans:
    #         print a
    #     id = raw_input("input an topic id:")


if __name__ == '__main__':
    main()
