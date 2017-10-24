import gensim.models as gm
import gensim.corpora as gc
from gensim.models import LdaModel
import time
import matplotlib.pyplot as plt


def main():
    # read file
    file = open("/Users/nali/PycharmProjects/LearningML/TrainData/user-played-album.csv")
    line = file.readline()
    userPlayDic = {}
    train = []
    ids = []
    count = 0
    print "step1: reading file"
    line = file.readline()
    while line and count < 1000000:
        ele = line.strip().split(',')
        if ele.__len__() < 2:
            continue
        id = ele[0]
        user = ele[1]
        if not userPlayDic.has_key(id):
            userPlayDic[id] = []
        userPlayDic[id].append(user)
        line = file.readline()
        count += 1
        if count % 100000 == 0:
            print count
    file.close()
    for i, v in userPlayDic.items():
        train.append(v)
        ids.append(i)
    print count
    # preprocess
    print "step2: build dict"
    dictionary = gc.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    print "step3: LDA"
    '''
    t1=time.time()
    lda=LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)
    lda.save('./ModelSave/userPlayLda.lda')
    t2=time.time()
    print t2-t1,"seconds"
    '''
    lda = LdaModel.load('./ModelSave/userPlayLda.lda')
    print "step:4 post-process"
    maxtopic = []
    for text in corpus:
        ans = lda.get_document_topics(bow=text)
        mx = max(ans, key=lambda x: x[1])
        maxtopic.append(mx)
    outfile = open("./album_topics", 'w')

    for i in range(ids.__len__()):
        mm = maxtopic[i]
        outstr = str(ids[i]) + '\t' + str(mm[0]) + '\t' + str(mm[1]) + '\n'
        outfile.write(outstr)
    outfile.close()

if __name__ == '__main__':
    main()
