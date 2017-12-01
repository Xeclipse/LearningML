
from gensim.models import Word2Vec, KeyedVectors


def main():
    f=open("./listenDoc.ldoc")
    sentence=[]
    for i in f.readlines():
        sentence.append(i.strip().split(' '))
    f.close()
    model = Word2Vec(sentences=sentence,sg=1,size=100, window=5,min_count=5,negative=3,sample=0.001,hs=1)
    model.save("../ModelSave/AlbumVector/albumVector.vec")
    # model = Word2Vec.load("../ModelSave/AlbumVector/albumVector.vec")
    # id= 'start'
    # while id!='exit':
    #     id = raw_input('album_id:')
    #     ans = model.most_similar(positive=[id],topn=10)
    #     ans = [i[0] for i in ans]
    #     print '\n'.join(ans)

if __name__ == '__main__':
    main()