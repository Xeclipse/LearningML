from gensim.models import Word2Vec
import sklearn as sk
from sklearn.cluster import KMeans

print 'loading data ....'
model = Word2Vec.load("../ModelSave/AlbumVector/albumVector.vec")
word_vectors = model.wv
count = 0
data = []
albumid = []
for i in word_vectors.vocab:
    albumid.append(i)
    data.append(word_vectors[i])
print 'data size:', data.__len__()
print 'kmeans ...'
n_digits = 100
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=100)
y = kmeans.fit_predict(X=data)
albumPred = [(albumid[i], y[i]) for i in range(y.__len__())]
albumPred = sorted(albumPred, key=lambda x: x[1])

print 'write result'
file = open("albumCategory", 'w')
for i in albumPred:
    outstr = i[0] + '\t' + str(i[1]) + '\n'
    file.write(outstr)
