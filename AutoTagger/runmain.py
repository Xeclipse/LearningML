# coding:utf-8

from AutoTagger.Data import dataAPI
from utils import textProcess as tp
from AutoTagger.Data.dataAPI import AlbumMeta
import pickle
from AutoTagger.models.AttentionRNNClassifier import AlbumTextClassifier

# todo: 在dataAPI 中存在一些问题，有时间需要查找问题来fix这个坑爹的bug
# todo：1。有的meta无法index到
# todo：2。有的tags、title、intro index完之后是空的

# print 'start loading'
# dic =tp.loadDict('./Data/dictionaries/ch2index')
# albums = dataAPI.getRawAlbums('./Data/albums.dump')
# print 'finish loading'
# voiceBook = {}
# for i,v in albums.items():
#     if v.album.category_id==3:
#         voiceBook[i] = v
# albums = None
# metaDic = tp.loadDict('./Data/dictionaries/meta2index')
# print 'indexing'
# indexedAlbums = dataAPI.indexAlbumTexts(voiceBook, dic, metaDic)
# print 'finishIndexing'
# with open('./Data/indexingAlbum3','w') as f:
#     pickle.dump(indexedAlbums,f)
# debugPoint =0
#
#
# albums=indexedAlbums
# # with open('./Data/indexingAlbum3') as f:
# #     albums = pickle.load(f)
# tags=[]
# titles=[]
# intros=[]
# metas=[]
# for i,v in albums.items():
#     tags.append(v['tag'])
#     titles.append(v['title'])
#     intros.append(v['intro'])
#     metas.append(v['meta'])
# X = [tags, titles, intros]
# Y= metas
# with open('./Data/trainX','w') as f:
#     albums = pickle.dump(X,f)
# with open('./Data/trainY','w') as f:
#     albums = pickle.dump(Y,f)

#
with open('./Data/trainX') as f:
    X = pickle.load(f)
with open('./Data/trainY') as f:
    Y = pickle.load(f)
print 'finish loading '
Y = tp.oneHotLabels(Y)
dic =tp.loadDict('./Data/dictionaries/ch2index')
atc = AlbumTextClassifier()
atc.displayStep=1
atc.tensorBoardPath = "./Record/TensorBoard/TestTrial"
atc.modelPath ="./Record/modelRecord/testModel.rnn"
atc.vocabDim = len(dic)+2
atc.numLabels = 220
atc.train(X,Y,1000,3)
