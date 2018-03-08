# coding:utf-8

import sys


sys.path.append('/usr/local/NeuralNetworkTask/LearningML3/')
from AutoTagger.models.HierarchicalLabelClassifier import HierarchicalAlbumTextClassifier
from AutoTagger.models.ComplexAttentionRNNClassifier import ComplexAlbumTextClassifier

from AutoTagger.models.MultiLayerAttentionRNNClassifier import MultiAlbumTextClassifier
from AutoTagger.Data import dataAPI
from utils import textProcess as tp
from AutoTagger.Data.dataAPI import AlbumMeta, generateAlbumMetaObjectsAndSaveToFile
import pickle
from AutoTagger.models.AttentionRNNClassifier import AlbumTextClassifier
from AutoTagger.Data.dataAPI import MetaInfo

# todo: 在dataAPI 中存在一些问题，有时间需要查找问题来fix这个坑爹的bug
# todo：1。有的meta无法index到: 解决，原因是因为有的专辑不止一个category——id， 所以混进了别的category_id 下的meta
# todo：2。有的tags、title、intro index完之后是空的




# print 'start loading'
# dic = tp.loadDict('./Data/dictionaries/ch2index')
# albums = dataAPI.getRawAlbums('./Data/albums.dump')
# print 'finish loading,', len(albums), 'has read'
# voiceBook = {}
# for i, v in albums.items():
#     if v.album.category_id == 3:
#         voiceBook[i] = v
# albums = None
# metaDic = tp.loadDict('./Data/dictionaries/meta2index')
#
# print 'indexing'
# indexedAlbums = dataAPI.indexAlbumTexts2Level(voiceBook, dic, metaDic)
# print 'finishIndexing', len(indexedAlbums), 'will be saved'
# with open('./Data/indexedAlbum3', 'w') as f:
#     pickle.dump(indexedAlbums, f)
# debugPoint = 0
# albums = indexedAlbums

# with open('./Data/indexedAlbum3') as f:
#     albums = pickle.load(f)
# tags = []
# titles = []
# intros = []
# metas1 = []
# metas2 = []
# albumItems = sorted(albums.items(), key=lambda x: len(x[1]['intro']))
# for i, v in albumItems:
#     tags.append(v['tag'])
#     titles.append(v['title'])
#     intros.append(v['intro'])
#     metas1.append(v['meta1'])
#     metas2.append(v['meta2'])
# X = [tags, titles, intros]
# Y = [metas1, metas2]
# with open('./Data/trainX','w') as f:
#     albums = pickle.dump(X,f)
# with open('./Data/trainY','w') as f:
#     albums = pickle.dump(Y,f)


with open('./Data/trainX') as f:
    X = pickle.load(f)
with open('./Data/trainY') as f:
    Y = pickle.load(f)
print 'finish loading '
Y[0] = tp.oneHotLabels(Y[0])
Y[1] = tp.oneHotLabels(Y[1])

#
# print X[0][0]
# print X[1][0]
# print X[2][0]
# print Y[0][0]
# print Y[1][0]

# metaDic = tp.reverseDic(tp.loadDict('./Data/dictionaries/meta2index')['l2meta2id'])
# albumIndex = 3
# print Y[0][albumIndex]
# print Y[1][albumIndex]
# dic = tp.loadDict('./Data/dictionaries/ch2index')
# id2chdic = tp.reverseDic(dic)
# print tp.id2String(X[0][albumIndex], id2chdic)
# print tp.id2String(X[1][albumIndex], id2chdic)
# print tp.id2String(X[2][albumIndex], id2chdic)


atc = HierarchicalAlbumTextClassifier()
dic = tp.loadDict('./Data/dictionaries/ch2index')
atc.displayStep = 0
atc.vecDim = 2
atc.tensorBoardPath = "./Record/TensorBoard/TestTrial"
atc.modelPath = "./Record/modelRecord/testModel.rnn"
atc.vocabDim = len(dic) + 2
atc.numLabelsLevel1 = len(Y[0][0])
atc.numLabelsLevel2 = len(Y[1][0])
atc.train(X, Y, True, 100, 1, startStep=0)
# atc.outputGraph(file = './Record')

# title = u"回到大明当才子"
# intro = u"自古有云，冬困秋乏夏打盹，睡不醒的春三月。古人诚不欺我，这话一点都错不了，太阳都已经照到屁股了，大明山东省东昌府临清城一座豪华土气得厉害的大宅院里，一个二十来岁的年轻人就还趴在床上呼呼大睡，嘴里除了发出一阵高过一阵的鼾声外，还不时的说几句梦话，念念几个临清城里鸳鸯楼红牌姑娘的名字，睡得十分香甜。而房间外面来往的丫鬟仆人虽多，却没有一个人敢于发出一点声音，全都是轻手轻脚仿佛做贼，连喘气都不敢大声，生怕惊醒了这位享福无比的大少爷，招来少爷或者老爷再或者夫人的一顿训斥，乃至毒打！ 水面平静的下面是暗流汹涌，张大少爷看上去是在床上睡得贼香贼甜，可谁也不知道的是，此时此刻，就在张大少爷的睡梦中，一场针对张大少爷身体控制权的争夺，已经在如火如荼的展开……"
# tags = u"回到大明当才子,奇幻,有声小说,穿越,言情"


# title = u"凤临天下：王妃十三岁"
# intro = u"她是冷面无敌的雇佣兵首领，一着穿越成为十三岁小王妃。他是帝国的绝色王爷，铁血冷酷，威震天下。天下风云涌动，七国争霸，群雄逐鹿，她与他金戈铁马、血腥沙场，一路平雪圣，扫傲国，吞陈国，灭后金，并南宋，夺赵国，完成称霸天下，一统七国的霸业。大婚典礼上，冥岛的半个月之约，却又将他们卷进一场惊天的阴谋中。他与她率万千船只扬帆起航，炸敌营、渡洛河、斩雪蛇、过迷阵，一路步步惊心，等待他们的又是怎样惊心动魄的迷局？ 乌龙穿越，废材晋升武术天才，　小虾米变身大白鲨，请务必小心谨慎！ 　　身为世界第一的佣兵首脑，竟然因为撞上车门而灵魂穿越，附身在天辰国慕容大将军庶出的孙女——慕容琉月身上！？这要是传出去，真会笑掉人家的大牙！ 　　不过，这场乌龙穿越好像也不全然是坏事，至少目前对她来说，是极具挑战性的——"
# tags = u"穿越,言情,王妃,奇幻,暖心"
#
# title = u"谁都别惹我作者张小花播讲有一头熊"
# intro = u"顺序已经调好 抱歉了 各位 四大天王掉下来了、吕洞宾掉下来了、李靖和哪吒掉下来了、七仙女掉下来了、阎王爷掉下来了…… 我在街边摊套了一个布娃娃会说话，它说它是天界娃娃，要吸取人间喜怒哀乐愁，还钦命我为第一帮凶…… 我是甄廷强，我很烦，你们谁都别惹我！ ......"
# tags = u"玄幻,科幻,穿越"

# title = u"一个：很高兴见到你"
# intro = u"2013年夏，韩寒亲任主编，经过14个月精心打磨，终于推出文艺主题书系《一个》。 系列的 第一部叫做《很高兴见到你》，收入28篇精选之作，其中有韩寒最新作品《一次告别》《井与陆地，海和岛屿》；有“ONE一个”APP人气文章，张晓晗、颜茹玉、咪蒙、荞麦、蔡崇达、暖小团等未来文学之星齐聚一堂；李海鹏、李娟、七堇年、那多一众实力派作家加盟；更收录陈坤、蔡康永、曾轶可、邵夷贝等跨界明星的文学佳作。而始终不变的是韩寒独有的文艺气息，摒弃无病呻吟，不卖弄技巧，崇尚“真心话+自然美”，简单好读又趣味盎然。"
# tags = u"韩寒,one人气作者,畅销书,有趣"


# title = u"小王子"
# intro = u"25岁的时候，读了小王子。然后爱上了那只狐狸。"
# tags = u"有声读物,小王子"

# title = u"驱魔人系列"
# intro = u"驱魔人第一季·你是谁014 驱魔人第二季·迷城017 驱魔人第三季·无间永生021 驱魔人第四季·鬼影023 驱魔人第五季·秘密060 驱魔人第六季·迷城042 驱魔人第七季·阴童037 驱魔人第八季·赌神026 驱魔人第十季·沉默的羔羊016"
# tags = u"恐怖,有声小说,畅销书,驱魔人,鬼故事"


# metaDic = tp.loadDict('./Data/dictionaries/meta2index')
# id2Meta1 = tp.reverseDic(metaDic['l1meta2id'])
# id2Meta2 = tp.reverseDic(metaDic['l2meta2id'])
#
# title, dic = tp.indexSentence(title, dic, addDict=False)
# intro, dic = tp.indexSentence(intro, dic, addDict=False)
# intro = tp.padding(intro, 150)
# tags, dic = tp.indexSentence(tags, dic, addDict=False)
# X = []
# X.append([tags])
# X.append([title])
# X.append([intro])
# X[0] = X[0][0:1]
# X[1] = X[1][0:1]
# X[2] = X[2][0:1]
# res = atc.predict(X=X)
# res0 = res[0][0]
# # print res[0][0]
# # print res[1][0]
# for id, prob in enumerate(res0):
#     if prob >=0.01:
#         print id2Meta1[id], ':', prob
# print '-' * 30

#
# res1 =res[1][0]
# for id, prob in enumerate(res1):
#     if prob>=0.1:
#         print id2Meta2[id],':',prob
# print '-'*30
