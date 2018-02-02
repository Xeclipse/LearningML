# coding:utf-8

import utils.textProcess as tp



def genModel(modelSaveFile, flag):
    with open("./Data/word_meta.csv") as f:
        f.readline()
        model = {}
        c = 0
        while 1:
            c+=1
            if c%1000000 == 0:
                print c,
            line = f.readline()
            if not line:
                break
            try:
                item = line.decode('utf-8').strip().split(',')
                if flag:
                    model = tp.genRelationFromSentence([item[0]],[item[1]], model)
                else:
                    model = tp.genRelationFromSentence([item[1]], [item[0]], model)
            except:
                pass
        tp.saveDict(model, modelSaveFile)
        print 'finish'
        
# genModel("./model/word2meta.model.dic", True)
# genModel("./model/meta2word.model.dic", False)