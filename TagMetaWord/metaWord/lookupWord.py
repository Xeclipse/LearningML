# coding:utf-8

import pickle

def loadDict(file):
    with open(file) as f:
        return pickle.load(f)


def main():
    print "loading model..."
    model = loadDict("./model/word2meta.model.dic")
    while 1:
        meta = raw_input("输入热词:")
        try:
            meta = meta.decode('utf-8')
        except:
            pass
        try:
            subdic = model[meta]
        except:
            print "未收录该热词"
            continue
        items = subdic.items()
        allnum = len(subdic)
        items = sorted(items, key=lambda x: x[1], reverse=True)
        for i in items:
            if i[1] * 1.0 / allnum >0.01:
                print i[0]


if __name__ == '__main__':
    main()
