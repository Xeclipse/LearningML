import utils.xNgram

f = open('../../SpamFilter/Data/search-filter-spam.format')
sta = utils.xNgram.statictic(f.readlines())
f.close()
counts = sorted(sta.vocab().items(), key=lambda x: x[1])
for i in counts:
    print i[0], ':', i[1]
