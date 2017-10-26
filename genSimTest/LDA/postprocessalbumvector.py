from collections import Counter
import matplotlib.pyplot as plt

f = open('./album_topics.t')
albumCat = []
for line in f.readlines():
    albumCat.append(line.strip().split('\t'))
f.close()

# ids=[int(i[1]) for i in albumCat]
# counter=Counter(ids)
# x=counter.keys()
# y=counter.values()
# plt.bar(x,y)
# plt.show()


albumCat = sorted(albumCat, key=lambda x: int(x[1]))

f = open('./album_topics_sorted.st', 'w')
for i in albumCat:
    outstr = i[0] + '\t' + i[1] + '\t' + i[2] + '\n'
    f.write(outstr)
f.close()
