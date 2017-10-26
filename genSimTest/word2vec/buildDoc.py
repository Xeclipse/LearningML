


def main():
    # read file1
    file = open("/Users/nali/PycharmProjects/LearningML/TrainData/user-played-album-3-day.csv")
    line = file.readline()
    userPlayDic = {}
    train = []
    ids = []
    count = 0
    print "step1: reading file"
    line = file.readline()
    while line:
        ele = line.strip().split(',')
        if ele.__len__() < 2:
            continue
        id = ele[0]
        user = ele[1]
        if not userPlayDic.has_key(user):
            userPlayDic[user] = []
        userPlayDic[user].append(id)
        line = file.readline()
        count += 1
        if count % 100000 == 0:
            print count
    file.close()

    outfile = open("./listenDoc.ldoc","w")
    for i in userPlayDic.values():
        userdoc=""
        for album in i:
            userdoc += album+' '
        userdoc+='\n'
        outfile.write(userdoc)
    outfile.close()

if __name__ == '__main__':
    main()