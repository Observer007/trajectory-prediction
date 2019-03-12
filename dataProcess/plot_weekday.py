import math
import time

# filePath = '../data/DIDI/train/'
filePath1 = '../data/train/'
# inFileName = 'final_11200_1_15_20.txt'
# inFileName = '11200_800_1_5.txt'
# inFileName = 'train-small-troad.txt'
inFileName = 'train-small-20.txt'
gridfileName = 'train-chengdu-grid50-20.txt'
# gridfileName = 'train-small-grid50-20.txt'

roadnet = '../data/roadnet/road-chengdu-50.txt'
grid2cor = []
with open(roadnet, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        if len(line) == 7:
            grid2cor.append([float(line[1]), float(line[2])])
# testFile = open(filePath + inFileName, 'r')

outFileName = inFileName.split('.')[0]

groundTruth = open('../data/view/' + 'weekday' + outFileName + '.geojson', 'w')
predict = open('../data/view/' + 'weekend' + outFileName + '.geojson', 'w')

preStr = "{ \n \"type\": \"FeatureCollection\", \n \"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } }, \"features\": [\n"
endStr = "] \n }"

pStr = "{  \"geometry\": { \"type\": \"LineString\", \n \"coordinates\": [\n"
eStr = "\n] } },\n"

groundTruth.write(preStr)
predict.write(preStr)


def change(l):
    l = l.split(',')
    length = len(l)
    #    lon_t = []
    #    lat_t = []
    tra = []
    for i in range(length / 2 - 1):
        lon_t = (float(l[i * 2].replace('[', '')))
        lat_t = (float(l[i * 2 + 1].replace(']', '')))
        tra.append([lon_t, lat_t])
    return tra


def calVelocity(tra):
    length = len(tra)
    lon_t = []
    lat_t = []
    for i in range(length):
        lon_t.append(tra[i][0])
        lat_t.append(tra[i][1])
    # lon_t = tra%lonDim
    #    lat_t = tra/lonDim
    v = 0
    for p in range(len(lon_t) - 1):
        v = v + math.sqrt((lon_t[p] - lon_t[p + 1]) ** 2 + (lat_t[p] - lat_t[p + 1]) ** 2)
    return v / length


# print 'begin the process'
# index=0
# for line in testFile:
#     # if index>239:
#     #     break
#     # if index<39:
#     #     index = index + 1
#     #     continue
#     line=line.strip()
#
#     if index%4==1:
#         line0=line.strip(',')
#         tra = change(line0)
#         v = calVelocity(tra)
#
#     if index%4==2:
#         line=line.strip(',')
#         tra1 = change(line)
#         v1 = calVelocity(tra1)
#         if abs(v-v1)<0.00005:
#             predict.write(pStr)
#             predict.write(line0)
#             predict.write(eStr)
#             groundTruth.write(pStr)
#             groundTruth.write(line)
#             groundTruth.write(eStr)
#     #if index%3==2:
#         #continue
#     index = index+1
#
# groundTruth.write(endStr)
# predict.write(endStr)
#
# groundTruth.close()
# predict.close()
#
# print 'all done...'

print 'begin the process'

index = 0
with open(filePath1 + inFileName, 'r') as file1:
    grids = {}
    new = {}
    for line in file1:
        line = line.strip()#.split(',')

        tmp2 = []
        date = int(float(line.split('\t')[3]))
        print line.split('\t')[0], line.split('\t')[1], line.split('\t')[2]
        print line.split('\t')[4], line.split('\t')[5]
        '''chengdu'''
        # date = date - 1477958400
        # date = int(date//86400)+1
        '''shanghai'''
        date = date//100000
        month = (date-1488)//31
        day = (date-1488)%31
        print date, month, day
        if month<10:
            month = '0'+str(month)
        else:
            month = str(month)
        if day < 10:
            if day==0:
                day = 1
            date = '2014'+month+str(day)
        else:
            date = '2014'+month+str(day)
        #print type(date)
        vec = time.strptime(date, '%Y%m%d').tm_wday
        line = line.split('\t')
        for i in range(1, len(line), 3):
            #tmp = line[i].split('\t')
            tmp2.append('['+line[i]+','+line[i+1]+']')
        if vec>=5:
            grids[line[0]] = tmp2
        else:
            new[line[0]] = tmp2
        index = index + 1

index1 = 0
index2 = 0
for traId in new:
    # if index>6:
    #     break
    # if index <=3:
    #     index = index + 1
    #     continue
    print traId
    groundTruth.write(pStr)

    roadtra = new[traId]

    if not len(roadtra) == 20:
        print len(roadtra)
    for i in range(len(roadtra)):
        if i:
            groundTruth.write(',')

        groundTruth.write(str(roadtra[i]))

    groundTruth.write(eStr)

    index1 += 1

for traId in grids:
    predict.write(pStr)
    gridtra = grids[traId]
    if not len(gridtra) == 20:
        print len(gridtra)
    for i in range(len(roadtra)):
        if i:
            predict.write(',')
        predict.write(str(gridtra[i]))
    predict.write(eStr)
    index2 += 1
groundTruth.write(endStr)
groundTruth.close()
predict.write(endStr)
predict.close()
print index1, index2
print 'all done...'