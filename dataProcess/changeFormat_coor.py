import math
# filePath = '../data/DIDI/train/'
# filePath = '../data/DIDI/test/'
city = 'beijing'
filePath = '../data/train/'
# filePath = '../data/test/'

# inFileName = 'train-chengdu-20.txt'
# inFileName = 'train-chengdu-road20.txt'
# inFileName = 'test-chengdu-20.txt'
# inFileName = 'test-chengdu-road20.txt'

# inFileName = 'train-small-10.txt'
# inFileName = 'test-beijing-road20.txt'

inFileName = 'train-'+city+'-weekday.txt'
# inFileName = 'train-'+city+'-weekend.txt'

threshold = 100000000
# roadnet = '../data/roadnet/road-chengdu-50.txt'
# grid2cor = []
# with open(roadnet, 'r') as f:
#     for line in f:
#         line = line.strip().split('\t')
#         if len(line)==7:
#             grid2cor.append([float(line[1]), float(line[2])])
testFile=open(filePath + inFileName, 'r')

outFileName = inFileName.split('.')[0]

groundTruth = open('../data/new_view/' + outFileName+'.geojson','w')
# predict = open('../data/view/'+'prediction' + outFileName+'.geojson','w')

preStr = "{ \n \"type\": \"FeatureCollection\", \n \"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } }, \"features\": [\n"
endStr = "] \n }"

pStr = "{  \"geometry\": { \"type\": \"LineString\", \n \"coordinates\": [\n"
eStr = "\n] } },\n"

groundTruth.write(preStr)


def change(l):
    l = l.split(',')
    length = len(l)

    tra = []
    for i in range(length/2-1):
        lon_t = (float(l[i*2].replace('[', '')))
        lat_t = (float(l[i*2+1].replace(']', '')))
        tra.append([lon_t, lat_t])
    return tra
def calVelocity(tra):
    length = len(tra)
    lon_t = []
    lat_t = []
    for i in range(length):
        lon_t.append(tra[i][0])
        lat_t.append(tra[i][1])
#    lon_t = tra%lonDim
#    lat_t = tra/lonDim
    v = 0
    for p in range(len(lon_t)-1):
        v = v + math.sqrt((lon_t[p]-lon_t[p+1])**2+(lat_t[p]-lat_t[p+1])**2)
    return v/length

print('begin the process')
index=0
for line in testFile:
    line=line.strip()
    groundTruth.write(pStr)
    line = line.split('\t')[1:]
    w_line = ''
    for i in range(0, len(line), 3):
        if i:
            w_line+=','
        w_line+='['+line[i]+','+line[i+1]+']'
    groundTruth.write(w_line)
    groundTruth.write(eStr)
    index = index+1
    if index>threshold:
        break
groundTruth.write(endStr)

groundTruth.close()

print('all done...')
    
# print('begin the process')
# index=0
# new = {}
# for line in testFile:
#     # if index>219:
#     #     break
#     # if index<=218:
#     #     index = index + 1
#     #     continue
#     line = line.strip()
#     line = line.split('\t')
#     # del(line[0])
#     #del(line[0])
#
#     tmp1 = []
#     for i in range(1, len(line), 3):
#         tmp1.append([float(line[i]), float(line[i+1])])
#     new[line[0]] = tmp1
#     index = index + 1
#
#
# index=0
# with open(filePath1+gridfileName, 'r') as file1:
#     grids = {}
#     for line in file1:
#         # if index>219:
#         #     break
#         # if index<=218:
#         #     index = index + 1
#         #     continue
#         line = line.strip().split(',')
#
#         tmp2 = []
#         for i in range(1, len(line)):
#             tmp = line[i].split(':')
#             tmp2.append(grid2cor[int(float(tmp[0]))])
#         grids[line[0]] = tmp2
#
#
#         index = index + 1
#
# index = 0
# for traId in grids:
#     # if index>6:
#     #     break
#     # if index <=3:
#     #     index = index + 1
#     #     continue
#     print traId
#     groundTruth.write(pStr)
#     predict.write(pStr)
#     roadtra = new[traId]
#     gridtra = grids[traId]
#     if not len(roadtra)==20:
#         print len(roadtra)
#     for i in range(len(roadtra)):
#         if i:
#             groundTruth.write(',')
#             predict.write(',')
#         groundTruth.write(str(roadtra[i]))
#         predict.write(str(gridtra[i]))
#     groundTruth.write(eStr)
#     predict.write(eStr)
#     index += 1
# groundTruth.write(endStr)
# groundTruth.close()
# predict.write(endStr)
# predict.close()
# print index
# print 'all done...'