import math

# ====================== chengdu ==============================
# filePath = '../data/train/'
# gridfileName = 'train-chengdu-grid50-20.txt'

# filePath = '../data/test/'
# gridfileName = 'test-chengdu-grid50-20.txt'

filePath = '../data/result/shanghai/'
gridfileName = 'result_hidden_128_inter_128_intra_38_ext_9_grid_50_numstep_10.txt'
roadnet = '../data/roadnet/road-shanghai-50.txt'


grid2cor = []
with open(roadnet, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        if len(line)==7:
            grid2cor.append([float(line[1]), float(line[2])])

outFileName = gridfileName.split('.')[0]

groundTruth = open('../data/new_view/' + outFileName + '.geojson', 'w')
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



index=0
with open(filePath+gridfileName, 'r') as file1:
    grids = {}
    for line in file1:
        # if index>219:
        #     break
        # if index<=218:
        #     index = index + 1
        #     continue
        line = line.strip().split(',')

        tmp2 = []
        for i in range(1, len(line)):
            if line[i].__len__() == 0:
                continue
            tmp = line[i].split(':')

            tmp2.append(grid2cor[int(float(tmp[0]))])

        grids[line[0]] = tmp2

        if index %1000:
            print(index)
        index = index + 1

index = 0
with open(filePath+gridfileName, 'r') as file:
    for line in file:
        # if index>6:
        #     break
        # if index <=3:
        #     index = index + 1
        #     continue
        groundTruth.write(pStr)
        line = line.strip().split(',')[1:]
        for i in range(len(line)):
            if line[i] == '':
                continue
            if i:
                groundTruth.write(',')
            groundTruth.write(str(grid2cor[int(line[i].split(':')[0])]))
        groundTruth.write(eStr)
        index += 1
groundTruth.write(endStr)
groundTruth.close()
print('all done...')