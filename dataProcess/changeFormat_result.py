
# inFileName = '../data/generate/result-chengdu-tgrid50-120-20.txt'
inFileName = '../data/train/train-small-road20.txt'
testFile = open(inFileName, 'r')
RoadNetFile = '../data/roadnet/road-chengdu-50.txt'
InfoFile = open(RoadNetFile, 'r')
grids = {}
index = 0
for line in InfoFile:
    line = line.strip().split('\t')
    if len(line) == 7:
        index = index + 1
        grids[int(line[0])] = [float(line[1]), float(line[2])]
InfoFile.close()
# print index
outFileName = inFileName.split('.')[0]

groundTruth = open('../data/view/groundTruth_' + outFileName+'.geojson', 'w')
predict = open('../data/view/prediction_' + outFileName+'.geojson', 'w')

preStr = "{ \n \"type\": \"FeatureCollection\", \n \"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } }, \"features\": [\n"
endStr = "] \n }"

pStr = "{  \"geometry\": { \"type\": \"LineString\", \n \"coordinates\": [\n"
eStr = "\n] } },\n"

groundTruth.write(preStr)
predict.write(preStr)


print('begin the process')
index=0
for line in testFile:
    # if index<=-1:
    #     index = index+1
    #     continue
    # if index>5:
    #     break
    line=line.strip()
    if index%2==0:
        line = line.strip(',').split(',')
        del(line[0])
        groundTruth.write(pStr)
        for i in range(len(line)):
            if i!=0:
                groundTruth.write(',')
            groundTruth.write(str(grids[int(line[i])]))
        groundTruth.write(eStr)
    if index%2==1:
        line = line.strip(',').split(',')
        del(line[0])
        predict.write(pStr)
        for i in range(len(line)):
            if i != 0:
                predict.write(',')
            print(grids[int(line[i])])
            predict.write(str(grids[int(line[i])]))
        predict.write(eStr)
    #if index%3==2:
        #continue
    index = index+1

groundTruth.write(endStr)
predict.write(endStr)

groundTruth.close()
predict.close()

print('all done...')
    