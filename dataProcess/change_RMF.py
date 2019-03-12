import math

# filePath = '../data/train/'
filePath = '../data/generate/'
# inFileName = 'final_11200_1_15_20.txt'
# inFileName = '11200_800_1_5.txt'
# inFileName = 'train-small-troad.txt'
inFileName = 'RMF-small.txt'
gridfileName = 'test-small-grid50.txt'
roadnet = '../data/roadnet/road-small-50.txt'
grid2cor = []
with open(roadnet, 'r') as f:
	for line in f:
		line = line.strip().split('\t')
		if len(line) == 3:
			grid2cor.append([float(line[1]), float(line[2])])
testFile = open(filePath + inFileName, 'r')

outFileName = inFileName.split('.')[0]

groundTruth = open('../data/view/' + 'groundTruth' + outFileName + '.geojson', 'w')
predict = open('../data/view/' + 'prediction' + outFileName + '.geojson', 'w')

preStr = "{ \n \"type\": \"FeatureCollection\", \n \"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } }, \"features\": [\n"
endStr = "] \n }"

pStr = "{  \"geometry\": { \"type\": \"LineString\", \n \"coordinates\": [\n"
eStr = "\n] } },\n"

groundTruth.write(preStr)
predict.write(preStr)


def change(l):
	l = l.split(',')
	l = l[1:]
	length = len(l)
	#    lon_t = []
	#    lat_t = []
	tra = []
	for i in range(length / 2 - 1):
		lon_t = (float(l[i * 2].replace('[', '')))
		lat_t = (float(l[i * 2 + 1].replace(']', '')))
		tra.append([lon_t, lat_t])
	return tra, l


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


print 'begin the process'
index=0
for line in testFile:
    if index>239:
        break
    if index<39:
        index = index + 1
        continue
    line=line.strip()

    if index%2==1:
        line0=line.strip(',')
        tra, line0= change(line0)
        predict.write(pStr)
        for i in range(len(tra)):
            if i!=0:
                predict.write(',')
            predict.write(str(tra[i]))
        predict.write(eStr)
    if index%2==0:
        line=line.strip(',')
        tra1, line = change(line)
        #v1 = calVelocity(tra1)
        groundTruth.write(pStr)
        for i in range(len(tra1)):
			if i!=0:
				groundTruth.write(',')
			groundTruth.write(str(tra1[i]))
        groundTruth.write(eStr)
    #if index%3==2:
        #continue
    index = index+1

groundTruth.write(endStr)
predict.write(endStr)

groundTruth.close()
predict.close()

print 'all done...'

