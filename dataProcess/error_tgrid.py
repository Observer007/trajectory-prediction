import math
import matplotlib.pyplot as plt
import os

grids = {}
RoadNetInfo = open('../data/roadnet/road-small-50.txt', 'r')
maxgrid = 0
tra2grid = {}
grid2tra = {}
grid2cor = {}
for line in RoadNetInfo:
	line = line.strip().split('\t')
	if len(line) == 2:
		tra2grid[int(line[0])] = int(line[1])
		grid2tra[int(line[1])] = int(line[0])
	#maxgrid = max(maxgrid, int(line[0]))
	if len(line) == 3:
		grid2cor[int(line[0])] = [float(line[1]), float(line[2])]
RoadNetInfo.close()
#print maxgrid


def calDistance(roads1, roads2, distance, pointNum, start, maxNum, minerror, maxerror):
	for t in range(min(len(roads1), len(roads2))):
		tra = roads1[t][start:]
		tra1 = roads2[t][start:]
		ptime = roads1[t][start-1][2]
		print tra
		min_len = min(len(tra), len(tra1))
		if (min_len)<maxNum:
			for i in range(min_len):
				tmp = math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				print tra[i][2]
				distance[tra[i][2]-ptime] = distance[tra[i][2]-ptime] + tmp
				pointNum[tra[i][2]-ptime] = pointNum[tra[i][2]-ptime] + 1
				if i == min_len-1:
				#minerror = min(minerror,math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
					maxerror = max(maxerror, math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
		else:
			for i in range(maxNum-start):
				# distance[i] = distance[i]+math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				# pointNum[i] = pointNum[i]+1
				tmp = math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				# if tmp>1000:
				# 	continue
				distance[start+i] = distance[start+i] + tmp
				pointNum[start+i] = pointNum[start+i] + 1
				if i == maxNum-1:
					minerror = min(minerror, math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
					maxerror = max(maxerror, math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))

	return minerror, maxerror

def change_grid(line):
	road = []
	line = line[1:]
	length = len(line)
	#print line
	for i in range(length-1):
		tmp = grid2cor[int(line[i])]
		road.append(tmp)
	return road

def change_coor(line):
	road = []
	line = line[1:]
	length = len(line)
	for i in range(length/3):
		if not i:
			road.append([float(line[3*i]), float(line[3*i+1]), 0])
			ptime = int(float(line[3*i+2]))
		else:
			road.append([float(line[3*i]), float(line[3*i+1]), int(float(line[3*i+2]))-ptime])
	return road

origindict = {}
predictdict = {}
originroad = []
predictroad = []

index = 0

# testFile = open('../data/generate/result-small-tgrid50.txt', 'r')
testFile = open('../data/seq_generate/result-small-tgrid50-20-20.txt', 'r')

origFile = open('../data/test/test-small-t20.txt', 'r')
for line in testFile:
	line = line.strip().split(',')
	road = change_grid(line)
	if index % 2 == 1:
		predictdict[line[0]] = road
	index = index + 1
for line in origFile:
	line = line.strip().split('\t')
	road = change_coor(line)
	origindict[line[0]] = road
for traId in predictdict:
	print traId
	predictroad.append(predictdict[traId])
	originroad.append(origindict[traId])
minerror = 1000000
maxerror = 0
start = 10
calculatelen = (20-start)*40
distance = [0.0]*calculatelen
pointNum = [0.00001]*calculatelen
print originroad[0]
calDistance(originroad, predictroad, distance, pointNum, start, calculatelen, minerror, maxerror)
x = []
y = []
for i in range(5, 51):
	if not pointNum[i]<0.1 and not i%5:
		x.append(i)
		y.append(distance[i]/pointNum[i])
		print i, 'th tru distance:\n', distance[i]/pointNum[i]
print pointNum[:]
print y
y4 = [81.81238461105947, 104.8583196458128, 119.24825099732867, 124.49362956096623, 121.30939586718576,
	  127.35297576347475, 123.30859695330476, 130.70545947761548, 136.54084370205823, 135.4983376795977]
# y3 = [38.52195660201118, 40.50111026240869, 39.34223575664907, 44.42406012401447, 42.93550895197809,
# 	  44.674811364746496, 39.43221779695715, 40.037343225242516, 37.50840638302457, 41.11076388895563,
# 	  44.55637858436172, 41.149954439578515, 45.871394962266876, 38.80941188920937, 39.594639656545915,
# 	  39.40232718869918, 42.89811252338822, 44.21682618099083, 43.046871688674834, 44.447333750531094]
y2 = [54.85674875904323, 51.81754552135408, 55.45600804993668, 49.24557219889665, 52.16098262521036,
	  54.49268020893064, 58.19954400974716, 56.28031944624678, 55.302708167474826, 63.38636238969011,
	  66.06159994352943, 65.44606827312403, 65.37397145322879, 56.979174632941636, 54.65442923605119,
	  53.560822490509466, 52.57636280581273, 55.960090442238254, 54.79988459141531, 54.94486800862001]

y1 = [62.537301023933345, 61.43684172211795, 62.874909020933245, 67.23866418241171, 62.323931285066585,
	   64.4995170820057, 62.34140372631919, 65.44226552388618, 76.6489143030613, 78.51910682795237,
	   77.44716053185427, 65.78338893035962, 66.63155908782197, 60.51547037219032, 64.59021248530685,
	   64.16935525587685, 64.80720659786796, 65.10654856215615, 60.68951667667953, 66.57600488508871]
# #plt.hold()
plt.plot(x, y, label="pre")
# plt.plot(range(0, calculatelen-start), y1, label="length=50")
# plt.plot(range(0, calculatelen-start), y2, label="length=30")
# plt.plot(range(0, calculatelen-start), y2, color="blue", label="remove=1000")
# plt.plot(range(0, calculatelen-start), y3, color="red", label="remove=200")
# plt.plot(range(0, calculatelen-start), y4, label="new")
plt.legend(loc='upper right')
plt.xlabel("time")
plt.ylabel("average error(m)")
plt.xlim(0, 55)
plt.ylim(0, 300)
plt.show()