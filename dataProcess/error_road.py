import math
import sympy
import matplotlib.pyplot as plt
import os
import numpy as np

grids = {}
RoadNetInfo = open('../data/roadnet/RoadNetInfoV.txt', 'r')
maxgrid = 0
begin = {}
end = {}
#grid2cor = {}
for line in RoadNetInfo:
	line = line.strip().split('\t')
	begin[int(line[0])] = [float(line[1]), float(line[2])]
	end[int(line[0])] = [float(line[3]), float(line[4])]

RoadNetInfo.close()
#print maxgrid

def calDistance(roads1, roads2, distance, pointNum, start, maxNum, minerror, maxerror):
	for t in range(len(roads1)):
		tra = roads1[t][start:]
		tra1 = roads2[t][start:]
		print tra
		print tra1
		min_len = min(len(tra),len(tra1))
		if (min_len)<maxNum:
			for i in range(min_len):
				distance[i]=distance[i]+math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				pointNum[i]=pointNum[i]+1
				if i==min_len-1:
				#minerror = min(minerror,math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
					maxerror = max(maxerror,math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
		else:
			for i in range(maxNum):
				distance[i]=distance[i]+math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				pointNum[i]=pointNum[i]+1
				if i==maxNum-1:
					minerror = min(minerror,math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
					maxerror = max(maxerror,math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))

	return minerror,maxerror

def velocity(tra):
	sum = 0
	for i in range(len(tra)-3, len(tra)-1):
		sum += math.sqrt((tra[i][0]-tra[i+1][0])**2 + (tra[i][1]-tra[i+1][1])**2)
	return sum/2


def angle(v1, v2):
	#print v1, v2
	tmp1 = np.sum(v1*v2)
	tmp2 = np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2))
	tmpangle = tmp1/tmp2
	#print tmpangle
	if tmpangle<-1:
		return 180
	if tmpangle>1:
		return 0
	return math.acos(tmpangle)


def inarea(point, vbegin, vend):
	lon = point[0]
	lat = point[1]
	lonmax = max(vbegin[0], vend[0])
	lonmin = min(vbegin[0], vend[0])
	latmax = max(vbegin[1], vend[1])
	latmin = min(vbegin[1], vend[1])
	if lon <= lonmax and lon >= lonmin and lat <= latmax and lat >= latmin:
		return True
	return False


def change(line, origroad):
	pregrid = []
	#print originroad
	pretra = origroad[:10]
	#line = line[1:]
	length = len(line)
	#print line


	for i in range(length-1):
		tmp = line[i].split(':')
		pregrid.append(int(tmp[0]))
	vlast = np.array(origroad[9])					#point coordinate
	plast = np.array(origroad[8])					#point coordinate
	#print vlast
	for i in range(10, length-1):
		#print pretra[:i]
		v = velocity(pretra[:i])
		#print 'v is: ', v
		vgrid = np.array(pregrid[i-1])
		#print pregrid[i-1]
		vbegin = np.array(begin[pregrid[i-1]])		#road start coordinate
		vend = np.array(end[pregrid[i-1]])			#road end coordinate
		#pgrid = np.array(pregrid[i-2])

		road = vend - vbegin
		if pregrid[i] == vgrid:
			# if angle(road, tra)<=90:
			direct = vlast - plast
			if angle(direct, road)<5:
				point = vlast + v*road
			else:
				point = vlast - v*road
			pretra.append(point.tolist())
		else:
			x = sympy.Symbol('x')
			y = sympy.Symbol('y')
			k = road[1]/float(road[0])
			x1 = vbegin[0]
			y1 = vbegin[1]
			x0 = vlast[0]
			y0 = vlast[1]
			#print type(x1), type(y1), type(x0), type(y0)
			#print x0, y0
			[p1, p2] = sympy.solve([y-y1-k*(x-x1), (x0-x)*(x0-x)+(y0-y)*(y0-y)-v], [x, y])
			tmp = []

			try:
				if inarea(p1, vbegin, vend):
					tmp.append([float(p1[0]),float(p1[1])])
				if inarea(p2, vbegin, vend):
					tmp.append([float(p2[0]),float(p2[1])])

			except TypeError:
				tmp = []
			if len(tmp):
				pretra.append(tmp[0])
			else:
				#p1 = np.array(p1)
				#print np.sum(np.square(p1 - vbegin))
				dis1 = math.sqrt(np.sum(np.square(vlast-vbegin)))
				dis2 = math.sqrt(np.sum(np.square(vlast-vend)))
				#print dis1
				#p2 = np.array(p2)
				# dis3 = math.sqrt(np.sum(np.square(p2-vbegin)))
				# dis4 = math.sqrt(np.sum(np.square(p2-vend)))
				if dis1< dis2:
					pretra.append(vbegin.tolist())
				else:
					pretra.append(vend.tolist())
	return pretra

originroad = []
predictroad = []

index = 0
testFile = open('../../../char-rnn-tf-master/data/generate/test.txt', 'r')
origFile = open('../data/gpstest-tf-road.txt', 'r')
for line in origFile:
	if index%2 == 0:
		line = line.strip().split('\t')
		line = line[1:]
		tmproad = []
		for i in range(len(line)/3):
			lon = float(line[i*3])
			lat = float(line[i*3+1])
			tmproad.append([lon, lat])
		originroad.append(tmproad)
	index = index + 1

index = 0
for line in testFile:
	line = line.strip().split(',')

	if index % 2 == 1:
		#print len(originroad[index / 2])
		road = change(line, originroad[index/2])
		predictroad.append(road)
	index = index + 1
	if index%100 == 0:
		break
minerror = 1000000
maxerror = 0
start = 4
calculatelen = 30
distance = [0.0]*calculatelen
pointNum = [0.00001]*calculatelen
calDistance(originroad[:30], predictroad[:30], distance, pointNum, start, calculatelen, minerror, maxerror)
for i in range(0, calculatelen):
#i+1
	print i,'th tru distance:\n', distance[i+start]/pointNum[i+start]