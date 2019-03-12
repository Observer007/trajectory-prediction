import math
import numpy as np
import matplotlib.pyplot as plt

start = 10


def calDistance(line, line1, distance, pointNum, maxNum):
	# line = line.split(',')
	# line1 = line1.split(',')
	min_len = min(len(line), len(line1))
	if (min_len / 2 - 1) < maxNum:
		for i in range(min_len / 2 - 1):
			p1_lon = float(line[i * 2])
			p1_lat = float(line[i * 2 + 1])
			p2_lon = float(line1[i * 2])
			p2_lat = float(line1[i * 2 + 1])
			# print p1_lon
			distance[i] = distance[i] + math.sqrt(((p1_lon - p2_lon) * 95218) ** 2 + ((p1_lat - p2_lat) * 111320) ** 2)
			pointNum[i] = pointNum[i] + 1
		# print p1_lon, p2_lon, p1_lat, p2_lat
	else:
		for i in range(maxNum):
			# print 'i',i
			p1_lon = float(line[i * 2])
			p1_lat = float(line[i * 2 + 1])
			p2_lon = float(line1[i * 2])
			p2_lat = float(line1[i * 2 + 1])
			# print p1_lon
			distance[i] = distance[i] + math.sqrt(((p1_lon - p2_lon) * 95218) ** 2 + ((p1_lat - p2_lat) * 111320) ** 2)
			pointNum[i] = pointNum[i] + 1

length = '20'
filePath = '../data/generate/'
inFileName = 'RMF-chengdu.txt'
originName = '../data/test/test-chengdu-'+length+'.txt'
outFileName = 'groundTruth'
outFileName1 = 'prediction'
testFile = open(filePath + inFileName, 'r')
orgFile = open(originName, 'r')
calculatelen = 20
distance = [0.0] * calculatelen
pointNum = [0.00001] * calculatelen
distance1 = [0.0] * calculatelen
pointNum1 = [0.00001] * calculatelen
print pointNum
l1 = ''
l2 = ''
print 'begin the process'
index = 0
number = 249
for line in testFile:
	line = line.strip()
	if index % 2 == 0:
		l1 = line.split(',')
		l1 = l1[1:]
		#print l1
	if index % 2 == 1:
		l2 = line.split(',')
		l2 = l2[1:]
		calDistance(l1, l2, distance, pointNum, calculatelen)
	index = index + 1
# print index
# print 'distance:', distance
# print 'point Num', pointNum
# for i in range(start,calculatelen):
#    print i,'th average distance:\n', (np.array(distance)/np.array(pointNum))[i]
# print ''
print 'tru Distance:', distance
print 'point Num:', pointNum
for i in range(start, calculatelen):
	print (np.array(distance) / np.array(pointNum))[i]
	# print i, 'th tru distance:\n', (np.array(distance) / np.array(pointNum))[i]
print (np.array(distance) / np.array(pointNum))

y = np.array(distance1) / np.array(pointNum) / 1000
plt.figure(1)
x = np.array([a for a in range(1, calculatelen - start + 1)])
plt.plot(x, y[start:], 'r', label='RMF')





plt.xlabel('time(s)')
plt.ylabel('average distance error(km)')
plt.legend()