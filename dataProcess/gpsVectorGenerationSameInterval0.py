import math

dataPath = '../data/'
gpsFile = open(dataPath + 'shanghaiGPS1407-1412.txt')
# gpsFile = open(dataPath+'shanghaiGPS1501.txt')
outFile = open(dataPath + 'train/train-small-t20.txt', 'w')
# outFile = open(dataPath + 'test/test-small-t20.txt', 'w')

lonMin = 121.44
lonMax = 121.48
latMin = 31.20
latMax = 31.24
# lonMin = 121.40
# lonMax = 121.50
# latMin = 30.92
# latMax = 30.98
# lonMin = 121.70
# lonMax = 121.80
# latMin = 31.00
# latMax = 31.06

lonGap = lonMax - lonMin
latGap = latMax - latMin

trajectory = {}
for line in gpsFile:
	line = line.strip()
	line = line.split(',')
	driverId = line[2]
	longitude = float(line[3])
	latitude = float(line[4])
	if longitude > lonMax or longitude < lonMin or latitude > latMax or latitude < latMin:
		continue
	time = line[8]
	time = time.split(' ')
	dataTime = time[0].split('-')
	hourTime = time[1].split(':')
	year = int(dataTime[0])
	month = int(dataTime[1])
	day = int(dataTime[2])
	hour = int(hourTime[0])
	minute = int(hourTime[1])
	second = int(hourTime[2])
	data = (year - 2010) * 372 + month * 31 + day
	if trajectory.has_key(driverId):
		# if trajectory[driverId].has_key(data):
		# trajectory[driverId][data].append([longitude, latitude, hour*3600+minute*60+second])
		trajectory[driverId].append([longitude, latitude, data * 100000 + hour * 3600 + minute * 60 + second])
	# else:
	# trajectory[driverId][data] = [[longitude, latitude, hour*3600+minute*60+second]]
	else:
		# trajectory[driverId] = {data: [[longitude, latitude, hour*3600+minute*60+second]]}
		trajectory[driverId] = [[longitude, latitude, data * 100000 + hour * 3600 + minute * 60 + second]]
trajectorySeg = 20
timeInterval = 30  # 60s
innerTimeInterval = 5  # 5s

new_trajectory = []
num = 0
x = 0

for t in trajectory:
	segId = 0
	count = 1
	new_one0 = str(t) + '_' + str(segId)
	new_tra = [new_one0]
	plast = trajectory[t][0]
	#plast5 = plast
	# outFile.write(t+'_'+str(0))
	for point in trajectory[t][1:]:
		if point[2] - plast[2] > timeInterval:  # or lenNow > trajectorySeg:
			count = 1
			new_tra.append(plast)
			segId = segId + 1
			new_trajectory.append(new_tra)
			new_one0 = str(t) + '_' + str(segId)
			new_tra = [new_one0]
		else:
			if not point[2]-plast[2]:
				continue
			if count % trajectorySeg == 0:
				count = 1
				new_tra.append(plast)
				# print new_tra[1:]
				segId = segId + 1
				# outFile.write(new_input+'\n')
				new_trajectory.append(new_tra)
				new_one0 = str(t) + '_' + str(segId)
				new_tra = [new_one0]
			new_tra.append(plast)
			count = count + 1
		plast = point
	new_tra.append(plast)
	# print new_tra[1:]
	segId = segId + 1
	new_trajectory.append(new_tra)

# trajectory = list(new_trajectory)
# print len(trajectory)
length = 20
maxtime = 0
mintime = 9999999
for tra in new_trajectory:
	new_input = tra[0]
	print len(tra[1:])
	if len(tra[1:]) == length:
		num = num + 1
		for i in range(1, len(tra)):
			new_input += '\t' + str(round(tra[i][0], 5)) + '\t' + str(round(tra[i][1], 5)) + '\t' + str(
				round(tra[i][2], 5))
			if i>1:
				maxtime = max(maxtime, tra[i][2]-tra[i-1][2])
				mintime = min(mintime, tra[i][2]-tra[i-1][2])
		outFile.write(new_input + '\n')
print maxtime, mintime
print "has 0: ", x
print "trajectory number is:", num
print "all done..."

