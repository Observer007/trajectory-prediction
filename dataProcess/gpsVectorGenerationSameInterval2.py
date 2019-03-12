import math
import numpy as np
dataPath = '../data/'
# gpsFile = open(dataPath+'shanghaiGPS1407-1412.txt')
gpsFile = open(dataPath+'shanghaiGPS1501.txt')
# outFile = open(dataPath+'train/train-small-10.txt', 'w')
outFile = open(dataPath+'test/test-small-10.txt', 'w')


# lonMin = 121.44
# lonMax = 121.48
# latMin = 31.20
# latMax = 31.24
#
lonMin = 121.40
lonMax = 121.50
latMin = 30.92
latMax = 30.98

# lonMin = 121.70
# lonMax = 121.80
# latMin = 31.00
# latMax = 31.06

lonGap = lonMax-lonMin
latGap = latMax-latMin


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
    data = (year-2010)*372 + month*31 + day
    if trajectory.has_key(driverId):
        #if trajectory[driverId].has_key(data):
            #trajectory[driverId][data].append([longitude, latitude, hour*3600+minute*60+second])
        trajectory[driverId].append([longitude, latitude, data*100000 + hour * 3600 + minute * 60 + second])
        #else:
            #trajectory[driverId][data] = [[longitude, latitude, hour*3600+minute*60+second]]
    else:
        #trajectory[driverId] = {data: [[longitude, latitude, hour*3600+minute*60+second]]}
        trajectory[driverId] = [[longitude, latitude, data*100000 + hour*3600 + minute*60 + second]]

print "trajectory number: ", len(trajectory)
trajectorySeg = 10
timeInterval = 30  #60s
innerTimeInterval = 10 #5s

def validity(tra):
    length = len(tra)
    if length<15:
        return False
    angle = []
    maxangle = 0
    pangle = 0
    index = 0
    for i in range(length-2):
        index = index + 1
        plon = (tra[i+1][0] - tra[i][0])*100000
        plat = (tra[i+1][1] - tra[i][1])*100000
        lon = (tra[i+2][0] - tra[i+1][0])*100000
        lat = (tra[i+2][1] - tra[i+1][1])*100000

        div = math.sqrt(plon**2 + plat**2)*math.sqrt(lon**2 + lat**2)
        #print plon, plat, lon, lat
        #print div
        if div == 0:
            temp = 1
        else:
            temp = (plon*lon + plat*lat)/div
            if temp > 1:
                temp = 1
            elif temp < -1:
                temp = -1
        angle.append(math.acos(temp)*180/math.pi)
    maxangle1 = 0
    maxangle2 = 0
    for i in range(len(angle)):
        maxangle1 = max(maxangle1, angle[i])

    for i in range(len(angle)-5):
        pmax = min(angle[i], angle[i + 1])
        lmax = max(angle[i], angle[i + 1])
        for j in range(2, 6):
            if angle[i+j]>lmax:
                pmax = lmax
                lmax = angle[i+j]
            elif angle[i+j]>pmax:
                pmax = angle[i+j]
        maxangle2 = max(maxangle2, pmax+lmax)
    #maxi = max(angle)
    #print maxi
    # temp = maxangle
    # index0 = 0
    # index1 = 1
    # #minangle = min(angle[0], angle[1])
    # for i in range(len(angle)):
    #     if angle[i]<5:
    #         continue
    #     if i==0:
    #         maxangle = angle[0]
    #         temp = maxangle
    #     else:
    #         temp = temp - min(angle[index0], angle[index1]) + angle[i]
    #     index0 = index1
    #     index1 = i
    #     maxangle = max(maxangle, temp)

    maxangle2 = maxangle2/2
    #print maxangle
    if maxangle2 > 90 or maxangle1 > 150:
        return False
    return True

new_trajectory = []
num = 0

for t in trajectory:
    segId = 0
    #for l in trajectory[t]:
        #flag = validity(trajectory[t][l])
        # if flag==False:
        #     continue
    count = 1
    new_one0 = str(t) + '_' + str(segId)
    new_tra = [new_one0]
    #lenNow = 0
    plast = trajectory[t][0]
    plast5 = plast
    #outFile.write(t+'_'+str(0))
    for point in trajectory[t]:
        if point[2]-plast[2] > timeInterval: #or lenNow > trajectorySeg:
            #lenNow = 1
            #print count
            count = 1
            #outFile.write('\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2]))
            #outFile.write('\n'+t+'_'+str(segId))
            #new_input += '\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2])
            new_tra.append(plast5)
            #print new_tra[1:]
            # flag = validity(new_tra[1:])

            #if flag:
            if 1==1:
                #num = num + 1
                segId = segId + 1
                #outFile.write(new_input+'\n')
                new_trajectory.append(new_tra)
            new_one0 = str(t)+'_'+str(segId)
            new_tra = [new_one0]
            plast5 = point
        else:
            #lenNow = lenNow+1
            if point[2]-plast5[2] < innerTimeInterval:
                continue
            while(point[2]-plast5[2] > innerTimeInterval):
                if count%trajectorySeg == 0:
                    count = 1
                    new_tra.append(plast5)
                    # print new_tra[1:]
                    # flag = validity(new_tra[1:])
                    if 1:
                        segId = segId + 1
                        # outFile.write(new_input+'\n')
                        new_trajectory.append(new_tra)
                    new_one0 = str(t) + '_' + str(segId)
                    new_tra = [new_one0]
                    #plast5 = point
                dt1 = point[2]-plast[2]
                if plast5[2] < plast[2]:
                    dt = innerTimeInterval-(plast[2]-plast5[2])
                else:
                    dt=plast5[2]-plast[2]+innerTimeInterval
                point5 = [0, 0, 0]
                point5[0]=plast[0]+(point[0]-plast[0])*dt/dt1
                point5[1]=plast[1]+(point[1]-plast[1])*dt/dt1
                point5[2]=plast5[2]+innerTimeInterval
                #outFile.write('\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2]))
                #new_input += '\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2])
                new_tra.append(plast5)
                count = count + 1
                plast5=point5
                
        plast = point
        #outFile.write('\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2]))
        #new_input += '\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2])
    new_tra.append(plast5)
    #print new_tra[1:]
    # flag = validity(new_tra[1:])
    #if flag:
    if 1:
        segId = segId + 1
        #num = num + 1
        #outFile.write(new_input)
        #outFile.write('\n')
        #print count
        #print ' '
        new_trajectory.append(new_tra)

#trajectory = list(new_trajectory)
#print len(trajectory)
trajectory = []
delnum = 0
v0 = []
for t in range(len(new_trajectory)):
    #print t

    f = 1
    tra = new_trajectory[t]
    if len(tra)<10:
        # print 'x'
        delnum = delnum + 1
        continue
    #del(tra[0])
    dis = 0
    time = 0
    for p in range(1, len(tra)-1):
        dis += math.sqrt(((tra[p][0]-tra[p+1][0])*94915)**2+((tra[p][1]-tra[p+1][1])*111122)**2)
        time += tra[p+1][2] - tra[p][2]
        if tra[p+1][2] - tra[p][2] == 0:
            print 'error'
            #print tra[p][2]
            #print t, p
        if time==0:
            print time
            continue

    v = dis/time
    v0.append(v)
    if v<=5:
        print 'p'
        delnum = delnum + 1
        f = 0
    if f:
        trajectory.append(tra)
v0 = np.array(v0)
print np.mean(v0), np.max(v0), np.min(v0)
for tra in trajectory:
    new_input = tra[0]
    # print len(tra[1:])
    if len(tra[1:]) == 10:
        for i in range(1, len(tra)):
            new_input += '\t'+str(round(tra[i][0], 5))+'\t'+str(round(tra[i][1], 5))+'\t'+str(round(tra[i][2], 5))
        outFile.write(new_input+'\n')

#print new_trajectory[0]
print "delete number is:", delnum
print "trajectory number is:", len(trajectory)
print "all done..."
    
