

def DistanceBetweenMeter(geo1, geo2):
    return pow(pow((float(geo1[0]) - float(geo2[0]))*95419.1, 2) + pow((float(geo1[1]) - float(geo2[1]))*111319.5, 2), 0.5)

#gpsFile = open('../data/shanghaiGPS07.txt')
#outFile = open('../data/gpsVector07111.txt','w')

# training data
#gpsFile = open('../data/shanghaiGPS1407-1412.txt')
#outFile = open('../data/gpsVector07-12.txt','w')

# test data
gpsFile = open('../data/shanghaiGPS1501.txt')
outFile = open('../data/gpsVector1501.txt','w')


lonMin=121.44
lonMax=121.51
latMin=31.20
latMax=31.24

lonGap=lonMax-lonMin
latGap=latMax-latMin


trajectory={}
for line in gpsFile:
    line = line.strip()
    line = line.split(',')
    driverId =  line[2]
    longitude = float(line[3])
    latitude = float(line[4])
    if longitude>lonMax or longitude<lonMin or latitude>latMax or latitude<latMin:
        continue      
    time = line[8]
    time = time.split(' ')
    hourTime = time[1].split(':')
    hour = int(hourTime[0])
    minute = int(hourTime[1])
    second = int(hourTime[2])
    if trajectory.has_key(driverId):
        trajectory[driverId].append([longitude, latitude, hour*3600+minute*60+second])
    else:
        trajectory[driverId]=[[longitude, latitude, hour*3600+minute*60+second]]
 
trajectorySeg = 100   
timeInterval = 60  #60s
innerTimeInterval = 10 #5s
for t in trajectory:
    lenNow = 0
    plast = trajectory[t][0]
    plast5 = plast
    segId = 0
    outFile.write(t+'_'+str(0))
    for point in trajectory[t]:
        if point[2]-plast5[2]>innerTimeInterval and DistanceBetweenMeter(point[0:2], plast5[0:2])<10:
            print 'time interval', point[2]-plast5[2]
            print 'distance',DistanceBetweenMeter(point[0:2], plast5[0:2])
            lenNow = 1
            segId = segId+1
            outFile.write('\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2]))  
            outFile.write('\n'+t+'_'+str(segId))
            plast5 = point
        if point[2]-plast[2]>timeInterval or lenNow > trajectorySeg or DistanceBetweenMeter(point[0:2], plast[0:2])>100:
            lenNow = 1
            segId = segId+1
            outFile.write('\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2]))  
            outFile.write('\n'+t+'_'+str(segId))
            plast5 = point
        else:
            lenNow = lenNow+1
            if point[2]-plast5[2]<innerTimeInterval:
                continue
            while(point[2]-plast5[2]>innerTimeInterval):
                dt1=point[2]-plast[2]
                if plast5[2]<plast[2]:
                    dt=innerTimeInterval-(plast[2]-plast5[2])
                else:
                    dt=plast5[2]-plast[2]+innerTimeInterval
                point5 = [0,0,0]
                point5[0]=plast[0]+(point[0]-plast[0])*dt/dt1
                point5[1]=plast[1]+(point[1]-plast[1])*dt/dt1
                point5[2]=plast5[2]+innerTimeInterval
                outFile.write('\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2]))                    
                plast5=point5
                
        plast = point
    outFile.write('\t'+str(plast5[0])+'\t'+str(plast5[1])+'\t'+str(plast5[2])) 
    outFile.write('\n')
    
print "all done..."
    