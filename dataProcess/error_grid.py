import math
import matplotlib.pyplot as plt
import os
import numpy as np

city = 'chengdu'
grid = '50'
grids = {}
RoadNetInfo = open('../data/roadnet/road-'+ city +'-'+grid+'.txt', 'r')
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
	if len(line) == 7:
		grid2cor[int(line[0])] = [float(line[1]), float(line[2])]
RoadNetInfo.close()
#print maxgrid


def calDistance(roads1, roads2, distance, pointNum, start, maxNum, minerror, maxerror):
	for t in range(min(len(roads1), len(roads2))):
		tra = roads1[t][start:]
		tra1 = roads2[t][start:]
		# print roads1[t][0], tra
		# print roads2[t][0], tra1
		min_len = min(len(tra), len(tra1))
		if (min_len)<maxNum-start:
			for i in range(min_len):
				tmp = math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				if tmp> 1000:
					continue
				distance[start+i] = distance[start+i] + tmp
				pointNum[start+i] = pointNum[start+i] + 1
				if i == min_len-1:
				#minerror = min(minerror,math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
					maxerror = max(maxerror, math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
		else:
			for i in range(maxNum-start):
				# distance[i] = distance[i]+math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				# pointNum[i] = pointNum[i]+1
				tmp = math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2)
				# if tmp>176:
				if tmp > 600:
					continue
				count[i].append(int(tmp)/50)
				# print int(tmp/50)
				if tmp>200:
					distribution[i][4]+=1
				else:
					distribution[i][int(tmp/50)] += 1
				distance[start+i] = distance[start+i] + tmp
				pointNum[start+i] = pointNum[start+i] + 1
				if i == maxNum-1:
					minerror = min(minerror, math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))
					maxerror = max(maxerror, math.sqrt(((tra[i][0]-tra1[i][0])*math.cos(tra[i][0]/180*math.pi)*111000)**2+((tra[i][1]-tra1[i][1])*111000)**2))

	return minerror, maxerror


distributed = {}
def change_grid(line):
	road = []
	line = line[1:]
	length = len(line)
	for i in range(length-1):
		if len(line) in distributed:
			distributed[len(line)] += 1
		else:
			distributed[len(line)] = 1
		if int(line[i])==7:
			tmp = [0,0]
			road.append(tmp)
		else:
			tmp = grid2cor[int(line[i])]
			road.append(tmp)
		#print tmp
	return road

def change_coor(line):
	road = []
	line = line[1:]
	length = len(line)
	for i in range(length/3):
		road.append([float(line[3*i]), float(line[3*i+1])])
	return road

origindict = {}
predictdict = {}
originroad = []
predictroad = []
originsection = []
predictsection = []

minerror = 1000000
maxerror = 0
start = 5
calculatelen = 10
count = [[] for i in range(calculatelen-start)]
distribution = [[0 for i in range(5)] for j in range(calculatelen-start)]
distance = [0.0]*calculatelen
pointNum = [0.00001]*calculatelen

correct = np.zeros(calculatelen-start)

# nearest = '120'
holiday = 'True'
add_dim = '33'
nearest = '20'
index = 0
length1 = '10'
input_dim = 256
'''holiday'''
# testFile = open('../data/generate/result-'+size+'-holiday.txt', 'r')
# testFile = open('../data/generate/result-'+size+'-weekday.txt', 'r')
'''peakhour'''
testFile = open('../data/generate/result-'+size+'-morning.txt', 'r')
# testFile = open('../data/generate/result-'+size+'-evening.txt', 'r')
# testFile = open('../data/generate/result-'+size+'-othertime.txt', 'r')

# testFile = open('../data/generate/result-'+size+'-noembedding'+add_dim+'.txt', 'r')
# testFile = open('../data/generate/result-'+size+'-'+holiday+'-'+add_dim+'-'+str(input_dim)+'-'+length1+'.txt', 'r')
# testFile = open('../data/seq_generate/result-'+size+'-tgrid'+grid+'-'+str(nearest)+'-'+length1+'.txt', 'r')
# testFile = open('../data/generate/test1.txt', 'r')
origFile = open('../data/test/test-'+size+'-'+length1+'.txt', 'r')
# origFile = open('../data/train/train-small-t20.txt', 'r')
for line in testFile:
	line = line.strip().split(',')
	road = change_grid(line)
	if index % 2 == 0:
		originsection.append(road)
	if index % 2 == 1:
		predictdict[line[0]] = road
		predictsection.append(road)
	index = index + 1

for line in origFile:
	line = line.strip().split('\t')
	road = change_coor(line)
	origindict[line[0]] = road
for traId in predictdict:
	#print traId
	if origindict.has_key(traId) and len(predictdict[traId])>start+1:
		predictroad.append(predictdict[traId])
		originroad.append(origindict[traId])

for key in distributed:
	print key, distributed[key]
print(len(originsection), len(predictsection))
_index, index = 100, 200
calDistance(originroad[_index:index], predictroad[_index:index], distance, pointNum, start, calculatelen, minerror, maxerror)
for i in range(len(originsection)):
	for j in range(start, min(calculatelen,len(predictsection[i]))):
		if originsection[i][j]==predictsection[i][j]:
			correct[j-start] += 1
print "correct rate is: ", correct/float(len(originsection))
print "total correct rate is: ", np.sum(correct)/float(len(originsection)*5)

y = []
for i in range(0, calculatelen-start):

	y.append(distance[i+start]/pointNum[i+start])

	print i, 'th tru distance:\n', y[i]
# plt.plot(range(0,5), y, color='#B22222', marker='^', markersize=10)
print "total correct rate is: ", np.sum(np.array(y[:1])), np.sum(np.array(y[:2])),\
	np.sum(np.array(y[:3])), np.sum(np.array(y[:4])), np.sum(np.array(y[:5])),\
np.sum(np.array(y[:6])), np.sum(np.array(y[:7])), np.sum(np.array(y[:8]))
print pointNum[start:]
print np.array(pointNum[start:])/len(predictroad)
print y

y4 = [81.81238461105947, 104.8583196458128, 119.24825099732867, 124.49362956096623, 121.30939586718576,
	  127.35297576347475, 123.30859695330476, 130.70545947761548, 136.54084370205823, 135.4983376795977]
y3 = [77.46693858719989, 79.45455436311563, 91.20112339631866, 115.70651029705215, 129.1288344178096]
'''shanghai RMF'''
y11 = [76.155138,     143.2786179,    231.25666447,   331.12486838,   457.79380771]
'''chengdu RMF'''
y12 = [43.61874717,   68.36418241,   97.56028121,  128.41090156,
  163.17121463,  204.16944481,  247.233177,    291.54086603]


y14 = [113.61017729352896,121.41672772400715,134.92658440842638,139.39805208083982,154.3295609145327]
y13 = [127.93862548079734,133.2611178148431,142.22541648926298,149.49565023543335,152.75521763848036]
# y12 = [69.7937139133999,79.93998489200378,89.74432371603653,98.94308851484436,
# 	   104.75229868069674,107.04142977006025,110.08937724171545,117.39010035107823]
'''shanghai R2-D2'''
y21 = [116.71025977086003, 132.67115336204077, 144.7861431864225, 160.94802661666571, 176.15222240901625]
'''chengdu R2-D2'''
y22 = [55.30247306, 64.6381737, 72.19838774, 76.7369201, 82.75042684]
y1 = [104.97955965293497,112.06243067663107,122.85508904497516,136.96355742193444,141.8674355866567,
	  142.70266209173403,153.94081593184748,147.3743912724778,141.8968555849219,132.98211559593167]
'''shanghai lstm'''
y31 = [34.44378478, 51.769, 74.194, 95.603, 126.084]
'''shanghai RA-lstm'''
y41 = [28.1515208576, 44.17, 64.254, 89.829, 117.66]
'''chengdu lstm'''
y32 = [43.8639311674, 51.3939, 60.1449, 69.86, 80.4127]
'''chengdu RA-lstm'''
y42 = [25.9319600268, 33.7371, 42.5015, 53.1572, 64.3129]
# plt.plot(range(5, (calculatelen-start+1)*5, 5), y, label="new")
plt.title("")
# plt.plot(range(1, (calculatelen-start+1)), y3, marker='^', label="RMF-LSTM")
plt.plot(range(1, (calculatelen-start+1)), y42[:calculatelen-start], marker='^', label="RA-LSTM")
plt.plot(range(1, (calculatelen-start+1)), y32[:calculatelen-start],  marker='x', label="LSTM")
# plt.plot(range(0, calculatelen-start), y2, color="blue", label="remove=1000")
print distribution
plt.plot(range(1, (calculatelen-start+1)), y22[:calculatelen-start], marker='o', label="R2-D2")
plt.plot(range(1, (calculatelen-start+1)), y12[:calculatelen-start], marker='s', label="RMF")
plt.legend(loc='upper left', fontsize=13)
plt.xlabel("Prediction step", fontsize=15)
plt.ylabel("Distance error(m)", fontsize=15)
plt.xticks(range(1, 11), fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.8, 5.2)
plt.ylim(0, 250)
plt.show()
distribution = [[distribution[i][j]/float(pointNum[start+1]) for j in range(len(distribution[i]))] for i in range(len(distribution))]
plt.figure(2)
print [i*50 for i in range(11)]
# temp = plt.bar([i for i in range(11)], distribution[0], normed=1, facecolor='yellowgreen')
print distribution
distribution = np.array(distribution)
plt.xlim(-0.4, 5.2)
plt.ylim(0, 1)
ind = np.arange(5)                # the x locations for the groups
width = 0.18
plt.bar(ind+width*0.5, distribution[:,0], width,color=('b'),
				label=('0~50m'), align="center", yerr=0.000001,edgecolor='black')
plt.bar(ind+width*1.5, distribution[:,1], width,color=('red'),
				label=('50~100m'), align="center", yerr=0.000001,edgecolor='black')
plt.bar(ind+width*2.5, distribution[:,2], width,color=('yellowgreen'),
				label=('100~150m'), align="center", yerr=0.000001,edgecolor='black')
plt.bar(ind+width*3.5, distribution[:,3], width,color=('green'),
				label=('150~200m'), align="center", yerr=0.000001,edgecolor='black')
plt.bar(ind+width*4.5, distribution[:,4], width,color=('m'),
				label=('>200m'), align="center", yerr=0.000001,edgecolor='black')
# plt.bar(ind+width*5, distribution[:,5], width,color=('b'),
# 				label=('step='+str(i+1)), align="center", yerr=0.000001,edgecolor='black')

# plt.bar(ind,y1,width,color = 'blue',label = 'm=2')
# plt.bar(ind+width,y2,width,color = 'g',label = 'm=4') # ind+width adjusts the left start location of the bar.
# plt.bar(ind+2*width,y3,width,color = 'c',label = 'm=6')
# plt.bar(ind+3*width,y4,width,color = 'r',label = 'm=8')
# plt.bar(ind+4*width,y5,width,color = 'm',label = 'm=10')
plt.xticks(np.arange(5) + 2.5*width, ('1','2','3','4','5'))

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Prediction step', fontsize=15)
plt.ylabel('Error distribution', fontsize=15)

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
# xticks = mtick.FormatStrFormatter(fmt)
# Set the formatter
# axes = plt.gca()   # get current axes
# axes.yaxis.set_major_formatter(xticks) # set % format to ystick.
# axes.grid(True)
plt.legend(loc="upper right",fontsize=13)
# plt.savefig('D:\\errorRate.eps', format='eps',dpi = 1000,bbox_inches='tight')

plt.show()