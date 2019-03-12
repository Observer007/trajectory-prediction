
import numpy as np
import pandas as pd
import time
grids = []
gridgap = 50
nearest = 40
size = 'small'

def caldistance(point1, point2):
	return pow(pow(point2[1]-point1[1], 2)+pow(point2[0]-point1[0], 2), 0.5)

with open("../data/roadnet/road-"+size+'-'+str(gridgap)+".txt", 'r') as file:
	for line in file:
		line = line.strip().split('\t')
		if len(line)==3:
			grids.append([float(line[1]), float(line[2])])

length = len(grids)
print length
distance = np.zeros((length, length))
total = []
for i in xrange(length):
	if not i%1000:
		print i
	for j in xrange(length):
		if j<i:
			distance[i][j] = distance[j][i]
		elif j==i:
			continue
		else:
			distance[i][j] = caldistance(grids[i], grids[j])
			#total.append(distance[i][j])
distance0 = pd.DataFrame(distance, index=range(length), columns=range(length))
#distance1 = pd.Series(total, index=range(length*(length-1)/2))


begin = time.time()
with open("../data/roadnet/road-"+size+"-"+str(gridgap)+".txt", 'a') as file:
	for i in range(length):
		if not i%100:
			print i
		tmp0 = distance0.sort_values(by=i).index[1:nearest+1]
		str0 = str(i)
		for t in tmp0:
			str0 += '\t' + str(t)
		str0 += '\n'
		file.write(str0)
end = time.time()
print end - begin

# begin = time.time()
# near = [[] for i in range(length)]
# compare = np.zeros(length).tolist()
# #print compare
# for i in range(length):
# 	compare[i] = (2*length-i-1)*i/2
# with open("../data/roadnet/road-"+size+"-"+str(gridgap)+".txt", 'a') as file:
# 	tmp = distance1.sort_values().index[:length*nearest/2]
# 	for t in range(len(tmp)):
# 		if not t%10000:
# 			print t
# 		if tmp[t] < length-1:
# 			i = 0
# 			j = tmp[t]+1
# 		else:
# 			i = 0
# 			while compare[i]<=tmp[t]:
# 				i = i + 1
# 			i = i-1
# 			j = tmp[t]-compare[i]+i+1
# 		if not len(near[i]) == 20:
# 			near[i].append(j)
# 		if not len(near[j]) == 20:
# 			near[j].append(i)
# 	for i in range(length):
# 		#if not i%10000:
# 		print i
# 		str0 = str(i)
# 		for t in range(nearest):
# 			print t
# 			str0 += '\t' + str(near[i][t])
# 		str0 += '\n'
# 		file.write(str0)
# end = time.time()
# print end - begin
