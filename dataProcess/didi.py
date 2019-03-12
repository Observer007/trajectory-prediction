
lonMin = 104.04
lonMax = 104.09
latMin = 30.64
latMax = 30.69
data = 0
# outputfile = open('../data/DIDI/train/diditrain.txt', 'w')
# trajectorys = []
#
# for i in range(4, 30):
#     print i
#     if i<10:
#         inputfile = '../data/DIDI/gps_2016110'+str(i)
#     else:
#         inputfile = '../data/DIDI/gps_201611'+str(i)
#     with open(inputfile, 'r') as file:
#         id = ''
#         trajectory = []
#         tralen = 0
#         flag = 1
#         for line in file:
#             line = line.strip().split(',')
#             if not len(id):
#                 index = 0
#                 id = line[1]
#                 trajectory = ['\t' + line[3] + '\t' + line[4] + '\t' + line[2]]
#                 tralen = 1
#                 pretime = int(line[2])
#             if not id==line[1]:
#                 if tralen >= 20 and flag:
#                     for i in range(tralen-20):
#                         id0 = id+str(i)
#                         outputfile.write(id0)
#                         for j in range(20):
#                             # print type(trajectory[i+j])
#                             outputfile.write(trajectory[i+j])
#                         outputfile.write('\n')
#                 id = line[1]
#                 flag = 1
#                 tralen = 1
#                 index = 0
#                 trajectory = ['\t' + line[3] + '\t' + line[4] + '\t' + line[2]]
#                 pretime = int(line[2])
#             else:
#                 if int(line[2])-pretime >=2 and int(line[2])-pretime <=4:
#                     index += 1
#                     pretime = int(line[2])
#                 else:
#                     flag = 0
#                 if index % 2 == 0:
#                     trajectory.append('\t' + line[3] + '\t' + line[4] + '\t'+ line[2])
#                     tralen+=1
new_tra = []
index = 0
outputfile = open('../data/DIDI/train/diditrain.txt', 'r')
newfile = open('../data/DIDI/train/diditrain-20.txt', 'w')
for line in outputfile:
    index+=1
    if index%100==0:
        print index
    line0 = line.strip().split('\t')
    flag = 1
    for i in range(1, len(line0), 3):
        if float(line0[i])>lonMax or float(line0[i])<lonMin or float(line0[i+1])>latMax or float(line0[i+1]<latMin):
            flag = 0
            break
    if flag:
        newfile.write(line)
