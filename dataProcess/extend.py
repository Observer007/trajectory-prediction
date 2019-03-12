import math
# ============================= grid 20 to 10 ===============================
size, grids, length, threshold = 'beijing', '100', '20', 100000
reduce_grid = True
# train_file = "../data/train/train-"+size+"-grid"+grids+"-"+length+".txt"
train_grid_file = "../data/train/train-"+size+"-grid"+grids+"-20.txt"
train_grid_output_file = "../data/train/train-"+size+"-grid"+grids+"-10.txt"
test_grid_file = "../data/test/test-"+size+"-grid"+grids+"-20.txt"
test_grid_output_file = "../data/test/test-"+size+"-grid"+grids+"-10.txt"
road_file = '../data/roadnet/road-'+size+'-'+grids+'.txt'
def DistanceBetweenMeter(geo1, geo2):
    R = 6378137
    lonA, latA = geo1[0]/180*math.pi, geo1[1]/180*math.pi
    lonB, latB = geo2[0]/180*math.pi, geo2[1]/180*math.pi
    return R*math.acos(min(1.0, math.sin(math.pi/2-latA)*math.sin(math.pi/2-latB)*
        math.cos(lonA-lonB) + math.cos(math.pi/2-latA)*math.cos(math.pi/2-latB)))
def read_road_info(road_file):
    grid_num = 0
    with open(road_file, 'r') as file:
        for line in file:
            if line.strip().__len__()>3:
                grid_num+=1
    grid2cor = []
    intra_feas = []
    with open(road_file, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            grid2cor.append([float(line[1]), float(line[2])])
            intra_feas.append([line[i] for i in range(3, line.__len__())])
    # print(DistanceBetweenMeter(grid2cor[4510], grid2cor[3082]))
    assert grid2cor.__len__() == grid_num
    return grid2cor, intra_feas

if reduce_grid:
    # ========= int to int dict =========================
    grid2new = dict()
    new2grid = dict()
    new_idx = 0
train_grids = set()
train_grid_output = open(train_grid_output_file, 'w')
with open(train_grid_file, 'r') as file:
    for line in file:
        line = line.strip().split(',')
        index = 0
        for i in range(1, 11):
            temp = line[0]+'_'+str(index)
            for j in range(11):
                if line[i+j].split(':')[0] not in train_grids:
                    train_grids.add(line[i+j].split(':')[0])
                    if reduce_grid:
                        new2grid[new_idx] = int(line[i+j].split(':')[0])
                        grid2new[int(line[i+j].split(':')[0])] = new_idx
                        new_idx += 1
                        if new_idx == 3000:
                            x = 1
                if reduce_grid:
                    temp += ','+str(grid2new[int(line[i+j].split(':')[0])])\
                            +':'+line[i+j].split(':')[1]
                else:
                    temp += ','+line[i+j]
            temp += '\n'
            train_grid_output.write(temp)
            index+=1
train_grid_output.close()
if reduce_grid:
    print('new idx is : %d' % new_idx)
    assert new_idx == train_grids.__len__()
print('train grids in unique is : %d' % train_grids.__len__())
grid2cor, intra_feas = read_road_info(road_file)
test_grid_output = open(test_grid_output_file, 'w')
with open(test_grid_file, 'r') as file:
    for line in file:
        line = line.strip().split(',')
        flag = True
        for i in range(1, 21):
            # tmp_grid = line[i].split(':')[0]
            # if tmp_grid not in train_grids:
            #     flag = False
            if i>1 and DistanceBetweenMeter(grid2cor[int(line[i].split(':')[0])],
                                            grid2cor[int(line[i-1].split(':')[0])])>threshold:
                print(DistanceBetweenMeter(grid2cor[int(line[i].split(':')[0])],
                                            grid2cor[int(line[i-1].split(':')[0])]))
                flag = False
        if flag:
            index = 0
            for i in range(1, 11):
                temp = line[0]+'_'+str(index)
                for j in range(11):
                    if reduce_grid:
                        if int(line[i+j].split(':')[0]) not in grid2new:
                            new2grid[new_idx] = int(line[i + j].split(':')[0])
                            grid2new[int(line[i + j].split(':')[0])] = new_idx
                            new_idx += 1
                        temp += ',' + str(grid2new[int(line[i + j].split(':')[0])]) \
                                + ':' + line[i + j].split(':')[1]
                    else:
                        temp += ',' + line[i + j]
                temp += '\n'
                test_grid_output.write(temp)
                index+=1
test_grid_output.close()
if reduce_grid:
    with open(road_file, 'w') as file:
        for i in range(new2grid.__len__()):
            tmp = [str(i)]
            tmp_coor = grid2cor[new2grid[i]]
            assert tmp_coor.__len__() == 2
            tmp_coor = [str(j) for j in tmp_coor]
            tmp += tmp_coor
            tmp += intra_feas[new2grid[i]]
            file.write('\t'.join(tmp) + '\n')
    print('update road info done')
# ========================== coor 20 to 10 =================================
if not size == 'beijing':
    coor_file = '../data/train/train-'+size+'-20-selected.txt'
else:
    coor_file = '../data/train/train-beijing-20.txt'
coor_output_file = '../data/train/train-'+size+'-10.txt'

coor_output = open(coor_output_file, 'w')
with open(coor_file, 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        index = 0
        for i in range(10):
            temp = line[0]+'_'+str(index)
            for j in range(11):
                # print i+j
                temp += '\t'+line[3*(i+j)+1]+'\t'+line[3*(i+j)+2]+'\t'+line[3*(i+j)+3]
            temp += '\n'
            coor_output.write(temp)
            index+=1
coor_output.close()
print('all done!')

if not size == 'beijing':
    coor_file = '../data/test/test-'+size+'-20-selected.txt'
else:
    coor_file = '../data/test/test-beijing-20.txt'
coor_output_file = '../data/test/test-'+size+'-10.txt'

coor_output = open(coor_output_file, 'w')
with open(coor_file, 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        index = 0
        for i in range(10):
            temp = line[0]+'_'+str(index)
            for j in range(11):
                # print i+j
                temp += '\t'+line[3*(i+j)+1]+'\t'+line[3*(i+j)+2]+'\t'+line[3*(i+j)+3]
            temp += '\n'
            coor_output.write(temp)
            index+=1
coor_output.close()
print('all done!')