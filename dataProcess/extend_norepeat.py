import math
# ============================= grid 20 to 10 ===============================
size, grids, length = 'chengdu', '50', '20'
reduce_grid = True
# train_file = "../data/train/train-"+size+"-grid"+grids+"-"+length+".txt"
grid_file = "../data/train/train-"+size+"-grid"+grids+"-20.txt"
grid_output_file = "../data/train/train-"+size+"-grid"+grids+"-10.txt"
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
    print(DistanceBetweenMeter(grid2cor[4510], grid2cor[3082]))
    assert grid2cor.__len__() == grid_num
    return grid2cor, intra_feas
grid2cor, intra_feas = read_road_info(road_file)

if reduce_grid:
    # ========= int to int dict =========================
    grid2new = dict()
    new2grid = dict()
    new_idx = 0
dis_count, total_dis, tra_count = 0, 0, 0
train_grid_set = set()
grid_output = open(grid_output_file, 'w')
with open(grid_file, 'r') as file:
    for line in file:
        line = line.strip().split(',')
        assert line.__len__() == 21
        grid_line = line[1:]
        tmp_grid_set = set()
        grid_index = []
        flag = False
        for i in range(20):
            if grid_line[i].split(':')[0] not in tmp_grid_set:
                tmp_grid_set.add(grid_line[i].split(':')[0])
                grid_index.append(i)

        assert grid_index.__len__() == tmp_grid_set.__len__()
        for i in range(grid_index.__len__()-1):
            tmp_dis = DistanceBetweenMeter(grid2cor[int(grid_line[grid_index[i]].split(':')[0])],
                                           grid2cor[int(grid_line[grid_index[i+1]].split(':')[0])])
            total_dis += tmp_dis
            dis_count += 1
        if tmp_grid_set.__len__()>=11:
            flag = True
            for item in tmp_grid_set:
                if item not in train_grid_set:
                    train_grid_set.add(item)
                    if reduce_grid:
                        grid2new[int(item)] = new_idx
                        new2grid[new_idx] = int(item)
                        new_idx += 1
        if flag:
            index = 0
            for i in range(tmp_grid_set.__len__()-10):
                tra_count += 1
                temp = line[0]+'_'+str(index)
                for j in range(11):
                    if reduce_grid:
                        temp += ','+str(grid2new[int(grid_line[grid_index[i+j]].split(':')[0])])\
                                +':'+grid_line[grid_index[i+j]].split(':')[1]
                    else:
                        temp += ','+grid_line[grid_index[i+j]]
                temp += '\n'
                grid_output.write(temp)
                index+=1
grid_output.close()
print('train average distance with no repeat is: %f' % (total_dis/dis_count))
print('train total tra is : %d' % tra_count)
print('train grid in unique is : %d/%d' % (train_grid_set.__len__(), grid2cor.__len__()))
grid_file = "../data/test/test-"+size+"-grid"+grids+"-20.txt"
grid_output_file = "../data/test/test-"+size+"-grid"+grids+"-10.txt"
road_file = '../data/roadnet/road-'+size+'-'+grids+'.txt'
dis_count, total_dis, tra_count = 0, 0, 0
test_grid_set = set()
grid_output = open(grid_output_file, 'w')
with open(grid_file, 'r') as file:
    for line in file:
        line = line.strip().split(',')
        assert line.__len__() == 21
        grid_line = line[1:]
        tmp_grid_set = set()
        grid_index = []
        flag = False
        for i in range(20):
            if grid_line[i].split(':')[0] not in tmp_grid_set:
                tmp_grid_set.add(grid_line[i].split(':')[0])
                grid_index.append(i)
        assert grid_index.__len__() == tmp_grid_set.__len__()
        for i in range(grid_index.__len__()-1):
            tmp_dis = DistanceBetweenMeter(grid2cor[int(grid_line[grid_index[i]].split(':')[0])],
                                           grid2cor[int(grid_line[grid_index[i+1]].split(':')[0])])
            total_dis += tmp_dis
            dis_count += 1
        if tmp_grid_set.__len__()>=11 and (tmp_grid_set-train_grid_set).__len__()==0:
            test_grid_set = tmp_grid_set|test_grid_set
            flag = True
        if flag:
            index = 0
            for i in range(tmp_grid_set.__len__()-10):
                tra_count += 1
                temp = line[0]+'_'+str(index)
                for j in range(11):
                    if reduce_grid:
                        temp += ','+str(grid2new[int(grid_line[grid_index[i+j]].split(':')[0])])\
                                +':'+grid_line[grid_index[i+j]].split(':')[1]
                    else:
                        temp += ','+grid_line[grid_index[i+j]]
                temp += '\n'
                grid_output.write(temp)
                index+=1
grid_output.close()
print('test average distance with no repeat is: %f' % (total_dis/dis_count))
print('test total tra is : %d' % tra_count)
print('test grid in unique is : %d/%d' % (test_grid_set.__len__(), grid2cor.__len__()))
print('train test grid in unique is : %d/%d' %((train_grid_set|test_grid_set).__len__(), grid2cor.__len__()))
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
coor_file = '../data/train/train-'+size+'-20-selected.txt'
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


coor_file = '../data/test/test-'+size+'-20-selected.txt'
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