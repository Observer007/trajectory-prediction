import numpy as np
import math
id = 'LnxMB9evozs6BouF_irH0cowsiNv1gir0_0'
size, grids = 'chengdu', '50'
grid_output_file = "../data/test/test-"+size+"-grid"+grids+"-10.txt"
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
grid2cor, intra_feas = read_road_info(road_file)

with open(grid_output_file, 'r') as file:
    for line in file:
        line = line.strip().split(',')
        grid_line = line[1:]
        dis_matrix = np.zeros([11,11])
        if id == line[0]:
            for i in range(grid_line.__len__()-1):
                for j in range(i+1, grid_line.__len__()):
                    dis_matrix[i][j] = \
            DistanceBetweenMeter(grid2cor[int(grid_line[i].split(':')[0])],
                                 grid2cor[int(grid_line[j].split(':')[0])])

            print(dis_matrix)