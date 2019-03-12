import numpy as np
import math

city = 'beijing'
grid = 300
threshold = 800
road_file = '../data/roadnet/road-' + city + '-' + str(grid)+'.txt'
near_road_file = '../data/roadnet/road-'+city+'-'+ str(grid)+'-near.txt'
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
    with open(road_file, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            grid2cor.append([float(line[1]), float(line[2])])
    # print(DistanceBetweenMeter(grid2cor[4510], grid2cor[3082]))
    assert grid2cor.__len__() == grid_num
    return grid2cor

def find_near(grid2_cor, threshold=100):
    grid_num = len(grid2_cor)
    print(grid_num)
    index = 0
    grid_near = [[i] for i in range(grid_num)]
    for i in range(grid_num):
        if i%500 == 0:
            print(i)
        for j in range(i+1, grid_num):
            dis = DistanceBetweenMeter(grid2_cor[i], grid2_cor[j])
            if dis < threshold:
                grid_near[i].append(j)
                grid_near[j].append(i)
    return grid_near
def output_near(near_road_file, grid_near):
    with open(near_road_file, 'w') as file:
        for i in range(len(grid_near)):
            for j in range(len(grid_near[i])):
                if j:
                    file.write('\t')
                file.write(str(grid_near[i][j]))
            file.write('\n')
    print('save done!')

if __name__ == '__main__':
    grid2cor = read_road_info(road_file)
    grid_near = find_near(grid2cor, threshold)
    output_near(near_road_file, grid_near)
