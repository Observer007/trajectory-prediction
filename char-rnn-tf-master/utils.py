#coding:utf-8

import numpy as np
import Config

def load_data_road(filename="../data/gpsVector0701.txt", roadnetname="./RoadNetInfo.txt"):
    '''load grid sequence data for trajectory, trajectory has been matched to the road network'''
    print("Reading trajectory file...")
    # gridIdMax  = 0
    netinfo = open(roadnetname, 'r')
    data_size = 0
    road_size = 0
    train_data = []
    for line in netinfo:
        road_size = road_size + 1

    with open(filename, 'rt') as f:
        for line in f:
            data_size = data_size + 1
            data = []
            line = line.strip()
            line = line.split(',')
            for i in range(1, len(line)):
                gpsG = line[i].split(':')
                gridId = int(gpsG[0])
                time = int(gpsG[1])
                data.append(gridId)
            train_data.append(data)
    data_size = len(train_data)
    return train_data, data_size, road_size

def load_data_grid(filename="../data/gpsVector0701.txt", roadnetname="./RoadNetInfo.txt"):
    '''load grid sequence data for trajectory, trajectory has been matched to the road network'''
    print("Reading trajectory file...")
    # gridIdMax  = 0
    netinfo = open(roadnetname, 'r')
    data_size = 0
    road_size = 0
    train_data = {}
    neargrid = []
    grid2cor = []
    types = Config.Config.add_dim - 6
    type, tunnel, bridge, oneway = [], [], [], []
    for line in netinfo:
        line = line.strip().split('\t')
        if len(line) == 7:
            grid2cor.append([float(line[1]), float(line[2])])
            if types>=0:
                tmptype, tmptunnel, tmpbridge, tmponeway = np.zeros(types), np.zeros(2), np.zeros(2), np.zeros(2)
                tmptype[int(line[3])], tmptunnel[int(line[4])], tmpbridge[int(line[5])], tmponeway[int(line[6])] = 1, 1, 1, 1
                type.append(tmptype.tolist()), tunnel.append(tmptunnel.tolist()), bridge.append(tmpbridge.tolist()), oneway.append(tmponeway.tolist())
            road_size = road_size + 1
        if len(line) >= 21:
            tmp = [int(x) for x in line[1:]]
            tmp.sort()
            neargrid.append(tmp)
    print(len(type), len(tunnel), len(bridge), len(oneway), len(grid2cor))
    with open(filename, 'r') as f:
        for line in f:
            data_size = data_size + 1
            data = []
            line = line.strip()
            line = line.split(',')
            for i in range(1, len(line)):
                gpsG = line[i].split(':')
                gridId = int(gpsG[0])
                time = int(gpsG[1])
                if types==0:
                    data.append(gridId)
                else:
                    data.append(gridId)
            train_data[line[0]] = data
    data_size = len(train_data)
    features = [grid2cor, type, tunnel, bridge, oneway]
    return train_data, features, data_size, road_size, neargrid