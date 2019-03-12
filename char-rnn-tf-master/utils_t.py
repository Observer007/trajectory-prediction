#coding:utf-8

#import numpy as np
import Config
import numpy as np
import time
from sklearn.model_selection import train_test_split
class Trajector_Grid:
    def __init__(self, id, grids, weekday, ground_truth=None):
        self.id = id
        self.grids = grids
        self.weekday = weekday
        self.ground_truth = ground_truth

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
                time0 = int(gpsG[1])
                data.append(gridId)
            train_data.append(data)
    data_size = len(train_data)
    return train_data, data_size, road_size
def read_intra_embeddings(road_file, city='chengdu'):
    embeddings = None
    grids = 0
    with open(road_file, 'r') as file:
        for line in file:
            if line.strip().__len__()>3:
                grids+=1
    if city == 'chengdu':
        emb_dim = 33
    elif city == 'shanghai':
        emb_dim = 38
    elif city == 'beijing':
        emb_dim = 34
    else:
        raise ValueError('wrong city')
    embeddings = np.zeros([grids, emb_dim], dtype=np.float32)
    grid2cor = []
    grid = 0
    with open(road_file, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            if len(line) == 7:
                grid2cor.append([float(line[1]), float(line[2])])
                embeddings[grid][int(line[3])] = 1
                embeddings[grid][int(line[4])+emb_dim-6] = 1
                embeddings[grid][int(line[5])+emb_dim-4] = 1
                try:
                    embeddings[grid][int(line[6])+emb_dim-2] = 1
                except:
                    if line[6] == 'yes':
                        embeddings[grid][emb_dim - 1] = 1
            else:
                raise ValueError('wrong road')
            grid += 1
    return grid2cor, embeddings

def read_inter_embeddings(embed_file):
    line_count = 0
    embeddings = None
    with open(embed_file, 'r') as file:
        for line in file:
            line_count+=1
            if line_count == 1:
                line = line.strip().split(' ')
                assert line.__len__()==2
                embeddings = np.zeros([int(line[0]), int(line[1])], dtype=np.float32)
                continue
            line = line.strip().split(' ')
            if len(line) < 2:
                continue
            elif len(line)<10:
                raise ValueError('wrong embeddings')
            line0 = line[1:]
            for i in range(len(line0)):
                embeddings[int(line[0])][i] = float(line0[i])
    return embeddings

def read_tra_grids(grid_file, max_grids=0, city='chengdu'):

    tra_num = 0
    num_step = 0
    with open(grid_file, 'r') as f:
        for line in f:
            if line.strip().__len__()>0:
                tra_num += 1
            if tra_num == 1:
                line = line.strip().split(',')
                num_step = line.__len__()-1
    all_tra_data = []
    with open(grid_file, 'r') as f:
        for line in f:
            data = []
            times = [0]
            line = line.strip().split(',')
            traId = line[0]
            date = int(float(line[1].split(':')[1]))
            grids = []
            '''shanghai'''
            if city == 'shanghai':
                time_interval = 10
                date = date // 100000
                year = '2014'
                month = (date - 1488) // 31
                day = (date - 1488) % 31
                if month < 10:
                    month = '0' + str(month)
                elif month > 12:
                    month = '0' + str(month % 12)
                    year = '2015'
                else:
                    month = str(month)
                if day < 10:
                    if day == 0:
                        day = 1
                    date = year + month + '0' + str(day)
                else:
                    date = year + month + str(day)
                # print(date)
                vec = time.strptime(date, '%Y%m%d').tm_wday
                tempweek = np.zeros(9)
                tempweek[vec] = 1
                if vec < 5:
                    tempweek[7] = 1
                else:
                    tempweek[8] = 1
                weekday = tempweek

            elif city == 'chengdu':
                time_interval = 6
                date = date - 1477988400
                date = int(date // 86400) + 1
                if date < 10:
                    date = '2016110' + str(date)
                else:
                    date = '201611' + str(date)
                vec = time.strptime(date, '%Y%m%d').tm_wday
                tempweek = np.zeros(9)
                tempweek[vec] = 1
                if vec < 5:
                    tempweek[7] = 1
                else:
                    tempweek[8] = 1
                weekday = tempweek
            elif city == 'beijing':
                date = date//86400
                year = date//365
                month = date%365//31
                day = date%365%31
                str_year = str(year)
                if month<10:
                    str_month = '0'+str(month)
                else:
                    str_month = str(month)
                if day<10:
                    str_day = '0'+str(day)
                else:
                    str_day = str(day)
                date = str_year+str_month+str_day
                vec = time.strptime(date, '%Y%m%d').tm_wday
                tempweek = np.zeros(9)
                tempweek[vec] = 1
                if vec < 5:
                    tempweek[7] = 1
                else:
                    tempweek[8] = 1
                weekday = tempweek
            else:
                raise ValueError('wrong city')

            for i in range(1, len(line)):
                # print(len(line))
                gpsG = line[i].split(':')
                gridId = int(gpsG[0])
                assert gridId<max_grids
                grids.append(gridId)

            assert grids.__len__()==num_step
            tra_grid = Trajector_Grid(traId, grids, weekday)
            all_tra_data.append(tra_grid)
    all_tra_data = np.array(all_tra_data)
    return all_tra_data

def load_data_grid(filename, roadnetname, embedname, city):
    '''load grid sequence data for trajectory, trajectory has been matched to the road network'''
    print("Reading trajectory file...")

    grid2cor, intra_embedings = read_intra_embeddings(roadnetname, city)
    inter_embeddings = read_inter_embeddings(embedname)
    tra_data = read_tra_grids(filename, len(grid2cor), city)

    return tra_data, inter_embeddings, intra_embedings, grid2cor
def read_road_near(filename, config=None):
    grid_near = []
    grid_count = 0
    with open(filename, 'r') as file:
        for line in file:
            if line.__len__() == 0:
                continue
            line = line.strip().split('\t')
            line = [int(i) for i in line]
            grid_near.append(line)
    return grid_near
def train_val_split(train_data, test_ratio=0.2):
    """

    :param train_data: type Trajector_Grid
    :return:
    """
    train_data, val_data = \
        train_test_split(train_data, test_size=test_ratio,
                         random_state=0, shuffle=False)
    train_data, _ = train_test_split(train_data, test_size=0,random_state=0, shuffle=True)
    # train_ids = set()
    # for tra in train_data:
    #     if tra.id.split('_')[0] not in train_ids:
    #         train_ids.add(tra.id.split('_')[0])
    # inter_num = 0
    # for tra in val_data:
    #     if tra.id.split('_')[0] in train_ids:
    #         inter_num+=1
    # print(inter_num)
    return train_data, val_data

def cal_acc(result, y):
    assert result.shape.__len__() == 2
    correct = (result == y).sum()
    total = result.shape[0]*result.shape[1]
    return correct/total