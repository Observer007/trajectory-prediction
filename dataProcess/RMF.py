# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:31:36 2017

@author: Administrator
"""

import numpy as np
from utils_RMF import *
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')

def DistanceBetweenMeter(geo1, geo2):
    R = 6378137
    lonA, latA = geo1[0]/180*math.pi, geo1[1]/180*math.pi
    lonB, latB = geo2[0]/180*math.pi, geo2[1]/180*math.pi
    return R*math.acos(min(1.0, math.sin(math.pi/2-latA)*math.sin(math.pi/2-latB)*
        math.cos(lonA-lonB) + math.cos(math.pi/2-latA)*math.cos(math.pi/2-latB)))
def calDistance(tras1, tras2, start=5):
    assert tras1.__len__() == tras2.__len__()
    dis_error = np.zeros([5])

    num_count = np.zeros([5])
    min_error, max_error = np.ones([5]) * 100000, np.zeros([5])
    for t in range(len(tras1)):
        tra1 = tras1[t][start:]
        tra2 = tras2[t][start:]
        for i in range(5):
            dis = DistanceBetweenMeter(tra1[i], tra2[i])

            dis_error[i] += dis
            num_count[i] += 1

            min_error[i] = min(min_error[i], dis)
            max_error[i] = max(max_error[i], dis)
    tra_num = num_count[0]
    dis_error = dis_error / num_count
    num_count = num_count / len(tras1)
    return tra_num, dis_error, num_count, min_error, max_error
city = 'beijing'

baseline_file = '../data/test/test-' + city + '-baseline.txt'


tras = load_data_RMF(baseline_file)
true_tras, rmf_tras = [], []
for traId in tras.id2tra:
    # states.append(x_test[traId])
    # data.append(trueTra[traId])
    tras.id2tra[traId] = test_RMF_model(tras.id2tra[traId], h=5, f=4)
    true_tras.append(tras.id2tra[traId].coors)
    rmf_tras.append(tras.id2tra[traId].pre_rmf)
print('calculate done!')
tra_num, dis_error, num_count, min_error, max_error = calDistance(true_tras, rmf_tras)
print(dis_error.mean())
print(dis_error)
print(min_error)
print(max_error)





