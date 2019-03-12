# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:53:47 2017

@author: Administrator
"""


"""
Created on Wed Mar 29 20:38:47 2017

@author: Administrator
"""

#! /usr/bin/env python

import csv
import itertools
import numpy as np
import time
import sys
import operator
import io
import array
from datetime import datetime

from RMF_calculate import *
#from gru_theano import GRUTheano
import math
reload(sys)  
sys.setdefaultencoding('utf8')


def load_data_RMF(filename="../data/gpsVector0701.txt", prePointNum = 10):

    print("Reading trajectory file...")
    with open(filename, 'rt') as f:
        data = {}
        x_train = {}
        for line in f:
            traSeg = []
            line =  line.strip()
            line =  line.split('\t')
            #print line
            if len(line)<4:
                print 'error'
                break
            for i in range(1, len(line), 3):
                longitude = float(line[i])
                latitude = float(line[i+1])
                traSeg.append([longitude, latitude])    # true longitude and latitude
                #print traSeg
            #print traSeg
            if len(traSeg)>prePointNum+1:
                data[line[0]] = traSeg
                x_train[line[0]] = np.asarray([np.array(traSeg[i-prePointNum:i][::-1]).flatten().tolist()
                                                for i in range(prePointNum, len(traSeg))])
    # x_train = np.asarray([[np.array(traSeg[i-prePointNum:i][::-1]).flatten().tolist() for i in range(prePointNum, len(traSeg))] for traSeg in data])  # should be range(prePointNum-1,...)

    return x_train, data
    
    
def test_RMF_model(x_test, tIndex, fileName, NUM_POINT, trueTra, traId):
    traNum = len(x_test)
    # print x_test
    outfile = open(fileName, 'a')
    for ti in range(1):

        testTra =  x_test
        truTra  =  trueTra
        #print truTra
        new_trajectory = testTra[:tIndex-NUM_POINT+1]    #fold state
        tru_trajectory = truTra[:tIndex]                 #location
        # print new_trajectory.shape, tru_trajectory.shape
        new_trajectory = new_trajectory.tolist()


        for tj in range(tIndex, len(truTra)):

            k = RMF_calculate(new_trajectory, tru_trajectory, NUM_POINT, tIndex, 2) #2 is dim

            next_loc = RMF_prediction(new_trajectory[-1], k).tolist()


            tru_trajectory.append(next_loc)
            new_trajectory.append(np.array(tru_trajectory[-5:][::-1]).flatten().tolist())
        outfile.write(traId+',')
        for point in truTra:
            longitude = point[0]
            latitude  = point[1]
            outfile.write(str(longitude)+','+ str(latitude)+',')
        
        outfile.write('\n')
        outfile.write(traId+',')
        for point in tru_trajectory:
            longitude = point[0]
            latitude  = point[1]
            outfile.write(str(round(longitude, 5))+','+ str(round(latitude, 5))+',')
        outfile.write('\n')
        print 'y_test len:', len(x_test), 'new_trajectory len:', len(tru_trajectory), 'tr len:', len(trueTra)
    outfile.close()


