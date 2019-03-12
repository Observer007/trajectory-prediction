# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:05:15 2017

@author: Administrator
"""

import numpy as np

#from numpy import *

def RMF_calculate_one_step(s,l):
   U,W,V = np.linalg.svd(s,full_matrices = False)
   #print U
   #print W
   #print V
   W = np.diag(W)
   _W = np.zeros_like(W)
   for i in range(len(W)):
       for j in range(len(W[0])):
           if W[i][j]<1e-2:
               continue
           _W[i][j] = 1/W[i][j]
   #k = 0
   #print U.shape,W.shape,V.shape
   k = V.T.dot(_W).dot(U.T).dot(l)
   #print k
   return k


def RMF_calculate(x_train, data, f, h, d):
    s = []
    l = []
    k = []
    #print x_train
    for i in range(h-f):
        s.append(x_train[-(i+2)])
        l.append(data[-(i+1)])
    l = np.array(l)
    #print l
    l = l.T
    # print(l.shape)
    c = np.zeros((d, d*f),dtype = float)
    _l = np.zeros((d, d),dtype = float)
    for i in range(d):
        _l[i][i] = 1            
        for j in range(d*f):
            if (j+i)%d==0:
                c[i][j] = 1
    #print c
    #print _l
    #print l
    s = np.vstack((np.array(s), c))
    l = np.hstack((l, _l))
    for i in range(d):
        temp = RMF_calculate_one_step(s, l[i])
        k.append(temp)
    k = np.array(k)
    #print k
    return k

def RMF_prediction(x_train,k):
    #k = np.array(k)
    return k.dot(x_train)