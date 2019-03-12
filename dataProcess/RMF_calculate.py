# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:05:15 2017

@author: Administrator
"""

import numpy as np
import sys
reload(sys)  
sys.setdefaultencoding('utf8')
#from numpy import *

def RMF_calculate_one_step(s,l):
   U,W,V = np.linalg.svd(s, full_matrices=False)
   #print U
   #print W
   #print V
   W = np.diag(W)
   _W = np.zeros_like(W)
   for i in range(len(W)):
       for j in range(len(W[0])):
           if W[i][j]<1e-3:
               continue
           _W[i][j] = 1/W[i][j]
   #k = 0
   #print U.shape,W.shape,V.shape
   k = V.T.dot(_W).dot(U.T).dot(l)
   #print k
   return k

def RMF_calculate(s,l,f,h,d):
    assert s.shape[0] == h-f and s.shape[1] == d*f
    assert l.shape[0] == h-f and l.shape[1] == d
    c = np.zeros((d, d*f), dtype=float)
    _l = np.zeros((d, d), dtype=float)
    for i in range(d):
        _l[i][i] = 1
        for j in range(d*f):
            if (j+i)%d == 0:
                c[i][j] = 1

    # s = np.vstack((np.array(s), c))
    # l = np.vstack((l, _l))
    k = []
    for i in range(d):
        temp = RMF_calculate_one_step(s, l[:, i])
        k.append(temp)
    k = np.array(k)
    k = k.T
    assert k.shape[0] == d*f and k.shape[1] == d
    return k

def RMF_prediction(x_train,k):
    return x_train.dot(k)