# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:31:36 2017

@author: Administrator
"""

import numpy as np
from utils_RMF import *
import os

sys.setdefaultencoding('utf8')


size = 'small'

TEST_OUT_FILE = os.environ.get("INfdfdPUT_DATA_FILE", "../data/generate/RMF-"+size+".txt")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "../data/test/test-"+size+"-30.txt")

NUM_POINT = int(os.environ.get("EACH_TIME_NUM_POINT", "5"))




states = []
data = []
x_test, trueTra = load_data_RMF(INPUT_DATA_FILE, NUM_POINT)

sys.stdout.flush()
for traId in trueTra:
	# states.append(x_test[traId])
	# data.append(trueTra[traId])
	test_RMF_model(x_test[traId], 10, TEST_OUT_FILE, NUM_POINT, trueTra[traId], traId)