import numpy as np
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
train = [ 108.088, 11.245, 5.441, 4.003, 3.289, 2.844, 2.581, 2.374, 2.224, 2.108, 2.009, 1.921, 1.832, 1.762, \
     1.705, 1.665, 1.637]
test = [ 35.071, 14.323, 10.991, 10.231, 10.117, 10.303, 10.990, 12.124, 12.634, 13.501, 14.120, 15.395, 16.103, 17.234, \
     18.032, 19.154, 19.852]
# train_file = "./data/gps-tf-train.txt"
# roadinfo_file = "./data/roadnet/RoadNetInfoV.txt"
# #trainfile = open(train_file, 'r')
# #roadinfo = open(roadinfo_file, 'r')
# train_data, data_size, _vocab_size = load_data_grid(train_file, roadinfo_file)
# for i in range(len(train_data)):
#     if not len(train_data[i]) == 30:
#         print(len(train_data[i]))
# print(type(train_data), type(train_data[0]))
# a = np.array(train_data)
# print(a[:, :2])
# x = range(len(train))
# plt.figure(1)
# plt.plot(x, train)
# plt.plot(x, test)
#
# plt.show()


# config_tf = tf.ConfigProto()
# config_tf.gpu_options.allow_growth = True
# config_tf.inter_op_parallelism_threads = 1
# config_tf.intra_op_parallelism_threads = 1
# with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
#      lstm = tf.contrib.rnn.BasicLSTMCell(100, state_is_tuple=True)
#      cell = tf.contrib.rnn.MultiRNNCell([lstm]*3, state_is_tuple=True)
#      state = cell.zero_state(50, dtype=tf.float32)
#      c, h, e = state
#      print(state)
#      print(c, h, e)
#      a = tf.concat(1, [[1], [2], [3]], [[4], [5], [6]])
#
#      session.run(a)
#coding=utf-8
import numpy as np

# 因为是生成随机数做测试，设置固定随机数种子，可以保证每次结果一致
a = [1, 2, 3]
b = [4, 5]
print(a+b)
