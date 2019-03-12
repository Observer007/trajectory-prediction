#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import pickle, os
import random
import pandas as pd
import Config
import T_Model
from utils_t import *
# from RMF_calculate import *

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()
types = config.add_dim
holiday = config.holiday
size = config.size
grids = '50'
batch_size = config.batch_size
hidden = str(config.hidden_size)
idim = input_dim = config.input_dim
nearest = config.nearest
num_steps = config.num_steps
length = str(10)
roadinfo = '../useroadnet/generate_data/data/roadnet/road-'+size+'-'+grids+'.txt'
testfile = '../useroadnet/generate_data/data/test/test-'+size+'-grid'+grids+'-'+length+'.txt'
# testfile = '../useroadnet/generate_data/data/test/test-small.txt'
genfile = '../useroadnet/generate_data/data/generate/result-'+size+'-'+holiday+'-'+str(types)+'-'+str(input_dim)+'-'+length+'.txt'
# genfile = '../useroadnet/generate_data/data/generate/result-small-noholiday.txt'
embedding_file = '../useroadnet/generate_data/data/roadnet/road-'+ size + '-' + grids + '-' + str(input_dim) + '-new.embedding'

with open(genfile, 'w') as f:
    print('build!')
# roadinfo = '../data/roadnet/road-'+size+'-'+grids+'.txt'
# testfile = '../data/test/test-'+size+'-grid'+grids+'.txt'
# genfile = '../data/generate/result-'+size+'-grid'+grids+'.txt'

test_data0, embedding, features, test_time0, data_size, vocab_size, neargrid = load_data_grid(testfile, roadinfo, embedding_file, size)
[grid2cor, type, tunnel, bridge, oneway] = features
# print(test_data0)
test_data = []
test_name = []
test_time = []
for traId in test_data0:
    test_data.append(test_data0[traId])
    test_name.append(traId)
    test_time.append(test_time0[traId])
mean, var = 6.36721020342, 30.8796707124
# print(test_time)
# print(neargrid)
pre_position = 5
config.vocab_size = vocab_size
is_sample = config.is_sample
is_beams = config.is_beams
beam_size = config.beam_size
len_of_generation = config.len_of_generation
predict = config.predict


def data_iterator(raw_data, test_name, test_time, batch_size, num_steps):
    if types!=0:
        raw_data0 = [[raw_data[i][j][0][:int(idim)]+raw_data[i][j][1]+raw_data[i][j][2]+raw_data[i][j][3]+raw_data[i][j][4] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    else:
        raw_data0 = [[raw_data[i][j][0][:int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data1 = [[raw_data[i][j][0][int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data0 = np.array(raw_data0)
    raw_data1 = np.array(raw_data1)
    test_name = np.array(test_name)
    test_time = np.array(test_time)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    union_data = []
    for i in range(int(batch_len)):
        tmp = raw_data0[i * batch_size:(i + 1) * batch_size]
        tmp1 = raw_data1[i * batch_size:(i + 1) * batch_size]
        tmp_name = test_name[i * batch_size:(i + 1) * batch_size]
        tmp_time = test_time[i * batch_size:(i + 1) * batch_size]
        x = tmp[:, :num_steps, :]
        g = tmp1[:, :num_steps]
        y = tmp_name  # y就是x的错一位，即下一个词
        t = tmp_time
        print(g.shape)
        # y = np.reshape(y, (y.shape[0], -1))
        union_data.append([x, g, y, t])
    return union_data



def run_epoch(session, m, data, time, eval_op, state=None):
    """Runs the model on the given data."""
    # x = data.reshape((1, 1, input_dim+types))
    # t = np.array(time).reshape((1, 9))
    # data = np.reshape(data[:, :, config.input_dim:], [-1, config.add_dim])
    # print(data.shape, time.shape)
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                         {m.input_data: data,
                          m.extra_data: time,
                          m.initial_state: state})
    return prob, _state

def embedding_all(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(np.float32(embedding[data[i]][:input_dim]+type[data[i]]+
                                   tunnel[data[i]]+bridge[data[i]]+oneway[data[i]]))
    new_data = np.reshape(np.array(new_data), [batch_size, 1, -1])
    # print(new_data.shape)
    return new_data

def only_embedding(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(np.float32(embedding[data[i]][:input_dim]))
    new_data = np.reshape(np.array(new_data), [batch_size, 1, -1])
    # print(new_data.shape)
    return new_data

def out_file(file, g, new_g, y):
    for i in range(len(y)):
        with open(file, 'a') as output:
            output.write(str(y[i])+',')
            for j in range(10):
                output.write(str(g[i][j])+',')
            output.write('\n')
            output.write(str(y[i]) + ',')
            for j in range(10):
                output.write(str(new_g[i][j]) + ',')
            output.write('\n')



def out_file1(file, tra, traId):
    with open(file, 'a') as output:
        output.write(str(traId)+',')
        for point in tra:
            output.write(str(point)+',')
        output.write('\n')

def out_file2(file, tra, traId):
    with open(file, 'a') as output:
        output.write(str(traId)+',')
        for point in tra:
            output.write(str(point)+',')
        output.write('\n')

def caldistance(point1, point2):
    return pow(pow(point2[1]-point1[1], 2)+pow(point2[0]-point1[0], 2), 0.5)


def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        # config.batch_size = 1
        config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = T_Model.T_Model(is_training=False, holiday=holiday, config=config)

        #tf.global_variables_initializer().run()

        model_saver = tf.train.Saver()
        print('model loading ...')
        # model_saver.restore(session, config.model_path+'-'+hidden+'-'+str(input_dim)+'-'+size+'-tgrid'+grids+'-%d' % config.save_time)
        print(config.model_path + '-' + hidden + '-' + str(
            input_dim + types) + '-' + size + '-tgrid' + grids + '-' + holiday + '-%d' % config.save_time)
        model_saver.restore(session, config.model_path+'-'+hidden+'-'+str(input_dim+types)+'-'+size+'-tgrid'+grids+'-'+holiday+'-%d' % config.save_time)
        f_data = data_iterator(test_data, test_name, test_time, batch_size, num_steps)
        print('Done!')
        timestamp = 0
        total_time = 0
        index = 0
        for step, temp in enumerate(f_data):
            x = temp[0]
            # print(x)
            g = temp[1]
            y = temp[2]
            t = temp[3]
            new_g = g[:, :5]
            # print(t.shape)
            if not index%50:
                print(index)
            # if index==700:
            #     break
            if not is_beams:

                # tra_grid = [test_data[traId][i][0][-1] for i in range(num_steps)]
                # tra_time = test_time[traId]

                _state = session.run(mtest.initial_state)

                new_tra = []
                starttime = time.time()
                # print(len(tra))
                for i in range(num_steps):
                    if i<pre_position:
                        # print(i)
                        # print(np.reshape(x[:,i,:], [batch_size, 1, -1]).shape)
                        prob, _state = run_epoch(session, mtest, np.reshape(x[:,i,:], [batch_size, 1, -1]),
                                                 t, tf.no_op(), _state)
                        # new_tra.append(tra_grid[i])
                        # pre_point = tra_grid[i]
                    elif i<pre_position+predict+1:
                        if is_sample:
                            pre_point = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                            pre_point = pre_point[0]
                        else:
                            prob = prob.reshape([batch_size, -1])
                            pre_point = np.argmax(prob, axis=1)
                            print(pre_point.shape)
                        if types!=0:
                            prob, _state = run_epoch(session, mtest, embedding_all(pre_point), t, tf.no_op(), _state)
                        else:
                            prob, _state = run_epoch(session, mtest, only_embedding(pre_point), t, tf.no_op(), _state)
                        new_g = np.concatenate([new_g, pre_point.reshape([-1,1])], axis=1)
                        print(new_g.shape)
                    else:
                        break
                endtime = time.time()
                if timestamp<10:
                    timestamp += 1
                    total_time+=endtime-starttime
                else:
                    pass
                    # print(total_time/10)
                out_file(genfile, g, new_g, y)
                # out_file1(genfile, new_tra, traId)


if __name__ == "__main__":
    print(os._exists('./model/Model-512-289-chengdu-tgrid50-True-45.meta'))
    main(1)