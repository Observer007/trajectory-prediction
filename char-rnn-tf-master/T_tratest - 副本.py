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

test_data, embedding, features, test_time, data_size, vocab_size, neargrid = load_data_grid(testfile, roadinfo, embedding_file, size)
[grid2cor, type, tunnel, bridge, oneway] = features

mean, var = 6.36721020342, 30.8796707124

print(neargrid)
pre_position = 5
config.vocab_size = vocab_size
is_sample = config.is_sample
is_beams = config.is_beams
beam_size = config.beam_size
len_of_generation = config.len_of_generation
predict = config.predict


def data_iterator(raw_data, time_data, batch_size, num_steps):
    if types!=0:
        raw_data0 = [[raw_data[i][j][0][:int(idim)]+raw_data[i][j][1]+raw_data[i][j][2]+raw_data[i][j][3]+raw_data[i][j][4] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    else:
        raw_data0 = [[raw_data[i][j][0][:int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data1 = [[raw_data[i][j][0][int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data0 = np.array(raw_data0)
    raw_data1 = np.array(raw_data1)
    time_data = np.array(time_data)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    union_data = []
    for i in range(int(batch_len)):
        tmp = raw_data0[i * batch_size:(i + 1) * batch_size]
        tmp1 = raw_data1[i * batch_size:(i + 1) * batch_size]
        tmp_time = time_data[i * batch_size:(i + 1) * batch_size]
        x = tmp[:, :num_steps, :]
        y = tmp1[:, 1: num_steps+1]  # y就是x的错一位，即下一个词
        t = tmp_time
        y = np.reshape(y, (y.shape[0], -1))
        union_data.append([x, y, t])
    return union_data



def run_epoch(session, m, data, time, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1, input_dim+types))
    # t = np.array(time).reshape((1, 9))
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                         {m.input_data: x,
                          m.extra_data: time,
                          m.initial_state: state})
    return prob, _state


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
        config.batch_size = 1
        config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = T_Model.T_Model(is_training=False, holiday=holiday, config=config)

        #tf.global_variables_initializer().run()

        model_saver = tf.train.Saver()
        print('model loading ...')
        # model_saver.restore(session, config.model_path+'-'+hidden+'-'+str(input_dim)+'-'+size+'-tgrid'+grids+'-%d' % config.save_time)
        model_saver.restore(session, config.model_path+'-'+hidden+'-'+str(input_dim+types)+'-'+size+'-tgrid'+grids+'-'+holiday+'-%d' % config.save_time)

        print('Done!')
        timestamp = 0
        total_time = 0
        index = 0
        for traId in test_data:
            index += 1
            if not index%50:
                print(index)
            if index==700:
                break
            if not is_beams:
                # sentence state
                if types!=0:
                    tra = [test_data[traId][i][0][:input_dim]+test_data[traId][i][1]+test_data[traId][i][2]+ \
                      test_data[traId][i][3]+test_data[traId][i][4] for i in range(num_steps)]
                else:
                    tra = [test_data[traId][i][0][:input_dim] for i in range(num_steps)]
                tra_grid = [test_data[traId][i][0][-1] for i in range(num_steps)]
                tra_time = test_time[traId]

                _state = session.run(mtest.initial_state)

                new_tra = []
                starttime = time.time()
                # print(len(tra))
                for i in range(len(tra)):
                    if i<pre_position:
                        # print(tra_time)
                        prob, _state = run_epoch(session, mtest, np.float32([tra[i]]),
                                                 [tra_time], tf.no_op(), _state)
                        new_tra.append(tra_grid[i])
                        pre_point = tra_grid[i]
                    elif i<pre_position+predict+1:
                        if is_sample:
                            pre_point = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                            pre_point = pre_point[0]
                        else:
                            # f = 3
                            # fold_states = np.array(
                            #     [np.array([grid2cor[new_tra[l - f - 1 + t]] for t in range(f)]).flatten().tolist()
                            #      for l in range(f - 1 + i - pre_position, i)])
                            # pre_tra = np.array([grid2cor[new_tra[t]] for t in range(i - pre_position, i)])
                            # k = RMF_calculate(fold_states, pre_tra, f, pre_position, 2)
                            # next_loc = RMF_prediction(fold_states[-1], k).tolist()
                            # maxgrid = []
                            # dis = []
                            # dis1 = 100
                            # for p in range(nearest):
                            #     tmp = caldistance(next_loc, grid2cor[neargrid[pre_point][p]])
                            #     dis.append(tmp)
                            #     if tmp < dis1:
                            #         dis1 = tmp
                            #         maxgrid = neargrid[pre_point][p]

                            # dis = pd.Series(dis, index=neargrid[pre_point])
                            # dis = dis.sort_values().index[:2]
                            # print(maxgrid, dis[0])
                            prob = prob.reshape(-1)
                            # prob[maxgrid] += 0.8
                            # for i in range(len(dis)):
                            #     prob[dis[i]] += 1e-4
                            # prob = [prob[i] for i in neargrid[pre_point]]
                            # prob = pd.Series(prob, index=neargrid[pre_point])
                            # pre_point = prob.sort_values().index[-1]
                            pre_point = np.argmax(prob)
                        if types!=0:
                            prob, _state = run_epoch(session, mtest, np.float32(embedding[pre_point][:input_dim]+type[pre_point]+
                                                 tunnel[pre_point]+bridge[pre_point]+oneway[pre_point]),
                                                 [tra_time], tf.no_op(), _state)
                        else:
                            prob, _state = run_epoch(session, mtest,
                                                 np.float32(embedding[pre_point][:input_dim]),
                                                 [tra_time], tf.no_op(), _state)
                        new_tra.append(pre_point)
                    else:
                        break
                endtime = time.time()
                if timestamp<10:
                    timestamp += 1
                    total_time+=endtime-starttime
                else:
                    pass
                    # print(total_time/10)
                out_file2(genfile, tra_grid, traId)
                out_file1(genfile, new_tra, traId)


if __name__ == "__main__":
    main(1)