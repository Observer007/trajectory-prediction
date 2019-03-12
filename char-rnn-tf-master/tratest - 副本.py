#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import pickle, os
import random
import pandas as pd
import Config
import Model
from utils import *
# from RMF_calculate import *
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()
types = config.add_dim
holiday = config.holiday
size = config.size
# size = 'small'
grids = '50'
hidden = str(config.hidden_size)
nearest = str(config.nearest)
length = str(10)
roadinfo = '../useroadnet/generate_data/data/roadnet/road-'+size+'-'+grids+'.txt'
# testfile = '../useroadnet/generate_data/data/test/test-'+size+'-grid'+grids+'-'+length+'.txt'
# genfile = '../useroadnet/generate_data/data/generate/result-'+size+'-noembedding'+str(types)+'.txt'

testfile = '../useroadnet/generate_data/data/test/test-small.txt'
genfile = '../useroadnet/generate_data/data/generate/result-small-none.txt'

# roadinfo = '../data/roadnet/road-'+size+'-'+grids+'.txt'
# testfile = '../data/test/test-'+size+'-grid'+grids+'.txt'
# genfile = '../data/generate/result-'+size+'-grid'+grids+'.txt'

# roadinfo = "../useroadnet/generate_data/data/roadnet/road-small-30.txt"
# testfile = "../useroadnet/generate_data/data/test/test-small-grid30.txt"
# genfile = "../useroadnet/generate_data/data/generate/result-small-grid30.txt"
# roadinfo = "../data/roadnet/road-small-30.txt"
# testfile = "../data/test/test-small-grid30.txt"
# genfile = "../data/generate/result-small-grid30.txt"

# test_data, data_size, grid2cor, vocab_size, neargrid = load_data_grid(testfile, roadinfo)
test_data, features, data_size, vocab_size, neargrid = load_data_grid(testfile, roadinfo)
[grid2cor, type, tunnel, bridge, oneway] = features
print(len(test_data), vocab_size)
# test_data = []
pre_position = 5
types = config.add_dim
config.vocab_size = vocab_size
is_sample = config.is_sample
is_beams = config.is_beams
beam_size = config.beam_size
len_of_generation = config.len_of_generation
# for traId in test_data0:
#     test_data.append(test_data0[traId])
with open(genfile, 'w') as output1:
    print("build")
def run_epoch(session, m, data, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1))
    # print(x)
    if types!=0:
        f = np.array([[type[x[0][0]] + tunnel[x[0][0]] + bridge[x[0][0]] + oneway[x[0][0]]]])
        feed_dict = {m.input_data: x,
                          m.input_features: f,
                          m.initial_state: state}
    else:
        f = []
        feed_dict = {m.input_data: x,
                          # m.input_features: f,
                          m.initial_state: state}
    # print('f is:', f)
    start_time = time.time()
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                         feed_dict)
    end_time = time.time()
    # print("time is: ", end_time - start_time)
    return prob, _state

def out_file(file, tra, traId):
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
            mtest = Model.Model(is_training=False, config=config)


        model_saver = tf.train.Saver()
        print('model loading ...')
        print(config.model_path+'-'+hidden+'-'+size+'-grid'+grids+'-%d' % config.save_time)
        model_saver.restore(session, config.model_path+'-'+hidden+'-'+str(config.input_dim+types)+'-'+size+'-grid'+grids+'-%d' % config.save_time)
        print('Done!')
        print(test_data)
        index = 0
        for traId in test_data:
            index += 1
            if not index % 50:
                print(index)
            if index == 700:
                break
            if not is_beams:
                tra = test_data[traId]
                # print(tra)
                # mtest.reset_state(mtest.batch_size)
                _state = mtest.initial_state.eval()
                new_tra = []
                starttime = time.time()
                for i in range(len(tra)):
                    if i<pre_position:
                        prob, _state = run_epoch(session, mtest, np.int32([tra[i]]), tf.no_op(), _state)
                        new_tra.append(tra[i])
                        pre_point = tra[i]
                    else:
                        if is_sample:
                            pre_point = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                            pre_point = pre_point[0]
                        else:
                            # f = 5
                            # fold_states = np.array(
                            #     [np.array([grid2cor[new_tra[l - f - 1 + t]] for t in range(f)]).flatten().tolist()
                            #      for l in range(f - 1 + i - pre_position, i)])
                            # # print(fold_states.shape)
                            # # print(fold_states)
                            # pre_tra = np.array([grid2cor[new_tra[t]] for t in range(i - pre_position, i)])
                            # k = RMF_calculate(fold_states, pre_tra, f, pre_position, 2)
                            # next_loc = RMF_prediction(fold_states[-1], k).tolist()
                            # maxgrid = -1
                            # dis = 100
                            # for p in range(int(nearest)):
                            #     tmp = caldistance(next_loc, grid2cor[neargrid[pre_point][p]])
                            #     # dis.append(tmp)
                            #     if tmp < dis:
                            #         dis = tmp
                            #         maxgrid = neargrid[pre_point][p]
                            prob = prob.reshape(-1)
                            # prob[maxgrid] += 0.0
                            # prob = [prob[i] for i in neargrid[pre_point]]
                            # prob = pd.Series(prob, index=neargrid[pre_point])
                            # pre_point = prob.sort_values().index[-1]
                            pre_point = np.argmax(prob)
                        prob, _state = run_epoch(session, mtest, np.int32([pre_point]), tf.no_op(), _state)
                        new_tra.append(pre_point)
                endtime = time.time()
                # print(endtime-starttime)
                out_file(genfile, tra, traId)
                out_file(genfile, new_tra, traId)


if __name__ == "__main__":
    main(1)