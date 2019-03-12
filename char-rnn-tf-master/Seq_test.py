#coding:utf-8
import tensorflow as tf
import sys, time
import numpy as np
import pickle, os
import random
import pandas as pd
import Config
import seq2seq
from utils_t import *
from RMF_calculate import *

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()

size = 'small'
grids = '50'
hidden = str(config.hidden_size)
input_dim = config.input_dim
nearest = config.nearest
num_steps = config.num_steps
idim = str(config.input_dim)
length = str(10)
roadinfo = '../useroadnet/generate_data/data/roadnet/road-'+size+'-'+grids+'.txt'
testfile = '../useroadnet/generate_data/data/test/test-'+size+'-grid'+grids+'-'+length+'.txt'
genfile = '../useroadnet/generate_data/data/seq_generate/result-'+size+'-tgrid'+grids+'-'+str(nearest)+'-'+length+'.txt'
embedding_file = '../useroadnet/generate_data/data/roadnet/road-'+ size + '-' + grids + '-' + str(input_dim) + '-new.embedding'

with open(genfile, 'w') as f:
    print('build!')
# roadinfo = '../data/roadnet/road-'+size+'-'+grids+'.txt'
# testfile = '../data/test/test-'+size+'-grid'+grids+'.txt'
# genfile = '../data/generate/result-'+size+'-grid'+grids+'.txt'

test_data, embedding, features, test_time, data_size, vocab_size, neargrid = load_data_grid(testfile, roadinfo, embedding_file, size)
[grid2cor, _type, tunnel, bridge, oneway] = features

mean, var = 6.36721020342, 30.8796707124

print(neargrid)
pre_position = 5
config.vocab_size = vocab_size
is_sample = config.is_sample
is_beams = config.is_beams
beam_size = config.beam_size
len_of_generation = config.len_of_generation
predict = config.predict
batch_size = config.batch_size
print(batch_size, num_steps)
if config.add_dim==0:
    embeddings = np.array([embedding[i][:int(idim)] for i in range(len(embedding))])
else:
    embeddings = np.concatenate((np.array([embedding[i][:int(idim)] for i in range(len(embedding))]), _type,
                             tunnel, bridge, oneway), axis=1)
print(embeddings.shape)

def run_epoch(session, m, data, time, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1, input_dim+types))
    t = time.reshape((1, 1, 1))
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                         {m.input_data: x,
                          # m.input_time: t,
                          m.initial_state: state})
    return prob, _state


def out_file1(file, tra, tra1, traId):
    with open(file, 'a') as output:
        output.write(str(traId)+',')
        for point in tra:
            output.write(str(point)+',')
        for point in tra1[:int(num_steps/2)]:
            output.write(str(point) + ',')
        output.write('\n')

def out_file2(file, tra, traId):
    with open(file, 'a') as output:
        output.write(str(traId)+',')
        for point in tra:
            output.write(str(point)+',')
        output.write('\n')

def caldistance(point1, point2):
    return pow(pow(point2[1]-point1[1], 2)+pow(point2[0]-point1[0], 2), 0.5)

types = config.add_dim
def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtest = seq2seq.Model(is_training=False, network_embeddings=embeddings, config=config)

        #tf.global_variables_initializer().run()

        model_saver = tf.train.Saver()
        print('model loading ...')
        # model_saver.restore(session, config.model_path+'-'+hidden+'-'+str(input_dim)+'-'+size+'-tgrid'+grids+'-%d' % config.save_time)
        model_saver.restore(session, config.seq2seq_path+'-'+str(config.num_layers)+'-'+str(hidden)+'-'+str(input_dim+types)+'-'+size+'-%d' % config.save_time)

        print('Done!')
        timestamp = 0
        total_time = 0
        index = 0
        for traId in test_data:
            index += 1
            if not index%50:
                print(index)
            if not is_beams:
                # sentence state
                if types!=0:
                    tra = [test_data[traId][i][0][:input_dim]+test_data[traId][i][1]+test_data[traId][i][2]+ \
                      test_data[traId][i][3]+test_data[traId][i][4] for i in range(int(num_steps/2))]
                else:
                    tra = [test_data[traId][i][0][:input_dim] for i in range(int(num_steps/2))]
                tra_grid = [test_data[traId][i][0][-1] for i in range(num_steps)]
                mid_grid = [test_data[traId][i][0][-1] for i in range(int(num_steps/2))]
                # _state = session.run(mtest.initial_state)

                new_tra = []
                starttime = time.time()
                answer_logits = session.run(mtest.logits, {mtest.encoder_input: [tra] * batch_size})[0]
                print(answer_logits)
                #new_tra.append(pre_point)
                endtime = time.time()
                if timestamp<1:
                    timestamp += 1
                    total_time+=endtime-starttime
                else:
                    print(total_time/10)
                out_file2(genfile, tra_grid, traId)
                out_file1(genfile, mid_grid, answer_logits, traId)


if __name__ == "__main__":
    main(1)