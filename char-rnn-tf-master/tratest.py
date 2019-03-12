#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import pickle, os
import random
import pandas as pd
import Config
import zx_model
from utils_t import *
from train import get_train_val_test, data_iterator
from tqdm import tqdm
# from RMF_calculate import *
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()






def predict(session, model, test_data, epoch, file, grid_near=None, grid2cor=None):
    """Runs the model on the given data."""
    start_time = time.time()
    iters = 0
    total_acc = 0
    pre_result_list = []
    total_val_acc = 0
    # ============================ train ====================================
    for step, (id, x, y, ext) in tqdm(enumerate(data_iterator(test_data, config.batch_size
            , config.num_steps, is_training=False))):

        tf.reset_default_graph()
        _, val_acc = model.val_one_batch(session, x, y, ext)
        result = model.test_one_batch(session, x[:, :config.given_step+config.gen_step],
                                    ext, config.given_step, config.gen_step, grid_near, grid2cor)
        tf.get_default_graph().finalize()
        iters += 1
        ground_truth = x[:, config.given_step:config.given_step+config.gen_step]
        result_pre = result[:, config.given_step:config.given_step+config.gen_step]

        acc = cal_acc(result_pre, ground_truth)
        print(acc, val_acc)
        total_acc += acc
        total_val_acc += val_acc
        for i in range(len(id)):
            tra = Trajector_Grid(id[i], result[i], None, x[i, : config.given_step+config.gen_step])
            pre_result_list.append(tra)
    print('val acc: %.4f' %(total_val_acc/iters))
    print('test epoch: %d, acc: %.4f, cost-time: %.2f' %
          (epoch, total_acc / iters, time.time() - start_time))
    out_file(file, pre_result_list)

def out_file(file, tras):
    with open(file, 'w') as output:
        for tra in tras:
            output.write(str(tra.id) + ',')
            for grid in tra.ground_truth:
                output.write(str(int(grid)) + ',')
            output.write('\n')
            output.write(str(tra.id)+',')
            for grid in tra.grids:
                output.write(str(int(grid))+',')
            output.write('\n')





if __name__ == "__main__":
    session = tf.Session(config=config_tf)
    _, val_data, test_data, inter_embeddings, intra_embeddings, grid2cor = \
        get_train_val_test(config)
    grid_near_file = '../data/roadnet/road-'+config.city+'-'+str(config.grids)+'-near.txt'
    grid_near = read_road_near(grid_near_file)
    # grid_near = None
    with tf.variable_scope("model", reuse=False):
        mtest = zx_model.Model(config, inter_embeddings=inter_embeddings,
            intra_embeddings=intra_embeddings, is_training=False)
    result_file = '../data/result/'+config.city+'/result_hidden_'+\
    str(config.hidden_size) + '_inter_' + str(config.inter_fea) + \
    '_intra_' + str(config.intra_fea) +'_ext_'+\
    str(config.ext_dim)+'_grid_' + str(config.grids)+\
    '_numstep_'+str(config.num_steps)+'.txt'
    model_saver = tf.train.Saver()
    print(config.inter_fea, config.intra_fea, config.ext_dim)
    print('model loading ...')
    print(config.model_path + '_hidden_' + str(config.hidden_size) + '_inter_' + str(
            config.inter_fea) + '_intra_' + str(config.intra_fea) +
          '_ext_'+str(config.ext_dim)+'_grid_' + str(config.grids) +
          '_numstep_'+str(config.num_steps)+'_%d' % (config.test_epoch))
    try:
        model_saver.restore(session, config.model_path + '_hidden_' + str(config.hidden_size) + '_inter_' + str(
            config.inter_fea) + '_intra_' + str(config.intra_fea) +
          '_ext_'+str(config.ext_dim)+'_grid_' + str(config.grids) +
          '_numstep_'+str(config.num_steps)+'_%d' % (config.test_epoch))
    except:
        model_saver.restore(session, config.model_path + '_hidden_' + str(config.hidden_size) + '_inter_' + str(
            config.inter_fea) + '_intra_' + str(config.intra_fea) +
                            '_ext_' + str(config.ext_dim) + '_grid_' + str(config.grids) +
                            '_%d' % (config.test_epoch))
    predict(session, mtest, test_data, config.test_epoch, result_file, grid_near, grid2cor)
    print('model loading Done!')

