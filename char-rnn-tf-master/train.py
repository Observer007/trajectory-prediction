#coding:utf-8
import tensorflow as tf
import sys, time
import numpy as np
import pickle
import Config
from zx_model import Model
from utils_t import *
from tqdm import tqdm
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()
tf.set_random_seed(0)

def get_train_val_test(config):
    try:
        train_file = "../data/train/train-" + config.city + "-grid" + str(config.grids) + "-" + str(
            config.length) + ".txt"
        test_file = "../data/test/test-" + config.city + "-grid" + str(config.grids) + "-" + str(config.length) + ".txt"
        roadinfo_file = '../data/roadnet/road-' + config.city + '-' + str(config.grids) + '.txt'
        embedding_file = '../data/roadnet/road-' + str(config.city) + '-' + str(config.grids) + '-' + str(
            config.inter_units) + '.embedding'
        train_data, inter_embeddings, intra_embeddings, grid2cor = \
            load_data_grid(train_file, roadinfo_file, embedding_file, config.city)
        test_data, _, _, _ = load_data_grid(test_file, roadinfo_file, embedding_file, config.city)
        train_data, val_data = train_val_split(train_data)
        print('train data size: %d, val data size: %d, test data size: %d' %
              (train_data.shape[0], val_data.shape[0], test_data.shape[0]))
        print('grid number is: %d' % len(grid2cor))
        config.grid_size = len(grid2cor)
    except:
        train_file = "../useroadnet/generate_data/data/train/train-" + config.city + "-grid" + str(
            config.grids) + "-" + str(config.length) + ".txt"
        test_file = "../useroadnet/generate_data/data/test/test-" + config.city + "-grid" + str(
            config.grids) + "-" + str(config.length) + ".txt"
        roadinfo_file = '../useroadnet/generate_data/data/roadnet/road-' + config.city + '-' + str(
            config.grids) + '.txt'
        embedding_file = '../useroadnet/generate_data/data/roadnet/road-' + str(config.city) + '-' + str(
            config.grids) + '-' + str(config.inter_units) + '.embedding'
        train_data, inter_embeddings, intra_embeddings, grid2cor = \
            load_data_grid(train_file, roadinfo_file, embedding_file, config.city)
        test_data, _, _, _ = load_data_grid(test_file, roadinfo_file, embedding_file, config.city)
        train_data, val_data = train_val_split(train_data)
        print('train data size: %d, val data size: %d, test data size: %d' %
              (train_data.shape[0], val_data.shape[0], test_data.shape[0]))
        print('grid number is: %d' % len(grid2cor))
        config.grid_size = len(grid2cor)

    return train_data, val_data, test_data, inter_embeddings, intra_embeddings, grid2cor


def data_iterator(raw_data, batch_size, num_steps, is_training=True):
    id_data = np.array([i.id for i in raw_data])
    grid_data = np.array([i.grids for i in raw_data])
    week_day_data = np.array([i.weekday for i in raw_data])
    assert grid_data.shape.__len__()==2 and grid_data.shape[1]>num_steps
    total_size = grid_data.shape[0]
    batch_num = int(np.ceil(total_size/batch_size))
    for i in range(batch_num):
        if is_training:
            yield (id_data[i*batch_size:(i+1)*batch_size],
            grid_data[i*batch_size:(i+1)*batch_size, :num_steps],
            grid_data[i*batch_size:(i+1)*batch_size, 1:num_steps+1],
            week_day_data[i*batch_size:(i+1)*batch_size, :])
        else:
            yield (id_data[i * batch_size:(i + 1) * batch_size],
            grid_data[i * batch_size:(i + 1) * batch_size, :config.given_step+config.gen_step],
            grid_data[i * batch_size:(i + 1) * batch_size, 1:config.given_step+config.gen_step + 1],
            week_day_data[i * batch_size:(i + 1) * batch_size, :])

def run_one_epoch(config, session, model, train_data, val_data, epoch=0):
    """Runs the model on the given data."""
    # ============================ train =====================================
    start_time = time.time()
    last_time = start_time
    costs = 0.0
    iters = 1
    total = len(train_data)
    print_step = (total//config.batch_size+1)//10
    total_loss = 0
    total_acc = 0
    print(print_step, total)
    # ============================ train ====================================
    for step, (_, x, y, ext) in tqdm(enumerate(data_iterator(train_data, config.batch_size
            , config.num_steps))):
        if step == 10:
            p = 1
        tf.reset_default_graph()
        loss, acc = model.train_one_batch(session, x, y, ext)
        tf.get_default_graph().finalize()
        total_loss += loss
        total_acc += acc
        if step % print_step == 0:
            print("%.2f, loss: %.3f, acc: %.4f, cost-time: %.2f s" %
                (step * 1.0 / total, total_loss / iters,
                 total_acc/iters, float(time.time() - last_time)))
            last_time = time.time()
        iters += 1
    train_loss = total_loss/iters
    print('train epoch: %d, loss: %.3f, acc: %.4f, cost-time: %.2f'%
          (epoch, total_loss/iters, total_acc/iters, time.time()-start_time))
    # ============================== val =====================================
    total_loss, total_acc = 0, 0
    iters = 1
    start_time = time.time()
    for step, (_, x, y, ext) in enumerate(data_iterator(val_data, config.batch_size,
                config.num_steps)):
        tf.reset_default_graph()
        loss, acc = model.val_one_batch(session, x, y, ext)
        tf.get_default_graph().finalize()
        total_loss += loss
        total_acc += acc
        iters += 1
    val_loss = total_loss/iters
    print('val epoch: %d, loss: %.3f, acc: %.4f, cost-time: %.2f'%
          (epoch, total_loss/iters, total_acc/iters, time.time()-start_time))


    return train_loss, val_loss



    

if __name__ == "__main__":
    print(config.inter_fea, config.intra_fea, config.ext_dim)
    train_data, val_data, test_data, inter_embeddings, intra_embeddings, _ = \
        get_train_val_test(config)
    # with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
    session = tf.Session(config=config_tf)
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=False, initializer=initializer):
        m = Model(config, inter_embeddings=inter_embeddings,
                  intra_embeddings=intra_embeddings, is_training=True)

    session.run(tf.global_variables_initializer())

    model_saver = tf.train.Saver(tf.global_variables())
    if config.train_from > 0:
        init_epoch = config.train_from + 1
        model_saver.restore(session, config.model_path + '_hidden_' + str(config.hidden_size) + '_inter_' + str(
            config.inter_fea) + '_intra_' + str(config.intra_fea) +
          '_ext_'+str(config.ext_dim)+'_grid_' + str(config.grids) + '_%d' % (config.train_from))
    else:
        init_epoch = 1
    for epoch in range(init_epoch, config.max_epoch+1):
        print("Training Epoch: %d ..." % (epoch))
        train_perplexity, test_perplexity = run_one_epoch(config, session, m, train_data, val_data, epoch)
        # print("Epoch: %d Train Perplexity: %.3f Test Perplexity: %.3f" % (i + 1, train_perplexity, test_perplexity))

        if epoch>config.save_after:
            print('model saving ...')
            model_saver.save(session, config.model_path + '_hidden_' + str(config.hidden_size) + '_inter_' + str(
                config.inter_fea) + '_intra_' + str(config.intra_fea) +
                '_ext_'+str(config.ext_dim)+'_grid_' + str(config.grids) +
                '_numstep_'+str(config.num_steps)+'_%d' % (epoch))
            print('Done!')
