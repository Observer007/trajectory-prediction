#coding:utf-8
import tensorflow as tf
import sys, time
import numpy as np
import pickle
import Config
import Model
from utils import *

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()

size = config.size
grids = '50'
hidden = str(config.hidden_size)
idim = str(config.input_dim)
types = config.add_dim
length = str(10)
train_file = "../useroadnet/generate_data/data/train/train-"+size+"-grid"+grids+"-"+length+".txt"
print(train_file)
test_file = "../useroadnet/generate_data/data/test/test-"+size+"-grid"+grids+"-"+length+".txt"
roadinfo_file = '../useroadnet/generate_data/data/roadnet/road-'+size+'-'+grids+'.txt'
embedding_file = '../useroadnet/generate_data/data/roadnet/road-'+ size + '-' + grids + '-' + idim + '-new.embedding'

# train_file = '../data/train/train-'+size+'-grid'+grids+'.txt'
# test_file = '../data/test/test-'+size+'-ngrid'+grids+'.txt'
# roadinfo_file = '../data/roadnet/road-'+size+'-'+grids+'.txt'
#trainfile = open(train_file, 'r')
#roadinfo = open(roadinfo_file, 'r')
train_data = []
test_data = []
train_data0, features, data_size, _vocab_size, neargrid = load_data_grid(train_file, roadinfo_file)
test_data0, __, data_size1, _vocab_size, _= load_data_grid(test_file, roadinfo_file)
[grid2cor, type, tunnel, bridge, oneway] = features
for traId in train_data0:
    train_data.append(train_data0[traId])
for traId in test_data0:
    test_data.append(test_data0[traId])

print('train data has %d characters, %d unique.' % (data_size , _vocab_size))
print(' test data has %d characters, %d unique.' % (data_size1, _vocab_size))
#char_to_idx = { ch:i for i, ch in enumerate(chars)}
#idx_to_char = { i:ch for i, ch in enumerate(chars)}


config.vocab_size = _vocab_size

#pickle.dump((char_to_idx, idx_to_char), open(config.model_path+'.voc', 'w'), protocol=pickle.HIGHEST_PROTOCOL)

#context_of_idx = [char_to_idx[ch] for ch in data]

def data_iterator(raw_data, batch_size, num_steps):
    #print(type(raw_data), type(raw_data[0]))
    if types != 0:
        features_data = [[type[raw_data[i][j]] + tunnel[raw_data[i][j]] + bridge[raw_data[i][j]] +
                      oneway[raw_data[i][j]] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    else:
        features_data = []
    raw_data = np.array(raw_data)

    features_data = np.array(features_data)
    #raw_data = np.array(raw_data)
#    raw_data = np.array(raw_data, dtype=np.int32)
    #print(raw_data)
    #print(type(raw_data), type(raw_data[0]))
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    # for i in range(batch_size):
    #     data[i] = raw_data[batch_len * i:batch_len * (i + 1)]#data的shape是(batch_size, batch_len)，每一行是连贯的一段，一次可输入多个段
    #
    # epoch_size = (batch_len - 1) // num_steps
    #
    # if epoch_size == 0:
    #     raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    #
    # for i in range(epoch_size):
    #     x = data[:, i*num_steps:(i+1)*num_steps]
    #     y = data[:, i*num_steps+1:(i+1)*num_steps+1]#y就是x的错一位，即下一个词
    #     yield (x, y)

    for i in range(int(batch_len*0.9)):
        tmp = raw_data[i*batch_size:(i+1)*batch_size]
        tmp0 = features_data[i * batch_size:(i + 1) * batch_size]
        #print(tmp[0, :1])
        x = tmp[:, :num_steps]
        y = tmp[:, 1:num_steps+1]              #y就是x的错一位，即下一个词
        if types!=0:
            f = tmp0[:, :num_steps, :]
        else:
            f = tmp0
        yield (x, y, f)


def run_epoch(session, m, data, test_data, eval_op):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size))
    print(epoch_size)
    start_time = time.time()
    costs = 0.0
    iters = 1

    #for i in range(len(state)):
        #print(len(state[i]))
    for step, (x, y, f) in enumerate(data_iterator(data, m.batch_size, m.num_steps)):
        #print(len(x[0]))
        state = m.initial_state.eval()
        if types:
            feed_dict = {m.input_data: x,
                         m.targets: y,
                         m.input_features: f,
                         m.initial_state: state}
        else:
            feed_dict = {m.input_data: x,
                         m.targets: y,
                         m.initial_state: state}
        # m.reset_state(m.batch_size)

        #print(state[0][0])
        cost, state, _ = session.run([m.cost, m.final_state, eval_op], #x和y的shape都是(batch_size, num_steps)
                                 feed_dict)
        costs += cost
        iters += m.num_steps
        if step and step % (epoch_size // 10) == 0:
            print("%.2f perplexity: %.3f cost-time: %.2f s" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 (time.time() - start_time)))
            start_time = time.time()


    costs1 = 0.0
    iters1 = 1
    for step, (x, y, f) in enumerate(data_iterator(test_data, m.batch_size, m.num_steps)):
        state = m.initial_state.eval()
        if types:
            feed_dict = {m.input_data: x,
                         m.targets: y,
                         m.input_features: f,
                         m.initial_state: state}
        else:
            feed_dict = {m.input_data: x,
                         m.targets: y,
                         m.initial_state: state}

        #print(state[0][0])
        cost, state, _ = session.run([m.cost, m.final_state, tf.no_op()], #x和y的shape都是(batch_size, num_steps)
                                 feed_dict)
        #print('gd is', type(_))
        #print(epoch_size)
        costs1 += cost
        iters1 += m.num_steps
        
    return np.exp(costs / iters), np.exp(costs1 / iters1)


def main(_):
    #train_data = context_of_idx
    
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=False, initializer=initializer):
            m = Model.Model(is_training=True, config=config)

        tf.global_variables_initializer().run()
        
        model_saver = tf.train.Saver(tf.global_variables())
        index = 1
        for i in range(config.iteration):
            # if index%5 == 0:
            #     config.learning_rate = config.learning_rate * 0.95
            #m.lr_decay(0.1)
            #print(session.run(m.lr))
            print("Training Epoch: %d ..." % (i+1))
            train_perplexity, test_perplexity = run_epoch(session, m, train_data, test_data, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f Test Perplexity: %.3f" % (i + 1, train_perplexity, test_perplexity))
            
            if (i+1) % config.save_freq == 0:
                print('model saving ...')
                model_saver.save(session, config.model_path+'-'+hidden+'-'+str(config.input_dim+types)+'-'+size+'-grid'+grids+'-%d'%(i+1))
                print('Done!')
            
if __name__ == "__main__":
    #tf.app.run()
    main(1)