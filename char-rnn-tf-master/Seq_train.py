# coding:utf-8
import tensorflow as tf
import sys, time
import numpy as np
import pickle
import Config
import seq2seq
from utils_t import *

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()
size = 'small'
grids = str(50)
hidden = str(config.hidden_size)
idim = str(config.input_dim)
length = '10'
train_file = "../useroadnet/generate_data/data/train/train-"+size+"-grid"+grids+"-"+length+".txt"
print(train_file)
test_file = "../useroadnet/generate_data/data/test/test-"+size+"-grid"+grids+"-"+length+".txt"
roadinfo_file = "../useroadnet/generate_data/data/roadnet/road-"+size+"-"+grids+".txt"
embedding_file = '../useroadnet/generate_data/data/roadnet/road-'+ size + '-' + grids + '-' + idim + '-new.embedding'

start_segment = 7
end_segment = 7
# train_file = "../data/train/train-big-grid50.txt"
# test_file = "../data/test/test-big-ngrid50.txt"
# roadinfo_file = "../data/roadnet/road-big-50.txt"
# embedding_file = '../data/roadnet/road-'+ size + '-' + grids + '-' + idim + '.embedding'

# trainfile = open(train_file, 'r')
# roadinfo = open(roadinfo_file, 'r')
train_data = []
test_data = []
train_mean = 0
train_var = 0
test_mean = 0
test_var = 0
train_data0, embedding, features, train_time0, data_size, _vocab_size, _ = load_data_grid(train_file, roadinfo_file, embedding_file, city=size)
test_data0, __, __1, test_time0, data_size1, _vocab_size, _ = load_data_grid(test_file, roadinfo_file, embedding_file, city=size)
[grid2cor, _type, tunnel, bridge, oneway] = features
# print(np.array(_type).shape)
for traId in train_data0:
    train_data.append(train_data0[traId])
# print(train_data[0][0][1])
for traId in test_data0:
    test_data.append(test_data0[traId])
if config.add_dim==0:
    embeddings = np.array([embedding[i][:int(idim)] for i in range(len(embedding))])
else:
    embeddings = np.concatenate((np.array([embedding[i][:int(idim)] for i in range(len(embedding))]), _type,
                             tunnel, bridge, oneway), axis=1)
print(embeddings.shape)
print('train data has %d characters, %d unique.' % (data_size, _vocab_size))
print(' test data has %d characters, %d unique.' % (data_size1, _vocab_size))

config.vocab_size = _vocab_size

types = config.add_dim

def data_iterator(raw_data, batch_size, num_steps):
    if types != 0:
        # print(raw_data[0][0][1])
        raw_data0 = [[raw_data[i][j][0][:int(idim)]+raw_data[i][j][1]+raw_data[i][j][2]+raw_data[i][j][3]+raw_data[i][j][4] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    else:
        raw_data0 = [[raw_data[i][j][0][:int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data1 = [[raw_data[i][j][0][int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data0 = np.array(raw_data0)
    raw_data1 = np.array(raw_data1)

    data_len = len(raw_data)
    batch_len = data_len // batch_size

    for i in range(batch_len):
        tmp = raw_data0[i * batch_size:(i + 1) * batch_size]
        tmp1 = raw_data1[i * batch_size:(i + 1) * batch_size]
        # print(type(tmp[:, :num_steps, 0][:int(idim)]), type(tmp[:, :num_steps, 1]))
        index = int(num_steps/2)
        x_e = tmp[:, :index, :]
        x_d = np.array([[embeddings[start_segment]] for _ in range(len(tmp))])
        x_d = np.concatenate((x_d, tmp[:, index:num_steps, :]), axis=1)
        y = np.array([[end_segment] for _ in range(len(tmp))])
        y = np.concatenate((tmp1[:, index: num_steps], y), axis=1)  # y就是x的错一位，即下一个词
        #y = np.reshape(y, (y.shape[0], -1))
        # print(x_d.shape, y.shape)
        # print(y[0])
        yield (x_e, x_d, y)


def run_epoch(session, m, data, test_data, eval_op):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size))
    print(epoch_size)
    start_time = time.time()
    costs = 0.0
    iters = 1

    for step, (x_e, x_d, y) in enumerate(data_iterator(data, m.batch_size, m.num_steps)):
        #state = session.run(m.initial_state)
        # cost, _, l2, weight = session.run([m.cost, eval_op, m.l2, m.weights],  # x和y的shape都是(batch_size, num_steps)
        cost, _ = session.run([m.cost, eval_op],
                                     {m.encoder_input: x_e,
                                      m.decoder_input: x_d,
                                      m.targets: y,
                                      })
        # print('costs is:', cost, l2)
        costs += cost
        iters += m.num_steps

        if step and step % (epoch_size // 10) == 0:
            print("%.2f perplexity: %.3f cost-time: %.2f s" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   (time.time() - start_time)))
            # print('weight is:', weight)
            start_time = time.time()

    costs1 = 0.0
    iters1 = 1
    for step, (x_e, x_d, y) in enumerate(data_iterator(test_data, m.batch_size, m.num_steps)):

        cost, _ = session.run([m.cost, tf.no_op()],  # x和y的shape都是(batch_size, num_steps)
                                     {m.encoder_input: x_e,
                                      m.decoder_input: x_d,
                                      m.targets: y,
                                      })
        costs1 += cost
        iters1 += m.num_steps

    return np.exp(costs / iters), np.exp(costs1 / iters1)


def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        print('1')
        model = seq2seq.Model(is_training=True, network_embeddings=embeddings, config=config)
        print('2')
        session.run(tf.global_variables_initializer())

        model_saver = tf.train.Saver(tf.global_variables())
        index = 1
        for i in range(config.iteration):
            print("Training Epoch: %d ..." % (i + 1))
            train_perplexity, test_perplexity = run_epoch(session, model, train_data,
                                                          test_data, model.train_op)
            print("Epoch: %d Train Perplexity: %.3f Test Perplexity: %.3f" % (i + 1, train_perplexity, test_perplexity))

            if (i + 1) % config.save_freq == 0:
                print('model saving ...')
                model_saver.save(session, config.seq2seq_path+'-'+str(config.num_layers)+'-'+str(hidden)+'-'+str(config.input_dim+types)+'-'+size + '-%d' % (i + 1))
                print('Done!')


if __name__ == "__main__":
    # tf.app.run()
    main(1)