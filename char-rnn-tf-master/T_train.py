# coding:utf-8
import tensorflow as tf
import sys, time
import numpy as np
import pickle
import Config
import T_Model
from utils_t import *

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1



config = Config.Config()
holiday = config.holiday
size = config.size
city = size
grids = str(50)
hidden = str(config.hidden_size)
idim = str(config.input_dim)
length = '10'
train_file = "../useroadnet/generate_data/data/train/train-"+size+"-grid"+grids+"-"+length+".txt"
print(train_file)
test_file = "../useroadnet/generate_data/data/test/test-"+size+"-grid"+grids+"-"+length+".txt"
roadinfo_file = "../useroadnet/generate_data/data/roadnet/road-"+size+"-"+grids+".txt"
embedding_file = '../useroadnet/generate_data/data/roadnet/road-'+ size + '-' + grids + '-' + idim + '-new.embedding'

# train_file = "../data/train/train-big-grid50.txt"
# test_file = "../data/test/test-big-ngrid50.txt"
# roadinfo_file = "../data/roadnet/road-big-50.txt"
# embedding_file = '../data/roadnet/road-'+ size + '-' + grids + '-' + idim + '.embedding'

# trainfile = open(train_file, 'r')
# roadinfo = open(roadinfo_file, 'r')
train_data = []
test_data = []
train_time = []
test_time = []
train_mean = 0
train_var = 0
test_mean = 0
test_var = 0
train_data0, embedding, features, weekday, data_size, _vocab_size, _ = load_data_grid(train_file, roadinfo_file, embedding_file, city)
print('tetjoiewjroiejrioewjri')
test_data0, __, __1, weekday1, data_size1, _vocab_size, _ = load_data_grid(test_file, roadinfo_file, embedding_file, city)
[grid2cor, type, tunnel, bridge, oneway] = features
for traId in train_data0:
    train_data.append(train_data0[traId])
    train_time.append(weekday[traId])
for traId in test_data0:
    test_data.append(test_data0[traId])
    test_time.append(weekday1[traId])

tmp_train = np.array(train_time)
tmp_train.reshape([-1])
tmp_test = np.array(test_time)
tmp_test.reshape([-1])
train_mean = np.mean(tmp_train)
train_var = np.var(tmp_train)
test_mean = np.mean(tmp_test)
test_var = np.var(tmp_test)
train_time = ((np.array(train_time)-train_mean)/train_var).tolist()
test_time = ((np.array(test_time)-train_mean)/train_var).tolist()
print(train_mean, train_var)

print('train data has %d characters, %d unique.' % (data_size, _vocab_size))
print(' test data has %d characters, %d unique.' % (data_size1, _vocab_size))
# char_to_idx = { ch:i for i, ch in enumerate(chars)}
# idx_to_char = { i:ch for i, ch in enumerate(chars)}



config.vocab_size = _vocab_size

types = config.add_dim
def data_iterator(raw_data, time_data, batch_size, num_steps):
    # print(raw_data[0][0][0][:int(idim)])
    if types!=0:
        raw_data0 = [[raw_data[i][j][0][:int(idim)]+raw_data[i][j][1]+raw_data[i][j][2]+raw_data[i][j][3]+raw_data[i][j][4] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    else:
        raw_data0 = [[raw_data[i][j][0][:int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data1 = [[raw_data[i][j][0][int(idim)] for j in range(len(raw_data[0]))] for i in range(len(raw_data))]
    raw_data0 = np.array(raw_data0)
    raw_data1 = np.array(raw_data1)
    time_data = np.array(time_data)
    #    raw_data = np.array(raw_data, dtype=np.int32)
    # print(raw_data0.shape, raw_data1.shape)
    # print(type(raw_data), type(raw_data[0]))
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    # data = np.zeros([batch_size, batch_len], dtype=np.int32)
    union_data = []
    valid_data = []
    for i in range(int(batch_len*0.9)):
        tmp = raw_data0[i * batch_size:(i + 1) * batch_size]
        tmp1 = raw_data1[i * batch_size:(i + 1) * batch_size]
        tmp_time = time_data[i * batch_size:(i + 1) * batch_size]
        # print(type(tmp[:, :num_steps, 0][:int(idim)]), type(tmp[:, :num_steps, 1]))
        x = tmp[:, :num_steps, :]
        y = tmp1[:, 1: num_steps+1]  # y就是x的错一位，即下一个词
        t = tmp_time
        y = np.reshape(y, (y.shape[0], -1))
        # print(x.shape, y.shape)
        union_data.append([x, y, t])
    for i in range(int(batch_len*0.9), batch_len):
        tmp = raw_data0[i * batch_size:(i + 1) * batch_size]
        tmp1 = raw_data1[i * batch_size:(i + 1) * batch_size]
        tmp_time = time_data[i * batch_size:(i + 1) * batch_size]
        # print(type(tmp[:, :num_steps, 0][:int(idim)]), type(tmp[:, :num_steps, 1]))
        x = tmp[:, :num_steps, :]
        y = tmp1[:, 1: num_steps + 1]  # y就是x的错一位，即下一个词
        t = tmp_time
        y = np.reshape(y, (y.shape[0], -1))
        # print(x.shape, y.shape)
        valid_data.append([x, y, t])
    return union_data, valid_data


def run_epoch(session, m, f_data, v_data, t_data, eval_op):
    """Runs the model on the given data."""
    epoch_size = len(f_data)
    print(epoch_size)
    start_time = time.time()
    costs = 0.0
    iters = 1

    # for i in range(len(state)):
    # print(len(state[i]))
    for step, temp in enumerate(f_data):
        x = temp[0]
        y = temp[1]
        t = temp[2]

        state = session.run(m.initial_state)

        cost, state, _ = session.run([m.cost, m.final_state, eval_op],  # x和y的shape都是(batch_size, num_steps)
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.extra_data: t,
                                      m.initial_state: state})

        costs += cost
        iters += m.num_steps

        if step and step % (epoch_size // 10) == 0:
            print("%.2f perplexity: %.3f cost-time: %.2f s" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   (time.time() - start_time)))
            start_time = time.time()

    costs1 = 0.0
    iters1 = 1
    for step, (x, y, t) in enumerate(v_data):

        state = session.run(m.initial_state)

        cost, state, _ = session.run([m.cost, m.final_state, tf.no_op()],  # x和y的shape都是(batch_size, num_steps)
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.extra_data: t,
                                      m.initial_state: state})
        # cost, state, _ = session.run([m.cost, m.final_state, eval_op],  # x和y的shape都是(batch_size, num_steps)
        #                              feed_dict)
        costs1 += cost
        iters1 += m.num_steps
    costs2 = 0.0
    iters2 = 1
    for step, (x, y, t) in enumerate(t_data):
        # print(len(x[0]))
        # print(t.shape)
        # m.reset_state(m.batch_size)
        state = session.run(m.initial_state)
        # feed_dict = {m.input_data: x, m.targets: y, m.input_time: t}
        # for i in range(len(m.initial_state)):
        #     feed_dict[m.initial_state[i][0]] = m.initial_state[i][0].eval()
        #     feed_dict[m.initial_state[i][1]] = m.initial_state[i][1].eval()
        cost, state, _ = session.run([m.cost, m.final_state, tf.no_op()],  # x和y的shape都是(batch_size, num_steps)
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.extra_data: t,
                                      m.initial_state: state})
        # cost, state, _ = session.run([m.cost, m.final_state, eval_op],  # x和y的shape都是(batch_size, num_steps)
        #                              feed_dict)
        costs2 += cost
        iters2 += m.num_steps
    return np.exp(costs / iters), np.exp(costs1 / iters1), np.exp(costs2/iters2)


def main(_):
    # train_data = context_of_idx

    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=False, initializer=initializer):
            m = T_Model.T_Model(is_training=True, holiday=holiday, config=config)

        tf.global_variables_initializer().run()

        model_saver = tf.train.Saver(tf.global_variables())
        index = 1
        train_data1, valid_data = data_iterator(train_data, train_time, m.batch_size, m.num_steps)
        # f_data = [x_f, y_f, t_f]
        test_data1, valid_data1 = data_iterator(test_data, test_time, m.batch_size, m.num_steps)
        # t_data = [x_t, y_t, t_t]
        for i in range(config.iteration):
            # if index%5 == 0:
            #     config.learning_rate = config.learning_rate * 0.95
            # m.lr_decay(0.1)
            # print(session.run(m.lr))
            print("Training Epoch: %d ..." % (i + 1))
            train_perplexity, valid_perplexity, test_perplexity = run_epoch(session, m, train_data1, valid_data, test_data1, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f Valid Perplexity: %.3f Test Perplexity: %.3f" % (i + 1, train_perplexity, valid_perplexity, test_perplexity))

            if (i + 1) % config.save_freq == 0:
                print('model saving ...')
                model_saver.save(session, config.model_path+'-'+hidden+'-'+str(config.input_dim+types)+'-'+size +'-tgrid'+grids+'-'+holiday+'-%d' % (i + 1))
                print('Done!')


if __name__ == "__main__":
    # tf.app.run()
    main(1)