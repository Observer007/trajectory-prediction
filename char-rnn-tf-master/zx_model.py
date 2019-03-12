import sys
import numpy as np
import pickle
import tensorflow as tf
import math
from utils_t import *

def DistanceBetweenMeter(geo1, geo2):
    R = 6378137
    lonA, latA = geo1[0]/180*math.pi, geo1[1]/180*math.pi
    lonB, latB = geo2[0]/180*math.pi, geo2[1]/180*math.pi
    return R*math.acos(min(1.0, math.sin(math.pi/2-latA)*math.sin(math.pi/2-latB)*
        math.cos(lonA-lonB) + math.cos(math.pi/2-latA)*math.cos(math.pi/2-latB)))

class Model():
    def __init__(self, args, inter_embeddings=None,
                 intra_embeddings=None, is_training=True):
        self.intra_fea = args.intra_fea
        self.inter_fea = args.inter_fea
        self.ext_dim = args.ext_dim
        self.inter_embeddings = inter_embeddings
        self.intra_embeddings = intra_embeddings
        #
        self.intra_units = args.intra_units
        self.inter_units = args.inter_units         # inter_units == inter_fea
        # self.ext_units = args.ext_units
        #
        self.hidden_size = args.hidden_size
        #
        # self.batch_size = args.batch_size
        if is_training:
            self.num_steps = args.num_steps
        else:
            self.num_steps = 1
        #
        self.grid_size = args.grid_size
        assert self.grid_size>0
        self.keep_prob = args.keep_prob
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        #

        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.ext_data = tf.placeholder(tf.float32, [None, 9])
        self.batch_size = tf.unstack(tf.shape(self.x))[0]
        if not is_training:
            self.state = tf.placeholder(tf.float32, [])
        #
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)

        self.initial_state = self.lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        self.lr = args.learning_rate
        # self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        output_probs, loss = self.build_easy_model(is_training)
        self.output_probs = tf.reshape(output_probs, [self.batch_size, -1, self.grid_size])
        self.output_vocabs = tf.argmax(self.output_probs, axis=-1, output_type=tf.int32)
        acc = tf.equal(self.output_vocabs, self.y)
        acc = tf.cast(acc, tf.float32)
        self.acc = tf.reduce_sum(acc)/tf.reduce_sum(tf.ones_like(acc, dtype=tf.float32))
        self.loss = loss
        self.get_optim(args, loss)

    def build_easy_model(self, is_training=True):
        if self.inter_fea>0:
            assert self.inter_embeddings is not None
            inter_embeddings = tf.get_variable(name='embeddings', shape=(self.grid_size, self.inter_units), dtype=tf.float32,
                                               initializer=tf.constant_initializer(self.inter_embeddings))
            inter_data = tf.nn.embedding_lookup(inter_embeddings, self.x)
        else:
            onehot_embeddings = tf.get_variable(name='embeddings', shape=(self.grid_size, self.inter_units), dtype=tf.float32, initializer=self.weight_initializer)
            inter_data = tf.nn.embedding_lookup(onehot_embeddings, self.x)
        if self.intra_fea>0:
            assert self.intra_embeddings is not None
            intra_data = tf.nn.embedding_lookup(self.intra_embeddings, self.x)
            intra_data = tf.layers.dense(intra_data, self.intra_units, activation=tf.nn.sigmoid)
            input_data = tf.concat([intra_data, inter_data], axis=-1)
        else:
            input_data = inter_data

        if is_training and self.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell, output_keep_prob=self.keep_prob)
            input_data = tf.nn.dropout(input_data, self.keep_prob)
            # self.state = self.initial_state
        else:
            lstm_cell = self.lstm_cell
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, input_data, initial_state=self.initial_state)
        output = tf.reshape(tf.stack(outputs), [-1, self.hidden_size])
        self.final_state = state
        output = tf.layers.dense(output, self.grid_size)
        output = tf.reshape(output, (self.batch_size, -1, self.grid_size))
        #
        if self.ext_dim>0:
            # self.ext_data = tf.cast(self.ext_data, dtype=tf.float32)
            ext_output = tf.layers.dense(self.ext_data, self.grid_size)
            output = tf.add(output, tf.expand_dims(ext_output, 1))

        _prob = tf.nn.softmax(output, dim=-1)

        loss = tf.contrib.seq2seq.sequence_loss(
            output,
            self.y,
            tf.ones_like(self.y, dtype=tf.float32),
            average_across_timesteps=False,
        )
        # logits [batch_size*num_steps, vocab],targets -> batch_size*num_steps
        # _loss = tf.reduce
        _loss = tf.reduce_sum(loss)

        return _prob, _loss
    def get_optim(self, arg, loss):
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
        #                                   arg.max_grad_norm)
        self._grads = grads
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = self.optimizer.apply_gradients(zip(grads, tvars))

    def reset_state(self, sess, x):
        return sess.run([self.initial_state],
                        feed_dict={
                            self.x: x
                        })

    def train_one_batch(self, sess, x, y, ext_data):
        self.reset_state(sess, x)
        loss,acc, _ = sess.run([self.loss, self.acc, self._train_op],
                           feed_dict={
                               self.x: x,
                               self.y: y,
                               self.ext_data: ext_data
                           })
        return loss, acc

    def val_one_batch(self, sess, x, y, ext_data):
        self.reset_state(sess, x)
        loss, acc = sess.run([self.loss, self.acc],
                        feed_dict={
                            self.x: x,
                            self.y: y,
                            self.ext_data: ext_data
                        })
        return loss, acc

    def test_one_batch(self, sess, x, ext_data, given_step, pred_step, grid_near=None, grid2cor=None):
        state = self.reset_state(sess, x)
        pre_grids = np.zeros([x.shape[0], given_step+pred_step])
        pre_grid = None
        if grid_near is None:
            for i in range(given_step):
                pre_grid, state = sess.run([self.output_vocabs, self.final_state],
                                 feed_dict={
                                     self.x: np.expand_dims(x[:, i], axis=-1),
                                     self.ext_data: ext_data,
                                     self.initial_state: state
                                 })
                pre_grids[:, i] = x[:, i]
            assert pre_grid is not None
            pre_grids[:, given_step] = pre_grid[:, 0]
            x_step = pre_grid
            result = pre_grids
            for i in range(given_step+1, given_step+pred_step):
                pre_grid, state = sess.run([self.output_vocabs, self.final_state],
                                    feed_dict={
                                        self.x: x_step,
                                        self.ext_data: ext_data,
                                        self.initial_state: state
                                    })

                x_step = pre_grid
                result[:, i] = x_step[:, 0]
        else:
            for i in range(given_step):
                pre_prob, state = sess.run([self.output_probs, self.final_state],
                                    feed_dict={
                                        self.x: np.expand_dims(x[:, i], axis=-1),
                                        self.ext_data: ext_data,
                                        self.initial_state: state
                                    })
                pre_grid = self.get_near_grid(pre_prob, x[:, i], grid_near, grid2cor)
                pre_grids[:, i] = x[:, i]
            assert pre_grid is not None
            pre_grids[:, given_step] = pre_grid[:, 0]
            x_step = pre_grid
            result = pre_grids
            for i in range(given_step + 1, given_step + pred_step):
                pre_prob, state = sess.run([self.output_probs, self.final_state],
                                           feed_dict={
                                               self.x: x_step,
                                               self.ext_data: ext_data,
                                               self.initial_state: state
                                           })
                pre_grid = self.get_near_grid(pre_prob, pre_grid[:, 0], grid_near, grid2cor)
                x_step = pre_grid
                result[:, i] = x_step[:, 0]
        return result

    @staticmethod
    def get_near_grid(pre_prob, x, grid_near, grid2cor):
        assert len(pre_prob.shape) == 3 and pre_prob.shape[0] == x.shape[0]
        pre_grid = np.zeros([pre_prob.shape[0], 1])
        pre_prob = np.squeeze(pre_prob)
        select_prob = np.zeros_like(pre_prob)
        for i in range(pre_prob.shape[0]):
            for j in grid_near[x[i]]:
                if DistanceBetweenMeter(grid2cor[j], grid2cor[x[i]])>=3000:
                    print('error', DistanceBetweenMeter(grid2cor[j], grid2cor[x[i]]))
                select_prob[i][j] = pre_prob[i][j]
        pre_grid = np.expand_dims(np.argmax(select_prob, axis=-1), -1)
        return pre_grid
