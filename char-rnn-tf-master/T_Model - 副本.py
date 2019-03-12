# coding:utf-8
import tensorflow as tf
import numpy as np

class T_Model(object):
    def __init__(self, is_training, holiday, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate
        self.input_dim = input_dim = config.input_dim + config.add_dim

        # self._input_time = tf.placeholder(tf.float32, [batch_size, num_steps, 1])
        self._input_data = tf.placeholder(tf.float32, [None, None, input_dim])


        self._extra_data = tf.placeholder(tf.float32, [None, 9])
        self._targets = tf.placeholder(tf.int32, [None, num_steps])  # 声明输入变量x, y
        if config.add_dim:
            inter_data = self._input_data[:, :, :config.input_dim]
            intra_data = tf.reshape(self._input_data[:, :, config.input_dim:], [-1, config.add_dim])
            print(inter_data.shape)
            print(intra_data.shape)
            intra_embedding = tf.get_variable("embedding", [config.add_dim, 16], dtype=tf.float32)        #size是wordembedding的维度
            intra_embedding_b = tf.get_variable("embedding_b", [16], dtype=tf.float32)
            print(intra_embedding.shape)
            intra_data = tf.nn.sigmoid(tf.matmul(intra_data, intra_embedding)+intra_embedding_b)
            intra_data = tf.reshape(intra_data, [batch_size, num_steps, 16])
            inputs = tf.concat([inter_data, intra_data], axis=2)
        else:
            inputs = self._input_data
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        # self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=False)
        self.cell = lstm_cell
        self._initial_state = self.cell.zero_state(batch_size, tf.float32)

        # with tf.device("/gpu:0"):
        #     embedding = tf.get_variable("embedding", [vocab_size, size])  # size是wordembedding的维度
        #     inputs = tf.nn.embedding_lookup(embedding,
        #                                     self._input_data)  # 返回一个tensor，shape是(batch_size, num_steps, size)
        # inputs = tf.concat([self._input_time, self._input_data], 2)

        print(inputs.shape)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)  # 随机扔掉一些隐层单元训练
        # c1 = tf.constant(1, dtype=tf.float32)
        # c2 = tf.constant(2.7182, dtype=tf.float32)
        # Ones = tf.ones([1, size], dtype=tf.float32)
        outputs = []
        state = self._initial_state
        #tmp = tf.Variable("tmp", [batch_size, 1])
        # T_w = tf.get_variable("T_w", [size, size])
        # T_b = tf.get_variable("T_b", [size])
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                (cell_output, state) = self.cell(inputs[:, time_step, :],
                                                 state)  # inputs[:, time_step, :]的shape是(batch_size, size)
                outputs.append(cell_output)
        
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        print(output.shape)

        if is_training:
            extra_output = []
            for i in range(10):
                extra_output.append(self._extra_data)
            extra_output = tf.reshape(tf.concat(extra_output, 1), [-1, 9])
        else:
            extra_output = tf.reshape(self._extra_data, [-1, 9])
        print(np.array(extra_output).shape)
        if holiday=='True':
            flag=True
        else:
            flag=False
        if flag:
            extra_weight = tf.get_variable('extra', [9, 4])
            extra_bias = tf.get_variable('extra_b', [4])
            extra_output = tf.nn.sigmoid(tf.matmul(extra_output, extra_weight)+extra_bias)
            print(extra_output.shape)
            output = tf.concat([output, extra_output], 1)
            print(output.shape)
            softmax_w = tf.get_variable("softmax_w", [size+4, vocab_size])  # size is the hidden layyer size.

        else:
            softmax_w = tf.get_variable("softmax_w", [size, vocab_size])  # size is the hidden layyer size.
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        # self._logits = logits
        self._final_state = state
        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return

        # logits = tf.nn.softmax(logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])        # logits [batch_size*num_steps, vocab],targets -> batch_size*num
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        tvars = tf.trainable_variables()
        # print(tvars)
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        self._grads = grads
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        # self._train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self._global_step)
        self._train_op = self.optimizer.apply_gradients(zip(grads, tvars))


    def reset_state(self, batch_size):
        self._initial_state = self.cell.zero_state(batch_size, tf.float32)

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op

    @property
    def extra_data(self):
        return self._extra_data

    @property
    def global_step(self):
        return self._global_step

    @property
    def input_time(self):
        return self._input_time

    @property
    def cell_p(self):
        return self._cell_p

    @property
    def cell_s(self):
        return self._cell_s

    @property
    def cell_l(self):
        return self._cell_l

    @property
    def g0(self):
        return self._g0

    @property
    def g(self):
        return self._g

    @property
    def grads(self):
        return self._grads

    @property
    def logits(self):
        return self._logits
