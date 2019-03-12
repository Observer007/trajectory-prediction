#coding:utf-8
import tensorflow as tf

class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate
        types = config.add_dim
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #声明输入变量x, y
        # if types:
        self._input_features = tf.placeholder(tf.float32, [batch_size, num_steps, types])
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = self.cell.zero_state(batch_size, tf.float32)

        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])        #size是wordembedding的维度
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)        #返回一个tensor，shape是(batch_size, num_steps, size)
        if types!=0:
            inputs = tf.concat([self._input_features, inputs], 2)
        print(tf.shape(inputs))
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)                    #随机扔掉一些隐层单元训练

        
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(inputs[:, time_step, :], state) #inputs[:, time_step, :]的shape是(batch_size, size)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])#size is the hidden layyer size.
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._final_state = state
        self._logits = logits
        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])#logits [batch_size*num_steps, vocab],targets -> batch_size*num
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        tvars = tf.trainable_variables()
        #print(tvars)
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        self._grads = grads
        #self._global_step = tf.Variable(0)
        #self.lr = tf.train.exponential_decay(self.lr, self._global_step, 2, 0.98, staircase=True)
        #self.lr
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        # self._train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self._global_step)
        self._train_op = self.optimizer.apply_gradients(zip(grads, tvars))


    def reset_state(self, batch_size):
        self._initial_state = self.cell.zero_state(batch_size, tf.float32)

    @property
    def input_data(self):
        return self._input_data

    @property
    def input_features(self):
        return self._input_features

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
    def global_step(self):
        return self._global_step

    @property
    def grads(self):
        return self._grads

    @property
    def logits(self):
        return self._logits