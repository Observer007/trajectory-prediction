# coding:utf-8
import tensorflow as tf


class Model(object):
    def __init__(self, is_training, network_embeddings, config):
        self.batch_size = batch_size = config.batch_size
        print('dfsdfsd', config.batch_size)
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate
        self.input_dim = input_dim = config.input_dim + config.add_dim
        index = int(num_steps/2)
        self._encoder_input = tf.placeholder(tf.float32, [batch_size, index, input_dim])
        self._decoder_input = tf.placeholder(tf.float32, [batch_size, index+1, input_dim])
        self._targets = tf.placeholder(tf.int32, [batch_size, index+1])  # 声明输入变量x, y

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        self.encoder = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        embeddings = tf.constant(network_embeddings, dtype=tf.float32)
        # with tf.device("/gpu:0"):
        #     embedding = tf.get_variable("embedding", [vocab_size, size])  # size是wordembedding的维度
        #     inputs = tf.nn.embedding_lookup(embedding,
        #                                     self._input_data)  # 返回一个tensor，shape是(batch_size, num_steps, size)
        # inputs = tf.concat([self._input_time, self._input_data], 2)
        #inputs = self._input_data
        #print(inputs.shape)
        # if is_training and config.keep_prob < 1:
        #     inputs = tf.nn.dropout(inputs, config.keep_prob)  # 随机扔掉一些隐层单元训练

        encoder_output, encoder_state = tf.nn.dynamic_rnn(self.encoder, self.encoder_input,
                                                        dtype=tf.float32, scope="encoder")

        self.decoder = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        output_layer = tf.layers.Dense(vocab_size,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        sequence_length = tf.constant([index+1] * batch_size, dtype=tf.int32)
        max_sequence_length = tf.reduce_max(sequence_length)
        with tf.variable_scope("decoder"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input,
                                                                sequence_length,
                                                                time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder,
                                                 training_helper,
                                                 encoder_state,
                                                 output_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                     impute_finished = True,
                                                     maximum_iterations = max_sequence_length)

        with tf.variable_scope("decoder", reuse=True):
            start_segment = tf.tile(tf.constant([7], dtype=tf.int32), [batch_size], name='start_segment')
            end_segment = tf.constant(13)
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                  start_segment,
                                                    end_segment)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder,
                                                 predicting_helper,
                                                 encoder_state,
                                                 output_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                           maximum_iterations=max_sequence_length)
        if is_training:
            training_logits = tf.identity(training_decoder_output.rnn_output, "logits")
            masks = tf.sequence_mask(sequence_length, max_sequence_length, dtype=tf.float32, name='masks')
            # cost = tf.contrib.seq2seq.sequence_loss(
            #     training_logits,
            #     self._targets,
            #     masks)
            cost = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(training_logits, [-1, vocab_size])],
                [tf.reshape(self._targets, [-1])],
                [tf.ones([batch_size * (index+1)])])
            tvars = tf.trainable_variables()
            self._cost = tf.reduce_sum(cost)/batch_size
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              config.max_grad_norm)
            #self._grads = grads
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self._train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')



    @property
    def encoder_input(self):
        return self._encoder_input

    @property
    def decoder_input(self):
        return self._decoder_input

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def logits(self):
        return self.predicting_logits

