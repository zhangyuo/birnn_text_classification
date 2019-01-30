#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/9 15:55
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : text_classify.py
@Software: PyCharm
@Github  : zhangyuo
"""

from tool.logger import logger
import tensorflow as tf
import time
import os
from config.config import IS_PRETRAINED_EMBEDDING, IS_MIDDLE_MODEL
import datetime
from process.data_process import batch_iter


class BiRNN(object):
    """
    A BiRNN with attention layer for text classification.
    """

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 num_layer,
                 vocab_size,
                 attention_size,
                 sequence_length,
                 num_classes,
                 grad_clip,
                 lr,
                 l2_reg_lambda,
                 dropout,
                 optimizer,
                 num_checkpoints=5,
                 batch_size=128,
                 num_epochs=200,
                 evaluate_every=100,
                 checkpoint_every=100):
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.vocab_size = vocab_size
        self.attention_size = attention_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.grad_clip = grad_clip
        self.lr = lr
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout = dropout
        self.optimizer = optimizer
        self.num_checkpoints = num_checkpoints
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every

    def build_graph(self):
        self.add_placeholders()
        self.regular_loss_op()
        self.embedding_layer_op()
        self.birnn_layer_op()
        self.attention_layer_op()
        self.loss_op()
        self.accuracy_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        """
        Placeholders for input, output and dropout
        :return:
        """
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def regular_loss_op(self):
        """
        Keeping track of l2 regularization loss (optional)
        :return:
        """
        self.l2_loss = tf.constant(0.0)

    def embedding_layer_op(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self._word_embeddings = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.embedding_dim], stddev=0.1, mean=0.5),
                # tf.initialize_variables([self.vocab_size, self.embedding_dim],),
                trainable=False,
                name="_word_embeddings")
            # self._word_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
            #                                     trainable=False,
            #                                     name="_word_embeddings")
            self.word_embeddings = tf.nn.embedding_lookup(params=self._word_embeddings,
                                                          ids=self.input_x,
                                                          name='word_embeddings')

    def birnn_layer_op(self):
        """
        define birnn layer
        :return:
        """
        # define forward cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            logger.info(tf.get_variable_scope().name)
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(self.hidden_size) for _ in list(range(self.num_layer))]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.dropout_keep_prob)
        # define backward cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            logger.info(tf.get_variable_scope().name)
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(self.hidden_size) for _ in list(range(self.num_layer))]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=self.dropout_keep_prob)
        # self.input_x shape: (batch_size , sequence_length)
        # self.word_embeddings shape: (batch_size , sequence_length, embedding_dim)
        # bidirection rnn requires input shape : (sequence_length, batch_size, hidden_size)
        hidden_inputs = tf.transpose(self.word_embeddings, [1, 0, 2])
        # bidirection rnn target input is a list
        hidden_inputs = tf.reshape(hidden_inputs, [-1, self.hidden_size])
        hidden_inputs = tf.split(hidden_inputs, self.sequence_length, 0)
        # build bidirection rnn
        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            self.hidden_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m,
                                                                                lstm_bw_cell_m,
                                                                                hidden_inputs,
                                                                                dtype=tf.float32)

    def attention_layer_op(self):
        """
        define attention layer
        :return:
        """
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2 * self.hidden_size, self.attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.attention_size]), name='attention_b')
            u_list = []
            for t in list(range(self.sequence_length)):
                u_t = tf.tanh(tf.matmul(self.hidden_outputs[t], attention_w) + attention_b)
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([self.attention_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in list(range(self.sequence_length)):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            self.alpha = tf.nn.softmax(attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(self.alpha, [1, 0]), [self.sequence_length, -1, 1])
            self.attention_output = tf.reduce_sum(self.hidden_outputs * alpha_trans, 0)
            logger.info('attention layer output shape is %s' % self.attention_output.shape)

        with tf.name_scope("output"):
            # outputs shape: (sequence_length, batch_size, 2*rnn_size)
            W = tf.Variable(tf.truncated_normal([2 * self.hidden_size, self.num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.zeros([self.num_classes]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.attention_output, W, b, name="logits")
            self.prob = tf.nn.softmax(self.logits, name='prob')
            self.predictions = tf.argmax(self.prob, 1, name="predictions")

    def loss_op(self):
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.losses.softmax_cross_entropy(self.input_y, self.logits)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def accuracy_op(self):
        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.input_y, axis=1), self.predictions)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    def trainstep_op(self):
        with tf.name_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

            # 法一
            self.grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v] for g, v in
                                   self.grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            # 法二
            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
            # self.train_op = optim.apply_gradients(zip(grads, tvars))

    def init_op(self):
        """
        Initialize all variables
        :return:
        """
        self.init_op = tf.global_variables_initializer()

    def train(self, vocab_processor, x_train, y_train, x_dev, y_dev, pre_embeddings=None, checkpoint_file=None):
        """
        model train process
        :return:
        """
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
        # GPU assign
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = False
        with tf.Session(config=tf_config) as sess:
            sess.run(self.init_op)
            self._add_summary(sess, vocab_processor)
            # using pre trained embeddings
            if IS_PRETRAINED_EMBEDDING:
                sess.run(self._word_embeddings.assign(pre_embeddings))
                del pre_embeddings
            # restore model
            if IS_MIDDLE_MODEL:
                saver.restore(sess, checkpoint_file.model_checkpoint_path)
            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.train_step(sess, x_batch, y_batch)
                current_step = tf.train.global_step(sess, self.global_step)
                if current_step % self.evaluate_every == 0:
                    logger.info("Evaluation:")
                    self.dev_step(sess, x_dev, y_dev)
                    logger.info("")
                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess, self.checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {}\n".format(path))

    def train_step(self, sess, x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.dropout
        }
        # scores, predictions = sess.run([self.scores, self.predictions], feed_dict)
        _, step, summaries, loss, accuracy = sess.run(
            [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def dev_step(self, sess, x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [self.global_step, self.dev_summary_op, self.loss, self.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.dev_summary_writer.add_summary(summaries, step)

    ###########################private func##############################
    def _add_summary(self, sess, vocab_processor):
        """
        Tesorboard 图形化展示
        :param sess:
        :return:
        """
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
