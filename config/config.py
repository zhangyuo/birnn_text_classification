#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/29 9:59
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : config.py
@Software: PyCharm
@Github  : zhangyuo
"""

TRAIN_DATA_PATH = '../data/rt_train.txt'
TEST_DATA_PATH = '../data/rt_test.txt'

# pretrained embedding
IS_PRETRAINED_EMBEDDING = False
PRETRAINED_EMBEDDING_PATH = ''

# saved middle process path
IS_MIDDLE_MODEL = False
MIDDLE_PATH = ''

# model Hyperparameters
# percentage of the training data to use for validation
dev_sample_percentage = 0.1
# word embedding dim
EMBEDDING_DIM = 100
# hidden units of RNN, as well as dimensionality of word embedding
hidden_size = 100
# number of layers of RNN
num_layer = 2
# attention layer size
attention_size = 200
# Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD
optimizer = "Adam"
# learning rate
lr = 1e-3
# gradient clipping
grad_clip = 5.0
# l2 regularization lambda
l2_reg_lambda = 0.5
# dropout keep_prob
dropout = 0.5

# training parameters
# number of checkpoints to store
num_checkpoints = 5
# batch Size
batch_size = 128
# number of training epochs
num_epochs = 200
# evaluate model on dev set after this many steps
evaluate_every = 100
# save model after this many steps
checkpoint_every = 100