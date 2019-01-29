#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/29 10:00
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : main_process.py
@Software: PyCharm
@Github  : zhangyuo
"""
from tool.logger import logger
from config.config import *
from process.data_process import *
from tensorflow.contrib import learn
import numpy as np
import os
import tensorflow as tf
from tool.text_classify import BiRNN


def train():
    """
    model train process
    :return:
    """
    logger.info('Loading train data...')
    # load train data
    x_text, y = load_data_and_labels(TRAIN_DATA_PATH)
    # build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_size = len(vocab_processor.vocabulary_)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x, y, x_shuffled, y_shuffled
    logger.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    # define pretrained embedding
    embedding_dim = EMBEDDING_DIM
    embeddings = None
    if IS_PRETRAINED_EMBEDDING:
        embeddings = np.load(PRETRAINED_EMBEDDING_PATH)
        logger.info("embedding shape {}".format(embeddings.shape))
        vocab_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
    # load saved middle model last time
    ckpt = None
    if IS_MIDDLE_MODEL:
        assert os.path.isdir(MIDDLE_PATH), '{} must be a directory'.format(MIDDLE_PATH)
        ckpt = tf.train.get_checkpoint_state(MIDDLE_PATH)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

    model = BiRNN(embedding_dim=embedding_dim,
                  hidden_size=hidden_size,
                  num_layer=num_layer,
                  vocab_size=vocab_size,
                  attention_size=attention_size,
                  sequence_length=x_train.shape[1],
                  num_classes=y_train.shape[1],
                  grad_clip=grad_clip,
                  lr=lr,
                  l2_reg_lambda=l2_reg_lambda,
                  dropout=dropout,
                  optimizer=optimizer,
                  num_checkpoints=num_checkpoints,
                  batch_size=batch_size,
                  num_epochs=num_epochs,
                  evaluate_every=evaluate_every,
                  checkpoint_every=checkpoint_every)

    model.build_graph()
    model.train(vocab_processor, x_train, y_train, x_dev, y_dev, pre_embeddings=embeddings, checkpoint_file=ckpt)


if __name__ == '__main__':
    train()
