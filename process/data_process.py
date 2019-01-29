#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/29 10:00
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : data_process.py
@Software: PyCharm
@Github  : zhangyuo
"""
import re
import numpy as np


def load_data_and_labels(path):
    """
    load data file, build input and output format of train or test file
    :param path:
    :return:
    """
    positive_examples = list(open(path, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip().split('\t') for s in positive_examples]
    # Split by words
    x_text = [k[0] for k in positive_examples]
    y_text = [k[1] for k in positive_examples]
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels: pos=[1,0], neg=[0,1]
    y = np.array([[1, 0] if k == 0 else [0, 1] for k in y_text])
    return [x_text, y]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
