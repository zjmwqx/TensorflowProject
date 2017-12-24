#!/usr/bin/python
# -*- coding:utf-8 -*-
class Config(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.5
    max_grad_norm = 10
    num_layers = 2
    num_steps = 21
    hidden_size = 128
    max_epoch = 5
    max_max_epoch = 10
    keep_prob = 0.95
    lr_decay = 1 / 1.15
    batch_size = 100  # single sample batch
    vocab_size = 5000
    target_num = 5  # POS tagging tag number for Chinese
    bi_direction = True  # LSTM or BiLSTM


def get_config():
    config = Config()
    return config
