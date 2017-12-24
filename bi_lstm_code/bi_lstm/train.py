#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os, sys
import numpy as np
import time
import tensorflow as tf

default_path = os.path.abspath("./..")
sys.path.append(default_path)
from utils import reader
from bi_lstm.model import BiLstmModel
from bi_lstm.setting import get_config

flags = tf.flags
logging = tf.logging

data_path = os.path.join(default_path, "dataset")  # 训练集的目录
train_dir = os.path.join(default_path, "bi_lstm", "ckpt")  # 训练参数目录

flags.DEFINE_string("pos_data_path", data_path, "data_path")
flags.DEFINE_string("pos_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("pos_scope_name", "pos_var_scope", "Define POS Tagging Variable Scope Name")
FLAGS = flags.FLAGS


def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps

    start_time = time.time()
    costs = 0.0
    iters = 0

    for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                  model.num_steps)):
        fetches = [model.cost, model.logits, eval_op]  # eval_op define the m.train_op or m.eval_op
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        cost, logits, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

        # Save Model to CheckPoint when is_training is True
        if model.is_training:
            if step % (epoch_size // 10) == 10:
                checkpoint_path = os.path.join(FLAGS.pos_train_dir, "pos_bilstm.ckpt")
                model.saver.save(session, checkpoint_path)
                print("Model Saved... at time step " + str(step))

    return np.exp(costs / iters)


def train():
    if not FLAGS.pos_data_path:
        raise ValueError("No data files found in 'data_path' folder")

    raw_data = reader.load_data(FLAGS.pos_data_path)
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocabulary = raw_data

    train_config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    # eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                    train_config.init_scale)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=None, initializer=initializer):
            train_model = BiLstmModel(is_training=True, config=train_config)
        with tf.variable_scope(FLAGS.pos_scope_name, reuse=True, initializer=initializer):
            valid_model = BiLstmModel(is_training=False, config=train_config)
            # test_model = BiLstmModel(is_training=False, config=eval_config)


        # CheckPoint State
        ckpt = tf.train.get_checkpoint_state(FLAGS.pos_train_dir)
        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.saver.restore(session, tf.train.latest_checkpoint(FLAGS.pos_train_dir))
            # tf.summary.FileWriter("logs/", session.graph) #创建tensorboard图
            # sys.exit(0)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        for i in range(train_config.max_max_epoch):
            lr_decay = train_config.lr_decay ** max(i - train_config.max_epoch, 0.0)
            train_model.assign_lr(session, train_config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model.lr)))
            train_perplexity = run_epoch(session, train_model, train_word, train_tag, train_model.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, valid_model, dev_word, dev_tag, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        # test_perplexity = run_epoch(session, test_model, test_word, test_tag, tf.no_op())
        # print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    train()
