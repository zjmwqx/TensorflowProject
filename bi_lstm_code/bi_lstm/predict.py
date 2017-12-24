#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os, sys
import glob
import tensorflow as tf
import numpy as np

default_path = os.path.join(os.path.dirname(__file__), "..")  # 设置默认路径
sys.path.append(default_path)
from utils import reader
from bi_lstm.setting import get_config
from bi_lstm.model import BiLstmModel

data_path = os.path.join(default_path, "dataset")
train_dir = os.path.join(default_path, "bi_lstm", "ckpt")





class PredictModel(object):
    """
    模型加载器
    """

    def __init__(self, data_path, ckpt_path):
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing pos_tagger class...")
        self.model = self._init_pos_model(self.session, self.ckpt_path)

    def _init_pos_model(self, session, ckpt_path):
        config = get_config()
        config.batch_size = 1
        config.num_steps = 1

        with tf.variable_scope("pos_var_scope"):  # Need to Change in Pos_Tagger Save Function
            bi_lstm_model = BiLstmModel(is_training=False, config=config)  # save object after is_training

        if len(glob.glob(ckpt_path + '.data*')) > 0:  # file exist with pattern: 'pos.ckpt.data*'
            print("Loading model parameters from %s" % ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("pos_var_scope")]
            tf.train.Saver(model_vars).restore(session, ckpt_path)
        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return bi_lstm_model

    def _bi_predict_pos_tags(self, session, model, words, data_path):
        '''
        Define prediction function of POS Tagging
        return tuples [(word, tag)]
        '''
        word_data = reader.sentence_to_word_ids(data_path, words)
        print(word_data)
        tag_data = [0] * len(word_data)
        predict_id = []
        result_list = []
        for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                      model.num_steps)):
            fetches = [model.cost, model.logits]  # eval_op define the m.train_op or m.eval_op
            feed_dict = {}
            feed_dict[model.input_data] = x
            feed_dict[model.targets] = y
            cost, logits = session.run(fetches, feed_dict)
            print(logits)
            result_list.append(logits)
            # predict_id.append(int(np.argmax(logits)))
        predict_id = self.get_tag(result_list)
        predict_tag = reader.word_ids_to_sentence(data_path, predict_id)
        return zip(words, predict_tag)  # [(word,tag),(word,tag)......]返回word与tag的tuple_list

    def get_tag(self,result_list):
        match_ids = {0:[0,1],1:[2,3],2:[0,1],3:[2,3]}
        predict_id = []
        pre_tag = np.argmax(result_list[0])
        predict_id.append(pre_tag)
        for logits in result_list[1:]:
            tag_one,tag_two = match_ids[pre_tag][0:2]
            if logits[0][tag_one]>logits[0][tag_two]:
                pre_tag = tag_one
                predict_id.append(tag_one)
            else:
                pre_tag = tag_two
                predict_id.append(tag_two)

        predict_tag = reader.word_ids_to_sentence(data_path, predict_id)
        return predict_id

    def predict(self, words):
        tagging = self._bi_predict_pos_tags(self.session, self.model, words, self.data_path)
        return tagging

def load_model():
    ''' data_path e.g.: ./deepnlp/pos/data/zh
        ckpt_path e.g.: ./deepnlp/pos/bi_ckpt/zh/pos.ckpt
        ckpt_file e.g.: ./deepnlp/pos/bi_ckpt/zh/pos.ckpt.data-00000-of-00001
    '''
    data_path = os.path.abspath('./../dataset')
    train_dir = os.path.abspath("./ckpt")
    data_path = os.path.join(data_path)  # POS vocabulary data path
    ckpt_path = os.path.join(train_dir, "pos_bilstm.ckpt")  # POS model checkpoint path
    return PredictModel( data_path, ckpt_path)


if __name__ == "__main__":
    predict_obj = load_model()
    num_step = 21
    words = [word for word in "安徽省宣城市宣州区百佳超市"]
    padding_list = ["_PAD" for i in range(0,(num_step-len(words)))]
    words.extend(padding_list)
    word_tags = predict_obj.predict(words)
    for (word, tag) in word_tags:
        print(word, "\t", tag)

