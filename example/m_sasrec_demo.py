"""
Created on Nov 20, 2021
Updated on Apr 23, 2022
train SASRec demo
@author: Ziyao Geng(zggzy1996@163.com)
"""
import os

import numpy as np
import tensorflow as tf
import pandas as pd
from absl import flags, app
from time import time
from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import SASRec
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg
from data.utils.data_loader import DataGenerator
from multiprocessing import Process, Queue
from collections import defaultdict
import datetime
from tqdm import tqdm

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Setting training parameters
flags.DEFINE_string("file_path", "data/ml-1m/ratings.dat", "file path.")
flags.DEFINE_string("train_path", "data/ml-1m/ml_seq_train.txt", "train path. If set to None, the program will split the dataset.")
flags.DEFINE_string("val_path", "data/ml-1m/ml_seq_val.txt", "val path.")
flags.DEFINE_string("test_path", "data/ml-1m/ml_seq_test.txt", "test path.")
flags.DEFINE_string("meta_path", "data/ml-1m/ml_seq_meta.txt", "meta path.")
flags.DEFINE_integer("item_dim", 64, "The size of item embedding dimension.")
flags.DEFINE_integer("user_dim", 50, "The size of user embedding dimension.")
flags.DEFINE_float("embed_reg", 0.0, "The value of embedding regularization.")
flags.DEFINE_integer("blocks", 2, "The Number of blocks.")
flags.DEFINE_integer("num_heads", 2, "The Number of attention heads.")
flags.DEFINE_integer("ffn_hidden_unit", 64, "Number of hidden unit in FFN.")
flags.DEFINE_float("dnn_dropout", 0.2, "Float between 0 and 1. Dropout of user and item MLP layer.")
flags.DEFINE_float("layer_norm_eps", 1e-6, "Small float added to variance to avoid dividing by zero.")
flags.DEFINE_boolean("use_l2norm", False, "Whether user embedding, item embedding should be normalized or not.")
flags.DEFINE_string("loss_name", "binary_cross_entropy_loss", "Loss Name.")
flags.DEFINE_float("gamma", 0.5, "If hinge_loss is selected as the loss function, you can specify the margin.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("neg_num", 4, "The number of negative sample for each positive sample.")
flags.DEFINE_integer("seq_len", 100, "The length of user's behavior sequence.")
flags.DEFINE_integer("epochs", 20, "train steps.")
flags.DEFINE_integer("batch_size", 512, "Batch Size.")
flags.DEFINE_integer("test_neg_num", 100, "The number of test negative samples.")
flags.DEFINE_integer("k", 10, "recall k items at test stage.")
flags.DEFINE_integer("seed", None, "random seed.")

def main(argv):
    # TODO: 1. Split Data
    if FLAGS.train_path == "None":
        train_path, val_path, test_path, meta_path = ml.split_seq_data(file_path=FLAGS.file_path)
    else:
        train_path, val_path, test_path, meta_path = FLAGS.train_path, FLAGS.val_path, FLAGS.test_path, FLAGS.meta_path
    # with open(meta_path) as f:
    with open("data/Beauty/Beauty_seq_meta.txt") as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split(' ')]
    # TODO: 2. Load Sequence Data
    user2seq = ml.load_user2seq("data/Beauty/Beauty.txt")
    sampler = WarpSampler(user2seq, FLAGS.neg_num, max_item_num, n_workers=3, queue_size=2)
    # train_data = ml.load_seq_data(train_path, "train", FLAGS.seq_len, FLAGS.neg_num, max_item_num)
    train_data = ml.load_txt_data("data/Beauty/Beauty.txt", "train", FLAGS.seq_len, FLAGS.neg_num, max_item_num, max_user_num)
    train_generator = DataGenerator(train_data, FLAGS.batch_size)
    # val_data = ml.load_seq_data(val_path, "val", FLAGS.seq_len, FLAGS.neg_num, max_item_num)
    val_data = ml.load_txt_data("data/Beauty/Beauty.txt", "val", FLAGS.seq_len, FLAGS.neg_num, max_item_num, max_user_num)
    val_generator = DataGenerator(val_data, FLAGS.batch_size)
    # test_data = ml.load_seq_data(test_path, "test", FLAGS.seq_len, FLAGS.test_neg_num, max_item_num)
    test_data = ml.load_txt_data("data/Beauty/Beauty.txt", "test", FLAGS.seq_len, FLAGS.test_neg_num, max_item_num, max_user_num)
    # TODO: 3. Set Model Hyper Parameters.
    model_params = {
        'item_num': max_item_num + 1,
        'user_num': max_user_num + 1,
        'item_dim': FLAGS.item_dim,
        'user_dim': FLAGS.user_dim,
        'seq_len': FLAGS.seq_len,
        'blocks': FLAGS.blocks,
        'num_heads': FLAGS.num_heads,
        'ffn_hidden_unit': FLAGS.ffn_hidden_unit,
        'dnn_dropout': FLAGS.dnn_dropout,
        'use_l2norm': FLAGS.use_l2norm,
        'loss_name': FLAGS.loss_name,
        'gamma': FLAGS.gamma,
        'embed_reg': FLAGS.embed_reg,
        'seed': FLAGS.seed,
        'neg_num': FLAGS.neg_num
    }
    # 获取当前时间作为模型文件名后缀
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：20241130_123456
    model_name = f"sasrec_{start_time}"
    # TODO: 4. Build Model
    model = SASRec(**model_params)
    model.compile(optimizer=Adam(learning_rate=FLAGS.learning_rate))
    # TODO: 5. Fit Model
    try:
        results = []
        for epoch in range(1, FLAGS.epochs + 1):
            t1 = time()
            model.fit(
                x=train_generator,
                epochs=1,
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=4,
                # batch_size=FLAGS.batch_size
            )
            t2 = time()
            eval_dict = eval_pos_neg(model, test_data, ['hr', 'mrr', 'ndcg'], FLAGS.k)
            # 每个epoch重新生成训练集、验证集、测试集的负样本
            transfer_neg_data(sampler, train_data, max_user_num)
            # @10, @20, @40
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR_10 = %.4f, MRR@10 = %.4f, NDCG@10 = %.4f,'
                  ' HR@20 = %.4f, MRR@20 = %.4f, NDCG@20 = %.4f, HR@40 = %.4f, MRR@40 = %.4f, NDCG@40 = %.4f'
                  % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg'], eval_dict['hr_20'], eval_dict['mrr_20'], eval_dict['ndcg_20'], eval_dict['hr_40'], eval_dict['mrr_40'], eval_dict['ndcg_40']))
            results.append([epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg'], eval_dict['hr_20'], eval_dict['mrr_20'], eval_dict['ndcg_20'], eval_dict['hr_40'], eval_dict['mrr_40'], eval_dict['ndcg_40']])
        # write logs
        pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hr@10', 'mrr@10', 'ndcg@10', 'hr@20', 'mrr@20', 'ndcg@20', 'hr@40', 'mrr@40', 'ndcg@40']).\
            to_csv("logs/SASRec_log_{}_maxlen_{}_blocks_{}_heads_{}.csv".format(start_time, FLAGS.seq_len, FLAGS.blocks, FLAGS.num_heads), index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        model.save(model_name, save_format='tf')
        sampler.close()


def transfer_neg_data(sampler, train_data, max_user_num):
    value_list = sampler.next_batch()
    # 生成一个初始值为0的列表，因为user编号从1开始，所以长度为max_user_num+1
    user_get_neg_index = [0] * (max_user_num + 1)
    # 三个集合要替换的负样本列表
    train_data_neg_item = []
    # 从value_list中取出user_neg_list和user_test_neg_list
    user_neg_list = value_list
    # train:(None, 1)->(None,)
    train_users = np.reshape(train_data['user'], (-1,))
    for u in train_users:
        train_data_neg_item.append(user_neg_list[u][user_get_neg_index[u]])
        user_get_neg_index[u] += 1
    train_data['neg_item'] = np.array(train_data_neg_item, dtype=np.int32)


def no_augmentation_sample_function(user2seq, neg_num, max_item_num, result_queue):
    def sample():
        user_neg_list = defaultdict(list)
        for user in user2seq:
            seq = user2seq[user]
            neg_item = ml.gen_negative_samples_except_pos(neg_num, seq, max_item_num)
            user_neg_list[user].append(neg_item)
        return user_neg_list

    while True:
        value_list = sample()
        result_queue.put(value_list)


def sample_function(user2seq, neg_num, max_item_num, result_queue):
    def sample():
        user_neg_list = defaultdict(list)
        for user in user2seq:
            seq = user2seq[user]
            # 训练集加验证集的负样本数目(排除：1、第一个项目不能作为正样本，2、最后一个项目属于测试集的正样本)
            train_val_neg_num = len(seq)-2
            for _ in range(train_val_neg_num):
                neg_item = ml.gen_negative_samples_except_pos(neg_num, seq, max_item_num)
                user_neg_list[user].append(neg_item)
        return user_neg_list

    while True:
        value_list = sample()
        result_queue.put(value_list)


class WarpSampler(object):
    def __init__(self, user2seq, neg_num, max_item_num, n_workers=2, queue_size=2):
        self.result_queue = Queue(maxsize=queue_size)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(user2seq, neg_num, max_item_num, self.result_queue)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


class EpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, outer_epoch=0):
        super(EpochCallback, self).__init__()
        self.model = model
        self.outer_epoch = outer_epoch

    def on_epoch_begin(self, epoch, logs=None):
        # 在每个 epoch 开始时，将当前 epoch 传递到模型的 `call` 方法中
        print(f"Epoch {self.outer_epoch} starts.")
        self.model.epoch = self.outer_epoch  # 存储 epoch 到模型中

    def get_epoch(self):
        return self.outer_epoch  # 提供获取epoch的方式


if __name__ == '__main__':
    app.run(main)