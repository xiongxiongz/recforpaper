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
from reclearn.evaluator import evaluate
from data.utils.data_loader import DataGenerator
from multiprocessing import Process, Queue
from collections import defaultdict
import datetime
from tqdm import tqdm
from data_process import data_partition
from reclearn.models.losses import get_loss_with_istarget


# Setting training parameters
flags.DEFINE_string("dataset", "ml1m", "The dataset name.")
flags.DEFINE_integer("gpu", 0, "The GPU number.")
flags.DEFINE_integer("print_freq", 5, "The print frequency.")
flags.DEFINE_integer("item_dim", 50, "The size of item embedding dimension.")
flags.DEFINE_integer("user_dim", 50, "The size of user embedding dimension.")
flags.DEFINE_float("embed_reg", 0.0, "The value of embedding regularization.")
flags.DEFINE_integer("blocks", 2, "The Number of blocks.")
flags.DEFINE_integer("num_heads", 1, "The Number of attention heads.")
flags.DEFINE_integer("ffn_hidden_unit", 64, "Number of hidden unit in FFN.")
flags.DEFINE_float("dnn_dropout", 0.2, "Float between 0 and 1. Dropout of user and item MLP layer.")
flags.DEFINE_float("layer_norm_eps", 1e-6, "Small float added to variance to avoid dividing by zero.")
flags.DEFINE_boolean("use_l2norm", False, "Whether user embedding, item embedding should be normalized or not.")
flags.DEFINE_string("loss_name", "binary_cross_entropy_loss", "Loss Name.")
flags.DEFINE_float("gamma", 0.5, "If hinge_loss is selected as the loss function, you can specify the margin.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("neg_num", 4, "The number of negative sample for each positive sample.")
flags.DEFINE_integer("seq_len", 200, "The length of user's behavior sequence.")
flags.DEFINE_integer("epochs", 20, "train steps.")
flags.DEFINE_integer("batch_size", 128, "Batch Size.")
flags.DEFINE_integer("test_neg_num", 100, "The number of test negative samples.")
flags.DEFINE_integer("k", 10, "recall k items at test stage.")
flags.DEFINE_integer("seed", None, "random seed.")

FLAGS = flags.FLAGS
def main(argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    # 1. Split Data
    dataset = data_partition(FLAGS.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, user_bucket_train, user_bucket_valid, user_bucket_test, user_bin_train, user_bin_valid, user_bin_test] = dataset
    # TODO:如果是对数据集扩充，以下代码需要修改
    num_batch = len(user_train) // FLAGS.batch_size
    cc = 0.0
    max_len = 0
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(max_len, len(user_train[u]))
    print("\nThere are {0} users {1} items \n".format(usernum, itemnum))
    print("Average sequence length: {0}\n".format(cc / len(user_train)))
    print("Maximum length of sequence: {0}\n".format(max_len))
    # 2. Load Sequence Data
    sampler = WarpSampler(user_train, usernum, itemnum, user_bucket_train, user_bin_train,
                          batch_size=FLAGS.batch_size, maxlen=FLAGS.seq_len,
                          n_workers=3)
    # 3. Set Model Hyper Parameters.
    model_params = {
        'item_num': itemnum + 1,
        'user_num': usernum + 1,
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
    # 4. Build Model
    model = SASRec(**model_params)
    model.compile(optimizer=Adam(learning_rate=FLAGS.learning_rate))
    optimizer = tf.keras.optimizers.Adam()
    # 5. Fit Model
    try:
        one_train_batch = sampler.next_batch()
        results = []
        t1 = time()
        step_loss = 0.0
        for epoch in range(1, FLAGS.epochs + 1):
            for step in range(num_batch):

                with tf.GradientTape() as tape:
                    predictions = model(inputs=one_train_batch, training=True)
                    step_loss = get_loss_with_istarget(predictions[0], predictions[1], predictions[2])
                # 更新梯度
                gradients = tape.gradient(step_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                one_train_batch = sampler.next_batch()

                '''
                weights = model.get_embedding_weights()  # (3417, item_dim)
                norm_weights = weights / tf.linalg.norm(weights, axis=1, keepdims=True)
                cosine = tf.matmul(norm_weights, norm_weights, transpose_b=True)
                # 对每一行进行排序
                sorted_values, sorted_indices = tf.math.top_k(cosine, k=5, sorted=True)
                item_map = sorted_indices[:, -1]  # 取最相似的item
                item_map = tf.concat([tf.constant([0]), item_map[1:]], axis=0)
                random_values = tf.random.uniform(
                    tf.shape(one_train_batch['click_seq']),
                    minval=0.0,
                    maxval=1.0
                )  # (None, seq_len)
                transfer_inputs = tf.gather(item_map, one_train_batch['click_seq'])
                one_train_batch['click_seq'] = tf.where(random_values < 0.3, transfer_inputs,
                                                        one_train_batch['click_seq'])
                '''
            print(f"Epoch {epoch}/{FLAGS.epochs}, Loss: {step_loss}")
            t2 = time()
            if epoch % FLAGS.print_freq == 0:
                test_result = evaluate(model, dataset, FLAGS.seq_len, FLAGS.k)
                print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR_10 = %.4f, MRR@10 = %.4f, NDCG@10 = %.4f'
                      % (epoch, t2 - t1, time() - t2, test_result[2], test_result[1], test_result[0]))
                results.append([epoch, t2 - t1, time() - t2, test_result[2], test_result[1], test_result[0]])
        # write logs
        pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hr@10', 'mrr@10', 'ndcg@10']).\
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


def origin_sample_function(user2seq, neg_num, max_item_num, result_queue):
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
    def __init__(self, User, usernum, itemnum, user_bucket_train, user_bin_train, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      user_bucket_train,
                                                      user_bin_train,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def sample_function(user_train, user_bucket_train, user_bin_train, usernum, itemnum, batch_size, maxlen,
                    result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = [0] * maxlen
        pos = [0] * maxlen
        neg = [0] * maxlen
        personality_id = [0] * maxlen
        popularity_id = [0] * maxlen
        pos_personality_id = [0] * maxlen
        nxt = user_train[user][-1]
        pos_nxt = user_bucket_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        # 先翻转
        reverse_user_train = reversed(user_train[user][:-1])
        reverse_user_bucket_train = reversed(user_bucket_train[user][:-1])
        reverse_user_bin_train = reversed(user_bin_train[user][:-1])
        for i, bucket, bin in zip(reverse_user_train, reverse_user_bucket_train, reverse_user_bin_train):
            seq[idx] = i
            personality_id[idx] = bucket
            popularity_id[idx] = bin
            pos[idx] = nxt
            pos_personality_id[idx] = pos_nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            pos_nxt = bucket
            idx -= 1
            if idx == -1: break

        # equivalent to hard parameter sharing
        # user = 1

        return [[user], seq, pos, neg, personality_id, popularity_id, pos_personality_id]

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        user_col, seq_col, pos_col, neg_col, personality_col, popularity_col, pos_personality_col = zip(*one_batch)
        one_batch_input = {
            'user': np.array(user_col, dtype=np.int32),
            'click_seq': np.array(seq_col, dtype=np.int32),
            'pos_item': np.array(pos_col, dtype=np.int32),
            'neg_item': np.array(neg_col, dtype=np.int32),
            'personality_id': np.array(personality_col, dtype=np.int32),
            'popularity_id': np.array(popularity_col, dtype=np.int32),
            'pos_personality_id': np.array(pos_personality_col, dtype=np.int32)
        }
        result_queue.put(one_batch_input)


def sample_function_0(user_train, usernum, itemnum, batch_size, maxlen,
                    result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = [0] * maxlen
        pos = [0] * maxlen
        neg = [0] * maxlen
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])

        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        # equivalent to hard parameter sharing
        # user = 1

        return [[user], seq, pos, neg]

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        user_col, seq_col, pos_col, neg_col = zip(*one_batch)
        one_batch_input = {
            'user': np.array(user_col, dtype=np.int32),
            'click_seq': np.array(seq_col, dtype=np.int32),
            'pos_item': np.array(pos_col, dtype=np.int32),
            'neg_item': np.array(neg_col, dtype=np.int32)
        }
        result_queue.put(one_batch_input)


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

if __name__ == '__main__':
    app.run(main)