"""
Created on Nov 14, 2021
Evaluate Functions.
@author: Ziyao Geng(zggzy1996@163.com)
"""
from reclearn.evaluator.metrics import *
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data.utils.data_loader import DataGenerator


def eval_pos_neg(model, test_data, metric_names, max_item_num, k=10, batch_size=None):
    """Evaluate the performance of Top-k recommendation algorithm.
    Note: Test data must contain some negative samples(>= k) and one positive samples.
    Args:
        :param model: A model built-by tensorflow.
        :param test_data: A dict.
        :param metric_names: A list like ['hr'].
        :param k: A scalar(int).
        :param batch_size: A scalar(int).
    :return: A result dict such as {'hr':, 'ndcg':, ...}
    """
    batch = 100
    predict_y = []
    batch_neg_item = [j for j in range(1, max_item_num + 1)]
    item_emb_map = model.get_embedding_weights()
    item_emb_map = tf.constant(item_emb_map)
    batch_item_emb = tf.gather(item_emb_map, batch_neg_item)
    # 扩展第一个维度用于concat
    batch_item_emb = tf.expand_dims(batch_item_emb, axis=0)
    # 生成DataGenerator用于获取batch数据
    test_generator = DataGenerator(test_data, batch)
    # 生成固定batch和最后一个batch的concat_item_emb
    general_batch_len = len(test_generator.__getitem__(0)['pos_item'])
    general_batch_item_emb = get_item_embedding(general_batch_len, batch_item_emb)
    last_batch_len = len(test_generator.__getitem__(test_generator.__len__() - 1)['pos_item'])
    last_batch_item_emb = get_item_embedding(last_batch_len, batch_item_emb)
    for a in tqdm(range(test_generator.__len__())):
        batch_test_data = test_generator.__getitem__(a)
        # 区分一般batch和最后一个batch
        all_item_emb = general_batch_item_emb
        if a == test_generator.__len__() - 1:
            all_item_emb = last_batch_item_emb
        # model.predict返回的是numpy.ndarray
        batch_predict_y = model.predict(batch_test_data, batch_size)
        # 矩阵密集型计算，gpu快于cpu
        batch_predict_y = tf.reduce_sum(tf.multiply(batch_predict_y, all_item_emb), axis=-1)  # (None, max_item_num)
        # 获取正样本的下标
        batch_pos_item_index = batch_test_data['pos_item'] - np.array(1, dtype=np.int32)
        batch_pos_item_index = tf.expand_dims(batch_pos_item_index, axis=-1)  # (None, 1)
        # 排名，获取正样本的排名
        batch_predict_y = - batch_predict_y
        batch_predict_y = tf.argsort(tf.argsort(batch_predict_y, axis=-1, direction='ASCENDING'), axis=-1, direction='ASCENDING')
        batch_predict_y = tf.gather(batch_predict_y, batch_pos_item_index, axis=1, batch_dims=1)  # (None, 1)
        batch_predict_y = batch_predict_y.numpy()
        # GPU显存不足，无法存储，转为ndarray存放在内存，放入列表
        predict_y.append(batch_predict_y)
    predict_y = np.concatenate(predict_y, axis=0)
    # 从(None, 1)->(None)
    predict_y = np.squeeze(predict_y, axis=1)
    return eval_rank(predict_y, metric_names, k)


def eval_rank(rank, metric_names, k=10):
    """Evaluate
        Args:
            :param rank: A ndarray.
            :param metric_names: A list like ['hr'].
            :param k: A scalar(int).
        :return: A result dict such as {'hr':, 'ndcg':, ...}
    """
    print("rank:", rank)
    print("len(rank):", len(rank))
    res_dict = {}
    for name in metric_names:
        if name == 'hr':
            res = hr(rank, k)
        elif name == 'ndcg':
            res = ndcg(rank, k)
        elif name == 'mrr':
            res = mrr(rank, k)
        else:
            break
        res_dict[name] = res[0]
        res_dict[name + '_20'] = res[1]
        res_dict[name + '_40'] = res[2]
        # res_dict[name] = res
    return res_dict


def get_item_embedding(batch_size, batch_item_emb):
    col = []
    for i in range(batch_size):
        col.append(batch_item_emb)
    return tf.concat(col, axis=0)
