"""
Created on Nov 14, 2021
Evaluate Functions.
@author: Ziyao Geng(zggzy1996@163.com)
"""
from reclearn.evaluator.metrics import *
import numpy as np
import copy


def evaluate(model, dataset, maxlen, k=10, batch_size=None):
    [train, valid, test, usernum, itemnum, user_bucket_train, user_bucket_valid, user_bucket_test,
     user_bin_train, user_bin_valid, user_bin_test] = copy.deepcopy(dataset)
    NDCG = 0.0
    MRR = 0.0
    HT = 0.0
    user_list = []
    seq_list = []
    pos_list = []
    neg_list = []
    pos_personality_list = []
    personality_list = []
    popularity_list = []
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = [0] * maxlen
        personality_id = [0] * maxlen
        popularity_id = [0] * maxlen
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        reverse_train = reversed(train[u])
        reverse_personality = reversed(user_bucket_train[u])
        reverse_popularity = reversed(user_bin_train[u])
        for i, personality, popularity in zip(reverse_train, reverse_personality, reverse_popularity):
            seq[idx] = i
            personality_id[idx] = personality
            popularity_id[idx] = popularity
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        pos_item = [test[u][0]]
        neg_item = []
        pos_personality_item = [user_bucket_test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            neg_item.append(t)
        user_list.append([u])
        seq_list.append(seq)
        personality_list.append(personality_id)
        popularity_list.append(popularity_id)
        pos_list.append(pos_item)
        neg_list.append(neg_item)
        pos_personality_list.append(pos_personality_item)
    # create evaluate inputs
    evaluate_inputs = {
        'user': np.array(user_list, dtype=np.int32),
        'click_seq': np.array(seq_list, dtype=np.int32),
        'pos_item': np.array(pos_list, dtype=np.int32),
        'neg_item': np.array(neg_list, dtype=np.int32),
        'personality_id': np.array(personality_list, dtype=np.int32),
        'popularity_id': np.array(popularity_list, dtype=np.int32),
        'pos_personality_id': np.array(pos_personality_list, dtype=np.int32)
    }
    predictions = -model.predict(evaluate_inputs, batch_size)
    print("predictions.argsort().argsort():", predictions.argsort().argsort())
    rank = predictions.argsort().argsort()[:, -1]
    valid_user = len(rank)
    # print("valid_user:", valid_user)
    for r in rank:
        if r < k:
            NDCG += 1 / np.log2(r + 2)
            MRR += 1 / (r + 1)
            HT += 1

    return NDCG / valid_user, MRR / valid_user, HT / valid_user


def evaluate_0(model, dataset, maxlen, k=10, batch_size=None):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    MRR = 0.0
    HT = 0.0
    user_list = []
    seq_list = []
    pos_list = []
    neg_list = []
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = [0] * maxlen
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        pos_item = [test[u][0]]
        neg_item = []
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            neg_item.append(t)
        user_list.append([u])
        seq_list.append(seq)
        pos_list.append(pos_item)
        neg_list.append(neg_item)
    # create evaluate inputs
    evaluate_inputs = {
        'user': np.array(user_list, dtype=np.int32),
        'click_seq': np.array(seq_list, dtype=np.int32),
        'pos_item': np.array(pos_list, dtype=np.int32),
        'neg_item': np.array(neg_list, dtype=np.int32)
    }
    predictions = -model.predict(evaluate_inputs, batch_size)
    print("predictions.argsort().argsort():", predictions.argsort().argsort())
    print("predictions.shape:", predictions.shape)
    rank = predictions.argsort().argsort()[:, -1]
    valid_user = len(rank)
    print("valid_user:", valid_user)
    for r in rank:
        if r < k:
            NDCG += 1 / np.log2(r + 2)
            MRR += 1 / (r + 1)
            HT += 1

    return NDCG / valid_user, MRR / valid_user, HT / valid_user

def eval_pos_neg(model, test_data, metric_names, k=10, batch_size=None):
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
    pred_y = - model.predict(test_data, batch_size)
    print("pred_y.argsort().argsort():", pred_y.argsort().argsort())
    return eval_rank(pred_y, metric_names, k)


def eval_rank(pred_y, metric_names, k=10):
    """Evaluate
        Args:
            :param pred_y: A ndarray.
            :param metric_names: A list like ['hr'].
            :param k: A scalar(int).
        :return: A result dict such as {'hr':, 'ndcg':, ...}
    """
    rank = pred_y.argsort().argsort()[:, -1]
    print("rank:", rank)
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