"""
Created on Nov 14, 2021
@author: Ziyao Geng(zggzy1996@163.com)
"""
import numpy as np


def hr(rank, k):
    """Hit Rate.
    Args:
        :param rank: A list.
        :param k: A scalar(int).
    :return: hit rate.
    """
    res = 0.0
    res_20 = 0.0
    res_40 = 0.0
    for r in rank:
        if r < k:
            res += 1
        if r < k * 2:
            res_20 += 1
        if r < k * 4:
            res_40 += 1
    print("res:", res)
    print("len(rank):", len(rank))
    print("res / len(rank):", res / len(rank))
    return [res / len(rank), res_20 / len(rank), res_40 / len(rank)]


def mrr(rank, k):
    """Mean Reciprocal Rank.
    Args:
        :param rank: A list.
        :param k: A scalar(int).
    :return: mrr.
    """
    mrr = 0.0
    mrr_20 = 0.0
    mrr_40 = 0.0
    for r in rank:
        if r < k:
            mrr += 1 / (r + 1)
        if r < k * 2:
            mrr_20 += 1 / (r + 1)
        if r < k * 4:
            mrr_40 += 1 / (r + 1)
    return [mrr / len(rank), mrr_20 / len(rank), mrr_40 / len(rank)]


def ndcg(rank, k):
    """Normalized Discounted Cumulative Gain.
    Args:
        :param rank: A list.
        :param k: A scalar(int).
    :return: ndcg.
    """
    res = 0.0
    res_20 = 0.0
    res_40 = 0.0
    for r in rank:
        if r < k:
            res += 1 / np.log2(r + 2)
        if r < k * 2:
            res_20 += 1 / np.log2(r + 2)
        if r < k * 4:
            res_40 += 1 / np.log2(r + 2)
    return [res / len(rank), res_20 / len(rank), res_40 / len(rank)]
