import random

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.regularizers import l2
import numpy as np
from tqdm import tqdm
import os


def generate_movie_seq():
    # 处理观影序列
    file_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ratings.dat"
    movie_seq_path = "/home/cqj/zzh/recforpaper/data/ml-1m/movie_seq_val.txt"
    users, items = set(), set()
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            user, item, score, timestamp = line.strip().split("::")
            users.add(int(user))
            items.add(int(item))
            history.setdefault(int(user), [])
            history[int(user)].append([item, timestamp])
        random.shuffle(list(users))
    with open(movie_seq_path, 'w') as f1:
        for user, hist in history.items():
            hist_u = history[int(user)]
            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]
            f1.write(' '.join(hist[:-1]) + '\n')


def generate_movie_vew_count():
    # 处理观影次数
    file_path = "/home/cqj/zzh/recforpaper/data/ml-1m/movie_seq_val.txt"
    movie_view_count_path = "/home/cqj/zzh/recforpaper/data/ml-1m/movie_view_count_test.txt"
    movie_view_count = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            items = line.strip().split(" ")
            for item in items:
                if int(item) not in movie_view_count:
                    movie_view_count[int(item)] = 1
                else:
                    movie_view_count[int(item)] += 1
        with open(movie_view_count_path, 'w') as f1:
            for line in tqdm(lines):
                items = line.strip().split(" ")
                temp = []
                for item in items:
                    temp.append(str(movie_view_count[int(item)]))
                f1.write(' '.join(temp) + '\n')


if __name__ == '__main__':
    # 向量空间：item_embedding
    item_embedding = Embedding(input_dim=3,
                                    input_length=1,
                                    output_dim=2,
                                    embeddings_initializer='random_normal',
                                    embeddings_regularizer=l2(0.0))
    # 向量空间：pos_embedding
    pos_embedding = Embedding(input_dim=3,
                                   input_length=1,
                                   output_dim=2,
                                   embeddings_initializer='random_normal',
                                   embeddings_regularizer=l2(0.0))
    temp = np.array([0, 1, 2])
    temp_emb = tf.expand_dims(pos_embedding(temp), axis=1)
    print("temp_emb:", temp_emb)
    a = tf.fill([3, 3, 4], 3.)
    b = tf.fill([3, 3, 4], 2.)

