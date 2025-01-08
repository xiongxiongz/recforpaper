import json
import random

import numpy as np
from tqdm import tqdm
import pickle
import os
import tensorflow as tf


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

def generate_movie_genre_json():
    movies_file = "/home/cqj/zzh/recforpaper/data/ml-1m/movies.dat"
    genres_index_pickle = "/home/cqj/zzh/recforpaper/data/ml-1m/genres_index.pkl"
    movies_genres_json = "/home/cqj/zzh/recforpaper/data/ml-1m/movies_genres.json"
    unique_genres = {}
    index = 0
    with open(movies_file, 'r', encoding='iso-8859-1') as f1:
        lines = f1.readlines()
        for line in tqdm(lines):
            item, item_name, genres = line.strip().split('::')
            genres_list = genres.strip().split('|')
            for gen in genres_list:
                if gen not in unique_genres:
                    unique_genres[gen] = index
                    index += 1
    print("unique_genres:", unique_genres)
    with open(genres_index_pickle, 'wb') as f2:
        pickle.dump(unique_genres, f2)
    item2genres_index = {}
    with open(movies_file, 'r', encoding='iso-8859-1') as f3:
        lines = f3.readlines()
        for line in tqdm(lines):
            item, item_name, genres = line.strip().split('::')
            genres_list = genres.strip().split('|')
            index_list = []
            for gen in genres_list:
                index_list.append(unique_genres[gen])
            item2genres_index[int(item)] = index_list
    with open(movies_genres_json, 'w', encoding='utf-8') as f4:
        json.dump(item2genres_index, f4, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    self.depthwise1 = DepthwiseConv2D(
        kernel_size=(3, 1),
        depth_multiplier=1,  # 每个通道独立卷积
        padding='same',
        use_bias=False
    )
    self.point_conv1 = Conv1D(filters=8, kernel_size=1, padding='same', activation='relu')
    self.recover_dense1 = Dense(units=32)
    self.depthwise2 = DepthwiseConv2D(
        kernel_size=(5, 1),
        depth_multiplier=1,  # 每个通道独立卷积
        padding='same',
        use_bias=False
    )
    self.point_conv2 = Conv1D(filters=8, kernel_size=1, padding='same', activation='relu')
    self.recover_dense2 = Dense(units=32)
    self.depthwise3 = DepthwiseConv2D(
        kernel_size=(7, 1),
        depth_multiplier=1,  # 每个通道独立卷积
        padding='same',
        use_bias=False
    )
    self.point_conv3 = Conv1D(filters=8, kernel_size=1, padding='same', activation='relu')
    self.recover_dense3 = Dense(units=32)
    self.depthwise4 = DepthwiseConv2D(
        kernel_size=(11, 1),
        depth_multiplier=1,  # 每个通道独立卷积
        padding='same',
        use_bias=False
    )
    self.point_conv4 = Conv1D(filters=8, kernel_size=1, padding='same', activation='relu')
    self.recover_dense4 = Dense(units=32)

    dimension_per_head = self.d_model // 4
    par_q_1 = q[..., dimension_per_head * 0: dimension_per_head * 1]
    par_k_1 = k[..., dimension_per_head * 0: dimension_per_head * 1]
    par_v_1 = v[..., dimension_per_head * 0: dimension_per_head * 1]
    scaled_attention_1_st = scaled_dot_product_attention(par_q_1, par_k_1, par_v_1,
                                                         mask)  # (None, num_heads, seq_len, d_model // num_heads)
    scaled_attention_1 = scaled_attention_1_st
    scaled_attention_1 = tf.expand_dims(scaled_attention_1, axis=-1)
    scaled_attention_1 = self.depthwise1(scaled_attention_1)
    scaled_attention_1 = tf.squeeze(scaled_attention_1, axis=-1)
    scaled_attention_1 = self.point_conv1(scaled_attention_1)
    scaled_attention_1 = self.recover_dense1(scaled_attention_1)

    par_q_2 = q[..., dimension_per_head * 1: dimension_per_head * 2]
    par_k_2 = k[..., dimension_per_head * 1: dimension_per_head * 2]
    par_v_2 = v[..., dimension_per_head * 1: dimension_per_head * 2]
    scaled_attention_2_st = scaled_dot_product_attention(par_q_2, par_k_2, par_v_2,
                                                         mask)  # (None, num_heads, seq_len, d_model // num_heads)
    scaled_attention_2 = scaled_attention_2_st
    scaled_attention_2 = tf.expand_dims(scaled_attention_2, axis=-1)
    scaled_attention_2 = self.depthwise2(scaled_attention_2)
    scaled_attention_2 = tf.squeeze(scaled_attention_2, axis=-1)
    scaled_attention_2 = self.point_conv2(scaled_attention_2)
    scaled_attention_2 = self.recover_dense2(scaled_attention_2)

    par_q_3 = q[..., dimension_per_head * 2: dimension_per_head * 3]
    par_k_3 = k[..., dimension_per_head * 2: dimension_per_head * 3]
    par_v_3 = v[..., dimension_per_head * 2: dimension_per_head * 3]
    scaled_attention_3_st = scaled_dot_product_attention(par_q_3, par_k_3, par_v_3,
                                                         mask)  # (None, num_heads, seq_len, d_model // num_heads)
    scaled_attention_3 = scaled_attention_3_st
    scaled_attention_3 = tf.expand_dims(scaled_attention_3, axis=-1)
    scaled_attention_3 = self.depthwise3(scaled_attention_3)
    scaled_attention_3 = tf.squeeze(scaled_attention_3, axis=-1)
    scaled_attention_3 = self.point_conv3(scaled_attention_3)
    scaled_attention_3 = self.recover_dense3(scaled_attention_3)

    par_q_4 = q[..., dimension_per_head * 3: dimension_per_head * 4]
    par_k_4 = k[..., dimension_per_head * 3: dimension_per_head * 4]
    par_v_4 = v[..., dimension_per_head * 3: dimension_per_head * 4]
    scaled_attention_4_st = scaled_dot_product_attention(par_q_4, par_k_4, par_v_4,
                                                         mask)  # (None, num_heads, seq_len, d_model // num_heads)
    scaled_attention_4 = scaled_attention_4_st
    scaled_attention_4 = tf.expand_dims(scaled_attention_4, axis=-1)
    scaled_attention_4 = self.depthwise4(scaled_attention_4)
    scaled_attention_4 = tf.squeeze(scaled_attention_4, axis=-1)
    scaled_attention_4 = self.point_conv4(scaled_attention_4)
    scaled_attention_4 = self.recover_dense4(scaled_attention_4)

    # merge
    scaled_attention_1_st = scaled_attention_1_st + scaled_attention_2 + scaled_attention_3 + scaled_attention_4
    scaled_attention_2_st = scaled_attention_2_st + scaled_attention_1 + scaled_attention_3 + scaled_attention_4
    scaled_attention_3_st = scaled_attention_3_st + scaled_attention_1 + scaled_attention_2 + scaled_attention_4
    scaled_attention_4_st = scaled_attention_4_st + scaled_attention_1 + scaled_attention_2 + scaled_attention_3
    par_output = [scaled_attention_1_st, scaled_attention_2_st, scaled_attention_3_st, scaled_attention_4_st]

    outputs = tf.concat(par_output, axis=-1)
    ##########################################################
    self.depthwise1 = DepthwiseConv2D(
        kernel_size=(3, 1),
        depth_multiplier=1,  # 每个通道独立卷积
        padding='same',
        use_bias=False
    )
    self.point_conv1 = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    self.recover_dense1 = Dense(units=d_model)

    self.depthwise2 = DepthwiseConv2D(
        kernel_size=(5, 1),
        depth_multiplier=1,  # 每个通道独立卷积
        padding='same',
        use_bias=False
    )
    self.point_conv2 = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    self.recover_dense2 = Dense(units=d_model)

    attention_encode = scaled_dot_product_attention(q, k, v, mask)
    scaled_attention1 = tf.expand_dims(attention_encode, axis=-1)
    scaled_attention1 = self.depthwise1(scaled_attention1)
    scaled_attention1 = tf.squeeze(scaled_attention1, axis=-1)
    scaled_attention1 = self.point_conv1(scaled_attention1)
    scaled_attention1 = self.recover_dense1(scaled_attention1)

    scaled_attention2 = tf.expand_dims(attention_encode, axis=-1)
    scaled_attention2 = self.depthwise2(scaled_attention2)
    scaled_attention2 = tf.squeeze(scaled_attention2, axis=-1)
    scaled_attention2 = self.point_conv2(scaled_attention2)
    scaled_attention2 = self.recover_dense2(scaled_attention2)
    outputs = attention_encode + scaled_attention1 + scaled_attention2
    #################################################
    self.depthwise_conv = DepthwiseConv2D(
        kernel_size=(1, 1),
        depth_multiplier=1,  # 每个通道独立卷积
        padding='same',
        use_bias=False
    )
    self.point_conv = Conv1D(filters=user_dim // 4, kernel_size=1, padding='same')
    self.dense = Dense(units=user_dim // 4, activation="relu")
    self.user_dropout = Dropout(0.3)
    self.conv = Conv1D(filters=user_dim, kernel_size=1)

    origin_user_encode = self.user_embedding(inputs['user'])  # (None, 1, dim)
    origin_user_embed = tf.expand_dims(origin_user_encode, axis=-1)  # (None, 1, dim, 1)
    origin_user_embed = self.depthwise_conv(origin_user_embed)
    origin_user_embed = tf.squeeze(origin_user_embed, axis=-1)
    origin_user_embed = self.point_conv(origin_user_embed)
    origin_user_embed = self.dense(origin_user_embed)
    origin_user_embed = self.user_dropout(origin_user_embed)
    origin_user_embed = self.conv(origin_user_embed)
    user_embed = tf.tile(origin_user_embed, [1, self.seq_len, 1])  # (None, seq_len, dim)
    seq_embed = tf.concat([seq_embed, user_embed], axis=-1)  # (None, seq_len, item_dim + user_dim)
    #################################################

    self.dense = Dense(units=user_dim // 4, activation="relu")
    self.user_dropout = Dropout(0.3)
    self.conv = Conv1D(filters=user_dim, kernel_size=1)

    origin_user_encode = self.user_embedding(inputs['user'])  # (None, 1, dim)
    origin_user_embed = self.dense(origin_user_encode)
    origin_user_embed = self.user_dropout(origin_user_embed)
    origin_user_embed = self.conv(origin_user_embed)
    user_embed = tf.tile(origin_user_embed, [1, self.seq_len, 1])  # (None, seq_len, dim)
    seq_embed = tf.concat([seq_embed, user_embed], axis=-1)  # (None, seq_len, item_dim + user_dim)
