import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import math
from multiprocessing import Pool

MAX_ITEM_NUM = 3953
MAX_USER_NUM = 6041


# general recommendation
def split_data(file_path):
    """split movielens for general recommendation
        Args:
            :param file_path: A string. The file path of 'ratings.dat'.
        :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "ml_train.txt")
    val_path = os.path.join(dst_path, "ml_val.txt")
    test_path = os.path.join(dst_path, "ml_test.txt")
    meta_path = os.path.join(dst_path, "ml_meta.txt")
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
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist = history[int(user)]
            hist.sort(key=lambda x: x[1])
            for idx, value in enumerate(hist):
                if idx == len(hist) - 1:
                    f3.write(str(user) + '\t' + value[0] + '\n')
                elif idx == len(hist) - 2:
                    f2.write(str(user) + '\t' + value[0] + '\n')
                else:
                    f1.write(str(user) + '\t' + value[0] + '\n')
    with open(meta_path, 'w') as f:
        f.write(str(max(users)) + '\t' + str(max(items)))
    return train_path, val_path, test_path, meta_path


# sequence recommendation
def split_seq_data(file_path):
    """split movielens for sequence recommendation
    Args:
        :param file_path: A string. The file path of 'ratings.dat'.
    :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "ml_seq_train.txt")
    val_path = os.path.join(dst_path, "ml_seq_val.txt")
    test_path = os.path.join(dst_path, "ml_seq_test.txt")
    meta_path = os.path.join(dst_path, "ml_seq_meta.txt")
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
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user, hist in history.items():
            hist_u = history[int(user)]
            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]
            f1.write(str(user) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + '\n')
            f2.write(str(user) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + "\t" + hist[-2] + '\n')
            f3.write(str(user) + "\t" + ' '.join(hist[:-1]) + "\t" + ' '.join(time[:-1]) + "\t" + hist[-1] + '\n')
    with open(meta_path, 'w') as f:
        f.write(str(max(users)) + '\t' + str(max(items)))
    return train_path, val_path, test_path, meta_path


def load_data(file_path, neg_num, max_item_num):
    """load movielens dataset.
    Args:
        :param file_path: A string. The file path.
        :param neg_num: A scalar(int). The negative num of one sample.
        :param max_item_num: A scalar(int). The max index of item.
    :return: A dict. data.
    """
    data = np.array(pd.read_csv(file_path, delimiter='\t'))
    np.random.shuffle(data)
    neg_items = []
    for i in tqdm(range(len(data))):
        neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
        neg_items.append(neg_item)
    return {'user': data[:, 0].astype(int), 'pos_item': data[:, 1].astype(int), 'neg_item': np.array(neg_items)}


def load_user2seq(file_path):
    usernum = 0
    itemnum = 0
    user2seq = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            u, i = line.strip().split(' ')
            u = int(u)
            i = int(i)
            user2seq[u].append(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
        print('usernum: ', usernum)
        print('itemnum: ', itemnum)
    return user2seq


def process_user_data(user, user_seq, mode, seq_len, neg_num, max_item_num, user_bucket, user_bin):
    users, click_seqs, pos_items, neg_items = [], [], [], []
    user_buckets = []
    user_bins = []
    if len(user_seq[user][:-1]) >= seq_len:
        if mode == 'train':
            generate_seq_count = len(user_seq[user][:-1]) - seq_len + 1
            for n in range(generate_seq_count):
                tmp = user_seq[user][n:n + seq_len]
                tmp_bucket = user_bucket[n:n + seq_len]
                tmp_bin = user_bin[n:n + seq_len]

                neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
                users.append([user])
                click_seqs.append(tmp)
                pos_items.append(user_seq[user][n + seq_len])
                neg_items.append(neg_item)
                user_buckets.append(tmp_bucket)
                user_bins.append(tmp_bin)
        else:
            tmp = user_seq[user][:-1][len(user_seq[user][:-1]) - seq_len:]
            tmp_bucket = user_bucket[:-1][len(user_bucket[:-1]) - seq_len:]
            tmp_bin = user_bin[:-1][len(user_bin[:-1]) - seq_len:]

            neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
            users.append([user])
            click_seqs.append(tmp)
            pos_items.append(user_seq[user][-1])
            neg_items.append(neg_item)
            user_buckets.append(tmp_bucket)
            user_bins.append(tmp_bin)
    else:
        tmp = [0] * (seq_len - len(user_seq[user][:-1])) + user_seq[user][:-1]
        tmp_bucket = [0] * (seq_len - len(user_bucket[:-1])) + user_bucket[:-1]
        tmp_bin = [0] * (seq_len - len(user_bin[:-1])) + user_bin[:-1]

        neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
        users.append([user])
        click_seqs.append(tmp)
        pos_items.append(user_seq[user][-1])
        neg_items.append(neg_item)
        user_buckets.append(tmp_bucket)
        user_bins.append(tmp_bin)

    return users, click_seqs, pos_items, neg_items, user_buckets, user_bins


def load_txt_data(file_path, mode, seq_len, neg_num, max_item_num, max_user_num):
    users, click_seqs, pos_items, neg_items = [], [], [], []
    all_tf_idfs, bucket_ids = [], []
    num_bins = 5
    argue_popularity, argue_bucket_ids = [], []
    argue_bins = 5
    user2seq = defaultdict(list)
    item2user = defaultdict(set)
    user_tf_idf = defaultdict(list)
    user_popularity = defaultdict(list)
    user_seq = {}
    user2bucket = {}
    user2bin = {}
    user_bucket = {}
    user_bin = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            u, i = line.strip().split(' ')
            u = int(u)
            i = int(i)
            user2seq[u].append(i)
            item2user[i].add(u)
    for u in tqdm(user2seq):
        tf_idf, popularity = [], []
        counter = Counter(user2seq[u])
        for i in user2seq[u]:
            temp_tf_idf = counter[i] / len(user2seq[u]) * math.log(max_user_num / len(item2user[i]))
            tf_idf.append(temp_tf_idf)
            temp_popularity = counter[i] / len(user2seq[u]) * math.exp(len(item2user[i]) / max_user_num)
            popularity.append(temp_popularity)
        all_tf_idfs.extend(tf_idf)
        argue_popularity.extend(popularity)
        user_tf_idf[u] = tf_idf
        user_popularity[u] = popularity
    quantiles = np.percentile(all_tf_idfs, np.linspace(0, 100, num_bins + 1))
    argue_quantiles = np.percentile(argue_popularity, np.linspace(0, 100, argue_bins + 1))
    for u in tqdm(user2seq):
        user2bucket[u] = np.digitize(user_tf_idf[u], quantiles, right=True).tolist()
        user2bin[u] = np.digitize(user_popularity[u], argue_quantiles, right=True).tolist()
    for user in tqdm(user2seq):
        nfeedback = len(user2seq[user])
        if nfeedback > 3:
            if mode == 'train':
                user_seq[user] = user2seq[user][:-2]
                user_bucket[user] = user2bucket[user][:-2]
                user_bin[user] = user2bin[user][:-2]
            elif mode == 'val':
                user_seq[user] = user2seq[user][:-1]
                user_bucket[user] = user2bucket[user][:-1]
                user_bin[user] = user2bin[user][:-1]
            else:
                user_seq[user] = user2seq[user]
                user_bucket[user] = user2bucket[user]
                user_bin[user] = user2bin[user]

    with Pool() as pool:
        results = pool.starmap(
            process_user_data,
            [(user, user_seq, mode, seq_len, neg_num, max_item_num, user_bucket[user], user_bin[user]) for user in user_seq]
        )

    # 汇总结果
    for result in tqdm(results):
        users.extend(result[0])
        click_seqs.extend(result[1])
        pos_items.extend(result[2])
        neg_items.extend(result[3])
        bucket_ids.extend(result[4])
        argue_bucket_ids.extend(result[5])
    data = list(zip(users, click_seqs, pos_items, neg_items, bucket_ids, argue_bucket_ids))
    random.shuffle(data)
    users, click_seqs, pos_items, neg_items, bucket_ids, argue_bucket_ids = zip(*data)
    data = {'user': np.array(users, dtype=np.int32),
            'click_seq': np.array(click_seqs, dtype=np.int32),
            'pos_item': np.array(pos_items, dtype=np.int32),
            'neg_item': np.array(neg_items, dtype=np.int32),
            'bucket_id': np.array(bucket_ids, dtype=np.int32),
            'argue_bucket_id': np.array(argue_bucket_ids, dtype=np.int32)
            }
    return data


def load_seq_data(file_path, mode, seq_len, neg_num, max_item_num, contain_user=True, contain_time=False):
    """load sequence movielens dataset.
    Args:
        :param file_path: A string. The file path.
        :param mode: A string. "train", "val" or "test".
        :param seq_len: A scalar(int). The length of sequence.
        :param neg_num: A scalar(int). The negative num of one sample.
        :param max_item_num: A scalar(int). The max index of item.
        :param contain_user: A boolean. Whether including user'id input or not.
        :param contain_time: A boolean. Whether including time sequence input or not.
    :return: A dict. data.
    """
    movies_genres = {}
    max_genre_num = 6
    with open('data/ml-1m/movies_genres.json', 'r', encoding='utf-8') as json_f:
        movies_genres = json.load(json_f)
    users, click_seqs, time_seqs, pos_items, neg_items, genre_index_seqs = [], [], [], [], [], []
    with open(file_path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if mode == "train":
                user, click_seq, time_seq = line.split('\t')
                click_seq = click_seq.split(' ')
                click_seq = [int(x) for x in click_seq]
                time_seq = time_seq.split(' ')
                for i in range(len(click_seq)-1):
                    if i + 1 >= seq_len:
                        tmp = click_seq[i+1-seq_len:i+1]
                        tmp2 = time_seq[i + 1 - seq_len:i + 1]
                    else:
                        tmp = [0] * (seq_len-i-1) + click_seq[:i+1]
                        tmp2 = [0] * (seq_len - i - 1) + time_seq[:i + 1]
                    # gen_neg = _gen_negative_samples(neg_num, click_seq, max_item_num)
                    # neg_item = [neg_item for neg_item in gen_neg]
                    # neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
                    '''
                    genre_index_seq = []
                    for item in tmp:
                        origin_genre = [18] * max_genre_num
                        if str(item) in movies_genres:
                            origin_genre[:len(movies_genres[str(item)])] = movies_genres[str(item)]
                        genre_index_seq.append(origin_genre)
                    '''
                    neg_item = gen_negative_samples_except_pos(neg_num, click_seq, max_item_num)
                    users.append([int(user)])
                    click_seqs.append(tmp)
                    time_seqs.append(tmp2)
                    pos_items.append(click_seq[i + 1])
                    neg_items.append(neg_item)
                    '''
                    genre_index_seqs.append(genre_index_seq)
                    '''
            else:  # "val", "test"
                user, click_seq, time_seq, pos_item = line.split('\t')
                click_seq = click_seq.split(' ')
                click_seq = [int(x) for x in click_seq]
                time_seq = time_seq.split(' ')
                if len(click_seq) >= seq_len:
                    tmp = click_seq[len(click_seq) - seq_len:]
                    tmp2 = time_seq[len(time_seq) - seq_len:]
                else:
                    tmp = [0] * (seq_len - len(click_seq)) + click_seq[:]
                    tmp2 = [0] * (seq_len - len(time_seq)) + time_seq[:]
                # gen_neg = _gen_negative_samples(neg_num, click_seq, max_item_num)
                # neg_item = [neg_item for neg_item in gen_neg]
                # neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
                '''
                genre_index_seq = []
                for item in tmp:
                    origin_genre = [18] * max_genre_num
                    if str(item) in movies_genres:
                        origin_genre[:len(movies_genres[str(item)])] = movies_genres[str(item)]
                    genre_index_seq.append(origin_genre)
                '''
                neg_item = gen_negative_samples_except_pos(neg_num, click_seq, max_item_num)
                users.append([int(user)])
                click_seqs.append(tmp)
                time_seqs.append(tmp2)
                pos_items.append(int(pos_item))
                neg_items.append(neg_item)
                '''
                genre_index_seqs.append(genre_index_seq)
                '''
    data = list(zip(users, click_seqs, time_seqs, pos_items, neg_items))
    random.shuffle(data)
    users, click_seqs, time_seqs, pos_items, neg_items = zip(*data)
    #       , 'genre_index_seq': np.array(genre_index_seqs)
    data = {'click_seq': np.array(click_seqs), 'pos_item': np.array(pos_items), 'neg_item': np.array(neg_items)}
    if contain_user:
        data['user'] = np.array(users)
    if contain_time:
        data['time_seq'] = np.array(time_seqs)
    return data


def _gen_negative_samples(neg_num, item_list, max_num):
    for i in range(neg_num):
        # neg = item_list[0]
        # while neg in set(item_list):
        neg = random.randint(1, max_num)
        yield neg


def gen_negative_samples_except_pos(neg_num, click_seq, max_num):
    neg_item = []
    while len(neg_item) != neg_num:
        neg = random.randint(1, max_num)
        while neg in click_seq:
            neg = random.randint(1, max_num)
        neg_item.append(neg)
    return neg_item


def gen_negative_samples_include_pos(neg_num, max_num):
    neg_item = []
    while len(neg_item) != neg_num:
        neg = random.randint(1, max_num)
        neg_item.append(neg)
    return neg_item


"""
def generate_movielens(file_path, neg_num):
    with open(file_path, 'r') as f:
        for line in f:
            user, pos_item = line.split('\t')
            neg_item = [random.randint(1, MAX_ITEM_NUM) for _ in range(neg_num)]
            yield int(user), int(pos_item), neg_item
"""

"""
def generate_ml(file_path, neg_num):
    return tf.data.Dataset.from_generator(
        generator=generate_movielens,
        args=[file_path, neg_num],
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(neg_num,), dtype=tf.int32),
        )
    )
"""


def generate_seq_data(file_path, seq_len, neg_num):
    with open(file_path, 'r') as f:
        user, hist = f.readline().split('\t')
        hist = hist.split(',')
        hist = [int(item) for item in hist]
        hist_len = len(hist)
        if hist_len < seq_len:
            hist = [0] * (seq_len - hist_len) + hist
            gen_neg = _gen_negative_samples(neg_num, hist, MAX_ITEM_NUM)
            neg_hist = [neg_item for neg_item in gen_neg]
            mask_hist = [0] * (seq_len - hist_len) + [1] * hist_len
        else:
            hist = hist[hist_len - seq_len:]
            gen_neg = _gen_negative_samples(neg_num, hist, MAX_ITEM_NUM)
            neg_hist = [neg_item for neg_item in gen_neg]
            mask_hist = [1] * seq_len
        yield {'user': int(user), 'hist': hist, 'neg_list': neg_hist, 'mask_hist': mask_hist}


def create_ml_1m_dataset(file, trans_score=2, test_neg_num=100):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()
        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num - 1)]
        for i in range(1, len(pos_list)):
            if i == len(pos_list) - 1:
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
            else:
                train_data['user_id'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = {'user': np.array(train_data['user_id']),
             'pos_item': np.array(train_data['pos_id']),
             'neg_item': np.array(train_data['neg_id']).reshape((-1, 1))}
    val = {'user': np.array(val_data['user_id']),
           'pos_item': np.array(val_data['pos_id']),
           'neg_item': np.array(val_data['neg_id']).reshape((-1, 1))}
    test = {'user': np.array(test_data['user_id']),
            'pos_item': np.array(test_data['pos_id']),
            'neg_item': np.array(test_data['neg_id'])}
    print('============Data Preprocess End=============')
    return train, val, test


def create_seq_ml_1m_dataset(file, trans_score=1, seq_len=40, test_neg_num=100):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param seq_len: A scalar. maxlen.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data['hist'].append(hist_i)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['hist'].append(hist_i)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append([neg_list[i]])
            else:
                train_data['hist'].append(hist_i)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append([neg_list[i]])
    # item feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    # padding
    print('==================Padding===================')
    train = {'click_seq': pad_sequences(train_data['hist'], maxlen=seq_len),
             'pos_item': np.array(train_data['pos_id']),
             'neg_item': np.array(train_data['neg_id'])}
    val = {'click_seq': pad_sequences(val_data['hist'], maxlen=seq_len),
           'pos_item': np.array(val_data['pos_id']),
           'neg_item': np.array(val_data['neg_id'])}
    test = {'click_seq': pad_sequences(test_data['hist'], maxlen=seq_len),
            'pos_item': np.array(test_data['pos_id']),
             'neg_item': np.array(test_data['neg_id'])}
    print('============Data Preprocess End=============')
    return user_num, item_num, train, val, test