import math
from collections import defaultdict, Counter
from tqdm import tqdm
import tensorflow as tf
import numpy as np

def process_user_data(user, user_seq, mode, seq_len, neg_num, max_item_num):
    users, click_seqs, pos_items, neg_items = [], [], [], []
    if mode == 'train':
        # 除了第一个item，其他的item都可以作为正样本
        for i in range(len(user_seq[user]) - 1):
            if i + 1 >= seq_len:
                tmp = user_seq[user][i + 1 - seq_len:i + 1]
            else:
                tmp = [0] * (seq_len - i - 1) + user_seq[user][:i + 1]
            neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
            users.append([user])
            click_seqs.append(tmp)
            pos_items.append(user_seq[user][i + 1])
            neg_items.append(neg_item)
    else:
        if len(user_seq[user][:-1]) >= seq_len:
            tmp = user_seq[user][:-1][len(user_seq[user][:-1]) - seq_len:]
        else:
            tmp = [0] * (seq_len - len(user_seq[user][:-1])) + user_seq[user][:-1]
        neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
        users.append([user])
        click_seqs.append(tmp)
        pos_items.append(user_seq[user][-1])
        neg_items.append(neg_item)

    return users, click_seqs, pos_items, neg_items

def load_txt_data(file_path, mode, seq_len, neg_num, max_item_num):
    users, click_seqs, pos_items, neg_items = [], [], [], []
    usernum = 0
    itemnum = 0
    user2seq = defaultdict(list)
    user_seq = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            u, i = line.strip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user2seq[u].append(i)
        print('usernum: ', usernum)
        print('itemnum: ', itemnum)
    for user in tqdm(user2seq):
        nfeedback = len(user2seq[user])
        if nfeedback > 3:
            if mode == 'train':
                user_seq[user] = user2seq[user][:-2]
            elif mode == 'val':
                user_seq[user] = user2seq[user][:-1]
            else:
                user_seq[user] = user2seq[user]
    with Pool() as pool:
        results = pool.starmap(
            process_user_data,
            [(user, user_seq, mode, seq_len, neg_num, max_item_num) for user in user_seq]
        )

    # 汇总结果
    for result in tqdm(results):
        users.extend(result[0])
        click_seqs.extend(result[1])
        pos_items.extend(result[2])
        neg_items.extend(result[3])
    # for user in tqdm(user_seq):
    #     if mode == 'train':
    #         # 除了第一个item，其他的item都可以作为正样本
    #         for i in range(len(user_seq[user])-1):
    #             if i + 1 >= seq_len:
    #                 tmp = user_seq[user][i + 1 - seq_len:i + 1]
    #             else:
    #                 tmp = [0] * (seq_len - i - 1) + user_seq[user][:i + 1]
    #             neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
    #             users.append([user])
    #             click_seqs.append(tmp)
    #             pos_items.append(user_seq[user][i + 1])
    #             neg_items.append(neg_item)
    #         '''
    #         if len(user_seq[user][:-1]) >= seq_len:
    #             for i in range(seq_len-1, len(user_seq[user][:-1])):
    #                 tmp = user_seq[user][i+1 - seq_len:i+1]
    #                 neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
    #                 users.append([user])
    #                 click_seqs.append(tmp)
    #                 pos_items.append(user_seq[user][i+1])
    #                 neg_items.append(neg_item)
    #         else:
    #             tmp = [0] * (seq_len - len(user_seq[user][:-1])) + user_seq[user][:-1]
    #             neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
    #             users.append([user])
    #             click_seqs.append(tmp)
    #             pos_items.append(user_seq[user][-1])
    #             neg_items.append(neg_item)
    #         '''
    #     else:
    #         if len(user_seq[user][:-1]) >= seq_len:
    #             tmp = user_seq[user][:-1][len(user_seq[user][:-1]) - seq_len:]
    #         else:
    #             tmp = [0] * (seq_len - len(user_seq[user][:-1])) + user_seq[user][:-1]
    #         neg_item = gen_negative_samples_except_pos(neg_num, user_seq[user], max_item_num)
    #         users.append([user])
    #         click_seqs.append(tmp)
    #         pos_items.append(user_seq[user][-1])
    #         neg_items.append(neg_item)
    data = list(zip(users, click_seqs, pos_items, neg_items))
    random.shuffle(data)
    users, click_seqs, pos_items, neg_items = zip(*data)
    data = {'user': np.array(users),
            'click_seq': np.array(click_seqs),
            'pos_item': np.array(pos_items),
            'neg_item': np.array(neg_items)
            }
    return data

def cal_tf_idf(file_path):
    usernum = 0
    itemnum = 0
    user2seq = defaultdict(list)
    item2user = defaultdict(set)
    user_tf_idf = defaultdict(list)
    user_avg_tf_idf = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            u, i = line.strip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user2seq[u].append(i)
            item2user[i].add(u)
        print('usernum: ', usernum)
        print('itemnum: ', itemnum)
    for u in user2seq:
        tf_idf = []
        sum_tf_idf = 0.0
        counter = Counter(user2seq[u])
        for i in counter:
            temp_tf_idf = counter[i] / len(user2seq[u]) * math.log(usernum / len(item2user[i]))
            tf_idf.append(temp_tf_idf)
            sum_tf_idf += temp_tf_idf
        user_tf_idf[u] = tf_idf
        print(user_tf_idf[u])
        user_avg_tf_idf[u] = sum_tf_idf / len(user2seq[u])
        print(user_avg_tf_idf[u])


if __name__ == '__main__':
    '''区分训练和推理，推理时不需要mask
    if training:
        last_one = tf.slice(mask, begin=[0, self.seq_len-1, 0], size=[-1, 1, -1])  # (None, 1, 1)
        random_values = tf.random.uniform(
            tf.shape(inputs['tf_idf']),
            minval=inputs['avg_tf_idf'],
            maxval=1.0
        )  # (None, seq_len)
        mask_tf_idf = tf.expand_dims(tf.cast(inputs['tf_idf'] < random_values, dtype=tf.float32), axis=-1)  # (None, seq_len, 1)
        mask = tf.math.multiply(mask, mask_tf_idf)
        mask = tf.concat([mask[:, :-1, :], last_one], axis=1)
    '''
    '''
    if training:
        weights = self.get_embedding_weights()  # (3417, item_dim)
        norm_weights = weights / tf.linalg.norm(weights, axis=1, keepdims=True)
        cosine = tf.matmul(norm_weights, norm_weights, transpose_b=True)
        # 对每一行进行排序
        sorted_values, sorted_indices = tf.math.top_k(cosine, k=3416, sorted=True)
        print("sorted_indices:", sorted_indices)
        item_map = sorted_indices[:, -1]  # 取最不相似的item
        item_map = tf.concat([tf.constant([0]), item_map[1:]], axis=0)
        print("item_map:", item_map)
        random_values = tf.random.uniform(
            tf.shape(inputs['tf_idf']),
            minval=0.0,
            maxval=inputs['avg_tf_idf'] * 0.5
        )  # (None, seq_len)
        transfer_inputs = tf.gather(item_map, inputs['click_seq'])
        generate_inputs = tf.where(inputs['tf_idf'] < random_values, transfer_inputs, inputs['click_seq'])
    '''
    for i, val in enumerate(reversed([1, 2, 3])):
        print(i, '_', val)

