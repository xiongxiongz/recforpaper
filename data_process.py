from collections import defaultdict, Counter
import math
import numpy as np


def data_partition(fname):
    all_tf_idfs, bucket_ids = [], []
    num_bins = 5
    argue_popularity, argue_bucket_ids = [], []
    argue_bins = 5
    item2user = defaultdict(set)
    user_tf_idf = defaultdict(list)
    user_popularity = defaultdict(list)
    user_bucket = {}
    user_bin = {}
    user_bucket_train = {}
    user_bucket_valid = {}
    user_bucket_test = {}
    user_bin_train = {}
    user_bin_valid = {}
    user_bin_test = {}
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    file_path = f"data/{fname}/{fname}.txt"
    f = open(file_path, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        item2user[i].add(u)
    for u in User:
        tf_idf, popularity = [], []
        counter = Counter(User[u])
        for i in counter:
            temp_tf_idf = counter[i] / len(User[u]) * math.log(usernum / len(item2user[i]))
            tf_idf.append(temp_tf_idf)
            temp_popularity = counter[i] / len(User[u]) * math.exp(len(item2user[i]) / usernum)
            popularity.append(temp_popularity)
        all_tf_idfs.extend(tf_idf)
        argue_popularity.extend(popularity)
        user_tf_idf[u] = tf_idf
        user_popularity[u] = popularity
    quantiles = np.percentile(all_tf_idfs, np.linspace(0, 100, num_bins + 1))
    argue_quantiles = np.percentile(argue_popularity, np.linspace(0, 100, argue_bins + 1))
    for u in User:
        user_bucket[u] = np.digitize(user_tf_idf[u], quantiles, right=True).tolist()
        user_bin[u] = np.digitize(user_popularity[u], argue_quantiles, right=True).tolist()
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
            user_bucket_train[user] = user_bucket[user]
            user_bucket_valid[user] = []
            user_bucket_test[user] = []
            user_bin_train[user] = user_bin[user]
            user_bin_valid[user] = []
            user_bin_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
            user_bucket_train[user] = user_bucket[user][:-2]
            user_bucket_valid[user] = []
            user_bucket_valid[user].append(user_bucket[user][-2])
            user_bucket_test[user] = []
            user_bucket_test[user].append(user_bucket[user][-1])
            user_bin_train[user] = user_bin[user][:-2]
            user_bin_valid[user] = []
            user_bin_valid[user].append(user_bin[user][-2])
            user_bin_test[user] = []
            user_bin_test[user].append(user_bin[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum, user_bucket_train, user_bucket_valid, user_bucket_test, user_bin_train, user_bin_valid, user_bin_test]


def data_partition_0(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    file_path = f"data/{fname}/{fname}.txt"
    f = open(file_path, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]