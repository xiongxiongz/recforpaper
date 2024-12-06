import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model


# 加载数据
def load_data(file_path):
    data = []
    max_item_seq = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            seq = [int(x) for x in line.strip().split(' ')]
            if len(seq) > max_item_seq:
                max_item_seq = len(seq)
            data.append(seq)
    return [data, max_item_seq]


# 数据准备
def generate_synthetic_data(origin_data, user_num, max_item_seq):
    # 模拟稀疏观影序列矩阵，每个用户有 max_item_seq 个观影记录，填充 0 表示无观影
    data = np.zeros((user_num, max_item_seq), dtype=np.int32)
    for i, seq in enumerate(origin_data):
        data[i, :len(seq)] = seq
    return data


def add_noise(data, mask_rate=0.3):
    noisy_data = data.copy()
    mask = np.random.rand(*data.shape) < mask_rate
    noisy_data[mask] = 0  # 随机将部分电影 ID 掩盖
    return noisy_data


def build_denoising_autoencoder(input_dim, encoding_dim):
    # 编码器
    input_seq = Input(shape=(input_dim,), name='input')
    encoded = Dense(encoding_dim, activation='relu', name='encoder')(input_seq)

    # 解码器
    decoded = Dense(input_dim, activation='sigmoid', name='decoder')(encoded)

    # 模型
    autoencoder = Model(inputs=input_seq, outputs=decoded)
    return autoencoder


if __name__ == '__main__':
    train_path = "data/ml-1m/ml_seq_train.txt"
    load_data("data/ml-1m/ml_seq_train.txt")

