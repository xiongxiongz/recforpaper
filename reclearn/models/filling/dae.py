import numpy as np
import tensorflow as tf
import datetime
from time import time
from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from reclearn.models.losses import deal_zero_loss


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
    epochs = 40
    train_path = "/home/cqj/zzh/recforpaper/data/ml-1m/movie_seq.txt"
    meta_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_meta.txt"
    data, max_item_seq = load_data(train_path)
    with open(meta_path) as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    data_matrix = generate_synthetic_data(origin_data=data, user_num=max_user_num, max_item_seq=max_item_seq)
    data_matrix_with_noise = add_noise(data_matrix, mask_rate=0.3)

    # 构建去噪自编码器
    encoding_dim = 128
    autoencoder = build_denoising_autoencoder(input_dim=max_item_seq, encoding_dim=encoding_dim)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=deal_zero_loss)
    # 数据归一化
    data_normalized = data_matrix / max_item_num
    noisy_data_normalized = data_matrix_with_noise / max_item_num
    # 划分训练集和验证集
    train_data = data_normalized[:int(0.8 * max_user_num)]
    val_data = data_normalized[int(0.8 * max_user_num):]
    train_noisy_data = noisy_data_normalized[:int(0.8 * max_user_num)]
    val_noisy_data = noisy_data_normalized[int(0.8 * max_user_num):]

    model = tf.keras.models.load_model(
        "/home/cqj/zzh/recforpaper/reclearn/models/filling/dae_20241206_213324.h5",
    custom_objects={'deal_zero_loss': deal_zero_loss})
    prediction = model.predict(data_normalized)
    prediction = prediction * max_item_num
    # 将浮点数四舍五入为最接近的整数
    movie_ids = np.rint(prediction).astype(np.int32)
    # 确保预测的电影 ID 在合理范围内
    movie_ids = np.clip(movie_ids, 1, max_item_num)
    print("movie_ids.shape:", movie_ids.shape)
    print(movie_ids)
    '''
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：20241130_123456
    model_name = f"dae_{start_time}.h5"
    try:
        for epoch in range(1, epochs + 1):
            t1 = time()
            autoencoder.fit(
                x=train_noisy_data,
                y=train_data,
                batch_size=512,
                validation_data=(val_noisy_data, val_data),
            )
            t2 = time()
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]'
                  % (epoch, t2 - t1, time() - t2))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        autoencoder.save(model_name)
    '''