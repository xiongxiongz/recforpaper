import tensorflow as tf
from tensorflow.keras.layers import Embedding
from reclearn.data.datasets import movielens as ml
from data.utils.data_loader import DataGenerator
from reclearn.models.filling.sasrec_demo import SASRec
from reclearn.models.filling.ppo import PPOModel
from reclearn.models.losses import cal_rl_loss
from time import time
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_epoch = 20
train_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_train.txt"
val_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_val.txt"
test_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_test.txt"
meta_path = "/home/cqj/zzh/recforpaper/data/ml-1m/ml_seq_meta.txt"
batch_size = 512
neg_num = 100
test_neg_num = 100
embed_dim = 64
blocks = 2
num_heads = 2
ffn_hidden_unit = 64
dnn_dropout = 0.2
use_l2norm = False
loss_name = "binary_cross_entropy_loss"
gamma = 0.5
seq_len = 100
embed_reg = 0.0
with open(meta_path) as f:
    max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
user_num = max_user_num + 1
item_num = max_item_num + 1

# Load Sequence Data
train_data = ml.load_seq_data(train_path, "train", seq_len, neg_num, max_item_num)
train_generator = DataGenerator(train_data, batch_size)
val_data = ml.load_seq_data(val_path, "val", seq_len, neg_num, max_item_num)
val_generator = DataGenerator(val_data, batch_size)
test_data = ml.load_seq_data(test_path, "test", seq_len, test_neg_num, max_item_num)

model_params = {
        'item_num': item_num,
        'user_num': user_num,
        'embed_dim': embed_dim,
        'seq_len': seq_len,
        'blocks': blocks,
        'num_heads': num_heads,
        'ffn_hidden_unit': ffn_hidden_unit,
        'dnn_dropout': dnn_dropout,
        'use_l2norm': use_l2norm,
        'loss_name': loss_name,
        'gamma': gamma,
        'embed_reg': embed_reg
    }
sasrec_model = SASRec(**model_params)
ppo_model = PPOModel(item_num, user_num, seq_len, embed_dim, embed_reg)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
def compute_ppo_loss(policies, next_policies, advantage, epsilon=0.2):
    # 策略比值
    old_probs = tf.reduce_sum(tf.nn.softmax(policies, axis=-1), axis=-1)
    probs = tf.reduce_sum(tf.nn.softmax(next_policies, axis=-1), axis=-1)
    ratio = probs / (old_probs + 1e-10)

    # 剪辑损失
    clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
    ppo_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))

    return ppo_loss

def binary_cross_entropy_loss(pos_scores, neg_scores):
    """binary cross entropy loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, neg_num].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    loss = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_scores)) - tf.math.log(1 - tf.nn.sigmoid(neg_scores))) / 2
    return loss

def cal_rewards(results):
    rewards = -cal_rl_loss(results)
    return rewards


def cal_hr(results):
    pred_y = - results
    pred_y = tf.argsort(tf.argsort(pred_y))
    pred_y = tf.slice(pred_y, begin=[0, 0], size=[-1, 1])
    pred_y = tf.cast(pred_y, tf.float32)
    loss_matrix = tf.where(pred_y < 10.0, tf.ones_like(pred_y), tf.zeros_like(pred_y))
    hr = tf.reduce_mean(loss_matrix)
    return hr

@tf.function
def train_step(inputs, next_states=1, ppo_gamma=0.95):
    with tf.GradientTape(persistent=True) as tape:
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32),
                              axis=-1)  # (None, seq_len, 1)
        # TODO:PPO预测权重矩阵
        policies, values, weighted_embed, pos_info, neg_info = ppo_model(inputs)  # policies (None, seq_len, dim), values (None, 1)
        # 掩码
        weighted_embed *= mask  # 去除填充为0部分的影响
        # TODO:SASRec Model
        # SASRec 模型
        att_outputs = sasrec_model(weighted_embed, mask)  # SASRec 提取特征

        # 计算 SASEec 损失
        pos_scores = tf.reduce_sum(tf.multiply(att_outputs, tf.expand_dims(pos_info, axis=1)), axis=-1)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(att_outputs, neg_info), axis=-1)  # (None, neg_num)
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        rewards = cal_rewards(logits)
        hr = cal_hr(logits)
        sas_loss = binary_cross_entropy_loss(pos_scores, neg_scores)
        # 计算 PPO 损失
        # next_policies, next_value, _, _, _ = ppo_model(next_states)
        # advantage = rewards + ppo_gamma * next_value - values  # TD 误差
        advantage = rewards - values
        # ppo_loss = compute_ppo_loss(policies, next_policies, advantage)  # 自定义函数
        value_loss = tf.reduce_mean(tf.square(values - rewards))  # 值函数损失
        # 总损失
        # total_loss = ppo_loss + value_loss + sas_loss
        total_loss = value_loss + sas_loss
    gradients = tape.gradient(total_loss, ppo_model.trainable_variables + sasrec_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ppo_model.trainable_variables + sasrec_model.trainable_variables))
    return total_loss, hr, rewards


if __name__ == '__main__':
    loss, hr, ndcg = None, None, None
    for epoch in range(1, train_epoch+1):
        t1 = time()
        for batch_data in tqdm(train_generator):
            loss, _, _ = train_step(batch_data)
        t2 = time()
        _, hr, ndcg = train_step(test_data)
        print(f"Epoch {epoch}, Fit {t2-t1}, Train Loss: {loss.numpy()}, hr@10: {hr}, ndcg@10: {ndcg}")
