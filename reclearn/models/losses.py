"""
Created on Nov 14, 2021
Loss function.
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence

def get_loss(pos_scores, neg_scores, loss_name, gamma=None):
    """Get loss scores.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, 1].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
        :param loss_name: A string such as 'bpr_loss', 'hing_loss' and etc.
        :param gamma: A scalar(int). If loss_name == 'hinge_loss', the gamma must be valid.
    :return:
    """
    # pos_scores = tf.tile(pos_scores, [1, neg_scores.shape[1]])
    if loss_name == 'bpr_loss':
        loss = bpr_loss(pos_scores, neg_scores)
    elif loss_name == 'hinge_loss':
        loss = hinge_loss(pos_scores, neg_scores, gamma)
    else:
        loss = binary_cross_entropy_loss(pos_scores, neg_scores)
    return loss


def get_loss_with_rl(pos_scores, neg_scores, loss_name, gamma=None):
    """Get loss scores.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, 1].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
        :param loss_name: A string such as 'bpr_loss', 'hing_loss' and etc.
        :param gamma: A scalar(int). If loss_name == 'hinge_loss', the gamma must be valid.
    :return:
    """
    # pos_scores = tf.tile(pos_scores, [1, neg_scores.shape[1]])
    if loss_name == 'bpr_loss':
        loss = bpr_loss(pos_scores, neg_scores)
    elif loss_name == 'hinge_loss':
        loss = hinge_loss(pos_scores, neg_scores, gamma)
    else:
        loss = binary_cross_entropy_loss_with_rl_loss(pos_scores, neg_scores)
    return loss


def get_loss_with_istarget(pos_logits, neg_logits, istarget):
    """Get loss scores.
    Args:
        :param pos_logits: A tensor with shape of [batch_size, neg_num].
        :param neg_logits: A tensor with shape of [batch_size, neg_num].
        :param istarget: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    loss = tf.reduce_sum(
        - tf.math.log(tf.sigmoid(pos_logits) + 1e-24) * istarget -
        tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-24) * istarget
    ) / tf.reduce_sum(istarget)
    return loss

def get_loss_with_emb(pos_scores, neg_scores, loss_name, y_pred, y_true, norm_emb, gamma=None):
    """Get loss scores.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, 1].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
        :param loss_name: A string such as 'bpr_loss', 'hing_loss' and etc.
        :param y_pred
        :param y_true
        :param norm_emb
        :param gamma: A scalar(int). If loss_name == 'hinge_loss', the gamma must be valid.
    :return:
    """
    # pos_scores = tf.tile(pos_scores, [1, neg_scores.shape[1]])
    if loss_name == 'bpr_loss':
        loss = bpr_loss(pos_scores, neg_scores)
    elif loss_name == 'hinge_loss':
        loss = hinge_loss(pos_scores, neg_scores, gamma)
    elif loss_name == 'infonce_loss':
        loss = infonce_loss(pos_scores, neg_scores)
    elif loss_name == 'hybrid_infonce_cross':
        loss = hybrid_loss(pos_scores, neg_scores)
    else:
        loss = binary_cross_entropy_loss_with_emb(pos_scores, neg_scores, y_pred, y_true, norm_emb)
    return loss


def bpr_loss(pos_scores, neg_scores):
    """bpr loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, 1].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
    return loss


def hinge_loss(pos_scores, neg_scores, gamma=0.5):
    """hinge loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, neg_num].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
        :param gamma: A scalar(int).
    :return:
    """
    loss = tf.reduce_mean(tf.nn.relu(neg_scores - pos_scores + gamma))
    return loss


def binary_cross_entropy_loss(pos_scores, neg_scores):
    """binary cross entropy loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, neg_num].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    loss = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_scores)) - tf.math.log(1 - tf.nn.sigmoid(neg_scores))) / 2
    return loss


def binary_cross_entropy_loss_with_rl_loss(pos_scores, neg_scores):
    """binary cross entropy loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, neg_num].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    # KL散度
    # a_probs = probs[0]
    # a_probs = tf.nn.softmax(a_probs, axis=-1)
    # b_probs = probs[1]
    # b_probs = tf.nn.softmax(b_probs, axis=-1)
    # reward_loss_divergence = tf.reduce_mean(KLDivergence()(a_probs, b_probs))
    # contra_loss = tf.constant(0.5, dtype=reward_loss_divergence.dtype) * reward_loss_divergence
    loss = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_scores)) - tf.math.log(1 - tf.nn.sigmoid(neg_scores))) / 2
    return loss

def cal_rl_loss(logits, k=10):
    pred_y = - logits
    pred_y = tf.argsort(tf.argsort(pred_y))
    pred_y = tf.slice(pred_y, begin=[0, 0], size=[-1, 1])
    pred_y = tf.cast(pred_y, tf.float32)
    reward_matrix = tf.where(pred_y < 10.0, 1.0 / (tf.math.log(pred_y + 2.0) / tf.math.log(2.0)), tf.zeros_like(pred_y))
    reward = tf.reduce_mean(reward_matrix)
    return tf.constant(1.0, dtype=tf.float32) - reward


def binary_cross_entropy_loss_with_emb(pos_scores, neg_scores, y_pred, y_true, norm_emb, alpha=0.8, beta=0.2, reg=0.01):
    """binary cross entropy loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, neg_num].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
        :param y_pred
        :param y_true
        :param norm_emb
    :return:
    """
    base_loss = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_scores)) - tf.math.log(1 - tf.nn.sigmoid(neg_scores))) / 2
    reconstruct_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    # reg_loss = tf.reduce_mean(tf.square(norm_emb))
    return alpha * base_loss + beta * reconstruct_loss


def infonce_loss(pos_scores, neg_scores, temperature=0.7):
    # 正样本 logits
    pos_logits = pos_scores / temperature

    # 负样本 logits，逐样本与负样本对比
    neg_logits = neg_scores / temperature
    # 拼接正负样本 logits
    logits = tf.concat([pos_logits, neg_logits], axis=-1)
    # 创建标签（正样本为第 0 类）
    labels = tf.zeros(shape=(tf.shape(pos_scores)[0],), dtype=tf.int32)

    # 使用交叉熵计算对比损失
    contrastive_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )

    return contrastive_loss


def hybrid_loss(pos_scores, neg_scores, alpha=0.2, temperature=0.05):
    """
    结合原始损失和 InfoNCE 的混合损失函数。

    :param pos_scores: 正样本分数，形状为 (batch_size,)
    :param neg_scores: 负样本分数，形状为 (batch_size, num_negatives)
    :param alpha: 原始损失和对比损失的权重平衡系数
    :param temperature: 温度参数，控制对比损失的分布锐度
    :return: 混合损失值
    """
    # 原始损失部分
    original_loss = tf.reduce_mean(
        -tf.math.log(tf.nn.sigmoid(pos_scores))
        - tf.math.log(1 - tf.nn.sigmoid(neg_scores))
    ) / 2

    # InfoNCE 损失部分
    # 正样本 logits
    pos_logits = pos_scores / temperature

    # 负样本 logits，逐样本与负样本对比
    neg_logits = neg_scores / temperature

    # 拼接正负样本 logits
    logits = tf.concat([tf.expand_dims(pos_logits, axis=1), neg_logits], axis=1)

    # 创建标签（正样本为第 0 类）
    labels = tf.zeros(shape=(tf.shape(pos_scores)[0],), dtype=tf.int32)

    # 使用交叉熵计算对比损失
    contrastive_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )

    # 结合两种损失
    total_loss = alpha * original_loss + (1 - alpha) * contrastive_loss
    return total_loss


def deal_zero_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss_non_zero = tf.reduce_mean(tf.square((y_true - y_pred) * mask))
    loss_zero = tf.reduce_mean(tf.square((y_true - y_pred) * (1 - mask)))
    return 0.7 * loss_non_zero + 0.3 * loss_zero

