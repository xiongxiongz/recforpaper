import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.regularizers import l2


class PPOModel(Model):
    def __init__(self, item_num, user_num, seq_len, embed_dim, embed_reg):
        super(PPOModel, self).__init__()
        self.actor = Sequential([
            Dense(128, activation='relu'),
            Dense(embed_dim, activation='softmax')
        ])
        self.critic = Sequential([
            Dense(128, activation='relu'),
            Dense(1)
        ])
        self.seq_len = seq_len
        # item embedding
        self.item_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

        self.user_embedding = Embedding(input_dim=user_num,
                                        output_dim=embed_dim,
                                        input_length=1,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

    def call(self, inputs):
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, dim)
        user_embed = self.user_embedding(inputs['user'])  # (None, 1, dim)
        # pos encoding
        seq_user_embed = tf.multiply(seq_embed, user_embed)  # (None, seq_len, dim)
        values = self.critic(seq_user_embed)
        policies = self.actor(seq_user_embed)
        # 采样动作（权重矩阵）
        action_probs = tf.nn.softmax(policies, axis=-1)  # 确保权重在[0, 1]范围
        weight_matrix = action_probs  # 权重矩阵 (None, seq_len, dim)
        # 应用权重矩阵
        weighted_embed = tf.multiply(seq_user_embed, weight_matrix)  # (None, seq_len, dim)
        # item info contain pos_info and neg_info.
        pos_info = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, dim)
        neg_info = self.item_embedding(inputs['neg_item'])  # (None, neg_num, dim)
        return policies, values, weighted_embed, pos_info, neg_info

