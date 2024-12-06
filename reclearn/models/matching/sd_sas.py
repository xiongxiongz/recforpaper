import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding
from tensorflow.keras.regularizers import l2

from reclearn.layers import TransformerEncoder
from reclearn.models.losses import get_loss, get_loss_with_emb


class SDSAS(Model):
    def __init__(self, item_num, embed_dim, seq_len=100, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dnn_dropout=0., layer_norm_eps=1e-6, use_l2norm=False,
                 loss_name="binary_cross_entropy_loss", gamma=0.5, embed_reg=0., seed=None):
        """Self-Attentive Sequential Recommendation
        :param item_num: An integer type. The largest item index + 1.
        :param embed_dim: An integer type. Embedding dimension of item vector.
        :param seq_len: An integer type. The length of the input sequence.
        :param blocks: An integer type. The Number of blocks.
        :param num_heads: An integer type. The Number of attention heads.
        :param ffn_hidden_unit: An integer type. Number of hidden unit in FFN.
        :param dnn_dropout: Float between 0 and 1. Dropout of user and item MLP layer.
        :param layer_norm_eps: A float type. Small float added to variance to avoid dividing by zero.
        :param use_l2norm: A boolean. Whether user embedding, item embedding should be normalized or not.
        :param loss_name: A string. You can specify the current point-loss function 'binary_cross_entropy_loss' or
        pair-loss function as 'bpr_loss'、'hinge_loss'.
        :param gamma: A float type. If hinge_loss is selected as the loss function, you can specify the margin.
        :param embed_reg: A float type. The regularizer of embedding.
        :param seed: A Python integer to use as random seed.
        """
        super(SDSAS, self).__init__()
        # item embedding
        self.item_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.pos_embedding = Embedding(input_dim=seq_len,
                                       input_length=1,
                                       output_dim=embed_dim,
                                       embeddings_initializer='random_normal',
                                       embeddings_regularizer=l2(embed_reg))
        # encode
        self.dense1 = Dense(128, activation='relu')
        self.encode_dense = Dense(64, activation='relu')
        self.decode_dense = Dense(item_num, activation='sigmoid')
        # multi encoder block
        self.encoder_layer = [TransformerEncoder(embed_dim, num_heads, ffn_hidden_unit,
                                                 dnn_dropout, layer_norm_eps) for _ in range(blocks)]
        self.dropout = Dropout(dnn_dropout)
        # norm
        self.use_l2norm = use_l2norm
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seq_len
        self.seq_len = seq_len
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # 嵌入层
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, dim)

        # 编码器
        encoded = self.dense1(seq_embed)  # (None, seq_len, 128)
        latent_space = self.encode_dense(encoded)   # (None, seq_len, 64)

        # 解码器（重建任务）
        reconstructed = self.decode_dense(latent_space)   # (None, seq_len, item_num)

        # 序列推荐模块（预测下一个行为）
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32),
                              axis=-1)  # (None, seq_len, 1)
        # pos encoding
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.seq_len)), axis=0)  # (1, seq_len, embed_dim)

        latent_space += pos_encoding  # (None, seq_len, embed_dim), broadcasting
        latent_space = self.dropout(latent_space)
        att_outputs = latent_space  # (None, seq_len, embed_dim)
        att_outputs *= mask
        # transformer encoder part
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, embed_dim)
            att_outputs *= mask
        # user_info. There are two ways to get the user vector.
        # user_info = tf.reduce_mean(att_outputs, axis=1)  # (None, dim)
        user_info = tf.slice(att_outputs, begin=[0, self.seq_len - 1, 0], size=[-1, 1, -1])  # (None, 1, embed_dim)

        pos_info = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, dim)
        neg_info = self.item_embedding(inputs['neg_item'])  # (None, neg_num, dim)
        # norm
        if self.use_l2norm:
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
            user_info = tf.math.l2_normalize(user_info, axis=-1)
        pos_scores = tf.reduce_sum(tf.multiply(user_info, tf.expand_dims(pos_info, axis=1)), axis=-1)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(user_info, neg_info), axis=-1)  # (None, neg_num)
        # loss
        self.add_loss(get_loss_with_emb(pos_scores, neg_scores, self.loss_name, reconstructed, inputs['click_seq'], latent_space, self.gamma))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits