"""
Created on Dec 20, 2020
Updated on Apr 22, 2022
Reference: "Self-Attentive Sequential Recommendation", ICDM, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Input, Conv1D, DepthwiseConv2D
from tensorflow.keras.regularizers import l2
from reclearn.layers import TransformerEncoder
from reclearn.models.losses import get_loss_with_istarget
from reclearn.layers.core import TransformerEncoder2


class SASRec(Model):
    def __init__(self, item_num, user_num, item_dim, user_dim, seq_len=100, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dnn_dropout=0., layer_norm_eps=1e-6, use_l2norm=False,
                 loss_name="binary_cross_entropy_loss", gamma=0.5, embed_reg=0., seed=None, neg_num=50):
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
        super(SASRec, self).__init__()
        self.hidden_unit = item_dim + user_dim
        '''
        self.genre_embedding = Embedding(
                                        input_dim=19,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        '''

        # item embedding
        self.item_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=item_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))

        self.tf_idf_embedding = Embedding(input_dim=6,
                                        input_length=1,
                                        output_dim=item_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        '''
        self.popularity_embedding = Embedding(input_dim=6,
                                          input_length=1,
                                          output_dim=item_dim,
                                          embeddings_initializer='random_normal',
                                          embeddings_regularizer=l2(embed_reg))

        self.pos_embedding = Embedding(input_dim=seq_len,
                                       input_length=1,
                                       output_dim=item_dim,
                                       embeddings_initializer='random_normal',
                                       embeddings_regularizer=l2(embed_reg))
        '''
        '''
        self.pos_embedding_trainable = self.add_weight(
            shape=[1, seq_len, self.hidden_unit],
            initializer='glorot_uniform',
            trainable=True,
            name='pos_embedding_trainable'
        )
        self.fc = Dense(self.hidden_unit // 2, activation='relu')
        self.gate_dense = Dense(self.hidden_unit, activation='sigmoid')
        '''

        self.dense = Dense(units=user_dim // 4, activation="relu")
        self.user_dropout = Dropout(0.3)
        self.conv = Conv1D(filters=user_dim, kernel_size=1)
        self.user_embedding = Embedding(input_dim=user_num,
                                       output_dim=user_dim,
                                       input_length=1,
                                       embeddings_initializer='random_normal',
                                       embeddings_regularizer=l2(embed_reg))

        self.dropout = Dropout(dnn_dropout)
        # multi encoder block
        self.encoder_layer = [TransformerEncoder2(self.hidden_unit, num_heads, self.hidden_unit,
                                                 dnn_dropout, layer_norm_eps) for _ in range(blocks)]
        # norm
        self.use_l2norm = use_l2norm
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seq_len
        self.seq_len = seq_len
        # seed
        tf.random.set_seed(seed)
        # neg_num
        self.neg_num = neg_num

    def call(self, inputs, training=None, mask=None):
        # seq info
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, item_dim)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32), axis=-1)  # (None, seq_len, 1)

        # pos encoding
        # squeeze = tf.reduce_mean(seq_embed, axis=1)
        # excitation = self.fc(squeeze)
        # gate = self.gate_dense(excitation)
        # gate = tf.expand_dims(gate, axis=1)
        # gate_pos_embed = gate * self.pos_embedding_trainable
        # seq_embed += gate_pos_embed

        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.seq_len)), axis=0)  # (1, seq_len, embed_dim)
        seq_embed += pos_encoding  # (None, seq_len, embed_dim), broadcasting

        origin_user_encode = self.user_embedding(inputs['user'])  # (None, 1, user_dim)
        origin_user_embed = self.dense(origin_user_encode)
        origin_user_embed = self.user_dropout(origin_user_embed)
        origin_user_embed = self.conv(origin_user_embed)
        user_embed = tf.tile(origin_user_embed, [1, self.seq_len, 1])  # (None, seq_len, user_dim)
        seq_embed = tf.concat([seq_embed, user_embed], axis=-1)  # (None, seq_len, item_dim + user_dim)

        '''
        origin_user_encode = self.user_embedding(inputs['user'])  # (None, 1, user_dim)
        origin_user_embed = self.dense(origin_user_encode)
        origin_user_embed = self.user_dropout(origin_user_embed)
        origin_user_embed = self.conv(origin_user_embed)
        user_embed = tf.tile(origin_user_embed, [1, self.seq_len, 1])  # (None, seq_len, user_dim)
        seq_embed = tf.concat([seq_embed, user_embed], axis=-1)  # (None, seq_len, item_dim + user_dim)
        squeeze = tf.reduce_mean(seq_embed, axis=1)
        excitation = self.fc(squeeze)
        gate = self.gate_dense(excitation)
        gate = tf.expand_dims(gate, axis=1)
        gate_pos_embed = gate * self.pos_embedding_trainable
        seq_embed += gate_pos_embed
        '''
        '''
        genre_encoding = self.genre_embedding(inputs['genre_index_seq'])       # (batch, seq_len, 6, emb_dim)
        genre_encoding = tf.reduce_mean(genre_encoding, axis=2)       # (batch, seq_len, emb_dim)
        seq_embed += genre_encoding
        '''
        seq_embed = self.dropout(seq_embed)
        att_outputs = seq_embed  # (None, seq_len, embed_dim)
        att_outputs *= mask
        # transformer encoder part
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, embed_dim)
            att_outputs *= mask
        # item info contain pos_info and neg_info.
        pos_info = self.item_embedding(inputs['pos_item'])  # (None, seq_len, item_dim) or (None, 1, item_dim)
        neg_info = self.item_embedding(inputs['neg_item'])  # (None, seq_len, item_dim) or (None, 100, item_dim)
        if training:
            # add user
            neg_num = neg_info.shape[1]
            if neg_num is None:
                neg_num = self.neg_num
            neg_user_emb = [origin_user_encode for _ in range(neg_num)]
            neg_user_emb = tf.concat(neg_user_emb, axis=1)
            neg_info = tf.concat([neg_info, neg_user_emb], axis=-1)
            pos_num = pos_info.shape[1]
            if pos_num is None:
                pos_num = self.neg_num
            pos_user_emb = [origin_user_encode for _ in range(pos_num)]
            pos_user_emb = tf.concat(pos_user_emb, axis=1)
            pos_info = tf.concat([pos_info, pos_user_emb], axis=-1)

            pos_scores = tf.reduce_sum(tf.multiply(att_outputs, pos_info), axis=-1)  # (None, seq_len)
            neg_scores = tf.reduce_sum(tf.multiply(att_outputs, neg_info), axis=-1)  # (None, seq_len)
            pos_logits = tf.reshape(pos_scores, [-1, ])
            neg_logits = tf.reshape(neg_scores, [-1, ])
            # 当用户序列长度不足时，正样本总会存在0
            istarget = tf.reshape(tf.cast(tf.not_equal(inputs['pos_item'], 0), dtype=tf.float32), [-1, ])
            return [pos_logits, neg_logits, istarget]

        if not training:
            # add user
            neg_user_emb = [origin_user_encode for _ in range(100)]
            neg_user_emb = tf.concat(neg_user_emb, axis=1)
            neg_info = tf.concat([neg_info, neg_user_emb], axis=-1)
            pos_info = tf.concat([pos_info, origin_user_encode], axis=-1)

            user_info = tf.slice(att_outputs, begin=[0, self.seq_len - 1, 0], size=[-1, 1, -1])  # (None, 1, item_dim)
            infer_neg_scores = tf.reduce_sum(tf.multiply(user_info, neg_info), axis=-1)  # (None, seq_len) or (None, 100)
            infer_pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1)  # (None, seq_len) or (None, 1)
            return tf.concat([infer_neg_scores, infer_pos_scores], axis=-1)  # (None, 100+1)

    def get_embedding_weights(self):
        return self.item_embedding.trainable_variables[0]
