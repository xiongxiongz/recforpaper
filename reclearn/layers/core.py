"""
Created on Nov 07, 2021
core layers
@author: Ziyao Geng(zggzy1996@163.com)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU
from tensorflow.keras.regularizers import l2

from reclearn.layers.utils import scaled_dot_product_attention, split_heads, index_mapping

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.reduction_ratio = reduction_ratio

        # Squeeze and Excitation layers
        self.fc1 = Dense(embedding_dim // reduction_ratio, activation='relu')
        self.fc2 = Dense(embedding_dim, activation='sigmoid')

    def call(self, inputs):
        """
        Args:
            inputs: Tensor, shape (batch_size, seq_len, embedding_dim)
        Returns:
            Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        # Squeeze: Global Max or Average Pooling on time dimension
        squeeze = tf.reduce_max(inputs, axis=1)  # shape: (batch_size, embedding_dim)

        # Excitation: Fully connected layers
        excitation = self.fc1(squeeze)  # shape: (batch_size, embedding_dim // reduction_ratio)
        excitation = self.fc2(excitation)  # shape: (batch_size, embedding_dim)

        # Reweight: Scale the original inputs
        excitation = tf.expand_dims(excitation, axis=1)  # shape: (batch_size, 1, embedding_dim)
        outputs = inputs * excitation  # shape: (batch_size, seq_len, embedding_dim)
        return outputs


class FeatureSEBlock(tf.keras.layers.Layer):
    def __init__(self, seq_len, reduction_ratio=25):
        super(FeatureSEBlock, self).__init__()
        self.seq_len = seq_len
        self.reduction_ratio = reduction_ratio

        # Squeeze and Excitation layers
        self.fc1 = Dense(seq_len // reduction_ratio, activation='relu')
        self.fc2 = Dense(seq_len, activation='sigmoid')

    def call(self, inputs):
        """
        Args:
            inputs: Tensor, shape (batch_size, seq_len, embedding_dim)
        Returns:
            Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        # Squeeze: Global Max or Average Pooling on feature dimension
        squeeze = tf.reduce_max(inputs, axis=2)  # shape: (batch_size, seq_len)

        # Excitation: Fully connected layers
        excitation = self.fc1(squeeze)  # shape: (batch_size, seq_len // reduction_ratio)
        excitation = self.fc2(excitation)  # shape: (batch_size, seq_len)

        # Reweight: Scale the original inputs
        excitation = tf.expand_dims(excitation, axis=2)  # shape: (batch_size, seq_len, 1)
        outputs = inputs * excitation  # shape: (batch_size, seq_len, embedding_dim)
        return outputs


class Linear(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """Linear Part.
        Args:
            :param feature_length: A scalar. The length of features.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
        :return:
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name="w",
                                 shape=(self.feature_length, 1),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs):
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        return result


class MLP(Layer):
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., use_batch_norm=False):
        """Multilayer Perceptron.
        Args:
            :param hidden_units: A list. The list of hidden layer units's numbers.
            :param activation: A string. The name of activation function, like 'relu', 'sigmoid' and so on.
            :param dnn_dropout: A scalar. The rate of dropout .
            :param use_batch_norm: A boolean. Whether using batch normalization or not.
        :return:
        """
        super(MLP, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.use_batch_norm = use_batch_norm
        self.bt = BatchNormalization()

    def call(self, inputs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        if self.use_batch_norm:
            x = self.bt(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        """Multi Head Attention Mechanism.
        Args:
            :param d_model: A scalar. The self-attention hidden size.
            :param num_heads: A scalar. Number of heads. If num_heads == 1, the layer is a single self-attention layer.
        :return:
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)

    def call(self, q, k, v, mask):
        q = self.wq(q)  # (None, seq_len, d_model)
        k = self.wk(k)  # (None, seq_len, d_model)
        v = self.wv(v)  # (None, seq_len, d_model)
        # split d_model into num_heads * depth
        '''
        seq_len, d_model = q.shape[1], q.shape[2]
        q = split_heads(q, seq_len, self.num_heads, q.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        k = split_heads(k, seq_len, self.num_heads, k.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        v = split_heads(v, seq_len, self.num_heads, v.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        # mask
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_heads, 1, 1])  # (None, num_heads, seq_len, 1)
        '''
        par_output = []
        dimension_per_head = self.d_model // self.num_heads
        for i in range(self.num_heads):
            # par_q = tf.slice(q, begin=[0, 0, dimension_per_head * i], size=[-1, -1, dimension_per_head])
            # par_k = tf.slice(k, begin=[0, 0, dimension_per_head * i], size=[-1, -1, dimension_per_head])
            # par_v = tf.slice(v, begin=[0, 0, dimension_per_head * i], size=[-1, -1, dimension_per_head])
            par_q = q[..., dimension_per_head * i: dimension_per_head * (i+1)]
            par_k = k[..., dimension_per_head * i: dimension_per_head * (i+1)]
            par_v = v[..., dimension_per_head * i: dimension_per_head * (i+1)]
            # attention
            scaled_attention = scaled_dot_product_attention(par_q, par_k, par_v, mask)  # (None, num_heads, seq_len, d_model // num_heads)
            par_output.append(scaled_attention)
        outputs = tf.concat(par_output, axis=-1)
        # reshape
        # outputs = tf.reshape(tf.transpose(scaled_attention, [0, 2, 1, 3]), [-1, seq_len, d_model])  # (None, seq_len, d_model)
        return outputs


class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """Feed Forward Network.
        Args:
            :param hidden_unit: A scalar.
            :param d_model: A scalar.
        :return:
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs):
        x = self.conv1(inputs)
        output = self.conv2(x)
        return output


class TransformerEncoder(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., layer_norm_eps=1e-6):
        """Encoder Layer.
        Args:
            :param d_model: A scalar. The self-attention hidden size.
            :param num_heads: A scalar. Number of heads.
            :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
            :param dropout: A scalar. Number of dropout.
            :param layer_norm_eps: A scalar. Small float added to variance to avoid dividing by zero.
        :return:
        """
        super(TransformerEncoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs):
        x, mask = inputs
        # self-attention
        att_out = self.mha(x, x, x, mask)  # (None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)  # (None, seq_len, d_model)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
        out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2


class SelfAttention(Layer):
    def __init__(self, add_pos=True):
        """Self Attention.
        :return:
        """
        super(SelfAttention, self).__init__()
        self.add_pos = add_pos

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=[self.dim, self.dim],
            name='att_weights',
            initializer='random_normal')

    def call(self, inputs):
        q, k, v, mask = inputs
        # pos encoding
        if self.add_pos:
            k += self.positional_encoding(k)
            q += self.positional_encoding(q)
        # Nonlinear transformation
        q = tf.nn.relu(tf.matmul(q, self.W))  # (None, seq_len, dim)
        k = tf.nn.relu(tf.matmul(k, self.W))  # (None, seq_len, dim)
        mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        mask = tf.tile(mask, [1, 1, q.shape[1]])  # (None, seq_len, seq_len)
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
        # softmax
        outputs = tf.nn.softmax(logits=outputs, axis=-1)  # (None, seq_len, seq_len)
        # output
        outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = tf.reduce_mean(outputs, axis=1)  # (None, dim)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        """Factorization Machines.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return:
        """
        super(FM_Layer, self).__init__()
        self.feature_columns = feature_columns
        self.field_num = len(feature_columns)
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs):
        # mapping
        inputs = index_mapping(inputs, self.map_dict)
        inputs = tf.concat([value for _, value in inputs.items()], axis=-1)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs


class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        """Field Factorization Machines Layer.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return:
        """
        super(FFM_Layer, self).__init__()
        self.feature_columns = feature_columns
        self.field_num = len(self.feature_columns)
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_length, self.field_num, self.k),
                                 initializer='random_normal',
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs):
        inputs = index_mapping(inputs, self.map_dict)
        inputs = tf.concat([value for _, value in inputs.items()], axis=-1)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # field second order
        second_order = 0
        latent_vector = tf.reduce_sum(tf.nn.embedding_lookup(self.v, inputs), axis=1)  # (batch_size, field_num, k)
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                second_order += tf.reduce_sum(latent_vector[:, i] * latent_vector[:, j], axis=1, keepdims=True)
        return first_order + second_order


class Residual_Units(Layer):
    def __init__(self, hidden_unit, dim_stack):
        """Residual Units.
        Args:
            :param hidden_unit: A list. Neural network hidden units.
            :param dim_stack: A scalar. The dimension of inputs unit.
        :return:
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs


class CrossNetwork(Layer):
    def __init__(self, layer_num, reg_w=0., reg_b=0.):
        """CrossNetwork.
        Args:
            :param layer_num: A scalar. The depth of cross network.
            :param reg_w: A scalar. The regularization coefficient of w.
            :param reg_b: A scalar. The regularization coefficient of b.
        :return:
        """
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_b),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs):
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        x_l = x_0  # (None, dim, 1)
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, 1, 1)
            print(x_l1)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        return x_l


class New_FM(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """New Factorization Machine for DeepFm.
        Note. In DeepFM, only the first order feature and second order feature intersect are included.
        Args:
            :param feature_length: A scalar. The length of features.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
        :return:
        """
        super(New_FM, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        :param inputs: A dict with shape `(batch_size, {'sparse_inputs', 'embed_inputs'})`:
          sparse_inputs is 2D tensor with shape `(batch_size, sum(field_num))`
          embed_inputs is 3D tensor with shape `(batch_size, fields, embed_dim)`
        """
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']
        sparse_inputs = tf.concat([value for _, value in sparse_inputs.items()], axis=-1)
        # first order
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1)  # (batch_size, 1)
        # second order
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        return first_order + second_order


class CIN(Layer):
    def __init__(self, cin_size, l2_reg=0.):
        """CIN
        Args:
            :param cin_size: A list. [H_1, H_2 ,..., H_k], a list of the number of layers
            :param l2_reg: A scalar. L2 regularization.
        :return:
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # get the number of embedding fields
        self.embedding_nums = input_shape[1]
        # a list of the number of CIN
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_normal',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for convenient calculation
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)  # dim * (None, field_nums[0], 1)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)  # dim * (None, filed_nums[i], 1)
            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)  # (dim, None, field_nums[0], field_nums[i])
            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])
            result_3 = tf.transpose(result_2, perm=[1, 0, 2])  # (None, dim, field_nums[0] * field_nums[i])
            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')
            result_5 = tf.transpose(result_4, perm=[0, 2, 1])  # (None, field_num[i+1], dim)
            hidden_layers_results.append(result_5)
        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)  # (None, H_1 + ... + H_K, dim)
        result = tf.reduce_sum(result,  axis=-1)  # (None, dim)
        return result


class LBA(Layer):
    def __init__(self, dnn_dropout=0.):
        """location-based attention layer. FISSA
        Args:
            dnn_dropout: A scalar.
        :return:
        """
        super(LBA, self).__init__()
        self.dropout = Dropout(dnn_dropout)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=[self.dim, self.dim],
            name='att_weights',
            initializer='random_normal')
        self.W_2 = self.add_weight(
            shape=[self.dim, self.dim],
            name='att_weights2',
            initializer='random_normal')
        self.q = self.add_weight(
            shape=[1, self.dim],
            name='query weight',
            initializer='random_normal')

    def call(self, inputs):
        k, v, mask = inputs  # (None, seq_len, dim), (None, seq_len, dim), (None, seq_len, 1)
        k = tf.matmul(k, self.W)  # (None, seq_len, dim)
        mat_qk = tf.matmul(k, self.q, transpose_b=True)  # (None, seq_len, 1)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, 1)
        # softmax
        outputs = tf.nn.softmax(logits=tf.transpose(outputs, [0, 2, 1]), axis=-1)  # (None, 1, seq_len)
        # output
        outputs = tf.squeeze(tf.matmul(tf.matmul(outputs, v), self.W_2), axis=1)  # (None, dim)
        outputs = self.dropout(outputs)
        return outputs


class Item_similarity_gating(Layer):
    def __init__(self, dnn_dropout=0.):
        """Item_similarity_gating, FISSA
        Args:
            dnn_dropout: A scalar.
        :return:
        """
        self.dropout = Dropout(dnn_dropout)
        super(Item_similarity_gating, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=[3 * self.dim, 1],
            name='att_weights',
            initializer='random_normal')

    def call(self, inputs):
        item_embed, global_info, candidate_embed = inputs
        inputs = tf.concat([item_embed, global_info, candidate_embed], -1)
        inputs = self.dropout(inputs)

        logits = tf.matmul(inputs, self.W)  # (None, neg_num + 1, 1)
        weights = tf.nn.sigmoid(logits)
        return weights


class CapsuleNetwork(Layer):
    def __init__(self, embed_dim, seq_len, bilinear_type=0, num_interest=4, stop_grad=True):
        super(CapsuleNetwork, self).__init__()
        self.bilinear_type = bilinear_type
        self.seq_len = seq_len
        self.num_interest = num_interest
        self.embed_dim = embed_dim
        self.stop_grad = stop_grad

    def build(self, input_shape):
        if self.bilinear_type >= 2:
            self.w = self.add_weight(
                shape=[1, self.seq_len, self.num_interest * self.embed_dim, self.embed_dim],
                initializer='random_normal',
                name='weights'
            )

    def call(self, hist_emb, mask):
        if self.bilinear_type == 0:
            hist_emb_hat = tf.tile(hist_emb, [1, 1, self.num_interest])  # (None, seq_len, num_inter * embed_dim)
        elif self.bilinear_type == 1:
            hist_emb_hat = Dense(self.dim * self.num_interest, activation=None)(hist_emb)
        else:
            u = tf.expand_dims(hist_emb, axis=2)  # (None, seq_len, 1, embed_dim)
            hist_emb_hat = tf.reduce_sum(self.w * u, axis=3)  # (None, seq_len, num_inter * embed_dim)
        hist_emb_hat = tf.reshape(hist_emb_hat, [-1, self.seq_len, self.num_interest, self.embed_dim])
        hist_emb_hat = tf.transpose(hist_emb_hat, [0, 2, 1, 3])  # (None, num_inter, seq_len, embed_dim)
        hist_emb_hat = tf.reshape(hist_emb_hat, [-1, self.num_interest, self.seq_len, self.embed_dim])
        if self.stop_grad:
            hist_emb_iter = tf.stop_gradient(hist_emb_hat)
        else:
            hist_emb_iter = hist_emb_hat  # (None, num_inter, seq_len, embed_dim)

        if self.bilinear_type > 0:
            self.capsule_weight = self.add_weight(
                shape=[tf.shape(hist_emb_hat)[0], self.num_interest, self.seq_len],
                initializer=tf.zeros_initializer()
            )
        else:
            self.capsule_weight = tf.random.truncated_normal(
                shape=[tf.shape(hist_emb_hat)[0], self.num_interest, self.seq_len],
                stddev=1.0)
        tf.stop_gradient(self.capsule_weight)

        for i in range(3):
            att_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])  # (None, num_inter, seq_len)
            paddings = tf.zeros_like(att_mask)

            capsule_softmax_weight = tf.nn.softmax(self.capsule_weight, axis=1)  # (None, num_inter, seq_len)
            capsule_softmax_weight = tf.where(tf.equal(att_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)  # (None, num_inter, 1, seq_len)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, hist_emb_iter)  # (None, num_inter, 1, embed_dim)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, keepdims=True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(hist_emb_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                self.capsule_weight = self.capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, hist_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.embed_dim])
        return interest_capsule






