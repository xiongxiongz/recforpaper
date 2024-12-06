import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.regularizers import l2
import numpy as np

if __name__ == '__main__':
    # 向量空间：item_embedding
    '''
    item_embedding = Embedding(input_dim=3,
                                    input_length=1,
                                    output_dim=2,
                                    embeddings_initializer='random_normal',
                                    embeddings_regularizer=l2(0.0))
    # 向量空间：pos_embedding
    pos_embedding = Embedding(input_dim=3,
                                   input_length=1,
                                   output_dim=2,
                                   embeddings_initializer='random_normal',
                                   embeddings_regularizer=l2(0.0))
    '''
    # a = tf.fill([3, 4], 3.)
    # b = tf.fill([3, 4], 2.)
    a = tf.fill([3, 2, 2], 3.)
    b = tf.fill([3, 1], 3.)
    batch_interest = tf.reduce_mean(a, axis=0)
    print("a:", a)
    print(batch_interest)
    print(a-batch_interest)

