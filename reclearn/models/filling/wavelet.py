import tensorflow as tf


class WaveletConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_sizes, dilation_steps, strides=1, activation='relu', **kwargs):
        super(WaveletConv1D, self).__init__(**kwargs)
        self.convs = [tf.keras.layers.Conv1D(filters=filters,
                                             kernel_size=k,
                                             dilation_rate=d,
                                             strides=strides,
                                             padding='same',
                                             activation=activation)
                      for k, d in zip(kernel_sizes, dilation_steps)]

    def call(self, inputs):
        outputs = [conv(inputs) for conv in self.convs]
        return tf.concat(outputs, axis=-1)  # Concatenate features from different kernel sizes