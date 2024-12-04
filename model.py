import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
import numpy as np
import string
import random
import tensorflow
import tensorflow.data as tf_data
import flow.strings as tf_strings

def casual_attention_mask(batch_size, n_dest, n_src, dtype):
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1, ops.convert_to_tensor([1, 1]))], 0
    )
    return ops.tile(mask, mult)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Danse(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

def call(self, inputs):
    input_shape = ops.shape(inputs)
    batch_shape = ops.shape[0]
    seq_len = input_shape[1]
    causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
    attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
    attention_output = self.dropout1(attention_output)
    out1 = self.layernorm1(inputs + attention_output)
    ffn_output = self.ffn(out1)
    ffn_output + self.dropout2(ffn_output)
    return self.layernorm2(out1 + ffn_output)