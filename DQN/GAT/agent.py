import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import warnings
from .networks import *
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]

        # self.output_layer = layers.Dense(output_dim)
        # self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)
        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(35,
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(0.1),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=0.01))
        self.Readout.add(keras.layers.Dense(35,
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(0.1),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=0.01))
        self.Readout.add(keras.layers.Dense(1, kernel_regularizer=regularizers.l2(0.1),
                                            name="Readout3"))



    def call(self, states_action, states_graph_ids, states_first, states_second, ordered_edges, training=False):

        link_state = states_action # (84,20)

        states_first = tf.reshape(states_first,[1,-1])
        states_second = tf.reshape(states_second,[1,-1])
        edges = tf.concat([states_first,states_second],axis=0)
        edges = tf.transpose(edges,[1,0])

        x = self.preprocess(link_state)
        for attention_layer in self.attention_layers:
            x = attention_layer([link_state,edges]) + x

        # 读出层
        edges_combi_outputs = tf.math.segment_sum(x, states_graph_ids, name=None)
        r = self.Readout(edges_combi_outputs,training=training)

        return r

    @staticmethod
    def _get_specific_number_weights(model):
        weights = model.get_weights()
        layer_dimensions = [(w.shape, w.size) for w in weights]
        return layer_dimensions, sum(w[1] for w in layer_dimensions)

    def get_message_number_weights(self):
        return self._get_specific_number_weights(self.Message)

    def get_update_number_weights(self):
        return self._get_specific_number_weights(self.Update)

    def get_message_update_number_weights(self):
        message_layer_dimensions, message_number_params = self._get_specific_number_weights(self.Message)
        update_layer_dimensions, update_number_params = self._get_specific_number_weights(self.Update)
        return message_layer_dimensions + update_layer_dimensions, message_number_params + update_number_params

    def get_readout_number_weights(self):
        return self._get_specific_number_weights(self.Readout)

    def get_number_weights(self):
        return self._get_specific_number_weights(super(GraphAttentionNetwork, self))

    @staticmethod
    def _get_specific_weights(model):
        weights = model.get_weights()
        for w in range(len(weights)):
            weights[w] = np.reshape(weights[w], (weights[w].size,))
        return np.concatenate(weights)

    def get_message_weights(self):
        return self._get_specific_weights(self.Message)

    def get_update_weights(self):
        return self._get_specific_weights(self.Update)

    def get_message_update_weights(self):
        return np.concatenate((self._get_specific_weights(self.Message), self._get_specific_weights(self.Update)))

    def get_readout_weights(self):
        return self._get_specific_weights(self.Readout)

    def get_weights(self):
        return self._get_specific_weights(super(GraphAttentionNetwork, self))

    @staticmethod
    def _set_weights(model, new_weights):
        weights = model.get_weights()
        layer_dimensions = [(w.shape, w.size) for w in weights]

        transformed_weights = []
        current_idx = 0
        for layer_shape, layer_size in layer_dimensions:
            layer_weights = np.reshape(new_weights[current_idx:current_idx + layer_size], layer_shape)
            transformed_weights.append(layer_weights)
            current_idx += layer_size

        model.set_weights(transformed_weights)

    def set_message_weights(self, new_weights):
        self._set_weights(self.Message, new_weights)

    def set_update_weights(self, new_weights):
        self._set_weights(self.Update, new_weights)

    def set_message_update_weights(self, new_weights):
        _, message_number_params = self.get_message_number_weights()
        self._set_weights(self.Message, new_weights[:message_number_params])
        self._set_weights(self.Update, new_weights[message_number_params:])

    def set_readout_weights(self, new_weights):
        self._set_weights(self.Readout, new_weights)

    def set_weights(self, new_weights):
        self._set_weights(super(GraphAttentionNetwork, self), new_weights)