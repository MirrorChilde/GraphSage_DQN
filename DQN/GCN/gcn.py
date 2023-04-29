import tensorflow as tf
from tensorflow import keras
from keras import regularizers

class GCN(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.w = self.add_weight(shape=(input_dim, output_dim),
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 trainable=True,
                                 name="w")
        self.b = self.add_weight(shape=(output_dim,),
                                 initializer=tf.keras.initializers.Zeros(),
                                 trainable=True,
                                 name="b")

    def call(self, x, adj):
        # x shape: (num_edges, 2 * link_state_dim)
        # adj shape: (num_edges,)
        h = tf.nn.relu(tf.matmul(x, self.w) + self.b)
        h = tf.math.unsorted_segment_sum(h, adj, tf.reduce_max(adj) + 1)
        h = tf.nn.dropout(h, rate=self.dropout)
        return h

class myModel(tf.keras.Model):
    def __init__(self, hparams):
        super(myModel, self).__init__()
        self.hparams = hparams

        # Define layers here
        self.Message = tf.keras.layers.Dense(self.hparams['link_state_dim'], activation=tf.nn.selu,
                                             name="Message")
        self.GCN = GCN(self.hparams['link_state_dim'], self.hparams['link_state_dim'], dropout=hparams['dropout_rate'])
        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'], activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'], activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(1, kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout3"))

    def call(self, states_action, states_graph_ids, states_first, states_second, sates_num_edges, training=False):
        # Define the forward pass
        link_state = states_action

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            # Message passing
            outputs = self.Message(edgesConcat)
            # GCN
            link_state = self.GCN(outputs, states_second)

        # Perform sum of all hidden states
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)

        r = self.Readout(edges_combi_outputs,training=training)
        return r

