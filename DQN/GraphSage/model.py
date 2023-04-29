import tensorflow as tf
from tensorflow import keras
from keras import regularizers

class myModel(tf.keras.Model):
    def __init__(self, hparams):
        super(myModel, self).__init__()
        self.hparams = hparams

        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            activation=tf.nn.selu, name="FirstLayer"))

        self.GraphSage = tf.keras.models.Sequential()
        self.GraphSage.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                          activation=tf.nn.selu,
                                          kernel_regularizer=regularizers.l2(hparams['l2']),
                                          name="Sage1"))
        self.GraphSage.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.GraphSage.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                          activation=tf.nn.selu,
                                          kernel_regularizer=regularizers.l2(hparams['l2']),
                                          name="Sage2"))
        self.GraphSage.add(keras.layers.Dropout(rate=hparams['dropout_rate']))

        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(1, kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout3"))

    def build(self, input_shape=None):
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.GraphSage.build(input_shape=[None, self.hparams['link_state_dim']*2])
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    @tf.function
    def call(self,
             states_action,
             states_graph_ids,
             states_first,
             states_second,
             sates_num_edges,
             training=False):
        # Define the forward pass
        link_state = states_action

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            ### 1.a Message passing for link with all it's neighbours
            outputs = self.Message(edgesConcat)

            ### 1.b Sum of output values according to link id index
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)

            ### 2. Update for each link
            sage_outputs = self.GraphSage(tf.concat([link_state, edges_inputs], axis=1))

            link_state = sage_outputs

        # Perform sum of all hidden states
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)

        r = self.Readout(edges_combi_outputs,training=training)
        return r
