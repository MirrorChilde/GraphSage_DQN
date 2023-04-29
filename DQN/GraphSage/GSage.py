# coding=gbk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.layers import Layer, Input
from keras.models import Model
import keras.activations as activations
import keras.backend as K
import numpy as np
import networkx as nx
from .minibatch import *

tf.config.experimental_run_functions_eagerly(True)

class myModel(tf.keras.Model):
	def __init__(self,
				 out_dim,
				 aggr_method,
				 activation_rounds,
				 use_bias,
				 sample_nums, # GraphSAGE各阶邻居采样数
				 **kwargs,):
		super().__init__(**kwargs)
		self.out_dim=out_dim
		self.aggr_method=aggr_method
		self.activation_rounds=activation_rounds
		self.use_bias=use_bias
		self.rounds = len(activation_rounds)
		self.sample_nums=sample_nums
		self.aggregators = [
			Aggregator(
				output_dim=self.out_dim[i],
				aggr_method=self.aggr_method,
				activation=self.activation_rounds[i],
				use_bias=self.use_bias
			) for i in range(self.rounds)
		]
		# self.aggregators = Aggregator(
		# 	output_dim=self.out_dim[1],
		# 	aggr_method=self.aggr_method,
		# 	activation=self.activation_rounds[0],
		# 	use_bias=self.use_bias
		# 	)

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
		link_state = states_action  # (84,20)
		# 获取邻接矩阵
		states_first = tf.reshape(states_first, [1, -1])
		states_second = tf.reshape(states_second, [1, -1])
		edges = tf.concat([states_first, states_second], axis=0)
		edges = tf.transpose(edges, [1, 0])
		edgesList=edges.numpy()
		G = nx.Graph()
		G.add_edges_from(edgesList)
		adj_matrix=nx.adjacency_matrix(G).todense()

		# 1.采样邻居列表
		neighbors_list = GraphSAGE_DataGenerator_method(
			link_state=link_state.numpy(),
			adj_matrix=adj_matrix,
			sample_nums=self.sample_nums
		)
		# mainEdges = tf.gather(link_state, states_first)
		# firstNeighEdges = tf.gather(link_state, states_second)
		# secondNeighEdges = tf.gather(link_state, states_second)

		# 2.一轮两层聚合
		# hidden=tf.convert_to_tensor(neighbors_list)
		hidden=tuple(neighbors_list)
		# next_hidden=[]
		# aggregator = self.aggregators
		# for i in range(self.rounds):
		# 	src_nodes = tf.convert_to_tensor(hidden[0])
		# 	neighbor_nodes =tf.convert_to_tensor(hidden[i+1])
		# 	aggr_nodes = aggregator([src_nodes, neighbor_nodes])
		# 	next_hidden.append(aggr_nodes)
		# hidden = next_hidden
		for r in range(self.rounds):
			next_hidden=[]
			aggregator = self.aggregators[r]
			for i in range(self.rounds-r):
				src_nodes = tf.convert_to_tensor(hidden[0])
				neighbor_nodes =tf.convert_to_tensor(hidden[i+1])
				aggr_nodes = aggregator([src_nodes, neighbor_nodes])
				next_hidden.append(aggr_nodes)
			hidden = next_hidden
		# 3.读出层
		edges_combi_outputs = tf.math.segment_sum(hidden[0], states_graph_ids, name=None)
		r = self.Readout(edges_combi_outputs, training=training)

		return r


class Aggregator(Layer):
    '''聚合函数
    参数：
        - outout_dim：输出维度
        - aggr_method：聚合方法，'mean'：平均聚合，'gcn'：类GCN聚合，'pooling'：max pooling聚合
        - activation：激活函数
        - use_bias：是否使用偏置
    输入：
        - [
        目标节点特征, (n, dim)，目标节点数为n
        邻居节点特征, (n*k, dim)，每个目标节点的邻节点数为k
        ]
    输出：
        - 更新后的目标节点特征， (n, outout_dim)
    '''
    def __init__(
        self,
        output_dim,
        aggr_method,
        activation,
        use_bias=True,
        **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.aggr_method = aggr_method
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        if self.aggr_method in ['mean', 'pooling']:
            self.w = self.add_weight(
                name = 'w',
                shape=(2*input_dim, self.output_dim),
                initializer = 'glorot_uniform',
                )
            if self.aggr_method == 'pooling':
                self.w_pool = self.add_weight(
                    name = 'w_pool',
                    shape=(input_dim, input_dim),
                    initializer = 'glorot_uniform',
                    )
                self.bias_pool = self.add_weight(
                    name = 'bias_pool',
                    shape=(input_dim, ),
                    initializer = 'zero',
                    )
        if self.aggr_method in ['gcn']:
            self.w = self.add_weight(
                name = 'w',
                shape=(input_dim, self.output_dim),
                initializer = 'glorot_uniform',
                )
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape=(self.output_dim,),
                initializer = 'zero',
                )

    def call(self, inputs):
        src_features = inputs[0]
        neighbor_features = inputs[1]

        # batch = K.shape(neighbor_features)[0]
        num = K.shape(src_features)[0]
        k = K.shape(neighbor_features)[0] // num
        input_dim = K.shape(src_features)[-1]
        neighbor_target_shape = ( num, k, input_dim)

        if self.aggr_method == 'mean':
            # (batch, num, k, input_dim)
            neighbor_features = K.reshape(neighbor_features, neighbor_target_shape)
            # (batch, num, input_dim)
            neighbor_features = K.mean(neighbor_features, axis = 1)
            # (batch, num, 2*input_dim)
            src_features = K.concatenate([src_features, neighbor_features])
            # (batch, num, output_dim)
            src_features = K.dot(src_features, self.w)

        elif self.aggr_method == 'gcn':
            # (batch, num, 1, input_dim)
            src_features = K.expand_dims(src_features, axis=2)
            # (batch, num, k, input_dim)
            neighbor_features = K.reshape(neighbor_features, neighbor_target_shape)
            #(batch, num, k+1, input_dim)
            src_features = K.concatenate([src_features, neighbor_features], axis = 2)
            #(batch, num, input_dim)
            src_features = K.mean(src_features, axis = 2)
            # (batch, num, output_dim)
            src_features = K.dot(src_features, self.w)

        elif self.aggr_method == 'pooling':
            # (batch, num*k, input_dim)
            neighbor_features = K.dot(neighbor_features, self.w_pool)
            # (batch, num*k, input_dim)
            neighbor_features = K.bias_add(neighbor_features, self.bias_pool)
            # (batch, num*k, input_dim)
            neighbor_features = self.activation(neighbor_features)
            # (batch, num, k, input_dim)
            neighbor_features = K.reshape(neighbor_features, neighbor_target_shape)
            # (batch, num, input_dim)
            neighbor_features = K.max(neighbor_features, axis=1)
            # (batch, num, 2*input_dim)
            src_features = K.concatenate([src_features, neighbor_features])
            # (batch, num, output_dim)
            src_features = K.dot(src_features, self.w)

        if self.use_bias:
            src_features = K.bias_add(src_features, self.bias)

        return self.activation(src_features)
