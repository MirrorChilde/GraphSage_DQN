import networkx as nx
import numpy as np
from tensorflow.keras.utils import Sequence


def sampling(src_nodes, sample_num, neighbor_table, seed=1):
	'''根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
	某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节
	参数:
		- src_nodes {list, ndarray} -- 源节点列表
		- sample_num {int} -- 需要采样的节点数
		- neighbor_table {dict} -- 节点到其邻居节点的映射表
		- seed：随机种子
	输出:
		np.ndarray -- 采样结果构成的列表
	'''
	results = []
	np.random.seed(seed)
	for sid in src_nodes:
		# 从节点的邻居中进行有放回地进行采样
		res = np.random.choice(neighbor_table[sid], size=(sample_num,))
		results.append(res)
	return np.asarray(results).flatten()

def multihop_sampling(src_nodes, sample_nums, neighbor_table, seed):
	'''根据源节点进行多阶采样

	参数:
		- src_nodes {list, np.ndarray} -- 源节点id
		- sample_nums {list of int} -- 每一阶需要采样的个数
		- neighbor_table {dict} -- 节点到其邻居节点的映射
		- seed：随机种子
	输出:
		[list of ndarray] -- 每一阶采样的结果
	'''
	sampling_result = [src_nodes]
	for k, hopk_num in enumerate(sample_nums):
		hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table, seed)
		sampling_result.append(hopk_result)
	return sampling_result

def GraphSAGE_DataGenerator_method(link_state,adj_matrix,sample_nums,indexes=None):
	'''针对GraphSAGE模型的数据数据生成器
	参数：
		- x_set：特征矩阵
		- adj_matrix：邻接矩阵
		- neighbor_batch_size：批尺寸
		- sample_nums：各阶邻居的采样数
		- indexes=None：采样的索引序列
		- seed=1：shuffle的随机种子
	'''

	sample_idx_list = np.array(indexes) if indexes else np.arange(len(link_state))
	g = nx.from_numpy_array(adj_matrix)
	neighbor_table = {n: list(g.neighbors(n)) for n in g.nodes}

	sample_result = multihop_sampling(sample_idx_list,sample_nums,neighbor_table,1)
	batch_x = []
	for arr in sample_result:
		neighbor_features = link_state[arr]
		# neighbor_features = np.array(np.split(neighbor_features, 1))  # 根据batch_size切分node_features
		batch_x.append(neighbor_features)
	return batch_x

class GraphSAGE_DataGenerator(Sequence):
	'''针对GraphSAGE模型的数据数据生成器
	参数：
		- x_set：特征矩阵
		- adj_matrix：邻接矩阵
		- neighbor_batch_size：批尺寸
		- sample_nums：各阶邻居的采样数
		- indexes=None：采样的索引序列
		- seed=1：shuffle的随机种子
	'''

	def __init__(
		self,
		link_state,
		adj_matrix,
		sample_nums,
		indexes=None,
	):
		self.x = link_state
		self.sample_nums = sample_nums
		self.indexes = np.array(indexes) if indexes else np.arange(len(self.x))
		# assert len(self.indexes) >= self.batch_size

		g = nx.from_numpy_array(adj_matrix)
		self.neighbor_table = {n: list(g.neighbors(n)) for n in g.nodes}

	# def __len__(self):
	# 	return int((len(self.indexes) / self.batch_size))

	def call(self, idx):
		sample_idx_list = self.indexes
		sample_result = multihop_sampling(
			sample_idx_list,
			self.sample_nums,
			self.neighbor_table,
			1
		)
		batch_x = []
		for arr in sample_result:
			neighbor_features = self.x[arr]
			# neighbor_features = np.array(np.split(neighbor_features, self.neighbor_batch_size))  # 根据batch_size切分node_features
			batch_x.append(neighbor_features)
		return batch_x
