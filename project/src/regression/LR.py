# -*- coding: utf-8 -*-

import csv
import random
import numpy as np
rng = np.random  # 随机数生成器

def  getData(filename):
	f = file(filename, 'rb')
	reader = csv.reader(f)
	count = False
	data = []
	#获取全部数据
	for line in reader:
		if count == False:
			count = True
			continue
		items = line
		size = len(items)
		temp  = [float(item) for item in items[1:size-1]]
		values = np.array([1] + temp) #在每个数据前面加上1,表示x0 = 1
		lable = float(items[-1])
		data.append((values, lable))

	#为了测试方便train_data只取10000个，test_data取100个;当然实际训练的时候，要取更多
	train_data =  data[0:1000]
	test_data = data[10000:10010]
	print  '加载数据完毕'
	return train_data, test_data

class LinearRegression(object):
	def __init__(self,dim,alpha):
		"""
		dim表示输入数据有多少维
		alpha 表示学习率
		"""
		#self.theta =  rng.randn(dim) #参数初始化
		self.theta = rng.randn(dim)
		self.dim = dim
		self.alpha = alpha

	def watch(self, i,train_data, test_data = None):
		#  每40轮查看一下损失值，方便监控，比如调学习率
		if i % 40 == 0:
			print 'times', i
			print ' cost:', self.cost(train_data)

		# 每500轮对测试数据进行测试一下
		if  i % 500 == 0:
			if test_data:
				print "Time {0}:{1}".format(i, self.evaluate(test_data))
			else:
				print 'complete {0}'.format(i)		

	def BGD_train(self, times, train_data, test_data = None):
		if test_data:
			n_test = len(test_data)
		m = len(train_data)
		for i in range(times):
			self.b_update(train_data)
			#监控与测试
			self.watch(i, train_data, test_data)

	def SGD_train(self, times, train_data, test_data = None):
		if test_data:
			n_test = len(test_data)
		random.shuffle(train_data) #一次进行混乱，来模拟每次都是随机的效果
		m = len(train_data)
		for i in range(times):
			count = 0
			for item in train_data:
				self.s_update(item)
				#监控与测试
				count = count + 1
				self.watch(i * m + count, train_data, test_data)			

	def MBGD_train(self, epochs, mini_batch_size, train_data, test_data = None):
		if test_data: 
			n_test = len(test_data)
		m = len(train_data)
		for j in range(epochs):
			random.shuffle(train_data)
			mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, m, mini_batch_size)] #对数据进行分割，得到所有的batch
			count = 0
			mini_batches_num = len(mini_batches)
			for mini_batch in mini_batches:
				self.mb_update( mini_batch)
				#监控与测试
				count = count + 1
				self.watch(j*mini_batches_num + count, train_data, test_data)	


	def b_update(self, batch):
		m = len(batch)
		#方法1：直接按公式的做法；因为需要反复计算self.hypothesis(x) - y ，所以比较慢
		"""
		for j in range(self.dim):
			partial = 0
			for x, y in batch:
				partial += (self.hypothesis(x) - y)*x[j]
			partial = 1.0 / m * partial
			self.theta[j] = self.theta[j] - self.alpha * partial
		"""

		#方法2：将公式转成矩阵计算，提高速度
		diff_sum  = [self.hypothesis(x) - y for x, y  in batch] #缓存全部测试用命的差值,结果是一个m维的向量
		diff_sum =  np.array(diff_sum)#转为numpy数组
		X = [[] for i in range(self.dim)] #建立一个dim维的的数组，每个元素也是一个数组，用于分别存放全部数据的每一列的元素
		
		for x, y  in batch:
			for index in range(len(x)):
				X[index].append(x[index])
		self.theta =  [theta_j - self.alpha / m * np.dot(diff_sum , x_j) for theta_j, x_j in zip(self.theta, X)]
		self.theta = np.array(self.theta)

	def s_update(self, data):
		x = data[0]
		y = data[1]
		for j in range(self.dim):
			partial = (self.hypothesis(x) - y) * x[j]
			self.theta[j] = self.theta[j] - self.alpha * partial

	def mb_update(self, mini_batch):
		self.b_update(mini_batch)

	def evaluate(self, test_data):
		"""
		输出测试数据的差值平方
		"""
		return self.cost(test_data)

	def cost(self, train_data):
		cost = 0
		size = len(train_data)
		for x ,y in train_data:
			diff = self.hypothesis(x) - y
			cost += diff * diff
		cost  =  1.0 / (2*size) * cost
		return cost

	def hypothesis(self, x):
		return np.dot(x, self.theta)

	def reset(self):
		self.theta = rng.randn(dim)

if __name__ == '__main__':
	train_data, test_data = getData('../../data/regression/train.csv')
	dim = len(train_data[0][0])
	alpha = 0.08
	model = LinearRegression(dim, alpha)
	#model.BGD_train(3000, train_data, test_data)
	#model.SGD_train(3000,train_data, test_data)
	model.MBGD_train(300, 100, train_data, test_data)