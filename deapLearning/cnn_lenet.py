#-*- coding:utf-8 -*-
import os
import sys
import timeit
import gzip

from six.moves import urllib
import six.moves.cPickle as pickle

import numpy
import theano
import theano.tensor as T
from  theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

class LeNetConvPoolLayer(object):
	"""
	卷积层
	"""
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
		"""
		rng:随机数生成器
		input:输入的批量测试数据
		filter_shape:list of lenght 4，(number of filters, num input feature maps, filter height, filter width)
		image_shape: list of length 4, (batch size[表示批量的数据，也就是一次计算使用的图片数], num input feature maps, image height, image width)
		poolsize:池化的核大小
		"""
		assert image_shape[1] == filter_shape[1]
		self.input = input

		fan_in = numpy.prod(filter_shape[1:]) #本层每个输出神经元与多少个输入神经元相连 num input feature maps* filter height * filter width
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))  / numpy.prod(poolsize) #每个输入神经元可以从几个输出神经元得到误差加传"num output feature maps * filter height * filter width / pooling size" 
		W_bound = numpy.sqrt(6.0/(fan_in+fan_out))
		self.W = theano.shared(
			numpy.asarray(
				rng.uniform(
					low = -W_bound,
					height = W_bound,
					size = filter_shape), #(number of filters, num input feature maps, filter height, filter width)；每个单体是一个hieght * width大小的矩阵，表示一个核，每number input feature maps 个单体组成一个个体，表示所有输入特征图的一组核，而一共有number of filtes组；其实这个也可以看作是每列都代表了一个输出特征图的一组核
				dtype = theano.config.floatX
			),
			borrow = True
		)
		b_values = numpy.zeros((filter_shape[0], ), dtype=theano.config.floatX)
		self.b  = theano.shared(value = b_values, borrow=True)

		#卷积输出
		conv_out = conv2d(
			input = input,
			filters = self.W,
			filter_shape = filter_shape,
			input_shape = image_shape
		)

		#池化
		pooled_out = pool.pool_2d(
			input=conv_out,
			ds = poolsize,
			ignore_border=True
		)

		#计算的结果还要加上偏置，但是我们的输出pool_out是一个(batch size, output feature map, height, width)这样一个四维的向量，而b是一个一维的向量，所以要将b进行扩展成(1, output feature map, 1, 1),再利用numpy的broadcasted进行相加；则每个批量的用例中的每个特征图的值都会加上这个特征图对应的偏置了
		self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		self.params = [self.W, self.b]

		self.input = input


class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		"""
		全连接类似的隐藏层
		rng: 随机数生成器
		input:输入数据（输入神经元）
		n_in:输入神经元的个数
		n_out:输出神经元的个数
		activation:激活函数
		"""
		self.input = input
		#加入判断W是否已经初始化，主要是因为我们有时可以用已经训练好的一些权重进行再训练；
		if W is None:
			W_bound = numpy.sqrt(6.0 / (n_in + n_out)) #这个公式是使用本层的每个输出神经元与输入神经元的连接数（n_in）加上每个输入神经元可以从多少个输出神经元得到误差加值（n_out）的值来设定上界
			W_values = numpy.asarray(
				rng.uniform(
					low = -W_bound,
					high = W_bound,
					size=(n_in,n_out) #全连接，每个输出神经元都有一个n_in大小的权重向量 ；第一列表示一个神经元的权重; 用这种方式初始化而不是使用size=(n_in, n_out)，是因为后面要使用权重计算加权值，我们又要进行转置,如何这样初始化的话，我们可以直接就使用T.dot进行矩阵乘法
				),
				dtype=theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = numpy.zeros((n_out,),dtype=theano.config.floatX) #一个一维的向量，每一列表示一个神经元的偏置
			b = theano.shared(value=b_values, name="b", borrow=True)

		self.W=W
		self.b=b

		#计算加权值
		lin_output = T.dot(input, self.W) + self.b #每一行的数据代表了一个测试用例的输出神经元的值；再加各一维的偏置就可以得到加权值了
		self.output = (
			lin_output if activation is None else activation(lin_output)
		)
		self.params = [self.W, self.b]

class LogisticRegression(object):
	"""
	多类别分类模型，激活函数使用softmax,损失函数使用对应的负log似然函数来提高更新速度
	"""
	def __init__(self, input, n_in, n_out):
		"""
		input:
			input@array:the input data (from previsous layer) to train
			n_in@int:the numbel the input nerual(the input data), the x data
			n_out@int: the number of the output nerual(the output data) the predict y data
		"""
		#初始化权重和偏置
		self.W = theano.shared(
			value=numpy.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
			),
			name="W",
			borrow=True
		)
		self.b = theano.shared(
			value=numpy.zeros(
				(n_out,)
				dtype = theano.config.floatX
			),
			name="b",
			borrow=True
		)

		#softmax生成各概率的预测模型图
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		#取最大的概率对应的下标
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		#整合所有参数
		self.params = [self.W, self.b]
		self.input = input

	def negative_log_likeihood(self, y):
		"""
		使用了负对数似然，可以参考http://deeplearning.stanford.edu/wiki/index.php/Softmax_Regression
		在这里我们的目标是是去加上所有正确标签的概率，然后把取个平均；T.arange(y.shape[0],y)产生了一个[[0,y[0]], [1,y[1]]... [n-1,y[n-1]]的对，应用上T.log(self.p_y_given_x)它是一个k*n（k为类别数，n为用例数）的矩阵就可以得到所有T.log上猜对的概率组成的数组了；在numpy中a[1,2]表示取第1行第2个数据；
		"""
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

	def errors(self, y):
		"""
		用于打印一些错误信息
		"""
		#测试用例数不一样；
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shapes as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type)
			)

		if y.dtype.startswith('int'):
			#T.neq函数用于查找出两个对象之间不相符的数目；T.mean用于去除以个数，得到正确率
			return T.mean(T.neq(self.y_pred,y)) 
		else:
			raise NotImplementedError()


#help function
def shared_dataset(data_xy, borrow=True):
	"""
	将数据一次转换成shared variables,防止在进行训练时，每次都将一个批量的数据考进GPU中，因为这个过程是非常消耗时间的; borrow=True表示使用原始数据作为数据缓存；提取数据时不用给副本 ；提速用的
	"""	
	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x, 
						dtype=theano.config.floatX),
				borrow = borrow)
	shared_y = theano.shared(numpy.asarray(data_y, 
						dtype=theano.config.floatX),
				borrow = borrow)
	#当我们把数据存在在显存的时候，我们需要使用floats型 的，但是当我们计算的时候 ，我们需要使用整形，所以这里再把数据转成int32
	return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset='mnist.pkl.gz'):
	'''
	MNIST数据加载函数
	input:
		dataset @string:the path to the dataset
	output:
		rsl @a array with three share value
	'''
	data_dir, data_file = os.path.split(dataset)
	#load from local
	if data_dir == '' and not os.path.isfile(dataset):
		#check if the dataset is in the data directory
		new_path = os.path.join(
			os.path.split(__file__)[0],
			"..",
			"dataset",
			dataset
		)
		if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
			dataset = new_path
	#load from internet
	if (not os.path.isfile(dataset)) and data_file == "mnist.pkl.gz":
		origin = (
			'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'			
		)
		print('Downloading data from %s' % origin)
		urllib.request.urlretrieve(origin, dataset)

	print ('..loading data')

	with gzip.open(dataset, 'rb') as f:
		try:
			train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
		except:
			train_set, valid_set, test_set = pickle.load(f)

	#将数据转为shared_variable
	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	return [(train_set_x, train_set_y),
			(valid_set_x, valid_set_y),
			(test_set_x, test_set_y)]

print load_data()