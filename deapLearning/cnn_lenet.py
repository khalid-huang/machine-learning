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

#layer model
class LeNetConvPoolLayer(object):
	"""
	卷积层
	"""
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
		"""
		rng:随机数生成器
		input:输入的批量测试数据
		filter_shape:list of lenght 4，(number of filters, num input feature maps, filter height, filter width); 这里要注意 的number of filters表示的是这个层有多少个输出特征图，也就是说每个输入特征图会有多少个卷积核；而不是这个层有的全部的卷积核数
		image_shape: list of length 4, (batch size[表示批量的数据，也就是一次计算使用的图片数], num input feature maps, image height, image width)
		poolsize:池化的核大小
		"""
		print image_shape[1], filter_shape[1]
		assert image_shape[1] == filter_shape[1]
		self.input = input

		fan_in = numpy.prod(filter_shape[1:]) #本层每个输出神经元与多少个输入神经元相连 num input feature maps* filter height * filter width
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))  / numpy.prod(poolsize) #每个输入神经元可以从几个输出神经元得到误差加传"num output feature maps * filter height * filter width / pooling size" 
		W_bound = numpy.sqrt(6.0/(fan_in+fan_out))
		self.W = theano.shared(
			numpy.asarray(
				rng.uniform(
					low = -W_bound,
					high= W_bound,
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
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

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
				(n_out,),
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
			#T.neq函数用于查找出两个对象之间不相符的数目；T.mean用于去除以个数，得到错误率
			return T.mean(T.neq(self.y_pred,y)) 
		else:
			raise NotImplementedError()

#network model
class simpleLenet(object):
	def __init__(self,learning_rate =0.1, n_epochs=200,
			dataset="mnist.pkl.gz",
			nkerns=[20,50], batch_size=500):
		"""
		根据传入的参数进行模型的初始化
		learning_rate@float:学习率
		n_epochs@int:表示进行批量迭代的次数
		dataset@string:数据集的路径
		nkerns@list:每个卷积-池化层使用在每个输入特征图的核的个数(也就是输出特征图的个数)；所以第n层会用到的总的核的个数是the number of input feature maps * the number of kernes(也就是上面传入的nkerns[n],n表示第几层)
		"""	
		print "...init model"
		self.rng = numpy.random.RandomState(23455)
		self.nkerns = nkerns
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs

		#测试用
		#self.n_epochs = 10
		#self.batch_size = 10

		#加载数据
		datasets = load_data(dataset)
		self.train_set_x, self.train_set_y = datasets[0] 
		self.valid_set_x,self.valid_set_y = datasets[1]
		self.test_set_x, self.test_set_y = datasets[2]
		#根据有的batch_size计算总共有多少个batch块
		n_train_number = self.train_set_x.get_value(borrow=True).shape[0]
		n_valid_number = self.valid_set_x.get_value(borrow=True).shape[0]
		n_test_number = self.test_set_x.get_value(borrow=True).shape[0]
		self.n_train_batches = n_train_number // self.batch_size #进行floor 除法
		self.n_valid_batches = n_valid_number // self.batch_size
		self.n_test_batches = n_test_number // self.batch_size

		#测试用
		#self.n_train_batches = 100
		#self.n_valid_batches = 10
		#self.n_test_batches = 10

		#建立一些符号
		self.index = T.lscalar() #index to a batch

		self.x = T.matrix('x') #x表示一个矩阵，每行代表了一个数据用例
		self.y = T.ivector('y') #用一个向量表示结果，每个值表示一个label

	def build(self):
		"""
		建立网络模型图
		"""
		print "..build lenet model"
		#我们要将我们数据进行维数变换；因为我们的输入数据是一个n*784的类型的矩阵，n表示行数；所以 我们需要把它变成一个n*1*28*28的，转成theano处理的4D数据；第一个n表示有多少个用例，也就是我们的batch size;第2个表示每个用例有一个特征图；28×28表示每个特征图的height * width
		x = self.x
		y = self.y
		index = self.index
		batch_size = self.batch_size
		nkerns = self.nkerns
		rng = self.rng
		learning_rate = self.learning_rate

		layer0_input = x.reshape((batch_size, 1, 28, 28))

		#第一个卷积池化层
		layer0 = LeNetConvPoolLayer(
			rng,
			input=layer0_input,
			image_shape=(batch_size,1,28,28),
			filter_shape=(nkerns[0],1,5,5),
			poolsize=(2,2)
		)

		#第二个卷积池化层
		#image_shape的设置为首先每个迭代有batch_size个用例，本轮的每个图片输入对应有nkers[0]个输入特征图；大小为(28-5+1) / 2
		layer1 = LeNetConvPoolLayer(
			rng,
			input=layer0.output,
			image_shape=(batch_size, nkerns[0],12,12),
			filter_shape=(nkerns[1], nkerns[0], 5,5),
			poolsize=(2,2)
		)
		#因为 后面要接下全连接层了，而全连接层的输入与卷积-池化是不一样的，它是一个类似n*m的两维数据，n表示用例数，而m表示每个用例有多少维数据（多少个x）;所以我们要把layer1的输出batch_size*nkers[1]*((12-5+1)/2))*((12-5+1)/2))这样的一个四维数据变成一个batch_size * (nkers[1]*((12-5+1)/2))*((12-5+1)/2)))这样一个两维数据；也就是将所有的输出特征图都连成一个一维的数据
		layer2_input = layer1.output.flatten(2)

		layer2 = HiddenLayer(
			rng,
			input=layer2_input,
			n_in = nkerns[1] * 4 * 4, #每个输入数据有the previous layer's numpy of output feature map * height * width 个输入神经元
			n_out = 500,
			activation = T.tanh
		)

		#分类器，使用softmax的logistic回归模型
		layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

		#测试模型
		self.test_model = theano.function(
			[index],
			layer3.errors(y),
			givens={
				x:self.test_set_x[index * batch_size:(index+1)*batch_size], #根据batch_size来进行选择数据
				y:self.test_set_y[index * batch_size:(index+1)*batch_size]
			}
		)

		#校验模型
		self.validation_model = theano.function(
			[index],
			layer3.errors(y),
			givens={
				x:self.valid_set_x[index * batch_size:(index+1)*batch_size], #根据batch_size来进行选择数据
				y:self.valid_set_y[index * batch_size:(index+1)*batch_size]
			}
		)

		#对应的NLL损失函数(负对数似然)
		cost = layer3.negative_log_likeihood(y)
		params = layer3.params + layer2.params + layer1.params + layer0.params
		#得到偏导
		grads = T.grad(cost, params)
		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		]

		self.train_model = theano.function(
			[index],
			cost,
			updates = updates,
			givens = {
				x:self.train_set_x[index*batch_size:(index+1)*batch_size],
				y:self.train_set_y[index*batch_size:(index+1)*batch_size]
			}
		)

	def train(self):
		"""
		进行训练
		"""
		print "... train model"
		#用以测试效果的，主要是减少
		#patience = 100
		
		patience = 10000 #做一个监控，是一个最大的可容忍的训练次数值; 它的值会更新；它的作用是，如果在迭代的过程中，出现了最好结果的话，就在patience和 patience_increase * iter之间选择一个最大値，然后在本轮迭代结束的时候，比较下patience 是否比iter小；如果小的话就则后续的迭代都不做了；可以看出这样的使用是在保证迭代一定做了patience / patience_increase的情况下，同时还根据出现最好的结果时提前结束迭代，也就是不做后续迭代
		patience_increase = 2 
		improvement_threshold = 0.995 #
		validation_frequency = min(self.n_train_batches, patience // 2)

		best_validation_loss = numpy.inf #记录最好的迭代结果的出现，也就是Loss最小的时候
		best_iter = 0 #对应的出现第几个批量迭代
		test_score = 0
		start_time = timeit.default_timer() #用于计时

		#关于迭代有一个说明，就是每个迭代的epoch(周期)都会用上所有的batches块，也就是说每个epoch有batches_number次iter(batches_number表示批量块的个数，它的值是train_number//batch_size)
		epoch = 0 #记录迭代的周期数
		done_looping = False #用于结束迭代的策略之一

		while (epoch < self.n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in range(self.n_train_batches):
				iter = (epoch - 1) * self.n_train_batches + minibatch_index #计算目前已经迭代了几次
				if iter % 10 == 0:
					print('training @ iter =  %i' % iter)
				cost_ij = self.train_model(minibatch_index)
				#判断是否需要进行校验
				if(iter + 1) % validation_frequency == 0:
					validation_losses = [self.validation_model(i) for i in range(self.n_valid_batches)]
					this_validation_loss = numpy.mean(validation_losses)
					#打印错误率，错误率是估计错误的结果除以全部的用例数
					print(('epoch %i,minibatch %i/%i, validation error rate %f %%') %
						(epoch, minibatch_index + 1, self.n_train_batches, this_validation_loss * 100.))
					#如果得到了一个目前最好的结果的话，首先就更新我们的best_validation_loss和best_iter，同时在测试集上进行测试；如果有本次的测试结果比上一次最好的imprement_threshold倍还好的话，就提高patience; 
					if this_validation_loss < best_validation_loss:
						if this_validation_loss < best_validation_loss * improvement_threshold:
							patience = max(patience, iter * patience_increase)

						best_validation_loss = this_validation_loss
						best_iter = iter

						test_losses =  [
							self.test_model(i) for i in range(self.n_test_batches)
						]
						test_score = numpy.mean(test_losses)
						print(('  epoch %i, minibatch %i/%i, test error rate of best model %f %%') %
							(epoch, minibatch_index + 1, self.n_train_batches, test_score * 100.))

				if patience <= iter:
					done_looping = True
					break

		end_time = timeit.default_timer()
		print 'Optimization complete' 
		print ('Best validation score of %f %% obtained at iteration %i, with test performance %f %%' % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
		print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)))


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
		rsl @a array with three share value; 每行表示一个数据集
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
	#每个x都是一个矩阵，每行表示一个数据用例；y是一个一维向量，每个元素表示一个用例的label
	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	return [(train_set_x, train_set_y),
			(valid_set_x, valid_set_y),
			(test_set_x, test_set_y)]

#test function
#test lenet
def experiment_lenet():
	lenet = simpleLenet()
	lenet.build()
	lenet.train()

experiment_lenet()
#print load_data()