# -*- coding: utf-8 -*-  
#使用theano实现简单的线性回归模型

import numpy
import theano
import theano.tensor as T
rng = numpy.random  # 随机数生成器


# help function
# 数据生成,返回一个数组，第一个是x数据矩阵，每行表示一个数据，第二个是y数组
def genData(N,feats):
	return (rng.randn(N,feats), 1+rng.randn(N))

#定义全局使用的参数
N = 500 #样本数
feats = 7 #特征数
train_loop = 1000 #测试的循环次数
a = 0.1
lamb =  0.1

#定义需要用到的符号
X = T.dmatrix('X') #数据矩阵
Y = T.dvector('Y') #标签数组

#定义并初始化权重向量和偏置
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0.0, name="b")

#构建我们的expression graph
Y_pred = T.dot(X, w) - b #预测结果
prediction = Y_pred
cost = T.mean(T.sqr(Y_pred-Y)) #损失函数
#使用正则化的话如下
#cost = T.mean(T.sqr(Y_pred-Y) + lamb * (w**2).sum())

gw, gb = T.grad(cost, [w,b]) #计算导数

#Compile
linear_regression_model = theano.function(
	inputs = [X,Y],
	outputs = [prediction, cost],
	updates = ((w, w-a*gw), (b, b-a*gb)))

predict = theano.function(
	inputs = [X],
	outputs = prediction)

# Train
D = genData(N,feats)
for i in range(train_loop):
	pred, cost = linear_regression_model(D[0], D[1])
	print cost

result = predict(D[0])
print result - D[1]
print result .shape
print D[1].shape
