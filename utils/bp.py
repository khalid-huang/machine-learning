# -*- coding: utf-8 -*- 

"""
使用SGD对一个前馈弄的神经网络进行学习；梯度利的计算使用反向传播算法
"""
#https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
import random


import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        sizes参数是一个列表，表示了网络的结构，比如[2,3,1]代表了这是一个三层的神经网络，其中第一层有2个神经元，第二层有3个神经元，第三层有1个神经元
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # 偏置数组
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] #权重矩阵数组，第一个层都有一个权重矩阵

    def feedforward(self, a):
        """
        前向传播阶段
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        使用mini-batch SGD进行训练；其中training_data是我们的训练数据格式为(x1,x2,x3, ..., y)；
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data) #进行数据混乱，提升效果
            mini_batches = [
                training_data[k:k+mini_batch_size] # 将数据切割成多个mini-batches
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        使用梯度下降进行更新；梯度利用反向传播算法进行计算；
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases] #用于保存批量数据中的b的梯度之和
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #计算每个数据的梯度
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        返回损失函数的梯度
        """
        #用于保存损失函数的梯度
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] #保存全部层的激活值
        zs = [] #保存全部层的加权值
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        #计算输出层的误差还有两个梯度
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        返回测试数据中预测正确的个数; simoid函数是取最高激活值对应的坐标那个
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        返回损失函数针对输出层激活值的偏导
        """
        return (output_activations-y)


def sigmoid(z):

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    sigmoid函数的导数
    """
    return sigmoid(z)*(1-sigmoid(z))
