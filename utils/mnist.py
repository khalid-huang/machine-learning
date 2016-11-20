# -*- coding: utf-8 -*-
import numpy as np
import struct
import os
import matplotlib.pyplot as plt

_tag = '>' #使用大端读取
_twoBytes = 'II' #读取数据格式是两个整数
_fourBytes =  'IIII' #读取的数据格式是四个整数
_pictureBytes =  '784B' #读取的图片的数据格式是784个字节，28*28
_lableByte = '1B' #标签是1个字节
_msb_twoBytes = _tag + _twoBytes
_msb_fourBytes = _tag +  _fourBytes
_msb_pictureBytes = _tag + _pictureBytes
_msb_lableByte = _tag + _lableByte

def getImage(filename = None):
	binfile = open(filename, 'rb') #以二进制读取的方式打开文件
	buf = binfile.read() #获取文件内容缓存区
	binfile.close()
	index = 0 #偏移量
	numMagic, numImgs, numRows, numCols = struct.unpack_from(_msb_fourBytes, buf, index)
	index += struct.calcsize(_fourBytes)
	images = []
	for i in xrange(numImgs):
		imgVal  = struct.unpack_from(_msb_pictureBytes, buf, index)
		index += struct.calcsize(_pictureBytes)
		#下面不知道在做什么
		imgVal	= list(imgVal)
		#for j in range(len(imgVal)):
		#	if imgVal[j] > 1:
		#		imgVal[j] = 1
		images.append(imgVal)
	return np.array(images)

def getlable(filename=None) :
	binfile = open(filename, 'rb')
	buf = binfile.read() #获取文件内容缓存区
	binfile.close()
	index = 0 #偏移量
	numMagic, numItems = struct.unpack_from(_msb_twoBytes,buf, index)
	index += struct.calcsize(_twoBytes)
	labels = []
	for i in range(numItems):
		value = struct.unpack_from(_msb_lableByte, buf, index)
		index += struct.calcsize(_lableByte)
		labels.append(value[0]) #获取值的内容
	return np.array(labels)

def outImg(arrX, arrY, order):
	"""
	根据指定的order来获取对应的图片和标签
	"""
	image = np.array(arrX[order])
	print image.shape
	image = image.reshape(28,28)
	label = arrY[order]
	print label
	outfile = str(order) + '_'+str(label) + '.png'
	plt.figure()
	plt.imshow(image, cmap = 'binary')
	plt.savefig("./" + outfile)

def test():
	trainfile_X = '../dataset/MNIST/train-images.idx3-ubyte'
        trainfile_y = '../dataset/MNIST/train-labels.idx1-ubyte'
        arrX = getImage(trainfile_X)
        arrY = getlable(trainfile_y)
        outImg(arrX, arrY, 3)

if __name__  == '__main__':
	test()