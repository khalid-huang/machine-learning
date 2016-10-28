#coding:utf-8
 
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2
 
lmdb_env = lmdb.open('lmdb_data')#打开数据文件
lmdb_txn = lmdb_env.begin()      #生成句柄
lmdb_cursor = lmdb_txn.cursor()  #生成迭代器指针
datum = caffe_pb2.Datum()        #caffe定义的数据类型
 
for key, value in lmdb_cursor:   #循环获取数据
    datum.ParseFromString(value) #从value中读取datum数据
 
    label = datum.label          #获取标签以及图像数据
    data = caffe.io.datum_to_array(datum)
    print data.shape
    print datum.channels
    image =data.transpose(1,2,0)
    #cv2.imshow('cv2.png', image) #显示
    #cv2.waitKey(0)
 
cv2.destroyAllWindows()
lmdb_env.close()