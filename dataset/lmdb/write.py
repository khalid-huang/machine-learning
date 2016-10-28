#coding:utf-8
import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2 
 
#basic setting
lmdb_file = 'lmdb_data'#期望生成的数据文件
batch_size = 4       #lmdb对于数据进行的是先缓存后一次性写入从而提高效率，因此定义一个batch_size控制每次写入的量。
 
# create the leveldb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))#生成一个数据文件，定义最大空间
lmdb_txn = lmdb_env.begin(write=True)              #打开数据库的句柄
datum = caffe_pb2.Datum()                          #这是caffe中定义数据的重要类型
 
for x in range(8):
    x+=1
    print x
    img=cv2.imread('zhengfang/'+str(x)+'.png')  #从zhengfang/文件夹中依次读取图像
 
    # save in datum
    #f1 = open("out1.txt", "w")   
    #print >> f1, "%s" %(img)
    
    data = img.astype('int').transpose(2,0,1)      #图像矩阵，注意需要调节维度，因为假设图片是sRGB的800*600（宽*高），caffe.io.array_to_datum接受的array是3*heigh*width,也就是3*600*800的，而原本cv2里面imread到的是600*800*3; 虽然格式不同，但是其代表的都是一样的

    #f1 = open("out2.txt", "w")   
    #print >> f1, "%s" %(data)    
    #print data.shape
    
    #data = np.array([img.convert('L').astype('int')]) #或者这样增加维度
    label = x                                      #图像的标签，为了方便存储，这个必须是整数。
    datum = caffe.io.array_to_datum(data, label)   #将数据以及标签整合为一个数据项这个函数的定义在caffe目录下的python/caffe/io.py里面
 
    keystr = '{:0>8d}'.format(x-1)                 #lmdb的每一个数据都是由键值对构成的，因此生成一个用递增顺序排列的定长唯一的key；关于format的函数使用可以自行google
    lmdb_txn.put( keystr, datum.SerializeToString())#调用句柄，写入内存。
 
    # write batch
    if x % batch_size == 0:                        #每当累计到一定的数据量，便用commit方法写入硬盘。
        lmdb_txn.commit()
        lmdb_txn = lmdb_env.begin(write=True)      #commit之后，之前的txn就不能用了，必须重新开一个。
        print 'batch {} writen'.format(x)
 
lmdb_env.close()                                   #结束后记住释放资源，否则下次用的时候打不开。。。
