#!/usr/bin/python
import os
import numpy as np 
import cv2
import caffe
import lmdb

def makelmdb(path):
	global m
	global v
	global y
	dir_list=os.listdir(path)
	for dir_name in dir_list:
		file_path=os.listdir(path+'/'+dir_name)
		y[v]=m
		m=m+1
		for file in file_path:
			full_path=path+'/'+dir_name+'/'+file
			read_avi(full_path)


def read_avi(avi_path):
	global X
	global v
	

	L=10
	f=0
	cap = cv2.VideoCapture(avi_path)
	for i in range(L):
		ret,img = cap.read()
		#cv2.imwrite("ttjpg.jpg",img)
		X[v,f]=img[:,:,0]
		f=f+1
		X[v,f]=img[:,:,1]
		f=f+1
		X[v,f]=img[:,:,2]
		f=f+1
	cap.release()
	v=v+1
# main function start here
N=100

X = np.zeros((N,30,240,320), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)
v=0
m=0
makelmdb('/home/zkk/caffe/python/UCF-test')

map_size = X.nbytes * 10

env = lmdb.open('mylmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)