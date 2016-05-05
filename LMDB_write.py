#!/usr/bin/python
import numpy as np

import caffe
import lmdb


N=100
X=np.zeros((N,227,227,3),dtype=np.uint8)
y=np.zeros(N,dtype=np.int64)

env = lmdb.open('test_lmdb',map_size = 1024**4,map_async=True,max_dbs=0)
for i in range(N):
	with env.begin(write=True) as txn:
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = X.shape[1]
		datum.height = X.shape[2]
		datum.width = X.shape[3]
		datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
		datum.label = int(y[i])

		str_id = '{:08}'.format(a[n]) 

		txn.put(str_id.encode('ascii'), datum.SerializeToString())
