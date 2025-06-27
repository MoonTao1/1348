import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio

import scipy.io as sio
import numpy as np

data_names = np.load('/data9102/workspace/mwt/Experiment/rainy/deeplab/test_rainy_deeplab.npz')

data_name = data_names['p'].astype(np.float32)
data_name1 = data_names['t'].astype(np.float32)

# 假设数据是 (N, C, H, W)，我们按 N 维度拆分
split_size = data_name.shape[0] // 2
data1, data2 = np.split(data_name, [split_size])
data1_t, data2_t = np.split(data_name1, [split_size])

# 分别保存
sio.savemat('/p1.mat', {'p1': data1})
sio.savemat('/p2.mat', {'p2': data2})
sio.savemat('/t1.mat', {'t1': data1_t})
sio.savemat('/t2.mat', {'t2': data2_t})


