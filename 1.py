import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
# data_names = np.load('/data9102/workspace/mwt/test_night/test_night.npz')
# data_name = (data_names['p'])
# np.save('/data9102/workspace/mwt/test_night/p.npy',data_name)
# data_name1 = (data_names['t'])
# np.save('/data9102/workspace/mwt/test_night/t.npy',data_name1)

# data_names = np.load('/data9102/workspace/mwt/test_night/test_night.npz')
#
# # 提取数据
# data_name = data_names['p']
# data_name1 = data_names['t']
#
# # 保存为 .mat 文件
# sio.savemat('/data9102/workspace/mwt/test_night/p.mat', {'p': data_name})
# sio.savemat('/data9102/workspace/mwt/test_night/t.mat', {'t': data_name1})
# import json
# import scipy.io as sio
#
# # 逐行读取 JSON 文件并处理
# test = []
# with open('/data9102/workspace/mwt/dataset/night/test.json', 'r') as f:
#     for line in f:
#         # 每行是一个 JSON 字符串，加载它并添加到列表中
#         test.append(json.loads(line.strip()))  # 每行加载一个 JSON 对象
#
# # 创建字典并准备保存为 .mat 文件
# data_dict = {
#     'test': test
# }
#
# # 保存为 .mat 文件
# sio.savemat('/data9102/workspace/mwt/dataset/night/test_night.mat', data_dict)
#
# print("MAT 文件已成功创建。")
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
sio.savemat('/data9102/workspace/mwt/Experiment/rainy/deeplab/p1.mat', {'p1': data1})
sio.savemat('/data9102/workspace/mwt/Experiment/rainy/deeplab/p2.mat', {'p2': data2})
sio.savemat('/data9102/workspace/mwt/Experiment/rainy/deeplab/t1.mat', {'t1': data1_t})
sio.savemat('/data9102/workspace/mwt/Experiment/rainy/deeplab/t2.mat', {'t2': data2_t})


