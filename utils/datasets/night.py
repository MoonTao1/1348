from torch.utils.data import Dataset
import imageio as io
import cv2
import torch
from scipy.ndimage import filters
from numpy import *
import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm


# class ImageList_DrFixD_rainy(Dataset):
#     def __init__(self, args, imgs, for_train=False):
#         self.root = args.root
#         self.imgs = imgs
#
#         self.img_shape = args.img_shape
#         self.for_train = for_train
#
#     def __getitem__(self, index):
#         img_name = self.imgs[index]
#         vid_index = int(img_name[0:2])
#         frame_index = int(img_name[3:9])
#         img_name = 'trafficframe/' + self.imgs[index]
#
#         image_name = os.path.join(self.root, img_name)
#         img = io.imread(image_name)
#         img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
#         img = img.astype('float32') / 255.0
#
#         lab_img = getLabel(self.root, vid_index, frame_index, self.img_shape)
#
#         if self.for_train:
#             img, lab_img = transform(img, lab_img)
#
#         img = img.transpose(2, 0, 1)
#         lab_img = lab_img[None, ...]
#         img = np.ascontiguousarray(img)
#
#         lab_img = np.ascontiguousarray(lab_img)
#
#         img_tensor, lab_img_tensor = torch.from_numpy(img), torch.from_numpy(lab_img)
#         # del img, lab_img
#         return img_tensor, lab_img_tensor
#
#     def __len__(self):
#         return len(self.imgs)


class ImageList_NIGHT(Dataset):
    def __init__(self, args, imgs, for_train=False):
        self.root = args.root
        self.imgs = imgs
        self.img_shape = args.img_shape
        self.for_train = for_train

        # 检查是否已经保存了数据
        self.data_dir = os.path.join(self.root, 'cache_data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.save_data()
        # self.save_data()

    def save_data(self):
        for img_name in tqdm(self.imgs, desc="Prepare Images"):
            img_dir = img_name[0:2]

            temp_dir = os.path.join(self.data_dir, img_dir)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            vid_index = int(img_name[0:2])
            frame_index = int(img_name[3:9])
            img_path = os.path.join(self.root, 'trafficframe/', img_name)

            # 加载并预处理图像
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
            img = img.astype('float32') / 255.0

            lab_img = getLabel(self.root, vid_index, frame_index, self.img_shape)

            if self.for_train:
                img, lab_img = transform(img, lab_img)

            img = img.transpose(2, 0, 1)
            lab_img = lab_img[None, ...]
            img = np.ascontiguousarray(img)
            lab_img = np.ascontiguousarray(lab_img)

            img_tensor, lab_img_tensor = torch.from_numpy(img), torch.from_numpy(lab_img)

            # 保存图像和标签为.pt文件
            img_save_path = os.path.join(self.data_dir, f"{img_name[0:-4]}_img.pt")
            lab_save_path = os.path.join(self.data_dir, f"{img_name[0:-4]}_lab.pt")
            torch.save(img_tensor, img_save_path)
            torch.save(lab_img_tensor, lab_save_path)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_path = os.path.join(self.data_dir, f"{img_name[0:-4]}_img.pt")
        lab_path = os.path.join(self.data_dir, f"{img_name[0:-4]}_lab.pt")

        # 加载图像和标签
        img_tensor = torch.load(img_path)
        lab_img_tensor = torch.load(lab_path)

        return img_tensor, lab_img_tensor

    def __len__(self):
        return len(self.imgs)


# class ImageList_DrFixD_rainy_memmap(Dataset):
#     def __init__(self, args, imgs, for_train=False):
#         self.root = args.root
#         self.imgs = imgs
#         self.img_shape = args.img_shape
#         self.for_train = for_train

#         # 创建内存映射文件的目录
#         # self.memmap_dir = 'memmap_cache'
#         self.memmap_dir = os.path.join(self.root, 'memmap_cache')
#         os.makedirs(self.memmap_dir, exist_ok=True)

#         # 内存映射文件路径
#         self.images_memmap_path = os.path.join(self.memmap_dir, 'images.dat')
#         self.labels_memmap_path = os.path.join(self.memmap_dir, 'labels.dat')

#         # 预处理并创建内存映射
#         self.prepare_memmap()

#     def prepare_memmap(self):
#         # 如果内存映射文件已存在，直接加载
#         # print(self.images_memmap_path)
#         # print(self.labels_memmap_path)
#         # exit(0)
#         if os.path.exists(self.images_memmap_path) and os.path.exists(self.labels_memmap_path):
#             print("加载已存在的内存映射文件")
#             self.images_memmap = np.memmap(
#                 self.images_memmap_path,
#                 dtype='float32',
#                 mode='r',
#                 shape=(len(self.imgs), 3, *self.img_shape)
#             )
#             self.labels_memmap = np.memmap(
#                 self.labels_memmap_path,
#                 dtype='float32',
#                 mode='r',
#                 shape=(len(self.imgs), 1, *self.img_shape)
#             )
#             return

#         # 创建可写的内存映射文件
#         print("创建新的内存映射文件")
#         self.images_memmap = np.memmap(
#             self.images_memmap_path,
#             dtype='float32',
#             mode='w+',
#             shape=(len(self.imgs), 3, *self.img_shape)
#         )
#         self.labels_memmap = np.memmap(
#             self.labels_memmap_path,
#             dtype='float32',
#             mode='w+',
#             shape=(len(self.imgs), 1, *self.img_shape)
#         )

#         # 预处理并存储数据
#         for i, img_name in enumerate(self.imgs):
#             # 加载图像
#             full_img_path = os.path.join(self.root, 'trafficframe', img_name)

#             # 解析图像和标签信息
#             vid_index = int(img_name[0:2])
#             frame_index = int(img_name[3:9])

#             # 读取图像
#             img = io.imread(full_img_path)
#             img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
#             img = img.astype('float32') / 255.0

#             # 获取标签
#             lab_img = getLabel(self.root, vid_index, frame_index, self.img_shape)

#             # 数据增强（如果是训练集）
#             if self.for_train:
#                 img, lab_img = transform(img, lab_img)

#             # 调整图像和标签格式
#             img = img.transpose(2, 0, 1)
#             lab_img = lab_img[None, ...]

#             img = np.ascontiguousarray(img)
#             lab_img = np.ascontiguousarray(lab_img)


#             # 存储到内存映射
#             # print(img.shape)
#             # print(img.transpose(1, 2, 0).shape)
#             # print(lab_img.shape)
#             # exit(0)
#             self.images_memmap[i] = img
#             self.labels_memmap[i] = lab_img

#             # 进度提示
#             if (i + 1) % 100 == 0:
#                 print(f"处理进度: {i+1}/{len(self.imgs)}")

#         # 刷新内存映射
#         self.images_memmap.flush()
#         self.labels_memmap.flush()

#     def __getitem__(self, index):
#         # 从内存映射读取数据
#         img = self.images_memmap[index]
#         lab_img = self.labels_memmap[index]

#         # 转换为张量
#         img_tensor = torch.from_numpy(img).float()
#         lab_img_tensor = torch.from_numpy(lab_img).float()

#         return img_tensor, lab_img_tensor

#     def __len__(self):
#         return len(self.imgs)

#     def cleanup(self):
#         # 删除内存映射文件
#         try:
#             os.remove(self.images_memmap_path)
#             os.remove(self.labels_memmap_path)
#             os.rmdir(self.memmap_dir)
#             print("已清理内存映射文件")
#         except Exception as e:
#             print(f"清理失败: {e}")


class ImageList_NIGHT_Continuous(Dataset):
    def __init__(self, args, imgs, for_train=False):
        self.args = args
        self.root = args.root
        self.imgs = imgs
        self.for_train = for_train
        self.seq_len = args.seq_len

    def __getitem__(self, index):
        img_name = self.imgs[index]

        imgarr = []
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])
        for m in range(self.seq_len):
            fra_index = frame_index - m
            vid_index = vid_index
            img_name = 'trafficframe/' + '%02d' % (vid_index) + "/" + '%06d' % (fra_index) + '.jpg'
            image_name = os.path.join(self.root, img_name)
            img = io.imread(image_name)
            img = cv2.resize(img, self.args.img_shape, interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            imgarr.append(torch.from_numpy(img))
        imgarr = torch.stack(imgarr)
        imgarr = imgarr.float() / 255.0
        mask = getLabel_Continuous(self.root, vid_index, frame_index, self.args.img_shape)

        if self.for_train:
            img, mask = transform(img, mask)

        mask = mask[None, ...]
        mask = np.ascontiguousarray(mask)
        return imgarr, torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)


def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y


def getLabel(root, vid_index, frame_index, img_shape):
    fixdatafile = (root + '/fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    fix_x = fix_x.astype('int')
    fix_y = fix_y.astype('int')
    mask = np.zeros((720, 1280), dtype='float32')

    for i in range(len(fix_x)):
        if (fix_x[i] < 0) or (fix_x[i] >= 720) or (fix_y[i] < 0) or (fix_y[i] >= 1280):
            continue
        mask[int(fix_x[i]), int(fix_y[i])] = 1

    mask = filters.gaussian_filter(mask, 40)
    mask = np.array(mask, dtype='float32')
    mask = cv2.resize(mask, img_shape, interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0

    if mask.max() == 0:
        pass
        # print(mask.max())
    else:
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def getLabel_Continuous(root, vid_index, frame_index, img_shape):
    fixdatafile = (root + '/fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    fix_x = fix_x.astype('int')
    fix_y = fix_y.astype('int')
    mask = np.zeros((720, 1280), dtype='float32')
    # print(len(fix_x),vid_index, frame_index)
    for i in range(len(fix_x)):
        # print(fix_x[i],fix_y[i])
        mask[fix_x[i], fix_y[i]] = 1
    mask = filters.gaussian_filter(mask, 40)
    mask = np.array(mask, dtype='float32')
    mask = cv2.resize(mask, img_shape, interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0

    if mask.max() == 0:
        # print(mask.max())
        # print img_name
        pass
    else:
        mask = mask / mask.max()
    return mask