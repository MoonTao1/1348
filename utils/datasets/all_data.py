from scipy.ndimage import filters
from numpy import *
import scipy.io as sio
import os
import numpy as np
import cv2
import numpy
import os
import imageio as io
import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader


img_shape = (256, 256)

def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y

#######################################################################################################################
def getLabel(root, vid_index, frame_index, img_shape):
    root = root.replace('traffic_frames', 'fixdata')
    fixdatafile = (root + 'fixdata' + str(vid_index) + '.mat')
    # print(fixdatafile)
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



def getLabel_Continuous(vid_index, frame_index,img_shape):
    fixdatafile = ('/data/workspace/mwt/traffic_dataset/fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    fix_x = fix_x.astype('int')
    fix_y = fix_y.astype('int')
    mask = numpy.zeros((720, 1280), dtype='float32')
    #print(len(fix_x),vid_index, frame_index)
    for i in range(len(fix_x)):
        #print(fix_x[i],fix_y[i])
        mask[fix_x[i], fix_y[i]] = 1
    mask = filters.gaussian_filter(mask, 40)
    mask = numpy.array(mask, dtype='float32')
    mask = cv2.resize(mask, img_shape, interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0
    if mask.max() == 0:
        print (mask.max())
        #print img_name
    else:
        mask = mask / mask.max()
    return mask
########################################################################################################################
'''
Make dataloader for dataset: BDDA
'''
class ImageList_BDDA(Dataset):
    def __init__(self, root, imgs, label_root, labels, img_shape=(256, 256), for_train=False):
        self.root = root
        self.imgs = imgs
        self.label_root = label_root
        self.labels = labels
        self.img_shape = img_shape
        self.for_train = for_train

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_name = os.path.join(self.root, img_name)

        img = io.imread(img_name)
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0

        lab_img_name = self.labels[index]
        lab_img_name = os.path.join(self.label_root, lab_img_name)

        lab_img = cv2.imread(lab_img_name, 0)
        lab_img = cv2.resize(lab_img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        lab_img = lab_img.astype('float32') / 255.0

        if np.max(lab_img) < 0.1:
            print(lab_img_name, np.max(lab_img))

        if self.for_train:
            img, lab_img = transform(img, lab_img)

        img = img.transpose(2, 0, 1)
        lab_img = lab_img[None, ...]
        img = np.ascontiguousarray(img)
        lab_img = np.ascontiguousarray(lab_img)

        img_tensor, lab_img_tensor = torch.from_numpy(img), torch.from_numpy(lab_img)
        del img, lab_img
        return img_tensor, lab_img_tensor

    def __len__(self):
        return len(self.imgs)


class ImageList_BDDA_Continuous(Dataset):
    def __init__(self, root, imgs, label_root, labels, img_shape, frames=1, for_train=False):
        self.root = root
        self.imgs = imgs
        self.for_train = for_train
        self.label_root = label_root
        self.labels = labels
        self.frames = frames
        self.img_shape = img_shape

    def __getitem__(self, index):
        img_name = self.imgs[index]
        vid_index, frame_index = img_name.split('/')

        vid_index = int(vid_index)
        frame_index = int(frame_index[:-4])

        imgarr = []
        for m in range(self.frames):
            temp_frame_index = frame_index - m

            if temp_frame_index < 0:
                continue  # 跳过无效帧索引

            img_name = f'{vid_index:04d}/{temp_frame_index:04d}.jpg'
            image_name = os.path.join(self.root, img_name)

            if not os.path.exists(image_name):
                raise FileNotFoundError(f"Image not found: {image_name}")

            img = io.imread(image_name)
            img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
            imgarr.append(torch.from_numpy(img))

        img_name = f'{vid_index:04d}/{frame_index:04d}.jpg'
        label_name = os.path.join(self.label_root, img_name)
        label = cv2.imread(label_name, 0)

        if label is None:
            raise FileNotFoundError(f"Label not found at {label_name}")

        label = cv2.resize(label, self.img_shape, interpolation=cv2.INTER_CUBIC)
        label = np.ascontiguousarray(label)
        label = torch.from_numpy(label).unsqueeze(0)

        imgarr = torch.stack(imgarr)
        imgarr = imgarr.float() / 255.0
        label_arr = label.float() / 255.0

        return imgarr, label_arr

    def __len__(self):
        return len(self.imgs)

'''
Make dataloader for dataset: Traffic_Gaze DrFixD-Rainy
'''
class ImageList_SunnyorRainy(Dataset):
    def __init__(self, root, imgs, img_shape=(256, 256), for_train=False):
        self.root = root
        self.imgs = imgs

        self.img_shape = img_shape
        self.for_train = for_train

    def __getitem__(self, index):
        img_name = self.imgs[index]
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])

        image_name = os.path.join(self.root, img_name)
        img = io.imread(image_name)
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
        # del img, lab_img
        return img_tensor, lab_img_tensor

    def __len__(self):
        return len(self.imgs)


class ImageList_SunnyorRainy_Continuous(Dataset):
    def __init__(self, root, imgs, img_shape, frames = 1,for_train=False):
        self.root = root
        self.imgs = imgs
        self.for_train = for_train
        self.frames = frames
    def __getitem__(self, index):
        img_name = self.imgs[index]
        imgarr = []
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])
        for m in range(self.frames):
            fra_index = frame_index - m
            # print(fra_index)
            vid_index = vid_index
            img_name = '%02d' % (vid_index) + "/" + '%06d' % (fra_index) + '.jpg'
            # print(img_name)
            image_name = os.path.join(self.root, img_name)
            img = io.imread(image_name)
            img = cv2.resize(img, img_shape, interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
            img = numpy.ascontiguousarray(img)
            # print(type(img))
            imgarr.append(torch.from_numpy(img))
        imgarr = torch.stack(imgarr)
        imgarr = imgarr.float() / 255.0
        mask = getLabel_Continuous(vid_index, frame_index,img_shape)
        # if self.for_train:
        # img, mask = transform(img, mask)
        mask = mask[None, ...]
        mask = numpy.ascontiguousarray(mask)
        # exit(0)
        return imgarr, torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)
##############################################################################################################################
def create_dataset_loader(not_continuous, dataset_type, root, imgs, label_root=None, labels=None, img_shape=(256, 256), frames = 1, for_train=False):
    if not_continuous:
        if dataset_type == "SunnyorRainy":
            return DataLoader(ImageList_SunnyorRainy(root, imgs, img_shape, for_train))
        elif dataset_type =="BDDA":
            return DataLoader(ImageList_BDDA(root, imgs, label_root, labels, img_shape, for_train))
        else:
            raise ValueError("Unsupported dataset type.")

    else:
        if dataset_type == "SunnyorRainy":
            return DataLoader(ImageList_SunnyorRainy_Continuous(root, imgs, img_shape, frames, for_train))
        elif dataset_type == "BDDA":
            return DataLoader(ImageList_BDDA_Continuous(root, imgs, label_root, labels, img_shape, frames, for_train))
        else:
            raise ValueError("Unsupported dataset type.")

###############################################################################################################################



if __name__ == '__main__':



    root_sunny = '/data/workspace/mwt/traffic_dataset/traffic_frames/'
    root_rainy = '/data/workspace/zcm/dataset/DrFixD-rainy/traffic frames/'
    root_BDDA = 'F:/Build_dataset/BDDA/'

    bdda_image_root = root_BDDA + 'training/camera_frames/'
    bdda_label_root = root_BDDA + 'training/gazemap_frames/'

    # sunny_train_imgs = [json.loads(line) for line in open(root_sunny + 'train.json')]
    # sunny_valid_imgs = [json.loads(line) for line in open(root_sunny + 'valid.json')]
    # sunny_test_imgs = [json.loads(line) for line in open(root_sunny + 'test.json')]
    #
    # rainy_train_imgs = [json.loads(line) for line in open(root_rainy + 'train.json')]
    # rainy_valid_imgs = [json.loads(line) for line in open(root_rainy + 'valid.json')]
    # rainy_test_imgs = [json.loads(line) for line in open(root_rainy + 'test.json')]


    bdda_train_imgs = [json.loads(line) for line in open(root_BDDA + 'training.json')]
    # bdda_valid_imgs = [json.loads(line) for line in open(image_root + 'validation_c.json')]
    # bdda_test_imgs = [json.loads(line) for line in open(image_root + 'test_c.json')]


    bdda_train_imgs_labels = [json.loads(line) for line in open(root_BDDA + 'training.json')]


    not_continuous = False  # 或 False
    dataset_type = "BDDA"  #SunnyorRainy 或 "BDDA"

    train_loader = create_dataset_loader(not_continuous, dataset_type, bdda_image_root, bdda_train_imgs,  bdda_label_root,         bdda_train_imgs_labels,    img_shape=(256, 256), frames=6, for_train=True)
    #######################################(是否连续，       数据集类型，       根目录，      图像数据，  bdda标签根目录（仅在bdda时有用），bdda标签数据（仅在bdda时有用），    图像形状，          帧数，      训练）

    for i, (input, target) in enumerate(train_loader):
        print(input.shape, target.shape)
