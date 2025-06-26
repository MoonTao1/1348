import argparse
import os
import time
# import cPickle as pickle
import pickle as pk

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model import Model
from data_loader import ImageList
import random
import warnings
import logging
import numpy as np
import json

import pickle
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 下面两行为自己添加测试跑代码时间长短用
# torch.multiprocessing.set_start_method('spawn', force=True)
import datetime
startTime = datetime.datetime.now()

torch.cuda.current_device()
torch.cuda._initialized = True

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
# workers default = 4
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# epochs之前为100，改为1了
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-g', '--gpu', default='0', type=str,
                    metavar='N', help='GPU NO. (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--split', default=0, type=int)
args = parser.parse_args()

name = 'traffic_net'
ckpts = 'ckpts/cdnn/'  #save model
if not os.path.exists(ckpts): os.makedirs(ckpts)

log_file = os.path.join(ckpts + "/train_log_%s.txt" % (name, ))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)


def main():
    # global args, best_score
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    model = Model()
    model = model.cuda()


    params = model.parameters()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(params, args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # D:/Install/PyCharm Community/pythonProject/CDNN_R
    root = 'E:/pycharm/Project/CDNN-traffic-saliency-master/traffic_dataset/traffic_frames/'  ### traffic_frames root
    test_imgs = [json.loads(line) for line in open(root + 'n_test.json')]
    test_loader = DataLoader(
            ImageList(root, test_imgs),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

    file_name = os.path.join(ckpts, 'model_epoch_3.tar' )


    checkpoint = torch.load(file_name)
    outputs, targets = predict(test_loader, model)


def predict(valid_loader, model):

    # switch to evaluate mode
    model.eval()

    targets = []
    outputs = []

    for i, (input, target) in enumerate(valid_loader):
        # ----------改变target的size--------------------
        # print('validating-第{}次,input_size:{}'.format(i, input.shape))
        # print('validating-第{}次,target_size:{}'.format(i, target.shape))
        # print(target)

        from torchvision.transforms import Resize
        torch_resize = Resize([192, 320])  # 定义Resize类对象
        target = torch_resize(target)
        # ---------------------------------------------

        targets.append(target.numpy().squeeze(1))

        input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output = model(input_var)

        # output = output.data.cpu().numpy().squeeze(1)
        # print(type(output))
        # plt.imshow(output[0], aspect='auto')
        # plt.show()
        # exit(0)

        outputs.append(output.data.cpu().numpy().squeeze(1))

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    return outputs, targets




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs//3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # main()

    # 测试跑代码所用时长
    # endTime = datetime.datetime.now()
    # fileName = 'codeRunTime.txt'
    # with open(fileName, 'w') as f:
    #     f.write('Code start  time: ' + str(startTime) + '\n')
    #     f.write('Code finish time: ' + str(endTime) + '\n')
    #     f.write('Code  run   time: ' + str(endTime-startTime))

    # 使用模型预测测试集
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    model = Model()
    model = model.cuda()

    file_name = os.path.join(ckpts, 'model_epoch_3.tar')
    checkpoint = torch.load(file_name)

    root = 'E:/pycharm/Project/CDNN-traffic-saliency-master/traffic_dataset/traffic_frames/'
    test_imgs = [json.loads(line) for line in open(root + 'n_test.json')]
    test_loader = DataLoader(
        ImageList(root, test_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    outputs, targets = predict(test_loader, model)
    out_img = targets[0]
    print(type(out_img))
    print(out_img.shape)
    orig_Img = cv2.imread('E:/pycharm/Project/CDNN-traffic-saliency-master/traffic_dataset/traffic_frames/1464.jpg')
    orig_Img = cv2.resize(orig_Img, (320, 192), interpolation=cv2.INTER_CUBIC)
    print(type(orig_Img))
    plt.subplot(121)
    plt.imshow(orig_Img, aspect='auto')
    plt.title('original image')
    plt.subplot(122)
    plt.imshow(out_img, aspect='auto')
    plt.title('output image')
    plt.show()