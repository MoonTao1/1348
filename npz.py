import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 这里必须在 torch 导入前执行
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json
import torch
import warnings

from utils.options import parser
# from utils.bulid_models import build_model
from utils.build_datasets import build_dataset
from networks.model import *
from utils.options import parser
from utils.build_datasets import build_dataset

model_root = 'models'
cuda = True
cudnn.benchmark = True
warnings.simplefilter("ignore")

args = parser.parse_args()

def main():

    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)
    ckpts = ''

    model = Model(num_classes=1, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    # print len(train_imgs),train_imgs[0]
    # print train_imgs
    # exit(0)
    file_name = os.path.join('/model_best.tar')
    _, _, test_rainy_loader = build_dataset(args=args)

    # dual_train_dataset = DualDataset(train_loader.dataset, train_rainy_loader.dataset)
    # dual_train_loader = DataLoader(dual_train_dataset, batch_size=args.batch_size, shuffle=True)
    # load model


    model = model.cuda()
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['state_dict'])
    outputs, targets = predict(test_rainy_loader, model)
    model.visualize_domain_distribution(save_path="domain_distribution.png")

    # np.savez(ckpts + 'test_rainy_deeplab.npz', p=outputs, t=targets)

def predict(test_loader, model):
    model.eval()
    targets = []
    outputs = []
    alpha= 0
    for i, (input, target) in enumerate(test_loader):

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        from torchvision.transforms import Resize
        torch_resize = Resize([256, 256])  # 定义Resize类对象
        target= torch_resize(target)
        # ---------------------------------------------
        # compute output
        output,_ = model(input=input,alpha=0)
        outputs.append(output.data.cpu().numpy().squeeze(1))
        targets.append(target.data.cpu().numpy().squeeze(1))

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    return outputs, targets

# class DualDataset(Dataset):
#     def __init__(self, dataset1, dataset2):
#         self.dataset1 = dataset1  # 第一个数据集
#         self.dataset2 = dataset2  # 第二个数据集
#         self.len = min(len(dataset1), len(dataset2))  # 保证每次训练两个数据集加载相同数量的样本
#
#     def __getitem__(self, index):
#         # 从第一个数据集加载数据
#         img1, label1 = self.dataset1[index]
#
#         # 从第二个数据集加载数据
#         img2, label2 = self.dataset2[index]
#
#         return img1, label1, img2, label2
#
#     def __len__(self):
#         return self.len
if __name__ == '__main__':
    main()
