import random
from utils.options import parser

import os
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# from torchvision import datasets
# from torchvision import transforms

import json
import torch
import warnings
import argparse
import pickle
from networks.model import *
import logging
import time
from utils.mid_metrics import cc, sim, kldiv

# from utils.bulid_models import build_model
from utils.build_datasets import build_dataset
import matplotlib.pyplot as plt
import tempfile



CUDA_LAUNCH_BLOCKING=1

cuda = True
cudnn.benchmark = True

warnings.simplefilter("ignore")




ckpts = ''

log_file = os.path.join(ckpts + "/train_log.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)
# load data
def main():

    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    tmp_path = '/data/pytorch-tmp/mwt'
    os.environ['TMPDIR'] = tmp_path
    tempfile.tempdir = tmp_path
    os.makedirs(tmp_path, exist_ok=True)
    os.chmod(tmp_path, 0o777)

    train_rainy_loader, valid_rainy_loader, test_rainy_loader = build_dataset(args=args)
    args.category = 'TrafficGaze'
    args.root = ''
    train_loader, valid_loader, _ = build_dataset(args=args)

    # model = build_model(args=args)
    model = Model(num_classes=1, backbone='mobilenet', output_stride=args.out_stride,
                                sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    params = model.parameters()
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')
    file_name = os.path.join(ckpts, 'model_best.tar')
    print('-------------- New training session, LR = %.3f ----------------' %(args.lr))

    # setup optimizer


    criterion = torch.nn.BCELoss()
    # criterion_domain = nn.BCEWithLogitsLoss()
    criterion_domain = nn.NLLLoss()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print(f"=> Loading checkpoint '{args.resume}'")
    #         # 加载检查点并映射到当前设备
    #         checkpoint = torch.load(args.resume, map_location=device)
    #         checkpoint_best = torch.load('/data9102/workspace/mwt/DANN-clip-cls/model_best.tar', map_location=device)
    #
    #         # 恢复关键参数
    #         start_epoch = checkpoint['epoch']
    #         best_loss = checkpoint_best.get('valid_loss', float('inf'))
    #
    #         # 确保模型在设备上并加载参数
    #         model = model.to(device)
    #         model.load_state_dict(checkpoint['state_dict'])
    #
    #         # 加载优化器状态并移动张量到设备
    #         optimizer.load_state_dict(checkpoint['optim_dict'])
    #         for state in optimizer.state.values():
    #             for k, v in state.items():
    #                 if isinstance(v, torch.Tensor):
    #                     state[k] = v.to(device)
    #
    #         print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, best_loss={best_loss:.4f})")
    #     else:
    #         print(f"=> No checkpoint found at '{args.resume}'")

    criterion = criterion.cuda()
    criterion_domain = criterion_domain.cuda()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_domain = criterion_domain.cuda()

    # for p in model.parameters():
    #     p.requires_grad = True

    # training
    for epoch in range(args.start_epoch,args.epochs):
        train_loss = train(train_loader,train_rainy_loader,model,criterion,criterion_domain,epoch,optimizer)
        valid_loss = validate(model, valid_rainy_loader, criterion,epoch)
        best_loss = min(valid_loss, best_loss)
        file_name_last = os.path.join(ckpts, 'model_epoch_%d.tar' % (epoch + 1,))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'valid_loss': valid_loss,
        }, file_name_last)

        if valid_loss == best_loss:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, file_name)
        logging.info('Epoch: {:%d} Train loss {:%.4f} | Valid loss {:%.4f}' % (epoch+1, train_loss, valid_loss))

        # checkpoint = torch.load(file_name)
        # model.load_state_dict(checkpoint['state_dict'])
        # outputs, targets = predict(test_rainy_loader, model)
        #
        # np.savez(ckpts + 'test.npz', p=outputs, t=targets)
        # with open(ckpts + 'test.pkl', 'wb') as f:
        #     pickle.dump(test_imgs, f)

        # accu_s = test(source_dataset_name)
        # print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
        # accu_t = test(target_dataset_name)
        # print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))
        # if accu_t > best_accu_t:
        #     best_accu_s = accu_s
        #     best_accu_t = accu_t
        #     torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

    print('============ Summary ============= \n')
# print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
# print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
    print('Corresponding model was save in ' +  '/model_epoch_best.tar')


def validate(model, valid_loader, criterion,epoch):
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    alpha = 0
    start = time.time()
    metrics = [0, 0, 0]
    domain_label = torch.tensor([1] * 16)
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):

            input = input.cuda()
            target = target.cuda()
            domain_label = domain_label.cuda()
            # compute output
            output, _ = model(input=input,alpha=alpha)

            loss = criterion(output, target)

            # measure accuracy and record loss

            losses.update(loss.data, target.size(0))
            # valid metrics printing
            output = output.squeeze(1)
            target = target.squeeze(1)
            metrics[0] = metrics[0] + cc(output, target)
            metrics[1] = metrics[1] + sim(output, target)
            metrics[2] = metrics[2] + kldiv(output, target)

            msg = 'epoch: {:03d} Validating Iter {:03d} Loss {:.6f} || CC {:4f}  SIM {:4f}  KLD {:4f} in {:.3f}s'.format(epoch+1,i + 1,
                                                                                                           losses.avg,
                                                                                                           metrics[
                                                                                                               0] / (
                                                                                                                   i + 1),
                                                                                                           metrics[
                                                                                                               1] / (
                                                                                                                   i + 1),
                                                                                                           metrics[
                                                                                                               2] / (
                                                                                                                   i + 1),
                                                                                                           time.time() - start)
            # print(msg)
            # logging.info(msg)
            start = time.time()

            del input, target, output
            # gc.collect()

            interval = 5
            if (i + 1) % interval == 0:
                logging.info(msg)

    model.train()

    return losses.avg

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


def train(train_loader, train_rainy_loader,model, criterion,criterion_domain,epoch,optimizer):
    erres = AverageMeter()
    model.train()
    # 开启训练模式
    len_dataloader = min(len(train_loader), len(train_rainy_loader))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(train_rainy_loader)


    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = next(data_source_iter)
        s_img, s_label = data_source

        batch_size = args.batch_size
        # 传入源域数据
        domain_label_s = torch.zeros(batch_size).long()
        # domain_label = domain_label.squeeze(dim=1)

        model.zero_grad()
        # 梯度清零
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label_s = domain_label_s.cuda()


        class_output, domain_output = model(input=s_img,alpha =alpha)

        err_s_label = criterion(class_output, s_label)

        err_s_domain = criterion_domain(domain_output, domain_label_s)
        # 前向传播确定损失
        data_target = next(data_target_iter)
        t_img, _ = data_target

        batch_size = args.batch_size

        domain_label_t = torch.ones(batch_size).long()

        # 加载目标域损失
        if cuda:
            t_img = t_img.cuda()
            domain_label_t = domain_label_t.cuda()

        _, domain_output = model(input=t_img,alpha=alpha)
        err_t_domain = criterion_domain(domain_output, domain_label_t)
        err = err_t_domain + err_s_domain + err_s_label
        erres.update(err.item(), s_label.size(0))
        # 计算损失值并记录
        err.backward()
        optimizer.step()
        # 反向传播并进行优化
        sys.stdout.write('\r train epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch+1, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()

    print('\n')
    return erres.avg

# def predict(test_loader, model):
#     model.eval()
#     len_dataloader = len(test_loader)
#     targets = []
#     outputs = []
#     data_source_iter = iter(test_loader)
#     for i in range(len_dataloader):
#         # ----------改变target的size--------------------
#         # print('validating-第{}次,input_size:{}'.format(i, input.shape))
#         # print('validating-第{}次,target_size:{}'.format(i, target.shape))
#         # print(target)
#         data_source = data_source_iter.__next__()
#         input, target = data_source
#         from torchvision.transforms import Resize
#         torch_resize = Resize([192, 320])  # 定义Resize类对象
#         target= torch_resize(target)
#         # ---------------------------------------------
#
#         targets.append(target.numpy().squeeze(1))
#
#         input = input.cuda()
#
#         # compute output
#         output = model(input)
#         outputs.append(output.data.cpu().numpy().squeeze(1))
#
#     targets = np.concatenate(targets)
#     outputs = np.concatenate(outputs)
#     return outputs, targets


if __name__ == '__main__':
    main()
