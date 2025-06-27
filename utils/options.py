import argparse

parser = argparse.ArgumentParser('SalMAE plus training', add_help=False)
parser.add_argument('--network', default='salmae', type=str)
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-g', '--gpu', default='1', type=str,
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
'''
Dataset parameters
'''
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--seq_len', default=1, type=int)
parser.add_argument('--img_shape', default=(256, 256), type=lambda s: tuple(map(int, s.split(','))))
# parser.add_argument('--category_rainy', default='DrFixD_rainy', type=str, help='select [BDDA or TrafficGaze or DrFixD_rainy]')
parser.add_argument('--category', default='DrFixD_rainy', type=str, help='select [BDDA or TrafficGaze or DrFixD_rainy]')
# parser.add_argument('--category_night', default='night', type=str, help='select [BDDA or TrafficGaze or DrFixD_rainy]')
# DrFixD_rainy salmm
# parser.add_argument('--root', default='/traffic_dataset/', type=str)
parser.add_argument('--root', default='/dataset/DrFixD-rainy/', type=str)
# parser.add_argument('--root', default='/dataset/night/', type=str)
parser.add_argument(
    '--out-stride',
    type=int,
    default=16,
    help='out-stride of deeplabv3+',
)
parser.add_argument(
    '--sync-bn',
    type=bool,
    default=True,
    help='sync-bn in deeplabv3+',
)
parser.add_argument(
    '--freeze-bn',
    type=bool,
    default=False,
    help='freeze batch normalization of deeplabv3+',
)
