
import json
from torch.utils.data import DataLoader
from .datasets.TrafficGaze import *
from .datasets.night import *
from .datasets.DrFixD_rainy import *
from .datasets.BDDA import *
from .datasets.domain_adaption import *

def make_dataset(args = None):
    train_imgs = [json.loads(line) for line in open(args.root + '/train.json')]
    valid_imgs = [json.loads(line) for line in open(args.root + '/valid.json')]
    test_imgs = [json.loads(line) for line in open(args.root + '/test.json')]

    train_rainy_imgs = [json.loads(line) for line in open(args.root_rainy + '/train.json')]
    valid_rainy_imgs = [json.loads(line) for line in open(args.root_rainy + '/valid.json')]
    test_rainy_imgs = [json.loads(line) for line in open(args.root_rainy + '/test.json')]

    train_loader = DataLoader(
        ImageList_TrafficGaze_DA(args, train_imgs,train_rainy_imgs, for_train=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    valid_loader = DataLoader(
        ImageList_TrafficGaze_DA(args, valid_imgs, valid_rainy_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = DataLoader(
        ImageList_TrafficGaze_DA(args, test_imgs, test_rainy_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)



    return train_loader, valid_loader, test_loader



def build_dataset(args=None):

    if args.seq_len > 1:
        dataset_classes = {
            'TrafficGaze': ImageList_TrafficGaze_Continuous,
            'DrFixD_rainy': ImageList_DrFixD_rainy_Continuous,
            'BDDA': ImageList_BDDA_Continuous,
            'night':ImageList_NIGHT_Continuous
        }
    else:
        dataset_classes = {
            'TrafficGaze': ImageList_TrafficGaze,
            'DrFixD_rainy': ImageList_DrFixD_rainy,
            'BDDA': ImageList_BDDA,
            'night':ImageList_NIGHT
        }

    train_imgs = [json.loads(line) for line in open(args.root + 'train.json')]
    valid_imgs = [json.loads(line) for line in open(args.root + 'valid.json')]
    test_imgs = [json.loads(line) for line in open(args.root + 'test.json')]

    dataset_class = dataset_classes.get(args.category)

    if dataset_class is None:
        raise ValueError(f"Unknown category: {args.category}")

    train_loader = DataLoader(
        dataset_class(args, train_imgs, for_train=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True)

    valid_loader = DataLoader(
        dataset_class(args, valid_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True)

    test_loader = DataLoader(
        dataset_class(args, test_imgs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('SalMAE training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--seq_len', default=2, type=int)
    parser.add_argument('--img_shape', default=(256, 256), type=lambda s: tuple(map(int, s.split(','))))
    parser.add_argument('--category', default='BDDA', type=str)
    parser.add_argument('--root', default='F:/Build_dataset/BDDA', type=str)

    args = parser.parse_args()

    train_loader, valid_loader, test_loader = build_dataset(args=args)
    for i, (input, target) in enumerate(train_loader):
        print('input: ', input.shape)
        print('target: ', target.shape)
        exit(0)
