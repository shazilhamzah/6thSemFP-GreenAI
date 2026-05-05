import os
from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long
import numpy as np


def CIFAR100(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    
    dataset_dir = args.data_path + '/cifar100'
    if not os.path.exists(dataset_dir):
        try:
            from src.re_play_it_straight.support.kaggle_utils import download_from_kaggle
            download_from_kaggle("fedesoriano/cifar100-python", dataset_dir)
        except Exception as e:
            print(f"[!] Kaggle download for CIFAR100 failed ({e}). Falling back to torchvision...")
        
    dst_train = datasets.CIFAR100(dataset_dir, train=True, download=True, transform=train_transform)
    dst_unlabeled = datasets.CIFAR100(dataset_dir, train=True, download=True, transform=test_transform)
    dst_test = datasets.CIFAR100(dataset_dir, train=False, download=True, transform=test_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test
