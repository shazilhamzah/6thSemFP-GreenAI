import os
from torchvision import datasets, transforms
from src.re_play_it_straight.support.kaggle_utils import download_from_kaggle


def FashionMNIST(args):
    data_path = args.data_path
    dataset_dir = os.path.join(data_path, 'fashionmnist')
    if not os.path.exists(dataset_dir):
        download_from_kaggle("zalando-research/fashionmnist", dataset_dir)

    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.2861]
    std = [0.3530]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.FashionMNIST(dataset_dir, train=True, download=True, transform=transform)
    dst_test = datasets.FashionMNIST(dataset_dir, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, None, dst_test
