from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long
import numpy as np


def MNIST(args, permuted=False, permutation_seed=None):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))])
    '''
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=28, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    
    dataset_dir = args.data_path + '/mnist'
    import os
    if not os.path.exists(dataset_dir):
        try:
            from src.re_play_it_straight.support.kaggle_utils import download_from_kaggle
            download_from_kaggle("hojjatk/mnist-dataset", dataset_dir)
        except Exception as e:
            print(f"[!] Kaggle download for MNIST failed ({e}). Falling back to torchvision...")
        
    dst_train = datasets.MNIST(dataset_dir, train=True, download=True, transform=train_transform)
    dst_unlabeled = datasets.MNIST(dataset_dir, train=True, download=True, transform=test_transform)
    dst_test = datasets.MNIST(dataset_dir, train=False, download=True, transform=test_transform)

    #dst_train = datasets_.MNIST(data_path, train=True, download=True, transform=transform)
    #dst_test = datasets_.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    #return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test


def permutedMNIST(data_path, permutation_seed=None):
    return MNIST(data_path, True, permutation_seed)
