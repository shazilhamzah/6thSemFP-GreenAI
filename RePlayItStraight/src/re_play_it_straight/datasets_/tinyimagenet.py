from torchvision import datasets, transforms
import os
from src.re_play_it_straight.support.kaggle_utils import download_from_kaggle


def TinyImageNet(args, downsize=False):
    data_path = args.data_path
    dataset_dir = os.path.join(data_path, "tiny-imagenet-200")
    
    if not os.path.exists(dataset_dir):
        # We download from Kaggle instead of the slow Stanford server
        download_from_kaggle("akash2sharma/tiny-imagenet", dataset_dir)

    channel = 3
    im_size = (32, 32) if downsize else (64, 64)
    num_classes = 200
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if downsize:
        transform = transforms.Compose([transforms.Resize(32), transform])

    dst_train = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=transform)
    dst_unlabeled = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/test'), transform=transform)

    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test
