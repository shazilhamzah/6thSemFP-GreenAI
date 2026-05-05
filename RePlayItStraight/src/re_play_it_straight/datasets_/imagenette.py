from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch import tensor, long
from PIL import Image
import os
from src.re_play_it_straight.support.kaggle_utils import download_from_kaggle


class ImagenetteDataset(Dataset):
    def __init__(self, file_path, transform=None, resolution=160):
        self.transform = transform
        self.resolution = resolution
        print(f"Resizing Initial Data into {self.resolution}x{self.resolution}")
        transform_resize = T.Resize(size=(self.resolution, self.resolution))
        self.data = ImageFolder(file_path, transform_resize, is_valid_file=self.checkImage)
        self.classes = self.data.classes
        self.targets = self.data.targets

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img, label = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, label#, index

    def __len__(self):
        return len(self.data)

    def checkImage(self, path):
        try:
            Image.open(path)
            return True

        except:
            return False


def get_augmentations_32(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])
    test_transform = T.Compose([T.ToTensor(), T_normalize])
    return train_transform, test_transform


def Imagenette(args):
    channel = 3
    im_size = (32, 32)  # (160, 160) TODO
    num_classes = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    T_normalize = T.Normalize(mean, std)
    #if args.resolution == 32: TODO
    train_transform, test_transform = get_augmentations_32(T_normalize) #TODO

    dataset_dir = os.path.join(args.data_path, 'Imagenette')
    if not os.path.exists(dataset_dir):
        try:
            # Try Kaggle first
            download_from_kaggle("frabbisw/imagenette", dataset_dir)
        except Exception as e:
            print(f"[!] Kaggle download for Imagenette failed ({e}). Falling back to Fast.ai servers...")
            import requests
            import tarfile
            
            # Use the 320px version as a reliable fallback
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
            os.makedirs(args.data_path, exist_ok=True)
            tgz_path = os.path.join(args.data_path, "imagenette2-320.tgz")
            
            print("Downloading Imagenette from Fast.ai...")
            r = requests.get(url, stream=True)
            with open(tgz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            print("Extracting Imagenette...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=args.data_path)
            
            # Rename the extracted folder to match expected structure
            extracted_path = os.path.join(args.data_path, "imagenette2-320")
            if os.path.exists(extracted_path):
                if os.path.exists(dataset_dir):
                    import shutil
                    shutil.rmtree(dataset_dir)
                os.rename(extracted_path, dataset_dir)
                
            os.remove(tgz_path)

    dst_train = ImagenetteDataset(dataset_dir + '/train/', transform=train_transform, resolution=args.resolution)
    dst_unlabeled = ImagenetteDataset(dataset_dir + '/train/', transform=test_transform, resolution=args.resolution)
    dst_test = ImagenetteDataset(dataset_dir + '/val/', transform=test_transform, resolution=args.resolution)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test
