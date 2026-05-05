from torchvision import datasets, transforms
import os
from src.re_play_it_straight.support.kaggle_utils import download_from_kaggle


def TinyImageNet(args, downsize=False):
    data_path = args.data_path
    dataset_dir = os.path.join(data_path, "tiny-imagenet-200")
    
    if not os.path.exists(dataset_dir):
        try:
            # Try Kaggle first (Fast)
            download_from_kaggle("akash2sharma/tiny-imagenet", dataset_dir)
        except Exception as e:
            print(f"[!] Kaggle download failed ({e}). Falling back to slow Stanford server...")
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            import requests
            import zipfile
            
            os.makedirs(data_path, exist_ok=True)
            zip_path = os.path.join(data_path, "tiny-imagenet-200.zip")
            
            print("Downloading Tiny-ImageNet from Stanford...")
            r = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            print("Unzipping Tiny-ImageNet...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(path=data_path)
            
            # Remove zip after extraction to save space
            os.remove(zip_path)

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
