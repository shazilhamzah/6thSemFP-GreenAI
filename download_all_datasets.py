import os
import sys

# Add RePlayItStraight to path to import kaggle_utils
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "RePlayItStraight"))

from src.re_play_it_straight.support.kaggle_utils import download_from_kaggle

def download_all():
    data_path = os.path.join(project_root, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    datasets = {
        "CIFAR10": {"handle": "pankrzysiu/cifar10-python", "path": "cifar10"},
        "CIFAR100": {"handle": "fedesoriano/cifar100", "path": "cifar100"},
        "MNIST": {"handle": "hojjatk/mnist-dataset", "path": "mnist"},
        "FashionMNIST": {"handle": "zalando-research/fashionmnist", "path": "fashionmnist"},
        "SVHN": {"handle": "stanfordu/street-view-house-numbers", "path": "svhn"},
        "TinyImageNet": {"handle": "akash2sharma/tiny-imagenet", "path": "tiny-imagenet-200"},
        "Imagenette": {"handle": "frabbisw/imagenette", "path": "Imagenette"},
        "Imagewoof": {"handle": "frabbisw/imagewoof", "path": "Imagewoof"},
    }
    
    print(f"Starting download of all datasets to: {data_path}")
    print("=" * 50)
    
    for name, info in datasets.items():
        target_dir = os.path.join(data_path, info["path"])
        print(f"\n[+] Processing {name}...")
        if os.path.exists(target_dir) and os.listdir(target_dir):
            print(f"[SKIP] {name} already exists.")
            continue
            
        try:
            download_from_kaggle(info["handle"], target_dir)
            print(f"[OK] {name} download complete.")
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")
            
    print("\n" + "=" * 50)
    print("All datasets processed!")
    print("=" * 50)

if __name__ == "__main__":
    download_all()
