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
            print(f"Attempting to download {name} via Kaggle API...")
            download_from_kaggle(info["handle"], target_dir)
            print(f"[OK] {name} download complete.")
        except Exception as e:
            print(f"[!] Kaggle download for {name} failed ({e}). Attempting original server fallback...")
            
            # Fallback logic based on dataset name
            try:
                import requests
                if name == "TinyImageNet":
                    import zipfile
                    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
                    zip_path = os.path.join(data_path, "tiny-imagenet-200.zip")
                    print(f"Downloading from {url}...")
                    r = requests.get(url, stream=True)
                    with open(zip_path, "wb") as f: f.write(r.content)
                    with zipfile.ZipFile(zip_path) as zf: zf.extractall(path=data_path)
                    os.remove(zip_path)
                    print(f"[OK] {name} ready via Stanford.")
                elif name in ["Imagenette", "Imagewoof"]:
                    import tarfile
                    slug = "imagenette2-320" if name == "Imagenette" else "imagewoof2-320"
                    url = f"https://s3.amazonaws.com/fast-ai-imageclas/{slug}.tgz"
                    tgz_path = os.path.join(data_path, f"{slug}.tgz")
                    print(f"Downloading from {url}...")
                    r = requests.get(url, stream=True)
                    with open(tgz_path, "wb") as f: f.write(r.content)
                    with tarfile.open(tgz_path, "r:gz") as tar: tar.extractall(path=data_path)
                    if os.path.exists(target_dir):
                        import shutil
                        shutil.rmtree(target_dir)
                    os.rename(os.path.join(data_path, slug), target_dir)
                    os.remove(tgz_path)
                    print(f"[OK] {name} ready via Fast.ai.")
                else:
                    print(f"No automated fallback for {name}. It may still download via torchvision during training.")
            except Exception as fe:
                print(f"[ERROR] Fallback failed for {name}: {fe}")
            
    print("\n" + "=" * 50)
    print("All datasets processed!")
    print("=" * 50)

if __name__ == "__main__":
    download_all()
