import os
import kagglehub
import shutil

def download_from_kaggle(dataset_handle, target_dir):
    """
    Downloads a dataset from Kaggle using kagglehub and moves it to the target directory.
    """
    if os.path.exists(target_dir):
        # Check if directory is not empty
        if os.listdir(target_dir):
            print(f"Directory {target_dir} already exists and is not empty. Skipping download.")
            return target_dir
            
    print(f"Downloading {dataset_handle} from Kaggle...")
    path = kagglehub.dataset_download(dataset_handle)
    
    print(f"Moving dataset from {path} to {target_dir}...")
    if not os.path.exists(os.path.dirname(target_dir)):
        os.makedirs(os.path.dirname(target_dir))
        
    # If kagglehub returns a path to a directory, move its contents
    if os.path.isdir(path):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(path, target_dir)
    else:
        # If it's a single file, just move it
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        shutil.copy(path, os.path.join(target_dir, os.path.basename(path)))
        
    print(f"Dataset ready at {target_dir}")
    return target_dir
