import os
import sys
import subprocess

# 1. Set Path (Replaces $env:PYTHONPATH)
# Colab runs on Linux, so we use forward slashes.
project_root = "./RePlayItStraight"
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{project_root}"

# Ensure the results directory exists
results_dir = "./RePlayItStraight/results"
os.makedirs(results_dir, exist_ok=True)

# 2. Define all datasets and their correct class counts
datasets = [
    {"Name": "CIFAR10", "Classes": 10},
    {"Name": "CIFAR20", "Classes": 20},
    {"Name": "CIFAR50", "Classes": 50},
    {"Name": "CIFAR100", "Classes": 100},
    {"Name": "FashionMNIST", "Classes": 10},
    {"Name": "ImageNet30", "Classes": 30},
    {"Name": "Imagenette", "Classes": 10},
    {"Name": "Imagewoof", "Classes": 10},
    {"Name": "MNIST", "Classes": 10},
    {"Name": "QMNIST", "Classes": 10},
    {"Name": "SVHN", "Classes": 10},
    {"Name": "TinyImageNet", "Classes": 200}
]

print("Starting sequential dataset evaluation...")

# 3. Loop through each dataset and run the training script
for ds in datasets:
    ds_name = ds["Name"]
    ds_classes = ds["Classes"]
    log_file = os.path.join(results_dir, f"{ds_name}_Log.txt")

    print(f"\n{'='*57}")
    print(f" Running Experiment for: {ds_name} (Classes: {ds_classes}) ")
    print(f" Logging to: {log_file}")
    print(f"{'='*57}")

    # Build the command just like in PowerShell
    cmd = [
        "python", "-u", "./RePlayItStraight/src/re_play_it_straight/main_re_play_it_straight.py",
        "--gpu", "0",
        "--data_path", "../data",
        "--dataset", ds_name,
        "--n-class", str(ds_classes),
        "--model", "ResNet18",
        "--method", "ImprovedUncertainty",
        "--uncertainty", "LeastConfidence",
        "--n-query", "1000",
        "--epochs", "10",
        "--batch-size", "128",
        "--n_split", "3",
        "--cycle", "10",
        "--boot_epochs", "20",
        "--discount_rs2", "2",
        "--boost_threshold", "0.07",
        "--workers", "0", # Kept at 0 as requested
        "--seed", "42",
        "--save_path", results_dir
    ]

    # Execute the script, capturing output to mimic Tee-Object
    with open(log_file, "w", encoding="utf-8") as f:
        # bufsize=1 and text=True ensures line-by-line real-time streaming
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            # Print to screen (Colab console)
            print(line, end="")
            # Write to log file
            f.write(line)
            f.flush()
        
        process.wait()

print("\n" + "="*57)
print("All dataset evaluations completed successfully!")
print("="*57)