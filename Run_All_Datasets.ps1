# 1. Activate Environment and Set Path
.\greenai\Scripts\Activate.ps1
$env:PYTHONPATH = ".\RePlayItStraight"

# Ensure the results directory exists
$results_dir = ".\RePlayItStraight\results"
if (!(Test-Path -Path $results_dir)) {
    New-Item -ItemType Directory -Path $results_dir | Out-Null
}

# 2. Define all datasets from your folder and their correct class counts
$datasets = @(
    @{ Name="CIFAR10"; Classes=10 },
    @{ Name="CIFAR20"; Classes=20 },
    @{ Name="CIFAR50"; Classes=50 },
    @{ Name="CIFAR100"; Classes=100 },
    @{ Name="FashionMNIST"; Classes=10 },
    @{ Name="ImageNet30"; Classes=30 },
    @{ Name="Imagenette"; Classes=10 },
    @{ Name="Imagewoof"; Classes=10 },
    @{ Name="MNIST"; Classes=10 },
    @{ Name="QMNIST"; Classes=10 },
    @{ Name="SVHN"; Classes=10 },
    @{ Name="TinyImageNet"; Classes=200 }
)

Write-Host "Starting sequential dataset evaluation..."

# 3. Loop through each dataset and run the training script
foreach ($ds in $datasets) {
    $ds_name = $ds.Name
    $ds_classes = $ds.Classes
    $log_file = "$results_dir\${ds_name}_Log.txt"

    Write-Host ""
    Write-Host "========================================================="
    Write-Host " Running Experiment for: $ds_name (Classes: $ds_classes) "
    Write-Host " Logging to: $log_file"
    Write-Host "========================================================="

    # The python -u and Tee-Object pipeline ensures you see it on screen AND saves to the text file
    python -u .\RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --gpu 0 --data_path "..\data" --dataset $ds_name --n-class $ds_classes --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query 1000 --epochs 10 --batch-size 128 --n_split 3 --cycle 10 --boot_epochs 20 --discount_rs2 2 --boost_threshold 0.07 --workers 0 --seed 42 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log_file
}

Write-Host "========================================================="
Write-Host "All dataset evaluations completed successfully!"
Write-Host "========================================================="