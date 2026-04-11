$ks = @(250, 500, 1000, 2000, 4000)
$epochs = @(3, 5, 10, 15, 20)
$discounts = @(0.5, 1, 2, 3, 4)
$boot_epochs = @(3, 7, 15, 25, 31, 37)

$dataset_name = "CIFAR10"
$seed = 42
$best_k = 1000
$best_epochs = 10
$best_boot_epochs = 20
$best_discount = 2

# 1. Activate Environment and Set Path
.\greenai\Scripts\Activate.ps1
$env:PYTHONPATH = ".\RePlayItStraight"

# Ensure the results directory exists
$results_dir = ".\RePlayItStraight\results"
if (!(Test-Path -Path $results_dir)) {
    New-Item -ItemType Directory -Path $results_dir | Out-Null
}

Write-Host "Dataset selected $($dataset_name)!"

Write-Host "Running Experiment for sensibility over k..."
foreach ($current_k in $ks) {
    $path_log_file = "$results_dir\res_$($dataset_name)_seed_$($seed)_k_$($current_k)_epochs_$($best_epochs)_discount_$($best_discount)_bootep_$($best_boot_epochs).log"
    Write-Host "Running experiment with k: $current_k, epochs: $best_epochs, boot epochs: $best_boot_epochs, discount: $best_discount..."
    
    # 2>&1 merges streams, ForEach-Object { "$_" } prevents red errors, Tee-Object splits the output to screen and file
    python -u .\RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --gpu 0 --data_path "..\data"  --workers 0 --dataset $dataset_name --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query $current_k --epochs $best_epochs --batch-size 128 --n_split 3 --cycle 20 --boot_epochs $best_boot_epochs --discount_rs2 $best_discount --boost_threshold 0.07 --seed $seed 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $path_log_file
}

Write-Host "Running Experiment for sensibility over epochs..."
foreach ($current_epochs in $epochs) {
    $path_log_file = "$results_dir\res_$($dataset_name)_seed_$($seed)_k_$($best_k)_epochs_$($current_epochs)_discount_$($best_discount)_bootep_$($best_boot_epochs).log"
    Write-Host "Running experiment with k: $best_k, epochs: $current_epochs, boot epochs: $best_boot_epochs, discount: $best_discount..."
    
    python -u .\RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --gpu 0 --data_path "..\data"  --workers 0 --dataset $dataset_name --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query $best_k --epochs $current_epochs --batch-size 128 --n_split 3 --cycle 20 --boot_epochs $best_boot_epochs --discount_rs2 $best_discount --boost_threshold 0.07 --seed $seed 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $path_log_file
}

Write-Host "Running Experiment for sensibility over discounts..."
foreach ($current_discount in $discounts) {
    $path_log_file = "$results_dir\res_$($dataset_name)_seed_$($seed)_k_$($best_k)_epochs_$($best_epochs)_discount_$($current_discount)_bootep_$($best_boot_epochs).log"
    Write-Host "Running experiment with k: $best_k, epochs: $best_epochs, boot epochs: $best_boot_epochs, discount: $current_discount..."
    
    python -u .\RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --gpu 0 --data_path "..\data"  --workers 0 --dataset $dataset_name --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query $best_k --epochs $best_epochs --batch-size 128 --n_split 3 --cycle 20 --boot_epochs $best_boot_epochs --discount_rs2 $current_discount --boost_threshold 0.07 --seed $seed 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $path_log_file
}

Write-Host "Running Experiment for sensibility over boot epochs..."
foreach ($current_boot_epochs in $boot_epochs) {
    $path_log_file = "$results_dir\res_$($dataset_name)_seed_$($seed)_k_$($best_k)_epochs_$($best_epochs)_discount_$($best_discount)_bootep_$($current_boot_epochs).log"
    Write-Host "Running experiment with k: $best_k, epochs: $best_epochs, boot epochs: $current_boot_epochs, discount: $best_discount..."
    
    python -u .\RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --gpu 0 --data_path "..\data"  --workers 0 --dataset $dataset_name --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query $best_k --epochs $best_epochs --batch-size 128 --n_split 3 --cycle 20 --boot_epochs $current_boot_epochs --discount_rs2 $best_discount --boost_threshold 0.07 --seed $seed 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $path_log_file
}

Write-Host "Completed!"