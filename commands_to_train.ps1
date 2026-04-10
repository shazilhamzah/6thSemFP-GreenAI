# Get the directory where the script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Change directory to RePlayItStraight
cd "$ScriptDir\RePlayItStraight"

# Activate the virtual environment
# We use the absolute path to be sure
$VenvPath = "$ScriptDir\greenai"
if (Test-Path "$VenvPath\Scripts\Activate.ps1") {
    . "$VenvPath\Scripts\Activate.ps1"
} else {
    Write-Error "Virtual environment not found at $VenvPath"
    exit 1
}

# Install requirements
pip install -r src\re_play_it_straight\requirements.txt

# Set PYTHONPATH to include the current directory
$env:PYTHONPATH = "$ScriptDir\RePlayItStraight"

# Run the standard training script with a small subset
# Workers 0 is essential for weak laptops to avoid memory pressure
# python src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --subset 1000 --batch-size 32 --target_accuracy 80 --epochs 5 --boot_epochs 2 --n_split 5 --n-query 100 --workers 0

# Run an ULTRA-SMALL version if the above is still slow
python src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --subset 200 --batch-size 16 --target_accuracy 70 --epochs 2 --boot_epochs 1 --n_split 2 --n-query 50 --workers 0

# Run the cross-validation script with reduced data
Write-Host "Starting Cross-Validation with reduced data..."
python src\re_play_it_straight\main_cross_validation_re_play_it_straight.py --dataset MNIST --subset 200 --batch-size 16 --target_accuracy 75 --epochs 2 --boot_epochs 1 --n_split 2 --n-query 50 --workers 0
