# Re-Play it Straight: Green AI Training Framework

This project implements Re-Play it Straight, a training framework for deep learning models that tries to reach target accuracies while burning as little energy as possible. It combines random subset sampling with Active Learning to cut down computation significantly.

## Prerequisites

Python `3.10.11` — install this before anything else.

## Setup

### Step 1: Create a virtual environment

```powershell
python -m venv greenai
```

### Step 2: Activate it

PowerShell:
```powershell
.\greenai\Scripts\Activate.ps1
```

CMD:
```cmd
.\greenai\Scripts\activate.bat
```

### Step 3: Install dependencies

```powershell
pip install -r RePlayItStraight\src\re_play_it_straight\requirements.txt
```

### Step 4: Set PYTHONPATH

```powershell
# PowerShell
$env:PYTHONPATH = ".\RePlayItStraight"
# CMD
set PYTHONPATH=.\RePlayItStraight
```

---

## Running the Code

### Full dataset (high-end hardware)

Runs on the complete MNIST dataset (60,000 samples):

```powershell
python RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --batch-size 128 --target_accuracy 99 --epochs 200
```

### Subset run (low-spec machine or quick testing)

Uses 1,000 samples instead of the full dataset:

```powershell
python RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --subset 1000 --batch-size 32 --epochs 5 --n-query 100
```

### Cross-validation

5-fold cross-validation on a 1,000-sample subset:

```powershell
python RePlayItStraight\src\re_play_it_straight\main_cross_validation_re_play_it_straight.py --dataset MNIST --subset 1000
```

---

## Automation

A PowerShell script handles setup and training in one shot:

```powershell
.\commands_to_train.ps1
```

## Notes

- If a CUDA GPU is available, the code will use it automatically.
- Energy usage is tracked via `codecarbon` and saved to `emissions.csv`.