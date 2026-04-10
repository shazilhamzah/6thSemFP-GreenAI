# Re-Play it Straight: Green AI Training Framework

This project implements the **Re-Play it Straight** framework, an efficient deep learning training methodology designed for **Green AI**. The framework combines random subset sampling with Active Learning to reach target accuracies with significantly lower energy consumption and computational cost.

## 🛠 Prerequisites

*   **Python Version:** `3.10.11` (Ensure this is installed before proceeding).

## 🚀 Setup & Installation

Follow these steps to set up your environment and prepare the project for training:

### Step 1: Create a Virtual Environment
Open your terminal (PowerShell or CMD) in the project root directory and run:
```powershell
python -m venv greenai
```

### Step 2: Activate the Virtual Environment
*   **PowerShell:**
    ```powershell
    .\greenai\Scripts\Activate.ps1
    ```
*   **CMD:**
    ```cmd
    .\greenai\Scripts\activate.bat
    ```

### Step 3: Install Dependencies
Ensure you are in the project root and install the required packages:
```powershell
pip install -r RePlayItStraight\src\re_play_it_straight\requirements.txt
```

### Step 4: Configure PYTHONPATH
The scripts require the `RePlayItStraight` directory to be in your Python path:
```powershell
# PowerShell
$env:PYTHONPATH = ".\RePlayItStraight"
# CMD
set PYTHONPATH=.\RePlayItStraight
```

---

## 🏃 Running the Code

### 1. Processing the Whole Dataset (Full Training)
Use this command for high-performance hardware or if you want to run the full research experiment on the complete MNIST dataset (60,000 samples).

```powershell
python RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --batch-size 128 --target_accuracy 99 --epochs 200
```

### 2. Processing a Subset (Weak Laptop / Fast Testing)
If you are running on a machine with limited CPU/Memory, use the `--subset` flag to reduce the data volume. This configuration uses only 1,000 samples (~98% reduction).

```powershell
python RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --subset 1000 --batch-size 32 --epochs 5 --n-query 100
```

### 3. Cross-Validation Run
To run the framework with 5-fold cross-validation:
```powershell
python RePlayItStraight\src\re_play_it_straight\main_cross_validation_re_play_it_straight.py --dataset MNIST --subset 1000
```

---

## ⚡ Automation
For convenience, you can use the provided PowerShell script to run the environment setup and training in one go:
```powershell
.\commands_to_train.ps1
```

## 📄 Notes
*   **GPU Support:** The code will automatically detect and use a CUDA-capable GPU if available.
*   **Energy Tracking:** Energy consumption is tracked via `codecarbon` and saved to `emissions.csv`.
