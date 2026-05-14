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

## Running the Code (Manual Execution)

### Full dataset (high-end hardware)

Runs on the complete MNIST dataset (60,000 samples):

```powershell
python RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --batch-size 128 --target_accuracy 99 --epochs 200
```

### Subset run (low-spec machine or quick testing)

Uses 1,000 samples instead of the full dataset. Set `--workers 0` to prevent memory issues:

```powershell
python RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --subset 1000 --batch-size 32 --epochs 5 --n-query 100 --workers 0
```

### Ultra-Light run (if 1,000 samples are still slow)

Uses only 200 samples and minimal epochs:

```powershell
python RePlayItStraight\src\re_play_it_straight\main_re_play_it_straight.py --dataset MNIST --subset 200 --batch-size 16 --epochs 2 --n-query 50 --workers 0
```

### Cross-validation

5-fold cross-validation on a 200-sample subset:

```powershell
python RePlayItStraight\src\re_play_it_straight\main_cross_validation_re_play_it_straight.py --dataset MNIST --subset 200 --batch-size 16 --workers 0
```
## Automated Batch Execution

### Google Colab Pipeline
If you are running this project on Google Colab (T4 GPU), the entire end-to-end pipeline is consolidated into a single Jupyter Notebook: `GREENAI_AI_PROJ.ipynb`.

Simply open the notebook in Colab and click "Run All". The notebook will automatically:

1. Clone the repository and install dependencies.

2. Download and format the required datasets.

3. Sequentially run the training orchestrator across the datasets.

4. Process the generated telemetry and output visual dashboards.

### Local PowerShell Automation
To automatically loop through all available datasets (CIFAR10, CIFAR100, TinyImageNet, etc.) sequentially, map the correct class counts, and output terminal logs:

```powershell
.\Run_All_Datasets.ps1
```

---

## Analytics & Dashboards
As the training runs (locally or in Colab), the framework extracts and saves comprehensive telemetry to the ./RePlayItStraight/results directory.

### Data Outputs
- {dataset}_Log.txt: The raw terminal output capturing the entire training process.
- {dataset}_cycle_log.csv: Active Learning cycle metrics (Labeled samples, Accuracy, F1, Cumulative Backward Steps).
- {dataset}_epoch_log.csv: Fine-grained training dynamics (Loss, Learning Rate decay, Epoch accuracy).
- best_model_{dataset}.pth: Model checkpoints saved upon accuracy improvement.

### Dashboards & Visualizations
You do not need to run separate analytics scripts. The final cells in `GREENAI_AI_PROJ.ipynb` (or your local Jupyter environment) automatically process the CSV logs to generate research-grade visualizations replicating the paper's metrics:

1. **Individual Dashboards**: Multi-panel plots (PNG/SVG) showing Computational Scaling, Data Usage, and Training Dynamics (Loss/Accuracy/Learning Rate) for each dataset.

2. **Cross-Dataset Comparison**: Generates a combined dashboard to compare the performance of algorithms across different datasets on a single axis.

3. **Excel Export**: Automatically exports clean analytical data to `.xlsx` files for native MS Office charting.

## Notes
- If a CUDA GPU is available, the code will hook into it automatically (cuda:0).
- Energy usage is tracked via codecarbon. Please note that consumer GPUs might lack NVML energy tracking support; however, Datacenter GPUs (like Colab's T4) will successfully log exact kg CO2eq emissions.
