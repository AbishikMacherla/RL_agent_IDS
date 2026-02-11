<p align="center">
  <h1 align="center">🛡️ RL-Enhanced Intrusion Detection System</h1>
  <p align="center">
    <strong>A Deep Reinforcement Learning approach to Network Intrusion Detection</strong>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#results">Results</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#usage">Usage</a>
  </p>
</p>

---

## Overview

This project explores the application of **Deep Reinforcement Learning (DRL)** for network intrusion detection, comparing a custom **Deep Q-Network (DQN)** agent against traditional machine learning baselines (**Random Forest** and **XGBoost**). The system is trained and evaluated on the [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset.

### Key Features

- 🤖 **Custom DQN Agent** — trained via a Gymnasium environment that simulates real-time traffic classification
- 🌲 **ML Baselines** — Random Forest and XGBoost classifiers for benchmark comparison
- 🧪 **Zero-Day Simulation** — label-exclusion mechanism to test detection of unseen attack types (e.g., DDoS)
- 📊 **Interactive Dashboard** — Streamlit-based visualisation of model performance metrics
- ⚙️ **Reproducible Pipeline** — data preprocessing, training, and evaluation scripts with documented hyperparameters

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CIC-IDS2017 Dataset                  │
│              (2.8M labelled traffic flows)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   Data Preprocessing    │
         │  (data_preprocessing.py)│
         │  • Feature scaling      │
         │  • Label encoding       │
         │  • Train/Test split     │
         └─────────┬───────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ┌──────────────┐  ┌──────────────┐
   │  ML Baselines │  │  RL Agent    │
   │ (ml_baselines │  │ (train_rl_  │
   │    .py)       │  │  agent.py)  │
   │               │  │             │
   │ • Random      │  │ • DQN with  │
   │   Forest      │  │   replay    │
   │ • XGBoost     │  │   buffer    │
   └──────┬───────┘  └──────┬──────┘
          │                  │
          └────────┬─────────┘
                   ▼
         ┌─────────────────────┐
         │  Unified Evaluation │
         │ (evaluate_all_      │
         │  models.py)         │
         │ • Standard scenario │
         │ • Zero-day scenario │
         └─────────────────────┘
```

---

## Results

### Standard Classification (CIC-IDS2017 Test Set)

| Model | Accuracy | Precision | Recall | F1 Score |
|:------|:--------:|:---------:|:------:|:--------:|
| **XGBoost** | 99.95% | 99.83% | 99.90% | 99.87% |
| **Random Forest** | 99.90% | 99.64% | 99.86% | 99.75% |
| **DQN Agent** | 91.20% | 69.92% | 97.04% | 81.28% |

> **Note:** The DQN agent achieves **97% recall** (attack detection rate) despite lower overall accuracy, demonstrating the RL agent's security-first reward design — it prioritises catching attacks over minimising false alarms.

---

## Project Structure

```
RL_agent_IDS/
├── Python_code/                # Source code
│   ├── data_preprocessing.py   # Dataset loading, cleaning, feature engineering
│   ├── ids_env.py              # Custom Gymnasium environment for RL training
│   ├── train_rl_agent.py       # DQN agent architecture & training loop
│   ├── ml_baselines.py         # Random Forest & XGBoost training
│   ├── evaluate_all_models.py  # Unified evaluation on untouched test set
│   ├── run_full_evaluation.py  # Multi-scenario evaluation (standard + zero-day)
│   ├── run_experiments.py      # Experiment runner
│   ├── hyperparameters.py      # Centralised hyperparameter definitions
│   └── dashboard.py            # Streamlit interactive dashboard
├── results/                    # Evaluation outputs
│   └── evaluation_results.json # Model performance metrics
├── requirements.txt            # Python dependencies
├── setup_env.sh                # One-command environment setup
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster DQN training

### Installation

```bash
# Clone the repository
git clone https://github.com/AbishikMacherla/RL_agent_IDS.git
cd RL_agent_IDS

# Set up virtual environment & install dependencies
chmod +x setup_env.sh
./setup_env.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset

This project uses the **CIC-IDS2017** dataset. Download it from the [official source](https://www.unb.ca/cic/datasets/ids-2017.html) and place the CSV files in a `data/` directory:

```
data/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── Wednesday-workingHours.pcap_ISCX.csv
```

---

## Usage

### 1. Preprocess Data

```bash
cd Python_code
python data_preprocessing.py
```

This generates scaled feature matrices and encoded labels in `processed_data/`.

### 2. Train ML Baselines

```bash
python ml_baselines.py
```

Trains Random Forest and XGBoost models, saves them to `models/`.

### 3. Train DQN Agent

```bash
# Standard training
python train_rl_agent.py

# Zero-day simulation (exclude DDoS from training)
python train_rl_agent.py --exclude ddos "dos hulk" --output dqn_no_ddos.pth
```

### 4. Evaluate All Models

```bash
python run_full_evaluation.py
```

Runs all models through standard and zero-day scenarios, outputs results to `results/`.

### 5. Interactive Dashboard

```bash
streamlit run dashboard.py
```

---

## Hyperparameters

### DQN Agent

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| Replay Buffer | 100,000 | Experience replay memory size |
| Batch Size | 64 | Minibatch size for training |
| Gamma (γ) | 0.99 | Discount factor |
| Tau (τ) | 0.001 | Soft update coefficient |
| Learning Rate | 5×10⁻⁴ | Adam optimiser learning rate |
| Epsilon Decay | 0.999 | Exploration decay rate |
| Episodes | 2,000 | Training episodes |
| Network | 78→64→64→2 | Fully connected layers |

### ML Baselines

| Parameter | Random Forest | XGBoost |
|:----------|:------------:|:-------:|
| n_estimators | 100 | 100 |
| max_depth | 20 | 6 |
| learning_rate | — | 0.1 |

---

## Reward Structure

The DQN agent uses an **asymmetric reward function** designed for security-critical applications:

| Prediction | Ground Truth | Reward | Rationale |
|:-----------|:------------|:------:|:----------|
| Block (1) | Malicious (1) | **+10** | True Positive — correctly detected attack |
| Allow (0) | Benign (0) | **+1** | True Negative — correct classification |
| Allow (0) | Malicious (1) | **-10** | False Negative — missed attack (dangerous) |
| Block (1) | Benign (0) | **-1** | False Positive — false alarm (acceptable) |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-0081A5)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)

---

## License

This project was developed as part of a BEng Honours dissertation at Edinburgh Napier University.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/AbishikMacherla">Abishik Macherla</a></sub>
</p>
