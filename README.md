<p align="center">
  <h1 align="center">RL-Enhanced Intrusion Detection System</h1>
  <p align="center">
    <strong>Deep Reinforcement Learning for Autonomous Network Defence</strong>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#experiments">Experiments</a> •
    <a href="#results">Results</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#dashboard">Dashboard</a>
  </p>
</p>

---

## Overview

This project explores **Deep Reinforcement Learning (DRL)** for network intrusion detection, comparing a custom **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** agent against traditional ML baselines (**Random Forest** and **XGBoost**).

The system is evaluated across **4 experimental scenarios** including standard classification, zero-day attack detection, and cross-dataset generalisation.

### Key Features

- **DQN + PPO Agents** — two RL approaches trained via custom Gymnasium environments
- **ML Baselines** — Random Forest and XGBoost for benchmark comparison
- **Zero-Day Simulation** — label-exclusion to test detection of unseen attack types
- **Cross-Dataset Generalisation** — train on CIC-IDS2017, test on CIC-IoT-2023
- **Streamlit Dashboard** — interactive visualisation with Plotly charts
- **8 DQN Experiments** — systematic hyperparameter tuning (reward structure, architecture, training)
- **Reproducible Pipeline** — documented hyperparameters and one-command experiment runner
---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           CIC-IDS2017 + CIC-IoT-2023 Datasets              │
│           (2.8M + 1.5M labelled traffic flows)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌──────────────────────────┐
           │    Data Preprocessing    │
           │  (data_preprocessing.py) │
           │  • Feature scaling       │
           │  • Label encoding        │
           │  • Feature mapping (12   │
           │    common features for   │
           │    cross-dataset)        │
           └──────────┬──────────────┘
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
    ┌───────────────┐   ┌───────────────┐
    │  ML Baselines │   │  RL Agents    │
    │ (ml_baselines │   │ • DQN (train_ │
    │    .py)       │   │   rl_agent)   │
    │               │   │ • PPO (train_ │
    │ • Random      │   │   ppo_agent)  │
    │   Forest      │   │               │
    │ • XGBoost     │   │ Gym Env:      │
    └───────┬───────┘   │ ids_env.py    │
            │           └───────┬───────┘
            └─────────┬─────────┘
                      ▼
        ┌───────────────────────────┐
        │  Master Experiment Runner │
        │  (run_all_scenarios.py)   │
        │  • 4 Scenarios            │
        │  • ROC-AUC + Latency      │
        │  • Confusion Matrices     │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────────┐
        │   Results Report          │
        │  (visualise_results.py)   │
        │   • Plotly charts         │
        │   • Confusion matrices    │
        │   • Self-contained HTML   │
        └───────────────────────────┘
```

---

## Experiments

### 4 Dissertation Scenarios

| # | Scenario | Purpose |
|:-:|:---------|:--------|
| 1 | **Standard Classification** | All models on CIC-IDS2017 (78 features) — baseline comparison |
| 2 | **Zero-Day DDoS** | RL trained WITHOUT DDoS → tested on DDoS-only data |
| 3 | **Zero-Day Web Attacks** | RL trained WITHOUT web attacks → tested on web-only data |
| 4 | **Cross-Dataset** | Train on CIC-IDS2017 → Test on CIC-IoT-2023 (12 features) |

### 8 DQN Hyperparameter Experiments

Systematic tuning of reward structure, network architecture, and training duration to understand how each factor affects agent behaviour.

---

## Results

Results are stored in `results/all_scenarios_results.json` and viewable via the dashboard.

> **Key Finding:** Reward structure is the primary driver of DQN agent behaviour. Symmetric rewards (+1/+1/-1/-1) achieve the best F1 score (93.1%) with only 8K false positives, while asymmetric (10:1) rewards achieve 98% recall but 94K false positives.

---

## Project Structure

```
HONOURS_PROJECT/
├── Python_code/                     # Source code
│   ├── data_preprocessing.py        # Dataset loading, cleaning, feature mapping
│   ├── ids_env.py                   # Custom Gymnasium environment for RL
│   ├── train_rl_agent.py            # DQN agent architecture & training
│   ├── train_ppo_agent.py           # PPO agent (Stable-Baselines3)
│   ├── ml_baselines.py              # Random Forest & XGBoost training
│   ├── run_dqn_experiments.py       # 8 DQN hyperparameter experiments
│   ├── run_all_scenarios.py         # Master experiment runner (4 scenarios)
│   ├── hyperparameters.py           # Hyperparameter reference documentation
│   ├── visualise_results.py         # Figure generator (matplotlib + seaborn)
│   └── visualise_results.ipynb      # Jupyter notebook version
├── environment_experiments/         # Alternative IDS environment experiments
│   ├── env_variants.py              # 3 environment variants
│   ├── run_env_experiments.py       # Experiment runner
│   └── README.md                    # Time estimates & usage
├── results/                         # Experimental results
│   ├── all_scenarios_results.json   # All 4 scenario metrics
│   ├── dqn_experiments.json         # 8 DQN experiment metrics
│   └── figures/                     # Generated PNG figures for dissertation
├── LaTex/                           # Dissertation source
│   ├── main.tex                     # Main document
│   ├── references.bib               # Bibliography
│   └── sections/                    # Section files (methodology, results, etc.)
├── requirements.txt                 # Python dependencies
├── setup_env.sh                     # One-command environment setup
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster training

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

This project uses **CIC-IDS2017** and **CIC-IoT-2023** datasets:

- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — place CSVs in `data/`
- [CIC-IoT-2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) — place CSVs in `data/CIC-IoT-2023/`

---

## Usage

### 1. Preprocess Data

```bash
cd Python_code
python data_preprocessing.py
```

### 2. Run All Experiments

```bash
# Run all 4 scenarios
python run_all_scenarios.py

# Or run a specific scenario (1-4)
python run_all_scenarios.py --scenario 1
```

### 3. Run DQN Hyperparameter Experiments

```bash
python run_dqn_experiments.py
```

### 4. Generate Figures

```bash
# Generate all figures for dissertation
python visualise_results.py

# Or use the Jupyter notebook
jupyter notebook visualise_results.ipynb
```

### 5. Run Environment Experiments (Optional)

```bash
# Run all 3 environment variants (~75 min)
python environment_experiments/run_env_experiments.py

# Or run a specific experiment
python environment_experiments/run_env_experiments.py --exp 1
```

---

## Results

Results are stored in `results/all_scenarios_results.json` and visualised as publication-quality figures in `results/figures/`.

- **Scenario charts** — grouped bar charts per scenario
- **Cross-scenario summary** — F1 comparison across all 4 scenarios
- **DQN experiments** — F1 progression, precision–recall trade-off, error analysis
- **Confusion matrices** — heatmaps with counts and percentages
- **Reward impact** — how reward structure affects detection behaviour

---

## Hyperparameters

### DQN Agent (Best Config — Exp 7)

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| Replay Buffer | 100,000 | Experience replay memory size |
| Batch Size | 64 | Minibatch size for training |
| Gamma (γ) | 0.99 | Discount factor |
| Tau (τ) | 0.001 | Soft update coefficient |
| Learning Rate | 5×10⁻⁴ | Adam optimiser learning rate |
| Epsilon Decay | 0.999 | Exploration decay rate |
| Episodes | 5,000 | Training episodes |
| Network | 78→64→64→2 | Fully connected layers |
| Rewards | +1/+1/-1/-1 | Symmetric (TP/TN/FN/FP) |

### PPO Agent

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| Timesteps | 1,000,000 | Total training steps |
| Learning Rate | 3×10⁻⁴ | Adam optimiser |
| Clip Range | 0.2 | PPO clipping parameter |
| GAE Lambda | 0.95 | Advantage estimation |
| Entropy Coef | 0.01 | Exploration encouragement |

### ML Baselines

| Parameter | Random Forest | XGBoost |
|:----------|:------------:|:-------:|
| n_estimators | 100 | 100 |
| max_depth | 20 | 6 |
| learning_rate | — | 0.1 |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-0081A5)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-4c72b0)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75?logo=plotly&logoColor=white)

---

## License

This project was developed as part of a BEng Honours dissertation at Edinburgh Napier University.

---

<p align="center">
  <sub>Built with by <a href="https://github.com/AbishikMacherla">Abishik Macherla</a></sub>
</p>
