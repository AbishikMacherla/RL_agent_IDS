# Experiment Log: RL-Enhanced IDS

## Experiment 1: Baseline Comparison (Standard Training)
**Date:** 2025-01-29
**Dataset:** CIC-IDS2017 (Train 80% / Test 20%)

### ML Baselines

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 99.90% | 99.64% | 99.86% | 99.75% |
| XGBoost | 99.95% | 99.83% | 99.90% | 99.87% |

**Random Forest Hyperparameters:**
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- random_state: 42

**XGBoost Hyperparameters:**
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- gamma: 0
- random_state: 42

---

### RL Agents

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|--------|----------|-------|
| DQN | 91.20% | 70.00% | **97.00%** | 81.00% | High recall! Security-first works |
| PPO | 69.23% | 31.55% | 48.20% | 38.14% | Needs more training |

**DQN Hyperparameters:**
- Episodes: 2000
- Max steps per episode: 1000
- Learning rate: 5e-4
- Epsilon start: 1.0
- Epsilon end: 0.01
- Epsilon decay: 0.999
- Batch size: 64
- Buffer size: 100,000
- Gamma: 0.99
- Tau (soft update): 1e-3
- Network: 2 hidden layers, 64 neurons each

**PPO Hyperparameters:**
- Total timesteps: 200,000
- Learning rate: 3e-4
- N steps: 2048
- Batch size: 64
- N epochs: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coefficient: 0.01

**Reward Structure:**
- True Positive (catch attack): +10
- True Negative (allow benign): +1
- False Positive (block benign): -1
- False Negative (miss attack): -10

---

## Experiment 2: Zero-Day Simulation (Exclude DDoS)
**Date:** 2025-01-29
**Status:** IN PROGRESS

**Setup:**
- Training: All attacks EXCEPT DDoS
- Testing: Specifically on DDoS attacks
- Command: `python Python_code/train_rl_agent.py --exclude "ddos" --output dqn_no_ddos.pth`

**Hyperparameters:** Same as Experiment 1 DQN

**Results:** _(To be filled after training)_

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|--------|----------|-------|
| DQN (no DDoS) | - | - | - | - | Zero-day test |

---

## Experiment Ideas (Future)

### Experiment 3: Aggressive Reward Tuning
- TP: +20, TN: +1, FP: -1, FN: -50
- Goal: Push recall even higher

### Experiment 4: Longer Training
- DQN: 5000 episodes
- PPO: 500k-1M timesteps

### Experiment 5: Network Architecture
- Try 128 neurons per layer
- Try 3 hidden layers

### Experiment 6: Cross-Dataset Generalization
- Train on CIC-IDS2017
- Test on CIC-IoT-2023

---

## Key Observations

1. **ML vs RL Trade-off:** ML achieves higher accuracy on static data, but DQN's 97% recall shows RL can prioritize security goals through reward shaping.

2. **PPO Challenges:** PPO requires significantly more training than DQN for this task. May need curriculum learning or better exploration.

3. **Class Imbalance:** Dataset is ~80% benign. This affects RL exploration - agents can get "stuck" predicting benign.

4. **Next Step:** Test zero-day detection to demonstrate RL's adaptability advantage.
