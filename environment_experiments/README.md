# IDS Environment Experiments

This folder contains **alternative IDS environment configurations** for testing how different environment designs affect RL agent performance.

## Time Estimates (20GB RAM, CPU only)

| Experiment | Environment Change | DQN Training | PPO Training | Total (per run) |
|---|---|---|---|---|
| `env_exp_1` | Feature Subset (top-20) | ~8 min | ~12 min | **~20 min** |
| `env_exp_2` | Sliding Window (N=5) | ~15 min | ~20 min | **~35 min** |
| `env_exp_3` | Sequential Episodes | ~8 min | ~12 min | **~20 min** |

**Total time for all 3 experiments: ~75 minutes**

## How to Run

```bash
cd /home/abishik/HONOURS_PROJECT

# Run all environment experiments
source venv/bin/activate
python environment_experiments/run_env_experiments.py

# Run a specific experiment
python environment_experiments/run_env_experiments.py --exp 1
python environment_experiments/run_env_experiments.py --exp 2
python environment_experiments/run_env_experiments.py --exp 3
```

## Experiment Descriptions

### Exp 1: Feature Subset (Top-20 Features)
- Uses Random Forest feature importance to select top-20 most discriminative features
- Tests whether reducing dimensionality improves or hurts RL generalisation
- Hypothesis: fewer, better features → less noise → better zero-day detection

### Exp 2: Sliding Window (Temporal Context)
- Instead of single sample → agent observes last 5 consecutive samples concatenated
- State space: 78 × 5 = 390 features (flattened)
- Tests whether temporal context helps detect multi-step attacks
- Hypothesis: temporal patterns → better web attack detection

### Exp 3: Sequential Episodes (Time-Ordered)
- Episodes follow dataset order instead of random sampling
- More realistic — simulates real-time traffic flow
- Tests whether order matters for learning
- Hypothesis: sequential context → more realistic training dynamics

## Results
Results will be saved to `environment_experiments/results/` after running.
