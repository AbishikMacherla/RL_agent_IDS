#!/usr/bin/env python3
"""
RL-Enhanced IDS — Environment Experiments Runner
Tests DQN agent performance across 3 different IDS environment configurations.

Usage:
    python run_env_experiments.py           # Run all experiments
    python run_env_experiments.py --exp 1   # Run experiment 1 only
    python run_env_experiments.py --exp 2   # Run experiment 2 only
    python run_env_experiments.py --exp 3   # Run experiment 3 only

Memory: Configured for ~20GB RAM usage. Uses chunked data loading where possible.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/abishik/HONOURS_PROJECT/Python_code')
sys.path.insert(0, '/home/abishik/HONOURS_PROJECT/environment_experiments')

from env_variants import IdsEnvFeatureSubset, IdsEnvSlidingWindow, IdsEnvSequential
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = '/home/abishik/HONOURS_PROJECT/environment_experiments/results'
MODEL_DIR = '/home/abishik/HONOURS_PROJECT/environment_experiments/models'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Memory management
torch.set_num_threads(4)  # Limit CPU threads to avoid memory spikes

# Use best reward config from main experiments (symmetric)
REWARD_CONFIG = {'tp': 1.0, 'tn': 1.0, 'fn': -1.0, 'fp': -1.0}

# Training config (matched to best main experiment: Exp 7)
DQN_CONFIG = {
    'n_episodes': 3000,
    'max_t': 1000,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.999,
    'buffer_size': 100000,
    'batch_size': 64,
    'gamma': 0.99,
    'tau': 0.001,
    'lr': 5e-4,
}


# ============================================================================
# DQN AGENT (self-contained to avoid import issues)
# ============================================================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.qnetwork_local = DQN(state_size, action_size)
        self.qnetwork_target = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['lr'])

        self.memory = ReplayBuffer(config['buffer_size'], config['batch_size'])
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % 4 == 0 and len(self.memory) > self.config['batch_size']:
            self._learn()

    def act(self, state, eps=0.0):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(range(self.action_size))

    def _learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.config['gamma'] * Q_targets_next * (1 - dones)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                              self.qnetwork_local.parameters()):
            target_param.data.copy_(
                self.config['tau'] * local_param.data + (1 - self.config['tau']) * target_param.data)


# ============================================================================
# TRAINING
# ============================================================================
def train_dqn(env, config, model_name='model.pth'):
    """Train DQN agent on given environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config)

    scores = []
    eps = config['eps_start']

    for episode in range(1, config['n_episodes'] + 1):
        state, _ = env.reset()
        score = 0
        for t in range(config['max_t']):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        eps = max(config['eps_end'], config['eps_decay'] * eps)

        if episode % 500 == 0:
            avg = np.mean(scores[-100:])
            print(f"    Episode {episode}/{config['n_episodes']} | Avg Score: {avg:.1f} | ε: {eps:.3f}")

    # Save model
    path = os.path.join(MODEL_DIR, model_name)
    torch.save(agent.qnetwork_local.state_dict(), path)
    print(f"    Model saved: {path}")
    return agent, scores


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_agent(agent, env):
    """Evaluate trained DQN agent on test data."""
    X_test = env.X_test
    y_test = env.y_test
    predictions = []

    for i in range(len(X_test)):
        state = X_test[i]
        action = agent.act(state, eps=0.0)  # greedy
        predictions.append(action)

    y_pred = np.array(predictions)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary',
                                                        zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1': round(f1 * 100, 2),
        'ConfusionMatrix': cm.tolist(),
        'TN': int(cm[0][0]),
        'FP': int(cm[0][1]),
        'FN': int(cm[1][0]),
        'TP': int(cm[1][1]),
    }

    print(f"    Accuracy: {results['Accuracy']}% | Precision: {results['Precision']}% | "
          f"Recall: {results['Recall']}% | F1: {results['F1']}%")
    print(f"    TN: {results['TN']:,} | FP: {results['FP']:,} | FN: {results['FN']:,} | TP: {results['TP']:,}")
    return results


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================
EXPERIMENTS = {
    1: {
        'name': 'Feature Subset (Top-20)',
        'env_class': IdsEnvFeatureSubset,
        'env_kwargs': {'top_k': 20, 'reward_config': REWARD_CONFIG},
        'model_name': 'env_exp1_feature_subset.pth',
    },
    2: {
        'name': 'Sliding Window (N=5)',
        'env_class': IdsEnvSlidingWindow,
        'env_kwargs': {'window_size': 5, 'reward_config': REWARD_CONFIG},
        'model_name': 'env_exp2_sliding_window.pth',
    },
    3: {
        'name': 'Sequential Episodes',
        'env_class': IdsEnvSequential,
        'env_kwargs': {'reward_config': REWARD_CONFIG},
        'model_name': 'env_exp3_sequential.pth',
    },
}


# ============================================================================
# MAIN
# ============================================================================
def run_experiment(exp_id):
    """Run a single environment experiment."""
    exp = EXPERIMENTS[exp_id]
    print(f"\n{'='*60}")
    print(f"  Environment Experiment {exp_id}: {exp['name']}")
    print(f"{'='*60}")

    # Create environment
    print("\n  Creating environment...")
    start_time = time.time()
    env = exp['env_class'](**exp['env_kwargs'])

    # Train DQN
    print(f"\n  Training DQN ({DQN_CONFIG['n_episodes']} episodes)...")
    agent, scores = train_dqn(env, DQN_CONFIG, exp['model_name'])
    train_time = time.time() - start_time

    # Evaluate
    print("\n  Evaluating...")
    results = evaluate_agent(agent, env)
    total_time = time.time() - start_time

    # Save results
    results['experiment'] = exp['name']
    results['train_time_seconds'] = round(train_time, 1)
    results['total_time_seconds'] = round(total_time, 1)
    results['config'] = DQN_CONFIG.copy()
    results['config']['reward_config'] = REWARD_CONFIG

    results_path = os.path.join(RESULTS_DIR, f'env_exp{exp_id}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total time: {total_time/60:.1f} minutes")

    return results


def main():
    parser = argparse.ArgumentParser(description='IDS Environment Experiments')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3],
                        help='Run specific experiment (1, 2, or 3)')
    args = parser.parse_args()

    print("=" * 60)
    print("  RL-Enhanced IDS — Environment Experiments")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}

    if args.exp:
        results = run_experiment(args.exp)
        all_results[f'exp_{args.exp}'] = results
    else:
        for exp_id in [1, 2, 3]:
            results = run_experiment(exp_id)
            all_results[f'exp_{exp_id}'] = results

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, 'env_experiments_combined.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nCombined results saved: {combined_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Experiment':<30} {'F1':>8} {'Recall':>8} {'Prec':>8} {'Time':>10}")
    print("  " + "-" * 70)
    for exp_key, r in all_results.items():
        print(f"  {r['experiment']:<30} {r['F1']:>7.1f}% {r['Recall']:>7.1f}% "
              f"{r['Precision']:>7.1f}% {r['total_time_seconds']/60:>8.1f} min")
    print()


if __name__ == '__main__':
    main()
