# DQN Agent Training Script (v2 - Multi-Scenario Support)
# Supports zero-day simulation via exclude_labels parameter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import argparse

from ids_env import IdsEnv

# ============== HYPERPARAMETERS (Documented for Methodology) ==============
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Soft update coefficient
LR = 5e-4               # Learning rate
UPDATE_EVERY = 4        # Network update frequency
# ==========================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    """
    Deep Q-Network Architecture:
    - Input: Network traffic features (78 dimensions)
    - Hidden1: 64 neurons with ReLU activation
    - Hidden2: 64 neurons with ReLU activation  
    - Output: 2 Q-values (Allow=0, Block=1)
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-size buffer for experience replay."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class Agent:
    """DQN Agent with experience replay and target network."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, learning_rate=LR):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        
        self.qnetwork_local = DQN(state_size, action_size, seed, fc1_units, fc2_units).to(device)
        self.qnetwork_target = DQN(state_size, action_size, seed, fc1_units, fc2_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def train_dqn(exclude_labels=None, n_episodes=2000, max_t=1000, 
              eps_start=1.0, eps_end=0.01, eps_decay=0.999,
              model_name='dqn_agent.pth', reward_config=None,
              fc1_units=64, fc2_units=64, learning_rate=LR):
    """
    Train DQN agent.
    
    Args:
        exclude_labels: List of attack labels to exclude (for zero-day simulation).
        n_episodes: Number of training episodes.
        max_t: Maximum steps per episode.
        model_name: Output model filename.
        reward_config: Dict with 'tp', 'tn', 'fn', 'fp' reward values.
        fc1_units: First hidden layer size.
        fc2_units: Second hidden layer size.
        learning_rate: Adam optimizer learning rate.
    """
    # Initialize environment with optional label exclusion and reward config
    env = IdsEnv(exclude_labels=exclude_labels, reward_config=reward_config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = Agent(state_size=state_size, action_size=action_size, seed=42,
                  fc1_units=fc1_units, fc2_units=fc2_units, learning_rate=learning_rate)
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    print(f"\n{'='*60}")
    print(f"Training DQN Agent")
    print(f"Excluded labels: {exclude_labels or 'None'}")
    print(f"Episodes: {n_episodes}, Max steps: {max_t}")
    print(f"Network: {fc1_units}-{fc2_units}, LR: {learning_rate}")
    print(f"Rewards: {env.rewards}")
    print(f"{'='*60}\n")
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
    
    # Save model
    model_dir = '/home/abishik/HONOURS_PROJECT/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    torch.save(agent.qnetwork_local.state_dict(), model_path)
    print(f"\nTraining complete. Model saved to {model_path}")
    
    return scores, agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN Agent for IDS')
    parser.add_argument('--exclude', nargs='+', default=None, 
                        help='Labels to exclude (e.g., --exclude ddos "dos hulk")')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--output', type=str, default='dqn_agent.pth', help='Model output name')
    args = parser.parse_args()
    
    train_dqn(
        exclude_labels=args.exclude,
        n_episodes=args.episodes,
        model_name=args.output
    )