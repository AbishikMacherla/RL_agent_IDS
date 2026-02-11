# PPO Agent Training Script
# Uses Stable-Baselines3 for Proximal Policy Optimization

import os
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ids_env import IdsEnv

# ============== HYPERPARAMETERS (Documented for Methodology) ==============
LEARNING_RATE = 3e-4    # PPO default
N_STEPS = 2048          # Steps per update
BATCH_SIZE = 64         # Minibatch size
N_EPOCHS = 10           # Optimization epochs per update
GAMMA = 0.99            # Discount factor
GAE_LAMBDA = 0.95       # GAE lambda for advantage estimation
CLIP_RANGE = 0.2        # PPO clipping parameter
ENT_COEF = 0.01         # Entropy coefficient for exploration
# ==========================================================================


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress."""
    
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                print(f"Step {self.n_calls}: Mean reward = {mean_reward:.2f}")
        return True


def train_ppo(exclude_labels=None, total_timesteps=100000, model_name='ppo_agent'):
    """
    Train PPO agent for Intrusion Detection.
    
    Args:
        exclude_labels: List of attack labels to exclude (for zero-day simulation).
        total_timesteps: Total training timesteps.
        model_name: Output model filename (without extension).
    
    Returns:
        Trained PPO model.
    """
    print(f"\n{'='*60}")
    print(f"Training PPO Agent (Stable-Baselines3)")
    print(f"Excluded labels: {exclude_labels or 'None'}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Hyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  N Steps: {N_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  GAE Lambda: {GAE_LAMBDA}")
    print(f"  Clip Range: {CLIP_RANGE}")
    print(f"{'='*60}\n")
    
    # Create environment
    def make_env():
        return IdsEnv(exclude_labels=exclude_labels)
    
    env = DummyVecEnv([make_env])
    
    # Initialize PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log="./logs/ppo_tensorboard/"
    )
    
    # Training callback
    callback = TrainingCallback(log_freq=10000)
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    model_dir = '/home/abishik/HONOURS_PROJECT/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f"\nTraining complete. Model saved to {model_path}.zip")
    
    return model


def evaluate_ppo(model_path, exclude_labels=None, n_episodes=10):
    """
    Evaluate trained PPO model.
    
    Args:
        model_path: Path to saved PPO model.
        exclude_labels: Labels to exclude (should match training).
        n_episodes: Number of evaluation episodes.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    env = IdsEnv(exclude_labels=exclude_labels)
    model = PPO.load(model_path)
    
    total_rewards = []
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Track metrics
            true_label = env.y[env.current_step - 1]
            if action == 1 and true_label == 1:
                true_positives += 1
            elif action == 0 and true_label == 0:
                true_negatives += 1
            elif action == 1 and true_label == 0:
                false_positives += 1
            else:
                false_negatives += 1
        
        total_rewards.append(episode_reward)
    
    # Calculate metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_reward': np.mean(total_rewards),
        'confusion_matrix': {
            'TP': true_positives,
            'TN': true_negatives,
            'FP': false_positives,
            'FN': false_negatives
        }
    }
    
    print(f"\n{'='*40}")
    print(f"PPO Evaluation Results ({n_episodes} episodes)")
    print(f"{'='*40}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Mean Reward: {np.mean(total_rewards):.2f}")
    print(f"{'='*40}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO Agent for IDS')
    parser.add_argument('--exclude', nargs='+', default=None,
                        help='Labels to exclude (e.g., --exclude ddos "dos hulk")')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Total training timesteps')
    parser.add_argument('--output', type=str, default='ppo_agent', 
                        help='Model output name (without extension)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after training')
    args = parser.parse_args()
    
    model = train_ppo(
        exclude_labels=args.exclude,
        total_timesteps=args.timesteps,
        model_name=args.output
    )
    
    if args.evaluate:
        model_path = f'/home/abishik/HONOURS_PROJECT/models/{args.output}'
        evaluate_ppo(model_path, exclude_labels=args.exclude)
