#!/usr/bin/env python3
"""
Master Experiment Runner for RL-Enhanced IDS
Runs all algorithms and scenarios for comparison.

Usage:
    python run_experiments.py --all           # Run everything
    python run_experiments.py --ml            # Only ML baselines
    python run_experiments.py --rl            # Only RL agents
    python run_experiments.py --standard      # Standard scenario only
"""

import os
import sys
import argparse
import json
from datetime import datetime
import subprocess

# Add project root to path
sys.path.insert(0, '/home/abishik/HONOURS_PROJECT')

RESULTS_DIR = '/home/abishik/HONOURS_PROJECT/results'
MODELS_DIR = '/home/abishik/HONOURS_PROJECT/models'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"[RUNNING] {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def run_ml_baselines():
    """Train and evaluate ML baselines (RF, XGBoost)."""
    print("\n" + "="*60)
    print("[ML] RUNNING ML BASELINES")
    print("="*60)
    
    from ml_baselines import run_all_baselines
    baselines = run_all_baselines()
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, 'ml_baselines_results.json')
    results = {}
    for name, metrics in baselines.results.items():
        results[name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    return results


def run_dqn_training(exclude_labels=None, episodes=2000, name='dqn_standard'):
    """Train DQN agent."""
    print("\n" + "="*60)
    print(f"[RL] TRAINING DQN AGENT: {name}")
    print("="*60)
    
    cmd = f"python train_rl_agent.py --episodes {episodes} --output {name}.pth"
    if exclude_labels:
        cmd += f" --exclude {' '.join(exclude_labels)}"
    
    return run_command(cmd, f"Training DQN ({name})")


def run_ppo_training(exclude_labels=None, timesteps=100000, name='ppo_standard'):
    """Train PPO agent."""
    print("\n" + "="*60)
    print(f"[RL] TRAINING PPO AGENT: {name}")
    print("="*60)
    
    cmd = f"python train_ppo_agent.py --timesteps {timesteps} --output {name}"
    if exclude_labels:
        cmd += f" --exclude {' '.join(exclude_labels)}"
    
    return run_command(cmd, f"Training PPO ({name})")


def run_standard_scenario():
    """Run standard training with all attack types."""
    print("\n" + "#"*60)
    print("# SCENARIO 1: STANDARD (All attack types)")
    print("#"*60)
    
    run_ml_baselines()
    run_dqn_training(name='dqn_standard', episodes=2000)
    # PPO (commented out - focusing on DQN, RF, XGBoost for now)
    # run_ppo_training(name='ppo_standard', timesteps=100000)


def run_zeroday_scenario(held_out_label='ddos'):
    """Run zero-day simulation: train without attack type, test on it."""
    print("\n" + "#"*60)
    print(f"# SCENARIO 2: ZERO-DAY (Held out: {held_out_label})")
    print("#"*60)
    
    run_dqn_training(
        exclude_labels=[held_out_label],
        name=f'dqn_zeroday_{held_out_label}',
        episodes=1500
    )
    # PPO (commented out - focusing on DQN, RF, XGBoost for now)
    # run_ppo_training(
    #     exclude_labels=[held_out_label],
    #     name=f'ppo_zeroday_{held_out_label}',
    #     timesteps=80000
    # )


def generate_comparison_report():
    """Generate final comparison report."""
    print("\n" + "="*60)
    print("[REPORT] GENERATING COMPARISON REPORT")
    print("="*60)
    
    # Load all results
    ml_results_path = os.path.join(RESULTS_DIR, 'ml_baselines_results.json')
    
    if os.path.exists(ml_results_path):
        with open(ml_results_path, 'r') as f:
            ml_results = json.load(f)
    else:
        ml_results = {}
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'ml_baselines': ml_results,
        'rl_agents': {
            'dqn': '(Run evaluation after training)',
            'ppo': '(Run evaluation after training)'
        }
    }
    
    report_path = os.path.join(RESULTS_DIR, 'experiment_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for model, metrics in ml_results.items():
        print(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run IDS experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--ml', action='store_true', help='Run ML baselines only')
    parser.add_argument('--dqn', action='store_true', help='Train DQN only')
    parser.add_argument('--ppo', action='store_true', help='Train PPO only')
    parser.add_argument('--standard', action='store_true', help='Standard scenario only')
    parser.add_argument('--zeroday', type=str, help='Zero-day scenario with held-out label')
    parser.add_argument('--report', action='store_true', help='Generate comparison report')
    args = parser.parse_args()
    
    if args.all:
        run_standard_scenario()
        run_zeroday_scenario('ddos')
        generate_comparison_report()
    elif args.ml:
        run_ml_baselines()
    elif args.dqn:
        run_dqn_training()
    elif args.ppo:
        run_ppo_training()
    elif args.standard:
        run_standard_scenario()
    elif args.zeroday:
        run_zeroday_scenario(args.zeroday)
    elif args.report:
        generate_comparison_report()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
