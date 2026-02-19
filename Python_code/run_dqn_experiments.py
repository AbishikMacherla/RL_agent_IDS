"""
DQN Hyperparameter Experiment Runner
Trains multiple DQN agents with different configurations and evaluates them.
Generates comparison table and per-experiment notes.

Usage:
    python run_dqn_experiments.py              # Run all experiments (1-5 first, then 6-8 with best)
    python run_dqn_experiments.py --phase1     # Run only reward experiments (1-5)
    python run_dqn_experiments.py --phase2     # Run only architecture experiments (6-8) using best from phase 1
    python run_dqn_experiments.py --eval-only  # Evaluate existing models without retraining
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

# Add Python_code to path
sys.path.insert(0, '/home/abishik/HONOURS_PROJECT/Python_code')
from train_rl_agent import train_dqn, DQN

# Paths
DATA_DIR = '/home/abishik/HONOURS_PROJECT/processed_data'
MODEL_DIR = '/home/abishik/HONOURS_PROJECT/models'
RESULTS_DIR = '/home/abishik/HONOURS_PROJECT/results'
NOTES_DIR = '/home/abishik/HONOURS_PROJECT/Notes'
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

PHASE1_EXPERIMENTS = {
    'exp1_baseline': {
        'name': 'Exp 1: Baseline (Current 10:1)',
        'reward_config': {'tp': 10.0, 'tn': 1.0, 'fn': -10.0, 'fp': -1.0},
        'fc1_units': 64,
        'fc2_units': 64,
        'learning_rate': 5e-4,
        'episodes': 2000,
        'description': 'Current security-first setup with 10:1 asymmetric rewards.',
        'hypothesis': 'High recall but many false positives due to extreme penalty asymmetry.',
    },
    'exp2_lower_fn': {
        'name': 'Exp 2: Lower FN Penalty (-5)',
        'reward_config': {'tp': 10.0, 'tn': 1.0, 'fn': -5.0, 'fp': -1.0},
        'fc1_units': 64,
        'fc2_units': 64,
        'learning_rate': 5e-4,
        'episodes': 2000,
        'description': 'Reduced FN penalty from -10 to -5. Agent should be less paranoid.',
        'hypothesis': 'Fewer false positives, slightly lower recall. Agent blocks when ~16.7% sure instead of ~9.1%.',
    },
    'exp3_higher_fp': {
        'name': 'Exp 3: Higher FP Penalty (-3)',
        'reward_config': {'tp': 10.0, 'tn': 1.0, 'fn': -10.0, 'fp': -3.0},
        'fc1_units': 64,
        'fc2_units': 64,
        'learning_rate': 5e-4,
        'episodes': 2000,
        'description': 'Raised FP penalty from -1 to -3. Agent penalised more for false alarms.',
        'hypothesis': 'Agent cares more about not blocking benign traffic. Should reduce FP significantly while keeping strong recall.',
    },
    'exp4_5to1_ratio': {
        'name': 'Exp 4: 5:1 Ratio',
        'reward_config': {'tp': 5.0, 'tn': 1.0, 'fn': -5.0, 'fp': -1.0},
        'fc1_units': 64,
        'fc2_units': 64,
        'learning_rate': 5e-4,
        'episodes': 2000,
        'description': 'Balanced 5:1 ratio. Still security-first but less extreme.',
        'hypothesis': 'Middle ground between security-first and balanced. Agent blocks when ~16.7% sure.',
    },
    'exp5_symmetric': {
        'name': 'Exp 5: Symmetric (No Bias)',
        'reward_config': {'tp': 1.0, 'tn': 1.0, 'fn': -1.0, 'fp': -1.0},
        'fc1_units': 64,
        'fc2_units': 64,
        'learning_rate': 5e-4,
        'episodes': 2000,
        'description': 'Fully symmetric rewards. No security bias — all decisions weighted equally.',
        'hypothesis': 'Should behave like supervised ML — balanced precision/recall, ~50% threshold. Highest accuracy but lower recall.',
    },
}

def get_phase2_experiments(best_reward_config):
    """Generate Phase 2 experiments using the best reward config from Phase 1."""
    return {
        'exp6_bigger_brain': {
            'name': 'Exp 6: Bigger Brain (128-128)',
            'reward_config': best_reward_config,
            'fc1_units': 128,
            'fc2_units': 128,
            'learning_rate': 5e-4,
            'episodes': 2000,
            'description': f'Doubled network size (128-128 neurons) with best reward config.',
            'hypothesis': 'Larger network can learn more complex patterns. May distinguish subtle differences between benign and malicious better, reducing both FP and FN.',
        },
        'exp7_more_episodes': {
            'name': 'Exp 7: More Episodes (5000)',
            'reward_config': best_reward_config,
            'fc1_units': 64,
            'fc2_units': 64,
            'learning_rate': 5e-4,
            'episodes': 5000,
            'description': f'2.5x more training episodes with best reward config.',
            'hypothesis': 'Agent sees more examples and refines its decision boundary. Should improve all metrics but with diminishing returns.',
        },
        'exp8_slower_lr': {
            'name': 'Exp 8: Slower Learning Rate (1e-4)',
            'reward_config': best_reward_config,
            'fc1_units': 64,
            'fc2_units': 64,
            'learning_rate': 1e-4,
            'episodes': 2000,
            'description': f'5x slower learning rate with best reward config.',
            'hypothesis': 'Slower, more careful learning. Agent makes smaller updates per step — more stable training but may need more episodes to converge.',
        },
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_experiment(exp_id, config):
    """Evaluate a trained DQN model on the standard test set."""
    model_path = os.path.join(MODEL_DIR, f'{exp_id}.pth')
    
    if not os.path.exists(model_path):
        print(f"  [SKIP] {exp_id}: model not found")
        return None
    
    # Load test data
    X_test = pd.read_parquet(os.path.join(DATA_DIR, 'X_test_2017.parquet')).to_numpy()
    y_test = np.load(os.path.join(DATA_DIR, 'y_test_2017.npy'))
    state_size = X_test.shape[1]
    
    # Load model with matching architecture
    fc1 = config['fc1_units']
    fc2 = config['fc2_units']
    qnetwork = DQN(state_size, action_size=2, seed=42, fc1_units=fc1, fc2_units=fc2).to(device)
    qnetwork.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    qnetwork.eval()
    
    # Run inference
    predictions = []
    X_tensor = torch.from_numpy(X_test).float().to(device)
    
    batch_size = 2048
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            output = qnetwork(batch)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    y_pred = np.array(predictions)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    return {
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


# ============================================================================
# NOTES GENERATION
# ============================================================================

def generate_experiment_note(exp_id, config, results):
    """Generate a markdown note for a single experiment with results explanation."""
    rewards = config['reward_config']
    
    # Calculate decision threshold
    tp_r = rewards['tp']
    fn_r = abs(rewards['fn'])
    fp_r = abs(rewards['fp'])
    tn_r = rewards['tn']
    
    # Threshold: agent blocks when p(malicious) > fp_r / (tp_r + fp_r) approximately
    # More precisely: Block when p > (fp_r + tn_r) / (tp_r + fn_r + fp_r + tn_r)
    threshold = (fp_r + tn_r) / (tp_r + fn_r + fp_r + tn_r)
    
    note = f"""# {config['name']}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Experiment ID**: {exp_id}

## Parameters
| Parameter | Value |
|-----------|-------|
| TP Reward | {rewards['tp']:+.1f} |
| TN Reward | {rewards['tn']:+.1f} |
| FN Penalty | {rewards['fn']:+.1f} |
| FP Penalty | {rewards['fp']:+.1f} |
| Network | {config['fc1_units']}-{config['fc2_units']} neurons |
| Learning Rate | {config['learning_rate']} |
| Episodes | {config['episodes']} |
| Reward Ratio | {int(tp_r)}:{int(fp_r)} (TP:FP penalty) |
| Estimated Block Threshold | ~{threshold:.1%} malicious probability |

## Description
{config['description']}

## Hypothesis (Before Training)
{config['hypothesis']}

"""
    
    if results:
        note += f"""## Results
| Metric | Value |
|--------|-------|
| **Accuracy** | {results['Accuracy']}% |
| **Precision** | {results['Precision']}% |
| **Recall** | {results['Recall']}% |
| **F1 Score** | {results['F1']}% |

### Confusion Matrix
|  | Predicted Allow | Predicted Block |
|--|-----------------|-----------------|
| **Actual Benign** | TN = {results['TN']:,} | FP = {results['FP']:,} |
| **Actual Attack** | FN = {results['FN']:,} | TP = {results['TP']:,} |

## Analysis — Why These Results?

"""
        # Generate analysis based on results
        total_benign = results['TN'] + results['FP']
        total_attack = results['FN'] + results['TP']
        fp_rate = results['FP'] / total_benign * 100 if total_benign > 0 else 0
        fn_rate = results['FN'] / total_attack * 100 if total_attack > 0 else 0
        
        note += f"""### False Positive Analysis
- Out of {total_benign:,} benign samples, {results['FP']:,} were incorrectly blocked ({fp_rate:.2f}%)
"""
        if fp_rate > 5:
            note += f"- **High FP rate**: The FP penalty ({rewards['fp']:+.1f}) is low relative to FN penalty ({rewards['fn']:+.1f}), so the agent favours blocking uncertain traffic.\n"
        elif fp_rate < 1:
            note += f"- **Low FP rate**: The reward structure balances false alarm costs well. Agent only blocks when confident.\n"
        else:
            note += f"- **Moderate FP rate**: Reasonable balance between caution and precision.\n"
        
        note += f"""
### False Negative Analysis
- Out of {total_attack:,} attack samples, {results['FN']:,} were missed ({fn_rate:.2f}%)
"""
        if fn_rate < 1:
            note += f"- **Very low FN rate**: Agent catches virtually all attacks. The high FN penalty ({rewards['fn']:+.1f}) drives strong recall.\n"
        elif fn_rate < 5:
            note += f"- **Low FN rate**: Agent catches most attacks but allows some edge cases through.\n"
        else:
            note += f"- **Higher FN rate**: With reduced FN penalty, the agent is less aggressive about blocking uncertain traffic.\n"
        
        note += f"""
### Reward Structure Impact
- The {int(tp_r)}:{int(fp_r)} reward ratio means the agent blocks when it estimates >{threshold:.1%} chance of malicious traffic.
- Precision ({results['Precision']}%) indicates how many of the agent's "Block" decisions were correct.
- Recall ({results['Recall']}%) indicates what proportion of actual attacks were caught.
"""
        
        # Compare to baseline if not baseline itself
        if exp_id != 'exp1_baseline':
            note += f"\n### Comparison to Baseline (Exp 1)\n"
            note += f"- See the comparison table in `results/dqn_experiments.json` for side-by-side metrics.\n"
    
    else:
        note += "## Results\n*Model not yet trained or evaluation failed.*\n"
    
    return note


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_experiments(experiments, phase_name):
    """Train and evaluate a set of experiments."""
    print(f"\n{'#' * 70}")
    print(f"# {phase_name}")
    print(f"{'#' * 70}")
    
    all_results = {}
    
    for exp_id, config in experiments.items():
        print(f"\n{'=' * 70}")
        print(f"  {config['name']}")
        print(f"{'=' * 70}")
        
        # Train
        model_name = f"{exp_id}.pth"
        print(f"\n  Training {exp_id}...")
        
        train_dqn(
            n_episodes=config['episodes'],
            model_name=model_name,
            reward_config=config['reward_config'],
            fc1_units=config['fc1_units'],
            fc2_units=config['fc2_units'],
            learning_rate=config['learning_rate'],
        )
        
        # Evaluate
        print(f"\n  Evaluating {exp_id}...")
        results = evaluate_experiment(exp_id, config)
        
        if results:
            all_results[exp_id] = {
                'config': config,
                'results': results,
            }
            print(f"  ✓ Acc={results['Accuracy']}%, Prec={results['Precision']}%, Rec={results['Recall']}%, F1={results['F1']}%")
            
            # Generate note for this experiment
            note_content = generate_experiment_note(exp_id, config, results)
            note_path = os.path.join(NOTES_DIR, f'{exp_id}_results.md')
            with open(note_path, 'w') as f:
                f.write(note_content)
            print(f"  ✓ Note saved to {note_path}")
    
    return all_results


def evaluate_only(experiments):
    """Evaluate existing models without retraining."""
    print(f"\n{'#' * 70}")
    print(f"# EVALUATING EXISTING MODELS")
    print(f"{'#' * 70}")
    
    all_results = {}
    
    for exp_id, config in experiments.items():
        model_path = os.path.join(MODEL_DIR, f'{exp_id}.pth')
        if not os.path.exists(model_path):
            print(f"  [SKIP] {exp_id}: model not found at {model_path}")
            continue
        
        print(f"\n  Evaluating {exp_id}...")
        results = evaluate_experiment(exp_id, config)
        
        if results:
            all_results[exp_id] = {
                'config': config,
                'results': results,
            }
            print(f"  ✓ Acc={results['Accuracy']}%, Prec={results['Precision']}%, Rec={results['Recall']}%, F1={results['F1']}%")
            
            # Generate note
            note_content = generate_experiment_note(exp_id, config, results)
            note_path = os.path.join(NOTES_DIR, f'{exp_id}_results.md')
            with open(note_path, 'w') as f:
                f.write(note_content)
            print(f"  ✓ Note saved to {note_path}")
    
    return all_results


def find_best_experiment(results):
    """Find the best experiment based on F1 score."""
    best_id = None
    best_f1 = 0
    
    for exp_id, data in results.items():
        f1 = data['results']['F1']
        if f1 > best_f1:
            best_f1 = f1
            best_id = exp_id
    
    return best_id


def print_comparison_table(all_results):
    """Print formatted comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  EXPERIMENT COMPARISON TABLE")
    print(f"{'=' * 90}")
    print(f"  {'Experiment':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>10} {'FN':>10}")
    print(f"  {'─' * 85}")
    
    for exp_id, data in all_results.items():
        r = data['results']
        name = data['config']['name'][:28]
        print(f"  {name:<30} {r['Accuracy']:>9.2f}% {r['Precision']:>9.2f}% {r['Recall']:>9.2f}% {r['F1']:>9.2f}% {r['FP']:>9,} {r['FN']:>9,}")
    
    print(f"{'=' * 90}")
    
    # Find best
    best_id = find_best_experiment(all_results)
    if best_id:
        print(f"\n  ★ Best F1 Score: {all_results[best_id]['config']['name']} ({all_results[best_id]['results']['F1']}%)")


def save_all_results(all_results):
    """Save all experiment results to JSON."""
    # Make results JSON-serializable
    serializable = {}
    for exp_id, data in all_results.items():
        serializable[exp_id] = {
            'name': data['config']['name'],
            'rewards': data['config']['reward_config'],
            'network': f"{data['config']['fc1_units']}-{data['config']['fc2_units']}",
            'learning_rate': data['config']['learning_rate'],
            'episodes': data['config']['episodes'],
            'results': data['results'],
        }
    
    results_path = os.path.join(RESULTS_DIR, 'dqn_experiments.json')
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DQN Hyperparameter Experiments')
    parser.add_argument('--phase1', action='store_true', help='Run only Phase 1 (reward experiments)')
    parser.add_argument('--phase2', action='store_true', help='Run only Phase 2 (architecture experiments)')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate existing models only')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DQN HYPERPARAMETER EXPERIMENTS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = {}
    
    if args.eval_only:
        # Evaluate whatever models exist
        all_experiments = {**PHASE1_EXPERIMENTS}
        # Try to load phase 2 experiments — need to guess best reward config
        results_path = os.path.join(RESULTS_DIR, 'dqn_experiments.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                prev_results = json.load(f)
            # Find best phase 1 reward config
            best_f1 = 0
            best_rewards = PHASE1_EXPERIMENTS['exp1_baseline']['reward_config']
            for exp_id, data in prev_results.items():
                if exp_id.startswith('exp') and int(exp_id[3]) <= 5:
                    if data['results']['F1'] > best_f1:
                        best_f1 = data['results']['F1']
                        best_rewards = data['rewards']
            all_experiments.update(get_phase2_experiments(best_rewards))
        
        all_results = evaluate_only(all_experiments)
    
    elif args.phase1:
        all_results = run_experiments(PHASE1_EXPERIMENTS, "PHASE 1: REWARD STRUCTURE EXPERIMENTS")
    
    elif args.phase2:
        # Load Phase 1 results to find best reward config
        results_path = os.path.join(RESULTS_DIR, 'dqn_experiments.json')
        if not os.path.exists(results_path):
            print("ERROR: Run Phase 1 first to determine best reward config!")
            return
        
        with open(results_path, 'r') as f:
            phase1_results = json.load(f)
        
        best_f1 = 0
        best_rewards = None
        for exp_id, data in phase1_results.items():
            if data['results']['F1'] > best_f1:
                best_f1 = data['results']['F1']
                best_rewards = data['rewards']
        
        print(f"\nBest Phase 1 reward config (F1={best_f1}%): {best_rewards}")
        
        phase2_exps = get_phase2_experiments(best_rewards)
        all_results = run_experiments(phase2_exps, "PHASE 2: ARCHITECTURE EXPERIMENTS")
        
        # Merge with phase 1 results
        for exp_id, data in phase1_results.items():
            all_results[exp_id] = {
                'config': {
                    'name': data['name'],
                    'reward_config': data['rewards'],
                    'fc1_units': int(data['network'].split('-')[0]),
                    'fc2_units': int(data['network'].split('-')[1]),
                    'learning_rate': data['learning_rate'],
                    'episodes': data['episodes'],
                },
                'results': data['results'],
            }
    
    else:
        # Run everything: Phase 1 then Phase 2
        phase1_results = run_experiments(PHASE1_EXPERIMENTS, "PHASE 1: REWARD STRUCTURE EXPERIMENTS")
        all_results.update(phase1_results)
        
        # Find best Phase 1 config
        best_id = find_best_experiment(phase1_results)
        if best_id:
            best_rewards = phase1_results[best_id]['config']['reward_config']
            print(f"\n★ Best Phase 1: {phase1_results[best_id]['config']['name']} (F1={phase1_results[best_id]['results']['F1']}%)")
            print(f"  Using reward config: {best_rewards}")
            
            # Run Phase 2 with best rewards
            phase2_exps = get_phase2_experiments(best_rewards)
            phase2_results = run_experiments(phase2_exps, "PHASE 2: ARCHITECTURE EXPERIMENTS")
            all_results.update(phase2_results)
    
    # Print comparison and save
    if all_results:
        print_comparison_table(all_results)
        save_all_results(all_results)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
