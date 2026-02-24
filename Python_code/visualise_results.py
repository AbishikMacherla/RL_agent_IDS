#!/usr/bin/env python3
"""
RL-Enhanced IDS — Figure Generator
Generates publication-quality matplotlib/seaborn figures for dissertation.

Usage:
    python visualise_results.py

Output:
    results/figures/*.png    — All figures for dissertation
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ============================================================================
# CONFIG
# ============================================================================
PROJECT_DIR = '/home/abishik/HONOURS_PROJECT'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'all_scenarios_results.json')
DQN_EXPERIMENTS_FILE = os.path.join(RESULTS_DIR, 'dqn_experiments.json')

os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
sns.set_theme(style='whitegrid', font_scale=1.15, rc={
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'axes.titleweight': 'bold',
})

MODEL_COLORS = {
    'Random Forest': '#27ae60',
    'XGBoost': '#2980b9',
    'DQN': '#e67e22',
    'DQN (Standard)': '#e67e22',
    'DQN (No DDoS)': '#f39c12',
    'DQN (No Web)': '#f39c12',
    'PPO': '#8e44ad',
}

SCENARIO_TITLES = {
    'scenario_1': 'Scenario 1: Standard Classification (CIC-IDS2017)',
    'scenario_2': 'Scenario 2: Zero-Day DDoS Detection',
    'scenario_3': 'Scenario 3: Zero-Day Web Attack Detection',
    'scenario_4': 'Scenario 4: Cross-Dataset Generalisation (2017→2023)',
}


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    scenarios, dqn_experiments = {}, {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            scenarios = json.load(f)
        print(f"  ✓ Loaded scenarios: {list(scenarios.keys())}")
    if os.path.exists(DQN_EXPERIMENTS_FILE):
        with open(DQN_EXPERIMENTS_FILE) as f:
            dqn_experiments = json.load(f)
        print(f"  ✓ Loaded {len(dqn_experiments)} DQN experiments")
    return scenarios, dqn_experiments


# ============================================================================
# FIGURE 1: Scenario Comparison Bar Charts
# ============================================================================
def fig_scenario_comparison(scenarios):
    """One grouped bar chart per scenario."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    metric_colors = ['#1abc9c', '#3498db', '#e74c3c', '#9b59b6']

    for s_key, s_data in scenarios.items():
        if s_key not in SCENARIO_TITLES:
            continue

        models = list(s_data.keys())
        n_models = len(models)
        x = np.arange(n_models)
        width = 0.2

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2.2), 5))
        for i, metric in enumerate(metrics):
            vals = [s_data[m].get(metric, 0) for m in models]
            bars = ax.bar(x + i * width, vals, width, label=metric,
                          color=metric_colors[i], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_ylim(0, max(max(s_data[m].get(met, 0) for m in models for met in metrics), 100) + 8)
        ax.set_title(SCENARIO_TITLES[s_key], fontsize=14, pad=15)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        path = os.path.join(FIGURES_DIR, f'fig_{s_key}_comparison.png')
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {path}")


# ============================================================================
# FIGURE 2: Cross-Scenario Summary
# ============================================================================
def fig_cross_scenario_summary(scenarios):
    """Best model vs best DQN F1 across all scenarios."""
    names, best_f1, best_models, dqn_f1 = [], [], [], []

    for s_key in ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4']:
        if s_key not in scenarios:
            continue
        data = scenarios[s_key]
        names.append(SCENARIO_TITLES.get(s_key, s_key).split(':')[0])

        best_m = max(data, key=lambda m: data[m].get('F1', 0))
        best_f1.append(data[best_m].get('F1', 0))
        best_models.append(best_m)

        dqn_keys = [k for k in data if 'DQN' in k]
        best_dqn = max(dqn_keys, key=lambda k: data[k].get('F1', 0)) if dqn_keys else None
        dqn_f1.append(data[best_dqn].get('F1', 0) if best_dqn else 0)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, best_f1, width, label='Best Model (ML or RL)',
                   color='#27ae60', edgecolor='white')
    bars2 = ax.bar(x + width / 2, dqn_f1, width, label='Best DQN Variant',
                   color='#e67e22', edgecolor='white')

    for bar, val, model in zip(bars1, best_f1, best_models):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%\n({model})', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, dqn_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_ylim(0, max(best_f1) + 18)
    ax.set_title('Cross-Scenario F1 Performance Summary', fontsize=14, pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'fig_cross_scenario_summary.png')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================================
# FIGURE 3: DQN Experiment F1 Progression
# ============================================================================
def fig_dqn_f1_progression(experiments):
    if not experiments:
        return
    names, f1s = [], []
    for exp_id, exp_data in experiments.items():
        names.append(exp_data.get('name', exp_id).replace('Exp ', 'E'))
        f1s.append(exp_data['results'].get('F1', 0))

    # Phase 1 (reward tuning) = red, Phase 2 (architecture) = blue
    colors = ['#e74c3c'] * 5 + ['#2980b9'] * 3
    colors = colors[:len(names)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, f1s, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Phase labels
    ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(2, max(f1s) + 5, 'Phase 1: Reward Tuning', ha='center',
            fontsize=11, color='#e74c3c', fontweight='bold')
    ax.text(6, max(f1s) + 5, 'Phase 2: Architecture', ha='center',
            fontsize=11, color='#2980b9', fontweight='bold')

    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_ylim(0, max(f1s) + 10)
    ax.set_title('DQN Hyperparameter Experiments — F1 Score Comparison', fontsize=14, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'fig_dqn_f1_progression.png')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================================
# FIGURE 4: DQN Precision vs Recall Scatter
# ============================================================================
def fig_dqn_precision_recall(experiments):
    if not experiments:
        return
    names, prec, rec, f1s = [], [], [], []
    for exp_id, exp_data in experiments.items():
        r = exp_data['results']
        names.append(exp_data.get('name', exp_id).split(': ')[-1] if ': ' in exp_data.get('name', '') else exp_id)
        prec.append(r.get('Precision', 0))
        rec.append(r.get('Recall', 0))
        f1s.append(r.get('F1', 0))

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(rec, prec, c=f1s, cmap='RdYlGn', s=150, edgecolors='black', linewidth=0.8, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (rec[i], prec[i]), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=8, fontweight='bold')

    cbar = fig.colorbar(sc, ax=ax, label='F1 Score (%)')
    ax.set_xlabel('Recall (%)', fontsize=12)
    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title('DQN Experiments — Precision vs Recall Trade-off', fontsize=14, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'fig_dqn_precision_recall.png')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================================
# FIGURE 5: DQN FP vs FN Error Analysis
# ============================================================================
def fig_dqn_fp_fn(experiments):
    if not experiments:
        return
    names, fps, fns = [], [], []
    for exp_id, exp_data in experiments.items():
        r = exp_data['results']
        names.append(exp_data.get('name', exp_id).replace('Exp ', 'E'))
        fps.append(r.get('FP', 0))
        fns.append(r.get('FN', 0))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, fps, width, label='False Positives', color='#e74c3c', edgecolor='white')
    ax.bar(x + width / 2, fns, width, label='False Negatives', color='#3498db', edgecolor='white')

    for i, (fp, fn) in enumerate(zip(fps, fns)):
        ax.text(x[i] - width / 2, fp + max(fps) * 0.02, f'{fp:,}', ha='center', va='bottom', fontsize=7)
        ax.text(x[i] + width / 2, fn + max(fns) * 0.02, f'{fn:,}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('DQN Experiments — False Positives vs False Negatives', fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'fig_dqn_fp_fn_analysis.png')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================================
# FIGURE 6: Reward Impact on Recall
# ============================================================================
def fig_reward_impact(experiments):
    """Show how reward structure affects recall, precision, and F1."""
    if not experiments:
        return
    # Phase 1 experiments only (reward tuning)
    phase1 = {k: v for i, (k, v) in enumerate(experiments.items()) if i < 5}
    if not phase1:
        return

    names, recalls, precs, f1s, reward_labels = [], [], [], [], []
    for exp_id, exp_data in phase1.items():
        r = exp_data['results']
        rewards = exp_data.get('rewards', {})
        names.append(exp_data.get('name', exp_id).split(': ')[-1] if ': ' in exp_data.get('name', '') else exp_id)
        recalls.append(r.get('Recall', 0))
        precs.append(r.get('Precision', 0))
        f1s.append(r.get('F1', 0))
        reward_labels.append(f"+{rewards.get('tp',0)}/+{rewards.get('tn',0)}/{rewards.get('fn',0)}/{rewards.get('fp',0)}")

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, recalls, width, label='Recall', color='#e74c3c')
    ax.bar(x, precs, width, label='Precision', color='#3498db')
    ax.bar(x + width, f1s, width, label='F1', color='#2ecc71')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n{rl}' for n, rl in zip(names, reward_labels)], fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title('Impact of Reward Structure on DQN Detection Behaviour', fontsize=14, pad=15)
    ax.legend(fontsize=10, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(FIGURES_DIR, 'fig_reward_impact.png')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {path}")


# ============================================================================
# FIGURE 7: Confusion Matrices
# ============================================================================
def fig_confusion_matrices(scenarios):
    """Heatmap confusion matrices per scenario."""
    for s_key, s_data in scenarios.items():
        if s_key not in SCENARIO_TITLES:
            continue

        models = list(s_data.keys())
        n = len(models)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, model in enumerate(models):
            ax = axes[i]
            cm = np.array(s_data[model].get('ConfusionMatrix', [[0, 0], [0, 0]]))
            total = cm.sum() if cm.sum() > 0 else 1

            annot = np.array([
                [f'TN\n{cm[0][0]:,}\n({cm[0][0]/total*100:.1f}%)',
                 f'FP\n{cm[0][1]:,}\n({cm[0][1]/total*100:.1f}%)'],
                [f'FN\n{cm[1][0]:,}\n({cm[1][0]/total*100:.1f}%)',
                 f'TP\n{cm[1][1]:,}\n({cm[1][1]/total*100:.1f}%)']
            ])

            color_vals = np.array([[0.75, 0.25], [0.25, 0.75]])
            sns.heatmap(color_vals, annot=annot, fmt='', ax=ax, cmap='RdYlGn',
                        vmin=0, vmax=1, cbar=False,
                        xticklabels=['Pred: Allow', 'Pred: Block'],
                        yticklabels=['True: Benign', 'True: Attack'],
                        annot_kws={'fontsize': 10, 'fontweight': 'bold'})
            ax.set_title(model, fontsize=13, fontweight='bold', pad=10)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(SCENARIO_TITLES[s_key] + '\nConfusion Matrices',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()

        path = os.path.join(FIGURES_DIR, f'fig_cm_{s_key}.png')
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 55)
    print("  RL-Enhanced IDS — Figure Generator")
    print("=" * 55)

    scenarios, experiments = load_data()
    if not scenarios and not experiments:
        print("\n  ✗ No data found. Run experiments first.")
        sys.exit(1)

    print("\nGenerating figures...")
    fig_scenario_comparison(scenarios)
    fig_cross_scenario_summary(scenarios)
    fig_dqn_f1_progression(experiments)
    fig_dqn_precision_recall(experiments)
    fig_dqn_fp_fn(experiments)
    fig_reward_impact(experiments)
    fig_confusion_matrices(scenarios)

    print(f"\n  ✓ All figures saved to {FIGURES_DIR}/")
    print("  Done!")


if __name__ == '__main__':
    main()
