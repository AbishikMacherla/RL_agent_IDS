#!/usr/bin/env python3
"""
RL-Enhanced IDS Dashboard — Streamlit Web Application
Interactive dashboard for viewing experiment results, comparing models,
and understanding RL agent performance for intrusion detection.

Usage:
    streamlit run dashboard.py
    or: ./dashboard
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================================
# CONFIG & DATA
# ============================================================================

PROJECT_DIR = '/home/abishik/HONOURS_PROJECT'
RESULTS_FILE = os.path.join(PROJECT_DIR, 'results', 'all_scenarios_results.json')
DQN_EXPERIMENTS_FILE = os.path.join(PROJECT_DIR, 'results', 'dqn_experiments.json')

SCENARIO_INFO = {
    'scenario_1': {
        'title': '📊 Scenario 1: Standard Classification',
        'desc': 'All models trained & tested on CIC-IDS2017 (78 features)',
        'short': 'Standard',
    },
    'scenario_2': {
        'title': '🛡️ Scenario 2: Zero-Day DDoS',
        'desc': 'RL trained WITHOUT DDoS → tested on DDoS-only data',
        'short': 'Zero-Day DDoS',
    },
    'scenario_3': {
        'title': '🌐 Scenario 3: Zero-Day Web Attacks',
        'desc': 'RL trained WITHOUT web attacks → tested on web-only data',
        'short': 'Zero-Day Web',
    },
    'scenario_4': {
        'title': '🔄 Scenario 4: Cross-Dataset (2017→2023)',
        'desc': 'Trained on CIC-IDS2017, tested on CIC-IoT-2023 (12 features)',
        'short': 'Cross-Dataset',
    },
}

MODEL_COLORS = {
    'Random Forest': '#2ecc71',
    'XGBoost': '#3498db',
    'DQN': '#f39c12',
    'DQN (Standard)': '#f39c12',
    'DQN (No DDoS)': '#e67e22',
    'DQN (No Web)': '#e67e22',
    'PPO': '#9b59b6',
    'PPO (Standard)': '#9b59b6',
    'PPO (No DDoS)': '#e74c3c',
    'PPO (No Web)': '#e74c3c',
}

FILTER_CYCLE = ['All', 'Random Forest', 'XGBoost', 'DQN', 'PPO']


@st.cache_data(ttl=30)
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=30)
def load_dqn_experiments():
    if os.path.exists(DQN_EXPERIMENTS_FILE):
        with open(DQN_EXPERIMENTS_FILE) as f:
            return json.load(f)
    return {}


# ============================================================================
# CHART BUILDERS
# ============================================================================

def build_metrics_comparison_chart(data, metrics=None):
    """Grouped bar chart comparing models across metrics."""
    if not data:
        return None
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    fig = go.Figure()
    for metric in metrics:
        names = list(data.keys())
        values = [data[n].get(metric, 0) for n in names]
        colors = [MODEL_COLORS.get(n, '#95a5a6') for n in names]
        fig.add_trace(go.Bar(
            name=metric,
            x=names,
            y=values,
            text=[f'{v:.1f}%' for v in values],
            textposition='outside',
        ))

    fig.update_layout(
        barmode='group',
        yaxis_title='Score (%)',
        yaxis_range=[0, 105],
        template='plotly_dark',
        height=400,
        margin=dict(t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(size=13),
    )
    return fig


def build_radar_chart(data):
    """Radar chart for multi-dimensional comparison."""
    if not data:
        return None

    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    fig = go.Figure()

    for name, m in data.items():
        values = [m.get(c, 0) for c in categories]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            line=dict(color=MODEL_COLORS.get(name, '#95a5a6')),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)',
        ),
        template='plotly_dark',
        height=400,
        margin=dict(t=30, b=30),
        font=dict(size=13),
    )
    return fig


def build_confusion_matrix_chart(metrics, model_name):
    """Heatmap confusion matrix."""
    cm = np.array(metrics.get('ConfusionMatrix', [[0, 0], [0, 0]]))
    labels = ['Benign (Allow)', 'Attack (Block)']

    # Normalised for colour, raw for text
    cm_norm = cm.astype(float) / cm.sum() * 100

    text = [[f'{cm[i][j]:,}<br>({cm_norm[i][j]:.1f}%)' for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=['Predicted Allow', 'Predicted Block'],
        y=['Actual Benign', 'Actual Attack'],
        text=text,
        texttemplate='%{text}',
        colorscale=[[0, '#1a1a2e'], [0.5, '#16213e'], [1, '#e94560']],
        showscale=False,
        hovertemplate='%{y} → %{x}<br>Count: %{text}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text=f'{model_name} — Confusion Matrix', font=dict(size=14)),
        template='plotly_dark',
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        font=dict(size=13),
        yaxis=dict(autorange='reversed'),
    )
    return fig


def build_roc_auc_bar(data):
    """Bar chart for ROC-AUC scores."""
    models = []
    scores = []
    colors = []
    for name, m in data.items():
        auc = m.get('ROC_AUC')
        if auc is not None:
            models.append(name)
            scores.append(auc)
            colors.append(MODEL_COLORS.get(name, '#95a5a6'))

    if not models:
        return None

    fig = go.Figure(go.Bar(
        x=models, y=scores,
        marker_color=colors,
        text=[f'{s:.1f}%' for s in scores],
        textposition='outside',
    ))
    fig.update_layout(
        yaxis_title='ROC-AUC (%)',
        yaxis_range=[0, 105],
        template='plotly_dark',
        height=300,
        margin=dict(t=20, b=30),
        font=dict(size=13),
    )
    return fig


def build_latency_chart(data):
    """Horizontal bar chart for inference latency."""
    models = []
    latencies = []
    throughputs = []
    colors = []
    for name, m in data.items():
        lat = m.get('Latency_us')
        thr = m.get('Throughput')
        if lat is not None:
            models.append(name)
            latencies.append(lat)
            throughputs.append(thr or 0)
            colors.append(MODEL_COLORS.get(name, '#95a5a6'))

    if not models:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Latency (μs/sample)', 'Throughput (samples/sec)'))

    fig.add_trace(go.Bar(
        y=models, x=latencies, orientation='h',
        marker_color=colors,
        text=[f'{l:.1f}μs' for l in latencies],
        textposition='outside',
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=models, x=throughputs, orientation='h',
        marker_color=colors,
        text=[f'{t:,.0f}/s' for t in throughputs],
        textposition='outside',
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        template='plotly_dark',
        height=250,
        margin=dict(t=40, b=20),
        font=dict(size=12),
    )
    return fig


def build_dqn_experiment_chart(experiments):
    """Chart showing F1 vs FP trade-off across experiments."""
    if not experiments:
        return None

    names = []
    f1_scores = []
    fp_counts = []
    precisions = []

    for key, exp in experiments.items():
        r = exp.get('results', {})
        names.append(exp.get('name', key).replace('Exp ', ''))
        f1_scores.append(r.get('F1', 0))
        fp_counts.append(r.get('FP', 0))
        precisions.append(r.get('Precision', 0))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('F1 Score by Experiment', 'False Positives by Experiment'),
    )

    # F1 bars
    colors = ['#2ecc71' if f > 92 else ('#f39c12' if f > 85 else '#e74c3c') for f in f1_scores]
    fig.add_trace(go.Bar(
        x=names, y=f1_scores,
        marker_color=colors,
        text=[f'{f:.1f}%' for f in f1_scores],
        textposition='outside',
        showlegend=False,
    ), row=1, col=1)

    # FP bars
    fp_colors = ['#2ecc71' if fp < 10000 else ('#f39c12' if fp < 50000 else '#e74c3c') for fp in fp_counts]
    fig.add_trace(go.Bar(
        x=names, y=fp_counts,
        marker_color=fp_colors,
        text=[f'{fp:,}' for fp in fp_counts],
        textposition='outside',
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        template='plotly_dark',
        height=350,
        margin=dict(t=40, b=40),
        font=dict(size=12),
    )
    fig.update_yaxes(title_text='F1 (%)', range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text='False Positives', row=1, col=2)
    return fig


# ============================================================================
# PAGE SECTIONS
# ============================================================================

def render_scenario_page(scenario_key, data, model_filter):
    """Render a single scenario's results page."""
    info = SCENARIO_INFO[scenario_key]
    st.header(info['title'])
    st.caption(info['desc'])

    if not data:
        st.warning(f'No results available. Run: `python run_all_scenarios.py --scenario {scenario_key[-1]}`')
        return

    # Apply filter
    if model_filter != 'All':
        data = {k: v for k, v in data.items() if model_filter.lower() in k.lower()}
        if not data:
            st.info(f'No results for filter "{model_filter}" in this scenario.')
            return

    # Metrics table
    df = pd.DataFrame(data).T
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'Latency_us', 'Throughput']
    display_cols = [c for c in metric_cols if c in df.columns]
    st.dataframe(
        df[display_cols].style.format('{:.2f}', subset=[c for c in display_cols if c != 'Throughput'])
        .format('{:,.0f}', subset=['Throughput'] if 'Throughput' in display_cols else [])
        .highlight_max(axis=0, props='background-color: #2ecc71; color: white', subset=['F1'] if 'F1' in display_cols else []),
        use_container_width=True,
    )

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Performance Comparison')
        fig = build_metrics_comparison_chart(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Radar View')
        fig = build_radar_chart(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ROC-AUC and Latency
    has_auc = any(m.get('ROC_AUC') for m in data.values())
    has_lat = any(m.get('Latency_us') for m in data.values())

    if has_auc or has_lat:
        col3, col4 = st.columns(2)
        if has_auc:
            with col3:
                st.subheader('ROC-AUC')
                fig = build_roc_auc_bar(data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        if has_lat:
            with col4:
                st.subheader('Inference Speed')
                fig = build_latency_chart(data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.subheader('Confusion Matrices')
    cm_cols = st.columns(min(len(data), 4))
    for i, (name, m) in enumerate(data.items()):
        with cm_cols[i % len(cm_cols)]:
            fig = build_confusion_matrix_chart(m, name)
            st.plotly_chart(fig, use_container_width=True)


def render_dqn_experiments_page(experiments):
    """Render the DQN hyperparameter experiments page."""
    st.header('🧪 DQN Hyperparameter Experiments')
    st.caption('8 systematic experiments tuning reward structure, architecture & training duration')

    if not experiments:
        st.warning('No DQN experiment results found.')
        return

    # Build table
    rows = []
    for key, exp in experiments.items():
        r = exp.get('results', {})
        rw = exp.get('rewards', {})
        rows.append({
            'Experiment': exp.get('name', key),
            'Rewards (TP/TN/FN/FP)': f"+{rw.get('tp',0)}/+{rw.get('tn',0)}/{rw.get('fn',0)}/{rw.get('fp',0)}",
            'Network': exp.get('network', '64-64'),
            'Episodes': exp.get('episodes', 2000),
            'LR': exp.get('learning_rate', 0.0005),
            'Accuracy': r.get('Accuracy', 0),
            'Precision': r.get('Precision', 0),
            'Recall': r.get('Recall', 0),
            'F1': r.get('F1', 0),
            'FP': r.get('FP', 0),
            'FN': r.get('FN', 0),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style
        .highlight_max(axis=0, props='background-color: #2ecc71; color: white', subset=['F1', 'Precision'])
        .highlight_min(axis=0, props='background-color: #2ecc71; color: white', subset=['FP'])
        .format({'Accuracy': '{:.1f}%', 'Precision': '{:.1f}%', 'Recall': '{:.1f}%', 'F1': '{:.1f}%', 'FP': '{:,}', 'FN': '{:,}', 'LR': '{:.4f}'}),
        use_container_width=True,
        hide_index=True,
    )

    # Chart
    fig = build_dqn_experiment_chart(experiments)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Key findings
    best = max(experiments.values(), key=lambda e: e.get('results', {}).get('F1', 0))
    best_r = best.get('results', {})

    st.success(f"**Best: {best.get('name')}** — F1={best_r.get('F1',0):.1f}%, Precision={best_r.get('Precision',0):.1f}%, FP={best_r.get('FP',0):,}")

    with st.expander('💡 Key Insights', expanded=True):
        st.markdown("""
        - **Reward structure is the primary driver** of agent behaviour
        - **Asymmetric (10:1)** rewards → 98% recall but 94K false positives (alert fatigue)
        - **Symmetric (+1/+1/-1/-1)** → 93% F1 with only 8K FP (balanced for real-world use)
        - **Larger network (128-128)** did not improve performance — 64-64 is sufficient
        - **More episodes (5K vs 2K)** gave the best result — DQN benefits from longer training
        """)


def render_guide_page():
    """Render the user guide page."""
    st.header('📖 Dashboard Guide')

    st.markdown("""
    ### How to Use This Dashboard

    **Sidebar Controls:**
    - **Page selector** — Switch between scenarios, experiments, and this guide
    - **Model filter** — View specific model types (RF, XGBoost, DQN, PPO)
    - **Reload** — Refresh data after running new experiments

    ---

    ### What Each Scenario Tests

    | Scenario | Purpose | Key Question |
    |----------|---------|--------------|
    | **1. Standard** | Baseline comparison on CIC-IDS2017 | How does RL compare to ML? |
    | **2. Zero-Day DDoS** | RL trained without DDoS | Can RL detect unseen attack types? |
    | **3. Zero-Day Web** | RL trained without web attacks | Generalisation to app-layer threats? |
    | **4. Cross-Dataset** | Train 2017 → Test 2023 | Do patterns transfer across networks? |

    ---

    ### Understanding the Metrics

    | Metric | What It Measures | Why It Matters |
    |--------|-----------------|----------------|
    | **Accuracy** | Overall correct rate | Can be misleading with imbalanced data |
    | **Precision** | Of "attack" predictions, % correct | High = fewer false alarms |
    | **Recall** | Of actual attacks, % detected | High = fewer missed attacks |
    | **F1** | Balance of precision & recall | Best single metric for comparison |
    | **ROC-AUC** | Discrimination ability | Higher = better at separating classes |
    | **Latency** | μs per sample | Real-time capability |
    | **Throughput** | Samples/second | Network capacity |

    ---

    ### Confusion Matrix Key

    |  | Predicted Allow | Predicted Block |
    |--|----------------|-----------------|
    | **Actually Benign** | ✅ TN (correct) | ⚠️ FP (false alarm) |
    | **Actually Attack** | 🚨 FN (missed!) | ✅ TP (caught!) |

    - **FN is most dangerous** — missed attacks
    - **FP causes alert fatigue** — too many false alarms

    ---

    ### Running Experiments

    ```bash
    source ~/HONOURS_PROJECT/venv/bin/activate

    # Run individual scenarios
    python run_all_scenarios.py --scenario 1
    python run_all_scenarios.py --scenario 2
    python run_all_scenarios.py --scenario 3
    python run_all_scenarios.py --scenario 4

    # Or: run a scenario and then open dashboard
    ./dashboard --run 1
    ```

    After training completes, click **🔄 Reload Data** in the sidebar.
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title='RL-Enhanced IDS Dashboard',
        page_icon='🛡️',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    # Custom CSS for dark premium look
    st.markdown("""
    <style>
        /* Dark theme enhancements */
        .stApp { background-color: #0e1117; }
        .stMetricValue { font-size: 1.4rem !important; }
        div[data-testid="stMetricDelta"] { font-size: 0.9rem; }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }

        /* Smooth transitions */
        .stPlotlyChart { transition: all 0.3s ease; }
        .stPlotlyChart:hover { transform: scale(1.01); }

        /* Header gradient */
        h1 { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             font-weight: 800 !important; }

        h2, h3 { color: #c9d1d9 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Load data
    results = load_results()
    dqn_experiments = load_dqn_experiments()

    # Sidebar
    with st.sidebar:
        st.title('🛡️ RL-Enhanced IDS')
        st.caption('Autonomous Network Defence')
        st.divider()

        # Page selector
        pages = ['📊 Scenario 1', '🛡️ Scenario 2', '🌐 Scenario 3', '🔄 Scenario 4',
                 '🧪 DQN Experiments', '📖 Guide']
        page = st.radio('Navigate', pages, label_visibility='collapsed')

        st.divider()

        # Model filter
        model_filter = st.selectbox('Model Filter', FILTER_CYCLE, index=0)

        st.divider()

        # Reload button
        if st.button('🔄 Reload Data', use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # Status
        n_scenarios = len(results)
        n_models = sum(len(v) for v in results.values())
        st.caption(f'📁 {n_scenarios} scenarios, {n_models} model results')

    # Route to page
    if page == '📊 Scenario 1':
        render_scenario_page('scenario_1', results.get('scenario_1', {}), model_filter)
    elif page == '🛡️ Scenario 2':
        render_scenario_page('scenario_2', results.get('scenario_2', {}), model_filter)
    elif page == '🌐 Scenario 3':
        render_scenario_page('scenario_3', results.get('scenario_3', {}), model_filter)
    elif page == '🔄 Scenario 4':
        render_scenario_page('scenario_4', results.get('scenario_4', {}), model_filter)
    elif page == '🧪 DQN Experiments':
        render_dqn_experiments_page(dqn_experiments)
    elif page == '📖 Guide':
        render_guide_page()


if __name__ == '__main__':
    main()
