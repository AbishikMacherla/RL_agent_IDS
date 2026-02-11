# IDS Dashboard - Live Training and Evaluation Interface
# Streamlit app for RL agent training, tuning, and comparison
#
# Purpose: Proof of concept dashboard for RL-Enhanced IDS
# - Real-time training visualization
# - Hyperparameter tuning controls
# - Model comparison (RL vs ML)
# - Simulated traffic classification

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import threading
import queue
from datetime import datetime
import joblib

# Configure page
st.set_page_config(
    page_title="RL-IDS Dashboard",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============== STATE MANAGEMENT ==============
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'training_data' not in st.session_state:
    st.session_state.training_data = {'episodes': [], 'rewards': [], 'losses': []}
if 'live_metrics' not in st.session_state:
    st.session_state.live_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}


# ============== SIDEBAR: AGENT CONTROLS ==============
st.sidebar.title("Agent Controls")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["DQN", "PPO"],
    help="DQN: Value-based RL, PPO: Policy-based RL"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")

# ============== DQN HYPERPARAMETERS ==============
# Note: These are tunable parameters for the DQN agent
# Default values are based on research best practices
# If training fails or converges slowly, adjust these:
#   - Learning rate: Lower if unstable (0.0001), higher if slow (0.001)
#   - Epsilon decay: Slower decay if exploring too little
#   - Buffer size: Larger prevents forgetting, smaller trains faster

if algorithm == "DQN":
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001, max_value=0.01, value=0.0005, step=0.0001,
        format="%.4f",
        help="Default: 0.0005. Lower if training is unstable."
    )
    epsilon_start = st.sidebar.slider(
        "Epsilon Start (Exploration)",
        min_value=0.1, max_value=1.0, value=1.0, step=0.1,
        help="Start with full exploration (1.0)"
    )
    epsilon_end = st.sidebar.slider(
        "Epsilon End",
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        help="Maintain some exploration (0.05)"
    )
    epsilon_decay = st.sidebar.slider(
        "Epsilon Decay",
        min_value=0.99, max_value=0.9999, value=0.999, step=0.001,
        format="%.4f",
        help="Slow decay for better exploration"
    )
    buffer_size = st.sidebar.select_slider(
        "Replay Buffer Size",
        options=[10000, 50000, 100000],
        value=100000,
        help="Larger buffer prevents catastrophic forgetting"
    )
    batch_size = st.sidebar.select_slider(
        "Batch Size",
        options=[32, 64, 128],
        value=64,
        help="Smaller batch for less bias to recent experiences"
    )
    
# ============== PPO HYPERPARAMETERS ==============
# Note: PPO is generally more stable than DQN
# Key parameters to tune:
#   - Clip range: Prevents too large policy updates
#   - Entropy coefficient: Higher = more exploration

else:  # PPO
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001, max_value=0.001, value=0.0003, step=0.0001,
        format="%.4f",
        help="Default: 0.0003. PPO is less sensitive to this."
    )
    clip_range = st.sidebar.slider(
        "Clip Range",
        min_value=0.1, max_value=0.4, value=0.2, step=0.05,
        help="Prevents too large policy updates. Default: 0.2"
    )
    n_epochs = st.sidebar.slider(
        "Optimization Epochs",
        min_value=1, max_value=20, value=10,
        help="Epochs per batch update. Default: 10"
    )
    entropy_coef = st.sidebar.slider(
        "Entropy Coefficient",
        min_value=0.0, max_value=0.1, value=0.01, step=0.005,
        format="%.3f",
        help="Higher = more exploration. Default: 0.01"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Reward Weights")
st.sidebar.caption("Asymmetric rewards for security-first approach")

# ============== REWARD STRUCTURE ==============
# Note: Security-first approach prioritizes catching attacks
# True Positive reward > True Negative (catching attacks is critical)
# False Negative penalty > False Positive (missing attacks is worse)

reward_tp = st.sidebar.slider(
    "True Positive (Catch Attack)",
    min_value=1, max_value=20, value=10,
    help="Reward for correctly blocking malicious traffic"
)
reward_tn = st.sidebar.slider(
    "True Negative (Allow Benign)",
    min_value=1, max_value=5, value=1,
    help="Reward for correctly allowing benign traffic"
)
penalty_fn = st.sidebar.slider(
    "False Negative (Miss Attack)",
    min_value=-20, max_value=-1, value=-10,
    help="Penalty for missing malicious traffic (critical!)"
)
penalty_fp = st.sidebar.slider(
    "False Positive (Block Benign)",
    min_value=-5, max_value=-1, value=-1,
    help="Penalty for blocking benign traffic (alert fatigue)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Training Settings")

n_episodes = st.sidebar.number_input(
    "Number of Episodes",
    min_value=100, max_value=10000, value=2000, step=100
)

# Training controls
col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("Start", use_container_width=True)
stop_btn = col2.button("Stop", use_container_width=True)


# ============== MAIN CONTENT ==============
st.title("RL-Enhanced Intrusion Detection System")
st.markdown("**Proof of Concept Dashboard for Autonomous Network Defence**")

# Top metrics row
st.subheader("Live Performance Metrics")
metric_cols = st.columns(4)

with metric_cols[0]:
    st.metric(
        label="Accuracy",
        value=f"{st.session_state.live_metrics['accuracy']:.2%}",
        delta="0.5%"
    )
with metric_cols[1]:
    st.metric(
        label="Precision",
        value=f"{st.session_state.live_metrics['precision']:.2%}",
        delta="0.3%"
    )
with metric_cols[2]:
    st.metric(
        label="Recall",
        value=f"{st.session_state.live_metrics['recall']:.2%}",
        delta="0.8%"
    )
with metric_cols[3]:
    st.metric(
        label="F1 Score",
        value=f"{st.session_state.live_metrics['f1']:.2%}",
        delta="0.4%"
    )

st.markdown("---")

# Charts row
chart_cols = st.columns(2)

with chart_cols[0]:
    st.subheader("Training Progress")
    
    # Sample training data for demonstration
    if len(st.session_state.training_data['episodes']) == 0:
        # Demo data
        demo_episodes = list(range(1, 101))
        demo_rewards = [50 + i * 0.5 + np.random.randn() * 10 for i in range(100)]
    else:
        demo_episodes = st.session_state.training_data['episodes']
        demo_rewards = st.session_state.training_data['rewards']
    
    fig_training = px.line(
        x=demo_episodes,
        y=demo_rewards,
        labels={'x': 'Episode', 'y': 'Reward'},
        title=f"{algorithm} Training Reward"
    )
    fig_training.update_layout(
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig_training, use_container_width=True)

with chart_cols[1]:
    st.subheader("Confusion Matrix")
    
    # Sample confusion matrix
    cm_data = np.array([[8500, 1500], [50, 9950]])  # TN, FP, FN, TP
    
    fig_cm = px.imshow(
        cm_data,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Benign', 'Malicious'],
        y=['Benign', 'Malicious'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig_cm.update_layout(
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# Model comparison section
st.subheader("Model Comparison: RL vs ML")
st.caption("Comparing RL agents against traditional ML baselines")

# Sample comparison data
comparison_data = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'DQN', 'PPO'],
    'Type': ['ML', 'ML', 'RL', 'RL'],
    'Accuracy': [0.9985, 0.9991, 0.7462, 0.8234],
    'Precision': [0.9987, 0.9993, 0.5012, 0.6521],
    'Recall': [0.9978, 0.9985, 1.0000, 0.9850],
    'F1 Score': [0.9982, 0.9989, 0.6680, 0.7845],
    'FP Rate': [0.001, 0.0007, 0.2538, 0.1523]
})

# Highlight: RL achieves 100% recall (catches all attacks)
st.info("Note: RL agents achieve near-perfect recall (100%) at the cost of higher false positives. "
        "This security-first approach ensures no attacks are missed.")

# Color-code by type
fig_comparison = px.bar(
    comparison_data,
    x='Model',
    y=['Accuracy', 'Recall', 'F1 Score'],
    barmode='group',
    color_discrete_sequence=['#2ecc71', '#3498db', '#9b59b6'],
    title="Algorithm Performance Comparison"
)
fig_comparison.update_layout(
    template="plotly_dark",
    height=350
)
st.plotly_chart(fig_comparison, use_container_width=True)

# Comparison table
st.dataframe(
    comparison_data.style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1 Score': '{:.2%}',
        'FP Rate': '{:.2%}'
    }).background_gradient(subset=['Recall'], cmap='Greens'),
    use_container_width=True
)

st.markdown("---")

# Live traffic simulation section
st.subheader("Simulated Traffic Classification")
st.caption("Dataset records streamed as simulated network traffic")

# Traffic feed placeholder
traffic_placeholder = st.empty()

with traffic_placeholder.container():
    traffic_cols = st.columns([1, 2, 1, 1])
    
    with traffic_cols[0]:
        st.markdown("**Timestamp**")
        st.text(datetime.now().strftime("%H:%M:%S"))
    
    with traffic_cols[1]:
        st.markdown("**Traffic Sample**")
        st.code("SYN->ACK flow | 192.168.1.x -> 10.0.0.x | 1.2KB", language=None)
    
    with traffic_cols[2]:
        st.markdown("**Prediction**")
        st.success("BENIGN")
    
    with traffic_cols[3]:
        st.markdown("**Confidence**")
        st.progress(0.95)
        st.caption("95%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>RL-Enhanced IDS Dashboard | Honours Project 2024-25</p>
    <p>Abishik Macherla Vijayakrishna</p>
</div>
""", unsafe_allow_html=True)


# ============== PARAMETER REFERENCE ==============
# This section documents all tunable parameters for easy reference
# Use these as starting points and adjust based on training results
#
# DQN Parameters:
#   BUFFER_SIZE = 100000    # Experience replay buffer
#   BATCH_SIZE = 64         # Training batch size
#   GAMMA = 0.99            # Discount factor
#   TAU = 0.001             # Soft update coefficient
#   LR = 0.0005             # Learning rate
#   UPDATE_EVERY = 4        # Steps between network updates
#   EPS_START = 1.0         # Initial exploration
#   EPS_END = 0.05          # Final exploration
#   EPS_DECAY = 0.999       # Exploration decay rate
#
# PPO Parameters:
#   LEARNING_RATE = 0.0003  # Adam optimizer learning rate
#   N_STEPS = 2048          # Steps per update
#   BATCH_SIZE = 64         # Minibatch size
#   N_EPOCHS = 10           # Optimization epochs
#   GAMMA = 0.99            # Discount factor
#   GAE_LAMBDA = 0.95       # GAE lambda
#   CLIP_RANGE = 0.2        # PPO clip parameter
#   ENT_COEF = 0.01         # Entropy bonus
#
# Reward Structure:
#   TP_REWARD = +10         # True Positive (catch attack)
#   TN_REWARD = +1          # True Negative (allow benign)
#   FN_PENALTY = -10        # False Negative (miss attack)
#   FP_PENALTY = -1         # False Positive (block benign)
