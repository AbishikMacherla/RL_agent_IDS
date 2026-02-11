# Sources Needed for Dissertation

Please search for academic papers matching these descriptions. Provide URLs (arXiv, IEEE, ACM, etc.) and I will add them to `references.bib`.

---

## 1. Traditional IDS Limitations
**Need**: Evidence that current IDSs still rely on static signatures or supervised ML and struggle with zero-day attacks.

**Search Terms**: "intrusion detection system zero-day limitations survey", "signature-based IDS limitations"

**Already Have**: Ring et al. (2019) survey on IDS datasets.

**🆕 New Research Found**:
- XGBoost achieves 99%+ accuracy but requires retraining for new attacks
- Traditional ML struggles with concept drift and adversarial settings

---

## 2. RL for Cybersecurity (General)
**Need**: Papers showing RL has been applied to network security with success.

**Search Terms**: "reinforcement learning intrusion detection", "deep Q-network cyber defense"

**Already Have**: Yang et al. (2024) survey, Hsu & Matsuoka (2020).

**🆕 New Research Found**:
- PPO-based IDS achieved 97.16% accuracy across NSL-KDD, CICIDS, TON-IoT, DDoS, UNSW-NB15
- PPO can control hyperparameters automatically based on network environment

---

## 3. Zero-Day Attack Detection
**Need**: Papers specifically on detecting previously unseen attacks.

**Search Terms**: "zero-day attack detection machine learning", "transfer learning intrusion detection"

**Why**: This supports our core thesis that adaptability (RL) beats static models (ML).

---

## 4. CybORG / CybORG++ / OpenAI Gym Environment
**Need**: The original papers for the simulation environment.

**Already Have**: Standen et al. (2021) for CybORG, Emerson et al. (2024) for CybORG++.

**🆕 To Add**: 
- Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540

---

## 5. CIC-IDS2017 Dataset
**Need**: The paper describing the dataset creation and validation.

**Already Have**: Sharafaldin et al. (2018).

---

## 6. DQN Algorithm
**Need**: The original DeepMind paper introducing DQN.

**Search Terms**: "Playing Atari with deep reinforcement learning Mnih"

**Citation**: Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature.

**🆕 Best Hyperparameters from Research**:
| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Epsilon start | 1.0 | Start fully random |
| Epsilon end | 0.05 | Maintain some exploration |
| Epsilon decay | 0.999 or 100k steps | Slow decay |
| Replay buffer | 10,000 - 100,000 | Prevents forgetting |
| Batch size | 25 - 50 | Small relative to buffer |
| Learning rate | 0.001 | Standard |
| Discount (γ) | 0.95 - 0.99 | Future reward weight |
| Hidden layers | 128 → 64 (ReLU) | Efficient architecture |

**Why This Matters (For Report)**:
DQN combines Q-Learning (learning action values) with Deep Neural Networks (handling complex inputs). The agent observes network traffic, passes it through hidden layers, and outputs Q-values for "Allow" or "Block".

---

## 7. PPO Algorithm
**Need**: The OpenAI paper introducing PPO.

**Already Have**: Schulman et al. (2017). arXiv:1707.06347

**🆕 PPO for IDS Research Found**:
| Source | Key Finding |
|--------|-------------|
| Semantic Scholar | 97.16% accuracy on multiple IDS datasets |
| MDPI | PPO for DNN/clustering hyperparameter control |

---

## 8. Random Forest / XGBoost for IDS
**Need**: Papers showing RF/XGBoost used for intrusion detection (baseline justification).

**Search Terms**: "random forest intrusion detection", "XGBoost network intrusion CIC-IDS"

**🆕 XGBoost Research Found**:
- 99%+ accuracy on CIC-IDS2017/2018
- 10x faster than some ML methods
- Outperforms RF, SVM, KNN, AdaBoost
- Key params: learning_rate, max_depth, n_estimators, gamma

---

## 9. Class Imbalance in Network Security
**Need**: Papers discussing handling imbalanced datasets in IDS (attack traffic is rare).

**Search Terms**: "class imbalance intrusion detection SMOTE"

**Why**: Justifies our reward engineering approach (asymmetric rewards).

---

## 10. Dwell Time / Incident Response Statistics
**Need**: Real-world incident reports showing dwell time.

**Search Terms**: "dwell time cyber attack statistics 2024", "IBM Cost of a Data Breach Report"

**Why**: Strengthens the motivation section.

---

## Alignment Check Summary

| Requirement | Status | Evidence |
|-------------|--------|----------|
| IPO: RL-enhanced IDS | ✅ | DQN + PPO implementation |
| IPO: Baseline comparison | ✅ | RF + XGBoost |
| IPO: CIC-IDS2017 | ✅ | Training dataset |
| IPO: Performance metrics | ✅ | Dashboard with live metrics |
| IPO: Dashboard | ✅ | Streamlit with RL controls |
| Guidelines: 6 intro elements | ✅ | Introduction rewritten |
| Guidelines: Lit review synthesis | 🔄 | In progress |
| Inspiration: Multi-algorithm | ✅ | 2 RL + 2 ML algorithms |

---

## 11. Microsoft CyberBattleSim Methodology (FROM FEEDBACK)
**Need**: How Microsoft structured RL experiments for cybersecurity

**Search Terms**: "Microsoft CyberBattleSim", "Microsoft RL cyber defense methodology"

**Why**: Feedback suggested following their concepts if we struggle with RL performance

**Status**: 🔄 Need to search

---

## 12. Online/Continual Learning for IDS (FROM FEEDBACK)
**Need**: Papers on updating IDS models incrementally without full retraining

**Search Terms**: "continual learning intrusion detection", "online learning IDS whitelist"

**Why**: Supports the "good traffic whitelist" scenario idea from feedback

**Status**: 🔄 Need to search

---

## Current Experiment Results (For Reference)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 99.95% | 99.83% | 99.90% | 99.87% |
| Random Forest | 99.90% | 99.64% | 99.86% | 99.75% |
| **DQN (RL)** | 91.20% | 70.00% | **97.00%** | 81.00% |
| PPO (RL) | 69.23% | 31.55% | 48.20% | 38.14% |

**Key Finding**: DQN achieves 97% Recall - security-first reward structure works!

---

## Still Missing (From Feedback)

- [ ] IBM 4-year trend data (not just costs)
- [ ] Evidence ML fails on zero-day specifically
- [ ] Real-time latency benchmarks
- [ ] Microsoft CyberBattleSim papers

---

*User: Please provide URLs for any papers you find. I will then add the BibTeX entries and integrate the citations.*

---

## 13. Threat Model Formalization (STRIDE/Attack Trees) - NEW FROM FEEDBACK

**Need**: Academic sources supporting formal threat modeling in IDS/autonomous defence systems

**Search Terms**:
- "STRIDE threat model intrusion detection"
- "attack tree cyber defense formal model"
- "attacker capability model network security"
- "threat modeling autonomous cyber operations"

**Specific Papers to Look For**:
1. **STRIDE methodology paper** - Microsoft's original threat classification (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)
2. **Attack trees for IDS** - Paper showing how threat scenarios can be formally structured
3. **Attacker capability taxonomy** - Any paper defining attacker knowledge/capability levels in network intrusion contexts

**Why**: Supervisor requires formal threat model to justify "autonomous defence" claims. Need academic backing for attacker goals, capabilities, and knowledge assumptions.

**Status**: 🔴 URGENT - Need to search

---

## 14. RL vs Classification Justification - NEW FROM FEEDBACK

**Need**: Papers justifying when RL is appropriate vs supervised learning for security tasks

**Search Terms**:
- "offline reinforcement learning intrusion detection"
- "contextual bandits network security"
- "reward shaping classification problems"
- "when to use RL vs supervised learning"
- "reinforcement learning advantages over classification"

**Specific Papers to Look For**:
1. **Offline RL for security** - Papers showing RL trained on static datasets (like ours)
2. **Contextual bandits for IDS** - If this framing is more accurate than full RL
3. **Reward shaping benefits** - Papers showing how asymmetric rewards achieve what loss functions cannot
4. **Sequential decision-making in network monitoring** - Papers arguing traffic analysis IS sequential

**Key Evidence Needed**:
- Why reward engineering provides flexibility that cross-entropy loss cannot
- How offline RL on logged data can still be called "RL"
- Examples of production IDS using RL (even if on static data)

**Why**: Supervisor questions whether our approach is "genuinely RL" or "cost-sensitive classification with RL vocabulary". We need academic evidence for both honest framing AND practical benefits.

**Status**: 🔴 URGENT - Need to search

---

## 15. Periodical/Curriculum/Online Learning for RL-IDS - NEW FROM YOUR QUESTION

**Need**: Papers on incrementally feeding data to RL agents for better performance

**Search Terms**:
- "curriculum learning reinforcement learning intrusion detection"
- "online learning RL network security"
- "incremental reinforcement learning IDS"
- "continual learning cyber defense"
- "episodic training intrusion detection"

**Specific Papers to Look For**:
1. **Curriculum learning for RL** - Training on easy cases first, then harder (could improve PPO)
2. **Online RL for IDS** - Agents that update as new traffic arrives
3. **Experience replay strategies** - Prioritized replay for rare attack cases
4. **Episodic vs continuous training** - Different ways to structure the learning process

**Why**: You asked if periodical feeding and parameter changes could beat ML. This is a valid research direction - could argue RL can continuously improve while ML stays static.

**Potential Experiment**: Train DQN/PPO with curriculum (benign first → easy attacks → hard attacks) and compare to standard random sampling.

**Status**: 🟠 Useful for Discussion section

---

## 16. Data Leakage Controls in IDS Research - NEW FROM FEEDBACK

**Need**: Papers documenting proper train/test split methodology for CIC-IDS datasets

**Search Terms**:
- "CIC-IDS2017 data leakage"
- "train test split intrusion detection temporal"
- "benchmark artifacts IDS evaluation"
- "proper evaluation methodology network intrusion"

**Specific Papers to Look For**:
1. **Temporal splits for CIC-IDS** - Papers using time-based rather than random splits
2. **Known issues with CIC-IDS2017** - Papers documenting benchmark artifacts
3. **Best practices for IDS evaluation** - Methodology papers

**Why**: Supervisor flagged "near-perfect accuracy" as red flag. Need to either document our controls OR acknowledge limitations and cite known issues.

**Status**: 🟠 Medium priority

---

## 17. System Architecture Diagrams for IDS (For Reference) - NEW

**Need**: Examples of well-drawn RL-IDS architecture diagrams in academic papers

**Search Terms**:
- "RL intrusion detection system architecture diagram"
- "deep reinforcement learning IDS framework"

**Why**: For generating our own figures based on established conventions.

**Status**: 🟢 Low priority - for inspiration only

---

## Updated Priority Summary

| # | Topic | Priority | Why |
|---|-------|----------|-----|
| 13 | Threat Model (STRIDE) | 🔴 URGENT | Supervisor mandate |
| 14 | RL vs Classification | 🔴 URGENT | Core validity question |
| 15 | Curriculum/Online Learning | 🟠 MEDIUM | Experiment idea + Discussion |
| 16 | Data Leakage Controls | 🟠 MEDIUM | Defend accuracy claims |
| 11 | Microsoft CyberBattleSim | 🟠 MEDIUM | Existing request |
| 12 | Continual Learning | 🟠 MEDIUM | Existing request |
| 17 | Architecture Examples | 🟢 LOW | Visual reference only |

