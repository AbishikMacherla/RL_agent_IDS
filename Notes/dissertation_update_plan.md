# Dissertation Update Plan (Based on Feedback)

## Summary of Feedback Points

I've organized your feedback into actionable items by section. Here's the plan:

---

## 1. INTRODUCTION UPDATES

### 1.1 Remove/Revise Cost Statistics
- **Action:** Remove specific dollar amounts from IBM breach report
- **Instead:** Focus on 4-year trend data (increasing breaches, new attack types)
- **Research Needed:** IBM reports 2021-2024 for trend data

### 1.2 Mention IDS Type
- **Action:** Clarify we're building an **Anomaly-based** IDS (not signature-based)
- **Rationale:** RL learns patterns, not signatures

### 1.3 Rephrase Section Structure
- **Action:** Review if "Section 2, 3, 4, 5" enumeration is needed or can flow naturally
- **Decision:** Keep structure but improve transitions

---

## 2. LITERATURE REVIEW UPDATES

### 2.1 Evidence for ML Failing on New Attacks
- **Action:** Find sources showing ML/signature IDS fail on zero-day attacks
- **Research Needed:** Add to sources_needed.md

### 2.2 Microsoft & Comparative Studies
- **Action:** Reference Microsoft CyberBattleSim methodology
- **Research Needed:** Microsoft RL security papers

### 2.3 Periodical Update of "Good Traffic" List
- **Action:** Research if anyone has done incremental whitelisting with RL
- **Research Needed:** Online/continual learning for IDS

---

## 3. METHODOLOGY UPDATES

### 3.1 Dataset Paragraph Sync
- **Action:** Update to reflect actual datasets used:
  - CIC-IDS2017 (Training)
  - CIC-IoT-2023 (Generalization Testing)

### 3.2 Two Scenario Design
- **Scenario 1:** Standard training (all attacks), test on held-out set
- **Scenario 2:** "Whitelist" approach - provide known-good traffic list as context

### 3.3 Heavy Penalty Experiment
- **Action:** Add experiment with FN penalty = -50 instead of -10
- **Goal:** Push RL recall even higher

---

## 4. DASHBOARD FEATURES

### 4.1 What Happens When Attack Detected?
- **Action:** In paper, explain: "Agent outputs Block action → traffic flagged/logged"
- **Dashboard displays:**
  - Real-time benign/malicious counts
  - Attack type breakdown
  - Live log of flagged packets
  - Confusion matrix updating in real-time

---

## 5. DISCUSSION UPDATES

### 5.1 Why Focus on Zero-Day (Not Just FP)?
- **Answer:** Zero-day detection demonstrates RL's *adaptability* - the core value proposition
- **Action:** Add this justification to Discussion

### 5.2 Latency vs Accuracy Trade-off
- **My Opinion:** For a dissertation proof-of-concept, **accuracy/recall matters more than latency**
- **Rationale:** We're proving RL *can* learn to detect attacks. Real-world latency optimization is future work.
- **Action:** Acknowledge latency as a limitation, mention as future work

### 5.3 Proof of Concept Validation
- **How we prove it works:**
  1. DQN achieves 97% recall (catches almost all attacks)
  2. Zero-day test shows adaptability
  3. Comparison with ML shows trade-offs
- **Action:** Add explicit "Proof of Concept Validation" subsection

---

## 6. SOURCES NEEDED

I'll update `sources_needed.md` with:
1. IBM Data Breach Reports 2021-2024 (trend data)
2. Evidence of ML/signature IDS failing on zero-day
3. Microsoft CyberBattleSim methodology papers
4. Online/continual learning for IDS
5. Studies on whitelisting approaches in IDS

---

## PRIORITY ORDER

| # | Task | Section | Complexity |
|---|------|---------|------------|
| 1 | Update Introduction (costs → trends, IDS type) | Intro | Medium |
| 2 | Add ML zero-day failure evidence | Lit Review | Research needed |
| 3 | Sync dataset paragraph | Methodology | Easy |
| 4 | Add heavy penalty experiment description | Methodology | Easy |
| 5 | Document dashboard features | Methodology | Easy |
| 6 | Add latency limitation + future work | Discussion | Easy |
| 7 | Add proof of concept validation | Discussion | Medium |
| 8 | Research whitelist/online learning | Lit Review | Research needed |

---

## NEXT STEPS

1. **You:** Confirm this plan looks good
2. **Me:** Update `sources_needed.md` with research requests
3. **You:** Provide the sources
4. **Me:** Update all dissertation sections

Ready to proceed?
