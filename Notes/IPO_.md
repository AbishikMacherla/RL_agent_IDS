Initial Project Overview
SOC10101 Honours Project (40 Credits)
Title of Project:
Reinforcement Learning enhanced Intrusion Detection System for Autonomous Network Defence.
Overview of Project Content
The aim of this project is to develop and evaluate proof-of-concept Intrusion Detection System
(IDS) enhanced by a Reinforcement learning (RL) agent. The concept is to integrate the RL agent
into a system that will support the IDS in analysing network traffic to detect and flag threats within
a simulated environment. This approach aims to create more adaptive defence mechanism that
reduce the reliance on static rules and manual analysis, to improve speed and accuracy of network
threat detection. Primary focus will be on using the RL methodology's effectiveness in a controlled
environment, rather than achieving a fully operating real-time system.
Milestones:
 Literature review & Resource gathering: Explore existing research papers on RL in
Cybersecurity and look into potential datasets(eg: CIC-IDS2017), RL models, and tools (e.g.,
CybORG, Mininet).
 Test Environment Setup: Build and configure a small network environment for a stable test
environment for the RL agent testing. Develop a baseline model for comparison, such as a
traditional rule-based IDS or static supervised ML(machine learning) classifier.
 RL Agent implementation & Training: Develop the core RL agent and start the initial training
process using the selected dataset.
 Integration & Fine-tuning: Integrate the agent with the test environment, conduct experiments,
and fine-tune model by reward system and adjust the parameters based on the performance.
 Evaluation & Analysis: Finally, with gathered data conduct a comparative analysis of the RL
agent’s effectiveness against the baseline. Performance will be measured using metircs such as
ROC-AUC, precision, recall, latency, and throughoutput. Results will be documented and
visualised to prove the proof of concept.
The Main Deliverables:
There are three primary deliverables in this project:
 A functional proof of concept system of an RL-enhanced IDS capable of threat classification.
 A performance dashboard to visualise key metrics such as accuracy, false positives, and detection
rates, latency, precision and recall.
 A final report detailing the project methodology, development process, findings, and final
conclusions.
B.Eng(Hons) Cybersecurity & Forensics
 1
Abishik Macherla Vijayakrishna
 40594078
The Target Audience for the Deliverable(s):
The target audience will mostly be technical and academic audience such as, Network security
researchers, Cybersecurity professionals and incident responders looking into AI implementation in
Network defence or IDS, Academics doing research or studying in related fields and organisations
developing AI/ML-enhanced products in threat detection.
The Work to be Undertaken:
The project is divided into 3 phases:
 Phase 1 (Research & Preparation): Based on the research and findings, commence with the
literature review and gather all necessary resources such as datasets, models & tools and create a
sketch/plan for the miniature test environment.
 Phase 2 (Implementation & Testing): Build the test environment, develop & train the RL agent,
and run repeated cycles of training and testing to gather results.
 Phase 3 (Analysis & Refinement): Analyse the results from testing, fine-tuning the agent’s
parameters to improve performance using the reward system, with Reinforcement Learning
Human Feedback(RLHF).
In Phases 2 & 3, the tuning and testing will be happening in repeated and analysed and gathering
results for documentation and data visualisation. In Meantime, the final report/project
documentation will be updated and documented through all the phases of the work.
Additional Information / Knowledge Required:
Topics related to Reinforcement Learning is essential, including a strong understanding of how RL
agents learn from reward systems and to perform detailed model analysis and tuning, which
involves in interpreting performance data to make informed adjustments. Furthermore, these AI-
focused concepts must be supported by a knowledge of Cybersecurity concepts- such as networking
protocols, common networking attacks, and threat detection frameworks and principles which will
be useful in the agent’s training and evaluation.
The project requires technical skills in handing the data for dashboard and displaying the large data
into the correct visualisation for better user. The network simulation to set up and manage the
environment for testing for Phase 2 & 3.
Information Sources that Provide a Context for the Project:
Google scholar, Databases from Library search like ACM digital library, IEEE Xplore, Web of
Science and ScienceDirect for academic findings and further research.
Open source datasets like CIC-IDS2017 & other CIC versions of network traffic for training and
testing agent.
Technical documentation for tools, frameworks or code used in the project.
B.Eng(Hons) Cybersecurity & Forensics
 2
Abishik Macherla Vijayakrishna
 40594078
The Importance of the Project:
Time is critical factor in responding to network-based attacks and network defence tools like IDS
play a vital role but still in standard incident response frameworks, the initial detection and
investigation phase is often the time consuming, as defenders work to confirm that an alert is not a
false positive. The longer time takes, the greater the damage to systems.
This project proposes using a RL agent to reducing this time consuming process in detection. By
training the agent on known malicious traffic, it can learn to analyse and detect attack patterns and
behaviors, raising high-confidence alerts for human investigation.
The novelty of this approach lies in the reinforcement aspect of the agent’s learning. When the
agent detects unknown traffic, it can take an action such as blocking or flagging it and then learn
from human feedback on its correctness. This reward system means the agent can continuously
improve its detection capabilities over time without needing to be fully retrained, making it an
adaptive defence. Additionally, this adaptive capability has forensic use, for instance high-
confidence alerts can trigger automated evidence preservation workflow and the system’s log can
provide patterns and other information for post-incident or further investigation.
The Key Challenge(s) to be Overcome:
The key challenges are
 Finding a high quality, latest dataset for effective model training and testing.
 RL model’s reward system will need careful fine-tuning to minimise false positives and learn
efficiently from human feedback.
 Ensuring the RL agent can analyse network traffic with minimal delay.
 Ensuring all the components- the RL agent, the network environment, and the dashboard work
together smoothly.
 Data visualisation - what to display and how to display for better understanding for the end user
in the dashboard.
 Data handling – ensuring all the experimentation will be under ethical guidelines, mostly by
using fully anonymised, publicly available datasets to protect privacy.
Key academic sources related to my Project
Emerson, H., Bates, L., Hicks, C., & Mavroudis, V. (2024). Cyborg++: An enhanced gym for the
development of autonomous cyber agents. arXiv preprint arXiv:2410.16324.
Hsu, Y.-F., & Matsuoka, M. (2020). A Deep Reinforcement Learning Approach for Anomaly
Network Intrusion Detection System. 2020 IEEE 9th International Conference on Cloud
Networking (CloudNet), 1–6. https://doi.org/10.1109/CloudNet51028.2020.9335796
Malik, M., & Singh Saini, K. (2023). Network Intrusion Detection System using Reinforcement
learning. 2023 4th International Conference for Emerging Technology, INCET 2023.
https://doi.org/10.1109/INCET57972.2023.10170630
B.Eng(Hons) Cybersecurity & Forensics
 3
Abishik Macherla Vijayakrishna
 40594078
Malik, M., & Saini, K. S. (2023). Network Intrusion Detection System Using Reinforcement
Learning Techniques. 2023 International Conference on Circuit Power and Computing
Technologies (ICCPCT), 1642–1649. https://doi.org/10.1109/ICCPCT58313.2023.10245608
Standen, M., Lucas, M., Bowman, D., Richer, T. J., Kim, J., & Marriott, D. (2021). CybORG: A
Gym for the Development of Autonomous Cyber Agents. CoRR, abs/2108.09118.
https://arxiv.org/abs/2108.09118
Suwannalai, E., & Polprasert, C. (2020). Network Intrusion Detection Systems Using Adversarial
Reinforcement Learning with Deep Q-network. 2020 18th International Conference on ICT and
Knowledge Engineering (ICT&KE), 1–7. https://doi.org/10.1109/ICTKE50349.2020.9289884
Tellache, A., Mokhtari, A., Korba, A. A., & Ghamri-Doudane, Y. (2024). Multi-agent
Reinforcement Learning-based Network Intrusion Detection System. NOMS 2024-2024 IEEE
Network Operations and Management Symposium, 1–9.
https://doi.org/10.1109/NOMS59830.2024.10575541
B.Eng(Hons) Cybersecurity & Forensics
 4