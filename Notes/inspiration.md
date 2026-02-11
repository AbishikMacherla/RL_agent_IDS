Ethan Campbell
 40483972
LLM-Driven Network Intrusion Detection
Ethan Campbell
Submitted in partial fulfilment of
the requirements of Napier University
for the Degree of
BEng (Hons) Cybersecurity & Forensics
School of Computing
November 2024
Ethan Campbell
 40483972
Authorship Declaration
I, Ethan Campbell, confirm that this dissertation and the work presented in it are my
own achievement.
Where I have consulted the published work of others this is always clearly attributed;
Where I have quoted from the work of others the source is always given. With the
exception of such quotations this dissertation is entirely my own work;
I have acknowledged all main sources of help;
If my research follows on from previous work or is part of a larger collaborative re-
search project I have made clear exactly what was done by others and what I have
contributed myself;
I have read and understand the penalties associated with Academic Misconduct.
I also confirm that I have obtained informed consent from all people I have involved
in the work in this dissertation following the School's ethical guidelines
Signed: Ethan Campbell
Date: 25/11/2024
Matriculation no: 40483972
Ethan Campbell
 40483972
General Data Protection Regulation Declaration
Under the General Data Protection Regulation (GDPR) (EU) 2016/679, the University
cannot disclose your grade to an unauthorised person. However, other students ben-
efit from studying dissertations that have their grades attached.
The University may make this dissertation, with indicative grade, available to others.
Ethan Campbell
 40483972
Abstract
This study investigates the application of Large Language Models (LLMs) to network intrusion
detection, evaluating their effectiveness in detecting both known and zero-day attacks. Multiple
LLM approaches are explored including GPT-3 and Mistral 7B, where the main LLM explored is
the RoBERTa-large model. Multiple benchmark datasets are used including TON-IoT,
CICDDoS2019, and CSE-CIC-IDS2018, an overall accuracy of 98.53% was achieved in multi-
class attack detection with a notably low false positive rate of 0.16%. In zero-day attack scenar-
ios, the LLM-based approach demonstrated superior performance compared to traditional ma-
chine learning methods in certain tests, achieving 99.67% accuracy in detecting previously un-
seen DDoS attacks versus XGBoost's 90.67%. The research also revealed limitations in general-
izing to custom network captures and highlighted challenges in processing multiple records
simultaneously. The findings suggest that LLMs can effectively complement existing Security
Operations Center (SOC) infrastructure, particularly when trained on environment-specific da-
tasets. The study concludes that while LLM-based approaches show promise for enhancing in-
trusion detection capabilities, optimal implementation requires integration with traditional se-
curity mechanisms and careful consideration of computational requirements and deployment
constraints.
Ethan Campbell
 40483972
Contents
1. Introduction ....................................................................................................................... 7
2. Background and Related Work ............................................................................................ 8
2.1. Traditional Network Security Approaches...................................................................... 8
2.2. Machine Learning in Intrusion Detection ....................................................................... 8
2.3. Large Language Models in Cybersecurity ...................................................................... 9
2.4 Zero-Day Attack Detection Approaches ......................................................................... 9
3. Data and Method .............................................................................................................. 10
3.1. Dataset Selection and Characteristics ........................................................................ 11
3.2. Data Pre-processing and Representation .................................................................... 14
3.3. Zero-Day Detection Dataset Processing Setup ............................................................ 19
4. LLM Model Selection, Setup and Training........................................................................... 21
4.1. Model Architecture Analysis ....................................................................................... 21
4.2. Model Selection and Training Methodology ................................................................. 22
4.3. Zero-Day Detection experimental setup using TON IoT dataset .................................... 24
4.4. Further Zero-Day testing using CICDDoS2019 Dataset ................................................ 25
5. Experimental Results and Analysis .................................................................................... 26
5.1. Primary Experiment Using TON-IoT Dataset ................................................................. 27
5.2. Other Validation Results Based on TON-IoT Experiment ............................................... 29
5.2.1. CICDDoS2019 results .......................................................................................... 29
5.2.2. CSE-CIC-IDS2018 results .................................................................................... 32
5.2.3. NSL-KDD replication study results........................................................................ 34
5.2.4. Multi-Source Data results .................................................................................... 36
5.3. Zero-Day Detection Performance ............................................................................... 37
5.4. LLM Model Comparison ............................................................................................. 39
6. Discussion ....................................................................................................................... 40
6.1. Implications for SOC Environments ............................................................................ 40
6.2. Main Findings, Limitations and Challenges.................................................................. 41
6.4. Future Research Directions ........................................................................................ 43
7. Conclusion ...................................................................................................................... 43
References .......................................................................................................................... 44
Appendix ............................................................................................................................. 47
Ethan Campbell
 40483972
List of Tables
Table 1, showing how many train/validation/test records were used for binary model test (text generation
tests and start of text classification tests) ........................................................................................... 15
Table 2, showing how many train/validation/test records were taken for each label for the 6 types text-
classification model ......................................................................................................................... 16
Table 3, number of records available per label from dataset ................................................................ 16
Table 4. distribution of attack types in the "processed network" datasets and their train/validation/test
splits ............................................................................................................................................... 17
Table 5, distribution of attack types in the "processed network" datasets and their train/validation/test
splits ............................................................................................................................................... 25
Table 6, performance metrics for different traffic types ........................................................................ 27
Table 7, comparison of different machine learning approaches ........................................................... 29
Table 8, performance metrics for CICDDoS2019 multi-class classification ........................................... 32
Table 9, performance metrics for CSE-CIC-IDS2018 multi-class classification ..................................... 32
Table 10, models where ddos records were excluded in training: 99.67% correct classification of 3000
ddos records .................................................................................................................................... 38
Table 11, models where password records were excluded in training: 100% correct classification of 3000
password records ............................................................................................................................. 38
Table 12, models where injection records were excluded in training: 100% correct classification of 3000
injection records .............................................................................................................................. 38
Table 13, models where scanning records were excluded in training: 32.87% correct classification of 3000
scanning records .............................................................................................................................. 38
Table 14, models where udp-lag records were excluded in training: 99.97% correct classification of 3000
udp-lag records ................................................................................................................................ 39
Table 15, models where syn records were excluded in training: 100% correct classification of 3000 syn
records ............................................................................................................................................ 39
Table 16, comparison of all fine-tuned model attempts throughout paper ............................................ 40
List of Figures
Figure 1, Methodology Pipeline .......................................................................................................... 11
Figure 2, Common preprocessing steps ............................................................................................. 14
Figure 3, Standard and zero-day data setup and training ...................................................................... 20
Figure 4, TON IoT standard results confusion matrix ............................................................................ 28
Figure 5, Mistral 7B malicious prediction example............................................................................... 30
Figure 6, Mistral 7B benign prediction example ................................................................................... 30
Figure 7, Mistral 7B multiple record prediction example ...................................................................... 31
Figure 8, GPT-3 multiple record prediction example ............................................................................ 31
Figure 9, CSE-CIC-IDS2018 results using newly-generated benign data ................................................ 33
Figure 10, CSE-CIC-IDS2018 results using original dataset only .......................................................... 34
Figure 11, results achieved from paper replication .............................................................................. 35
Figure 12, Multi-source data results confusion matrix.......................................................................... 36
Figure 13, Multi-source data results for individual captures ................................................................. 37
Ethan Campbell
 40483972
1. Introduction
Intrusion detection has long been a challenge for cybersecurity experts and as cyber threats
continue to evolve in scale and sophistication, the need for robust and adaptive Intrusion
Detection Systems (IDS) continues. Traditional IDS approaches, including signature-based and
anomaly-based detection methods, have shown limitations in identifying novel attacks and
adapting to rapidly changing threat landscapes [1]. In response to these challenges, recent
years have shown a rise of machine learning and artificial intelligence in cybersecurity, offering
promising results in enhancing intrusion detection capabilities. However, these solutions are
not without their own limitations. They often struggle with high false positive rates, require
extensive training data, and may lack the contextual understanding necessary to discern
sophisticated attack patterns. Other ML challenges include concept drift, adversarial settings,
and data confidentiality issues [2], which hinder the long-term reliability of current systems.
This has led researchers to explore more advanced AI techniques to address these persistent
challenges in network security.
Large Language Models (LLMs) and Generative AI have demonstrated remarkable success in
solving complex problems across various domains, including cybersecurity. Their ability to
understand context, recognize patterns, and generate human-like text offers new possibilities
for enhancing intrusion detection capabilities [3]. This research investigates the application of
LLMs to the intrusion detection problem, exploring both the potential benefits and the
challenges associated with this novel approach.
The motivation behind this study stems from the need to address the limitations of current IDS
technologies and leverage the power of advanced language models in the cybersecurity
domain. By exploring LLM-based approaches, the aim is to enhance the accuracy, adaptability,
and contextual understanding of intrusion detection systems. This research seeks to bridge the
gap between traditional machine learning methods and the latest advancements in natural
language processing (NLP), offering new perspectives and potential network security
techniques.
To guide the investigation, the following research questions were formulated:
1)2)3)How do LLM-based approaches compare to traditional machine learning methods in
network intrusion detection?
To what accuracy can LLMs fine-tuned on specific network traffic datasets detect attack
records that they have not been trained on?
How would our proposed system detect zero-day vulnerabilities?
The remainder of this paper is organized as follows. Section 2 presents background information
related to traditional network security approaches, the application of machine learning in
intrusion detection, and the emerging role of Large Language Models in cybersecurity. Section 3
describes our experimental setup and methodology, including dataset selection, data
preprocessing techniques. Section 4 shows the process for why a model was selected and its
training process. In Section 5, the results and analysis are presented, focusing on performance
comparisons between LLM-based approaches and traditional methods, the zero-day detection
results, and the challenges encountered in generalizing these models to new environments.
Section 6 discusses the implications of the findings for SOC environments, addresses
limitations and challenges, and outlines directions for future research. Finally, Section 7
concludes the paper by summarizing the key findings and their significance for the field of
network intrusion detection.
Ethan Campbell
 40483972
2. Background and Related Work
2.1. Traditional Network Security Approaches
Network security has evolved significantly, with various approaches developed to detect and
prevent intrusions. Wang et al. [4] proposed a method combining signature analysis, entropy
protocol analysis, and machine learning for comprehensive traffic behaviour analysis. This
approach not only detects attacks but also identifies their types, enabling more targeted
defence strategies.
An analysis of both host-based and network-based intrusion detection systems was provided by
Singh et al. [5], comparing their strengths and applications in comprehensive network security
strategies. As networks have grown more complex, the distinction between these systems has
become increasingly relevant.
Signature-based intrusion detection systems (IDS) have been a cornerstone, relying on
predefined rules to identify known malicious behaviour [6]. Complementing this, anomaly-
based detection aims to identify deviations from normal network behaviour, potentially catching
novel attacks [7].
Heuristic-based detection techniques have gained prominence as an adaptable approach to
threat identification [8]. These techniques analyse the behaviour of files, systems, and network
activities to detect anomalies indicative of potential threats. The integration of heuristic based
detection with artificial intelligence and machine learning has further enhanced detection
capabilities.
2.2. Machine Learning in Intrusion Detection
Machine learning techniques have significantly advanced intrusion detection systems,
addressing limitations of traditional approaches. Zhang et al. [9] demonstrated that ensemble
methods, particularly random forests, have shown superior performance in feature selection
and parameter optimization for intrusion detection. The success of these methods highlights
the importance of combining multiple models to improve overall detection accuracy and
robustness.
Unsupervised learning methods, such as Autoencoders and Restricted Boltzmann Machines
combined with clustering, have shown promise in identifying novel attack types without relying
on predefined signatures [10]. This approach is particularly relevant to the challenges faced in
detecting evolving threats.
The adaptation of complex neural network architectures, including CNNs and hybrid models
like CNNLSTM, has demonstrated improved accuracy in intrusion detection [11]. These models’
ability to process timeseries data effectively aligns with the temporal nature of network traffic
patterns, a crucial aspect in identifying sophisticated attack sequences.
Recent comprehensive reviews [12], [13] emphasize the diversity of machine learning
approaches in IDS, underlining the importance of selecting appropriate data sources and
algorithms based on specific attack characteristics. To address the common issue of
imbalanced datasets in cybersecurity, innovative techniques like Generative Adversarial
Networks (GANs) have been employed to generate synthetic data for underrepresented attack
types [14]. This approach enhances the detection of rare but potentially critical security
breaches, a persistent challenge in network security. However, fundamental challenges persist,
including the continuous evolution of systems, the presence of adversaries, and data privacy
Ethan Campbell
 40483972
concerns [2]. These issues underscore the need for adaptive and privacy-preserving machine
learning approaches in cybersecurity.
2.3. Large Language Models in Cybersecurity
The application of Large Language Models (LLMs) in cybersecurity represents a significant
evolution from traditional and machine learning-based approaches. Aghaei et al. [15] developed
SecureBERT, a domain-specific language model trained on extensive cybersecurity resources.
Its success in tasks like named entity recognition highlights the potential of LLMs in
understanding complex cybersecurity language.
The integration of LLMs with other AI techniques has shown promising results. Ali et al. [16]
developed HuntGPT, which combines a Random Forest classifier with explainable AI
frameworks and a GPT-3.5 Turbo agent, demonstrating the potential for more interpretable and
user-friendly intrusion detection systems.
In intrusion detection, BERT-based models have shown promise by transforming network data
into a natural language format, enabling more nuanced feature extraction and anomaly
detection [17]. This approach has outperformed conventional machine learning methods in
accuracy and detection rates, particularly when tested on standard datasets like NSL-KDD.
LLMs, with their transformer architectures, have demonstrated potential in processing and
comprehending large volumes of network log data, offering autonomous learning and
adaptation to evolving network behaviours [18], [19]. Recent work by researchers has shown
particular promise in applying LLMs to intrusion detection systems through fine-tuning
approaches on network security datasets. A notable study [19] demonstrated the effectiveness
of LLM-based IDS using the NSL-KDD dataset, achieving superior anomaly detection accuracy
compared to traditional deep learning approaches such as DNNs, CNNs, and Autoencoders.
Their methodology emphasized the LLM's ability to learn patterns with limited data availability,
addressing a common challenge in traditional ML-based IDS where imbalanced datasets often
hinder effective pattern recognition.
However, several gaps remain in the current research landscape. While studies like [19]
demonstrate promising results, questions persist about the reproducibility of their findings and
the scalability of their approaches. The challenges of model interpretability, computational
resource requirements, and potential overfitting issues in LLM-based IDS implementations
remain areas requiring further investigation. These limitations, coupled with the rapid evolution
of network threats, suggest the need for additional research into hybrid approaches that
combine the pattern recognition capabilities of LLMs with traditional IDS methodologies.
The application of LLMs in operational cybersecurity environments remains an emerging field.
Many studies are preliminary, and further research is needed to validate their effectiveness in
real-world scenarios. This background sets the stage for exploring how LLMs can be effectively
integrated into existing intrusion detection frameworks, potentially addressing the limitations of
both traditional and machine learning-based approaches.
2.4 Zero-Day Attack Detection Approaches
Zero-day attack detection remains one of the most challenging aspects of network security, as
these attacks exploit unknown vulnerabilities before defensive measures can be implemented.
Traditional signature-based detection methods are inherently unable to identify such previously
unknown threats, leading researchers to explore more sophisticated approaches.
Ethan Campbell
 40483972
A zero-day detection methodology was proposed by Salehi et al. [20], introducing the MAAR
(Malware Analysis using API and Arguments Runtime) approach. Their research demonstrated
that analysing behavioural patterns, specifically API calls along with their arguments and return
values, could achieve 99.4% accuracy in detecting previously unseen malicious behaviour. The
robustness of this approach was validated through testing against new malware families not
present in the training data, maintaining 96.3% accuracy and establishing the importance of
focusing on behavioural patterns rather than specific signatures.
Recent research has highlighted the emergence of deep learning techniques as particularly
promising for zero-day attack detection. The adaptable deep learning framework proposed by
[21] introduced an innovative approach combining open set recognition with clustering
optimization. This method achieved accuracy above 99% for most attack types while
maintaining the ability to identify and categorize previously unseen attacks. Their work
demonstrated that combining deep learning models with clustering algorithms can effectively
adapt to traffic concept drifts while maintaining high detection accuracy.
Transfer learning has emerged as another approach for zero-day attack detection. As
demonstrated by [22], deep transductive transfer learning can effectively detect zero-day
attacks by leveraging knowledge from known attack patterns to identify novel threats. This
approach is particularly valuable when labelled data for new attack types is unavailable, as it
can transfer knowledge from existing attack patterns to detect previously unseen variants. The
research showed promising results even when source and target domains had different feature
spaces and probability distributions.
More recently, federated learning has been explored as a solution for zero-day attack detection
in IoT environments. Research by [23] proposed a novel federated learning-enabled anomaly-
based IDS that addresses privacy concerns while enabling collaborative model training across
multiple networks. This approach is particularly relevant for IoT networks, where device
heterogeneity and resource limitations make traditional detection methods challenging to
implement.
These developments in zero-day attack detection demonstrate a clear evolution from traditional
signature-based approaches toward more sophisticated methods that can adapt to and identify
previously unknown threats. However, challenges remain, particularly in reducing false positive
rates and managing the computational resources required for these advanced detection
methods.
3. Data and Method
This section outlines the systematic approach used for applying Large Language Models (LLMs)
to network intrusion detection. Figure 1 presents our methodology pipeline, which consists of
three primary phases: data collection, preprocessing, and model selection/training. The
pipeline was designed to ensure consistent handling of multiple datasets while maintaining
data integrity and promoting model generalization.
The data collection phase incorporated several established datasets: CICDDoS2019, CSE-IDS-
2018, NSL-KDD, and TON-IoT, as well as targeted samples from IoT23 and CTU-13. Each dataset
Ethan Campbell
 40483972
was selected to address specific aspects of the research objectives, particularly regarding zero-
day attack detection and model generalization capabilities.
Figure 1, Methodology Pipeline
The preprocessing phase, detailed in the central portion of Figure 1, implemented a
standardized approach across all datasets. This included the removal of temporal data to
prevent temporal bias, anonymization of IP addresses to ensure privacy and prevent overfitting
to specific network configurations, standardization to a consistent CSV format with text and
label columns, and numerical encoding of labels for efficient model processing.
The final phase of model selection and training described in section 4, utilized the pre-
processed datasets to effectively evaluated various LLM architectures, ultimately focusing on
RoBERTa-large for its superior performance in preliminary testing. The following subsections
detail each component of this pipeline, beginning with the dataset selection criteria and
characteristics.
3.1. Dataset Selection and Characteristics
The datasets selected for this study were chosen to systematically evaluate the LLM's
capabilities across different aspects of network intrusion detection. The combination of
CICDDoS2019 [24], CSE-IDS-2018 [25], NSL-KDD [26], TON-IoT [27], [28], [29], [30], [31], [32],
[33], [34], and samples from IoT23 [35] and CTU-13 [36] provides a comprehensive testing
ground for both the model's general classification abilities and its potential for zero-day attack
detection. These datasets represent different generations of network security research, ranging
from the established NSL-KDD benchmark to modern IoT-focused datasets, allowing us to
assess the model's adaptability across different types of evolving network environments.
The selection criteria prioritized datasets that offer distinct advantages for the research goals:
CICDDoS2019 provides focused DDoS attack scenarios ideal for initial model validation as it is
generally an easier test to detect DDoS packets/patterns; CSE-IDS-2018 offers broader attack
varieties for testing generalization; TON-IoT presents modern IoT-specific attack patterns crucial
for contemporary relevance; and the NSL-KDD dataset enables direct comparison with previous
research. The inclusion of IoT23 and CTU-13 samples, along with real-world network captures,
helps validate the model's performance against different data sources and collection
methodologies.
Ethan Campbell
 40483972
Importantly, these datasets collectively enable a graduated testing approach - from basic binary
classification to multi-class detection and ultimately to zero-day attack scenarios - while
maintaining sufficient overlap in their fundamental network characteristics to allow meaningful
cross-dataset validation of the results.
CICDDoS2019
This dataset was selected as the initial testing ground for the LLM-based approach due to its
focused scope and well-structured attack scenarios. By starting with this dataset, the
methodology could be validated using clearly defined DDoS attack patterns before progressing
to more complex classification tasks.
The dataset contains benign network data and the most up-to-date common DDoS attacks,
which resembles true real-world data (PCAPs). It also includes the results of network traffic
analysis using CICFlowMeter-V3 with labelled flows based on the time stamp, source, and
destination IPs, source and destination ports, protocols and attack (CSV files).
Out of the 13 types of DDoS attacks collected in the CICDDoS2019 dataset, for the purposes of
testing an LLM models' classification abilities I first only included benign data and one DDoS
attack type (UDP-Lag), then in later tests to see if the model could deal with many possibilities,
it was trained on five DDoS types, UDP-lag, Syn, NetBIOS, MSSQL and LDAP. This gradual
increase in classification complexity helped establish the fundamental capabilities of the
approach before moving on to more diverse attack scenarios in other datasets.
CSE-CIC-IDS2018
Following initial experimentation with CICDDoS2019's focused DDoS attack scenarios, the
CSE-CIC-IDS2018 dataset was selected to evaluate the model's capability to handle a broader
spectrum of attack types. This progression from CICDDoS2019 to CSE-CIC-IDS2018's more
comprehensive attack scenarios was crucial for testing the model's ability to generalize across
different types of malicious network behaviour, better reflecting real-world security
environments where threats are rarely limited to a single attack category and helping assess the
LLM-based approach's practical applicability.
This dataset utilizes profiles to systematically generate network traffic data, encompassing both
normal user traffic and attack scenarios. The normal traffic patterns span multiple common
protocols including HTTPS, HTTP, SMTP, POP3, IMAP, SSH, and FTP, providing a robust baseline
for benign network behaviour. The attack scenarios are particularly valuable for research
purposes as they include six distinct categories: internal network infiltration, HTTP denial of
service, web application attacks, brute force attempts, and exploits of recent vulnerabilities like
Heartbleed.
NSL-KDD (for paper replication)
The NSL-KDD dataset was specifically selected to validate and extend upon recent research in
LLM-based intrusion detection systems. This dataset's inclusion was motivated by a recent
study [19] that demonstrated promising results using BERT-based models for network security.
By incorporating the NSL-KDD dataset, the aim is to both reproduce these findings and address
questions about the reproducibility and scalability of LLM approaches in intrusion detection.
The dataset is particularly valuable for research as it provides a refined version of the KDD Cup
99 dataset, having addressed various statistical anomalies and redundancy issues [28]. It
comprises approximately 148,517 records (125,973 training, 22,544 testing), with each record
Ethan Campbell
 40483972
containing 41 features describing different aspects of network connections [29]. The data
encompasses various attack types, including Denial of Service (DoS), User to Root (U2R),
Remote to Local (R2L), and Probing attacks, providing a comprehensive range of threat
scenarios for model evaluation.
A key characteristic that made this dataset suitable for the study was its inclusion of novel
attack types in the test set that are not present in the training data. This feature naturally aligns
with the research goals of evaluating LLM effectiveness in detecting previously unseen attack
patterns. Furthermore, the dataset's established position in the research community provides a
solid benchmark for comparing the approach against both traditional machine learning
methods and newer LLM-based solutions.
ToN IoT
The TON IoT dataset was selected as the primary dataset for several key reasons that align with
the research objectives. Created in 2020, its recency ensures any findings remain relevant to
current cybersecurity challenges, particularly in modern IoT-enabled networks where traditional
intrusion detection methods often struggle.
The dataset's architecture proved ideal for a multi-faceted evaluation approach. Its diverse
attack categories - ranging from DDoS and data injection to scanning activities - provided the
variety needed to test the hypothesis that LLMs can effectively generalize across different attack
patterns. This diversity was particularly valuable for zero-day attack detection experiments,
where specific attack types could be systematically excluded during training while maintaining
a robust set of other attack patterns for the model to learn from.
Furthermore, the dataset's extensive use in traditional machine learning studies provided a
valuable baseline for comparing the LLM-based approach against established methodologies.
Its structured organization, with well-documented attack patterns and clear labelling, also
facilitated the experimental design, particularly for the zero-day attack simulation scenarios
where precise control is needed over which attack patterns were exposed to the model during
training.
Multi-Source Data Test and Real-World Data Collection
While structured datasets like TON-IoT and CICDDoS2019 provide excellent foundations for
model development, incorporating diverse PCAP sources and real-world network traffic was
crucial for validating the LLM's practical applicability. This approach was considered essential
for testing the model's generalization capabilities across different network environments and
exploring the potential for environment-specific training to enhance detection accuracy.
The primary datasets used were IoT23 [35] and CTU-13 [36], which are well-established in the
field of network security research. From the IoT23 dataset, specifically the "only5000" pcap
sample was used, which provides a focused subset of IoT network traffic. The CTU-13 dataset
contributed seven different botnet captures, offering a variety of malicious network behaviours
for the model to learn from.
To enhance the diversity and realism of the dataset, 19 additional pcap files were incorporated
from the Malware Traffic Analysis website [37]. These files represent various network scenarios
and attack types, further enriching training data.
For the benign network traffic source, records from a home network environment were
collected. This addition provides the model with examples of typical, non-malicious network
Ethan Campbell
 40483972
activities and aims to show that using training data based on a specific network will increase
accuracy of the model. This combination of varied data sources enables us to evaluate not just
the model's classification accuracy, but also its adaptability to different network contexts and
its potential for practical deployment in real-world security operations centers (SOCs).
3.2. Data Pre-processing and Representation
Data preprocessing is an essential step in preparing network traffic data for LLM training and
analysis, and while there are variations on how this data needs to be processed for the different
datasets, there are many commonalities needed to both be compatible with LLM fine-tuning
and enhance the model's pattern recognition capabilities. The primary goals of preprocessing
are to enhance the model's ability to identify patterns, reduce noise, and ensure data privacy. By
standardizing the format and content of the data, the aim is to improve the model's learning
efficiency and generalization capabilities while addressing potential confidentiality concerns.
Common preprocessing steps
In the context of network intrusion detection, preprocessing serves several key purposes, as
illustrated in Figure 2. First, as shown in the Initial Processing stage, it helps to remove or
obfuscate sensitive information such as IP addresses, which not only protects privacy but also
encourages the model to focus on underlying patterns rather than specific network identifiers.
Second, through the Data Transformation stage, it transforms raw network data into a format
that is compatible with LLM input requirements, typically involving the creation of structured
Figure 2, Common preprocessing steps
Ethan Campbell
 40483972
text representations. Finally, preprocessing can help to balance the dataset, ensuring that the
model is exposed to a representative sample of both normal and anomalous network
behaviours.
The aim was to create a dataset that would enable the LLM to learn robust, generalizable
patterns of network behaviour while mitigating potential issues related to data confidentiality
and model bias. For this reason, all data formatting followed these prerequisites; any specific
contextual data would not be included such as IP address or timestamps, the IP being replaced
with only “local” or “foreign”; all dataset need to be in a csv format with two columns, “text” and
“label” as these are a requirement for finetuning the models selected; the "label" column is
populated with integer values rather than string descriptions. This encoding allows for more
efficient processing by the model and enables straightforward multi-class classification.
Dataset-specific considerations for CICDDoS2019
In the original dataset there are around 88 attributes for each network record, of which only the
relative timestamp attribute was removed.
In the initial attempts of the training process, the idea of using text generation models was
explored. The dataset was processed to generate a dataset that seemed suitable for this type of
model, where the name of each column in the original dataset was taken and formatted to
describe the number/attribute it represented in one human readable sentence for each record.
A shortened example of which can be seen:
Example record of CIC-DDoS2019 for text generation models:
The sender is a foreign ip address from port 58445. The destination is the local ip address on
port 4463. Protocol number 17. [... records generally have 1500-3000 characters, ~700 tokens
...] Forward average bytes per bulk 0 bytes. This record is labelled as being Malicious – UDP-Lag.
It was decided to finetune two different models, Mistral's 7B Instruct v0.2 and GPT-3, both of
which needed datasets processed slightly differently. For Mistral's 7B Instruct v0.2, only one
column is needed named “description” where each column was one whole record like seen in
the example. For GPT-3, a “system content” was needed where I included the majority of a
record for each, then an “assistant reply” was needed in a JSON format where I included the
label (e.g. “This record is labelled as being Benign”).
For attempts using the text generation models, each record would always end in either "This
record is labelled as being Benign" or "This record is labelled as being Malicious – UDP-Lag"
which acted as each record's malicious or benign label. While the CICDDoS2019 dataset
contained various DDoS attack types, initial experiments focused on binary classification using
only benign and UDP-Lag records. Table 1 shows the distribution of records used in these initial
binary classification tests, where 6,000 records of each type were used for training and
validation, with 12,000 records reserved for testing.
Table 1, showing how many train/validation/test records were used for binary model test (text generation
tests and start of text classification tests)
Attack Type
 Total Count
 Train
 Validation
 Test
Benign
 49,554
 3000
 3000
 6000
UDP-Lag
 366,461
 3000
 3000
 6000
Ethan Campbell
 40483972
In later experiments, the focus shifted to using the RoBERTa-large text classification model,
which allowed for both binary and multi-class classification tasks. The dataset format was
refined to focus purely on pattern recognition rather than natural language understanding. This
involved representing records as space-separated numerical values from the original attributes,
without any descriptive text about what these values represented. A more comprehensive set of
attack types was also incorporated, as shown in Table 2. This refined approach to data
formatting, focusing on numerical patterns rather than natural language descriptions, was
adopted for all subsequent fine-tuning experiments across different datasets.
Table 2, showing how many train/validation/test records were taken for each label for the 6 types text-
classification model
Attack Type
 Total Count
 Train
 Validation
 Test
Benign
 49,554
 1500
 1500
 3000
UDP-Lag
 366,461
 1500
 1500
 3000
DrDoS_LDAP
 2,179,930
 1500
 1500
 3000
DrDoS_MSSQL
 4,522,492
 1500
 1500
 3000
DrDoS_NetBIOS
 4,093,279
 1500
 1500
 3000
Syn
 4,284,751
 1500
 1500
 3000
Dataset-specific considerations for CSE-CIC-IDS2018
The dataset used the same tool as the CIC-DDoS2019 dataset to process the packet capture
files, CICFlowmeter-V3, so the initial data pre-processing steps remained consistent with those
previously described. The CSE-CIC-IDS2018 dataset presented additional preprocessing
considerations due to its broader attack diversity, including infiltration attacks, brute force
attempts, web attacks, and DoS/DDoS scenarios.
From the original dataset containing 80 network traffic attributes, the same preprocessing
approach was maintained, where temporal data and contextual identifiers are removed. The
dataset was refined to focus on four primary attack categories for the experiments: brute force
attacks, web-based attacks, DoS/DDoS attacks, and infiltration attempts, the total sample
count and specific name of each of these labels can be seen in Table 3. This categorization
helped maintain consistency with the broader experimental goals while providing sufficient
diversity in attack patterns.
Table 3, number of records available per label from dataset
Attack Type
 Total Count
 Train
 Validation
 Test
Benign
 1,663,703
 1500
 1500
 3000
SSH-Bruteforce
 187,589
 1500
 1500
 3000
DoS attacks-GoldenEye
 41,508
 1500
 1500
 3000
DoS attacks-Slowloris
 10,990
 1500
 1500
 3000
To validate the approach and address potential dataset-specific biases, additional test data was
generated following the methodology outlined in the CSE-CIC-IDS2018 dataset documentation.
The generated data was processed using CICFlowmeter-V3 to maintain consistency with the
original dataset's feature extraction methodology. This approach allowed us to validate the
model's performance against both the standardized dataset and newly generated, controlled
Ethan Campbell
 40483972
test cases while maintaining the same preprocessing pipeline and data representation format
used with the CIC-DDoS2019 dataset.
Dataset-specific considerations for TON IoT
The csv files used from the dataset were specifically from the “Processed Network dataset”
section. A detailed record count encompassing all csv files from the section can be seen in
Table 4.
Table 4. distribution of attack types in the "processed network" datasets and their train/validation/test
splits
Attack Type
 Total Count
 Train
 Validation
 Test
DOS
 3,375,328
 1500
 1500
 3000
Normal
 796,380
 1500
 1500
 3000
Scanning
 7,140,161
 1500
 1500
 3000
Injection
 452,659
 1500
 1500
 3000
DDoS
 6,165,008
 1500
 1500
 3000
Password
 1,718,568
 1500
 1500
 3000
XSS
 2,108,944
 1500
 1500
 3000
Backdoor
 508,116
 1500
 1500
 3000
Ransomware
 72,805
 1500
 1500
 3000
MITM
 1,052
 263
 263
 526
Originally the csv files contained 46 attributes, and following the principle of not including
contextual or redundant data, the dataset was refined to 16 key features most relevant for
pattern-based LLM training: src ip, src port, dst ip, dst port, proto, service, duration, src bytes,
dst bytes, conn state, missed bytes, src pkts, src ip bytes, dst pkts, dst ip bytes, and type.
Two datasets were created for training and validation, each comprising 15,000 records (1500
records per attack type, randomly selected across all 10 unique labels seen in the dataset). For
the dataset used to test the model, a further 3000 records per label were selected. It was
ensured that there were no duplicate records selected during this phase.
To prepare for the RoBERTa-large text-classification model, labels were encoded numerically:
normal network data (0, “normal”), DDoS (1, “ddos”), data injection attempts (2, “injection”),
password brute-force attempts (3, “password”), information gathering port scans (4,
“scanning”), denial of service (5, “dos”), ransomware attacks (6, “ransomware”), backdoor
intrusions (7, “backdoor”), cross-site scripting (8, “xss”), and man-in-the-middle attacks (9,
“mitm”).
Dataset-specific considerations for NSL-KDD dataset (for paper replication)
To prepare the data for the BERT-based model, log creation was implemented following the
procedure described in the original paper [19], concatenating selected features to create
synthetic log entries. Feature selection was based on correlation matrices and statistical
analysis to identify the most informative attributes for attack detection.
While attempting to faithfully replicate the experimental conditions of the original study, several
notable discrepancies were encountered in the dataset composition. The original paper
reported working with 23 distinct attack classes in their training data, with only 12 of these
classes appearing in their test dataset. However, the analysis revealed a more complex
Ethan Campbell
 40483972
distribution: the training dataset contained 22 unique attack labels, while the test dataset
contained 38 distinct attack types, with 22 labels being shared between both sets.
This divergence from the reported numbers suggests potential differences in how attack types
were categorized or aggregated in the original study. Some attack categories in the analysis
contained very sparse data, with certain types having only 1-10 records. This distribution pattern
could explain why the original paper might have chosen to focus on a subset of the attack types,
though this reasoning wasn't explicitly stated in their methodology.
For the recreated preprocessing pipeline, it was chosen to work with all 22 shared attack types
between the training and test sets to maintain data consistency and provide a more
comprehensive evaluation. This decision, while differing from the original paper's approach,
allows for a more thorough assessment of the model's classification capabilities across a
broader range of attack patterns.
These preprocessing choices impact how recreated results were interpreted and compared to
the original study, particularly regarding the model's performance across different attack
categories. The discrepancy in label counts and distribution highlights the importance of
transparent reporting of dataset characteristics in replication studies.
Dataset-specific considerations for Multi-Source Data and Real-World Data Collection
As there were many different datasets and other sources of pcap files used in this test, the data
pre-processing pipeline was designed to standardize and prepare the pcap files for use in
training the model. The process involved several key steps:
Filtering: For datasets that were not pre-filtered (excluding IoT23, which was already processed),
filters to isolate traffic associated with known malicious IP addresses. This step ensured that
the model focused on learning patterns specific to malicious activities.
ZEEK Processing: All pcap files were processed using the ZEEK network security monitor. This
step converted the raw packet captures into structured logs, providing a rich set of features for
the model to analyse. The processed data was then converted to the CSV format needed for the
models training process.
As previously described in the data formatting prerequisites, all specific contextual data was
removed such as IP addresses and timestamps, ensuring the model learns from general
network behaviour patterns rather than specific identifiers.
Labelling: Each record in the final dataset was labelled as either benign (0) or malicious (1),
providing clear classification targets for a supervised learning approach.
This pre-processing pipeline ensured that the diverse set of input data was transformed into a
consistent, anonymized, and labelled dataset suitable for training and evaluating the model.
For the collection of benign network traffic, a home network environment was utilized consisting
of typical consumer devices including computers, smartphones, and IoT devices. Traffic was
captured over a 48-hour period during normal network usage, ensuring a representative sample
of legitimate network behavior. This capture included common protocols such as HTTP/HTTPS,
DNS, DHCP, and various streaming services, providing a diverse dataset of benign traffic
patterns.
To generate test data for malicious traffic classification, a controlled testing was established
environment within the same home network. This setup included:
Ethan Campbell
 40483972
•
•
•
A dedicated attacking machine running Kali Linux
A target machine running common services (HTTP, FTP, SSH)
Network segregation to prevent unintended impact on other devices
Several common attack scenarios were executed against the target machine:
•
•
•
•
•
TCP SYN flood attacks using hping3
UDP flood attacks targeting various services
DNS amplification attacks
Port scanning activities using nmap
Basic brute force attempts against SSH
Each attack was conducted in isolation, with precise timing recorded to ensure accurate
labeling of the captured traffic. The packet captures were collected using tcpdump and
processed through the same pipeline described above. This approach provided us with ground
truth data where classification of both benign and malicious traffic was a certainty, as there was
complete control over when and how the attacks were executed and who executed them.
This methodology of using real network traffic rather than simulated data for both benign and
attack scenarios provide several advantages:
1.2.3.It captures the natural variations and inconsistencies present in actual network
environments
It includes the complex interactions between different network protocols and services
It represents realistic attack patterns as they would appear in production environments
The resulting dataset comprised 3426 records of benign traffic and 3427 records of malicious
traffic, that will be split into relevant parts for training, validation and testing (25%, 25%, 50%
respectively).
3.3. Zero-Day Detection Dataset Processing Setup
The preparation of datasets for zero-day attack detection required a specialized approach to
data processing and organization, building upon the preprocessing methodology outlined in
Section 3.2. This process was designed to evaluate the model's ability to detect previously
unseen attack patterns while maintaining consistency with the overall experimental framework.
For the TON IoT dataset, five distinct traffic types were selected: normal network data, DDoS
attacks, data injection attempts, password brute-force attempts, and information gathering port
scans. These categories were chosen to represent a diverse range of attack patterns while
maintaining sufficient sample sizes for robust training and testing. The data was processed
following the standard preprocessing steps described earlier, including the removal of temporal
data and IP address anonymization.
The zero-day simulation datasets were structured as follows:
Training/Validation Datasets:
•
•
•
•
Binary classification format (0 for normal, 1 for malicious)
Four attack types included in each training set
One attack type systematically excluded to simulate zero-day conditions
12,000 total records (3,000 normal, 9,000 malicious)
Ethan Campbell
 40483972
Testing Datasets:
•
•
•
•
Included all five traffic types
15,000 total records (3,000 per type)
Maintained consistent format with training data
Included the excluded "zero-day" attack type
As shown in Figure 3, this process created two parallel dataset paths: one for standard multi-
class classification and another specifically for zero-day attack simulation. Both paths
converged at the dataset split stage, where they were divided into training, validation, and test
sets.
Figure 3, Standard and zero-day data setup and training
Ethan Campbell
 40483972
To validate this approach, the same methodology was applied the to the CICDDoS2019 dataset,
focusing on specific DDoS attack types. This secondary validation helped ensure the
robustness of the zero-day detection approach across different attack scenarios and network
environments.
The processed datasets were then used for model training and evaluation as detailed in Section
4, with particular attention paid to the model's performance in detecting the simulated zero-day
attack patterns.
4. LLM Model Selection, Setup and Training
The selection of an appropriate LLM for network intrusion detection using the TON IoT dataset
was driven by the goal of leveraging language understanding capabilities in a non-traditional
domain. While LLMs are primarily designed for natural language tasks, their ability to recognize
complex patterns and generalize to new contexts makes them intriguing candidates for
cybersecurity applications. Among the various types of LLMs, those focused on Natural
Language Processing (NLP) tasks were identified as most relevant for this study. Specifically,
models capable of Text Classification and Text Generation were considered, as these
capabilities could potentially be adapted to categorize network traffic data when appropriately
formatted. The choice between these two approaches - Text Classification vs. Text Generation
Models - was crucial in determining how to best leverage the pattern recognition capabilities of
LLMs for the novel task of network intrusion detection.
4.1. Model Architecture Analysis
Large Language Models can be categorized into many architectural types based on their primary
functions and design approaches ranging from specialized Computer Vision models, NLP
models or Multimodal models. Base models, such as GPT (Generative Pre-trained Transformer)
and BERT (Bidirectional Encoder Representations from Transformers), serve as foundations for
more specialized implementations. These base architectures can be adapted into instruction-
tuned models, which are trained to follow specific directives, and domain-adapted models,
which are fine-tuned for fields like cybersecurity.
For the network intrusion detection study, the focus was primarily on two main model types:
text classification models and text generation models. Classification models, exemplified by
BERT and its variants like RoBERTa, excel at categorizing input into predefined classes. These
models utilize bidirectional attention mechanisms to understand context from both directions,
making them particularly suitable for pattern recognition tasks. Generation models, such as
GPT variants, use autoregressive approaches to predict sequential patterns, potentially offering
different insights into network traffic analysis.
Text Classification vs. Generation approaches
While text generation models like Mistral’s 7B Instruct v0.2 [38] and OpenAI’s GPT-3 [39] have
shown remarkable capabilities in producing human-like text, they are not optimally suited for
the task of network intrusion detection. The primary reasons for this are:
Task Alignment: Intrusion detection is fundamentally a classification problem, where the goal is
to categorize network traffic into predefined classes (normal or various attack types). Text
Ethan Campbell
 40483972
generation models are designed to produce coherent text, which is not directly applicable to
this task.
Efficiency: Text classification models are generally more computationally efficient for this type
of task, as they don’t need to generate full text responses but rather output probability
distributions over a fixed set of classes.
Precision: Classification models can be fine-tuned to optimize for specific metrics relevant to
intrusion detection, such as minimizing false positives and false negatives.
Input Format: Network traffic data, even when pre-processed, doesn’t resemble natural
language text. Classification models can be more easily adapted to work with this structured,
non-linguistic input.
4.2. Model Selection and Training Methodology
For each model, standard evaluation metrics were obtained after training, including accuracy,
precision, recall, and F1-score. Every evaluation was completed on a different sample of the
respective dataset that wasn't used during training the model to ensure an unbiased
assessment.
Model Selection: Mistral 7B and GPT-3
For original testing with text generation models, transformer based systems designed to
produce human-like text based on input prompts, I used Mistral's 7B Instruct v0.2 [38] and
OpenAI’s GPT-3 [39] were fine-tuned. As described in the pre-processing section the datasets
were created with the natural English language generation in mind, with the idea that patterns
between malicious and benign records could be inferred from the training data.
While each attribute was described in English, the descriptions remained constant across
records, with only the numerical values changing. After training the two text generation models,
it became apparent that focusing solely on the original numerical data might be more effective
for the task at hand.
The Mistral-7B-Instruct-v0.2 model was configured with a block size of 1024 tokens and a
maximum sequence length of 2048 tokens. The training utilized the SFT (Supervised Fine-
Tuning) trainer with tensorboard logging. A learning rate of 3e-05 was applied for a single epoch.
To manage memory constraints, a small batch size of 2 was used with gradient accumulation of
4. The AdamW optimizer and a linear learning rate scheduler were employed for optimization.
After 725 training steps, the model achieved a loss value of 0.1725.
For the GPT-3 model, more specific parameters weren't accessible like with mistral, but it was
trained for 3 epochs with a batch size of 12 and a learning rate multiplier of 2. A seed value of
1303128212 was used for reproducibility. Training progressed for 725 steps, reaching a minimal
loss value. However, token limit issues were encountered, where individual records exceeded
the model's input capacity without truncation.
Despite achieving acceptable accuracy, these text generation models were not selected for
further testing due to operational limitations. The prediction speed was notably slow, with each
classification taking approximately 3-4 seconds even with GPU acceleration, compared to near-
instantaneous results from BERT-based models. Additionally, while BERT-based text
classification models provide predetermined response categories, text generation models
could produce unpredictable responses that complicated automated classification. These
Ethan Campbell
 40483972
operational constraints, combined with higher computational costs and comparable accuracy
to faster alternatives, led to the decision to focus on more efficient text classification
approaches for subsequent experiments.
Model Selection: RoBERTa-large
Among text classification models, RoBERTa-large [12] was selected for this study. RoBERTa,
introduced by Facebook AI in 2019, is an optimized version of BERT that has shown superior
performance across various NLP tasks. RoBERTa was pretrained on a large corpus of diverse
text data which should provide a strong foundation for transfer learning when applied to
structured data like network logs. Despite being initially designed for natural language,
RoBERTa's architecture should be adaptable to structured network data if formatted
appropriately.
The model was configured with consideration of computational efficiency and performance
optimization. Key hyperparameters included: train and evaluation batch sizes of 8 per device,
balancing between memory constraints and training speed; a learning rate of 2e-5, chosen to
allow for fine-tuning of the pretrained model without overwriting its learned features too quickly;
training over 5 epochs, with evaluation and model saving after each epoch to capture the best-
performing checkpoints.
Training outcomes varied by dataset:
•
•
•
•
CIC-DDoS2019 (2 labels): 3750 steps with minimal loss.
CIC-DDoS2019 (6 labels): 11250 steps, final training loss 0.1229, validation loss 0.1318.
CIC-IDS-2018 (4 labels): 4500 steps with minimal loss.
TON IoT (5 labels): 4690 steps, final training loss 0.0159, validation loss 0.038011.
In this context, a "step" represents one iteration of training where the model processes a single
batch of data and updates its parameters. The total number of steps is calculated by multiplying
the number of epochs by the number of batches needed to process the entire dataset. For
example, the 3750 steps in the CIC-DDoS2019 binary classification task indicates the total
number of parameter updates performed during training.
The loss values provide a quantitative measure of the model's prediction accuracy during
training, where lower values indicate better performance. Training loss reflects the model's
performance on the training data, while validation loss indicates its performance on unseen
validation data. The difference between training and validation loss can reveal whether the
model is overfitting – when the validation loss is significantly higher than the training loss, it
suggests the model may be memorizing training data rather than learning generalizable
patterns. In the results, the close alignment between training and validation loss (e.g., 0.1229 vs
0.1318 for CIC-DDoS2019 with 6 labels) suggests good generalization without overfitting.
As previously described, the model was fine-tuned for five different cases including: One model
trained on all labels (normal, DDoS, injection, password, scanning, DoS, ransomware,
backdoor, xss, mitm); Four different binary classification models, each excluding one attack
type to simulate zero-day attack detection using 5 labels (normal, DDoS, injection, password,
scanning). This approach allowed for a comprehensive evaluation of the model's capabilities in
both multi-class classification and its ability to generalize to unseen attack patterns.
Ethan Campbell
 40483972
By leveraging the strengths of the RoBERTa architecture and carefully tuning the training
process, this approach aimed to create a robust and efficient intrusion detection system
capable of handling the complexities present in the TON IoT dataset.
Model Used for Paper Replication: bert-based-uncased
For the paper replication study using NSL-KDD, BERT-base-uncased was implemented,
following the architecture described in the original study [19]. The model configuration closely
adhered to the original implementation while incorporating some refinements to enhance
reproducibility and performance.
The BERT-base-uncased model was selected to match the original study's architecture. The
implementation maintained the core architectural choices while providing more detailed
documentation of the training process. The model was configured with a maximum sequence
length of 512 tokens and a batch size of 32 to balance memory constraints with training
efficiency.
These modifications resulted in the model successfully classifying 13 attack classes accurately,
compared to the 11 reported in the original study. The improved performance can be attributed
to the extended training duration and more robust validation process, though direct comparison
is challenging due to discrepancies in the reported number of unique attack classes between
the replicated paper analysis and the original paper.
4.3. Zero-Day Detection experimental setup using TON IoT dataset
To further assess the model's ability to detect zero-day attacks, a series of experiments were
conducted focusing on the generalization capabilities of both LLMs and traditional machine
learning approaches using the idea of [20], removing specific labels from training data. Four
distinct scenarios were created, each simulating a potential zero-day attack by excluding one
specific attack type from the training data.
For each scenario, specialized datasets were prepared specialized datasets derived from the
original TON-IoT data. Instead of using every label type seen In Table 4, the datasets included
five traffic types: normal, DDoS, injection, password, and scanning attacks. These datasets
would then be used to train 4 different models, where malicious labels would be excluded one
at a time for each models training.
To streamline the experimental process and provide a focused comparison, consolidated the
attack types into a binary classification problem. Normal traffic was labelled as “0”, while all
attack types were collectively labelled as “1”. This approach allowed us to evaluate the models'
performance in distinguishing between benign and malicious traffic, with particular emphasis
on their ability to identify attack patterns not present in the training data.
Each experimental setup for this zero-day attack test involved creating three distinct datasets:
two for training and validation, and another for testing where the datasets all took different
samples from the original TON-IoT dataset. This separation ensures a true evaluation of the
models' generalization capabilities, as they are tested on network traffic patterns they haven't
encountered during training. The training dataset consisted of 12,000 records in total, with
3,000 normal records labelled as “0” and 9,000 attack records labelled as “1”, excluding the
specific attack type being simulated as a zero-day threat. The testing dataset comprised 15,000
different records, including 3,000 records for each of the five traffic types, ensuring a
Ethan Campbell
 40483972
comprehensive evaluation of the model's performance across all categories, including the
simulated zero-day attack.
This approach serves multiple purposes: it simulates a real-world scenario where new,
previously unseen attack types emerge; it challenges the model to identify anomalies based on
learned patterns of normal traffic and known attack types; it allows us to assess the model's
capability to generalize its understanding of “malicious” behaviour to novel attack patterns.
By excluding each attack type in turn, a controlled environment was create to test the model's
performance on “zero-day” attacks, providing insights into its potential real-world applicability
in detecting emerging threats. This method aligns with the research goal of evaluating LLMs’
effectiveness in adapting to evolving cyber threats and their potential in enhancing intrusion
detection systems' resilience against novel attacks.
4.4. Further Zero-Day testing using CICDDoS2019 Dataset
To further assess the model’s ability to detect zero-day attacks and give another comparison
perspective, the CICDDoS2019 dataset [24] was used. The dataset represents a comprehensive
collection of benign and modern DDoS attack traffic. The dataset addresses previous
limitations in DDoS evaluation data by including both reflection-based and exploitation-based
attacks across TCP/UDP protocols. Generated using the B-Profile system to simulate realistic
network behaviour, it contains labelled network flows from 25 users' activities across common
protocols (HTTP, HTTPS, FTP, SSH, email). The dataset includes detailed packet captures
(PCAPs) and labelled flow features extracted using CICFlowMeter-V3 [40].
From the dataset, I used a subset of 6 specific labels taken from the following files: UDPLag.csv,
Syn.csv, DrDoS_NetBIOS.csv, DrDoS_MSSQL.csv, DrDoS_LDAP.csv, including the benign traffic
data from these, where the amount of available relevant records can be seen in Table 2. This
dataset was used exclusively for zero-day attack detection experiments, focusing specifically
on UDP-lag and SYN attack scenarios. Following the preprocessing approach from the TON-IoT
dataset, the data format was standardized by removing temporal and specific contextual
information, with IP addresses encoded as "local" or "foreign" to maintain consistency and
privacy while preserving the underlying pattern recognition capabilities.
Table 5, distribution of attack types in the "processed network" datasets and their train/validation/test
splits
Attack Type
 Total Count
 Train
 Validation Test
BENIGN
 44,820
 2000
 2000
 3000
LDAP
 2,179,930
 2000
 2000
 3000
MSSQL
 4,522,492
 2000
 2000
 3000
NetBIOS
 4,093,279
 2000
 2000
 3000
SYN*
 4,284,751
 2000
 2000
 3000
UDP-lag*
 366,461
 2000
 2000
 3000
* Each served as zero-day attack and were excluded in train/validation in separate experiments
While previous tests used the CICDDoS2019 dataset, in this case a different subset was
created exclusively for this zero-day attack detection testing. The pre-processed dataset was
structured into a binary classification format, with benign traffic labelled as "0" and attack
traffic as "1". For each zero-day experiment, one attack type was systematically excluded from
the training process. The experimental datasets were constructed by sampling 2,000 records
from each included label for training (10,000 total, as one label is excluded), 2,000 records per
Ethan Campbell
 40483972
included label for validation (10,000 total), and for testing 3,000 records from each label were
included (15,000 total, as no labels are excluded). This sampling strategy ensures
comprehensive evaluation of the model's capability to detect previously unseen attack
patterns.
5. Experimental Results and Analysis
This section presents experimental results across multiple datasets and testing scenarios,
progressing from primary findings to various validation experiments. Section 5.1 details core
results using the TON-IoT dataset, where a notably high accuracy (98.53%) was achieved in
multi-class attack detection. Section 5.2 explores validation experiments across different
datasets where a testing process similar to Section 5.1 was used with varying degrees of
success - from excellent performance with CICDDoS2019 (100% accuracy in binary
classification) to significant challenges with CSE-CIC-IDS2018 and real-world network
captures. Section 5.3 examines zero-day attack detection capabilities, demonstrating strong
performance in most scenarios, while Section 5.4 provides a comprehensive comparison of
model performance across all experiments. This structured presentation allows us to examine
both the strengths and limitations of the LLM-based approach to network intrusion detection.
To evaluate the performance of the models, several key metrics were employed. The formulas
for these metrics are as follows:
Where:
•
•
•
•
•
TP is True Positives
FP is False Positives
TN is True Negatives
FN is False Negatives
N is Total Samples
Precision (P) measures the accuracy of positive predictions, while Recall (R) quantifies the
proportion of actual positives correctly identified. The F1-score balances precision and recall in
a single metric. The False Positive Rate (FPR) indicates the proportion of negative instances
incorrectly classified as positive. Accuracy (A) represents the overall correctness of the model
across all classes.
These metrics were calculated for each class individually and then aggregated to provide an
overall performance assessment.
Ethan Campbell
 40483972
5.1. Primary Experiment Using TON-IoT Dataset
Promising results were demonstrated by the text classification model in identifying various
types of network traffic. This approach focused on pattern recognition within the network data,
training the model on the numerical features of each packet record without relying on human-
readable attributes.
The model was trained to classify traffic into ten distinct categories: normal network data (0,
"normal"), DDoS (1, "ddos"), data injection attempts (2, "injection"), password brute-force
attempts (3, "password"), and information gathering port scans (4, "scanning"), denial of service
(5, "dos"), ransomware attacks (6, "ransomware"), backdoor intrusions (7, "backdoor"), cross-
site scripting (8, "xss"), and man-in-the-middle attacks (9, "mitm"). This multi-class
classification task presented an even more challenging and realistic scenario compared to the
previous five-class model, as it required the model to distinguish between a wider range of
traffic types that may share similarities. This multi-class classification task presented a more
challenging and realistic scenario compared to binary classification, as it required the model to
distinguish between multiple traffic types that may share similarities.
Table 6, performance metrics for different traffic types
Type
 Accuracy
 Precision
 Recall
 F1-score
 FP rate
normal
 0.9983
 0.9900
 0.9947
 0.9924
 0.0012
ddos
 0.9966
 0.9799
 0.9893
 0.9846
 0.0025
inject
 0.9410
 0.9850
 0.9410
 0.9625
 0.0018
passwd
 0.9923
 0.9780
 0.9923
 0.9851
 0.0027
scan
 0.9893
 0.9966
 0.9893
 0.9930
 0.0004
dos
 0.9953
 0.9940
 0.9953
 0.9947
 0.0007
ransom
 0.9963
 0.9973
 0.9963
 0.9968
 0.0003
back
 0.9983
 1.0000
 0.9983
 0.9992
 0.0000
xss
 0.9853
 0.9576
 0.9853
 0.9713
 0.0053
mitm
 0.9011
 0.9258
 0.9011
 0.9133
 0.0014
Overall
 0.9853
 0.9853
 0.9853
 0.9852
 0.0016
Ethan Campbell
 40483972
Figure 4, TON IoT standard results confusion matrix
The model achieved an impressive overall accuracy of 98.53% across the test dataset, as
evidenced by the confusion matrix presented in Table 6 and Figure 4. This visualization clearly
demonstrates the model's high performance, with most categories showing correct
classifications close to or exceeding 2,900 out of 3,000 test samples. For the 'mitm' category,
which had a smaller sample size of 512, the model also correctly identified 474 instances. This
high level of accuracy suggests that the model was able to effectively learn and distinguish the
patterns associated with each type of network traffic, even when dealing with multiple
categories that may have overlapping characteristics.
These results, as depicted in Table 6 and Figure 4, indicate that the text classification approach
using the RoBERTa-large model is highly effective for network traffic analysis, particularly when
dealing with complex, multi-class scenarios. The model's ability to maintain such high accuracy
across multiple traffic types, with most diagonal elements in the confusion matrix showing
values above 2,900, demonstrates its potential for real-world applications in network security
and traffic management. The overall false positive rate of 0.16% is particularly noteworthy, as it
suggests that the model rarely misclassifies benign traffic as malicious. This is crucial in
practical applications to minimize false alarms and reduce the workload on security analysts,
as evidenced by the low off-diagonal values in the confusion matrix.
Ethan Campbell
 40483972
To contextualize the performance of the LLM-based approach, results were compared with
those from traditional machine learning methods applied to the same TON-IoT dataset. The
comparison data is taken from a journal article [41] which uses the same Network Data subset
of the TON IoT dataset. The results are presented in Table 7.
Table 7, comparison of different machine learning approaches
Approach
 Accuracy
 Precision
 Recall
 F1-score
 FP rate
AdaBoost (Multi-class, All attributes)
 0.399
 0.339
 0.229
 0.274
 0.505
KNN (Multi-class, All attributes)
 0.979
 0.933
 0.925
 0.929
 0.009
XGBoost (Multi-class, All attributes)
 0.983
 0.945
 0.953
 0.949
 0.008
AdaBoost (Multi-class, Chi2)
 0.497
 0.352
 0.363
 0.358
 0.424
KNN (Multi-class, Chi2)
 0.977
 0.929
 0.928
 0.929
 0.014
KNN (Multi-class, SMOTE)
 0.976
 0.901
 0.956
 0.928
 0.018
XGBoost (Multi-class, SMOTE)
 0.979
 0.907
 0.968
 0.937
 0.018
RoBERTa-large TON IoT
 0.9853
 0.9853
 0.9853
 0.9852
 0.0016
The RoBERTa-large model, trained on the TON IoT dataset with 5 traffic types, demonstrates
competitive performance compared to the XGBoost [42] models. The LLM-based approach
achieves an accuracy of 98.75%, which is comparable to the best-performing XGBoost models
in both binary and multi-class classifications.
Notably, the model shows superior performance in terms of the false positive rate (FP rate).
With an FP rate of 0.0031, it significantly outperforms all XGBoost variants, which have FP rates
ranging from 0.008 to 0.018. This lower false positive rate is particularly important in the context
of intrusion detection systems, as it reduces the number of false alarms that security analysts
need to investigate.
It's important to highlight that our model achieves this performance in a multi-class setting with
10 traffic types, which is generally a more challenging task than binary classification. The fact
that it outperforms or matches the multi-class XGBoost models while maintaining a lower false
positive rate underscores the potential of LLMs in this domain.
Moreover, the LLM-based approach demonstrates this high performance without the need for
extensive feature engineering or selection methods (such as Chi2) or data balancing techniques
(like SMOTE) that were applied to some of the XGBoost models. This suggests that the LLM's
ability to learn patterns from the raw data representation might reduce the need for some of the
preprocessing steps typically required in traditional machine learning approaches.
These results suggest that LLM-based approaches, such as our RoBERTa-large model, can be
highly effective for network intrusion detection tasks. They offer competitive performance
compared to traditional machine learning methods while potentially providing additional
benefits in terms of generalization and adaptability to novel attack patterns.
5.2. Other Validation Results Based on TON-IoT Experiment
5.2.1. CICDDoS2019 results
The CICDDoS2019 dataset was used to evaluate two distinct methodological approaches.
Initially, text generation models (Mistral 7B and GPT-3) were employed to assess their capability
in classifying network traffic through natural language processing. Following these experiments,
the methodology shifted to using RoBERTa-large as a text classification model, which proved to
Ethan Campbell
 40483972
be more efficient and accurate in identifying attack patterns. This progression allowed us to
compare the effectiveness of different LLM architectures and approaches in the context of
network intrusion detection.
Mistral 7B and GPT-3 text generation model results
The initial testing phase evaluated the Mistral model using two distinct approaches. The first
approach involved testing individual records not present in the training dataset but sourced
from the same distribution. The model demonstrated high accuracy in classifying single
records, achieving correct classification in all test cases for both malicious and benign
samples.
single malicious record outside dataset used for training, correct 10/10 times:
Figure 5, Mistral 7B malicious prediction example
single benign record from outside the dataset used for training:
Figure 6, Mistral 7B benign prediction example
The second approach tested the model's ability to classify multiple records simultaneously.
Due to token length constraints, testing was limited to batches of 9-12 records. This approach
proved less successful, with the model incorrectly classifying groups of malicious records as
benign in approximately 90% of cases. While the classification accuracy was poor, the model
provided explanatory justification for its decisions, offering insight into its classification
reasoning.
Below is an example of a series of 10 malicious records in a row (from outside dataset) being
categorised as benign. It also had the same outcome if malicious records from inside the
dataset where used. While the example shows the incorrect classification, a difference in this
version compared to single records was that it tried explaining why it thought all the records fell
under a specific category.
Ethan Campbell
 40483972
Figure 7, Mistral 7B multiple record prediction example
In the second version of the model using GPT-3, It had basically the same results categorisation
wise as it very accurately guessed the label for a single record.
Figure 8, GPT-3 multiple record prediction example
Testing of the GPT-3 model yielded similar results for single-record classification, matching the
high accuracy demonstrated by the Mistral model. However, when processing multiple records
simultaneously, GPT-3 exhibited the same tendency to misclassify batches as benign. A notable
difference between the models emerged in their explanatory capabilities - while Mistral
provided coherent reasoning for its classifications, GPT-3 often produced seemingly random
string patterns when prompted to explain its decisions. This disparity in explanation quality
suggests fundamental differences in how the models process and interpret network traffic
patterns.
roBERTa-large text classification model results
Following the text generation experiments, the approach shifted to utilizing a text classification
model with the CIC-DDoS2019 dataset. This methodology focused on pattern recognition within
the network traffic data, where attribute descriptions were omitted in favour of structured
numerical features. The streamlined data representation resulted in both improved accuracy
and reduced computational overhead due to decreased token counts per record.
Initial binary classification tests using only benign and UDP-Lag DDoS labels demonstrated
perfect accuracy, achieving 100% correct classification across 6,000 test records not present in
the training dataset. While these results were promising, the distinct characteristics between
Ethan Campbell
 40483972
these two traffic types made classification relatively straightforward. This led to the
development of a more challenging multi-class classification scenario.
The subsequent experiment expanded to six distinct traffic types: benign traffic, UDP-Lag DDoS,
SYN flood attacks, NetBIOS protocol DDoS, MSSQL-targeted DDoS, and LDAP-targeted DDoS
attacks. As shown in Table 8, the model achieved an overall precision of 80.78% with varying
performance across different attack types. UDP-Lag detection maintained perfect precision and
recall, while other attack types showed more variation in their detection rates. The overall false
positive rate of 0.0429 indicates relatively good discrimination between attack types, though not
as strong as the binary classification results.
Table 8, performance metrics for CICDDoS2019 multi-class classification
Label
 Precision
 Recall
 F1-score
 FP rate
Benign
 0.6963
 0.9997
 0.8209
 0.0872
UDP-lag
 1.0
 1.0
 1.0
 0.0
Syn
 1.0
 0.5673
 0.7239
 0.0
DrDoS-NetBIOS
 0.6940
 0.9637
 0.8069
 0.0850
DrDoS-MSSQL
 0.6184
 0.4650
 0.5308
 0.0574
DrDoS-LDAP
 0.8381
 0.7177
 0.7732
 0.0277
Overall
 0.8078
 0.7856
 0.7760
 0.0429
5.2.2. CSE-CIC-IDS2018 results
Initial testing using the CSE-CIC-IDS2018 dataset yielded notably different results compared to
experiments using the CICDDoS2019 dataset. When evaluating the model using different
samples from the original dataset (not used in training), an overall accuracy of 78.02% was
observed. This performance, while significantly better than random chance, fell short of the high
accuracy rates achieved with the CICDDoS2019 dataset.
Table 9, performance metrics for CSE-CIC-IDS2018 multi-class classification
Label
 Precision
 Recall
 F1-score
 FP rate
Benign
 0.3453
 0.9923
 0.5123
 0.6272
SSH-Bruteforce
 0.8593
 0.3970
 0.5431
 0.0217
DoS-Slowloris
 0.9177
 0.2380
 0.3780
 0.0071
DoS-GoldenEye
 0.9984
 0.4040
 0.5752
 0.0002
Overall
 0.7802
 0.5078
 0.5021
 0.1641
Ethan Campbell
 40483972
When testing the model against custom-generated network traffic data, collected from a
controlled testing environment described in Section 3.2, the results seen in Figure 9 were
attained. The confusion matrix for these tests revealed near-complete failure in attack
detection, with the model classifying almost all traffic as benign, regardless of its actual nature.
Of the approximately 500 records captured for each attack type, only a handful were classified
as non-benign, and even these classifications were incorrect when compared to the ground
truth labels.
Figure 9, CSE-CIC-IDS2018 results using newly-generated benign data
The model's performance using only the original CSE-CIC-IDS2018 dataset’s data can be seen
in Figure 10, and while it was better than when custom data was used, it still showed significant
limitations. The Figure 10 confusion matrix demonstrates that even with familiar data patterns,
the model struggled to maintain consistent classification accuracy across different attacks for
the type of data in this dataset.
The contrast between the model's performance on the original dataset (50.78% accuracy) and
the custom-generated data (effectively 0% accuracy) suggests several potential issues. First,
the significant drop in performance when testing against real-world traffic may primarily stem
from limitations in the experimental setup for generating attack data. Despite efforts to recreate
Ethan Campbell
 40483972
Figure 10, CSE-CIC-IDS2018 results using original dataset only
attack scenarios in a controlled home network environment, the generated traffic patterns likely
differed substantially from those in the training dataset. This could be attributed to variations in
network configuration, differences in attack tool implementations, or other environmental
factors that weren't perfectly aligned with the original dataset's collection methodology.
Additionally, though secondary, there's the possibility that the model may be overfitting to
specific characteristics of the CSE-CIC-IDS2018 dataset rather than learning generalizable
patterns of attack behaviour.
These results highlight a crucial consideration: the challenges in accurately reproducing attack
scenarios that match the characteristics of established benchmark datasets. While the poor
performance on the custom-generated data initially appears to suggest limitations in the
model's real-world applicability, it more likely reflects the difficulties in creating comparable
attack traffic in a controlled environment. Future work would benefit from more sophisticated
attack simulation methodologies or, ideally, access to real-world attack data collected from
production environments.
5.2.3. NSL-KDD replication study results
The replication study of the NSL-KDD dataset experiments revealed several notable
discrepancies from the original research [19]. While the original paper reported working with 23
distinct attack classes in their training and testing data, analysis during recreation of the
experiment identified 22 unique attack labels in the training dataset and 38 distinct attack types
in the test dataset, with 22 labels shared between both sets. This finding contrasts with the
original paper's assertion that "The testing subset comprises only 12 out of the total 23 training
classes."
Ethan Campbell
 40483972
The confusion matrix for the replication of the study (Figure 11) illustrates the model's
classification performance across the shared attack types. The implementation achieved
accurate prediction for 13 distinct attack classes, surpassing the original paper's reported
success rate of 11 classes. This improved performance persisted even when their single epoch
training approach was replicated, suggesting that the difference in results stems from factors
beyond training duration.
Figure 11, results achieved from paper replication
Several factors may contribute to these discrepancies. The analysis revealed that some attack
categories in the dataset contained very sparse data, with certain types having only 1-10
records. This observation suggests that the original study might have excluded these low-
frequency attack types from their analysis, though this methodology wasn't explicitly stated in
their paper.
The confusion matrix demonstrates particularly strong performance in identifying major attack
categories, with notably high precision for denial-of-service (DoS) attacks and probe attempts.
However, the matrix also reveals some classification challenges for attack types with limited
training samples, reflecting the inherent difficulty of learning from imbalanced datasets.
These findings highlight important considerations for replication studies in network security
research, particularly regarding dataset composition and the handling of imbalanced attack
categories. While the recreated implementation achieved superior classification performance,
Ethan Campbell
 40483972
the discrepancies in reported attack class numbers underscore the need for detailed
methodology reporting in security research, especially regarding dataset preprocessing and
class selection criteria.
5.2.4. Multi-Source Data results
The evaluation of the model's performance on real-world network traffic data collected from a
home network environment provided insights into both the potential and limitations of LLM-
based network intrusion detection in practical settings. The test dataset comprised network
captures during normal operations and during controlled attack scenarios, providing a realistic
evaluation environment.
Figure 12 demonstrates the model's classification performance across the collected dataset.
When examining benign traffic patterns, the model showed strong accuracy in correctly
identifying normal network behaviour. However, the results for malicious traffic classification
completely failed and only classified traffic as benign.
Figure 12, Multi-source data results confusion matrix
Ethan Campbell
 40483972
Figure 13, Multi-source data results for individual captures
The overall accuracy achieved was 63.18%, with the model performing notably better at
identifying benign traffic compared to malicious traffic patterns. This disparity in classification
performance raises important considerations about the methodology used for labelling and the
inherent challenges in real-world network traffic classification.
A significant limitation of this experiment stems from the data labelling methodology employed.
Traffic was labelled as malicious based solely on the presence of specific IP addresses
associated with the attack-generating machine in either the source or destination fields. This
approach, while pragmatic for initial testing, presents several potential issues:
1.2.3.Not all traffic involving the attack-generating machine is necessarily malicious (e.g.,
initial connection establishment, DNS queries)
The binary classification based on IP addresses may oversimplify complex network
interactions
The model may learn to associate specific network characteristics unique to the attack
machine rather than actual attack patterns
These findings highlight an important consideration for real-world applications: while the model
shows promise in identifying benign traffic patterns specific to a network environment, its ability
to generalize to malicious traffic detection may require more sophisticated labelling
mechanisms and training methodologies. This observation suggests a potential avenue for
future research, particularly in developing more nuanced approaches to real-world traffic
labelling and classification that go beyond simple IP-based heuristics.
Despite these limitations, the experiment provides valuable insights into the practical
application of LLM-based network intrusion detection in real-world environments. The results
suggest that while current approaches show promise, further research is needed to bridge the
gap between controlled experimental environments and practical deployment scenarios.
5.3. Zero-Day Detection Performance
To assess the model's ability to detect zero-day attacks, experiments were conducted
simulating four zero-day scenarios using the TON-IoT dataset, as well as a further two scenarios
using the CICDDoS2019 dataset, each excluding a specific attack type from the training data.
Ethan Campbell
 40483972
These experiments were designed to evaluate how well the models could identify previously
unseen attack patterns. The results provide valuable insights into the generalization capabilities
of both the LLM approach using RoBERTa-large and the traditional machine learning approach
using XGBoost.
For each TON-IoT zero-day scenario, while the training and validation datasets would only ever
include 4/5 labels, the test dataset included all 5 possible labels used in the experiment
(normal, ddos, password, injection and scanning). This comprised a total of 15,000 records:
3,000 records of normal, benign traffic (labelled as "0"); 12,000 records of malicious traffic
(labelled as "1"). This balanced distribution ensures a comprehensive evaluation of the models'
ability to distinguish between normal traffic and both known and unknown attack patterns.
Tables 10-13 present the performance of both models in classifying unseen attack types,
comparing the accuracy and classification breakdown of the LLM and XGBoost models when
faced with attack patterns absent from their training data. This analysis allows us to evaluate
the effectiveness of language model-based approaches against traditional machine learning
methods in the context of emerging threat detection.
Table 10, models where ddos records were excluded in training: 99.67% correct classification of 3000
ddos records
Model
XGB
LLM
Accuracy
 Predicted Malicious
 Predicted Benign
0.9067
 2720/3000
 280/3000
0.9967
 2990/3000
 10/3000
Table 11, models where password records were excluded in training: 100% correct classification of 3000
password records
Model
XGB
LLM
Accuracy
 Predicted Malicious
 Predicted Benign
1.0
 3000/3000
 0/3000
1.0
 3000/3000
 0/3000
Table 12, models where injection records were excluded in training: 100% correct classification of 3000
injection records
Model
XGB
LLM
Accuracy
 Predicted Malicious
 Predicted Benign
0.9970
 2991/3000
 9/3000
1.0
 3000/3000
 0/3000
Table 13, models where scanning records were excluded in training: 32.87% correct classification of 3000
scanning records
Model
XGB
LLM
Accuracy
 Predicted Malicious
 Predicted Benign
0.3233
 970/3000
 2030/3000
0.3287
 986/3000
 2014/3000
To validate these findings in a different context, Tables 14 and 15 show results of additional
zero-day detection experiments that were conducted using the CICDDoS2019 dataset,
Ethan Campbell
 40483972
specifically focusing on UDP-lag and SYN attacks. The results from these experiments
demonstrated similarly high detection capabilities
Table 14, models where udp-lag records were excluded in training: 99.97% correct classification of 3000
udp-lag records
Model
XGB
LLM
Accuracy
 Predicted Malicious
 Predicted Benign
1.0000
 3000/3000
 0/3000
0.9997
 2999/3000
 1/3000
Table 15, models where syn records were excluded in training: 100% correct classification of 3000 syn
records
Model
XGB
LLM
Accuracy
 Predicted Malicious
 Predicted Benign
1.0
 3000/3000
 0/3000
1.0
 3000/3000
 0/3000
The high accuracy in classifying unseen attack types (zero-day) demonstrates the LLMs robust
generalization capabilities, which are comparable to or exceed those of traditional machine
learning methods like XGBoost. In the TON-IoT experiments, four out of four zero-day scenarios
showed the LLM outperforming XGBoost, showcasing its strong ability to detect novel threats.
The LLM's performance was particularly impressive for DDoS classification, achieving a 9%
increase over XGBoost.
In the CICDDoS2019 experiments, both the LLM and XGBoost achieved exceptionally high
accuracy, with nearly perfect detection rates for both UDP-lag and SYN attacks. While the
traditional XGBoost approach matched the LLM's performance in these scenarios, the LLM's
ability to achieve comparable results is noteworthy, particularly considering that it operates on
natural language representations of network traffic rather than traditional feature engineering.
The consistently high accuracy in the CICDDoS2019 experiments can be attributed to the
distinct characteristics of benign traffic in this dataset, which provides a clearer separation
between normal and attack patterns compared to the more diverse traffic patterns present in
the TON-IoT dataset.
The lower performance in TON-IoT scanning classification suggests similarities between
scanning and normal traffic patterns. This is not unexpected given that port scanning can
resemble normal network exploration or service discovery processes. Overall, as evident from
the results across both datasets, the LLM demonstrates remarkable ability to correctly classify
network traffic records it hasn't been explicitly trained on. This capability likely stems from the
model's keen sensitivity to patterns in benign records, allowing it to flag any deviations as
potentially malicious, while matching the performance of specialized traditional machine
learning approaches.
5.4. LLM Model Comparison
The performance metrics across different models and datasets reveal several key insights about
the application of LLMs to network intrusion detection. The RoBERTa-large model achieved
perfect accuracy (1.0) in binary classification using the CICDDoS2019 dataset, though this great
performance could be attributed to the distinctive characteristics of DDoS traffic patterns and
the simplified binary classification task.
Ethan Campbell
 40483972
Table 16, comparison of all fine-tuned model attempts throughout paper
Approach
 Accuracy
 Precision
 Recall
 F1-score
 FP rate
roberta-large CICDDoS2019 binary
 1.0
 1.0
 1.0
 1.0
 0
roberta-large CICDDoS2019 6 types
 0.7856
 0.8078
 0.7856
 0.7760
 0.0429
roberta-large CICIDS2018 4 types
 0.0
 0.0
 0.0
 0.0
 0.2501
trying to use my own capture for
predictions
roberta-large CICIDS2018 4 types
 0.5078
 0.7802
 0.5078
 0.5021
 0.1641
roberta-large Multi-Source Data
 0.6318
 0.0
 0.0
 0.0
 0.3682
model
roberta-large TON IoT 5 types
 0.9875
 0.9875
 0.9875
 0.9875
 0.0031
When tested against more complex scenarios, such as the six-type classification in
CICDDoS2019, the model's performance decreased notably (accuracy: 0.7856, precision:
0.8078), highlighting the increased challenge of distinguishing between multiple attack types.
The model demonstrated its strongest practical performance with the TON IoT dataset,
achieving 98.75% accuracy across five traffic types while maintaining a remarkably low false
positive rate of 0.0031.
The relatively poor performance on the CSE-CIC-IDS2018 dataset (accuracy: 0.5078) and
subsequent failure to generalize to custom captures (accuracy: 0) suggests that the model's
effectiveness is highly dependent on the similarity between training and test data distributions.
This observation aligns with findings from previous studies regarding the challenges of
transferring learned patterns across different network environments [31].
6. Discussion
6.1. Implications for SOC Environments
This study explores the use of LLMs for intrusion detection, showing that the TON IoT dataset
offers significant insights that could be particularly relevant to Security Operations Center
(SOC) environments. The RoBERTa-large model's performance in classifying both known and
simulated zero-day attacks suggests that LLMs could have a valuable role in complementing
existing security infrastructure.
One noteworthy aspect of this approach is its generalization capability. The model showed
proficiency in correctly classifying network traffic records it hadn't been explicitly trained on.
This ability to detect deviations from learned benign patterns could be especially valuable in a
SOC context, particularly when trained on environment-specific datasets. The findings suggest
that by creating datasets specific to a production environment using methodologies similar to
those employed in benchmark datasets like TON-IoT or CICDDoS2019, the model could achieve
highly accurate detection rates for both known and novel attacks within that environment.
The implications for SOC environments are multifaceted:
1.Enhanced Zero-Day Detection: The model's ability to identify anomalies that deviate
from normal patterns might aid in discovering novel threats not present in the training
data. The results imply that any network behaviour significantly different from the usual
benign traffic patterns could be effectively flagged as potentially malicious, providing an
additional layer of protection against zero-day attacks.
Ethan Campbell
 40483972
2.3.4.5.Environment-Specific Optimization: By training the model on environment-specific
benign traffic patterns, SOCs could potentially create more accurate detection systems
tailored to their specific network characteristics. This approach could be particularly
effective as the results demonstrate the model's strong capability in distinguishing
between normal and anomalous traffic when trained on well-defined baseline patterns.
Possible Reduction in False Negatives: The model's generalization ability might help
catch subtle or sophisticated attacks that could potentially slip through traditional rule-
based systems. This is particularly evident in the zero-day detection experiments, where
the model successfully identified attack patterns it wasn't explicitly trained to recognize.
Contextual Analysis: LLMs' capacity for understanding context could potentially assist
in distinguishing between genuine threats and unusual but benign network activities,
which might help in managing alert volumes for SOC analysts. This is supported by the
experimental results showing low false positive rates across multiple datasets.
Adaptability: If implemented with continuous learning capabilities, such a system could
potentially adapt to the evolving state of a network, maintaining its effectiveness over
time. The experiments suggest that when similar attack scenarios to those used in
creating benchmark datasets are executed in a production environment, the model can
effectively detect them, particularly when trained on environment-specific data.
The practical implications of these findings suggest that SOCs could benefit from implementing
LLM-based detection systems as a complementary layer to their existing security infrastructure.
By training these systems on a combination of environment-specific benign traffic and known
attack patterns, organizations could potentially create more robust and adaptive intrusion
detection capabilities. This approach would be particularly valuable in environments where the
characteristics of normal network traffic are well-defined and consistent, allowing the model to
more effectively identify deviations that could indicate potential security threats.
It's important to note that these potential benefits must be weighed against the challenges and
limitations discussed in subsequent sections, particularly regarding computational
requirements and the need for careful integration with existing security tools and processes.
6.2. Main Findings, Limitations and Challenges
The application of LLMs to network intrusion detection presents both promising capabilities and
notable limitations that warrant careful consideration. Through experimental analysis, several
key findings and challenges emerged that have significant implications for practical
implementation.
Dataset Dependencies and Training Considerations: A fundamental limitation of the findings
stems from the potential overlap between LLM training data and the experimental datasets. The
strong performance observed might be partially attributed to the possibility that the LLMs'
original training datasets included or contained similar content to the benchmark datasets used
in the experiments (TON-IoT, CICDDoS2019, etc.). This overlap could artificially enhance the
models' pattern recognition capabilities for these specific datasets. However, experiments with
custom-generated traffic data, particularly in the CSE-CIC-IDS2018 tests, revealed significant
performance degradation when dealing with traffic patterns outside these established
benchmarks.
The implications of this limitation suggest that optimal implementation in production
environments would require creating environment-specific datasets using methodologies
similar to those employed in generating benchmark datasets. By replicating similar attack
Ethan Campbell
 40483972
scenarios and collecting corresponding network traffic data, organizations could potentially
enhance detection accuracy for their specific network contexts. This approach could help
bridge the gap between laboratory performance and real-world effectiveness.
Operational Limitations: The current implementation faces several operational constraints that
could impact its practical utility:
1.Single Record Analysis: A significant limitation lies in the model's ability to only analyse
individual network traffic records in isolation. This constraint makes it challenging to
detect attack patterns that manifest across multiple records or time periods, such as
sophisticated DDoS attacks or data exfiltration attempts. While using IDS logs (e.g.,
Zeek connection logs) that summarize entire packet flows provides more
comprehensive context, the fundamental limitation of single-record analysis remains a
barrier to detecting certain types of attacks.
2.False Positive Risk: The model's sensitivity to anomalies, while beneficial for detecting
unknown threats, introduces a risk of elevated false positive rates. This is particularly
pronounced when encountering benign but unusual network activities that deviate from
typical patterns. Achieving an optimal balance between detection sensitivity and false
positive reduction remains a significant challenge.
3.Explainability Challenges: The inherent "black box" nature of LLMs poses challenges for
operational deployment [43]. In SOC environments, where understanding alert rationale
is crucial for effective response, the limited explainability of model decisions could
hinder incident investigation and response processes.
Resource and Implementation Challenges: The implementation of LLM-based detection
systems presents several practical challenges:
1.Computational Requirements: LLMs, particularly models like RoBERTa-large, demand
significantly higher computational resources compared to traditional intrusion
detection methods. This resource intensity could impact real-time analysis capabilities
in high-traffic networks, necessitating careful consideration of performance trade-offs.
2.Dataset Specificity: The model's performance exhibits strong dependency on training
dataset characteristics. While the TON IoT dataset provided comprehensive coverage,
real-world network environments may present traffic patterns that differ significantly
from training data. This specificity challenge suggests that successful deployment
would require ongoing model adaptation and training with environment-specific data.
3.Integration Complexity: Implementing LLM-based detection alongside existing security
infrastructure requires careful consideration of integration points and operational
workflows. The model's limitations in processing multiple records simultaneously could
complicate integration with existing security monitoring systems that rely on holistic
traffic analysis.
These limitations and challenges highlight the importance of viewing LLM-based intrusion
detection as a complementary tool rather than a replacement for existing security
infrastructure. Success in practical deployment likely depends on carefully designed
implementation strategies that account for these constraints while leveraging the model's
demonstrated strengths in pattern recognition and zero-day attack detection.
Ethan Campbell
 40483972
The findings suggest that optimal implementation would involve using LLMs as part of a layered
detection approach, where they augment rather than replace traditional signature-based
systems and rule-based blocking mechanisms. This hybrid approach could potentially combine
the pattern recognition capabilities of LLMs with the explicit detection capabilities of
conventional systems, providing enhanced protection against both known and emerging
threats.
6.4. Future Research Directions
The findings from this study suggest several promising avenues for future research in LLM-based
network intrusion detection. These directions focus on addressing current limitations while
expanding the potential applications of this technology.
Multi-Record Analysis Enhancement: A critical area for future investigation is developing
methodologies to analyse multiple network records simultaneously. This could involve
architectural modifications to existing LLM frameworks or novel approaches to sequence
modelling that better capture temporal relationships in network traffic. Such advancements
would be particularly valuable for detecting sophisticated attacks that manifest across multiple
connections or time periods.
Model Interpretability: Future research should explore techniques to enhance the explainability
of LLM decisions in network security contexts. This could include developing visualization tools
or interpretation frameworks specifically designed for network traffic analysis, making the
model's decision-making process more transparent to security analysts.
Environment-Specific Training: Investigation into efficient methods for environment-specific
model adaptation could significantly improve practical applications. This includes developing
techniques for model fine-tuning using production network data while maintaining detection
accuracy for known attack patterns. Research could focus on determining the optimal balance
between general and environment-specific training data.
Real-Time Processing Optimization: Future work should address the computational challenges
of deploying LLM-based detection in high-traffic networks. This could include exploring model
compression techniques, developing more efficient architectures, or investigating hybrid
approaches that combine lightweight initial filtering with more comprehensive LLM analysis for
suspicious traffic.
These research directions aim to enhance the practical applicability of LLM-based intrusion
detection while addressing the current limitations identified in this study. Success in these
areas could significantly advance the integration of LLMs into operational security
environments.
7. Conclusion
This study explored the application of Large Language Models (LLMs), exploring multiple LLM
approaches including GPT-3, Mistral 7B and BERT based LLMs, then testing them with multiple
established network intrusion detection datasets, including TON IoT, CICDDoS2019, NSL-KDD
among others. The findings demonstrate that LLMs, specifically the RoBERTa-large model, can
be effectively applied in this domain, achieving 98.53% overall accuracy compared to traditional
methods using the same data, like XGBoost (98.3%), KNN (97.9%), and AdaBoost (39.9%). The
model's superior false positive rate of 0.16% compared to XGBoost's 0.8% highlights its
practical advantages for real-world applications. Particularly noteworthy was the model's
Ethan Campbell
 40483972
performance in a zero-day attack scenario, where it achieved 99.67% accuracy for unseen
DDoS attacks compared to XGBoosts' 90.67%. These results were further validated using the
CICDDoS2019 dataset, where the model maintained comparable performance to XGBoost in
detecting novel attacks.
Based on the findings in this paper, an effective application of LLMs in SOC environments would
involve a binary classification approach. This would entail training a model on a combination of
environment-specific benign traffic and malicious network traffic from various datasets. The
model would classify traffic as benign (0) or potentially malicious (1), with low-confidence
benign classifications also flagged as potentially malicious. This conservative approach could
enhance anomaly detection while integrating with existing Intrusion Detection Systems. Such
an implementation could leverage LLMs' pattern recognition capabilities while complementing
traditional rule-based systems, potentially offering a balanced solution for enhancing network
security in SOC environments.
References
[1]
[2]
[3]
[4]
[5]
[6]
[7]
[8]
[9]
A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman, “Survey of intrusion detection
systems: techniques, datasets and challenges,” Cybersecurity, vol. 2, no. 1, pp. 1–22,
Dec. 2019, doi: 10.1186/S42400-019-0038-7/FIGURES/8.
G. Apruzzese et al., “The Role of Machine Learning in Cybersecurity,” Digital Threats:
Research and Practice, vol. 4, no. 1, Mar. 2023, doi: 10.1145/3545574/ASSET/C4586A69-
4437-435A-9DE6-E5F12828847D/ASSETS/GRAPHIC/DTRAP-2021-0064-F15.JPG.
F. N. Motlagh, M. Hajizadeh, M. Majd, P. Najafi, F. Cheng, and C. Meinel, “Large Language
Models in Cybersecurity: State-of-the-Art,” Jan. 2024, Accessed: Feb. 29, 2024. [Online].
Available: http://arxiv.org/abs/2402.00891
Y. Wang and H. Kobayashi, “High Performance Pattern Matching Algorithm for Network
Security,” IJCSNS International Journal of Computer Science and Network Security, vol.
6, no. 10, 2006.
M. D. Singh, “Analysis of Host-Based and Network-BasedIntrusion Detection System,”
Computer Network and Information Security, vol. 8, pp. 41–47, 2014, doi:
10.5815/ijcnis.2014.08.06.
T. Sommestad, H. Holm, and D. Steinvall, “Variables influencing the effectiveness of
signature-based network intrusion detection systems,” Information Security Journal: A
Global Perspective, vol. 31, no. 6, pp. 711–728, Nov. 2022, doi:
10.1080/19393555.2021.1975853.
Z. Yang et al., “A systematic literature review of methods and datasets for anomaly-based
network intrusion detection,” Comput Secur, vol. 116, p. 102675, May 2022, doi:
10.1016/J.COSE.2022.102675.
S. Banik, S. Surya, M. Dandyala, S. V. Nadimpalli, and S. Cybersecurity, “Heuristic-Based
Detection Techniques,” International Journal of Advanced Engineering Technologies and
Innovations, vol. 1, no. 2, pp. 352–362, Dec. 2022, Accessed: Sep. 26, 2024. [Online].
Available: https://ijaeti.com/index.php/Journal/article/view/573
J. Zhang and M. Zulkernine, “Network Intrusion Detection using Random Forests,” 2005.
Ethan Campbell
 40483972
[10]
[11]
[12]
[13]
[14]
[15]
[16]
[17]
[18]
[19]
[20]
[21]
[22]
M. Z. Alom and T. M. Taha, “Network intrusion detection for cyber security using
unsupervised deep learning approaches,” Proceedings of the IEEE National Aerospace
Electronics Conference, NAECON, vol. 2017-June, pp. 63–69, Jul. 2017, doi:
10.1109/NAECON.2017.8268746.
R. Vinayakumar, K. P. Soman, and P. Poornachandrany, “Applying convolutional neural
network for network intrusion detection,” 2017 International Conference on Advances in
Computing, Communications and Informatics, ICACCI 2017, vol. 2017-January, pp.
1222–1228, Nov. 2017, doi: 10.1109/ICACCI.2017.8126009.
Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” 2019,
Accessed: Oct. 15, 2024. [Online]. Available: https://github.com/pytorch/fairseq
G. Kocher and G. Kumar, “Machine learning and deep learning methods for intrusion
detection systems: recent developments and challenges,” Soft comput, vol. 25, no. 15,
pp. 9731–9763, Aug. 2021, doi: 10.1007/S00500-021-05893-0/TABLES/17.
C. Park, J. Lee, Y. Kim, J. G. Park, H. Kim, and D. Hong, “An Enhanced AI-Based Network
Intrusion Detection System Using Generative Adversarial Networks,” IEEE Internet Things
J, vol. 10, no. 3, pp. 2330–2345, Feb. 2023, doi: 10.1109/JIOT.2022.3211346.
E. Aghaei, X. Niu, W. Shadid, and E. Al-Shaer, “SecureBERT: A Domain-Specific Language
Model for Cybersecurity,” Apr. 2022, Accessed: Mar. 11, 2024. [Online]. Available:
http://arxiv.org/abs/2204.02685
T. Ali and P. Kostakos, “HuntGPT: Integrating Machine Learning-Based Anomaly Detection
and Explainable AI with Large Language Models (LLMs),” Sep. 2023, Accessed: Feb. 19,
2024. [Online]. Available: https://arxiv.org/abs/2309.16021v1
H. Lai, “Intrusion Detection Technology Based on Large Language Models,” 2023
International Conference on Evolutionary Algorithms and Soft Computing Techniques
(EASCT), pp. 1–5, Oct. 2023, doi: 10.1109/EASCT59475.2023.10393509.
D. M. Divakaran and S. T. Peddinti, “LLMs for Cyber Security: New Opportunities,” Apr.
2024, Accessed: May 06, 2024. [Online]. Available: http://arxiv.org/abs/2404.11338
O. G. Lira, A. Marroquin, and M. A. To, “Harnessing the Advanced Capabilities of LLM
for Adaptive Intrusion Detection Systems,” pp. 453–464, Apr. 2024, doi: 10.1007/978-3-
031-57942-4_44.
Z. Salehi, A. Sami, and M. Ghiasi, “MAAR: Robust features to detect malicious activity
based on API calls, their arguments and return values,” Eng Appl Artif Intell, vol. 59, pp.
93–102, 2017, doi: https://doi.org/10.1016/j.engappai.2016.12.016.
M. Soltani, B. Ousat, M. Jafari Siavoshani, and A. H. Jahangir, “An adaptable deep
learning-based intrusion detection system to zero-day attacks,” Journal of Information
Security and Applications, vol. 76, p. 103516, 2023, doi:
https://doi.org/10.1016/j.jisa.2023.103516.
N. Sameera and M. Shashi, “Deep transductive transfer learning framework for zero-day
attack detection,” ICT Express, vol. 6, no. 4, pp. 361–367, 2020, doi:
https://doi.org/10.1016/j.icte.2020.03.003.
Ethan Campbell
 40483972
[23]
[24]
[25]
[26]
[27]
[28]
[29]
[30]
[31]
[32]
[33]
[34]
T. Ohtani, R. Yamamoto, and S. Ohzahata, “Detecting Zero-Day Attack with Federated
Learning Using Autonomously Extracted Anomalies in IoT,” in 2024 IEEE 21st Consumer
Communications & Networking Conference (CCNC), 2024, pp. 356–359. doi:
10.1109/CCNC51664.2024.10454669.
I. Sharafaldin, A. H. Lashkari, S. Hakak, and A. A. Ghorbani, “Developing Realistic
Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy,” in 2019 International
Carnahan Conference on Security Technology (ICCST), 2019, pp. 1–8. doi:
10.1109/CCST.2019.8888419.
I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward Generating a New Intrusion
Detection Dataset and Intrusion Traffic Characterization,” in International Conference on
Information Systems Security and Privacy, 2018. [Online]. Available:
https://api.semanticscholar.org/CorpusID:4707749
G. Mohi-ud-din, “NSL-KDD,” 2018, IEEE Dataport. doi: 10.21227/425a-3e55.
N. Moustafa, “A new distributed architecture for evaluating AI-based security systems at
the edge: Network TON_IoT datasets,” Sustain Cities Soc, vol. 72, p. 102994, Sep. 2021,
doi: 10.1016/J.SCS.2021.102994.
N. Moustafa, “A Systemic IoT-Fog-Cloud Architecture for Big-Data Analytics and Cyber
Security Systems: A Review of Fog Computing,” May 2019, Accessed: Oct. 12, 2024.
[Online]. Available: https://arxiv.org/abs/1906.01055v1
N. Moustafa, “New Generations of Internet of Things Datasets for Cybersecurity
Applications based Machine Learning: TON_IoT Datasets,” in eResearch Australasia
Conference, Brisbane, Oct. 2019.
N. Moustafa, M. Ahmed, and S. Ahmed, “Data Analytics-enabled Intrusion Detection:
Evaluations of ToN IoT Linux Datasets,” Proceedings - 2020 IEEE 19th International
Conference on Trust, Security and Privacy in Computing and Communications, TrustCom
2020, pp. 727–735, Dec. 2020, doi: 10.1109/TRUSTCOM50675.2020.00100.
N. Moustafa, M. Keshk, E. Debie, and H. Janicke, “Federated TON_IoT windows datasets
for evaluating AI-based security applications,” Proceedings - 2020 IEEE 19th International
Conference on Trust, Security and Privacy in Computing and Communications, TrustCom
2020, pp. 848–855, Dec. 2020, doi: 10.1109/TRUSTCOM50675.2020.00114.
A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and Adna N Anwar, “TON-IoT telemetry
dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection
systems,” IEEE Access, vol. 8, pp. 165130–165150, 2020, doi:
10.1109/ACCESS.2020.3022862.
T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa, and F. T. H. D. Hartog, “ToN_IoT: The
Role of Heterogeneity and the Need for Standardization of Features and Attack Types in
IoT Network Intrusion Data Sets,” IEEE Internet Things J, vol. 9, no. 1, pp. 485–496, Jan.
2022, doi: 10.1109/JIOT.2021.3085194.
J. Ashraf et al., “IoTBoT-IDS: A novel statistical learning-enabled botnet detection
framework for protecting networks of smart cities,” Sustain Cities Soc, vol. 72, p. 103041,
Sep. 2021, doi: 10.1016/J.SCS.2021.103041.
Ethan Campbell
 40483972
[35]
[36]
[37]
[38]
[39]
[40]
[41]
[42]
[43]
“IoT-23: A labeled dataset with malicious and benign IoT network traffic”, doi:
10.5281/ZENODO.4743746.
S. García, M. Grill, J. Stiborek, and A. Zunino, “An empirical comparison of botnet
detection methods,” Comput Secur, vol. 45, pp. 100–123, 2014, doi:
https://doi.org/10.1016/j.cose.2014.05.011.
“malware-traffic-analysis.net.” Accessed: Sep. 25, 2024. [Online]. Available:
https://www.malware-traffic-analysis.net/
A. Q. Jiang et al., “Mistral 7B,” Oct. 2023.
T. B. Brown et al., “Language Models are Few-Shot Learners,” 2020.
“CICFlowMeter Repo.” [Online]. Available:
https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter
A. R. Gad, M. Haggag, A. Nashat, A. A. Nashat, and T. M. Barakat, “A Distributed Intrusion
Detection System using Machine Learning for IoT based on ToN-IoT Dataset,” Article in
International Journal of Advanced Computer Science and Applications, vol. 13, no. 6,
2022, doi: 10.14569/IJACSA.2022.0130667.
T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of
the 22nd acm sigkdd international conference on knowledge discovery and data mining,
2016, pp. 785–794.
A. Bhattacharjee, R. Moraffah, J. Garland, and H. Liu, “Towards LLM-guided Causal
Explainability for Black-box Text Classifiers,” 2024. [Online]. Available:
https://arxiv.org/abs/2309.13340
Appendix
A
 Initial Project Overview
Initial Project Overview SOC10101 Honours Project (40 Credits)
Title of Project: Using LLMs for SOC Management and Adaptive Threat Detection
Overview of Project Content and Milestones
The project aims to develop a system based on Large Language Model (LLM) models that can
operate in real-time for adaptive threat detection in cyber security systems, with related
research around how/if current systems have the capability to be adaptive, similar to
machine learning methods. To achieve this, the following key milestones will be established:
Literature Review: Investigate existing methodologies within SOC frameworks for threat
detection, focusing on Machine Learning and Deep Learning methods, also seeing if there is
previous work on the integration and performance of LLMs. The review should also address
how LLMs differ from and can augment traditional machine learning methods. A focus will be
designated to adaptive learning techniques and their operationalization within SOCs and
SIEM systems.
Technical Overview: Synthesize insights from the literature review to define the project scope,
which includes system constraints, desired outcomes, and potentials for SOC application.
Identification of data sources and training methodologies suitable for LLMs and other
machine learning models applied to SOC-related cyber security tasks.
Ethan Campbell
 40483972
Proof-of-Concept Development: Establish initial LLM prototypes to demonstrate the
conceptual viability, focusing on SOC contexts. This may encompass simulations of cyber-
attacks to test the prototype’s detection capabilities, as well as how an LLM would handle the
internal state of a SOC (i.e. managing users, permissions etc.) and compare its performance
with conventional machine learning models.
Further Development/Fine-Tuning: Refine and enhance the proof-of-concept model,
emphasizing improvements in real-time adaptability for previously unseen threats.
Investigate the efficacy of a hybrid LLM-machine learning system in SOC environments,
identifying optimization opportunities.
System Evaluation: Assess the developed system’s performance against standard SOC
solutions, such as SIEM systems, through simulated attack scenarios. Evaluate the
proficiency of detecting unknown threats and the model's ability to integrate with existing
SOC workflows.
The Main Deliverable(s):
The primary deliverable are the results of whether a large language model can enhance an
SOC system by providing real-time and adaptive threat detection. Research on comparing
machine learning and LLMs may influence the project, e.g. can they be used together to
improve the detection rate of threats on a network.
The Target Audience for the Deliverable(s):
Academics involved in research around the application of LLMs within cyber security
Network administrators that use intrusion detection systems.
Cybersecurity professionals and SOC operators seeking to leverage AI-driven tools for threat
detection and incident response.
The Work to be Undertaken:
- A thorough analysis in the literature review of LLMs, machine learning techniques, SOC
systems, and their roles in threat detection and analysis.
- A technical synopsis to describe project objectives, limitations, and envisaged impacts
on SOCs.
- Comparative analysis of the models against traditional SOC tools (e.g. SIEM systems) to
determine possible advancements in threat detection and response.
- A technical overview to scope the project's objectives, constraints, and intended
outcomes.
- Development of system based on LLM and machine learning models (depending on
research/literature review outcomes) to evaluate their effectiveness in threat detection in
a controlled environment.
- Refinement based on proof-of-concept findings to enhance detection accuracy and real-
time adaptability.
- Detailed understanding of adaptive mechanisms for LLMs and machine learning models
to keep up with evolving threats.
Additional Information / Knowledge Required:
The project will require knowledge of machine learning, deep learning, network security,
cybersecurity principles, software development and the latest advancements in artificial
intelligence aimed at threat detection. Developing an understanding of how different systems
such as machine learning integrate into a SOC system/SIEM is also essential.
Information Sources that Provide a Context for the Project:
Observations on how the deployment of LLMs may parallel usage within SOC operations can
be drawn from the following literature:
- Using a number of different LLM models to perform pentesting, “Using LLMs in pentesting
shares many similarities using LLMs in SOC operations” [1]
- Example of creating and using an LLM (BERT model) to be used for intrusion detection [2]
Ethan Campbell
 40483972
-
 Comparing LLM model to conventional ML and DL models [3]
-
 Possible way to mitigate out-of-date problem without retraining the entire model using
the RAG methodology [4]
- Extending the context window by millions of tokens – another approach to adding new
information without retraining the model [5]
- Overview of SIEM systems and how they will be affected by evolving technology such as
LLMs, ML, DL [6]
- Exploring GPT and BARD Strengths and limitations in discovering CWEs under various
threat models [7]
- Insights about how LLMs could be used for risk management/categorisation, and how
anomaly detection for network event logs could be implemented [8]
The Importance of the Project:
By introducing adaptability and real-time analysis through LLMs and machine learning the
importance of this project underscores the importance of cyber defence mechanisms, which
could be greatly improved by incorporating real-time, adaptable LLMs - potentially enhancing
the efficacy of SOC operations. The project aims to provide new insights into the combination
of LLMs, machine learning and its incorporation into a SOC system.
The Key Challenge(s) to be Overcome:
- Ensuring the LLM can adapt to evolving threats in real-time, possibly utilising the
strengths of both an LLM and machine learning within SOC operations.
- Developing a system that maintains high accuracy, e.g. LLMs are known to hallucinate.
- Retaining system adaptability in the face of an ever-changing threat landscape.
- Processing large volumes of network data typical to SOC environments efficiently.
References
[1] K. Shashwat et al., “A Preliminary Study on Using Large Language Models in Software
Pentesting,” Jan. 2024, Accessed: Feb. 29, 2024. [Online]. Available:
https://arxiv.org/abs/2401.17459v1
[2] H. Lai, “Intrusion Detection Technology Based on Large Language Models,” 2023
International Conference on Evolutionary Algorithms and Soft Computing Techniques
(EASCT), pp. 1–5, Oct. 2023, doi: 10.1109/EASCT59475.2023.10393509.
[3] M. A. Ferrag et al., “Revolutionizing Cyber Threat Detection with Large Language Models: A
privacy-preserving BERT-based Lightweight Model for IoT/IIoT Devices,” Jun. 2023, Accessed:
Feb. 29, 2024. [Online]. Available: http://arxiv.org/abs/2306.14263
[4] “Retrieval augmented generation: Keeping LLMs relevant and current - Stack Overflow.”
Accessed: Feb. 28, 2024. [Online]. Available: https://stackoverflow.blog/2023/10/18/retrieval-
augmented-generation-keeping-llms relevant-and-current/
[5] Y. Ding et al., “LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens,” Feb.
2024, Accessed: Feb. 29, 2024. [Online]. Available: http://arxiv.org/abs/2402.13753
[6] J. M. López Velásquez, S. M. Martínez Monterrubio, L. E. Sánchez Crespo, and D. Garcia
Rosado, “Systematic review of SIEM technology: SIEM-SC birth,” Int J Inf Secur, vol. 22, no. 3,
pp. 691–711, Jun. 2023, doi: 10.1007/S10207-022-00657-9/METRICS.
[7] S. Paria, A. Dasgupta, and S. Bhunia, “DIVAS: An LLM-based End-to-End Framework for
SoC Security Analysis and Policy-based Protection,” Aug. 2023, Accessed: Feb. 29, 2024.
[Online]. Available: http://arxiv.org/abs/2308.06932
[8] F. N. Motlagh, M. Hajizadeh, M. Majd, P. Najafi, F. Cheng, and C. Meinel, “Large Language
Models in Cybersecurity: State-of-the-Art,” Jan. 2024, Accessed: Feb. 29, 2024. [Online].
Available: http://arxiv.org/abs/2402.00891
B
Diary Examples
SCHOOL OF COMPUTING
Ethan Campbell
 40483972
Student: Ethan Campbell
Date: 08/04/2024
Objectives:
PROJECT DIARY
Supervisor: Sami Ashkan
•
•
•
Try to replicate the newly released “Harnessing the Advanced Capabilities of LLM for
Adaptive Intrusion Detection Systems” paper experiment that had a similar goal to my
project
Document results for LLM training involving CICDDoS 2019 dataset
Explain why datasets were used compared to others in more detail
Progress:
•
•
•
Was able to replicate experiment as closely as the paper allowed, getting basically the
same results, e.g. in paper, accuracy = 98.01, precision = 98.31 and in the replication
accuracy = 98.22, precision = 97.89
For CICDDoS 2019, tables with model results and a table showing dataset distribution and
sample sizes were created
Expanded explanation on why datasets were used as per supervisor comment, and added
an overall explanation of my aims in using datasets at the start of the dataset section
Supervisor’s Comments:
•
•
For each dataset, you should provide an overview of the number of records and features
and labels. It would be nice if you have some justification for why you chose the dataset
that you chose. It can be another paper, some forum? or based on specific search that you
have done.
You do not need to provide a lot of details about the dataset. However, it would be nice if
the dataset is not very common like KDD 2009, you provide more details of the features
and lables. For features and labels provide what they mean you do not need to specify
the exact name that dataset uses.
SCHOOL OF COMPUTING
PROJECT DIARY
Student: Ethan Campbell
 Supervisor: Sami Ashkan
Date: 10/09/2024
Objectives:
•
 Conduct a new experiment using TON IoT and CICDDoS2019 datasets, where specific
attack label is excluded from training data, and included in testing data to see what way it
is categorised
•
 Compare this experiment to results using the best traditional method XGBoost
Progress:
Ethan Campbell
 40483972
•
•
LLM model had very accurate results – the test involved two possible outcomes 0 benign,
1 malicious (encompassing 4 labels as 1), and for example in the first test "ddos" records
were excluded in training. had 0.9967 accuracy classifying ddos as “1”/malicious
I did the same label exclusion training/zero-day test using XGBoost, where that approach
showed 0.9067 (i.e. 2720/3000 were classified as 1/malicious)
Supervisor’s Comments:
•
•
While the performance of LLMs and traditional methods of IDS is similar, zero-day attacks
could be another aspect to explore
other simple algorithms like XGBoost would obtain excellent results in the order 99.XX %
accuracy. However, it would be nice to see if they also lose their competitiveness when
you test them for zero-day attacks. That is if you find time to finalize the experiments for
IoT dataset first.
