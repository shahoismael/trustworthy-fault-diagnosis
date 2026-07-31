# Unsupervised Anomaly Detection in Process-Complex Industrial Time Series: A Real-World Case Study

1<sup>st</sup> Sergej Krasnikov<sup>\*</sup> Universitat Augsburg¨ Augsburg, Germany krasnikov.research@gmail.com 0009-0008-1034-7290

4<sup>th</sup> Michael Heider Universitat Augsburg¨ Augsburg, Germany michael.heider@uni-a.de 0000-0003-3140-1993

2<sup>nd</sup> Lukas Meitz<sup>\*</sup> Technische Hochschule Augsburg Augsburg, Germany Lukas.Meitz@tha.de

5<sup>th</sup> Thorsten Scholer¨ Technische Hochschule Augsburg Augsburg, Germany email address or ORCID

3<sup>rd</sup> Samineh Bagheri Independent Researcher Karlsruhe, Germany email address or ORCID

6<sup>th</sup> Jorg H ¨ ahner ¨

Universitat Augsburg¨

Augsburg, Germany

joerg.haehner@uni-a.de

0000-0003-0107-264X

<sup>\*</sup>These authors contributed equally to this work.

Abstract—Industrial time-series data from real production environments exhibits substantially higher complexity than commonly used benchmark datasets, primarily due to heterogeneous, multi-stage operational processes. As a result, anomaly detection methods validated under simplified conditions often fail to generalize to industrial settings. This work presents an empirical study on a unique dataset collected from fully operational industrial machinery, explicitly capturing pronounced processinduced variability.

We evaluate which model classes are capable of capturing this complexity, starting with a classical Isolation Forest baseline and extending to multiple autoencoder architectures. Experimental results show that Isolation Forest is insufficient for modeling the non-periodic, multi-scale dynamics present in the data, whereas autoencoders consistently perform better. Among them, temporal convolutional autoencoders achieve the most robust performance, while recurrent and variational variants require more careful tuning.

Index Terms—Industrial, Anomaly Detection, Autoencoders, Complexity

## I. INTRODUCTION

Industrial environments generate large volumes of multivariate time-series data from heterogeneous sensors monitoring machinery and its executed processes. These streams are often high-dimensional, non-periodic, and exhibit multi-scale temporal dynamics, reflecting complex operational sequences and noise-prone conditions. Detecting deviations from normal behavior in such settings is a key prerequisite for reliable Machine Health monitoring and Predictive Maintenance.

Classical anomaly detection methods such as Isolation Forest (IF) are attractive in industrial practice due to their simplicity, low computational cost, and label-free training. However, they typically rely on learning static feature distributions and struggle when normal behavior is defined by non-stationary, multi-stage processes with variable temporal structure. This raises the question of whether classical methods remain viable under pronounced process-induced complexity, or whether representation-learning models are required.

Our study is built upon a proprietary dataset, that has been collected from more than 100 connected devices of the same type. This real-world dataset contains all of the complexity that naturally comes with operation and usage in production scenarios, which makes it a valuable resource for studying the effect of complexity on different model architectures.

To address this subject effectively, we evaluate the capability of a simpler model, the Isolation Forest (IF), to serve as a basis of comparison and implement six Autoencoder (AE) configurations: three standard and three variational—implemented with Temporal Convolutional Network (TCN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) architectures. The models are trained and tested on our proprietary industrial dataset characterized by non-periodic, multiscale process dynamics and sensor-rich recordings, reflecting typical challenges in machine monitoring environments.

In summary, this work makes the following contributions:

1) We systematically evaluate anomaly detection methods of varying complexity on industrial time-series.

2) We present a controlled empirical study comparing autoencoder architectures—convolutional versus recurrent, deterministic versus variational—in their ability to model process-complex (see II-B) industrial sequences, addressing a gap in the literature where such systematic comparisons remain rare.

3) We show that architectural alignment with temporal structure is more critical than model complexity: convolutional networks outperform recurrent alternatives through better capture of multi-scale process dynamics in real-world industrial data.

Our findings highlight architectural and representational trade-offs, with convolutional standard AEs demonstrating superior performance on complex industrial data. While the results are derived from a single industrial application, the observed performance differences are driven by structural properties—non-periodicity, variable phase ordering, and process-induced non-stationarity—that are common across many real-world industrial processes.

## II. BACKGROUND

Anomaly detection in time-series data has been extensively studied across domains such as healthcare, manufacturing, and infrastructure monitoring. In industrial settings, both classical and deep learning–based approaches are commonly employed, often under limited labeling and strict deployment constraints. However, a systematic understanding of how different model classes and architectures behave under realistic, processinduced complexity remains limited.

## A. Classical and Autoencoder-Based Anomaly Detection

Classical anomaly detection methods, including IFs and related feature-based techniques, have long been used in industrial monitoring due to their simplicity, scalability, and low data requirements. These methods typically operate on static feature representations and implicitly assume stationarity, which limits their ability to model non-periodic, multistage process dynamics commonly observed in real production environments.

In recent years, deep learning approaches—particularly autoencoder-based models—have demonstrated their potential for unsupervised time-series anomaly detection. Autoencoders learn compact latent representations from unlabeled data and identify anomalies via reconstruction error [1]. Their effectiveness has been shown across a wide range of applications, including industrial systems and predictive maintenance. Recent surveys provide comprehensive overviews of deep anomaly detection methods and highlight the large design space of autoencoder architectures [2].

A variety of autoencoder variants have been proposed to model temporal dependencies in time-series data. Recurrent architectures based on LSTM and GRU units are widely used for sequential modeling and have been applied to feature extraction and anomaly detection in industrial and energy systems [3]. Extensions incorporating memory mechanisms [4] or attention-based designs, such as OmniAnomaly [5], aim to improve robustness in multivariate settings. In parallel, TCNs leverage dilated convolutions to capture long-range dependencies with stable training behavior and have shown strong performance in healthcare, manufacturing, and general anomaly detection tasks [6], [7], [8], [9].

Variational Autoencoders (VAEs) extend standard autoencoders by learning probabilistic latent representations and have been explored for dimensionality reduction [10], weakly supervised anomaly detection [11], and data augmentation [12]. Despite their theoretical appeal, empirical evidence for consistent advantages of variational formulations over standard AEs in industrial anomaly detection remains mixed.

Recent industrial-focused surveys further emphasize that many deep anomaly detection methods are still predominantly evaluated on simplified or benchmark-style datasets, limiting conclusions about their robustness under realistic operational variability [13].

## B. Need for Complexity-Aware Evaluation

Although a wide range of anomaly detection architectures has been proposed, few studies systematically compare classical and AE-based models across convolutional, recurrent, and variational designs under realistic industrial conditions. Prior work has noted that general-purpose models often underperform when confronted with heterogeneous, non-stationary processes [6].

Recent research has argued that such evaluation gaps are closely tied to unmodeled sources of machine- and processinduced complexity, motivating the use of explicit complexity taxonomies to reason about dataset characteristics and model assumptions [14]. Following this taxonomy, we call our dataset process-complex, as it includes complexity predominantly induced by the process that is being monitored. Addressing this gap is essential for deriving practical guidance on model selection for industrial anomaly detection beyond benchmarkoriented evaluations.

## III. METHOD

In order to contribute to the research gap identified in the previous section, we will implement and compare multiple proven models in the anomaly detection domain. This section presents the experimental methodology used to investigate the performance of different models and configurations for anomaly detection in complex industrial time-series data.

The evaluation follows a two-stage process. First, models are assessed for their reconstruction performance using process data captured from industrial machines. Second, topperforming models are applied to an independent and labeled dataset from the same type of machinery containing real anomalies, enabling quantitative evaluation of their anomaly detection capabilities. Using this process, our goal is to evaluate the anomaly detection capability of different models on a dataset with high process-complexity, reflecting a realistic industrial application.

## A. Description of the Used Dataset

The dataset used in this study originates from an industrial process involving automated product creation. It consists of multivariate time-series recordings collected from 118 fielddeployed machines, encompassing 334 individual process instances. Each instance captures the full temporal evolution of a manufacturing cycle, including signals from a heterogeneous set of sensors and actuators. An illustrative example of a multivariate process instance is shown in Figure 1, highlighting the variability and heterogeneity across sensor signals during a typical production cycle.

![](images/3d6429fa81e3d566370dd554f587b17244e56110bdd9ead4afb850e4fa0204e2.jpg)

Fig. 1: Plot of a process sample from the dataset, showing readings from four selected sensors captured during a single run executed on one machine. Each sensor records different physical quantities over time, exhibiting distinct behaviors including stable operational phases, gradual drifts, and abrupt changes indicative of process transitions or anomalies. The view of the data has been deliberately perturbed to protect sensitive operational information, while preserving the underlying structure and complexity.

<table><tr><td>Model</td><td>Encoder Configuration</td><td>Decoder Configuration</td></tr><tr><td>TCN-AE</td><td>Conv1D + TCN blocks + Avg Pooling</td><td>Inverse TCN blocks + Conv1D</td></tr><tr><td>TCN-VAE</td><td>Conv1D + TCN blocks + Avg Pooling + Normal ( $\mu$ ,  $\sigma$ )</td><td>Inverse TCN blocks + Conv1D</td></tr><tr><td>LSTM-AE</td><td>2 LSTM layers</td><td>2 LSTM layers</td></tr><tr><td>LSTM-VAE</td><td>2 LSTM layers + Normal ( $\mu$ ,  $\sigma$ )</td><td>2 LSTM layers</td></tr><tr><td>GRU-AE</td><td>2 GRU layers</td><td>2 GRU layers</td></tr><tr><td>GRU-VAE</td><td>2 GRU layers + Normal ( $\mu$ ,  $\sigma$ )</td><td>2 GRU layers</td></tr></table>

TABLE I: Architectural configurations of the AE models used in the study. VAE models include an encoded normal distribution signified by mean (µ) and standard deviation (σ).

This dataset exhibits several characteristics that distinguish it from commonly used benchmarks in time-series modeling. Most notably, it features numerous sensor readings, nonperiodic behavior, and multi-scale process dynamics. These attributes arise from the complex structure of the underlying process, which comprises multiple sequential and interleaved operations with variable durations and response patterns. The data is further affected by sensor inaccuracies and interdevice variability, introducing realistic noise patterns typical of industrial deployments.

In contrast to widely used benchmark datasets such as NASA Turbofan or bearing test rigs, which typically exhibit repetitive cycles or monotonic degradation patterns, the recorded processes show variable phase orderings, durations, and actuator interactions, violating common assumptions of temporal alignment and stationarity.

Following the complexity taxonomy proposed by [14], the dataset qualifies as process-complex due to the concurrent presence of actuator diversity, multi-phase control logic, heterogeneous timing profiles, and non-repetitive process executions. These properties make the dataset particularly challenging for classical anomaly detection methods that rely on static feature distributions, while favoring models capable of learning hierarchical and temporal representations.

To ensure data quality, only newly commissioned machines were selected and the recordings were limited to the first two months of operation. This sampling strategy reduces the likelihood of degraded behavior contaminating the training data, allowing the models to focus on learning normal operational patterns. Preprocessing steps included normalization of all sensor channels, selective downsampling to reduce redundancy, and segmentation using a sliding window to preserve local temporal dependencies.

## B. Model Architectures and Training

To assess which model types can capture process-induced complexity in industrial time-series data, we evaluate a classical anomaly detection baseline alongside a set of representation-learning models based on AEs. This enables a direct comparison between feature-agnostic methods and sequence-aware architectures under identical data conditions.

a) Baseline Model: Isolation Forest was selected as a representative classical baseline due to its widespread use in industrial monitoring and scalability to high-dimensional data, and ability to operate without labeled anomalies. The method isolates samples via random feature partitioning and is effective for detecting deviations in static feature distributions. Each process instance was represented using aggregated statistical features over the full sequence, following standard practice for classical anomaly detection on time-series data. This baseline serves to assess whether feature-agnostic methods remain viable for process-driven industrial data with complex temporal dynamics.

b) Autoencoder Architectures: As primary models, we implemented six AE variants: standard and variational based on TCN, LSTM, and GRU. These architectures cover both convolutional and recurrent design paradigms commonly used in time-series modeling. Encoder and decoder configurations are summarized in Table I.

All AE models were implemented in PyTorch and trained using a unified protocol. Hyperparameter optimization was conducted using Optuna, with standard AEs minimizing reconstruction error (MSE) and VAEs maximizing the Evidence Lower Bound (ELBO). Training was performed on a singlenode Databricks cluster with an NVIDIA T4 GPU.

Models were trained exclusively on data from normal machine operation to reflect realistic industrial conditions. Each variant was trained 50 times until convergence with different random seeds and hyperparameter configurations. The top five configurations per model, selected based on validation performance, were retained for downstream anomaly detection. The evaluated hyperparameter options are reported in Table II.

<table><tr><td>Hyperparameter Description</td><td>Values</td></tr><tr><td>Gradient Descent Learning Rate</td><td>(0.001, 0.0001)</td></tr><tr><td>Batch Size</td><td>{64, 128, 256, 512}</td></tr><tr><td>Input Sequence Length</td><td>{480, 600, 720, 840, 960, 1080}</td></tr><tr><td>Hidden Layer Size</td><td>{32, 64}</td></tr><tr><td>Number of encoder/decoder Layers</td><td>(1, 3)</td></tr><tr><td>Latent Dimension (VAE)</td><td>{16, 32, 64}</td></tr><tr><td>Kernel Size (TCN)</td><td>(2, 10)</td></tr><tr><td>Downsampling Factor (TCN)</td><td>{4, 6, 8, 10}</td></tr></table>

TABLE II: Hyperparameter search space for model training. Values represent ranges (min, max) for continuous parameters and discrete sets {...} for categorical parameters. Architecturespecific parameters are shown for VAE and TCN models.

## C. Model Evaluation on Anomaly Detection

To assess the practical utility of the trained models, we evaluate their anomaly detection capabilities using a separate, labeled dataset that was not involved in training or hyperparameter optimization. This evaluation dataset consists of 46 complete process instances, 22 of which were manually labeled as anomalous and 24 as normal. The dataset exhibits similar structure and sensor diversity as the training data but includes known deviations introduced through real operational failures or irregular process conditions.

Anomaly detection was performed by computing reconstruction error for each process instance. A decision threshold was then applied to the aggregated error to classify each instance as anomalous or normal. Since reconstruction errors vary across models and configurations, the threshold was optimized individually based on the separate dataset for each model to balance detection performance.

To this end, a multi-objective optimization was applied using NSGA-II to find Pareto-optimal threshold values across four standard classification metrics: precision, recall, accuracy, and F1-score. These metrics were selected to provide a comprehensive evaluation of model performance, particularly in the context of imbalanced classification tasks such as anomaly detection [15, p. 412].

Detection thresholds were optimized post hoc for each model to estimate best achievable anomaly detection performance and do not represent fixed deployment thresholds. The best-performing threshold for each model was retained to represent its potential performance in a real-world deployment scenario. This evaluation setup enables a direct comparison of models in terms of both their reconstruction-based learning capabilities and their practical effectiveness in detecting anomalous machine behavior.

## IV. RESULTS

Following the experimental methodology described in the previous section, multiple models of increasing complexity were trained and evaluated on the dataset. Table III summarizes the average performance across all approaches.

<table><tr><td rowspan="2">Model</td><td>Anomaly Detection</td><td>Reconstruction</td></tr><tr><td>F1 Score</td><td>MSE / ELBO ( $\times 10^{-4}$ )</td></tr><tr><td>IF</td><td> $0.120 \pm 0.126$ </td><td>-</td></tr><tr><td>TCN-AE</td><td> $\mathbf{0.991 \pm 0.009}$ </td><td> $\mathbf{0.22 \pm 0.06}$ </td></tr><tr><td>LSTM-AE</td><td> $0.853 \pm 0.102$ </td><td> $1.23 \pm 0.25$ </td></tr><tr><td>GRU-AE</td><td> $0.918 \pm 0.066$ </td><td> $0.84 \pm 0.15$ </td></tr><tr><td>TCN-VAE</td><td> $\mathbf{0.968 \pm 0.010}$ </td><td> $-7.29 \pm 0.47$ </td></tr><tr><td>LSTM-VAE</td><td> $0.945 \pm 0.027$ </td><td> $-1.89 \pm 0.16$ </td></tr><tr><td>GRU-VAE</td><td> $0.876 \pm 0.085$ </td><td> $-2.24 \pm 0.07$ </td></tr></table>

TABLE III: Average performance metrics for the evaluated models in the presented industrial case study. Reported values are means ± standard deviation (MSE for standard AE, ELBO for VAE).

## A. Isolation Forest (Baseline)

The results reveal that IF is fundamentally unsuited for this industrial anomaly detection task. Across all runs, the approach achieved an average F1-score of $0 . 1 2 0 \pm 0 . 1 2 6$ Performance varied dramatically between runs: five runs detected no anomalies whatsoever, while the best performing run achieved only an F1-Score of 0.308 by detecting four out of 22 anomalous processes. Although the models achieved a precision of 1.0 in the runs in which anomalies were detected, the maximum recall of 0.182 indicates systematic failure to identify the majority of anomalous instances.

## B. Autoencoder Models

a) Reconstruction Performance: All AE architectures learned to replicate the multivariate sensor sequences, demonstrating their fundamental suitability for this complex industrial dataset. However, reconstruction fidelity varied significantly between architectures.

The TCN-AE consistently achieved the lowest reconstruction error, outperforming both recurrent alternatives by a substantial margin. Beyond superior reconstruction quality, the convolutional model exhibited the highest consistency across training runs, underscoring its robustness to hyperparameter variations. GRU-AE achieved moderate reconstruction quality, while LSTM-AE showed both higher reconstruction error and greater variability across runs, indicating less stable convergence behavior.

b) Anomaly Detection Performance: The primary evaluation criterion is practical applicability for anomaly detection. The results demonstrate a substantial improvement over the IF across all architectures. TCN-AE achieved most robust performance in this setting with some models even achieving an F1-Score of 1, detecting all anomalies correctly. GRU-AE followed with strong but more variable results, while LSTM-AE showed the highest variability and was more prone to both false positives and negatives. Figure 2 illustrates these performance differences through F1-score distributions across the top 5 configurations for each architecture, for both standard and variational variants.

## C. Variational Autoencoder

VAEs consistently underperformed their deterministic counterparts in anomaly detection. While LSTM-VAE and TCN-VAE achieved respectable performance, they fell short of the corresponding standard AEs. GRU-VAE showed the most pronounced degradation.

TCN-VAE in particular showed occasional degraded reconstructions marked by noise artifacts or spikes, reflected in its high ELBO variance. Among variational variants, LSTM-VAE achieved the best ELBO and most stable training behavior, though this did not translate to superior anomaly detection performance compared to the TCN models.

Training time analysis over variational and standard AEs revealed practical trade-offs: GRU-based models converged fastest (15–30 minutes), while LSTM and TCN variants required 40–60 minutes per model. However, given the scales at which industrial applications operate, these differences are likely negligible in practice, as inference performance is the more critical factor for deployment.

## V. DISCUSSION

This section interprets the experimental results with respect to the two central research questions, focusing on how architectural inductive biases affect anomaly detection performance under pronounced process-induced complexity.

![](images/546533a51f2346e51b0e0afdae90be71a901b450980633074d9863d6617915fb.jpg)  
Fig. 2: Differences in anomaly detection performance across models with TCN variants achieving the most desirable results. Shown are the F1-scores of the top 5 models for each architecture. Note: IF is excluded from this visualization due to its significantly lower performance (cf. Table III), which would distort the scale and clarity of the plot.

The reported results reflect best-case detection performance under controlled evaluation and should be interpreted as an assessment of model capacity rather than as a finalized deployment configuration. Rather than reiterating quantitative results, the discussion emphasizes explanatory factors and practical implications for model selection in industrial monitoring scenarios.

While the presented results are derived from a single industrial application, the observed performance differences are driven by structural properties—non-periodicity, variable phase ordering, and process-induced non-stationarity—that are common across many real-world industrial processes.

## A. Architectural Effects under Process-Induced Complexity

The results indicate a clear advantage of convolutional AE architectures, particularly TCN-based models, when applied to complex industrial time-series data. Their superior performance can be attributed to the ability of temporal convolutions to capture local and mid-range temporal dependencies while remaining robust to variable phase lengths and non-repetitive process structures. This inductive bias is well aligned with the characteristics of the studied dataset, which exhibits heterogeneous, multi-stage operational behavior rather than periodic or stationary dynamics.

Recurrent architectures based on LSTM and GRU were decent at modeling the data but showed increased sensitivity to hyperparameter choices and training conditions. Their sequential processing nature makes them more susceptible to instabilities when process stages vary in duration or ordering, which limits robustness in settings where extensive tuning is impractical.

Across all architectures, standard AEs consistently outperformed their variational counterparts. While VAEs provide theoretical advantages through regularized latent spaces, the introduced stochasticity proved detrimental in a reconstructiondriven anomaly detection setting. In particular, increased variance and reconstruction artifacts reduced threshold stability, indicating that precise deterministic reconstruction is more critical than latent expressiveness for detecting subtle process deviations.

Taken together, these observations suggest that architectural robustness and alignment with process structure are more decisive than model complexity alone. For process-complex industrial time-series, convolutional AEs offer a favorable balance between representational power, stability, and practical deployability.

## B. Limitations and Outlook

This study is limited to a single proprietary industrial dataset, which restricts direct comparison with publicly available benchmarks. Consequently, the findings should be interpreted as qualitative guidance on architectural suitability under process-induced complexity rather than as universal performance rankings. In addition, the evaluated model set was restricted to classical baselines and AE-based architectures.

While attention-based architectures such as transformerbased AEs are promising for modeling state-rich or highly variable processes, their data requirements, tuning complexity, and computational cost often conflict with the limited fault data and resource constraints typical of industrial anomaly detection deployments.

## VI. CONCLUSION

This work evaluated anomaly detection models on a realworld industrial time-series dataset characterized by pronounced process-induced complexity, including non-periodic behavior, heterogeneous sensors, and multi-stage operational dynamics. Such characteristics are largely absent from commonly used benchmark datasets and pose a substantial challenge for anomaly detection methods.

By comparing a classical IF baseline with multiple AE architectures, the study demonstrates that feature-agnostic classical methods are insufficient for reliably modeling complex industrial processes. AE-based models consistently achieved superior performance, with architectural choice playing a decisive role. In particular, TCN-AEs provided the most robust and stable results, while recurrent and variational variants required more careful tuning and showed higher performance variability.

These findings underline, within the studied industrial application, the necessity of representation-learning approaches for anomaly detection in process-complex industrial time-series and highlight temporal convolution as a particularly effective design choice. Overall, the results provide practical guidance for model selection in industrial monitoring scenarios where realistic process variability must be addressed.

## REFERENCES

[1] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, Learning Internal Representations by Error Propagation. Cambridge, MA, USA: MIT Press, 1986, p. 318–362.

[2] Z. Zamanzadeh Darban, G. I. Webb, S. Pan, C. Aggarwal, and M. Salehi, “Deep learning for time series anomaly detection: A survey,” ACM Computing Surveys, vol. 57, no. 1, p. 1–42, Oct. 2024. [Online]. Available: http://dx.doi.org/10.1145/3691338

[3] W. Yu, I. Kim, and C. Mechefske, “Analysis of different rnn autoencoder variants for time series classification and machine prognostics,” Mechanical Systems and Signal Processing, vol. 149, p. 107322, 2021.

[4] H. Gao, B. Qiu, R. J. D. Barroso, W. Hussain, Y. Xu, and X. Wang, “TSMAE: A novel anomaly detection approach for internet of things time series data using memory-augmented autoencoder,” IEEE Transactions on Network Science and Engineering, vol. 10, pp. 2978–2990, 2023.

[5] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust anomaly detection for multivariate time series through stochastic recurrent neural network,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, ser. KDD ’19. New York, NY, USA: Association for Computing Machinery, 2019, p. 2828–2837. [Online]. Available: https://doi.org/10.1145/3292500.3330672

[6] M. Thill, W. Konen, H. Wang, and T. Back, “Temporal convolutional¨ autoencoder for unsupervised anomaly detection in time series,” Applied Soft Computing, vol. 112, p. 107751, 2021. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1568494621006724

[7] S. Asahi, C. Karadogan, S. Tamura, S. Hayamizu, and M. Liewald, “Process data based estimation of tool wear on punching machines using TCN-autoencoder from raw time-series information,” IOP Conference Series: Materials Science and Engineering, vol. 1157, 2021.

[8] J. Park, Y.-S. Park, and C.-I. Kim, “TCAE: Temporal convolutional autoencoders for time series anomaly detection,” 2022 Thirteenth International Conference on Ubiquitous and Future Networks (ICUFN), pp. 421–426, 2022.

[9] S. Gopali, F. Abri, S. Siami-Namini, and A. Siami Namin, “A comparative study of detecting anomalies in time series data using lstm and tcn models,” arXiv preprint arXiv:2112.09293, 2021. [Online]. Available: https://arxiv.org/abs/2112.09293

[10] W. Todo, B. Laurent, J.-M. Loubes, and M. Selmani, “Dimension reduction for time series with variational autoencoders,” ArXiv, vol. abs/2204.11060, 2022.

[11] Z. Wu, L. Cao, Q. Zhang, J. Zhou, and H. Chen, “Weakly augmented variational autoencoder in time series anomaly detection,” ArXiv, vol. abs/2401.03341, 2024.

[12] S. Dodda, “Exploring variational autoencoders and generative latent time-series models for synthetic data generation and forecasting,” 2024 Control Instrumentation System Conference (CISCON), pp. 1–6, 2024.

[13] L. Meitz, J. Senge, T. Wagenhals, T. Scholer, J. H¨ ahner, J. Edinger,¨ and C. Krupitzer, “A literature review framework and open research challenges for predictive maintenance in industry 4.0,” Computers and Industrial Engineering, vol. 206, p. 111193, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0360835225003390

[14] L. Meitz, M. Heider, T. Scholer, and J. H¨ ahner, “A taxonomy for¨ complexity estimation of machine data in machine health applications,” in Proceedings of the 21st International Conference on Informatics in Control, Automation and Robotics - Volume 1: ICINCO, INSTICC. SciTePress, 2024, pp. 341–350.

[15] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA: MIT Press, 2016, eLBO definition on p. 624.