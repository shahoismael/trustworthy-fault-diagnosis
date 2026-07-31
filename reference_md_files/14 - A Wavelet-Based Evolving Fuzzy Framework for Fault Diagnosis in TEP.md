Article

# A Wavelet-Based Evolving Fuzzy Framework for Fault Diagnosis in the Tennessee Eastman Process

Marco Antonio Márquez-Vera <sup>1,</sup>\* , Jorge A. Ruiz-Vanoye <sup>1</sup> , Carlos Antonio Márquez-Vera <sup>2</sup> , Alfian Ma’arif <sup>3</sup> and Edith Mendoza-Ramírez <sup>1</sup>

Mechatronics and Automotive Engineering, Polytechnic University of Pachuca,

Zempoala 43830, Hidalgo, Mexico; jorge@ruizvanoye.com (J.A.R.-V.); edith.mendoza@upp.edu.mx (E.M.-R.)

<sup>2</sup> Chemical Engineering, Universidad Veracruzana, Poza Rica 93390, Veracruz, Mexico; carmarquez@uv.mx

3 Electrical Engineering, Universitas Ahmad Dahlan, Yogyakarta 55166, Indonesia; alfian.maarif@te.uad.ac.id

Correspondence: marquez@upp.edu.mx

## Abstract

Evolving fuzzy systems (EFS) offer an incremental learning, making them promising for fault diagnosis (FD) in industrial processes, where unknown faults and changing operation conditions are common. The evolving fuzzy structure enables incremental rule adaptation while maintaining interpretability and reduced computational complexity compared with deep learning approaches. However, the performance of EFS depends heavily on the preprocessing of input data. This study evaluates eight preprocessing strategies for EFS applied to the Tennessee Eastman benchmark process. A one-vs-rest EFS architecture was implemented for ten representative faults (IDV1, IDV2, IDV4, IDV5, IDV6, IDV7, IDV8, IDV10, IDV13 and IDV14) in order to make a comparison with other FD techniques. This approach uses seven variables selected by using the least angle regression. Preprocessing methods were applied to highlight fault signatures. Using the Daubechies-4 in the preprocessing achieved the best overall F1-score (73.68%) with a sensitivity of 97.37%, outperforming the no-preprocessing baseline (F1 = 70.67%). Per-fault analysis showed high performance for faults IDV6, IDV7, and IDV14, while IDV1, IDV2, IDV5, and IDV8 exhibited high sensitivity but lower specificity. These findings indicate that wavelet preprocessing significantly enhances EFS for FD, and that the choice of wavelet should be guided by application priorities: Daubechies-4 is recommended for maximum detection and fewer false alarms. The obtained results demonstrate that wavelet preprocessing substantially improves classification robustness and fault discrimination compared with the non-preprocessed baseline.

## Check for updates

Academic Editors: Martin Valtierra-Rodriguez, Maximiliano Bueno-Lopez and Afshin Rahimi

Received: 8 May 2026 Revised: 1 June 2026 Accepted: 10 June 2026 Published: 17 June 2026

Copyright: © 2026 by the authors. Licensee MDPI. Basel, Switzerland This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license.

Keywords: evolving fuzzy systems; fault diagnosis; one-vs-rest classification; industrial process monitoring

## 1. Introduction

In modern industrial processes, fault diagnosis (FD) is essential to detect anomalies that can lead to equipment degradation, safety risks, environmental harm, expensive corrective maintenance, and expensive unplanned shutdowns [1,2]. The growing complexity of industrial systems, tighter profit margins, stricter product specifications, and more stringent safety regulations have fueled the need for advanced, reliable, and interpretable fault diagnosis solutions [3].

The Tennessee Eastman (TE) process continues to be a standard benchmark for evalu ating FD algorithms [4]. The TE process, which was first proposed by Downs and Vogel in 1993 [5], simulates a realistic chemical plant with 20 predefined faults, 41 measured variables, 12 manipulated variables, and five major unit operations. Its widespread use has enabled comparisons of various FD approaches, ranging from statistical methods (PCA, PLS, ICA) [6] to machine learning methods (support vector machines, neural networks) [7,8] and fuzzy systems [9,10].

Conventional FD techniques often operate in offline batch mode, where a model is trained once using historical data and then deployed for monitoring [11]. While effective in stationary settings, batch-trained models suffer from several drawbacks in practical industrial environments:

Incapacity to handle unknown faults: The model typically fails to identify a fault not presented in the training data, often misclassifying it as normal operation [12].

Lack of flexibility: Changes in process dynamics, raw material characteristics, environmental factors, set-points, or equipment wear can render the original model obsolete over time [13].

Computational limitations: Retraining from scratch on growing historical datasets becomes increasing costly in terms of memory and time [14].

Evolving fuzzy systems (EFS) have emerged as an interesting approach for adaptive learning from data streams in order to overcome these constraints [15,16]. Unlike traditional fuzzy systems built in batch mode, EFS offer:

Incremental learning: Samples are processed one at a time (or in blocks) without storing the entire dataset, making EFS suitable for streaming data and big data scenarios [17]; no epochs for training are needed.

Structural evolution: New rules can be added when new operating conditions or system states emerge, and outdated or redundant rules can be eliminated to preserve compactness and interpretability [18].

Rapid adaptation: Recursive updating of consequent parameters (e.g., linear functions in Takagi–Sugeno systems) enables fast adaptation to process changes [19].

Several EFS architectures have been proposed in the literature, including eTS [20], FLEXFIS [21], SAFIS [22], PANFIS [23], and GENEFIS [24]. Extensive surveys are available in [25,26]. Significantly, EFS has been effectively applied to fault diagnosis in various industries, such as rolling mills [27], chemical processes like the Tennessee Eastman process [28], and NO emission monitoring [29].

Incipient faults, those that initially cause minor deviations from normal behavior and are often masked by measurement noise, control actions, and typical process variability, are difficult to detect until they produce noticeable process changes. These faults represent a recurring challenge in FD [30]. Wavelet transforms offer multi-resolution analysis capable of simultaneously capturing high-frequency transients and low-frequency trends, making them highly successful for FD [31].

Wavelet-based techniques have produced impressive results on the TE process. For example, D’Angelo et al. [32] achieved an average sensitivity of 88.52% by combining fuzzy clustering with wavelets and beta distributions. Hajihosseini et al. [33] achieved a sensitivity of 98.6% for specific faults by using discrete wavelet transform (DWT) to extract bi-dimensional features fed into a neural network classifier. Wavelet-based deep learning architectures have been proposed for TE fault diagnosis [34,35].

In [36], Daubechies-4 and Polywoq4 wavelets were used to preprocess TE signals prior to classification using an inverse fuzzy model. Although DWT with eight decomposition levels proved particularly effective for highlighting fault signatures, it required 256 samples (approximately 12.8 h) before fault isolation can be performed. This delay motivated the investigation of alternative preprocessing techniques and an examination of detection speed, sensitivity, and specificity.

Despite the individual successes of wavelet preprocessing and EFS, several research gaps remain:

Limited integration: Few studies have systematically integrated EFS with wavelet preprocessing for adaptive FD, despite the complementary advantages of both methods.

Absence of comparative studies: No thorough comparison exists within the EFS framework of various preprocessing techniques (smoothing, differentiation, peak detection, and standard wavelets).

Open-source implementations: Most EFS implementations are proprietary or only accessible in commercial environments like MATLAB© 2015b, hindering reproducibility and adoption in academic and industrial settings.

This article fills these gaps with the following contributions:

Integration of EFS with wavelet preprocessing: A complete, modular implementation of EFS with one-vs-rest architecture for multi-fault classification in the TE process.

Comparative study: Evaluation of eight preprocessing methods, including none, smoothing, peak detection, balanced preprocessing, strong smoothing, and the Daubechies, Haar, and Coiflet wavelets (all at level 4).

Open-source implementation: The entire codebase is implemented in Octave©, a free and open-source MATLAB© alternative, ensuring full reproducibility.

Detailed per-fault analysis: Results are reported not only as averages but also per individual fault (IDV1, IDV2, IDV4, IDV5, IDV6, IDV7, IDV8, IDV10, IDV13 and IDV14), revealing which faults are most challenging for EFS.

Unlike conventional FD approaches based exclusively on static classifiers or deep learning architectures requiring extensive retraining, the proposed methodology integrates evolving fuzzy systems with wavelet-based preprocessing in an adaptive framework oriented to industrial fault diagnosis. The contribution of this work is not centered on the development of a new standalone classifier, but rather on combining online rule evolution, incremental structural adaptation, and multi-resolution signal preprocessing for handling nonlinear process dynamics under changing operating conditions.

Furthermore, the proposed framework evaluates the influence of different wavelet families on the performance of evolving fuzzy diagnosis systems on the Tennessee Eastman benchmark. The results demonstrate that the preprocessing stage strongly affects the sensitivity, specificity, and robustness of the evolving classifier. Therefore, this work provides practical insights regarding the integration of signal processing and adaptive fuzzy inference for fault diagnosis in complex industrial systems.

The experimental results, summarized in Table 1, demonstrate that:

The Daubechies wavelet approach greatly outperforms the baseline without preprocessing (F1 = 70.67%), achieving the best overall F1-score (75.50%) with sensitivity of 97.17%.

When false alarms are expensive, the Haar wavelet is preferred because it has the highest specificity (44.37%).

The Coiflet wavelet performs poorly (F1 = 65.15%), especially on fault IDV7 (sensitivity = 17.33%), indicating that this application is not a good fit for it.

While some faults (IDV1, IDV2, IDV5, IDV8 and IDV13) are still difficult to diagnose, frequently showing 100% of sensitivity but the specificity is around 50%, others (IDV6, IDV7, and IDV14) are diagnosed with extremely high accuracy (F1 > 94% with the best method).

Table 1. Selected faults in the TE process.

<table><tr><td>Fault ID</td><td>Description</td><td>Type</td><td>Difficulty</td></tr><tr><td>IDV1</td><td>A/C feed ratio change</td><td>Step</td><td>Easy</td></tr><tr><td>IDV2</td><td>B composition change</td><td>Step</td><td>Easy</td></tr><tr><td>IDV4</td><td>Cooler water temperature change</td><td>Step</td><td>Easy</td></tr><tr><td>IDV5</td><td>Condenser temperature change</td><td>Step</td><td>Easy</td></tr><tr><td>IDV6</td><td>A feed loss</td><td>Step</td><td>Medium</td></tr><tr><td>IDV7</td><td>Separator pressure change</td><td>Step</td><td>Medium</td></tr><tr><td>IDV8</td><td>Reactant composition change</td><td>Random</td><td>Medium</td></tr><tr><td>IDV10</td><td>C feed temperature</td><td>Random</td><td>Hard</td></tr><tr><td>IDV13</td><td>Slow drift in reaction kinetics</td><td>Drift</td><td>Hard</td></tr><tr><td>IDV14</td><td>Sticking valve</td><td>Structural</td><td>Hard</td></tr></table>

These results emphasize how crucial it is to adjust preprocessing techniques to particular fault kinds and application needs (sensitivity vs. specificity trade-offs).

The remainder of the paper is structured as follows. Section 2 describes the Tennessee Eastman process, the selected faults and dataset, the principles of evolving fuzzy systems, the one-versus-rest architecture, the evaluated wavelet preprocessing techniques, and the experimental setup. Section 3 reports and discusses the results, including per-fault analysis and method comparisons. Section 4 presents the conclusions and contributions. Finally, Section 5 discusses limitations and future directions.

## 2. Materials and Methods

This section describes the materials and methods used in this investigation. First, the Tennessee Eastman (TE) benchmark process is introduced, including its key components, the ten faults selected for comparison with similar techniques, and the characteristics of the dataset used. Second, the principles of evolving fuzzy systems (EFS) are presented, including the recursive parameter adaptation mechanism, the Takagi–Sugeno rule struc ture, the one-vs-rest architecture for multi-fault classification, and the rule evolution and pruning criteria. Third, the wavelet-based preprocessing techniques evaluated in this study are described, including the Haar, Coiflet, and Daubechies wavelets, along with several traditional preprocessing methods that served as baselines. Finally, the experimental setup, including data partitioning, hyperparameter selection, evaluation metrics, and implementation details in the GNU Octave© 5.1.0 environment, is described in detail.

## 2.1. Tennessee Eastman Process Description

The Tennessee Eastman (TE) process is a widely recognized benchmark for fault diagnosis (FD) studies [5]. Originally it was developed by Downs and Vogel; it simulates a realistic chemical plant with five major unit operations: a reactor, a product condenser, a recycle compressor, a vapor–liquid separator, and a product stripper [5]. The process produces two liquid products (G and H) from four gaseous reactants (A, C, D, and E), with an inert component B present in the feed streams. The overall reaction scheme is as follows:

$$
A (g) + C (g) + D (g) \rightarrow G (l i q),\tag{1}
$$

$$
A (g) + C (g) + E (g) \rightarrow H (l i q),\tag{2}
$$

$$
A (g) + E (g) \rightarrow F (l i q),\tag{3}
$$

$$
3 D (g) \rightarrow 2 F (l i q),\tag{4}
$$

The TE process model includes 41 measured variables $\left( x _ { 1 } \ \mathrm { t o } \ x _ { 4 1 } \right)$ and 12 manipulated variables. The first 22 measurements are sampled every 3 min, while the remaining 19 composition variables have longer delay times (6–15 min) due to analysis requirements [4]. A total of 20 predefined faults (IDV1 to IDV20) are available for simulation, covering different fault types including step changes, random variations, slow drifts, and valve sticking [5].

In this work, ten representative faults were selected based on their prevalence in the literature, to compare results, and because of their varying difficulty levels $[ 6 , 7 , 3 2 ]$ . Table 1 summarizes the selected faults and their characteristics.

The TE process simulations under both normal operating conditions and each of the ten chosen faults provided are used. Three severity levels are available for each fault. In accordance with standard procedure, the medium severity level was chosen to balance practical relevance and detection difficulty [36]. Each simulation run contains 2001 samples. Faults are introduced at sample 501 and removed at sample 1011 (500 normal samples, 511 fault samples, and 990 post-fault samples). The sampling time in the TE process is three minutes. The data used were obtained from simulations because original data available for non-Simulink© users contain fault data, and the post-fault behavior is also relevant for FD.

Seven measured variables were chosen as inputs for the EFS classifier in accordance with the variable selection methodology described in [15]; the variables are $x _ { 1 } , x _ { 4 } , x _ { 1 8 } , x _ { 2 1 } .$ x<sub>23</sub>, x<sub>25</sub>, and $x _ { 2 8 }$ . Least angle regression (LARS) was used to determine which variables were most informative for fault isolation. After normalizing the data to zero mean and unit standard deviation using parameters estimated from normal operating data, clipping was used to restrict extreme values to the range [−3, 3].

LARS builds a model incrementally by adding the predictor most correlated with the current residual. Also, LARS moves in a direction equiangular between the selected predictors, ensuring monotonic progression of the $L _ { 1 }$ norm of the coefficient vector while maintaining the same computational cost as ordinary least squares. The algorithm proceeds as follows:

1. Computation of the correlations between each predictor and the current residual: $\mathbf { \boldsymbol { c } } = \mathbf { \boldsymbol { X } } ^ { \mathrm { { T } } } ( \mathbf { \boldsymbol { y } } - \mathbf { \boldsymbol { X } } \beta )$

2. To identify the most correlated predictor: $\hat { \mathbf { c } } = \mathbf { m a x _ { j } } | \mathbf { c _ { j } } |$

3. To move the coefficient vector in the least angle direction until another predictor attains the same correlation.

4. To add the new predictor to the active set and return to step 1.

For the TE process, LARS was applied to the full set of 41 measured variables. The seven variables with the largest absolute regression coefficients were from the solution of

$$
m i n _ {\beta} \left\| y - X \beta \right\| _ {2} ^ {2} + \alpha \| \beta \| _ {1},\tag{5}
$$

with X being the normalized data matrix with zero mean, y the target output vector, and $\beta$ the coefficient to find. α is a regularization parameter that controls the sparsity of the solution. A scheme of fault locations in the TE process is shown in Figure 1, illustrating the components of the industrial process.

![](images/f9c8a936a59682af0d0165d72643b39f69d6c9cbc81cb6ed19214a12583c68d6.jpg)  
Figure 1. Fault locations in the TE process.

## 2.2. Evolving Fuzzy Systems

A class of adaptive fuzzy inference systems known as evolving fuzzy systems (EFS) gains incremental, single-pass knowledge from data streams [15]. In contrast to traditional fuzzy systems that are trained in batch mode, EFS offers three essential features:

Incremental learning: They allow incremental operation and handling of unbounded data streams by processing data samples one at a time (or in blocks) without storing the complete dataset [17]. In this work, the data were split to evaluate our approach: 85% of the data were used to evolve the fuzzy system after certain preprocessing, and the remaining 15% were used to evaluate the methodology.

Structural evolution: When the existing system cannot adequately represent new data patterns, the rule base can grow by adding new rules or shrink by removing unnecessary or redundant rules [18].

Parameter adaptation: Forgetting mechanisms are used to recursively update both antecedent (membership function parameters) and consequent parameters to monitor evolving process dynamics [19].

A Takagi–Sugeno (TS) fuzzy model was used with local learning in this work, in which every rule has a linear consequent function. For regression tasks, the fuzzy rule has the following form:

$$
\begin{array}{c} \text {Rule} ^ {i}: \text {If (x_{1} is \mu_{i1}) and \ldots and (x_{p} is \mu_{ip}), then y = w_{i0} +w_{i1} x_{1} +} \\ \ldots + w _ {i p} x _ {p}, \end{array}\tag{6}
$$

where $\mu _ { i }$ is a fuzzy set, x is the input data, and $w _ { i }$ are the weights of the consequent. The consequent reduces to a singleton value that represents the estimated probability of the fault class for binary classification (fault vs. normal), which is then thresholded to yield a clear decision.

A one-vs-rest classification architecture was used to manage various fault types [26]. Separate binary EFS classifiers are trained for each of the ten selected faults. During operation, all classifiers are evaluated concurrently, and the classifier with the highest activation, or the first to surpass a confidence threshold, determines the detected fault. This strategy offers several benefits:

Each binary classifier can be independently optimized for its specific fault.

New fault types can be added by training additional binary classifiers without retraining existing ones.

Training each fault against all normal and other fault samples naturally addresses class imbalance.

New rule generation is controlled by a vigilance criterion based on the Mahalanobis distance between the current data sample and existing rule centers [21]. If the minimum Mahalanobis distance is greater than a predetermined threshold (vigilance), which indicates that the sample is in an uncovered area of the input space, a new rule is generated. The current sample is used to initialize the new rule’s center, and a diagonal matrix scaled by $\sigma _ { \mathrm { i n i t } } ^ { 2 }$ is used to initialize its covariance matrix.

Periodically, a pruning mechanism preserves interpretability and compactness. Rules are assessed based on their age (time since last update) and accuracy (ratio of correct predictions to total samples assigned to the rule). Rules that are too old without updates or have low accuracy (less than 50%) are eliminated. To reduce redundancy, rules with very similar centers (Euclidean distance below a threshold) are merged.

## 2.2.1. Rule Evolution and Pruning

A new rule is created when the current sample x lies outside the coverage region of all existing rules. In this case, a new data behavior can be detected and represented by the new fuzzy rule. As mentioned above, the Mahalanobis distance is used to determine the coverage area between the sample and the rule center c. The Mahalanobis distance for a rule i is computed as follows:

$$
d _ {M} (i, x) = \sqrt {(x - c _ {i}) ^ {T} \Sigma_ {i} ^ {- 1} (x - c _ {i})},\tag{7}
$$

where $\Sigma _ { i }$ is the covariance matrix. The sample is considered to be covered by rule i if $d _ { M } ( i , x ) \leq \rho , \rho$ being the vigilance threshold, and if for all existing rules min $d _ { M } ( i , x ) > \rho ,$ then a new rule is added with center $c _ { n e w } \ = \ x .$ , the inverse covariance matrix will be $\begin{array} { r } { \Sigma _ { i } ^ { - 1 } = \frac { 1 } { \sigma _ { i n i t } ^ { 2 } } \mathrm { I } , } \end{array}$ , and the consequent rule will be $\theta _ { n e w } = y _ { t r u e } .$

To maintain compactness and to prevent unbounded rule base growth, the pruning procedure is executed every $T _ { p r u n e } = 5 0$ samples. Algorithm 1 shows the pruning of rules when necessary. A rule is considered redundant and removed if it satisfies any of the following criteria:

Low accuracy: The rule classification accuracy falls below 50%. The accuracy of rule i computed as

$$
\mathrm{acc} _ {\mathrm{i}} = \frac {\text { correct   predictions }}{\text { total   samples   assigned   to   the   rule }},\tag{8}
$$

and the rule is pruned if acc $< 0 . 5$

• Obsolescence: The rule has not been updated for a large number of samples, i.e., $\mathrm { a g e _ { i } } > \mathrm { a g e _ { m a x } , }$ where $\mathrm { a g e } _ { \mathrm { m a x } } = 5 0 0$ samples in this case.

Redundancy due to similarity: Two rules are considered too similar (their centers are close) and their consequents are nearly identical. Then rules i and j are merged if

$$
\left\| c _ {i} - c _ {j} \right\| _ {2} <   \theta_ {\text { merge }}, \text {   and   } \left| \theta_ {i} - \theta_ {j} \right| <   \theta_ {\theta},\tag{9}
$$

with $\theta _ { m e r g e } = 0 . 1 5$ being the distance threshold, and the consequent similarity threshold being $\theta _ { \theta } = 0 . 1$ . When rules are merged, the one with higher accuracy is retained, and its center is updated as the weighted average of both centers using the number of samples assigned to each rule.

## 2.2.2. Parameter Adaptation

When a new sample x falls within the coverage region of existing rules, the center of the winning rule (the rule with the smallest $d _ { M } )$ is updated using a learning rate η according to

$$
c _ {w i n} (t) = c _ {w i n} (t - 1) + \eta (x - c _ {w i n} (t - 1)).\tag{10}
$$

The inverse covariance matrix $\Sigma _ { w i n } ^ { - 1 }$ is adaptively updated using a weighted recursive formulation to capture the local data distribution:

$$
\Sigma_ {w i n} ^ {- 1} (t) = (1 - \lambda) \Sigma_ {w i n} ^ {- 1} (t - 1) + \lambda Q (t),\tag{11}
$$

where

$$
Q (t) = \frac {(x - c _ {w i n}) (x - c _ {w i n}) ^ {T}}{\| x - c _ {w i n} \| ^ {2} + \epsilon},
$$

$\lambda = 1 /  { n _ { w i n } } + 1$ is a forgetting factor, and $n _ { w i n }$ is the number of samples assigned to the winning rule. This recursive approximation allows the evolving fuzzy rule to adapt its local covariance structure online while avoiding matrix inversion at every iteration.

```txt
Algorithm 1: Pseudocode for the periodic rule pruning
1: Input: Current rule base R, pruning interval T_prune
2: Output: Updated rule base R
3: if (sample_counter mod T_prune = 0), then
4:    for each rule i in R do
5:    Compute acc_i = correct_i / total_i
6:    if acc_i < 0.5, then
7:    Remove rule i
8:    else if age_i > age_max, then
9:    Remove rule i
10:    end if
11:    end for
12:    for each pair of rules (i, j) in R do
13:    if (distance(i, j) < θ_merge) and (|θ_i - θ_j| < θ_θ), then
14:    Remove rule with lower acc_i
15:    end if
16:    end for
17: end if
```

The consequent parameters $\theta _ { i }$ for each rule are updated using recursive fuzzily weighted least squares (RFWLS) [21], which performs local learning by weighting each sam ple’s contribution based on the corresponding rule’s normalized activation. The recursive update equation is

$$
\theta_ {i} (t) = \theta_ {i} (t - 1) + K _ {i} (t) \left(y (t) - \varphi_ {i} ^ {T} (t) \theta_ {i} (t - 1)\right),\tag{12}
$$

where $K _ { i }$ is the Kalman gain matrix calculated from the fuzzily weighted covariance matrix $P _ { \mathrm { i } } , \varphi _ { i } ( t ) = \exp \bigl ( - 0 . 5 d _ { M } ^ { 2 } ( i , x ) \bigr )$ is the regressor vector, and $\theta _ { i }$ is the parameter vector for rule $i , ( w _ { i 1 } , w _ { i 2 } , \ldots , w _ { i p } )$ . This local approach ensures that each rule’s consequent is updated only based on samples that activate that rule, weighted by their activation degree.

The normalized activation of rule i is

$$
\varphi_ {n o r m} (i, t) = \frac {\varphi_ {i} (t)}{\sum_ {j = 1} ^ {C} \varphi_ {j} (t)}.\tag{13}
$$

A modified version of eVQ (evolving Vector Quantization) [21] is used to update antecedent parameters (centers and covariance matrices).

The final decision is obtained from the aggregated output of the EFS. Each fuzzy rule produces a local consequent value based on the activation level of its premise, and these local outputs are combined using a weighted average aggregation mechanism. The resulting aggregated output represents the classifier’s confidence regarding the presence of a specific fault under the one-vs-rest scheme.

For FD, the aggregated output is compared against a fixed decision threshold. In this work, a threshold of 0.5 was employed, such that outputs greater than or equal to 0.5 are classified as fault conditions, while lower values were associated with normal operation. This threshold was selected because it provided stable behavior across the evaluated faults and maintained a suitable balance between sensitivity and specificity during the experimental analysis.

This adaptive behavior is particularly useful in industrial FD because process dynamics may evolve over time due to operating condition changes, sensor degradation, or previously unseen disturbances.

## 2.3. Wavelet-Based Preprocessing Methods

Because fault signatures are initially subtle and easily obscured by noise and control actions, wavelet transforms offer a multi-resolution analysis capability that is especially helpful for identifying incipient faults [30,31]. In this work, we compare several traditional techniques with three wavelet-based preprocessing methods.

In the proposed methodology, each process variable is preprocessed independently using discrete wavelet decomposition before entering the evolving fuzzy classifier. The original signals were decomposed using Haar, Daubechies, and Coiflet mother wavelets. After decomposition, the approximation coefficients are retained as filtered representations of the process dynamics, while high-frequency components associated mainly with noise were attenuated. These reconstructed signals are subsequently normalized and used as inputs for the EFS. This preprocessing stage aims to improve fault-related feature representation while preserving the temporal behavior of the original process variables.

The preprocessing stage was based on discrete wavelet transform (DWT), where each process signal x(nT) was decomposed into approximation and detail coefficients through successive low-pass and high-pass filtering operations:

$$
x (n T) \rightarrow \left\{A _ {j} (n T), D _ {j} (n T) \right\},\tag{14}
$$

where $A _ { j } ( n T )$ represents the approximation coefficients, $D _ { j } ( n T )$ are the detail coefficients at decomposition level j, and T is the sample time of three minutes. These reconstructed signals were subsequently normalized and used as inputs to the EFS.

## 2.3.1. Daubechies-4 (db4) Wavelet

The db4 wavelet transform uses differences in moving averages at several scales to approximate the effect of discrete wavelet decomposition. The detail coefficient at scale k for a given signal x(t) is calculated as follows:

$$
d _ {k} (t) = | \mathrm{MA} _ {2 k} (x (t)) - \mathrm{MA} _ {2 k + 1} (x (t)) |,\tag{15}
$$

where a moving average with window size w is indicated by $\mathrm { M A } _ { w }$ . The total detail coefficients for all scales, normalized to [0, 1], constitute the final preprocessed signal. This approach can be used in typical scientific computing environments instead of classical db4 wavelet because it captures multiscale transitions without the need for a specialized wavelet toolbox.

## 2.3.2. Haar Wavelet

The Haar wavelet, which has a square-shaped scaling function, is the most basic orthogonal wavelet [37]. It works well for identifying sudden changes (step-type faults) and is computationally efficient. The discrete Haar wavelet transform decomposes the signal into approximation and detail coefficients at several levels. In this work, we use level-4 decomposition and reconstruct the signal using only the detail coefficients, which emphasize fault onsets and transients.

## 2.3.3. Coiflet Wavelet

Coiflet wavelets, especially coif4, are more symmetric than Daubechies wavelets and offer superior reconstruction capabilities. They are frequently chosen when the phase of signal components needs to be preserved. As with the Haar wavelet, we use level-4 decomposition and retain the detail coefficients for fault highlighting.

## 2.3.4. Conventional Preprocessing Techniques

For comparison, we also evaluate five conventional preprocessing methods:

• None: Normalized raw data (baseline).

• Smoothing: Moving average with window size 3.

Peak detection: Deviations greater than two standard deviations determine the binary threshold.

• Adaptive thresholding comes after normalization in balanced preprocessing.

• Moving average with window size 7 is a strong smoothing technique.

## 2.4. Experimental Setup

A training set (85%, first 1700 samples) was used to evolve the fuzzy system, not for training, allowing the fuzzy system to be adapted according to incoming data. The validation set (15%, remaining 301 samples) was created from the available data (2001 samples) for each fault to evaluate the proposed framework according to its evolving property. The partitioning of normal operating data was comparable. To address class imbalance, we used undersampling of normal samples with a ratio of two normal samples for every fault sample during training.

The hyperparameters shown in Table 2 were selected empirically based on preliminary experiments and stability analysis over several TE fault scenarios. Parameters related to rule evolution, such as vigilance threshold and pruning frequency, were adjusted to balance model adaptability and rule base compactness. Learning rate parameters were selected to ensure stable incremental adaptation without excessive oscillations. The same hyperparameter configuration was maintained for all experiments to ensure consistency and reproducibility.

The final configuration was chosen as a trade-off between diagnostic performance and model complexity, avoiding excessive rule growth and overfitting effects.

Standard classification metrics calculated from the confusion matrix are used to evaluate the performance:

Sensitivity (detection rate) $) = \mathrm { T P } / ( \mathrm { T P } + \mathrm { F N } ) \times 1 0 0 \%$

• Specificity = TN/(TN + FP) × 100%.

## • F1-Score = 2 × (Precision × Sensitivity)/(Precision + Sensitivity) × 100%

where FP (false positives) are false alarms, FN (false negatives) are missed faults, TP (true positives) are correctly identified fault samples, and TN (true negatives) are correctly identified normal samples.

Table 2. ESF hyperparameters (default values).

<table><tr><td>Parameter</td><td>Symbol</td><td>Default Value</td><td>Description</td></tr><tr><td>Vigilance</td><td> $P$ </td><td>1.2</td><td>Threshold for creating new rules</td></tr><tr><td>Learning rate</td><td> $H$ </td><td>0.05</td><td>Step size for center/consequent updates</td></tr><tr><td>Initial spread</td><td> $\sigma_{init}$ </td><td>0.5</td><td>Initial width of Gaussian membership functions</td></tr><tr><td>Maximum rules</td><td> $R_{max}$ </td><td>40</td><td>Upper bound on rule base size</td></tr><tr><td>Minimum samples per rule</td><td> $N_{min}$ </td><td>20</td><td>Samples required before pruning</td></tr><tr><td>Pruning interval</td><td> $T_{prune}$ </td><td>50</td><td>Frequency of pruning (samples)</td></tr><tr><td>Persistence</td><td> $N_{persist}$ </td><td>2</td><td>Consecutive detections required</td></tr><tr><td>Detection threshold</td><td> $\lambda$ </td><td>0.25</td><td>Decision threshold for binary classification</td></tr><tr><td>Max age before pruning</td><td> $age_{max}$ </td><td>500</td><td>Samples without update</td></tr><tr><td>Merge distance threshold</td><td> $\theta_{merge}$ </td><td>0.15</td><td>Max center distance for merging</td></tr><tr><td>Merge consequent threshold</td><td> $\theta_{\theta}$ </td><td>0.1</td><td>Max consequent difference for merging</td></tr><tr><td>Persistence window</td><td> $N_{persist}$ </td><td>2</td><td>Consecutive detection required</td></tr></table>

The open-source numerical computing environment GNU Octave©, version 7.1.0, was used to implement each algorithm. The code is organized modularly with distinct functions for EFS training, prediction, rule pruning, and wavelet preprocessing.

Figure 2 illustrates the proposed EFS-based fault diagnosis framework with wavelet preprocessing. Algorithm 2 shows the same information for programming.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2: Pseudocode for EFS incremental learning in fault detection.

1: Initialize: $R = \{ \}$, sample$_{counter} = 0$
2: for each new sample ($x, y_{true}$) do
3:    sample$_{counter} = \text{sample}_{\text{counter}} + 1$
4:    if $R$ is empty, then
5:    Create first rule: $c_1 = x$, $\sigma_1 = \sigma_{\text{init}}$, $\theta_1 = y_{\text{true}}$
6:    else
7:    Compute activations $\varphi_1 = \exp(-0.5 * d_M(i, x)^2)$
8:    Let $i^* = \argmin_i d_M(i, x)$
9:    if $d_M(i^*, x) &gt; \rho$ and $|R| &lt; R_{max}$, then
10:    Create new rule: $c_{new} = x$, $\sigma_{new} = \sigma_{init}$, $\theta_{new} = y_{true}$
11:    else
12:    Update winning rule $i^*: c_i^* = c_i^* + \eta(x - c_i^*)$
13:    Update $\sigma_i^*$ using rank-one, Equation (11)
14:    Update $\theta_i^*$ using RFWLS
15:    end if
16:    if sample$_{counter}$ mod T$_{prune} == 0$, then
17:    Prune rules with ($acc_i &lt; 0.5$ or $age_i &gt; age_{max}$)
18:    Merge rules with $|c_i - c_j| |_2 &lt; \theta_{merge}$ and $|\theta_i - \theta_j| &lt; \theta_\theta$
19:    end if
20:    end if
21:    $y_{pred} = \Sigma_i \varphi_{norm} \theta_i$
22:    fault$_{detected} = (y_{pred} &gt; \lambda)$
23: end for
</div>

![](images/bcb84c64ac2618b552337ce842602f37c73d140414899efeafa17fde5abe0bed.jpg)  
Figure 2. Wavelet-based evolving fuzzy system for fault diagnosis scheme.

## 3. Results and Discussion

This section presents the experimental results of applying the evolving fuzzy system (EFS) with various preprocessing techniques to the Tennessee Eastman process. We first present the overall effectiveness of conventional preprocessing techniques, and then provide a comparison of wavelet-based methods, followed by a thorough per-fault analysis to identify which faults present the greatest difficulty for the proposed method. Finally, we discuss the implications of the results in light of the current literature.

The EFS itself is deterministic once the training and validation datasets are defined. However, small variations may appear between executions if random data partitioning is employed during the train/validation split stage. In order to improve reproducibility, the same experimental configuration and hyperparameters were maintained throughout all experiments. Another consideration is the pruning stage, because if more fuzzy rules are generated, the results may improve but the interpretability and computational cost will be compromised.

One important characteristic of evolving fuzzy systems is their capability to adapt their rule base incrementally as new data become available. In the proposed framework, incoming samples are processed sequentially without requiring complete retraining of the model structure, allowing the system to evolve its fuzzy rules according to new operating conditions or when data reveal new fault types.

Although, the present work did not explicitly evaluate the incorporation of entirely new fault classes during operation, the evolving architecture provides the structural basis for future adaptive extensions involving unknown or emerging faults under online industrial scenarios. The processed simulated data were added to the EFS to merge fuzzy rules indicting a fault, and with the 15% of data the framework was evaluated to obtain the F1-score.

Table 3 shows the performance of five standard preprocessing methods (none, smoothing, peak detection, balanced preprocessing, and strong smoothing) averaged over all ten selected faults. The baseline method (no preprocessing) achieved a sensitivity of 94.03% and an F1-score of 70.67%, setting a standard for comparison. Strong smoothing achieved the highest specificity (59.99%) among conventional methods, but also the lowest sensitivity (72.13%). Balanced preprocessing achieved the highest sensitivity (98.93%) but the lowest specificity (21.33%), indicating a high number of false alarms.

Table 3. Performance of conventional preprocessing methods (averaged over 10 faults).

<table><tr><td>Method</td><td>Sensitivity (%)</td><td>Specificity (%)</td><td>F1-Score (%)</td></tr><tr><td>None (baseline)</td><td>94.03</td><td>56.93</td><td>70.67</td></tr><tr><td>Smoothing (window = 3)</td><td>73.83</td><td>38.60</td><td>50.70</td></tr><tr><td>Peak detection</td><td>92.33</td><td>16.43</td><td>27.89</td></tr><tr><td>Balanced preprocessing</td><td>98.93</td><td>21.33</td><td>35.09</td></tr><tr><td>Strong smoothing (window = 7)</td><td>72.13</td><td>59.99</td><td>65.51</td></tr></table>

## 3.1. Performance of Wavelet-Based Preprocessing Methods

Table 4 shows the performance of three wavelet-based methods: Haar, Daubechies, and Coiflet wavelets (all at level 4 of decomposition). The Daubechies wavelet method achieved the highest overall F1-score (73.68%) with a sensitivity of 97.37%, substantially outperforming the baseline. The Haar wavelet achieved the best specificity (55.37%) and a competitive F1-score of 68.67%. The Coiflet wavelet, in contrast, performed poorly (F1-score 57.56%), especially on faults IDV1, IDV2, IDV10, and IDV13, where sensitivity dropped to less than 5%.

Table 4. Performance of wavelet-based preprocessing methods (averaged over 10 faults).

<table><tr><td>Wavelet Method</td><td>Sensitivity (%)</td><td>Specificity (%)</td><td>F1-Score (%)</td></tr><tr><td>Daubechies wavelet (level 4)</td><td>97.37</td><td>59.25</td><td>73.68</td></tr><tr><td>Haar wavelet (level 4)</td><td>90.39</td><td>55.37</td><td>68.67</td></tr><tr><td>Coiflet wavelet (level 4)</td><td>79.27</td><td>45.19</td><td>57.56</td></tr></table>

Figure 3 provides a side-by-side comparison of all eight methods tested in this work, including both conventional and wavelet-based methods. The Daubechies and Haar wavelets clearly outperform the others in terms of F1-score, demonstrating the utility of multiscale analysis for fault detection.

![](images/b70bcab9f742b75fbfe2ba3752dd3111b41f5f497d457bdd713b04f23dad2988.jpg)  
Figure 3. Comparative performance of all preprocessing methods.

The lower performance obtained with the Coiflet wavelet may be related to its longer support length and smoother basis functions. Although Coiflets are effective for representing smooth signals with higher regularity [38], several Tennessee Eastman faults produce abrupt transient changes and localized dynamic variations. In such cases, shorter and less smooth wavelets such as Daubechies preserve transient information more effectively.

Furthermore, the fault signatures in the TE process are highly nonstationary and may contain sharp temporal deviations associated with process disturbances. The Coiflet decomposition tends to distribute these local variations across multiple coefficients, reducing class separability in the EFS. This behavior may explain the lower sensitivity and specificity values observed in this approach. Nevertheless, Coiflets may still be useful primarily for incipient faults.

Daubechies wavelets exhibited superior performance due to their compact support and favorable time-frequency localization properties, which allow better preservation of transient fault signatures in nonlinear industrial signals.

Table 5 shows the detailed performance of the Daubechies wavelet method for each of the ten selected faults. The results reveal substantial variation across fault types. IDV4, IDV6, and IDV7 achieved excellent F1-scores (80.80%, 94.89%, and 98.36%, respectively). IDV7 had almost perfect sensitivity (99.67%) and specificity (97.00%).

Table 5. Detailed results for the Daubechies wavelet method per fault.

<table><tr><td>Fault</td><td>Sensitivity (%)</td><td>Specificity (%)</td><td>F1-Score (%)</td></tr><tr><td>IDV1</td><td>100.00</td><td>54.33</td><td>70.40</td></tr><tr><td>IDV2</td><td>100.00</td><td>54.33</td><td>70.40</td></tr><tr><td>IDV4</td><td>99.33</td><td>68.10</td><td>80.80</td></tr><tr><td>IDV5</td><td>100.00</td><td>54.33</td><td>70.40</td></tr><tr><td>IDV6</td><td>96.00</td><td>93.67</td><td>94.89</td></tr><tr><td>IDV7</td><td>99.67</td><td>97.00</td><td>98.36</td></tr><tr><td>IDV8</td><td>100.00</td><td>54.33</td><td>70.40</td></tr><tr><td>IDV10</td><td>89.45</td><td>65.37</td><td>75.53</td></tr><tr><td>IDV13</td><td>93.66</td><td>28.66</td><td>43.88</td></tr><tr><td>IDV14</td><td>95.67</td><td>45.67</td><td>61.83</td></tr></table>

In contrast, faults IDV1, IDV2, IDV4, IDV5, IDV7, IDV8, and IDV14 have near-100% sensitivity but specificity around 50% (indicating many false alarms). This means the model identified almost all samples as faults in these cases. This behavior suggests that these faults create persistent deviations that are readily identifiable yet challenging to differentiate from normal operation without triggering false alarms (incipient faults).

Faults IDV13 and IDV14 exhibited lower F1-score values compared to the remaining fault classes. Although the proposed framework achieved high sensitivity values above 93% for both cases, specificity remained limited, indicating an increased false alarm rate during classification. This is because IDV13 and IDV14 are known to generate process variations with temporal patterns that partially overlap normal operating fluctuations, making fault discrimination significantly more challenging than for abrupt or highly separable faults. Therefore, the F1-score is an important metric for evaluating framework performance because near-100% sensitivity does not imply an efficient FD scheme.

The evolving fuzzy framework tends to favor conservative detection behavior, maintaining high sensitivity while sacrificing specificity. Although this trade-off increases false alarms, it reduces the probability of missed detections, which is often preferable in safety-critical industrial environments [39].

These results also suggest that additional feature extraction strategies, such as multifractal descriptors, entropy-based measures, or hybrid adaptive thresholds, could further improve discrimination capability for complex incipient faults; this will be a future work.

Figure 4 shows the per-fault F1-scores for the three wavelet methods. Most of the time, the Daubechies and Haar wavelets perform similarly across faults. However, Coiflet performs much worse on IDV7, IDV8, and IDV13. This observation suggests that selecting the appropriate wavelet is essential for particular fault types, and no single wavelet is universally superior.

Table 6 shows how the best results from this study compare to those found in the literature for the TE process. The Daubechies wavelet method has a sensitivity of 97.37%, which is better than or as good as most other data-driven methods. This performance was achieved with a fully incremental, evolving fuzzy system that does not require retraining when new fault types emerge; a feature absent in static methods. Although deep learning approaches such as LSTM and autoencoders have shown high performance, they require retraining and lack interpretability. An interesting approach to achieving interpretability is given in [40], where a Transformer–LSTM denoising autoencoder achieved competitive fault sensitivities in the TE process through spatiotemporal attention; these results are also compared in Table 6. However, Ref. [40] does not show results for IDV3 because this fault only presents deviations from normal operating conditions.

![](images/480da341fe78204e2cbf9dd5a8073563066ff7eedb0602ae617a1f9b2139b4fb.jpg)  
Figure 4. Per-fault F1-score comparison for Daubechies, Haar, and Coiflet wavelets.

Table 6. Results comparison among different approaches.

<table><tr><td>Study</td><td>Method</td><td>Sensitivity (%)</td><td>Notes</td></tr><tr><td>Yin et al. [6]</td><td>PCA, PLS, ICA</td><td>73.8–84.4</td><td>Statistical methods (batch)</td></tr><tr><td>D&#x27;Angelo et al. [32]</td><td>Fuzzy clustering + wavelets</td><td>88.5</td><td>Batch learning, fuzzy logic</td></tr><tr><td>Hajihosseini et al. [33]</td><td>Wavelets + neural networks</td><td>98.6</td><td>Requires retraining, black box</td></tr><tr><td>Guo et al. [40]</td><td>Transformer-LSTM denoising autoencoder</td><td>96.50</td><td>Semi-supervised, batch learning, requires retraining</td></tr><tr><td>Márquez et al. [41]</td><td>LSTM + elastic-net</td><td>96.45</td><td>Deep learning</td></tr><tr><td>This approach</td><td>EFS + wavelets (one-vs-rest)</td><td>97.37</td><td>Incremental, no retraining</td></tr></table>

Compared to deep learning approaches such as Transformer–LSTM models [40], the proposed EFS framework achieves comparable sensitivity while maintaining full interpretability and supporting incremental learning without retraining.

Deep learning methods such as convolutional neural networks (CNNs) and long shortterm memory networks (LSTMs) have demonstrated high fault classification performance in industrial processes [41]. However, these approaches usually require extensive training datasets, high computational resources, and retraining procedures when new operating conditions or faults appear.

In contrast, EFSs provide adaptive rule evolution, lower computational complexity, and improved interpretability, making them attractive for industrial applications where transparency and incremental adaptation are required.

## 3.2. Discussion

The results demonstrate four main findings:

(1) Wavelet-based preprocessing consistently improves the performance of EFS for FD compared to conventional methods. The Daubechies wavelet achieved the highest F1-score (73.68%) and the best specificity (59.25%).

(2) The choice of wavelet matters significantly: the Coiflet wavelet underperformed, particularly on faults with abrupt dynamics (e.g., IDV7).

(3) Certain faults (IDV1, IDV2, IDV5, IDV8) remain challenging, exhibiting high sensitivity but very low specificity. These faults are expected to cause persistent changes in the measured variables that persist even after the fault is removed, making them difficult to distinguish from normal operation without generating false alarms.

(4) Interestingly, using no preprocessing achieved good sensitivity results, even better than using Haar and Coiflet wavelets to preprocess data for EFS improvement.

From a practical perspective, the selection of preprocessing method should be guided by the specific requirements of the application. If maximizing fault detection (sensitivity) is the priority, for example, in safety-critical systems, the Daubechies wavelet method is recommended for preprocessing. Whereas, if minimizing false alarms (specificity) is more important, for instance, in systems where alarms cause operator fatigue or unnecessary shutdowns, the Haar wavelet method is preferable. Also, in this work we selected the variables that made the most contribution in the signals once a fault is presented. A different set of variables could be chosen for FD. However, some signals present a different behavior when a fault is simulated after some time delay [36].

The window size used for wavelet preprocessing and feature extraction should be selected according to the dominant process dynamics. Small windows increase sensitivity to abrupt changes but may also amplify noise, whereas larger windows provide smoother responses at the expense of slower fault detection.

Additionally, normalization of process variables prior to preprocessing is recommended to avoid dominance of variables with larger amplitudes. The EFS hyperparameters should initially be adjusted using representative operating conditions before deployment in online monitoring scenarios. Although the proposed framework supports adaptive rule evolution, practitioners should periodically inspect newly generated fuzzy rules to determine whether they correspond to novel operating conditions, sensor drift, or genuinely new fault patterns. Such supervision is important to maintain interpretability and to reduce the risk of false alarms during long-term industrial operation.

Several limitations of this study should be acknowledged:

The evaluation was conducted on simulated data; real-world performance may differ due to unmodeled noise, sensor drift, and other practical factors.

Only ten of the twenty available TE faults were analyzed; generalization to the remaining faults requires further validation.

The one-vs-rest architecture, while effective, requires training separate classifiers for each fault, which may become computationally expensive as the number of fault types grows.

Although the TE process is one of the most widely accepted industrial benchmarks for fault diagnosis research, the experiments presented in this work were conducted using simulated data. Therefore, the reported results should be interpreted as a controlled validation of the proposed methodology under representative nonlinear process conditions.

## 4. Conclusions

This paper showed a comparative study of preprocessing methods for evolving fuzzy systems (EFS) applied to fault diagnosis (FD) in the Tennessee Eastman (TE) process. This work was motivated by the limitations of static fault diagnosis methods, particularly their inability to handle unknown faults and the need for adaptation when process conditions change. This is particularly important when the FD is model-based, when a control action is taken, and the operating point of the system changes over time; in such cases, a static or statistical approach will detect a non-existent fault.

EFS offer a paradigm shift by supporting incremental learning, structural evolution, and parameter adaptation, making them well-suited for adaptive FD applications, princi pally, when no prior fault information is available.

The main contributions of this work can be summarized in the following way:

Integration of EFS with wavelet preprocessing. A modular implementation of EFS with one-vs-rest architecture was developed for multi-fault classification. The system processes data incrementally, evolves new fuzzy rules when previously unseen patterns appear, and prunes redundant rules to maintain compactness and interpretability by limiting the number of fuzzy rules.

Comparative study. Eight preprocessing methods were evaluated to highlight the presence of a fault, and the one with the best result was compared with results reported in the literature.

Open-source implementation. The codebase was implemented in GNU Octave© to ensure reproducibility.

The Daubechies-4 (db4) wavelet approximation achieved the best overall performance: an F1-score of 73.68% with a sensitivity of 97.37%. The obtained results suggest that multiscale analysis effectively highlights relevant fault signatures in nonlinear industrial process signals. The db4 wavelet achieved the best overall sensitivity and F1-score, whereas the Haar wavelet produced the highest specificity and lowest false alarm rate. However, if no preprocessing is made or a strong smoothing is applied, the sensibility and specificity obtained were 94.03 and 56.93% respectively, but with a lower F1-score.

The one-vs-rest architecture, made here, requires training separate classifiers for each fault. As the number of fault types grows, the computational cost increases linearly. The proposed approach has significantly lower computational complexity compared to deep learning architectures. To maintain stable fault diagnosis performance and preserve interpretability, it is important to control the growth of the fuzzy rule base by pruning redundant or weakly activated rules.

An important advantage of the proposed evolving fuzzy framework is the interpretability of its rule-based structure. Unlike black-box models such as deep neural networks, the evolving fuzzy system generates linguistic fuzzy rules that can be associated with different operating conditions of the process. The number of rules changes dynamically according to the complexity of the monitored data, allowing the model to adapt to new operating regions while maintaining an interpretable structure. In practice, the system typically generates a compact set of rules, which facilitated the analysis of abnormal operating behaviors without requiring complete retraining.

Additionally, the use of wavelet preprocessing contributes to highlighting transient and multiscale characteristics of the process signals before rule generation. Although wavelet coefficients do not always have a direct physical interpretation, they preserve rele vant information associated with process dynamics and fault evolution. Consequently, the resulting fuzzy rules can still provide meaningful information regarding changes in process behavior. Nevertheless, as the number of operating conditions and fault classes increases, some overlap between fuzzy premises may appear, which can reduce transparency. For this reason, periodic inspection and pruning of redundant rules may be beneficial in long-term industrial applications.

As industrial processes continue to generate massive streams of data, adaptive, incremental, and interpretable methods like EFS will play an increasingly important role in ensuring safe and efficient operations. This work presents an interpretable and adaptive framework that can be applied when no data history is available, and when an interpretable model is required.

## 5. Limitations and Future Work

Although the proposed framework demonstrated promising performance in the Tennessee Eastman benchmark, several limitations remain. First, the experiments were conducted using simulated process data rather than measurements obtained from real industrial plants. Consequently, additional validation under real operating conditions is required. Additionally, human supervision may still be required when newly generated fuzzy rules appear, particularly to determine whether they correspond to new operating conditions, sensor drift, or previously unseen fault patterns.

Second, although the evolving fuzzy architecture supports adaptive rule modification, the present study evaluated sequential offline partitions instead of fully online streaming scenarios involving concept drift or dynamic fault incorporation.

Future work will focus on:

• Online adaptive learning;

• Multifractal feature extraction;

• Hybrid entropy–fractal descriptors;

• Using an interpretable framework like Kolmogorov–Arnold networks.

Author Contributions: Conceptualization, M.A.M.-V. and C.A.M.-V.; methodology, M.A.M.-V., J.A.R.-V. and A.M.; software, M.A.M.-V., J.A.R.-V. and E.M.-R.; validation, C.A.M.-V., A.M. and E.M.-R.; formal analysis, M.A.M.-V., J.A.R.-V. and A.M.; investigation, M.A.M.-V., C.A.M.-V. and E.M.-R.; data curation, A.M. and E.M.-R.; writing—original draft preparation, M.A.M.-V., C.A.M.-V. and A.M.; writing—review and editing, J.A.R.-V. and E.M.-R.; visualization, M.A.M.-V. and C.A.M.-V.; supervision, M.A.M.-V. and J.A.R.-V. All authors have read and agreed to the published version of the manuscript.

Funding: This research received no external funding.

Data Availability Statement: Data supporting reported results can be found in the following link, where fault data and example code were added: https://drive.google.com/drive/folders/1dgDc-MYkI2yXWpqxSXspk7dKmzFC2-Hj?usp=sharing (accessed on 9 June 2026).

Conflicts of Interest: The authors declare no conflicts of interest.

## References

1. Yimam, A.W.; Vafaeipour, M.; Messagie, M.; Fante, K.A.; Molla, E.M.; Azerefegn, T.M.; Coosemans, T. State-of-the-art of machine learning methods for fault detection and health monitoring of wind turbine system components: A comprehensive review. Eng. Appl. Artif. Intell. 2025, 162, 112645. [CrossRef]

Soleimani, M.; Irani, F.N.; Yadegar, M.; Meskin, N. Comprehensive review of gas turbine fault diagnostic strategies. Appl. Energy 2025, 401, 126801. [CrossRef]

3. Asmone, A.S.; Goh, Y.M.; Lim, M.S.H. Prioritization of industry level interventions to improve implementation of design for safety regulations. J. Saf. Res. 2022, 82, 352–366. [CrossRef] [PubMed]

4. Bathelt, A.; Ricker, N.L.; Jelali, M. Revision of the Tennessee Eastman process model. IFAC-PapersOnLine 2015, 48, 309–314. [CrossRef]

5. Downs, J.J.; Vogel, E.F. A plant-wide industrial process control problem. Comput. Chem. Eng. 1993, 17, 245–255. [CrossRef]

6. Yin, S.; Ding, S.X.; Haghani, A.; Hao, H.; Zhang, P. A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process. J. Process Control 2012, 22, 1567–1581. [CrossRef]

7. Cho, S.; Jiang, J. A fault detection and isolation technique using nonlinear support vectors dichotomizing multi-class parity space residuals. J. Process Control 2019, 82, 31–43. [CrossRef]

8. Yang, C.; Huang, J.; Wu, S.; Liu, Q. Neural-network-based practical specified-time resilient formation maneuver control for second-order nonlinear multi-robot systems under FDI attacks. Neural Netw. 2025, 186, 107288. [CrossRef] [PubMed]

9. Cao, X.; Liu, S.; Cen, J. Observer-based adaptive neural asynchronous H∞ Control for fuzzy Markov jump systems under FDI attacks. J. Frankl. Inst. 2024, 361, 107147. [CrossRef]

10. Lemos, A.; Caminhas, W.; Gomide, F. Adaptive fault detection and diagnosis using an evolving fuzzy classifier. Inf. Sci. 2013, 220, 64–85. [CrossRef]

11. Schmidt, S.; Gryllias, K.C. An anomalous frequency band identification method utilising available healthy historical data for gearbox fault detection. Measurement 2023, 222, 113515. [CrossRef]

12. Venkatasubramanian, V.; Rengaswamy, R.; Yin, K.; Kavuri, S.N. A review of process fault detection and diagnosis: Part I: Quantitative model-based methods. Comput. Chem. Eng. 2003, 27, 293–311. [CrossRef]

13. Lughofer, E. On-line assurance of interpretability criteria in evolving fuzzy systems—Achievements, new concepts and open issues. Inf. Sci. 2013, 251, 22–46.

14. Kumar, A.A.; Srikumar, D.K.; Rao, D.G.N. A high-dimensional data-driven approach for enhancing cyber-physical attack detection in PV-connected distribution power grids using deep Q-networks. Comput. Electr. Eng. 2026, 129, 110814. [CrossRef]

15. Lughofer, E.; Pratama, M. Evolving multi-user fuzzy classifier system with advanced explainability and interpretability aspects. Inf. Fusion 2023, 91, 458–476. [CrossRef]

16. Huang, H.; Rong, H.J.; Yang, Z.X.; Vong, C.M. Jointly evolving and compressing fuzzy system for feature reduction and classification. Inf. Sci. 2021, 579, 218–230. [CrossRef]

17. Lughofer, E.; Guardiola, C. On-line fault detection with data-driven evolving fuzzy models. Control. Intell. Syst. 2008, 36, 307.

18. Lughofer, E.; Cernuda, C.; Kinderman, S.; Pratama, M. Generalized smart evolving fuzzy systems. Evol. Syst. 2015, 6, 269–292. [CrossRef]

19. Pratama, M.; Lu, J.; Zhang, G. Evolving type-2 fuzzy classifier. IEEE Trans. Fuzzy Syst. 2016, 24, 574–589.

20. Angelov, P.; Filev, D. An approach to online identification of Takagi-Sugeno fuzzy models. IEEE Trans. Syst. Man Cybern. Part B 2004, 34, 484–498. [CrossRef] [PubMed]

21. Lughofer, E. FLEXFIS: A robust incremental learning approach for evolving Takagi-Sugeno fuzzy models. IEEE Trans. Fuzzy Syst. 2008, 16, 1393–1410.

22. Kasabov, N.; Song, Q. DENFIS: Dynamic evolving neural-fuzzy inference system and its application for time-series prediction. IEEE Trans. Fuzzy Syst. 2002, 10, 144–154. [CrossRef]

23. Pratama, M.; Anavatti, S.G.; Angelov, P.P.; Lughofer, E. PANFIS: A novel incremental learning machine. IEEE Trans. Neural Netw. Learn. Syst. 2014, 25, 55–68. [CrossRef] [PubMed]

24. Pratama, M.; Anavatti, S.G.; Lughofer, E. GENEFIS: Toward an effective localist network. IEEE Trans. Fuzzy Syst. 2014, 22, 547–562. [CrossRef]

25. Lughofer, E. On Improving Performance and Increasing Useability of EFS. In Evolving Fuzzy Systems—Methodologies, Advanced Concepts and Applications; Springer: Cham, Switzerland, 2011; pp. 213–259.

26. Lughofer, E. Evolving fuzzy systems: Fundamentals, reliability, interpretability, and applications. In Handbook of Computational Intelligence; World Scientific: Singapore, 2016; pp. 67–135.

27. Serdio, F.; Lughofer, E.; Pichler, K.; Pichler, M.; Buchegger, T.; Efendic, H. Fuzzy fault isolation using gradient information and quality criteria from system identification models. Inf. Sci. 2015, 316, 18–39. [CrossRef]

28. Adonovski, G.; Muiˇc, G.; Blažiˇc, S.; Škrjanc, I. Evolving model identification for process monitoring and prediction of non-linear systems. Eng. Appl. Artif. Intell. 2018, 68, 214–221. [CrossRef]

29. Lughofer, E.; Macián, V.; Guardiola, C.; Klement, E.P. Identifying static and dynamic prediction models for NOx emissions with evolving fuzzy systems. Appl. Soft Comput. 2011, 11, 2487–2500. [CrossRef]

30. Yan, R.; Gao, R.X.; Chen, X. Wavelets for fault diagnosis of rotary machines: A review with applications. Signal Process. 2014, 96, 1–15. [CrossRef]

31. Mallat, S. A Wavelet Tour of Signal Processing: The Sparse Way; Academic Press: Cambridge, MA, USA, 2009.

32. D’Angelo, M.F.S.V.; Palhares, R.M.; Camargos-Filho, M.C.O.; Maia, R.D.; Mendes, J.B.; Ekel, P.Y. A new fault classification approach applied to Tennessee Eastman benchmark process. Appl. Soft Comput. 2016, 49, 676–686. [CrossRef]

33. Hajihosseini, P.; Anzehaee, M.M.; Behnam, B. Fault detection and isolation in the challenging Tennessee Eastman process by using image processing techniques. ISA Trans. 2018, 79, 137–146. [CrossRef] [PubMed]

34. He, Y.L.; Zhao, Y.; Hu, X.; Yan, X.N.; Zhu, Q.X.; Xu, Y. Fault diagnosis using novel AdaBoost based discriminant locality preserving projection with resamples. Eng. Appl. Artif. Intell. 2020, 91, 103631. [CrossRef]

35. Liang, P.; Deng, C.; Wu, J.; Yang, Z.; Zhu, J.; Zhang, Z. Compound fault diagnosis of gearboxes via multi-label convolutional neural network and wavelet transform. Comput. Ind. 2019, 113, 103132. [CrossRef]

36. Márquez-Vera, M.A.; Ramos-Velasco, L.E.; López-Ortega, O.; Zúñiga-Peña, N.S.; Ramos-Fernández, J.C.; Ortega-Mendoza, R.M. Inverse fuzzy fault model for fault detection and isolation with least angle regression for variable selection. Comput. Ind. Eng. 2021, 159, 107499. [CrossRef]

37. Chen, J.; Li, Z.; Pan, J.; Chen, G.; Zi, Y.; Yuan, J.; Chen, B.; He, Z. Wavelet transform based on inner product in fault diagnosis of rotating machinery: A review. Mech. Syst. Signal Process 2016, 70, 1–35. [CrossRef]

38. Mansi, A.; Dunai, L.; Cao, M. Wavelet-based denoising of structural health monitoring strain measurements. Meas. Sens. 2025, 42, 101974. [CrossRef]

39. Toribio, L.; Veloso, B.; Gama, J.; Zafra, A. A two-stage framework for early failure detection in predictive maintenance: A case study on metro trains. Neurocomputing 2026, 670, 132506.

40. Guo, L.; Shi, J.; Kang, J.; Li, A. An interpretable Transformer–LSTM denoising autoencoder for semi-supervised fault diagnosis in chemical processes. Eng. Appl. Artif. Intell. 2026, 172, 114358. [CrossRef]

41. Márquez-Vera, M.A.; López-Ortega, O.; Ramos-Velasco, L.E.; Ortega-Mendoza, R.M.; Fernández-Neri, B.J.; Zúñiga-Peña, N.S. Fault diagnosis using an LSTM and an elastic net. Rev. Iberoam. Autom. Inf. Ind. 2021, 18, 164–175.

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.