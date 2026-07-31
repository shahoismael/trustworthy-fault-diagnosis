# Title: Open-Set Fault Diagnosis in Multimode Processes via Fine-Grained Deep Feature Representation

Names of authors: Guangqiang Li <sup>a</sup>, M. Amine Atoui <sup>b</sup>, Xiangshun Li <sup>a∗</sup>

Afiliations and addresses: <sup>a</sup> School of Automation, Wuhan University of Technology, Wuhan, China <sup>b</sup> School of Information Technology, Halmstad University, Halmstad, Sweden

Corresponding author

Name: Xiangshun Li

E-mail: lixiangshun@whut.edu.cn

# Open-Set Fault Diagnosis in Multimode Processes via Fine-Grained Deep Feature Representation

Guangqiang Li<sup>a</sup>, M. Amine Atoui<sup>b</sup>, Xiangshun Li<sup>a,∗</sup>

<sup>a</sup>School of Automation, Wuhan University of Technology, Wuhan, 430070, PR China <sup>b</sup>The School of Information Technology, Halmstad University, Halmstad, Sweden

## Abstract

A reliable fault diagnosis system should not only accurately classify known health states but also efectively identify unknown faults. In multimode processes, samples belonging to the same health state often show multiple cluster distributions, making it dificult to construct compact and accurate decision boundaries for that state. To address this challenge, a novel openset fault diagnosis model named fine-grained clustering and rejection network (FGCRN) is proposed. It combines multiscale depthwise convolution, bidirectional gated recurrent unit and temporal attention mechanism to capture discriminative features. A distance-based loss function is designed to enhance the intra-class compactness. Fine-grained feature representations are constructed through unsupervised learning to uncover the intrinsic structures of each health state. Extreme value theory is employed to model the distance between sample features and their corresponding fine-grained representations, enabling efective identification of unknown faults. Extensive experiments demonstrate the superior performance of the proposed method.

Keywords: Fault Diagnosis, Open set, Multimode processes

## 1. Introduction

Safety is the fundamental prerequisite for modern industrial production and manufacturing. With the growing scale and complexity of modern industrial systems, the probability of system faults has risen significantly [1, 2, 3].

Local faults may propagate within the system, potentially triggering cascading failures and catastrophic accidents, causing serious economic losses and threatening human safety [4, 5]. Fault diagnosis models aim to identify anomalies and assess system health state in real time, thereby enabling early warning and maintenance actions [6]. This timely intervention can efectively prevent fault propagation and escalation. Therefore, fault diagnosis models serve as the “doctor” of industrial systems, providing critical technical support for their long-term safe, stable, and reliable operation.

With the development of advanced sensing, communication, and storage technologies, industrial systems have accumulated a large amount historical data containing important diagnostic information [7]. Based on the collected data, data-driven methods aim to learn fault-related features to accurately identify health state categories. Compared to model-based and knowledgebased methods, data-driven methods have significant strengths as they do not rely on precise mathematical models and extensive expert knowledge [8]. As a representative of data-driven methods, machine learning methods have obtained widespread attention in fault diagnosis and achieved promising outcomes. Traditional machine learning methods such as support vector machine (SVM) [9], Bayesian network [10], discriminant analysis [11, 12], and random forest [13] have been extensively employed. As industrial systems become increasingly complex, the extraction of fault features has become increasingly challenging. As an important branch of machine learning, deep learning has gained increasing interest for its great capability in automatic feature extraction and high diagnostic accuracy [14, 15]. Various network architectures, including convolution neural network [16], long short-term memory [17, 18], transformer [19], auto-encoder [20, 21], generative adversarial network [22, 23], and prototypical network [24], have been employed in fault diagnosis. However, most existing studies assume that industrial systems operate under a single working condition.

Due to changes in environment conditions and production strategies, industrial plants frequently switch between diferent operating modes. The coupling between operating modes and fault types significantly increases the complexity of fault diagnosis in multimode conditions. Li et al. [25] employed instance normalization to suppress mode-specific features and constructed a temporal attention to attend to critical moments with high mode-invariant information. Qin et al. [26] designed the adaptive attention mechanism to enhance critical features and introduced a triplet loss to improve discriminative performance under multimode conditions. Wu et al. [27] combined fine-tuning with the joint adaptation network to facilitate fault knowledge transfer across diferent modes. Yang et al. [28] used adversarial training to efectively capture shared representations by minimizing the Wasserstein distance between diferent modes, thereby improving classification performance.

Deep learning-based methods typically rely on the assumption that the training and test datasets have the same label space, but this is usually dificult to satisfy in practical systems. The historical data collected for training usually fail to cover all possible fault categories, especially rare and infrequent faults. In this context, efectively classifying known health states and identifying unknown faults is crucial for ensuring reliable decision-making and operational management [29]. Yu et al. [30] employed extreme value theory to identify unknown faults, while Peng et al. [31] introduced the soft Brownian ofset to generate synthetic samples for unknown fault identification. Outof-distribution (OOD) detection and open-set recognition in computer vision are highly similar to unknown fault identification in fault diagnosis and have been extensively studied. Bendale et al. [32] proposed OpenMax, a method that estimates the probability of a sample belonging to an unknown category based on the distance between its activation vector and class activation vectors. Additionally, various scoring functions have been proposed to distinguish between known and unknown categories, including maximum softmax probability (MSP) [33], maximum logit (MaxLogit) [34], KL matching score [34], generalized entropy (GEN) score [35] and virtual-logit matching (ViM) [36] score.

However, identifying unknown faults in multimode processes presents significant challenges. First, samples from the same category often exhibit a multicluster structure in the feature space. This discontinuity weakens the performance of threshold-based open-set recognition methods that rely on distance, probability, or confidence scores. Second, the significant intra-class variation further hinders the construction of compact, closed and continuous decision boundaries that strictly enclose samples of a single category, thereby increasing the risk of misidentification.

For instance, Fig. 1 shows the t-SNE visualizations of the features and Logit outputs generated by the proposed model under operating modes 1 and 4 of the Tennessee Eastman (TE) process dataset. The hollow circles and hollow triangles represent the samples from modes 1 and 4, respectively. The dashed region highlights the samples corresponding to Fault 2. It is evident that these samples do not form a single cohesive cluster in either the feature or Logit space. Instead, they appear to be divided into two sub-clusters corresponding to diferent operating modes. This discontinuity in both spaces makes it dificult to define a reliable threshold for unknown fault identification and to construct a continuous decision boundary that only encloses the Fault 2 category.

![](images/96c60ca8705ed328b5af367d5518a3df074ebf03d30c734d3118639ec0fc3017.jpg)  
Fig. 1. Visualization of the TE process.

To address the above problems, a fine-grained clustering and rejection network (FGCRN) is proposed. Unlike traditional methods that assign a single feature representation to each health state, this work constructs multiple representations for each state. Deep discriminative features are extracted using a combination of multiscale depthwise convolution (MSDC), bidirectional gated recurrent unit (BiGRU), and temporal attention mechanism (TAM), while a distance-based loss is incorporated to enhance feature compactness. Furthermore, unsupervised clustering is applied to divide each health state into multiple sub-clusters, using the cluster centroids as fine-grained representations. Based on extreme value theory (EVT), the rejection probability is estimated using the Mahalanobis distance between the sample and its corresponding cluster centroid, thereby achieving efective identification of unknown faults.

The main contributions of this paper are detailed as follows:

(1) A novel fine-grained clustering and rejection network is proposed for open-set fault diagnosis in multimode processes. It is capable of efectively classifying known health states and accurately identifying unknown faults.

(2) Multiscale depthwise convolution, bidirectional gated recurrent unit, and temporal attention mechanism are integrated to capture key temporal features, while the combination of batch normalization and self-adaptive instance normalization adaptively preserves discriminative statistical information, thereby facilitating accurate classification of known health states.

(3) Unsupervised clustering is employed to construct fine-grained category representations, and combined with extreme value theory to achieve efective identification of unknown faults.

(4) Extensive experiments conducted on two simulated datasets and an actual industrial process dataset demonstrate that the proposed method outperforms existing advanced methods in identifying unknown faults.

The remaining sections of this paper are structured as follows. Section 2 covers the problem formulation, the basics of clustering methods, and the application of extreme value theory in unknown fault identification. Section 3 details the construction and optimization process of the proposed method, and Section 4 validates its performance across multiple datasets. Section 5 outlines the main conclusions.

## 2. Preliminaries

## 2.1. Problem formulation

Let the training set be denoted as $D _ { \mathrm { t r } } ~ = ~ \{ x _ { \mathrm { t r } } ^ { i } , y _ { \mathrm { t r } } ^ { i } \} _ { i = 1 } ^ { N _ { \mathrm { t r } } }$ , containing $N _ { \mathrm { t r } }$ samples collected under M diferent operating modes. Each sample $x _ { \mathrm { t r } } ^ { i } \in$ $\mathbb { R } ^ { V \times T }$ represents a multivariate time window with V monitored variables and a window length of T ; $y _ { \mathrm { t r } } ^ { i } \in Y _ { \mathrm { t r } } = \{ Y _ { 1 } , Y _ { 2 } , \ldots , Y _ { k } \}$ denotes its health state label. The label set consists of one normal (healthy) category and $k - 1$ known fault categories. The test set is denoted as $D _ { \mathrm { t e } } = \{ x _ { \mathrm { t e } } ^ { i } \} _ { i = 1 } ^ { N _ { \mathrm { t e } } }$ , also collected under the same M operating modes. However, its label set expands to $Y _ { \mathrm { t e } } = \{ Y _ { 1 } , Y _ { 2 } , \dots , Y _ { k } , Y _ { k + 1 } , \dots , Y _ { k + u } \}$ , where the additional u categories represent unknown faults not seen during training. Note that neither the training set nor the test set includes operating mode labels. The objective of open-set fault diagnosis in this context is to learn a model H from $D _ { \mathrm { t r } }$ that can not only accurately classify samples from the known health state categories $\{ Y _ { 1 } , Y _ { 2 } , \ldots , Y _ { k } \}$ but also reliably identify and reject samples from the unknown fault categories $\{ Y _ { k + 1 } , \dotsc , Y _ { k + u } \}$ , as unseen or novel.

## 2.2. K-means++

K-means++ [37] is an unsupervised clustering model designed to automatically capture latent group structures in unlabeled data, ensuring that intra-cluster similarity is higher than inter-cluster similarity. It assigns each sample to the nearest cluster based on distance metrics such as Euclidean or Mahalanobis distance, and iteratively updates each cluster center as the average of the samples within the cluster until convergence. Compared to the K-means model, which initializes cluster centers randomly and may yield unstable results, K-means++ improves initialization by selecting centers that are maximally distant from each other. This strategy increases the likelihood that the initial cluster centers locate in diferent actual clusters, thereby improving the stability of the clustering results. The detailed steps of the K-means++ cluster process are presented in Algorithm 1.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 K-means++.  
1: Randomly select one sample as the first cluster center.  
2: Calculate the minimum distance $d_i$ between each sample and the existing cluster centers.  
3: Compute the probability of selecting each sample $d_i / \sum_k d_k$.  
4: Choose the next cluster center based on these calculated probabilities.  
5: Repeat Steps 2-4 until $K$ cluster centers are obtained.  
6: Assign each sample to the nearest cluster based on the distances.  
7: Update the cluster centers using the mean of the samples within each cluster.  
8: Repeat Steps 6 and 7 until the cluster centers converge.
</div>

## 2.3. Extreme value theory for unknown fault identification

Extreme value theory (EVT) is an efective method for analyzing the distribution of abnormally high or low values [38, 39]. It estimates the probability of a sample belonging to the unknown category by modeling the distance distribution between samples and the center of its corresponding known category[40]. The overall procedure is illustrated in Fig. 2. First, for each known health state, the mean feature vector of correctly classified samples is taken as the category representation. Subsequently, the distances between each correctly classified sample and its corresponding category representation are calculated. The tail of this distance distribution $\left\{ d \left( r _ { c , 1 } ^ { i } , \pmb { \mu } _ { c } \right) \right\} _ { \alpha }$ , defined as the top α largest distances, is fitted using a Weibull distribution. The rejection probability for each sample is given by the cumulative distribution function of the Weibull model. During testing, samples with rejection probabilities exceeding the predefined threshold are identified as the unknown fault.

![](images/a7809916d747e312a45a66c050ea1131211e11744fb13a01a03c553d5e83f601.jpg)  
Fit the Weibull distribution to the tail of the distance distribution for each category  
Fig. 2. EVT Modeling for unknown fault identification.

## 3. Proposed method

## 3.1. Overall architecture of FGCRN

The overall architecture of the proposed model is illustrated in Fig. 3. It includes a feature extractor, a classifier, and a fine-grained feature representation module. The feature extractor captures discriminative features closely associated with the health states. The classifier determines the health state category based on the extracted features. The fine-grained feature representation module divides the feature space of each health state based on distance diferences. EVT is applied to model the known categories, enabling the identification of previously unseen faults.

## 3.2. Feature extractor

The feature extractor is composed of three main components: a MSDC to extract local features across multiple scales, a BiGRU to capture temporal dependencies in both forward and backward directions, and a TAM to enhance the representation of critical time points.

## 3.2.1. MSDC

The MSDC employs four kernel sizes to extract local features across different temporal scales. Each channel is processed by an independent convolution kernel, which efectively reduces computational complexity. The convolution output at the jth channel and tth time step is formulated as,

![](images/41237592ef5a3dc0c3b2226ac7705845dafa9b50c0e711a8fbef09f50f0dc455.jpg)  
Fig. 3. Overall architecture of FGCRN.

$$
f _ {t, j} ^ {\mathrm{DC} _ {2 l + 1}} = \sum_ {k = t - l} ^ {t + l} x _ {k, j} w _ {k + l - t + 1, j}, \quad l = 1, 2, 3, 4,\tag{1}
$$

where w denotes the convolution kernel and the kernel size is $2 l + 1$

Batch normalization (BN) is applied after depthwise convolutions with kernel sizes of 3 and 5 to accelerate convergence. The output features after BN processing are formulated as,

$$
\begin{array}{l} f _ {t, j} ^ {\mathrm{BN}} = \gamma_ {j} ^ {\mathrm{BN}} \left(\frac {f _ {t , j} ^ {\mathrm{DC}} - \mu_ {j} ^ {\mathrm{BN}}}{\sqrt {\left(\sigma_ {j} ^ {\mathrm{BN}}\right) ^ {2} + \varepsilon}}\right) + \beta_ {j} ^ {\mathrm{BN}}, \\ \mu_ {j} ^ {\mathrm{BN}} = \frac {1}{B T} \sum_ {b = 1} ^ {B} \sum_ {t = 1} ^ {T} f _ {t, j} ^ {\mathrm{DC}}, \\ \sigma_ {j} ^ {\mathrm{BN}} = \sqrt {\frac {1}{B T} \sum_ {b = 1} ^ {B} \sum_ {t = 1} ^ {T} \left(f _ {t , j} ^ {\mathrm{DC}} - \mu_ {j} ^ {\mathrm{DC}}\right) ^ {2}}, \end{array}\tag{2}
$$

where B denotes batch size, $\gamma _ { j } ^ { \mathrm { B N } }$ and $\beta _ { j } ^ { \mathrm { B N } }$ are learnable weights. $\mu _ { j } ^ { \mathrm { B N } }$ and $\sigma _ { j } ^ { \mathrm { B N } }$ denote the mean and standard deviation of the jth channel within a

batch.

For kernel sizes of 7 and 9, self-adaptive instance normalization (SAIN) [41] is employed to weaken statistical features unrelated to the healthy state category. The output features after SAIN processing are formulated as,

$$
\begin{array}{l} f _ {t, j} ^ {\mathrm{SAIN}} = \gamma_ {j} ^ {\mathrm{SAIN}} \left(\frac {f _ {t , j} ^ {\mathrm{DC}} - \mu_ {j} ^ {\mathrm{SAIN}}}{\sqrt {\left(\sigma_ {j} ^ {\mathrm{SAIN}}\right) ^ {2} + \varepsilon}}\right) + \beta_ {j} ^ {\mathrm{SAIN}}, \\ \mu_ {j} ^ {\mathrm{SAIN}} = \frac {1}{T} \sum_ {t = 1} ^ {T} f _ {t, j} ^ {\mathrm{DC}}, \\ \sigma_ {j} ^ {\mathrm{DC}} = \sqrt {\frac {1}{T} \sum_ {t = 1} ^ {T} \left(f _ {t , j} ^ {\mathrm{DC}} - \mu_ {j} ^ {\mathrm{DC}}\right) ^ {2}}, \\ \gamma^ {\mathrm{SAIN}} = g _ {2} \left(\mathrm{ReLU} \left(g _ {1} \left(\mu_ {j} ^ {\mathrm{SAIN}}\right)\right)\right), \\ \beta^ {\mathrm{SAIN}} = g _ {4} \left(\mathrm{ReLU} \left(g _ {3} \left(\sigma_ {j} ^ {\mathrm{SAIN}}\right)\right)\right), \end{array}\tag{3}
$$

where g1, g2, g3 and g4 denote fully connected layers, $\mu _ { j } ^ { \mathrm { S A I N } }$ and $\sigma _ { j } ^ { \mathrm { S A I N } }$ denote the mean and standard deviation of each sample in the jth channel.

The combination of BN and SAIN contributes to enhancing feature diversity. SAIN adaptively suppresses statistical interference, while BN preserves statistical discriminative information. By concatenating the outputs of both operations along the channel dimension, the network is enabled to automatically select the most discriminative features for classification. The concatenated feature is formulated as,

$$
f _ {t} = \mathrm{ReLU} \left(\left[ f _ {t} ^ {\mathrm{BN} _ {3}}; f _ {t} ^ {\mathrm{BN} _ {5}}; f _ {t} ^ {\mathrm{SAIN} _ {7}}; f _ {t} ^ {\mathrm{SAIN} _ {9}} \right] _ {\mathrm{C}}\right),\tag{4}
$$

where $[ \mathrm { ~ \int _ { C } ~ }$ denotes concatenation along the channel dimension, and $f _ { t } \in$ $\mathbb { R } ^ { 4 V }$

## 3.2.2. BiGRU

As an important variant of recurrent neural network, GRU transmits information through hidden states and are superior to CNN in capturing longterm dependencies. The internal information flow of the GRU is illustrated in Fig. 4 (a). The hidden state at the current time step $h _ { t }$ is determined by the previous hidden state $h _ { t - 1 }$ and the current input feature $f _ { t }$ . This process is formulated as,

$$
\begin{array}{l} \overrightarrow {r} _ {t} = \sigma (\overrightarrow {W} _ {r} f _ {t} + \overrightarrow {U} _ {r} \overrightarrow {h} _ {t - 1}), \\ \overrightarrow {h} _ {t} ^ {0} = \tanh (\overrightarrow {W} f _ {t} + \overrightarrow {U} (\overrightarrow {r} _ {t} \odot \overrightarrow {h} _ {t - 1}), \\ \overrightarrow {z} _ {t} = \sigma (\overrightarrow {W} _ {z} f _ {t} + \overrightarrow {U} _ {z} \overrightarrow {h} _ {t - 1}), \\ \overrightarrow {h} _ {t} = (1 - \overrightarrow {z} _ {t}) \odot \overrightarrow {h} _ {t - 1} + \overrightarrow {z} _ {t} \odot \overrightarrow {h} _ {t} ^ {0}, \end{array}\tag{5}
$$

where $\vec { \boldsymbol { r } } _ { t }$ and $\vec { z } _ { t }$ control information reset and update, respectively. $\sigma$ is the sigmoid activation function; $\vec { W } _ { r } , \vec { W } _ { z } , \vec { W } , \vec { U } _ { r } , \vec { U } _ { z }$ and $\vec { U }$ are weight matrices. Compared to GRU, BiGRU combines forward and backward flows, enabling a more comprehensive capture of temporal context information in sequence data. The forward output of BiGRU is given in Eq. 5, and the backward output is formulated as,

$$
\begin{array}{r l} & {\overleftarrow {r} _ {t} = \sigma (\overleftarrow {W} _ {r} f _ {t} + \overleftarrow {U} _ {r} \overleftarrow {h} _ {t + 1}),} \\ & {\overleftarrow {h} _ {t} ^ {0} = \tanh (\overleftarrow {W} f _ {t} + \overleftarrow {U} (\overleftarrow {r} _ {t} \odot \overleftarrow {h} _ {t + 1}),} \\ & {\overleftarrow {z} _ {t} = \sigma (\overleftarrow {W} _ {z} f _ {t} + \overleftarrow {U} _ {z} \overleftarrow {h} _ {t + 1}),} \\ & {\overleftarrow {h} _ {t} = (1 - \overleftarrow {z} _ {t}) \odot \overleftarrow {h} _ {t + 1} + \overleftarrow {z} _ {t} \odot \overleftarrow {h} _ {t} ^ {0}.} \end{array}\tag{6}
$$

As shown in Fig. 4 (b), during the backward information flow, the hidden state at the current time step $\smash { \overleftarrow { h } } _ { t }$ is determined by the subsequent hidden state $\overleftarrow { h } _ { t + 1 }$ and the current input feature $f _ { t }$

![](images/cc60de6885f6c80841ca6b159bc7779ea38243e5b07b4b36f50b65848f41e084.jpg)  
Fig. 4. Forward and backward flows of BiGRU.

Finally, the forward and backward outputs are concatenated along the time dimension to fully preserve the context information. The resulting con-

catenated features are formulated as,

$$
h = \left[ \overrightarrow {h} _ {1}, \ldots , \overrightarrow {h} _ {T}, \overleftarrow {h} _ {1}, \ldots , \overleftarrow {h} _ {T} \right] _ {\mathrm{T}},\tag{7}
$$

where $[ \mathrm { ~ \int _ { T } ~ }$ denotes concatenation along the time dimension.

## 3.2.3. TAM

The forward and backward outputs of BiGRU contribute diferently to health state classification. To enhance the model’s focus on critical moments enriched with discriminative contextual information, the TAM [25] is introduced. The temporal attention map is formulated as follows,

$$
\begin{array}{r l} & {a _ {t} = \mathrm{ReLU} \left(\mathrm{Conv} \left([ a _ {1}; a _ {2} ] _ {\mathrm{C}})\right), \right.} \\ & {a _ {t} ^ {1} = g _ {6} \left(\mathrm{ReLU} \left(g _ {5} (\mathrm{Avg} (h _ {t}))\right)\right),} \\ & {a _ {t} ^ {2} = g _ {8} \left(\mathrm{ReLU} \left(g _ {7} (\mathrm{Std} (h _ {t}))\right)\right),} \end{array}\tag{8}
$$

where Avg and Std are average pooling and standard deviation operations; g , g , $g _ { 7 }$ and $g _ { 8 }$ are fully connected layers; Conv denotes convolution operation.

The output features of the BiGRU are weighted by the temporal attention map to emphasize critical time steps, and subsequently aggregated along the time dimension. The final representation is formulated as,

$$
r = \sum_ {t = 1} ^ {2 T} h _ {t} a _ {t}.\tag{9}
$$

## 3.3. Classifier

The classifier maps the captured features r to the health state predictions through a fully connected layer. The probability that the ith sample is assigned to the cth health state category is formulated as,

$$
\begin{array}{l} {p _ {c} ^ {i} = \frac {\exp (o _ {c} ^ {i})}{\sum_ {c = 1} ^ {k} \exp (o _ {c} ^ {i})},} \\ {o ^ {i} = g _ {9} (r ^ {i}),} \end{array}\tag{10}
$$

where $g _ { 9 }$ denotes the fully connected layer and k denotes the number of known health state categories.

## 3.4. Fine-grained feature representation module

In multimode processes, changes in operating conditions often lead to significant distribution diferences among samples with the same health states. Constrained by the Softmax activation function, traditional fault diagnosis models can only classify samples into predefined health states. While this setup can distinguish known health states, it inevitably expands the feature space occupied by each state. As a result, the model learns coarse-grained feature representations, increasing the risk of misidentifying unknown faults as known health states. Motivated by this limitation, the fine-grained feature representations corresponding to each health state in diferent operating modes are modeled separately to improve the identification ability of unknown faults. K-means++ is employed to cluster samples for each health condition and reveal their intrinsic grouping structure. It is applied here to partition the samples into potential M modes (line 3 of Algorithm 2).

Subsequently, the Mahalanobis distance between each correctly classified sample and its corresponding cluster center is calculated (line 6 of Algorithm 2). The distance is formulated as,

$$
d \left(r _ {c, 1} ^ {i}, \mu_ {c} ^ {m}\right) = \sqrt {\max \{\left(r _ {c , 1} ^ {i} - \mu_ {c} ^ {m}\right) ^ {\mathrm{T}} \left(\Sigma_ {c} ^ {m} + \epsilon \mathrm{I}\right) ^ {- 1} \left(r _ {c , 1} ^ {i} - \mu_ {c} ^ {m}\right) , d _ {0} \}},\tag{11}
$$

where $r _ { c , 1 } ^ { i }$ denotes the feature representation of the ith sample that is correctly classified into cth category, $\mu _ { c } ^ { m }$ denotes the mean vector of the mth operating mode cluster within cth category, $\Sigma _ { c } ^ { m }$ is the covariance matrix, ϵI is the regularization term, and $d _ { 0 }$ denotes the minimum distance threshold.

The tail distribution of Mahalanobis distances is then fitted using the Weibull distribution (line 7 of Algorithm 2), and its cumulative distribution function is used to estimate the rejection probability. The rejection probability is formulated as,

$$
q _ {c} ^ {i} = 1 - \exp \left(\frac {- \left\| d (r _ {c , 1} ^ {i} , \mu_ {c} ^ {m}) - \tau_ {c} ^ {m} \right\|}{\lambda_ {c} ^ {m}}\right) ^ {\kappa_ {c} ^ {m}},\tag{12}
$$

where $\tau _ { c } ^ { m } , \lambda _ { c } ^ { m }$ and $\kappa _ { c } ^ { m }$ are the parameters of the Weibull distribution.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Fine-grained feature representation modeling.

Input: Sample feature r, real category y, predicted category argmax{p}.
Output: Cluster center μ, cluster covariance Σ, fitted Weibull distribution's parameters τ, λ and κ.

1: for c = 1 to k do
2: Select correctly classified sample features $r_{c,1}$.
3: Partition samples into M clusters using the K-means++ algorithm.
4: for m = 1 to M do
5: Compute mean vector and covariance matrix for each cluster.
6: Compute the Mahalanobis distance between each sample and its corresponding cluster distribution via Eq. 11.
7: Fit the Weibull distribution to the tail of the distance distribution and obtain the parameters $\tau_c^m$, $\lambda_c^m$ and $\kappa_c^m$.
8: end for
9: end for

3.5. Optimization object
The cross-entropy loss is employed to assess the model's classification performance on known categories and is formulated as,

$L_1 = \sum_{i=1}^{N_{\text{tr}}} \log \left(p_{y_{\text{tr}}^i}\right)$, (13)
where $p_{y_{\text{tr}}^i}$ denotes the $y_{\text{tr}}^i$th output of p. The distance loss is employed to enhance intra-class cohesion by minimizing the Mahalanobis distance between samples and their corresponding cluster centers, and is formulated as,

$L_2 = \frac{1}{N_{\text{tr},1}} \sum_{i=1}^{N_{\text{tr},1}} \sqrt{\max\left\{\left(r_{c,1}^i - \mu_c^m\right)^{\text{T}} \left(\Sigma_c^m + \epsilon \text{I}\right)^{-1} \left(r_{c,1}^i - \mu_c^m\right), d_0\right\}}$, (14)
where $N_{\text{tr},1}$ denotes the number of correctly classified samples.
The total loss is composed of the cross-entropy loss and the distance loss, and is formulated as,

$L = L_1 + \lambda L_2$, (15)
where λ denotes the hyperparameter that balances the contributions of cross-entropy and distance losses.
</div>

## 3.6. Application workflow of FGCRN

The application process of FGCRN is illustrated in Fig. 5, which is partitioned into training and test stages.

![](images/565c155f23786b39ba8b42ef25dd9444e5b7dd8321e3f78d16f9800311fee58b.jpg)  
Fig. 5. Application workflow of FGCRN.

During the training stage, historical data are first standardized using zscore transformation to eliminate scale diferences among variables. Then, the model’s feature extractor and classifier are optimized by minimizing training loss (see Eq. 15). Model performance is evaluated using both the training and validation datasets. Subsequently, fine-grained feature representations are constructed. Correctly classified samples in each health state are selected and clustered into M groups using the K-means++ model. The tail distribution of Mahalanobis distances between samples and their corresponding cluster centers is modeled using the Weibull distribution. This process is iteratively repeated until model convergence.

During the test stage, online data are standardized using statistical parameters derived from historical data. Then, the standardized data are input into the trained model to obtain classification results. Subsequently, each test sample is allocated to the cluster within its predicted category that has the smallest Mahalanobis distance. Finally, the rejection probability of the sample belonging to the assigned cluster is calculated using Eq. 12. If the sample’s rejection probability exceeds a predefined threshold, it is identified as the unknown fault; otherwise, it is assigned a label according on the classification output.

## 4. Experimental study

## 4.1. Evaluation metrics

To clearly describe the evaluation metrics for open-set fault diagnosis, confusion matrix for category $Y _ { c }$ is provided in Table 1. $T P _ { c } , T N _ { c } , F P _ { c } .$ , and $F N _ { c }$ denote the number of known-category samples correctly classified as $Y _ { c } ,$ correctly classified as $Y _ { c }$ <sub>−</sub>, misclassified as $Y _ { c } ,$ and misclassified as $Y _ { c }$ <sub>−</sub>, respectively. The number of samples correctly identified $\mathrm { a s }$ the known category is calculated as $T K = T P _ { c } + T N _ { c } + F P _ { c } + F N _ { c }$ . Additionally, T U, F K, and F U denote the number of samples correctly identified as unknown category, misidentified as known category, and misidentified as unknown category, respectively.

Table 1 Confusion matrix.

<table><tr><td></td><td>PC is  $Y_c$ </td><td>PC is  $Y_{c-}$ </td><td>PC is unknown</td></tr><tr><td>RC is  $Y_c$ </td><td> $TP_c$ </td><td> $FN_c$ </td><td rowspan="2">FU</td></tr><tr><td>RC is  $Y_{c-}$ </td><td> $FP_c$ </td><td> $TN_c$ </td></tr><tr><td>RC is unknown</td><td colspan="2">FK</td><td>TU</td></tr></table>

<sup>a</sup> Note: RC denotes the real category, PC denotes the predicted category, $Y _ { c - }$ denotes the known categories that are not $Y _ { c } .$

To comprehensively assess the classification performance for known health states and the identification capability for unknown faults, the accuracy for open set identification is adopted, formulated as follows,

$$
A C C = \frac {\sum_ {c = 1} ^ {k} T P _ {c} + T U}{\sum_ {c = 1} ^ {k} (T P _ {c} + F N _ {c}) + T U + F K + F U},\tag{16}
$$

Additionally, misidentifying known health states as unknown fault may increase the workload of field operators, while misidentifying unknown fault as known health states could lead to incorrect decisions. To quantify these errors, the false acceptance rate (FAR) and false rejection rate (FRR) are introduced, which are formulated as follows,

$$
\begin{array}{l} F A R = \frac {F K}{F K + T U}, \\ F R R = \frac {F U}{\sum_ {c = 1} ^ {k} \left(T P _ {c} + F N _ {c}\right) + F U}. \end{array}\tag{17}
$$

The F AR denotes the proportion of unknown faults that are incorrectly identified as known health states, while the F RR refers to the proportion of known health states that are mistakenly identified as unknown faults. Higher ACC, lower F AR, and lower F RR indicate better model performance.

## 4.2. Implementation details

The Adam optimizer was employed to update the weights of the FGCRN over 50 training epochs. The learning rate was initialized at 0.01 and decayed gradually according to the following schedule $0 . 0 1 \times 0 . 3 ^ { e p o c h / / 3 }$ . The batch size was 512, and the GRU hidden layer size was 100.

Several advanced methods were selected for comparison, including MSP [33], OpenMax [32], MaxLogit [34], KL Match [34], GEN [35] and ViM [36]. These methods have been demonstrated as capable of efectively identifying out-of-distribution samples or unknown categories. In this study, these methods are employed to evaluate the challenges of identifying unknown faults in multimode processes and validate the efectiveness of the proposed model.

## 4.3. TE process

TE process [42] is commonly employed to assess the performance of fault diagnosis models. This process has been implemented in Simulink [43], and its corresponding system structure is shown in Fig. 6. TE process includes six operating modes (see Table 2) and can simulate 28 types of faults. Sixteen health state categories were selected for the experiment, as listed in Table 3. The task settings are listed in Table 4. Six tasks were designed by combining diferent operating modes, where step fault F6, random variation fault F12, and unknown fault F20 were used as the unknown fault in subtasks A, B, and C, respectively. The experimental data were generated by TE process simulation platform, with a total simulation time of 100 hours, a sampling interval of 3 minutes, and fault injection occurring at the 30th hour. For the known health state samples, an 8:1:1 split was applied to generate training, validation, and test sets, whereas all unknown fault samples were used exclusively for test.

![](images/f3db4bca559e898e8367a0d6f79dffa2a3c0fe588937bfbdadda757f20b76ace.jpg)  
Fig. 6. P&ID of the revised process model [43].

Table 2 Modes of TE process [42].

<table><tr><td>No.</td><td>G/H mass ratio</td><td>Production rate</td></tr><tr><td>M1</td><td>50/50</td><td>G: 7038kg/h, H: 7038kg/h</td></tr><tr><td>M2</td><td>10/90</td><td>G: 1408kg/h, H: 12669kg/h</td></tr><tr><td>M3</td><td>90/10</td><td>G: 10000kg/h, H: 1111kg/h</td></tr><tr><td>M4</td><td>50/50</td><td>maximum production rate</td></tr><tr><td>M5</td><td>10/90</td><td>maximum production rate</td></tr><tr><td>M6</td><td>90/10</td><td>maximum production rate</td></tr></table>

The comparison of diagnostic accuracy among diferent models is shown in Table 5. The proposed model achieves the highest accuracy across all 18 subtasks. The average accuracy of all comparison models does not exceed 83%, indicating that existing methods still face significant challenges in open-set fault diagnosis for multimode processes. As shown in Fig. 1, the feature distributions of diferent health states tend to cluster according to operating modes. Such complex data distributions make it dificult to accurately identify unknown faults using a single scoring function or category representation. Compared with MSP, OpenMax, MaxLogit, KL Match, GEN, and ViM, the proposed model improves average accuracy by 15.17%, 24.88%, 25.42%, 19.46%, 15.19%, and 24.16%, respectively. These results demonstrate its superior performance in open-set fault diagnosis for multimode processes.

Table 3 Faults of TE process used in open set fault diagnosis [42].

<table><tr><td>No.</td><td>Description</td><td>Type</td></tr><tr><td>N</td><td>Normal</td><td>-</td></tr><tr><td>F1</td><td>A/C feed ratio, B composition constant (stream 4)</td><td>Step</td></tr><tr><td>F2</td><td>B composition, A/C ratio constant (Stream 4)</td><td>Step</td></tr><tr><td>F4</td><td>Reactor cooling water inlet temperature</td><td>Step</td></tr><tr><td>F6</td><td>A feed loss (stream 1)</td><td>Step</td></tr><tr><td>F7</td><td>C header pressure loss - reduced availability (stream 4)</td><td>Step</td></tr><tr><td>F8</td><td>A, B, C feed composition (stream 4)</td><td>Random variation</td></tr><tr><td>F10</td><td>C feed temperature (stream 4)</td><td>Random variation</td></tr><tr><td>F11</td><td>Reactor cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>F12</td><td>Condenser cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>F13</td><td>Reaction kinetics</td><td>Drift</td></tr><tr><td>F14</td><td>Reactor cooling water valve</td><td>Sticking</td></tr><tr><td>F17-20</td><td>Unknown</td><td>Unknown</td></tr></table>

Table 4 Task settings on TE process dataset.

<table><tr><td>Modes</td><td>Known category</td><td>Unknown category (Task label)</td></tr><tr><td>M1,M4</td><td>N,F1,F2,F4,F7,F8,F10,F11,F13,F14,F17-F19</td><td>F6 (T1A) / F12 (T1B) / F20 (T1C)</td></tr><tr><td>M2,M5</td><td>N,F1,F2,F4,F7,F8,F10,F11,F13,F14,F17-F19</td><td>F6 (T2A) / F12 (T2B) / F20 (T2C)</td></tr><tr><td>M3,M6</td><td>N,F1,F2,F4,F7,F8,F10,F11,F13,F14,F17-F19</td><td>F6 (T3A) / F12 (T3B) / F20 (T3C)</td></tr><tr><td>M1,M2</td><td>N,F1,F2,F4,F7,F8,F10,F11,F13,F14,F17-F19</td><td>F6 (T4A) / F12 (T4B) / F20 (T4C)</td></tr><tr><td>M3,M4</td><td>N,F1,F2,F4,F7,F8,F10,F11,F13,F14,F17-F19</td><td>F6 (T5A) / F12 (T5B) / F20 (T5C)</td></tr><tr><td>M5,M6</td><td>N,F1,F2,F4,F7,F8,F10,F11,F13,F14,F17-F19</td><td>F6 (T6A) / F12 (T6B) / F20 (T6C)</td></tr></table>

The FRR results are summarized in Table 6. All models maintain a low misidentification rate of known categories as unknown faults, with average FRRs between 1% and 2%. This demonstrates that these models are able to accurately identify known health state categories. The proposed model achieves the lowest average FRR of 1.12%, efectively reducing the workload of field personnel caused by unnecessary verification of unknown fault categories when known health states are mistakenly identified as unknown faults.

Table 5 Accuracy for open set fault diagnosis on TE process dataset.

<table><tr><td></td><td>MSP</td><td>OpenMax</td><td>MaxLogit</td><td>KL Match</td><td>GEN</td><td>ViM</td><td>Proposed</td></tr><tr><td>T1A</td><td>94.35%</td><td>93.02%</td><td>91.78%</td><td>95.38%</td><td>94.74%</td><td>94.41%</td><td>97.51%</td></tr><tr><td>T1B</td><td>95.48%</td><td>69.81%</td><td>69.94%</td><td>84.43%</td><td>92.12%</td><td>58.85%</td><td>97.85%</td></tr><tr><td>T1C</td><td>83.81%</td><td>53.86%</td><td>55.11%</td><td>67.25%</td><td>81.03%</td><td>76.35%</td><td>97.31%</td></tr><tr><td>T2A</td><td>94.84%</td><td>93.43%</td><td>92.40%</td><td>94.58%</td><td>95.12%</td><td>94.81%</td><td>98.05%</td></tr><tr><td>T2B</td><td>82.61%</td><td>63.52%</td><td>89.74%</td><td>65.06%</td><td>91.10%</td><td>55.42%</td><td>98.02%</td></tr><tr><td>T2C</td><td>63.66%</td><td>57.10%</td><td>57.73%</td><td>60.05%</td><td>62.20%</td><td>55.50%</td><td>98.42%</td></tr><tr><td>T3A</td><td>94.38%</td><td>93.35%</td><td>92.99%</td><td>94.95%</td><td>95.36%</td><td>97.27%</td><td>98.56%</td></tr><tr><td>T3B</td><td>79.72%</td><td>88.82%</td><td>56.15%</td><td>88.54%</td><td>86.96%</td><td>95.71%</td><td>99.05%</td></tr><tr><td>T3C</td><td>93.45%</td><td>66.51%</td><td>68.54%</td><td>86.10%</td><td>88.87%</td><td>65.12%</td><td>98.26%</td></tr><tr><td>T4A</td><td>91.93%</td><td>91.80%</td><td>91.14%</td><td>92.19%</td><td>91.86%</td><td>93.87%</td><td>98.19%</td></tr><tr><td>T4B</td><td>87.42%</td><td>68.23%</td><td>77.30%</td><td>84.75%</td><td>88.37%</td><td>55.53%</td><td>98.62%</td></tr><tr><td>T4C</td><td>56.86%</td><td>56.69%</td><td>56.38%</td><td>56.72%</td><td>56.80%</td><td>62.98%</td><td>98.09%</td></tr><tr><td>T5A</td><td>92.80%</td><td>91.77%</td><td>91.85%</td><td>94.19%</td><td>93.35%</td><td>94.72%</td><td>97.25%</td></tr><tr><td>T5B</td><td>72.33%</td><td>53.50%</td><td>53.75%</td><td>60.74%</td><td>67.78%</td><td>57.03%</td><td>98.22%</td></tr><tr><td>T5C</td><td>74.50%</td><td>53.65%</td><td>54.95%</td><td>64.69%</td><td>69.87%</td><td>56.90%</td><td>97.04%</td></tr><tr><td>T6A</td><td>94.58%</td><td>94.09%</td><td>92.90%</td><td>93.75%</td><td>94.14%</td><td>96.90%</td><td>98.30%</td></tr><tr><td>T6B</td><td>80.57%</td><td>71.09%</td><td>57.19%</td><td>73.40%</td><td>82.89%</td><td>62.64%</td><td>98.20%</td></tr><tr><td>T6C</td><td>58.26%</td><td>56.54%</td><td>57.20%</td><td>57.53%</td><td>58.63%</td><td>55.70%</td><td>97.66%</td></tr><tr><td>Avg</td><td>82.86%</td><td>73.15%</td><td>72.61%</td><td>78.57%</td><td>82.84%</td><td>73.87%</td><td>98.03%</td></tr></table>

The average FAR results are presented in Fig. 7. The comparison models show average FARs exceeding 50%, indicating a high risk of misidentifying unknown faults as known health states. In contrast, the proposed model achieves a significantly lower FAR of only 1.28%, efectively avoiding the misidentification of unknown faults and reducing the risk of incorrect decisions.

## 4.4. CSTR

Continuous stirred tank reactor (CSTR) process [44] is also widely used to access fault diagnosis models, and its structure is presented in Fig. 8. The control of reactor temperature T is achieved through the regulation of cooling water flow rate Q. Nine fault types are simulated, as listed in Table 7. The original setpoint is defined as operating mode M1, with setpoints increased by 5 K and 10 K defined as modes M2 and M3, respectively. The task settings are listed in Table 8. Each task consists of a pairwise combination of two operating modes, and faults F8 and F9 are set as unknown faults, corresponding to subtasks A and B, respectively. The experimental data were generated using the Simulink model of the CSTR process over a total of 20 hours, sampled once per minute, with fault injection starting at the 200th minute. The division of training, validation and test sets follows the same strategy used in TE process.

Table 6 FRR for open set fault diagnosis on TE process dataset.

<table><tr><td></td><td>MSP</td><td>OpenMax</td><td>MaxLogit</td><td>KL Match</td><td>GEN</td><td>ViM</td><td>Proposed</td></tr><tr><td>T1A</td><td>2.28%</td><td>2.43%</td><td>2.13%</td><td>2.22%</td><td>2.28%</td><td>1.13%</td><td>1.15%</td></tr><tr><td>T1B</td><td>2.28%</td><td>2.43%</td><td>2.13%</td><td>2.22%</td><td>2.28%</td><td>1.13%</td><td>1.26%</td></tr><tr><td>T1C</td><td>2.28%</td><td>2.43%</td><td>2.13%</td><td>2.22%</td><td>2.28%</td><td>1.13%</td><td>1.26%</td></tr><tr><td>T2A</td><td>1.40%</td><td>1.24%</td><td>1.37%</td><td>1.24%</td><td>1.32%</td><td>1.21%</td><td>1.26%</td></tr><tr><td>T2B</td><td>1.40%</td><td>1.24%</td><td>1.37%</td><td>1.24%</td><td>1.32%</td><td>1.21%</td><td>1.15%</td></tr><tr><td>T2C</td><td>1.40%</td><td>1.24%</td><td>1.37%</td><td>1.24%</td><td>1.32%</td><td>1.21%</td><td>1.15%</td></tr><tr><td>T3A</td><td>1.26%</td><td>0.85%</td><td>1.13%</td><td>1.15%</td><td>1.15%</td><td>1.18%</td><td>1.15%</td></tr><tr><td>T3B</td><td>1.26%</td><td>0.85%</td><td>1.13%</td><td>1.15%</td><td>1.15%</td><td>1.18%</td><td>0.93%</td></tr><tr><td>T3C</td><td>1.26%</td><td>0.85%</td><td>1.13%</td><td>1.15%</td><td>1.15%</td><td>1.18%</td><td>0.93%</td></tr><tr><td>T4A</td><td>1.90%</td><td>1.46%</td><td>2.01%</td><td>1.59%</td><td>1.98%</td><td>1.13%</td><td>0.93%</td></tr><tr><td>T4B</td><td>1.90%</td><td>1.46%</td><td>2.01%</td><td>1.59%</td><td>1.98%</td><td>1.13%</td><td>1.15%</td></tr><tr><td>T4C</td><td>1.90%</td><td>1.46%</td><td>2.01%</td><td>1.59%</td><td>1.98%</td><td>1.13%</td><td>1.15%</td></tr><tr><td>T5A</td><td>2.31%</td><td>1.42%</td><td>2.34%</td><td>2.40%</td><td>2.28%</td><td>1.10%</td><td>1.15%</td></tr><tr><td>T5B</td><td>2.31%</td><td>1.42%</td><td>2.34%</td><td>2.40%</td><td>2.28%</td><td>1.10%</td><td>1.07%</td></tr><tr><td>T5C</td><td>2.31%</td><td>1.42%</td><td>2.34%</td><td>2.40%</td><td>2.28%</td><td>1.10%</td><td>1.07%</td></tr><tr><td>T6A</td><td>1.54%</td><td>1.43%</td><td>1.40%</td><td>1.62%</td><td>1.48%</td><td>1.10%</td><td>1.07%</td></tr><tr><td>T6B</td><td>1.54%</td><td>1.43%</td><td>1.40%</td><td>1.62%</td><td>1.48%</td><td>1.10%</td><td>1.15%</td></tr><tr><td>T6C</td><td>1.54%</td><td>1.43%</td><td>1.40%</td><td>1.62%</td><td>1.48%</td><td>1.10%</td><td>1.15%</td></tr><tr><td>Avg</td><td>1.78%</td><td>1.47%</td><td>1.73%</td><td>1.70%</td><td>1.75%</td><td>1.14%</td><td>1.12%</td></tr></table>

The diagnostic accuracies of diferent models are compared in Table 9. The proposed model consistently attains the top accuracies across all subtasks. Compared with MSP, OpenMax, MaxLogit, KL Match, GEN, and ViM, the proposed method shows average accuracy improvements of 40.51%, 27.52%, 45.54%, 37.67%, 40.52% and 33.02%, respectively. These results highlight the strong capability of the proposed model in handling open-set fault diagnosis for multimode processes.

![](images/d0c739654c63fe6a80145ff5693dbe8bc9e5b56feee17c61aa6ff5cecfb34933.jpg)

Fig. 7. Average FAR for open set fault diagnosis on TE process dataset.  
![](images/291cfbb5cf69e3110cd7bd39a6f3c425aeff489b86e0073b89ceeda1d84b359b.jpg)  
Fig. 8. Structure of CSTR [44].

The average FRR and FAR of each model are shown in Fig. 9. While all models exhibit relatively low average FRRs, there are significant diferences in average FARs. The proposed model achieves the lowest average FAR, indicating its strong capability in accurately identifying unknown faults, whereas the comparison models are more prone to misidentifying unknown faults as known health states.

## 4.5. IPCTF

The intelligent process control-test facility developed by Wuhan University of Technology (see Fig. 10) was employed to evaluate the performance of the proposed model in the real system. This system controls the temperature of the heat source loop via the condenser. Experimental data were collected under three health states: normal (N), pump blockage in the heat source loop (F1), and pipeline blockage in the condenser loop (F2). The operating modes and task settings are listed in Table 10.

Table 7 Faults of CSTR used in open set fault diagnosis[44].

<table><tr><td>Health state</td><td>Description</td></tr><tr><td>N</td><td>Normal</td></tr><tr><td>F1</td><td> $C_i = C_{i,0} + 0.001t$ </td></tr><tr><td>F2</td><td> $T_i = T_{i,0} + 0.05t$ </td></tr><tr><td>F3</td><td> $C = C_0 + 0.001t$ </td></tr><tr><td>F4</td><td> $T = T_0 + 0.05t$ </td></tr><tr><td>F5</td><td> $Q_c = Q_{c,0} - 0.1t$ </td></tr><tr><td>F6</td><td> $T_{ci} = T_{ci,0} + 0.05t$ </td></tr><tr><td>F7</td><td> $T_c = T_{c,0} + 0.05t$ </td></tr><tr><td>F8</td><td> $a = a_0\exp(-0.0005t)$ </td></tr><tr><td>F9</td><td> $b = b_0\exp(-0.001t)$ </td></tr></table>

Table 8 Task settings on CSTR dataset.

<table><tr><td>Modes</td><td>Known category</td><td>Unknown category (Task label)</td></tr><tr><td>M1,M2</td><td>N,F1,F2,F3,F4,F5,F6,F7</td><td>F8 (T7A) / F9 (T7B)</td></tr><tr><td>M1,M3</td><td>N,F1,F2,F3,F4,F5,F6,F7</td><td>F8 (T8A) / F9 (T8B)</td></tr><tr><td>M2,M3</td><td>N,F1,F2,F3,F4,F5,F6,F7</td><td>F8 (T9A) / F9 (T9B)</td></tr></table>

Table 9 Accuracy for open set fault diagnosis on CSTR dataset.

<table><tr><td></td><td>MSP</td><td>OpenMax</td><td>MaxLogit</td><td>KL Match</td><td>GEN</td><td>ViM</td><td>Proposed</td></tr><tr><td>T1A</td><td>61.30%</td><td>68.10%</td><td>57.18%</td><td>63.19%</td><td>61.30%</td><td>62.04%</td><td>99.07%</td></tr><tr><td>T1B</td><td>44.03%</td><td>98.61%</td><td>44.03%</td><td>44.31%</td><td>44.03%</td><td>87.92%</td><td>99.07%</td></tr><tr><td>T2A</td><td>67.87%</td><td>56.71%</td><td>76.94%</td><td>67.78%</td><td>67.82%</td><td>56.30%</td><td>98.33%</td></tr><tr><td>T2B</td><td>54.21%</td><td>75.69%</td><td>44.03%</td><td>69.86%</td><td>54.21%</td><td>67.04%</td><td>98.56%</td></tr><tr><td>T3A</td><td>61.71%</td><td>43.94%</td><td>50.05%</td><td>56.62%</td><td>61.71%</td><td>43.98%</td><td>99.21%</td></tr><tr><td>T3B</td><td>57.27%</td><td>81.30%</td><td>43.98%</td><td>61.67%</td><td>57.27%</td><td>74.03%</td><td>95.19%</td></tr><tr><td>Avg</td><td>57.73%</td><td>70.73%</td><td>52.70%</td><td>60.57%</td><td>57.72%</td><td>65.22%</td><td>98.24%</td></tr></table>

The diagnostic performance comparison among the evaluated models is shown in Table 11. The proposed method consistently outperforms the others in both subtasks. Moreover, the average accuracy of the proposed model and

![](images/b3873c9eb562b835567cd0d027807dc23c865e7c6abd8f2bada38faa33e48f8d.jpg)

![](images/ac3dbe2f7e0582800d5b6b11063a330800e0491ced9b4c7765e5aa085381b4ce.jpg)  
Fig. 9. Average FRR and FAR for open set fault diagnosis on CSTR dataset.

Table 10 Operating modes and task settings for open set fault diagnosis on IPCTF dataset.

<table><tr><td>Mode No.</td><td>Power of the heat source (kW)</td><td>Setpoint of the cooling loop (°C)</td></tr><tr><td>M1</td><td>10</td><td>29</td></tr><tr><td>M2</td><td>8</td><td>24</td></tr><tr><td>Task No.</td><td>Known category</td><td>Unknown category</td></tr><tr><td>T10A</td><td>N,F1</td><td>F2</td></tr><tr><td>T10B</td><td>N,F2</td><td>F1</td></tr></table>

ViM significantly outperform all other comparison methods, demonstrating strong potential for practical system deployment.

The confusion matrices of our method and ViM are presented in Fig. 11.

![](images/c83ccbab510bf6d528c8948f4f6eceea555637d408f70611d6b43c93db28cf75.jpg)  
Fig. 10. Structure of IPCTF.

Table 11 Accuracy for open set fault diagnosis on IPCTF dataset.

<table><tr><td></td><td>MSP</td><td>OpenMax</td><td>MaxLogit</td><td>KL Match</td><td>GEN</td><td>ViM</td><td>Proposed</td></tr><tr><td>T10A</td><td>28.97%</td><td>35.72%</td><td>28.97%</td><td>28.97%</td><td>28.97%</td><td>99.51%</td><td>99.72%</td></tr><tr><td>T10B</td><td>16.58%</td><td>59.80%</td><td>16.72%</td><td>16.63%</td><td>16.58%</td><td>99.40%</td><td>99.65%</td></tr><tr><td>Avg</td><td>22.78%</td><td>47.76%</td><td>22.85%</td><td>22.80%</td><td>22.78%</td><td>99.46%</td><td>99.69%</td></tr></table>

Labels 0 and 1 correspond to known health states, while label 2 denotes the unknown fault. In subtask T10A, both models efectively identify all unknown faults, with a FAR of 0%. However, ViM exhibits a higher FRR than the proposed model. In subtask T10B, the proposed method outperforms ViM in both FAR and FRR, further demonstrating its practical efectiveness.

t-SNE was employed to visualize the feature representations generated by the proposed model, as illustrated in Fig. 12. In subtasks T10A and T10B, the features of normal samples exhibit clustered distributions corresponding to two operating modes, while the unknown-category samples are more dispersed in the feature space. This makes it challenging to construct a unified decision hyperplane that includes only normal samples. The pentagrams in the figure indicate the cluster centers. The results clearly demonstrate that the clusters associated with these centers are well-separated and highly distinguishable. This study quantifies the confidence that a sample belongs to the unknown category based on its Mahalanobis distance to cluster centers, efectively addressing the open-set fault diagnosis problem under multimode conditions.

![](images/810b8ccde8461aadc021629d78b5f7212f22e12f09a4a1629f74e46c3e49ce58.jpg)

![](images/e432197a3eab48a7da21e0fd83c1e88eba7665562e2617be70fc74dc1eba9977.jpg)

![](images/44708623908b29154dab7c07848e5eca169cfd0107f2103311220c07a900de27.jpg)

![](images/850acd080783fc15e6f834a44687b6d30eb5cc246c786c3ebb09b046fde2acc2.jpg)  
Fig. 11. Confusion matrix for open set fault diagnosis on IPCTF dataset.

![](images/0eb79bd3fd5b2d4ba95f9d2810f6f47c49d1f87656f6bb09ff5bb26fd61d0588.jpg)

![](images/79e33a8862bde8e23b3b17a8c0691e7306f37ed7df9ccf3bbe4faa7ee1a64008.jpg)  
Fig. 12. Feature visualization on IPCTF dataset.

## 4.6. Ablation Study

Ablation studies were carried out to examine the efectiveness of each component in the proposed method. The configurations of diferent model variants are listed in Table 12. Specifically, A1 replaces BiGRU with GRU; A2 removes the TAM module; A3 employs BN for normalization; A4 uses SAIN for normalization; A5 removes the $L _ { 2 }$ loss term; and A6 constructs a single category representation for each health state.

Table 12 Configurations of diferent model variants for ablation experiments.

<table><tr><td></td><td>A1</td><td>A2</td><td>A3</td><td>A4</td><td>A5</td><td>A6</td></tr><tr><td>BiGRU</td><td>GRU</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>TAM</td><td>√</td><td>-</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>BN+SAIN</td><td>√</td><td>√</td><td>BN</td><td>SAIN</td><td>√</td><td>√</td></tr><tr><td> $L_2$ </td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>√</td></tr><tr><td>Cluster</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td></tr></table>

The diagnosis performance of diferent models is compared in Table 13. The proposed FGCRN obtains the top average accuracy of 98.03%. Additionally, it attains the highest minimum accuracy across tasks at 96.81%, outperforming models A1 to A6 by 12.65%, 33.93%, 4.05%, 9.90%, 33.11%, and 40.52%, respectively. These results indicate that the proposed method has strong identification capabilities for unknown faults under multiple operating modes, while demonstrating greater stability and robustness. Compared with models A1 and A2, the proposed model achieves average classification accuracy increases of 2.27% and 5.98%, respectively. This reveals that the backward flow in BiGRU efectively captures health-related temporal features, and the TAM further enhances the model’s discriminative ability by focusing on critical forward and backward time steps. Compared with A3 and A4, the proposed model shows improvements of 0.57% and 2.08%, respectively, which validates the complementarity of SAIN and BN. SAIN adaptively eliminates statistical interference, while BN preserves discriminative statistical features, thereby enhancing the model’s reliability across multiple scenarios. For example, the accuracy of A3 on subtask T1B improved from 92.99% to 97.85% after incorporating the SAIN module. Similarly, the accuracy of A4 on subtask T2B increased from 87.14% to 98.02% after integrating the BN module. Furthermore, the proposed model outperformed A5 by 5.09%, confirming that the Mahalanobis distance loss enhances feature compactness and aids in identifying unknown faults. It also outperformed A6 by 8.09%, indicating that constructing fine-grained representations for each health state facilitates the formation of a reliable feature space constrained to corresponding category samples, thereby enhancing the identification of unknown faults. Overall, the combination of these components enables the proposed method to provide reliable and stable open-set fault diagnosis performance across multiple operating modes.

Table 13 Accuracy for ablation experiments on TE process dataset.

<table><tr><td></td><td>A1</td><td>A2</td><td>A3</td><td>A4</td><td>A5</td><td>A6</td><td>Proposed</td></tr><tr><td>T1A</td><td>93.10%</td><td>94.07%</td><td>96.98%</td><td>95.51%</td><td>97.51%</td><td>97.12%</td><td>97.51%</td></tr><tr><td>T1B</td><td>84.39%</td><td>77.24%</td><td>92.99%</td><td>96.97%</td><td>97.72%</td><td>98.33%</td><td>97.85%</td></tr><tr><td>T1C</td><td>95.77%</td><td>96.73%</td><td>97.46%</td><td>96.13%</td><td>97.07%</td><td>97.51%</td><td>97.31%</td></tr><tr><td>T2A</td><td>97.41%</td><td>98.43%</td><td>97.56%</td><td>98.82%</td><td>98.59%</td><td>94.84%</td><td>98.05%</td></tr><tr><td>T2B</td><td>98.18%</td><td>98.26%</td><td>98.23%</td><td>87.14%</td><td>77.13%</td><td>64.35%</td><td>98.02%</td></tr><tr><td>T2C</td><td>97.30%</td><td>63.11%</td><td>97.34%</td><td>97.97%</td><td>63.93%</td><td>56.52%</td><td>98.42%</td></tr><tr><td>T3A</td><td>98.38%</td><td>97.22%</td><td>97.91%</td><td>98.74%</td><td>99.30%</td><td>99.05%</td><td>98.56%</td></tr><tr><td>T3B</td><td>98.87%</td><td>98.94%</td><td>98.63%</td><td>99.11%</td><td>99.49%</td><td>99.49%</td><td>99.05%</td></tr><tr><td>T3C</td><td>97.95%</td><td>97.98%</td><td>97.50%</td><td>97.95%</td><td>98.52%</td><td>96.48%</td><td>98.26%</td></tr><tr><td>T4A</td><td>97.22%</td><td>96.22%</td><td>97.68%</td><td>98.34%</td><td>96.45%</td><td>93.41%</td><td>98.19%</td></tr><tr><td>T4B</td><td>98.70%</td><td>98.46%</td><td>98.32%</td><td>98.25%</td><td>93.32%</td><td>95.59%</td><td>98.62%</td></tr><tr><td>T4C</td><td>88.80%</td><td>77.20%</td><td>97.72%</td><td>98.18%</td><td>82.48%</td><td>65.30%</td><td>98.09%</td></tr><tr><td>T5A</td><td>96.80%</td><td>94.94%</td><td>96.97%</td><td>92.21%</td><td>97.86%</td><td>96.64%</td><td>97.25%</td></tr><tr><td>T5B</td><td>95.66%</td><td>92.80%</td><td>97.93%</td><td>87.45%</td><td>85.49%</td><td>82.42%</td><td>98.22%</td></tr><tr><td>T5C</td><td>96.28%</td><td>96.62%</td><td>96.57%</td><td>94.20%</td><td>93.59%</td><td>94.79%</td><td>97.04%</td></tr><tr><td>T6A</td><td>97.96%</td><td>96.80%</td><td>98.27%</td><td>97.62%</td><td>98.92%</td><td>98.94%</td><td>98.30%</td></tr><tr><td>T6B</td><td>93.59%</td><td>92.70%</td><td>98.68%</td><td>97.56%</td><td>97.42%</td><td>90.06%</td><td>98.20%</td></tr><tr><td>T6C</td><td>97.34%</td><td>89.21%</td><td>97.59%</td><td>95.03%</td><td>98.06%</td><td>98.04%</td><td>97.66%</td></tr><tr><td>Min</td><td>84.39%</td><td>63.11%</td><td>92.99%</td><td>87.14%</td><td>63.93%</td><td>56.52%</td><td>97.04%</td></tr><tr><td>Avg</td><td>95.76%</td><td>92.05%</td><td>97.46%</td><td>95.95%</td><td>92.94%</td><td>89.94%</td><td>98.03%</td></tr></table>

## 5. Conclusion

This paper proposes a novel fine-grained clustering and rejection network for open-set fault diagnosis in multimode industrial processes. This method focuses on constructing multiple feature representations for each known health state to enhance the identification of unknown faults. Extensive experiments validate that the proposed method outperforms other advanced models. The principal conclusions are outlined as follows.

• The integration of MSDC, BiGRU, and TAM enables eficient extraction of deep feature representations associated with known health states from complex data distributions.

• The combination of BN and SAIN adaptively preserves discriminative statistical features, enhancing the model’s adaptability to diverse operating modes.

• Distance loss enhances the discriminability of the feature space by compressing the feature distributions within the same health state, thereby facilitating the identification of unknown faults based on feature distance.

• Constructing fine-grained representations for each health state further depicts more detailed internal structures, which significantly strengthens the model’s capability to identify unknown faults.

Although this study has achieved encouraging results, there are still some limitations. The proposed model categorizes all unknown faults into a single ”unknown” category without performing a more detailed unsupervised classification. Moreover, once these unknown faults are labeled and incorporated into known categories, they could be utilized to further refine the diagnostic model. Future work could focus on developing automated incremental learning-based fault diagnosis models for multimode processes.

## References

[1] C. Lou, X. Li, M. A. Atoui, J. Jiang, Enhanced fault diagnosis method using conditional gaussian network for dynamic processes, Engineering Applications of Artificial Intelligence 93 (2020). doi:10.1016/j. engappai.2020.103704.

[2] X. T. Bi, R. S. Qin, D. Y. Wu, S. D. Zheng, J. S. Zhao, One step forward for smart chemical process fault detection and diagnosis, Computers & Chemical Engineering 164 (2022) 19. doi:10.1016/j.compchemeng. 2022.107884.

[3] C. Shang, S. X. Ding, H. Ye, Distributionally robust fault detection design and assessment for dynamical systems, Automatica 125 (2021). doi:10.1016/j.automatica.2020.109434.

[4] K. Zhong, M. Han, B. Han, Data-driven based fault prognosis for industrial systems: a concise overview, IEEE/CAA Journal of Automatica Sinica 7 (2) (2020) 330–345. doi:10.1109/JAS.2019.1911804.

[5] R. Arunthavanathan, F. Khan, S. Ahmed, S. Imtiaz, An analysis of process fault diagnosis methods from safety perspectives, Computers & Chemical Engineering 145 (2021). doi:10.1016/j.compchemeng.2020. 107197.

[6] Z. Mian, X. Deng, X. Dong, Y. Tian, T. Cao, K. Chen, T. Al Jaber, A literature review of fault diagnosis based on ensemble learning, Engineering Applications of Artificial Intelligence 127 (2024). doi:10.1016/ j.engappai.2023.107357.

[7] W. J. Wang, Q. C. Jiang, X. F. Yan, W. M. Zhong, Vgmtnet: A variational gaussian mixture label transfer network for industrial fault diagnosis, Expert Systems with Applications 291 (2025). doi:10.1016/j. eswa.2025.128472.

[8] Y.-L. He, Y. Zhao, X. Hu, X.-N. Yan, Q.-X. Zhu, Y. Xu, Fault diagnosis using novel adaboost based discriminant locality preserving projection with resamples, Engineering Applications of Artificial Intelligence 91 (2020). doi:10.1016/j.engappai.2020.103631.

[9] L. H. Chiang, M. E. Kotanchek, A. K. Kordon, Fault diagnosis based on fisher discriminant analysis and support vector machines, Computers & Chemical Engineering 28 (8) (2004) 1389–1401. doi:10.1016/j. compchemeng.2003.10.002.

[10] M. A. Atoui, Fault diagnosis using pca-bayesian network classifier with unknown faults, in: 2020 European Control Conference (ECC), IEEE, 2020, pp. 2039–2044.

[11] S. Yin, S. X. Ding, A. Haghani, H. Hao, P. Zhang, A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark tennessee eastman process, Journal of Process Control 22 (9) (2012) 1567–1581. doi:10.1016/j.jprocont.2012.06.009.

[12] C. Shang, L. Zhao, X. Huang, H. Ye, D. Huang, Group-sparsityenforcing fault discrimination and estimation with dynamic process data, Journal of Process Control 105 (2021) 236–249. doi:10.1016/ j.jprocont.2021.08.003.

[13] Z. Chai, C. Zhao, Enhanced random forest with concurrent analysis of static and dynamic nodes for industrial fault classification, IEEE Transactions on Industrial Informatics 16 (1) (2020) 54–66. doi: 10.1109/tii.2019.2915559.

[14] C. Lou, M. A. Atoui, Unknown health states recognition with collectivedecision-based deep learning networks in predictive maintenance applications, Mathematics 12 (1) (2024). doi:10.3390/math12010089.

[15] Y. Chen, R. Zhang, Deep multiscale convolutional model with multihead self-attention for industrial process fault diagnosis, IEEE Transactions on Systems Man Cybernetics-Systems 55 (4) (2025) 2503–2512. doi: 10.1109/tsmc.2024.3523708.

[16] H. Wu, J. Zhao, Deep convolutional neural network model based chemical process fault diagnosis, Computers & Chemical Engineering 115 (2018) 185–197. doi:https://doi.org/10.1016/j.compchemeng. 2018.04.009.

[17] H. Zhao, S. Sun, B. Jin, Sequential fault diagnosis based on lstm neural network, IEEE Access 6 (2018) 12929–12939. doi:10.1109/access. 2018.2794765.

[18] T. Huang, Q. Zhang, X. Tang, S. Zhao, X. Lu, A novel fault diagnosis method based on cnn and lstm and its application in fault diagnosis for complex systems, Artificial Intelligence Review 55 (2) (2022) 1289–1315. doi:10.1007/s10462-021-09993-z.

[19] K. Zhou, Y. Tong, X. Li, X. Wei, H. Huang, K. Song, X. Chen, Exploring global attention mechanism on fault detection and diagnosis for complex engineering processes, Process Safety and Environmental Protection 170 (2023) 660–669. doi:10.1016/j.psep.2022.12.055.

[20] S. Chen, J. Yu, S. Wang, One-dimensional convolutional auto-encoderbased feature learning for fault diagnosis of multivariate processes, Jour-

nal of Process Control 87 (2020) 54–67. doi:10.1016/j.jprocont. 2020.01.004.

[21] Y. Ma, H. Shi, S. Tan, Y. Tao, B. Song, Consistency regularization auto-encoder network for semi-supervised process fault diagnosis, IEEE Transactions on Instrumentation and Measurement 71 (2022). doi: 10.1109/tim.2022.3184346.

[22] R. Qin, J. Zhao, High-eficiency generative adversarial network model for chemical process fault diagnosis, in: 13th IFAC Symposium on Dynamics and Control of Process Systems, including Biosystems (DYCOPS), Vol. 55, 2022, pp. 732–737. doi:10.1016/j.ifacol.2022.07.531.

[23] W. Du, J. Yang, G. Meng, Fault diagnosis for dynamic system based on the independent latent space reconstruction of generative adversarial network, Journal of Process Control 125 (2023) 28–40. doi:10.1016/ j.jprocont.2023.04.001.

[24] K. Li, C. Shang, H. Ye, Reweighted regularized prototypical network for few-shot fault diagnosis, IEEE Transactions on Neural Networks and Learning Systems 35 (5) (2024) 6206–6217. doi:10.1109/tnnls.2022. 3232394.

[25] G. Li, M. A. Atoui, X. Li, Attention-based multiscale temporal fusion network for uncertain-mode fault diagnosis in multimode processes, Process Safety and Environmental Protection 201 (2025) 107554. doi:https://doi.org/10.1016/j.psep.2025.107554.

[26] R. Qin, J. Zhao, Adaptive multiscale convolutional neural network model for chemical process fault diagnosis, Chinese Journal of Chemical Engineering 50 (2022) 398–411. doi:10.1016/j.cjche.2022.10.001.

[27] H. Wu, J. Zhao, Fault detection and diagnosis based on transfer learning for multimode chemical processes, Computers & Chemical Engineering 135 (2020) 106731. doi:https://doi.org/10.1016/j.compchemeng. 2020.106731.

[28] J. F. Yang, N. Zhang, Y. L. He, Q. X. Zhu, Y. Xu, Novel dual-network autoencoder based adversarial domain adaptation with wasserstein divergence for fault diagnosis of unlabeled data, Expert Systems with Applications 238 (2024) 11. doi:10.1016/j.eswa.2023.122393.

[29] C. Lou, M. A. Atoui, X. Li, Novel online discriminant analysis based schemes to deal with observations from known and new classes: Application to industrial systems, Engineering Applications of Artificial Intelligence 111 (2022) 104811.

[30] X. Yu, Z. Zhao, X. Zhang, Q. Zhang, Y. Liu, C. Sun, X. Chen, Deeplearning-based open set fault diagnosis by extreme value theory, IEEE Transactions on Industrial Informatics 18 (1) (2022) 185–196. doi: 10.1109/TII.2021.3070324.

[31] P. Peng, J. Lu, T. Xie, S. Tao, H. Wang, H. Zhang, Open-set fault diagnosis via supervised contrastive learning with negative outof-distribution data augmentation, IEEE Transactions on Industrial Informatics 19 (3) (2023) 2463–2473. doi:10.1109/TII.2022.3149935.

[32] A. Bendale, T. E. Boult, Towards open set deep networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 1563–1572.

[33] D. Hendrycks, K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples in neural networks, in: International Conference on Learning Representations, 2017.

[34] D. Hendrycks, S. Basart, M. Mazeika, A. Zou, J. Kwon, M. Mostajabi, J. Steinhardt, D. Song, Scaling out-of-distribution detection for real-world settings, in: International Conference on Machine Learning, PMLR, 2022, pp. 8759–8773.

[35] X. Liu, Y. Lochman, C. Zach, Gen: Pushing the limits of softmaxbased out-of-distribution detection, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 23946– 23955.

[36] H. Wang, Z. Li, L. Feng, W. Zhang, Vim: Out-of-distribution with virtual-logit matching, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4921–4930.

[37] D. Arthur, S. Vassilvitskii, k-means++ the advantages of careful seeding, in: Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms, 2007, pp. 1027–1035.

[38] X. Yu, Z. Zhao, X. Zhang, X. Chen, J. Cai, Statistical identification guided open-set domain adaptation in fault diagnosis, Reliability Engineering & System Safety 232 (2023). doi:10.1016/j.ress.2022. 109047.

[39] X. Yu, Z. Zhao, X. Zhang, Q. Zhang, Y. Liu, C. Sun, X. Chen, Deeplearning-based open set fault diagnosis by extreme value theory, IEEE Transactions on Industrial Informatics 18 (1) (2022) 185–196. doi: 10.1109/tii.2021.3070324.

[40] F. Gao, X. Peng, D. Yang, C. Su, L. Li, W. Zhong, A novel distributed fault diagnosis scheme toward open-set scenarios based on extreme value theory, IEEE Transactions on Industrial Informatics 19 (10) (2023) 10454–10466. doi:10.1109/tii.2023.3240919.

[41] G. Li, M. Atoui, X. Li, Fault diagnosis across heterogeneous domains via self-adaptive temporal-spatial attention and sample generation, arXiv preprint arXiv:2505.11083 (2025).

[42] J. J. Downs, E. F. Vogel, A plant-wide industrial-process control problem, Computers & Chemical Engineering 17 (3) (1993) 245–255. doi: 10.1016/0098-1354(93)80018-i.

[43] A. Bathelt, N. L. Ricker, M. Jelali, Revision of the tennessee eastman process model, IFAC-PapersOnLine 48 (8) (2015) 309–314. doi:https: //doi.org/10.1016/j.ifacol.2015.08.199.

[44] K. E. S. Pilario, Y. Cao, Canonical variate dissimilarity analysis for process incipient fault detection, IEEE Transactions on Industrial Informatics 14 (12) (2018) 5308–5315. doi:10.1109/tii.2018.2810822.