# Generalized Out-of-distribution Fault Diagnosis (GOOFD) via Internal Contrastive Learning

Xingyue Wang<sup>†</sup>, Hanrong Zhang<sup>†</sup>, Xinlong Qiao, Ke Ma, Shuting Tao, Peng Peng<sup>∗</sup>, Member, IEEE, Hongwei Wang<sup>∗</sup>, Member, IEEE

Abstract—Fault diagnosis is crucial in monitoring machines within industrial processes. With the increasing complexity of working conditions and demand for safety during production, diverse diagnosis methods are required, and an integrated fault diagnosis system capable of handling multiple tasks is highly desired. However, the diagnosis subtasks are often studied separately, and the current methods still need improvement for such a generalized system. To address this issue, we propose the Generalized Out-of-distribution Fault Diagnosis (GOOFD) framework to integrate diagnosis subtasks. Additionally, a unified fault diagnosis method based on internal contrastive learning and Mahalanobis distance is put forward to underpin the proposed generalized framework. The method involves feature extraction through internal contrastive learning and outlier recognition based on the Mahalanobis distance. Our proposed method can be applied to multiple faults diagnosis tasks and achieve better performance than the existing single-task methods. Experiments are conducted on benchmark and practical process datasets, indicating the effectiveness of the proposed framework.

Index Terms—Fault diagnosis, internal contrastive learning, open-set classification, process monitoring.

## I. INTRODUCTION

D <sup>URING</sup> <sup>the</sup> <sup>industrial</sup> <sup>process</sup> <sup>and</sup> <sup>mechanical</sup> <sup>operation,</sup>faults inevitably occur, resulting in declining industrial faults inevitably occur, resulting in declining industrial efficiency and severe losses [1]–[3]. Accurate fault diagnosis has become increasingly crucial in system design and maintenance to ensure safety in machinery production [4]–[6] using the burgeoning deep learning technology [7]–[11].

Currently, the research in fault diagnosis mainly contains two tasks: 1) determining whether the system is normal. The technique for this task is called process monitoring, which aims to detect the abnormal during production. 2) distinguishing bluefaults that occurred while finding their reasons. The methods for this task are fault detection and fault classification. In process monitoring, the traditional approaches applied the multivariate statistical method, such as Principal Components

Analysis (PCA) [12], Partial Least Squares (PLS) [13], and Independent Component Analysis (ICA) [14], which extract features by transforming the raw data into a lower dimension and detecting occurred faults simply by statistical variance. In recent years, some new tasks have emerged to deal with more complex production situations, such as a novel task called open set fault diagnosis (OSFD), which detects unseen faults that occur during industrial processes. It is a relatively new task in fault diagnosis, aiming to detect unknown fault samples while correctly classifying known classes. Transfer Learning (TL) and domain adaptation (DA) combined with other deep learning methods have been widely utilized in the OSFD task [15], [16]. Chen et al. [17] proposed a multi-source openset DA diagnosis approach to tackle the OSFD task. Mao et al. [18] introduced an interactive dual adversarial neural network (IDANN), which uses weighted unknown classification items to distinguish the outliers. Li et al. [19] established a deep adversarial transfer learning network (DATLN), where a classifier is trained to learn the decision boundary and detect new faults. The tasks and techniques mentioned above are often discussed and tested separately. Nevertheless, since practical production is complicated, it leads to two problems: 1) A generalized fault diagnostic framework for multiple tasks is still lacking 2) The training on different techniques is timeconsuming since they have different feature extraction methods and network structures.

To address the problems, it is crucial to develop a generalized fault diagnosis system that considers several situations with a unified method to deal with multiple tasks. It enables comprehensive monitoring of a machine’s status, which facilitates the timely detection and identification of any faults that may arise, allowing for prompt action to be taken. Moreover, the process monitoring and OSFD methods mentioned above still need to be improved for the generalized framework. First, these methods can only solve a single task. They cannot simultaneously deal with all subtasks in a generalized framework, which does not match the comprehensive approach we need. Secondly, the working environment changes during the practical production process, and the above methods do not perform well in detecting faults under variable condition scenarios due to the lack of a powerful feature extractor. As a result, internal contrastive learning (ICL), as an effective representation learning technique, can be consequently used in our method.

In recent years, contrastive learning has been used to extract features from data [20], [21]. It is a stochastic data augmentation method in self-supervised learning, aiming to pull similar samples closer while pushing different samples apart in the embedding space. This method has proven effective in feature representation learning [22] in numerous research studies. Currently, some studies have applied contrastive learning to fault diagnosis methods to extract features and improve performance under changing working conditions. The contrastivelearning-based diagnosis methods are able to learn invariant features and improve performance under changing operating conditions. Zhang [23] et al. introduced a semi-supervised contrastive learning method to enhance the feature mapping of limited data. Li [24] et al. used contrastive learning for the fault diagnosis of rolling element bearings to extract interclass features. Zhang et al. [25] proposed a contrastive learningbased diagnosis model to learn domain-invariant fault features and improve performance under variable working conditions. In recent research, Tom and Wolf [26] have utilized ICL for anomaly detection. This method uses subvectors from given samples as positive/negative pairs to generate the contrastive loss, which depends little on the data structure and can better detect outliers.

Inspired by the above, this paper proposes a novel framework, and the main contributions are summarized as follows:

• We are the first to propose an integrated diagnostic system called the Generalized Out-of-distribution Fault Diagnosis (GOOFD) Framework, including process monitoring, fault classification, and OSFD tasks, which offers significant room for expansion and exploration.

• The paper introduces a novel integrated diagnosis method ICL-OD to 1) address the multi-tasks issue in the GOOFD framework and 2) learn more distinctive features for unknown classes based on internal contrastive learning and the Mahalanobis-distance approach.

• Extensive experiments on benchmark datasets and actual mechanical datasets are conducted to indicate that our proposed method can be applied to different fault diagnosis tasks and has a better performance compared to the existing single-task methods.

The rest of the paper is organized as follows: Section II presents the motivation and background of our proposed framework and methods. Section III introduces the integrated fault diagnosis framework and the proposed methods. Section IV presents the experimental results and the validation of the proposed model. Finally, the conclusion of the study is presented in Section V.

## II. BACKGROUND AND MOTIVATION

## A. Motivation

There are increasing numbers of new tasks and demands for fault diagnosis in industrial production, and many of these tasks have numerous subtasks that require various techniques. In the production process, there are often situations where many different types of faults need to be diagnosed, such as fault detection, fault classification, and unseen fault detection, which cannot be achieved through a single task alone. A fault diagnosis system that can comprehensively solve multiple tasks is required to diagnose faults more efficiently.

Currently, subtasks are discussed separately and have yet to be integrated to implement a comprehensive fault diagnosis system, and specific steps in the tasks, such as feature extraction and fault recognition, are also carried out separately. Thus, it is essential to build a comprehensive diagnostic system. We have found that the diagnosis subtasks have some commonalities, such as all involving the detection of outliers, indicating that it is feasible to combine these subtasks to solve them, and integrating them can also improve the efficiency of model training by using a unified feature map. Therefore, we propose this integrated framework to consolidate these subtasks.

Our method treats the unified problem in the proposed framework as the outlier detection task. In process monitoring, fault samples are defined as the outliners. In OSFD, the outliners are the unknown fault samples. This way, the generalized fault diagnosis issue can be treated as the out-of-distribution detection problem, in which the outliers vary from specific subtasks. Therefore, we used a contrastive learning method that is more suitable for outlier detection: ICL. The ICL technique generates positive and negative samples from the internal sub-vectors of the given sample, which focuses more on learning its characteristics and is more suitable for outlier detection problems in binary classification. By integrating ICL into our method, the detection of outliers is improved, achieving better fault detection performance.

## B. Contrastive Learning

Contrastive learning is a stochastic data augmentation method in self-supervised learning, which is proven effective in feature representation learning [22]. For a batch of N samples, the model generates 2N augmented data, where each source data x has two augmentations ${ \tilde { x } } _ { i }$ and $\tilde { x } _ { j }$ . For each augmented data $\tilde { x } _ { i }$ , the other augmented data $\tilde { x } _ { j }$ generated from the same source data x is regarded as the positive pair. The remaining 2(N − 1) augmentations $\tilde { x } _ { j } ( m \neq j )$ generated from other $( N - 1 )$ source data is deemed as the negative pair of $\tilde { x } _ { i }$ . The model is trained to minimize the distance between every positive pair and maximize the distance between every negative pair. The contrastive loss is defined as:

$$
\mathcal {L} = - \sum_ {i \in A} l o g \frac {e x p (f (\tilde {x} _ {i}) \cdot f (\tilde {x} _ {j}) / \tau)}{\sum_ {m \neq j} e x p (f (\tilde {x} _ {i}) \cdot f (\tilde {x} _ {m}) / \tau)},\tag{1}
$$

where $f ( \tilde { x } _ { k } )$ represents the embedding of augmented data $\tilde { x } _ { k }$ by encoder $f ( \cdot )$ . The index $i \in A$ ranges from (0, 2N ).

## C. Internal Contrastive Learning

In [26], Tom and Wolf proposed an anomaly detection method based on ICL. The method sliced the input vector into different dimensions. The network is trained to produce similar embeddings for complementary pairs and push away every sub-vector pair in the input vector. Based on the internal contrastive loss, the method generates positive and negative pairs from the subvectors of a sample $x _ { i }$ with dimension D to learn the representations of each sample. To split the vectors, here, we set a hyperparameter d to represent the starting index of a sub-vector, also called the internal dimension of the subset pairs. First, given the parameter $d ,$ extract consecutive l-length sub-vector $\mathbf { \bar { \rho } } _ { p _ { i } ^ { d } } ^ { d } = \{ x _ { i } ^ { d } , . . . , x _ { i } ^ { d + l - 1 } \}$ } from $x _ { i }$ . Then, the complementary part $\check { q _ { i } ^ { d } } = \{ \bar { x _ { i } ^ { 1 } } , . . . , \bar { x _ { i } ^ { d - 1 } } , \bar { x _ { i } ^ { d + l } } , . . . , x _ { i } ^ { D } \}$ with the length of $D - l ,$ is defined as the positive pair of $p _ { i } ^ { d }$ . For index d ${ \bf \omega } ^ { \prime } \neq d , p _ { i } ^ { d ^ { \prime } }$ is regarded as the negative pair of $q _ { i } ^ { d }$ . The method has two encoders $F$ and $G$ to learn the feature map of $p _ { i } ^ { d }$ and $q _ { i } ^ { d }$ , respectively. Here, we normalized the networks and obtained the normalized networks $F ^ { N }$ and $G ^ { N }$ . Let $\mathcal { D } ( q _ { i } ^ { d } , p _ { i } ^ { d } )$ represent the dot product between $p _ { i } ^ { d }$ and $q _ { i } ^ { d }$ , which is shown in (2) :

$$
\mathcal {D} (q _ {i} ^ {d}, p _ {i} ^ {d}) = F ^ {N} (q _ {i} ^ {d}) \cdot G ^ {N} (p _ {i} ^ {d}).\tag{2}
$$

The two networks are trained to learn similar embeddings between positive pairs $F ^ { N } ( p _ { i } ^ { d } )$ and $G ^ { N } ( q _ { i } ^ { d } )$ while minimizing the mutual information between negative pairs $p _ { i } ^ { d ^ { \prime } }$ and $q _ { i } ^ { d }$ . The internal contrastive loss for sample $x _ { i }$ in internal dimension d is defined in (3), where $k = D - l + 1$ :

$$
\ell (x _ {i}, j) = - l n \frac {e ^ {\mathcal {D} (q _ {i} ^ {d} , p _ {i} ^ {d}) / \tau}}{\sum_ {d ^ {\prime} \neq d} ^ {k} e ^ {\mathcal {D} (q _ {i} ^ {d} , p _ {i} ^ {d ^ {\prime}}) / \tau}}.\tag{3}
$$

In the internal contrastive loss, for a given sample $x _ { i }$ and internal dimension $d ,$ the method learns to produce similar embeddings $q _ { i } ^ { d }$ and $p _ { i } ^ { d }$ between the matched sub-vectors from the same dimension $j$ and separate the l-length sub-vectors $q _ { i } ^ { d }$ and $p _ { i } ^ { d ^ { \prime } }$ from different dimensions. Then the model can generate scores $S ( x _ { i } )$ for $x _ { i }$ considering all the internal dimensions $d = \{ 0 , . . . , k \}$ , as shown in (4):

$$
S (x _ {i}) = \sum_ {d} \ell (x _ {i}, d),\tag{4}
$$

where the final output score $S ( x _ { i } )$ denotes the representation of the sample $x _ { i }$

## III. GENERALIZED OUT-OF-DISTRIBUTION FAULT DIAGNOSIS

The framework of the proposed GOOFD system is shown in Fig.1. The framework uses the ICL-OD method to learn the boundary for outliers and then detect outliers to complete both process monitoring and OSFD tasks. The following subsections first introduce the definition of the unified tasks in the GOOFD system and then describe the methodology of the ICL-OD approach and how it is used to tackle multiple tasks in the GOOFD system.

## A. Task Definition

Given a set of n normal data $\mathcal { D } _ { 0 } = \{ x _ { 1 } , . . . , x _ { n } \}$ , categoried in the normal class $X _ { 0 }$ , and a set of known fault data $\mathcal { D } _ { f } = \{ f _ { 1 } , . . . , f _ { m } \}$ , which belong to $( N - 1 )$ fault classes $X _ { 1 } , . . . , X _ { N - 1 }$ . The embedding space for normal data $\mathcal { D } _ { 0 }$ is called $ { \boldsymbol { S } } _ { 0 }$ and the embedding space for the known fault data $\mathcal { D } _ { f }$ is called $ { \boldsymbol { S } } _ { f }$ . For the unknown fault ${ \mathcal { D } } _ { u } = \{ u _ { 1 } , . . . , u _ { k } \}$ with embedding space $\boldsymbol { S } _ { u } .$ , the label of $u _ { i }$ is class $X _ { N } .$ The embedding space of known data is defined as $\scriptstyle { S _ { k } }$ . Its corresponding open space is defined as $\mathcal { O } _ { k }$

In the proposed GOOFD, The embedding space $\quad S _ { 0 } , S _ { f }$ , and $ { \boldsymbol { S } } _ { u }$ collectively form the entire embedding space. The process monitor and OSFD tasks are fused by specifying different boundaries for the known embedding space $\scriptstyle { S _ { k } }$ . In process monitoring, the known embedding space contains only normal data $\mathcal { D } _ { 0 }$ . Thus it is defined as $S _ { k } = S _ { 0 }$ . Its corresponding open space $\mathcal { O } _ { 0 }$ is $\cal { S } _ { f } \cup \cal { S } _ { u }$ , which contains all the faults. While in OSFD, the known embedding space $\boldsymbol { S _ { k } }$ consists of normal data $\mathcal { D } _ { 0 }$ and known faults $\mathcal { D } _ { f }$ , and its corresponding $\mathcal { O } _ { k } =  { S _ { u } }$

## B. ICL for Outlier Detection

The proposed ICL for Outlier Detection (ICL-OD) method can be divided into two steps: generate the embedding map based on ICL and detect the outliers. The ICL-OD applied the ICL-based method for fault diagnosis in [26] to extract features from the input samples. Given a set of input samples $X = \{ x _ { 1 } , . . . , x _ { n } \}$ , obtain score $S ( x _ { i } )$ as the feature space for each sample $x _ { i }$ based on (4).

After obtaining the score $S ( x _ { i } )$ the method uses a Mahalanobis-distance-based classifier to detect the outlier. The Mahalanobis distance measures the distance between a sample and a distribution. Given an input sample x and a distribution $Q ,$ , the Mahalanobis distance $d _ { M } ( x )$ between them is defined in (5):

$$
d _ {M} (x) = \sqrt {(x - \mu) \sum^ {- 1} (x - \mu)},\tag{5}
$$

where $\mu$ represents the means of $Q$ and $\displaystyle \sum$ represents the covariance matrix of $Q .$ Unlike Euclidean distance, the Mahalanobis distance considers associations between correlated variables. It has been proven to be an effective graphic tool for identifying outliers. In our method, the Mahalanobis distance can be applied to measure the distance $d _ { S } ( x )$ between sample x and the distribution of the training set (normal data) in the score embedding space, as shown in (6):

$$
d _ {S} (x) = \sqrt {(S (x) - \mu_ {S}) \sum_ {S} ^ {- 1} (S (x) - \mu_ {S})},\tag{6}
$$

where $S ( x )$ denotes the output score of sample x, $\mu _ { S }$ and $\textstyle \sum _ { S }$ represent the means and covariance matrix of score $S$ for training sets, respectively. In the one-classification task, the training set contains only normal data, so the output score S can be deemed as the distribution of normal data.

Given training samples, the method first implements the ICL method to obtain the output score S for training input samples and calculate the means $\mu _ { t r }$ and covariance matrix $\sum _ { t r }$ of $S .$ Then, for every sample $\{ x _ { i } ^ { t r } | i \in ( 1 , n ) \}$ in the training set, calculate the Mahalanobis distance $d _ { S } ^ { t r a i { \bar { n } } } = \{ d _ { S } ( x _ { i } ^ { t r } ) | i \in$ $( 1 , n ) \}$ of each training sample to the normal distribution, as shown in (7):

$$
d _ {S} (x _ {i} ^ {t r}) = \sqrt {(S (x _ {i} ^ {t r}) - \mu_ {t r}) \sum_ {t r} ^ {- 1} (S (x _ {i} ^ {t r}) - \mu_ {t r})}.\tag{7}
$$

Similarly, for every sample $\{ x _ { i } ^ { t e } | i \in ( 1 , m ) \}$ in the testing set, calculate the Mahalanobis distance $d _ { S } ^ { t e s t } = \{ d _ { S } ( x _ { i } ^ { t e } ) | i \in$ $( 1 , m ) \}$ of each test sample to the normal distribution. Based on the distance, the outliers can be identified, as shown in (8):

![](images/e9f89aba082d1f110085f90e67446f86e3abc6ff6b841ebfabcd39f04d75707c.jpg)  
Fig. 1. (a) The difference between the GOOFD framework and existing single-task methods. A single-task approach can only handle a specific task. However the GOOFD framework is dedicated to integrating and solving various fault diagnosis tasks through a unified method. (b) The basic process of our approach <sup>. .. .</sup>. <sup>.</sup>.<sub>.</sub>addresses multi-task issues.

$$
y = \left\{ \begin{array}{l l} i n l i n e r, & d _ {S} (x ^ {t e}) \leq \theta , \\ o u t l i e r, & d _ {S} (x ^ {t e}) > \theta , \end{array} \right.\tag{8}
$$

where threshold θ is used to reject outliers, and its value is equal to the k% quantile of Mahalanobis distance in $d _ { S } ^ { t r a i n }$ . If $d _ { S } ( x _ { i } ^ { t e } )$ exceeds threshold θ, the sample $x _ { i } ^ { t e }$ will be determined as the outlier, and vice versa, is determined as the normal.

## C. Generalized Out-of-distribution Detection

As previously described, the GOOFD fault diagnosis task contains both process monitoring and OSFD tasks for fault diagnosis. The ICL Outlier Detection (ICL-OD) method proposed in this paper can be used in GOOFD fault diagnosis for process monitoring and OSFD, as shown in Figure 1. Since the process monitoring task aims at detecting the fault from normal, the input data x only consists of normal data of class $X _ { 0 }$ . Then, the ICL-OD method helps to generate the score S for normal data and reject the anomaly based on $d _ { S } ( x )$ , as shown in (9):

$$
d _ {S} (x) = \sqrt {(S (x) - \mu_ {0}) \sum_ {0} ^ {- 1} (S (x) - \mu_ {0})},\tag{9}
$$

where $\mu _ { 0 }$ and $\scriptstyle \sum _ { 0 }$ represent the means and covariance matrix of score S for normal samples. Then, use the k% quantile of Mahalanobis distance of normal samples as the threshold $\theta _ { 0 }$ to find the outliers, as shown in (10):

$$
y = \left\{ \begin{array}{l l} 0, & d _ {S} (x) \leq \theta_ {0}, \\ 1, & d _ {S} (x) > \theta_ {0}. \end{array} \right.\tag{10}
$$

For a sample x in the test set, if the Mahalanobis distance $d _ { S } ( x )$ of x exceeds the threshold $\theta _ { 0 } ,$ , the sample x is labeled as 1 and rejected as the outliers. Otherwise, it will be labeled as 0 and classified as normal.

In OSFD, which is a multi-class classification task, the training data contains normal data and known faults, regarded as known classes. First, the method trains a classifier C to classify known classes, which include the normal data class $X _ { 0 }$ and N −1 known fault class $X _ { 1 } , . . . X _ { N }$ . The input samples are regarded as the known classes in OSFD. Then, the ICL-OD method is used to learn the boundary of the empirical data in Mahalanobis distance. Similar to Anomaly Detection, for a given sample x from the testing set, the method uses the Mahalanobis-distance-based threshold to reject the unknown, shown in (11):

$$
d _ {S} (x) = \sqrt {(S (x) - \mu_ {k}) \sum_ {k} ^ {- 1} (S (x) - \mu_ {k})},\tag{11}
$$

where $\mu _ { k }$ and $\textstyle \sum _ { k }$ denote the means and covariance matrix of the known training samples, respectively. The method uses the k% quantile of Mahalanobis distance of training samples as the threshold $\theta _ { u }$ to reject the unknown. Finally, the method can distinguish between known and unknown, shown in (12):

$$
y = \left\{ \begin{array}{c c} a r g m a x (\mathcal {C} (x)) \in (1, N), & d _ {S} (x) \leq \theta_ {u}, \\ N + 1, & d _ {S} (x) > \theta_ {u}. \end{array} \right.\tag{12}
$$

The trained classifier C firstly outputs the softmax score of x and classifies them into known classes $1 \sim N$ . Then, for all the samples whose Mahalanobis distance is greater than the threshold $\theta _ { u } .$ , the method will treat these points as outliers and classify them into the unknown class (class N + 1).

## IV. EXPERIMENT STUDY

In this section, we evaluate and compare our proposed method with baseline methods and set up two experiments to prove the effectiveness of the proposed method. In Experiment I, we first use a simulated benchmark dataset to perform a preliminary validation of the proposed method. In Experiment II, the evaluation of our approach is further verified using two datasets generated from the real-world process. The details of the two experiments are listed below.

\- Experiment I: The proposed method will be evaluated in the Tennessee Eastman process [27] to measure the performance in process monitoring and OSFD for fault diagnosis. In process monitoring, the dataset uses normal data as the training set and fault data as the test set. The OSFD task divides the dataset into open and closed sets to test its ability to detect unseen faults.

\- Experiment II: To further evaluate the effectiveness of our method, Experiment II implements the proposed technique with a real-world dataset: Multiphase Flow Facility (MFF) Dataset [28]. The MFF dataset is used in process monitoring and OSFD tasks, respectively.

Baseline. In Experiment I, for the process monitoring task, we compare the proposed method with the following seven baseline methods: PCA [12], Kernel PCA (KPCA) [12], ICA [14], Kernel ICA (KICA) [29], Dynamic PCA (DPCA) [30], Structured Joint Sparse Canonical Correlation Analysis (SJCCA) [31], Orthonormal Subspace Analysis (OSA) [32], and Decentralized Support Vector Data Descriptions (DSVDD) [33]. In the OSFD task, we compare the proposed method with the following six baselines: SoftMax, OpenMax [34], CenterLoss [35], EOW-SoftMax [36], Generalized ENtropy score (GEN) [37], and Positive and Unlabeled learning and Label Shift Estimation (PULSE) [38]. In addition, we compare our proposed method with the ICL anomaly detection approach [26] to validate the superiority of our method. Since the ICL anomaly detection approach cannot be directly applied to the OSFD task, we add a classifier to the method to complete the experiment, which we call ICL+ in the later section.

![](images/4b8b674c99b244311ca9e3269a3be7749ffc1409fa87722fc42bc9460ab06f34.jpg)  
Fig. 2. Flow chart of the TE process [27]

Experiment II implements PCA, DPCA, PLS, Enhanced Dynamic Latent Variable (EDLV) analysis [39], and Global–local Preserving Projection based on Optimal Active Relative Entropy (OARE-GLPP) [40] as the baseline methods in the process monitoring task. The baseline used in the OSFD task remains SoftMax, OpenMax, CenterLoss, EOW-Softmax, GEN, PULSE, and ICL+.

Metrics. In process monitoring, our experiments use the Fault Detection Rate (FDR) to evaluate a model’s ability to detect faults. In OSFD, we use the F1 score and AUROC to assess performance.

## A. Experiment I: evaluation with TE Process

Dataset. Based on actual chemical reaction processes, Eastman Chemical Company has developed an open and challenging chemical model simulation platform, the Tennessee Eastman (TE) [27] simulation platform 2, which generates time-varying data for testing control and diagnostic models of complex industrial processes.

The TE dataset contains 21 faults. In process monitoring, the experiment uses normal data as the training set and 21 fault data as test data to evaluate the performance. In the OSFD task, the training set consists of normal data and known faults (fault 1, fault 2, fault 4, and fault 5). Then, the unseen faults (faults 6, 7, 13, 14) are randomly picked and used as test sets to assess the model’s performance.

Training Scheme. In the ICL-OD method, the model employs Adma Optimizer with a learning rate of 0.001. We set the subvector’s length l = 2 and temperature $\tau = 0 . 0 1$ . The F and G are fully-connected networks. The hidden layers in the F network have u and 2u units. A LeakyRelu activation and batch normalization layer is applied for each hidden layer. In the G network, hidden units are $u / 4$ and $u / 2 ,$ , with the LeakyRelu activation for each layer. Unlike network F , G applied batch normalization only in the first hidden layer. We set hidden unit $u = 2 0 0$ for both $F$ and G. The classifier C for the TE dataset is a fully connected network with one hidden layer of 20 units. The classifier applies Adam Optimizer and the CrossEntrophy loss in the training phase. The threshold $\theta _ { u }$ is set at values when $k \% = 9 8 \%$ to reject the unknown samples.

Results. In process monitoring, our proposed method outperforms the baseline methods in most cases, highlighted in green in Table I. $\mathbf { \mu ^ { 6 6 } O T ^ { 5 } } )$ in Table I refers to the cases where the event of fault detection was impossible to determine because the indicator value was already over the threshold before the fault introduction. In most cases, our model achieves the highest FDR compared to other baselines. Especially in cases of fault 3, fault 9, and fault 15, our model dramatically improves the results compared to the poor performance of baseline methods. Even in the case where the FDR of our model is not the highest, our proposed method can still achieve relatively good results.

In OSFD, we repeat experiments for each method five times to report the average F1-score and AUROC. The results in Table II show that our proposed model outperforms the baseline methods. Fig. 3 presents the detailed results where Fault 6 is seen as the unknown. From the confusion matrix shown in Fig. 3 (a), we can see that our proposed method has a higher recognition rate for the unknown, while the classification of the closed-set data is still guaranteed.

Fig. 3 (b) presents the distribution of the prediction scores to further analyze the superiority of our method. The prediction scores generated by the Softmax and CenterLoss methods exhibit significant overlap between known and unknown classes, leading to the misclassification of most unknown classes as known ones. The OpenMax method demonstrates a clearer distinction between scores of unknown and known classes, but there are still instances where the scores for unknown categories are too high, resulting in misclassification as known categories. The EOW-SoftMax method presents a similar issue, producing overly low scores for unknown data, resulting in misidentification as known classes. The GEN method has difficulty generating predictive scores with high discrimination, and most of its normal samples (class 0) and unknown samples have similar scores, resulting in poor classification. The PULSE method has better classification results, but some unknown samples have overly high confidence scores, leading to their misclassification into known categories.

However, our method can generate prediction scores with greater distinctiveness, enabling better identification of unknown classes. As shown in Fig. 3 (b), our method has minimal overlap between the distribution of prediction scores for unknown and known classes. In contrast, in the ICL+ method, there is still a significant overlap between the prediction scores of unknown and known classes, resulting in a higher mislabeling rate for unknown classes. Fig. 4 shows the histogram of the prediction scores for unknown classes in both the ICL+ method and our approach. Our method can generate more “extreme” prediction scores for unknown classes, thereby enhancing differentiation from known categories. These results indicate that, in contrast to other baseline methods, our approach with an effective feature extractor can generate prediction scores with greater distribution disparity between unknown and known categories, thereby achieving superior performance.

TABLE I  
RESULTS OF PROCESS MONITORING IN TE PROCESS

<table><tr><td rowspan="2">fault</td><td>Ours</td><td colspan="2">PCA [12]</td><td colspan="2">KPCA [12]</td><td colspan="2">ICA [14]</td><td colspan="2">KICA [29]</td><td colspan="2">DPCA [30]</td><td>SJCCA [31]</td><td>OSA [32]</td><td>DSVDD [33]</td></tr><tr><td>-</td><td> $T^2$ </td><td>Q</td><td> $T^2$ </td><td>Q</td><td> $I^2$ </td><td>SPE</td><td> $T^2$ </td><td>SPE</td><td> $T^2$ </td><td>Q</td><td>-</td><td>SPE</td><td>-</td></tr><tr><td>1</td><td>100</td><td>99.50</td><td>99.80</td><td>99.80</td><td>99.80</td><td>99.77</td><td>99.78</td><td>100</td><td>100</td><td>99.00</td><td>99.40</td><td>99.75</td><td>99.90</td><td>99.75</td></tr><tr><td>2</td><td>99.12</td><td>98.30</td><td>98.80</td><td>98.80</td><td>98.50</td><td>98.22</td><td>98.27</td><td>98.00</td><td>98.00</td><td>98.40</td><td>98.10</td><td>99.47</td><td>95.60</td><td>98.63</td></tr><tr><td>3</td><td>61.88</td><td>8.10</td><td>7.80</td><td>8.00</td><td>7.50</td><td>3.46</td><td>9.34</td><td>6.00</td><td>3.00</td><td>3.50</td><td>1.00</td><td>41.38</td><td>3.3</td><td>13.25</td></tr><tr><td>4</td><td>76.75</td><td>28.90</td><td>100</td><td>100</td><td>37.30</td><td>1.96</td><td>4.82</td><td>82.00</td><td>100</td><td>16.50</td><td>99.90</td><td>100</td><td>100</td><td>100</td></tr><tr><td>5</td><td>73.75</td><td>30.60</td><td>31.30</td><td>28.60</td><td>99.50</td><td>22.63</td><td>28.51</td><td>29.00</td><td>27.00</td><td>29.30</td><td>22.80</td><td>36.62</td><td>23.00</td><td>33.00</td></tr><tr><td>6</td><td>100</td><td>99.30</td><td>100</td><td>99.50</td><td>100</td><td>99.81</td><td>99.96</td><td>100</td><td>100</td><td>98.90</td><td>99.90</td><td>100</td><td>100</td><td>100</td></tr><tr><td>7</td><td>100</td><td>100</td><td>100</td><td>100</td><td>99.90</td><td>36.90</td><td>42.52</td><td>100</td><td>100</td><td>98.60</td><td>99.90</td><td>100</td><td>100</td><td>100</td></tr><tr><td>8</td><td>99.75</td><td>97.50</td><td>97.90</td><td>98.30</td><td>98.10</td><td>95.85</td><td>98.16</td><td>97.00</td><td>98.00</td><td>97.30</td><td>97.40</td><td>97.85</td><td>88.9</td><td>98</td></tr><tr><td>9</td><td>56.12</td><td>7.40</td><td>6.00</td><td>6.50</td><td>4.50</td><td>3.18</td><td>8.59</td><td>5.00</td><td>3.00</td><td>3.00</td><td>0.20</td><td>40.87</td><td>1.90</td><td>10.63</td></tr><tr><td>10</td><td>87.00</td><td>49.80</td><td>53.90</td><td>54.90</td><td>86.90</td><td>57.70</td><td>68.15</td><td>81.00</td><td>80.00</td><td>43.90</td><td>17.20</td><td>39.58</td><td>35.1</td><td>59.75</td></tr><tr><td>11</td><td>79.50</td><td>47.40</td><td>73.30</td><td>79.30</td><td>51.80</td><td>32.16</td><td>39.24</td><td>81.00</td><td>77.00</td><td>34.00</td><td>82.90</td><td>78.50</td><td>76.5</td><td>73.88</td></tr><tr><td>12</td><td>99.50</td><td>99.00</td><td>97.80</td><td>99.10</td><td>99.50</td><td>95.05</td><td>99.00</td><td>97.00</td><td>98.00</td><td>99.00</td><td>96.40</td><td>96.37</td><td>89.6</td><td>99.13</td></tr><tr><td>13</td><td>98.38</td><td>95.00</td><td>95.40</td><td>95.50</td><td>95.90</td><td>94.16</td><td>94.82</td><td>95.00</td><td>95.00</td><td>94.30</td><td>95.00</td><td>95.29</td><td>95.3</td><td>95.00</td></tr><tr><td>14</td><td>95.75</td><td>99.00</td><td>100</td><td>100</td><td>99.90</td><td>99.76</td><td>99.92</td><td>100</td><td>100</td><td>99.00</td><td>99.90</td><td>89.82</td><td>100</td><td>100</td></tr><tr><td>15</td><td>52.50</td><td>12.40</td><td>8.80</td><td>13.30</td><td>14.00</td><td>6.49</td><td>12.97</td><td>5.00</td><td>7.00</td><td>5.90</td><td>0.90</td><td>42.37</td><td>2.90</td><td>18.50</td></tr><tr><td>16</td><td>93.50</td><td>32.50</td><td>48.30</td><td>37.00</td><td>90.00</td><td>24.20</td><td>39.89</td><td>80.00</td><td>52.00</td><td>21.70</td><td>14.50</td><td>26.37</td><td>35.10</td><td>52.63</td></tr><tr><td>17</td><td>85.75</td><td>81.60</td><td>93.90</td><td>96.10</td><td>90.50</td><td>88.51</td><td>95.38</td><td>95.00</td><td>95.00</td><td>79.00</td><td>95.30</td><td>41.75</td><td>95.9</td><td>91.38</td></tr><tr><td>18</td><td>95.88</td><td>89.50</td><td>91.40</td><td>91.30</td><td>89.40</td><td>90.06</td><td>90.08</td><td>90.00</td><td>80.00</td><td>89.00</td><td>89.80</td><td>94.12</td><td>90.3</td><td>90.13</td></tr><tr><td>19</td><td>65.62</td><td>8.40</td><td>29.10</td><td>19.10</td><td>80.80</td><td>7.68</td><td>22.99</td><td>75.00</td><td>69.00</td><td>4.60</td><td>29.80</td><td>29.93</td><td>18.8</td><td>13.63</td></tr><tr><td>20</td><td>84.00</td><td>47.00</td><td>57.30</td><td>68.40</td><td>72.30</td><td>49.79</td><td>57.42</td><td>58.00</td><td>55.00</td><td>40.80</td><td>49.30</td><td>55.75</td><td>53.1</td><td>58.63</td></tr><tr><td>21</td><td>70.75</td><td>39.40</td><td>51.10</td><td>54.50</td><td>44.30</td><td>38.17</td><td>43.74</td><td>61.00</td><td>58.00</td><td>42.90</td><td>40.90</td><td>96.90</td><td>57.6</td><td>42.88</td></tr><tr><td>Average</td><td>84.55</td><td>60.50</td><td>68.66</td><td>68.95</td><td>74.30</td><td>54.55</td><td>59.69</td><td>73.10</td><td>71.19</td><td>57.08</td><td>63.36</td><td>71.54</td><td>64.90</td><td>68.99</td></tr></table>

TABLE II  
THE RESULTS OF OSFD IN TE PROCESS

<table><tr><td>Unknown fault</td><td>Metric</td><td>SoftMax</td><td>OpenMax</td><td>CenterLoss</td><td>EOW-Softmax</td><td>GEN</td><td>PULSE</td><td>ICL+</td><td>Ours</td></tr><tr><td rowspan="2">Fault 6</td><td>AUROC</td><td>.305 ± .090</td><td>.904 ± .057</td><td>.363 ± .102</td><td>.958 ± .027</td><td>.251 ± .112</td><td>.790 ± .276</td><td>.999 ± .001</td><td>1.0 ± 0.0</td></tr><tr><td>F1</td><td>.815 ± .006</td><td>.912 ± .058</td><td>.804 ± .014</td><td>.964 ± .041</td><td>.825 ± .002</td><td>.918 ± .067</td><td>.980 ± .002</td><td>.987 ± .003</td></tr><tr><td rowspan="2">Fault 7</td><td>AUROC</td><td>.858 ± .122</td><td>.897 ± .042</td><td>.865 ± .040</td><td>.744 ± .088</td><td>.788 ± .035</td><td>.925 ± .035</td><td>.995 ± .010</td><td>1.0 ± .001</td></tr><tr><td>F1</td><td>.856 ± .018</td><td>.885 ± .027</td><td>.854 ± .016</td><td>.832 ± .005</td><td>.825 ± .003</td><td>.858 ± .023</td><td>.975 ± .030</td><td>.985 ± .003</td></tr><tr><td rowspan="2">Fault 13</td><td>AUROC</td><td>.718 ± .072</td><td>.798 ± .046</td><td>.789 ± .097</td><td>.731 ± .057</td><td>.643 ± .080</td><td>.797 ± .073</td><td>.891 ± .009</td><td>.926 ± .005</td></tr><tr><td>F1</td><td>.835 ± .007</td><td>.865 ± .012</td><td>.842 ± .028</td><td>.856 ± .016</td><td>.825 ± .003</td><td>.855 ± .026</td><td>.902 ± .003</td><td>.921 ± .011</td></tr><tr><td rowspan="2">Fault 14</td><td>AUROC</td><td>.567 ± .114</td><td>.707 ± .044</td><td>.771 ± .074</td><td>.743 ± .103</td><td>.422± .041</td><td>.844 ± .056</td><td>.815 ± .036</td><td>.888 ± .023</td></tr><tr><td>F1</td><td>.828 ± .006</td><td>.868 ± .005</td><td>.836 ± .019</td><td>.862 ± .013</td><td>.828 ± .002</td><td>.870 ± .015</td><td>.914 ± .036</td><td>.930 ± .001</td></tr></table>

Experiment I trains and validates the model with the TE dataset and demonstrates that our proposed model can combine the process monitoring and OSFD tasks while achieving better performance than baseline methods in both tasks.

## B. Experiment II: evaluation with Multi-phase Flow Facility

Datasets. To further demonstrate the effectiveness of our method, Experiment II uses a real-world dataset TE and the Multiphase Flow Facility (MFF) dataset [28] to validate the model.

The MMF dataset is derived from the Three-phase Flow Facility system at Cranfield University. The data has 24 process variables and is acquired at the sampling rate of 1 Hz. The MFF dataset contains normal data and six fault cases. In normal data, three datasets (T1, T2, T3) are captured from the system to represent normal operating conditions adequately. For each fault case, the process data is generated under different operation conditions and used as different sets. In process monitoring, we use the T1 and T3 sets of normal data to train our method. The description of the test set is shown in Table III. All test datasets are from Set 1 with changing operating conditions. In the OSFD test of Experiment II, we use the T2 set of normal data and Set-2 of each fault case to train and evaluate our model. We randomly select normal data and three fault cases as the training set.

TABLE III  
THE DESCRIPTION OF TEST SETS OF MMF IN PROCESS MONITOR EXPERIMENT

<table><tr><td>Test Set</td><td>Fault Description</td><td>Dataset</td></tr><tr><td>1</td><td>Air line blockage</td><td>1.1</td></tr><tr><td>2</td><td>Water line blockage</td><td>3.1</td></tr><tr><td>3</td><td>Top separator input blockage</td><td>4.1</td></tr><tr><td>4</td><td>Open direct bypass</td><td>5.1</td></tr></table>

Training Scheme. The parameter settings of the ICL-OD

![](images/400019e09b9914ecb303ed35342084b976a7f62e383c0b35e192b83ae3eb430b.jpg)  
(b) The distribution of prediction scores  
Fig. 3. (a) The confusion matrix for the OSFD task in the TE process with Fault 6 as the unknown class. (b) we visualize the distribution of prediction scores generated by each method. The images in the second row are the enlarged display of the yellow region in the first-row images, providing a better view of details. The prediction scores of the Softmax, OpenMax, and CenterLoss methods are softmax probabilities. Scores below the threshold are classified as unknown. The prediction scores of the EOW-Softmax method constitute the (K+1)th-dimensional probability of its output, estimating open-world uncertainty. Scores surpassing the threshold are classified as unknown. The GEN method can be applied to any pre-trained softmax-based classifier to generate the entropy-based score, and scores below a threshold are determined to be unknown. The prediction score of the PULSE method is the output of the discriminator. Scores below the threshold are labeled as unknown classes. The prediction score of ICL+ and our method are the opposites of the output score. Data with prediction scores below the threshold are classified as the unknown class. The smaller the overlap between the prediction scores of known and unknown categories, the better the method’s performance.

![](images/a853c39dee18e26191425ae1717bff89574495d4110e0cfd7112243f1511615b.jpg)  
Fig. 4. The histogram of prediction scores for unknown classes in ICL+ and our methods. The prediction scores of the ICL+ and our method shown in Fig. 3 (b) are overly dense, making it difficult to discern specific distributional features. Therefore, we present a clearer display of the distribution of the prediction scores for unknown classes.

method in Experiment II are the same as those in Experiment I. The classifier C for the MFF dataset is a fully connected network with one hidden layer. The hidden layer has 12 units, followed by a ReLU activation layer. During the training phase, the classifier applies Adam Optimizer and CrossEntrophy loss.

Results. The FDR results of the process monitor experiment are shown in Table IV. It can be seen that our method outperforms all other baseline methods in the MFF dataset.

For the OSFD evaluation, we ran the experiment five times, and the average results of the MFF process are shown in Table V. The Known-Unknown in Table V shows the known classes for training and their corresponding unknown fault for evaluation in each case. Compared with the other six baseline methods, our approach shows a significant improvement in the F1-score and AUROC metrics for each case. Furthermore, our method outperforms ICL+. With unknown faults of 2 and $^ { 4 , }$ both the AUROC and F1-score of our method are significantly better than those of the ICL+ method. Especially when the unknown fault is 2, our method improves by 12% on the F1-score compared to ICL+, which is a great classification improvement. In the case of unknown faults of 3 and 5, even though AUROC is both 1.0 (achieving the best result), the F1-score of our method still exceeds the ICL+ method, which suggests that our method can better predict the boundaries of known classes and generate more accurate thresholds to reject unknown classes. The results of Experiment II indicate that even in real-world data, our method can achieve remarkable performance in the GOOFD task and can surpass baseline methods in both tasks.

## C. Ablation study

In the ablation study, the experiments are conducted on the TE dataset in the OSFD task. The known class consists of normal data, faults 1, 2, 4, and 5. The experiments are repeated five times to present the average F1-score and AUROC.

Threshold for outliers. The threshold $\theta _ { u }$ is used to determine and reject unknown classes. Therefore, it is crucial to select an appropriate threshold, which will determine the performance of unknown fault detection. We set different values of $\theta _ { u }$ (80%, 85%, 90%, 95%, 98% and 100%) on the TE dataset in OSFD task. The results in Fig. 5 show that it can reach the best performance when $\theta _ { u } = 9 8 \%$ . When the threshold is within the range of 80% to 98%, the F1-score shows an increasing trend, indicating that raising the threshold for rejecting unknown categories has a positive effect on the method’s performance. However, when the threshold exceeds 98%, the F1-score decreases, suggesting that setting an overly high threshold for unknown categories may mistakenly identify them as known, leading to a decline in model performance.

TABLE IV  
THE RESULTS OF PROCESS MONITORING IN MFF

<table><tr><td rowspan="2">Test Set</td><td colspan="2">PCA</td><td colspan="2">DPCA</td><td colspan="2">PLS</td><td>EDLV</td><td>OARE-GLPP</td><td>Ours</td></tr><tr><td> $T^2$ </td><td>Q</td><td> $T^2$ </td><td>Q</td><td> $T^2$ </td><td>Q</td><td> $T_S^2$ </td><td>-</td><td>-</td></tr><tr><td>1</td><td>22.65</td><td>52.72</td><td>23.1</td><td>60.17</td><td>36.49</td><td>30.23</td><td>79.56</td><td>76.15</td><td>100</td></tr><tr><td>2</td><td>98.37</td><td>99.72</td><td>98.44</td><td>100</td><td>99.36</td><td>98.67</td><td>99.14</td><td>98.16</td><td>100</td></tr><tr><td>3</td><td>34.5</td><td>92.64</td><td>35.61</td><td>94.14</td><td>40.34</td><td>43.02</td><td>91.61</td><td>66.01</td><td>99.42</td></tr><tr><td>4</td><td>70.63</td><td>OT</td><td>71.87</td><td>OT</td><td>94.72</td><td>58.63</td><td>89.64</td><td>42.36</td><td>98.97</td></tr></table>

TABLE V  
THE RESULTS OF OSFD IN MMF

<table><tr><td>Known-Unknown</td><td>Metric</td><td>SoftMax</td><td>OpenMax</td><td>CenterLoss</td><td>EOW-SoftMax</td><td>GEN</td><td>PULSE</td><td>ICL+</td><td>Ours</td></tr><tr><td rowspan="2">0, 1, 3, 4 - 2</td><td>AUROC</td><td>.925 ± .036</td><td>.928 ± .059</td><td>.865 ± .146</td><td>.824 ± .127</td><td>.922 ± .017</td><td>.840 ± .173</td><td>.938 ± .038</td><td>.960 ± .016</td></tr><tr><td>F1</td><td>.793 ± .000</td><td>.849 ± .045</td><td>.789 ± .075</td><td>.803 ± .129</td><td>.822 ± .103</td><td>.890 ± .105</td><td>.821 ± .084</td><td>.941 ± .038</td></tr><tr><td rowspan="2">0, 1, 2, 6 - 3</td><td>AUROC</td><td>.939 ± .070</td><td>.759 ± .093</td><td>.925 ± .046</td><td>.967 ± .027</td><td>.930 ± .056</td><td>.503 ± .252</td><td>1. ± .000</td><td>1. ± .000</td></tr><tr><td>F1</td><td>.839 ± .060</td><td>.726 ± .029</td><td>.869 ± .051</td><td>.958 ± .018</td><td>.884 ± .054</td><td>.678 ± .080</td><td>.957 ± .082</td><td>.998 ± .000</td></tr><tr><td rowspan="2">0, 1, 2, 6 - 4</td><td>AUROC</td><td>.884 ± .199</td><td>.881 ± .052</td><td>.738 ± .126</td><td>.891 ± .070</td><td>.936 ± .041</td><td>.496 ± .028</td><td>.960 ± .072</td><td>.998 ± .005</td></tr><tr><td>F1</td><td>.822 ± .040</td><td>.780 ± .063</td><td>.813 ± .031</td><td>.892 ± .046</td><td>.896 ± .041</td><td>.796 ± .033</td><td>.907 ± .072</td><td>.969 ± .072</td></tr><tr><td rowspan="2">0, 1, 2, 6 - 5</td><td>AUROC</td><td>.398 ± .094</td><td>.809 ± .041</td><td>.988 ± .012</td><td>.951 ± .030</td><td>.232 ± .052</td><td>.928 ± .063</td><td>1. ± .000</td><td>1. ± .000</td></tr><tr><td>F1</td><td>.800 ± .006</td><td>.715 ± .053</td><td>.936 ± .052</td><td>.897 ± .061</td><td>.785 ± .026</td><td>.887 ± .055</td><td>.983 ± .033</td><td>1. ± .000</td></tr></table>

outlier detection than other distances.

![](images/431af925606afea9419a6e94a05e2ce1db502bfba79c2bc6f95249f04367356d.jpg)  
Fig. 5. The average F1-score of OSFD tasks in TE process when applying various threshold $\theta _ { u }$

## V. CONCLUSION

In this paper, we are the first to introduce the novel idea of an integrated fault diagnosis framework and present the GOOFD framework to integrate multiple fault diagnosis tasks. Compared to existing single-task fault diagnosis approaches, our ICL-OD method can be applied to multiple tasks and can also better extract features and generate prediction scores with greater discriminability. Extensive experiments are conducted to demonstrate that our method leads to better performance for each sub-task. In future work, we intend to expand GOOFD to cover more diagnostic subtasks, enhancing fault detection capabilities and system robustness. This development aims to improve industrial safety, and machine reliability, and reduce maintenance costs. Furthermore, we will work on incorporating interpretability within the framework to improve transparency and user trust, which are expected to deliver significant societal advantages, such as safer industrial operations and economic efficiencies.

Distance for outlier detection. In order to find a more suitable measurement for unknown class detection, we compared the Mahalanobis distance with other measures of distance (cityblock distance, Canberra distance, and Euclidean distance). The results in Table VI show that the Mahalanobis distance leads to better classification performance. Both AUROC and F1-score are significantly better than the rest of the distances in the case of the Mahalanobis distance. Particularly, in the case of the unknown fault 13, the AUROC of the Mahalanobis distance is 12.6%-22% higher compared to the rest of the distances. This suggests that the Mahalanobis distance generates more discriminatory confidence scores compared to the other distances. Theoretically, the Mahalanobis distance can be regarded as a modification of the Euclidean distance, which corrects the problem of inconsistent and correlated dimensional scales, as shown in (5). Therefore, since the Mahalanobis distance can eliminate the problem of different scales between different dimensions, and make dimensional corrections to the sample distribution, it is more suitable for

## REFERENCES

[1] P. Peng, H. Zhang, X. Wang, W. Huang, and H. Wang, “Imbalanced chemical process fault diagnosis using balancing gan with active sample selection,” IEEE Sensors Journal, vol. 23, no. 13, pp. 14 826–14 833, 2023.

[2] W. Huang, H. Zhang, P. Peng, and H. Wang, “Multi-gate mixture-ofexpert combined with synthetic minority over-sampling technique for multimode imbalanced fault diagnosis,” in 2023 26th International Conference on Computer Supported Cooperative Work in Design (CSCWD), 2023, pp. 456–461.

[3] M. Li, P. Peng, J. Zhang, H. Wang, and W. Shen, “Sccam: Supervised contrastive convolutional attention mechanism for ante-hoc interpretable fault diagnosis with limited fault samples,” IEEE Transactions on Neural Networks and Learning Systems, 2023.

[4] H. Momeni, N. Sadoogi, M. Farrokhifar, and H. F. Gharibeh, “Fault diagnosis in photovoltaic arrays using gbssl method and proposing a fault correction system,” IEEE Transactions on Industrial Informatics, vol. 16, no. 8, pp. 5300–5308, 2020.

[5] L. Feng and C. Zhao, “Fault description based attribute transfer for zero-sample industrial fault diagnosis,” IEEE Transactions on Industrial Informatics, vol. 17, no. 3, pp. 1852–1862, 2021.

[6] A. Glowacz, “Thermographic fault diagnosis of shaft of bldc motor,” Sensors, vol. 22, no. 21, 2022.

TABLE VI  
ABLATION STUDY ON DIFFERENT MEASUREMENTS OF DISTANCE

<table><tr><td>Unknown</td><td colspan="2">Fault 6</td><td colspan="2">Fault 7</td><td colspan="2">Fault 13</td><td colspan="2">Fault 14</td></tr><tr><td>Metirc</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td></tr><tr><td>City-block</td><td>.977 ± .012</td><td>.919 ± .064</td><td>.994 ± .013</td><td>.961 ± .061</td><td>.785 ± .122</td><td>.883 ± .030</td><td>.782 ± .069</td><td>.814 ± .033</td></tr><tr><td>Euclidean</td><td>.999 ± .001</td><td>.976 ± .019</td><td>.992 ± .019</td><td>.960 ± .046</td><td>.756 ± .092</td><td>.880 ± .038</td><td>.775 ± .134</td><td>.895 ± .048</td></tr><tr><td>Canberra</td><td>.984 ± .010</td><td>.973 ± .009</td><td>.907 ± .163</td><td>.928 ± .075</td><td>.691 ± .120</td><td>.870 ± .024</td><td>.790 ± .057</td><td>.869 ± .038</td></tr><tr><td>Mahalanobis</td><td>1.0 ± 0</td><td>.987 ± .003</td><td>1.0 ± .001</td><td>.985 ± .003</td><td>.911 ± .017</td><td>.928 ± .007</td><td>.803 ± .081</td><td>.908 ± .026</td></tr></table>

[7] M. Li, P. Peng, H. Sun, M. Wang, and H. Wang, “An order-invariant and interpretable dilated convolution neural network for chemical process fault detection and diagnosis,” IEEE Transactions on Automation Science and Engineering, 2023.

[8] H. Zhang, X. Wang, J. Pan, and H. Wang, “Saka: An intelligent platform for semi-automated knowledge graph construction and application,” Service Oriented Computing and Applications, vol. 17, no. 3, pp. 201– 212, 2023.

[9] Z. Wang, B. Qin, M. Li, M. D. Butala, H. Wang, P. Peng, and H. Wang, “Hard sample mining enabled contrastive feature learning for wind turbine pitch system fault diagnosis,” arXiv preprint arXiv:2306.14701, 2023.

[10] Z. Wang, H. Tang, H. Wang, B. Qin, M. D. Butala, W. Shen, and H. Wang, “Weighted joint maximum mean discrepancy enabled multisource-multi-target unsupervised domain adaptation fault diagnosis,” arXiv preprint arXiv:2310.14790, 2023.

[11] Z. Wang, B. Qin, H. Sun, J. Zhang, M. D. Butala, C. Demartino, P. Peng, and H. Wang, “An imbalanced semi-supervised wind turbine blade icing detection method based on contrastive learning,” Renewable Energy, vol. 212, pp. 251–262, 2023.

[12] X. Deng, X. Tian, S. Chen, and C. J. Harris, “Nonlinear process fault diagnosis based on serial principal component analysis,” IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 3, pp. 560–572, 2018.

[13] H. Abdi and L. J. Williams, “Partial least squares methods: partial least squares correlation and partial least square regression,” in Computational Toxicology: Volume II, 2013, pp. 549–579.

[14] Z. Ge and Z. Song, “Performance-driven ensemble learning ica model for improved non-gaussian process monitoring,” Chemometrics and Intelligent Laboratory Systems, vol. 123, pp. 1–8, 2013.

[15] Y. Shi, A. Deng, M. Deng, J. Li, M. Xu, S. Zhang, X. Ding, and S. Xu, “Domain transferability-based deep domain generalization method towards actual fault diagnosis scenarios,” IEEE Transactions on Industrial Informatics, vol. 19, no. 6, pp. 7355–7366, 2023.

[16] Y. Zhou, Y. Dong, and G. Tang, “Time-varying online transfer learning for intelligent bearing fault diagnosis with incomplete unlabeled target data,” IEEE Transactions on Industrial Informatics, vol. 19, no. 6, pp. 7733–7741, 2023.

[17] Z. Chen, Y. Liao, J. Li, R. Huang, L. Xu, G. Jin, and W. Li, “A multisource weighted deep transfer network for open-set fault diagnosis of rotary machinery,” IEEE Transactions on Cybernetics, vol. 53, no. 3, pp. 1982–1993, 2023.

[18] G. Mao, Y. Li, S. Jia, and K. Noman, “Interactive dual adversarial neural network framework: An open-set domain adaptation intelligent fault diagnosis method of rotating machinery,” Measurement, vol. 195, p. 111125, 2022.

[19] J. Li, R. Huang, G. He, S. Wang, G. Li, and W. Li, “A deep adversarial transfer learning network for machinery emerging fault detection,” IEEE Sensors Journal, vol. 20, no. 15, pp. 8413–8422, 2020.

[20] C. Yang, J. Liu, Q. Xu, and K. Zhou, “A generalized graph contrastive learning framework for few-shot machine fault diagnosis,” IEEE Transactions on Industrial Informatics, pp. 1–10, 2023.

[21] P. Peng, H. Zhang, M. Li, G. Peng, H. Wang, and W. Shen, “Sclifd: Supervised contrastive knowledge distillation for incremental fault diagnosis under limited fault data,” arXiv preprint arXiv:2302.05929, 2023.

[22] C. Qiu, T. Pfrommer, M. Kloft, S. Mandt, and M. Rudolph, “Neural transformation learning for deep anomaly detection beyond images,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139, 2021, pp. 8703–8714.

[23] W. Zhang, D. Chen, Y. Xiao, and H. Yin, “Semi-supervised contrast learning based on multiscale attention and multitarget contrast learning for bearing fault diagnosis,” IEEE Transactions on Industrial Informatics, vol. 19, no. 10, pp. 10 056–10 068, 2023.

[24] C. Li, X. Lei, Y. Huang, F. Nazeer, J. Long, and Z. Yang, “Incrementally contrastive learning of homologous and interclass features for the fault diagnosis of rolling element bearings,” IEEE Transactions on Industrial Informatics, pp. 1–9, 2023.

[25] T. Zhang, J. Chen, S. Liu, and Z. Liu, “Domain discrepancy-guided contrastive feature learning for few-shot industrial fault diagnosis under variable working conditions,” IEEE Transactions on Industrial Informatics, pp. 1–11, 2023.

[26] T. Shenkar and L. Wolf, “Anomaly detection for tabular data with internal contrastive learning,” in International Conference on Learning Representations, 2021.

[27] N. L. Ricker, “Decentralized control of the tennessee eastman challenge process,” Journal of Process Control, vol. 6, no. 4, pp. 205–221, 1996.

[28] C. Ruiz-Carcel, Y. Cao, D. Mba, L. Lao, and R. Samuel, “Statistical´ process monitoring of a multiphase flow facility,” Control Engineering Practice, vol. 42, pp. 74–88, 2015.

[29] Y. Zhang, “Fault detection and diagnosis of nonlinear processes using improved kernel independent component analysis (kica) and support vector machine (svm),” Industrial & Engineering Chemistry Research, vol. 47, no. 18, pp. 6961–6971, 2008.

[30] T. J. Rato and M. S. Reis, “Fault detection in the tennessee eastman benchmark process using dynamic principal components analysis based on decorrelated residuals (dpca-dr),” Chemometrics and Intelligent Laboratory Systems, vol. 125, pp. 101–108, 2013.

[31] X. Xiu, Y. Yang, L. Kong, and W. Liu, “Data-driven process monitoring using structured joint sparse canonical correlation analysis,” IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 68, no. 1, pp. 361–365, 2021.

[32] Z. Lou, Y. Wang, Y. Si, and S. Lu, “A novel multivariate statistical process monitoring algorithm: Orthonormal subspace analysis,” Automatica, vol. 138, p. 110148, 2022.

[33] J. Wang, P. Liu, S. Lu, M. Zhou, and X. Chen, “Decentralized plantwide monitoring based on mutual information-louvain decomposition and support vector data description diagnosis,” ISA transactions, vol. 133, pp. 42–52, 2023.

[34] A. Bendale and T. E. Boult, “Towards open set deep networks,” CoRR, vol. abs/1511.06233, 2015.

[35] Y. Wen, K. Zhang, Z. Li, and Y. Qiao, “A discriminative feature learning approach for deep face recognition,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11– 14, 2016, Proceedings, Part VII 14. Springer, 2016, pp. 499–515.

[36] Y. Wang, B. Li, T. Che, K. Zhou, Z. Liu, and D. Li, “Energy-based open-world uncertainty modeling for confidence calibration,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9282–9291.

[37] X. Liu, Y. Lochman, and Z. Christopher, “Gen: Pushing the limits of softmax-based out-of-distribution detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[38] S. Garg, S. Balakrishnan, and Z. Lipton, “Domain adaptation under open set label shift,” Advances in Neural Information Processing Systems, vol. 35, pp. 22 531–22 546, 2022.

[39] X. Wang, H. Shi, B. Song, Y. Tao, and S. Tan, “Enhanced dynamic latent variable analysis for dynamic process monitoring,” Journal of the Taiwan Institute of Chemical Engineers, vol. 156, p. 105292, 2024.

[40] B. Liu, Y. Chai, C. Huang, X. Fang, Q. Tang, and Y. Wang, “Industrial process monitoring based on optimal active relative entropy components,” Measurement, vol. 197, p. 111160, 2022.

![](images/bbea4a409539413a6772c9715b1ab5729cd403ff9e2345565cc397e8e0ca51e5.jpg)

Xinyue Wang is currently a graduate student at the Zhejiang University—University of Illinois at Urbana-Champaign Joint Institute, Zhejiang University, Haining, China. Also, she received her undergraduate degree from the Beijing University of Posts and Telecommunications in the major of Digital Media Technology in 2022. Her field interests are artificial intelligence, natural language processing, and fault diagnosis.

![](images/cb7bf24d01e9733ccc5c3956ab67d586c34a8ea851b5619ed6cbb7dae7fda332.jpg)

Peng Peng is currently a doctoral student at the National Engineering Research Centre of Computer Integrated Manufacturing System (CIMS-ERC) in Tsinghua University, Beijing, China. He received his Bachelor degree at the Department of Automation from Northeastern University in 2016. His research interests are process monitoring and prognostic and health management.

![](images/163545922612c957b719a8d7a1b96fb58b90778c467e4f1d6d3a21898da05d64.jpg)

Hanrong Zhang received a dual B.S. degree in Computer Science from the University of Leeds, Leeds, United Kingdom, and Southwest Jiaotong University, Chengdu, China, in 2022. He is currently pursuing a Master degree with the Zhejiang University—University of Illinois at Urbana-Champaign Joint Institute, Zhejiang University, Haining, China. His current research interests include deep learning, fault diagnosis, and knowledge graph.

![](images/296006b1ea390315e1a68cbbfd8865850aed1acf494035e5a7b2593aca99fa0b.jpg)

Xinlong Qiao received the B.E. degree in software engineering from the Harbin Institute of Technology, Weihai, China, in 2023. He is currently pursuing a Master’s degree at the Zhejiang University-University of Illinois at Urbana-Champaign Joint Institute, Zhejiang University, Haining, China. His current research interests include deep learning, named entity recognition, and fault diagnosis.

![](images/b5d9a4acd6f0d400de56c8e4e2f55eba6472fb8ca02ac5cf477ce5ad92d5876e.jpg)

![](images/5e4054e9373e4a1e3681e89a749d864a7e0c02603dbf39b3274944f55b58a26c.jpg)

Ke Ma is currently a doctoral student in Computer Science at Zhejiang University. He received his M.S. degree in Structural Engineering, Mechanics and Materials from Department of Civil and Environmental Engineering, University of California, Berkeley in 2020.

Hongwei Wang received the B.S. degree in information technology and instrumentation from Zhejiang University, China, in 2004, the M.S. degree in control science and engineering from Tsinghua University, China, in 2007, and the Ph.D. degree in design knowledge retrieval from the University of Cambridge. From 2011 to 2018, he was a Lecturer and then, a Senior Lecturer in engineering design with the University of Portsmouth. He is currently a Tenured Professor with Zhejiang University and the University of Illinois at Urbana–Champaign Joint

Institute. His research interests include knowledge engineering, industrial knowledge graph, intelligent and collaborative systems, and data-driven fault diagnosis. His research in these areas has been published in over 120 peerreviewed papers in well-established journals and international conferences.

![](images/5662c525510a1a193d3d0ed35ec11ee03c71fb3f7ef986f66df6cea414369a06.jpg)

Shuting Tao is currently a doctoral student at the College of Computer Science and Technology in Zhejiang University, China. She received dual Bachelor degree in Computer Engineering from Zhejiang University and University of Illinois at Urbana-Champaign in 2021. Her current research interests include machine learning, knowledge graph, and imbalanced learning.