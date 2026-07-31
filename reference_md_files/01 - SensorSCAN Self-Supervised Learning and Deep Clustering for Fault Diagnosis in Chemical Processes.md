# SENSORSCAN: SELF-SUPERVISED LEARNING AND DEEP CLUSTERING FOR FAULT DIAGNOSIS IN CHEMICAL PROCESSES

Maksim Golyadkin AIRI HSE University Moscow, Russia mgolyadkin@hse.ru

Vitaliy Pozdnyakov I AIRI H HSE University M Moscow, Russia 1z pozdnyakov@airi.net ABSTRACT

Leonid Zhukov HSE University Moscow, Russia lzhukov@hse.ru

Ilya Makarov AIRI Moscow, Russia makarov@airi.net

Modern industrial facilities generate large volumes of raw sensor data during the production process. This data is used to monitor and control the processes and can be analyzed to detect and predict process abnormalities. Typically, the data has to be annotated by experts in order to be used in predictive modeling. However, manual annotation of large amounts of data can be difficult in industrial settings.

In this paper, we propose SensorSCAN, a novel method for unsupervised fault detection and diagnosis, designed for industrial chemical process monitoring. We demonstrate our model’s performance on two publicly available datasets of the Tennessee Eastman Process with various faults. The results show that our method significantly outperforms existing approaches (+0.2-0.3 TPR for a fixed FPR) and effectively detects most of the process faults without expert annotation. Moreover, we show that the model fine-tuned on a small fraction of labeled data nearly reaches the performance of a SOTA model trained on the full dataset. We also demonstrate that our method is suitable for real-world applications where the number of faults is not known in advance. The code is available at https://github.com/AIRI-Institute/sensorscan.

## 1 Introduction

Chemical processing plants use specialized equipment and technology in the manufacturing process. The stability of the production process is usually maintained by a closed-loop control system that can automatically perform small corrections to the operation control parameters to keep the process variables within the desired production range (e.g., reactor temperature, flow velocity, etc.). Process lines are typically well instrumented, with sensors providing live feed to monitoring and control systems. Nevertheless, unexpected process behavior occasionally occurs, both within routine operations (either due to the variability of feedstock or when switching to a new target product) or due to some external factors. This may lead to a decrease in the process yield, to process interruption, and eventually to increased equipment wear or even breakdown.

An important part of a modern process monitoring system is early fault detection and diagnostics. A fault is typically defined as a deviation of a process variable from the acceptable production range [1] – for example, increased reactor temperature or decreased feed velocity. Identification of faults helps select recovery procedures to return the process to a normal state.

Most modern data-driven fault detection and diagnosis (FDD) methods are developed in a supervised learning setting that requires all sensor data for each time interval to be labeled with the corresponding process state. However, manual labeling of large amounts of data can be expensive and difficult in industrial settings. For example, it may be challenging to determine the exact moment when a process fault starts, due to smoothness of change in the process behavior. In addition, some equipment faults may remain unnoticed due to an early correction during regular technical service. As an alternative, unsupervised FDD methods were proposed for detecting sensor data patterns grouped according to the process states. Traditional unsupervised methods such as principal component analysis (PCA) and Fisher discriminant analysis (FDA) are used to reduce the dimensionality of sensor data, while clustering methods such as k-means and DBSCAN are used to group samples according to process states. The clusters are then manually mapped to fault types so that the trained model can be used as a diagnostic system.

Deep learning methods were proposed to model complex nonlinear relationships between sensors and efficiently process high-dimensional sensor data. Deep learning can be applied to unsupervised FDD by means of feature extraction and subsequent deep clustering, which are alternatives to the traditional dimensionality reduction followed by clustering approach. In recent years, several deep clustering methods have been proposed for unsupervised image classification [2, 3, 4, 5, 6]. However, in these works, the feature extractor is randomly initialized and trained in order to improve the quality. As a result, the feature extractor relies on low-level features, thus missing high-level hidden properties of the input data. Self-supervised learning (SSL) methods are aimed at pretraining the feature extractor on pretext tasks to represent the high-level properties of input data [7, 8, 9].

In this paper, we propose a novel unsupervised FDD method, SensorSCAN. The method is based on deep learning techniques designed to achieve high accuracy on chemical sensor data — namely, SSL and deep clustering. First, a Transformer-based feature extractor [10] is pretrained to embed sensor data into a latent space using SSL methods designed for sensor data. Second, a small feedforward network called clustering head is trained to map embeddings onto cluster indices using modified Semantic Clustering by Adopting Nearest Neighbors (SCAN) algorithm [11]. The clustering training also involves updating feature extraction weights, which makes the latent space separable and leads to faster fault detection. Both techniques (SSL and deep clustering) are combined into a model, which ensures consistency between them.

We utilized a label matching technique in our experiments to simulate the manual mapping process performed by experts (see Subsection 3.4). In the basic case, a cluster is assigned to a fault that occurs most frequently in that cluster. A schematic view of SensorSCAN is depicted in Figure 1.

![](images/445c283c83a790e74cafdc58d448a3a16a38912b019b346a7de39a221d935ab4.jpg)  
Figure 1: Overview of the SensorSCAN method. First, the feature extractor is trained to map unlabeled sensor data onto latent representations via self-supervised learning. Second, the feature extractor and the clustering head are jointly trained, which makes the latent space well separable. The final step is the manual mapping of the clusters’ indices onto the process states.

Our method demonstrates the best results measured with multiple clustering evaluation metrics on the Tennessee Eastman Process (TEP) benchmarks compared to modern unsupervised FDD methods, such as ConvAE [12]. Additionally, we implement the semi-supervised setting in which we fine-tune the pretrained model on a small fraction of labeled data. Experimental results show that a model pretrained using SSL methods and then fine-tuned on very small number of labeled examples is close in performance in terms of the True Positive Rate to the SOTA supervised models trained on the full TEP dataset.

The paper is organized as follows. Section 2 overviews traditional and state-of-the-art data-driven methods for both supervised and unsupervised FDD, as well as SSL and deep clustering methods designed for sensor data. Section 3 contains a detailed description of the proposed model. Section 4 presents Tennessee Eastman Process simulators and corresponding datasets. Section 5 describes the evaluation metrics and their practical significance. In Section 6, we present the results of an ablation study and discuss the sensitivity of the performance to methodological choices, i.e. explore the relationship between the performance and the stability of our model by removing certain components and changing the pretraining setting in order to understand its contribution to the overall model. Section 7 describes the models that we consider as the baseline in the experiments. Section 8 overviews the experimental results. In Section 9, we discuss the applicability of our model to real-world industrial settings. Finally, Section 10 contains the summary of the results and outlines directions for future research.

## 2 Related work

Recently, numerous data-driven methods have been proposed for industrial process fault detection and diagnosis [13, 14, 15]. The prevalence and success of data-driven methods are due to the fact that they operate on information extracted from process history, and modern production facilities generate large amounts of such data. Yet, in most cases, this data is not publicly shared for security reasons, therefore researchers use simulated data [16, 17, 18] for reproducibility of results and comparison with existing methods.

A vast majority of data-driven approaches are built utilizing feature extraction from raw historical data. However, data from industrial sensors is highly redundant and intercorrelated, which impedes many data-driven approaches. Feature extraction is beneficial for robustness and improvement of performance since valuable information is often contained in manifolds of lower dimensionality. The most frequently used feature extraction methods are based on Canonical Correlation Analysis (CCA) [19, 20], Principal Component Analysis (PCA) [21, 22, 23, 24, 25], Partial Least Squares (PLS) [26, 27, 28], and different variations of autoencoders (AE) [29, 30, 31, 32, 33, 34, 35, 36, 37, 38]. Over the past decade, deep neural networks have gained popularity in application to industrial processes because they are simultaneously trained to perform feature extraction and classification, which leads to better performances [39, 40, 41, 42, 43, 44, 45].

There are two general approaches to fault detection and diagnosis - the supervised and the unsupervised ones. The supervised setting requires a labeled dataset, which means that each time interval must be marked as belonging to a normal or an abnormal process state. As a result, the trained model distinguishes only between the abnormal states observed in the process history. In [46], the authors propose a method based on a deep convolutional neural network. Zhang et al. [47] show that the Deep Belief Network (DBN) is able to effectively extract features in complex chemical processes. Wang [48] presents a modification of the latter architecture called Extended DBN, which alleviates some of the training imperfections. Park et al. [49] use a combination of autoencoders and LSTM to perform fault detection and fault diagnosis in two stages. Lomov et al. [50] report the Temporal CNN1D2D architecture to simultaneously capture long-running patterns in a single sensor as well as relationships between different sensors.

However, the data which is typically available does not contain all the possible faults; therefore, there is a motivation to use unsupervised methods that are not restricted to a set of previously observed and labeled faults. In [51, 52, 53, 54, 55], alternative approaches to time series clustering and segmentation have been proposed. Application of the CatGAN [56] architecture for unsupervised fault diagnosis is reported in [57, 58]. The DeepAnT [59] architecture utilizes time series prediction for unsupervised anomaly detection. Yu and Yan [60] propose a modified autoencoder architecture with LSTM blocks, called unLSTM, for unsupervised fault detection. Yang et al. [61] and Li et al. [62] demonstrate models based on discrimination and reconstruction anomaly score of a pretrained GAN. Zheng and Zhao [12] present a method employing stacked AE, t-SNE, and DBSCAN for clustering and generation of pseudolabels used for supervised training. Rajeevan et al. [63] propose an approach that utilizes an incremental one-class neural network for unsupervised anomaly detection and a dynamic shallow neural network for supervised fault diagnosis.

## 2.1 Self-supervised learning for time series

Self-supervised learning methods for time series can be categorized into two groups. Methods from the first group employ pretext tasks that exploit the structural features specific to time series. Oorg et al. [64] perform time series forecasting in the latent space for representation learning. Another temporal-aware pretext task for self-supervised change point detection is proposed in [7]; it consists in determining whether one time series is an extension of another. Approaches from the second group use general representation learning methods which can be applied to any data type and demonstrated the best performance on a variety of machine learning tasks in recent years [65, 66]. Fortuin et al. [67] develop a modification of variational autoencoder that produces topologically interpretable discrete representations for time series. Mohsenvand et al. [8] apply contrastive learning [68] to EEG time series. Adaptation of the pretraining routine of a transformer-based BERT [69] model to multivariate time series is shown in [9]. TS-TCC [70] combines contrastive learning and time series forecasting for representation learning.

## 2.2 Deep clustering

Basic deep clustering methods are based on application of classical clustering algorithms to the features extracted with a pretrained feature extractor [71, 72, 73, 74]. Typically, feature extractors are autoencoders or models trained with representation learning. These models can be trained with unlabeled data, and their operation is based on extraction of semantically meaningful features from raw data. It is reasonable to expect embeddings of objects to be distributed in the latent space according to the closeness of their semantic meanings rather than raw similarity. However, it often turns out in practice that object embeddings are placed too densely, which complicates the work of the classic clustering algorithms. Alternatively, there exist models trained with end-to-end indirect loss functions to map inputs to cluster indices [2, 3, 4, 5, 6]. The key drawback of this approach is that the same loss is used for the feature extraction and the class assignment training processes, i.e. these processes are jointly optimized while having inherently different goals. At the beginning of training, the network is only able to rely on low-level features because the feature extractor is randomly initialized. As learning continues, the model is not able to significantly change the decision-making process. As a result, the network operates only with low-level features ignoring global features, which leads to a suboptimal solution. This shortcoming may be addressed by pretraining the feature extractor and subsequently training the network for cluster assignment with the extracted features. TSUC [75] was the first method employing this idea: the feature extractor is pretrained with a self-supervised learning algorithm, and the classification head is trained with a combination of mutual information loss and contrastive loss. Van Gansbeke et al. proposed the SCAN algorithm [11], which performs clustering by enforcing the similarity of cluster predictions for neighboring objects with the closeness calculated in the embedding space of the pretrained feature extractor. The results of the arbitrary end-to-end deep clustering algorithm can be improved with the RUC [76] add-on module that is based on robust learning techniques

With respect to existing approaches, our work makes the following contributions:

• We introduce a novel unsupervised learning approach designed specifically for the fault detection domain. The approach involves adapting the proven SCAN method to the sensor data previously unexplored in this context. To our knowledge, we are the first to apply deep clustering methods to fault detection.

• We propose a new self-supervised learning task based on contrastive learning and masked values reconstruction tailored to the fault detection task by means of appropriate augmentations and mask generation. The original SCAN for images is built upon existing widely available self-supervised pretraining methods. However, there is a need for pretraining methods designed purposely for time series, and we address this gap.

• We identify a crucial challenge when applying the SCAN method to the time series domain – namely, that the chunks of time series generated by the sliding window exhibit a higher correlation than images. To mitigate this issue, we develop a subsampling method that effectively handles the high similarities in time series data.

## 3 Model description

The objective of our method is to produce clustering for a given set of unlabeled multivariate time series samples $\mathcal { X } = \{ X _ { 1 } , \ldots , X _ { N } \} , X _ { i } \in \mathbb { R } ^ { L \times D }$ , where L is the sample length and D is the number of sensors. The set X has a corresponding set of labels $Y = \{ y _ { 1 } , \dots , y _ { N } \} , y _ { i } \in \{ 1 , \dots , Q \}$ that are not available for training. The produced clusters have to be consistent with the label distribution, which means that samples with the same label should be referred to the same cluster, and samples with different labels should belong to different clusters. This way, the ground truth labels are correctly restored with human assistance. It is also assumed that Q may not be known.

![](images/1acd297248b0b946b841ccd48ddd89b1ce4ef6739c3bfad43e2a675a51889555.jpg)  
Figure 2: A schematic view of training SensorSCAN. Left: self-supervised pretraining of feature extractor F with reconstruction and constartive losses. Right: clustering training of feature extractor $\mathcal { F }$ and clustering network C with SCAN loss.

Our model consists of two parts: a large feature extractor $\mathcal { F }$ and a small clustering network C. The feature extractor retrieves characteristic patterns from the data and maps the input samples onto an embedding space of low dimensionality. The clustering network maps an embedding vector onto a vector of probability distribution over $\tilde { Q }$ clusters, where $\tilde { Q }$ is determined in advance. The training procedure consists of two steps (see Figure 2). First, the feature extractor is pretrained with selfsupervised learning methods, and the number of clusters is determined via visual analysis of the latent space distribution. Second, the clustering network and the feature extractor are trained using SCAN loss [11], and information about the neighbors is obtained from the embedding space. In what follows, we describe each training step in more detail.

## 3.1 Self-supervised pretraining

We use self-supervised learning methods to enable our model to distinguish samples from each other. With these approaches, the neural network explores the internal structure of the data without ground truth labels. The model finds the data-specific patterns containing information about the corresponding process, which are then utilized for clustering.

We employ the feature extractor based on the Transformer [10] architecture. This architecture is highly suitable for sequence processing due to the self-attention modules that provide a global receptive field and the capability to find sophisticated dependencies between the sequence elements.The feature extractor consists of four parts: encoder $\tau _ { \ast }$ , sequential pooling layer $\bar { \mathcal { P } } _ { \cdot }$ , projection head H, and reconstruction head R.The encoder $\tau$ is a three-layer Transformer encoder with the embedding dimensionality H. It maps a sample of length $L$ to a sequence of embeddings of the same length, $\tau$ $\mathbb { R } ^ { L \times D } \xrightarrow { } \mathbb { R } ^ { L \times H }$ . Sinusoidal positional encoding is used to provides information about the position of each timestamp. The sequential pooling layer $\mathcal { P }$ maps a sequence of the embedding vectors $\tilde { h } = [ h _ { 1 } , \dots , h _ { L } ] \in \mathbb { R } ^ { L \times H }$ onto a single vector $\hat { h } \in \mathbb { R } ^ { H }$ . The pooling is performed with weighted sum, where the weights $w \in \mathbb { R } ^ { L }$ are obtained with learnable affine transformation and softmax operation:

$$
w = \operatorname{softmax} (W _ {p o o l} ^ {\top} \tilde {h} ^ {\top}), W _ {p o o l} \in \mathbb {R} ^ {H \times 1},\tag{1}
$$

$$
\hat {h} = w \tilde {h}.\tag{2}
$$

The projection head H is a 2-layer MLP with intermediate BatchNormalization [77] and ReLU activation. Its goal is to reduce the dimensionality of the encoder’s embedding space to encourage the model to encode more informative features and to improve the embedding quality, H : $\mathbb { R } ^ { H } \overset { \sim } { \to } \mathbb { R } ^ { F }$ where $F$ is the size of the feature extractor embeddings. Finally, the reconstruction head R is meant to reconstruct raw sensor data from the encoder’s embedding vectors, $\mathcal { R } \colon \mathbb { R } ^ { H } \to \mathbb { R } ^ { D }$ . It is used exclusively for training and consists of one linear layer.

In order to improve the robustness and to achieve higher discriminative power, we employ a combination of two self-supervised learning tasks – masked input reconstruction [9] and contrastive learning [68]. These methods use different procedures to explore data distribution – namely, masked values retrieval and distinguishing between similar and dissimilar samples, respectively. Each of the two methods performs rather well when applied separately, but their pretext tasks effectively complement each other, so we choose to combine them in data preprocessing and loss calculation.

The goal of the first task, mask input reconstruction, is to reconstruct values from the partially masked input. The idea behind this task is that the model has to properly learn the internal structure of the data to recover the values, thus positively affecting the quality of feature extraction and the embeddings. In addition, industrial data may contain missing values due to sensor malfunctions or missing logs and thus, masking adapts our model to such disruptions. For a given sample $\boldsymbol { X } \in \mathbb { R } ^ { L \times D }$ , the binary mask $M \in \{ 0 , 1 \forall ^ { L \times D }$ is generated to produce masked samples with element-wise multiplication ${ \hat { X } } = X \odot M$ . The mask generation process is based on geometric distribution: for each column corresponding to the measurements of a single sensor, sequences of zeros and ones are consecutively generated, with sequence lengths being independently sampled from geometric distributions with expectations $l _ { m }$ and $l _ { u }$

$$
l _ {u} = \frac {1 - r}{r} l _ {m},\tag{3}
$$

where r is the ratio of the masked input. Parameters $l _ { m }$ and r determine the complexity of the task and directly affect the characteristics of the obtained embedding space. The geometric distribution is preferred to the Bernoulli distribution because masks generated with the Bernoulli distribution often result in sequences of zeros of unit length. Reconstruction of values from the neighboring points is a relatively simple task in which no informative features are learnt. The loss function for this task is MSE, calculated only for the masked values:

$$
\mathcal {L} _ {r e c} (\hat {X}, X) = \frac {1}{B} \sum_ {i = 1} ^ {B} \frac {1}{| M ^ {i} |} \sum_ {{l, d: M _ {l d} ^ {i} = 0}} (\mathcal {R} (\mathcal {T} (\hat {X})) _ {l d} - X _ {l d}) ^ {2},\tag{4}
$$

where B is the batch size, and l and d correspond to the number of timestamps and the number of sensors, respectively.

The second task of the self-supervised pretraining is contrastive learning; its core objective is to train the feature extractor so that the embedding vectors of similar samples are closer in the embedding space than those of dissimilar samples. To achieve this, pairs of similar samples are created using two augmentations applied to the same sample. Next, the model is trained to find mutual patterns within the paired samples and discriminate these samples from the others. Thus, the choice of augmentation type is crucial and depends on the downstream task and the data type. Moreover, augmentation is beneficial in the context of application to industrial data since the amount of data for anomalies is limited and all the possible disturbances are not represented in the historical data.

We use the jitter, scaling, and permutation operations [78] for data augmentation. Jitter augmentation is an injection of additive Gaussian noise. Scaling augmentation is multiplication by a value sampled from some random distribution, with sampling being independent for each variable in a multivariate time series. Finally, the permutation augmentation consists in splitting a sample into a predetermined number of chunks of random length that are then shuffled and concatenated back into a single sample. Following the approach proposed in [70], we split augmentations into weak and strong ones to diversify the data so as to improve the model robustness. For the first sample of the pair, we use a combination of scaling and jitter as a weak augmentation, since it modifies the sample slightly. For the second sample, we use a combination of permutation and scaling as a strong augmentation; it breaks time dependencies but preserves the semantic information such as process state.

The training is performed via NT-Xent loss [68] minimization (normalized temperature-scaled cross entropy). We randomly sample a minibatch of B samples and apply the augmentations to transform it into a minibatch of 2B samples. Thus, for each sample there is one positive pair and $\left( 2 B - 2 \right)$ negative pairs. The NT-Xent loss makes the distance between positive pairs smaller than the distance between negative pairs. For the positive pair (i, j) it is defined as:

$$
l _ {i, j} = - \log \frac {\exp (\mathrm{sim} (\pmb {z} _ {i} , \pmb {z} _ {j}) / \tau)}{\sum_ {k = 1} ^ {2 B} [ k \neq i ] \exp (\mathrm{sim} (\pmb {z} _ {i} , \pmb {z} _ {k}) / \tau)},\tag{5}
$$

where $z _ { i } , z _ { j } , z _ { k } \in \mathbb { R } ^ { F }$ denotes the outputs of the projection head, sim $( \mathbf { u } , \mathbf { v } ) = \mathbf { u } ^ { \top } \mathbf { v } / ( | | \mathbf { u } | | | | \mathbf { v } | | )$ denotes the cosine similarity, and τ denotes the temperature parameter. The final loss $\mathcal { L } _ { c o n t }$ is calculated across all the 2B positive pairs.

The two above tasks affect different latent representations. The first task targets the embeddings generated by the encoder T for every time series element. The second task influences the embedding of the whole sample, produced from element-wise embeddings with a sequential pooling P and a projection head $\bar { \mathcal { P } } _ { * }$ . Thus, there is a potential for combining these methods to improve the efficiency of the pretraining procedure. By reducing the number of errors and improving the quality of the pretraining, we directly affect the final performance; this is crucial because the subsequent clustering training alone would not be able to resolve the fundamental imprecisions of feature extraction.

Eventually, the SSL pretraining is performed over $E$ epochs. The iteration of pretraining is as follows:

1. Randomly sample a minibatch of size B.

$$
X = [ X _ {1}, \dots , X _ {B} ]\tag{6}
$$

2. Transform every sample with weak augmentation α and strong augmentation $\beta .$

$$
\tilde {X} = [ \alpha (X _ {1}), \beta (X _ {1}), \ldots , \alpha (X _ {B}), \beta (X _ {B}) ]\tag{7}
$$

3. Independently for each sample, generate and apply a binary mask M<sub>i</sub>.

$$
\hat {X} _ {i} = \tilde {X} _ {i} \odot M _ {i}, i = \overline {{1 , 2 B}}\tag{8}
$$

4. Reconstruct the masked values and calculate the reconstruction score.

$$
\mathcal {L} _ {r e c} = \mathcal {L} _ {r e c} (\hat {X}, \tilde {X})\tag{9}
$$

5. Produce embedding vectors for the masked samples and calculate the NT-Xent loss.

$$
\pmb {z} _ {i} = \mathcal {H} (\mathcal {P} (\mathcal {T} (\hat {X} _ {i}))), i = \overline {{1 , 2 B}}\tag{10}
$$

$$
\mathcal {L} _ {c o n t} = \frac {1}{2 B} \sum_ {k = 1} ^ {B} (l _ {2 k - 1, 2 k} + l _ {2 k, 2 k - 1})\tag{11}
$$

6. Calculate the total loss by the weighted sum over the two losses.

$$
\mathcal {L} = \mathcal {L} _ {r e c} + \lambda_ {c o n t} \mathcal {L} _ {c o n t},\tag{12}
$$

where $\lambda _ { c o n t }$ denotes the weight that adjusts the impact of contrastive loss on joint training.

7. The feature extractor weights are updated according to the step of the optimization algorithm minimizing loss.

After the training, we discard the reconstruction head, and the feature extractor is defined: ${ \mathcal { F } } =$ $\mathcal { T } \circ \mathcal { P } \circ \mathcal { H } , \mathcal { F } \colon \breve { \mathbb { R } } ^ { L \times D } \to \mathbb { R } ^ { F }$ ; masking and augmentations are no longer applied.

## 3.2 Clustering training

Clustering groups samples according to their similarity, thus separating different process states in the absence of ground truth labels. The SCAN algorithm, initially designed for images, has been shown to exploit the high-quality embeddings of the pretrained feature extractor in the most efficient way, hence we choose it to adapt for our domain.

The training requires preprocessing called nearest neighbors mining. For every sample $X _ { i } ,$ , we retrieve its K nearest neighbors $\mathcal { \bar { N } } _ { X _ { i } }$ in the feature extractor’s embedding space. Unlike the original SCAN approach, we do not search for nearest neighbors over the entire training set but over a subsample of the nearest neighbors. The reason is that time series data is usually obtained by applying a sliding window of a small step size. As a result, the nearest neighbors in the embedding space are highly overlapped time series which are not sufficiently representative of the diversity of the data.

The nearest neighbors mining algorithm is as follows:

1. Randomly shuffle the training dataset: ${ \mathcal { X } } = [ X _ { 1 } , \ldots X _ { n } ]$

2. Split the dataset into T chunks of equal size: $\mathcal { X } = [ X _ { j _ { 1 } } , \ldots , X _ { j _ { T } } ] .$

3. For each $X _ { i } ,$ the nearest neighbors are found within the chunk $X _ { k }$ it belongs to: $\mathcal { N } _ { X _ { i } } =$ NearestNeighbors $( X _ { i } , X _ { k } )$

The clustering network C is a 2-layer MLP with intermediate BatchNormalization and ReLU activation, and final softmax activation, $\mathcal { C } \colon \mathbb { R } ^ { F } \to \mathbb { R } ^ { \tilde { M } }$ . The learning process is designed to enforce the same label prediction distribution for both the sample and its neighbors. To avoid the trivial solution when all samples are referred to the same class, the entropy term that penalizes uneven sizes of clusters is used. In total, the loss is computed as follows:

$$
\mathcal {L} _ {S C A N} = - \frac {1}{B} \sum_ {i = 1} ^ {B} \log \left\langle \mathcal {C} (\mathcal {F} (X _ {i})), \mathcal {C} (\mathcal {F} (X _ {i} ^ {N N})) \right\rangle + \lambda_ {e n t} H (\mathcal {C} ^ {\prime}),\tag{13}
$$

$$
\mathcal {C} ^ {\prime} = \frac {1}{B} \sum_ {i = 1} ^ {B} \mathcal {C} (\mathcal {F} (X _ {i})),\tag{14}
$$

where $\langle \cdot , \cdot \rangle$ denotes the dot product, $\lambda _ { e n t }$ denotes entropy loss weight, $X _ { i } ^ { N N }$ denotes the neighbour randomly sampled from $\mathcal { N } _ { X _ { i } }$ , and $H ( { \mathcal { C } } ^ { \prime } )$ denotes the entropy over discrete distribution $\scriptstyle { \mathcal { C } } ^ { \prime }$

We would like to notice that end-to-end learning not only provides clustering training but also updates the feature extractor weights, thereby improving disentanglement of its embedding space.

## 3.3 Number of clusters

In practice, the number of underlying classes is not known upfront, but the SCAN algorithm requires it to be set in advance. As shown in the original paper [11], it is possible to employ an overclustering approach by roughly estimating the number of classes. A lower bound on the number of clusters can be obtained by visual inspection of the data after dimensionality reduction with t-SNE [79] applied to time series representations after self-supervised pretraining.

Since the largest cluster in t-SNE visualization corresponds to the normal behavior, we subsample it to make equally-sized classes for training. The reason is that the entropy term in the loss is sensitive to highly imbalanced data, which is usual for real process history.

In Section 6, we analyze the performance of the model with the number of clusters different from the number of classes and advocate for the choice of t-SNE among the other dimensionality reduction approaches, such as PCA [80] and UMAP [81].

The real-world setting requires domain experts to identify faults in the corresponding clusters. It is much easier for an expert to determine that two clusters contain entries with the same fault than to split a cluster containing many faults; therefore, we consider the overclustering approach optimal and practically feasible.

## 3.4 Label matching

Since one of the fault detection and diagnosis (FDD) goals is to determine the process state, the algorithm is to assign a label to each cluster.

In our experiments, we map cluster index l to a corresponding process state by the weighted maximum occurrence as follows:

$$
\operatorname{LM} (l) = \operatorname{argmax} _ {q} \sum_ {i = 1} ^ {n} \alpha_ {q} ^ {l} [ c _ {i} = l \wedge y _ {i} = q ]\tag{15}
$$

where $\alpha _ { q } ^ { l }$ is the weight of the label $q$ in the cluster l. In our experiments, we set $\alpha _ { q } ^ { l } = 1$ for all faulty states and $\alpha _ { q } ^ { l } = Q _ { l } + 1$ for the normal state, where $Q _ { l }$ is the number of process states in the cluster l. In other words, a cluster is assigned to a normal state if the share of normal samples in the cluster is at least $1 / ( Q _ { l } + 1 )$ ; this parameter’s value helps to reduce false alarms. The label matching procedure is performed only on the training set to obtain the cluster index, i.e. the matching process state that is later used for labeling the test samples. Note that in real-world industrial settings, the cluster-to-label matching has to be performed manually by experts. There exist many labeling approaches that are beyond the scope of the present work, but we assume that labeling just a few samples in each cluster produces the correct matching if the unsupervised model is sufficiently accurate.

## 3.5 Fine-tuning on few labeled runs

Utilization of self-supervised learning techniques makes it possible to efficiently retrieve hidden data distribution without human-made annotations. However, the process history may contain few examples of labeled data that can be used for supervised learning instead of the clustering and label matching approach. In other words, we first perform self-supervised pretraining on unlabeled data. Then, we perform fine-tuning by means of supervised training on the few labeled examples using cross-entropy loss and regularization methods, e.g. label smoothing [82, 83]. In Section 8, we show that our model fine-tuned with one simulation run for each fault is able to exhibit results that are almost comparable to the model trained with all data and labels.

## 4 Tennessee Eastman Process

The Tennessee Eastman Process (TEP) is a well-known benchmark for testing process control and FDD methods. It was created by Eastman Chemical Company and is presented in [84]. The TEP model is based on a real chemical plant process and allows simulating various processes and process faults; a detailed description of TEP can be found in A.

Two distinct numerical simulators of the TEP process are available. The first one employs the control scheme from [85]. The corresponding dataset (available at <sup>1</sup>) with 20 faults was presented in [86]; it consists of a process simulation run for normal operations and for each fault in both the train and the test sets. To increase the diversity of data, an extended version of this dataset <sup>2</sup> (hereafter $\mathrm { T E P _ { R i e t h } } )$ was proposed and employed in [87]. It was simulated under the same settings as the previous one, but it contains as many as 500 runs with different random seeds for every faulty and normal operation. The second simulator is based on different control schemes [88, 89, 90] and is available from the Tennessee Eastman Challenge Archive <sup>3</sup>. Primarily, this simulator includes the modified TEP model suggested in [91]. First, it solves the problem where the result of the simulation depends on the chosen ODE solver if random variation disturbances are activated. Besides, it features 8 additional faults, process measurements, and simulation options. The extended dataset (hereafter $\mathrm { T E P _ { R i c k e r } } )$ generated with this model is presented in [92].

In $\mathrm { T E P _ { R i e t h } }$ , there are 500 simulation runs for each process state in both the training and the testing sets, which totals 21,000 simulation runs. A single simulation run consists of 500 and 960 time stamps for training and testing, respectively. The sampling rate is equal to 3 minutes. To make the experimental setup consistent with real fault diagnosis situations, we unbalance the training set by cropping it to 500 runs for the normal state and 5 runs for each faulty state. For testing, we keep the testing set as is.

$\mathrm { T E P _ { R i c k e r } }$ consists of 2000 time stamps for each simulation run with the sampling rate of 3 minutes. The total run duration is 100 hours, with a fault introduced in the 30-th hour. There are 100 simulation runs for each process state, which totals 2,800 runs. Since there is no train/test split in the original dataset, we split the simulation runs by the ratio of 80/20. Note that this dataset does not have runs that consist only of normal behavior without fault introduction; consequently, we maintain the superiority of the normal state.

## 5 Evaluation metrics

We perform FDD on multivariate time series data $\mathcal { X } = \{ X _ { 1 } , \ldots , X _ { N } \} , X _ { i } \in \mathbb { R } ^ { L \times D }$ , where L is the sample length and D is the number of sensors. Each sample is generated by a sliding window of size L with a certain step size. Thereby, each sample is a matrix where every row corresponds to a single time stamp and the column to a sensor. Each sample is assigned with a process state, the set of states being $Y \overset {  } { = } \{ y _ { 1 } , \ldots , y _ { N } \} , y _ { i } \in \{ 1 , \ldots , Q \}$ . There is one normal state and $Q - 1$ faulty states. In our experiments, we set the window size to 100 and the step size to 1; that is, a model has to collect 100 time stamps to make a first prediction. After that, a model is able to predict the process state for every subsequent time stamp. For both datasets, $\mathrm { T E P _ { R i e t h } }$ and $\mathrm { T E P _ { R i c k e r } } ,$ , the window size and the step size correspond to 300 minutes and 3 minutes, respectively. We call a sequence of samples a “run”. Each run starts from the initial process state and continues until the process stops. Runs can be normal or faulty: a normal run contains only normal samples; a faulty run contains several normal samples, while the rest are faulty. That is, we can explicitly define the first faulty sample in each faulty run, which is used in the evaluation of the detection delay.

## 5.1 Clustering metrics

Clustering metrics measure the discrepancy between ground truth labels Y and cluster indices C without supevision steps such as label matching.

• Unsupervised Clustering Accuracy (ACC) [93] is similar to classification accuracy and is calculated as the maximum accuracy over all possible matches between cluster indices and ground truth labels:

$$
\operatorname{ACC} (Y, C) = \max _ {f} \frac {\sum_ {i = 1} ^ {n} [ y _ {i} = f (c _ {i}) ]}{n},\tag{16}
$$

where n is the size of the test dataset and f is a matching function. Calculation of ACC involves solving a maximization problem that can be formulated as the Linear Assignment Problem. The best matching function can be found using the Hungarian Algorithm [94].

• Normalized Mutual Information (NMI) [93] is an information-theoretic measure that is equal to the mutual information between ground truth labels and cluster indices normalized by the average of their entropies:

$$
\operatorname{NMI} (Y, C) = \frac {2 I (Y , C)}{H (Y) + H (C)},\tag{17}
$$

where $I ( \cdot , \cdot )$ is mutual information and $H ( \cdot )$ is entropy.

• Adjusted Rand Index (ARI) [95]. The Rand Index (RI) considers all pairs of samples and takes into account the ratio of pairs with the correct cluster index in respect to the ground truth labels:

$$
\operatorname{RI}(Y,C) = \frac{\sum_{i = 1}^{n}\sum_{\substack{j = 1\\ i <   j}}^{n}[c_{i} = c_{j} \wedge y_{i} = y_{j}] + [c_{i}\neq c_{j} \wedge y_{i}\neq y_{j}]}{\binom{n}{2}}\tag{18}
$$

where [·] is the indicator function. The Adjusted Rand Index is the corrected-for-chance version of the Rand index.

## 5.2 Detection and diagnosis metrics

To calculate FDD metrics, we need to compare the ground truth labels and the predicted ones. In the unsupervised setting, each class is linked to some specific cluster. In our experiments, we interconnect classes and clusters using the label matching procedure described in Subsection 3.4.

To evaluate the quality of diagnosis, we look at the TPR, FPR, and CDR metrics, where TPR and FPR are calculated separately for each fault, and faulty samples are regarded as positive and normal samples as negative examples:

$\mathrm { T P R } _ { i } ,$ True Positive Rate, aka Detection Rate – the number of detected faulty samples of the type i divided by the number of faulty samples of the type i.

• FPR<sub>i</sub>, False Positive Rate, aka False Alarm Rate – the number of false alarms of the type i divided by the number of normal samples. We assume that a model with FPR greater than 0.05 is not applicable in real cases due to inadequately frequent false alarms.

• CDR, Correct Diagnosis Rate – the total number of correctly diagnosed faulty samples divided by the number of detected faulty samples.

In addition, we separately measure detection metrics in order to assess the model’s ability to detect whatever faults and then correctly diagnose the type. The detection metrics are:

• Detection TPR and Detection FPR – the TPR and FPR in the binary classification task where all faulty samples count as the positive class and all normal samples as the negative class.

• ADD, Average Detection Delay – the average number of samples between the first groundtruth faulty sample and the first detected faulty sample. The averaging is performed across all the faulty runs, excluding the runs with undetected faults (false negatives); the motivation is to keep this metric explicitly interpretable. If the step size is greater than one, then the number of samples is multiplied by the step size. Here we evaluate the delay in the fault detection task, since we consider it important in real-world cases to detect a fault as soon as possible, even if it is misdiagnosed. This allows the operator to prevent accidents by stopping the process or turning on a protection system.

## 6 Ablation study and sensitivity to methodological choices

In this section, we present an overview of the set of experiments conducted to investigate the sensitivity of our method to the removal of its components, on the one hand, and to methodological choices, on the other. We reviewed the performance of the model under several conditions: by removing the tasks in SSL, by substituting the nearest neighbors mining approach with the original one, and by changing the number of clusters, the training set, and the dimensionality reduction technique. The study was performed on $\mathrm { T E P _ { R i e t h } }$

We carried out the pretraining with masked input reconstruction and the contrastive learning tasks separately to show that they complement each other; we also compared the performance of our nearest neighbors mining approach to the original one (proposed in [11]) for the model pretrained with both of the self-supervised tasks. The visual results of the study are depicted in Figure 5, and the numerical outcomes are presented in Table 3. We used the clustering method proposed in [12] for the first three configurations.

It may be noticed that the combination of self-supervised methods significantly improves disentanglement of the embedding space and increases the number of discriminated classes (Figure 5(c)). It is also noticeable that SCAN with nearest neighbors mining (Figure 5(d)) proposed in the original paper yields worse results than the method proposed in [12]. However, SCAN with our modification of nearest neighbors mining outperforms the latter. Substantial improvement can be seen in Figure 5(e), when Fault 10 and Fault 16 (the orange and the green dots) are separated into two clusters.

We also conducted a series of experiments to evaluate how the predetermined number of clusters M<sup>˜</sup> affects the final performance; $\mathrm { T E P _ { R i c k e r } }$ was used since it contains a higher number of faults. In Figure 3 and Table 1, we show that overclustering does not drop the performance significantly (the FPR values are omitted from the chart since they all are below the 0.05 threshold; the exact TPR and FPR values can be found in C). When using twice as many clusters as the classes, the result is almost identical to using the correct number of clusters — however, twice as much human labor is required for cluster labeling. With a small increase in the number of clusters, the performance of the model even improves where the need for human labor remains essentially the same. Imperfect pretraining produces an embedding space with blobs containing samples of two or more classes; however, overclustering allows separating such blobs into pure clusters. In contrast, underclustering significantly reduces performance due to the fact that at least one cluster will contain multiple ground truth classes.

To demonstrate the high generalizability of the pretrained feature extractor, we compared the performance of the models trained in the first step with different subsets of faults; the results are reported in Figure 4 and Table 2 (the FPR values are omitted from the chart since they all are below the 0.05 threshold; the exact TPR and FPR values are found in C). Faults were divided into two equal subsets based on their difficulty, as shown in Table 19: Faults 4, 5, 6, 7, 8, 11, 12, 14, 19, 20, 23, 24, and 27 were regarded as easy, and the rest were considered difficult. In the second training step, all the faults were used. We also used an untrained model with randomly initialized weights for comparison. The results show that the majority of faults that were not previously seen by the feature extractor are separable in its latent space. However, the presence of a fault in the pretraining dataset does not guarantee its detection. In addition, training can even deteriorate the quality of the embeddings for the samples of this fault (see Fault 26 from the "Difficult faults" model), which is clearly seen in comparison with the untrained model. However, training increases the CDR for detected errors and improves fault diagnosis, especially when training with the difficult faults. To summarize, our model benefits from fault diversity that facilitates productive pretraining and results in high retrieval capability.

Finally, we carried out a visual comparison to justify the choice of the dimensionality reduction algorithm (see Figure 7). PCA fails to reduce the dimensionality to two dimensions whilst preserving the data distribution. On the opposite, t-SNE and UMAP adequately represent the data division into groups according to the ground truth labels, which are unknown to the algorithms. Examination of hyperparameters found that UMAP divides samples of the same process state into several groups more frequently than t-SNE, so we decided to utilize the latter in our experiments.

![](images/eb20999dbc3ba1bce10dc60b4dc4e6a60bfed4fad99d798aa83f0feccdd63147.jpg)  
Figure 3: Radar chart with TPR values evaluated on $\mathrm { T E P _ { R i c k e r } }$ for various number of clusters. Faults are numbered on the circle; the distance between 0 and the points represents TPR values; the points in 0 represent undetected faults. FPR values are omitted from the chart since they all are below the 0.05 threshold.

Table 1: Aggregated detection and diagnosis metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$ for various numbers of clusters.

<table><tr><td></td><td>10</td><td>29</td><td>33</td><td>58</td></tr><tr><td>Detection TPR</td><td>0.82</td><td>0.87</td><td>0.89</td><td>0.86</td></tr><tr><td>Detection FPR</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.01</td></tr><tr><td>CDR</td><td>0.34</td><td>0.96</td><td>0.92</td><td>0.91</td></tr><tr><td>ADD</td><td>54.08</td><td>28.47</td><td>45.50</td><td>61.91</td></tr></table>

![](images/396b9ed119f4fb665b41e83155bd59f5238ddc45af5dbca04a82471ae97eca4d.jpg)  
Figure 4: Radar chart with TPR values evaluated on $\mathrm { T E P _ { R i c k e r } }$ for various faults. Faults are numbered on the circle, the distance between 0 and the points represents TPR values; the points in 0 represent undetected faults. FPR values are omitted from the chart since they all are below the 0.05 threshold.

Table 2: Aggregated detection and diagnosis metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$ for various faults.

<table><tr><td></td><td>Untrained model</td><td>Easy faults</td><td>Difficult faults</td><td>All faults (ours)</td></tr><tr><td>Detection TPR</td><td>0.62</td><td>0.68</td><td>0.66</td><td>0.87</td></tr><tr><td>Detection FPR</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td></tr><tr><td>CDR</td><td>0.83</td><td>0.87</td><td>0.95</td><td>0.96</td></tr><tr><td>ADD</td><td>67.36</td><td>150.14</td><td>124.86</td><td>28.47</td></tr></table>

Table 3: Results of ablation study. Clustering metrics evaluated on $\mathrm { T E P _ { R i e t h } }$

<table><tr><td></td><td>ACC</td><td>ARI</td><td>NMI</td></tr><tr><td>Only reconstruction task</td><td>0.632</td><td>0.546</td><td>0.711</td></tr><tr><td>Only contrastive learning</td><td>0.730</td><td>0.531</td><td>0.804</td></tr><tr><td>Both tasks</td><td>0.780</td><td>0.697</td><td>0.838</td></tr><tr><td>Both tasks with naive SCAN</td><td>0.756</td><td>0.659</td><td>0.812</td></tr><tr><td>Ours</td><td>0.785</td><td>0.703</td><td>0.846</td></tr></table>

![](images/f4f63f43c0ce7866ff5c8aa28934850fe20a1210d8abd15a919680944455153b.jpg)  
(a)

![](images/4ded36385521f89af9092c018f80d7257a99751381d30ce13ff536edaf06ccca.jpg)  
(b)

![](images/aa641aac1aa538f0ea8d43db7e0a61d1892a2aedcb8a96085f5080ca8d8e08da.jpg)  
(c)

![](images/58bbb48fce0f3b57f4aafc565ce48067987332e136b47d092e89938723ae900c.jpg)  
(d)

![](images/ca3ff1880cf83cad8d64dccd034149b3d93da729a256eb23ba7f8a1cb39f034f.jpg)  
(e)

Figure 5: Visualization of the embedding space with t-SNE on $\mathrm { T E P _ { R i e t h } } .$ . The colors correspond to the ground truth labels. Left to right: reconstruction task (a), contrastive learning (b), both tasks (c), both tasks with naive SCAN (d), ours (e). Our model produces denser clusters and decreases the size of the large mixed group. The red ellipses highlight the key differences between the figures.  
![](images/a0ce49b6a7ef7bbed5ccee3c410fbb74a964f614e3065a88c462e7d88554537d.jpg)  
Figure 6: Comparison of feature extractor pretraining with various faults for $\mathrm { T E P } _ { \mathrm { R i c k e r } } .$ The colors correspond to the ground truth labels. Left to right: untrained model, easy faults, difficult faults, all faults (ours). Pretraining on all faults decreases the size of the large mixed group.

## 7 Baselines

We compared SensorSCAN with unsupervised models of different types to cover both traditional and latest data-driven methods. We assume that an accurate unsupervised model can detect all faults by dividing samples into separate groups, so we set the number of clusters equal to the number of classes. For more detail on the training setup and the hyperparameter tuning for SensorSCAN, see B.

PCA. We consider PCA with k-means clustering as a simple model with cheap computational costs; the approach was proposed in [25]. We perform PCA to represent the samples in a 25-dimensional embedding space and then apply k-means to determine the clusters of samples. The dimensionality of the embedding space is selected according to the largest TPR on the training set.

ST-CatGAN. ST-CatGAN is a deep learning FDD method based on a convolutional neural network with adversarial training. The values of hyperparameters were taken from [58] and adapted to the TEP datasets. Samples of the shape (100, D) where D is the number of sensors, are converted with the short-time Fourier transformation (STFT) into multi-channel matrices of the shape (16, 16, D). The window size of the STFT is 30, the step size is 23. The values of the hyperparameters are chosen according to the largest TPR values on the training set.

![](images/4178d95abc398705b6bdd5405c443d6393526dea96fee8f30334ab8e65fceb3f.jpg)  
Figure 7: Comparison of dimensionality reduction algorithms on $\mathrm { T E P _ { R i e t h } }$ (here, we use the full dataset to improve the quality of visualisation and to demonstrate the reasonableness of our choice). The colors correspond to the ground truth labels. Left to right: PCA, t-SNE, UMAP. PCA failed to preserve the data distribution, while UMAP has spread the samples of the same process state far apart (as indicated with red circles).

ConvAE. The approach consists of three steps: feature extraction by convolutional stacked autoencoders, converting the obtained embeddings into 2-dimensional vectors with the t-SNE algorithm, and performing k-means to determine the clusters of samples. The method was proposed in [12] as a state-of-the-art unsupervised FDD for TEP; we borrow the hyperparameter values from the paper.

We also consider the following supervised model to compare with fine-tuned SensorSCAN.

GRU. The Gated Recurrent Unit (GRU) was adapted to FDD in [50]. Following the paper, we regard the model as a state-of-the-art supervised FDD for TEP. The model we use is referred to as ”GRU type:2”, with the values of hyperparameters proposed in the paper.

## 8 Experimental results

## 8.1 Unsupervised setting

The unsupervised setting consists of two steps – namely, clustering of samples which is followed by label matching. The clustering metrics were calculated based on cluster indices without the use of label matching; the results are reported in Table 4 and Table 5 for $\mathrm { T E P _ { R i e t h } }$ and $\mathrm { T E P _ { R i c k e r } } ,$ , respectively. PCA and ST-CatGAN showed the worst results compared to their competitors across all the clustering metrics, while SensorSCAN yielded the best results.

Table 4: Clustering metrics evaluated on $\mathrm { T E P _ { R i e t h } }$

<table><tr><td></td><td>ACC</td><td>ARI</td><td>NMI</td></tr><tr><td>PCA</td><td>0.274</td><td>0.110</td><td>0.363</td></tr><tr><td>ST-CatGAN</td><td>0.175</td><td>0.113</td><td>0.222</td></tr><tr><td>ConvAE</td><td>0.402</td><td>0.124</td><td>0.467</td></tr><tr><td>Ours</td><td>0.785</td><td>0.703</td><td>0.846</td></tr></table>

Table 5: Clustering metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$

<table><tr><td></td><td>ACC</td><td>ARI</td><td>NMI</td></tr><tr><td>PCA</td><td>0.352</td><td>0.132</td><td>0.448</td></tr><tr><td>ST-CatGAN</td><td>0.302</td><td>0.129</td><td>0.361</td></tr><tr><td>ConvAE</td><td>0.523</td><td>0.239</td><td>0.573</td></tr><tr><td>Ours</td><td>0.736</td><td>0.481</td><td>0.850</td></tr></table>

The FDD metrics are calculated using the label matching output – see Figure 8 and Tables 6, 7. The FPR values are omitted from the chart since they all are below the 0.05 threshold (the exact TPR and FPR

values can be found in C). We observe that PCA is able to effectively classify the normal type to prevent false alarms, but at the same time the majority of faults (12 out of 20 and 17 out of 28, respectively) remain undetected. ConvAE significantly outperformed PCA: it is able to detect all the faults that can be detected by PCA, including faults 5, 7, 14 in $\mathrm { T E P _ { R i e t h } }$ and faults 5, 10, 11, 24, 25, 26, 27 in $\mathrm { T E P _ { R i c k e r } }$ (the description of faults can be found in Table 15). The majoriy of the detected faults has high TPR values. However, there still remain undetected faults (8 out of 20 and 11 out of 28, correspondingly). SensorSCAN is able to detect almost all the faults except faults 9 and 15 in $\mathrm { T E P _ { R i e t h } }$ and faults 9, 15, 16, and 21 in $\mathrm { T E P } _ { \mathrm { R i c k e r } } .$ . Note that these faults remain undetected by all the unsupervised models, so we term them hard-to-detect faults; in the next subsection, we show that SensorSCAN is able to detect some of them with the help of fine-tuning. All the models are operating with almost zero Detection FPR; thus, they can be applied in real cases since they do not cause excessive numbers of false alarms. SensorSCAN showed the best results in the aggregated detection and diagnosis metrics, Detection TPR and Detection CDR, detecting 90% of faults in $\mathrm { T E P _ { R i e t h } }$ and 87% in $\mathrm { T E P _ { R i c k e r } }$ . The detected faults were correctly diagnosed almost always (96% on both datasets). The average detection delay was significantly shorter with respect to the other models, amounting to 27.7 and 28.47 time stamps, respectively. The next fastest model was ConvAE, which detects faults within about 49.95 and 52.28 time stamps, respectively.

![](images/9a23c5e59a6ca851fadb8a71405a73d967d84a5a81ffd4ba43f95aaa1b2484d9.jpg)

![](images/3e80676415541011b59bf85658a5bb7b6038415943010fce452e097c4511ce77.jpg)  
Figure 8: Radar chart with TPR values evaluated on $\mathrm { T E P _ { R i e t h } }$ (left) and $\mathrm { T E P _ { R i c k e r } }$ (right) in the unsupervised setting. Faults are enumerated on the circle, the distance between 0 and the points represents TPR values, the points in 0 represent undetected faults.

Table 6: Aggregated detection and diagnosis metrics evaluated on $\mathrm { T E P _ { R i e t h } }$ in the unsupervised setting.

<table><tr><td></td><td>PCA</td><td>ST-CatGAN</td><td>ConvAE</td><td>Ours</td></tr><tr><td>Detection TPR</td><td>0.36</td><td>0.30</td><td>0.48</td><td>0.84</td></tr><tr><td>Detection FPR</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>CDR</td><td>0.79</td><td>0.32</td><td>0.93</td><td>0.92</td></tr><tr><td>ADD</td><td>113.95</td><td>102.63</td><td>49.95</td><td>5.21</td></tr></table>

Table 7: Aggregated detection and diagnosis metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$ in the unsupervised setting.

<table><tr><td></td><td>PCA</td><td>ST-CatGAN</td><td>ConvAE</td><td>Ours</td></tr><tr><td>Detection TPR</td><td>0.36</td><td>0.36</td><td>0.64</td><td>0.87</td></tr><tr><td>Detection FPR</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>CDR</td><td>0.95</td><td>0.89</td><td>0.89</td><td>0.96</td></tr><tr><td>ADD</td><td>111.49</td><td>135.04</td><td>52.28</td><td>28.47</td></tr></table>

In order to evaluate the ability of the unsupervised models to diagnose faults, we conduct a 2- dimensional t-SNE visualization where the dots represent embeddings and are colored in accordance with the ground truth labels. First, let us consider $\mathrm { T E P _ { R i e t h } }$ in Figure 9. We observe that PCA is able to effectively distinguish about 6 faults, while ConvAE increases this number up to 11. All the other faults are referred to a large, mixed group. In contrast, SensorSCAN not only distinguishes a larger number of faults, but it also constructs groups that almost entirely consist of only two faults. This potentially means that such a group can be effectively split into two by fine-tuning. Next, let us consider $\mathrm { T E P _ { R i c k e r } }$ in Figure 10. Again, we observe that SensorSCAN distinguishes a larger number of faults with respect to the competitors, while also constructing a mixed group with only two faults (the orange-blue group below the main mixed group).

![](images/41d490fd78f41d87ee27d3b0c66a21e5c7f6239879d04a897381d7da608bc197.jpg)  
Figure 9: Model comparison on $\mathrm { T E P _ { R i e t h } ; }$ visualization of the embedding space with t-SNE. The colors correspond to the ground truth labels. Left to right: PCA, ST-CatGAN, ConvAE, ours. Our model is able to separate most faults from the normal state cluster (blue blob), while the other models fail to separate most of the faults from the normal state and from each other.

![](images/e379d992670a6e9ae85215f3d26d96ce7215dd18a4f0acd7e2495194794adc0a.jpg)  
Figure 10: Model comparison on $\mathrm { T E P _ { R i c k e r } ; }$ visualization of the embedding space with t-SNE. The colors correspond to the ground truth labels. Left to right: PCA, ST-CatGAN, ConvAE, ours. We can see that our model produces denser clusters as well as decreases the size of the large mixed group.

## 8.2 Semi-supervised setting

We performed experiments in a setting that we call semi-supervised, in order to show the fine-tuning ability of SensorSCAN. We compared the results with the model based on Gated Recurrent Units (GRU) that was reported to show superior results on TEP in a supervised setting [50].

First, we fine-tuned pretrained SensorSCAN and trained GRU on the dataset with a single labeled run to show that self-supervised pretraining benefits our model’s performance in the semi-supervised setting $( \mathrm { S e n s o r S C A N _ { \mathrm { s i n g l e } } }$ and $\mathrm { G R U _ { \mathrm { s i n g l e } } }$ , respectively – see Tables 8, 9). Second, we trained GRU on the full labeled dataset $( \mathrm { G R U _ { f u l l } } )$ and compared it with $\mathrm { S e n s o r S C A N _ { s i n g l e } }$ to illustrate that self-supervised pretraining and fine-tuning on a single run helps achieve performance close to a SOTA model trained on the full dataset.

Remarkably, GRU with a limited number of labeled samples is disadvantaged over our approach since our model has access to all sensor data (for the semi-supervised pretraining part), whereas GRU only uses the labeled sensor data. We can observe that SensorSCAN is better than the single-run GRU in detecting most faults and is at least as good in the other faults. In addition, our model’s values are close to those of the all-runs GRU, which is valid for both datasets. SensorSCAN showed the highest TPR in 14 faults, while the all-runs GRU proved superior in 15 faults in $\mathrm { T E P _ { R i e t h } }$ (and 17 vs. 26 in $\mathrm { T E P _ { R i c k e r } } )$ Important observations are to be made on the ability of the models to handle the hard-to-detect faults. Notably, only the all-runs GRU successfully detected fault 9 in $\mathrm { T E P _ { R i e t h } }$ , while fault 15 persisted across all the models: the single-run GRU did not detect it, whereas the others yielded inadequately high FPR (0.10 for SensorSCAN, 0.11 for the all-runs GRU) – assuming that FPR greater than 0.05 is impractical in real cases due to unacceptably frequent false alarms. In ${ \bar { \mathrm { T E P } } } _ { \mathrm { R i c k e r } } ,$ fault 9 was now detected by all the models, with the all-runs GRU showing the highest TPR. Fault 15 was successfully detected by the all-runs GRU, while the others showed inadequately low TPR (0.01 for the single-run GRU and 0.04 for SensorSCAN). Fault 16 was detected by SensorSCAN and the all-runs GRU, while the single-run GRU produced the highest TPR. Fault 21 remained almost undetected by all the models: only SensorSCAN was able to detect a few samples (with the TPR of 0.04). The hard-to-detect faults significantly increase Detection FPR, thus making almost all of the models trained in the semi-supervised setting impractical in real-world cases. Only the all-runs GRU resulted with the threshold Detection FPR value of 0.05 on $\mathrm { T E P } _ { \mathrm { R i c k e r } } .$ . Nevertheless, we can conclude that fine-tuning helps SensorSCAN not only achieve results similar to the all-runs GRU but also to cope with the hard-to-detect faults 9 and 16 in $\mathrm { T E P _ { R i c k e r } }$ . In real cases, we can decrease Detection FPR of SensorSCAN to acceptable values by turning off the detection of the remaining hard-to-detect faults and treating them with other methods – for example, the methods based on expert knowledge.

Table 8: FDD metrics evaluated on $\mathrm { T E P _ { R i e t h } }$ in the semi-supervised setting. Top down: TPR/FPR for each fault, followed by aggregated detection and diagnosis metrics. The largest TPR values are highlighted on the condition that FPR is no greater than 0.05. The confidence interval is evaluated on randomly selected runs for training. SensorSCAN is fine-tuned on a single run. Results of GRU trained on a single run (left column) and on all runs (right column) are presented for comparison. Our fine-tuned model is either on a par with or is better than the single-run GRU and shows the best results on the number of faults that is similar to the all-runs GRU.

<table><tr><td rowspan="2"></td><td colspan="2">GRUsingle</td><td colspan="2">SensorSCANsingle</td><td colspan="2">GRUfull</td></tr><tr><td>TPR</td><td>FPR</td><td>TPR</td><td>FPR</td><td>TPR</td><td>FPR</td></tr><tr><td>Fault 1</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 2</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 3</td><td> $0.15 \pm 0.03$ </td><td>0.07</td><td> $\mathbf{0.27} \pm 0.04$ </td><td>0.05</td><td>0.79</td><td>0.13</td></tr><tr><td>Fault 4</td><td> $0.97 \pm 0.01$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 5</td><td> $0.93 \pm 0.02$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 6</td><td> $0.85 \pm 0.03$ </td><td>0.00</td><td> $0.97 \pm 0.01$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 7</td><td> $0.99 \pm 0.01$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 8</td><td> $0.60 \pm 0.05$ </td><td>0.00</td><td> $\mathbf{0.90} \pm 0.02$ </td><td>0.00</td><td>0.89</td><td>0.00</td></tr><tr><td>Fault 9</td><td> $\mathbf{0.03} \pm 0.01$ </td><td>0.02</td><td> $0.10 \pm 0.04$ </td><td>0.08</td><td>0.00</td><td>0.00</td></tr><tr><td>Fault 10</td><td> $0.47 \pm 0.19$ </td><td>0.01</td><td> $\mathbf{0.68} \pm 0.15$ </td><td>0.00</td><td>0.56</td><td>0.00</td></tr><tr><td>Fault 11</td><td> $0.99 \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 12</td><td> $0.83 \pm 0.03$ </td><td>0.00</td><td> $\mathbf{0.96} \pm 0.01$ </td><td>0.00</td><td>0.88</td><td>0.00</td></tr><tr><td>Fault 13</td><td> $0.30 \pm 0.09$ </td><td>0.00</td><td> $\mathbf{0.80} \pm 0.03$ </td><td>0.00</td><td>0.70</td><td>0.00</td></tr><tr><td>Fault 14</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.01</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 15</td><td> $0.04 \pm 0.01$ </td><td>0.03</td><td> $0.06 \pm 0.03$ </td><td>0.04</td><td>0.00</td><td>0.00</td></tr><tr><td>Fault 16</td><td> $0.32 \pm 0.15$ </td><td>0.08</td><td> $\mathbf{0.60} \pm 0.17$ </td><td>0.01</td><td>0.25</td><td>0.00</td></tr><tr><td>Fault 17</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 18</td><td> $0.83 \pm 0.02$ </td><td>0.00</td><td> $0.72 \pm 0.05$ </td><td>0.00</td><td> $\mathbf{0.98}$ </td><td>0.00</td></tr><tr><td>Fault 19</td><td> $0.99 \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Fault 20</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00} \pm 0.00$ </td><td>0.00</td><td> $\mathbf{1.00}$ </td><td>0.00</td></tr><tr><td>Detection TPR</td><td colspan="2"> $0.85 \pm 0.01$ </td><td colspan="2"> $\mathbf{0.89} \pm 0.01$ </td><td colspan="2">0.87</td></tr><tr><td>Detection FPR</td><td colspan="2"> $0.21 \pm 0.06$ </td><td colspan="2"> $0.19 \pm 0.02$ </td><td colspan="2"> $\mathbf{0.16}$ </td></tr><tr><td>CDR</td><td colspan="2"> $0.84 \pm 0.01$ </td><td colspan="2"> $0.89 \pm 0.02$ </td><td colspan="2"> $\mathbf{0.92}$ </td></tr><tr><td>ADD</td><td colspan="2"> $30.78 \pm 11.58$ </td><td colspan="2"> $17.46 \pm 3.08$ </td><td colspan="2"> $\mathbf{16.70}$ </td></tr></table>

## 9 Discussion

The experimental results presented in Figure 8 demonstrate the ability of our model to outperform the latest deep learning data-driven FDD models in the unsupervised setting. Note that our model is able to successfully detect the faults that remain undetected by other models. It becomes possible thanks to the powerful feature extractor that can retrieve inherent patterns from the process behavior and map them as separate groups in the latent space. Subsequent deep clustering increases the density of such groups, which forces the feature extractor not only to map the process patterns in the latent space but also to distinguish between them, making the diagnostics more accurate.

Table 9: FDD metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$ in the semi-supervised setting. Top down: TPR/FPR for each fault, followed by aggregated detection and diagnosis metrics. The largest TPR values are highlighted on the condition that FPR is no greater than 0.05. The confidence interval is evaluated on randomly selected runs for training. SensorSCAN is fine-tuned on a single run. Results of GRU trained on a single run (left column) and on all runs (right column) are presented for comparison. Our fine-tuned model is either on par with or is better than the single-run GRU and shows the best results on the number of faults that is similar to the all-runs GRU.

<table><tr><td rowspan="2"></td><td colspan="2">GRUsingle</td><td colspan="2">SensorSCANsingle</td><td colspan="2">GRUfull</td></tr><tr><td>TPR</td><td>FPR</td><td>TPR</td><td>FPR</td><td>TPR</td><td>FPR</td></tr><tr><td>Fault 1</td><td>0.99 ± 0.00</td><td>0.00</td><td>1.00 ± 0.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>Fault 2</td><td>0.98 ± 0.00</td><td>0.00</td><td>1.00 ± 0.00</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 3</td><td>0.82 ± 0.12</td><td>0.00</td><td>0.97 ± 0.00</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 4</td><td>1.00 ± 0.00</td><td>0.00</td><td>1.00 ± 0.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>Fault 5</td><td>0.94 ± 0.01</td><td>0.01</td><td>0.99 ± 0.00</td><td>0.01</td><td>1.00</td><td>0.00</td></tr><tr><td>Fault 6</td><td>0.99 ± 0.00</td><td>0.00</td><td>1.00 ± 0.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>Fault 7</td><td>1.00 ± 0.00</td><td>0.00</td><td>1.00 ± 0.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>Fault 8</td><td>0.87 ± 0.01</td><td>0.00</td><td>0.97 ± 0.01</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 9</td><td>0.25 ± 0.20</td><td>0.01</td><td>0.54 ± 0.10</td><td>0.00</td><td>0.73</td><td>0.00</td></tr><tr><td>Fault 10</td><td>0.95 ± 0.01</td><td>0.01</td><td>0.98 ± 0.00</td><td>0.00</td><td>0.98</td><td>0.00</td></tr><tr><td>Fault 11</td><td>0.98 ± 0.01</td><td>0.00</td><td>0.99 ± 0.00</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 12</td><td>0.95 ± 0.01</td><td>0.01</td><td>0.99 ± 0.00</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 13</td><td>0.85 ± 0.01</td><td>0.00</td><td>0.97 ± 0.00</td><td>0.00</td><td>0.97</td><td>0.00</td></tr><tr><td>Fault 14</td><td>0.99 ± 0.00</td><td>0.00</td><td>1.00 ± 0.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>Fault 15</td><td>0.00 ± 0.01</td><td>0.02</td><td>0.06 ± 0.02</td><td>0.03</td><td>0.65</td><td>0.05</td></tr><tr><td>Fault 16</td><td>0.02 ± 0.02</td><td>0.01</td><td>0.82 ± 0.04</td><td>0.01</td><td>0.95</td><td>0.01</td></tr><tr><td>Fault 17</td><td>0.97 ± 0.00</td><td>0.00</td><td>0.98 ± 0.00</td><td>0.00</td><td>0.98</td><td>0.00</td></tr><tr><td>Fault 18</td><td>0.96 ± 0.00</td><td>0.00</td><td>0.96 ± 0.00</td><td>0.00</td><td>0.97</td><td>0.00</td></tr><tr><td>Fault 19</td><td>0.60 ± 0.16</td><td>0.00</td><td>0.89 ± 0.08</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 20</td><td>0.97 ± 0.00</td><td>0.00</td><td>0.97 ± 0.00</td><td>0.01</td><td>0.97</td><td>0.00</td></tr><tr><td>Fault 21</td><td>0.01 ± 0.02</td><td>0.00</td><td>0.06 ± 0.01</td><td>0.04</td><td>0.00</td><td>0.00</td></tr><tr><td>Fault 22</td><td>0.25 ± 0.11</td><td>0.02</td><td>0.52 ± 0.13</td><td>0.01</td><td>0.74</td><td>0.00</td></tr><tr><td>Fault 23</td><td>0.03 ± 0.04</td><td>0.00</td><td>0.98 ± 0.00</td><td>0.00</td><td>0.98</td><td>0.01</td></tr><tr><td>Fault 24</td><td>0.96 ± 0.00</td><td>0.00</td><td>0.99 ± 0.00</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 25</td><td>0.95 ± 0.00</td><td>0.00</td><td>0.99 ± 0.00</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 26</td><td>0.89 ± 0.06</td><td>0.01</td><td>0.98 ± 0.00</td><td>0.00</td><td>0.98</td><td>0.00</td></tr><tr><td>Fault 27</td><td>0.97 ± 0.01</td><td>0.00</td><td>0.99 ± 0.00</td><td>0.00</td><td>0.99</td><td>0.00</td></tr><tr><td>Fault 28</td><td>0.28 ± 0.13</td><td>0.06</td><td>0.96 ± 0.00</td><td>0.01</td><td>0.97</td><td>0.01</td></tr><tr><td>Detection TPR</td><td>0.83 ± 0.02</td><td></td><td>0.92 ± 0.00</td><td></td><td>0.94</td><td></td></tr><tr><td>Detection FPR</td><td>0.11 ± 0.05</td><td></td><td>0.09 ± 0.02</td><td></td><td>0.05</td><td></td></tr><tr><td>CDR</td><td>0.88 ± 0.01</td><td></td><td>0.95 ± 0.00</td><td></td><td>0.97</td><td></td></tr><tr><td>ADD</td><td>59.76 ± 18.49</td><td></td><td>35.47 ± 1.59</td><td></td><td>31.00</td><td></td></tr></table>

As can be seen in Table 6 and Table 7, SensorSCAN detects faults much faster in the unsupervised setting in terms of ADD. A large ADD may occur in the other models because of smooth transitions between process states in the latent space. Samples that are uniformly distant from the cluster centers increase the uncertainty of the model and do not allow timely fault detection. We showed that the deep clustering approach used in our model makes groups of samples denser, thus reducing the uncertainty of the model, which subsequently leads to an abrupt transition between the predicted process states, followed by undelayed detection.

Considering the application of FDD models in real-world industrial cases, it is important to assess the computational time and the memory costs. Deep-learning methods are more expensive compared to classical methods like PCA, but this is mainly due to the training process, since it requires performing a complicated backpropagation procedure and implementing optimization steps over a large number of samples with stochastic gradient descent (SGD) or similar methods.

The time complexity of the training process of deep learning models depends on many factors, such as the size of the dataset, the size of mini-batches, the learning rate, the number of steps, and the optimization algorithm. Let us look at the difficulty of training such deep neural networks if the number of sensors significantly increases, which is a realistic scenario for large-scale complex processes. The number of sensors determines only the number of neurons in the input layer in a deep neural network. That is, increasing the number of sensors does not substantially affect the time complexity of a single optimization step. For example, increasing the number of sensors from 52 to 1000 in SensorSCAN leads to the increase in the execution time of one optimization step with the mini-batch size of 512 from $0 . 4 6 \pm 0 . 0 3$ seconds to $0 . 4 9 \pm 0 . 0 4$ seconds.

Time costs in the inference are low since the samples are received in the one-by-one fashion with an intermediate delay. For example, prediction of the process state of a single sample by SensorSCAN takes approximately 0.01 seconds on a laptop with the 2.7 GHz Dual-Core Intel Core i5 CPU, while samples arrive every 3 minutes in TEP. As a result, we can process each sample without any delay.

Memory costs depend on the number of trainable parameters; the exact values (in megabytes) used in our experiments are represented in Table 10.

Table 10: Memory costs of models on TEP dataset.

<table><tr><td></td><td>MB</td></tr><tr><td>PCA</td><td>0.41</td></tr><tr><td>ST-CatGAN</td><td>0.82</td></tr><tr><td>ConvAE</td><td>4.97</td></tr><tr><td>Ours</td><td>2.38</td></tr></table>

Summing up, we conclude that usage of deep-learning FDD models is possible in real-time monitoring even on a laptop with CPU cores.

## 10 Conclusion

In this paper, we proposed an unsupervised FDD model based on self-supervised learning and deep clustering. The predicted faults were obtained from cluster indices by means of label matching procedure, a simple heuristic technique that in real-world industrial setting can be performed by experts. The model was evaluated on the datasets simulated by Tennessee Eastman Process benchmarks, $\mathrm { T E P _ { R i e t h } }$ [87] and $\mathrm { T E P _ { R i c k e r } } \ [ 9 2 ]$ . We evaluated our model using a wide range of metrics assessing the ability to cluster faults in the latent space, the ability to detect different faults, and the speed of detection. The empirical evaluation showed that our method outperforms other unsupervised approaches on both datasets due to the ability of the feature extractor to represent samples of sensor data in the latent space while preserving important high-level properties and the ability of deep clustering to produce dense clusters in the latent space.

Data with insufficient labeling of faults or even without labeling is common in the real-world industrial setting. Most of the existing FDD approaches are supervised [13, 14], which basically ignores this fact and makes their application impractical. SensorSCAN is based on SSL and deep clustering, which enables training on unlabeled data and taking into account the expected number of faults, while fine-tuning allows us to take advantage of the labeled examples in the data. Moreover, if the expected number of faults is changed or new faults are incorporated, then the feature extractor can be further fine-tuned without needing to train it from scratch. Thus, our method not only advances the state-ofthe-art metrics on the chosen datasets but also covers the machine learning settings used in real-world monitoring and control of industrial processes, which is essential for adapting data-driven models to the production stage.

We showed that our model can be applied in the case of unlabeled data, which is a typical scenario in real industrial processes. However, we observed that some faults consistently resist detection in the absence of labels; to address this obstacle, we proposed semi-supervised fine-tuning, the technique that helps detect difficult faults using very few labeled data.

We envisage further research in two directions. The first is to explore other SSL techniques, including the methods that incorporate knowledge from the domains of chemistry and physics. The second is the analysis of deep semi-supervised methods for performing fine-tuning. We think that the promising results recently achieved by semi-supervised methods for time series classification [96, 97, 98] can be adapted for FDD on sensor data.

## Acknowledgements

The work on the Related work and the Ablation sections was supported by the Russian Science Foundation under grant 22-11-00323 and performed at the National Research University Higher School of Economics (Moscow, Russia).

## References

[1] V. Venkatasubramanian, R. Rengaswamy, K. Yin, S. N. Kavuri, A review of process fault detection and diagnosis: Part i: Quantitative model-based methods, Computers & Chemical Engineering 27 (3) (2003) 293–311. doi:https://doi.org/10.1016/S0098-1354(02) 00160-6.

[2] X. Ji, A. Vedaldi, J. Henriques, Invariant information clustering for unsupervised image classification and segmentation, in: 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9864–9873. doi:https://doi.org/10.1109/ICCV.2019.00996.

[3] W. Hu, T. Miyato, S. Tokui, E. Matsumoto, M. Sugiyama, Learning discrete representations via information maximizing self-augmented training, in: Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17, JMLR.org, 2017, p. 1558–1567.

[4] J. Huang, S. Gong, X. Zhu, Deep semantic clustering by partition confidence maximisation, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 8846–8855. doi:https://doi.org/10.1109/CVPR42600.2020.00887.

[5] Y. Li, P. Hu, Z. Liu, D. Peng, J. T. Zhou, X. Peng, Contrastive clustering, Proceedings of the AAAI Conference on Artificial Intelligence 35 (10) (2021) 8547–8555. doi:https://doi. org/10.48550/arXiv.2009.09687.

[6] C. Niu, J. Zhang, G. Wang, J. Liang, Gatcluster: Self-supervised gaussian-attention network for image clustering, in: A. Vedaldi, H. Bischof, T. Brox, J.-M. Frahm (Eds.), Computer Vision – ECCV 2020, Springer International Publishing, Cham, 2020, pp. 735–751.

[7] S. Deldari, D. V. Smith, H. Xue, F. D. Salim, Time series change point detection with selfsupervised contrastive predictive coding, in: Proceedings of the Web Conference 2021, WWW ’21, Association for Computing Machinery, New York, NY, USA, 2021, p. 3124–3135. doi: https://doi.org/10.1145/3442381.3449903.

[8] M. N. Mohsenvand, M. R. Izadi, P. Maes, Contrastive representation learning for electroencephalogram classification, in: E. Alsentzer, M. B. A. McDermott, F. Falck, S. K. Sarkar, S. Roy, S. L. Hyland (Eds.), Proceedings of the Machine Learning for Health NeurIPS Workshop, Vol. 136 of Proceedings of Machine Learning Research, PMLR, 2020, pp. 238–253.

[9] G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, C. Eickhoff, A transformer-based framework for multivariate time series representation learning, in: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’21, Association for Computing Machinery, New York, NY, USA, 2021, p. 2114–2124.

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, Curran Associates Inc., Red Hook, NY, USA, 2017, p. 6000–6010.

[11] W. Van Gansbeke, S. Vandenhende, S. Georgoulis, M. Proesmans, L. Van Gool, Scan: Learning to classify images without labels, in: A. Vedaldi, H. Bischof, T. Brox, J.-M. Frahm (Eds.), Computer Vision – ECCV 2020, Springer International Publishing, Cham, 2020, pp. 268–285.

[12] S. Zheng, J. Zhao, A new unsupervised data mining method based on the stacked autoencoder for chemical process fault diagnosis, Computers & Chemical Engineering 135 (2020) 106755. doi:https://doi.org/10.1016/j.compchemeng.2020.106755.

[13] Y. Lei, B. Yang, X. Jiang, F. Jia, N. Li, A. K. Nandi, Applications of machine learning to machine fault diagnosis: A review and roadmap, Mechanical Systems and Signal Processing 138 (2020) 106587. doi:https://doi.org/10.1016/j.ymssp.2019.106587.

[14] S. A. A. Taqvi, H. Zabiri, L. D. Tufa, F. Uddin, S. A. Fatima, A. S. Maulud, A review on datadriven learning approaches for fault detection and diagnosis in chemical processes, ChemBioEng Reviews 8 (3) (2021) 239–259. doi:https://doi.org/10.1002/cben.202000027.

[15] X. Bi, R. Qin, D. Wu, S. Zheng, J. Zhao, One step forward for smart chemical process fault detection and diagnosis, Computers I& Chemical Engineering 164 (2022) 107884. doi:https: //doi.org/10.1016/j.compchemeng.2022.107884.

[16] J. Van Impe, G. Gins, An extensive reference dataset for fault detection and identification in batch processes, Chemometrics and Intelligent Laboratory Systems 148 (2015) 20–31. doi: https://doi.org/10.1016/j.chemolab.2015.08.019.

[17] S. Goldrick, A. ¸Stefan, D. Lovett, G. Montague, B. Lennox, The development of an industrialscale fed-batch fermentation simulation, Journal of Biotechnology 193 (2015) 70–82. doi: https://doi.org/10.1016/j.jbiotec.2014.10.029.

[18] B. Bruijn, T. A. Nguyen, D. Bucur, K. Tei, Benchmark datasets for fault detection and classification in sensor data, in: Proceedings of the 5th International Confererence on Sensor Networks - SENSORNETS„ 2016, pp. 185–195. doi:https://doi.org/10.5220/ 0005637901850195.

[19] Z. Chen, S. X. Ding, K. Zhang, Z. Li, Z. Hu, Canonical correlation analysis-based fault detection methods with application to alumina evaporation process, Control Engineering Practice 46 (2016) 51–58. doi:https://doi.org/10.1016/j.conengprac.2015.10.006.

[20] Z. Chen, S. X. Ding, T. Peng, C. Yang, W. Gui, Fault detection for non-gaussian processes using generalized canonical correlation analysis and randomized algorithms, IEEE Transactions on Industrial Electronics 65 (2) (2018) 1559–1567. doi:https://doi.org/10.1109/TIE. 2017.2733501.

[21] X. Gao, J. Hou, An improved svm integrated gs-pca fault diagnosis approach of tennessee eastman process, Neurocomputing 174 (2016) 906–911. doi:https://doi.org/10.1016/j. neucom.2015.10.018.

[22] M. Noruzi Nashalji, M. Aliyari Shoorehdeli, M. Teshnehlab, Fault detection of the tennessee eastman process using improved pca and neural classifier, in: X.-Z. Gao, A. Gaspar-Cunha, M. Köppen, G. Schaefer, J. Wang (Eds.), Soft Computing in Industrial Applications, Springer Berlin Heidelberg, Berlin, Heidelberg, 2010, pp. 41–50. doi:https://doi.org/10.1007/ 978-3-642-11282-9\_5.

[23] C. Lau, K. Ghosh, M. Hussain, C. Che Hassan, Fault diagnosis of tennessee eastman process with multi-scale pca and anfis, Chemometrics and Intelligent Laboratory Systems 120 (2013) 1–14. doi:https://doi.org/10.1016/j.chemolab.2012.10.005.

[24] T. J. Rato, M. S. Reis, Fault detection in the tennessee eastman benchmark process using dynamic principal components analysis based on decorrelated residuals (dpca-dr), Chemometrics and Intelligent Laboratory Systems 125 (2013) 101–108. doi:https://doi.org/10.1016/j. chemolab.2013.04.002.

[25] Q. P. He, S. J. Qin, J. Wang, A new fault diagnosis method using fault directions in fisher discriminant analysis, AIChE Journal 51 (2) (2005) 555–571. arXiv:https:// aiche.onlinelibrary.wiley.com/doi/pdf/10.1002/aic.10325, doi:https://doi. org/10.1002/aic.10325.

[26] J. Dong, K. Zhang, Y. Huang, G. Li, K. Peng, Adaptive total pls based quality-relevant process monitoring with application to the tennessee eastman process, Neurocomputing 154 (2015) 77–85. doi:https://doi.org/10.1016/j.neucom.2014.12.017.

[27] C. Hu, Z. Xu, X. Kong, J. Luo, Recursive-cpls-based quality-relevant and process-relevant fault monitoring with application to the tennessee eastman process, IEEE Access 7 (2019) 128746–128757. doi:https://doi.org/10.1109/ACCESS.2019.2939163.

[28] G. Lee, C. Han, E. S. Yoon, Multiple-fault diagnosis of the tennessee eastman process based on system decomposition and dynamic pls, Industrial & Engineering Chemistry Research 43 (25) (2004) 8037–8048. doi:https://doi.org/10.1021/ie049624u.

[29] P. Peng, Y. Zhang, H. Wang, H. Zhang, Towards robust and understandable fault detection and diagnosis using denoising sparse autoencoder and smooth integrated gradients, ISA Transactions (2021). doi:https://doi.org/10.1016/j.isatra.2021.06.005.

[30] S. Yan, X. Yan, Using labeled autoencoder to supervise neural network combined with k-nearest neighbor for visual industrial process monitoring, Industrial & Engineering Chemistry Research 58 (23) (2019) 9952–9958. doi:https://doi.org/10.1021/acs.iecr.9b01325.

[31] S. Yan, X. Yan, Design teacher and supervised dual stacked auto-encoders for quality-relevant fault detection in industrial process, Applied Soft Computing 81 (2019) 105526. doi:https: //doi.org/10.1016/j.asoc.2019.105526.

[32] J. Yu, X. Yan, Multiscale intelligent fault detection system based on agglomerative hierarchical clustering using stacked denoising autoencoder with temporal information, Applied Soft Computing 95 (2020) 106525. doi:https://doi.org/10.1016/j.asoc.2020.106525.

[33] F. Cheng, Q. P. He, J. Zhao, A novel process monitoring approach based on variational recurrent autoencoder, Computers & Chemical Engineering 129 (2019) 106515. doi:https://doi. org/10.1016/j.compchemeng.2019.106515.

[34] C. Li, D. Zhao, S. Mu, W. Zhang, N. Shi, L. Li, Fault diagnosis for distillation process based on cnn–dae, Chinese Journal of Chemical Engineering 27 (3) (2019) 598–604. doi:https: //doi.org/10.1016/j.cjche.2018.12.021.

[35] H. Shao, H. Jiang, Y. Lin, X. Li, A novel method for intelligent fault diagnosis of rolling bearings using ensemble deep auto-encoders, Mechanical Systems and Signal Processing 102 (2018) 278–297. doi:https://doi.org/10.1016/j.ymssp.2017.09.026.

[36] X. Bi, J. Zhao, A novel orthogonal self-attentive variational autoencoder method for interpretable chemical process fault detection and identification, Process Safety and Environmental Protection 156 (2021) 581–597. doi:https://doi.org/10.1016/j.psep.2021.10.036.

[37] J. Yu, X. Yan, Deep unlstm network: Features with memory information extracted from unlabeled data and their application on industrial unsupervised industrial fault detection, Applied Soft Computing 108 (2021) 107382. doi:https://doi.org/10.1016/j.asoc.2021.107382.

[38] S. Zheng, J. Zhao, High-fidelity positive-unlabeled deep learning for semi-supervised fault detection of chemical processes, Process Safety and Environmental Protection 165 (2022) 191– 204. doi:https://doi.org/10.1016/j.psep.2022.06.058. URL https://www.sciencedirect.com/science/article/pii/S0957582022005900

[39] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, D. Pei, Robust anomaly detection for multivariate time series through stochastic recurrent neural network, in: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, Association for Computing Machinery, New York, NY, USA, 2019, p. 2828–2837. doi:https://doi.org/ 10.1145/3292500.3330672.

[40] Y. Han, N. Ding, Z. Geng, Z. Wang, C. Chu, An optimized long short-term memory network based fault diagnosis model for chemical processes, Journal of Process Control 92 (2020) 161–168. doi:https://doi.org/10.1016/j.jprocont.2020.06.005.

[41] D. Xie, L. Bai, A hierarchical deep neural network for fault diagnosis on tennessee-eastman process, in: 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), 2015, pp. 745–748. doi:https://doi.org/10.1109/ICMLA.2015.208.

[42] Y. Wang, Z. Xiao, G. Cao, A convolutional neural network method based on adam optimizer with power-exponential learning rate for bearing fault diagnosis, Journal of Vibroengineering 24 (4) (2022) 666–678. doi:10.21595/jve.2022.22271.

[43] Y. Liu, R. Young, B. Jafarpour, Long–short-term memory encoder–decoder with regularized hidden dynamics for fault detection in industrial processes, Journal of Process Control 124 (2023) 166–178. doi:https://doi.org/10.1016/j.jprocont.2023.01.015.

[44] A. Kovalenko, V. Pozdnyakov, I. Makarov, Graph neural networks with trainable adjacency matrices for fault diagnosis on multivariate sensor data (2022). arXiv:2210.11164.

[45] R. Verma, R. Yerolla, C. S. Besta, Deep learning-based fault detection in the tennessee eastman process, in: 2022 Second International Conference on Artificial Intelligence and Smart Energy (ICAIS), 2022, pp. 228–233. doi:10.1109/ICAIS53314.2022.9743021.

[46] H. Wu, J. Zhao, Deep convolutional neural network model based chemical process fault diagnosis, Computers & Chemical Engineering 115 (2018) 185–197. doi:https://doi.org/10.1016/ j.compchemeng.2018.04.009.

[47] Z. Zhang, J. Zhao, A deep belief network based fault diagnosis model for complex chemical processes, Computers & Chemical Engineering 107 (2017) 395–407, in honor of Professor Rafiqul Gani. doi:https://doi.org/10.1016/j.compchemeng.2017.02.041.

[48] Y. Wang, Z. Pan, X. Yuan, C. Yang, W. Gui, A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network, ISA Transactions 96 (2020) 457–467. doi:https://doi.org/10.1016/j.isatra.2019.07.001.

[49] P. Park, P. D. Marco, H. Shin, J. Bang, Fault detection and diagnosis using combined autoencoder and long short-term memory network, Sensors 19 (21) (2019). doi:https://doi.org/10. 3390/s19214612.

[50] I. Lomov, M. Lyubimov, I. Makarov, L. E. Zhukov, Fault detection in tennessee eastman process with temporal deep learning models, Journal of Industrial Information Integration 23 (2021) 100216. doi:https://doi.org/10.1016/j.jii.2021.100216.

[51] S. Bahrampour, B. Moshiri, K. Salahshoor, Weighted and constrained possibilistic c-means clustering for online fault detection and isolation, Applied Intelligence 35 (2) (2011) 269–284. doi:https://doi.org/10.1007/s10489-010-0219-2.

[52] J. Yu, A support vector clustering-based probabilistic method for unsupervised fault detection and classification of complex chemical processes using unlabeled data, AIChE Journal 59 (2) (2013) 407–419. doi:https://doi.org/10.1002/aic.13816.

[53] H. K. Alaei, K. Salahshoor, H. K. Alaei, A new integrated on-line fuzzy clustering and segmentation methodology with adaptive pca approach for process monitoring and fault detection and diagnosis, soft computing 17 (3) (2013) 345–362. doi:https://doi.org/10.1007/ s00500-012-0910-9.

[54] M. S. Escobar, H. Kaneko, K. Funatsu, On generative topographic mapping and graph theory combined approach for unsupervised non-linear data visualization and fault identification, Computers & Chemical Engineering 98 (2017) 113–127. doi:https://doi.org/10.1016/ j.compchemeng.2016.12.009.

[55] J. An, P. Ai, C. Liu, S. Xu, D. Liu, Deep clustering bearing fault diagnosis method based on local manifold learning of an autoencoded embedding, IEEE Access 9 (2021) 30154–30168. doi:10.1109/ACCESS.2021.3059459.

[56] J. T. Springenberg, Unsupervised and semi-supervised learning with categorical generative adversarial networks, arXiv preprint arXiv:1511.06390 (2015).

[57] H. Liu, J. Zhou, Y. Xu, Y. Zheng, X. Peng, W. Jiang, Unsupervised fault diagnosis of rolling bearings using a deep neural network based on generative adversarial networks, Neurocomputing 315 (2018) 412–424. doi:https://doi.org/10.1016/j.neucom.2018.07.034.

[58] H. Tao, P. Wang, Y. Chen, V. Stojanovic, H. Yang, An unsupervised fault diagnosis method for rolling bearing using stft and generative neural networks, Journal of the Franklin Institute 357 (11) (2020) 7286–7307. doi:https://doi.org/10.1016/j.jfranklin.2020.04.024.

[59] M. Munir, S. A. Siddiqui, A. Dengel, S. Ahmed, Deepant: A deep learning approach for unsupervised anomaly detection in time series, IEEE Access 7 (2019) 1991–2005. doi:https: //doi.org/10.1109/ACCESS.2018.2886457.

[60] J. Yu, X. Yan, Deep unlstm network: Features with memory information extracted from unlabeled data and their application on industrial unsupervised industrial fault detection, Applied Soft Computing 108 (2021) 107382. doi:https://doi.org/10.1016/j.asoc.2021.107382.

[61] X. Yang, D. Feng, Generative adversarial network based anomaly detection on the benchmark tennessee eastman process, in: 2019 5th International Conference on Control, Automation and Robotics (ICCAR), 2019, pp. 644–648. doi:https://doi.org/10.1109/ICCAR.2019. 8813415.

[62] D. Li, D. Chen, B. Jin, L. Shi, J. Goh, S.-K. Ng, Mad-gan: Multivariate anomaly detection for time series data with generative adversarial networks, in: I. V. Tetko, V. K˚urková, P. Karpov, F. Theis (Eds.), Artificial Neural Networks and Machine Learning – ICANN 2019: Text and

Time Series, Springer International Publishing, Cham, 2019, pp. 703–716. doi:https://doi. org/10.1007/978-3-030-30490-4\_56.

[63] R. Arunthavanathan, F. Khan, S. Ahmed, S. Imtiaz, R. Rusli, Fault detection and diagnosis in process system using artificial intelligence-based cognitive technique, Computers & Chemical Engineering 134 (2020) 106697. doi:https://doi.org/10.1016/j.compchemeng.2019. 106697.

[64] A. v. d. Oord, Y. Li, O. Vinyals, Representation learning with contrastive predictive coding (2018). doi:https://doi.org/10.48550/ARXIV.1807.03748.

[65] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, D. Amodei, Language models are few-shot learners, CoRR abs/2005.14165 (2020). arXiv:2005.14165, doi:https://doi. org/10.48550/arXiv.2005.14165.

[66] Z. Dai, H. Liu, Q. V. Le, M. Tan, Coatnet: Marrying convolution and attention for all data sizes, Advances in Neural Information Processing Systems 34 (2021) 3965–3977. doi:https: //doi.org/10.48550/arXiv.2106.04803.

[67] V. Fortuin, M. Hüser, F. Locatello, H. Strathmann, G. Rätsch, Som-vae: Interpretable discrete representation learning on time series, in: 7th International Conference on Learning Representations (ICLR), ICLR, 2019, pp. 1–18. doi:https://doi.org/10.48550/arXiv.1806.02199. URL https://openreview.net/pdf?id=rygjcsR9Y7

[68] T. Chen, S. Kornblith, M. Norouzi, G. E. Hinton, A simple framework for contrastive learning of visual representations, CoRR abs/2002.05709 (2020). arXiv:2002.05709, doi:https: //doi.org/10.48550/arXiv.2002.05709.

[69] J. Devlin, M. Chang, K. Lee, K. Toutanova, BERT: pre-training of deep bidirectional transformers for language understanding, CoRR abs/1810.04805 (2018). arXiv:1810.04805, doi:https: //doi.org/10.48550/arXiv.1810.04805.

[70] E. Eldele, M. Ragab, Z. Chen, M. Wu, C. K. Kwoh, X. Li, C. Guan, Time-series representation learning via temporal and contextual contrasting, in: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, 2021, pp. 2352–2359.

[71] P. Huang, Y. Huang, W. Wang, L. Wang, Deep embedding network for clustering, in: 2014 22nd International Conference on Pattern Recognition, 2014, pp. 1532–1537. doi:https: //doi.org/10.1109/ICPR.2014.272.

[72] J. Xie, R. Girshick, A. Farhadi, Unsupervised deep embedding for clustering analysis, in: M. F. Balcan, K. Q. Weinberger (Eds.), Proceedings of The 33rd International Conference on Machine Learning, Vol. 48 of Proceedings of Machine Learning Research, PMLR, New York, New York, USA, 2016, pp. 478–487.

[73] B. Yang, X. Fu, N. D. Sidiropoulos, M. Hong, Towards k-means-friendly spaces: Simultaneous deep learning and clustering, in: Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17, JMLR.org, 2017, p. 3861–3870.

[74] F. Li, H. Qiao, B. Zhang, Discriminatively boosted image clustering with fully convolutional auto-encoders, Pattern Recognition 83 (2018) 161–173. doi:https://doi.org/10.1016/j. patcog.2018.05.019.

[75] S. Han, S. Park, S. Park, S. Kim, M. Cha, Mitigating embedding and class assignment mismatch in unsupervised image classification, in: A. Vedaldi, H. Bischof, T. Brox, J.-M. Frahm (Eds.), Computer Vision – ECCV 2020, Springer International Publishing, Cham, 2020, pp. 768–784.

[76] S. Park, S. Han, S. Kim, D. Kim, S. Park, S. Hong, M. Cha, Improving unsupervised image clustering with robust learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 12278–12287.

[77] S. Ioffe, C. Szegedy, Batch normalization: Accelerating deep network training by reducing internal covariate shift, in: F. Bach, D. Blei (Eds.), Proceedings of the 32nd International Conference on Machine Learning, Vol. 37 of Proceedings of Machine Learning Research, PMLR, Lille, France, 2015, pp. 448–456.

[78] T. T. Um, F. M. J. Pfister, D. Pichler, S. Endo, M. Lang, S. Hirche, U. Fietzek, D. Kulic, Data´ augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks, in: Proceedings of the 19th ACM International Conference on Multimodal Interaction, ICMI ’17, Association for Computing Machinery, New York, NY, USA, 2017, p. 216–220.

[79] L. van der Maaten, G. Hinton, Visualizing data using t-sne, Journal of Machine Learning Research 9 (86) (2008) 2579–2605.

[80] K. P. F.R.S., Liii. on lines and planes of closest fit to systems of points in space, The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2 (11) (1901) 559–572. doi:https://doi.org/10.1080/14786440109462720.

[81] L. McInnes, J. Healy, N. Saul, L. Großberger, Umap: Uniform manifold approximation and projection, Journal of Open Source Software 3 (29) (2018) 861. doi:https://doi.org/10. 21105/joss.00861.

[82] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna, Rethinking the inception architecture for computer vision, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2818–2826. doi:https://doi.org/10.1109/CVPR.2016.308.

[83] R. Müller, S. Kornblith, G. Hinton, When does label smoothing help?, in: Proceedings of the 33rd International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA, 2019, pp. 4694—-4703.

[84] J. Downs, E. Vogel, A plant-wide industrial process control problem, Computers & Chemical Engineering 17 (3) (1993) 245–255, industrial challenge problems in process control. doi: https://doi.org/10.1016/0098-1354(93)80018-I.

[85] P. Lyman, C. Georgakis, Plant-wide control of the tennessee eastman problem, Computers & Chemical Engineering 19 (3) (1995) 321–331. doi:https://doi.org/10.1016/ 0098-1354(94)00057-U.

[86] L. H. Chiang, E. L. Russell, R. D. Braatz, Fault diagnosis in chemical processes using fisher discriminant analysis, discriminant partial least squares, and principal component analysis, Chemometrics and Intelligent Laboratory Systems 50 (2) (2000) 243–252. doi:https://doi. org/10.1016/S0169-7439(99)00061-1.

[87] C. A. Rieth, B. D. Amsel, R. Tran, M. B. Cook, Issues and advances in anomaly detection evaluation for joint human-automated systems, in: Advances in Human Factors in Robots and Unmanned Systems, Springer International Publishing, 2018, pp. 52–63. doi:https: //doi.org/10.1007/978-3-319-60384-1\_6.

[88] N. Lawrence Ricker, Decentralized control of the tennessee eastman challenge process, Journal of Process Control 6 (4) (1996) 205–221. doi:https://doi.org/10.1016/0959-1524(96) 00031-5.

[89] N. Ricker, J. Lee, Nonlinear modeling and state estimation for the tennessee eastman challenge process, Computers & Chemical Engineering 19 (9) (1995) 983–1005. doi:https://doi. org/10.1016/0098-1354(94)00113-3.

[90] T. Larsson, K. Hestetun, E. Hovland, S. Skogestad, Self-optimizing control of a large-scale plant: The tennessee eastman process, Industrial & Engineering Chemistry Research 40 (22) (2001) 4889–4901. doi:https://doi.org/10.1021/ie000586y.

[91] A. Bathelt, N. L. Ricker, M. Jelali, Revision of the tennessee eastman process model, IFAC-PapersOnLine 48 (8) (2015) 309–314, 9th IFAC Symposium on Advanced Control of Chemical Processes ADCHEM 2015. doi:https://doi.org/10.1016/j.ifacol.2015.08.199.

[92] C. Reinartz, M. Kulahci, O. Ravn, An extended tennessee eastman simulation dataset for faultdetection and decision support systems, Computers & Chemical Engineering 149 (2021) 107281. doi:https://doi.org/10.1016/j.compchemeng.2021.107281.

[93] D. Cai, X. He, J. Han, Locally consistent concept factorization for document clustering, IEEE Transactions on Knowledge and Data Engineering 23 (6) (2010) 902–913.

[94] H. W. Kuhn, The hungarian method for the assignment problem, Naval research logistics quarterly 2 (1-2) (1955) 83–97.

[95] K. Y. Yeung, W. L. Ruzzo, Details of the adjusted rand index and clustering algorithms, supplement to the paper an empirical study on principal component analysis for clustering gene expression data, Bioinformatics 17 (9) (2001) 763–774.

[96] H. Fan, F. Zhang, R. Wang, X. Huang, Z. Li, Semi-supervised time series classification by temporal relation prediction, in: ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 3545–3549. doi:https://doi. org/10.1109/ICASSP39728.2021.9413883.

[97] X. Zhang, Y. Gao, J. Lin, C.-T. Lu, Tapnet: Multivariate time series classification with attentional prototypical network, Proceedings of the AAAI Conference on Artificial Intelligence 34 (04) (2020) 6845–6852. doi:https://doi.org/10.1609/aaai.v34i04.6165.

[98] J. Goschenhofer, R. Hvingelby, D. Ruegamer, J. Thomas, M. Wagner, B. Bischl, Deep semisupervised learning for time series classification, in: 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA), 2021, pp. 422–428. doi:https://doi. org/10.1109/ICMLA52953.2021.00072.

[99] N. Ricker, Optimal steady-state operation of the tennessee eastman challenge process, Computers & Chemical Engineering 19 (9) (1995) 949–959. doi:https://doi.org/10.1016/ 0098-1354(94)00043-N.

[100] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, Pytorch: An imperative style, high-performance deep learning library, in: Advances in Neural Information Processing Systems 32, Curran Associates, Inc., 2019, pp. 8024–8035.

[101] D. Kingma, J. Ba, Adam: A method for stochastic optimization, International Conference on Learning Representations (12 2014). doi:https://doi.org/10.48550/arXiv.1412.6980.

## A Detailed description of Tennessee Eastman Process

The Tennessee Eastman Process consists of five major units (reactor, product condenser, vapor-liquid separator, recycle compressor, and product stripper) to produce liquids G and H from gaseous reactants A, C, D, E with inert B and byproduct F. The process’s reactions are exothermic and irreversible and are described by:

A(gas) + C(gas) + D(gas) −→ G(liq), Product 1,

A(gas) + C(gas) + E(gas) −→ H(liq), Product 2,

A(gas) + E(gas) −→ F(liq), Byproduct,

There are 52 process variables: 22 continuously measured (Table 12), 19 sampled measured (Table 13) and 11 manipulated variables (Table 14). The measured variables are affected by measurement noise and the sampled variables are obtained with a certain delay. The process undergoes 20 predefined faults (Table 15) with 15 knowns and 5 unknowns.

Faults 1-7 are associated with the step change in a process variable, faults 8-12 are related to increased variability of specific process variables, fault 13 is a slow drift in reaction kinetics, and faults 14 and 15 are caused by sticking valves.

To generate historical data, the process is simulated with a control scheme. Originally, the TEP simulation presented by Eastman Chemical Company was available in open-loop operation. However, the TEP process is open-loop unstable even with initialization from [99]. Besides, real industrial plants operate in a closed loop, so the employment of a closed-loop control system seems natural. There are six process operation modes (see Table 11) which correspond to various G/H mass ratios and production rates in Stream 11.

Table 11: Process operation modes proposed in [84].

<table><tr><td>Mode</td><td>G/H mass ratio</td><td>G production rate (stream 11)</td></tr><tr><td>1</td><td>50/50</td><td> $7038 \text{ kg h}^{-1}$ </td></tr><tr><td>2</td><td>10/90</td><td> $1408 \text{ kg h}^{-1}$ </td></tr><tr><td>3</td><td>90/10</td><td> $10000 \text{ kg h}^{-1}$ </td></tr><tr><td>4</td><td>50/50</td><td>maximum production rate</td></tr><tr><td>5</td><td>10/90</td><td>maximum production rate</td></tr><tr><td>6</td><td>90/10</td><td>maximum production rate</td></tr></table>

![](images/24a17c60bce69569909acc293b935b5cf84d91ba55db43abf5ff43993b666750.jpg)  
Figure 11: Diagram of the Tennessee Eastman Process simulator [86].

Table 12: Continuous process measurements.

<table><tr><td>Variable</td><td>Description</td><td>Units</td></tr><tr><td>XMEAS(1)</td><td>A feed (stream 1)</td><td>ks cm h</td></tr><tr><td>XMEAS(2)</td><td>D feed (stream 2)</td><td>kg  $h^{-1}$ </td></tr><tr><td>XMEAS(3)</td><td>E feed (stream 3)</td><td>kg  $h^{-1}$ </td></tr><tr><td>XMEAS(4)</td><td>A and C feed (stream 4)</td><td>ks cm h</td></tr><tr><td>XMEAS(5)</td><td>Recycle flow (stream 8)</td><td>ks cm h</td></tr><tr><td>XMEAS(6)</td><td>Reactor feed rate (stream 6)</td><td>ks cm h</td></tr><tr><td>XMEAS(7)</td><td>Reactor pressure</td><td>kPa gauge</td></tr><tr><td>XMEAS(8)</td><td>Reactor level</td><td>%</td></tr><tr><td>XMEAS(9)</td><td>Reactor temperature</td><td>°C</td></tr><tr><td>XMEAS(10)</td><td>Purge rate (stream 9)</td><td>ks cm h</td></tr><tr><td>XMEAS(11)</td><td>Product separator temperature</td><td>°C</td></tr><tr><td>XMEAS(12)</td><td>Product separator level</td><td>%</td></tr><tr><td>XMEAS(13)</td><td>Product separator pressure</td><td>kPa gauge</td></tr><tr><td>XMEAS(14)</td><td>Product separator underflow (stream 10)</td><td> $m^{3}$   $h^{-1}$ </td></tr><tr><td>XMEAS(15)</td><td>Stripper level</td><td>%</td></tr><tr><td>XMEAS(16)</td><td>Stripper pressure</td><td>kPa gauge</td></tr><tr><td>XMEAS(17)</td><td>Stripper underflow (stream 11)</td><td> $m^{3}$   $h^{-1}$ </td></tr><tr><td>XMEAS(18)</td><td>Stripper temperature</td><td>°C</td></tr><tr><td>XMEAS(19)</td><td>Stripper steam flow</td><td>kg  $h^{-1}$ </td></tr><tr><td>XMEAS(20)</td><td>Compressor work</td><td>°C</td></tr><tr><td>XMEAS(21)</td><td>Reactor cooling water outlet temperature</td><td>°C</td></tr><tr><td>XMEAS(22)</td><td>Stripper temperature</td><td>°C</td></tr></table>

Table 13: Sampled process measurements, in mole %

<table><tr><td>Block</td><td>Variable</td><td>Description</td></tr><tr><td rowspan="6">Reactor feed analysis</td><td>XMEAS(23)</td><td>Component A</td></tr><tr><td>XMEAS(24)</td><td>Component B</td></tr><tr><td>XMEAS(25)</td><td>Component C</td></tr><tr><td>XMEAS(26)</td><td>Component D</td></tr><tr><td>XMEAS(27)</td><td>Component E</td></tr><tr><td>XMEAS(28)</td><td>Component F</td></tr><tr><td rowspan="8">Purge gas analysis</td><td>XMEAS(29)</td><td>Component A</td></tr><tr><td>XMEAS(30)</td><td>Component B</td></tr><tr><td>XMEAS(31)</td><td>Component C</td></tr><tr><td>XMEAS(32)</td><td>Component D</td></tr><tr><td>XMEAS(33)</td><td>Component E</td></tr><tr><td>XMEAS(34)</td><td>Component F</td></tr><tr><td>XMEAS(35)</td><td>Component G</td></tr><tr><td>XMEAS(36)</td><td>Component H</td></tr><tr><td rowspan="5">Product analysis</td><td>XMEAS(37)</td><td>Component D</td></tr><tr><td>XMEAS(38)</td><td>Component E</td></tr><tr><td>XMEAS(39)</td><td>Component F</td></tr><tr><td>XMEAS(40)</td><td>Component G</td></tr><tr><td>XMEAS(41)</td><td>Component H</td></tr></table>

Table 14: Manipulated variables.

<table><tr><td>Variable</td><td>Description</td><td>Units</td></tr><tr><td>XMV(1)</td><td>D feed flow (stream 2)</td><td> $\text{kg h}^{-1}$ </td></tr><tr><td>XMV(2)</td><td>E feed flow (stream 3)</td><td> $\text{kg h}^{-1}$ </td></tr><tr><td>XMV(3)</td><td>A feed flow (stream 1)</td><td>ks cm h</td></tr><tr><td>XMV(4)</td><td>A and C feed flow (stream 4)</td><td>ks cm h</td></tr><tr><td>XMV(5)</td><td>Compressor recycle valve</td><td>%</td></tr><tr><td>XMV(6)</td><td>D feed flow (stream 2)</td><td>%</td></tr><tr><td>XMV(7)</td><td>D feed flow (stream 2)</td><td> $\text{m}^{3}\text{h}^{-1}$ </td></tr><tr><td>XMV(8)</td><td>D feed flow (stream 2)</td><td> $\text{m}^{3}\text{h}^{-1}$ </td></tr><tr><td>XMV(9)</td><td>D feed flow (stream 2)</td><td>%</td></tr><tr><td>XMV(10)</td><td>D feed flow (stream 2)</td><td> $\text{m}^{3}\text{h}^{-1}$ </td></tr><tr><td>XMV(11)</td><td>D feed flow (stream 2)</td><td> $\text{m}^{3}\text{h}^{-1}$ </td></tr></table>

Table 15: Process faults. Faults introduced in [91] are located under the dashed line.

<table><tr><td>Fault number</td><td>Process variable</td><td>Type</td></tr><tr><td>IDV(1)</td><td>A/C feed ratio, B composition constant (stream 4)</td><td>Step</td></tr><tr><td>IDV(2)</td><td>B composition, A/C ration constant (stream 4)</td><td>Step</td></tr><tr><td>IDV(3)</td><td>D feed temperature (stream 2)</td><td>Step</td></tr><tr><td>IDV(4)</td><td>Reactor cooling water inlet temperature</td><td>Step</td></tr><tr><td>IDV(5)</td><td>Condenser cooling water inlet temperature</td><td>Step</td></tr><tr><td>IDV(6)</td><td>A feed loss (stream 1)</td><td>Step</td></tr><tr><td>IDV(7)</td><td>C header pressure loss - reduced availability</td><td>Step</td></tr><tr><td>IDV(8)</td><td>A, B, C feed composition (stream 4)</td><td>Random variation</td></tr><tr><td>IDV(9)</td><td>D feed temperature (stream 2)</td><td>Random variation</td></tr><tr><td>IDV(10)</td><td>C feed temperature (stream 4)</td><td>Random variation</td></tr><tr><td>IDV(11)</td><td>Reactor cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>IDV(12)</td><td>Condenser cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>IDV(13)</td><td>Reaction kinetics</td><td>Slow drift</td></tr><tr><td>IDV(14)</td><td>Reactor cooling water valve</td><td>Sticking</td></tr><tr><td>IDV(15)</td><td>Condencer cooling water valve</td><td>Sticking</td></tr><tr><td>IDV(16)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV(17)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV(18)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV(19)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV(20)</td><td>Unknown</td><td>Random variation</td></tr><tr><td>IDV(21)</td><td>A feed (stream 1) temperature</td><td>Random variation</td></tr><tr><td>IDV(22)</td><td>E feed (stream 3) temperature</td><td>Random variation</td></tr><tr><td>IDV(23)</td><td>A feed flow (stream 1)</td><td>Random variation</td></tr><tr><td>IDV(24)</td><td>D feed flow (stream 2)</td><td>Random variation</td></tr><tr><td>IDV(25)</td><td>E feed flow (stream 3)</td><td>Random variation</td></tr><tr><td>IDV(26)</td><td>A and C feed flow (stream 4)</td><td>Random variation</td></tr><tr><td>IDV(27)</td><td>Reactor cooling water flow</td><td>Random variation</td></tr><tr><td>IDV(28)</td><td>Condenser cooling water flow</td><td>Random variation</td></tr></table>

## B Training setup for SensorSCAN

The neural network is implemented with the PyTorch [100] library for Python. We initialize the weights with Xavier initialization and perform optimization with Adam [101] for all models. We use the NVIDIA Tesla A100 80Gb for calculations. The hyperparameters are selected using grid search.

To generate the training dataset, we use the sliding window size of L equal to 100 and the step size of 1 for both datasets. The sensor data is normalized with feature-wise standardization (for every feature, the mean and the scale are calculated over all timesteps). We follow [12] in utilizing only a subset of process measurements. We removed 19 sampled process measurements and the measurements that remain constant. As a result, the overall number of sensors D totals 30.

## B.1 First step

We employ only the training dataset for feature extractor pretraining. The best combination of hyperparameters is picked via visual evaluation of the training dataset projection onto the embedding space produced with t-SNE without fault-based coloring since the latter is not available in the real world scenario. The smoother and more non-intersecting clusters are indicative of the optimal combination of hyperparameters.

The training takes $E = 8$ epochs. The learning rate is equal to $1 0 ^ { - 3 }$ , the batch size B is equal to 1024 and the weight decay is equal to $1 0 ^ { - 4 }$ . We found that the large batch size is critical for achieving peak performance. Surprisingly, this finding is connected with the reconstruction task but not with contrastive learning. Usually, contrastive learning requires utilization of a large minibatch.

The other hyperparameters are set as follows: $\lambda _ { \mathrm { c o n t } } = 0 . 7$ (see Eq. 12), $\tau = 0 . 2$ (see Eq. 5), $r = 0 . 5$ and $l _ { m } = 6 ( \sec \mathrm { E q } . 3 )$ . We conducted experiments with gradually increasing the length of the mask $l _ { m }$ and the masking ratio r but have not noticed a statistically significant difference. Regarding the architecture of the model, the encoder embedding dimension H is set to 128, the feed-forward layer dimension is 512, the projection head dimension F is 32, and the dropout rate is equal to 0.1.

For the augmentations, the following hyperparameters are found with grid search. For jitter augmentation, the noise is sampled from a normal distribution with the zero mean and the standard deviation equaling 0.08. For scaling augmentation, the scaling factor is also sampled from the normal distribution with the standard deviation set to 0.1 and the mean equal to 2 and 0.5 for weak and strong augmentations, respectively. During hyperparameter tuning, we found that the effectiveness of learning significantly depends on the number of chunks into which the time series is divided for permutation. Splitting into about 15 chunks exhibits superior results, which is quite counterintuitive since such transformation seriously distorts the temporal information. Presumably, this happens because a large number of chunks induces greater variability in augmentations, which makes the contrastive learning task difficult enough for the model to learn meaningful features.

## B.2 Second step

The model is trained for 5 epochs. The weights of the feature extractor are frozen for three epochs to avoid distortions since the clustering network is randomly initialized and the learning process is quite unstable at the beginning. The best combination of hyperparameters is chosen according to the value of SCAN loss, which does not employ ground truth labels. This way, both steps do not use information about the ground truth fault distribution. Therefore, we consider our hyperparameter tuning approach fair. The learning rate is equal to 1e-2 for the clustering network and to 4e-5 for the feature extractor. The batch size is set to be 128, $\lambda _ { e n t }$ is equal to 2, the number of neighbors K is equal to 12, and the number of chunks T is 20.

## C FDD metrics

This section provides the TPR and FPR values from the radar charts in Sections 6 and 8. Tables 16 and 17 show the results of the ablation study for selection of the number of clusters and of the fault subset, respectively. Tables 18 and 19 show the results of the methods evaluation in the unsupervised setting on $\mathrm { T E P _ { R i e t h } }$ and $\mathrm { T E P _ { R i c k e r } } ,$ , respectively.

Table 16: FDD metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$ for various numbers of clusters. Top down: TPR/FPR for each fault, followed by aggregated detection and diagnosis metrics. The largest TPR values are highlighted on the condition that FPR is not greater than 0.05.

<table><tr><td></td><td>10</td><td>29</td><td>33</td><td>58</td></tr><tr><td>Fault 1</td><td>0.00/0.00</td><td>0.98/0.00</td><td>0.87/0.00</td><td>0.65/0.00</td></tr><tr><td>Fault 2</td><td>0.99/0.00</td><td>0.99/0.00</td><td>0.87/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 3</td><td>0.00/0.00</td><td>0.95/0.00</td><td>0.77/0.00</td><td>0.95/0.00</td></tr><tr><td>Fault 4</td><td>0.99/0.00</td><td>1.00/0.00</td><td>0.93/0.00</td><td>0.59/0.00</td></tr><tr><td>Fault 5</td><td>0.99/0.00</td><td>1.00/0.00</td><td>0.98/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 6</td><td>0.98/0.00</td><td>0.99/0.00</td><td>0.93/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 7</td><td>0.00/0.00</td><td>0.93/0.00</td><td>0.93/0.00</td><td>0.93/0.00</td></tr><tr><td>Fault 8</td><td>0.00/0.00</td><td>0.99/0.00</td><td>0.92/0.00</td><td>0.92/0.00</td></tr><tr><td>Fault 9</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.53/0.00</td><td>0.27/0.00</td></tr><tr><td>Fault 10</td><td>0.00/0.00</td><td>0.98/0.00</td><td>0.92/0.00</td><td>0.78/0.00</td></tr><tr><td>Fault 11</td><td>0.93/0.00</td><td>0.99/0.00</td><td>0.93/0.00</td><td>0.93/0.00</td></tr><tr><td>Fault 12</td><td>0.00/0.00</td><td>0.99/0.00</td><td>0.93/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 13</td><td>0.00/0.00</td><td>0.96/0.00</td><td>0.91/0.00</td><td>0.88/0.00</td></tr><tr><td>Fault 14</td><td>1.00/0.00</td><td>1.00/0.00</td><td>0.93/0.00</td><td>0.93/0.00</td></tr><tr><td>Fault 15</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 16</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.75/0.00</td><td>0.69/0.01</td></tr><tr><td>Fault 17</td><td>0.00/0.00</td><td>0.91/0.00</td><td>0.91/0.00</td><td>0.91/0.00</td></tr><tr><td>Fault 18</td><td>0.00/0.00</td><td>0.96/0.00</td><td>0.96/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 19</td><td>0.00/0.00</td><td>0.99/0.00</td><td>0.98/0.00</td><td>0.89/0.00</td></tr><tr><td>Fault 20</td><td>0.00/0.00</td><td>0.97/0.00</td><td>0.97/0.00</td><td>0.97/0.00</td></tr><tr><td>Fault 21</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 22</td><td>0.96/0.00</td><td>0.96/0.00</td><td>0.44/0.00</td><td>0.59/0.00</td></tr><tr><td>Fault 23</td><td>0.00/0.00</td><td>0.96/0.00</td><td>0.91/0.00</td><td>0.58/0.01</td></tr><tr><td>Fault 24</td><td>0.00/0.00</td><td>0.99/0.00</td><td>0.92/0.00</td><td>0.92/0.00</td></tr><tr><td>Fault 25</td><td>0.00/0.00</td><td>0.95/0.00</td><td>0.92/0.00</td><td>0.85/0.00</td></tr><tr><td>Fault 26</td><td>0.97/0.00</td><td>0.98/0.00</td><td>0.90/0.00</td><td>0.97/0.00</td></tr><tr><td>Fault 27</td><td>0.00/0.00</td><td>0.99/0.00</td><td>0.94/0.00</td><td>0.93/0.00</td></tr><tr><td>Fault 28</td><td>0.00/0.00</td><td>0.96/0.01</td><td>0.95/0.00</td><td>0.96/0.01</td></tr></table>

Table 17: FDD metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$ for feature extractor pretraining with various faults. Top down: TPR/FPR for each fault. The largest TPR values are highlighted on the condition that FPR is no more than 0.05.

<table><tr><td></td><td>Untrained model</td><td>Easy faults</td><td>Difficult faults</td><td>All faults (ours)</td></tr><tr><td>Fault 1</td><td>0.90/0.00</td><td>0.84/0.00</td><td>0.89/0.00</td><td>0.98/0.00</td></tr><tr><td>Fault 2</td><td>0.89/0.00</td><td>0.74/0.00</td><td>0.92/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 3</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.95/0.00</td></tr><tr><td>Fault 4</td><td>0.98/0.00</td><td>0.93/0.00</td><td>0.97/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 5</td><td>0.96/0.00</td><td>0.96/0.00</td><td>0.00/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 6</td><td>0.97/0.00</td><td>0.93/0.00</td><td>0.98/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 7</td><td>1.00/0.00</td><td>0.93/0.00</td><td>0.94/0.01</td><td>0.93/0.00</td></tr><tr><td>Fault 8</td><td>0.42/0.00</td><td>0.89/0.00</td><td>0.89/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 9</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 10</td><td>0.60/0.00</td><td>0.94/0.00</td><td>0.97/0.01</td><td>0.98/0.00</td></tr><tr><td>Fault 11</td><td>0.98/0.00</td><td>0.93/0.00</td><td>0.94/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 12</td><td>0.00/0.00</td><td>0.91/0.00</td><td>0.92/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 13</td><td>0.50/0.00</td><td>0.88/0.00</td><td>0.92/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 14</td><td>0.98/0.00</td><td>0.52/0.00</td><td>0.96/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 15</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 16</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 17</td><td>0.96/0.00</td><td>0.94/0.00</td><td>0.92/0.00</td><td>0.91/0.00</td></tr><tr><td>Fault 18</td><td>0.94/0.00</td><td>0.93/0.00</td><td>0.95/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 19</td><td>0.49/0.00</td><td>0.55/0.00</td><td>0.96/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 20</td><td>0.96/0.00</td><td>0.95/0.00</td><td>0.95/0.00</td><td>0.97/0.00</td></tr><tr><td>Fault 21</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 22</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.56/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 23</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 24</td><td>0.98/0.00</td><td>0.93/0.00</td><td>0.97/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 25</td><td>0.96/0.00</td><td>0.90/0.00</td><td>0.98/0.00</td><td>0.95/0.00</td></tr><tr><td>Fault 26</td><td>0.00/0.00</td><td>0.55/0.00</td><td>0.00/0.00</td><td>0.98/0.00</td></tr><tr><td>Fault 27</td><td>0.00/0.00</td><td>0.55/0.00</td><td>0.93/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 28</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.96/0.01</td></tr></table>

Table 18: FDD metrics evaluated on $\mathrm { T E P _ { R i e t h } }$ in the unsupervised setting. Top down: TPR/FPR for each fault, followed by aggregated detection and diagnosis metrics. The largest TPR values are highlighted on the condition that FPR is not greater than 0.05

<table><tr><td></td><td>PCA</td><td>ST-CatGAN</td><td>ConvAE</td><td>Ours</td></tr><tr><td>Fault 1</td><td>0.92/0.00</td><td>0.00/0.00</td><td>0.95/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 2</td><td>0.88/0.00</td><td>0.85/0.00</td><td>0.95/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 3</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 4</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.97/0.00</td></tr><tr><td>Fault 5</td><td>0.00/0.00</td><td>0.13/0.00</td><td>0.07/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 6</td><td>0.86/0.00</td><td>0.99/0.00</td><td>0.81/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 7</td><td>0.74/0.00</td><td>0.00/0.00</td><td>0.94/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 8</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.67/0.00</td><td>0.78/0.00</td></tr><tr><td>Fault 9</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 10</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.69/0.00</td></tr><tr><td>Fault 11</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 12</td><td>0.37/0.00</td><td>0.00/0.00</td><td>0.08/0.00</td><td>0.95/0.00</td></tr><tr><td>Fault 13</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.38/0.00</td><td>0.76/0.00</td></tr><tr><td>Fault 14</td><td>0.33/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.98/0.00</td></tr><tr><td>Fault 15</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 16</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.71/0.00</td></tr><tr><td>Fault 17</td><td>0.89/0.00</td><td>0.00/0.00</td><td>0.92/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 18</td><td>0.65/0.00</td><td>0.00/0.00</td><td>0.40/0.00</td><td>0.69/0.00</td></tr><tr><td>Fault 19</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.98/0.00</td></tr><tr><td>Fault 20</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>1.00/0.00</td></tr></table>

Table 19: FDD metrics evaluated on $\mathrm { T E P _ { R i c k e r } }$ in the unsupervised setting. Top down: TPR/FPR for each fault, then aggregated detection and diagnosis metrics. The largest TPR values are highlighted on the condition that FPR is not greater than 0.05.

<table><tr><td></td><td>PCA</td><td>ST-CatGAN</td><td>ConvAE</td><td>Ours</td></tr><tr><td>Fault 1</td><td>0.90/0.00</td><td>0.88/0.00</td><td>0.79/0.00</td><td>0.98/0.00</td></tr><tr><td>Fault 2</td><td>0.93/0.00</td><td>0.97/0.00</td><td>0.93/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 3</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.95/0.00</td></tr><tr><td>Fault 4</td><td>0.96/0.00</td><td>0.00/0.00</td><td>0.70/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 5</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.97/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 6</td><td>0.93/0.00</td><td>0.97/0.00</td><td>0.89/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 7</td><td>0.96/0.00</td><td>0.94/0.00</td><td>0.99/0.01</td><td>0.93/0.00</td></tr><tr><td>Fault 8</td><td>0.55/0.00</td><td>0.67/0.00</td><td>0.00/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 9</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 10</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.95/0.01</td><td>0.98/0.00</td></tr><tr><td>Fault 11</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.96/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 12</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 13</td><td>0.73/0.00</td><td>0.86/0.00</td><td>0.95/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 14</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>1.00/0.00</td></tr><tr><td>Fault 15</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 16</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 17</td><td>0.94/0.00</td><td>0.94/0.00</td><td>0.96/0.00</td><td>0.91/0.00</td></tr><tr><td>Fault 18</td><td>0.93/0.00</td><td>0.00/0.00</td><td>0.95/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 19</td><td>0.79/0.00</td><td>0.92/0.00</td><td>0.98/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 20</td><td>0.93/0.00</td><td>0.93/0.00</td><td>0.94/0.00</td><td>0.97/0.00</td></tr><tr><td>Fault 21</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Fault 22</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 23</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.96/0.00</td></tr><tr><td>Fault 24</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.98/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 25</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.98/0.00</td><td>0.95/0.00</td></tr><tr><td>Fault 26</td><td>0.00/0.00</td><td>0.94/0.00</td><td>0.95/0.00</td><td>0.98/0.00</td></tr><tr><td>Fault 27</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.98/0.00</td><td>0.99/0.00</td></tr><tr><td>Fault 28</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.96/0.01</td></tr></table>