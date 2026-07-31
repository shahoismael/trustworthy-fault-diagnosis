# Smart filter aided domain adversarial neural network for fault diagnosis in noisy

industrial scenarios

Baorui Dai<sup>1,2</sup>, Gaëtan Frusque<sup>2</sup>, Tianfu Li<sup>2,3</sup>, Qi Li<sup>1,\*</sup>, Olga Fink<sup>2</sup>

<sup>1</sup>Department of Bridge Engineering, Tongji University, Shanghai 200092, China

<sup>2</sup>Laboratory of Intelligent Maintenance and Operations Systems, EPFL, 1015 Lausanne, Switzerland <sup>3</sup>School of Mechanical Engineering, Xi’an Jiaotong University, Xi’an 710049, China

Corresponding author. E-mail address: liqi\_bridge@tongji.edu.cn (Qi Li).

## Abstract

The application of unsupervised domain adaptation (UDA)-based fault diagnosis methods has shown significant efficacy in industrial settings, facilitating the transfer of operationa experience and fault signatures between different operating conditions, different units of a fleet or between simulated and real data. However, in real industrial scenarios, unknown levels and types of noise can amplify the difficulty of domain alignment, thus severely affecting the diagnostic performance of deep learning models. To address this issue, we propose an UDA method called Smart Filter-Aided Domain Adversarial Neural Network (SFDANN) for fault diagnosis in noisy industrial scenarios. The proposed methodology comprises two steps. In the first step, we develop a smart filter that dynamically enforces similarity between the source and target domain data in the time-frequency domain. This is achieved by combining a learnable wavelet packet transform network (LWPT) and a traditional wavelet packet transform module. In the second step, we input the data reconstructed by the smart filter into a domain adversarial neural network (DANN). To learn domain-invariant and discriminative features, the learnable modules of SFDANN are trained in a unified manner with three objectives: time frequency feature proximity, domain alignment, and fault classification. We validate the effectiveness of the proposed SFDANN method based on two fault diagnosis cases: one involving fault diagnosis of bearings in noisy environments and another involving fault diagnosis of slab tracks in a train-track-bridge coupling vibration system, where the transfe task involves transferring from numerical simulations to field measurements. Results show that compared to other representative state of the art UDA methods, SFDANN exhibits superior performance and remarkable stability.

Keywords: Intelligent fault diagnosis, unsupervised domain adaptation, learnable wavelet packet transform, noisy industrial scenarios.

## 1. Introduction

Intelligent fault diagnosis technology has recently received significant attention and has been widely applied to detect and monitor the health status of mechanical equipment and engineering structures [1-3]. Deep learning algorithms have become a popular choice for intelligent fault diagnosis tasks, with their extensive application being a notable development in this field [4]. Compared to traditional machine learning algorithms that require manual feature engineering, deep learning algorithms can automatically learn meaningful features from raw signals and often achieve better fault diagnosis performance. In particular, deep learning algorithms such as Long Short-term Memory Networks (LSTMs) [5], Generative Adversarial Networks (GANs) [6], and Variational Autoencoders (VAEs) [7] are increasingly gaining prominence in the field of intelligent fault diagnosis. Moreover, in few-shot fault diagnosis scenarios, meta-learning [8] and generative models [9] have recently showcased their effectiveness by learning directly from limited samples and providing data augmentation capabilities, respectively.

Although deep learning-based methods have achieved remarkable success in intelligent fault diagnosis, they still face several challenges that particularly arise from real world limitations [10, 11]. One of the major challenges faced by deep learning-based methods for intelligent fault diagnosis is the difficulty in obtaining sufficient amounts of labeled data. This is particularly due to the fact that faults in safety-critical systems are rare. Insufficient training data can result in overfitting of deep learning models, which limits their applicability in real world deployments. To address the challenge of limited labeled data availability, one potential approach is to use domain adaptation techniques between a labeled source dataset and an unlabeled target dataset, which may be similar but still exhibit a domain gap due to differences in their distributions [12, 13]. In the context of fault diagnosis, domain adaptation has been employed either between different units of a fleet [14, 15], different operating conditions [16- 18], or between simulated and real data [19-22].

Unsupervised domain adaptation (UDA) techniques provide an effective method to address these challenges in fault diagnosis without requiring any labels in target domain [4, 23- 25]. The goal of using these techniques is to align the distributions of the source and target datasets, making the features learned from them indistinguishable between the two domains. Leveraging a classifier trained through either the process of feature alignment or based on learned domain-invariant features can be applied to both domains. This approach leads to better fault diagnosis performance in the target domain compared to directly applying models trained on the source domain. In a comprehensive review paper, various UDA methods were compared, showcasing their strong performance in tackling fault diagnosis challenges associated with domain-shifts [10]. Among these methods, those employing adversarial training techniques have shown significant effectiveness in enhancing classification accuracy and have found widespread application across various domains.

While the majority of developed transfer learning approaches has been focusing on closing the domain gap between data captured under real conditions for different operating conditions or different units of a fleet, recently, there has been an increasing emphasis on closing the domain gap between real and simulated data [19, 20, 22, 26]. In the context of simulation-to-real transfer, there are two main directions: one where labels are available in both the source and target domains [26], and another scenario where labels are only available in the source (simulated) domain [22]. Domain transfer from a fully controlled simulated dataset to a real dataset presents several challenges, including unknown class imbalance between different fault types [22] and unknown noise levels. Such challenges can significantly impact the performance of UDA algorithms, and unfortunately, they have not yet been sufficiently addressed. To address the impact of unknown noise, the deep negative correlation multisource domains adaptation networks [27] and marginal denoising autoencoders [28] have been proposed. However, these methods are either tailored for multiple source domain situations or rely on noise-free signals, which limits their broad applicability in real-world industrial settings.

Given that noise can significantly impact the analysis of time-frequency characteristics in signals, one potential approach to mitigate domain gaps is to dynamically modify raw signals in the time-frequency domain during the training of domain adaptation models. This training approach aims to facilitate the learning of shared features between the source and target domains. In the field of signal processing, the recently proposed Denoising Sparse Wavelet Network (DeSpaWN) [29] and the Learnable Wavelet Packet Transform network (LWPT) [30] have shown superior performance in automatically learning meaningful and sparse features from raw signals. This makes them well-suited for unsupervised signal denoising. DeSpaWN primarily emphasizes low-frequency denoising and feature representation, while LWPT evenly prioritizes all frequency ranges. Inspired by these advancements in signal processing, we recently proposed a framework for acoustic signal denoising based on DeSpaWN [31]. By considering vibration signals as denoised variants of acoustic signals, we proposed an effective acoustic signal filtering technique. Additionally, we discovered that by guiding the signa representations, both DeSpaWN and LWPT can approximate one set of signals to another in the time-frequency domain, thereby achieving signal denoising or enhancement. However, it is worth noting that our recently proposed framework requires target labels, which reduces its feasibility in many real-world applications.

In this research, our primary focus is on tackling the challenge of bridging the domain gap between two noisy datasets of high frequency condition monitoring signals, where we only have labels for the source dataset but not for the target dataset. These datasets are characterize by unknown levels and types of noise that are different between the two domains. This significantly affects the alignment between source and target datasets. Specifically, we addres the problem of UDA within the context of simulated-to-real domain gap scenarios. In this setup, the source (simulated) dataset would be either noise-free or would only contain controlled and known noise levels, while the target (real) dataset could be severely impacted by noise which is not known a priori. To address this challenge, we propose an unsupervised domain adaptation method called the Smart Filter-Aided Domain Adversarial Neural Network (SFDANN). SFDANN starts with two main components: a learnable wavelet packet transform (LWPT) network and a traditional wavelet packet transform (WPT) module to effectivel handle the substantial gap between two domains with different noise levels. By feeding the raw signals from the noisy target domain into the LWPT module, specific coefficients that generat a filtered version of the input signals are produced. Conversely, the raw signals from the source domain are inputted into the WPT module. It gives traditional wavelet coefficients and reconstructs the raw input signals. Then, we introduce a guidance loss that acts on the learnable wavelet coefficients to promote a closer frequency content between the source and (noisy) target domains during the training process. Finally, we input the reconstructed signals of the source and target domains into the latter part of the SFDANN, a typical domain adversarial neural network (DANN) [32], which includes three main parts: a feature extractor, a domain discriminator, and a classifier. The feature extractor extracts features from the source and target domains using a convolutional neural network, the domain discriminator forces the extracted features to lose domain discriminability, and the classifier is trained using the features an labels of the source domain data to classify the aligned source and target domain data. The task of DANN becomes easier due to the similarity in frequency content of both source and target signals, which is achieved with the aid of our proposed smart filter. The proposed SFDANN builds the smart filter, feature extractor, domain discriminator, and classifier in a unified dee neural network framework, allowing the training process of each learnable module to be conducted simultaneously. The main contributions of our research are summarized as follows:

(1) We propose a smart filter that combines the LWPT and WPT techniques with our developed guidance loss. This smart filter dynamically enforces the similarity between the source and (noisy) target domain signals in the time-frequency domain during the training process. To the best of the authors' knowledge, it is the first time that the LWPT and WPT techniques are used in the context of UDA to denoise signals.

(2) Our proposed UDA method utilizes the smart filter to enhance the performance of the domain adversarial neural network for end-to-end intelligent fault diagnosis. This approach effectively captures aligned features between the source and target domains, especially in noisy industrial scenarios.

(3) In our extensive evaluations, we specifically focus on two main application scenarios that have been rarely studied: (a) UDA between two datasets with different levels and types of noise impact in the source and target domain signals, and (b) UDA between a noise-free simulated dataset and a real-world dataset with an unknown level and type of noise. In our application, we consider the train-track-bridge coupling system as a specific case study.

The remaining content of this paper is structured as follows. In Section 2, we provide the background on the DANN and the LWPT. Section 3 presents our proposed method, SFDANN. In Section 4, we demonstrate the effectiveness of SFDANN in cross-domain fault diagnosis through two case studies. Section 4 also includes ablation studies on SFDANN and explores the specific role of the smart filter. Finally, in Section 5, we present the main conclusions of our work.

## 2. Background

## 2.1. Domain adversarial neural network (DANN)

DANN is a neural network inspired by generative adversarial networks and has found widespread applications in various fields, including fault diagnosis [33] and prognosis [34], speech recognition [35], and sentiment analysis [36]. It comprises a feature extractor $( \mathsf { G } _ { \mathtt { F } } )$ , a domain discriminator $( \mathsf { G } _ { \mathsf { D } } )$ , and a classifier $( \mathsf { G } _ { \mathsf { C } } )$ as depicted in Fig. 1 [32]. DANN captures transferable features that are invariant to domain changes through a minimax game between $\mathrm { G } _ { \mathrm { D } }$ and $\mathsf { G } _ { \mathtt { F } }$ . It learns features that are effective for classification by leveraging the relationship between $\mathsf { G } _ { \mathsf { C } }$ and $\mathsf { G } _ { \mathtt { F } }$ . Consequently, DANN can extract features that are both discriminative for classification tasks and invariant to domain changes, making it well-suited for various crossdomain fault diagnosis applications.

F: Feature extractor, D: Domain discriminator, C: Classifie  
![](images/a06738b64f52278954bc21051d14fbecaa1ea940590406accb268ad876801dcb.jpg)  
Fig. 1. Architecture of classic DANN.

The labeled data from the source domain is defined as $\{ x _ { i } ^ { s } ; y _ { i } ^ { s } \} _ { i = 1 } ^ { N _ { s } }$ , and the unlabeled data from the target domain is defined as $\{ x _ { i } ^ { \mathrm { t } } \} _ { i = 1 } ^ { N _ { \mathrm { t } } }$ . Here, $x _ { i } ^ { s } , y _ { i } ^ { s }$ , and $x _ { i } ^ { \mathrm { t } }$ represent the i-th sample from the source domain, the corresponding label for the i-th sample from the source domain, and the i-th sample from the target domain, respectively. $N _ { s }$ and $N _ { \mathrm { t } }$ represent the number of samples from the source domain and target domain. DANN uses a loss function to minimize classification error and maximize domain discrimination error, thereby improving the classification ability of models and reducing the differences in domain distribution of extracted features. This loss function consists of the classification loss $L _ { \mathrm { C } }$ , defined by the cross-entropy loss, and the domain discrimination loss $L _ { \mathrm { D } }$ , defined by the binary cross-entropy loss. It can be expressed as:

$$
L (\theta_ {\mathrm{F}}, \theta_ {\mathrm{C}}, \theta_ {\mathrm{D}}) = L _ {\mathrm{c}} (\theta_ {\mathrm{F}}, \theta_ {\mathrm{C}}) - \lambda L _ {\mathrm{D}} (\theta_ {\mathrm{F}}, \theta_ {\mathrm{D}})\tag{1}
$$

$$
L _ {\mathrm{C}} (\theta_ {\mathrm{F}}, \theta_ {\mathrm{C}}) = - \frac {1}{N _ {s}} \sum_ {i = 1} ^ {N _ {s}} \sum_ {c = 0} ^ {C - 1} I [ y _ {i} ^ {s} = c ] \log \bigl (\mathrm{G} _ {\mathrm{C}} (\mathrm{G} _ {\mathrm{F}} (x _ {i} ^ {s}; \theta_ {\mathrm{F}}); \theta_ {\mathrm{C}}) \bigr)\tag{2}
$$

$$
L _ {\mathrm{D}} (\theta_ {\mathrm{F}}, \theta_ {\mathrm{D}}) = - \frac {1}{N _ {\mathrm{s}}} \sum_ {i = 1} ^ {N _ {\mathrm{s}}} \log \bigl (\mathrm{G} _ {\mathrm{D}} (\mathrm{G} _ {\mathrm{F}} (x _ {i} ^ {\mathrm{s}}; \theta_ {\mathrm{F}}); \theta_ {\mathrm{D}}) \bigr) - \frac {1}{N _ {\mathrm{t}}} \sum_ {i = 1} ^ {N _ {\mathrm{t}}} \log \Bigl (1 - \mathrm{G} _ {\mathrm{D}} (\mathrm{G} _ {\mathrm{F}} (x _ {i} ^ {\mathrm{t}}; \theta_ {\mathrm{F}}); \theta_ {\mathrm{D}}) \Bigr)\tag{3}
$$

where $\theta _ { \mathrm { F } } , \theta _ { \mathrm { C } }$ , and $\theta _ { \mathrm { D } }$ are the parameters of $\mathrm { G } _ { \mathrm { F } } , \mathrm { G } _ { \mathrm { C } }$ , and $\mathrm { G } _ { \mathrm { D } }$ , respectively; ?? is a trade-off parameter; ??[·] is a symbol function that takes a value of 1 if the true label of $x _ { i } ^ { s }$ is $c _ { \mathrm { { ; } } }$ , and 0 otherwise; and ?? is the number of class labels.

## 2.2. Learnable wavelet packet transform network (LWPT)

LWPT is a recently proposed deep learning framework that draws inspiration from WPT and aims to automatically learn meaningful and sparse representations of raw signals [30]. Fig. 2 depicts the cascade algorithm associated with WPT, which serves as the fundamental architecture of LWPT. This algorithm decomposes the input signal into detail and approximation coefficients by applying a low-pass and a high-pass filter, followed by a subsampling step. In a recursive manner, the detail and approximation coefficients of the previous layer are decomposed using a similar process. The detail and approximation coefficients of the final decomposition layer form the time-frequency representation of the input signal. By using the inverse WPT, the input signal can be perfectly reconstructed from the obtained representation.

![](images/6d263f3eb8e4b31e715d8dde954ee3e3bc97a45558e3ace570db801b414604d4.jpg)

Fig. 2. Architecture of the cascade algorithm related to the WPT.  
![](images/53235803cf6326b92ddb3e363f65a4ef43c7fc9581f2231b8b1677d73343369c.jpg)  
Fig. 3. Encoding blocks of the WPT and the LWPT.

LWPT adopts an encoder-decoder architecture that sequentially applies learnable signal decomposition and learnable signal reconstruction. It incorporates a fully learnable version of the cascade algorithm, enabling the learning of the kernel shared by both filters at each layer The resulting detail and approximation coefficients from each layer are then passed through learnable hard thresholding (HT) activation functions originally proposed for DeSpaWN [29]. Fig. 3 illustrates the encoding blocks of WPT and LWPT, with elements shared with Fig. 2. The HT activation function is a combination of two sigmoid functions with opposite characteristics, and it is mathematically expressed as follows:

$$
\mathrm{HT} (x) = x \left[ \frac {1}{1 + \exp (1 0 (x + b))} + \frac {1}{1 + \exp (- 1 0 (x - b))} \right]\tag{4}
$$

where ?? is the learnable bias acting as the thresholds on both sides of the origin.

The decoding blocks of LWPT exhibit a reverse architecture compared to the encoding blocks but without the HT activation functions. In each layer, the learnable HT activation functions operate independently to automatically denoise the wavelet coefficients in the encoding blocks. However, due to the denoising effect of HT activation function, achieving perfect signal reconstruction with the decoding part of LWPT is not feasible anymore.

## 3. Proposed method

## 3.1. Overall framework of the proposed method

When the signals are significantly affected by high levels of noise, current domain adaptation methods may face challenges in effectively learning the shared features between the source and target domains, leading to misclassification of a significant number of samples in the target domain. In this research, we focus on high frequency condition monitoring signals. To facilitate the learning of domain-invariant and discriminative features, we propose dynamically enforcing the similarity between the source and target domain signals in the timefrequency domain during the training process while learning to filter the noisy target domain signals. The overall framework of our proposed SFDANN is depicted in Fig. 4, comprising four modules: a smart filter, a feature extractor, a domain discriminator, and a classifier. To clarify the operation mechanism of the SFDANN, the architecture in Fig. 4 is divided into two parts: Part A and Part B. Part A comprises the data input and the smart filter, which can filter signals without requiring neither labels nor ground truth noise-free signals. Part A inputs the source domain data into the WPT module and the (noisy) target domain data into the LWPT module. We also evaluate the opposite data input strategy, where we input the (noisy) target domain signals into the WPT and the source domain data into the LWPT. It is important to note that Fig. 4 displays only the first of the data input strategies. The optimal data input strategy will be further discussed in the ablation study section based on the evaluation of the obtained results. Part B involves the typical DANN architecture. During the training stage, the data from the source and target domains undergo filtering in Part A before being passed to Part B. Additionally, loss functions in Part B affect the parameter learning in Part A through error backpropagation.

As depicted in Fig. 4, firstly, raw signals from the source domain are input into the WPT module’s encoder for signal decomposition, while the raw signals from the target domain are input into the LWPT module’s encoder for signal decomposition. This process generates traditional wavelet coefficients and learnable wavelet coefficients, respectively. Next, the traditional wavelet coefficients and learnable wavelet coefficients are reconstructed into time domain signals by the decoders of the WPT module and the LWPT modules. Guided by the guidance loss, the features of the learnable wavelet coefficients dynamically converge towards those of the traditional wavelet coefficients during the learning process. This convergence enforces the similarity between the source and target domain signals in the time-frequency domain while filtering noisy signals.

After being processed by the smart filter in Part A, the time-domain signals from the source and target domains are fed into the Part B of SFDANN. Part B consists of three typical components found in a DANN: feature extractor, domain discriminator, and classifier. The feature extractor utilizes a convolutional neural network to extract features from the source and target domains. The domain discriminator, guided by the domain discrimination loss, encourages the extracted features to lose domain distinction and become indistinguishable between the two domains. Meanwhile, the classifier is trained to minimize the misclassification loss associated with the source domain data. Moreover, the learning process of the smart filter is also guided by the domain discrimination loss and the classification loss, ensuring that the filter adjusts the time-frequency characteristics to facilitate the learning of domain-invariant and discriminative features.

![](images/2f2413de627aa4c5b73cca0697296c2e800df966034c7570f4d8c43fd3ec6452.jpg)  
Fig. 4. Architecture of the proposed SFDANN.

## 3.2. Smart filter

The smart filter ${ \sf G } _ { s }$ consists of two modules: $\mathsf { G } _ { s - \mathsf { W P T } }$ (WPT module) and $\mathsf { G } _ { \mathsf { s } - \mathsf { L W P T } }$ (LWPT module). Two distinct data input strategies are viable for the smart filter, the optimal of which will be discussed in the ablation study section. For simplicity, we will explain the functioning of the smart filter using an example where data from the source domain is fed into $\mathsf { G } _ { s - \mathsf { W P T } }$ , and data from the target domain is fed into $\mathsf { G } _ { \mathsf { s } - \mathsf { L W P T } }$ , consistent with depicted in Fig. 4.

First, $x _ { i } ^ { s }$ is decomposed into traditional wavelet coefficients $\left\{ \left\{ c _ { i , j , k } ^ { s } \right\} _ { k = 1 } ^ { K } \right\} _ { j = 1 } ^ { 2 ^ { L } }$ using the encoder in $\complement _ { s \mathrm { - } \mathrm { { W P T } } }$ , which follows an L-layer signal decomposition framework. Similarly, $x _ { i } ^ { \mathrm { t } }$ is decomposed into learnable wavelet coefficients $\left\{ \left\{ c _ { i , j , k } ^ { \mathrm { t } } \right\} _ { k = 1 } ^ { K } \right\} _ { j = 1 } ^ { 2 ^ { L } }$ using the encoder in $\mathrm { G } _ { \mathrm { s - L W P T } }$ , which also has an L-layer signal decomposition framework. Here, ?? is the number of wavelet coefficients. The determination of the number of decomposition layers $L$ is influenced by the sampling frequency $F _ { s }$ and the frequency bandwidth $F _ { r }$ , which are important for fault diagnosis of the input signal, as stated in [29, 30]. The determination of $F _ { r }$ is dependent on the inherent characteristics of the fault diagnosis object and will be further discussed in the case studies section. It is essential to highlight that when increasing the number of signal decomposition layers $L ,$ , it directly adds complexity to the convolutional neural network. While this can result in finer decomposition of source and target domain signals, encouraging similarity in these smaller frequency bands, it is crucial to note that these fine-grained frequency bands do not possess specific physical significance for fault diagnosis. Consequently, they do not contribute to improving the smart filter's performance. On the contrary, the augmented learnable parameters can elevate the learning difficulty due to potential overfitting effect, especially when the dataset has limited samples.

In real industrial scenarios, significant differences in time-frequency features between the source and target domain signals often arise due to factors such as noise interference. These differences typically stem from the distinct environments of each domain. Consequently, it is desirable to achieve overall similarity in the time-frequency characteristics of source and target domain samples. The expectation of the features of all samples from the source domain provides an overall representation of the data characteristics in that domain. However, during training, deep learning models typically process data in batches rather than the entire dataset at once. A batch size, which is smaller than the total number of samples, is determined, and a random batch of samples is selected for each training iteration. To ensure that the expectation of the features in a batch approximates that of the entire dataset, the ratio of samples corresponding to each class in the batch should be roughly equivalent to that in the whole dataset. According to the law of large numbers (specifically, Bernoulli’s law of large numbers), it is important to ensure that the batch size is not too small. Further details regarding the appropriate batch size will be presented in the case studies section.

Assuming that the batch size of SFDANN is B, we define a guidance loss to encourage the expectation of the learnable wavelet coefficients $\left\{ \left\{ \{ c _ { i , j , k } ^ { \mathrm { t } } \} _ { k = 1 } ^ { K } \right\} _ { j = 1 } ^ { 2 ^ { L } } \right\} _ { i = 1 } ^ { B }$ for a batch of target 1 domain samples $\{ x _ { i } ^ { \mathrm { t } } \} _ { i = 1 } ^ { B }$ to be close to the expectation of the traditional wavelet coefficients $\left\{ \left\{ \{ c _ { i , j , k } ^ { s } \} _ { k = 1 } ^ { K } \right\} _ { j = 1 } ^ { 2 ^ { L } } \right\} _ { i = 1 } ^ { B }$ for a batch of source domain samples $\{ x _ { i } ^ { s } \} _ { i = 1 } ^ { B }$ . The guidance loss is expressed as follows:

$$
L _ {\mathrm{G}} (\theta_ {S}) = \frac {1}{2 ^ {L}} \sum_ {j = 1} ^ {2 ^ {L}} \left(\operatorname{E} \left[ \left\{\left\{c _ {i, j, k} ^ {\mathrm{s}} \right\} _ {k = 1} ^ {K} \right\} _ {i = 1} ^ {B} \right] - \operatorname{E} \left[ \left\{\left\{c _ {i, j, k} ^ {\mathrm{t}} \right\} _ {k = 1} ^ {K} \right\} _ {i = 1} ^ {B} \right]\right) ^ {2}\tag{5}
$$

$$
\operatorname{E} \left[ \left\{\left\{c _ {i, j, k} ^ {s} \right\} _ {k = 1} ^ {K} \right\} _ {i = 1} ^ {B} \right] = \frac {1}{B} \sum_ {i = 1} ^ {B} \frac {1}{K} \sum_ {k = 1} ^ {K} \left| c _ {i, j, k} ^ {s} \right|\tag{6}
$$

$$
\operatorname{E} \left[ \left\{\left\{c _ {i, j, k} ^ {\mathrm{t}} \right\} _ {k = 1} ^ {K} \right\} _ {i = 1} ^ {B} \right] = \frac {1}{B} \sum_ {i = 1} ^ {B} \frac {1}{K} \sum_ {k = 1} ^ {K} \Bigl | c _ {i, j, k} ^ {\mathrm{t}} \Bigr |\tag{7}
$$

where $\theta _ { S }$ represents the parameters of the smart filter ${ \sf G } _ { s } ;$ ?? is the wavelet coefficient number, which is related to time.

While being guided by the guidance loss $L _ { \mathrm { G } } ( \theta _ { \mathrm { S } } )$ and obtaining a similar signal content representation between source and target, the traditional wavelet coefficients $\left\{ \left\{ \{ c _ { i , j , k } ^ { s } \} _ { k = 1 } ^ { K } \right\} _ { j = 1 } ^ { 2 ^ { L } } \right\} _ { i = 1 } ^ { B }$ and learnable wavelet coefficients $\{ \{ ( c _ { i , j , k } ^ { \mathrm { t } } \} _ { k = 1 } ^ { K } \} _ { j = 1 } ^ { 2 ^ { L } } \} _ { i = 1 } ^ { B }$ are input into 1 1 the separate decoders with L-layer signal reconstruction frameworks in $\complement _ { s \mathrm { - } \mathrm { { W P T } } }$ and $\mathrm { G } _ { \mathrm { s - L W P T } }$ respectively, to generate reconstructed signals $\{ \widehat { x } _ { i } ^ { s } \} _ { i = 1 } ^ { B }$ and $\{ \hat { x } _ { i } ^ { \mathrm { t } } \} _ { i = 1 } ^ { B }$ in the source and target domains. Then, $\{ \hat { x } _ { i } ^ { s } ; y _ { i } ^ { s } \} _ { i = 1 } ^ { B }$ and $\{ \hat { x } _ { i } ^ { \mathrm { t } } \} _ { i = 1 } ^ { B }$ are input into the subsequent DANN framework.

## 3.3. Objective function

To facilitate the learning of domain-invariant and discriminative features, our SFDANN is trained using three loss functions: the guidance loss, classification loss, and domain alignment loss. These loss functions directly and individually impact the smart filter, classifier, and domain discriminator components of SFDANN, as depicted in Fig. 4.

## 3.3.1. Guidance loss

The guidance loss serves the purpose of dynamically aligning the time-frequency features of the source and target domains, facilitating the learning of shared features between them. The expression for the guidance loss is shown in Eq. (5).

## 3.3.2. Classification loss

The classification loss is calculated using cross entropy loss, which measures the discrepancy between the predicted label and the true label. This loss function guides the classifier to improve its classification performance and make accurate predictions. The expression for the classification loss is provided in Eq. (2). However, in our proposed SFDANN, the input $x _ { i } ^ { s }$ in Eq. (2) is replaced by $\widehat { x } _ { i } ^ { s }$ , which corresponds to $\mathrm { G } _ { \mathrm { S } } ( x _ { i } ^ { s } ; \theta _ { S } )$ ). As a result, $L _ { \mathrm { C } } ( \theta _ { \mathrm { F } } , \theta _ { \mathrm { C } } )$ in Eq. (2) is modified to $L _ { \mathrm { C } } ( \theta _ { \mathrm { S } } , \theta _ { \mathrm { F } } , \theta _ { \mathrm { C } } )$

## 3.3.3. Domain alignment loss

The domain alignment loss is calculated using binary cross-entropy loss, which quantifies the discrepancy between the source and target domain labels. Its objective is to ensure that the features extracted by the feature extractor are distinguishable by the domain discriminator. The expression for the domain alignment loss is presented in Eq. (3). However, in our proposed SFDANN, the terms $x _ { i } ^ { s }$ and $x _ { i } ^ { \mathrm { t } }$ in Eq. (3) are substituted by $\widehat { x } _ { i } ^ { s }$ and $\hat { x } _ { i } ^ { \mathrm { t } }$ , respectively, denoting the outputs of $\mathsf { G } _ { \mathsf { S } } ( x _ { i } ^ { s } ; \theta _ { \mathsf { S } } )$ and $\mathrm { G } _ { \mathrm { S } } ( x _ { i } ^ { \mathrm { t } } ; \theta _ { S } )$ . As a result, $L _ { \mathrm { D } } ( \theta _ { \mathrm { F } } , \theta _ { \mathrm { D } } )$ in Eq. (3) is modified to $L _ { \mathrm { D } } ( \theta _ { \mathrm { S } } , \theta _ { \mathrm { F } } , \theta _ { \mathrm { D } } )$

## 3.3.4. Overall objective function

The overall objective function of SFDANN is a combination of the three loss functions and can be expressed as follows:

$$
L (\theta_ {\mathrm{S}}, \theta_ {\mathrm{F}}, \theta_ {\mathrm{D}}, \theta_ {\mathrm{C}}) = L _ {\mathrm{C}} (\theta_ {\mathrm{S}}, \theta_ {\mathrm{F}}, \theta_ {\mathrm{C}}) - \lambda L _ {\mathrm{D}} (\theta_ {\mathrm{S}}, \theta_ {\mathrm{F}}, \theta_ {\mathrm{D}}) + \mu L _ {\mathrm{G}} (\theta_ {\mathrm{S}})\tag{8}
$$

where ?? and $\mu$ are the trade-off parameters.

By assigning higher importance to $L _ { \mathrm { G } }$ than $L _ { \mathrm { D } }$ in the initial stages of training, we can facilitate the alignment of time-frequency features between the source and target domains and create favorable conditions for extracting shared features from them. Following the approach proposed by [37], we introduce a weight expression for $L _ { \mathrm { D } }$ , denoted as ??, which increases from 0 to 1 as the training progresses:

$$
\lambda = \frac {2}{1 + e ^ {- 1 0 p}} - 1\tag{9}
$$

where $p$ is the ratio of the current epochs to the total number of epochs in the training procedure.

## 4. Case studies

## 4.1. Introduction of case studies

We validate the effectiveness of the proposed SFDANN for fault diagnosis in industrial scenarios with unknown types and levels of noise through two cases. The first case study focuses on fault diagnosis of bearings, where the source and target domain data are constructed with a modified version of CWRU dataset [38] to match the case of significantly different levels of noise interference. The second case study involves the recognition of the health states of slab track in a train-track-bridge coupling system, where the source data are obtained from numerical simulation and target domain data comprise field measurements. In both cases, the data utilized consists of acceleration signals.

To evaluate the performance of SFDANN, we compare it with five other UDA methods: joint adaptation networks (JAN) [39], multi kernels maximum mean discrepancy (MK-MMD) [40], correlation alignment (CORAL) [41], DANN [32], and conditional DANN (CDANN) [42]. To ensure fair and effective comparisons, we use a unified testing framework and consistent parameter settings for all methods, following the structure outlined in the review paper by [10]. A four-layer CNN serves as the feature extractor for all methods, and we also use this feature extractor in combination with a classifier as a baseline for comparison wit UDA methods. In the case of the SFDANN method, unless stated otherwise, the source domain data is input into the WPT module and target domain data into the LWPT module, as depicted in Fig. 4. Since the classes in both case studies are balanced, the performance evaluation metric used in this paper is the overall accuracy of classification, calculated as the number of accurately classified samples divided by the total number of samples. To minimize result variability, we perform calculations with five random seeds and report the average and standard deviation of the classification results.

## 4.1.1. Case Study 1: bearing fault diagnosis

## (1) Data description

The CWRU dataset has been one of the most commonly used open-source datasets in UDA research. A diagram illustrating the experimental setup is provided in Fig. 5 [38]. In this research, we use the 12 kHz drive-end bearing data, with detailed parameters provided in Table 1. The bearing data consists of four different operating conditions, each corresponding to a different combination of load and rotational speed parameters. The dataset contains 10 different health conditions of the bearing, including one healthy state and three different fault types (inner ring faults (IF), outer ring faults (OF) and ball faults (BF)) with three different fault sizes each.

However, unlike previous studies, our case considers the impact of environmental noise on the target domain data, which is considered more realistic for industrial applications. In practical industrial settings, various levels of noise interference are unavoidable, and UDA techniques always involve generalizing data to unknown levels of noise. To simulate realworld scenarios, we add Gaussian white noise to target domain data, resulting in noisy signals with signal-to-noise ratios (SNRs) of 0 or -5. The SNR is defined as follows:

$$
\mathrm {SNR_ {dB}} = 1 0 \log_ {1 0} {\frac {P _ {\mathrm{s}}}{P _ {\mathrm{n}}}}\tag{10}
$$

where $P _ { s }$ and $P _ { \mathrm { n } }$ are the power of the signal and noise, respectively.

![](images/aee93b058898ce52c54ef95ea31e5647df121aad262158231d3f908e115a6ac3.jpg)  
Fig. 5. Diagram of the test rig for collecting CWRU data.

Table 1. Parameters for the CWRU dataset.

<table><tr><td>Operation condition</td><td>Rotational speed (RPM)</td><td>Class label (Fault type-severity in mm)</td></tr><tr><td>0</td><td>1797</td><td rowspan="4">0(NA), 1(IF-7), 2(BF-7), 3(OF-7), 4(IF-14), 5(BF-14), 6(OF-14), 7(IF-21), 8(BF-21), 9(OF-21)</td></tr><tr><td>1</td><td>1772</td></tr><tr><td>2</td><td>1750</td></tr><tr><td>3</td><td>1730</td></tr></table>

## (2) Parameter settings

In this study, the Z-score normalization method is used to normalize the data. The normalized signal is then segmented into samples using a sliding window with a window length of 2048, which coincides with the approach and parameters adopted by [43] and [44]. The segmentation process does not involve any overlap between the samples. Consequently, each fault state contains approximately 60 samples, while the healthy state contains approximately 120 samples under operation condition 0 and 240 samples under other operation conditions. For both the source and target domains, 80% of the total samples are used for training, and 20% are reserved for testing. The cross-domain task of 0→1 is denoted as T01, indicating that the model is trained using labeled training data from operation condition 0 and unlabeled training data from operation condition 1, and subsequently tested using testing data from operation condition 1.

The model training process is conducted over a total of 300 epochs. The initial learning rate is set to 0.001, which is reduced by a factor of 0.1 at epochs 150 and 250 to facilitate learning rate decay. To ensure the inclusion of meaningful frequency bandwidth for bearing fault diagnosis, the smart filter is configured with five signal decomposition layers, resulting in a frequency resolution of $1 2 0 0 0 / 2 / 2 ^ { 5 } = 1 8 7 . 5 \mathrm { { H z } }$ . The batch size is set to 64. The tradeoff parameter ?? is determined using Eq. (9), while the value of $\mu$ is adjusted based on the severity of the noise. Specifically, $\mu$ is set to 1 when no Gaussian white noise is present, 2 at a SNR of 0, and 10 at a SNR of -5.

## 4.1.2. Case Study 2: State recognition of slab tracks

Slab track has been increasingly built in urban rail transit and high-speed railways. The mortar layer, which connects the slab track and the foundation, is susceptible to degradation due to train dynamic loads, temperature fluctuations, and other factors. Recognizing the state of the slab track is crucial for ensuring the safe operation of trains. However, acquiring an adequate amount of labeled data from field measurements poses challenges. Moreover, measurements are usually taken under noisy environments. Therefore, conducting numerical simulation and then applying the UDA methods to transfer from labeled and noise-free numerical simulation data to unlabeled and noisy measurement data is very important.

## (1) Field measurements

Since the primary cause of slab track deterioration is the degradation of the mortar layer, which alters the support conditions of the slab track, we have chosen three different support conditions of slab track from a railway test line to represent three different deterioration states, as depicted in Fig. 6. The connections between the three track slabs and the foundation are supported by mortar, rubber, and discrete spring, respectively, representing the healthy state, moderate deterioration state, and severe deterioration state.

![](images/9a6b2c8e7426b4768a4a839fa9876c7a65c03ffceb27bd4f8f96f29f246788f4.jpg)  
Fig. 6. Three types of slab track with different support conditions.

The railway test line is installed on simply supported girder bridges. A six-vehicle metro train with a total length of 140 meters operates on the railway test line. Acceleration signals are collected from acceleration sensors installed on the three types of slab track, as depicted in Fig. 7. The signals are sampled at a frequency of 20 kHz. The train operates at speeds of 20, 40, 60, and 80 km/h, and six passes are conducted at each speed.

![](images/01a52da9139cde9203b50962218d0f36de59b6387f4d51db2b2c7852c3217be2.jpg)  
Fig. 7. Placement of acceleration sensors.

## (2) Numerical simulation

To obtain the numerical simulation signals of slab track vibration, a coupling vibration numerical model is established based on the mechanical parameters of the train, track, and bridge obtained from the field experiment. The model development is divided as two steps to reduce computational costs. Firstly, the numerical model of the train, steel rail, and bridge is established using ANSYS software with the beam elements, as depicted in Fig. 8. To generate track irregularity, we use the irregularity spectrum from ISO 3095-2013 [45] and then obtain the wheel-rail force time-history data using the modal superposition method [46] when the train travels at the four different speeds.

![](images/0f77fbb7dd46452e3c77872587fbcd4e26f4c04f71424236e7552f48f221cc74.jpg)  
(b)  
Fig. 8. Numerical models with beam element: (a) Single vehicle of metro train; (b) Rail and bridge.

Secondly, a refined numerical model of the steel rail, slab track, and bridge is developed using the ABAQUS software. Beam elements are used to model the steel rails, while solid elements are used to model the slab track and bridge structures, as illustrated in Fig. 9. The wheel-rail force time-history data obtained previously is loaded onto the numerical model in ABAQUS as a moving load. The simulated acceleration signals of the slab track are then calculated corresponding to the placement position of the acceleration sensors in the field measurement.

![](images/66e000348771c6386e0fb0b572883ac0d9074d099830a79a20838a33fdceeb58.jpg)  
Fig. 9. Refined numerical models of rail, slab track and bridge.

## (3) Parameter Settings

In the time domain, the acceleration signal of the slab track exhibits six cycles of waveforms due to the continuous excitation caused by the six vehicles of the metro train. To enhance the dataset, each measured or simulated sample is divided into six sub-samples based on the time when each vehicle passes through the slab track. Consequently, for both measured and simulated data, there are a total of 36 sub-samples for each track state at each speed.

In the frequency domain, the computational cost significantly increases when calculating high-frequency vibrations. Therefore, we set the maximum frequency of the numerical simulation to 800Hz. To ensure consistency with the frequency range of the numerical simulation signal, we apply a bandpass filter to the measured signal and retain the frequency components below 800Hz.

For data normalization, we use the Z-score normalization method to globally normalize the data, using the numerical simulation as the source domain and the field measurement as the target domain. In both domains, 75% of the total samples are allocated for the training set, while the remaining 25% are assigned to the testing set. To approximate more realistic scenarios, sub-samples from the same train ride are either all included in the training dataset or all included in the testing dataset.

In this case study, we use healthy and faulty acceleration signals generated from numerical simulations as input source domain data, while healthy and faulty signals collected from acceleration sensors during field experiments serve as input target domain data. The resulting output of the SFDANN provides classification labels. The model training parameters are the same as those used in Case Study 1. The smart filter includes six signal decomposition layers, which corresponds to a frequency resolution of $8 0 0 / 2 ^ { 6 } = 1 2 . 5 \mathrm { { H z } }$ . The batch size is set to 27, ensuring that it can be evenly divided by $3 6 \times 3 \times 7 5 \% = 8 1$ training samples at each speed. The trade-off parameter ?? is set according to Eq. (9), and $\mu$ is set to 5.

## 4.2. Results and analysis

## 4.2.1. Diagnosis results

## (1) Diagnosis results of Case Study 1 (Bearing dataset)

Diagnosis results in the Case Study 1 are obtained considering noise added in the target domain data with SNRs of 0 and -5. Given the extensive diagnosis results obtained through experiments conducted with various noise levels and based on different UDA methods, we selectively focus on a subset of transfer tasks. Specifically, we present transfer diagnosis results from operating condition 0 to other operating conditions, as well as from operating condition 1 to other operating conditions. This subset constitutes half of all transfer tasks and provides a suitable basis for comparing the performance of different UDA methods. Tables 2 showcases these results. The findings in Tables 2 clearly demonstrate the superior fault diagnosis accuracy achieved by the SFDANN method proposed in this paper. When using the original noise-free CWRU dataset, all UDA methods attain fault diagnosis accuracy close to 100%, indicating the effective resolution of domain transfer challenges through conventional UDA methods in the original CWRU dataset. However, the introduction of noise interference into the target domain dataset adversely impacts the performance of all fault diagnosis methods. At an SNR of 0, the SFDANN method exhibits an average diagnostic accuracy of 96.3%, surpassing other fault diagnosis methods by 2.0%-50.8%. At the SNR of -5, the superiority of the SFDANN method over other fault diagnosis methods becomes more pronounced. Despite the average diagnostic accuracy of the SFDANN method decreasing to 83.9%, it continues to outperform other faul diagnosis methods by 10.7%-52.9%. These results underscore the exceptional diagnostic performance of the SFDANN method in noisy environments.

Table 2. Bearing fault diagnosis accuracy considering various transfer tasks and noise levels.

<table><tr><td>Noise level</td><td>Task</td><td>CNN</td><td>CORAL</td><td>MK-MMD</td><td>JAN</td><td>DANN</td><td>CDANN</td><td>SFDANN</td></tr><tr><td rowspan="7">No noise</td><td>T01</td><td>99.4%±1.3%</td><td>100.0%±0</td><td>100.0%±0</td><td>99.9%±0.3%</td><td>99.9%±0.3%</td><td>100.0%±0</td><td>99.9%±0.3%</td></tr><tr><td>T02</td><td>99.6%±0.3%</td><td>99.9%±0.3%</td><td>99.8%±0.3%</td><td>99.7%±0.3%</td><td>99.8%±0.3%</td><td>99.6%±0.3%</td><td>99.7%±0.3%</td></tr><tr><td>T03</td><td>86.5%±2.5%</td><td>100.0%±0</td><td>99.9%±0.3%</td><td>99.7%±0.4%</td><td>99.8%±0.3%</td><td>99.8%±0.3%</td><td>99.7%±0.3%</td></tr><tr><td>T10</td><td>98.9%±1.1%</td><td>99.7%±0.4%</td><td>99.8%±0.3%</td><td>99.5%±0.4%</td><td>99.8%±0.3%</td><td>99.5%±0.4%</td><td>99.9%±0.3%</td></tr><tr><td>T12</td><td>99.5%±0.3%</td><td>99.7%±0.3%</td><td>99.7%±0.3%</td><td>99.7%±0.3%</td><td>99.9%±0.3%</td><td>99.9%±0.3%</td><td>99.9%±0.3%</td></tr><tr><td>T13</td><td>97.1%±1.2%</td><td>99.8%±0.3%</td><td>100.0%±0</td><td>99.9%±0.2%</td><td>99.8%±0.2%</td><td>100.0%±0</td><td>99.9%±0.3%</td></tr><tr><td>Average</td><td>96.8%±1.1%</td><td>99.9%±0.2%</td><td>99.9%±0.2%</td><td>99.7%±0.3%</td><td>99.8%±0.3%</td><td>99.8%±0.2%</td><td>99.8%±0.3%</td></tr><tr><td rowspan="7">SNR=0</td><td>T01</td><td>43.1%±12.7%</td><td>31.1%±21.0%</td><td>93.4%±2.3%</td><td>93.8%±1.2%</td><td>95.5%±1.8%</td><td>93.6%±2.4%</td><td>96.1%±1.0%</td></tr><tr><td>T02</td><td>43.1%±8.1%</td><td>56.1%±24.1%</td><td>93.7%±3.3%</td><td>94.8%±1.7%</td><td>95.1%±1.6%</td><td>93.1%±1.8%</td><td>97.1%±1.0%</td></tr><tr><td>T03</td><td>30.3%±5.2%</td><td>22.7%±5.1%</td><td>75.8%±2.8%</td><td>90.5%±3.0%</td><td>91.9%±4.5%</td><td>88.3%±4.1%</td><td>96.2%±1.3%</td></tr><tr><td>T10</td><td>50.6%±10.7%</td><td>58.6%±1.5%</td><td>82.9%±2.9%</td><td>77.9%±4.9%</td><td>93.9%±2.3%</td><td>88.9%±2.4%</td><td>94.0%±0.8%</td></tr><tr><td>T12</td><td>60.8%±2.6%</td><td>82.4%±2.3%</td><td>95.1%±2.0%</td><td>95.7%±2.1%</td><td>96.1%±1.1%</td><td>95.6%±0.8%</td><td>97.8%±1.3%</td></tr><tr><td>T13</td><td>45.3%±4.5%</td><td>48.8%±10.4%</td><td>94.3%±1.7%</td><td>96.0%±0.6%</td><td>93.4%±1.4%</td><td>93.4%±1.6%</td><td>96.7%±1.2%</td></tr><tr><td>Average</td><td>45.5%±7.3%</td><td>50.0%±10.7%</td><td>89.2%±2.5%</td><td>91.5%±2.3%</td><td>94.3%±2.1%</td><td>92.2%±2.2%</td><td>96.3%±1.1%</td></tr><tr><td rowspan="7">SNR=-5</td><td>T01</td><td>40.3%±9.9%</td><td>35.2%±4.3%</td><td>56.4%±8.5%</td><td>59.1%±2.8%</td><td>74.5%±5.5%</td><td>58.0%±6.1%</td><td>86.1%±2.4%</td></tr><tr><td>T02</td><td>37.2%±4.0%</td><td>38.0%±5.6%</td><td>44.8%±6.4%</td><td>57.5%±6.5%</td><td>79.7%±2.0%</td><td>79.5%±5.0%</td><td>85.1%±1.9%</td></tr><tr><td>T03</td><td>16.8%±1.9%</td><td>36.3%±3.5%</td><td>49.3%±1.3%</td><td>58.9%±5.3%</td><td>62.4%±2.0%</td><td>47.9%±2.9%</td><td>80.8%±1.7%</td></tr><tr><td>T10</td><td>31.5%±3.1%</td><td>35.3%±3.3%</td><td>35.7%±2.2%</td><td>49.1%±7.2%</td><td>73.3%±2.4%</td><td>35.1%±1.7%</td><td>87.5%±1.8%</td></tr><tr><td>T12</td><td>31.9%±7.9%</td><td>34.1%±3.8%</td><td>81.4%±3.6%</td><td>80.8%±1.4%</td><td>79.1%±2.5%</td><td>45.3%±4.1%</td><td>82.7%±2.0%</td></tr><tr><td>T13</td><td>28.4%±5.4%</td><td>38.9%±7.0%</td><td>38.8%±5.4%</td><td>55.6%±4.4%</td><td>70.2%±3.9%</td><td>46.7%±6.4%</td><td>81.3%±0.9%</td></tr><tr><td>Average</td><td>31.0%±5.4%</td><td>36.3%±4.6%</td><td>51.1%±4.6%</td><td>60.2%±4.6%</td><td>73.2%±3.1%</td><td>52.1%±4.4%</td><td>83.9%±1.8%</td></tr></table>

To visualize the extracted features and validate the domain alignment capability of different methods, we used the t-distributed stochastic neighbor embedding (t-SNE) method [47]. Fig. 10 illustrates the visualization results of the feature distribution on a randomly selected task T01 using various methods, with an SNR of -5 for the target domain data. For clarity, we only display the results from four methods. Different fault types are represented b different colors, while different domains are distinguished by different shapes. This transfer learning task is particularly challenging due to the presence of severe noise interference. A depicted in Fig. 10, conventional UDA methods exhibit significant domain misalignment indicating a difficulty in aligning the feature distributions between the source and target domains due to the noise impact. In contrast, the SFDANN method demonstrates a remarkable ability to align the domains and perform accurate classification. The visualization confirms the strong domain alignment and classification capabilities of the SFDANN method in the face of challenging noise interference.

<table><tr><td>●</td><td>S0</td><td>●</td><td>S1</td><td>●</td><td>S2</td><td>●</td><td>S3</td><td>●</td><td>S4</td><td>●</td><td>S5</td><td>●</td><td>S6</td><td>●</td><td>S7</td><td>●</td><td>S8</td><td>●</td><td>S9</td></tr><tr><td>×</td><td>T0</td><td>×</td><td>T1</td><td>×</td><td>T2</td><td>×</td><td>T3</td><td>×</td><td>T4</td><td>×</td><td>T5</td><td>×</td><td>T6</td><td>×</td><td>T7</td><td>×</td><td>T8</td><td>×</td><td>T9</td></tr></table>

![](images/804164bdb4ce5e3c9eb11218e3a50f3f22654409e026cd9db5e1e0a70f19c3f6.jpg)  
(a) CORAL

![](images/97305375c73055f9a14f0ae3423a2a38b1b7050808784036f309f5c59c852483.jpg)  
(b) MK-MMD

![](images/adbfea30f31a59da58d45fe0124ccf88cfb0f5097f16b28c4cbc978b840c8afa.jpg)  
(c) DANN

![](images/65bd10df11e148a0c910a971f3bb6b1ee994765683d58a167d238c3958fca2c9.jpg)  
(d) SFDANN  
Fig. 10. Visualization of features using t-SNE for the bearing case with transfer task T01.

## (2) Diagnosis results of Case Study 2 (Slab track dataset)

Table 3 presents the diagnosis results for Case Study 2, where the UDA methods are applied on transfer tasks from labeled numerical simulation data (source domain) to unlabeled field measurement data (target domain) at different train speeds. The proposed SFDANN method achieves the highest fault diagnosis accuracy across all transfer tasks. SFDANN demonstrates an average accuracy of 96.1% for all cross-domain tasks, surpassing the best performing method by 13.3% and the worst performing method by 26.3%. These results illustrate the exceptional fault diagnosis performance of the SFDANN method in transferring knowledge from simulation scenarios to real-world scenarios in complex industrial systems.

Table 3. Recognition accuracy of slab track health state.

<table><tr><td>Speed</td><td>CNN</td><td>CORAL</td><td>MK-MMD</td><td>JAN</td><td>DANN</td><td>CDANN</td><td>SFDANN</td></tr><tr><td>20 km/h</td><td>60.0%±8.2%</td><td>83.7%±9.3%</td><td>79.3%±3.8%</td><td>73.3%±3.6%</td><td>63.0%±7.4%</td><td>68.2%±3.0%</td><td>100.0%±0</td></tr><tr><td>40 km/h</td><td>71.1%±4.3%</td><td>80.0%±10.1%</td><td>77.8%±12.6%</td><td>77.8%±8.4%</td><td>72.6%±1.8%</td><td>69.6%±3.6%</td><td>96.3%±4.1%</td></tr><tr><td>60 km/h</td><td>68.2%±3.0%</td><td>81.5%±8.1%</td><td>78.5%±6.4%</td><td>70.4%±5.7%</td><td>79.3%±6.0%</td><td>80.0%±5.0%</td><td>96.3%±4.1%</td></tr><tr><td>80 km/h</td><td>80.0%±4.4%</td><td>85.9%±3.6%</td><td>84.4%±5.4%</td><td>90.4%±1.8%</td><td>85.2%±6.6%</td><td>85.2%±5.2%</td><td>91.9%±2.8%</td></tr><tr><td>Average</td><td>69.8%±5.0%</td><td>82.8%±7.8%</td><td>80.0%±7.1%</td><td>78.0%±4.9%</td><td>75.0%±5.5%</td><td>75.8%±4.2%</td><td>96.1%±2.8%</td></tr></table>

![](images/4b3c057ebe73e26dec8eb07759889777c33f8e1405d893ba1571f1302532d0dc.jpg)  
(a) CORAL

![](images/e45f6f06cc36f60945d13493bc66e886c7f7489694d4c1c93e056eac1fc9853e.jpg)  
(b) MK-MMD

![](images/a201a3593109582851d27a4744f8782a40081d379f1ecd2326de010f59c8802b.jpg)  
(c) DANN

![](images/580e06bc4ad9da979ed479ec2f29e1f5af1dbeb93966d316f5732e6942897d25.jpg)  
(d) SFDANN  
Fig. 11. Visualization of features using t-SNE for the slab track case at a train speed of 60km/h.

Similar to Case Study 1, we used the t-SNE method to evaluate the domain alignment capabilities of different methods by visualizing the extracted features. To ensure clarity, in Fig. 11, we only present the results for four methods, using the case of train speed 60km/h as an example. As observed from the figure, the conventional UDA methods fail to achieve complete domain alignment. However, the SFDANN method demonstrates robust domain alignment and classification capability.

## 4.2.2. Ablation study

To assess the influence of the LWPT and WPT modules on the performance of the SFDANN model, we conducted an ablation study in this section. In the Section 4.2.1, we already compared SFDANN with a variant that uses two WPT modules instead of the combination of LWPT and WPT modules, which corresponds to the classical DANN. In this section, we introduce another variant of SFDANN, referred to as SFDANN-v, where the combination of LWPT and WPT modules in SFDANN is replaced with two LWPT modules. Moreover, we added different levels of Gaussian white noise to source domain of CWRU dataset as a supplement of Case study 1, where the noisy environments is only considered for target domain. The performance of SFDANN with two distinct data input strategies is evaluated in this section to analyze the impact of data input strategy on the alignment of two domains.

Table 4. Fault diagnosis accuracy using CWRU dataset in ablation study.

<table><tr><td rowspan="2" colspan="3">Task</td><td rowspan="2">SFDANN-v</td><td colspan="2">SFDANN</td></tr><tr><td>S→LWPT</td><td>T→LWPT</td></tr><tr><td rowspan="7" colspan="2">No noise</td><td>T01</td><td>99.7%±0.3%</td><td>99.9%±0.3%</td><td>99.9%±0.3%</td></tr><tr><td>T02</td><td>99.7%±0.3%</td><td>99.9%±0.3%</td><td>99.7%±0.3%</td></tr><tr><td>T03</td><td>99.9%±0.3%</td><td>99.7%±0.3%</td><td>99.7%±0.3%</td></tr><tr><td>T10</td><td>99.7%±0.3%</td><td>99.9%±0.3%</td><td>99.9%±0.3%</td></tr><tr><td>T12</td><td>99.7%±0.3%</td><td>99.7%±0.3%</td><td>99.9%±0.3%</td></tr><tr><td>T13</td><td>99.8%±0.3%</td><td>99.9%±0.2%</td><td>99.9%±0.3%</td></tr><tr><td>Average</td><td>99.8%±0.3%</td><td>99.8%±0.3%</td><td>99.8%±0.3%</td></tr><tr><td rowspan="13">SNR=0</td><td rowspan="6">Source</td><td>T01</td><td>98.1%±0.6%</td><td>98.4%±0.5%</td><td>98.1%±1.1%</td></tr><tr><td>T02</td><td>98.7%±1.1%</td><td>98.8%±1.0%</td><td>98.4%±0.8%</td></tr><tr><td>T03</td><td>99.5%±0.5%</td><td>99.2%±0.5%</td><td>97.7%±1.9%</td></tr><tr><td>T10</td><td>96.2%±2.1%</td><td>96.3%±0.9%</td><td>95.5%±0.9%</td></tr><tr><td>T12</td><td>97.6%±1.1%</td><td>98.3%±0.9%</td><td>98.6%±0.6%</td></tr><tr><td>T13</td><td>99.5%±0.6%</td><td>99.9%±0.3%</td><td>99.7%±0.3%</td></tr><tr><td rowspan="6">Target</td><td>T01</td><td>94.4%±2.1%</td><td>95.4%±1.0%</td><td>96.1%±1.0%</td></tr><tr><td>T02</td><td>95.8%±1.7%</td><td>96.3%±1.1%</td><td>97.1%±1.0%</td></tr><tr><td>T03</td><td>93.4%±1.5%</td><td>93.7%±2.5%</td><td>96.2%±1.3%</td></tr><tr><td>T10</td><td>94.4%±1.4%</td><td>93.8%±1.3%</td><td>94.0%±0.8%</td></tr><tr><td>T12</td><td>96.9%±1.4%</td><td>96.7%±0.5%</td><td>97.8%±1.3%</td></tr><tr><td>T13</td><td>95.8%±0.8%</td><td>96.4%±2.2%</td><td>96.7%±1.2%</td></tr><tr><td colspan="2">Average</td><td>96.7%±1.2%</td><td>96.9%±1.1%</td><td>97.2%±1.0%</td></tr><tr><td rowspan="12">SNR=-5</td><td rowspan="6">Source</td><td>T01</td><td>91.0%±0.9%</td><td>97.0%±1.0%</td><td>95.5%±0.7%</td></tr><tr><td>T02</td><td>87.6%±3.2%</td><td>96.5%±0.7%</td><td>94.4%±1.8%</td></tr><tr><td>T03</td><td>91.1%±1.1%</td><td>94.2%±1.9%</td><td>92.5%±0.7%</td></tr><tr><td>T10</td><td>88.8%±2.0%</td><td>93.9%±1.1%</td><td>90.9%±1.5%</td></tr><tr><td>T12</td><td>92.2%±0.6%</td><td>96.2%±1.5%</td><td>93.4%±1.4%</td></tr><tr><td>T13</td><td>92.5%±0.9%</td><td>97.2%±1.1%</td><td>93.5%±2.7%</td></tr><tr><td rowspan="6">Target</td><td>T01</td><td>79.6%±4.7%</td><td>83.5%±3.0%</td><td>86.1%±2.4%</td></tr><tr><td>T02</td><td>74.7%±3.3%</td><td>81.3%±1.0%</td><td>85.1%±1.9%</td></tr><tr><td>T03</td><td>72.1%±5.7%</td><td>78.7%±1.0%</td><td>80.8%±1.7%</td></tr><tr><td>T10</td><td>83.2%±1.5%</td><td>84.6%±1.5%</td><td>87.5%±1.8%</td></tr><tr><td>T12</td><td>78.2%±1.7%</td><td>80.3%±1.7%</td><td>82.7%±2.0%</td></tr><tr><td>T13</td><td>77.1%±4.0%</td><td>79.1%±3.2%</td><td>81.3%±0.9%</td></tr></table>

<table><tr><td></td><td>Average</td><td>84.0%±2.5%</td><td>88.5%±1.6%</td><td>88.6%±1.6%</td></tr></table>

Note: “S→LWPT” indicates the data input strategy of feeding the source domain data into the LWPT module of smart filter; “T→LWPT” indicates the data input strategy of feeding the target domain data into the LWPT module of smart filter; “Source” means that the noise is added to the source domain; “Target” means that the noise is added to the target domain

Table 5. Fault diagnosis accuracy using slab track dataset in ablation study.

<table><tr><td rowspan="2">Speed</td><td rowspan="2">SFDANN-v</td><td colspan="2">SFDANN</td></tr><tr><td>S→LWPT</td><td>T→LWPT</td></tr><tr><td>20 km/h</td><td>87.4%±5.5%</td><td>99.3%±1.5%</td><td>100.0%±0</td></tr><tr><td>40 km/h</td><td>75.6%±1.8%</td><td>91.1%±6.5%</td><td>96.3%±4.1%</td></tr><tr><td>60 km/h</td><td>82.2%±1.5%</td><td>91.9%±8.2%</td><td>96.3%±4.1%</td></tr><tr><td>80 km/h</td><td>86.7%±3.8%</td><td>89.6%±1.5%</td><td>91.9%±2.8%</td></tr><tr><td>Average</td><td>83.0%±3.2%</td><td>93.0%±4.4%</td><td>96.1%±2.8%</td></tr></table>

Note: “S→LWPT” indicates the data input strategy of feeding the source domain data into the LWPT module of smart filter; “T→LWPT” indicates the data input strategy of feeding the target domain data into the LWPT module of smart filter.

The experimental results of SFDANN-v and SFDANN on the bearing dataset and the slab track dataset are presented in Table 4 and 5. These results demonstrate that the overall performance of SFDANN is superior to that of SFDANN-v. On the bearing dataset, when there is no Gaussian white noise in the data, both SFDANN and SFDANN-v exhibit nearly identical fault diagnosis accuracy. However, in the presence of Gaussian white noise in either the source domain or the target domain, SFDANN consistently outperforms SFDANN-v in most cases. For the slab track dataset, SFDANN demonstrates better domain adaptation ability compared to SFDANN-v. This suggests that preserving either the source domain data or the target domain data, and promoting the other domain data to resemble the preserved data, yields improved domain alignment results compared to using two LWPT modules to align both domains simultaneously. This observation can be attributed to the fact that the combination of LWPT and WPT modules has fewer learnable parameters compared to two LWPT modules. As a result, it reduces training difficulty and enhances training stability, leading to better performance in SFDANN.

Table 4 additionally exhibits the fault diagnosis results of bearings using two data input strategies with the SFDANN method. It can be observed that the diagnostic accuracy is similar when there is no Gaussian noise in the dataset, regardless of the input strategy employed. However, when noise is present in the source domain, the overall diagnostic accuracy is higher when the noisy source domain data is inputted into the LWPT module of the smart filter, while the noise-free target domain data is inputted into the WPT module. Conversely, when noise exists in the target domain, the diagnostic performance is generally better when the noisy target domain data is inputted into the LWPT module. This phenomenon can be attributed to the guidance loss, which drives the time-frequency characteristics of the input data in the LWPT module to closely align with those of the input data in the WPT module. If the data inputted to the WPT module is less affected by noise, the data inputted to the LWPT module will undergo a certain level of denoising during the model training process, leading to improved fault diagnosis performance compared to conventional UDA methods. However, if the data inputted to the WPT module is heavily impacted by noise, the data inputted to the LWPT module experiences a certain level of noise augmentation during training. While the SFDANN with this data input strategy still outperforms conventional UDA methods in most cases, the degree of performance improvement is not as significant as that achieved by the data input strategy capable of denoising the noisy data.

Similar phenomenon can be observed in the state recognition results of slab tracks. Table 5 indicates that the state recognition accuracy obtained by inputting the field measurement data into the LWPT module of the smart filter and the numerical simulation data into the WPT module is higher than that obtained by the other data input strategy. This is mainly because numerical simulation signals are purer and easier to be classified correctly by the classifier than field measurement signals. Therefore, promoting the field measurement signals similar to the numerical simulation signals is more conducive to fault diagnosis than promoting the numerica simulation signals similar to the field measurement signals. As a result, it can be concluded from Table 4 and 5 that the optimal data input strategy for SFDANN is to feed the data from the noisier domain into the LWPT module of the smart filter, while inputting the data from another domain into the WPT module of the smart filter.

## 4.2.3. Effect of smart filter

To further investigate the effectiveness of the smart filter in SFDANN, we analyzed the spectral changes of data passing through the smart filter. We focused a UDA task from operation condition 0 to 2 in the bearing case study, where the Gaussian white noise was added to the target domain data to create noisy signals with an SNR of -5. In this analysis, we input noisy target domain data (i.e., data from operation condition 2) into the LWPT module, while the source domain data (i.e., data from operation condition 0) was input into the WPT module. After training the SFDANN model, we selected the original source domain data, original target domain data, reconstructed source domain data, and reconstructed target domain data corresponding to label 2 as examples to explore the effect of the smart filter. To assess the impact of the smart filter, we calculated the mean frequency spectra for these four groups of data and plotted them in Fig. 12(a), while the corresponding one-third octave spectra were shown in Fig. 12(b).

![](images/b919b3453a969d61a1a44c6f87853791d08f9f7f1d47453fc94d68fc430e1c47.jpg)  
(a) Frequency spectra

![](images/6256caf9b12bdf9a0154eae7e72f3e6911d99c858985e1df20e6abd46d18c0dd.jpg)  
(b) One-third octave spectra  
Fig. 12. Source and target domain data corresponding to label 2 from CWRU dataset: “Rec” denotes “Reconstructed”.

The frequency spectra of the original and reconstructed source domain data completely overlap in Fig. 12, indicating that there is no change in the source domain data before and after passing through the smart filter. However, the original target domain data is severely distorted by noise, leading to a significant deviation in its frequency spectrum compared to that of the source domain data. The reconstructed target domain data exhibits a frequency spectrum shape that is closer to that of the source domain data when compared to the original target domain data. This suggests that the target domain data has increased similarity with the source domain data after passing through the smart filter. Such alignment between the source and target domains is beneficial for subsequent feature extractors to capture shared features and further improve fault diagnosis accuracy.

![](images/ee1c107bc6d20d4c7b6722a2958e839741c61232d2eef08a6db6ae613175e43f.jpg)  
(a) Frequency spectra

![](images/bd60bce69f762ee0615202abfab3077051555da4c549d0213a9e5d0b632a2bab.jpg)  
(b) One-third octave spectra  
Fig. 13. Source and target domain data corresponding to label 2 from slab track dataset: “Rec” denotes “Reconstructed”.

For the slab track case, we select a UDA task with a train speed of 60 km/h as an example. In this case, the measurement data serves as the target domain data and is input into the LWPT module, while the numerical simulation data serves as the source domain data and is input into the WPT module. After the model training, we obtain original and reconstructed data with label 2 for both domains. Their mean frequency spectra and corresponding one-third octave spectra are plotted in Fig. 13. From Fig. 13(b), it is evident that, at the one-third octave scale, the shape of the reconstructed source domain data is closer to that of the target domain data compared to the original source data. This demonstrates the effectiveness of the smart filter in promoting the similarity between the data from the two domains. This increased similarity is beneficial for SFDANN model to learn domain-invariant and discriminative features, leading to improved performance in fault diagnosis tasks.

## 5. Conclusion

We propose SFDANN, an unsupervised domain adaptation method to address the challenge of domain misalignment in fault diagnosis scenarios characterized by different noise levels. SFDANN consists of a smart filter based on LWPT and WPT, a feature extractor, a domain discriminator, and a classifier. The effectiveness of SFDANN is demonstrated through case studies on bearing fault diagnosis in noisy environments and on simulation to real transfer of slab track fault diagnosis in a train-track-bridge coupling vibration system. The latter case study involves the transfer of models from labeled numerical simulations to unlabeled field measurements. Furthermore, this study analyzes the impact of two data input strategies, (1) inputting source and target domain data separately into LWPT and WPT modules, and (2) the reverse input strategy, on fault diagnosis results.

The results indicate that the designed smart filter can dynamically approximate the source and target signals in the time-frequency domain, allowing SFDANN to learn domain-invariant and discriminative features. Inputting the data from the domain with more severe noise interference into the LWPT module of the smart filter, and the data from the other domain into the WPT module, proves more beneficial for improving the overall final fault diagnosis accuracy. Furthermore, the proposed SFDANN method outperforms other UDA methods in terms of domain adaptability and diagnostic performance in industrial scenarios with unknown types and levels of noise.

In comparison to other UDA methods, the primary drawback of our proposed method is the additional computational cost incurred during the training of the newly introduced module, smart filter. However, it is important to note that the increase in the computational cost is only significant when the number of signal decomposition layers becomes excessively large. Ou future research will focus on partial domain adaptation for fault diagnosis. Additionally, we plan to evaluate the performance of SFDANN on other challenging fault knowledge transfe tasks, such as utilizing signals measured by different types of sensors. The proposed SFDANN algorithm is very versatile and flexible which makes it applicable across a wide range of potential applications, especially those susceptible to noise interference. This applicability extends beyond the specific fault diagnosis scenarios showcased in this study. Illustrative examples include transfer learning tasks in noisy speech recognition and health-oriented applications, particularly involving the processing of electroencephalogram (EEG) and electrocardiogram (ECG) data.

## Acknowledgement

This study was supported by the National Natural Science Foundation of China (grant number 52178432) and China Scholarship Council (grant number 202106260178).

## Reference

[1] Zhao B, Zhang X, Wu Q, Yang Z, Zhan Z. A novel unsupervised directed hierarchical graph network with clustering representation for intelligent fault diagnosis of machines. Mechanical Systems and Signal Processing. 2023;183:109615.

[2] Zhang G, Kong X, Du J, Wang J, Yang S, Ma H. Adaptive multispace adjustable sparse filtering: A sparse feature learning method for intelligent fault diagnosis of rotating machinery. Engineering Applications of Artificial Intelligence. 2023;120:105847.

[3] Shao H, Li W, Cai B, Wan J, Xiao Y, Yan S. Dual-Threshold Attention-Guided Gan and Limited Infrared Thermal Images for Rotating Machinery Fault Diagnosis Under Speed Fluctuation. IEEE Transactions on Industrial Informatics. 2023:1-10.

[4] Fink O, Wang Q, Svensén M, Dersin P, Lee W-J, Ducoffe M. Potential, challenges and future directions for deep learning in prognostics and health management applications. Engineering Applications of Artificial Intelligence. 2020;92:103678.

[5] Han Y, Qi W, Ding N, Geng Z. Short-Time Wavelet Entropy Integrating Improved LSTM for Fault Diagnosis of Modular Multilevel Converter. IEEE Transactions on Cybernetics. 2022;52:7504-12.

[6] Rombach K, Michau G, Fink O. Controlled generation of unseen faults for Partial and Open-Partial domain adaptation. Reliability Engineering & System Safety. 2023;230:108857.

[7] Yan X, She D, Xu Y, Jia M. Deep regularized variational autoencoder for intelligent fault diagnosis of rotor–bearing system within entire life-cycle process. Knowledge-Based Systems. 2021;226:107142.

[8] Che C, Wang H, Ni X, Xiong M. Few-shot structural repair decision of civil aircraft based on deep meta-learning. Engineering Applications of Artificial Intelligence. 2023;125:106695.

[9] Shao H, Li W, Cai B, Wan J, Xiao Y, Yan S. Dual-threshold attention-guided GAN and limited infrared thermal images for rotating machinery fault diagnosis under speed fluctuation. IEEE Transactions on Industrial Informatics. 2023.

[10] Zhao Z, Zhang Q, Yu X, Sun C, Wang S, Yan R, et al. Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study. IEEE Transactions on Instrumentation and Measurement. 2021;70:1-28.

[11] Li W, Huang R, Li J, Liao Y, Chen Z, He G, et al. A perspective survey on deep transfer learning for fault diagnosis in industrial scenarios: Theories, applications and challenges. Mechanical Systems and Signal Processing. 2022;167:108487.

[12] Li T, Zhao Z, Sun C, Yan R, Chen X. Domain Adversarial Graph Convolutional Network for Fault Diagnosis Under Variable Working Conditions. IEEE Transactions on Instrumentation and Measurement. 2021;70:1-10.

[13] Lu W, Fan H, Zeng K, Li Z, Chen J. Self‐supervised domain adaptation for cross‐domain fault diagnosis. International Journal of Intelligent Systems. 2022;37:10903-23.

[14] Michau G, Fink O. Unsupervised transfer learning for anomaly detection: Application to complementary operating condition transfer. Knowledge-Based Systems. 2021;216:106816.

[15] Zhu J, Chen N, Shen C. A New Multiple Source Domain Adaptation Fault Diagnosis Method Between Different Rotating Machines. IEEE Transactions on Industrial Informatics. 2021;17:4788-97.

[16] Wang Q, Michau G, Fink O. Domain Adaptive Transfer Learning for Fault Diagnosis. 2019 Prognostics and System Health Management Conference (PHM-Paris)2019. p. 279-85.

[17] Wang Q, Michau G, Fink O. Missing-Class-Robust Domain Adaptation by Unilateral Alignment. IEEE Transactions on Industrial Electronics. 2021;68:663-71.

[18] Chen P, Zhao R, He T, Wei K, Yuan J. Unsupervised structure subdomain adaptation based the Contrastive Cluster Center for bearing fault diagnosis. Engineering Applications of Artificial Intelligence. 2023;122:106141.

[19] Xiao Y, Shao H, Han S, Huo Z, Wan J. Novel Joint Transfer Network for Unsupervised Bearing Fault Diagnosis From Simulation Domain to Experimental Domain. IEEE/ASME Transactions on Mechatronics. 2022;27:5254-63.

[20] Lou Y, Kumar A, Xiang J. Machinery Fault Diagnosis Based on Domain Adaptation to Bridge the Gap Between Simulation and Measured Signals. IEEE Transactions on Instrumentation and Measurement. 2022;71:1-9.

[21] Liu J, Cao H, Su S, Chen X. Simulation-Driven Subdomain Adaptation Network for bearing fault diagnosis with missing samples. Engineering Applications of Artificial Intelligence. 2023;123:106201.

[22] Wang Q, Taal C, Fink O. Integrating Expert Knowledge With Domain Adaptation for Unsupervised Fault Diagnosis. IEEE Transactions on Instrumentation and Measurement. 2021;71:1-12.

[23] Lu N, Xiao H, Sun Y, Han M, Wang Y. A new method for intelligent fault diagnosis of machines based on unsupervised domain adaptation. Neurocomputing. 2021;427:96-109.

[24] Chen Z, Liao Y, Li J, Huang R, Xu L, Jin G, et al. A Multi-Source Weighted Deep Transfer Network for Open-Set Fault Diagnosis of Rotary Machinery. IEEE Transactions on Cybernetics. 2023;53:1982-93.

[25] Yao Y, Chen Q, Gui G, Yang S, Zhang S. A hierarchical adversarial multi-target domain adaptation for gear fault diagnosis under variable working condition based on raw acoustic signal. Engineering Applications of Artificial Intelligence. 2023;123:106449.

[26] Biggio L, Bendinelli T, Kulkarni C, Fink O. Dynaformer: A Deep Learning Model for Ageing-aware Battery Discharge Prediction. arXiv preprint arXiv:220602555. 2022.

[27] Ye Z, Yu J. Deep Negative Correlation Multisource Domains Adaptation Network for Machinery Fault Diagnosis Under Different Working Conditions. IEEE/ASME Transactions on Mechatronics. 2022;27:5914-25.

[28] Xiao D, Qin C, Yu H, Huang Y, Liu C, Zhang J. Unsupervised machine fault diagnosis for noisy domain adaptation using marginal denoising autoencoder based on acoustic signals. Measurement. 2021;176:109186.

[29] Michau G, Frusque G, Fink O. Fully learnable deep wavelet transform for unsupervised monitoring of high-frequency time series. Proceedings of the National Academy of Sciences. 2022;119:e2106598119.

[30] Frusque G, Fink O. Learnable Wavelet Packet Transform for Data-Adapted Spectrograms. ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal

Processing (ICASSP)2022. p. 3119-23.

[31] Dai B, Frusque G, Li Q, Fink O. Acceleration-Guided Acoustic Signal Denoising Framework Based on Learnable Wavelet Transform Applied to Slab Track Condition Monitoring. IEEE Sensors Journal. 2022;22:24140-9.

[32] Ganin Y, Lempitsky V. Unsupervised Domain Adaptation by Backpropagation. In: Francis B, David B, editors. Proceedings of the 32nd International Conference on Machine Learning. Proceedings of Machine Learning Research: PMLR; 2015. p. 1180--9.

[33] Zhang Y, Ji JC, Ren Z, Ni Q, Gu F, Feng K, et al. Digital twin-driven partial domain adaptation network for intelligent fault diagnosis of rolling bearing. Reliability Engineering & System Safety. 2023;234:109186.

[34] Nejjar I, Geissmann F, Zhao M, Taal C, Fink O. Domain adaptation via alignment of operation profile for remaining useful lifetime prediction. arXiv preprint arXiv:230201704. 2023.

[35] Tu Y, Mak MW, Chien JT. Variational Domain Adversarial Learning With Mutual Information Maximization for Speaker Verification. IEEE/ACM Transactions on Audio, Speech, and Language Processing. 2020;28:2013-24.

[36] Yao F, Wang Y. Domain-specific sentiment analysis for tweets during hurricanes (DSSA-H): A domain-adversarial neural-network-based approach. Computers, Environment and Urban Systems. 2020;83:101522.

[37] Ganin Y, Ustinova E, Ajakan H, Germain P, Larochelle H, Laviolette F, et al. Domainadversarial training of neural networks. The journal of machine learning research. 2016;17:2096-30.

[38] Smith WA, Randall RB. Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. Mechanical Systems and Signal Processing. 2015;64-65:100-31.

[39] Long M, Zhu H, Wang J, Jordan MI. Deep transfer learning with joint adaptation networks. International conference on machine learning: PMLR; 2017. p. 2208-17.

[40] Long M, Cao Y, Cao Z, Wang J, Jordan MI. Transferable Representation Learning with Deep Adaptation Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2019;41:3071-85.

[41] Sun B, Saenko K. Deep coral: Correlation alignment for deep domain adaptation. Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14: Springer; 2016. p. 443-50.

[42] Long M, Cao Z, Wang J, Jordan MI. Conditional adversarial domain adaptation. Advances in neural information processing systems. 2018;31.

[43] Liu C, Gryllias K. Simulation-Driven Domain Adaptation for Rolling Element Bearing Fault Diagnosis. IEEE Transactions on Industrial Informatics. 2022;18:5760-70.

[44] Zhang Y, Han D, Tian J, Shi P. Domain adaptation meta-learning network with discardsupplement module for few-shot cross-domain rotating machinery fault diagnosis. Knowledge Based Systems. 2023;268:110484.

[45] Standardization IOf. ISO3095:2013 - Acoustics - Railway applications - Measurement of noise emitted by railbound vehicles. Geneva, Switzerland: International Organization for Standardization; 2013.

[46] Xu YL, Li Q, Wu DJ, Chen ZW. Stress and acceleration analysis of coupled vehicle and long-span bridge systems using the mode superposition method. Engineering Structures. 2010;32:1356-68.

[47] van der Maaten L, Hinton G. Viualizing data using t-SNE. Journal of Machine Learning Research. 2008;9:2579-605.