## Accepted Manuscript

A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network

Yalin Wang, Zhuofu Pan, Xiaofeng Yuan, Chunhua Yang, Weihua Gui

![](images/07f5968c8a60003b065095d7e9ef6b691ccc735aad2bc92a0f3c3942f5dd9527.jpg)

PII: S0019-0578(19)30290-3

DOI: https://doi.org/10.1016/j.isatra.2019.07.001

Reference: ISATRA 3254

To appear in: ISA Transactions

Received date : 25 January 2019

Revised date : 1 July 2019

Accepted date : 1 July 2019

Please cite this article as: Y. Wang, Z. Pan, X. Yuan et al., A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network. ISA Transactions (2019), https://doi.org/10.1016/j.isatra.2019.07.001

This is a PDF file of an unedited manuscript that has been accepted for publication. As a service to our customers we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and review of the resulting proof before it is published in its final form. Please note that during the production process errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.

# A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network

Yalin Wang, Zhuofu Pan, Xiaofeng Yuan<sup>\*</sup>, Chunhua Yang, Weihua Gui

School of Automation, Central South University, Changsha, 410083, Hunan, P. R. China

## Abstract

Deep learning networks have been recently utilized for fault detection and diagnosis (FDD) due to its effectiveness in handling industrial process data, which are often with high nonlinearities and strong correlations. However, the valuable information in the raw data may be filtered with the layer-wise feature compression in traditional deep networks. This cannot benefit for the subsequent fine-tuning phase of fault classification. To allevi this problem, an extended deep belief network (EDBN) is proposed to fully exploit usefu l information in the raw data, in which raw data is combined with the hidden features as inputs to each extended restricted Boltzmann machine (ERBM) during the pre-training phase. Then, a dynamic EDBN-based fault classifier is constructed to take the dynamic characteristics of process data into consideration. Finally, to test the performance of the proposed method, it is $\mathrm { \cdot p } \mathrm { p } _ { \star } ^ { \star } \mathrm { \ r ^ { \ v } d }$ to the Tennessee Eastman (TE) process for fault classification. By comparing EDBN and DBN under different network structures, the results show that EDBN has better feature extra ction and fault classification performance than traditional DBN.

Keywords: Fault Detection and Diagnosis; Deep learning; Deep belief network; Extended DBN.

## 1. Introduction

Thanks to advanced process control systems, modern chemical processes have become highly automated, which allows continuous production with low cost and high safety. However, even for chemical plants equipped with distributed control systems and safety instrument systems, there still have many accidents reported in recent years, which can easily result in serious fatalities, asset damage and environmental destruction [1, 2]. This is due to the reason that it often largely dependent on process operators to monitor and handle abnormal situations in chemical processes. However, it is difficult to ensure that operators can always discover abnormal situations and make the right operation timely. Therefore, as a real-time, reliable and efficient advanced monitoring means, fault detection and diagnosis (FDD) [3] has played an important role in process monitoring and system fault tracking, which has received extensive attention from academia and industry. Moreover, FDD can give valuable information for timely process control and optimization [4-10].

In FDD, fault detection aims to monitor a system and identify when a fault occurs in the processes. Fault diagnosis concerns about which fault type the new identified fault status belongs to and what the root fault cause is. Hence, it is an important step to carry out fault classification to category. Generally, fault classification can be treated as a multiclassification task, which can not only detect whether there is a fault, but also identify the category of the fault type. Typically, fault classification approaches can be divided into the model-based, knowledge-based and data-driven approaches [3, 11]. Compared to model-based and knowledgebased met 1。 which strongly depend on process physicochemical knowledge, data-driven methods can build fault classification models by extracting valuable information from process history data. By far, it has become more and more popular for data-driven methods in handling fault classification problems due to the large amount of data that can be collected in modern chemical plants.

In general, the massive process data collected in chemical plants are with high redundancy and strong correlations, in which valuable information is submerged and needs to be mined effectively. How to extract useful information is the key step to establish a stable and robust model, as well as improve the classification performance [12]. During the past years, many feature extractors have been developed based on machine learning approaches and multivariate statistical methods. Canonical correlation analysis (CCA) [13, 14], principal component analysis (PCA) [15-17], Fisher discriminant analysis (FDA) [18]and partial least squares (PLS) [19], and are some of the most widely used linear feature representation method Alternatively, nonlinear feature extractors like kernel PCA [20], kernel PLS [21], artificial neural networks (ANN) [22, 23] and support vector machine (SVM) [24] have also been applied for describing more complicated data characteristics. These traditional feature extractors can be regarded as shallow learning networks, which learn useful features for many pattern recognition tasks. However, they are limited in their representation for massive data in large-scale complex processes, especially in the era of big data. In contrast, deep multi-layer networks are more capable of having excellent expression capability, which can extract more useful and predictive features for highly complex systems.

Actually, multi-layer networks did not function well in the early days owing to the gradient exploding and vanishing problems during the network training stage. This bottleneck is not broken until the de p learning technique was developed by Hinton et al. in 2006, with which the deep networks are first unsupervised layer-wise pre-trained and then fine-tuned by back-propagation (BP) algorithm. Since then, deep learning has raised great attention in areas like speech recognition, image recognition, and natural language processing, in which it has shown powerful ability in identifying and expressing deep features with different complex tasks [25].

In the last five years, deep learning networks have also been used in chemical industry for process data modeling like soft sensor and fault classification applications [26-30]. For example, Jiang et al. [31] utilized stacked denoising auto-encoder (SDAE) to build a chemical fault classification model, in which an active learning strategy was proposed that allows the model to select the informative data for online fine-tuning. Then, Zhang et al. [11] introduced deep belief network (DBN) for fault classification in benchmarked Tennessee Eastman (TE) process. They employed mutual information approach to extract features and established DBN sub-networks for each state in TE process. Also, Jiang et al. [32] ap plied stacked sparse auto-encoders (SSAE) for TE process fault classification. They properly considered the dynamic characteristics of the process data and provided a semi-supervised learning strategy for the deep network. Moreover, Wu et al. [1] exploited convolutional neural network (CNN) in TE process for fault classification application. Recently, Yu et al. [33] used DBN to monitor and enhance the abnormal fluctuation information, and the enhanced features extracted via DBN are used for fault detection in TE process. These methods show the great potential and outstanding performance in process data modeling with deep learning technique

Although many research works have been published on deep learning for fault classification problems, there are still some specific limitations in existing deep learning models. For instance, the reported de ep-learning-based techniques are often directly used to divide fault category without seeking a better feature representation to fit the dynamic and nonlinear characteristics of industria data. In addition, as hierarchical features are extracted layer by layer in deep learning, successive

##

feature compression may result in the loss of valuable information in the raw data, which is disadvantageous for fault classification in the subsequent fine-tuning phase. To fully exploit the potential valuable information in the raw data, an extended deep belief network (EDBN) is designed for feature representation and fault classification in this paper. In EDBN, features are progressively extracted through stacking multiple extended restricted Boltzmann machines (ERBM) during the pre-training phase. Then, the fault category results are outputted by an additional classification layer located at the top hidden layer of the last ERBM in the fine-tuning phase. For each extended RBM (ERBM), the hidden features from the previous RBM/ERBM are combined with the raw input data to serve as the new inputs to the current ERBM. By adding the raw data to each ERBM, the layerwise features related to the raw data can be $\mathrm { ~ \textit ~ { ~ P ~ } ~ } { } \cdot _ { \infty } \dot { \mathrm { ~ \textit ~ { ~ o ~ } ~ } }$ extracted, which is helpful to mine consideration, a dynamic EDBN-based modeling framework is further built for fault classification. The performance of the designed EDBN is shown on the well-known benchmarked TE process.

The rest of the paper is structured as follows. Section 2 simply revisits the structure of RBM and DBN. The extended deep belief network is then introduced and explained in Section 3, in which the detailed procedure of EDBN-based fault classification modeling is also described. After that, the proposed ED BN is utilized for fault classification on the TE benchmark in Section 4. At last, conclusions are summarized for this paper in Section 5.

## 2. Deep belief network

## 2.1.Restricted Boltzmann machine

Restricted Boltzmann machine (RBM) [34], as the basic module of deep belief network, mainly consists of one visible layer and a hidden layer. Different from Boltzmann machine (BM) with a fully connected graph structure, RBM limits the interconnection of peer nodes to ensure their mutual independence. The network structure of RBM is given in Fig. 1.

![](images/5606279e0ca5b755675ea25a568c002512a6cb6cbc6a1f3c9e1c9d2701e4beab.jpg)  
Fig. 1. The diagram of network structure for RBM

RBM is a probabilistic model, whose parameters are composed of the weights and biases. density can be represented as

$$
p (\mathbf {v}, \mathbf {h}) = e ^ {- E (\mathbf {v}, \mathbf {h})} / \iint_ {\mathbf {v}, \mathbf {h}} e ^ {- E (\mathbf {v}, \mathbf {h})},\tag{1}
$$

where E v h,  is the so-called energy function, whose form depends on the unit type of RBM. The commonly used types are binary unit and $r _ { \mathrm { \cdots \thinspace c s i a n . } }$ unit. When dealing with continuous-value data, a good choice is to apply Gaussian-Gaussian energy function (Eq. (2)) rather than binary-binary energy function (Eq. (3)). The forms can be expressed as [35]

$$
E (\mathbf {v}, \mathbf {h}) = \sum_ {i \in v i s} \frac {\left(v _ {i} - a _ {i}\right) ^ {2}}{2 \sigma_ {i} ^ {2}} + \sum_ {j \in h u d} \frac {\left(h _ {j} - b _ {j}\right) ^ {2}}{2 \sigma_ {j} ^ {2}} - \sum_ {i \in v i s, j \in h i d} \frac {v _ {i}}{\sigma_ {i}} \frac {h _ {j}}{\sigma_ {j}} w _ {i j},\tag{2}
$$

$$
E (\mathbf {v}, \mathbf {h}) = - \sum_ {i \in v i s} a _ {i} v _ {i} - \sum_ {j \in v i s, j \in h i d} h _ {j} h _ {j} - \sum_ {i \in v i s, j \in h i d} v _ {i} h _ {j} w _ {i j},\tag{3}
$$

where $h _ { j }$ and $\nu _ { i }$ respectively; b and $a _ { i } .$ refer to their corresponding bias terms; $w _ { i j }$ is the weight that connects $\nu _ { i }$ and $h _ { j }$ . In Gaussian energy function, $\sigma _ { i }$ and $\sigma _ { j }$ are the standard deviation terms of the Gaussian noises duced in the $i ^ { t h }$ visible unit and the $j ^ { t h }$ hidden unit, which usually take the value as $\sigma _ { i } = \sigma _ { j } = 1$ With the joint probabilistic distribution $p ( \mathbf { v } , \mathbf { h } )$ and its marginal probabilistic distributions p p( ), ( )v h , the conditional probabilistic distribution p p( | ), ( | )h v v h can be obtained

by Bayesian inference as

$$
\begin{array}{l} p (\mathbf {h} \mid \mathbf {v}) = p (\mathbf {v}, \mathbf {h}) / p (\mathbf {v}) = e ^ {- E (\mathbf {v}, \mathbf {h})} / \int_ {\mathbf {h}} e ^ {- E (\mathbf {v}, \mathbf {h})} \\ p (\mathbf {v} \mid \mathbf {h}) = p (\mathbf {v}, \mathbf {h}) / p (\mathbf {h}) = e ^ {- E (\mathbf {v}, \mathbf {h})} / \int_ {\mathbf {v}} e ^ {- E (\mathbf {v}, \mathbf {h})}. \end{array}\tag{4}
$$

For the Gaussian units, their conditional distribution should obey a normal distribution with following forms [11]

$$
p \left(h _ {j} \mid \mathbf {v}\right) \sim N \left(\mu_ {j}, \sigma_ {j}\right), \mu_ {j} = b _ {j} + \sigma_ {j} \sum_ {i \in v i s} \frac {v _ {i}}{\sigma_ {i}} w _ {i j} = l i n e (\mathbf {v})
$$

$$
p \left(v _ {i} \mid \mathbf {h}\right) \sim N \left(\mu_ {i}, \sigma_ {i}\right), \mu_ {i} = a _ {i} + \sigma_ {i} \sum_ {j \in h i d} \frac {h _ {j}}{\sigma_ {j}} w _ {i j} = l i n e (\mathbf {h})\tag{5}
$$

## 2.2.Deep belief network

Deep belief network is composed of multiple stacked RBMs and an output layer added on the last RBM, the structure of which is illustrated in Fig. 2. The training procedure of DBN includes two stages: the layer-wise unpervised pre-training and fine-tuning. During the pre-training phase, a layer-wise greedy technique is employed to train the RBMs. Once the training of the previous RBM is completed, its hidden layer is used as the visible layer for the next RBM. In such a way, RBMs can be trained one by one until the last RBM is trained. Each RBM is trained by maximizing the probability of its input $\mathrm { d } \mathfrak { e } ^ { \prime } { \bf \Pi } _ {  } \dot { \mathfrak { n } }$ which contrastive divergence (CD) algorithm [36, 37] is exploited to update the parameters. After pretraining, a classification layer is added to the last hidden layer and DBN will be further fine-tuned via minimizing the error between estimated output values and labels. The BP algorithm is implemented to progressively pass the error from the last layer to the bottom input layer. In this way, the parameters of the whole network can be updated.

![](images/e66894245b15244ef8709237b9d0589218713a7333b584fee30e0f280b5a45f6.jpg)  
Fig. 2. An illustration of DBN with two hidden layers

## 3. Extended deep belief network for fault classification

## 3.1.Extended deep belief network

Although DBN can extract effective deep features and achieve fast convergence by performing pre-training and fine-tuning, there is still room for improvement of learning performance. According to the information bottleneck theory, as the number of neural network layers increases, the relevant information between the extracted deep features and the raw data will be less and less. Therefore, in the layer-wise compression procedure of most existing deep networks, a lot of useful information in the raw data may be usually lost in high layers. To alleviate this problem, an extended deep belief network (EDBN) is proposed to sufficiently capture the valuable information in the raw data, which is stacked by multiple ERBMs. By utilizing the raw data as additional inputs to the visible layer of each ERBM for pre-training, the raw input data can participate in the whole compression procedure. Thus, he racted deep features are highly related with the raw data, where the potential valuable information is fully reserved. Compared with existing methods, EDBN can repeatedly distill valuable information from raw data and can provide deep compressed representations that are highly correlated with the raw data. Moreover, EDBN can achieve higher accuracy and lower false positive rate for classification task. The structure of EDBN can be seen in Fig. 3.

The training procedure of EDBN is also composed of pre-training and fine-tuning stages. During the pre-training stage, the raw data is added to the visible layer of each extended RBM (ERBM). In the left subfigure of Fig. 3, the inputs of each ERBM is composed of the raw data expressed by purple circles and the previous hidden features described by blue circles. Also, its weight matrix consists of two parts of $\mathbf { w } _ { I }$ and ${ \bf w } _ { H }$ , in which the former connects the raw data with the hidden layer, and the latter connects the hidden feature of the previous ERBM with the hidden layer. Then, the maximum likelihood rule and CD algorithm will be used to update the parameters of each ERBM. In this way, network ameters can be well learned, and their values will be used as the initialization for the fine-tu In the fine-tuning stage, another output layer is added for classification, and the extended raw variable nodes will be dropped out in each hidden layer. Finally, the loss between predicted outputs and labels will be calculated and BP algorithm is iteratively executed to update the network parameters by minimizing the loss function. The detailed training procedures are described in Section 3.2 and 3.3.

![](images/823abcbd339f071765067b2adb12ab0908801fd7883f8340370ff1907e815ebb.jpg)  
Fig. 3 The structure of EDBN during the pre-training phase and fine-tuning phase

## 3.2.Pre-training

The pre-training procedure of EDBN is to train these ERBMs one by one. As for the first ERBM, it is actually an RBM since there is no need to extend raw data. Every ERBM updates their weight and biases based on the maximum likelihood theory and k -step contrastive divergence obtain proper network parameters $( \mathbf { \boldsymbol { \theta } } = \{ \mathbf { w } , \mathbf { a } , \mathbf { b } \}$ in ERBM) that best fit the distribution of the input data. By calculating the logarithmic partial derivatives of $p ( \mathbf { v } = \mathbf { v } _ { \mathbf { \delta } _ { \mathit { u t a } } } )$ , the gradient update formula of network parameters can be obtained as Eq. (6), which is given by Fischer [40]. Since it is quite difficult to calculate the second terms in Eq. (6) accurately, CD- k algorithm based on Markov Chain and Monte Carlo sampling is employed to obtain the approximate solution of this problem. The main idea is to transfer the data between visible and hidden layer for k times, so that the network states h and v can mostly represent the distribution of the model after k iterations. As shown in Fig. 4, the input of the ERBM ${ \mathbf { v } } _ { d a _ { \mathrm { t } } }$ can be expressed as $\mathbf { v } ^ { ( 0 ) }$ . Via sampling from the probability $p \big ( \mathbf { h } \mid \mathbf { v } ^ { ( 0 ) } \big )$ $\mathbf { h } ^ { ( 0 ) }$ . Similarly, $\mathbf { v } ^ { ( 1 ) }$ can be obtained via sampling from $p \big ( \mathbf { v } \mid \mathbf { \mid } \big )$ . The learning process from $\mathrm { ~ \bf ~ v ~ } ^ { ( 0 ) } \mathrm { ~ \bf ~ t o ~ } \mathrm { ~ \bf ~ v ~ } ^ { ( 1 ) }$ is also called one step to Markov theory, after k -step Gibbs sampling (k →∞) , it will $\mathbf { v } ^ { ( k ) }$ at this time will reflect the distribution of model. By further approximating the expectations over $p ( \mathbf { v } )$ in Eq. (6) with the single sample $\mathbf { v } ^ { ( k ) }$ (Monte Carlo sampling), Eq. (6) can be simply converted to Eq. (7) as

$$
\frac {\partial \log_ {2} (\mathbf {v} - \mathbf {v} _ {\text {data}})}{\partial \theta} = - \int_ {\mathbf {h}} p (\mathbf {h} \mid \mathbf {v} _ {\text {data}}) \frac {\partial E (\mathbf {v} _ {\text {data}} , \mathbf {h})}{\partial \boldsymbol {\theta}} + \int_ {\mathbf {v}} p (\mathbf {v}) \int_ {\mathbf {h}} p (\mathbf {h} \mid \mathbf {v}) \frac {\partial E (\mathbf {v} , \mathbf {h})}{\partial \boldsymbol {\theta}},\tag{6}
$$

$$
C D _ {k} \left(\boldsymbol {\theta}, \mathbf {v} ^ {(0)}\right) = - \frac {\partial E \left(\mathbf {v} ^ {(0)} , \mathbf {h} ^ {(0)}\right)}{\partial \boldsymbol {\theta}} + \frac {\partial E \left(\mathbf {v} ^ {(k)} , \mathbf {h} ^ {(k)}\right)}{\partial \boldsymbol {\theta}}.\tag{7}
$$

In Eq. (7), the first and second terms are called negative term and positive term, respectively. They can reflect the raw data distribution and the model distribution. In the training process, the CD step can simply take k 1 , which can meet the requirements of calculation accura As mentioned in section 2.1, if the units type of the RBM/ERBM are Gaussian-Gaussian, their conditional probabilistic distribution $p ( \mathbf { h } \mid \mathbf { v } )$ and $p ( \mathbf { v } \mid \mathbf { h } )$ will obey a normal de nsity the model states v and h can take the means of the normal distributions as their samplin values.

![](images/04b8c96fd213422c120216e98b7f0ab51e256908b58f49d8bb9e755c6723a316.jpg)  
Fig. 4 The process of training RBM/ERBM with contrast divergence algorithm

By substituting Eq. (2) into Eq. (7), E . (8)-(10) can be further obtained. Then, the parameters $\mathbf { \boldsymbol { \Theta } } _ { p r e } = \{ \mathbf { w } _ { p r e } , \mathbf { a } _ { p r e } , \mathbf { b } _ { p r e } \}$ of ERBM can be updated by Eqs. (8) to (11) as

$$
\Delta w _ {i j} = v _ {i} ^ {(0)} h _ {j} ^ {(0)} - v _ {i} ^ {(k)} h _ {j} ^ {(k)},\tag{8}
$$

$$
\Delta a _ {i} = v _ {i} ^ {(0)} - v _ {i} ^ {(k)},\tag{9}
$$

$$
\Delta b _ {j} = h _ {j} ^ {(0)} - h _ {j} ^ {(k)},\tag{10}
$$

$$
\boldsymbol {\theta} _ {p r e} ^ {(e p o c h + 1)} = \boldsymbol {\theta} _ {p r e} ^ {(e p o c h)} + m \Delta \boldsymbol {\theta} _ {p r e} ^ {(p o c h - 1)} + r \Delta \boldsymbol {\theta} _ {p r e} ^ {(e p o c h)},\tag{11}
$$

where m is the momentum for accelerating the learning procedure; r is the learning rate; epoch represents the epoch of iterations; weight matrix consists of two parts as $\mathbf { w } _ { p r e } = [ \mathbf { w } _ { H } , \mathbf { w } _ { I } ]$ . In algorithm 1, a batch version of CD- k is explained.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: k-step contrastive divergence (CD-k)

Input: RBM(v, h), training batch B

Output: approximate gradient  $\Delta w, \Delta a, \Delta b$ 

1 init  $\Delta w_{ij} = \Delta a_i = \Delta b_j = 0$  for  $i = 1, ..., n, j = 1, ..., m$ 

2 forall the sample  $\in B$  do

3    $v^{(0)} \leftarrow \text{sample}$ 

4    for  $t = 0, ..., k - 1$  do

5    for  $j = 1, ..., m$  do sample  $h_j^{(t)}$  from  $p(h_j \mid v^{(t)})$ 

6    for  $i = 1, ..., n$  do sample  $v_i^{(t+1)}$  from  $p(v_i \mid h^{(t)})$ 

7    sample  $h_j^{(k)}$  from  $p(h_j \mid v^{(k)})$  for  $j = 1, ..., m$ 

8    for  $i = 1, ..., n, j = 1, ..., m$  do

9    $\Delta w_{ij} \leftarrow \Delta w_{ij} + v_i^{(0)} \cdot h_j^{(0)} - v_i^{(k)} \cdot h_j^{(k)}$ 

10    $\Delta a_i \leftarrow \Delta a_i + v_i^{(0)} - v_i^{(k)}$ 

11    $\Delta b_j \leftarrow \Delta b_j + h_j^{(0)} - h_j^{(k)}$
</div>

The pre-training process for the whole EDBN is described as follows. Assume $\mathbf { v } ^ { ( i , j ) }$ and $\mathbf { h } ^ { ( i , j ) }$ stand for the activation states of the visible and hidden units in the $i ^ { t h }$ ERBM at the $j ^ { t h }$ Gibbs sampling step, respectively. As sho $\therefore \sin \cdot \sin \cdot 5 ,$ the raw observed data $\mathbf { v } ^ { ( 1 , 0 ) }$ is first fed into the input layer of RBM-1. Then, hidden features $\mathbf { h } ^ { ( 1 , 0 ) }$ are generated by sampling from conditional probability $p \big ( \mathbf { h } | \mathbf { v } ^ { ( 1 , 0 ) } \big )$ . After that, $\textbf { v } ^ { ( 1 ) }$ and $\mathbf { h } ^ { ( 1 , 1 ) }$ can be available by further sampling from $p { \Big ( } \mathbf { v } \mid \mathbf { h } ^ { ( 1 , 0 ) } { \Big ) }$ and $p { \Big ( } \mathbf { h } | \mathbf { v } ^ { ( 1 , 1 ) } { \Big ) }$ By substituting the states into Eqs. (8)-(10), the parameters of the first RBM can be well trained. After RBM-1 is trained, its network parameters will be saved and used for transmitting the raw data $\mathbf { v } _ { d a t a } ^ { ( 1 ) }$ into hidden feature $\mathbf { h } _ { d a t a } ^ { ( 1 ) }$ . Then, the raw data and the hidden feature of RBM-1 are $\mathbf { \Omega } _ { \mathbf { V } } , \mathbf { r }$ bined as the inputs $\mathbf { v } _ { d a t a } ^ { ( 2 ) } = [ \mathbf { h } _ { d a t a } ^ { ( 1 ) } , \mathbf { v } _ { d a t a } ^ { ( 1 ) } ]$ for ERBM-2, and ERBM-2 will be trained in the same manner. In such a way, the remaining ERBMs will be trained one by one until the last one is trained. Finally, the features of the raw data will be extracted in a hierarchical and progressive manner, and the well-trained parameters will be used as the initial value for further finetuning. The whole pre-training process of EDBN can be summarized in algorithm 2.

![](images/fbe0c3a008eeece8865142929784e592902283ff4826dd066bfdb8716f71637b.jpg)

Fig. 5 Pre-training of EDBN: training RBM/ERBM layer by layer  
```txt
Algorithm 2: pre-training EDBN layer-by-layer

Input: EDBN, training set X
Output: pre-trained EDBN

1 forall the RBM/ERBM in EDBN do
2 init network parameters w, a, b
3 if the model to be trained is RBM then Input ← X
4 else Input ← combine H with λ
5 for epoch = 1, ..., e do
6    for k = 1, ..., floor(N_sample/N_batch_size) do
7    B ← take a batch from Input
8    Δw, Δa, Δb ← algorithm 1: CD-k
9    w ← w + r · Δv
10    a ← a + r · Δu
11    b ← b + r · Δb
12    H ← Input × output b
```

## 3.3.Fine-tuning

During the fine-tuning phase of EDBN, an additional layer for output is added at the last hidden layer of EDBN to obtain the probabilities of the sample into different categories. The parameters of hidden la ers in EDBN $\{ \mathbf { w } _ { f t } ^ { ( i ) } , \mathbf { b } _ { f t } ^ { ( i ) } \} _ { i = 1 , 2 , . . . , l }$ are initialized by the pre-trained parameters $\{ \mathbf { w } _ { H } ^ { ( i ) } , \mathbf { b } _ { p r e } ^ { ( i ) } \} _ { i = 1 , 2 , . . . , l }$ as

#

$$
\left\{ \begin{array}{l} \mathbf {w} _ {f t} ^ {(i)} = \mathbf {w} _ {H} ^ {(i)} \\ \mathbf {b} _ {f t} ^ {(i)} = \mathbf {b} _ {p r e} ^ {(i)} \end{array} , i = 1, 2, \ldots , l \right.,\tag{12}
$$

where l represents the number of hidden layers; the extended raw data nodes and their corresponding parameters $\{ \mathbf { w } _ { I } ^ { i } \} _ { i = 2 , 3 , . . . , l }$ are dropped out after pretraining. Fo the parameters $\{ \mathbf { w } _ { \boldsymbol { f t } } ^ { ( o ) } , b _ { \boldsymbol { f t } } ^ { ( o ) } \}$ of the output layer, random values are initialized for them. Therefore, the parameters need to be fine-tuned are $\mathbf { \boldsymbol { \Theta } } _ { f t } = \{ \mathbf { w } _ { f t } ^ { ( i ) } , \mathbf { \boldsymbol { b } } _ { f t } ^ { ( i ) } , \mathbf { w } _ { f t } ^ { ( o ) } , \mathbf { \boldsymbol { b } } _ { f t } ^ { ( o ) } \} _ { i = 1 , 2 , \dots , l }$ . Through forward propagation, the classification loss error C is calculated between the predicted outputs and the real labels. Then, the partial derivatives of the loss function with regard to the network parameters $\Delta \mathbf { \boldsymbol { \mathbf { 0 } } } _ { f t } = \partial \mathbf { C } / \partial \mathbf { \boldsymbol { \mathbf { 0 } } } _ { f t }$ can be further obtained by the BP algorithm. At last, based on the adaptive moment estimation (Adam) algorithm proposed by Kingma [41], the parameters of EDBN can be further tuned with Eq. (13) to (16).

$$
r ^ {(e p o c h + 1)} = r ^ {(0)} \sqrt {\left(1 - \beta_ {2} ^ {e p o c h}\right) / \left(1 - \beta_ {1} ^ {e p o c h}\right)}\tag{13}
$$

$$
\mathbf {m} _ {1} ^ {(e p o c h + 1)} = \beta_ {1} \mathbf {m} _ {1} ^ {(e p o c h)} + (1 - \beta_ {1}) \Delta \boldsymbol {\theta} _ {j t} ^ {(e p o c h)}\tag{14}
$$

$$
\mathbf {m} _ {2} ^ {(e p o c h + 1)} = \beta_ {2} \mathbf {m} _ {2} ^ {(e p o c h)} + \left(1 - \beta_ {2}\right) \left(\Delta \boldsymbol {\theta} _ {f t} ^ {(e p o c h)}\right) ^ {2},\tag{15}
$$

$$
\boldsymbol {\theta} _ {f t} ^ {(e p o c h + 1)} = \boldsymbol {\theta} _ {f t} ^ {(e p o c h)} - r ^ {(e p o c h + 1)} \mathbf {m} _ {1} ^ {(e _ {r}, e _ {h + 1})} / \sqrt {\mathbf {m} _ {2} ^ {(e p o c h + 1)} + \varepsilon},\tag{16}
$$

where epoch represents the number of iterations; $r ^ { ( e p o c h ) }$ refers to the learning rate, which has an initial learning rate ${ \boldsymbol { 0 } } ^ { \prime \prime } { \boldsymbol { r } } ^ { ( 0 ) }$ set by user; $\mathbf { m } _ { 1 } ^ { ( e p o c h ) }$ $\mathbf { m } _ { 2 } ^ { ( e p o c h ) }$ are first moment estimate and second raw moment estimate, respectively, both of which have an initial values of $\mathbf { 0 } ~ ; ~ \beta _ { 1 } , \beta _ { 2 } , \varepsilon$ are the parameters of $A a _ { 2 } \ldots$ hich are recommended to set to 0.9, 0.999, 10<sup>-8</sup>, separately.

## 3.4.Dynamic EDBN-based fault classification model

The proposed EDBN model can be used to extract deep features and predict classification results in the fault classification task of industrial chemical processes. As process data often have strong temporal correlations, the established fault classification model must consider the dynamic characteristics of process data. Hence, augmented dynamic data are constructed to build a dynamic EDBN model. Fig. 6 shows the whole augmentation procedure, which can be described as follows. Assume there are m raw measured variables. Then, a single data at sampling time t can be denoted as $\mathbf { x } ^ { ( t ) } = \left[ x _ { 1 } ^ { ( t ) } , x _ { 2 } ^ { ( t ) } , . . . , x _ { m } ^ { ( t ) } \right]$ , which acts as the fundamental element in a dynamic data. If the time length of the dynamic augmentation is taken as n , then a dynamic data from instant t n  1 to t is denoted as $\mathbf { d } ^ { ( t ) } = \left[ \mathbf { X } ^ { ( t ) } , \mathbf { X } ^ { ( t - 1 ) } , . . . , \mathbf { X } ^ { ( t - n + 1 ) } \right]$ . By expanding the dynamic data matrix into a onedimensional form, we can obtain the single input of the EDBN. In batch training, the batch data set $\mathbf { x } _ { i n }$ is used as the input of the EDBN in a training step, which is randomly selected from dynamic data set with a size of b . Correspondingly, the one-hot coded data set $\mathbf { y } _ { o u t }$ with the same size of b is utilized as the labels for dynamic EDBN mod

![](images/aab7bd032acac4378f5865e2e312eb5a69e0b9c215c63aa58f446cef61ccbff7.jpg)  
Fig. 6 The data augmentation for dynamic EDBN modeling

Based on the EDBN-based classifier, the fault classification framework can be established, which $\yen 123,456$ the offline training and online classification. In offline training, dynamic data are first generated from the historical database and transferred to the dynamic form for pre-training and fine-tuning of dynamic EDBN model. After the classifier is well-trained, it will be used for online fault classification. The classifier outputs the probabilities of all fault states and takes the maximum one as the fault category. Fig. 7 shows the modeling framework of EDBN-based classifier.

![](images/e7bc68047804a06a5554178f37d2e8026d39e7f0de469d58d8de1007095c1f07.jpg)  
Fig. 7 The procedure of offline modeling and online classification for dynamic EDBN-based fault classification model

## 4. Case study in Tennessee Eastman benchmark

## 4.1.Tennessee Eastman benchmark

![](images/81ad8ae88875a9ee5448afdf979d9e0c5a8eb62cc1d8a5fc8abfb06296bd12f3.jpg)  
Fig. 8 The flowchart of TE process [42]  
TE process [43] is a simulated one that has been extensively used as a chemical benchmark for

soft sensing, fault detection and process control researches. It obtains two main products from four reactants with five major operation units: the reactor, condenser, compressor, separator and stripper. A basic illustration is provided in Fig. 8 for the Tennessee Eastman process. process, there are fifty-two measured variables in total, which include nineteen composition measurements, twenty-two process measurements and eleven manipulated variabl For fault classification modeling, the first 33 measurement variables (see Table 1) are usually selected to build the classifier. The benchmark data sets can be downloaded from https://github.com/camaramm/tennesseeeastman-profBraatz. There are one normal state and 21 fault states simulated in the datasets, in which the sampling frequency is 3 minutes per sample. For each state, both training and testing datasets are collected. 500 training samples are rded for the normal state and 480 training every state, where the fault happens at the 160<sup>th</sup> sampling instant in each fault data set. Table 2 shows the sample number in the collected data set and their dynamic augmentation with time length n .

Table 1 Selected variables and process faults in the TE benchmark

<table><tr><td>No.</td><td>Variable type</td><td>No.</td><td>Variable</td><td>Status</td></tr><tr><td>1~22</td><td>XMEA (1~22)</td><td>Process measurements</td><td>0</td><td>Normal</td></tr><tr><td>23~33</td><td>YMV (1~33)</td><td>Manipulated variables</td><td>1~21</td><td>IDV (1~21) Fault (1~21)</td></tr></table>

Table 2 Sample counts of TE process data set

<table><tr><td rowspan="2">Status</td><td colspan="2">Collected data set</td><td colspan="2">Dynamic data set</td></tr><tr><td>Training</td><td>Test</td><td>Training</td><td>Test</td></tr><tr><td>Normal</td><td>500</td><td>960</td><td>501-n</td><td>961-n</td></tr><tr><td colspan="5">ACCEPTED MANUSCRIPT</td></tr><tr><td>Fault (1~21)</td><td>480</td><td> $160_{Normal} + 800_{Fault}$ </td><td>481 – n</td><td> $(161 - n)_{Normal} + (801 - n)_{Fault}$ </td></tr></table>

## 4.2.Fault classification result and performance comparison

After the classifier model is established and well-trained, the testing dataset is utilized to assess the model performance for fault classification. Usually, the model classification performance is evaluated by the false positive rate (FPR) and fault diagnosis rate $( \mathrm { F } _ { \mathbf { L } } ^ { \mathrm { ~ \tiny ~ D ~ } } )$ each fault type [1]. $F D R _ { i }$ represents the correct classification rate of samples with the label of i , while FPR represents the misclassification proportion of $ { \mathrm { i m } }  { \boldsymbol { r } } { \mathrm { ~ \ e s ~ \ . ~ } } 1$ the other class. They are calculated as

$$
F D R _ {i} = \frac {d _ {i}}{d _ {i} + r _ {i}},\tag{17}
$$

$$
F P R _ {i} = \frac {p _ {i}}{p _ {i} + q _ {i}},\tag{18}
$$

Table 3 Statistical indicators for the $i ^ { t h }$ class

<table><tr><td></td><td>Number of samples with predicted label  $i$ </td><td>Number of samples with predicted label other than  $i$ </td></tr><tr><td>Number of samples with real label  $i$ </td><td> $d_i$ </td><td> $r_i$ </td></tr><tr><td>Number of samples with real label other than  $i$ </td><td> $p_i$ </td><td> $q_i$ </td></tr></table>

where the detail symbol repr sentation is explained in Table 3. Moreover, $\overline { { F D R } }$ , the average fault diagnosis rate, and $\overrightarrow { F P ? }$ the average false positive rate, are used for evaluating the general classification performance on the entire data set, which are defined as

$$
\overline {{F D R}} = \frac {\sum_ {i} a _ {i}}{\sum_ {i} d _ {i} + \dot {d} _ {i}},\tag{19}
$$

$$
\overline {{F P R}} = \sum_ {i} \frac {\sum_ {i} p _ {i}}{p _ {i} + q _ {i}} = 1 - \overline {{F D R}}.\tag{20}
$$

The simulations were implemented on a computer with configuration as follows: 64-bit

Microsoft Windows 7 operating system, Intel Xeon E5-2630 2.4 GHz processor, 32 GB RAM, and Nvidia GeForce GTX 1080 Ti Graphics card. The algorithm of DBN is developed on Anaconda platform based on the open source python package tensorflow-gpu developed b oogle, which can be found from https://github.com/fuzimaoxinan/Tensorflow-Deep-Neural-Networks. The proposed EDBN algorithm is implemented based on the open DBN codes.

During the training, the epoch e is set to 35 and 240 for pre-training and fine-tuning, respectively. In each epoch, the min-batch gradient descend strategy is utilized. Every dynamic data set $\mathbf { x } _ { i n }$ with a batch sample size b is fed to the model at one time. In this study, the batch size is determined as 16. Moreover, the learning rate of pre-training and fine-tuning are both 0.0001 by trial and error. In the fine-tuning phase, mean- error (MSE) is used to calculate the loss between the estimated outputs and the label $\mathrm { { \bf O } , \mathrm { { \bf \Phi } \Psi _ { \cdot } \Psi _ { \cdot } \Psi _ { \cdot } } }$ The time window length n is set to 40 to include sufficient process dynamic information. In addition, dropout strategy is for each hidden layer during the training phase, which has a dropout rate of 0.382. Furthermore, linear and Gaussian functions are selected for activation ones as

$$
\left\{ \begin{array}{l} \text { Gaussian } (x) = 1 - e ^ {- x ^ {2}} \\ \text { Linear } (x) = x \end{array} \right.\tag{21}
$$

As a matter of fact, it is usually difficult to determine the best network architecture due to the lack of scientific guidance. In order to find a suitable model, some different DBN / EDBN models are taken for example for performance evaluation, the architectures of which are shown in Table 4. These network structures are designed entirely by experience, from which the most outstanding one will be selected. Parameters of the selected architecture mainly contain the number of layers and the activation function of each layer. In the Table 4, “G” stands for Gaussian activation function, and “L” refers to the linear activation function whose output value is equal to the input value. Take the second network in Table 4 for example, its network structure is designed as 1330-600-200-19, which means that there are 1330 and 19 neurons in the input layer and output layer, respectively. Meanwhile, 600 and 200 refers to the number of neurons in the 1<sup>st</sup> and 2<sup>nd</sup> hidde lay ers, respectively. As shown in the column of activation function, “G, L” represent the first and second hidden layer of it are selected as "Gaussian" and "Linear", while the last “G” stands "Gaussian" activation function chosen for its output layer.

Table 4 DBN/EDBN fault classification model with several candidate structures

<table><tr><td>DBN</td><td>EDBN</td><td>Architecture</td><td>Activation function</td></tr><tr><td>DBN-1</td><td>EDBN-1</td><td>1320, 600, 19</td><td>G, G</td></tr><tr><td>DBN-2</td><td>EDBN-2</td><td>1320, 600, 200, 19</td><td>G, L, G</td></tr><tr><td>DBN-3</td><td>EDBN-3</td><td>1320, 600, 200, 19</td><td>G, G, G</td></tr><tr><td>DBN-4</td><td>EDBN-4</td><td>1320, 600, 200, 200, 19</td><td>G, L, L, G</td></tr><tr><td>DBN-5</td><td>EDBN-5</td><td>320, 600, 400, 200, 19</td><td>G, L, G, G</td></tr><tr><td>DBN-6</td><td>EDBN-6</td><td>320, 600, 400, 200, 19</td><td>G, G, G, G</td></tr><tr><td>DBN-7</td><td>EDBN-7</td><td>1320, 600, 400, 200, 100, 19</td><td>G, L, G, L, G</td></tr><tr><td>DBN-8</td><td>EDBN°</td><td>1320, 600, 400, 200, 100, 19</td><td>G, G, G, G, G</td></tr></table>

Then, the classification performance and running time spent on pre-training are compared on these networks, the results of which are listed in Table 5. In general, the performance of EDBN can outperform DBN under the same hidden network structure. DBN achieves the highest average PDR value in DBN-4 with network structure of [1320, 600, 400, 200, 19], while EDBN-2 performs best in all designed EDBN models. In terms of pre-training time, EDBN is approximately 1.02 to 1.30 times of DBN. This is because the extended raw input data to each ERBM increases the network structure and the parameters, which naturally costs more pre-training time.

Table 5 Average FDR and time spent in the pre-training phase for several DBN/EDBN models

<table><tr><td>DBN</td><td>Running time in pre-training (s)</td><td> $\overline{FDR}$ (%)</td><td>EDBN</td><td>Running time in pre-training (s)</td><td> $\overline{FDR}$ (%)</td></tr><tr><td>DBN-1</td><td>130</td><td>92.05</td><td>EDBN-1</td><td>133</td><td>94.18</td></tr><tr><td>DBN-2</td><td>236</td><td>93.89</td><td>EDBN-2</td><td>163</td><td>94.31</td></tr><tr><td>DBN-3</td><td>220</td><td>92.66</td><td>EDBN-3</td><td>252</td><td>92.88</td></tr><tr><td>DBN-4</td><td>307</td><td>93.95</td><td>EDBN-4</td><td>376</td><td>94.02</td></tr><tr><td>DBN-5</td><td>315</td><td>93.26</td><td>EDBN-5</td><td>379</td><td>94.22</td></tr><tr><td>DBN-6</td><td>311</td><td>92.24</td><td>EDBN-6</td><td>369</td><td>92.56</td></tr><tr><td>DBN-7</td><td>387</td><td>92.96</td><td>EDBN-7</td><td>506</td><td>94.11</td></tr><tr><td>DBN-8</td><td>394</td><td>92.02</td><td>EDBN-8</td><td>493</td><td>92.47</td></tr></table>

From Table 5, we can find that when u aussian activation function, EDBN can achieve better results even with a simple three $\mathbf { \mu } ^ { \mathrm { { i } } \mathrm { { a } } } { \mathbf { y } } \cdot \mathbf { \mu }$ er neural network. Nevertheless, with the increase of the number of hidden layers, EDBNs with more Gaussian activation function show lower average FDR than those with few Gaussian. This may be due to the reason that excessive use of nonlinear structures leads to overfi $\mathbf { \Omega } _ { \mathbf { \Omega } ^ { \mathrm { L I I I } } \dot { \mathbf { z } } }$ while the original problem may not be that complicated. Hence, when using a partial of linear activation function instead of Gaussian activation function, the EDBN model achieves the best results with structure of EDBN-2.

Then, the second network structure is taken for example for further performance analysis. The detailed FDR and FPR of EDBN-2 and DBN-2 in all 19 categories are provided in Table 6. From Table 6, the detection rate of normal state and most of the fault diagnosis rates of EDBN-2 are higher than DBN-2, and the average FDR of EDBN-2 is increased by 0.42% compared with DBN-2. From the comparison of FPR, EDBN-2 can also achieve better performance since it has a lower FPR than

DBN-2 almost in all categories. At last, Fig. 9 shows the detailed classification results on the test data set by taking EDBN-2 for example, where the red circles represent real labels and blue circles represent predicted ones. It can be seen from Fig. 9 that almost all the samples can be classified correctly by EDBN-2. In detail, Fig. 10 shows the proportion of samples in test data set divided into each category, where the column refers to the real labels of the dat the row denotes the predicted one. From the simulation results, it can be concluded that EDBN-2 has superior performance in feature learning and fault classification than DBN-

Table 6 The comparison of classification performance on DBN-2 and EDBN-2 (19-category)

<table><tr><td rowspan="2">Fault type</td><td colspan="2">FDR (%)</td><td colspan="2">FPR (%)</td></tr><tr><td>DBN-2</td><td>EDBN-2</td><td>DBN-2</td><td>EDBN-2</td></tr><tr><td>0 (Normal data)</td><td>87.54</td><td>90.83</td><td>2.76</td><td>3.34</td></tr><tr><td>1</td><td>100</td><td>100</td><td>0</td><td>0</td></tr><tr><td>2</td><td>100</td><td>100</td><td>0.07</td><td>0.08</td></tr><tr><td>4</td><td>100</td><td>100</td><td>0.01</td><td>0.01</td></tr><tr><td>5</td><td>100</td><td>100</td><td>0</td><td>0</td></tr><tr><td>6</td><td>100</td><td>100</td><td>0</td><td>0</td></tr><tr><td>7</td><td>100</td><td>100</td><td>0</td><td>0</td></tr><tr><td>8</td><td>98.42</td><td>98.29</td><td>0.3</td><td>0.22</td></tr><tr><td>10</td><td>78.19</td><td>80.81</td><td>0.78</td><td>0.67</td></tr><tr><td>11</td><td>99.74</td><td>99.74</td><td>0</td><td>0</td></tr><tr><td>12</td><td>100</td><td>100</td><td>0.03</td><td>0</td></tr><tr><td>13</td><td>90.28</td><td>91.98</td><td>0</td><td>0</td></tr><tr><td>14</td><td>100</td><td>100</td><td>0</td><td>0</td></tr><tr><td>16</td><td>79.89</td><td>75.56</td><td>0.97</td><td>0.67</td></tr><tr><td>17</td><td>100</td><td>100</td><td>0</td><td>0</td></tr><tr><td>18</td><td>93.56</td><td>93.43</td><td>0</td><td>0</td></tr><tr><td>19</td><td>97.37</td><td>95.53</td><td>0.56</td><td>0.27</td></tr><tr><td>20</td><td>93.04</td><td>93.17</td><td>0.02</td><td>0</td></tr><tr><td>21</td><td>85.28</td><td>83.44</td><td>1.3</td><td>1.17</td></tr><tr><td>Average</td><td>93.89</td><td>94.31</td><td>6.11</td><td>5.69</td></tr></table>

![](images/419b7e83480348b36d266454e237eead5cb1ca3d52ac3020afe311849dfb3bbf.jpg)  
Fig. 9 Fault classification results o $\yen 123,456$ with 19-category

<table><tr><td></td><td>0.91</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.14</td><td>0</td><td>0.02</td><td>0</td><td>0.11</td><td>0</td><td>0.07</td><td>0.04</td><td>0.07</td><td>0.17</td></tr><tr><td>Normal</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 01</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 02</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 04</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 05</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 06</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 07</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 08</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.8</td><td>0</td><td>0</td><td>0</td><td>0.05</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 10</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.81</td><td>0</td><td>0</td><td>0.01</td><td>0</td><td>0.12</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 11</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 12</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 13</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.92</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 14</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 16</td><td>0.02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.06</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.76</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 17</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Fault 18</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.93</td><td>0</td><td>0</td></tr><tr><td>Fault 19</td><td>0.01</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.96</td><td>0</td><td>0</td></tr><tr><td>Fault 20</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.93</td><td>0</td><td>0</td></tr><tr><td>Fault 21</td><td>0.06</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.01</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

$\textbf { I } _ { \because } \hat { \textbf { \textit { \textbf { r } } } } _ { \approx }$ details of the proportion of samples in each fault divided into other category

## 5. Concluding remarks

A novel extended DBN is designed for feature representation and fault diagnosis in chemical

#

processes in this paper. Since most deep learning models do not consider the loss of potential valuable information in the raw data caused by layer-wise feature compressing, EDBN is developed to alleviate this problem. The strategy of repeatedly stacking raw data is used in the pre-training phase of EDBN to adequately extract useful information. By combining the previous hidden features with the original observed data as the inputs of the next ERBM, layer- wise raw data related feature can be extracted. Then, a dynamic EDBN-based fault classification modeling framework is built to consider the dynamics of process data. TE benchmark is used to assess the fault classification performance of the extended DBN. By testing the classification performance of DBN and EDBN with different network structures on the 19-category fault datasets, it shows that EDBN can almost obtain higher FDR and lower FPR than DBN, w it is on a single category or whole data set. In the best EDBN model, the average fault diagnosis rate achieves 94.31%, which is increased by 0.42% compared with DBN. Therefore, EDBN can extract more valuable features from raw data for further fault classification performance than the original DBN, which shows great potential for fault diagnosis in chemical processes

## Acknowledgment

This paper is supported in part by the National Natural Science Foundation of China of China (2018JJ3687), and in part by Innovation-driven Plan in Central South University (2018CX011, 2019zzts274).

## References

1. Wu H, Zhao JS. Deep convolutional neural network model based chemical process fault diagnosis. Comput Chem Eng. 2018, 115, 185-97.

2. Castro MAL, Escobar RF, Torres L, Aguilar JFG, Hernández JA, Olivares-Peregrino VH. Sensor

fault detection and isolation system for a condensation process. ISA Trans. 2016, 65, 456-67.

3. Liu Y, Bazzi AM. A review and comparison of fault detection and diagnosis methods for squirrelcage induction motors: State of the art. ISA Trans. 2017, 70, 400-9.

4. Yuan X, Li L, Wang Y. Nonlinear dynamic soft sensor modeling with supervised long short-term memory network. IEEE T Ind Inf. 2019, 10.1109/TII.2019.2902129

5. Wang S, Ren X, Na J, Zeng T. Extended-State-Observer-Based Funnel Control for Nonlinear Servomechanisms With Prescribed Tracking Performance. IEEE Transactions on Automation Science and Engineering. 2017, 14, 98-108.

6. Liu L, Liu Y, Tong S. Fuzzy Based Multi-Error Constraint Control for Switched Nonlinear Systems and Its Applications. IEEE Transactions on Fuzzy Systems. 2018, 1-.

7. Yuan X, Ge Z, Huang B, Song Z. A probabilistic just-in-time learning framework for soft sensor development with missing data. IEEE T Contr Syst T. 2017, 25, 1124-32.

8. Chen N, Dai J, Yuan X, Gui W, Ren W, Koivo HN. Temperature Prediction Model for Roller Kiln by ALD-Based Double Locally Weighted Kernel Principal Component Regression IEEE T Instrum Meas. 2018, 67, 2001 - 10

9. Yuan X, Chen Z, Wang Y. Probabilistic nonlinear soft sensor modeling based on generative topographic mapping regression. IEEE Access. 2018, 6, 10445-52.

10. Yuan X, Wang Y, Yang C, Ge Z, Song Z, Gui W. Weighted Linear Dynamic System for Feature Electron. 2018, 65, 1508-17.

11. Zhang ZP, Zhao JS. A deep belief network based fault diagnosis model for complex chemical processes. Comput Chem Eng. 2017, 107, 395-407.

12. Ge ZQ. Process Data Analytics via Probabilistic Latent Variable Models: A Tutorial Review. Ind Eng Chem Res. 2018, 57, 12646-61.

13. Chen ZW, Ding SX, Zhang K, Li ZB, Hu ZK. Canonical correlation analysis-based fault detection methods with application to alumina evaporation process. Control Engineering Practice. 2016, 46, 51-8.

14. Chen Z, Ding SX, Peng T, Yang C, Gui W. Fault Detection for Non-Gaussian Processes Using Generalized Canonical Correlation Analysis and Randomized Algorithms. IEEE T Ind Electron. 2018, 65, 1559-67.

15. Hu Z, Chen Z, Gui W, Jiang B. Adaptive PCA based fault diagnosis scheme in imperial smelting process. ISA T. 2014, 53, 1446-55.

16. Jiang QC, Yan XF, Huang BA. Performance-Driven Distributed PCA Process Monitoring Based on Fault-Relevant Variable Selection and Bayesian Inference. Ieee T Ind Electron. 2016, 63, 377- 86.

17. Yuan X, Ge Z, Huang B, Song Z, Wang Y. Semisupervised JITL Framework for Nonlinear Industrial Soft Sensing Based on Locally Semisupervised Weighted PCR. IEEE T Ind Inf. 2017, 13, 532-41.

18. Chiang LH, Kotanchek ME, Kordon AK. Fault diagnosis based on Fisher discriminant analysis and support vector machines. Comput Chem Eng. 2004, 28, 1389-401.

19. Yuan X, Zhou J, Wang Y, Yang C. Multi-similarity measurement driven ensemble just-in-time learning for soft sensing of industrial processes. J Chemometr. 2018, 32, e3040.

20. Yuan X, Ge Z, Song Z. Locally Weighted Kernel Principal Component Regression Model for Soft Sensing of Nonlinear Time-Variant Processes. Ind Eng Chem Res. 2014, 53, 13736-49.

21. Jiao J, Zhao N, Wang G, Yin S. A nonlinear quality-related fault detection approach based on modified kernel partial least squares. ISA Trans. 2017, 66, 275-83.

22. Shen Y, Wu ZG, Shi P, Su HY, Huang TW. Asynchronous Filtering for Markov Jump Neural Networks With Quantized Outputs. Ieee T Syst Man Cy-S. 2019, 49, 433-43.

23. Dong SL, Wu ZG, Shi P, Karimi HR, Su HY. Networked Fault Detection for Markov Jump Nonlinear Systems. Ieee T Fuzzy Syst. 2018, 26, 3368-78.

24. Ben Salem S, Bacha K, Chaari A. Support vector machine based decision for mechanical fault condition monitoring in induction motor using an advanced Hilbert-Park transform. ISA Trans. 2012, 51, 566-72.

25. Yuan XF, Huang B, Wang YL, Yang CH, Gui WH. Deep Learning-Based Feature Representation and Its Application for Soft Sensor Modeling With Variable-Wise Weighted SAE. Ieee Transactions on Industrial Informatics. 2018, 14, 3235-43.

26. Liu Y, Yang C, Gao Z, Yao Y. Ensemble deep kernel learning with application to quality prediction in industrial polymerization processes. Chemometr Intell Lab Syst. 2018, 174, 15-21.

27. Xuan Q, Fang B, Liu Y, Wang J, Zhang J, Zheng Y, et al. Automatic Pearl Classification Machine Based on a Multistream Convolutional Neural Network. IEEE Transactions on Industrial Electronics.

2018, 65, 6538-47.

28. Liu Y, Fan Y, Chen J. Flame Images for Oxygen Content Prediction of Combustion Systems Using DBN. Energy Fuels. 2017, 31, 8776-83.

29. Xuan Q, Chen Z, Liu Y, Huang H, Bao G, Zhang D. Multi-View Generative Adversarial Network and Its Application in Pearl Classification. IEEE Transactions on Industrial Electronics. 2018, 1-.

30. Yuan X, Ou C, Wang Y, Yang C, Gui W. Deep quality-related feature extraction for soft sensing modeling: A deep learning approach with hybrid VW-SAE. Neurocomputing. 2019.

31. Jiang P, Hu ZX, Liu J, Yu SN, Wu F. Fault Diagnosis Based on Chemical Sensor Data with an Active Deep Neural Network. Sensors. 2016, 16.

32. Jiang L, Ge ZQ, Song ZH. Semi-supervised fault classification based on dynamic Sparse Stacked auto-encoders model. Chemometrics Intellig Lab Syst. 2017, 168, 72-83.

33. Yu JB, Yan XF. Layer-by-Layer Enhancement Strategy of Favorable Features of the Deep Belief Network for Industrial Process Monitoring. Ind Eng Chem Res. 2018, 57, 15479-90.

of the 27th International Conference on International Conference on Machine Learning. Omnipress, Haifa, Israel, 2010. p. 807-14.

35. Shang C, Yang F, Huang DX, Lyu WX. Data-driven soft sensor development based on deep learning technique. Journal of Process Control. 2014, 24, 223-33.

36. Hinton GE. Training products of experts by minimizing contrastive divergence. Neural Comput. 2002, 14, 1771-800.

37. Hinton GE. A Practical Guide to Training Restricted Boltzmann Machines. In: Montavon G, Orr GB, Müller K-R, editors. Neural Networks: Tricks of the Trade: Second Edition. Springer Berlin Heidelberg, Berlin, Heidelberg, 2012. p. 599-619.

38. Hinton GE, Osindero S, Teh YW. A fast learning algorithm for deep belief nets. Neural Comput. 2006, 18, 1527-54.

39. Carreira-Perpinan MA, Hinton GE. On contrastive divergence learning. Aistats2005. p. 33-40.

40. Fischer A, Igel C. An Introduction to Restricted Boltzmann Machines. Springer Berlin Heidelberg, Berlin, Heidelberg, 2012. p. 14-36.

41. Kingma DP, Ba J. Adam: A method for stochastic optimization. arXiv:14126980. 2014.

42. Yin S, Ding SX, Haghani A, Hao HY, Zhang P. A comparison study of basic data-driven fault

diagnosis and process monitoring methods on the benchmark Tennessee Eastman process. Journal of Process Control. 2012, 22, 1567-81.

43. Chiang LH, Russell EL, Braatz RD. Fault detection and diagnosis in industrial systems. Springer2000.

##

## Title page showing Author Details

![](images/173fb4a3d9a8c61c1e843c01dec79a38ea2649cc7b5abbbae838b334efdb3e3d.jpg)

from the Department of Control Science and Engineerin entral South University, Changsha, China, in 1995 and 2001, respe

Since 2003, she has been with the School of Information Science and Engineering, Central south University, where she was at first an Associate Professor and is control for complex industrial processes, intelligient control, and process simulation.

![](images/1226ef1898242d852fa79a6e6185f48d2a6d28e4c0fd309ea132f459f1e3ad62.jpg)

Zhuofu Pan is currently a Ph.D. student at the School of Information Science and Engineering, Central South University, Changsha, China. He received his the B.Eng. degree in School of Civil Engineering from Changsha University of Science & Technology and the M.Eng. degree in School of Civil Engineering from Central South University, Changsha, China, in 2017 and 2014, respectively. His research interests include deep learning and artificial intelligence, industrial big data analysis, fault detection and diagnosis, intelligent optimization algorithm, etc.

![](images/2b1bcb14ae0f80d1ae614dcac3c36f168852faa09ee50bb435335d903ba19de8.jpg)

Xiaofeng Yuan received the B.Eng. and Ph.D. degrees from the Department of Control Science and Engineering, Zhejiang University, Hangzhou, China, in 2011 and 2016, respectively.

He was a visiting scholar with the Department of Chemical and Materials Engineering, University of Alberta, Edmonton, AB, Canada, from November 2014 to May 2015. He is currently an Associate Professor with the School of Information Science and Engineering, Central south University. His research interests include deep learning and artificial intelligence, machine learning recognition, industrial process soft sensor modeling, process data analysis, etc.

![](images/efb40e9a7c37068a034297cf0f7a2e7c0723e98b486f9e05b9051971066eccd8.jpg)

Chunhua Yang received the M.Eng. degree in automatic control engineering and the Ph.D. degree in control science and engineering from Central South University, Changsha, China, in 1988 and 2002, respectively. She was with the Department of Electrical Engineering, Katholieke Universiteit Leuven, Leuven, Belgium, from 1999 to 2001. She is currently a Full Professor with Central South University. Her current research interests include modeling and optimal control of complex industrial process, intelligent control system, and fault-tolerant computing of real-time systems.

![](images/cb622d9bf993cdd4e066dc9f69d554958227bdc5a3ea423aabb7b631c5f25119.jpg)

<sup>Weihua</sup> <sup>Gui</sup> <sup>received</sup> <sup>the</sup> <sup>degree</sup> <sup>of</sup> <sup>the</sup> <sup>B.Eng.</sup> <sup>and</sup> <sup>the</sup> <sup>M.Eng.</sup> <sup>at</sup>Department of Control Science and Engineering from Central South University, Changsha, China, in 1976 and 1981, respectively. From 1986 to 1988 he was a visiting scholar at Universität-GH-Duisburg, Germany. He has been South University, Changsha, China, since 1991. His main research interests include modeling and optimal control of complex industrial processes, distributed robust control, and fault diagnoses. He was elected as an academician of Chinese Academy of Engineering in 2013.

##

There is no conflict of interest with this paper entitled with “A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network”.

## Highlights

> An extended DBN is proposed for feature extraction and fault classification with hierarchically stacked extend RBMs (ERBM).

> In each ERBM, the hidden features of the previous RBM and inputs to the visible layer of current RBM for better feature learning.

> The extended DBN is beneficial for retaining enough valuable information from raw data while extracting deep compression features.

> High classification performance of the extended DBN is validated on TE process.