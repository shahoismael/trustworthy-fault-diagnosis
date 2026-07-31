Article

# Hierarchical Deep LSTM for Fault Detection and Diagnosis for a Chemical Process

Piyush Agarwal , Jorge Ivan Mireles Gonzalez, Ali Elkamel and Hector Budman \*

Department of Chemical Engineering, University of Waterloo, Waterloo, ON N2L 3G1, Canada \* Correspondence: hbudman@uwaterloo.ca

![](images/6215ae5b91090202c5fc22daf09c31e6bc9d55b892900227a3376802bf9c7879.jpg)

Citation: Agarwal, P.; Gonzalez, J.I.M.; Elkamel, A.; Budman, H. Hierarchical Deep LSTM for Fault Detection and Diagnosis for a Chemical Process. Processes 2022, 10, 2557. https://doi.org/10.3390/ pr10122557

Abstract: A hierarchical structure based on a Deep LSTM Supervised Autoencoder Neural Network (Deep LSTM-SAE NN) is presented for the detection and classification of faults in industrial plants. The proposed methodology has the ability to classify incipient faults that are difficult to detect and diagnose with traditional and many recent methods. Faults are grouped into different subsets according to the degree of difficulty to classify them accurately in the proposed hierarchical structure. External pseudo-random binary signals (PRBS) are injected in the system to enhance the identification of incipient faults. The approach is illustrated on the benchmark process (Tennessee Eastman Process) in order to compare across different methodologies. The efficacy of the proposed method is shown by a comprehensive comparison between many recent and traditional fault detection and diagnosis methods in the literature for Tennessee Eastman Process. The proposed work results in significant improvements in the classification of faults over both multivariate linear model-based strategies and non-hierarchical nonlinear model-based strategies.

Academic Editor: Bernard Riera, Nadhir Messai

Received: 3 November 2022 Accepted: 17 November 2022 Published: 1 December 2022

Publisher’s Note: MDPI stays neutral with regard to jurisdictional claims in published maps and institutional affiliations.

![](images/9dc018a687e719f1e063089528f83288ff440933566eb71a0cd7a71efe228316.jpg)

Copyright: © 2022 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https:// creativecommons.org/licenses/by/ 4.0/).

Keywords: fault detection and diagnosis; statistical process monitoring (SPC); classification; autoencoders; deep learning; Tennessee Eastman Process; LSTM; incipient faults

## 1. Introduction

The faults in a chemical plant often propagate along the process, significantly impacting the profit of chemical plants. Hence, it is imperative to detect them soon upon their occurrence. The operation of industrial plants employs sensors and control loops to mitigate the economic losses resulting from these faults. However, in the presence of pro cess faults and manipulated variable constraints, these control schemes are not sufficiently resilient to avoid abnormal operation [1,2]. Thus, process faults must be diagnosed and addressed by implementing a suitable corrective measure.

A typical process monitoring system consists of two parts: fault detection and diagnosis methodology. The objective of a fault detection system is to make a binary decision whether the current state of the process is in a normal or faulty operation region. Once an abnormal operation is detected, the fault diagnosis system is used to infer the type of fault or identify the root cause of the process fault. In the current study, we perform both detection and classification with a single algorithm by considering the normal operation condition as an additional fault class to be identified in the classification step.

Process monitoring schemes rely on estimated process models using historical data to infer faults. Based on the type of model, the methodologies are divided into two main approaches: mechanistic model-based (e.g., using first principles models) and data-driven model-based approaches [1]. Data-driven models for FDD, such as the one used in the current study, are based on a comparison between different sensor measurements under normal operation versus faulty operation [3–8]. Within the class of data-driven approaches, several reported algorithms are based on multivariate statistical methods such as Principal Component Analysis (PCA) [9–12] or its dynamic version such as Dynamic Principal Component Analysis (DPCA) [1,10,13–15]. These methods assume process behavior is linear. However, most chemical processes are inherently non-linear in nature. Thus, nonlinear modeling techniques such as Deep Neural Networks (DNNs) are employed in the current work. In the last decade, a new generation of Deep Neural Networks (DNNs) algorithms has emerged that capitalizes both the significant increase in computational power and novel algorithmic developments that facilitate the training and calibration of these networks. The use of these algorithms for fault detection in the process industry has recently received increased attention. However, despite the improvements in detection accuracy obtained with these techniques, some faults are still difficult to detect and diagnose (incipient faults). The current study focuses on the detection and diagnosis of such difficult to detect faults while maintaining good detection accuracy for the other faults. The difficult to observe/detect faults will be referred to as incipient faults.

Lack of observability often arises due to the low signal to noise ratio in the measurements used for fault detection and diagnosis (FDD) and feedback control [16,17]. Specifically, the controller forces the controlled variables to remain close to their set-points at all times. Furthermore, with the addition of noise, the effects of faults are masked. In addition, the lack of distinguishability between different process faults is related to the fact that various process faults have a similar effect on the dynamic responses of the measured variables.

FDD algorithms that rely on data collected from the process operation are referred to as passive, while active FDD approaches have also been proposed to improve detection [18]. Active FDD involves injecting persistently exciting input signals into the system and using the resulting input–output data for incipient fault detection and diagnosis [19–21]. The disadvantage of active FDD is that it introduces an external disturbance to the process which may temporarily impact the operation, and thus, its use should be limited. To the knowledge of the authors, the combination of active and passive FDD approaches into one algorithm for detecting a mix of non-incipient and incipient faults have not been studied.

Following the above, the focus of the current work is on developing deep learning techniques for the detection of faults with an emphasis on the detection of incipient faults. However, faults and their effects on process variables are strongly coupled with each other. Thus, improving the detection of incipient faults should be achieved without degrading the detection of the regular faults. Toward this goal, a novel hierarchical classification strategy based on DNN models is proposed that involves identifying separate models for different subsets of faults with different degrees of difficulty to detect. A combination of both passive and active FDD approaches is used. The DNN models used for the passive FDD component are of Recursive Neural Network (RNN) type to exploit the dynamic information in the data. It is also demonstrated that the detection accuracy of most faults can be enhanced by increasing the time horizon of the LSTM-based model. While the passive approach is used in the higher level of the hierarchy, the active approach involving the injection of external signals is only used in the last level of the hierarchy for detecting incipient faults that cannot be diagnosed otherwise. It is shown that the passive FDD approach is effective for identifying most faults, but the active approach is required for detecting incipient faults.

All studies in this work are conducted with a standard set of simulated data from the Tennessee Eastman Process (TEP) for a fair comparison with several algorithms reported for this system [22–26,26]. Since its introduction, the TEP has served as a benchmark problem for testing control and fault detection algorithms, and it is thus ideal for comparing existing approaches to our proposed algorithm. It should be emphasized that due to the difficulty in detecting a set of incipient faults for TEP (faults 3, 9 and 15), many studies on FDD for this system were carried out by ignoring these faults altogether [23,24]. For those studies of FDD for the TEP process that consider all the faults together, the regular faults were detected with an acceptable level of success, but the detection of incipient faults was very inaccurate [27]. In this work, we address the gap related to the miss-classification of incipient faults by proposing a novel hierarchical structure that combines a deep learning approach with an active FDD approach. Furthermore, it was also demonstrated that just the hierarchical structure along with deep NNs are not enough for classifying incipient faults through an ablation study. Thus, an active FDD (introduction of excitation signal) approach in combination with a hierarchical deep NN structure are both required for efficient fault diagnosis. Additional reported methods applied to TEP are further reviewed and discussed in Results and Discussions section (Sections 4 and 5). The comparison of our approach to several reported methods shows that our approach provides comparable or superior FDD accuracy for regular faults but clear superiority for incipient faults.

The main contributions of the current study are:

1. A novel hierarchical structure was developed combining passive FDD with active FDD to enhance the detection and classification accuracy for incipient faults.

2. Design of PRBS signal for improving the observability of the incipient faults.

3. The LSTM model was optimized with respect to data horizon for better classification of faults.

4. A comprehensive comparison of the proposed method to several other methods was carried out to demonstrate the efficacy of the proposed method for both regular and incipient faults.

The paper is organized as follows. Fundamentals used in the work are presented in Section 2. Explanation on the hierarchical structure of the proposed methodology is presented in Section 3. The results are presented in Section 4. Discussions and comparisons with previously reported approaches are presented in Section 5 followed by conclusions in Section 6.

## 2. Preliminaries

## 2.1. Recurrent Neural Networks (RNNs)

The current study uses a Recurrent Neural Network (RNN) type model that was originally developed for handling dynamic data by using time sequences of data ${ \bf x } _ { t } ^ { i } ,$ $t = 1 , 2 , \ldots , T \in \hat { \mathbb { R } } ^ { d _ { h } \times d _ { x } }$ as inputs to the network [28]. Parameters associated with RNN are shared along a time horizon to capture temporal correlations in data. This enhances the generalization capability of the model to time sequences that were not used for model calibration. A well-known challenge for training RNNs is the vanishing gradient or exploding gradient problem arising from the use of gradient descent algorithms in combination with sigmoid activation functions [29]. To deal with this problem, the best practice is to use gated-type unit structures within RNN models such as Long-Short Term Memory units (LSTM) [30] and Gated Recurrent Units (GRU) [31]. LSTM is reviewed in the following sub-section since they serve as the basis for the models used in the current study for FDD.

## 2.2. Long Short-Term Memory (LSTM) Units

The LSTM unit is composed of three gated units and a memory cell [30]. Figure 1 shows a single LSTM unit that includes four major gates: the forget gate $( \mathbf { f } _ { t } ) ,$ , the input gate (i ), the output gate $\left( \mathbf { o } _ { t } \right)$ and the update gate $( { \bf g } _ { t } )$ . The key component of the LSTM unit is the memory cell $( \mathbf { c } _ { t } \in \mathbb { R } ^ { d _ { h } \times 1 } )$ that is responsible for storing critical long-term dependencies learned over time. The input gate (i ) is responsible for evaluating which part, if any, of the past historical data should be kept. Thus, the function of the input gate is to allow the network to keep only relevant information from the previous time steps and discard the rest for a sample i.

Subsequently, the information that is worth recording is determined by the memory cell (c<sub>t</sub>). The process of identifying information and storing in the memory cell consists of two parts: new information that is recorded and information that is discarded. The information that should be discarded from previous cell state $\mathbf { c } _ { t - 1 } ^ { i }$ is determined by the forget gate $\left( \mathbf { f } _ { t } \right) .$ which is responsible for forgetting previously stored cell state values that have lost their relevance. Then, new relevant information is added, and existing cell-state values are updated by first selecting which values to update using the input gate $\mathbf { i } _ { t } ^ { i } ,$ , and the output from the input gate is then multiplied by the new information generated by the update gate $\mathbf { g } _ { t } ^ { \ i }$ . Ultimately, the output $\mathbf { h } _ { t }$ is computed at every time step from the information contained in the memory cell and it is further gated by an output gate according to its importance or relevance. The mathematical equations describing these gating operations are as follows:

$$
\mathbf {i} _ {t} ^ {i} = \sigma (\mathbf {W} _ {i} \mathbf {x} _ {t} ^ {i} + \mathbf {R} _ {i} \mathbf {h} _ {t - 1} ^ {i} + \mathbf {b} _ {i})
$$

$$
\mathbf {g} _ {t} ^ {i} = \tanh (\mathbf {W} _ {g} \mathbf {x} _ {t} ^ {i} + \mathbf {R} _ {g} \mathbf {h} _ {t - 1} ^ {i} + \mathbf {b} _ {g})\tag{1}
$$

$$
\mathbf {c} _ {t} ^ {i} = \mathbf {f} _ {t} ^ {i} \odot \mathbf {c} _ {t - 1} ^ {i} + \mathbf {i} _ {t} ^ {i} \odot \mathbf {g} _ {t} ^ {i}\tag{2}
$$

where $\mathbf { \alpha } \mathbf { e } ( \mathbf { \beta } )$ and tanh() are the element-wise sigmoid and hyperbolic tangent functions, respectively.

$$
\begin{array}{l} \mathbf {o} _ {t} ^ {i} = \sigma (\mathbf {W} _ {o} \mathbf {x} _ {t} ^ {i} + \mathbf {R} _ {o} \mathbf {h} _ {t - 1} ^ {i} + \mathbf {b} _ {o}) \\ \mathbf {h} _ {t} ^ {i} = \mathbf {o} _ {t} ^ {i} \odot \tanh (\mathbf {c} _ {t} ^ {i}) \end{array}\tag{3}
$$

where $\mathbf { R } = [ \mathbf { R } _ { f } \mathbf { R } _ { i } \mathbf { R } _ { g } \mathbf { R } _ { o } ] ^ { T } \in \mathbb { R } ^ { 4 d _ { h } \times d _ { h } }$ are known as recurrent weights, ${ \displaystyle { \bf W } = [ { \bf W } _ { f } { \bf W } _ { i } { \bf W } _ { g } { \bf W } _ { o } ] ^ { T } \in }$ $\mathbb { R } ^ { 4 d _ { h } \times d _ { x } }$ are all the input weights, $\boldsymbol { \mathbf { b } } = [ \boldsymbol { \mathbf { b } } _ { f } \ \boldsymbol { \mathbf { b } } _ { i } \ \boldsymbol { \mathbf { b } } _ { g } \ \boldsymbol { \mathbf { b } } _ { o } ] ^ { T } \in \mathbb { R } ^ { d _ { h } \times 1 }$ are the bias parameters.

![](images/311a850bac33956bc6489ecf14c5e2801a49081a659471cd26860611a4b37a5d.jpg)  
Figure 1. Schematic of an LSTM memory cell.

## 2.3. Deep LSTM Supervised Autoencoder Neural Network (DLSTM-SAE NN)

The training of a Deep Supervised Autoencoder Neural Network (DSAE-NN) model, as schematically shown in Figure 2, is based on the minimization of a weighted sum of the reconstruction loss function and the supervised classification loss corresponding to the first and second terms in Equation (4), respectively. Addition of the unsupervised loss function i i ii.e., reconstruction loss function, improves the generalization of supervised autoencoder t  t  t<sub>−</sub>1  <sub>model</sub> <sub>[32].</sub> <sub>Furthermore,</sub> <sub>it</sub> <sub>serves</sub> <sub>as</sub> <sub>the</sub> <sub>regularization</sub> <sub>term</sub> <sub>which</sub> <sub>constraints</sub> <sub>the</sub> <sub>problem</sub> i i iin terms of latent variables, thus reducing over-fitting. Meanwhile, the minimization of t  t ⊙ t the classification loss function, i.e., multi-class cross-entropy loss function, ensures the nonlinear latent variables extracted are the predictors of the output label. The mean squared T 4d derror function is used as a reconstruction loss and softmax cross-entropy is used as the ∈<sub>classification</sub> <sub>loss.</sub> <sub>The</sub> <sub>overall</sub> <sub>goal</sub> <sub>is</sub> <sub>to</sub> <sub>learn</sub> <sub>a</sub> <sub>function</sub> <sub>that</sub> <sub>predicts</sub> <sub>the</sub> <sub>class</sub> <sub>labels</sub> <sub>in</sub> one-hot encoded form $\mathbf { y } _ { i } \in \mathbb { R } ^ { m }$ from inputs $\mathbf { x } _ { i } \in \mathbb { R } ^ { d _ { x } \times 1 }$

For training DSAE-NN, the following loss function is minimized:

$$
l _ {D S A E} = \frac {\lambda_ {1}}{N} | | \mathbf {x} _ {s} - \hat {\mathbf {x}} _ {s} | | _ {2} ^ {2} + \frac {1}{N} \sum_ {s = 1} ^ {N} \sum_ {c = 1} ^ {m} - y _ {s, c} l o g (p _ {s, c})\tag{4}
$$

In this work, we use LSTM units instead of dense layers for both the encoder and decoder, as shown in Figure 3. The goal is to reconstruct and classify input sequences at time t simultaneously. The encoder transforms the input time sequences using Equations (1)–(3) to learn important features and encode these features $\mathbf { z } \in \mathbb { R } ^ { d _ { h } \times 1 }$ . The decoder function reconstructs the input using the extracted feature vectors. The operation performed by the encoder for a single LSTM layer between the input variables to the latent variables $\mathbf { z } _ { t } ^ { i } \in \mathbb { R } ^ { d _ { h } \times 1 }$ can be mathematically described as follows:

$$
\mathbf {z} _ {t} ^ {i} = \zeta_ {e} (\mathbf {x} _ {t} ^ {i})\tag{5}
$$

The latent variables $\mathbf { z } _ { t } ^ { i }$ are used both to predict the class labels and to reconstruct back the inputs x as follows:

$$
\hat {\mathbf {x}} _ {t} ^ {i} = \zeta_ {d} (z _ {t} ^ {i})\tag{6}
$$

$$
\hat {\mathbf {y}} _ {t} ^ {i} = f _ {c} (\mathbf {W} _ {c} \mathbf {z} _ {t} ^ {i} + \mathbf {b} _ {c})\tag{7}
$$

where $\zeta _ { e }$ and $\zeta _ { d }$ is the LSTM encoder and decoder function, respectively. $f _ { c }$ is a non-linear activation function (softmax layer) for the output layer. $\mathbf { W } _ { c } \in \hat { \mathbb { R } } ^ { m \times d _ { z } }$ and $\mathbf { b } _ { c } \in \mathbb { R } ^ { m }$ are the output weight matrix and bias vector, respectively.

$$
p _ {s, c} = \frac {e ^ {(\hat {y} _ {s , c})}}{\sum_ {c = 1} ^ {m} e ^ {(\hat {y} _ {s , c})}}\tag{8}
$$

where $\lambda _ { 1 }$ is the weight multiplying the reconstruction loss $L _ { r }$ in the cost to be minimized, m is the number of classes, $y _ { s , c }$ is a binary indicator (0 or 1) equal to 1 if the class label c is the correct one for observation s and 0 otherwise, $\hat { y _ { s , c } }$ is the non-normalized log probabilities and $p _ { s , c }$ is the predicted probability for a sample s of class c. Moreover, to avoid over fitting, a regularization term is added to the objective function in Equation (4). Accordingly, the objective function for Deep LSTM SAE NNs used for FDD is as follows:

$$
\min _ {\mathbf {W}} l _ {D L S T M - S A E} = \min \frac {1}{N} \left[ \lambda_ {1} | | \mathbf {x} _ {s} - \hat {\mathbf {x}} _ {s} | | _ {2} ^ {2} + \lambda_ {2} \sum_ {s = 1} ^ {N} \sum_ {c = 1} ^ {m} - y _ {s, c} l o g (p _ {s, c}) + \lambda_ {3} \sum_ {L} \sum_ {k} \sum_ {j} \mathbf {W} _ {k j} ^ {[ L ] 2} \right]\tag{9}
$$

where $\mathbf { W } _ { k j } ^ { [ L ] }$ represents the weight matrices for each layer L in the network and the weights on the individual objective functions $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ are chosen using validation data.

![](images/86b2dca24220e695e9d5f21eb7589a522557b4c3ed1d458c7f25d103e50c5241.jpg)

Figure 2. Schematic of a single layer Supervised Autoencoder Neural Network (SAE-NN).  
![](images/10ea5fa37515c6c0d1b905c0b3ada28fd2959ab1b7cc9482d63c8474ee0c4ba9.jpg)  
Figure 3. Schematic of a Deep LSTM Supervised Autoencoder Neural Network (DLSTM-SAE NN).

## 2.4. Model Structure and Specifications

The Deep LSTM-SAE model used in the current study was developed with training and testing data sets generated from the Tennessee Eastman Process (TEP: schematic shown in Figure 4) simulation. The data are extracted from simulations of the system conducted at either the normal state or when each of the 20 different faults is occurring in the process. It is assumed that at each sampling interval, 52 different variables (refer Table 1) are measured and organized into a vector. Each such vector of measurements is acquired every 3 min. It should be noticed that during testing of the methods proposed in this study, the normal state is considered as a different separate class, and hence, a total of 21 different classes (refer Table 2, i.e., 20 faulty plus one normal operations, are considered for classification. The standard dataset can be downloaded from http://depts.washington.edu/control/ LARRY/TE/download.html (accessed on 9 April 2022). The simulator is ran for 72 h (training: 24 h; testing: 48 h) for each fault, generating 1440 samples for each fault class and normal class. The data are then divided between calibration and validation data sets where the first 480 samples are used as training data and the rest are used for testing for each class. This results in a total of 10,080 training samples and 19,200 testing samples. A small fraction of training dataset is used as validation dataset for selecting the optimal hyper-parameters. It is important to note that the number of training, validation and testing samples vary depending on the time horizon used in the DLSTM-SAE model. The results reported in the following section are based on the classification accuracy of the test dataset, i.e., on data that were not used for model calibration. The experiments in this paper have been implemented on an Intel Core i7-7700HQ PC (2.80 GHz, 16 GB RAM) and NVIDIA

GeForce GTX 1060 (6 GB) 64 Bit Windows 10 operating system in Python ® environment. The models are developed using Keras [33] (an open deep learning library) on TensorFlow platform [34]. All hyper-parameters such as the number of LSTM encoder layers, LSTM units in each layer, weights and learning rate are optimized using Keras tuner.

Table 1. Measured and manipulated variables (from Downs and Vogel, 1993).

<table><tr><td>Variable Name</td><td>Variable Number</td><td>Units</td><td>Variable Name</td><td>Variable Number</td><td>Units</td></tr><tr><td>A feed (stream 1)</td><td>XMEAS (1)</td><td>kscmh</td><td>Reactor cooling water outlet temperature</td><td>XMEAS (21)</td><td>°C</td></tr><tr><td>D feed (stream 2)</td><td>XMEAS (2)</td><td> $\mathrm{kg h^{-1}}$ </td><td>Separator cooling water outlet temperature</td><td>XMEAS (22)</td><td>°C</td></tr><tr><td>E feed (stream 3)</td><td>XMEAS (3)</td><td> $\mathrm{kg h^{-1}}$ </td><td>Feed %A</td><td>XMEAS (23)</td><td>mol%</td></tr><tr><td>A and C feed (stream 4)</td><td>XMEAS (4)</td><td>kscmh</td><td>Feed %B</td><td>XMEAS (24)</td><td>mol%</td></tr><tr><td>Recycle flow (stream 8)</td><td>XMEAS (5)</td><td>kscmh</td><td>Feed %C</td><td>XMEAS (25)</td><td>mol%</td></tr><tr><td>Reactor feed rate (stream 6)</td><td>XMEAS (6)</td><td>kscmh</td><td>Feed %D</td><td>XMEAS (26)</td><td>mol%</td></tr><tr><td>Reactor pressure</td><td>XMEAS (7)</td><td>kPa guage</td><td>Feed %E</td><td>XMEAS (27)</td><td>mol%</td></tr><tr><td>Reactor level</td><td>XMEAS (8)</td><td>%</td><td>Feed %F</td><td>XMEAS (28)</td><td>mol%</td></tr><tr><td>Reactor temperature</td><td>XMEAS (9)</td><td>°C</td><td>Purge %A</td><td>XMEAS (29)</td><td>mol%</td></tr><tr><td>Purge rate (stream 9)</td><td>XMEAS (10)</td><td>kscmh</td><td>Purge %B</td><td>XMEAS (30)</td><td>mol%</td></tr><tr><td>Product separator temperature</td><td>XMEAS (11)</td><td>°C</td><td>Purge %C</td><td>XMEAS (31)</td><td>mol%</td></tr><tr><td>Product separator level</td><td>XMEAS (12)</td><td>%</td><td>Purge %D</td><td>XMEAS (32)</td><td>mol%</td></tr><tr><td>Product separator pressure</td><td>XMEAS (13)</td><td>kPa guage</td><td>Purge %E</td><td>XMEAS (33)</td><td>mol%</td></tr><tr><td>Product separator underflow (stream 10)</td><td>XMEAS (14)</td><td> $\mathrm{m^3 h^{-1}}$ </td><td>Purge %F</td><td>XMEAS (34)</td><td>mol%</td></tr><tr><td>Stripper level</td><td>XMEAS (15)</td><td>%</td><td>Purge %G</td><td>XMEAS(35)</td><td>mol%</td></tr><tr><td>Stripper pressure</td><td>XMEAS (16)</td><td>kPa guage</td><td>Purge %H</td><td>XMEAS (36)</td><td>mol%</td></tr><tr><td>Stripper underflow (stream 11)</td><td>XMEAS (17)</td><td> $\mathrm{m^3 h^{-1}}$ </td><td>Product %D</td><td>XMEAS (37)</td><td>mol%</td></tr><tr><td>Stripper temperature</td><td>XMEAS (18)</td><td>°C</td><td>Product %E</td><td>XMEAS (38)</td><td>mol%</td></tr><tr><td>Stripper steam flow</td><td>XMEAS (19)</td><td> $\mathrm{kg h^{-1}}$ </td><td>Product %F</td><td>XMEAS (39)</td><td>mol%</td></tr><tr><td>Compressor Work</td><td>XMEAS (20)</td><td>kW</td><td>Product %G</td><td>XMEAS (40)</td><td>mol%</td></tr><tr><td>D feed flow</td><td>XMV (1)</td><td> $\mathrm{kg h^{-1}}$ </td><td>Product %H</td><td>XMEAS (41)</td><td>mol%</td></tr><tr><td>E feed flow</td><td>XMV (2)</td><td> $\mathrm{kg h^{-1}}$ </td><td>A feed flow</td><td>XMV (3)</td><td>kscmh</td></tr><tr><td>A + C feed flow</td><td>XMV (4)</td><td>kscmh</td><td>Compressor recycle valve</td><td>XMV (5)</td><td>%</td></tr><tr><td>Purge valve</td><td>XMV (6)</td><td>%</td><td>Separator pot liquid flow</td><td>XMV (7)</td><td> $\mathrm{m^3 h^{-1}}$ </td></tr><tr><td>Stripper liquid product flow</td><td>XMV (8)</td><td> $\mathrm{m^3 h^{-1}}$ </td><td>Stripper steam valve</td><td>XMV (9)</td><td>%</td></tr><tr><td>Reactor cooling water flow</td><td>XMV (10)</td><td> $\mathrm{m^3 h^{-1}}$ </td><td>Condenser cooling water flow</td><td>XMV (11)</td><td> $\mathrm{m^3 h^{-1}}$ </td></tr></table>

Table 2. Process faults for classification in the TE process.

<table><tr><td>Fault</td><td>Description</td><td>Type</td></tr><tr><td>IDV(1)</td><td>A/C feed ratio, B composition constant (stream 4)</td><td>step</td></tr><tr><td>IDV(2)</td><td>B composition, A/C ratio constant (stream 4)</td><td>step</td></tr><tr><td>IDV(3)</td><td>D Feed temperature</td><td>step</td></tr><tr><td>IDV(4)</td><td>Reactor cooling water inlet temperature</td><td>step</td></tr><tr><td>IDV(5)</td><td>Condenser cooling water inlet temperature (stream 2)</td><td>step</td></tr><tr><td>IDV(6)</td><td>A feed loss (stream 1)</td><td>step</td></tr><tr><td>IDV(7)</td><td>C header pressure loss reduced availability (stream 4)</td><td>step</td></tr><tr><td>IDV(8)</td><td>A, B, C feed composition (stream 4)</td><td>random variation</td></tr><tr><td>IDV(9)</td><td>D feed temperature</td><td>random variation</td></tr><tr><td>IDV(10)</td><td>C feed temperature (stream 4)</td><td>random variation</td></tr><tr><td>IDV(11)</td><td>Reactor cooling water inlet temperature</td><td>random variation</td></tr><tr><td>IDV(12)</td><td>Condenser cooling water inlet temperature</td><td>random variation</td></tr><tr><td>IDV(13)</td><td>Reaction kinetics</td><td>slow drift</td></tr><tr><td>IDV(14)</td><td>Reactor cooling water</td><td>valve sticking</td></tr><tr><td>IDV(15)</td><td>Condenser cooling water valve</td><td>stiction</td></tr><tr><td>IDV(16)</td><td>Deviations of heat transfer within stripper</td><td>random variation</td></tr><tr><td>IDV(17)</td><td>Deviations of heat transfer within reactor</td><td>random variation</td></tr><tr><td>IDV(18)</td><td>Deviations of heat transfer within condenser</td><td>random variation</td></tr><tr><td>IDV(19)</td><td>Recycle valve of compressor, underflow stripper and steam valve stripper</td><td>stiction</td></tr><tr><td>IDV(20)</td><td>unknown</td><td>random variation</td></tr></table>

![](images/83bc59292692d4c8cdab409a0f663f89cb784cb4e8dd55933f8eff096e2f754c.jpg)  
Figure 4. Schematic of Tennessee Eastman Process.

## 3. Hierarchical Structure

The key goal of the work is to improve the detection and diagnosis of incipient faults but without sacrificing the detection accuracy for the regular (non-incipient) faults. Thus, we need to increase the sensitivity of the non-linear FDD algorithm with respect to the incipient faults but without losing sensitivity with respect to the non-incipient faults. The sensitivity of non-linear models such as deep neural networks is highly dependent on the variability of the data used for calibration. Accordingly, a key data pre-processing step toward model calibration involves data standardization, i.e., mean centering and normalization. It is hypothesized that by building separate models for different groups of faults, it is possible to increase the sensitivity of different models and distinguishability between faults because of the different re-normalization conducted within each group.

Following the above, a hierarchical structure is proposed as shown in Figure 5. This structure includes the following sequential steps for training of the model with a training data set:

1. The training data are mean centered and normalized.

2. The faults are classified into two groups: group 1—easily distinguishable faults and group 2—difficult to distinguish faults, which include the incipient faults along with normal operation data class.

3. A Deep LSTM-SAE model denoted as M1 is designed for identifying the faults of group 1 or identifying all faults in group 2 as a single fault

4. The data for group 2 identified in the previous step are mean centered and re-normalized.

5. A neural network model is designed specifically for group 2 denoted as M2.

6. For faults that are not accurately identified by M2, a PRBS is designed and injected into locations in the system that are informative about these faults.

Based on the trained hierarchical structure, online detection and diagnosis for any new sample proceeds as follows:

1. The data corresponding to the sample is mean centered and normalized as in step 1 of the training procedure.

2. The sample is classified as either in group 1 of easy to observe faults or group 2 of difficult to identify faults.

3. If sample is in group 1, it is classified accordingly by model M1. If it is in group 2, it is re-normalized according to the re-normalization in step 4 of the training procedure.

4. If the sample is within group 2, it is identified by model M2 in step 5 of the training procedure.

5. If the sample is not identified accurately by the model for group 2, PRBS signals are injected as specified in step 6 of the training procedure, and the corresponding faults are diagnosed from the resulting data.

![](images/ba3479f09dd702f216962793833264b173e6101806c25b0b08849261ca5333df.jpg)  
Figure 5. Hierarchical structure used for fault detection and diagnosis.

It should be noticed that in this algorithm, the normal operation is treated as an additional fault class denoted as Class 1. The incipient faults are characterized by responses that are very similar to the normal state (TEP: faults 3, 9 and 15). It should also be noted that the incipient faults are grouped along with the normal state as per step 1 of the training procedure; it may also result in miss-classification as other faults. Hence, the overall classification accuracy for the incipient faults must be assessed after the execution of the entire hierarchical procedure.

For model M1, the normalized data are fed to a first-level model where the softmax layer of LSTM-SAE NN uses 18 units instead of the 21 units (incipient faults and normal state grouped as one) as used in the non-hierarchical type model. The structure of model M2 is similar to model M1, but the difference is that the softmax layer involves only 4 units each for one of the incipient faults (3, 9, 15) and for the normal state (fault 0). The PRBS is injected only when the incipient fault cannot be properly identified with either models M1 or M2. Additional details about the PRBS signal design are given in the following section.

## Design: Pseudo-Random Binary Signal (PRBS)

Although the hierarchical structure proposed in the previous section enhances the diagnosibility of few faults, the detection of incipient faults is still challenging due to the lack of excitation to detect these faults in the presence of noise. This problem is particularly acute in the TEP since the dataset contains variables that are used in closed-loop control, thus exhibiting a small variation with respect to their set-point values, making it difficult to estimate the occurrence of faults from such variables. To increase the diagnosibility of incipient faults, the use of active fault detection, as reviewed in the Introduction, is proposed for the TEP process. The lack of diagnosibility/distinguishability of the incipient faults can be viewed as a problem of inaccurate identification of a model relating variability in measured values to faults. To improve the identification accuracy, it is required to use inputs that sufficiently excite the system dynamics in the presence of noise [35], which will result in larger changes in the measured quantities and larger sensitivity to fault changes. Thus, it is required to introduce additional excitation to the one available in regular operation of the system. Accordingly, external forcing signals are injected at particular points of the control loops, e.g., an excitation signal to the set-points of the loops that involve variables related to the difficult to detect faults. The addition of such excitation signals in combination with a separate deep neural network model (second level) in the hierarchical structure described in the previous section is investigated in the current study for detecting and diagnosing incipient faults that cannot be accurately identified with the regular operating data collected from the process.

To avoid a large negative impact of the external signals on the profitability of the plant, the input signals should meet certain constraints as follows:

1. Reduce input move sizes (to reduce wear and tear on actuators).

2. Reduce input and output amplitudes, power, or variance.

3. Short experimental time to prevent losses

In a practical implementation, the added excitation signal should result in variations in the measured quantities that will be large in magnitude relative to the noise. Toward this goal, it is necessary to include information of frequencies lower than the crossover frequency of the closed loop transfer function [36]. PRBS signals are used as excitation signals in this study, since they have a finite length that can be synthesized repeatedly with simple generators while presenting favorable spectra. The spectrum at low frequencies are flat and constant, while at high frequencies, the spectra drop off. Thus, the PRBS can be designed to have a specific bandwidth, which can be utilized for exciting the processes within the required range of frequencies [37]. The analytical expression for the power spectrum of a PRBS is given by:

$$
s (\omega) = \frac {A ^ {2} (R + 1) t _ {c l}}{R} \left[ \frac {\sin \omega t _ {c l} / 2}{\omega t _ {c l}} \right] ^ {2}\tag{10}
$$

where $\omega$ is the frequency, $t _ { c l }$ is the clock period (minimum time between a change in levels) which is a multiple of the sampling time $( T _ { s } )$ and A is the amplitude of the signal. The sequence repeats itself after $T = R \times t _ { c l }$ units of time, where $R = 2 n - 1$ and n is the number of shift registers used to generate the sequence. Thus, for designing the PRBS signal, it is necessary to estimate the amplitude and the frequency range.

$$
\frac {2 \pi}{T} \leq \omega \leq \frac {2 . 8}{t _ {c l}}\tag{11}
$$

Rivera and Gaikwad (1995) [36] Lee and Rivera, 2005 [38] and Garcia-Gabin and Lundh [37] provided practical guidelines for estimating the range of frequency needed for process closed-loop identification using time-domain information. The primary frequency band of interest for excitation is determined by the dominant time constants of the system.

$$
\omega_ {l o w} = \frac {1}{S _ {f} t ^ {o l}}\tag{12}
$$

where $t ^ { o l } = 4 \tau ^ { o l } + t _ { d } ^ { o l }$

$$
\omega_ {h i g h} = \frac {4 S _ {f}}{t ^ {c l}}\tag{13}
$$

$$
\omega_ {h i g h} \leq \omega_ {N}\tag{14}
$$

where $S _ { f }$ is a safety factor used to augment the bandwidth of the excitation signal, $t ^ { o l }$ is the open loop settling time and $t ^ { c l }$ is the settling time of the closed loop process without considering the time delays. $t _ { d } ^ { o l }$ is the time delay of the open loop process. In addition, the upper value of the frequency must be lower than the Nyquist frequency $\omega _ { N }$ to avoid aliasing. Although the magnitude of the signal has not been optimized in the current work, it could be further optimized by taking a profit function of the plant into consideration for minimal losses and using the validation data used for the FDD model.

## 4. Results

In this section, the industrial benchmark TEP is used to validate and demonstrate the effectiveness of the proposed method. Three tables are presented in this section to summarize the results. The tables show comparisons based on a standard set of simulated data from the TEP between our proposed algorithm with several algorithms reported in the literature. First, fault detection rates for all non-incipient faults are shown in Table 3 for different linear multivariate methods and DL-based methodologies. Secondly, Table 4 shows comparisons with those results that consider incipient fault detection for TEP along with non-incipient faults, $\mathrm { i . e . , }$ all faults.

Table 3. Comparison of Fault Detection Rate with different methods with non-incipient faults only.

<table><tr><td>Fault</td><td colspan="2">PCA(15 comp.)</td><td>DPCA(22 comp.)</td><td colspan="2">ICA(9 comp.)</td><td>DL(2017)</td><td>DL(2017)</td><td>DL(2018)</td><td>DL(2018)</td><td>DL(2019)</td><td>Proposed DL</td></tr><tr><td></td><td> $T^2$ </td><td>SPE</td><td> $T^2$ </td><td> $I^2$ </td><td>AO</td><td>SAE-NN</td><td>DSN</td><td>GAN</td><td>OCSVM</td><td>CNN</td><td>Deep LSTM-SAE</td></tr><tr><td>1</td><td>99.2%</td><td>99.8%</td><td>99%</td><td>100%</td><td>100%</td><td>77.6%</td><td>90.8%</td><td>99.62%</td><td>99.5%</td><td>91.39%</td><td>100%</td></tr><tr><td>2</td><td>98%</td><td>98.6%</td><td>98%</td><td>98%</td><td>98%</td><td>85%</td><td>89.6%</td><td>98.5%</td><td>98.5%</td><td>87.96%</td><td>100%</td></tr><tr><td>4</td><td>4.4%</td><td>96.2%</td><td>26%</td><td>61%</td><td>84%</td><td>56.6%</td><td>47.6%</td><td>56.25%</td><td>50.37%</td><td>99.73%</td><td>100%</td></tr><tr><td>5</td><td>22.5%</td><td>25.4%</td><td>36%</td><td>100%</td><td>100%</td><td>76%</td><td>31.6%</td><td>32.37%</td><td>30.5%</td><td>90.35%</td><td>100%</td></tr><tr><td>6</td><td>98.9%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>82.8%</td><td>91.6%</td><td>100%</td><td>100%</td><td>91.5%</td><td>100%</td></tr><tr><td>7</td><td>91.5%</td><td>100%</td><td>100%</td><td>99%</td><td>100%</td><td>80.6%</td><td>91%</td><td>99.99%</td><td>99.62%</td><td>91.55%</td><td>100%</td></tr><tr><td>8</td><td>96.6%</td><td>97.6%</td><td>98%</td><td>97%</td><td>97%</td><td>83%</td><td>90.2%</td><td>97.87%</td><td>97.37%</td><td>82.95%</td><td>100%</td></tr><tr><td>10</td><td>33.4%</td><td>34.1%</td><td>55%</td><td>78%</td><td>82%</td><td>75.3%</td><td>63.2%</td><td>50.87%</td><td>53.25%</td><td>70.05%</td><td>42.8%</td></tr><tr><td>11</td><td>20.6%</td><td>64.4%</td><td>48%</td><td>52%</td><td>70</td><td>75.9%</td><td>54.2%</td><td>58%</td><td>54.75%</td><td>60.16%</td><td>100%</td></tr><tr><td>12</td><td>97.1%</td><td>97.5%</td><td>99%</td><td>99%</td><td>100%</td><td>83.3%</td><td>87.8%</td><td>98.75%</td><td>98.63%</td><td>85.56%</td><td>100%</td></tr><tr><td>13</td><td>94%</td><td>95.5%</td><td>94%</td><td>94%</td><td>95%</td><td>83.3%</td><td>85.5%</td><td>95%</td><td>94.87%</td><td>46.92%</td><td>100%</td></tr><tr><td>14</td><td>84.2%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>77.8%</td><td>89%</td><td>100%</td><td>100%</td><td>88.88%</td><td>100%</td></tr><tr><td>16</td><td>16.6%</td><td>24.5%</td><td>49%</td><td>71%</td><td>78%</td><td>78.3%</td><td>74.8%</td><td>34.37%</td><td>36.37%</td><td>66.84%</td><td>100%</td></tr><tr><td>17</td><td>74.1%</td><td>89.2%</td><td>82%</td><td>89%</td><td>94%</td><td>78%</td><td>83.3%</td><td>91.12%</td><td>87.25%</td><td>77.11%</td><td>100%</td></tr><tr><td>18</td><td>88.7%</td><td>89.9%</td><td>90%</td><td>90%</td><td>90%</td><td>83.3%</td><td>82.4%</td><td>90.37%</td><td>90.12%</td><td>82.74%</td><td>100%</td></tr><tr><td>19</td><td>0.4%</td><td>12.7%</td><td>3%</td><td>69%</td><td>80%</td><td>67.7%</td><td>52.4%</td><td>11.8%</td><td>3.75%</td><td>70.87%</td><td>40.4%</td></tr><tr><td>20</td><td>29.9%</td><td>45%</td><td>53%</td><td>87%</td><td>91%</td><td>77.1%</td><td>44.1%</td><td>58.37%</td><td>52.75%</td><td>72.88%</td><td>100%</td></tr><tr><td>Average</td><td>61.77%</td><td>74.72%</td><td>72.35%</td><td>87.29%</td><td>91.70%</td><td>77.7%</td><td>76.84%</td><td>74.04%</td><td>62.78%</td><td>85.47%</td><td>93.13%</td></tr></table>

Table 4. Comparison of Fault Detection Rate with different methods (with all faults).

<table><tr><td>Fault</td><td>DL(2017)</td><td>DL(2017)</td><td>DL(2018)</td><td>DL(2018)</td><td>DL(2019)</td><td>DL(2018)</td><td>DL(2021)</td><td>Proposed DL</td></tr><tr><td></td><td>SAE-NN</td><td>DSN</td><td>GAN</td><td>OCSVM</td><td>CNN</td><td>Optimized LSTM</td><td>LSTM (attention)</td><td>Deep LSTM-SAE</td></tr><tr><td>1</td><td>77.6%</td><td>90.8%</td><td>99.62%</td><td>99.5%</td><td>91.39%</td><td>68%</td><td>100%</td><td>100%</td></tr><tr><td>2</td><td>85%</td><td>89.6%</td><td>98.5%</td><td>98.5%</td><td>87.96%</td><td>78%</td><td>89%</td><td>100%</td></tr><tr><td>3</td><td>79.4%</td><td>14.4%</td><td>10.375%</td><td>7.62%</td><td>50.59%</td><td>45%</td><td>94%</td><td>81.58%</td></tr><tr><td>4</td><td>56.6%</td><td>47.6%</td><td>56.25%</td><td>50.37%</td><td>99.73%</td><td>75%</td><td>99%</td><td>100%</td></tr><tr><td>5</td><td>76%</td><td>31.6%</td><td>32.37%</td><td>30.5%</td><td>90.35%</td><td>45%</td><td>94%</td><td>100%</td></tr><tr><td>6</td><td>82.8%</td><td>91.6%</td><td>100%</td><td>100%</td><td>91.5%</td><td>75%</td><td>100%</td><td>100%</td></tr><tr><td>7</td><td>80.6%</td><td>91%</td><td>99.99%</td><td>99.62%</td><td>91.55%</td><td>89%</td><td>100%</td><td>100%</td></tr><tr><td>8</td><td>83%</td><td>90.2%</td><td>97.87%</td><td>97.37%</td><td>82.95%</td><td>100%</td><td>99%</td><td>100%</td></tr><tr><td>9</td><td>50.6%</td><td>16.3%</td><td>8.625%</td><td>7.125%</td><td>49.53%</td><td>89%</td><td>81%</td><td>99.38%</td></tr><tr><td>10</td><td>75.3%</td><td>63.2%</td><td>50.87%</td><td>53.25%</td><td>70.05%</td><td>71%</td><td>99%</td><td>42.84%</td></tr><tr><td>11</td><td>75.9%</td><td>54.2%</td><td>58%</td><td>54.75%</td><td>60.16%</td><td>67%</td><td>88%</td><td>100%</td></tr><tr><td>12</td><td>83.3%</td><td>87.8%</td><td>98.75%</td><td>98.63%</td><td>85.56%</td><td>77%</td><td>99%</td><td>100%</td></tr><tr><td>13</td><td>83.3%</td><td>85.5%</td><td>95%</td><td>94.87%</td><td>46.92%</td><td>83%</td><td>89%</td><td>100%</td></tr><tr><td>14</td><td>77.8%</td><td>89%</td><td>100%</td><td>100%</td><td>88.88%</td><td>56%</td><td>99%</td><td>100%</td></tr><tr><td>15</td><td>55.5%</td><td>26.7%</td><td>12.5%</td><td>14%</td><td>43.54%</td><td>89%</td><td>22%</td><td>100%</td></tr><tr><td>16</td><td>78.3%</td><td>74.8%</td><td>34.37%</td><td>36.37%</td><td>66.84%</td><td>99%</td><td>31%</td><td>100%</td></tr><tr><td>17</td><td>78%</td><td>83.3%</td><td>91.12%</td><td>87.25%</td><td>77.11%</td><td>0%</td><td>97%</td><td>100%</td></tr><tr><td>18</td><td>83.3%</td><td>82.4%</td><td>90.37%</td><td>90.12%</td><td>82.74%</td><td>89%</td><td>95%</td><td>100%</td></tr><tr><td>19</td><td>67.7%</td><td>52.4%</td><td>11.8%</td><td>3.75%</td><td>70.87%</td><td>20%</td><td>97%</td><td>40.4%</td></tr><tr><td>20</td><td>77.1%</td><td>44.1%</td><td>58.37%</td><td>52.75%</td><td>72.88%</td><td>88%</td><td>85%</td><td>100%</td></tr><tr><td>Average</td><td>75.355%</td><td>65.32%</td><td>64.51%</td><td>62.78%</td><td>79.84%</td><td>70.15%</td><td>87.85%</td><td>93.23%</td></tr></table>

Finally, a systematic ablation study is conducted in Table 5 to demonstrate gradual improvements following the proposed methodology. Thus, in this table, the different levels of the proposed hierarchical algorithm are added one by one to observe their relative contribution to the FDD accuracy. In-depth details about the comparisons and ablation study are discussed in the next section.

Table 5. Ablation study for the proposed method.

<table><tr><td>Faults</td><td>Non-Hierarchical DL NN</td><td>Hierarchical DL NN (No PRBS)</td><td>Hierarchical DL NN+PRBS Addition for Fault 15</td><td>Hierarchical + PRBS Addition for Fault 15 and Fault 9</td></tr><tr><td>Fault 3</td><td>36%</td><td>42%</td><td>88.7%</td><td>81.5%</td></tr><tr><td>Fault 9</td><td>32%</td><td>18%</td><td>38.4%</td><td>99.3%</td></tr><tr><td>Fault 15</td><td>12%</td><td>30%</td><td>99.4%</td><td>100%</td></tr><tr><td>Normal Operation</td><td>18%</td><td>25%</td><td>100%</td><td>98.1%</td></tr><tr><td>Average of all other Faults</td><td>85%</td><td>87%</td><td>93.1%</td><td>93.1%</td></tr><tr><td>Averaged Test Accuracy</td><td>73.4%</td><td>75.90%</td><td>90.9%</td><td>93.4%</td></tr></table>

## 5. Discussions

We investigated the multi-class classification performance using a total of 20 fault modes presented in Table 2 which involve all of the compositions, manipulated and measurement variables in the TE process (Table 1). For an individual class IDV(i), the performance was typically evaluated by a confusion matrix which consists of true positives (TP ), false positives $( \mathrm { F P } _ { i } )$ , true negatives $( \mathrm { T N } _ { i } )$ and false negatives (FN ). The notation used in the confusion matrix (refer Table 6) is as follows:

Table 6. Confusion matrix for each fault (IDV(i)).

<table><tr><td></td><td>Counts of Predicted Label i</td><td>Counts of Predicted Label other than i</td></tr><tr><td>Counts of real label i</td><td> $\text{TP}_i$ </td><td> $\text{TN}_i$ </td></tr><tr><td>Counts of real label other than i</td><td> $\text{FP}_i$ </td><td> $\text{FN}_i$ </td></tr></table>

Two main important metrics for quantifying the performance of the proposed process monitoring methodology are as follows:

Fault Detection Rate (FDR):

$$
\begin{array}{r l} \mathrm{FDR} & = \frac {\text { number   of   fault   data   that   have   been   detected   as   fault }}{\text { total   number   of   faulty   samples }} \\ & = \frac {T P _ {i}}{T P _ {i} + F P _ {i}} \end{array}\tag{15}
$$

FDR represents the probability that the abnormal conditions are correctly detected, which is an important criterion to compare between different methods in terms of their detection efficiency. Evidently, a very high FDR is desirable.

• False Alarm Rate (FAR):

$$
\begin{array}{r l} \text { FAR } & = \frac {\text { number   of   normal   data   that   have   been   detected   as   fault }}{\text { total   number   of   normal   samples }} \\ & = \frac {F P _ {i}}{T P _ {i} + T N _ {i}} \end{array}\tag{16}
$$

where the class corresponding to normal operation is considered as the positive class. FAR represents the probability that the normal operation is wrongly identified as abnormal, and thus, a very low FAR is desired and necessary.

The fault detection results obtained with the hierarchical LSTM SAE NN model are compared with both linear multivariate statistical methods and deep learning methods reported in previous studies. For a fair comparison between the methods, for studies where only non-incipient faults were considered, the results were compared to fault detection results obtained from the first level of the hierarchical structure model, whereas for studies where all the faults were considered, the comparisons were made for results obtained from the second level of the hierarchical structure model. The Fault Detection Rate (FDR) for all the faults is compared for the proposed method, PCA [23], DPCA [23], ICA [24], Convolutional NN (CNN) [25], Deep Stacked Network (DSN) [26], Stacked Autoencoder (SAE) [26], Generative Adversarial Network (GAN) [22] and One-Class SVM (OCSVM) [22]. The Fault Detection Rates for all non-incipient faults and incipient faults are shown in Tables 3 and $^ { 4 , }$ respectively, for different methodologies along with the results from the proposed method. It can been seen from Table 3 that the proposed method outperformed the linear multivariate methods and other DL-based methods for most fault modes. For ex ample, for PCA with 15 principal components, the average fault detection rates are 61.77% and 74.72% using the ${ \dot { T } } ^ { 2 }$ and Q statistic, respectively. Since the principal components extracted using PCA capture static correlations between variables, DPCA is used to ac count for temporal correlations (both auto-correlations and cross-correlations) in the data. The effect of increasing the number of time samples in the Tennessee Eastman simulation is also investigated following the hypothesis that increasing the time horizon will enhance classification accuracy. In the case of DPCA, the number of lags used in the observation matrix is a key parameter. Since DPCA is only a data compression technique, it must be combined with a classification model for the purpose of fault detection. Accordingly, the output features from the DPCA model are fed into an SVM model that is used for final classification. Different time horizons were tried for training the DPCA model. Based on validation results, the best DPCA model was obtained with 22 lags. The average detection rate obtained was 72.35%. The ICA [24]-based monitoring scheme performs better than both PCA and DPCA-based methods with an averaged accuracy of approximately 90%. It should be noted that all these methods (PCA, DPCA and ICA) perform poorly for detecting incipient faults.

In addition to the comparison to linear methods, the proposed methodology was also compared with different DNN architectures such as CNN [26], DSN [26], SAE-NN (results reported in Chadha and Schwung, 2017 [26]), GAN [22], and OCSVM (results reported in Spyridon and Boutalis, 2018 [22]) reported previously. It can be seen that the proposed method also outperforms these DNN-based methodologies. The relative advantage of our method versus these other DNN architectures (Table 4) is due to the inclusion of the incipient faults within the normal class (hierarchical structure) and the supervised autoencoder (SAE) DNN architecture. This reduces the confusion between the normal samples with other non-incipient faults. However, the additional advantage of the proposed method over the other DNN architectures is realized when the hierarchical structure is used in combination with the PRBS signals as further discussed below. It should be noted that all these comparisons were based on an identical data set. Similarly, the fault detection rate for all faults is compared with different DL-based models in Table 4 including SAE-NN, DSN, GAN, OCSVM, CNN, Optimized LSTM [39] and LSTM along with an attention mechanism [27]. It can be seen that the proposed methodology improves the averaged test classification accuracy for all faults significantly.

Subsequently, the faults were diagnosed using the proposed hierarchical structure where the first level model of the hierarchical structure classifies non-incipient faults and the second level model classifies incipient faults. For the first level model, there are 7382 training samples and 17,442 testing samples in total with a time horizon of 150 timesteps. The model consists of 182 encoder LSTM units, which is followed by 116 LSTM units for processing of the output of the encoding layer. Thereafter, the output of the second LSTM layer is passed through a dense layer for classification. Hyper-parameters such as number of layers, number of LSTM units in each layer, classification weights, learning rate, time-horizon etc. are selected using validation data that are part of the training dataset. The confusion matrix for level 1 model is presented in Figure 6. The hyper-parameter search is implemented using a Keras-tuner. Firstly, a grid of hyper-parameters is defined, for example the number of encoder layers = [1, 2, 3], number of LSTM units for each of these layers ranging from 2 to 200 with an interval of 2 = [10:2:200], learning rate = [0.1, 0.2, 0.3, 0.01], value of weights in the objective function, etc. Keras-tuner trains the model using different combinations of these hyper-parameters values, and the averaged validation accuracy is evaluated at every epoch. The models are trained with a few epochs at the start, and the selected models with high validation accuracy are chosen to be trained for more epochs. A study was conducted to select the optimal time horizon for the LSTM-based model. It can be seen from Figure 7 that the classification averages can be enhanced by extending the length of the time horizon of past data fed to the LSTM-based model. A total of 150 time steps were chosen as the optimal time-horizon.

![](images/b32d982aa0362a0b43ba78ff60384d7bb20fc6c61ab8bf9cf665eacc3d307d9c.jpg)  
Figure 6. Confusion matrix for the first level model of the hierarchical structure (i.e., classification of non-incipient faults and considering incipient faults as a normal class).

The next important design parameter for the second level hierarchical model is the location in the process at which the external excitation signal should be introduced to maximize information about the occurring incipient fault. In this work, this choice is based on the flow-sheet and by identifying which variables are mostly correlated to the incipient faults under consideration. Specifically, the excitation signals were added to process set-points in control loops that are most correlated to the incipient faults. When the selection of the variable to be excited by a PRBS is not obvious from the process flowsheet, a more systematic approach is to use sensitivity analysis, e.g., the sensitivity of changes in the variable connected to the fault to all process variables. Since it may be detrimental to perturb the set-point continuously by the PRBS signal, the latter can be introduced intermittently into the process. In the current work, an excitation signal of length 40 time-steps was intermittently introduced every 4 h into the process by assuming that such an event will not impact significantly the profitability of the process (for test data). Changes in the separator temperature set-point will force changes in the condenser temperature. Since the fault to be identified is stiction in the valve that affects the condenser temperature, the imposed PRBS in the separator set-point indirectly helps in identifying fault 15. For fault 9, i.e., a random variation in D feed temperature (refer to Table 2), the PRBS excitation $( \omega \in [ \omega _ { c l } , \omega _ { n } ]$ where $\omega _ { c l } = 0 . 0 0 8 7$ rad/s and $\omega _ { n } = 1 . 7 4$ rad/s) signal is introduced to the D feed ratio in order to create a suitable excitation. After developing this PRBS signal, we added both signals to the process at different times during the simulation. For fault 15, the PRBS signal is designed with a frequency range of $\omega \in [ \omega _ { c l } , \omega _ { n } ]$ where $\omega _ { c l } = 0 . 0 0 5$ rad/s and $\omega _ { n } = 1 . 7 4$ rad/s.

A systematic ablation study is conducted in Table 5 in order to demonstrate the gradual improvements in the results by showing fault detection rates of incipient faults, normal operation and non-incipient faults for 4 cases: i—without the hierarchical structure with one DL model, ii—with the hierarchical structure and iii—with the hierarchical structure and with the addition of one PRBS signal related to fault 15 and iv—with the hierarchical structure and with the addition of two PRBS signals related to fault 9 and fault 15. Other than a slight decrease in the detection of the Normal operation with the hierarchical structure and the addition of the two PRBS signals, the improvements in all other faults and in the average test accuracy are evident.

For the second level model, there are 1796 training samples and 4196 testing samples in total with a time horizon of 150 time-steps. The model consists of 284 encoder LSTM units in the first hidden layer, and the second layer consists of 100 LSTM units, which is followed by 278 LSTM units for processing of the output of the encoding layer. Thereafter, the output of the third LSTM layer is passed through a dense layer for classification. Hyperparameters such as the number of layers, number of LSTM units in each layer, classification weights, learning rate, time-horizon, weights in the loss function, etc. are selected using the validation data which is part of the training dataset. The hyper-parameter search is implemented again using the Keras-tuner. For the second level model, the samples corresponding to fault 0 (normal) and incipient faults are considered. Figure 8 shows the confusion matrix after introducing the PRBS signal that was designed for identifying fault 15 and Figure 9 shows the confusion matrix after introducing both PRBS signals that were designed for identifying fault 15 and fault 9. The total FAR calculated using Equation (16) was 2.41%.

The averaged fault classification rates for all non-incipient faults and for all faults (including incipient faults) are shown in Figures 10 and 11, respectively. Figure 10 shows a bar-chart comparison of the proposed method with several non-linear methods such as sparse representation [40], SVM [41], hierarchical model based method [42], Random Forest, and structural SVM. It can be seen that the hierarchical deep RNN-based method outperforms other methods with a significant margin. It should be noted that the comparisons made in Figure 10 do not consider incipient faults. In Figure 11, the averaged test accuracy of all faults (both incipient and non-incipient faults) are compared with other DL-based methods [43]. It can be seen that the second level hierarchical model combined with the introduction of the designed PRBS signals significantly improves the classification of the incipient faults, and thus, the averaged test accuracy for fault diagnosis increases significantly.

![](images/6a3ce554215dd15a29ceeffd2898cd203c02e4a4f59beb2cb0a8c32512c85266.jpg)  
Figure 7. Selection of optimal time horizon for Hierarchical LSTM-SAE Level 1 model.

Confusion Matrix

<table><tr><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>0.06</td><td>0.887</td><td>0.051</td><td>0</td></tr><tr><td>0.34</td><td>0.271</td><td>0.384</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>1</td></tr></table>

Predicted Label

Figure 8. Confusion matrix on test data for the second level model of the hierarchical structure: after adding designed PRBS signal with respect to fault 15.

<table><tr><td colspan="4">Confusion Matrix</td></tr><tr><td>0.98</td><td>0.018</td><td>0</td><td>0</td></tr><tr><td>0.18</td><td>0.815</td><td>0</td><td>0</td></tr><tr><td>0</td><td>0.006</td><td>0.994</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>1</td></tr></table>

Predicted Label  
Figure 9. Confusion matrix on test data for the second level model of the hierarchical structure: after adding designed PRBS signal with respect to fault 9 and fault 15.

![](images/c9340de99608264b3635bca486dcc5a973a5a18d89b45da0c3046f8425a18e33.jpg)

Figure 10. Comparison of averaged fault classification rates (non-incipient faults only).  
![](images/b9a7eaa6d9777350939aec7268c5c0bfb1f1a1d363aeead7e592d279c3a766c3.jpg)  
Figure 11. Comparison of averaged fault classification rates (all faults).

## 6. Conclusions

This work studied the application of a deep learning model within a hierarchical structure as a way to increase the detection and classification of faults in the Tennessee Eastman Process (TEP). The TEP simulation contains 20 different faults that were used during this study to make the classification problem. As previously reported by other researchers, a subset of these faults—referred to in this study as incipient—is particularly difficult to diagnose due to low signal-to-noise ratio and similarities in the resulting dynamic responses corresponding to different faults.

A comparison between deep learning techniques to a multivariate linear technique for fault detection such as PCA, DPCA, ICA and other deep learning methods is also presented. It is observed that a hierarchical LSTM-based model is superior to traditional linear and other deep learning-based methods for fault classification due to their ability to capture nonlinear dynamic behavior. It was also shown that the classification averages can be enhanced by extending the length of the time horizon of past data fed to the RNN-based model. However, most of these improvements in classification occurred for the non-incipient faults. Therefore, an active fault detection approach was pursued where a hierarchical model structure combined with external PRBS signals was proposed that proved to be particularly effective for classifying incipient faults. Future studies will address the trade-off between the impact of the injected PRBS signals on quality and productivity versus the benefit from the early detection of incipient faults.

Author Contributions: P.A.: methodology, formal analysis, data curation, writing—review and editing, software, visulaization; J.I.M.G.: methodology, writing and editing; A.E.: writing—review, supervision; H.B.: methodology, writing-editing and review, supervision, project administration, funding acquisition. All authors have read and agreed to the published version of the manuscript.

Funding: This work is the result of the research project supported by MITACS grant IT10393 through MITACS-Accelerate Program.

Institutional Review Board Statement: Not applicable.

Data Availability Statement: Not applicable.

Conflicts of Interest: The authors declare no conflict of interest.

## Abbreviations

The following abbreviations are used in this manuscript:

<table><tr><td>FDD</td><td>Fault Detection and Diagnosis</td></tr><tr><td>PCA</td><td>Principal Component Analysis</td></tr><tr><td>DPCA</td><td>Dynamic Principal Component Analysis</td></tr><tr><td>DNN</td><td>Deep Neural Network</td></tr><tr><td>NN</td><td>Neural Network</td></tr><tr><td>RNN</td><td>Recurrent Neural Network</td></tr><tr><td>TEP</td><td>Tennessee Eastman Process</td></tr><tr><td>LSTM</td><td>Long Short-Term Memory</td></tr><tr><td>GRU</td><td>Gated Recurrent Units</td></tr><tr><td>DLSTM-SAE NN</td><td>Deep LSTM Supervised Autoencoder Neural Network</td></tr><tr><td>DSAE-NN</td><td>Deep Supervised Autoencoder Neural Network</td></tr><tr><td>PRBS</td><td>Pseudo-Random Binary Signal</td></tr><tr><td>FDR</td><td>Fault Detection Rate</td></tr><tr><td>FAR</td><td>False Alarm Rate</td></tr><tr><td>GAN</td><td>Generative Adversarial Network</td></tr><tr><td>DSN</td><td>Deep Stacked Network</td></tr><tr><td>SAE</td><td>Stacked Autoencoder</td></tr><tr><td>OCSVM</td><td>One-Class SVM</td></tr><tr><td>SVM</td><td>Support Vector Machines</td></tr><tr><td>DL</td><td>Deep Learning</td></tr><tr><td>CNN</td><td>Convolutional Neural Network</td></tr></table>

## References

1. Chiang, L.H.; Russell, E.L.; Braatz, R.D. Fault diagnosis in chemical processes using Fisher discriminant analysis, discriminant partial least squares, and principal component analysis. Chemom. Intell. Lab. Syst. 2000, 50, 243–252. [CrossRef]

Hematillake, D.; Freethy, D.; McGivern, J.; McCready, C.; Agarwal, P.; Budman, H. Design and Optimization of a Penicillin Fed-Batch Reactor Based on a Deep Learning Fault Detection and Diagnostic Model. Ind. Eng. Chem. Res. 2022, 61, 4625–4637. [CrossRef]

3. Yin, S.; Ding, S.X.; Xie, X.; Luo, H. A review on basic data-driven approaches for industrial process monitoring. IEEE Trans. Ind. Electron. 2014, 61, 6418–6428. [CrossRef]

4. Agarwal, P.; Budman, H. Classification of Profit-Based Operating Regions for the Tennessee Eastman Process using Deep Learning Methods. IFAC-PapersOnLine 2019, 52, 556–561. [CrossRef]

5. Agarwal, P.; Tamer, M.; Sahraei, M.H.; Budman, H. Deep Learning for Classification of Profit-Based Operating Regions in Industrial Processes. Ind. Eng. Chem. Res. 2019, 59, 2378–2395. [CrossRef]

6. Agarwal, P.; Tamer, M.; Budman, H. Explainability: Relevance based dynamic deep learning algorithm for fault detection and diagnosis in chemical processes. Comput. Chem. Eng. 2021, 154, 107467. [CrossRef]

7. Agarwal, P.; Aghaee, M.; Tamer, M.; Budman, H. A novel unsupervised approach for batch process monitoring using deep learning. Comput. Chem. Eng. 2022, 159, 107694. [CrossRef]

8. Agarwal, P. Application of Deep Learning in Chemical Processes: Explainability, Monitoring and Observability. Ph.D. Thesis, University of Waterloo, Waterloo, ON, Canada, 2022.

9. Zhang, Y. Enhanced statistical analysis of nonlinear processes using KPCA, KICA and SVM. Chem. Eng. Sci. 2009, 64, 801–811. [CrossRef]

10. Yin, S.; Ding, S.X.; Haghani, A.; Hao, H.; Zhang, P. A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process. J. Process Control 2012, 22, 1567–1581. [CrossRef]

11. Lau, C.; Ghosh, K.; Hussain, M.A.; Hassan, C.C. Fault diagnosis of Tennessee Eastman process with multi-scale PCA and ANFIS. Chemom. Intell. Lab. Syst. 2013, 120, 1–14. [CrossRef]

12. Shams, M.B.; Budman, H.; Duever, T. Fault detection using CUSUM based techniques with application to the Tennessee Eastman Process. IFAC Proc. Vol. 2010, 43, 109–114. [CrossRef]

13. Ku, W.; Storer, R.H.; Georgakis, C. Disturbance detection and isolation by dynamic principal component analysis. Chemom. Intell. Lab. Syst. 1995, 30, 179–196. [CrossRef]

14. Rato, T.J.; Reis, M.S. Fault detection in the Tennessee Eastman benchmark process using dynamic principal components analysis based on decorrelated residuals (DPCA-DR). Chemom. Intell. Lab. Syst. 2013, 125, 101–108. [CrossRef]

15. Odiowei, P.E.P.; Cao, Y. Nonlinear dynamic process monitoring using canonical variate analysis and kernel density estimations. IEEE Trans. Ind. Inform. 2009, 6, 36–45. [CrossRef]

16. Isermann, R. Fault-Diagnosis Systems: An Introduction from Fault Detection to Fault Tolerance; Springer Science & Business Media: Berlin, Germany, 2005.

17. Shams, M.B.; Budman, H.; Duever, T. Finding a trade-off between observability and economics in the fault detection of chemical processes. Comput. Chem. Eng. 2011, 35, 319–328. [CrossRef]

18. Mhaskar, P.; Gani, A.; El-Farra, N.H.; McFall, C.; Christofides, P.D.; Davis, J.F. Integrated fault-detection and fault-tolerant control of process systems. AIChE J. 2006, 52, 2129–2148. [CrossRef]

19. Heirung, T.A.N.; Mesbah, A. Input design for active fault diagnosis. Annu. Rev. Control 2019, 47, 35–50. [CrossRef]

20. Cusidó, J.; Romeral, L.; Ortega, J.A.; Garcia, A.; Riba, J. Signal injection as a fault detection technique. Sensors 2011, 11, 3356–3380. [CrossRef]

21. Busch, R.; Peddle, I.K. Active fault detection for open loop stable LTI SISO systems. Int. J. Control Autom. Syst. 2014, 12, 324–332. [CrossRef]

22. Spyridon, P.; Boutalis, Y.S. Generative adversarial networks for unsupervised fault detection. In Proceedings of the 2018 European Control Conference (ECC), Limassol, Cyprus, 12–15 June 2018; pp. 691–696.

23. Lv, F.; Wen, C.; Bao, Z.; Liu, M. Fault diagnosis based on deep learning. In Proceedings of the 2016 American Control Conference (ACC), Boston, MA, USA, 6–8 July 2016; pp. 6851–6856.

24. Hsu, C.C.; Chen, M.C.; Chen, L.S. A novel process monitoring approach with dynamic independent component analysis. Control Eng. Pract. 2010, 18, 242–253. [CrossRef]

25. Singh Chadha, G.; Krishnamoorthy, M.; Schwung, A. Time Series based Fault Detection in Industrial Processes using Convolutional Neural Networks. In Proceedings of the IECON 2019—45th Annual Conference of the IEEE Industrial Electronics Society, Lisbon, Portugal, 14–17 October 2019; Volume 1, pp. 173–178. [CrossRef]

26. Chadha, G.S.; Schwung, A. Comparison of deep neural network architectures for fault detection in Tennessee Eastman process. In Proceedings of the 2017 22nd IEEE International Conference on Emerging Technologies and Factory Automation (ETFA), Limassol, Cyprus, 12–15 September 2017; pp. 1–8.

27. Li, Y. A fault prediction and cause identification approach in complex industrial processes based on deep learning. Comput. Intell. Neurosci. 2021, 2021, 6612342. [CrossRef] [PubMed]

28. Rumelhart, D.E.; Hinton, G.E.; Williams, R.J. Learning representations by back-propagating errors. Nature 1986, 323, 533–536. [CrossRef]

29. Bengio, Y.; Simard, P.; Frasconi, P. Learning long-term dependencies with gradient descent is difficult. IEEE Trans. Neural Netw. 1994, 5, 157–166. [CrossRef]

30. Hochreiter, S.; Schmidhuber, J. Long short-term memory. Neural Comput. 1997, 9, 1735–1780. [CrossRef]

31. Cho, K.; Van Merriënboer, B.; Gulcehre, C.; Bahdanau, D.; Bougares, F.; Schwenk, H.; Bengio, Y. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv 2014, arXiv:1406.1078.

32. Le, L.; Patterson, A.; White, M. Supervised autoencoders: Improving generalization performance with unsupervised regularizers. Adv. Neural Inf. Process. Syst. 2018, 31, 107–117.

33. Chollet, F. Keras: The Python Deep Learning library. Astrophys. Source Code Libr. 2018, ascl-1806.

34. Abadi, M.; Agarwal, A.; Barham, P.; Brevdo, E.; Chen, Z.; Citro, C.; Corrado, G.S.; Davis, A.; Dean, J.; Devin, M.; et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv 2016, arXiv:1603.04467.

35. Ljung, L. System identification. In Wiley Encyclopedia of Electrical and Electronics Engineering; 1999; pp. 1–19. Available online: https://www.diva-portal.org/smash/get/diva2:316967/FULLTEXT01.pdf (accessed on 19 October 2022).

36. Rivera, D.E.; Gaikwad, S.V. Systematic techniques for determining modelling requirements for SISO and MIMO feedback control. J. Process Control 1995, 5, 213–224. [CrossRef]

37. Garcia-Gabin, W.; Lundh, M. Input PRBS Design for Identification of Multivariable Systems. Available online: http://users.abo. fi/khaggblo/npcw21/submissions/27\_Garcia-Gabin&Lundh.pdf (accessed on 19 Ocotober 2022).

38. Lee, H.J.; Rivera, D.E. An Integrated Methodology for Plant-Friendly Input Signal Design and Control-Relevant Estimation of Highly Interactive Processes; American Institute of Chemical Engineers: New York, NY, USA, 2005.

39. Zhao, H.; Sun, S.; Jin, B. Sequential fault diagnosis based on LSTM neural network. IEEE Access 2018, 6, 12929–12939. [CrossRef]

40. Wu, L.; Chen, X.; Peng, Y.; Ye, Q.; Jiao, J. Fault detection and diagnosis based on sparse representation classification (SRC). In Proceedings of the 2012 IEEE International Conference on Robotics and Biomimetics (ROBIO), Guangzhou, China, 11–14 December 2012; pp. 926–931.

41. Yin, S.; Gao, X.; Karimi, H.R.; Zhu, X. Study on support vector machine-based fault detection in tennessee eastman process. In Abstract and Applied Analysis; Hindawi: London, UK, 2014; Volume 2014

42. Xie, D.; Bai, L. A Hierarchical Deep Neural Network for Fault Diagnosis on Tennessee-Eastman Process. In Proceedings of the 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), Miami, FL, USA, 9–11 December 2015; pp. 745–748. [CrossRef]

43. Luo, L.; Xie, L.; Su, H. Deep Learning With Tensor Factorization Layers for Sequential Fault Diagnosis and Industrial Process Monitoring. IEEE Access 2020, 8, 105494–105506. [CrossRef]