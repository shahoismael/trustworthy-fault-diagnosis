# Fault Diagnosis of the Dynamic Chemical Process Based on the Optimized CNN-LSTM Network

Honghua Chen, Jian Cen,\* Zhuohong Yang, Weiwei Si, and Hongchao Cheng

![](images/c5da9adc83968c630c360b1735fc10a351b214b3bdf045b97a9c055e44917e82.jpg)

Cite This: ACS Omega 2022, 7, 34389−34400

![](images/754255f8a4f54fc18a8cc051a7773c8fade98acf7eb46e0bcd74661b30781bdc.jpg)

Read Online

ACCESS

Metrics & More

ABSTRACT: Deep learning provides new ideas for chemical process fault diagnosis, reducing potential risks and ensuring safe process operation in recent years. To address the problem that existing methods have dificulty extracting the dynamic fault features of a chemical process, a fusion model (CS-IMLSTM) based on a convolutional neural network (CNN), squeeze-andexcitation (SE) attention mechanism, and improved long shortterm memory network (IMLSTM) is developed for chemical process fault diagnosis in this paper. First, an extended sliding window is utilized to transform data into augmented dynamic data to enhance the dynamic features. Second, the SE is utilized to optimize the key fault features of augmented dynamic data extracted by CNN. Then, IMLSTM is used to balance fault

Article Recommendations  
![](images/c3f7df29a5aa3a6ce0a2c4aaf99947cd3e65210ad07090ead1fefb6c094201a8.jpg)  
information and further mine the dynamic features of time series data. Finally. the feasibility of the proposed method is verified in the Tennessee-Eastman process (TEP). The average accuracies of this method in two subdata sets of TEP are 98.29% and 97.74%, respectively. Compared with the traditional CNN-LSTM model, the proposed method improves the average accuracies by 5.18% and 2.10%, respectively. Experimental results confirm that the method developed in this paper is suitable for chemical process fault diagnosis.

## 1. INTRODUCTION

Chemical processes play a pivotal role in the development of the world economy and in the lives of people. New technologies, new equipment, and new materials are emerging, production scales are expanding, processes are becoming more complex, and operating environments are harsh, resulting in chemical process risks everywhere. Once a safety accident occurs, it will bring serious damage to people’s lives and health, the ecological environment, social stability, and the enterprise economy.

Abnormal situation management (ASM) provides an early warning for abnormal situations, timely diagnoses the causes, and provides decision support for technicians to take measures and restore the process to normal, which has made a great contribution to improving process safety.<sup>1</sup> Proper risk assessment (RA) helps to control the risks before they occur. Fault detection and fault diagnosis (FDD) means detecting whether faults have occurred and, if so, classifying the fault. From a process safety perspective, FDD, RA, and ASM can form a closed loop. Among them, FDD is a key step to identify potential risks. RA evaluates the risk margin based on fault information provided by FDD and reports risk events to ASM. ASM makes decisions to ensure process security based on the feedback.<sup>2</sup> Khan et al.<sup>3</sup> pointed out that process security could be improved by integrating dynamic FDD with RA. Dai et al.<sup>4</sup> proposed that FDD is an efective way to control and mitigate process risks.

From the perspective of risk engineering, process safety usability is to efectively detect and diagnose faults.<sup>5</sup> Therefore, the development of an intelligent and eficient FDD system is the key to maintain the ideal performance of digital industrial processes and safety in production.

The integration of FDD with process safety and risk assessment is an interesting research area. Amin et al.<sup>6</sup> proposed a risk-based FDD method. This method carried out fault detection and diagnosis in the monitored risk profile. Experimental results showed that this method has better diagnostic performance than PCA and transfer entropy. Bao et al.<sup>7</sup> proposed a risk-based process safety fault diagnosis technology. The advantage of this method is to identify and determine potential faults by risk index and realize the development of fault diagnosis technology from single variable to multivariable monitoring. Bhadriraju et al.<sup>8</sup> designed an operational adaptive sparse system recognition to solve the problem that an ofline training model has dificulty capturing the dynamic behavior of a process. The experimental result showed that the process behavior prediction based on this system can efectively predict faults and assess risks. From the perspective of process safety, it is again clear that FDD is an efective initiative to minimize risk and guarantee the safe operation of complex industrial processes.

![](images/456ebfce8144166cadaf06fc3ac9542926cdf28de172733624c72cd2af524a2a.jpg)

Data-driven FDD methods can avoid the dependence on a complex process mechanism and mine the high-value information hidden in process data. Data driven methods can be further divided into multivariate statistical methods, machine learning methods, and deep learning methods. Multivariate statistical methods include principal component analysis (PCA), independent component analysis (ICA), partial least-squares (PLS), a Gaussian mixture model (GMM), and their variants. At present, multivariate statistical methods are mainly used for fault detection. Deng et al.<sup>10</sup> proposed two local kernel PCA (KPCA) to solve the problem of missing local data information on KPCA in the case of an early fault. The method has been verified efectively in a continuous stirred tank reactor (CSTR). For fault detection of complex processes, a single method is usually not as superior as the detection results obtained by hybrid methods.<sup>4</sup> Han et al.<sup>11</sup> adopted the hybrid fault detection method of adaptive kernel PCA and gray correlation analysis, which is superior to single kernel PCA and can provide a basis for ASM. Fault detection based on multivariate statistical learning methods usually depend on the threshold value of calculation to judge whether the fault exists. If the threshold value calculation slightly deviates, it may lead to the wrong result of fault diagnosis and increase the process risk.

Machine learning methods include locality preserving projection (LPP), naïve bayes (NB), and support vector machine (SVM). He et al.<sup>12</sup> proposed a new discriminant LPP algorithm (DLPP) combined with Monte Carlo sampling, which not only solves the problem of high-dimensional process data but also solves the problem that DLPP is limited by a small sample size, thus efectively improving the fault diagnosis performance of industrial processes. Zhang et al.<sup>13</sup> proposed an improved LPP and AdaBoost integration method. The improved LPP based on the heat-kernel and cosine weights can efectively extract the internal structure feature of data, so high fault diagnosis accuracy can be achieved in two chemical processes. Zhang et al.<sup>14</sup> constructed a new farthest−nearest distance neighborhood and locality projections method and used it to reduce the dimension of high-dimensional process data to extract discriminant features. NB was adopted as a classifier for process fault diagnosis. Amin and Khan et al.<sup>15</sup> proposed a hybrid diagnosis method of PCA and BN. This method achieved good diagnostic performance in a continuous stirred tank heater and binary distillation column because it used the correlation dimension to select the principal component and combined a vine copula and the BN theorem to capture the nonlinear dependence of high dimensional process data. Deng et al.<sup>16</sup> proposed a fault detection method based on the integration of the spatial compression matrix and NB, which reduces the complexity of learning and helps to speed up the management of production risks. Machine learning can achieve a better efect of FDD when a small sample is used. However, the fault diagnosis ability of these methods depends on the quality of feature extraction, which has certain limitations on the dynamic feature representation of process data.

Deep learning is widely regarded as an efective tool for fault diagnosis in modern industrial applications. Diagnostic models based on classical deep learning include convolutional neural network (CNN), deep belief network (DBN), stacked autoencoder (SAE), and long short-term memory network (LSTM). Among them, CNN has achieved a more advanced performance. In 2018, Wu et al.<sup>17</sup> used deep CNN (DCNN) for fault diagnosis of TEP. The time-frequency domain features of process variables are converted into two-dimensional (2D) matrices, which are input into DCNN to extract spatial features of variables, and then fault classification is carried out. This method has achieved 88.2% classification accuracy. Song et al.<sup>18</sup> used matrix maps and multiscale CNN for chemical process fault diagnosis, and the classification accuracy is 88.54%. For the above models, process variables need to be converted into 2D matrices or complex images as inputs of CNN, which leads to a large consumption of computing resources. Yu et al.<sup>19</sup> designed a multichannel 1D CNN model (MC1-DCNN) on the basis of a wavelet transform and applied it to batch-fed fermentation of penicillin and TEP. The result shows that MC1-DCNN has the ability to learn high-dimensional process signal characteristics and a good performance of fault diagnosis. Yu et al.<sup>20</sup> designed a broad CNN with incremental learning ability, which is characterized by self-renewal in the face of new faults.

In addition, LSTM has also attracted scholars’ attention in industrial process fault diagnosis as a result of its stronger adaptability in time series data analysis. Zhao et al.<sup>21</sup> came up with an end-to-end sequential fault diagnosis method based on LSTM to address the problem that most conventional fault diagnosis techniques cannot learn dynamic features from raw data. Han et al.<sup>22</sup> presented an optimized LSTM, which improves the accuracy of diagnosis of single and multiple faults in TEP by optimizing the number of hidden laver nodes of the LSTM. Park et al.<sup>23</sup> proposed an integration method of convolutional LSTM and autoencoder to detect rare faults in industrial processes. Gravanis et al.<sup>24</sup> combined the feature reduction method with LSTM and a time delay network, respectively, to conduct FDD of nonlinear processes. The fault diagnosis model based on the combination of CNN and LSTM has become a research hotspot because of its ability to extract spatial and temporal features of industrial data and improve diagnosis performance. Shao et al.<sup>25</sup> used a multichannel LSTM-CNN (MCLSTM-CNN) fault diagnosis model. This method inputs a data set into LSTM and then uses multiple paralle convolution layers to mine the output features of the hidden layer at the same time. The research indicates that the fault diagnosis accuracy of applying MCLSTM-CNN to TEP is as high as 92.06%. Wang et al.<sup>26</sup> designed the feature extraction method of the LSTM-CNN parallel structure and then fused and compressed the features by MLP. This method can extract the temporal and spatial features of process variables. so as to improve the diagnostic performance. Yuan et al.<sup>27</sup> used a chemical process monitoring and fault diagnosis scheme based on multiscale CNN-LSTM, with the purpose of mining highdimensional fault features in a multiscale and hierarchical manner. Huang et al.<sup>28</sup> transformed the process data into twodimensional data and input it into CNN-LSTM to extract the spatial and delay characteristics of the data. This method improved the diagnostic accuracy and noise sensitivity. However, the following problems still exist in industrial process fault diagnosis based on CNN, LSTM, or CNN-LSTM:

(1) CNN in the above literature is only a series of convolution layers, connecting features in the channel dimension. However, fault data is usually composed of many variables collected by many sensors in an industrial process. Each variable provides a diferent degree of importance of distinguishing features for fault diagnosis. Therefore, the above research methods lack the proper mechanism to reflect the correlation and importance of fault dynamic characteristics between diferent channels.

![](images/5bdc0a10a76f24e9056273628d8e155bfeb0f7bac1df432083dc76196238ac3d.jpg)  
Figure 1. Basic structure of the CNN.

(2) The above methods used LSTM to extract time series characteristics of chemical process data. However, there will be the problem of an unbalanced distribution of fault dynamic information because of the special gating mechanism. Therefore, the fault information on time series data cannot be extracted eficiently.

The problems mentioned above can make it dificult for the traditional CNN-LSTM network and other forms of this network to extract dynamic fault features of chemical process dynamic data. Therefore, this paper designs a model which combines CNN, the SE attention mechanism, and improved LSTM (CS-IMLSTM) for the fault diagnosis of TEP. First, the time series of industrial process variables contains the dynamic evolution process of the faults. Therefore, to enhance the dynamic characteristics of sequential fault data, extended sliding window preprocessing technology is proposed to obtain the augmented dynamic data, which provide suficient fault diagnosis information for the proposed model. Second, aiming at the problem that a single network CNN cannot automatically select important channel features, a network architecture combining the CNN and SE attention mechanism (CS) is proposed, which makes the proposed model give more weight to critical channel fault features and reduce attention to redundant features. Finally, an improved LSTM is proposed to optimize the gating mechanism of original LSTM and balance the characteristic information on the industrial process in the time dimension. It is helpful for the proposed model to further mine dynamic information on industrial process fault data. The proposed method can not only adaptively extract dynamic fault features, weighting the features of diferent channels, but also balance fault information. Cascaded CS-IMLSTM can simultaneously extract the spatial and temporal dynamic features of process data, so as to enhance the capabilities of industrial process fault diagnosis. In terms of process safety, the proposed method can minimize the risk of industrial process operation and improve the safety of chemical process production.

## 2. RELATED THEORIES

2.1. Convolutional Neural Network. CNN has received extensive attention in the field of industrial fault diagnosis. CNN adaptively learns the spatial features of data by back-propagating using multiple blocks such as the convolution layer and pooling layer.<sup>29</sup> The basic structure is shown in Figure 1.

The convolution layer is mainly used to mine the local features of input. The mathematical expression for the convolution layer is

$$
x ^ {i} = f \left(\sum w ^ {i} \times x ^ {(i - 1)} + b ^ {i}\right)\tag{1}
$$

where i represents layer i of the network and $x ^ { i }$ denotes the output of feature data at the layer i. Similarly, $x ^ { i - 1 }$ is the input data of layer i. b<sup>i</sup> denotes the bias of layer $i ,$ and ω<sup>i</sup> is the convolution kernel $\cdot f ( \cdot )$ denotes activation function. LeakyReLU is used as the activation function, and its mathematical model can be represented as

$$
x ^ {j} = \left\{ \begin{array}{l} x ^ {j ^ {\prime}},   x ^ {j ^ {\prime}} > 0 \\ \alpha x ^ {j ^ {\prime}},   x ^ {j ^ {\prime}} \leq 0 \end{array} \right\}\tag{2}
$$

Here α is the fixed parameter, and $\alpha = 0 . 0 1$

The pooling layer can reduce data redundancy, preserving the key elements of the feature map and controlling overfitting. The mathematical model of pool operation can be expressed as

$$
x ^ {m ^ {\prime}, n ^ {\prime}} = p o o l (x ^ {m, n})\tag{3}
$$

where $x ^ { m , n }$ and $x ^ { m / n \prime }$ respectively represent the values before and after the pooling operation of the point $( m , n )$ in the output feature graph of the convolution layer.

The fully connected (FC) layer map features are extracted from the convolutional layer and down-sampled by the pooling layer to the sample label space. For specific information about CNN. please refer to the literature.30

2.2. Squeeze-and-Excitation Attention Mechanism. Recently, the benefits of attention mechanisms have been demonstrated in a variety of tasks. However, the advantage of the attention mechanism in chemical process fault diagnosis has not been fully exploited. Therefore, this paper uses an attention mechanism to mine important features of fault data. The emergence of the SE block is to work out a loss problem caused by diferent proportions of feature map channels in convolution operation and improve the depth representation ability of the

![](images/438ca1a9583f9a42a26d96db701787a8d59921421fad7673d661a2b1d2de2979.jpg)  
Figure 2. Basic structure of the SE block.

![](images/11b5cfa0df3fc4900dd90f50fe0c3c1186ae9b0f78af7bf9354e02afd5f0e5b9.jpg)  
Figure 3. Structures of the (a) LSTM and (b) IMLSTM.

CNN. The SE block can model dynamic nonlinear depend encies between channels using global information learned by the CNN. Thus, it can enhance feature information that is efective for fault classification and suppress the inefective feature information. The structure of the SE block is presented in Figure 2. For the detailed process of the SE block, the reader is referred to ref 31, and its brief process is as follows:

G i v e n t r a n s f o r m a t i o n $\boldsymbol { F } _ { t r } ,$ l e t $\begin{array} { r c l } { \textbf { U } } & { = } & { F _ { t r } \big ( \textbf X \big ) } \end{array}$ $\mathbf { X } \in \mathbb { R } ^ { H \times W ^ { \prime } \times C \prime } , \mathbf { U } \in \mathbb { R } ^ { H \times W \times C }$ . Assuming that $\mathbf { F } _ { t r }$ is a convolution operator, the feature map of X can be expressed as ${ \bf U } = { \bf \Psi }$ $\big [ \mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } , . . . , \mathbf { u } _ { C } \big ] ,$ , and

$$
\mathbf {u} _ {c} = \mathbf {k} _ {c} ^ {*} \mathbf {X}\tag{4}
$$

where $\mathbf { u } _ { c } \in \mathbb { R } ^ { H \times W }$ and $\mathbf { K } = \left[ \mathbf { k } _ { 1 } , \mathbf { k } _ { 2 } , . . . , \mathbf { k } _ { C } \right]$ represents the learned filter kernels. \* refers to the convolution operation.

In the squeeze operation, the spatial dimension H × W of U is compressed by global average pooling to obtain the channel statistic $z \in \mathbb { R } ^ { C }$ . The cth element in z can be expressed as

$$
\mathbf {z} _ {c} = F _ {s q} (\mathbf {u} _ {c})\tag{5}
$$

In the excitation operation, a gating mechanism with sigmoid function is utilized to obtain the dependencies between channels. The operation can be expressed as

$$
\mathbf {s} = F _ {e x} (\mathbf {z}, \mathbf {W})\tag{6}
$$

Here, $\pmb { \mathscr { s } } \in \mathbb { R } ^ { C }$ is the channel weight. W is a parameter that needs to be learned. The final output $\widetilde { \mathbf { X } } = [ \widetilde { \mathbf { x } } _ { 1 } , \widetilde { \widetilde { \mathbf { x } } } _ { 2 } , \widetilde { \mathbf { \ x } } _ { C } ]$ of the SE block is generated by the scaling operation

$$
\tilde {\mathbf {x}} _ {c} = \mathbf {F} _ {s c} (\mathbf {u} _ {c}, s _ {c}) = s _ {c} \mathbf {u} _ {c}\tag{7}
$$

Here, $\tilde { \mathbf { X } } \in \mathbb { R } ^ { H \times W \times C } , \mathbf { F } _ { s c } ( \mathbf { u } _ { c } , s _ { c } )$ is channel-wise between scalar $s _ { c }$ and the feature map $\mathbf { u } _ { c } \in \mathbb { R } ^ { H \times W }$

2.3. Improved Long Short-Term Memory Network. Hochreiter et $\mathrm { a l . } ^ { 3 2 }$ proposed LSTM, which can maintain the nondispersion of a data gradient over a long time span. LSTM has recently been successful in various areas of sequence modeling, including but not limited to speech recognition and machine translation. The basic structure of LSTM is presented in Figure 3a. Key elements in the LSTM layer include input gate $i _ { t }$ forget gate $f _ { t }$ output gate $o _ { t }$ and internal memory cell $c _ { t } .$ Moreover, each logic gate has its own parameters $( U , { \dot { W } } , b ) ,$ , so that information can be filtered at the corresponding position, the weight of useful information can be enhanced, and redundant information can be efectively filtered.

![](images/c9e28992ee83bf673a6672dc0b0ee2c685a9970ecc6f0eedc064aee02540bd2e.jpg)

(1) The forget gate $f _ { t }$ is expressed by the following equation:

$$
f _ {t} = \delta (W _ {f} x _ {t} + U _ {f} h _ {t - 1} + b _ {f})\tag{8}
$$

where $\delta ( \cdot )$ is a sigmoid function and $0 < f _ { t } < 1$

(2) The input gate $i _ { t }$ can be expressed by the following equation:

$$
i _ {t} = \delta (W _ {i} x _ {t} + U _ {i} h _ {t - 1} + b _ {i})\tag{9}
$$

$$
c _ {t} ^ {\prime} = \tanh (W _ {c} x _ {t} + U _ {c} h _ {t - 1} + b _ {c})\tag{10}
$$

where tanh(·) denotes the hyperbolic tangent activation function and $0 < i _ { t } < 1$

(3) The internal memory cell $c _ { t }$ is expressed by the following equation:

$$
c _ {t} = f _ {t} \odot c _ {t - 1} + i _ {t} \odot c _ {t} ^ {\prime}\tag{11}
$$

(4) The output gate $o _ { t }$ can be expressed by the following equation:

$$
o _ {t} = \delta (W _ {o} x _ {t} + U _ {o} h _ {t - 1} + b _ {o})\tag{12}
$$

$$
h _ {t} = o _ {t} \odot \tanh (c _ {t})\tag{13}
$$

The expression formula of the improved internal memory cell $c _ { t }$ is as follows:

$$
g _ {t} = 1 / (f _ {t} + i _ {t})\tag{14}
$$

![](images/d6baa32b2873d8e9a5c953058987eff070c6c124d3be255d782aff572a1884ef.jpg)  
Figure 4. Extended sliding window mechanism schematic.

![](images/d73db12c8aba3be1887df16a44176aad9e29d683705f99638ca8419073c31f70.jpg)  
Figure 5. Fault diagnosis flow based on the CS-IMLSTM model.

$$
c _ {t} = f _ {t} g _ {t} \odot c _ {t - 1} + i _ {t} g _ {t} \odot c _ {t} ^ {\prime}\tag{15}
$$

It can be seen from eqs 8−13 that the forgetting gate and input gate of LSTM are independent of each other. However, the values $\operatorname { o f } f _ { t }$ and $i _ { t }$ respectively determine the degree of retention for the previous moment internal memory cell $c _ { t - 1 }$ and current moment memory cell in eq 11.<sup>22</sup> This also means that when it is applied to complex chemical process fault diagnosis, the internal memory cell $c _ { t }$ at the current moment will excessively rely on $c _ { t - 1 }$ or $c _ { t } ^ { \prime } \ \mathrm { i f } \ f _ { t }$ or $i _ { t }$ approaches 1, which will lead to the problem of unbalanced fault features of the chemical process.<sup>33</sup>

The internal memory cell $c _ { t } \operatorname { o f L S T M }$ is improved to solve the above problem. The structure of IMLSTM is shown in Figure 3b, where improved $c _ { t }$ is shown in eqs 14 and 15. The introduction of g in $c _ { t }$ makes the degree of information retention in $c _ { t - 1 }$ dependent on $f _ { t } / f _ { t } + i _ { t } )$ and not only on $f _ { t }$ Similarly, the degree of information retention in $c _ { t } ^ { \prime }$ depends on $i _ { t } / \left( f _ { t } + i _ { t } \right)$ . By balancing the information on forgetting and input gates, IMLSTM can process the dynamic features of temporal data more eficiently.

![](images/005a7b6ed1e0e9072e055d0921b916e0115bd64d42257dd2017b612a24ebf1af.jpg)  
Figure 6. Flowchart of TEP.<sup>36</sup> Reprinted with permission from ref 36. Copyright 2019 Elsevier.

## 3. PROPOSED METHOD

3.1. Data Preprocessing. Data collected by industrial processes are usually dynamic; that is, faults occurring at the current moment may depend on changes in system state at the previous moment.<sup>28</sup> It is dificult to describe the change characteristics of industrial processes accurately by establishing a single global diagnostic model. In this paper, an extended sliding window mechanism is introduced to transform raw data into augmented dynamic data. The whole process is transformed into a time-varying dynamic process, and a local model is established. With the continuous change of the process, the model needs to be constantly updated to adapt to this change, which can be more accurate in the analysis of new samples and is more conducive to the proposed model to mine the dynamic feature information on time series data. The principle of the extended sliding window mechanism is shown in Figure 4. Formula 16 represents raw data set X and its corresponding labels $\mathbf { Y } ,$ where n and m respectively refer to the number of observed samples and variables and $\mathbf { x } _ { t } = \left( x _ { t 1 } , \ x _ { t 2 } , \ . . . . , \ x _ { t m } \right)$ denotes observed variables collected by industrial process at moment t. Let the sliding step of the sliding window be $S \left( S \in \mathbf { N } ^ { * } \right.$ and $S \le L )$ , and L is the length of the sliding window. As shown in Formula 17, dynamic data D and corresponding label $\mathbf { Y _ { d } }$ can be obtained by extended sliding window operation on the raw data set, which is the input of the proposed model.

After the extended sliding window operation, the raw data set is transformed into an augmented dynamic data set, which allows the proposed model to be able to capture the features of small changes in observed variables and learn dynamic information. Thus, the performance of fault diagnosis in an industrial process can be greatly improved.

$$
\mathbf {X} = \left( \begin{array}{c c c c} x _ {1 1} & x _ {1 2} & \dots & x _ {1 m} \\ \vdots & \vdots & \dots & \vdots \\ x _ {t 1} & x _ {t 2} & \dots & x _ {t m} \\ \vdots & \vdots & \dots & \vdots \\ a _ {n 1} & x _ {n 2} & \dots & a _ {n m} \end{array} \right) = (\mathbf {x} _ {1} \dots \mathbf {x} _ {t} \dots \mathbf {x} _ {n}) ^ {T} \quad \text {and} \quad \mathbf {Y} = \left( \begin{array}{c} y _ {1} \\ \vdots \\ y _ {t} \\ \vdots \\ y _ {n} \end{array} \right)\tag{16}
$$

$$
\mathbf {D} = \left( \begin{array}{c c c c} \mathbf {x _ {1}} & \mathbf {x _ {2}} & \dots & \mathbf {x _ {L}} \\ \vdots & \vdots & \dots & \vdots \\ \mathbf {x _ {t}} & \mathbf {x _ {t + 1}} & \dots & \mathbf {x _ {t + L - 1}} \\ \vdots & \vdots & \dots & \vdots \\ \mathbf {x _ {n - L + 1}} & \mathbf {x _ {n - L + 2}} & \dots & \mathbf {x _ {n}} \end{array} \right) \quad \text {and} \quad \mathbf {Y} _ {d} = \left( \begin{array}{c} y _ {L} \\ \vdots \\ y _ {t + L - 1} \\ \vdots \\ y _ {n} \end{array} \right)\tag{17}
$$

3.2. Diagnostic Process of the Proposed Method. In recent years, CNN, LSTM, CNN-LSTM, and their variants have been widely used in the field of fault diagnosis, but these deep learning methods have dificulty capturing the dynamic characteristics of dynamic data in the process industry. In this paper, we aim to build a diagnostic model of dynamic chemical processes based on an optimized CNN-LSTM (CS-IMLSTM) network. The fault diagnosis flowchart based on CS-IMLSTM is shown in Figure 5. The convolution layer extracts spatial features of data. The batch normalization (BN) layer improves the training speed and mitigates the risk of overfitting. LeakyReLU increases the network sparsity. The pooling layer reduces the number of model parameters and optimizes the workload. The SE block weights important channel features. The IMLSTM balances the fault information and extracts the temporal dynamic features of the data. The FC layer bridges all features and feeds the output values into classifiers for classification. CS-IMLSTM is an efective improvement of CNN-LSTM. It is worth noting that the proposed method uses CS-IMLSTM combined with the extended sliding window mechanism, which can not only automatically extract spatial and temporal features from the original industrial data but also perceive the deep dynamic information, so as to realize the identification of diferent fault types, optimize decision-making for risk assessment and ASM, and help the process run safely and steadily for a long period. The proposed method consists of the following five core steps:

Table 1. Fault Modes of Case 1 and Case 2

<table><tr><td>case</td><td>fault</td><td>fault cause</td><td>fault type</td></tr><tr><td rowspan="5">Case 1</td><td>1</td><td>A/C feed ratio fluctuates, B feed is stable</td><td>Step</td></tr><tr><td>2</td><td>B feed fluctuates, A/C feed ratio is stable</td><td>Step</td></tr><tr><td>6</td><td>A material leak</td><td>Step</td></tr><tr><td>7</td><td>Feed C inlet pressure loss: availability reduction</td><td>Step</td></tr><tr><td>8</td><td>A, B, C feed composition fluctuation</td><td>Random variable</td></tr><tr><td rowspan="5">Case 2</td><td>4</td><td>Temperature disturbance at reactor cooling water inlet</td><td>Step</td></tr><tr><td>5</td><td>Temperature disturbance at reactor cooling water inlet</td><td>Step</td></tr><tr><td>10</td><td>C feed temperature disturbance</td><td>Random variable</td></tr><tr><td>11</td><td>Inlet temperature fluctuation of reactor cooling water</td><td>Random variable</td></tr><tr><td>12</td><td>Inlet temperature fluctuation of condenser cooling water</td><td>Random variable</td></tr></table>

(1) Industrial process fault data and corresponding labels are collected.

(2) The extended sliding window mechanism is used to generate augmented dynamic data by setting the sliding step S and sliding window length L.

(3) The training set and corresponding label are fed into the CS-IMLSTM network. CS is used to extract spatial features of data and enhance critical fault features. The spatial feature vector of the data is transformed and input to IMLSTM. IMLSTM is used to balance the fault information and further extract dynamic features of augmented industrial data.

(4) The extracted features are input into the classifier for fault classification, and the trained model is saved.

(5) After extended sliding window processing, the test data and corresponding label are input to the trained model to prove the eficiency of the model.

## 4. EXPERIMENTAL VERIFICATION

4.1. Introduction to the Tennessee-Eastman Process. TEP is a simulation process developed by Eastman Chemical process. Therefore, the TEP is often taken as a simulation example to assess the feasibility of fault detection and diagnosis methods for industrial processes. The flowchart of the TEP is displayed in Figure 6. The TEP mainly consists of five operation units: reactor, condenser, gas−liquid separator, vapor extraction tower, and circulating compressor. The chemical reactions occurring in the TEP involve a total of eight components, where the reactants include gaseous substances $\begin{array} { r } { \dot { \bf A } , { \bf C } , { \bf D } , } \end{array}$ , and E and inert catalyst B and the products include liquid products G and H and byproduct F.

There are 52 variables in the overall process, including 11 control variables, 19 component measurement variables, and 22 continuous process variables. TEP can generate a data set of 1 normal state and 21 diferent fault states. Referring to the into two cases with the aim of verifying the generalization ability and robustness of the proposed method. Generally, the selected 10 fault data have large overlap and are dificult to classify.<sup>14,35</sup> The 10 fault types and descriptions are shown in Table 1. The fault type in case 1 is afected by feed and flow, and case 2 is afected by temperature. The faults in both cases occur under diferent operating conditions. Therefore, industrial process faults under diferent working conditions are diagnosed to verify the feasibility of our method. Each fault state includes 480 raw training samples and 800 raw test samples, respectively. Each sample is sampled at a frequency of 3 min.

4.2. Application Research of the Proposed Method in TEP Fault Diagnosis. With the goal of verifying the feasibility of the proposed method, we tested two subdata sets of TEP, and the accuracy of the test sets is taken as the efective performance of the industrial process fault diagnosis. All experiments are performed in Python 3.8 and Pytorch, running on Ubantu 18.04 with 64GB RAM and an NVIDIA Quadro P4000 GPU.

4.2.1. Experimental Setup. The extended sliding window mechanism is adopted to convert the raw data set X into the augmented dynamic data, and the sliding step S = 1 and the sliding window length $L \ = \ 2 0$ are set to ensure that the augmented data D has enough dynamic information for neural network learning. Table 2 clearly reflects the number of samples

Table 2. Sample Size of Raw Data and Augmented Dynamic Data

<table><tr><td>fault</td><td>data set</td><td>sample size of raw data</td><td>sample size of augmented dynamic data</td></tr><tr><td rowspan="2">Each fault</td><td>Train</td><td>480</td><td>461</td></tr><tr><td>Test</td><td>800</td><td>781</td></tr></table>

in the raw data set as well as the number of samples processed by the extended sliding window. Thus, the total number of train samples in each case is 461 $\times 5 = 2 3 0 5 ,$ , and the total number of test samples is $7 8 1 \times 5 = 3 9 0 5 .$ . It is worth noting that here each sample has $5 2 \times 2 0 = 1 0 4 0$ . In addition, we will draw 25% of data from the training sets of each case as the validation set during training.

In the training of the proposed model, the batch size is $^ { 3 2 , }$ learning rate is 0.001, convolution kernel is $1 \times 3 ,$ and maxpooling kernel is set to 1 × 2. We choose Adam as the optimizer, use a cross-entropy loss function to evaluate the performance of the network, and use back-propagation to update the weights. For the sake of verifying the superiority of the proposed method, we set up five ablation comparison experiments. The hyper parameters of the models are approximately the same, and the complexity is approximately equal. The structure and other parameters of the diferent models are set as shown in Table 3.

Table 3. Model Structure and Parameter Settings

<table><tr><td>model</td><td> $structure^a$ </td></tr><tr><td>CS-IMLSTM</td><td>CONV(32)-SE(32)-CONV(64)-SE(64)-CONV(64)-SE(64)-FC*(512)-IMLSTM(1024)-FC(5)</td></tr><tr><td>CNN-IMLSTM</td><td>CONV(32)-CONV(64)-CONV(64)-FC*(512)-IMLSTM(1024)-FC(5)</td></tr><tr><td>CS-LSTM</td><td>CONV(32)-SE(32)-CONV(64)-SE(64)-CONV(64)-SE(64)-FC*(512)-LSTM(1024)-FC(5)</td></tr><tr><td>CNN-LSTM</td><td>CONV(32)-CONV(64)-CONV(64)-FC*(512)-LSTM(1024)-FC(5)</td></tr><tr><td>LSTM</td><td>Lstm1(1024)-lstm2(1024)-lstm3(1024)-lstm4(1024)-FC(5)</td></tr><tr><td>CS-CNN</td><td>CONV(32)-SE(32)-CONV(64)-SE(64)-CONV(64)-SE(64)-FC*(512)-CNN(512)-FC(5)</td></tr></table>

<sup>a</sup>For convenience, the CONV(@) module is used to denote Conv1d(@)-BN(@)-LeakyReLU-maxpooling(@), where @ denotes the output channel. \* indicates FC layer with dropout rate of p = 0.5.

![](images/0e36396ea5872ca3139bcc8ce702ec92785f47b5f8871f601d25bca090c0b790.jpg)

![](images/9b14aca152a0edeb882e63e81ab332fe9ea5642dd50ccb1668fd7e02d23c113f.jpg)

(a)  
Figure 7. The 10 times average training loss curves of (a) case 1 and (b) case 2 on diferent models.  
![](images/0cdddf6195e8323f0dee0e506ee521d3f90eb4281e1abf6748a00603b5c4d74a.jpg)  
(a)

(b)  
![](images/3700c2996f481ae62669e5554dbb2c06dd34d941de91da2aad8c4ee4eab2417a.jpg)  
(b)  
Figure 8. The 10 times average validation loss curves of (a) case 1 and (b) case 2 on diferent models.

All experiments are repeated 10 times with the same terms. Finally, we use the accuracy of the test sets to evaluate the fault diagnosis capabilities of the diferent models.

4.2.2. Results and Discussion. The case 1 and case 2 training sets after extended sliding window processing are input to diferent models for training. After five epochs, the average training loss curve of 10 times obtained by each model are presented in Figure 7. From Figure 7a,b, it can be seen that the models with LSTM structure or improved LSTM structure have a stronger convergence ability compared with the CS-CNN models. It shows that LSTM or improved LSTM can handle the time series data problem of TEP very well. From Figure 7, it can be observed that the proposed model has the most stable training loss value and the strongest convergence ability in both case 1 and case 2. Besides, from Figure 7 and Figure 8, the convergence ability of the proposed model is significantly better than the traditional CNN-LSTM in terms of training loss and validation loss. Therefore, the proposed model has the strongest convergence and generalization ability compared to other models such as CNN-LSTM. This is mainly because SE can give more weight to the key channel features from CNN, and IMLSTM can balance historical fault information and adaptively capture the dynamic features of fault data through the updated gating mechanism.

Table 4. Classification Accuracy of Each Fault in Each Model

<table><tr><td>case</td><td>fault</td><td>proposed model (%)</td><td>CS-LSTM (%)</td><td>CNN-IMLSTM (%)</td><td>CNN-LSTM (%)</td><td>LSTM (%)</td><td>CS-CNN (%)</td></tr><tr><td rowspan="5">Case 1</td><td>1</td><td>99.36</td><td>99.35</td><td>98.85</td><td>99.36</td><td>99.74</td><td>97.57</td></tr><tr><td>2</td><td>100.0</td><td>99.74</td><td>99.87</td><td>99.62</td><td>99.62</td><td>99.49</td></tr><tr><td>6</td><td>99.23</td><td>99.87</td><td>99.87</td><td>100.0</td><td>98.98</td><td>99.74</td></tr><tr><td>7</td><td>100.0</td><td>90.39</td><td>97.95</td><td>86.94</td><td>94.88</td><td>84.76</td></tr><tr><td>8</td><td>94.24</td><td>94.37</td><td>88.35</td><td>88.99</td><td>85.53</td><td>92.70</td></tr><tr><td rowspan="5">Case 2</td><td>4</td><td>100.0</td><td>99.74</td><td>100.0</td><td>99.87</td><td>95.26</td><td>97.95</td></tr><tr><td>5</td><td>99.23</td><td>97.70</td><td>98.98</td><td>98.72</td><td>91.93</td><td>97.18</td></tr><tr><td>10</td><td>99.49</td><td>97.95</td><td>96.41</td><td>92.70</td><td>73.24</td><td>83.99</td></tr><tr><td>11</td><td>93.85</td><td>90.01</td><td>91.17</td><td>90.78</td><td>47.25</td><td>89.88</td></tr><tr><td>12</td><td>98.21</td><td>96.67</td><td>96.93</td><td>97.18</td><td>70.04</td><td>97.06</td></tr></table>

![](images/99a254eeed548eb2c59dab6788b85229cb0a8a9d166029516603bc7941860c1e.jpg)  
(a)  
Figure 9. Classification results of (a) case 1 and (b) case 2 under each model.

The trained model is utilized to classify test sets and obtain classification accuracy. Table 4 shows the classification accuracy of each fault in the best results of each model. The best results are highlighted in bold in the table. From Table 4, the accuracy of the proposed method in case 1 and case 2 is more than 93.85%. The recognition accuracies of CS-LSTM, CNN-IMLSTM, CNN-LSTM, LSTM, and CS-CNN are more than 90.01%, 88.35%, 86.94%, 47.25%, and 83.99%, respectively. In the proposed model, fault 2, fault 4, and fault 7 can achieve 100% prediction accuracy. Compared with the other five models, fault 5, fault 10, fault 11, and fault 12 can get the best prediction accuracies, which are 99.23%, 99.49%, 93.85%, and 98.21%, respectively. The performance of LSTM in case 2 is not as good as that in case 1, which shows that the generalization performance of the LSTM model for chemical process fault diagnosis is poor. It is dificult for LSTM to mine the spatial information on industrial data without the assistance of CNN. Therefore, it is shown again that the fusion model CS-IMLSTM can pay attention to the important characteristics of industrial process fault data and adaptively process the dynamic information on data. From Table 4, the fault identification results of other models are not as stable as those of the proposed model, indicating that the proposed model can learn more advanced features from extended dynamic data and improve the level of risk perception.

![](images/9e8ecce13657e708bf7687fdf65adc54e63a33bd3cc6e2f859e6baa76b1bb221.jpg)  
(b)

The work is repeated 10 times with the same terms, and the max accuracy, min accuracy, average accuracy, and standard deviation (std) are calculated. The diagnostic results are presented in Figure 9. The bold black text represents the average accuracy, and the bold red text represents the std. CS-IMLSTM achieved the highest average accuracy in both case 1 and case 2 test data sets with 98.29% ± 0.0014 and 97.74% ± 0.0018, respectively. The results demonstrate that the proposed model has high prediction accuracy and an excellent general ization performance. Specifically, the minimum accuracy obtained by the proposed model in case 1 is 1.15% higher than the maximum accuracy of CNN-IMLSTM, while the minimum accuracy obtained by the proposed model in case 2 is 0.85% higher than the maximum accuracy of CNN-IMLSTM. This indicates that the SE attention mechanism can focus on important channel features and boost the fault diagnosis performance of the model. The minimum accuracy obtained by the proposed model in case 1 is 0.46% higher than that of CS LSTM, while the minimum accuracy obtained by the proposed model in case 2 is 1.13% higher than that of CS-LSTM. This indicates that IMLSTM can balance the fault information on industrial process data and capture the dynamic features of the temporal data more adequately than LSTM. The minimum accuracy obtained by the proposed model in case 1 is 3.15% higher than the maximum accuracy of CNN-LSTM, while the minimum accuracy obtained by the proposed model in case 2 is 1.69% higher than the maximum accuracy of CNN-LSTM. This indicates that the organic combination of the SE attention mechanism, IMLSTM, and CNN can more fully exploit the feature information on augmented dynamic industrial data, achieve eficient flow of information, and improve the security of the process.

In the confusion matrix, the row stands for predicted fault labels, the column stands for actual fault labels, and the diagonal line indicates that predicted results are consistent with the real labels. Figure 10 provides the confusion matrix of the worst

![](images/651fcb0a44214866137b98596205e6733fd1391ba3393419f29dbda73734bb96.jpg)  
Figure 10. Confusion matrix for worst case prediction in case 1.

result of the proposed method in case 1, and its prediction accuracy is 98.13%. From Figure 10, fault 1, fault 2, and fault 6 are correctly predicted with 773 samples and above, while 760 samples are correctly predicted and 21 samples are incorrectly predicted as fault 8 in fault 7. Only 741 samples are correctly predicted, and 40 samples are misclassified as fault 2 in fault 8. In addition, we analyze the positive predictive value (PPV), true positive rate (TPR), and F1\_Score<sup>37</sup> of this confusion matrix. It is worth noting that F1\_Score here returns the score for each fault category. MacroF1\_Score is the simple arithmetic mean of F1\_ Score. The results are presented in Table 5. The proposed

Table 5. Analytical Results of the Worst Confusion Matrix in Case 1

<table><tr><td>indicator</td><td>fault 1</td><td>fault 2</td><td>fault 6</td><td>fault 7</td><td>fault 8</td></tr><tr><td>PPV (%)</td><td>100</td><td>95.13</td><td>100</td><td>98.96</td><td>96.74</td></tr><tr><td>TPR (%)</td><td>99.49</td><td>100</td><td>98.98</td><td>97.31</td><td>94.88</td></tr><tr><td>F1_Score (%)</td><td>99.74</td><td>97.50</td><td>99.49</td><td>98.13</td><td>95.80</td></tr><tr><td>MacroF1_Score (%)</td><td></td><td></td><td>98.13</td><td></td><td></td></tr></table>

method has high PPV, TPR, and F1\_Score, and MacroF1\_Score is 98.13%, which indicates that CS-IMLSTM can adequately extract the dynamic features of the data, thus enhancing the efectiveness of the fault diagnosis, improving the safety risk status of process industrial processes, guaranteeing process safety production, and increasing the economic eficiency of enterprises.

Figure 11 shows the confusion matrix for the worst result of the proposed method in case 2, and its prediction accuracy is 97.54%. All samples of fault 10 were correctly predicted. A total of 778 and 777 samples were correctly predicted for faults 4 and

![](images/b925e83ec81363e8b665c30256c24d6ef906c05cbc8b1b5bc756588e631e6740.jpg)  
Figure 11. Confusion matrix for worst case prediction in case 2.

5, respectively. Fault 11 and fault 12 are inlet temperature fluctuations of the reactor and condenser cooling water, respectively, and are consistent with the fault descriptions of fault 4 and fault 5, respectively. Fault 11 and fault 12 are random variable types, and fault 4 and fault 5 are step variable types. Therefore, faults 11 and 12 are easily confused with faults 4 and 5, respectively. As can be seen from the figure, 45 samples of fault 11 are misclassified as fault 4, and 10 samples of fault 12 are misclassified as fault 5. Similarly, PPV, TPR, and F1\_Score of the confusion matrix are analyzed, respectively. The analysis results are shown in Table 6. The method has high indicated PPV, TPR, and F1\_Score, and MacroF1\_Score is 97.52%, which indicates the efectiveness of using CS-IMLSTM for fault diagnosis.

Table 6. Analytical Results of the Worst Confusion Matrix in Case 2

<table><tr><td>indicator</td><td>fault 4</td><td>fault 5</td><td>fault 10</td><td>fault 11</td><td>fault 12</td></tr><tr><td>PPV (%)</td><td>94.53</td><td>98.10</td><td>95.71</td><td>100</td><td>99.87</td></tr><tr><td>TPR (%)</td><td>99.62</td><td>99.49</td><td>100</td><td>90.78</td><td>97.82</td></tr><tr><td>F1_Score (%)</td><td>97.01</td><td>98.79</td><td>97.81</td><td>95.17</td><td>98.84</td></tr><tr><td>MacroF1_Score (%)</td><td></td><td></td><td>98.13</td><td></td><td></td></tr></table>

4.2.3. Comparison with Existing Advanced Methods. To further verify the superiority of the proposed method in extracting dynamic features of the chemical process industry, this paper compares it with dynamic PCA-SVM (DPCA-SVM) and transformer neural network. In DPCA-SVM, DPCA is a classical method for extracting dynamic features of data, and SVM is used for fault identification. The dynamic order h of DPCA is 2, and the contribution rate of the principal component is 0.99. The kernel function of SVM is RBF. Transformer is a neural network based on a pure attention mechanism to reflect the global dependence between input and output and has good identification performance in chemical process fault diagnosis.38 Transformer’s network architecture and hyperparametric references<sup>39</sup> take the enhanced dynamic data as the input, and the size of the input subsequence is 20. Similarly, all experiments are repeated 10 times under the same conditions, and the average accuracy is taken as the experimental result.

The results are shown in Table 7. The proposed method achieves the best performance in both case 1 and case 2. Compared with DPCA-SVM, the average accuracies of the proposed method in case 1 and case 2 are improved by 8.56% and 36.91%, respectively. Compared with the results of Transformer, the average accuracies of the proposed method in case 1 and case 2 are improved by 1.06% and 19.23%, respectively. Experimental results show that, compared with these advanced fault diagnosis methods, the proposed method can extract the dynamic features of process data better and has the highest fault diagnosis results and the best generalization performance.

Table 7. Compare with Existing Advanced Methods

<table><tr><td>method</td><td>case 1</td><td>case 2</td><td>average accuracy</td></tr><tr><td>DPCA-SVM</td><td>89.73%</td><td>60.83%</td><td>75.28%</td></tr><tr><td>transformer</td><td>97.23%</td><td>78.51%</td><td>87.87%</td></tr><tr><td>proposed method</td><td>98.29%</td><td>97.74%</td><td>98.02%</td></tr></table>

From all the above results and analysis, it can be concluded that the proposed method has the most desirable fault diagnosis performance compared to all comparison experiments. This is mainly because, before classification, the extended sliding window is used to generate expanded dynamic data. The CS-IMLSTM model is used to learn spatial, channel, and temporal information on industrial process data and deeply excavates the dynamic information on the data. Therefore, the classification performance is improved.

## 5. CONCLUSION

In this paper, the CS-IMLSTM model is designed for chemical industrial process fault diagnosis, which solves the problem that the traditional CNN-LSTM model and other forms of this model have dificulty extracting dynamic fault features of chemical processes. The contributions are specified as follows:

(1) In terms of data preprocessing, an extended sliding window mechanism is proposed. The mechanism provides raw data with strong dynamic information for the proposed model and lays a foundation for the highest accuracy of the proposed model on the TEP data set.

(2) In terms of feature extraction, the CS-IMLSTM model is proposed. We introduce he SE attention mechanism into the CNN, which can adaptively assign more weight to key fault features to optimize fault features. In addition, we propose an IMLSTM, which alleviates the excessive dependence of LSTM on the current or previous fault information, so that LSTM can pay more attention to the features of industrial data in the time dimension, balance the fault information, and adaptively extract the dynamic information of the data. Finally, CS-IMLSTM is constructed by integrating the CS network and IMLSTM, which can extract the spatial and temporal dynamic characteristics of process industry data simultaneously.

(3) The efectiveness of the proposed method is verified in TEP. Compared with five comparison experiments including CNN-LSTM, CS-IMLSTM obtain the highest average accuracies of 98.29% ± 0.0014 and 97.74% ± 0.0018 in both subdata of the TEP. The simulation results verify the feasibility of the proposed method

The proposed method can better capture the dynamic fault information of a chemical process and enhance the performance of fault diagnosis. Therefore, CS-IMLSTM can provide RA and ASM with a more favorable decision-making basis based on the dynamic fault information of chemical processes and deploy remedial actions and implement safety measures in time to minimize process risks and avoid safety accidents.

The extended sliding window mechanism and deep learning network need to occupy a large amount of computer memory resources. Therefore, in future research, from the perspective of data preprocessing, it is an efective approach to improve the data quality by variable screening of multivariable industrial process data. In terms of network architecture design, network quantization, network decomposition, and lightweight network design are worthy of future research.

## ■ AUTHOR INFORMATION

## Corresponding Author

Jian Cen − School of Automation, Guangdong Polytechnic Normal University, Guangzhou 510665, China; Guangzhou Intelligent Building Equipment Information Integration and Control Key Laboratory, Guangzhou 510665, China; orcid.org/0000-0002-1714-7397; Email: mmcjian@ 163.com

## Authors

Honghua Chen − School of Automation, Guangdong Polytechnic Normal University, Guangzhou 510665, China; Guangzhou Intelligent Building Equipment Information Integration and Control Key Laboratory, Guangzhou 510665, China

Zhuohong Yang − School of Automation, Guangdong Polytechnic Normal University, Guangzhou 510665, China; Guangzhou Intelligent Building Equipment Information Integration and Control Key Laboratory, Guangzhou 510665, China

Weiwei Si − School of Automation, Guangdong Polytechnic Normal University, Guangzhou 510665, China; Guangzhou Intelligent Building Equipment Information Integration and Control Key Laboratory, Guangzhou 510665, China

Hongchao Cheng − School of Automation, Guangdong Polytechnic Normal University, Guangzhou 510665, China; Guangzhou Intelligent Building Equipment Information Integration and Control Key Laboratory, Guangzhou 510665, China

Complete contact information is available at:

https://pubs.acs.org/10.1021/acsomega.2c04017

## Notes

The authors declare no competing financial interest.

## ■ ACKNOWLEDGMENTS

This work was supported by the Innovative Team Project of the Ordinary University of Guangdong Province [Grant Number 2020KCXTD017]; the Guangdong Special Project in Key Field of Artificial Intelligence for the Ordinary University [Grant Number 2019KZDZX1004]; the Guangzhou Key Laboratory Construction Project [Grant Number 202002010003]; and the Guangzhou Key Research and Development Project [Grant Number 202206010022].

## ■ REFERENCES

(1) Shu, Y.; Ming, L.; Cheng, F.; Zhang, Z.; Zhao, J. Abnormal situation management: challenges and opportunities in the big data era. Comput. Chem. Eng. 2016, 91, 104−113.

(2) Arunthavanathan, R.; Khan, F.; Ahmed, S.; Imtiaz, S. An analysis of process fault diagnosis methods from safety perspectives. Comput. Chem. Eng. 2021, 145, 107197.

(3) Khan, F.; Rathnayaka, S.; Ahmed, S. Methods and models in process safety and risk management: past, present and future. Process Saf. Environ. Prot. 2015, 98, 116−147.

(4) Dai, Y.; Wang, H.; Khan, F.; Zhao, J. Abnormal situation management for smart chemical process operation. Curr. Opin Chem. Eng. 2016, 14, 49−55.

(5) Benson, C.; Argyropoulos, C. D.; Dimopoulos, C.; Mikellidou, C. V.; Boustras, G. Safety and risk analysis in digitalized process operations warning of possible deviating conditions in the process environment. Process Saf. Environ. Prot. 2021, 149, 750−757.

(6) Amin, M. T.; Khan, F.; Ahmed, S.; Imtiaz, S. Risk-based fault detection and diagnosis for nonlinear and non-Gaussian process systems using R-vine copula. Process Saf. Environ. Prot. 2021, 150, 123− 136.

(7) Bao, H.; Khan, F.; Iqbal, T.; Chang, Y. Risk-based fault diagnosis and safety management for process systems. Process Saf Prog. 2011, 30 (1), 6−17.

(8) Bhadriraju, B.; Kwon, J. S.-I.; Khan, F. Risk-based fault prediction of chemical processes using operable adaptive sparse identification of systems (OASIS). Comput. Chem. Eng. 2021, 152, 107378.

(9) Md Nor, N.; Che Hassan, C. R.; Hussain, M. A. A review of datadriven fault detection and diagnosis methods: applications in chemical process systems. Rev. Chem. Eng. 2020, 36 (4), 513−553.

(10) Deng, X.; Cai, P.; Cao, Y.; Wang, P. Two-step localized kernel principal component analysis based incipient fault diagnosis for nonlinear industrial processes. Ind. Eng. Chem. Res. 2020, 59 (13), 5956−5968.

(11) Han, Y.; Song, G.; Liu, F.; Geng, Z.; Ma, B.; Xu, W. Fault monitoring using novel adaptive kernel principal component analysis integrating grey relational analysis. Process Saf. Environ. Prot. 2022, 157, 397−410.

(12) He, Y.-L.; Li, K.; Liang, L.-L.; Xu, Y.; Zhu, Q.-X. Novel discriminant locality preserving projection integrated with monte carlo sampling for fault diagnosis. IEEE T Reliab. 2021, 1−11.

(13) Zhang, N.; Xu, Y.; Zhu, Q.-X.; He, Y.-L. Improved localitypreserving projections based on heat-kernel and cosine weights for fault classification in complex industrial processes. IEEE T Reliab 2022, 1.

(14) Zhang, N.; Xu, Y.; Zhu, Q.-X.; He, Y.-L. Farthest-nearest distance neighborhood and locality projections integrated with bootstrap for industrial process fault diagnosis. IEEE T Ind. Inform. 2022, DOI: 10.1109/TII.2022.3182774.

(15) Amin, M. T.; Khan, F.; Ahmed, S.; Imtiaz, S. A data-driven Bayesian network learning method for process fault diagnosis. Process Saf. Environ. Prot. 2021, 150, 110−122.

(16) Deng, Z.; Han, T.; Cheng, Z.; Jiang, J.; Duan, F. Fault detection of petrochemical process based on space-time compressed matrix and Naive Bayes. Process Saf. Environ. Prot. 2022, 160, 327−340.

(17) Wu, H.; Zhao, J. Deep convolutional neural network model based chemical process fault diagnosis. Comput. Chem. Eng. 2018, 115, 185− 197.

(18) Song, Q.; Jiang, P. A multi-scale convolutional neural network based fault diagnosis model for complex chemical processes. Process Saf. Environ. Prot. 2022, 159, 575−584.

(19) Yu, J.; Zhang, C.; Wang, S. Multichannel one-dimensional convolutional neural network-based feature learning for fault diagnosis of industrial processes. Neural Comput. Appl. 2021, 33 (8), 3085−3104.

(20) Yu, W.; Zhao, C. Broad Convolutional Neural Network based Industrial Process Fault Diagnosis with Incremental Learning Capability. IEEE Trans Ind. Electron. 2020, 67 (6), 5081−5091.

(21) Zhao, H.; Sun, S.; Jin, B. Sequential Fault Diagnosis based on LSTM Neural Network. IEEE Access. 2018, 6, 12929−12939.

(22) Han, Y.; Ding, N.; Geng, Z.; Wang, Z.; Chu, C. An optimized long short-term memory network based fault diagnosis model for chemical processes. J. Process Control. 2020, 92, 161−168.

(23) Park, P.; Marco, P. D.; Shin, H.; Bang, J. Fault detection and diagnosis using combined autoencoder and long short-term memory network. Sensors. 2019, 19 (21), 4612.

(24) Gravanis, G.; Dragogias, I.; Papakiriakos, K.; Ziogou, C.; Diamantaras, K. Fault detection and diagnosis for non-linear processes empowered by dynamic neural networks. Comput. Chem. Eng. 2022, 156, 107531.

(25) Shao, B.; Hu, X.; Bian, G.; Zhao, Y. A multichannel LSTM-CNN method for fault diagnosis of chemical process. Math Probl Eng. 2019, 2019 (3), 1032480.

(26) Wang, N.; Yang, F.; Zhang, R.; Gao, F. Intelligent fault diagnosis for chemical processes using deep learning multimodel fusion. IEEE Trans Cybern. 2020, 7121−7135.

(27) Yuan, J.; Tian, Y. A multiscale feature learning scheme based on deep learning for industrial process monitoring and fault diagnosis. IEEE Access. 2019, 7, 151189−151202.

(28) Huang, T.; Zhang, Q.; Tang, X.; Zhao, S.; Lu, X. A novel fault diagnosis method based on CNN and LSTM and its application in fault diagnosis for complex systems. Artif Intell Rev. 2022, 55 (2), 1289− 1315.

(29) Chen, Z.; Cen, J.; Xiong, J. Rolling bearing fault diagnosis using time-frequency analysis and deep transfer convolutional neural network. IEEE Access. 2020, 8, 150248−150261.

(30) Albawi, S.; Mohammed, T. A.; Al-Zawi, S. Understanding of a convolutional neural network. Proceedings of the 2017 International Conference on Engineering and Technology (ICET); IEEE: 2017; pp 1−6.

(31) Hu, J.; Shen, L.; Albanie, S.; Sun, G.; Wu, E. Squeeze-andexcitation networks. IEEE Trans Pattern Anal Mach Intell. 2020, 42 (8), 2011−2023.

(32) Hochreiter, S.; Schmidhuber, J. Long short-term memory. Neural Comput. 1997, 9 (8), 1735−1780.

(33) Han, Y.; Qi, W.; Ding, N.; Geng, Z. Short-time wavelet entropy integrating lmproved LSTM for fault diagnosis of modular multilevel converter. IEEE Trans Cybern. 2021, 7504−7512.

(34) Downs, J. J.; Vogel, E. F. A plant-wide industrial process control problem. Comput. Chem. Eng. 1993, 17 (3), 245−255.

(35) Arunthavanathan, R.; Khan, F.; Ahmed, S.; Imtiaz, S.; Rusli, R. Fault detection and diagnosis in process system using artificial intelligence-based cognitive technique. Comput. Chem. Eng. 2020, 134, 106697.

(36) Ma, L.; Dong, J.; Peng, K. A novel key performance indicator oriented hierarchical monitoring and propagation path identification framework for complex industrial processes. ISA Trans. 2020, 96, 1−13.

(37) Azamfar, M.; Li, X.; Lee, J. Deep Learning-based domain adaptation method for fault diagnosis in semiconductor manufacturing. IEEE Trans Semicond Manuf. 2020, 33 (3), 445−453.

(38) Zhang, L.; Song, Z.; Zhang, Q.; Peng, Z. Generalized transformer in fault diagnosis of Tennessee Eastman process. Neural Comput. appl. 2022, 34 (11), 8575−8585.

(39) Yang, Z.; Cen, J.; Liu, X.; Xiong, J.; Chen, H. Research on bearing fault diagnosis method based on transformer neural network. Meas. Sci. Technol. 2022, 33 (8), 085111.