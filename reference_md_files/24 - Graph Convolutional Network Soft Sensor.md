# Graph convolutional network soft sensor for process quality prediction

Mingwei Jia <sup>a</sup>, Danya Xu <sup>b</sup>, Tao Yang <sup>b</sup>, Yi Liu <sup>a,∗</sup>, Yuan Yao <sup>c,∗</sup>

![](images/83dcd5ee77a09b3459e85460049813828f7190cd9b05e2c595bc1c08541463a6.jpg)

<sup>a</sup> Institute of Process Equipment and Control Engineering, Zhejiang University of Technology, Hangzhou, 310023, People’s Republic of China

<sup>b</sup> State Key Laboratory of Synthetical Automation for Process Industries, Northeastern University, Shenyang 110819, People’s Republic of China

<sup>c</sup> Department of Chemical Engineering, National Tsing Hua University, Hsinchu 30013, Taiwan

## a r t i c l e i n f o

Article history: Received 28 July 2022 Received in revised form 14 December 2022 Accepted 17 January 2023 Available online 24 January 2023

Keywords: Soft sensor Graph convolutional network Quality prediction Fermentation process

## a b s t r a c t

The nonlinear time-varying characteristics of the process industry can be modeled using numerous data-driven soft sensor methods. However, the intrinsic relationships among the variables, especially the localized spatial-temporal correlations that shed light on model behavior. have received little attention. In this study, a soft sensor based on a graph convolutional network is constructed by introducing the concept of graph to process modeling. The focus is on obtaining localized spatial– temporal correlations that aid in comprehending the intricate interactions among the variables included in the soft sensor. The model is trained by considering the regularization terms and it learns distinctive localized spatial–temporal correlations in an end-to-end manner. Furthermore, longterm dependence is established via temporal convolution. Thus, both the localized spatial–temporal correlations and time-series properties are captured. The feasibility of the proposed soft sensor is illustrated using two fermentation processes. The localized spatial–temporal correlations of this case study are visualized, and they demonstrate that the soft sensor is not a black-box model; instead, it is consistent with process knowledge.

© 2023 Elsevier Ltd. All rights reserved.

## 1. Introduction

A highly accurate measurement of product quality is essential for monitoring, control, and optimization of modern industrial processes [1]. However, crucial quality indices/variables are not always measurable in real time, leading to a significantly delayed quality feedback. In such situations, soft sensors often play an important role in estimating the difficult-to-measure quality variables using easily accessible process knowledge or data [2]. First-principle models are preferred by some process engineers owing to their transparency. However, due to their complex process mechanisms, it is difficult to build these models quickly. [3]. On the other hand, data-driven models are gaining popularity, especially for complex processes where the available prior knowledge is either unavailable or insufficient [4–6].

In the past decade, data-driven models, including support vector regression (SVR) [7,8], various shallow neural network (NN) [9], partial least square (PLS) [10], and Bayesian network model (BNM) [11], have grown in popularity. Patanè and Xibilia adopted a recurrent NN to estimate key process variables in the sulfur recovery unit [12]. Desai et al. demonstrated that, in some cases, the SVR soft sensor is superior in comparison to the shallow NNs in terms of performance [13]. PLS-based soft sensors are popular in chemical engineering and chemometrics owing to their decent performance and simple design [14]. To improve the local prediction performance, these methods can be combined with the just-in-time learning techniques [15,16]. BNM can mine the causal relationship among variables and express it in the form of probability. Mohammadi et al. developed a soft sensor under the BNM framework for predicting sulfur content in a gas sweetening process [17]. However, in BNMs, parameter learning is a challenging task. In particular, both exact and approximate algorithms for calculating posteriors may lead to NP-hard problems, which are fatal for large models [11]. Additionally, with the increasing complexity of process industry, the structural limitations of these models make them insufficient for fully extracting representative features from historical data [18].

Recently, deep learning (DL) has received increasing attention in various fields including pattern recognition and nondestructive testing [19–21], and fault diagnosis [22]. DL exhibits powerful ability in historical data mining and helps describe complex processes [23,24]. Hence, DL has also been utilized for soft sensor development [25–28]. Xie et al. used long short-term memory (LSTM) to predict key variables and achieved satisfactory performance [29]. Huang et al. stacked autoencoder to identify significant variables and develop a soft sensor [30]. Chang et al. used contrastive learning to build a temporality-aware soft sensor that was robust to anomalies [31]. However, most current deep learning methods can be considered as ‘‘black boxes’’, which do not integrate prior knowledge and their behavior cannot be convincingly explained. Exploring variable relationships explicitly is beneficial for explaining model behavior and improving its interpretability.

Graphs can be extracted from various real-world relations among numerous entities and be defined using the vertex and edge sets, which represent the entities and their relationships, respectively [32]. Because graphs exploit essential and relevant relations among vertices, graph neural networks (GNNs) [33], which allow for the explicit study of variable relationships, have gained increasing popularity for capturing complex relationships. As a variant of traditional GNN, the graph convolutional network (GCN) [34] has exhibited powerful learning capability in several fields [35,36]. In the perspective of helping DL implement an explicit study of variable relationships, GCN have become noticeable as a representative model in the field of soft sensors.

Recent studies have introduced GCN into the field of soft sensors. Fang et al. used GCN to predict the quality of elements in the steelmaking process [37]. While the integration of relationships among elements in the form of prior knowledge improves model interpretability, the capturing of time-series properties alone is insufficient. Wang et al. established a soft sensor based on GCN and the gated recurrent unit for establishing relationships among variables and capturing their time-series properties during anaerobic digestion of kitchen waste [38]. As pointed by Song et al. it is indispensable to consider the localized spatial–temporal correlations and the cross-correlations of variables at different times [39]. Although the gated recurrent unit can capture cross-correlations to a certain extent, its capture process is unconstrained and not unique, which reduces model interpretability. Meanwhile, cross-correlation, as an adjunct of the gated recurrent unit has been rarely studied. The study of localized spatial–temporal correlations is meaningful for understanding the complex variable relationships in soft sensors.

In this study, we developed a novel GCN-based soft sensor utilizing localized spatial–temporal correlations. To reflect the relationship among variables at the same time and at different times, the proposed method uses spatial and temporal edges, respectively. The main contributions of this work are summarized as follows.

(a) We proposed a GCN-based soft sensor that aims to capture unique localized spatial–temporal correlations among variables by implementing the concept of graphs in the process industry.

(b) The proposed model autonomously learns the unique localized spatial–temporal graph that captures correlations. In addition, we used a regularization loss to constrain its learning process. Subsequently, the temporal convolution is developed to capture the time-series properties. Consequently, the GCN-based soft sensor facilitates a prediction that is consistent with the laws of physics and ensures model transparency.

(c) The performance of the GCN-based soft sensor is evaluated on two fermentation processes and is compared with that of several popular methods to prove its superiority. Contrary to the case of using a black-box model, the visualization of the localized spatial–temporal correlations using the proposed model exhibits its consistency with the prior knowledge.

## 2. Preliminaries

Definition 1. $G = ( { \bf V } , { \bf A } , { \bf E } )$ denotes the graph data, where V is the set of vertices (or nodes), E denotes the set of edges, and $\pmb { \mathrm { A } } \in \{ 0 , 1 \} ^ { V \times V }$ is the adjacency matrix of the graph G. G denotes the relationship of nodes in the spatial dimension, and the network structure does not change with time. In this work, the structure of G is that of a directed graph.

Definition 2. Graph node signal matrix of each sample is defined as $\mathbf { X } _ { G } ~ \in ~ \mathbb { R } ^ { C \times V \times \mathbf { \hat { T } } }$ , where C denotes the number of channels (i.e., the number of convolution kernels), V denotes the number of nodes, and T denotes the time step of each observed node. Additionally, each sample shares the same adjacency matrix A.

## 2.1. Graph convolutional network

There are two key steps to set up a GCN. The first step is to perform the local information fusion on graph-structure data, and the second is to achieve graph representation learning, which embeds nodes or edges into vectors using a deep model. To perform these steps, hidden layer vectors are added in GCN through information transfer between adjacent nodes, and tasks such as classification and regression are completed through parameter learning. Fig. 1 shows the information transfer process of a twolayer GCN. For example, Node A in each hidden layer is derived from the information transfer of its neighboring nodes C, D, and itself (A). Then, the rectified linear unit (ReLU) activation function is exerted on Node A. Node A in Hidden Layer 2 is derived from hidden layer 1, and contains C, D, and itself.

GCN generalizes the traditional convolutional network from the Euclidean space to the vertex domain. This graph convolutional operation is based on Fourier transform and Laplacian matrix, and it can be formulated as follow:

$$
\operatorname{GCN} \left(\mathbf {X} _ {G}, \mathbf {A}\right) = \sigma \left(\mathbf {D} ^ {- \frac {1}{2}} \mathbf {A D} ^ {- \frac {1}{2}} \mathbf {X} _ {G} \mathbf {Q}\right)\tag{1}
$$

where, D<sup>−1/2</sup>AD<sup>−1/2</sup> denotes the convolutional kernel; D is the degree matrix of the adjacency matrix $\mathbf { A } ; \mathbf { X } _ { G }$ is the input of the graph convolutional layer; Q is a weight matrix; σ denotes the activation function, such as ReLU.

## 2.2. Temporal convolutional network

Temporal convolutional networks (TCNs) [40] are based on two principles: (a) the network produces an output of the same length as the input; and (b) any leakage from the future into the past is impossible. First, the TCN is implemented as a regular onedimensional convolution with a kernel of size (1×a), where, a denotes the size of convolution kernel and each hidden layer has the same length as the input layer. Second, the TCN uses causal convolutions where output at the current time is convolved only with elements from the current time and earlier time. Fig. 2 shows the TCN framework with two hidden layers. Through the extraction of two hidden layers, the data at time t of the output layer captures the characteristics of the three previous moments and itself (t). TCN emphasizes building long and effective history sizes using a combination of deep networks and improving the ability of the networks to look as far as possible into the past to make a prediction.

## 3. Proposed soft sensor method

In this section, a GCN-based soft sensor method is proposed as follows. (1) The maximum information coefficients (MICs) [38] of each variable aimed at the target variable are calculated in the training set. (2) The validation set is used to determine the number of variables in the dataset. (3) The spatial–temporal convolutional layers (STCLs) are used to learn localized spatial– temporal correlations and to model long-term dependencies. (4) Multi-layer STCLs are stacked to form an encoding structure. (5) A fully connected layer (FCL) is added as a decoder to establish the mapping relationship between the data encoded by the stacked multi-layer STCLs and the target. It should be noted that the adjacency matrix of all layers is shared to ensure that the graph structures learned by the model are unique. The following sections detail the involved algorithms and model structures.

![](images/4bdd7c3040fd896935153c89b4c8c646501b9dff6bcec0bed53f6523d2ca3519.jpg)  
Fig. 1. A GCN structure with multiple graph convolutional layers.

![](images/a1b70887b7d7668e0e55537c4b5adb041bbd3aa1e9c03fe5951c225c7e5875ff.jpg)  
Fig. 2. A TCN framework with two hidden layers.

## 3.1. Variable selection based on MIC

An MIC [41] is derived from the mutual information (MI) that indicates the reduction in the uncertainty of a random variable caused by the introduction of another random variable. Here, only the MI between two discrete variables is considered. Given two discrete variables j and u, the joint probability distribution p(j, u) and MI can be obtained:

$$
I (\mathbf {j}, \mathbf {u}) = \sum_ {j \in \mathbf {j}} \sum_ {u \in \mathbf {u}} p (j, u) \log_ {2} \frac {p (j , u)}{p (j) p (u)}.\tag{2}
$$

In addition, the MIC between j and u is the maximal normalized MI among all states:

$$
\operatorname{MIC} _ {j, u} = \max _ {| j | \cdot | u | <   B} \frac {\mathrm{I} (\mathbf {j} , \mathbf {u})}{\log_ {2} (\min (| j | , | u |))},\tag{3}
$$

where, $B \ = \ m ^ { 0 . 5 5 }$ , and m denotes the size of the dataset. MIC gives each variable a full illustration of its importance relative to the key variable. After calculating the MIC between the variables and the target, an appropriate variable is selected as the model input.

## 3.2. Spatial–temporal convolutional layer

The STCL is divided into a localized spatial–temporal correlation module (LSCM) and a temporal convolution module (TCM). LSCM can directly capture the impact of each node on its neighbors that belong to both the current and the adjacent time steps. The most intuitive idea to achieve this goal is to connect all nodes with themselves at the adjacent time steps. As shown in Fig. 3(a), a localized spatial–temporal graph can be obtained by adding arrows between variables in two moments. For example, its effect of variable x on the remaining variables at the next moment follows a localized spatial–temporal correlation. According to the topological structure of the localized spatial–temporal graph, the correlations between each node and its spatial–temporal neighbors can be captured directly.

The relationship matrix $( \pmb { \cal A } \in \mathbb { R } ^ { V \times \bar { V } } )$ denotes the variable relationship matrix of the spatial graph, and the localized matrix $\mathbf { \Psi } ( \mathbf { A } ^ { \prime } \in \mathbf { \Gamma } \mathbb { R } ^ { 2 \dot { V } \times 2 V } )$ denotes the adjacency matrix of the localized spatial–temporal graph constructed on two continuous spatial graphs. Both these matrices are obtained through model training. As shown in Fig. 3(b), the A’ is composed of three sub-matrices: A, zero matrix O, and time correlation matrix $\pmb { \mathrm { A } } _ { \mathrm { t } } , \pmb { \mathrm { A } }$ implies that variables at different time steps share the same spatial structure, O implies that the previous time step is unaffected by the next time step, and $\mathbf { A } _ { \mathrm { t } }$ embodies the influence of the previous time step on the next time step. STCL built on a localized matrix can simultaneously capture the spatial and temporal relationships of variables.

(b)  
![](images/e3b366eb9f6937c8302bdd37aef92b5abc5157328d951d7cf5d701e41bbf98ac.jpg)  
(a)

Fig. 3. (a) An example of a localized spatial–temporal graph, and (b) its adjacency matrix.  
![](images/1abdc13aea9d495b52d4cac78c91ed388634011b5ae255ac2c5049c81e94165d.jpg)  
Fig. 4. For localized spatial–temporal graph: (a) the localized graph signal matrix, and (b) its cropping operation.

The graph signal matrix $\mathbf { X } _ { G } \in \mathbb { R } ^ { C \times V \times T }$ needs to be processed to correspond to $\pmb { \mathrm { A } } ^ { \prime } \in \mathbb { R } ^ { 2 V \times 2 V }$ . As shown in $\mathrm { F i g . } ~ 4 ( \mathsf { a } ) ,$ , the graph signal matrix for each channel is $\{ \mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } , . . . , \mathbf { X } _ { T } \} , \mathbf { X } _ { i } \stackrel { . . } { \in } \mathbb { R } ^ { V \times \mathbf { \breve { 1 } } }$ , and it is transformed into $\{ \{ { \bf X } _ { 1 } , ~ { \bf X } _ { 2 } , . . . , ~ { \bf X } _ { T - 1 } \} , ~ \{ { \bf X } _ { 2 } , ~ { \bf X } _ { 3 } , . . . , ~ { \bf X } _ { T } \} \} , ~ { \bf X } _ { i }$ ∈ $\mathbb { R } ^ { V \times 1 }$ . Finally, the localized graph signal matrix $\bar { \mathbf { X } _ { G } } \in \mathbb { R } ^ { C \times \bar { 2 } \bar { V } \times ( \bar { T } - 1 ) }$ is obtained to correspond to A’. The formula of LSCM can be expressed based on GCN as follows:

$$
\operatorname{LSCM} \left(\mathbf {X} _ {G} ^ {\prime}, \mathbf {A} ^ {\prime}\right) = \sigma \left(\mathbf {D} ^ {- \frac {1}{2}} \mathbf {A} ^ {\prime} \mathbf {D} ^ {- \frac {1}{2}} \mathbf {X} _ {G} ^ {\prime} \mathbf {Q}\right),\tag{4}
$$

where $\mathbf { Q } \in \mathbb { R } ^ { ( T \cdot }$ <sup>T−1)×T</sup> is a learnable matrix set for ensuring that the dimension remains unchanged after LSCM Subsequently, the cropping operation (Fig. 4(b)) removes all the features of the nodes in the previous time steps, and only the nodes in the next moment are retained. This is because LSCM has already aggregated the previous information. Each node contains the localized spatial–temporal correlations even after the previous time step is cropped.

The root mean square error (RMSE) is selected as the model loss. Because A’ is obtained through independent training of the model, the loss is extended with a regularization term:

$$
\begin{array}{c} \text {Loss} = \sqrt {\frac {1}{T _ {m}} \left(\mathbf {Y} - \mathbf {Y} ^ {\prime}\right) ^ {2}} + \beta \text {H} \left(\mathbf {A} ^ {\prime}\right) \\ \text {H} \left(\mathbf {A} ^ {\prime}\right) = - \mathbf {A} ^ {\prime} \log 2 \left(\mathbf {A} ^ {\prime}\right) - \left(1 - \mathbf {A} ^ {\prime}\right) \log 2 \left(1 - \mathbf {A} ^ {\prime}\right), \end{array}\tag{5}
$$

where, Y and Y’ denote the label and prediction, respectively. $T _ { m }$ is the number of time steps in a sample, β denotes a regularization coefficient, and H(.) indicates entropy. During training, the model may assign too much attention to irrelevant edges, resulting in the learning of wrong variable relationships. Therefore, a regularization term used to encourage discretization of the graph structure is added to the loss function, which improves the explainability of the learned graph structure.

TCM then encodes data that establish localized spatial– temporal correlations to capture temporal dynamic dependencies. An STCL is developed by combining LSCM and TCM. As presented in Fig. 5, all modules are followed by a batch normalization (BN) layer and an activation function.

## 3.3. Framework of GCN-based soft sensor

This section establishes the connection between the input variables and the target variable. Fig. 6 shows the GCN-based soft sensor framework and proposes a GCN-based modeling strategy. The proposed algorithm is summarized in Algorithm 1. After the

![](images/781c5587538440d62f3ad08aac5f227bf64af7b0b4f644586b3db4ff74f12626.jpg)  
Fig. 5. The STCL framework.

Algorithm 1 The proposed algorithm.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: The GCN-based soft sensor.

Input: The input variable  $X_{G} \in R^{1 \times V \times T}$ , label Y.

Output: The target variable Y'.

Hyperparameters: Channel C, epochs, learning rate.

1 Initialize parameters the relationship matrix  $A \in R^{V \times V}$ , the time correlation matrix  $A_{t} \in R^{V \times V}$ , the zero matrix  $O \in R^{V \times V}$ , and the model parameters (Q,  $\theta_{TCM}$ ,  $\theta_{FCL}$ ).

2  $A' \in R^{2V \times 2V} \leftarrow \text{concat}(A, A_t, O, A)$ 

3  $X'_{G} \in R^{1 \times 2V \times (T-1)} \leftarrow \text{transform}(X_G)$ 

4 for i = 1, 2, ..., epoch do

5    $A'[:, [0, V]]$  and  $A'[V:2V-1, V:2V-1] \leftarrow 0$ 

6    $D \in R^{2V \times 2V} \leftarrow \text{diagonal(sum}(A', axis=0))$ 

7    $X'_{G} \in R^{I \times V \times T} \leftarrow \sigma(D^{-1/2} A'D^{-1/2} X'_{G} Q)$ 

8    $X'_{G} \in R^{C \times V \times T} \leftarrow TCM_{with channel c}(X'_{G} | \theta_{TCM})$ 

9    $Y' \leftarrow FCL(X'_{G} | \theta_{FCL})$ 

10    $(Q, \theta_{TCM}, \theta_{FCL}) \leftarrow (Q, \theta_{TCM}, \theta_{FCL}) - learning rate \times \nabla Loss(Y, Y', A')$ 

11 end

12 Return Y'
</div>

local matrices are initialized, The STCL captures localized spatial– temporal correlations and time dynamic dependence. Because TCM involves convolution operations, multiple layers of STCLs are stacked to enlarge the receptive field of the model. In addition, the graph structure is shared by the multiple STCLs to ensure the uniqueness of the structure. The topological structure of the graph is captured by the model when the variable relationship is regarded as a graph. As a result of stacking multiple STCLs, each node contains the localized spatial–temporal correlations centered by itself.

## 4. Results and discussion

The production of secondary metabolites has been the subject of many studies due to its academic and industrial importance. Nevertheless, a primary obstacle to the implementation of control strategies is the lack of reliable sensors to measure the key variables, for example, biomass and production concentrations [1]. In this study, to verify the performance of the proposed method, we used a benchmark Pensim [42] and an industrial-scale Pensim (IndPensim) [43], both of which are simulation platforms for the penicillin fed-batch fermentation process. The code of this work is presented at https://github.com/MingweiJia/GCN-based\_soft\_ sensor, including all code and Pensim/IndPensim datasets used.

## 4.1. Pensim

Typically, the penicillin fermentation process in Pensim has two operational phases, that is, batch and fed-batch. In general, the system switches to the fed-batch mode after approximately 44 h. Then, during the fed-batch operation, a constant feed is used. We observed that the bacteria did not secrete penicillin in the first stage. Therefore, we only established the soft sensor for the second stage. The following 15 variables can be obtained from the penicillin fermentation process: aeration rate, agitator power, substrate feed flowrate, temperature, substrate concentration, dissolved oxygen concentration, biomass concentration, bacterial concentration, carbon dioxide concentration, pH value, generated temperature, and the flowrates of acid, base, cold water, and hot water. Among these, because the hot water flowrate is a constant value, and the bacterial concentration is an unmeasurable variable, these variables were omitted. Penicillin concentration (P) was considered as the target variable in this experiment.

![](images/7eceea1749f5a26f438e70d6c9732f7802a60371c921e3c1d293e44e3a98490b.jpg)  
Fig. 6. The framework of GCN-based soft sensor model

The software Pensim [42] can be obtained from: http://www. chee.iit.edu/\~control/software.html. It is used to generate an entire dataset containing 10 batches. The fermentation time of each batch was set to 400 h. In the actual fermentation process, to reduce the damage to the broth components, it is forbidden to frequently extract the broth to detect the product concentration. However, a relatively long detection interval would make it difficult to obtain dense labels required for model training in practice. Hence, to simulate the actual industrial production situation, the measurement frequency of penicillin concentration was set to once every 4 h, and that of the remaining variables was set to once every 1 h. We obtained one hundred pairs of data in every batch. After the first stage, 48 h were excluded, and data for 352 h per batch was obtained; 89 non-overlapping time windows were used to segment the 356 h data evenly in the time dimension. Finally, the dataset of dimension 10 × 89 × 4 (batches × samples × times) is acquired. The generated data are divided into the training, validation, and test sets in a ratio of 4:3:3. We observed that disturbances and noise affected the training of the localized matrix. In particular, these may interfere with the data distribution, causing the model to learn a wrong relationship among variables, which decreases the reliability of the model. Although preprocessing, such as noise reduction, is necessary, this step was ignored in this specific simulation case. In the feature selection process, the training set was used to calculate the MIC between each process variable and the penicillin concentration P. The results thus obtained are listed in Table 1.

Table 1  
MIC of each variable for penicillin concentration (P).

<table><tr><td>Variable</td><td>MIC</td></tr><tr><td>Aeration rate</td><td>0.9998</td></tr><tr><td>Agitator power</td><td>0.9997</td></tr><tr><td>Substrate feed flowrate</td><td>0.9999</td></tr><tr><td>Temperature</td><td>0.9780</td></tr><tr><td>Substrate concentration</td><td>0.9980</td></tr><tr><td>Dissolved oxygen concentration</td><td>0.9999</td></tr><tr><td>Biomass concentration</td><td>1</td></tr><tr><td>Carbon dioxide concentration</td><td>0.9999</td></tr><tr><td>pH value</td><td>0.9880</td></tr><tr><td>Generated temperature</td><td>1</td></tr><tr><td>Acid flowrate</td><td>0.1947</td></tr><tr><td>Base flowrate</td><td>0.9949</td></tr><tr><td>Cold water flowrate</td><td>0.8293</td></tr></table>

MIC is a tool of measuring information. A larger MIC value indicates a more significant linear or nonlinear correlation between two variables. In the feature selection phase of the proposed method, process variables are sorted according to the MIC values. Then, the SVR soft sensor model is built for quality prediction. The number of selected variables can be determined through model validation. As shown in Fig. 7, the validation loss calculated based on RMSE reaches a minimum value when the number of selected variables is five, six, or seven. According to Occam’s Razor, to maintain the simplicity of the model and reduce the number of model parameters, we selected the following five variables: substrate feed flowrate $( V _ { \mathrm { S F } } ) ,$ , biomass concentration $\left( E _ { \mathrm { B } } \right)$ , dissolved oxygen concentration $\left( E _ { \mathrm { D 0 } } \right)$ , carbon dioxide concentration $( E _ { \mathsf { C O 2 } } )$ and generated temperature $( T _ { \mathrm { G } } ) \mathrm { . }$ . Here, $V _ { \mathrm { S F } }$ is the manipulated variable. Plots of the input variables are shown in $\mathrm { F i g . }$ . 8. Additionally, to describe the dynamic characteristics of the system more accurately, the value of $P$ at the previous moment is added to the current soft sensor modeling. Therefore, $P _ { t - 4 }$ is added to the input as an autoregressive term of the previous moment for $P _ { t }$

![](images/eb4c06cc22f5deb6d5135b307a222a797bc10e97d8605f9678219d31ff0544c8.jpg)  
Fig. 7. Relationship between RMSE and the number of subsets.

![](images/e9773e12178f4dd3300a09484fa00f5aacae5e5d253461e22e9fc73727f0cf33.jpg)

![](images/b3622b7f4ac2de85075629a22daf0fb59e92fd4b3b8817d5112661baeafcb733.jpg)

![](images/42b6cc4ec59636e960ec0a75304e67d2e2556721012729d809216ff49499691f.jpg)

![](images/2a865afbc6e2c59774280f1d2857b99afe0eed72b3ba7a44e33d5d8b2eb3c81c.jpg)

![](images/43516627df9904d9554ca13febc8a0197b38e58ccf36e66d206ce798bede6dd9.jpg)  
Fig. 8. Trajectories of input variables.

The inputs and outputs of the model are as follows:

$$
\left\{ \begin{array}{l l} \mathbf {x} _ {t} = \left[ \begin{array}{c} V _ {S F _ {t, t - 1, t - 2, t - 3}}, E _ {B _ {t, t - 1, t - 2 t, t - 3}}, E _ {D O _ {t, t - 1 t, t - 2, t - 3}}, \\ E _ {c o _ {2 _ {t, t - 1, t - 2, t - 3}}}, T _ {G _ {t, t - 1, t - 2, t - 3}}, P _ {t - 4} \end{array} \right] \\ y _ {t} = [ P _ {t} ] \end{array} \right.\tag{6}
$$

Because each sample contains data collected over the past 4 h, the kernel size of TCM is set to $1 \times 3 .$ . Then, the validation set is used to test the performance of models with different combinations of channel numbers, where the channel number represents the number of convolution kernels in the TCM, and different channels contain information extracted by different convolution kernels. The final model will be determined from the three combinations of channel numbers (2, 4, 8), (4, 8, 16), and (8, 16, 32). RMSE of the three combinations of channel models is shown in Table 2. The model with channels (4, 8, 16) showed the best performance in the validation set. This is because the number of channels represents the size of the model capacity to a certain extent. Models with significantly small capacity are prone to underfitting, whereas those with considerably large capacity are prone to overfitting. Hence, (4, 8, 16) was selected as the number of model channels. We selected Adam-optimizer for gradient descent while setting its learning rate to 0.001. The loss function consists of two parts: prediction loss and entropy loss. $\beta$ denotes a regularization coefficient of the entropy loss, which is used to constrain the sparsity of the learned localized matrix. During the model training process, a significantly large value of $\beta$ results in an over-sparse matrix, whereas a considerably small value of $\beta$ leads to an over-dense matrix. Both cases negatively affect the performance of the GNN-based model. In this work, the value of $\beta$ in Eq. (5) is determined to be 0.08 according to the model performance on the validation set. Generally, the value on an edge connecting two uncorrelated variables is approximately zero. Therefore, a small number close to zero was assigned. Such edges should be removed according to a threshold when visualizing the adjacency matrix. In this work, this threshold is specified based on the $3 { - } \sigma$ principle. In detail, the values smaller than the lower 3-σ limit are set to zero. Additionally, 5% and 10% Gaussian noise is added to the data to analyze the robustness of the model. All experiments were repeated five times, and the calculated average performance was considered as the final value. Four time-steps of historical data were used to predict the critical product quality of the present single time-step. The experimental environment consists of i7-9750H 2.6 Hz × 12 (CPU), RTX3090 24 GiB (GPU), 16 $\mathbf { G B } \times 2$ memory (DDR4), and Linux (OS).

Table 2  
RMSE values of model with different channels.

<table><tr><td>Model</td><td>RMSE</td></tr><tr><td>GCN-based model with channels (2, 4, 8)</td><td>0.039</td></tr><tr><td>GCN-based model with channels (4, 8, 16)</td><td>0.020</td></tr><tr><td>GCN-based model with channels (8, 16, 32)</td><td>0.052</td></tr></table>

We chose the following baseline methods for comparison: the common SVR [8], PLS [10], LSTM [29], and GCN [34]. Among these, PLS and SVM are popular, and LSTM is a commonly used time series model in DL. GCNs are known for their ability to capture variable correlations and were used in this study to verify the effectiveness of localized spatial–temporal correlations. Both PLS and SVM use grid search to select model parameters. The number of layers of LSTM and GCN was set to three, and the embedding dimension of each layer is (16, 32, 64) and the rest of the settings are consistent with the GCN-based model. The RMSE values of the baseline models and the GCN-based model are listed in Table 3. Compared with the SVR and PLS, the RMSE of the GCN-based model in Test 1 of free noise is lower 67.2% and 71.3%, respectively. This is mainly because SVM and PLS cannot effectively capture the spatial and temporal characteristics of the data. To show the complete fermentation process, the first stage, which is not modeled, is also added to the display in Fig. 9. Additionally, models emphasizing the importance of time, such as LSTM, showed good prediction performance. However, LSTM only considers its temporal characteristics and ignores spatial characteristics, resulting in an inferior prediction performance than the GCN-based model.

Furthermore, the GCN-based model outperforms others when the data contain 5% and 10% noise, which mainly relies on graph sparsification [44] and the low-pass filtering of GCN [45]. The former allows the model to capture more essential variable relationships by removing task-independent edges in the graph, which improves its generalization [44]. The latter allows the model to retain low-frequency and ignore high-frequency signals during multiple nodal signal propagations [45]. Because high-frequency signals correspond to noise in this study, the model exhibits improved noise reduction performance. It should be noted that the anti-noise performance discussed in this study is stabilization robustness, which is an adjunct capability of the model rather than a specific capability designed for noise. Therefore, the proposed model has good anti-noise performance in the process of learning localized spatial–temporal correlation, which further improves the prediction accuracy and reliability of the model.

The localized spatial–temporal correlation information learned by the model is visualized to prove its superior performance. The initial and output spatial topology structures of the relationship matrix are illustrated in Fig. 10, where $V _ { \mathrm { S F } }$ is the operation variable and the other four are the process variables. The arrows represent the influence relationships among variables (including positive and negative correlations). Thus, by directly impacting $E _ { \mathrm { B } } ,$ , the operating variable $V _ { \mathrm { S F } }$ affects the remaining three process variables. Fig. 10(b) is explained as follows in combination with the fermentation mechanism [42]:

(a) The nutrient concentration per unit area is increased as the substrate feed flowrate $V _ { \mathrm { S F } }$ increases. The biomass concentration $E _ { B }$ increases when a more favorable growth environment is received by bacteria. Therefore, an increase in $V _ { \mathrm { S F } }$ is accompanied by an increase in $E _ { \mathrm { B } } .$

(b) Due to bacterial respiration, the dissolved oxygen concentration $E _ { \mathrm { D 0 } }$ decreases and the ${ \mathrm { C O } } _ { 2 }$ concentration increases. Simultaneously, the generated temperature $T _ { \mathrm { G } }$ increases as a result of the heat generated by respiration.

(c) With increasing temperature, the solubility of oxygen in a solvent diminishes. As a result, increasing $T _ { G }$ results in a decrease in $E _ { \mathrm { D 0 } }$

(d) An increase in dissolved oxygen concentration and decrease in ${ \mathsf { C O } } _ { 2 }$ concentration and temperature allow bacteria to multiply rapidly. As a result, an increase in $E _ { \mathrm { D 0 } }$ , and decrease in $E _ { \mathsf { C O 2 } }$ and $T _ { \mathrm { G } }$ contribute to an increase in $E _ { \mathrm { B } } .$

The topology structures of the localized matrix are illustrated in Fig. 11. Similar to the conclusion obtained from the relation matrix, in the localized spatial–temporal correlations (a) $V _ { \mathrm { S F } }$ only affects $E _ { \mathrm { B } }$ and (b) $E _ { \mathrm { B } }$ affects the remaining variables.

The test RMSE values of baseline models and GCN-based model.  
Table 3

<table><tr><td>Model</td><td>Free noise in test 1</td><td>5% Noise in test 1</td><td>10% Noise in test 1</td></tr><tr><td>SVR [8]</td><td>0.055</td><td>0.097</td><td>0.179</td></tr><tr><td>PLS [10]</td><td>0.063</td><td>0.098</td><td>0.186</td></tr><tr><td>LSTM [29]</td><td>0.032</td><td>0.076</td><td>0.134</td></tr><tr><td>GCN [34]</td><td>0.033</td><td>0.072</td><td>0.127</td></tr><tr><td>Proposed GCN</td><td>0.018</td><td>0.051</td><td>0.099</td></tr><tr><td>Model</td><td>Free noise in test 2</td><td>5% Noise in test 2</td><td>10% Noise in test 2</td></tr><tr><td>SVR [8]</td><td>0.046</td><td>0.082</td><td>0.167</td></tr><tr><td>PLS [10]</td><td>0.075</td><td>0.095</td><td>0.174</td></tr><tr><td>LSTM [29]</td><td>0.037</td><td>0.069</td><td>0.144</td></tr><tr><td>GCN [34]</td><td>0.038</td><td>0.076</td><td>0.137</td></tr><tr><td>Proposed GCN</td><td>0.012</td><td>0.053</td><td>0.093</td></tr><tr><td>Model</td><td>Free noise in test 3</td><td>5% Noise in test 3</td><td>10% Noise in test 3</td></tr><tr><td>SVR [8]</td><td>0.052</td><td>0.089</td><td>0.169</td></tr><tr><td>PLS [10]</td><td>0.063</td><td>0.091</td><td>0.172</td></tr><tr><td>LSTM [29]</td><td>0.035</td><td>0.061</td><td>0.135</td></tr><tr><td>GCN [34]</td><td>0.032</td><td>0.062</td><td>0.133</td></tr><tr><td>Proposed GCN</td><td>0.010</td><td>0.049</td><td>0.087</td></tr></table>

![](images/7ed2c2ba4116f77714dfa5bd438959d9b3082ae6c2d77f2a374f35a08879f92b.jpg)

(a)  
![](images/0396387e92d34a8ba8756defee8960e758a7288869363de1c58c361dcc689b3d.jpg)  
(b)

![](images/48f35e386833614652c9e76ead17ae49f99028d7f751d6197c9b07f966e367ec.jpg)  
(c)  
Fig. 9. Prediction in different scenarios: (a) free noise. (b) 5% noise, and (c) 10% noise.

![](images/1406e85856b22ef2a703a87de950040117e3fe1ffa0457df4e88cecb6265d794.jpg)  
Fig. 10. (a) The initial topology structure of data, and (b) its output topology structure.

![](images/1410a623e9dafa2e0a98064d0f204c3883f29ca527c52c18c1cbf69cafb0c945.jpg)  
Fig. 11. The localized spatial–temporal correlation.

## 4.2. IndPensim

IndPensim is a simulator of industrial fed-batch fermentation processes (the dataset can be obtained from: www.industrialpe nicillinsimulation.com) [43]. Compared with Pensim, soft sensor modeling in this case is more difficult because it is closer to the actual industrial process. The fermentation duration is between 168 h and 232 h, and the sensor records data every 0.2 h [43].

We used four normal batches (226 h, 230 h, 229 h, and 232 h) in this case. Similar to the Pensim experiment setup, the penicillin concentration measurement frequency is set to once every 1 h, and that for the remaining variables is set to once every 0.2 h. In IndPensim, the first phase lasts 24 h. Therefore, only the data of 24–226 h in each batch is selected to ensure the consistency of the dataset. Finally, the dataset of dimension 4 × 202 × 5 (batches × samples × times) is obtained. As opposed to the

![](images/945ad40e50041346818f3736e1d1465d18906a3132f696cd775c618374fbc1e3.jpg)  
Fig. 12. Relationship between RMSE and number of subsets.

Table 4  
MIC of each variable for penicillin concentration (P)

<table><tr><td>Variable</td><td>MIC</td></tr><tr><td>Basal flow rate</td><td>0.7998</td></tr><tr><td>Heating/cooling water flowrate</td><td>0.9997</td></tr><tr><td>Dissolved oxygen concentration</td><td>0.9282</td></tr><tr><td>Vessel volume</td><td>0.9980</td></tr><tr><td>Vessel weight</td><td>0.9980</td></tr><tr><td>pH value</td><td>0.1923</td></tr><tr><td>Temperature</td><td>0.9014</td></tr><tr><td>Generated temperature</td><td>0.8999</td></tr><tr><td>Carbon dioxide concentration in the off-gas</td><td>0.9980</td></tr><tr><td>Oxygen concentration in the off-gas</td><td>1</td></tr><tr><td>Carbon evolution rate</td><td>0.9947</td></tr><tr><td>Oxygen uptake rate</td><td>0.9949</td></tr></table>

Pensim experimental setup, only two batches were used as the training set, which further demonstrates the effectiveness of the GCN-based model in the scenario of limited training samples. Twelve variables in IndPensim were considered after removing the constant and step variables. The MIC between each process variable and the penicillin concentration P is calculated using the training set shown in Table 4.

Similarly, SVR soft sensor model was built for quality prediction. As shown in Fig. 12. the validation loss calculated based on RMSE reaches a minimum value when the number of selected variables is seven. According to Occam’s Razor, to maintain the simplicity of the model and reduce the number of model parameters, we selected the following seven variables: heating/cooling water flow rate $( R _ { \mathsf { W } } )$ , vessel volume $( V _ { \mathrm { V } } )$ , vessel weight $( V _ { \mathrm { W } } )$ carbon dioxide concentration in the off-gas $\left( E _ { \mathsf { C O 2 } } \right)$ , oxygen concentration in the off-gas $( E _ { 0 2 } ) _ { \mathrm { \Omega } }$ , carbon evolution rate $\left( R _ { C O 2 } \right)$ , and oxygen uptake rate $( R _ { 0 2 } ) .$ . Here, $R _ { \mathrm { { W } } }$ is the manipulated variable. $P _ { t - 5 }$ is added to the input as an autoregressive term of the previous moment for $P _ { t }$ . The inputs and outputs of the model are as follows:

$$
\left\{ \begin{array}{c} \mathbf {x} _ {t} = \left[ \begin{array}{c} R _ {\mathrm{W} _ {t, t - 1, t - 2, t - 3, t - 4}}, V _ {\mathrm{V} _ {t, t - 1, t - 2 t, t - 3, t - 4}}, \\ V _ {\mathrm{W} _ {t, t - 1 t, t - 2, t - 3, t - 4}}, R _ {\mathrm{O2} _ {t, t - 1, t - 2 t, t - 3, t - 4}} \\ E _ {\mathrm{CO2} _ {t, t - 1, t - 2 t, t - 3, t - 4}}, R _ {\mathrm{CO2} _ {t, t - 1, t - 2 t, t - 3, t - 4}}, \\ E _ {\mathrm{O2} _ {t, t - 1, t - 2 t, t - 3, t - 4}}, P _ {t - 5} \end{array} \right] \\ y _ {t} = [ P _ {t} ] \end{array} \right.\tag{7}
$$

Because each sample contains only 1 h data, the kernel size of TCM is set to $1 \times 3 .$ . Using the trial-and-error method, the number of layers and channels in the GCN-based model was determined to be three and (32, 64, 128), respectively. Adam-optimizer is selected for gradient descent, and its learning rate was set to 0.001. The regularization coefficient $\beta$ in Eq. (5) was determined to be 0.2. Additionally, 5% and 10% Gaussian noise was added to the data to investigate the robustness of the model in the case of noisy data.

Table 5  
The test RMSE values of baseline models and GCN-based model

<table><tr><td>Model</td><td>Free noise in test</td><td>5% Noise in test</td><td>10% Noise in test</td></tr><tr><td>SVR [8]</td><td>1.298</td><td>1.513</td><td>1.825</td></tr><tr><td>PLS [10]</td><td>1.515</td><td>1.794</td><td>1.985</td></tr><tr><td>LSTM [29]</td><td>1.279</td><td>1.505</td><td>1.702</td></tr><tr><td>GCN [34]</td><td>0.994</td><td>1.237</td><td>1.593</td></tr><tr><td>Proposed GCN</td><td>0.552</td><td>0.779</td><td>1.108</td></tr></table>

Similar to the previous experiment, SVR [8], PLS [10], LSTM [29], and GCN [34] were chosen for comparison. Moreover, grid search was used to select model hyperparameters for PLS and SVM. The number of layers of LSTM and GCN is set to three, embedding dimension of each layer is (64, 128, 256), and the rest of the settings are consistent with the GCN-based model. The RMSE values of the baseline and GCN-based models are listed in Table 5, with the prediction results shown in Fig. 13. As shown, compared with the SVR and PLS, the RMSE of the GCN-based model in the test of free noise is lower by 57.2% and 63.9%, respectively. Additionally, the prediction performance of GCN in this scenario is better than that of LSTM, which indicates that the importance of variable correlation in soft sensors increases as the system complexity increases and the dataset size decreases. Although a reduced dataset size negatively affects the performance of DL models, the GCN-based model exhibited decent performance with limited data by capturing spatial–temporal localized correlation.

The localized matrix learned by the model is visualized in Fig. 14. Edges with high strength co-existing in the relationship matrix and time correlation matrix should be given greater attention (the edge framed by the red box in Fig. 14). Some examples are: $R _ { \mathrm { W } } {  } R _ { 0 2 } , R _ { \mathrm { W } } {  } R _ { \mathrm { C 0 2 } } , R _ { \mathrm { W } } {  } V _ { \mathrm { V } } , R _ { \mathrm { W } } {  } V _ { \mathrm { W } } , V _ { \mathrm { V } } {  } V _ { \mathrm { W } } , R _ { 0 2 } {  } E _ { 0 2 } ,$ and $R _ { \mathsf { C O 2 } } {  } E _ { \mathsf { C O 2 } }$ . This can be explained in combination with the process mechanism [43]:

(a) $R _ { \mathrm { { W } } }$ changes the solution temperature according to oxygen uptake rate $R _ { 0 2 }$ and carbon evolution rate $R _ { \mathbb { C } 0 2 }$ to control bacterial activity. Meanwhile, $R _ { \mathrm { { W } } }$ affects $V _ { \mathrm { V } }$ and $V _ { \mathrm { { W } } }$ by simultaneously changing solution volume and weight.

(b) Changes in the oxygen uptake rate $R _ { 0 2 }$ and carbon evolution rate $R _ { \mathbb { C } 0 2 }$ are mainly a result of bacterial respiration and can lead to a significant effect on the concentrations of oxygen $E _ { 0 2 }$ and carbon dioxide $E _ { \mathsf { C O 2 } }$ in the off-gas.

## 5. Conclusions

Most data-driven soft sensor models have low explainability. In this work, a GCN-based soft sensor framework is proposed, which exhibits its model superiority by visualizing variable relationships. This model characterizes the cross-correlation among variables by capturing localized spatial–temporal correlations. Variables are used as nodes in the construction of the localized spatial–temporal graph and the network is trained in an end-toend manner. By stacking multiple spatial–temporal convolution layers, the model learns the underlying relationships and temporal correlations among variables in the form of relationships and localized matrices. Thus, it overcomes the limitations of conventional soft sensors. The case study on the penicillin fermentation process demonstrates the efficacy and superiority of the proposed model. The visualization of localized spatial–temporal correlations among variables demonstrates that the information extracted by the model is consistent with process mechanisms, which demonstrates the superior explainability of the proposed GCN-based mode. However, some limitations still exist in the proposed framework, such as variable relationships should be captured in a dynamic form to fit time-varying processes for further study because these relationships may change as the process progresses.

![](images/217e1bc838ab7b7d17761fd06e1761d0a59dff5547c185ea1ed697c357500aa8.jpg)

(a)  
![](images/2a0a3ad8337a1230d641d97093c94bac294c7db25f56f3e9f58bb87f01c57773.jpg)

(b)  
![](images/ba3c417f1b0560deb12539e7df321dc53489ce76be678424981099b1568c61ec.jpg)  
(c)  
Fig. 13. Prediction in different scenarios: (a) free noise, (b) 5% noise, and (c) 10% noise.

![](images/dc8de8c68f271b33de2d83f4e6fc5eccd9919fac7dcfe34d801ce7e9d5e9de16.jpg)  
RW Vv VW ECo2R02EO2 RCo2 RW Vv Vw ECo2RO2EO2RCo2  
Fig. 14. The localized matrix.

## Nomenclature

ANN artificial neural network BN batch normalization BNM bayesian network model CGC conditional Granger-cause DL deep learning FCL fully connected layer GCN graph convolutional network GC test Granger-cause test GNN graph neural network LSCM localized spatial–temporal correlation module LSTM long short-term memory MI mutual information MIC maximum information coefficient PLS partial least squares ReLU rectified linear unit RMSE root mean square error S2S sequence-to-sequence STCL spatial–temporal convolutional layer SVR support vector regression TCM temporal convolution module TCN temporal convolutional network

## Symbol

a the size of convolutional kernel A the adjacency matrix of the graph G A’ the localized matrix of variables

$\mathbf { A } _ { \mathrm { t } }$ the time correlation matrix of variables C the number of channels D the degree matrix of the adjacency matrix A E the edge of edges $E _ { \mathrm { B } }$ the biomass concentration $E _ { \mathsf { C O 2 } }$ the carbon dioxide concentration $E _ { \mathrm { D 0 } }$ the dissolved oxygen concentration $E _ { 0 2 }$ the oxygen concentration in the off-gas $G$ the graph $\mathrm { H } ( . )$ the element-level entropy $\mathbf { j } , \mathbf { u }$ two discrete variables $\mathbf { o }$ the zero matrix $P$ the penicillin concentration $\mathbf { Q }$ the trainable weight of GCN $R _ { \mathbb { C } 0 2 }$ the carbon evolution rate $R _ { 0 2 }$ the oxygen uptake rate $R _ { \mathrm { { W } } }$ the heating/cooling water flow rate $T$ the time step of the observation $T _ { \mathrm { G } }$ the generated temperature $T _ { m }$ the number of time step $V$ the number of nodes $\mathbf { v }$ the set of vertices $V _ { \mathrm { V } }$ the vessel volume $V _ { \mathrm { { W } } }$ the vessel weight $V _ { \mathrm { S F } }$ the substrate feed flowrate $\mathbf { X } _ { \mathrm { G } }$ the graph node signal matrix $\mathbf { X } _ { G } ^ { \prime }$ the localized graph signal matrix $\beta$ the coefficient of regularization

## CRediT authorship contribution statement

Mingwei Jia: Methodology, Software, Data curation, Visualization, Writing – original draft. Danya Xu: Methodology, Software, Validation. Tao Yang: Conceptualization, Resources, Writing – review & editing. Yi Liu: Conceptualization, Resources, Writing – review & editing, Supervision, Funding acquisition. Yuan Yao: Conceptualization, Writing – review & editing, Supervision, Funding acquisition.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

Data will be made available on request.

## Acknowledgment

The work was supported by the National Natural Science Foundation of China (Grant Nos. 62022073 and 61873241) and National Science and Technology Council, ROC (Grant No. NSTC 111-2221-E-007-005).

## References

[1] L. Fortuna, S. Graziani, A. Rizzo, M.G. Xibilia, Soft Sensors for Monitoring and Control of Industrial Processes, Springer, London, UK, 2007.

[2] F.A.A. Souza, R. Araújo, J. Mendes, Review of soft sensor methods for regression applications, Chemom. Intell. Lab. Syst. 152 (15) (2016) 69–79.

[3] B. Lin, B. Recke, J.K.H. Knudsen, S.B. Jørgensen, A systematic approach for soft sensor development, Comput. Chem. Eng. 31 (5/6) (2007) 419–425.

[4] M. Kano, M. Ogawa, The state of the art in chemical process control in Japan: Good practice and questionnaire survey, J. Process Control 20 (9) (2010) 969–982.

[5] S.J. Qin, L.H. Chiang, Advances and opportunities in machine learning for process data analytics, Comput. Chem. Eng. 126 (2019) 465–473.

[6] L.F. Fuentes-Cortes, A. Flores-Tlacuahuac, K.D.P. Nigam, Machine learning algorithms used in PSE environments: A didactic approach and critica perspective, Ind. Eng. Chem. Res. 61 (2022) 8932–8962.

[7] S.B. Chitralekha, S.L. Shah, Application of support vector regression for developing soft sensors for nonlinear processes, Can. J. Chem. Eng. 88 (2010) 696–709.

[8] Z. Li, H.P. Jin, S.L. Dong, B. Qian, B. Yang, X.G. Chen, Semi-supervised ensemble support vector regression based soft sensor for key quality variable estimation of nonlinear industrial processes with limited labeled data, Chem. Eng. Res. Des. 179 (2022) 510–526.

[9] W.W. Yan, D. Tang, Y.J. Lin, A data-driven soft sensor modeling method based on deep learning and its application, IEEE Trans. Ind. Electron. 64 (5) (2017) 4237–4245.

[10] J.H. Zheng, Z.H. Song, Mixture modeling for industrial soft sensor application based on semi-supervised probabilistic PLS, J. Process Control 84 (4) (2019) 46–55.

[11] A. Khosbayar, J. Valluru, B. Huang, Multi-rate Gaussian Bayesian network soft sensor development with noisy input and missing data, J. Process Control 105 (2021) 48–61.

[12] L. Patanè, M.G. Xibilia, Echo-state networks for soft sensor design in an SRU process, Inform. Sci. 566 (2021) 195–214.

[13] K. Desai, Y. Badhe, S.S. Tambe, B.D. Kulkarni, Soft-sensor development for fed-batch bioreactors using support vector regression, Biochem. Eng. J. 27 (3) (2006) 225–239.

[14] P. Kadlec, B. Gabrys, S. Strandt, Data-driven soft sensors in the process industry, Comput, Chem, Eng, 33 (4) (2009) 795–814

[15] Y.Q. Liu, M. Xie, Rebooting data-driven soft-sensors in process industries: A review of kernel methods, J. Process Control 89 (2020) 58–73.

[16] P. Zhou, W.Q. Chen, C.M. Yi, Z.H. Jiang, T. Yang, T.Y. Chai, Fast just-in-timelearning recursive multi-output LSSVR for quality prediction and control of multivariable dynamic systems. Eng. Appl. Artif, Intell. 100 (2021) 104168.

[17] A. Mohammadi, R. Zarghami, D. Lefebvre, S. Golshan, N. Mostoufi, Soft sensor design and fault detection using Bayesian network and probabilistic principal component analysis, J. Adv. Manu. Process. 1 (4) (2019) 10027.

[18] T.B. Lopez-Garcia, A. Coronado-Mendoza, J.A. Domínguez-Navarro, Artificia neural networks in microgrids: A review, Eng. Appl. Artif. Intell. 95 (2020) 103894.

[19] K.X. Liu, M.K. Zheng, Y. Liu, J.G. Yang, Y. Yao, Deep autoencoder thermography for defect detection of carbon fiber composites, IEEE Trans. Ind Inform. (2022) http://dx.doi.org/10.1109/TII.2022.3172902, in press.

[20] S. Gao, Y. Dai, Y.J. Li, K.X. Liu, K. Chen, Y. Liu, Multiview Wasserstein generative adversarial network for imbalanced pearl classification. Meas. Sci. Technol. 33 (8) (2022) 085406.

[21] Y. Liu, M.K. Zheng, K.X. Liu, Y. Yao, S. Sfarra, TriMap thermography with convolutional autoencoder for enhanced defect detection of polymer composites, J. Appl. Phys. 131 (14) (2022) 144901.

[22] Q. Liu, Y. Zhang, G. Wu, Z. Fan, Disturbance robust abnormality diagnosis of fused magnesium furnaces using deep neural networks, IEEE Trans. Artif Intell. (2022) http://dx.doi.org/10.1109/TAI.2022.3168251, in press.

[23] D.X. Chen, X.L. Liu, W.W. Yu, L. Zhu, Q.P. Tang, Neural-network based adaptive self-triggered consensus of nonlinear multi-agent systems with sensor saturation, IEEE Trans. Netw. Sci. Eng. 8 (2) (2021) 1531–1541.

[24] Q.Q. Sun, Z.Q. Ge, Probabilistic sequential network for deep learning of complex process data and soft sensor application, IEEE Trans. Ind. Inform. 15 (5) (2019) 2700–2709.

[25] C. Shang, F. Yang, D. Huang, W.X. Lyu, Data-driven soft sensor development based on deep learning technique, J. Process Control 24 (3) (2014) 223–233.

[26] Y. Liu, C. Yang, Z.L. Gao, Y. Yao, Ensemble deep kernel learning with application to quality prediction in industrial polymerization processes, Chemom. Intell. Lab. Syst. 174 (15) (2018) 15–21.

[27] T. Yang, J.L. Ding, K.G. Vamvoudakis, S.J. Qin, Guest editorial: Industrial artificial intelligence for smart manufacturing, IEEE Trans. Ind. Inform. 17 (12) (2021) 8319–8323.

[28] H.G. Han, Q.L. Chen, J.F. Qiao, An efficient self-organizing RBF neural network for water quality prediction, Neural Netw. 24 (7) (2019) 717–725.

[29] W. Xie, J. Wang, C. Xing, S. Guo, M. Guo, L. Zhu, Variational autoencoder bidirectional long and short-term memory neural network soft-sensor model based on batch training strategy, IEEE Trans. Ind. Inform. 17 (8) (2021) 5325–5334.

[30] X.F. Yuan, B. Huang, Y.L. Wang, C.H. Yang, W.H. Gui, Deep learningbased feature representation and its application for soft sensor modeling with variable-wise weighted SAE, IEEE Trans. Ind. Inform. 14 (7) (2018) 3235-3243.

[31] S.C. Chang, C.H. Zhao, K. Li, Consistent-contrastive network with temporality-awareness for robust-to-anomaly industrial soft sensor, IEEE Trans. Instrum. Meas. 71 (2021) 2502512.

[32] F. Xia, K. Se, S. Yu, A. Aziz, L.T. Wan, S.R. Pan, H. Liu, Graph learning: A survey, IEEE Trans. Artif. Intell. 2 (2) (2021) 109–127.

[33] L. Shi, Y. Zhang, J. Cheng, H. Lu, Two-stream adaptive spectral graph convolutional networks for skeleton-based action recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, 2019, pp. 12026–12035.

[34] F.T.N. Kip, M. Welling, Semi-supervised classification with graph convolutional networks, in: 5th International Conference on Learning Representations (ICLR), Toulon, 2016.

[35] Z.W. Chen, Q. Deng, H. Ren, Z.R. Zhao, T. Peng, C.H. Yang, W.H. Gui, A new energy consumption prediction method for chillers based on GraphSAGE by combining empirical knowledge and operating data, Appl. Energy 310 (15) (2022) 118410.

[36] J.M. Xu, H.B. Ke, Z.W. Chen, X.Y. Fan, T. Peng, C.H. Yang, Over-smoothing relief graph convolutional network-based fault diagnosis method with application to the rectifier of high-speed trains, IEEE Trans. Ind. Inform. 19 (2022) 771–779.

[37] L.J. Feng, C.H. Zhao, Y.L. Li, M. Zhou, C. Fu, Multichannel diffusion graph convolutional network for the prediction of end-point composition in the converter steelmaking process, IEEE Trans. Instrum. Meas. 70 (2020) 1–13.

[38] Y.H. Wang, P.F. Yan, M.G. Gai, Dynamic soft sensor for anaerobic digestion of kitchen waste based on SGSTGAT, IEEE Sens. J. 21 (17) (2021) 19198–19208.

[39] C. Song, Y. Lin, S. Guo, H. Wan, Spatial–temporal synchronous graph convolutional networks: a new framework for spatial–temporal network data forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34. (1) New York, 2020, pp. 914–921.

[40] S. Bai, J.Z. Kolter, V. Koltun, Trellis networks for sequence modeling, in: 8th International Conference on Learning Representations (ICLR), New Orleans, 2019.

[41] D.N. Reshef, Y.A. Reshef, H.K. Finucane, S.R. Grossman, G. McVean, P.J. Turnbaugh, E.S. Lander, M. Mitzenmacher, P.C. Sabeti, Detecting novel associations in large data sets, Science 6062 (2011) 1518–1524.

[42] G. Birol, C. Undey, A. Cinar, A modular simulation package for fedbatch fermentation: penicillin production, Comput. Chem. Eng. 11 (2002) 1553-1565

[43] S. Goldrick, A. Stefan, D. Lovett, G. Montague, B. Lennox, The development of an industrial-scale fed-batch fermentation simulation, J. Biotechnol. 193 (1) (2015) 70–82.

[44] C. Zheng, B. Zheng, W. Cheng, D.J. Song, J.C. Ni, W.C. Yu, H.F. Chen, W. Wang, Robust graph representation learning via neural sparsification. in: 37th International Conference on Machine Learning, ICML, 2020, pp. 11458–11468.

[45] H. NT, T. Maehara, T. Murata, Revisiting graph neural networks: graph filtering perspective, in: 25th International Conference on Pattern Recognition, ICPR, 2021, pp. 8376–8383.