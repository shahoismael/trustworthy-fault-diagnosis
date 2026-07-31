# A comprehensive hybrid first principles/machine learning modeling framework for complex industrial processes

![](images/2540c59eb0983154c794c1f23b875081ff2da353c0a14c66b4967a2552aee0a2.jpg)

Bei Sun<sup>a,b</sup>, Chunhua Yang<sup>a,∗</sup>, Yalin Wang<sup>a</sup>, Weihua Gui <sup>a</sup>, Ian Craig<sup>c</sup>, Laurentz Olivier <sup>c</sup>

<sup>a</sup> School of Automation, Central South University, Changsha, 410083, China

<sup>b</sup> the Peng Cheng Laboratory, Shenzhen, 518000, China

<sup>c</sup> Department of Electrical, Electronic, and Computer Engineering, University of Pretoria, Pretoria 0002, South Africa

## a r t i c l e i n f o

Article history: Received 29 August 2019 Revised 26 November 2019 Accepted 30 November 2019 Available online 20 December 2019

Keywords: Comprehensive state space Descriptive system Modeling Machine learning

## a b s t r a c t

The selection of an appropriate descriptive system and modeling framework to capture system dynamics and support process control applications is a fundamental problem in the operation of industrial processes. In this study, to account for the highly complex dynamics of industrial process and additional requirements imposed by smart and optimal manufacturing systems, an extended state space descriptive system, named comprehensive state space, is first designed. Then, based on the descriptive system, a hybrid first principles/machine learning modeling framework is proposed. The hybrid model is formulated as a combination of a nominal term and a deviation term. The nominal term covers the underlying physicochemical principles. The deviation term handles the effects of high-dimensional influence factors using regression of low-dimensional deep process features. To handle the multimodal and time-varying properties of process dynamics, the comprehensive state space is divided into subspaces indicating different operating conditions. The model parameters are identified and trained for each operating condition to form the sub-models. Then the system dynamics are formulated as a weighted sum of sub-models, with the weights being the probabilities that the current operating point belongs to different operating conditions. The weights update with the movement of the operating point in the comprehensive state space. Moreover, the descriptive system provides a platform for visualization, and can act as a digital twin of the physical process. A case study illustrates the feasibility and performance of the proposed descriptive system.

© 2019 Elsevier Ltd. All rights reserved.

## 1. Introduction

Process industries, which mainly include iron and steel, nonferrous metals, petrochemical and architectural materials, are the cornerstones of the economies of many countries [12]. By using continuous and complex physicochemical processes, they transform elementary raw materials into products that support economic development and major engineering projects. The utilization rate of raw materials, energy consumption, production costs and environmental effects are the main concerns in the daily operation of process industries. With the aim of high-eficiency and green production, modeling, optimization and control of industrial processes have long been recognized as important and challenging problems.

Descriptive systems and modeling approaches play a fundamental role in capturing process dynamics and support process control applications [28]. Traditionally, a description of system dynamics involves the use of a set of mathematical models to represent the physical and chemical phenomena, static and dynamic behaviors, as well as the causal relationships among observed quantities, including manipulated inputs and technical indexes. These mathematical models are widely used in model predictive control (MPC), state estimation, soft sensing, process monitoring, fault detection and diagnosis (FDD), operational optimization, plant design, process simulation, control performance evaluation, etc. [25,26]. The types of system models are many, caused by differences in the modeling approach and the purpose of the model. According to Ljung [24], system models can be categorized using a whole palette of grey shades from white to black, including white models (first principle models described by e.g. differential/algebraic equations (DAEs)), off-white models (white models with unknown or uncertain parameters described by e.g. state-space models), smoke-grey models (e.g., semi-physical modeling), steel-grey models (models linearized around an operating point), slate-grey models (e.g., hybrid models, block-oriented models) and black models (e.g., neural network, support vector machines). All these models are essentially aimed at extracting facts about the process dynamics from the measured data, information and knowledge.

Therefore, determination of the model structure, utilization of measured data, information and knowledge of reaction mechanisms are vital in process modeling. Among these models, hybrid models provide more flexibility in defining the model structure and utilizing available information [30]. Akkisetty et al. [2] proposed a semi-empirical hybrid model which integrates a population balance model with a neural network model to predict the milled particle size distribution. In Alavi et al. [3], a neural network model is incorporated within the mass balance model of an adiabatic fixed-bed reactor to reduce the computation load. Keskitalo and Leiviskä [18] combined several artificial neural networks into ensembles to account for the difference between the mechanistic model outputs and the real process values. Chaffart and Ricardez-Sandoval [10] proposed a hybrid model for the simulation of thin film deposition. The multiscale mechanistic model combined continuum differential equations describing the transport of the precursor gas phase and a stochastic partial differential equation (SPDE) predicting the evolution of the thin film surface. In order to accurately predict the thin film growth over wide operating ranges, an artificial neural network was trained to predict the coeficients of the mechanistic model. These hybrid modeling approaches mainly combine first-principle models with neural network models that are used to predict the modeling error or the coeficients of the first principle model. However, the multimodal and timevarying properties of the process dynamics are not considered.

With the shrinking of high grade mineral resources, mixed mineral supply, and fluctuating feed conditions, industrial processes in the current age are highly complex. The process dynamics exhibit different modes and vary with time. It is often not possible for traditional process models to include all the required information of such systems. Varying degrees of model uncertainty therefore exist, depending on the modeling effort. A descriptive model structure that comprehensively describes such a system and that reduces model uncertainty to a minimum, should be able to:

(i) Support the comprehensive and precise description of an industrial process.

(ii) Be ”flexible enough to cover many relevant nonlinear phenomena, at the same time as they allow inclusion of physical insight in order not to be too flexible” [24].

(iii) ”Describe static and dynamic process characteristics in the whole operational range” [16].

Correspondingly, a descriptive system supporting such a model type should comprehensively cover the essential factors influencing process dynamics. In addition, in order to realize smart and optimal manufacturing, such descriptive system should also be able to:

(i) Serve as an interface for digitalization and visualization.

(ii) Be compatible with both data analytics and control design.

(iii) Serve as a container for different sources of information collected along the production life-cycle.

(iv) Systematically support process control applications along the production life-cycle, e.g., modeling, optimization, FDD and control.

In this study, an extended state space descriptive system, namely Comprehensive State Space (CSS), is first designed to cope with the high complexity of industrial processes. The CSS descriptive system is a vector space which covers the essential factors that influence the process dynamics including inlet conditions, reaction conditions and output states. Then, under the CSS descriptive system, a hybrid first principles/machine learning modeling framework, is proposed. In the hybrid modeling framework, a practical industrial process is viewed as an interconnected system whose dynamics are formulated as a combination of a nominal term and a deviation term. The nominal term is a first-principle model (FPM) of the process. The deviation term accounts for the affects of the high-dimensional influence factors uncovered by the FPM. It is expressed in a regression form with its inputs being the low-dimensional deep process features extracted from production data via stacked auto-encoders (SAE) [34]. This model structure contains the underlying physicochemical principles while incorporating high-dimensional influence factors in a low-dimensional manner. To handle the multimodal and time-varying properties of process dynamics, the deep process features are first used to divide the CSS into subspaces. A subspace is a partition of CSS indicating certain operating condition. By identifying model parameters under different operating conditions, sub-models of the process are obtained. Then, the process model is expressed as a weighted sum of the sub-models. The weights are the probabilities that the current operating point belongs to different operating conditions and changes with the movement of the operating point in the CSS. Moreover, the CSS description enables the visualization of the evolution of the operating point and the digitalization of the process.

![](images/53bf2754ca762d7df4de723c2212fff97f0ae4953313ffd5e10d1601fc6e8e1c.jpg)  
Fig. 1. A plant composed of N unit processes

The rest of this paper is organized as follows. The complexity of industrial processes is first analyzed in Section 2. Then, the definition of the comprehensive state space descriptive system is given in Section 3. Based on the CSS descriptive system, a hybrid first principles/machine learning modeling framework is proposed in Section 4. For demonstration purpose, the feasibility and capability of the hybrid modeling framework and CSS descriptive system is illustrated via a case study in Section 5. Conclusions are drawn in Section 6.

## 2. Highly complex industrial processes

A plant is designed as a combination of multiple unit processes with specific functions. An entire plant could be viewed as a material flow and processing network in which a unit process or a single piece of equipment acts as a node. Each unit process is interconnected with other unit processes through mass/heat transfers, recycles or reentrances (Fig. 1) [21]. As a consequence, the inlet conditions of a unit process is prone to variations caused by:

(i) The fluctuations of the physicochemical properties and composition of the raw materials caused by the change of suppliers, price volatility of raw material and reagents, etc.

(ii) Fluctuations in the feed flow rate.

(iii) Disturbances in upstream unit processes.

(iv) Plant-wide adjustment caused by malfunctions, maintenance, etc.

In a unit process or a single piece of equipment, there can exist multiple main reactions, each reaction corresponding to a reaction step, especially for processes with slow dynamics or with complex production technology. In addition, the heterogeneous nature of the raw feed materials with its associated elements/compounds causes side reactions to occur besides the main reactions for which the process was designed. Whereas the kinetic model of the main reactions can be derived by applying chemical and physical principles, the practical dynamics of the main reactions are more complex (Fig. 2):

![](images/ab756ac5070633b058d2646e8c3f91be3bab87e08bbb945f153381eb8f7ed1c9.jpg)  
Fig. 2. The interactions between the main reactions, side reactions and auxiliary system in a unit process.

(i) The side reactions can interact with the main reactions by competition or promotion.

(ii) The dynamics of the side reactions and their interaction mechanisms with the main reactions are only partially known in most cases.

(iii) The variation of the physicochemical properties and composition of the feed material can change the type of the side reactions, and the proportions of the side and main reactions.

Take the zinc hydrometallurgy process as an example. In its cobalt removal process, the main reaction that occurs is the electrode reaction between cobalt and zinc catalyzed by arsenic trioxide [31]. The copper ions from the previous copper removal processes are involved in the following two reactions:

$$
\mathrm{Cu} ^ {2 +} + \mathrm{Zn} = \mathrm{Cu} + \mathrm{Zn} ^ {2 +}\tag{1}
$$

$$
\mathrm{HAsO} _ {2} + 3 \mathrm{Cu} ^ {2 +} + 3 \mathrm{H} ^ {+} + 4. 5 \mathrm{Zn} = \mathrm{Cu} _ {3} \mathrm{As} + 4. 5 \mathrm{Zn} ^ {2 +} + 2 \mathrm{H} _ {2} \mathrm{O}\tag{2}
$$

Reaction (1) consumes zinc dust. Reaction (2) produces a Cu-As alloy on the surface of the zinc dust which then serves as a substrate for the cementation of cobalt ions.

$$
\mathrm{HAsO} _ {2} + \mathrm{Co} ^ {2 +} + 3 \mathrm{H} ^ {+} + 2. 5 \mathrm{Zn} = \mathrm{CoAs} + 2. 5 \mathrm{Zn} ^ {2 +} + 2 \mathrm{H} _ {2} \mathrm{O}\tag{3}
$$

Thus, an appropriate amount of copper ions would promote cobalt removal. An excessive amount of copper ions will however compete with the cobalt ions by consuming zinc dust, and hence impede the cobalt removal process.

Moreover, kinetic models are often not a reflection of current practice as production conditions can differ from the experimental conditions used to derive these models. Auxiliary systems often form part of unit process to supply necessary reaction conditions, e.g., temperature and pH (Fig. 2). So,

(i) Changing the configuration of the auxiliary system can also affect the dynamics of the controlled reactions.

(ii) In addition, change of the reaction conditions can hinder part of the reactions and promote others, resulting in the variation of the process dynamics.

Therefore, for each unit process or a single piece of equipment, the main reactions are exposed to the inlet and reaction conditions. The inlet conditions are affected by the upstream and downstream unit processes, while the reaction conditions are determined by the configuration of the auxiliary system and type and portion of the side reactions. In this sense, the practical dynamics of the main reactions is an outcome of the intricate interactions among the main reactions, internal environment (side reactions, auxiliary system) and external environment (interconnected unit processes) (Fig. 3).

To sum up, an industrial process forms part of a multi-scale system ranging from molecular reactions, particle collisions, equipment interactions (basic production unit), plant-wide interactions (whole production life-cycle) to the global market (the global value-chain that affects plant production and profit). In addition, due to the inherent complexity and the aforementioned intricate interactions, the dynamics of the main reactions are nonlinear, containing uncertainties and are not consistent under different inlet and reaction conditions. Therefore, building a comprehensive and precise model of an industrial process, which can cover the dynamics of an industrial process under various operating conditions, requires knowledge from multiple disciplines. On the other hand, various restrictions in plant testing result in insuficient information about many aspects of the system dynamics, that could lead to significant variance errors in the resulting model [22]. In order to more accurately describe highly complex industrial processes, additional descriptive methods, beyond those described in Ljung [24], are required to increase the utilization of information.

![](images/b98e96e58a138b1cb020dc3f06cd82c380bb75f3cc410683e02ef1eba2edee69.jpg)  
Fig. 3. Interaction between the main reactions, internal environment and externa environment.

![](images/0cd19323caad0c5a101fed75193ee9deef36bcbe86f579ed9331b505afa44f08.jpg)  
Fig. 4. Unit process interactions.

## 3. Definition of comprehensive state space

As discussed in Section 2, the system dynamics and the intricate interactions that exist in a unit process can be illustrated by Fig. 4. If the output states are denoted as $\mathbf { x } _ { 0 }$ , the inlet conditions as $\mathbf { x } _ { \mathrm { I } } .$ , the reaction conditions as $\mathbf { x } _ { \mathrm { R } }$ , then the dynamics of the output states can be described by

$$
\dot {\mathbf {x}} _ {0} = \mathbf {g} (\mathbf {x} _ {0}, \mathbf {x} _ {\mathrm{I}}, \mathbf {x} _ {\mathrm{R}})\tag{4}
$$

where $\pmb { \mathrm { g } } ( \cdot )$ is a function representing the relation between $\dot { \mathbf { x } } _ { 0 }$ and $\mathbf { x } _ { 0 }$ , x , x . In this sense, the traditional state-space, which has its states being the output states $\mathbf { x } _ { 0 }$ , can be extended, by incorporating the inlet conditions $\mathbf { x } _ { \mathrm { I } }$ and reaction conditions $\mathbf { x } _ { \mathrm { { R } } } .$ , to a ’Comprehensive state space’ (Fig. 5).

Comprehensive state space: Comprehensive state space is a three dimensional vector space with each dimension being the codes of:

Output states $( \mathbf { x } _ { 0 } ) \colon$ The output states that describe the dynamic behavior of the main reactions, which are mainly the controlled technical indexes, e.g., outlet ion concentration of a continuous stirred tank reactor.

![](images/1cd8af995ac356984985a53afa70932f074ae7fba96150265b598265bbc0ec10.jpg)

Fig. 5. Illustration of comprehensive state space $( \mathsf { P } _ { 0 } , \mathsf { P } _ { 1 } , \mathsf { P } _ { \mathrm { k } }$ are different points in comprehensive state space).  
![](images/83d0019c58f61503c662a2469be53a6e4a24f8ef3cd304003b69d703a8c3e688.jpg)  
Fig. 6. Coordinate and attributes of a point in comprehensive state space.

<sub>•</sub> Inlet conditions (x ): The quantity, physicochemical properties, composition of the feed material, $\mathrm { e . g . , }$ , inlet flow rate, species concentration.

<sub>•</sub> Reaction conditions (x ): The conditions under which the main and side reactions take place. e.g. temperature, pH, stirring rate.

As $\mathbf { x } _ { 0 } .$ , x and $\mathbf { x } _ { \mathrm { R } }$ are high-dimensional, they are compressed into one-dimensional variables according to certain coding rules [7]. As shown in Fig. 6, x , $\mathbf { x } _ { \mathrm { I } }$ and $\mathbf { x } _ { \mathrm { R } }$ are coded as hexadecimal numbers. Different digits of the hexadecimal code represent different variables, e.g., three variables in the reaction conditions $\mathbf { x } _ { \mathrm { R 1 } }$ x , x take the value of ’06’, ’2A’ and ’05’ respectively.

The CSS description system covers more influence factors than the traditional state space composed solely of output states, which reduces the uncertainty in modeling. As shown in Figs. 5 and 6, each point in the comprehensive state space represents different operating points with corresponding attributes, including model structure, value of model parameters, type of operating condition, etc. The coordinate of a operating point depends on the value of output states, inlet conditions and reaction conditions (Fig. 6). The ’coordinate-attributes’ information vectors of each point in the CSS form a comprehensive description of the process. On the other hand, the CSS serves as a container for the ’coordinate-attributes information pairs, which can then act as a digital twin of the physical process.

In the following two sections, a study is performed to show how the CSS can support operating condition recognition and process modeling, or in other words, how to obtain the attributes of the operating condition and process model under the CSS framework.

Remark 1. Consider the output states $\mathbf { x } _ { 0 }$ and a fixed-length time interval [t , t ]. If the initial values of the output states $\mathbf { x } _ { 0 } ( t _ { 0 } )$ are the same, and the inlet conditions and reaction conditions are the same during [t<sub>0</sub>, t ], then starting from any $t _ { 0 } ,$ the final value of ${ \bf x } _ { 0 } ( t _ { f } )$ is the same.

Remark 2. In most cases, the manipulated variables affect the system dynamics by changing the reaction conditions $\mathbf { x } _ { \mathrm { R } } ,$ e.g., in the cobalt removal process, the flow rate of spent acid affects the pH, the dosage of zinc dust affects the overall oxidation–reduction atmosphere which is represented by the oxidation–reduction potential [32]. However, if under certain working conditions the inlet conditions have to be changed, some variables that determine the inlet conditions then become manipulated variables.

Remark 3. Inside a suficiently small subspace in the entire ’comprehensive state space’, the system dynamics, i.e. the model structure and parameters can be regarded as constant. As in different subspaces, the inlet conditions, reaction conditions and output states are different. Therefore, different subspaces indicate different operating conditions.

Remark 4. If the process behavior inside each subspace can be obtained, then the comprehensive state space can serve as a digital twin of the physical process. The operators can then access the location of the current operating condition in the comprehensive state space to follow and monitor the evolution of the process.

## 4. A hybrid first principles/machine learning modeling framework

In this section, a CSS-based hybrid first principles/machine learning modeling framework is presented. The process is modeled as a combination of a nominal first-principle model and a machine-learning based input-output model accounting for the deviation between the nominal model and actual process dynamics. The low-dimensional deep process features are first extracted to divide the CSS into different partitions indicating different operating conditions. Then, the sub-models under each operating condition are obtained by identifying the model parameters. The dynamic process model is formulated as a weighted sum of the sub-models. The weights are the probabilities that the current operating point belongs to different operating conditions and update continuously with the movement of the working point in the CSS.

## 4.1. Deep feature extraction

Deep feature extraction is used to transform high-dimensional raw data into a low-dimensional deep process feature set which can be used to detect the patterns in the inputs via multiple levels of representation [11,20,29]. Deep learning methods with higher layers of representation can learn very complex nonlinear functions. SAE is a type of unsupervised deep network, which is composed of multiple-level and stacked auto-encoders. As shown in Fig. 7, an auto-encoder has an input layer, output layer and one hidden layer. It aims to reproduce the input signal in the output layer via encoding and decoding, and extract the latent representation of inputs by limiting the number of hidden units. An auto-encoder is configured by minimizing the ’input-output approximation error, while the extracted latent representation serves as the inputs of its successive auto-encoder.

Consider the ith auto-encoder that first extracts a set of latent variables from the inputs,

$$
\mathbf {Y} _ {i} = \mathbf {H} (\mathbf {W} _ {i} \mathbf {X} _ {i} + \mathbf {b} _ {i})\tag{5}
$$

then the decoder reconstructs the outputs from the latent variables

$$
\hat {\mathbf {Y}} _ {i} = \mathbf {H} ^ {'} (\mathbf {W} _ {i} ^ {'} \mathbf {Y} _ {i} + \mathbf {b} _ {i} ^ {'})\tag{6}
$$

![](images/85b700c2792fcdebf8cf2d79e58cb9bd0f212fbef84e0fb9c028021556a310c8.jpg)  
Fig. 7. Deep feature extraction, regression, and subspace dividing.

where $\mathbf { X } _ { i } \in \mathbb { R } ^ { n _ { i } }$ represents the input, $\mathbf { Y } _ { i } \in \mathbb { R } ^ { m _ { i } }$ represents the hidden variables, $\hat { \mathbf { Y } } _ { i } \in \mathbb { R } ^ { n _ { i } }$ represents the output, H(·) and $\boldsymbol { \mathbf { \mathit { H } } } ^ { \prime } ( \cdot )$ are the element-wise activation functions, e.g., sigmoid function, ${ \bf W } _ { i } \in  \qquad $ $\mathbb { R } ^ { m _ { i } \times n _ { i } }$ $\mathbf { W } _ { i } ^ { ' } \in \mathbb { R } ^ { n _ { i } \times m _ { i } }$ and $\mathbf { b } _ { i } \in \mathbb { R } ^ { m _ { i } } , \ \mathbf { b } _ { i } ^ { ' } \in \mathbb { R } ^ { n _ { i } }$ are the weighting matrices and bias vectors respectively, that can be identified by minimizing the reconstruction error on a training set with M samples:

$$
\ell_ {\mathrm{SAE}} (\mathbf {W} _ {i}, \mathbf {W} _ {i} ^ {\prime}, \mathbf {b} _ {i}, \mathbf {b} _ {i} ^ {\prime}) = \sum_ {j = 1} ^ {M} \| \mathbf {X} _ {i} ^ {(j)} - \hat {\mathbf {Y}} _ {i} ^ {(j)} \| ^ {2}\tag{7}
$$

where $\mathbf { X } _ { i } ^ { ( j ) }$ and $\hat { \mathbf { Y } } _ { i } ^ { ( j ) }$ are the inputs and the reconstruction of the inputs of the jth training sample.

The pre-training of the SAE is conducted in a layer-wise manner. After all the auto-encoders are pre-trained, the hidden variables of the last auto-encoder are taken as the deep process feature set $\begin{array} { r } { \mathbf { z } = \mathbf { Y } _ { \mathrm { L } } } \end{array}$ . The initial regression function parameters of the deep process features α are identified by minimizing the regression error:

$$
\ell_ {\text { Regression }} (\boldsymbol {\alpha} _ {1}, \dots , \boldsymbol {\alpha} _ {N}) = \sum_ {j = 1} ^ {M} \| \mathbf {x} _ {0} ^ {\text { deviation } (j)} - \mathbf {E} (\mathbf {z} ^ {(j)}) \| ^ {2}\tag{8}
$$

where $\mathbf { x } _ { 0 } ^ { \mathrm { d e v i a t i o n } ^ { ( j ) } }$ and $z _ { i } ^ { ( j ) }$ are the deviation terms and the deep process feature variables of the jth training sample. E(·) is a regression function set of the deep process feature z. After pre-training, $[ { \pmb { \alpha } } _ { 1 } , \ \cdots , \ { \pmb { \alpha } } _ { N } ]$ and $[ { \bf W } _ { i } , { \bf W } _ { i } ^ { ' } , { \bf b } _ { i } , { \bf b } _ { i } ^ { ' } ]$ are used as initial solutions to fine-tune the weights and minimize the overall regression modeling error via back-propagation.

## 4.2. Deep feature space partitioning

After the extraction, the deep process features are fed to the operating condition classifier to divide the deep process feature space, e.g. Support Vector Machine (SVM), Logistic Regression (LR), k-Nearest Neighbors (k-NN), Decision Tree (DT), and Random Forest (RF). In this study, a sequential-manner feature space dividing approach is proposed. At first a rough division of the feature space is done using a k-Dimensional Tree (KD-Tree) [6,37]. Then a LR classifier is adopted for finer divisions [15]. After the feature space is divided, the partitions of the deep process feature space are ’decoded’ to the subspaces of the comprehensive state space. The steps for subdividing the process state are briefly introduced in Section 4.2.1 and 4.2.2.

## 4.2.1. Rough division using a KD-Tree

A KD-Tree is a space partitioning algorithm for a k-dimensional space. It is a variant of a binary tree with each node being kdimensional. It splits the space into two half-spaces at each level iteratively until a desired number of partitions is achieved. The detailed steps are as follows (Fig. 8):

![](images/b84f60b5bd5653463704cca5b73a2f0e2a61119eba0c2675b6f86001149a2355.jpg)  
Fig. 8. Rough division using a KD-Tree.

Step 1: Determine the desired numbers of partitions $N _ { P } .$

Step 2: Calculate the required number of features and the number of splits. If $2 ^ { N - 1 } < \bar { N _ { P } } \leqslant 2 ^ { N } , ~ N ~ \ge ~ 1 ,$ , then N features and $N _ { S }$ number of splits are required.

$$
N _ {S} = (2 ^ {N - 1} - 1) + (N _ {P} - 2 ^ {N - 1}) = N _ {P} - 1\tag{9}
$$

Step 3: Normalize the feature data. Calculate the variances of the features, and rank the features according to the magnitude of their variances. Let j = 1.

Step 4: For the jth feature, find its median value, for the partition ${ \sf S } _ { ( \mathrm { j - 1 } ) \mathrm { q } } , q = 1 \mathrm { t o } 2 ^ { ( j - 1 ) }$ , generate a splitting hyperplane which is perpendicular to the corresponding axis and taking the median value on the axis of the jth feature in the feature space. Rank the resulting new half-spaces $\mathsf { S } _ { \mathrm { j } ( 2 \mathsf { q } - 1 ) } , \mathsf { S } _ { \mathrm { j } ( 2 \mathsf { q } ) }$ according to the magnitude of the variance of the ( j + 1)th feature.

Step $5 \colon \mathrm { I f } \ j < N - 1$ , then let $j = j + 1$ , and repeat Step 4 until $j = N - 1$

Step 6: $\begin{array} { r l r } { \mathrm { I f } } & { { } 2 ^ { j } } & { < } & { N _ { P } , } \end{array}$ then from q = 1 to $q = 2 ^ { j }$ , divide the partition $\mathsf { S } _ { j q }$ using the Nth feature until 2q + $( 2 ^ { j } - q ) = N _ { P }$ . The original feature space S is partitioned into $\{ S _ { \mathrm { N 1 } } , S _ { \mathrm { N 2 } } , \cdot \cdot \cdot ~ , S _ { \mathrm { N ( 2 q ) } } , S _ { \mathrm { j ( q + 1 ) } } , \cdot \cdot \cdot ~ , S _ { \mathrm { j ( 2 j ) } } \}$ (denoted as $\{ \mathsf { S } ^ { ( 1 ) } , \mathsf { S } ^ { ( 2 ) } , \cdots , \mathsf { S } ^ { ( \mathsf { N } _ { \mathsf { P } } ) } \}$ hereafter).

Remark 5. The feature variance is used as criterion when doing rough division. A large feature variance indicates a wide distribution of the data points on that feature dimension, which has a higher dividing eficiency compared with other features with smaller variances.

Remark 6. The variance and covariance of z determines the first and second order properties of the deep process features [23], so that the statistical features of z can also be used for process monitoring and fault detection.

## 4.2.2. Fine division based on LR

The KD-Tree based rough division process provides an initial solution for the LR-based fine division. LR classifier estimates and discriminates among the probabilities that an observation belongs to different classes:

$$
P (y | \mathbf {z}) = \frac {1}{1 + e ^ {- \sigma (\mathbf {z})}}\tag{10}
$$

$$
\sigma (\mathbf {z}) = \beta_ {0} + \sum_ {i = 1} ^ {N} \beta_ {i} z _ {i}\tag{11}
$$

where $P ( y | \mathbf { z } )$ is the probability that input z belongs to the class with label y, e.g. if $P ( y | \mathbf { z } ) > 0 . 5 .$ i.e. $\sigma ( { \pmb z } ) > 0 ,$ then z belongs to class $y _ { \cdot } \sigma ( \mathbf { z } ) = 0$ is the decision boundary. $\pmb { \beta } = [ \beta _ { 0 } , \beta _ { 1 } , \dots , \beta _ { N } ]$ are the coeficients to be learned for each class. The detailed steps are as follows:

Step 1: Let $r = 1$

Step 2: For partition $S ^ { ( r ) } ,$ , train the coeficient set $\beta ^ { ( r ) }$ by minimizing the regression error:

$$
\begin{array}{l} \ell_ {\mathrm{LR}} ^ {(r)} (\boldsymbol {\beta} ^ {(r)}) = \sum_ {j = 1} ^ {M} [ \frac {1}{1 + \mathrm{e} ^ {- \sigma^ {(r)} (\mathbf {z} _ {j})}} - y ^ {(j)} ] ^ {2} \\ = \sum_ {j = 1} ^ {M} [ \frac {1}{1 + \mathrm{e} ^ {- [ \beta_ {0} ^ {(r)} + \sum_ {i = 1} ^ {N} \beta_ {i} ^ {(r)} z _ {i} ^ {(j)} ]}} - y ^ {(j)} ] ^ {2} \end{array}\tag{12}
$$

where $y ^ { ( j ) }$ is the class label of the jth sample obtained by rough division.

Step 3: $\mathrm { I f } \ r < N _ { P } , r = r + 1$ , repeat Step2.

Step 4: The partition with boundary $\sigma ^ { ( r ) } ( { \bf z } ) = 0$ is the rth subspace of the entire feature space.

Remark 7: The partitioning of the deep process feature space can be transformed to the partitioning of the original CSS. The borders of the subspaces are not time-invariant. When a suficient amount of new data samples appear, the process of roughly and finely dividing the feature space, should be repeated to update the partition.

## 4.3. A hybrid modeling framework

This section presents the hybrid modeling framework in the context of CSS. To start with, two typical methods to describe process dynamics, i.e. state-space based first principle modeling (SS-FPM) and machine learning based input/output modeling (ML-IOM), are revisited.

## 4.3.1. First principle based state-space modeling

First principle models that are based on the inherent physical or chemical laws of a process, e.g. mass and energy balances, reaction kinetics, hydrodynamics, and thermodynamics, can often be described in state-space form. State-space models consist of a set of input, output and state variables that are related by first-order differential or difference equations. The state-space approach is powerful as it provides a unifying framework for representing many classes of system equations including stochastic, nonlinear and time-varying multivariable systems [8].

The model parameters in such state-space models are typically physical coeficients or constants [23]. Examples of such models include a continuously-stirred tank heater model [33], a flotation cell model [5], and a grinding circuit model [19]. However, this modeling approach requires significant process knowledge, which is not always available [17]. As a result not all the factors that influence the process are captured in the model.

![](images/59b0effc020c7ebf7e2fe3df2b9444eb1796d8bdd59e4f1bf060b2890d735d8e.jpg)  
Fig. 9. Differences and connections between state-space and data-driven modeling

## 4.3.2. Machine learning based input/output modeling

Machine learning based input/output models represent empirical mathematical or statistical correlations between input and output variables derived from plant data, especially for complex systems whose rigorous theoretical model involves a large number of DAEs and unknown parameters [13,28].

ML-IOM involves the use of machine learning approaches (unsupervised learning, supervised learning, semi-supervised learning, reinforcement learning) which project the real process to a data space or latent variable space and construct statistical models for correlation analysis, prediction, soft sensing, process monitoring, pattern recognition, and FDD [14]. The main actions involved in ML-IOM approaches include regression, classification, clustering, coordinate transformation, and statistical properties analysis. These methods make use of a large volume of production data that contain useful information and knowledge about the process dynamics and running status. In addition, by projection, the evolution of operating conditions can be visualized in the data space. However, the performance of data driven modeling relies on the quality and operation ranges covered by the training data.

## 4.3.3. Differences and connections

The two modeling methods described in Sections 4.3.1 and 4.3.2 use different sources of data, information and knowledge (Fig. 9), and have their pros and cons:

(i) The SS-FPM utilizes inherent physical or chemical laws such as the underlying physicochemical laws of a process. However, not all the factors that influence the process are captured in the model.

(ii) The ML-IOM utilizes routinely collected production data and machine learning algorithms. It can approximate the relationship between process inputs and outputs with very high accuracy. However, it does not capture the internal dynamics of a system.

However, both methods are informed by probability theory and statistics, e.g., conservation laws are derived from experimental data, and first principle models often contain empirical relationships. The combination of these two methods could increase the utilization rate of data, information and knowledge, thus leading to the discovery of more facets of the process dynamics.

## 4.3.4. CSS-based hybrid modeling

The system dynamics described by (4) can be decomposed as a combination of a nominal term and a deviation term

$$
\dot {\mathbf {x}} _ {0} = \mathbf {f} (\mathbf {x} _ {0}, \boldsymbol {\Theta}, \mathbf {u}) + \epsilon (\mathbf {x} _ {0}, \mathbf {x} _ {\mathrm{I}}, \mathbf {x} _ {\mathrm{R}})\tag{13}
$$

where f(·) is the nominal term or the first principle model containing the main reactions (white model),  represents the model parameters which take different values under different configurations of the inlet and reaction conditions, which can be identified using historical operation data. $\pmb { \epsilon } ( \cdot ) = \mathbf { g } ( \cdot ) - \mathbf { f } ( \cdot )$ is the deviation between the nominal model and the real dynamics.

![](images/983fc7a101db4fd88a12bf856ecf2bd518ca4fb50cc6e660b76375a17ac02f9d.jpg)  
Fig. 10. Dynamic modeling framework in the context of CSS

This formulation keeps the nominal dynamics of the main reactions in a state-space form, and can account for the intricate interactions by introducing $\epsilon ( \mathbf { \theta } \cdot \mathbf { \theta } )$ which then needs to be determined using data analytics (grey or black model), as shown in Fig. 10.

Considering each operating condition, the process dynamics can be realized as following:

$$
\mathbf {x} _ {0} (t _ {f}) = \mathbf {x} _ {0} ^ {\mathrm{nominal}} (t _ {f}) + \mathbf {x} _ {0} ^ {\mathrm{deviation}} (t _ {f})\tag{14}
$$

where

$$
\mathbf {x} _ {0} ^ {\text { nominal }} (t _ {f}) = \mathbf {x} _ {0} ^ {\text { nominal }} (t _ {0}) + \int_ {t _ {0}} ^ {t _ {f}} \mathbf {f} (\mathbf {x} _ {0}, \Theta) d t\tag{15}
$$

$$
\mathbf {x} _ {0} ^ {\text { deviation }} (t _ {f}) = \mathbf {x} _ {0} ^ {\text { deviation }} (t _ {0}) + \int_ {t _ {0}} ^ {t _ {f}} \boldsymbol {\epsilon} (\mathbf {x} _ {0}, \mathbf {x} _ {\mathrm{I}}, \mathbf {x} _ {\mathrm{R}}) d t\tag{16}
$$

$\mathbf { x } _ { 0 } ^ { \mathrm { n o m i n a l } } ( t _ { f } )$ is the nominal term derived from the FPM of the main reactions $[ 9 , 2 6 ] . \ \mathbf { x } _ { 0 } ^ { \mathrm { d e v i a t i o n } } ( t _ { f } )$ is the deviation term with the output states $\mathbf { x } _ { 0 }$ , inlet conditions $\mathbf { x } _ { \mathrm { I } }$ and reaction conditions $\mathbf { x } _ { \mathrm { R } }$ as its inputs. Thus, the value of $\mathbf { x } _ { 0 }$ can be expressed as

$$
\mathbf {x} _ {0} (t _ {f}) = \mathbf {x} _ {0} (t _ {0}) + \int_ {t _ {0}} ^ {t _ {f}} [ \mathbf {f} (\mathbf {x} _ {0}, \Theta) + \boldsymbol {\epsilon} (\mathbf {x} _ {0}, \mathbf {x} _ {\mathrm{I}}, \mathbf {x} _ {\mathrm{R}}) ] d t\tag{17}
$$

or, more practically, in a nonlinear discrete form

$$
\begin{array}{r l} \mathbf {x} _ {0} (k + 1) = \mathbf {x} _ {0} (k) + \mathbf {f} _ {k} (\mathbf {x} _ {0} (k), \Theta (k)) \\ & + \boldsymbol {\epsilon} _ {k} (\mathbf {x} _ {0} (k), \mathbf {x} _ {\mathrm{I}} (k), \mathbf {x} _ {\mathrm{R}} (k)) \end{array}\tag{18}
$$

where k indicates the kth sampling instant, $\mathbf { f } _ { k } ( \mathbf { x } _ { 0 } ( k ) , \Theta ( k ) )$ and $\epsilon _ { k } ( \mathbf { x } _ { 0 } ( k ) , \mathbf { x } _ { \mathrm { I } } ( k ) , \mathbf { x } _ { \mathrm { R } } ( k ) )$ are discretized counterparts of $\begin{array} { r } { \int _ { t _ { k } } ^ { t _ { k + 1 } } \mathbf { f } ( \mathbf { x } _ { 0 } , \Theta ) ( } \end{array}$ dt and $\begin{array} { r } { \int _ { t _ { k } } ^ { t _ { k + 1 } } \epsilon ( \mathbf { x } _ { 0 } , \mathbf { x } _ { \mathrm { I } } , \mathbf { x } _ { \mathrm { R } } ) d t } \end{array}$ , respectively.

To make the model usable, the high-dimensional matrix $\epsilon _ { k } ( \mathbf { x } _ { 0 } ( k ) , \mathbf { x } _ { \mathrm { I } } ( k ) , \mathbf { x } _ { \mathrm { R } } ( k ) )$ is approximated in this study by the nonlinear regression form

$$
\boldsymbol {\epsilon} _ {k} = \mathbf {U} (\mathbf {z} (k)) = [ \mathrm{U} _ {1} (\mathbf {z} (k)) \quad \mathrm{U} _ {2} (\mathbf {z} (k)) \dots \mathrm{U} _ {\mathrm{N}} (\mathbf {z} (k)) ] ^ {\mathrm{T}}\tag{19}
$$

where z is the vector of low-dimensional deep process features extracted from the high-dimensional contributors, and $\mathbf { U } ( \cdot )$ is the regression function set of the deep process feature variables.

As the process evolves in the CSS, the model parameters are subject to the inlet conditions, reaction conditions and output states. However, these parameters are identified for each operating condition, not each point in the CSS. Consider the time-varying characteristic of the production environment, the probability information provided by the LR can be incorporated into the dynamic process model (Fig. 11).

$$
\mathbf {x} _ {0} (k + 1) = \mathbf {x} _ {0} (k) + \sum_ {i = 1} ^ {N _ {p}} P (i | \mathbf {z}) [ \mathbf {f} _ {k} ^ {(i)} (\mathbf {x} _ {0} (k), \Theta (k)) + \mathbf {U} ^ {(i)} (\mathbf {z} (k)) ]\tag{20}
$$

where $\mathbf { f } _ { k } ^ { ( i ) } ( \mathbf { x } _ { 0 } ( k ) , \Theta ( k ) )$ and $\mathbf { U } ^ { ( i ) } ( \pmb { z } ( k ) )$ indicate the corresponding terms under operating condition i, i.e. sub-models. Compared with some typical hybrid modeling approaches [1,27], the proposed CSS-based modeling approach considers the multimodality and updates the weights of each sub-model to enable the dynamic modeling of the process.

## 5. Case study

In this section the CSS descriptive framework is used to develop a process model for a cobalt removal process.

## 5.1. Process description

Cobalt removal is performed in a unit process in zinc hydrometallurgy. Its main function is to decrease the cobalt ion concentration in the zinc sulfate solution such that the cobalt ion concentration after removal is lower than a safety limit. The cobalt removal process usually consists of four continuously stirred tank reactors and a thickener (Fig. 12). Zinc dust is added into each reactor to replace the cobalt ions. Steam and spent acid are used to maintain the required reaction conditions (e.g., temperature 65 \~ 75<sup>◦</sup>C and pH 4.5 \~ 5.5). Catalyst is provided to promote and accelerate cobalt removal. After retention in the consecutive reactors, the solution enters the thickener where solid-liquid separation takes place. The resultant solids, which can be used as crystal nuclei for cementation, is deposited and recycled to promote impurity removal. The overflow is further filtered and delivered to a subsequent process.

![](images/571602a1d35699a62d2bff48526d34a11c48892bf1836112d14ff14f1fd3a895.jpg)  
Fig. 11. Hybrid modeling framework in the context of CSS.

![](images/2b9304432391ad6f61147748e857d4eb48abbd2ed467f1e464393ebd8705c72c.jpg)  
Fig. 12. A cobalt removal process

The principle of cobalt removal is to use zinc dust to replace the cobalt ion impurities. However, the replacement is not trivial. As shown in Fig. 13, cobalt removal is catalysed by electro-positive metal salts, e.g., arsenic trioxide, which reacts with residual copper ions and zinc dust (Reaction 1), and provides reaction surfaces for cobalt removal (Reaction 2). Besides the main reactions, the residual copper ions and hydrogen ions in the solution also react with zinc dust (Reactions 3 and 4). Reactions 3 and 4 affect cobalt removal by competing with cobalt ions for zinc dust and changing the oxidation–reduction atmosphere. If zinc dust is overdosed, Reactions 5 and 6 will take place. These two reactions are undesirable, and produce elemental arsenic and highly toxic arsine gas. Besides, other elements and compounds in the inlet solution can result in other side reactions. The flow rates of the inlet solution, spent acid, arsenic trioxide, underflow, steam and zinc dust can affect the process dynamics. In addition, the interactions with its preceding and subsequent unit processes can also affect cobalt removal [32].

## 5.2. Experiment setup

In the case study, 1050 data samples were collected over a 4 months period from an industrial zinc smelting plant (The data is scaled and desensitized for confidentiality reasons). The data were divided into two parts: 1000 samples for training and 50 samples for verification. Each data sample contains the values of the output states, inlet conditions and reaction conditions. The time span of the data samples ensure that a majority of the operating conditions are covered. In addition, with this amount of data samples, deep process feature extraction approaches can be applied. Knowledge of the reaction kinetics and the production process is obtained from studying the process mechanisms, data analysis, as well as discussions with experienced operators and technicians.

![](images/1889737beb90d9d2abc7d846d4090532faadff1e8c1019e63c6b23f4ac1cb855.jpg)  
Fig. 13. Main and side reactions in a cobalt removal process.

Table 2  
Table 1  
Variables selected for the CSS.

<table><tr><td>State type</td><td>Variables</td></tr><tr><td>Output states</td><td> $c_1, c_2, c_3, c_4$ </td></tr><tr><td>Reaction conditions</td><td> $v_1, v_2, v_3, v_4, f_{As}, f_{Acid}$ </td></tr><tr><td>Inlet conditions</td><td> $f_0, m_{\text{pH}}, c_0, c_{\text{Cu}}, c_{\text{Zn}}, c_{\text{Cd}}, c_{\text{Ni}}, c_{\text{As}}, c_{\text{Sb}}, c_{\text{Ge}}, c_{\text{Fe}}, c_{\text{Ca}}$ </td></tr></table>

The aim of this case study is twofold, i.e. to test if the working conditions can be recognized and how well the model performs under the CSS framework. The steps of the case study are listed as follows:

(i) Extract the low dimensional deep process features from the high dimensional process variables.

(ii) Divide the deep process feature space into subspaces with each of them indicating different operating conditions.

(iii) Test if the working conditions can be recognized and how well the model performs under the CSS framework.

(iv) Build the nominal kinetic model, and formulate the process dynamics using the CSS-based modeling framework.

(v) Compare the performance of the nominal kinetic model and the CSS-based modeling approach.

## 5.3. Results

According to the reaction mechanism, the output states were selected as the cobalt ion outlet concentrations of the four reactors. The variables indicating reaction conditions were chosen as the oxidation reduction potential (ORP) of each reactor, flow rate of the arsenic trioxide and spent acid. ORP presents the overall oxidation–reduction atmosphere inside the reactor, while the flow rate of the arsenic trioxide and spent acid determine the catalytic condition and the pH of the solution respectively. The temperature affects the reaction rate. However, the solution temperature in the reactor rarely change, and was therefore omitted from the simulation. The flow rate, pH, and concentrations of the metallic ions $( \mathrm { e . g . , C o ^ { 2 + } , C u ^ { 2 + } , Z n ^ { 2 + } }$ , etc.) in the feed solution demonstrate the inlet conditions (Table 1). To sum up, there are 4 output states, 12 variables indicating inlet conditions and 6 for the reaction conditions. Therefore, the dimension of output states, inlet conditions and reaction conditions is 22 in total. The correlations among the condition variables and output states are obtained by Pearson correlation analysis, as shown in Table 2.

Based on a process analysis, the number of operating conditions was determined as $N _ { P } = 7$ . Three deep process features are therefore required to divide the deep process feature space into 7 subspaces. To extract the 3 deep process features, two auto encoders were used to form an SAE. The number of latent variables in the two auto encoders are 18 and 3 respectively. The activation function adopted is the sigmoid function:

$$
f _ {\mathrm{sig}} (x) = \frac {1}{1 + \mathrm{e} ^ {- x}}\tag{21}
$$

Then, rough division was conducted based on the variance of the 3 deep process features, and the resulting subspaces were denoted as $\mathsf { S } _ { 3 1 }$ (Working condition type 1), $\mathsf { S } _ { 3 2 }$ (Type 2), $\mathsf { S } _ { 3 3 }$ (Type 3), $\mathsf { S } _ { 3 4 }$ (Type 4), $S _ { 3 5 }$ (Type 5), $S _ { 3 6 }$ (Type 6), $S _ { 2 4 }$ (Type 7) (Figs. 8, 14). Then, fine division based on LR followed. The LR provides probability class information, which is useful especially for the junctions of two or more subspaces (Fig. 14). Fig. 14 shows the location and the probability that the operating point belongs to different operating conditions of the 50 test samples in the deep process feature space. The 30th sample is on the border of operating conditions S35 and S36. The probability that it belongs to these two operating conditions are 85% and 15%, respectively. The corresponding partitioning of the CSS is shown in Fig. 15.

The correlations between the condition variables and output states.

<table><tr><td>Variables</td><td> $c_1$ </td><td> $c_2$ </td><td> $c_3$ </td><td> $c_4$ </td></tr><tr><td> $v_1$ </td><td>0.2143</td><td>0.1476</td><td>0.1572</td><td>0.3085</td></tr><tr><td> $v_2$ </td><td>0.0580</td><td>0.0419</td><td>0.0206</td><td>-0.0863</td></tr><tr><td> $v_3$ </td><td>-0.1221</td><td>-0.1567</td><td>-0.2421</td><td>-0.1648</td></tr><tr><td> $v_4$ </td><td>0.1511</td><td>0.0595</td><td>0.0494</td><td>0.0129</td></tr><tr><td> $f_{\text{As}}$ </td><td>0.3712</td><td>0.3427</td><td>0.3759</td><td>0.3942</td></tr><tr><td> $f_{\text{Acid}}$ </td><td>0.0128</td><td>-0.0555</td><td>0.0420</td><td>0.1625</td></tr><tr><td> $f_0$ </td><td>0.3467</td><td>0.2015</td><td>0.2529</td><td>0.2664</td></tr><tr><td> $m_{\text{pH}}$ </td><td>0.1390</td><td>0.0683</td><td>0.0469</td><td>0.0354</td></tr><tr><td> $c_0$ </td><td>0.5910</td><td>0.3890</td><td>0.3085</td><td>0.4931</td></tr><tr><td> $c_{\text{Cu}}$ </td><td>0.1819</td><td>0.2220</td><td>0.2317</td><td>0.3073</td></tr><tr><td> $c_{\text{Zn}}$ </td><td>0.0481</td><td>0.0938</td><td>0.0696</td><td>0.0772</td></tr><tr><td> $c_{\text{Cd}}$ </td><td>-0.2684</td><td>-0.1675</td><td>-0.1493</td><td>-0.1023</td></tr><tr><td> $c_{\text{Ni}}$ </td><td>0.0182</td><td>-0.0515</td><td>-0.0377</td><td>-0.0005</td></tr><tr><td> $c_{\text{As}}$ </td><td>0.1889</td><td>0.1144</td><td>0.1432</td><td>0.2587</td></tr><tr><td> $c_{\text{Sb}}$ </td><td>-0.0666</td><td>-0.0958</td><td>-0.0936</td><td>-0.0247</td></tr><tr><td> $c_{\text{Ge}}$ </td><td>-0.1512</td><td>-0.2045</td><td>-0.1987</td><td>-0.1254</td></tr><tr><td> $c_{\text{Fe}}$ </td><td>0.1264</td><td>0.0946</td><td>0.0443</td><td>-0.0388</td></tr><tr><td> $c_{\text{Ca}}$ </td><td>0.1299</td><td>0.0040</td><td>-0.0385</td><td>-0.0737</td></tr></table>

![](images/5c19a071cfa9643b9e855e23728e157314efdc0ad5c0c5ced729ae667c7870e0.jpg)  
Fig. 14. Partitioning of the deep process feature space.

The nominal part of the model is derived using mass balances and first order reaction kinetics. Assuming ideal mixing conditions, i.e. the reaction rate and solution temperature are uniform throughout the reactor, then:

$$
\frac {d c _ {i}}{d t} = \frac {f _ {i - 1}}{V} c _ {i - 1} - \frac {f _ {i}}{V} c _ {i} - r _ {i} c _ {i}\tag{22}
$$

where $c _ { i } , r _ { i }$ and $f _ { i }$ are the outlet cobalt ion concentration, reaction rate, and outlet flow rate of the ith reactor respectively, with i = $1 , 2 , \cdots , N .$ V is the volume of the reactor. $c _ { 0 }$ and $f _ { 0 }$ are the cobalt ion concentration and the flow rate of the inlet solution of the first reactor, respectively.

The reaction rate term r can be derived using the Arrhenius equation and electrode kinetics [4,36]:

$$
r = A _ {0} \beta g _ {s} e ^ {- \frac {E _ {e} + 2 \gamma F (\nu - \nu_ {e q})}{R T}}\tag{23}
$$

where v is the Oxidation Reduction Potential (ORP). It represents the oxidation-reduction ability of the solution and is controlled by the addition of zinc dust, u. The physical meanings of the parameters in the process model are listed in Table 3. To increase the identification accuracy, model parameters $A _ { 0 } , \beta$ and $g _ { s }$ were combined as a new parameter $A _ { \beta }$ in the identification of model parameters [9]. The value of model parameters under different operating conditions are shown in Table 4. The performance of the nominal kinetic model is shown in Fig. 16 and Table 5. For comparison purpose, the performance of pure data-driven model (PDDM) is also included. The PDDM is a combination of SAE and radius basis neural network (RBFNN). The input of the RBFNN is the deep process features extracted using SAE. The output of the RBFNN is the estimated value of cobalt ion concentrations of each reactor. As indicated by the results, compared with kinetic model, PDDM is less sensitive to the change of operating conditions.

Table 5  
Table 3  
Physical meaning of the parameters in the process model.

<table><tr><td>Parameter (unit)</td><td>Physical meaning</td></tr><tr><td> $A_0(s^{-1})$ </td><td>frequency factor of the reaction</td></tr><tr><td> $\beta$ </td><td>reaction surface area available on a unit area of the crystal nucleus</td></tr><tr><td> $g_s$ </td><td>weight of crystal nucleus per unit volume of the reactor</td></tr><tr><td> $E_e(J \cdot mol^{-1})$ </td><td>standard activation energy of the reaction</td></tr><tr><td> $\gamma$ </td><td>variation factor between the electrode potential and the cathode activation energy</td></tr><tr><td> $v_{eq}(V)$ </td><td>equilibrium potential of the cathode reaction</td></tr><tr><td> $v(V)$ </td><td>oxidation reduction potential of the solution in the reactor</td></tr><tr><td> $T(K)$ </td><td>reaction temperature</td></tr><tr><td> $F(C \cdot mol^{-1})$ </td><td>Faraday constant,  $F = 96485$ </td></tr><tr><td> $R(J \cdot mol^{-1}K^{-1})$ </td><td>ideal gas constant,  $R = 8.314$ </td></tr></table>

![](images/c6f09221ea4f3c1f3bc869203053ebb7deb6c8a6db227705056da3c17d875f4f.jpg)  
Table 4  
Fig. 15. Partitioning of the CSS.

To account for the unknown dynamics, a deviation term $c _ { i } ^ { \mathrm { d e v i a t i o n } }$ is added to the nominal model, then under certain operating condition, for each reactor:

$$
c _ {i} (t) = c _ {i} (0) + \int_ {0} ^ {t} \left(\frac {f _ {i - 1}}{V} c _ {i - 1} - \frac {f _ {i}}{V} c _ {i} - r _ {i} c _ {i}\right) d \tau + c _ {i} ^ {\text { deviation }}\tag{24}
$$

The deviation term is a neural network model with its inputs being the deep process features. Its output is the estimated deviation between the nominal model and practical system dynamics. Eq. (24) is a basic model of process dynamics. Its model parameters take different values under different working conditions. The formulation of the overall system dynamics was the same as Eq. (20). The performance of the integrated model is shown in Fig. 17. The performance comparison between the integrated model and the nominal model is given in Table 5, and Fig. 18 using a four plot figure [35], respectively. The proposed modeling approach has higher accuracy on average, and its performance is more stable. This indicates by combining reaction kinetics, production data and information about the current operating conditions, the system dynamics can be described more comprehensively. This is due to the higher information utilization rate under the CSS framework.

Values of model parameters under different operating conditions

<table><tr><td>Working condition</td><td>Reactor</td><td> $A_{\beta}$ </td><td> $E_e$ </td><td> $\gamma$ </td><td> $v_{eq}$ </td></tr><tr><td rowspan="4">S31</td><td>Reactor 1</td><td>6981106</td><td>72932</td><td>0.618</td><td>-0.300</td></tr><tr><td>Reactor 2</td><td>7286393</td><td>79847</td><td>0.625</td><td>-0.332</td></tr><tr><td>Reactor 3</td><td>8073669</td><td>79998</td><td>0.706</td><td>-0.373</td></tr><tr><td>Reactor 4</td><td>8983144</td><td>71945</td><td>0.664</td><td>-0.436</td></tr><tr><td rowspan="4">S32</td><td>Reactor 1</td><td>7362697</td><td>72747</td><td>0.600</td><td>-0.300</td></tr><tr><td>Reactor 2</td><td>7832393</td><td>79968</td><td>0.689</td><td>-0.367</td></tr><tr><td>Reactor 3</td><td>8668537</td><td>77202</td><td>0.671</td><td>-0.392</td></tr><tr><td>Reactor 4</td><td>6562077</td><td>72362</td><td>0.603</td><td>-0.404</td></tr><tr><td rowspan="4">S33</td><td>Reactor 1</td><td>5984033</td><td>72400</td><td>0.600</td><td>-0.300</td></tr><tr><td>Reactor 2</td><td>8365405</td><td>78822</td><td>0.601</td><td>-0.348</td></tr><tr><td>Reactor 3</td><td>5308410</td><td>79043</td><td>0.608</td><td>-0.348</td></tr><tr><td>Reactor 4</td><td>8186106</td><td>72876</td><td>0.750</td><td>-0.463</td></tr><tr><td rowspan="4">S34</td><td>Reactor 1</td><td>7638676</td><td>73391</td><td>0.600</td><td>-0.300</td></tr><tr><td>Reactor 2</td><td>8858660</td><td>77157</td><td>0.797</td><td>-0.427</td></tr><tr><td>Reactor 3</td><td>6213033</td><td>77216</td><td>0.791</td><td>-0.429</td></tr><tr><td>Reactor 4</td><td>5874675</td><td>78829</td><td>0.621</td><td>-0.368</td></tr><tr><td rowspan="4">S35</td><td>Reactor 1</td><td>5957729</td><td>72265</td><td>0.600</td><td>-0.300</td></tr><tr><td>Reactor 2</td><td>7970208</td><td>79557</td><td>0.630</td><td>-0.352</td></tr><tr><td>Reactor 3</td><td>8510620</td><td>77102</td><td>0.734</td><td>-0.417</td></tr><tr><td>Reactor 4</td><td>8817141</td><td>73734</td><td>0.695</td><td>-0.440</td></tr><tr><td rowspan="4">S36</td><td>Reactor 1</td><td>5349874</td><td>72240</td><td>0.600</td><td>-0.300</td></tr><tr><td>Reactor 2</td><td>6516429</td><td>74700</td><td>0.616</td><td>-0.385</td></tr><tr><td>Reactor 3</td><td>5255406</td><td>78698</td><td>0.772</td><td>-0.414</td></tr><tr><td>Reactor 4</td><td>7004144</td><td>71521</td><td>0.704</td><td>-0.457</td></tr><tr><td rowspan="4">S31</td><td>Reactor 1</td><td>5311958</td><td>72624</td><td>0.600</td><td>-0.300</td></tr><tr><td>Reactor 2</td><td>5438328</td><td>77854</td><td>0.671</td><td>-0.382</td></tr><tr><td>Reactor 3</td><td>8817776</td><td>70562</td><td>0.617</td><td>-0.440</td></tr><tr><td>Reactor 4</td><td>8756903</td><td>79996</td><td>0.640</td><td>-0.379</td></tr></table>

Performance comparison between nominal kinetic model, pure data-driven model and CSS modeling framework using average relative error (ARE) and root mean square error (RMSE)

<table><tr><td>Performance measure</td><td>Reactor 1</td><td>Reactor 2</td><td>Reactor 3</td><td>Reactor 4</td></tr><tr><td>ARE of FPM</td><td>15.19%</td><td>15.55%</td><td>16.14%</td><td>17.21%</td></tr><tr><td>ARE of PDDM</td><td>8.38%</td><td>10.70%</td><td>6.67%</td><td>8.32%</td></tr><tr><td>ARE of CSS model</td><td>5.7%</td><td>4.02%</td><td>3.33%</td><td>7.90%</td></tr><tr><td>RMSE of FPM</td><td>1.3967</td><td>0.6132</td><td>0.2506</td><td>0.0706</td></tr><tr><td>RMSE of PDDM</td><td>0.6296</td><td>0.3877</td><td>0.1069</td><td>0.0397</td></tr><tr><td>RMSE of CSS model</td><td>0.4166</td><td>0.1568</td><td>0.0679</td><td>0.0314</td></tr></table>

## 5.4. Discussion

It can be observed from the above results that the CSS framework provides:

(i) Improved modeling accuracy: By using the CSS modeling approach, the ARE of the four reactors are lower than 10%, which is satisfactory for an industrial application. The norm of the Fisher information matrix of the kinetic model and the integrated model have orders of magnitude of $1 0 ^ { 4 }$ and $1 0 ^ { 1 5 }$ , respectively. This indicates that more information is utilized in the integrated modeling approach. Therefore, higher model accuracy is obtained as an outcome.

Reactor 3  
![](images/e03e6ad9d9d83e66f29331f6b76d01049609de68a9d578bf34e35cd80998233c.jpg)

![](images/39c84a7fc468ef2e4e1dddd27d983b85dba2aa47eb7c0acc8270be1ea41ee0da.jpg)

![](images/d878ccb564e2d502e1890ed7cae49b6d157f97a5c90b56c472df889d4bae38f0.jpg)

![](images/aa2c2cfef84ecd0398f2d6e62cbe1a4788ee5b24d0a1e8d5ea1e4323e00f88c3.jpg)  
Fig. 16. Performance of the nominal kinetic model and pure data-driven model.

![](images/e6f9337f5c3787d907e0ae80b18f986414ed347bdffe2f6078b8c0ec07cda0d6.jpg)

![](images/5d564c590cc6ed200161c63f0a4236c06ffd78b26fcd919037158ac63d9b589e.jpg)

![](images/085b96967d038808cb561e0b027ddd6e3280cac0d87cd6f57498932ae2769a99.jpg)

![](images/e78ba3e55bb7aeedc66d33f81581383c8cdc316f6b4ff2909c306f616a44940e.jpg)  
Fig. 17. Performance of the integrated model under the CSS modeling framework

![](images/76b2a0192a48663396649badf401300e144401e4cfbb63ebabd2c0820a81696b.jpg)

![](images/1aaef8adad0145b48e2e0891c7ca4599a936915e541f338e0549c9bc4dd9301d.jpg)

![](images/920b1af79a969a162cb9b3dbd54a12cce980b111972b78df3fd85590647690b7.jpg)

![](images/f1d470228997693d4098ffbadfc18e612c7844160a311e3e32773edd60273ff5.jpg)  
Fig. 18. Four plot performance comparison between the integrated model and the nominal kinetic model.

![](images/b2fea9f1f3a5f7815153da70d19eb16e90aa1ea3d06a8e705ab2f21e33def403.jpg)  
Fig. 19. Trajectory of the test samples in the CSS.

(ii) Intuitive process monitoring: CSS provides a visual abstraction of the physical process. The location and the evolution of the process in the CSS can be monitored intuitively. Fig. 19 shows the evolution trajectory of the 50 test samples. Different marker colors indicate different operating conditions. The variation of model parameters (see Table 4) and operating conditions along the evolution trajectory can be observed from Fig. 19.

(iii) Comprehensive descriptive capability: The CSS descriptive framework covers the essential influence factors of the process. These factors are organized to form a digitized container of the physical process which can accommodate various attributes of an operating point under many different operating conditions, including e.g., model parameters, controller gain, status, and suggested operation.

## 6. Conclusions

In this study, a comprehensive state space descriptive system and a corresponding hybrid first principles/machine learning modeling framework were proposed. As a wide variety of process control applications is model-based, the CSS descriptive system can better support process control in terms of better understanding of process dynamics. In addition, its intuitive nature provides a platform for process monitoring. It provides a comprehensive yet ’not overly complex’ way to describe the dynamics of a process that enables the digitalization and visualization of a physical process. The CSS descriptive system is an open framework, which can serve as a container for the fusion of data, information and knowledge from various sources. Due to the above characteristics of CSS, it can also serve as a ’digital twin’ of a physical process. The physical process and its CSS counterpart can therefore act as a ’cyber-physical component’ in a smart factory. The results presented in this paper are preliminary and mainly considers process modeling in the context of CSS. For future extension, the following should be considered:

<sub>•</sub> The dividing of the deep process feature space is based on the variance of deep process features. This dividing approach is easy to implement. However, the physical meaning of the subspaces are not clear. Therefore, generating subspaces with concrete physical meanings can provide more useful information to the operators.

Transforming the system dynamics into a linear time varying (LTV) format by using a U-model [38] approach to describe the nominal term and deviation term respectively. This would resulted in an LTV model of the physical process, which can bridge the gap between the nonlinear system description and linear controller design approaches with well established properties.

Integrating modeling, control, and estimation in an interactive and systematic framework to gradually increase the understanding of the system dynamics and achieve intelligent autonomous control.

Solutions to the above problems and the methods proposed in this study can enrich the theoretical foundations for the smart and optimal manufacturing in the process industries.

## Declaration of Competing Interest

We certify that all co-authors have seen and agree with the contents of the submission “A comprehensive hybrid first principles/machine learning modeling framework for complex industrial processes” by Bei Sun et al., which is submitted to Journal of Process Control as a research article, and there is no financial interest to report. We also certify that the submission is our original work and is not under review at any other publications.

## CRediT authorship contribution statement

Bei Sun: Conceptualization, Methodology, Software. Chunhua Yang: Resources, Project administration. Yalin Wang: Investigation. Weihua Gui: Supervision. Ian Craig: Conceptualization, Writing - review & editing. Laurentz Olivier: Writing - review & editing.

## Acknowledgments

This work was supported by the Projects of International Cooperation and Exchanges NSFC (grant no. 61860206014), the National Natural Science Foundation of China (grant nos. 61603418, 61973321, 61703441), the 111 Project (B17048), the Natural Science Foundation of Hunan Province (grant no. 2019JJ50823), the Foundation for Innovative Research Groups of the National Natural Science Foundation of China (grant no. 61621062), and the Major Program of the National Natural Science Foundation of China (grant no. 61590921).

## Supplementary material

Supplementary material associated with this article can be found, in the online version, at 10.1016/j.jprocont.2019.11.012.

## References

[1] J. Abonyi, J. Madar, F. Szeifert, Combining first principles models and neural networks for generic model control, in: R. Roy, M. Köppen, S. Ovaska, T. Fu ruhashi, F. Hoffmann (Eds.), Soft computing and industry, Springer, London, 2002, pp. 111–122.

[2] P.K. Akkisetty, U. Lee, G.V. Reklaitis, V. Venkatasubramanian, Population balance model-based hybrid neural network for a pharmaceutical milling process, . of Pharmaceutical Innov. 5 (4) (2010) 161–168.

[3] M. Alavi, H. Jazayeri-Rad, R.M. Behbahani, Optimizing the feed conditions in a dimethyl ether production process to maximize methanol conversion using a hybrid first principle neural network approach, Chem. Eng. Commun. 201 (5) (2014) 650–673.

[4] L.I. Antropov, Theoretical Electrochemistry, Mir Publishers, Moscow, 1977.

[5] O.A. Bascur, Modelling and computer control of a flotation cell, University of Utah, Salt Lake City, Utah, US, 1982 Ph.D. thesis..

[6] J.L. Bentley, Multidimensional binary search trees used for associative searching, Commun. ACM 18 (9) (1975) 509–517.

[7] J. Bierbrauer, Introduction to Coding Theory, Chapman and Hall/CRC, New York, 2016.

[8] W.L. Brogan, Modern Control Theory, 3rd, Prentice Hall, Upper Saddle River, NJ, 1991.

[9] G. Buzzi-Ferraris, F. Manenti, Kinetic models analysis, Chem. Eng. Sci. 64 (5) (2009) 1061–1074.

[10] D. Chaffart, L.A. Ricardez-Sandoval, Optimization and control of a thin film growth process: a hybrid first principles/artificial neural network based multiscale modelling approach, Comput. Chem. Eng. 119 (2018) 465–479.

[11] Y. Chen, H. Jiang, C. Li, X. Jia, P. Ghamisi, Deep feature extraction and classification of hyperspectral images based on convolutional neural networks, IEEE Trans. Geosci. Remote Sens. 54 (10) (2016) 6232–6251.

[12] I. Craig, C. Aldrich, R. Braatz, F. Cuzzola, E. Domlan, S. Engell, J. Hahn, V. Havlena, A. Horch, B. Huang, et al., Control in the process industries, in: T. Samad, A.M. Annaswamy (Eds.), The impact of control technology, IEEE Control Systems Society, 2011.

[13] Z. Ge, Review on data-driven modeling and monitoring for plant-wide industrial processes, Chemom. Intell. Lab. Syst. 171 (2017) 16–25.

[14] Z. Ge, Z. Song, S.X. Ding, B. Huang, Data mining and analytics in the process industry: the role of machine learning, IEEE Access 5 (2017) 20590–20616.

[15] F.E. Harrell, Regression Modeling Strategies, Springer, New York, 2015

[16] D. Hodouin, S.-L. Jämsä-Jounela, M. Carvalho, L. Bergh, State of the art and challenges in mineral processing control, Control Eng. Pract. 9 (9) (2001) 995–1005.

[17] P. Kadlec, B. Gabrys, S. Strandt, Data-driven soft sensors in the process industry, Comput. Chem. Eng. 33 (4) (2009) 795–814.

[18] J. Keskitalo, K. Leiviskä, Artificial neural network ensembles in hybrid modelling of activated sludge plant, in: P. Angelov, K.T. Atanassov, L. Doukovska, M. Hadjiski, V. Jotsov, J. Kacprzyk, N. Kasabov, S. Sotirov, E. Szmidt, S. Zadrozny´ (Eds.) Intelligent systems' 2014, Springer Cham 2015 pp. 683–694

[19] J.D. Le Roux, I.K. Craig, D. Hulbert, A. Hinde, Analysis and validation of a run-of-mine ore grinding mill circuit model for process control, Minerals Eng. 43 (2013) 121–134.

[20] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, Nature 521 (7553) (2015) 436–444.

[21] J.H. Lee, J.M. Lee, Progress and challenges in control of chemical processes, Annu. Rev. Chem. Biomol. Eng. 5 (2014) 383–404.

[22] J.M. Lee, J.H. Lee, Approximate dynamic programming-based approaches for input–output data-driven control of nonlinear processes, Automatica 41 (7) (2005) 1281–1288.

[23] L. Ljung, System Identification: Theory for The User, 2nd, Prentice Hall, Upper Saddle River, NJ, 1999.

[24] L. Ljung, Perspectives on system identification, Annu. Rev. Control 34 (1) (2010) 1–12.

[25] L.E. Olivier, I.K. Craig, Model-plant mismatch detection and model update for a run-of-mine ore milling circuit under model predictive control, J. Process Control 23 (2) (2013) 100–107

[26] C.C. Pantelides, J. Renfro, The online use of first-principles models in process operations: review, current status and future needs, Comput. Chem. Eng. 51 (2013) 136–148.

[27] D.C. Psichogios, L.H. Ungar, A hybrid neural network-first principles approach to process modeling, AIChE J. 38 (10) (1992) 1499–1511.

[28] D.E. Seborg, D.A. Mellichamp, T.F. Edgar, F.J. Doyle III, Process Dynamics and Control, 4th, John Wiley & Sons, Hoboken, NJ, 2016.

[29] C. Shang, F. Yang, D. Huang, W. Lyu, Data-driven soft sensor development based on deep learning technique, J. Process Control 24 (3) (2014) 223–233.

[30] M. von Stosch, J. Zhang, M. Willis, Hybrid neural network modelling for process monitoring and control, in: A. Basile, M. Alavi, S. Curcio (Eds.), Artificial neural networks in chemical engineering, Nova Science Publishers, Inc., New York, 2017, pp. 205–228.

[31] B. Sun, W. Gui, Y. Wang, C. Yang, Intelligent optimal setting control of a cobalt removal process, J. Process Control 24 (5) (2014) 586–599.

[32] B. Sun, W. Gui, T. Wu, Y. Wang, C. Yang, An integrated prediction model of cobalt ion concentration based on oxidation–reduction potential, Hydrometallurgy 140 (2013) 102–110.

[33] N.F. Thornhill, S.C. Patwardhan, S.L. Shah, A continuous stirred tank heater simulation model with applications, J. Process Control 18 (3–4) (2008) 347–360.

[34] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P.-A. Manzagol, Stacked denoising autoencoders: learning useful representations in a deep network with a loca denoising criterion, J. Mach. Learn. Res. 11 (Dec) (2010) 3371–3408.

[35] X. Yuan, B. Huang, Y. Wang, C. Yang, W. Gui, Deep learning based feature representation and its application for soft sensor modeling with variable-wise weighted SAE, IEEE Trans. Ind. Inform. 14 (2018) 3235–3243.

[36] Q. Zha, Introduction to the Kinetics of Electrode Processes, Science Press, Beijing, 2002.

[37] J. Zhang, H. Guo, F. Hong, X. Yuan, T. Peterka, Dynamic load balancing based on constrained K-D tree decomposition for parallel particle tracing, IEEE Trans. Visual. Comput. Graphics 24 (1) (2018) 954–963.

[38] Q. Zhu, W. Zhang, J. Zhang, B. Sun, U-Neural network-enhanced control of nonlinear dynamic systems, Neurocomputing 352 (2019) 12–21.