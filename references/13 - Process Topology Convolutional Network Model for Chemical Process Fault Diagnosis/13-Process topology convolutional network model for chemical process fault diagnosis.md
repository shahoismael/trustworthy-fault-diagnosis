# Process topology convolutional network model for chemical process fault diagnosis

![](images/19f95432cba164a248063798bb3d031986d7cc4871009f08cc204ef9fe8afa72.jpg)

Deyang Wu<sup>a</sup>, Jinsong Zhao<sup>a,b,∗</sup>

<sup>a</sup> State Key Laboratory of Chemical Engineering, Department of Chemical Engineering, Tsinghua University, Beijing, China <sup>b</sup> Beijing Key Laboratory of Industrial Big Data System and Application, Tsinghua University, Beijing, China

## a r t i c l e i n f o

Article history: Received 8 January 2021 Received in revised form 29 March 2021 Accepted 30 March 2021 Available online 8 April 2021

Keywords: Fault diagnosis Chemical process Process topology convolutional network Explainable deep learning Process safety

## a b s t r a c t

There always exists potential safety risk in chemical processes. Abnormalities or faults of the processes can lead to severe accidents with unexpected loss of life and property. Early and accurate fault detection and diagnosis (FDD) is essential to prevent these accidents. Many data-driven FDD models have been developed to identify process faults. However, most of the models are black-box models with poor explainability. In this paper, a process topology convolutional network (PTCN) model is proposed for fault diagnosis of complex chemical processes. Experiments on the benchmark Tennessee Eastman process showed that PTCN improved the fault diagnosis accuracy with simpler network structure and less reliance on the amount of training data and computation resources. In the meantime, the model building process becomes much more rational and the model itself is much more understandable.

© 2021 Institution of Chemical Engineers. Published by Elsevier B.V. All rights reserved.

## 1. Introduction

Modern chemical processes are generally complex and integrated systems with many different unit operations. Under distributed control systems (DCS), chemical processes can be operated steadily in most of the time. But there always exists potential risk that abnormal events or process faults may happen, which is beyond the control capability of DCS and requires the intervention of operators. They can lead to severe accidents with unexpected loss of life and property if no actions are taken to bring the processes back to normal operation.

Abnormal situation management (ASM) of chemical processes is critical to handle these abnormal events or faults and prevent processes from further accidents. (Venkatasubramanian et al., 2003) defined ASM as a key component of supervisory control. ASM involves the timely detection of an abnormal event, diagnosing causal origins and then taking appropriate decisions and actions to bring the process back to a normal, safe, operating state. (Arunthavanathan et al., 2020a) proposed a framework from safety perspectives to analyze the interconnections between fault detection and diagnosis (FDD), risk assessment (RA) and ASM. In this framework, FDD is the initial step to identify the possible hazard.

RA uses failure models, accident models and risk models to provide the basis of decision making for ASM. With the feedback information, ASM changes the decision to control the operation and brings the processes back to normal state. (Khan et al., 2015) mentioned that integration of dynamic fault detection and diagnosis with risk assessment had significantly improved safety in process facilities. According to (Dai et al., 2016), process monitoring including fault detection and diagnosis should be effectively integrated to achieve a reliable, robust, scalable ASM platform for smart chemical process operations. As can be seen, FDD is highly valued in ASM and the field of process safety.

Although DCS makes chemical processes highly automatic nowadays, the automation of ASM has not been realized yet (Shu et al., 2016). It heavily relies on human operators’ monitoring. Additionally, real-time FDD is getting more and more difficult for the increasingly complex chemical processes (Venkatasubramanian et al., 2003). As the chemical industries evolve towards smart manufacturing, it’s necessary to build intelligent FDD system for safer and more automatic process operations.

From the perspectives of process safety and risk assessment, many researchers have paid attention to the applications of FDD, especially in the field of dynamic risk assessment these years. (Zadakbar et al., 2013) extended principle component analysis (PCA) based FDD to a risk based FDD framework targeting the safety issues of a process system. The combination of PCA and a quantitative operational risk assessment model made this method robust to false alarms. (Madakyaru et al., 2017) proposed a statistical approach that exploited the advantages of multiscale partial least squares (MSPLS) models and generalized likelihood ratio (GLR) tests for fault detection in processes. (Amin et al., 2020) proposed a novel methodology for dynamic risk analysis, integrating the multivariate data-based process monitoring and logical dynamic failure prediction model. This framework combined the naïve Bayes classifier, Bayesian network, and event tree analysis. It provided robust performance on the application of a binary distillation column and the RT 580 experimental setup in four fault scenarios.

From the perspective of modeling, many FDD models have been proposed by researchers over the past few decades. (Venkatasubramanian et al., 2003) classifies FDD models into three categories: quantitative model-based, qualitative model-based and process history based. Process history-based models are further classified into qualitative and quantitative models. The latter is commonly termed as data-driven models. Easy access to big data and the rapid development of computing power have made datadriven models get excellent performance in terms of accuracy. Additionally, data-driven models can be easily constructed without first principles and expert knowledge.

Basic data-driven FDD methods are mainly statistical methods, such as principle component analysis (PCA), independent component analysis (ICA), fisher discriminant analysis (FDA), partial least squares (PLS), canonical variate analysis (CVA), subspace aided approach (SAP), etc. and their variants. There’re some machine learning FDD models that treat fault diagnosis as a classification problem, such as support vector machine (SVM), artificial neural network (ANN), artificial immune system (AIS) (Dai and Zhao, 2011), etc. In order to compare different models’ performance, the Tennessee Eastman (TE) process (Downs and Vogel, 1993) was gradually exploited by researchers as a benchmark. A comparison study on basic data-driven fault diagnosis methods has been conducted by (Yin et al., 2012).

With the recent development of artificial intelligence and big data technologies. deep learning-based FDD has attracted a lot of interests from researchers. Deep learning is a general term for a large class of algorithms utilizing multi-layer neural network models. Alternate linear transformations and nonlinear activation functions give deep learning models strong ability of nonlinear fitting. (Xie and Bai, 2015) proposed a hierarchical deep neural network (HDNN) based on deep belief network (DBN). HDNN reached an average fault classification accuracy of 80.5 % excluding faults 3, 9, 15 on the TE process. (Lv et al., 2016) used sparse auto encoder (SAE) and support vector machine (SVM) for fault detection and classification. (Zhang and Zhao, 2017) proposed a fault diagnosis model based on DBN. It utilized DBN sub-networks for features extraction and a global two-layer back-propagation network for fault classification. The model reached an average fault classification accuracy of 82.1 % for all the 20 fault classes of the TE process. (Wu and Zhao, 2018) treated two-dimension data matrices as images and proposed a deep convolutional neural network (DCNN) model. DCNN extracted features in spatial and temporal domains simultaneously. It reached an average fault classification accuracy of 88.2 % for all of the 20 fault classes. (Zhang et al., 2020a) utilized bidirectional recurrent neural network (BiRNN) to process time series from both positive and negative directions. The model reached an average fault classification accuracy of 92.7 % considering all the 20 fault classes. These’re all supervised deep learning-based approaches. For unsupervised deep learning, (Cheng et al., 2019) proposed a process monitoring method based on variational recurrent autoencoder with a new monitoring metric named negative variational score, proving that unsupervised deep learning methods can be adapted for complex process monitoring. (Zheng and Zhao, 2020) proposed an unsupervised data mining method that can make use of unlabeled historical data to deal with the problem of lacking labeled data in real situations.

Despite of the advantages mentioned above, data-driven fault diagnosis models still have room for improvement. Firstly, these models are mainly based on pure data and utilize no process knowledge. Without extra information from the process, pure data-driven models tend to have higher degree of freedom thus more model parameters. Overly complex models generally suffer from the problem of overfitting (Lever et al., 2016). The model building process is not rational without guidance from process knowledge. Secondly, these pure data-driven models lack explainability and are still black boxes to humans, which limits their application in real industrial processes. Thirdly, the training process and online application of these models are prone to consume more resources on computation.

As techniques in machine learning develop quickly these years, there is a consensus among many researchers that innovation is needed to integrate data analytics tools with fundamental knowledge to create robust and scalable solutions for industrial processes (Qin and Chiang, 2019). The current status of machine learning is more like alchemy, a collection of ad hoc methods. But this limitation could be offset by the use of first-principles knowledge, which can impose some rigor and discipline on purely data-driven models (Venkatasubramanian, 2019). (Bikmukhametov and Jäschke, 2020) proposed 5 methods to enhance accuracy and explainability of data-driven models through combining machine learning and process engineering physics. However, these methods mainly made use of feature engineering or the mismatch between the developed models and first-principle models. Process knowledge was not utilized in the building phase of machine learning models.

There’re some hybrid approaches for FDD combining both model-based approaches and data-driven approaches. Among them, the Bayesian network (BN) has been used widely for FDD and root cause analysis of process systems in the past few decades. (Yu et al., 2015) proposed a dynamic operational risk management framework integrating modified independent component analysis and a BN model. (Gharahbagheri et al., 2017) proposed a methodology for operators to diagnose the root cause of abnormal. It was based on kernel principal component analysis, and process knowledge was combined through Bayesian belief network. (Lou et al., 2020) utilized a condition Gaussian network (CGN), a special form of BN, with an adaptive threshold scheme to detect and diagnose the process faults. For cognitive modeling, (Chen et al., 2014) employed deterministic reservoir models to fit the multiple-input and multiple-output signals in the TE process, which firstly investigated the TE process in a cognitive way. (Arunthavanathan et al., 2020b) focused on the integration of unsupervised learning and cognitive modeling to detect and diagnose unknown fault conditions. However, hybrid approaches generally take more effort in the phrase of modeling and can’t be trained end-to-end easily.

The graph neural network (GNN) is a type of deep learning model that can process data in the graph domain or non-Euclidean space, which has drawn much attention of researchers recently. In mathematics, graph is a kind of data structure consisting of nodes and edges. These nodes are connected with edges to describe their relationships. Differences of data in Euclidean space and graph domain are showed in Fig. 1. GNN has a wide range of applications because data in the graph domain are far more common than structured data in the real world. Among GNNs, graph convolutional network (GCN) (Kipf and Welling, 2017) has been studied widely in the field of deep learning. GCN defines graph convolutions by information propagation. It requires input of a general data matrix with stacked node features and an extra adjacent matrix that describes nodes connectivity in the graph. Within a graph convolutional layer in GCN, a node in the graph will aggregate feature information from its neighborhood nodes (Wu et al., 2020). Stacked graph convolutional layers enable GCN to propagate information of nodes along the edges in the graph as predefined. This helps GCN learn about the relationships among different nodes under clear guidance. The predefined graph is an entry where we can integrate extra knowledge or information from the processes. However, to the authors knowledge, GCN’s applications in chemical processes have been rarely reported.

![](images/0083cc0f3f72f6509cf47ce718ba1dedb5a8cf26d9a5b03d27b8373766c984f8.jpg)

![](images/b2d29d3ef93b6c4f17abc21885202ec5aaf858d7c92a42360835274ae4fe969b.jpg)  
(a)  
(b)  
Fig. 1. Data and convolution operations in (a) Euclidean space and (b) graph domain

The major contributions of this work are summarized as followed:

(1) A fault diagnosis model named process topology convolutional network (PTCN) is proposed based on GCN. Detailed steps are given to describe how to construct graphs from chemical processes that are required by PTCN.

(2) The framework of fault diagnosis method based on PTCN is designed for practical application. Case study on the TE process showed that PTCN improved fault diagnosis performance and reduced the reliance on model parameters, training data and computation resources.

The rest of the paper is organized as followed: Section 2 introduces the brief history and the basic theory of graph convolutiona network. Section 3 introduces graph construction steps, PTCN model structure, data structure and the fault diagnosis method in detail. Section 4 is a case study on the TE process that introduces the dataset preparation and experiments result of the PTCN model. Finally, conclusions are drawn in Section 5.

## 2. Graph convolutional network

In recent years, convolutional neural networks (CNNs) have received considerable attention from researchers in the field of machine learning. CNNs have performed much better than many traditional algorithms in many tasks, such as image classification (Krizhevsky et al., 2017) and object detection (Redmon et al., 2016). However, CNNs are good at processing the data in the Euclidean space, such as images, but can hardly fulfil the tasks where the data are represented in the non-Euclidean space, such as graphs or networks (Wu et al., 2020). Fig. 1 shows the differences between the data and convolutional operations represented in the Euclidean space and graph domain.

In practical application, the semi-structured data represented as graph or network are more common. For instance, the relationships among the users in some social media platforms can be represented as a graph. In chemical researches, a molecular structural formula can also be regarded as a graph, where the nodes represent the atoms and the edges represent the chemical bonds. To process data in graph domain, graph neural networks (GNNs) were invented (Wu et al., 2020), which have attracted the interests of many researchers in the field of deep learning.

Table 1  
Commonly used notations.

<table><tr><td>Notations</td><td>Descriptions</td></tr><tr><td> $G$ </td><td>A graph</td></tr><tr><td> $V$ </td><td>The set of nodes in a graph</td></tr><tr><td> $\nu$ </td><td>A node  $\nu \in V$ </td></tr><tr><td> $E$ </td><td>The set of edges in a graph</td></tr><tr><td> $e_{ij}$ </td><td>An edge  $e_{ij} \in E$ </td></tr><tr><td> $\mathcal{N}(\nu)$ </td><td>The neighborhoods of node  $\nu$ </td></tr><tr><td> $n$ </td><td>The number of nodes in a graph</td></tr><tr><td> $m$ </td><td>The number of edges in a graph</td></tr><tr><td> $d$ </td><td>The dimension of a node feature vector</td></tr><tr><td> $b$ </td><td>The dimension of a hidden state vector</td></tr><tr><td> $c$ </td><td>The dimension of an edge feature vector</td></tr><tr><td> $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ </td><td>The graph adjacency matrix</td></tr><tr><td> $\boldsymbol{A}^T$ </td><td>The transpose of the matrix  $\boldsymbol{A}$ </td></tr><tr><td> $\boldsymbol{A}_i$ </td><td>The  $i^{th}$  row of the matrix  $\boldsymbol{A}$ </td></tr><tr><td> $A_{ij}$ </td><td>The element in the  $i^{th}$  row and  $j^{th}$  column of the matrix  $\boldsymbol{A}$ </td></tr><tr><td> $\boldsymbol{D} \in \mathbb{R}^{n \times n}$ </td><td>The degree matrix of  $\boldsymbol{A}$ .  $D_{ii} = \sum_{j=1}^{n} A_{ij}$ </td></tr><tr><td> $\boldsymbol{X} \in \mathbb{R}^{n \times d}$ </td><td>The node feature matrix</td></tr><tr><td> $\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$ </td><td>The node feature matrix at time step  $t$ </td></tr><tr><td> $\boldsymbol{x}_\nu \in \mathbb{R}^d$ </td><td>The node feature vector of node  $\nu$ </td></tr><tr><td> $\boldsymbol{x}_{(v,u)} \in \mathbb{R}^c$ </td><td>The edge feature vector of edge  $(v,u)$ </td></tr><tr><td> $x_i \in \mathbb{R}$ </td><td>The  $i^{th}$  element of vector  $\boldsymbol{x}$ </td></tr><tr><td> $\boldsymbol{H}^k \in \mathbb{R}^{n \times b}$ </td><td>The hidden state matrix in the  $k^{th}$  graph convolutional layer</td></tr><tr><td> $\boldsymbol{h}_\nu \in \mathbb{R}^b$ </td><td>The hidden state vector of node  $\nu$ </td></tr></table>

## 2.1. The definition and notations of graph

A graph is a set of vertices (or nodes) and edges, and is applied as a special data structure in computing sciences that can represent objects and the connections among them. In this paper, the commonly used notations are illustrated in Table 1 unless otherwise specified. And the formal definitions of graphs and directed graphs are in accordance with (Wu et al., 2020).

## 2.1.1. Definition of graph

A graph is represented as $G = ( V , E )$ where V is the set of vertices or nodes (we will use nodes throughout the paper) and E is the set of edges. Let $\nu _ { i } ~ \in$ V to denote a node and $e _ { i j } = \left( \nu _ { i } , \nu _ { j } \right) \in \mathcal { I }$ E to denote an edge pointing from v to $\nu _ { i } .$ . The neighborhood of a node v is defined as $\mathcal { N } ( \nu ) = \left\{ u \in V \mid ( \nu , u ) \in E \right\}$ . The adjacency matrix is a $n \times n$ matrix with $A _ { i j } = 1 \mathrm { i f } e _ { i j } \in E$ and $A _ { i j } = 0 { \mathrm { i f } } e _ { i j } \not \in E . A $ graph may have node attributes $\pmb { X } ,$ where $\pmb { X } \in \mathbb { R } ^ { n \times d }$ is a node feature matrix with $\pmb { x } _ { \nu } \in \mathbb { R } ^ { d }$ <sup>X X</sup>representing the feature vector of a node v. Meanwhile, a <sup>x</sup>graph may have edge attributes $\pmb { X } ^ { e }$ , where $\pmb { X } ^ { e } \in \mathbb { R } ^ { m \times c }$ is an edge feature matrix with $\pmb { x } _ { ( \nu , u ) } \in \mathbb { R } ^ { c }$ <sup>X X</sup>representing the feature vector of an edge (v, u).

## 2.1.2. Definition of directed graph

A directed graph is a graph with all edges directed from one node to another. An undirected graph is considered as a special case of directed graphs where there is a pair of edges with inverse directions if two nodes are connected. A graph is undirected if and only if the adjacency matrix is symmetric.

## 2.2. A brief history of graph neural network

Early study on neural networks applied to graphs can date back to (Sperduti and Starita, 1997). After first proposed by (Gori et al., 2005), graph neural network (GNN)<sup>1</sup> was further developed by (Scarselli et al., 2009), which was extended from neural networks for processing the data represented in graph domains. GNNs for supervised learning can be mainly categorized into recursive graph neural networks (RecGNNs) and convolutional graph neural networks (ConvGNNs).

## 2.2.1. Recursive graph neural network

Many graph neural network models proposed in early time were extended from recursive neural network (RNN). A typical case, GNN (Scarselli et al., 2009), is an extension of RNN with random walk model, and is based on an information diffusion mechanism. In GNN, the nodes of a graph will exchange information through the connections between them and update their hidden states until a stable equilibrium is reached, and a unique stable equilibrium can always be guaranteed by Banach’s fixed point theorem (Khamsi and Kirk, 2001). The hidden state corresponding to a node v can be updated by

$$
\boldsymbol {h} _ {\nu} ^ {t} = \sum_ {u \in N (\nu)} f _ {\boldsymbol {w}} (\boldsymbol {x} _ {\nu}, \boldsymbol {x} _ {(\nu , u)}, \boldsymbol {h} _ {u} ^ {t - 1}, \boldsymbol {x} _ {u})\tag{1}
$$

Where $\pmb { h } _ { \nu } ^ { t }$ is the hidden state of the node v in the $t ^ { t h }$ iteration. The <sup>h</sup>hidden state of a node can be regarded as a high-level node representation that combines the information of neighbor nodes and the graph’s topological structure. The information diffusion mechanism in RecGNNs has made profound effect on the development of ConvGNNs.

## 2.2.2. Convolutional graph neural network

The convolutional graph neural network (ConvGNN) is another main category of graph neural network, and draws far more attention from researchers than RecGNNs. Convolutional graph neural networks introduce convolutional operation from Euclidean space to data in graph domains, which can hardly be processed by CNNs directly. Based on the domain that convolutional operations are performed on, ConvGNNs can be categorized further into spectralbased ConvGNNs and spatial-based ConvGNNs.

## 2.2.3. Spectral-based ConvGNN

Spectral-based ConvGNNs utilize the knowledge from graph signal processing. For a graph, the normalized Laplacian matrix is

$$
\pmb {L} = \pmb {I} _ {n} - \pmb {D} ^ {\left(- \frac {1}{2}\right)} \pmb {A D} ^ {\left(- \frac {1}{2}\right)}\tag{2}
$$

$$
\boldsymbol {D} = \sum_ {\boldsymbol {j} = 1} ^ {\boldsymbol {n}} A _ {i j}\tag{3}
$$

And is called degree matrix of the adjacency matrix . ${ \cal I } _ { n }$ is a n - <sup>D A I</sup>order identity matrix. When we perform eigenvalue decomposition to the normalized Laplacian matrix

$$
\boldsymbol {L} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {T}\tag{4}
$$

We can get an orthonormal matrix $\pmb { U } = [ \pmb { u } _ { 1 } , \pmb { u } _ { 2 } , . . . , \pmb { u } _ { n } ] \in \mathbb { R } ^ { n \times n }$ where ${ \pmb u } _ { i }$ <sup>U u u u</sup>is the eigenvector arranged by the value of eigenvalues, and $\pmb { A }$ <sup>u</sup>stands for the diagonal matrix where $\varLambda _ { i i } = \lambda _ { i } .$ . If we have a <sup>-</sup>graph signal $\pmb { x } = [ x _ { 1 } , x _ { 2 } , . . . x _ { n } ] ^ { T } \in \mathbb { R } ^ { n }$ where $x _ { i }$ is the signal value of <sup>x</sup>node i, the graph Fourier transformation and its inverse transformation can be defined as

$$
\pmb {x} ^ {\prime} = \mathcal {F} (\pmb {x}) = \pmb {U} ^ {T} \pmb {x}\tag{5}
$$

$$
\pmb {x} = \mathcal {F} ^ {- 1} (\pmb {x} ^ {\prime}) = \pmb {U x} ^ {\prime}\tag{6}
$$

This can be derived from $\pmb { U } \pmb { U } ^ { T } = \pmb { I }$ since is an orthonormal matrix. And a graph convolution of graph signal with filter can be defined as (Henaff et al., 2015)

$$
\chi_ {* G} \mathbf {g} = \mathbf {U} (\mathbf {U} ^ {T} \mathbf {x} \bigodot \mathbf {U} ^ {T} \mathbf {g})\tag{7}
$$

$^ { * } G$ stands for the convolutional operator performed on graphs and $\check { \odot }$ means element-wise product.

Since ${ \pmb U } ^ { T } { \pmb x } , { \pmb U } ^ { T } { \pmb g } \in \mathbb { R } ^ { n }$ , the spectral-based graph convolutional operation can be simplified as

$$
\chi_ {* G} \mathbf {g} _ {\theta} = \mathbf {U} (\mathbf {g} _ {\theta} \mathbf {U} ^ {T} \mathbf {x})\tag{8}
$$

Where $\pmb { \mathrm { g } } _ { \pmb { \theta } } = d i a g \left( \pmb { U } ^ { T } \pmb { g } \right)$ is a parametric filter. And nearly all the <sup>g U g</sup>spectral-based ConvGNNs identify with this formulation (Wu et al., 2020).

The efficiency of spectral-based ConvGNNs is affected by the eigenvalue decomposition of the normalized Laplacian matrix . (Defferrard et $\mathrm { a l . }$ <sup>L</sup>, 2016) proposed Chebyshev Spectral CNN (Cheb-Net) to solve the efficiency problem through approximating the filter ${ \pmb g } _ { \pmb \theta }$ by Chebyshev polynomials of . And (Kipf and Welling, <sup>g -</sup>2017) utilized a first-order approximation of ChebNet to propose Graph Convolutional Network (GCN). For a graph G of n nodes with d -dimensional features, the graph convolutional operation of GCN on a graph signal $\pmb { X } \in \mathbb { R } ^ { n \times d }$ is

$$
\mathbf {Z} = \overset {\sim} {\mathbf {D}} \left( \begin{array}{c} - \frac {1}{2} \\ \mathbf {A D} \end{array} \right) \overset {\sim} {\underset {} {\sim}} \left( \begin{array}{c} - \frac {1}{2} \\ \mathbf {X} \boldsymbol {\Theta} \end{array} \right)
$$

$$
\tilde {\pmb {A}} = \pmb {A} + \pmb {I} _ {n}\tag{9}
$$

(10)

$$
\tilde {D} _ {i i} = \sum_ {\boldsymbol {j} = 1} ^ {\boldsymbol {n}} \tilde {A} _ {i j}\tag{11}
$$

$\pmb { A }$ is the adjacent matrix of graph G added with self-loops, and $\tilde { \bf \delta p }$ is the degree matrix of $\check { A } . \Theta \in \mathbb { R } ^ { d \times b }$ is a parametric matrix that <sup>A </sup>embeds d -dimensional node features to a b -dimensional space.

## 2.2.4. Spatial-based ConvGNN

Different from spectral-based ConvGNNs, spatial-based ConvGNNs inherit more from the information diffusion mechanism of RecGNNs in early time. Spatial-based ConvGNNs propagate information through the connections, and aggregates the information of neighbor nodes and corresponding edges to a certain node to update the hidden state of it. However, spatial-based ConvGNNs do not require a unique stable equilibrium of the graph under strict constraint conditions. They define convolutional layers for information passing and aggregation operations, then use graph readout technologies to obtain node-level or graph-level representation of the information. From Fig. 2 we can see that the spatial-based convolutional operation on a directed graph updates a node’s hidden state with its neighbor nodes’ representations. This is similar to convolutional operation on images in CNNs if we consider the image as a special type of graph with nodes fixed in the grid. So spatial-based ConvGNNs can be regarded as a generalization of traditional CNNs.

(Gilmer et al., 2017) proposed message passing neural network (MPNN) as a framework for spatial-based ConvGNNs. It describes the process of message passing and aggregation on a graph. In MPNN, the spatial-based convolutional operation on graphs can be formulated as

$$
\boldsymbol {m} _ {\nu} ^ {t + 1} = \sum_ {u \in N (\nu)} M _ {t} (\boldsymbol {h} _ {\nu} ^ {t}, \boldsymbol {h} _ {u} ^ {t}, \boldsymbol {x} _ {(\nu , u)})\tag{12}
$$

$$
\boldsymbol {h} _ {\nu} ^ {t + 1} = U _ {t} (\boldsymbol {h} _ {\nu} ^ {t}, \boldsymbol {m} _ {\nu} ^ {t + 1})\tag{13}
$$

The function M is called message functions for passing the hidden states of neighbor nodes and the features of connected edges to node v. $U _ { t }$ is a node update function for updating the hidden state of node v, and t means the $t ^ { t h }$ iteration or the $t ^ { \bar { t } h }$ graph convolutional operation. After T convolutional operations, readout function

![](images/cdb3e1b11a1ef76522a59b03ab42301fa06be6afc6401ef3bdff3c127cbff2f2.jpg)  
Fig. 2. Spatial-based convolutional operation on a directed graph.

$R \big ( \{ h _ { \nu } ^ { T } | \nu \in G \} \big )$ can generate the representation of the graph from al <sup>h</sup>the nodes, or generate node-level representations.

If we look into the graph convolutional operation of GCN we can find that it’s in accordance with the MPNN framework. The equation Eq. (9) can be expressed as

$$
\mathbf {Z} _ {i} = \sum_ {j} \frac {1}{\sqrt {\tilde {D} _ {i i}} \cdot \sqrt {\tilde {D} _ {j j}}} \tilde {A} _ {i j} \mathbf {X} _ {j} \boldsymbol {\Theta} = \sum_ {j \in (i) \cup N (i)} \frac {1}{\sqrt {\tilde {D} _ {i i}} \cdot \sqrt {\tilde {D} _ {j j}}} \mathbf {X} _ {j} \boldsymbol {\Theta}\tag{14}
$$

$$
\pmb {h} _ {i} = U (\pmb {Z} _ {i}) = U (M _ {\pmb {\Theta}} (\frac {1}{\sqrt {\widetilde {D} _ {i i}} \cdot \sqrt {\widetilde {D} _ {i i}}} \pmb {X} _ {i}, \sum_ {j \in \mathcal {N} (i)} \frac {1}{\sqrt {\widetilde {D} _ {i i}} \cdot \sqrt {\widetilde {D} _ {j j}}} \pmb {X} _ {j}))\tag{15}
$$

The graph convolutional operation of GCN has two phases of message passing and aggregation like general spatial-based ConvGNNs. From this perspective, GCN can be regarded as a connection between spectral-based and spatial-based ConvGNNs, when the spectral filter is approximated to first-order (Zhang et al., 2020b). The spectral-based and spatial-based graph convolutions are equivalent under specific conditions, but spatial-based ones can benefit from the research progresses of traditional CNNs. This maybe the reason why there’re more researches studying spatial-based ConvGNNs than spectral-based ConvGNNs.

## 3. PTCN based fault diagnosis method

## 3.1. Graph and chemical processes topology

In mathematics, topology describes the way how nodes and edges are arranged within a network. In this paper, chemical process topology is used to describe the way how different parts of the process are arranged. In chemical processes, unit operations and streams are physically connected with pipes. Control loops link up the process variables and manipulated variables with sensors, controllers, actuators, etc. Process topology contains a large amount of knowledge about the relationships between different variables, since they are physical connected with pipes or signal transmission wires. To utilize the information of process topology, the process should be transformed to a graph first.

To construct a graph from a chemical process, the P&ID is required to figure out different unit operations, streams and corresponding measurements. Control loops should also be studied to confirm process variables, manipulated variables, sensors, controllers, actuators, etc. Detailed steps are listed below to describe how to construct graphs from chemical processes. For better understanding, a subgraph of the graph constructed from Tennessee Eastman process is showed in Fig. 3 as an example. And the complete graph is showed as Fig. 7 in Section 4.2.

Step 1: Figure out all the unit operations and streams that we concern about. Create nodes for every unit operation and stream. These nodes are called unit operation nodes and stream nodes respectively.

Step 2: Figure out all the corresponding measurements of these unit operations and streams in Step 1. Create nodes for every measurement. These nodes are called measurement nodes.

Step 3: Create directed edges pointing from every unit operation node or stream node to corresponding measurement node.

Step 4: For every stream node, create a directed edge pointing from the upstream unit operation node to the stream node, and a directed edge pointing from the stream node to the downstream unit operation node.

Step 5: Figure out all the process variables and manipulated variables of concerned control loops. Process variables are included in the measurement nodes in Step 2. Create nodes for every manipulated variable. These nodes are called manipulated variable node. For every control loop, create an abstract node called controller node.

Step 6: In a control loop, create a directed edge pointing from the corresponding measurement node to the controller node, and a directed edge pointing from the controller node to the corresponding manipulated variable node.

Step 7: For simplicity of the graph, delete the stream nodes that have no corresponding measurements, and then connect the upstream and downstream unit operation nodes with a directed edge directly.

The graph describes the physical connectivity among the streams, unit operations and control loops. From the perspective of information transferring, different nodes generate or carry information about the process. The directed edges describe the directions of information transferring or massage passing. Compared to complicated first-principle mathematical models, the graph only concerns about the connections between different parts of the process, which can qualitatively indicate the relationships among different variables.

Following the steps above, the graph constructed from the process will be cyclic graph considering recycle streams and control loops in the process. This is reminiscent of other fault diagnosis methods based on graphs such as the Bayesian network. But different from PTCN, the Bayesian network is based on acyclic graphs. The graph in PTCN is not used for inferring the probability of the presence of a variable, so it can be a cyclic graph. In another word, PTCN is not a probabilistic graphical model.

The graph in PTCN is mainly used to describe the physical relationships between different variables. It doesn’t intend to model the exact dependency between variables. Though the physical connections can’t accurately model the dependency between variables, the dependency must be based on physical links. The exact relationships between variables are decided by learnable network parameters, which are learnt from data following the paradigm of deep learning. Thus, the graph in PTCN works as loose constraints for the relationships between variables.

## 3.2. Data structures and data preprocessing

Generally, in data-driven fault diagnosis methods, a data sample in the training dataset is consist of n observation variables and w observations for each variable, which forms a data matrix $\pmb { X } _ { t } \in \mathbb { R } ^ { n \times w }$ at time step t like Eq. (16).

![](images/ddf6cdabd48429d98ec50a51443628baf58cc7afe9ebf655661e4ac48364ba1e.jpg)  
Fig. 3. Subgraph of the graph constructed from Tennessee Eastman process.

$$
\boldsymbol {X} _ {t} = \left[ \begin{array}{c c c c} x _ {1, t} & x _ {1, (t - 1)} & \dots & x _ {1, (t - w + 1)} \\ x _ {2, t} & x _ {2, (t - 1)} & \dots & x _ {2, (t - w + 1)} \\ \vdots & \vdots & \ddots & \vdots \\ x _ {n, t} & x _ {n, (t - 1)} & \dots & x _ {n, (t - w + 1)} \end{array} \right]\tag{16}
$$

Every column of the matrix is a n -dimensional observation vector at a certain time. Every row is a w -dimensional time serial of a certain observation variable. While columns in $\pmb { X } _ { t }$ should be arranged in chronological order to exploit serial correlation, the rows do not have to follow a certain sequence. In another word, all the observation variables are equally treated in because no <sup>X</sup>prior knowledge or constraint conditions are given. However, all the observation variables are collected from a certain chemical process, so they are highly correlated under the constraints of material balance, heat balance, energy balance and decentralized control system. And the process knowledge can hardly be involved in a single $\pmb { X } _ { t } .$ Thus, the data-driven model may suffer from overfitting <sup>X</sup>of the data without enough prior knowledge.

To integrate process knowledge about the correlation among the observation variables, the data matrix $\pmb { X } _ { t }$ is input with an adjacency matrix $\pmb { A } ^ { ' }$ of the graph G constructed from the concerned <sup>A</sup>chemical process topology. According to Definition 1 in Section 2.1, the adjacency matrix <sup>’</sup> can be calculated as Eq. (17).

$$
A _ {i j} ^ {\prime} = \left\{ \begin{array}{l} 1, i f e _ {i j} \in E \\ 0, i f e _ {i j} \notin E \end{array} \right.\tag{17}
$$

E is the set of edges in the graph G.

It needs to be pointed out carefully that the best directions for message passing are not necessary the same as the physical transferring directions of flows and control signals. They can also be the reverse directions. Experiments in Section 4.6 indicates that the reverse directions are more appropriate for message passing. Thus, the actual adjacency matrix input with data matrix $\pmb { X } _ { t }$ is, the adjacency matrix of the reverse graph of G. And in mathematics, can be easily deviated from $\pmb { A } ^ { ' }$ as Eq. (18).

$$
\boldsymbol {A} = \left(\boldsymbol {A} ^ {\prime}\right) ^ {T}\tag{18}
$$

After the graph is prepared, the historical data of the process should be collected for further training and testing a fault diagnosis model. Historical data of measurements and manipulated variables form time series and are sampled at a certain frequency. The number of measurements and manipulated variables is m. For every variable, all the original historical data $\pmb { X } _ { i } ^ { \dagger }$ should first be normalized with the sample mean $\overline { { x } } _ { i }$ and sample standard deviation $s _ { i }$ calculated using data in normal state as Eq. (19), which is also known as standard score normalization.

$$
\pmb {X} _ {i} = \frac {\pmb {X} _ {i} ^ {'} - \overline {{x}} _ {i}}{s _ {i}} (i = 1, 2, \dots m)\tag{19}
$$

The normalized data $\pmb { X } _ { i }$ are then sliced with a time window. At <sup>X</sup>a certain time t, the data slice for every variable will be a vector $\pmb { X } _ { i , t } \in \mathbb { R } ^ { 1 \times w }$ of w observations including current sample and $w - 1$ <sup>X</sup>samples before t.

$$
\mathbf {X} _ {i, t} = \left[ \begin{array}{c c c c} x _ {i, t} & x _ {i, (t - 1)} & \dots & x _ {i, (t - w + 1)} \end{array} \right] (i = 1, 2, \ldots m)\tag{20}
$$

It should be noted that the n nodes in the graph include not only measurements and manipulated variables that are observed in the process, but also some abstract nodes such as stream nodes, unit operation nodes and controller nodes. These abstract nodes can’t be observed directly so have no corresponding historical data. For computational requirements, the time series or node features of these abstract nodes should be initialized properly. The appropriate initialization technologies will be further discussed in Section 4.5. Then all the time series $\pmb { X } _ { i , t } ( i = 1 , 2 , . . . n )$ ) are stacked into the data matrix $\pmb { X } _ { t } \in \mathbb { R } ^ { n \times w }$ . The data matrix Xt can also be regarded as the <sup>X X</sup>node feature matrix, and every row is a node feature vector, which is also a time serial of w observations.

Labels are essential in supervised learning. Since the fault diagnosis is treated as a multi-classification problem, it has to be figured out when a fault happened in the historical data and what the fault type is. If there exists c types of faults, every data matrix $\pmb { X } _ { t }$ will be assigned with a corresponding label $\pmb { y } \in \mathbb { Z }$ <sup>X</sup>varying from 0 to c that indicates the fault type of the process at time t. And $\pmb { y } = 0$ means the process is in normal state, while other integers stand for different types of faults. As a result, a data sample input to the model will be $\pmb { X } _ { t } , \pmb { A } , \pmb { y } .$ . After all the historical data are sliced and arranged into data samples, the whole dataset $\left\{ { \pmb X } _ { t } , { \pmb A } , { \pmb y } \right\} _ { i = 1 } ^ { N }$ will be divided into training dataset and testing dataset with the ratio of 4:1 for further training and testing process.

## 3.3. PTCN model for fault diagnosis

The proposed model for fault diagnosis is named as process topology convolutional network (PTCN) in this paper, and the structure is showed as Fig. 4. The model name PTCN comes from that the graph convolutional operations are applied on the graph constructed from the chemical process topology. PTCN is mainly composed of stacked graph convolutional layers and a multi-layer perceptron classifier. A graph convolutional layer will include a complete graph convolutional operation as Eq. (9).

Within a graph convolutional layer, every node in the graph will aggregate feature information from its neighborhood nodes. When graph convolutions are applied to the chemical process topology, information will propagate among different parts of the process along the physical connections. This is how the knowledge of the chemical process topology can be utilized in the building of the model. Compared to pure data-driven models, the feature extraction of PTCN is more understandable and reasonable to humans, because the graph convolutions are based on the physics of the chemical process. Thus, PTCN is more of explainability with the guidance of the chemical process topology.

Within a training epoch, dataset $\left\{ { \pmb X } _ { t } , { \pmb A } , { \pmb y } \right\} ^ { N }$ is input to the network model. For every data sample, self-loops are first added to to get as Eq. (21) and corresponding degree matrix is calculated as Eq. (22).

$$
\tilde {\boldsymbol {A}} = \boldsymbol {A} + \boldsymbol {I} _ {n}\tag{21}
$$

$$
\widetilde {D} _ {i j} = \left\{ \begin{array}{l} \sum_ {k = 1} ^ {n} \widetilde {A} _ {k j}, i f i = j \\ 0, e l s e \end{array} \right.\tag{22}
$$

It’s worth mentioning that $\overset { \sim } { \pmb { A } }$ is not symmetric since G is a directed graph. The degree matrix $\pmb { D }$ is calculated as an out-degree <sup>D</sup>matrix, which counts how many nodes a node j is pointing to after self-loops are added.

The hidden states of nodes in the graph are initialized with the node feature matrix $\pmb { X } _ { t }$

$$
\pmb {H} ^ {0} = \pmb {X} _ {t}\tag{23}
$$

Every graph convolutional layer applies the graph convolutional operation to the hidden state matrix, and the embedding dimension of the hidden states in the $k ^ { t h }$ layer is decided by the parameter matrix $\pmb { \theta } ^ { k } \in \mathbb { R } ^ { d \times b }$ . The embedding dimension should <sup></sup>be determined according to the scale of the problem. What’s more, a nonlinear activation function Rectified Linear Unit (ReLU) (Glorot et al., 2011) is applied after every graph convolutional layer to give the network ability of nonlinear fitting:

$$
\boldsymbol {Z} ^ {k} = \widetilde {\boldsymbol {D}} ^ {\left(- \frac {1}{2}\right)} \boldsymbol {A} \boldsymbol {D} ^ {\sim \left(- \frac {1}{2}\right)} \boldsymbol {H} ^ {k - 1} \boldsymbol {\Theta} ^ {k} (k = 1, 2, \dots , K)\tag{24}
$$

$$
\boldsymbol {H} ^ {k} = \operatorname{ReLU} \left(\boldsymbol {Z} ^ {k}\right)\tag{25}
$$

After K graph convolutional layers, the final high-level representations of node features are extracted as $\pmb { H } ^ { K }$ . Then a multi-layer perceptron classifier is applied for the task of fault classification, which can be regarded as stacked fully connected (FC) neural layers. $\pmb { H } ^ { K }$ is firstly flatten to a vector ${ \pmb a } ^ { 0 }$ and then input to the classifier:

$$
\pmb {a} ^ {0} = f l a t \left(\pmb {H} ^ {K}\right) = \left[ \pmb {H} _ {1} ^ {K}, \pmb {H} _ {2} ^ {K}, \dots , \pmb {H} _ {n} ^ {K} \right] ^ {T}\tag{26}
$$

$$
\pmb {z} ^ {l} = \pmb {\Theta} ^ {l} \pmb {a} ^ {l - 1} + \pmb {b} ^ {l} (l = 1, 2,.., L)\tag{27}
$$

$$
\boldsymbol {a} ^ {l} = R e L U \left(\boldsymbol {z} ^ {l}\right) (l = 1, 2,.., L - 1)\tag{28}
$$

$$
\boldsymbol {a} ^ {l} = \text { Dropout } \left(\boldsymbol {a} ^ {l}\right) (l = 1, 2,.., L - 1)\tag{29}
$$

To avoid overfitting in the multi-layer perceptron classifier, a technology named Dropout (Srivastava et al., 2014) is applied. Simply, the dropout technology randomly ignores a certain ratio $p$ of neural units in a FC layer when training the network model, and the neural layers of $1 - p$ left neurons form the valid network structure for forward and backward propagation.

The $L ^ { t h }$ FC layer in the classifier is consist of $c + 1$ neurons that stand for 1 normal state and c fault states in the TE process respectively. The output vector $\pmb { z } ^ { L } \in \mathbb { R } ^ { c + 1 }$ of the $L ^ { t h }$ neural layer is

$$
\pmb {z} ^ {L} = [ z _ {0}, z _ {1}, \dots , z _ {c} ] ^ {T}\tag{30}
$$

To further judge the class or type of the faults, a Softmax layer follows the $L ^ { t h }$ fully connected neural layer, which transforms the real numbers of output vector $z ^ { L }$ into probability distribution as

$$
\sigma \left(z _ {i}\right) = \frac {e ^ {z _ {i}}}{\sum_ {j} e ^ {z _ {j}}} (i = 0, 1, \dots , c)\tag{31}
$$

$$
\sum_ {i} \sigma (z _ {i}) = \frac {\sum_ {i} e ^ {z _ {i}}}{\sum_ {j} e ^ {z _ {j}}} = 1\tag{32}
$$

$$
\sigma \left(\boldsymbol {z} ^ {L}\right) = \left[ \sigma \left(z _ {0}\right), \sigma \left(z _ {1}\right), \dots , \sigma \left(z _ {c}\right) \right] ^ {T}\tag{33}
$$

The corresponding index of maximal value in  $\left( z ^ { L } \right)$ is regarded <sup>z</sup>as the fault class for the input data sample. After the forward propagation, the loss is calculated using $\sigma \left( z ^ { L } \right)$ and one-hot encoded vector $\pmb { y } \in \mathbb { R } ^ { c + 1 }$ of the actual fault class. If the actual fault class <sup>y</sup>corresponding to the input is $i ( 0 , 1 , 2 , . . . c ) ,$ , the $i ^ { t h }$ element of is <sup>y</sup>set to 1 and other elements are all set to 0. The loss function commonly used in multi-class classification problem is cross entropy loss function, which is also used in this paper. And the loss can be calculated for backward propagation then for parameters updating with Adam optimizer (Kingma and Ba, 2017).

$$
\mathcal {L} \left(\boldsymbol {\Theta} ^ {k}, \boldsymbol {\Theta} ^ {l}, \boldsymbol {b} ^ {l}\right) = - \frac {1}{c + 1} \sum_ {i = 0} ^ {c} \left[ y _ {i} \log \sigma \left(z _ {i}\right) + \left(1 - y _ {i}\right) \log \left(1 - \sigma \left(z _ {i}\right)\right) \right]\tag{34}
$$

The complete algorithm for training PTCN is show as Table 2.

After trained for certain epochs, the PTCN model is ready for fault type inference. Given a test data sample $\left\{ { \pmb X } _ { t } , { \pmb A } \right\}$ , the data will <sup>X A</sup>be forward propagated through the layers with all the calculation procedures as ${ \mathsf { E q . } } ( 2 1 ) \sim ( 3 2 )$ , and the model will output the vector $\dot { \sigma } \left( z ^ { L } \right)$ . The index for the maximal element in $\sigma \left( z ^ { L } \right)$ will be the fault <sup>z z</sup>type that the model infers in term of the test data sample.

## 3.4. Fault diagnosis method based on PTCN

The fault diagnosis method based on PTCN is divided into offline stage and on-line stage showed as Fig. 5. The procedures are described in detail as below.

## 3.4.1. Off-line stage

Step 1: Historical process data in normal state and different faulty states are collected for PTCN model building. Pipes & instruments diagram is collected for graph modeling.

![](images/8fb4e1de33f634117bb5b68b8ca29ac84cd113e9e3d088e82f3b62a5f9f3cfb4.jpg)  
Fig. 4. PTCN model structure.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Table 2
PTCN training algorithm.

INPUT: Dataset  $\{X_{t}, A, y\}_{n=1}^{N}$ $\Theta^{k}(k = 1, 2, \ldots, K), \Theta^{l}(l = 1, 2, \ldots, L), b^{l}(l = 1, 2, \ldots, L) \leftarrow$  Initialize parameters

for epoch = 1 to E do

for n = 1 to N do

 $\widetilde{A} = A + I$ $\widetilde{D}_{jj} = \sum_{k} \widetilde{A}_{kj}$ $H^{0} = X_{t}$ 

for k = 1 to K do

 $Z^{k} = (\widetilde{D})^{-\frac{1}{2}} A(\widetilde{D})^{-\frac{1}{2}} H^{k-1} \Theta^{k}$ $H^{k} = ReLU(Z^{k})$ 

end for

 $a^{0} = flat(H^{K}) = [H_{1}^{K}, H_{2}^{K}, \ldots, H_{n}^{K}]^{T}$ 

for l = 1 to L - 1 do

 $z^{l} = \Theta^{l} a^{l-1} + b^{l}$ $a^{l} = ReLU(z^{l})$ $a^{l} = Dropout(a^{l})$ 

end for

 $z^{L} = \Theta^{L} a^{L-1} + b^{L} = [z_{0}, z_{1}, \ldots, z_{c}]^{T}$ 

for i = 1 to c do

 $\sigma(z_{i}) = \frac{e^{z_{i}}}{\sum_{j} e^{z_{j}}}$ 

end for

end for

 $\mathcal{L}(\Theta^{k}, \Theta^{l}, b^{l}) = \frac{1}{N} \sum_{N} \left( -\frac{1}{c+1} \sum_{i=0}^{c} [y_{i} \log \sigma(z_{i}) + (1 - y_{i}) \log (1 - \sigma(z_{i}))] \right)$ $\Theta^{k}(k = 1, 2, \ldots, K), \Theta^{l}(l = 1, 2, \ldots, L), b^{l}(l = 1, 2, \ldots, L) \leftarrow$  Backpropagation and Update parameters using gradients of  $\mathcal{L}(\Theta^{k}, \Theta^{l}, b^{l})$  (e.g. Adam optimizer)

end for
</div>

Step 2: Historical process data are normalized with mean and standard deviation of every variable. Different types of faults are recognized from the historical data. A graph is constructed from the chemical process topology according to P&ID and the design of decentralized control system.

Step 3: Preprocessed historical data are cut into data matrices $\pmb { X } _ { t } \in \bar { \mathbb { R } } ^ { n \times w }$ and every data matrix is labeled with a one-hot encoded <sup>X</sup>vector according to its fault type. The adjacency matrix is calculated from the graph. Dataset $\left\{ { \pmb X } _ { t } , { \pmb A } , { \pmb y } \right\} _ { i = 1 } ^ { N }$ is divided into training dataset and testing dataset then.

![](images/297679e420a36aec5d0cf9668225101a6abb26d9daafea614aa4f4419897d4b7.jpg)  
Fig. 5. The framework of fault diagnosis method based on PTCN.

![](images/49fb77af8436ff413e036dfd85c6b5955db6090a63f438464bf148f4d90a710b.jpg)  
Fig. 6. P&ID of the revised process model; additional measurements in red (Bathelt et al., 2015) (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article).

Step 4: The PTCN model is built as the structure showed in Fig. $4 .$ The number of graph convolutional layers and the dimension of node embedding should be chosen according to the scale of the problem.

Step 5: The model is trained with training dataset.

Step 6: The model is tested with testing dataset.

Step 7: If the testing results are not satisfactory, the number of graph convolutional layers and the dimension of node embedding should be modified in Step 4 before retrained and retested. Otherwise, the model should be saved and prepared for further application at the on-line stage.

## 3.4.2. On-line stage

Step 1: The PTCN model saved at the off-line stage is loaded and prepared for real-time fault diagnosis.

Step 2: The real-time process data are collected continuously from the process for fault diagnosis.

Step 3: The real-time process data are normalized with the same mean and standard deviation at the off-line stage.

Step 4: The preprocessed real-time data are cut into data matrices and packed with the adjacency matrix calculated at the off-line stage.

Step 5: The data matrices and the adjacency matrix are input into the loaded PTCN model.

Step 6: The PTCN model outputs the diagnosis results. If a fault occurs, the model will recognize and output the fault type.

## 4. Case study: Tennessee Eastman process

## 4.1. Introduction

Tennessee Eastman (TE) process is utilized as a benchmark to show the advantages of the proposed PTCN model in this paper. (Downs and Vogel, 1993) first introduced TE process that was modified based on an actual industrial process of Eastman Chemical Company in Tennessee, USA for testing process control technologies. It developed as a platform to examine or compare different algorithms in chemical process study later. The process model describes the non-linear relationships in the unit operation and the material and energy balances. (Ricker, 1995) studied optimal steady-state operations and (Lawrence Ricker, 1996) developed a decentralized control system of TE process shortly after it was published. After two decades, (Bathelt et al., 2015) revised TE process model for inconsistent computational results with different solvers, and extended the process model with additional process measurements and runtime outputs, which is available at https://depts. washington.edu/control/LARRY/TE/download.html. The process is mainly consisted of five unit operations: a reactor, a condenser, a vapor-liquid separator, a recycle compressor and a product stripper. The P&ID of the revised process model is showed as Fig. 6.

The process produces two products from four reactants, including an inert and a byproduct. There’re six modes of process operations at different products mass ratio and production rate. The process has 41 measurements and 12 manipulated variables in (Downs and Vogel, 1993), and (Bathelt et al., 2015) added 32 more measurements. For further testing and evaluation of different algorithms, 20 process disturbances showed in Table 3 can be added to the process, which will make the process operate at abnormal conditions of faults. Although 8 more kinds of disturbances were designed in (Bathelt et al., 2015), most papers used only the original 20 process disturbances to test their proposed models. And in this paper, only the original set of 41 measurements, 11 manipulated variables (agitator speed XMV (12) was excluded because it stays constant during simulations) and 20 process disturbances were utilized. The data for evaluating the proposed model were simulated from the MATLAB/Simulink model provided in (Bathelt et al., 2015) at mode 1 (base case).

Table 3  
Process disturbances in TE process.

<table><tr><td>Variable Number</td><td>Process variable</td><td>Type</td></tr><tr><td>IDV (1)</td><td>A/C feed ratio, B composition constant (stream 4)</td><td>Step</td></tr><tr><td>IDV (2)</td><td>B composition. A/C ratio constant (stream 4)</td><td>Step</td></tr><tr><td>IDV (3)</td><td>D feed temperature (stream 2)</td><td>Step</td></tr><tr><td>IDV (4)</td><td>Reactor cooling water inlet temperature</td><td>Step</td></tr><tr><td>IDV (5)</td><td>Condenser cooling water inlet temperature</td><td>Step</td></tr><tr><td>IDV (6)</td><td>A feed loss (stream 1)</td><td>Step</td></tr><tr><td>IDV (7)</td><td>C header pressure loss-reduced availability (stream 4)</td><td>Step</td></tr><tr><td>IDV (8)</td><td>A, B, C feed composition (stream 4)</td><td>Random variation</td></tr><tr><td>IDV (9)</td><td>D feed temperature (stream 2)</td><td>Random variation</td></tr><tr><td>IDV (10)</td><td>C feed temperature (stream 4)</td><td>Random variation</td></tr><tr><td>IDV (11)</td><td>Reactor cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>IDV (12)</td><td>Condenser cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>IDV (13)</td><td>Reaction kinetics</td><td>Slow drift</td></tr><tr><td>IDV (14)</td><td>Reactor cooling water valve</td><td>Sticking</td></tr><tr><td>IDV (15)</td><td>Condenser cooling water valve</td><td>Sticking</td></tr><tr><td>IDV (16)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV (17)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV (18)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV (19)</td><td>Unknown</td><td>Unknown</td></tr><tr><td>IDV (20)</td><td>Unknown</td><td>Unknown</td></tr></table>

## 4.2. Graph constructed from TE process topology

As mentioned in Section 3.2, PTCN requires an adjacency matrix as a part of the input data structures. To generate the adjacency <sup>A</sup>matrix, the TE process topology should be formulated as a graph first. Referring to the P&ID diagram in Fig. 6 and the decentralized control system designed by (Lawrence Ricker, 1996), the steps mentioned in Section 3.1 were followed to construct a graph from the TE process topology. The graph was named as TE graph $G ^ { \mathrm { T E } }$ in this paper. The complete TE graph $\stackrel { \cdot } { G } ^ { \mathrm { T E } }$ is showed as Fig. $^ { 7 , }$ , which consists of 82 nodes and 113 edges. The TE graph $G ^ { \mathrm { T E } }$ was then transformed into corresponding adjacency matrix $\bar { \pmb { A } } ^ { T E } \in$ <sup>R82×82</sup> with Eq. (17) for <sup>A</sup>further usage in the training process of PTCN.

As mentioned in Section 3.2, the set of’ s nodes V includes not only the measurements and manipulated variables mentioned in TE process, but also some abstract nodes that are not observed. But in the PTCN model, every node in the graph has a related node feature to satisfy the requirement for message passing. For those abstract nodes that are not observed in the TE process, the node features should be initialized with certain initialization technologies, which will be discussed in Section 4.5.

## 4.3. TE process simulation and data preprocessing

As mentioned above, (Bathelt et al., 2015) provided a revised version of TE process simulation code on the platform of MATLAB/Simulink on the Tennessee Eastman Challenge Archive website https://depts.washington.edu/control/LARRY/TE/ download.html#Updated TE Code. To generate enough training and testing process data for PTCN, the TE process model was simulated with MATLAB/Simulink referring to the methods in (Zhang and Zhao, 2017) and (Wu and Zhao, 2018). The simulation mode was set to mode 1 (base case). The simulation duration and data samples are showed in Table 4. For 20 different faulty states, the process was simulated for 10 times. In every run, the simulation generated data of 28 h and the fault was inserted into the process after running in normal state for 8 h, which means there are 20 h of simulation data for every run of a fault type. The simulation data were sampled every 3 min. It should be noted that the simulation can only last for 7 h after the fault IDV 6 is inserted, because some variables will exceed allowable limits after 7 h’ run, which will cause an early stop of the simulation program. For normal state, the model was simulated for 3750 h all at once without any fault inserted.

![](images/133874dcbc1c98c70c76b6468547cea223524e15f2598219767e0f9ce619ad0e.jpg)  
Fig. 7. The complete TE graph.

Simulation dataset for training and testing.

<table><tr><td>Fault types</td><td>Simulation length (training) / h</td><td>Number of Data Matrices (training)</td><td>Simulation length (testing) / h</td><td>Number of Data Matrices (testing)</td></tr><tr><td>IDV 1~5 &amp; 7~20</td><td>20 × 19 × 8</td><td>60800</td><td>20 × 19 × 2</td><td>15200</td></tr><tr><td>IDV 6</td><td>7 × 1 × 8</td><td>1120</td><td>7 × 1 × 2</td><td>280</td></tr><tr><td>Normal</td><td>3000</td><td>59981</td><td>750</td><td>14981</td></tr><tr><td>Total</td><td>6096</td><td>121901</td><td>1524</td><td>30461</td></tr></table>

The simulation data was normalized with sample mean and sample standard deviation calculated with all the simulation data in normal state. As described in Section 3.2, the data matrix $\pmb { X } _ { t }$ in a data sample $\{ { \pmb X } _ { t } , { \pmb A } , { \pmb y } \}$ <sup>X</sup>was stacked time serials in a certain time window. In this paper, the time window was set to 60 min, which means $\pmb { X } _ { t }$ includes 20 data points for every variable.

<sup>X</sup>For faulty data, 8 runs of simulation data were randomly chosen as training data and 2 runs were chosen as testing data. For normal data, 3000 h of simulation data were chosen as training data and 750 h were chosen as testing data. The ratio of faulty samples to normal samples was about 1:1 either in training dataset or testing dataset, which is considered to balance the numbers of faulty samples and normal samples for less false alarms.

## 4.4. Training parameters and evaluation criterion

To ensure the consistency of training conditions, all PTCN models were trained for 50 epochs. In every epoch, the dataset was divided into mini-batches with the size of 128 data samples. Cross entropy loss function was chosen for this multi-classification problem. For backpropagation, Adam optimizer (Kingma and Ba, 2017) was used to update the trainable parameters with the learning rate of 0.001 (Table 5).

Table 5  
Training parameters for the PTCN model

<table><tr><td>Epochs</td><td>Mini-batch size</td><td>Loss function</td><td>Optimizer</td><td>Learning rate</td></tr><tr><td>50</td><td>128</td><td>Cross entropy loss</td><td>Adam</td><td>0.001</td></tr></table>

Table 6  
Confusion matrix in multi-class classification

<table><tr><td colspan="2"></td><td colspan="2">Predicted class</td></tr><tr><td>&#x27;</td><td></td><td>Class c</td><td>Other Classes</td></tr><tr><td>Actual</td><td>Class c</td><td>True Positive (TP)</td><td>False Negative (FN)</td></tr><tr><td>class</td><td>Other Classes</td><td>False Positive (FP)</td><td>True Negative (TN)</td></tr></table>

There should be some criterions to evaluate the performance of fault diagnosis methods, so fault diagnosis rate (FDR) and accurate classification rate (ACR) (Zhang and Zhao, 2017) are introduced as followed. In multi-class classification, the confusion matrix for class c is defined as Table 6.

FDR and ACR are defined as

$$
F D R = \frac {N _ {T P}}{N _ {T P} + N _ {F N}}\tag{35}
$$

$$
A C R = \frac {\sum_ {c} N _ {T P}}{\sum_ {c} N _ {T P} + \sum_ {c} N _ {F N}} = \frac {\sum_ {c} N _ {T P}}{N _ {t o t a l}}\tag{36}
$$

FDR concerns how many samples of a certain class are recognized as the class itself, it’s also called true positive rate (TPR) generally. The higher FDR is, the more accurate the fault diagnosis model is for recognize class c. ACR is a total classification accuracy of this fault diagnosis model considering all the testing samples. It’s also a class-weighted average of FDR and will be used to evaluate the general performance of a model in this paper.

## 4.5. Hyper-parameters tuning

## 4.5.1. Direction of TE graph

In Section $4 . 2 ,$ , the TE process topology is formulated as a directed graph $G ^ { \mathrm { T E } }$ , where all the edges’ directions are determined by streams’ flow directions or control signals’ transmission directions. However, the best directions for message passing on $G ^ { \mathrm { T E } }$ are not necessarily the same as the physical conditions. Message may also be passed on a reverse graph of $G ^ { \mathrm { T E } }$ or an undirected graph of $G ^ { \mathrm { T E } }$ Fig. 8 shows the difference among a directed graph, its reverse graph and its undirected graph.

To choose the best directions for message passing, three types of adjacency matrix $\pmb { A } ^ { \mathrm { o r i g i n a l } } , \pmb { A } ^ { \mathrm { r e v e r s e } } , \pmb { A } ^ { \mathrm { u n d i r e c t e d } }$ are added to the dataset for testing.

$$
\pmb {A} ^ {\mathrm{original}} = \pmb {A} ^ {\mathrm{TE}}\tag{37}
$$

$$
\boldsymbol {A} ^ {\text { reverse }} = \left(\boldsymbol {A} ^ {\text { TE }}\right) ^ {T}\tag{38}
$$

$$
\boldsymbol {A} ^ {\text { undirected }} = \boldsymbol {A} ^ {\mathrm{TE}} + \left(\boldsymbol {A} ^ {\mathrm{TE}}\right) ^ {T}\tag{39}
$$

The base model structure is Model 3 in Table 8, which consists of 3 graph convolutional layers with 20-dimension embedding and a 2-layer perceptron classifier with 300 neurons and a Dropout rate of 0.5 in the first layer. The last layer of the classifier has 21 neurons (1 for normal state and 20 for different types of fault). The variables in $\pmb { X } _ { t }$ that are not observed in TE process are all initial-<sup>X</sup>ized with 0. The training parameters are introduced in Section 4.4. Table 7 shows the experiments results trained using the dataset in Section 4.3 with different type of graphs. The results indicate that the best message passing directions should be the same as the reverse of the original TE graph. Along the directions of streams flow or control signals’ transmission, the nodes in the downstream will carry the information propagated through the edges from the nodes in the upstream. With a reverse message passing directions in the PTCN model, one node’s hidden state will fuse information of itself and the nodes under its influence that belongs to a serial of time since the downstream nodes carry its previous information. This helps the model to extract node representations from spatial and temporal domains simultaneously. Thus, the reverse TE graph was chosen for PTCN model in the experiments below and the final testing procedure in Section 4.6.

Table 7  
Experiments results using different type of graphs

<table><tr><td>Graph types</td><td>Original TE graph</td><td>Reverse TE graph</td><td>Undirected TE graph</td></tr><tr><td>ACR</td><td>0.9246</td><td>0.9392</td><td>0.9168</td></tr></table>

Table 8  
PTCN model with different number of graph convolutional layers.

<table><tr><td>Model Name</td><td>Model structure</td></tr><tr><td>Model 1</td><td>GConv(20)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 2</td><td>GConv(20)-GConv(20)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 3</td><td>GConv(20)-GConv(20)-GConv(20)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 4</td><td>GConv(20)-GConv(20)-GConv(20)-GConv(20)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 5</td><td>GConv(20)-GConv(20)-GConv(20)-GConv(20)-GConv(20)-FC(300, 0.5)-FC(21)</td></tr></table>

Table 9

Experiments result of PTCN model with different number of graph convolutional layers.

<table><tr><td>Model Name</td><td>Model 1</td><td>Model 2</td><td>Model 3</td><td>Model 4</td><td>Model 5</td></tr><tr><td>ACR</td><td>0.9294</td><td>0.9283</td><td>0.9392</td><td>0.9311</td><td>0.9271</td></tr></table>

## 4.5.2. Number of graph convolutional layers

The number of graph convolutional layers has a strong impact on the performance of PTCN since it decides the width of message passing. With too few graph convolutional layers, a node can only perceive nearby neighbor nodes so the information is not fully spread through the graph. With too many graph convolutional layers, every node will contain information from nearly the whole graph, which eliminates the differences among nodes and makes the graph too smooth to keep the distinguish information of nodes. To determine the appropriate number, 5 PTCN models with 1∼5 different number of graph convolutional layers are trained and tested with the dataset. The model candidates are shown as Table 8. In the structure of models, GConv(n) means a graph convolutional layer with hidden state embedding dimension of n, FC(n, p) is a fully connected layer with n neurons using a dropout ratio of p. The variables in that are not observed in TE process are all initialized with <sup>X</sup>0. Table 9 shows that 3 graph convolutional layers is appropriate for PTCN, and less GConv layers or more GConv layers will have an impact on decreasing the overall performance of the model.

## 4.5.3. Embedding dimension of node features

The embedding dimension of node features is another important parameter for graph convolutional layers, which has to match with the scale of problem. With a high embedding dimension, the hidden states can be very sparse and thus increases the computational cost for redundancy dimensions. With a low embedding dimension, there will be information loss of node features that can degrade the performance of PTCN. The initialized node feature is 20-dimension for every variable, which is determined by the time window we choose to form a data matrix $\pmb { X } _ { t } .$ To choose an appro-<sup>X</sup>priate dimension of embedding, different PTCN model candidates are set as Table 10. The variables in $\pmb { X } _ { t }$ that are not observed in TE <sup>X</sup>process are all initialized with 0. The testing results are shown in Table 11 and it indicates that the dimension of hidden states should be kept to 20 all the time.

![](images/f8d1cce716b0138defa402b493104f23598b9e419da90f983d100473b0c73133.jpg)

![](images/4e24bc44c44651ac4be8596c74062771f0ae77655484e048f9ceed55128ff8cd.jpg)

![](images/3753112cf2f789fe23e8a9f2f976cca80483a8d82fbc2c97f8c327471424b398.jpg)  
Fig. 8. Example graph with different edge directions. (a) original graph G; (b) the reverse graph of G; (c) undirected graph of G.

Table 10  
PTCN model with different embedding dimension of graph convolutional layers.

<table><tr><td>Model Name</td><td>Model structure</td></tr><tr><td>Model 6</td><td>GConv(16)-GConv(8)-GConv(4)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 7</td><td>GConv(16)-GConv(10)-GConv(8)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 3</td><td>GConv(20)-GConv(20)-GConv(20)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 8</td><td>GConv(24)-GConv(28)-GConv(32)-FC(300, 0.5)-FC(21)</td></tr><tr><td>Model 9</td><td>GConv(32)-GConv(48)-GConv(64)-FC(300, 0.5)-FC(21)</td></tr></table>

Table 11  
Experiments result of PTCN model with different embedding dimension.

<table><tr><td>Model name</td><td>Model 6</td><td>Model 7</td><td>Model 3</td><td>Model 8</td><td>Model 9</td></tr><tr><td>ACR</td><td>0.9264</td><td>0.9269</td><td>0.9392</td><td>0.9305</td><td>0.9312</td></tr></table>

## 4.5.4. Initialization methods of unobserved variables

Section 3.2 mentioned that the data matrix $\pmb { X } _ { t } ~ \in ~ \mathbb { R } ^ { 8 2 \times 2 0 }$ <sup>Xt</sup>includes not only the 41 measurements and the 11 manipulated variables (XMV (12) excluded), but also some nodes without corresponding observations in the TE process. These nodes should be initialized with some values for computational requirements, which is also known as missing value filling. Three initialization methods are tested to find an appropriate one, namely constant initialization, random initialization and standard normal initialization. Constant initialization will initialize the missing values with the constant 0, since all the historical data are normalized with standard score normalization. Random initialization will initialize the miss values with data randomly sampled from [−1, 1], and standard normal initialization will use the data sampled from standard normal distribution N (0, 1). Model 3 was chosen as the base model for testing the initialization methods. The results are shown as Table 12. It can be seen that initialization with constant 0 is an appropriate method to fill the missing values relatively.

Table 12  
Experiments result of PTCN model with different initialization methods for unobserved variables.

<table><tr><td>Initialization methods</td><td>Constant initialization</td><td>Standard normal initialization</td><td>Random initialization</td></tr><tr><td>ACR</td><td>0.9392</td><td>0.9250</td><td>0.9274</td></tr></table>

Table 13  
ACR and average time cost for training and testing.

<table><tr><td>Model</td><td>Testing ACR</td><td>Time cost for training /second/epoch</td><td>Time cost for testing /second/epoch</td></tr><tr><td>Model 1</td><td>0.9294</td><td>23</td><td>7</td></tr><tr><td>Model 2</td><td>0.9283</td><td>24</td><td>7</td></tr><tr><td>Model 3</td><td>0.9392</td><td>25</td><td>7</td></tr><tr><td>Model 4</td><td>0.9311</td><td>29</td><td>8</td></tr><tr><td>Model 5</td><td>0.9271</td><td>31</td><td>8</td></tr><tr><td>Model 6</td><td>0.9264</td><td>26</td><td>8</td></tr><tr><td>Model 7</td><td>0.9269</td><td>26</td><td>8</td></tr><tr><td>Model 8</td><td>0.9305</td><td>27</td><td>8</td></tr><tr><td>Model 9</td><td>0.9312</td><td>27</td><td>8</td></tr></table>

## 4.6. Experiment results

Table 13 concludes all the ACR and the average time cost for training and testing within every epoch. All the experiments were conducted on the platform of CentOS Linux release 7.7 with Intel Xeon Gold 5118 processor and Titan RTX graphics card. To avoid the error caused by randomness, one type of model structure was trained for 10 times and the results showed in Table 13 are the average of all parallel experiments. It shows that the average time cost for training within every epoch is around 27 s, which shows no big difference between different models. And the PTCN with Model 3 structure reaches the highest testing ACR of 0.9392.

The final chosen structure for PTCN is Model 3 in Table 8. The adjacency matrix $\pmb { A } ^ { \mathrm { r e v e r s e } }$ required by PTCN is calculated with the <sup>A</sup>reverse of original TE graph. And the missing values in data matrices are all initialized with the constant 0. With the training parameters introduced in Section 4.4, PTCN is trained and tested and Fig. 9 show the changes of average accuracy with the number of training epochs for both training and testing dataset. It can be seen that the curves are converged steadily. Even with 3 epochs, the testing accuracy almost reaches the maximum.

The confusion matrix of the testing results shown as Fig. 10 illustrates the diagnosis performance of PTCN for every fault class respectively. The labels in vertical axis are actual labels for fault types from 0 to 20, and fault 0 means normal state. The labels in horizontal axis are predicted labels. In previous studies of fault diagnosis models, FDRs for fault 3, 9, 15 are relatively low and the 3 types of faults can hardly be distinguished from normal state. Fig. 10 visually display that PTCN greatly improve the diagnosis for fault 3, while fault 9 and 15 are still hard to recognized, especially for fault 15.

![](images/448e2534814a68d99f94bca02bd30d5411fb1c362ff907cf1d6ec9d3a0b94120.jpg)  
Fig. 9. Change of testing ACR with the number of training epochs.

![](images/689455b2e38d533196ae52a40a17cb90c1715f37fe0744c26475a3ead5b86cd8.jpg)  
Fig. 10. Confusion matrix of the testing results.

Table 14 gives the detailed experiments results for every fault type and the comparison with DBN-based (Zhang and Zhao, 2017), CNN-based (Wu and Zhao, 2018) and RNN-based (Zhang et al., 2020a) fault diagnosis models. These are all deep learning-based models. And the other approaches in Table 14 utilize graphs either, including Bayesian-based CGN (Lou et al., 2020) and SOM-based DFAE-SOM (Lu and Yan, 2020). The FDR for normal state of PTCN is much higher than other models, which will decrease false alarms in actual application. This is quite importance since a high frequency of false alarm will indeed increase the burden of the operators to confirm and thus lower their trust on the fault diagnosis models, which goes against the original intension. In all the 21 classes, there are 17 classes that corresponding FDR is or higher than 0.9. For fault 8 and 16 that FDRs are not satisfactory in some of other models, PTCN reaches the FDR of 0.9160 or higher, especially for fault 16 that the FDR reaches 0.9685. For the overall performance of the models, PTCN reaches the highest 0.9392 average accuracy among all the models, which is 5.72 % higher than DCNN and 1.22 % higher than BiGRU. Without the fault 9 and 15 that are constitutionally similar to normal state, the average accuracy of PTCN reaches 0.9729, and is 2.09 % higher than BiGRU, 3.89 % than CNN, 3.52 % higher than CGN and 2.32 % higher than DFAE-SOM. In the experiments of other graph-based approaches, CGN and DFAE-SOM ignored more than fault 9 and 15, which made the classification problem even easier.

With the knowledge of the chemical process topology, PTCN outperforms other pure data-driven fault diagnosis models in terms of the average performance. The adjacency matrix of the graph input to the model is indispensable, because the graph describes the connectivity of different parts of the chemical process. Essentially, the connectivity indicates the correlations of different process variables. This provides a guidance when the model is trained with data to fit the underlying relationships among different variables in the process, and it will reduce the risk of overfitting the training dataset. Compared to other approaches using graphs, PTCN also gets better results, which indicates its strong ability to learn about the relationships between variables from data.

## 4.7. Network parameters, computation cost and requirement for training data

To dig into the benefits of the utilization of the process topology, it’s important to pay attention to time cost in training and testing, the number of network parameters and the requirement of training data. Time cost in training and testing is about the PTCN’s efficiency. In practical applications, the faster PTCN can make inferences, the less time it’ll take to detect existing faults. The number of network parameters decides the model’s complexity. This is related to the consumption of computing resources and the required amount of training data. A simple model is always preferred because the fault data we can acquire for training are generally limited in real industrial applications.

Table 15 shows the numbers of network parameters and average time cost of training and testing of PTCN with different model structures. The table shows that most of the candidate models have less than 1 million trainable network parameters and the numbers are around 0.6 million. For the chosen structure Model 3, the number of network parameters is 0.597 million, while Model 7 of DCNN in (Wu and Zhao, 2018) has 2.537 million parameters in total, which is more than 4 times than that of Model 3 of PTCN. This indicates that the complexity of the network can be reduced with the guidance of process knowledge. Because the knowledge can reduce the degree of freedom and then cut down a large proportion of network parameters. Without any prior knowledge, the design of a network model is relatively blind so more trainable parameters will be added for uncertainty. In addition, a more lightweight network model with less parameters means less computing resources consumption including memory and time cost.

Table 15 shows that the average time cost for training within every epoch is around 27 s and it takes 25 s for Model 3 to be trained for an epoch. Under the condition of parallel computing, the average time cost for testing a data matrix is no more than 0.263 milliseconds when the total number of testing samples is 30,461 and the mini-batch size is set to 128. But Model 7 of DCNN in (Wu and Zhao, 2018) takes 30 s to train for every epoch. Average time cost for testing a data matrix is 1.5 milliseconds, which is about 6 times as long as PTCN takes. This indicates that the time cost for training and testing is closely related to the number of network parameters. With less parameters, the PTCN can be trained and tested faster, which is critical in the practical application.

To study the requirement of training data, Model 3 of PTCN and Model 7 of DCNN was trained with different number of runs of simulation data. And all the trained models were tested with the same testing dataset mentioned in Section 4.3. Table 16 shows the experiments results. The number of training data matrices and the testing ACRs are listed with different number of runs of simulation data. 8 runs of simulation data formed the original training dataset in Section 4.3. With less runs of simulation, the number of data matrices in the training dataset will decrease proportionately, but the ratio of data in normal state to faulty state is kept the same as original training dataset. And all the testing ACRs are the average testing result of 10 independent repeated training processes. Fig. 11 also shows the change of ACR of PTCN and DCNN with different number of runs of training data. The confidence interval is calculated under the significance level of 0.05.

Table 14  
FDR and ACR comparison of fault diagnosis experiments results on testing dataset.

<table><tr><td>Fault type</td><td>DBN (Zhang and Zhao, 2017)</td><td>DCNN (Wu and Zhao, 2018)</td><td>BiGRU (S. Zhang et al., 2020a)</td><td>CGN (Lou et al., 2020)</td><td>DFAE-SOM (Lu and Yan, 2020)</td><td>PTCN (ours)</td></tr><tr><td>Normal</td><td>-</td><td>0.978</td><td>0.969</td><td>0.985</td><td>0.748</td><td>0.9924</td></tr><tr><td>Fault 1</td><td>1</td><td>0.986</td><td>0.986</td><td>0.975</td><td>0.995</td><td>0.9931</td></tr><tr><td>Fault 2</td><td>0.99</td><td>0.985</td><td>0.972</td><td>0.980</td><td>0.987</td><td>0.9819</td></tr><tr><td>Fault 3</td><td>0.95</td><td>0.917</td><td>0.935</td><td>-</td><td>-</td><td>0.8804</td></tr><tr><td>Fault 4</td><td>0.98</td><td>0.976</td><td>0.974</td><td>0.824</td><td>0.996</td><td>0.9956</td></tr><tr><td>Fault 5</td><td>0.86</td><td>0.915</td><td>0.998</td><td>0.980</td><td>1</td><td>0.9786</td></tr><tr><td>Fault 6</td><td>1</td><td>0.975</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Fault 7</td><td>1</td><td>0.999</td><td>1</td><td>1</td><td>0.998</td><td>1</td></tr><tr><td>Fault 8</td><td>0.78</td><td>0.922</td><td>0.753</td><td>0.966</td><td>-</td><td>0.9160</td></tr><tr><td>Fault 9</td><td>0.57</td><td>0.584</td><td>0.807</td><td>-</td><td>-</td><td>0.6601</td></tr><tr><td>Fault 10</td><td>0.98</td><td>0.964</td><td>1</td><td>0.881</td><td>-</td><td>0.9276</td></tr><tr><td>Fault 11</td><td>0.87</td><td>0.984</td><td>0.965</td><td>0.778</td><td>-</td><td>0.9798</td></tr><tr><td>Fault 12</td><td>0.85</td><td>0.956</td><td>0.961</td><td>0.981</td><td>0.781</td><td>0.9704</td></tr><tr><td>Fault 13</td><td>0.88</td><td>0.957</td><td>0.953</td><td>0.758</td><td>-</td><td>0.8969</td></tr><tr><td>Fault 14</td><td>0.87</td><td>0.987</td><td>0.996</td><td>0.986</td><td>0.978</td><td>0.9964</td></tr><tr><td>Fault 15</td><td>0</td><td>0.28</td><td>0.541</td><td>-</td><td>-</td><td>0.0035</td></tr><tr><td>Fault 16</td><td>0</td><td>0.442</td><td>0.788</td><td>0.814</td><td>0.855</td><td>0.9685</td></tr><tr><td>Fault 17</td><td>1</td><td>0.945</td><td>0.97</td><td>0.848</td><td>0.928</td><td>0.9254</td></tr><tr><td>Fault 18</td><td>0.98</td><td>0.939</td><td>0.923</td><td>0.685</td><td>-</td><td>0.9049</td></tr><tr><td>Fault 19</td><td>0.93</td><td>0.986</td><td>0.926</td><td>0.964</td><td>0.860</td><td>0.9650</td></tr><tr><td>Fault 20</td><td>0.93</td><td>0.933</td><td>0.981</td><td>0.871</td><td>0.788</td><td>0.8825</td></tr><tr><td>ACR</td><td>0.821</td><td>0.882</td><td>0.927</td><td>-</td><td>-</td><td>0.9392</td></tr><tr><td>ACR w/o Fault 9 &amp; 15</td><td>0.889</td><td>0.934</td><td>0.952</td><td>0.904</td><td>0.916</td><td>0.9729</td></tr></table>

Table 15  
Numbers of network parameters and average time cost for training and testing.

<table><tr><td>Model</td><td>Numbers of network parameters/M</td><td>Time cost for training /(s/epoch)</td><td>Time cost for testing /(ms/matrix)</td></tr><tr><td>Model 1</td><td>0.531</td><td>23</td><td>0.230</td></tr><tr><td>Model 2</td><td>0.564</td><td>24</td><td>0.230</td></tr><tr><td>Model 3</td><td>0.597</td><td>25</td><td>0.230</td></tr><tr><td>Model 4</td><td>0.630</td><td>29</td><td>0.263</td></tr><tr><td>Model 5</td><td>0.663</td><td>31</td><td>0.263</td></tr><tr><td>Model 6</td><td>0.144</td><td>26</td><td>0.263</td></tr><tr><td>Model 7</td><td>0.249</td><td>26</td><td>0.263</td></tr><tr><td>Model 8</td><td>0.962</td><td>27</td><td>0.263</td></tr><tr><td>Model 9</td><td>2.011</td><td>27</td><td>0.263</td></tr><tr><td>Model 7 of DCNN(Wu and Zhao, 2018)</td><td>2.537</td><td>30</td><td>1.5</td></tr></table>

Table 16  
ACR of PTCN models trained with different number of runs of simulation data.

<table><tr><td>Number of runs of simulation data</td><td>Number of data matrices</td><td>Testing ACR (PTCN, ours)</td><td>ACR drop compared to the original training dataset (%)</td><td>Testing ACR (DCNN, Wu and Zhao, 2018)</td><td>ACR drop compared to the original training dataset (%)</td></tr><tr><td>8</td><td>121,901</td><td>0.9392</td><td>-</td><td>0.9309</td><td>-</td></tr><tr><td>7</td><td>106,661</td><td>0.9389</td><td>0.03</td><td>0.9294</td><td>0.15</td></tr><tr><td>6</td><td>91,421</td><td>0.9373</td><td>0.19</td><td>0.9260</td><td>0.49</td></tr><tr><td>5</td><td>76,181</td><td>0.9347</td><td>0.45</td><td>0.9227</td><td>0.82</td></tr><tr><td>4</td><td>60,941</td><td>0.9281</td><td>1.11</td><td>0.9165</td><td>1.44</td></tr><tr><td>3</td><td>45,701</td><td>0.9227</td><td>1.65</td><td>0.9133</td><td>1.76</td></tr><tr><td>2</td><td>30,461</td><td>0.9142</td><td>2.50</td><td>0.9040</td><td>2.69</td></tr><tr><td>1</td><td>15,221</td><td>0.8943</td><td>4.49</td><td>0.8738</td><td>5.71</td></tr></table>

For both PTCN and DCNN, the testing ACR will drop with the reduction of training data, which is as expected because datadriven models rely on patterns learnt from data. The testing ACR will drop more quickly when more training data are reduced. This means that the more data the training dataset has, the more redundant information the dataset has. In another word, the marginal benefit of adding data to training dataset will diminish in terms of the model’s performance.

Trained with different number of runs of simulation data, the testing ACR of PTCN is always about 1% higher than DCNN. As the training data decrease, the testing ACR of DCNN drops more quickly than PTCN. This means that the knowledge of process topology integrated in PTCN plays a role in reducing the impact of the decrease of training data. To get the same testing ACR, training data required by PTCN is less than DCNN. For example, to get the testing ACR of 0.93, it can be estimated from Fig. 11 that PTCN requires only 54 % of training data needed by DCNN. Namely, PTCN relies less on training data with the guidance of information from the chemical process. This feature of PTCN is very critical for the situations where data are relatively difficult to obtain but the topology of the process can be utilized to construct the model.

![](images/d4924256dd2b2c8f248dad661065aa8eeb9bec2d94e8422c47f8d339e9b0552c.jpg)  
Fig. 11. ACR of PTCN and DCNN trained with different number of runs of simulation data.

## 5. Conclusion

In this paper, process topology convolutional network (PTCN) model is proposed for fault diagnosis of chemical processes. Different from pure data-driven models, PTCN utilizes the knowledge of the chemical process topology. The process topology is transformed into a graph, which reflects the relationships among different process variables. The graph plays a role of bridge to integrate process knowledge to the model.

To illustrate the performance of PTCN, its application to the benchmark TE process is used as a case study. The TE process topology is first transformed into a directed graph named TE graph. Corresponding adjacency matrix which describes nodes and their connectivity in the graph is established. The training and testing data are generated by running the simulation model of the TE process. The experiments show that the ACR of the PTCN reaches 0.9392, which is higher than existing fault diagnosis models such as CNN-based, RNN-based, Bayesian-based and SOM-based models.

PTCN has only 0.597 million of network parameters, which is less than a quarter of the number of DCNN (Wu and Zhao, 2018). This means less memory consumption and time cost in training and testing processes. Integrated with knowledge of process topology, the PTCN model relies less on training data. Trained with different amount of data, the testing ACR of PTCN is always about 1% higher than DCNN. To get the testing ACR of 0.93, PTCN requires only 54 % of training data needed by DCNN.

From the perspective of process safety, the PTCN model can be further applied in RA and ASM. As an initial step in ASM, the PTCN model can monitor the operating states of chemical processes in real time. Once a fault happens, the model will complete detection and diagnosis simultaneously and will give out the possibilities of different types of fault. These possibilities can be used for quantitative dynamic risk assessment. The combination of PTCN and quantitative risk assessment models can provide the basis of decision for ASM, which is the final step to bring the processes back to normal states.

The knowledge from chemical process topology helps improve the performance of the data-driven model in terms of diagnosis accuracy and the requirements for training data and computation resources. What’s more, the feature extraction process of the PTCN model is more rational and understandable than other data-driven fault diagnosis models, since it follows the guidance of the process topology. The future research work will be focused on analyzing PTCN’s inference process and enhancing explainability for the PTCN model.

## Declaration of Competing Interest

The authors report no declarations of interest.

## Acknowledgements

The authors gratefully acknowledge support from the National Natural Science Foundation of China (No. 21878171) and the National Science and Technology Innovation 2030 Major Project (2018AAA0101605) of the Ministry of Science and Technology of China.

## References

Amin, M.T., Khan, F., Ahmed, S., Imtiaz, S., 2020. A novel data-driven methodology for fault detection and dynamic risk assessment. Can. J. Chem. Eng. 98, 2397–2416, http://dx.doi.org/10.1002/cjce.23760.

Arunthavanathan, R., Khan, F., Ahmed, S., Imtiaz, S., 2020a. An analysis of process fault diagnosis methods from safety perspectives. Comput. Chem. Eng., http:// dx.doi.org/10.1016/j.compchemeng.2020.107197, 107197.

Arunthavanathan, R., Khan, F., Ahmed, S., Imtiaz, S., Rusli, R., 2020b. Fault detection and diagnosis in process system using artificialintelligence-based cognitive technique. Comput. Chem. Eng. 134, http://dx.doi.org/10.1016/j.compchemeng. 2019.106697, 106697.

Bathelt, A., Ricker, N.L., Jelali, M., 2015. Revision of the Tennessee eastman process model. IFACPapersOnLine 48, 309–314, http://dx.doi.org/10.1016/j.ifacol.2015. 08.199.

Bikmukhametov, T., Jäschke, J., 2020. Combining machine learning and process engineering physics towards enhanced accuracy and explainability of datadriven models. Comput. Chem. Eng. 138, 106834, http://dx.doi.org/10.1016/j. compchemeng.2020.106834

Chen, H., Tino,ˇ P., Yao, X., 2014. Cognitive fault diagnosis in Tennessee Eastman process using learning in the model space. Comput. Chem. Eng. 67, 33–42, http:// dx.doiorg/10.1016/i.compchemeng.2014.03.015.

Cheng, F., He, Q.P., Zhao, J., 2019. A novel process monitoring approach based on variational recurrent autoencoder. Comput. Chem. Eng. 129, 106515, http://dx. doi.org/10.1016/j.compchemeng.2019.106515.

Dai, Y., Wang, H., Khan, F., Zhao, J., 2016. Abnormal situation management for smart chemical process operation. Curr. Opinion Chem. Eng. Biotechnol. Bioprocess Eng. Proc. Syst. Eng. 14, 49–55, http://dx.doi.org/10.1016/j.coche.2016.07.009.

Dai, Y., Zhao, J., 2011. Fault diagnosis of batch chemical processes using a dynamic time warping (DTW)-Based artificial immune system. Ind. Eng. Chem. Res. 50, 4534–4544, http://dx.doi.org/10.1021/ie101465b

Defferrard, M., Bresson, X., Vandergheynst, P., 2016. Convolutional neural networks on graphs with fast localized spectral filtering, In: Lee. D.D., Sugivama. M. Luxburg, U.V, Guvon. I., Garnett. R. (Eds.). Adyances in Neural Information Processing Systems 29. Curran Associates, Inc., pp. 3844–3852.

Downs, J.J., Vogel, E.F., 1993. A plant-wide industrial process control problem. Comput. Chem. Eng. 17, 245–255.

Gharahbagheri, H., Imtiaz, S.A., Khan, F., 2017. Root cause diagnosis of process fault using KPCA and bayesian network. Ind. Eng. Chem. Res. 56, 2054–2070, http:// dx.doi.org/10.1021/acs.iecr.6b01916.

Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., Dahl, G.E., 2017. Neural message passing for quantum chemistry. arXiv, 1704.01212 [cs].

Glorot, X., Bordes, A., Bengio, Y., 2011. Deep sparse rectifier neural networks. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. Presented at the Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, JMLR Workshop and Conference ProceedIngs. 315-323

Gori, M., Monfardini, G., Scarselli, F., 2005. A new model for learning in graph domains, In: Proceedings. 2005 IFFE International Joint Conference on Neural Networks, 2005. Presented at the International Joint Conference on Neural Networks 2005, IEEE, Montreal, Que., Canada, pp. 729–734, http://dx.doi.org/10. 1109/IJCNN.2005.1555942.

Henaff, M., Bruna, J., LeCun, Y., 2015. Deep convolutional networks on graphstructured data, arXiv, 1506.05163 [cs]

Khamsi M.A. Kirk W.A. 2001, An Introduction to Metric Spaces and Fixed Point Theory. John Wiley & Sons, Inc., http://dx.doi.org/10.1002/9781118033074.

Khan, F., Rathnayaka, S., Ahmed, S., 2015. Methods and models in process safety and risk management: past, present and future. Process. Saf. Environ. Prot. 98, 116–147, http://dx.doi.org/10.1016/j.psep.2015.07.005.

Kingma. D.P.. Ba. I. 2017. Adam: a method for stochastic optimization. arXiv. 1412.6980[csl.

Kipf, T.N., Welling, M., 2017. Semi-supervised classification with graph convolutional networks. arXiv, 1609.02907 [cs, stat].

Krizhevsky, A., Sutskever, I., Hinton, G.E., 2017. ImageNet classification with deep convolutional neural networks. Commun. ACM 60, 84–90, http://dx.doi.org/10. 1145/3065386

Lawrence Ricker, N., 1996. Decentralized control of the Tennessee eastman challenge process. J. Process Control 6, 205–221, http://dx.doi.org/10.1016/0959- 1524(96)00031-5.

Lever, J., Krzywinski, M., Altman, N., 2016. Model selection and overfitting. Nat Methods 13, 703–704, http://dx.doi.org/10.1038/nmeth.3968.

Lou, C., Li, X., Atoui, M.A., 2020. Bayesian network based on an adaptive threshold scheme for fault detection and classification. Ind. Eng. Chem. Res. 59 15155–15164, http://dx.doi.org/10.1021/acs.iecr.0c02762

Lu, W., Yan, X., 2020. Deep fisher autoencoder combined with self-organizing map for visual industrial process monitoring. J. Manuf. Syst. 56, 241–251, http://dx. doi.org/10.1016/j.jmsy.2020.05.005.

Lv, F., Wen, C., Bao, Z., Liu, M., 2016. Fault diagnosis based on deep learning. 2016 American Control Conference (ACC). Presented at the 2016 American Control Conference (ACC), 6851–6856, http://dx.doi.org/10.1109/ACC.2016.7526751.

Madakyaru, M., Harrou, F., Sun, Y., 2017. Improved data-based fault detection strategy and application to distillation columns. Process. Saf. Environ. Prot. 107, 22–34, http://dx.doi.org/10.1016/j.psep.2017.01.017.

Qin, S.J., Chiang, L.H., 2019. Advances and opportunities in machine learning for process data analytics. Comput. Chem. Eng. 126, 465–473, http://dx.doi.org/10. 1016/j.compchemeng.2019.04.003.

Redmon, J., Divvala, S., Girshick, R., Farhadi, A., 2016. You only look once: unified, real-time object detection. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Presented at the 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Las Vegas, NV, USA, pp. 779–788, http://dx.doi.org/10.1109/CVPR.2016.91.

Ricker, N.L., 1995. Optimal steady-state operation of the Tennessee Eastman challenge process. Comput. Chem. Eng. 19, 949–959, http://dx.doi.org/10.1016 0098-1354(94)00043-N.

Scarselli, F., Gori, M., Chung Tsoi, Ah, Hagenbuchner, M., Monfardini, G., 2009. The graph neural network model. IEEE Trans. Neural Netw. 20, 61–80, http://dx.doi. org/10.1109/TNN.2008.2005605.

Shu, Y., Ming, L., Cheng, F., Zhang, Z., Zhao, J., 2016. Abnormal situation management: challenges and opportunities in the big data era. Comput. Chem. Eng. 91, 104–113, http://dx.doi.org/10.1016/j.compchemeng.2016.04.011.

Sperduti, A., Starita, A., 1997. Supervised neural networks for the classification of structures. IEEE Trans. Neural Netw. 8, 714–735, http://dx.doi.org/10.1109/72. 572108.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R., 2014. Dropout: a simple way to prevent neural networks from overfitting. J. Mach. Learn. Res. 15, 1929–1958.

Venkatasubramanian, V., 2019. The promise of artificial intelligence in chemical engineering: Is it here, finally? Aiche J. 65, 466–478, http://dx.doi.org/10.1002 aic.16489.

Venkatasubramanian, V., Rengaswamy, R., Yin, K., Ka, S.N., 2003. A review of process fault detection and diagnosis Part I: quantitative model-based methods. Comput. Chem. Eng., 19.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S., 2020. A comprehensive survey on graph neural networks. IEEE Trans. Neural Netw. Learn. Syst., 1–21, http:// dx.doi.org/10.1109/TNNLS.2020.2978386.

Wu, H., Zhao, J., 2018. Deep convolutional neural network model based chemical process fault diagnosis. Comput. Chem. Eng. 115, 185–197, http://dx.doi.org/10. 1016/j.compchemeng.2018.04.009.

Xie, D., Bai, L., 2015. A hierarchical deep neural network for fault diagnosis on Tennessee-eastman process. 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA). Presented at the 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), 745–748, http://dx.doi.org/10.1109/ICMLA.2015.208.

Yin, S., Ding, S.X., Haghani, A., Hao, H., Zhang, P., 2012. A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process. J. Process Control 22, 1567–1581, http://dx.doi.org/ 10.1016/j.jprocont.2012.06.009.

Yu, H., Khan, F., Garaniya, V., 2015. Modified independent component analysis and bayesian network-based two-stage fault diagnosis of process operations. Ind. Eng. Chem. Res. 54, 2724–2742, http://dx.doi.org/10.1021/ie503530v

Zadakbar, O., Imtiaz, S., Khan, F., 2013. Dynamic risk assessment and fault detection using principal component analysis. Ind. Eng. Chem. Res. 52, 809–816, http:// dx.doi.org/10.1021/ie202880w.

Zhang, S., Bi, K., Qiu, T., 2020a. Bidirectional recurrent neural network-based chemical process fault diagnosis. Ind. Eng. Chem. Res. 59, 824–834, http://dx.doi.org/ 10.1021/acs.iecr.9b05885.

Zhang, Z., Cui, P., Zhu, W., 2020b. Deep learning on graphs: a survey. IEEE Trans. Knowl. Data Eng., http://dx.doi.org/10.1109/TKDE.2020.2981333, 1–1.

Zhang, Z., Zhao, J., 2017. A deep belief network based fault diagnosis model for complex chemical processes. Comput. Chem. Eng. 107, 395–407, http://dx.doi.org/ 10.1016/j.compchemeng.2017.02.041.

Zheng, S., Zhao, J., 2020. A new unsupervised data mining method based on the stacked autoencoder for chemical process fault diagnosis. Comput. Chem. Eng. 135, 106755, http://dx.doi.org/10.1016/j.compchemeng.2020.106755.