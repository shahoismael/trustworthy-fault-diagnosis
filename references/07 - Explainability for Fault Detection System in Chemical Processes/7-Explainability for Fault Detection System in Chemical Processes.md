# Explainability for Fault Detection System in Chemical Processes

Georgios Gravanis<sup>a,c</sup>, Dimitrios Kyriakou<sup>a</sup>, Spyros Voutetakis<sup>c</sup>, Simira Papadopoulou<sup>b,c</sup>, Konstantinos Diamantaras<sup>a</sup>

<sup>a</sup>Department of Information and Electronic Engineering, International Hellenic University, Greece

<sup>b</sup>Department of Industrial Engineering and Management, International Hellenic University, Greece

<sup>c</sup>Chemical Process and Energy Resources Institute, Centre for Research and Technology Hellas, Greece

## Abstract

In this work, we apply and compare two state-of-the-art eXplainability Artificial Intelligence (XAI) methods, the Integrated Gradients (IG) and the SHapley Additive exPlanations (SHAP), that explain the fault diagnosis decisions of a highly accurate Long Short-Time Memory (LSTM) classifier. The classifier is trained to detect faults in a benchmark non-linear chemical process, the Tennessee Eastman Process (TEP). It is highlighted how XAI methods can help identify the subsystem of the process where the fault occurred. Using our knowledge of the process, we note that in most cases the same features are indicated as the most important for the decision, while in some cases the SHAP method seems to be more informative and closer to the root cause of the fault. Finally, since the used XAI methods are modelagnostic, the proposed approach is not limited to the specific process and can also be used in similar problems.

## Keywords:

Explainable Deep Learning, Neural Networks, Fault Detection, Chemical Processes Explainability

## 1. Introduction

During the so-called 4th Industrial revolution we are going through, new technologies including Augmented Reality, Big Data mining, and Deep Learning are used to optimize the operation pipelines of production facilities. Deep Learning (DL) is highly applied with the digitization of industries and is used to enhance technologies such as Image Recognition for quality assurance (Deshpande et al. (2020); Yang et al. (2020)), production variables forecasting (Brunelli et al. (2019)), and Fault Detection and Diagnosis (Abid et al. (2021); Dai & Gao (2013); Saufi et al. (2019)).

However, there are still open issues concerning the adaptation of such solutions to real-life working environments. That is because the way insights are produced by such technologies is not transparent to the end users. Subsequently, the trustworthiness of the information produced by Deep Learning models is under question by the end users.

To tackle this issue, several algorithms have been developed to provide insights into Deep Learning model decisions. This relatively new research area is named eXplainable Artificial Intelligence (XAI) Arrieta et al. (2020). XAI methods mainly focus on Image classification algorithms, for example, in medical applications where images are the main diagnostic tool for several diseases (van der Velden et al., 2022), and in Natural Language Process (NLP) tasks Danilevsky et al. (2020).

However, little research has been conducted for developing XAI methods that focus on DL models handling multivariate time series data. As this research domain is new, there are not many works published implementing explainability methods for chemical processes. Next, we present the works most related to this one.

Agarwal et al. (2021) propose the use of an autoencoder along with the Layerwise Relevance Propagation (LRP) algorithm to enhance the accuracy of a Fault Detection and Diagnosis (FDD) framework. The authors use a version of TEP to test the proposed architecture, with good results.

Another approach for the explainability of a DNN in chemical processes is that of Wu & Zhao (2021). The authors in their study present a Process Topology Convolutional Network (PTCN) that is based on Graph Convolutional Networks (GCN) in order to improve classification accuracy in a more transparent way. Both works propose architectures that enhance transparency and accuracy. However, the question still remains the same: Why the classifier made its decision?

Bhakte et al. (2022) uses the SHAP explainability method over a DNN on TEP but only for a few faults i.e. IDV1, IDV2, IDV4, IDV5, IDV7, and IDV 14 (see Table 2). For those faults, most FDD frameworks achieve great accuracy results since, in general, they are easily recognizable.

With this work, we propose an approach to establish trustworthiness between the end user and the machine learning model decisions. We focus on ambiguous results and we explain the decisions of an FDD applied to a complex chemical process. Moreover, we compare two XAI methods and we validate the results’ plausibility according to the physical interpretation of each fault. The primary motivation for this work is described by the two following Research Questions (RQ):

• RQ1: Do the XAI methods results reach an agreement in the explaining variables?

• RQ2: Are the XAI results reasonable when applied to a chemical process time series data?

Next, we describe how this paper is organized: Section 2 briefly describes the methods used in this study to explain the deep learning algorithm decisions, whereas section 3 describes the benchmark TE process with all its variables that are used as a case study for this work. Finally, in section 4 we present the experimental procedure and the evaluation of the results and in Section 5 we summarize the conclusions of this work.

## 2. Explainability methods

According to Rojat et al. (2021), explainability methods can be separated into two main categories, namely the Ante hoc and the Post hoc that in general are applied either with backpropagation algorithms or with feature perturbation. The Ante hoc explainable methods are “embedded” into the Neural Network model algorithms, while the Post hoc methods are applied after the training phase of the models.

What actually diferentiates the two categories is the result interpretation ability of non-experts. In the first case, when Ante hoc methods are applied to the Neural Network models, the results are mostly useful to the Machine Learning Engineer with the insights provided being valuable for algorithm optimization. On the contrary, Post hoc methods are more generic and aim to produce information about the decision of the Neural Network model that can be useful to a domain expert (the end-user) that utilizes AI to solve a specific problem.

In this work, we evaluate the explanations of two of the most popular post hoc methods namely SHAP and IG. That is because we target to build trustworthiness between the end user and the DNN model. Next, a short description of the two methods namely IG and SHAP will be presented.

## 2.1. Integrated Gradients

Integrated Gradients (IG) is a method for attributing the prediction of a neural network to its input features and it is mostly used to explain the decisions of image classifiers. The method was first introduced by Sundararajan et al. (2017) and it belongs to the Post-hoc explainability methods. Thus, it can be applied to pre-trained ML models without any limitation on the model architecture.

According to the authors, IG ( eq. 1) complies with two basic axioms that explainability methods should fulfill, namely the Sensitivity and the Implementation Invariance. Sensitivity axiom is satisfied if an attribution method produces a non-zero attribution score when there are diferent predictions for every input and baseline that difer in one feature. Implementation Invariance axiom describes that an explainability method should produce identical attributions when two diferent Neural Networks have the same results.

$$
I n t e g r a t e d G r a d s _ {i} (x) := (x _ {i} - x _ {i} ^ {\prime}) \times \int_ {a = 0} ^ {1} \frac {\partial F (x ^ {\prime} + a \times (x - x ^ {\prime}))}{\partial x _ {i}} d a\tag{1}
$$

IG attributes an importance score to each feature $x _ { i }$ by accumulating gradients between the current input and a baseline value.

## 2.2. SHapley Additive exPlanations

SHAP method for explaining machine learning model decisions was introduced by Lundberg & Lee (2017). Equation 2 describes how SHAP method calculates the contribution score for each feature i,

$$
\phi_ {i} (f, x) = \sum_ {z ^ {\prime} \subseteq x ^ {\prime}} \frac {| z ^ {\prime} | ! (M - | z ^ {\prime} | - 1) !}{M !} [ f _ {x} (z ^ {\prime}) - f _ {x} (z ^ {\prime} \backslash i) ]\tag{2}
$$

where f is the model function, x is the input vector, M is the total number of features, $\left| z ^ { \prime } \right|$ is the number of non-zero entries in $z ^ { \prime } ,$ , and $z ^ { \prime } \subseteq x ^ { \prime }$ represents all $z ^ { \prime }$ where the non-zero entries are a subset of the non-zero entries in $x ^ { \prime }$

Briefly, the method estimates the contribution of each feature to the decision of any machine-learning model, by using Shapley values from game theory. This categorizes SHAP as a post hoc explainability method because it can be implemented after the training phase of any AI model.

## 3. TEP: A base case study

The Tennessee Eastman Process is a benchmark problem originally introduced by Downs & Vogel (1993), while an updated version was introduced recently by Bathelt et al. (2015). Figure 1 depicts the Piping and Instrumentation Diagram (P&ID). The variables and the faults of the process are described in Table 1 & Table 2, respectively. The initial goal of TEP was to provide a case study for the development and optimization of control methods and strategies. However, during the AI explosion era, TEP is utilized to implement frameworks for fault detection in chemical processes. (Zhang & Zhao (2017); Wu & Zhao (2018); Zhang et al. (2019)).

This work aims to explain such DL model decisions. To achieve that, we use a highly accurate model introduced in our work Gravanis et al. (2022), and we apply two of the most prominent XAI methods such as IG and SHAP in that model. Next, the experimental procedure and the results are presented.

![](images/08b62f66d6d2494cd4bbd7a27dbbddb97bbf951fe229f7d876b3dd68440bf6a4.jpg)  
Figure 1: Tennessee Eastman Process Piping and Instrumentation Diagram.

Table 1: TEP variables.

<table><tr><td>Variable</td><td>Description</td><td>Units</td><td>Variable</td><td>Description</td><td>Units</td></tr><tr><td>xmeas 1</td><td>A Feed (Stream 1)</td><td> $kSm^{3}/hr$ </td><td>xmeas 27</td><td>Component E</td><td>Stream 6</td></tr><tr><td>xmeas 2</td><td>D Feed (Stream 2)</td><td>kg/hr</td><td>xmeas 28</td><td>Component F</td><td>Stream 6</td></tr><tr><td>xmeas 3</td><td>E Feed (Stream 3)</td><td>kg/hr</td><td>xmeas 29</td><td>Component A</td><td>Stream 9</td></tr><tr><td>xmeas4</td><td>Total Feed (Stream 4)</td><td> $kSm^{3}/hr$ </td><td>xmeas 30</td><td>Component B</td><td>Stream 9</td></tr><tr><td>xmeas 5</td><td>Recycle Flow(Stream 8)</td><td> $kSm^{3}/hr$ </td><td>xmeas 31</td><td>Component C</td><td>Stream 9</td></tr><tr><td>xmeas 6</td><td>Reactor Feed Rate (Stream 6)</td><td> $kSm^{3}/hr$ </td><td>xmeas 32</td><td>Component D</td><td>Stream 9</td></tr><tr><td>xmeas 7</td><td>Reactor Pressure</td><td>kPa gauge</td><td>xmeas 33</td><td>Component E</td><td>Stream 9</td></tr><tr><td>xmeas 8</td><td>Reactor Level</td><td>%</td><td>xmeas 34</td><td>Component F</td><td>Stream 9</td></tr><tr><td>xmeas 9</td><td>Reactor Temperature</td><td>°C</td><td>xmeas 35</td><td>Component G</td><td>Stream 9</td></tr><tr><td>xmeas 10</td><td>Purge Rate (Stream 9)</td><td>kscmh</td><td>xmeas 36</td><td>Component H</td><td>Stream 9</td></tr><tr><td>xmeas 11</td><td>Product Sep Temp</td><td>°C</td><td>xmeas 37</td><td>Component D</td><td>Stream 11</td></tr><tr><td>xmeas 12</td><td>Product Sep Level</td><td>%</td><td>xmeas 38</td><td>Component E</td><td>Stream 11</td></tr><tr><td>xmeas 13</td><td>Product Sep Pressure</td><td>kPa gauge</td><td>xmeas 39</td><td>Component F</td><td>Stream 11</td></tr><tr><td>xmeas 14</td><td>Product Sep Underflow (Stream 10)</td><td> $m^{3}/hr$ </td><td>xmeas 40</td><td>Component G</td><td>Stream 11</td></tr><tr><td>xmeas 15</td><td>Stripper Level</td><td>%</td><td>xmeas 41</td><td>Component H</td><td>Stream 11</td></tr><tr><td>xmeas 16</td><td>Stripper Pressure</td><td>kPa gauge</td><td>xmv 1</td><td>D Feed (Stream 2)</td><td>kg/hr</td></tr><tr><td>xmeas 17</td><td>Stripper Underflow (Stream 11)</td><td> $m^{3}/hr$ </td><td>xmv 2</td><td>E Feed (Stream 3)</td><td>kg/hr</td></tr><tr><td>xmeas 18</td><td>Stripper Temperature</td><td>°C</td><td>xmv 3</td><td>A Feed (Stream 1)</td><td> $kSm^{3}/hr$ </td></tr><tr><td>xmeas 19</td><td>Stripper Steam Flow</td><td>kg/hr</td><td>xmv 4</td><td>Total Feed (Stream 4)</td><td> $kSm^{3}/hr$ </td></tr><tr><td>xmeas 20</td><td>Compressor Work</td><td>kW</td><td>xmv 5</td><td>Compressor Recycle Valve</td><td>%</td></tr><tr><td>xmeas 21</td><td>Reactor Cooling Water Outlet Temp</td><td>°C</td><td>xmv 6</td><td>Purge Valve (Stream 9)</td><td> $kSm^{3}/hr$ </td></tr><tr><td>xmeas 22</td><td>Separator Cooling Water Outlet Temp</td><td>°C</td><td>xmv 7</td><td>Separator Pot Liquid Flow (Stream 10)</td><td> $m^{3}/hr$ </td></tr><tr><td>xmeas 23</td><td>Component A</td><td>Stream 6</td><td>xmv 8</td><td>Stripper Liquid Product Flow (Stream 11)</td><td> $m^{3}/hr$ </td></tr><tr><td>xmeas 24</td><td>Component B</td><td>Stream 6</td><td>xmv 9</td><td>Stripper Steam Valve</td><td>%</td></tr><tr><td>xmeas 25</td><td>Component C</td><td>Stream 6</td><td>xmv 10</td><td>Reactor Cooling Water Flow</td><td> $kSm^{3}/hr$ </td></tr><tr><td>xmeas 26</td><td>Component D</td><td>Stream 6</td><td>xmv 11</td><td>Condenser Cooling Water Flow</td><td>°C</td></tr><tr><td></td><td></td><td></td><td>xmv 12</td><td>Agitator Speed</td><td>%</td></tr></table>

<sup>1</sup> Units of Composition measurements are mole %

Table 2: Predefined disturbances in TEP.

<table><tr><td>Disturbance</td><td>Description</td><td>Type</td></tr><tr><td>IDV 1</td><td>A/C feed ratio, B composition constant (Stream 4)</td><td>step</td></tr><tr><td>IDV 2</td><td>B composition, A/C ratio constant (Stream 4)</td><td>step</td></tr><tr><td>IDV 3</td><td>D feed temperature (Stream 2)</td><td>step</td></tr><tr><td>IDV 4</td><td>reactor cooling water inlet temperature</td><td>step</td></tr><tr><td>IDV 5</td><td>condenser cooling water inlet temperature</td><td>step</td></tr><tr><td>IDV 6</td><td>A feed loss (Stream 1)</td><td>step</td></tr><tr><td>IDV 7</td><td>C header pressure loss - Reduced availability (Stream 4)</td><td>step</td></tr><tr><td>IDV 8</td><td>A, B, C, feed composition (Stream 4)</td><td>random</td></tr><tr><td>IDV 9</td><td>D feed temperature (Stream 2)</td><td>random</td></tr><tr><td>IDV 10</td><td>C feed temperature (Stream 4)</td><td>random</td></tr><tr><td>IDV 11</td><td>Reactor cooling water inlet temperature</td><td>random</td></tr><tr><td>IDV 12</td><td>Condenser cooling water inlet temperature</td><td>random</td></tr><tr><td>IDV 13</td><td>Reaction kinetics</td><td>slow drift</td></tr><tr><td>IDV 14</td><td>Reactor cooling water valve</td><td>sticking</td></tr><tr><td>IDV 15</td><td>Condenser cooling water valve</td><td>sticking</td></tr><tr><td>IDV 16</td><td>(unknown) Deviations of heat transfer within stripper (heat exchanger)</td><td>random</td></tr><tr><td>IDV 17</td><td>(unknown) Deviations of heat transfer within reactor</td><td>random</td></tr><tr><td>IDV 18</td><td>(unknown) Deviations of heat transfer within condenser</td><td>random</td></tr><tr><td>IDV 19</td><td>(unknown) re-cycle valve of compressor, underflow separator (stream 10), underflow stripper (stream 11) and steam valve stripper</td><td>sticking</td></tr><tr><td>IDV 20</td><td>Unknown</td><td>random</td></tr></table>

## 4. Explainability experiments and results

As described previously, this work applies two post hoc explainability methods to a state-of-the-art LSTM architecture model developed for a multiclass and multivariate time series classification problem. The LSTM model architecture for Fault Detection and Diagnosis is described thoroughly in our previous work Gravanis et al. (2022)

To ease the evaluation process, we present the results of the explainability methods for the first 100 samples after the introduction of the disturbance in the process. Since process control is implemented in TEP, we can consider that measurements follow logical sequences, without large variations between neighbor ones. With that consideration in mind, we averaged the attribution scores for each feature for a given number of input sequences.

In order to be able to compare the two methods, we normalized the results to identify the highly attributed features. Those features are the ones that the explainability methods indicate as important for the decision of the classifier. The results of this process are displayed in Figure 2 in a qualitative way. We have to note that as proved in our previous work, not all features are important for the classifier to make its decision. To ease the reader, we excluded those features from the representation of the results.

![](images/e3e989ee91d116b32e3fadc7cee72d038ba2106d8756f143762925baef27c39d.jpg)  
Figure 2: Heatmap with the most contributing features. The darker the color, the higher the contribution. A columns for IG , B columns for SHAP

## 4.1. Result evaluation

To evaluate the results, we grouped the faults according to the part of the process that is expected to be directly afected (Table 3). With this grouping, we can recognize the variables that are most probable to present variation when the system is under disturbance compared to the normal operation.

Table 3: TEP disturbances grouped by the area afected.

<table><tr><td>Disturbance</td><td>Description</td><td>type</td><td>Part of the process affected</td></tr><tr><td>IDV 3</td><td>D feed temperature (Stream 2)</td><td>step</td><td></td></tr><tr><td>IDV 4</td><td>Reactor cooling water inlet temperature</td><td>step</td><td></td></tr><tr><td>IDV 9</td><td>D feed temperature (Stream 2)</td><td>random</td><td></td></tr><tr><td>IDV 10</td><td>C feed temperature (Stream 4)</td><td>random</td><td>Reactor (direct)</td></tr><tr><td>IDV 11</td><td>Reactor cooling water inlet temperature</td><td>random</td><td></td></tr><tr><td>IDV 14</td><td>Reactor cooling water valve</td><td>sticking</td><td></td></tr><tr><td>IDV 17</td><td>Deviations of heat transfer within reactor</td><td>random</td><td></td></tr><tr><td>IDV 2</td><td>B composition, A/C ratio constant (Stream 4)</td><td>step</td><td></td></tr><tr><td>IDV 8</td><td>A, B, C, feed composition (Stream 4)</td><td>random</td><td></td></tr><tr><td>IDV 13</td><td>Reaction kinetics</td><td>slow drift</td><td>Reactor (indirect)</td></tr><tr><td>IDV 5</td><td>Condenser cooling water inlet temperature</td><td>step</td><td></td></tr><tr><td>IDV 12</td><td>Condenser cooling water inlet temperature</td><td>random</td><td></td></tr><tr><td>IDV 18</td><td>Deviations of heat transfer within condenser</td><td>random</td><td></td></tr><tr><td>IDV 1</td><td>A/C feed ratio, B composition constant (Stream 4)</td><td>step</td><td>Stream 4 composition</td></tr><tr><td>IDV 6</td><td>A feed loss (Stream 1)</td><td>step</td><td>Stream 1 Feed</td></tr><tr><td>IDV 7</td><td>C header pressure loss - Reduced availability (Stream 4)</td><td>step</td><td>Stream 4 Feed</td></tr><tr><td>IDV 19</td><td>Recycle valve of the compressor, underflow separator (stream 10),underflow stripper (stream 11), and steam valve stripper</td><td>sticking</td><td>Multiple areas</td></tr><tr><td>IDV 20</td><td>Unknown</td><td>random</td><td>unknown</td></tr></table>

Given that the process is under control with the strategy defined and implemented by Bathelt et al. (2015), it is normal for both the classifier and the explanation method to recognize important features both in measured and manipulated values.

Another important observation is that the stability of the system, which is of high complexity, is controlled mostly through the operation of its main component i.e. the reactor. Subsequently, for most disturbances, the implemented control strategy takes countermeasures that afect the temperature and the operation of the reactor.

Table 4: Reactor highly afected variables.

<table><tr><td>Variable</td><td>Description</td></tr><tr><td>xmeas 9</td><td>Reactor temperature</td></tr><tr><td>xmeas 21</td><td>Reactors&#x27; cooling water temperature</td></tr><tr><td>xmv 10</td><td>Reactors&#x27; cooling water valve operation</td></tr></table>

That is clearly depicted in Figure 2 where it is shown that for faults that afect the reactor, there are three features (Table 4) that according to XAI methods play a crucial role in classifier decisions.

For the other faults that cannot be grouped into a specific category, the results are also reasonable. For example, when there is A feed loss (IDV 6), the methods indicate variable xmv 3 that actually controls the operation of the feeding valve for reactant A, when there is pressure loss in C header (IDV 7) they indicate xmv 4 that is the valve for reactant C, etc.

Another important observation from the results displayed in Figure 2, is that there is some diferentiation between IG and SHAP methods in faults IDV 8, IDV 12, IDV 18 & IDV 20. After a careful examination of the results, this diferentiation indicates that the SHAP method might be more informative compared to IG. Next, the results of two diferent example cases will be explained. More specifically, we will focus on the following two cases:

• the IDV 11 case which is a fault that directly afects the operation of the reactor heat exchanger subsystem and the XAI methods produced the same results

• the IDV 8 case where the fault afects mostly the reaction of the process, and there is some diferentiation between the methods as described before.

## 4.2. IDV 11 example case

In this case, the fault is a step variation of the cool water in the inlet of the reactor’s heat exchanger (Figure 3). The classifier recognizes the fault with 99% accuracy, while both XAI methods indicate as most important for the classifier decisions the features xmeas 9, xmeas 21 & xmv 10 (Table 5 & Figure 4). As displayed in Figure 5 the efect of the fault is clearly recognizable to the most important variables.

![](images/0f6cbc724100d9f3af8d12c3115d5dbdf5f81645de159fda6a3d0e904b805c96.jpg)  
Figure 3: IG and SHAP most important features for IDV 11.

## 4.3. IDV 8 example case

In the case of IDV 8 (Figure 6), the fault is a random variation in the composition of the reactants of stream 4, and the classifier achieves 99% accuracy. Here there is some diferentiation between the methods since SHAP indicates xmeas 20 & xmeas 22 as the most important features while for IG the most attributed features from IG are xmeas 21 & xmv 10. As shown in Figure 7 all four variables have a relatively large diferentiation from the normal operation. However, the variables that the SHAP method indicates as the most important, have greater error variation compared to the variables indicated by the IG method (Figure 8). In general, the error is expected to be greater for variables belonging to subsystems adjacent to the root cause, while the error of variables which are further away is absorbed by the control system of the process. This could be an indication that the SHAP method in some cases may be more informative than the IG.

Table 5: Fault 11 IG and SHAP attribution scores after normalization.

<table><tr><td>feature</td><td>IG</td><td>SHAP</td><td>feature</td><td>IG</td><td>SHAP</td><td>feature</td><td>IG</td><td>SHAP</td><td>feature</td><td>IG</td><td>SHAP</td></tr><tr><td>xmeas_1</td><td>-0.48</td><td>0.08</td><td>xmeas_15</td><td>-0.21</td><td>-0.24</td><td>xmeas_29</td><td>-0.20</td><td>-0.29</td><td>xmv_1</td><td>-0.40</td><td>-0.32</td></tr><tr><td>xmeas_2</td><td>-0.19</td><td>-0.32</td><td>xmeas_16</td><td>-0.08</td><td>-0.11</td><td>xmeas_30</td><td>-0.36</td><td>-0.19</td><td>xmv_2</td><td>-0.22</td><td>-0.23</td></tr><tr><td>xmeas_3</td><td>-0.25</td><td>-0.33</td><td>xmeas_17</td><td>-0.20</td><td>-0.37</td><td>xmeas_31</td><td>-0.19</td><td>-0.30</td><td>xmv_3</td><td>-0.74</td><td>0.01</td></tr><tr><td>xmeas_4</td><td>-0.16</td><td>-0.16</td><td>xmeas_18</td><td>0.01</td><td>-0.10</td><td>xmeas_32</td><td>-0.18</td><td>-0.32</td><td>xmv_4</td><td>-1.24</td><td>0.39</td></tr><tr><td>xmeas_5</td><td>-0.16</td><td>-0.27</td><td>xmeas_19</td><td>-0.26</td><td>-0.33</td><td>xmeas_33</td><td>-0.15</td><td>-0.31</td><td>xmv_5</td><td>-0.20</td><td>-0.32</td></tr><tr><td>xmeas_6</td><td>-0.21</td><td>-0.25</td><td>xmeas_20</td><td>-0.15</td><td>-0.08</td><td>xmeas_34</td><td>-0.28</td><td>-0.14</td><td>xmv_6</td><td>-0.17</td><td>-0.15</td></tr><tr><td>xmeas_7</td><td>0.01</td><td>-0.29</td><td>xmeas_21</td><td>2.11</td><td>0.65</td><td>xmeas_35</td><td>-0.20</td><td>-0.38</td><td>xmv_7</td><td>-0.14</td><td>0.09</td></tr><tr><td>xmeas_8</td><td>-0.16</td><td>-0.33</td><td>xmeas_22</td><td>0.23</td><td>0.35</td><td>xmeas_36</td><td>-0.23</td><td>-0.34</td><td>xmv_8</td><td>-0.18</td><td>0.09</td></tr><tr><td>xmeas_9</td><td>4.34</td><td>5.74</td><td>xmeas_23</td><td>-0.18</td><td>-0.34</td><td>xmeas_37</td><td>-0.21</td><td>-0.33</td><td>xmv_9</td><td>-0.20</td><td>-0.32</td></tr><tr><td>xmeas_10</td><td>-0.18</td><td>-0.19</td><td>xmeas_24</td><td>-0.20</td><td>-0.23</td><td>xmeas_38</td><td>-0.22</td><td>-0.30</td><td>xmv_10</td><td>5.01</td><td>4.00</td></tr><tr><td>xmeas_11</td><td>-0.30</td><td>0.11</td><td>xmeas_25</td><td>-0.16</td><td>-0.33</td><td>xmeas_39</td><td>-0.20</td><td>-0.29</td><td>xmv_11</td><td>-0.26</td><td>-0.20</td></tr><tr><td>xmeas_12</td><td>-0.19</td><td>-0.32</td><td>xmeas_26</td><td>-0.15</td><td>-0.31</td><td>xmeas_40</td><td>-0.20</td><td>-0.34</td><td>xmv_12</td><td>-0.20</td><td>-0.32</td></tr><tr><td>xmeas_13</td><td>-0.53</td><td>-0.12</td><td>xmeas_27</td><td>-0.20</td><td>-0.29</td><td>xmeas_41</td><td>-0.14</td><td>-0.28</td><td></td><td></td><td></td></tr><tr><td>xmeas_14</td><td>-0.18</td><td>-0.31</td><td>xmeas_28</td><td>-0.24</td><td>-0.23</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/fe51f57f7dc1517fbbb6648074ccd271daea2a23479923ba4e12955f78c2b078.jpg)  
Figure 4: IG attributions for IDV 11. Left: most important features based on IG score, right: less important features based on IG score.

![](images/54fa44a3d87e4d7d6c2e72a7fcc3fcafe905eb61e83ba964f5cf31c0ca9f782c.jpg)  
Figure 5: Most and less contributing variable behavior plots for Fault 11.

Table 6: Fault 8 IG and SHAP attribution scores after normalization.

<table><tr><td>feature</td><td>IG</td><td>SHAP</td><td>feature</td><td>IG</td><td>SHAP</td><td>feature</td><td>IG</td><td>SHAP</td><td>feature</td><td>IG</td><td>SHAP</td></tr><tr><td>xmeas_1</td><td>-0.27</td><td>0.98</td><td>xmeas_15</td><td>-0.37</td><td>-0.22</td><td>xmeas_29</td><td>0.80</td><td>-0.03</td><td>xmv_1</td><td>-0.60</td><td>-0.77</td></tr><tr><td>xmeas_2</td><td>-0.20</td><td>-0.39</td><td>xmeas_16</td><td>0.93</td><td>0.33</td><td>xmeas_30</td><td>-0.31</td><td>-0.31</td><td>xmv_2</td><td>-0.34</td><td>-0.32</td></tr><tr><td>xmeas_3</td><td>-0.28</td><td>-0.34</td><td>xmeas_17</td><td>-0.29</td><td>-0.40</td><td>xmeas_31</td><td>0.57</td><td>-0.18</td><td>xmv_3</td><td>-0.79</td><td>0.15</td></tr><tr><td>xmeas_4</td><td>-0.26</td><td>-0.36</td><td>xmeas_18</td><td>0.27</td><td>-3.19</td><td>xmeas_32</td><td>-0.19</td><td>-0.59</td><td>xmv_4</td><td>-1.40</td><td>0.48</td></tr><tr><td>xmeas_5</td><td>-0.20</td><td>0.18</td><td>xmeas_19</td><td>-0.18</td><td>-0.43</td><td>xmeas_33</td><td>-0.15</td><td>-0.37</td><td>xmv_5</td><td>-0.18</td><td>-0.41</td></tr><tr><td>xmeas_6</td><td>-0.31</td><td>0.32</td><td>xmeas_20</td><td>-0.18</td><td>5.04</td><td>xmeas_34</td><td>-0.40</td><td>-0.06</td><td>xmv_6</td><td>0.38</td><td>-0.22</td></tr><tr><td>xmeas_7</td><td>0.38</td><td>0.38</td><td>xmeas_21</td><td>6.37</td><td>1.60</td><td>xmeas_35</td><td>-0.22</td><td>-0.37</td><td>xmv_7</td><td>-0.29</td><td>0.74</td></tr><tr><td>xmeas_8</td><td>0.03</td><td>-0.35</td><td>xmeas_22</td><td>-0.24</td><td>2.28</td><td>xmeas_36</td><td>-0.33</td><td>-0.49</td><td>xmv_8</td><td>-0.50</td><td>0.98</td></tr><tr><td>xmeas_9</td><td>-0.39</td><td>0.93</td><td>xmeas_23</td><td>0.39</td><td>-0.13</td><td>xmeas_37</td><td>-0.12</td><td>-0.39</td><td>xmv_9</td><td>-0.18</td><td>-0.41</td></tr><tr><td>xmeas_10</td><td>0.44</td><td>-0.37</td><td>xmeas_24</td><td>-0.18</td><td>-0.27</td><td>xmeas_38</td><td>-0.20</td><td>-0.17</td><td>xmv_10</td><td>1.62</td><td>0.85</td></tr><tr><td>xmeas_11</td><td>-1.31</td><td>-0.12</td><td>xmeas_25</td><td>0.43</td><td>0.09</td><td>xmeas_39</td><td>-0.17</td><td>-0.47</td><td>xmv_11</td><td>-0.25</td><td>-0.53</td></tr><tr><td>xmeas_12</td><td>-0.12</td><td>-0.37</td><td>xmeas_26</td><td>-0.10</td><td>-0.37</td><td>xmeas_40</td><td>-0.19</td><td>-0.40</td><td>xmv_12</td><td>-0.18</td><td>-0.41</td></tr><tr><td>xmeas_13</td><td>0.16</td><td>0.04</td><td>xmeas_27</td><td>-0.14</td><td>-0.18</td><td>xmeas_41</td><td>-0.16</td><td>-0.42</td><td></td><td></td><td></td></tr><tr><td>xmeas_14</td><td>-0.23</td><td>-0.39</td><td>xmeas_28</td><td>-0.36</td><td>-0.16</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/11c6aeb971e082d9211ab24d1599b82ac0ee960226355e9dd95ddc77b54dd447.jpg)

Figure 6: IG and SHAP most important features for IDV 8  
![](images/af70fb861d090a029189c71dd0517232402502551c4653352356368d2baeabfd.jpg)  
Figure 7: IDV 8 most important variables according to IG and SHAP

![](images/379a7e849594da3b58f7d9a4f57f5440a8b3b00b7c83b836ffc93fadaafe8675.jpg)  
Figure 8: IDV 8 most important variables according to IG and SHAP

## 5. Conclusions

In this work two state-of-the-art methods namely IG and SHAP are used to explain the decisions of a highly accurate LSTM model trained to identify faults in a non-linear chemical process. The results of both methods are reasonable and in most cases, they agree in the list with the same features as the most important. In some cases i.e. IDV 8, IDV 12 & IDV 18, the SHAP method seems to be more informative than IG. However, both IG and SHAP showed eficacy and consistency in producing valuable insights and in enlightening any obscure points on the decisions of Deep Learning models in a chemical process application. Finally, since the proposed approach uses Post - hoc XAI methods, it can be adapted and used in a large variety of industrial and chemical process applications.

## References

Abid, A., Khan, M. T., & Iqbal, J. (2021). A review on fault detection and diagnosis techniques: basics and beyond. Artificial Intelligence Review, 54 , 3639–3664.

Agarwal, P., Tamer, M., & Budman, H. (2021). Explainability: Relevance based dynamic deep learning algorithm for fault detection and diagnosis in chemical processes. Computers & Chemical Engineering, 154 , 107467.

Arrieta, A. B., D´ıaz-Rodr´ıguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., Garc´ıa, S., Gil-L´opez, S., Molina, D., Benjamins, R. et al. (2020). Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai. Information fusion, 58 , 82–115.

Bathelt, A., Ricker, N. L., & Jelali, M. (2015). Revision of the tennessee eastman process model. IFAC-PapersOnLine, 48 , 309 – 314. URL: http:// www.sciencedirect.com/science/article/pii/S2405896315010666. doi:https://doi.org/10.1016/j.ifacol.2015.08.199. 9th IFAC Symposium on Advanced Control of Chemical Processes ADCHEM 2015.

Bhakte, A., Pakkiriswamy, V., & Srinivasan, R. (2022). An explainable artificial intelligence based approach for interpretation of fault classification results from deep neural networks. Chemical Engineering Science, 250 , 117373.

Brunelli, L., Masiero, C., Tosato, D., Beghi, A., & Susto, G. A. (2019). Deep learning-based production forecasting in manufacturing: a packaging equipment case study. Procedia Manufacturing, 38 , 248–255. URL: https://www.sciencedirect.com/ science/article/pii/S2351978920300342. doi:https://doi.org/10. 1016/j.promfg.2020.01.033. 29th International Conference on Flexible Automation and Intelligent Manufacturing ( FAIM 2019), June 24-28, 2019, Limerick, Ireland, Beyond Industry 4.0: Industrial Advances, Engineering Education and Intelligent Manufacturing.

Dai, X., & Gao, Z. (2013). From model, signal to knowledge: A datadriven perspective of fault detection and diagnosis. IEEE Transactions on Industrial Informatics, 9 , 2226–2238.

Danilevsky, M., Qian, K., Aharonov, R., Katsis, Y., Kawas, B., & Sen, P. (2020). A survey of the state of explainable ai for natural language processing. arXiv preprint arXiv:2010.00711 , .

Deshpande, A. M., Minai, A. A., & Kumar, M. (2020). Oneshot recognition of manufacturing defects in steel surfaces. Procedia Manufacturing, 48 , 1064–1071. URL: https://www.sciencedirect. com/science/article/pii/S2351978920315985. doi:https://doi.org/ 10.1016/j.promfg.2020.05.146. 48th SME North American Manufacturing Research Conference, NAMRC 48.

Downs, J. J., & Vogel, E. F. (1993). A plant-wide industrial process control problem. Computers & Chemical Engineering, 17 , 245–255. doi:10.1016/ 0098-1354(93)80018-I.

Gravanis, G., Dragogias, I., Papakiriakos, K., Ziogou, C., & Diamantaras, K. (2022). Fault detection and diagnosis for non-linear processes empowered by dynamic neural networks. Computers & Chemical Engineering, 156 , 107531. URL: https://www.sciencedirect. com/science/article/pii/S0098135421003094. doi:https://doi.org/ 10.1016/j.compchemeng.2021.107531.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 30 (pp. 4765–4774). Curran Associates, Inc. URL: http://papers.nips.cc/paper/7062-a-unified-approachto-interpreting-model-predictions.pdf.

Rojat, T., Puget, R., Filliat, D., Del Ser, J., Gelin, R., & D´ıaz-Rodr´ıguez, N. (2021). Explainable artificial intelligence (xai) on timeseries data: A survey. arXiv preprint arXiv:2104.00950 , .

Saufi, S. R., Ahmad, Z. A. B., Leong, M. S., & Lim, M. H. (2019). Challenges and opportunities of deep learning models for machinery fault detection and diagnosis: A review. IEEE Access, 7 , 122644–122662. doi:10.1109/ ACCESS.2019.2938227.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. In International conference on machine learning (pp. 3319– 3328). PMLR.

van der Velden, B. H., Kuijf, H. J., Gilhuijs, K. G., & Viergever, M. A. (2022). Explainable artificial intelligence (xai) in deep learning-based medical image analysis. Medical Image Analysis, (p. 102470).

Wu, D., & Zhao, J. (2021). Process topology convolutional network model for chemical process fault diagnosis. Process Safety and Environmental Protection, 150 , 93–109.

Wu, H., & Zhao, J. (2018). Deep convolutional neural network model based chemical process fault diagnosis. Computers & Chemical Engineering, 115 , 185–197.

Yang, J., Li, S., Wang, Z., Dong, H., Wang, J., & Tang, S. (2020). Using deep learning to detect defects in manufacturing: A comprehensive survey and

current challenges. Materials, 13 . URL: https://www.mdpi.com/1996- 1944/13/24/5755. doi:10.3390/ma13245755.

Zhang, S., Bi, K., & Qiu, T. (2019). Bidirectional recurrent neural networkbased chemical process fault diagnosis. Industrial & Engineering Chemistry Research, 59 , 824–834.

Zhang, Z., & Zhao, J. (2017). A deep belief network based fault diagnosis model for complex chemical processes. Computers & Chemical Engineering, 107 , 395–407. doi:10.1016/j.compchemeng.2017.02.041.