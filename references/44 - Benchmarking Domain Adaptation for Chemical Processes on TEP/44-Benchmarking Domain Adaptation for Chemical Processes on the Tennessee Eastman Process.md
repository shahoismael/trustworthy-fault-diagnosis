# Benchmarking Domain Adaptation for Chemical Processes on the Tennessee Eastman Process

Eduardo Fernandes Montesuma<sup>1</sup>, Michela Mulas<sup>2</sup>, Fred Ngolè Mboula<sup>1</sup>, Francesco Corona<sup>3</sup>, and Antoine Souloumiac<sup>1</sup>

<sup>1</sup> Université Paris-Saclay, CEA, LIST, F-91120, Palaiseau, France 2 Department of Teleinformatics Engineering, Federal University of Ceará, Brazil 3 School of Chemical Engineering, Aalto University, Finland

Abstract. In system monitoring, automatic fault diagnosis seeks to infer the systems’ state based on sensor readings, e.g., through machine learning models. In this context, it is of key importance that, based on historical data, these systems are able to generalize to incoming data. In parallel, many factors may induce changes in the data probability distribution, hindering the possibility of such models to generalize. In this sense, domain adaptation is an important framework for adapting models to diferent probability distributions. In this paper, we propose a new benchmark, based on the Tennessee Eastman Process of Downs and Vogel (1993), for benchmarking domain adaptation methods in the context of chemical processes. Besides describing the process, and its relevance for domain adaptation, we describe a series of data processing steps for reproducing our benchmark. We then test 11 domain adaptation strategies on this novel benchmark, showing that optimal transportbased techniques outperform other strategies.<sup>4</sup>.

Keywords: Transfer Learning · Domain Adaptation · Optimal Transport · Tennessee Eastman Process.

## 1 Introduction

Within process supervision, faults are unpermitted deviations of a characteristic property or variables of a system [16]. Furthermore, there is an increasing demand on reliability and safety of technical plants, motivating the necessity of methods for supervision and monitoring. These are Fault Detection and Diagnosis (FDD) methods, which comprise the detection, i.e., if and when a fault has occurred, and the diagnosis, i.e., the determination of which fault has occurred. In this paper, we focus on Automatic Fault Diagnosis (AFD) systems, assuming that faults were previously detected accordingly.

In parallel, Machine Learning (ML) is a field of artificial intelligence, that defines predictive models based on data. Nonetheless, these models make an implicit assumption, that training and test data come from the same probability distribution, which is seldom verified in practice [26], as both training and test data may be collected under heterogeneous conditions that drive shifts in probability distributions. This phenomenon motivates the field of Transfer Learning (TL) [23] to propose algorithms that are robust to distributional shift.

There is a straightforward link between ML and AFD systems, as one can understand fault diagnosis as a classification problem. In this sense, one uses sensor data (e.g., temperature, concentration, flow-rate) as inputs to a classifier, which predicts the corresponding fault, or its absence [34]. Further, TL is a broad field within ML, in which knowledge must be transferred from a source to a target context. Within TL, Domain Adaptation (DA) is a common framework where one has access to labeled data from a source domain, and unlabeled data from a target domain. Thus, DA seeks improving classification accuracy on target domain data. In many cases, source data is itself heterogeneous, following multiple probability distributions. This setting is known as Multi-Source DA (MSDA).

In this paper, we propose a new benchmark, based on the Tennessee Eastman Process (TEP) [7,28], a complex, large-scale chemical process used by the chemical engineering community for benchmarking control systems, as well as FDD techniques. This process is interesting for DA, as it may operate at diferent modes of production. As we show in our case study (section 3), the diferent modes of production induce diferent data probability distributions, thus the need for DA techniques for improving generalization. We further benchmark existing techniques in DA, either based on pre-extracted features (shallow DA), or through deep learning (deep DA).

The rest of this paper is divided as follows. Section 2 covers the theoretical foundations of our work. Section 3 presents a case study of the TEP. In this section, we present the system, analyze the properties of the diferent modes of production, and benchmark diferent strategies in DA. Finally, section 4 concludes this paper.

## 2 Classification and Domain Adaptation

In supervised learning, one is provided with a dataset $\{ \mathbf { x } _ { i } ^ { ( P ) } , y _ { i } ^ { ( P ) } \} _ { i = 1 } ^ { n }$ , where $\mathbf { x } _ { i } ^ { ( P ) } \overset { i i d } { \sim } P ,$ and $y _ { i } ^ { ( P ) } = h _ { 0 } ( \mathbf { x } _ { i } ^ { ( P ) } )$ , for a distribution P and a ground-truth labeling function $h _ { 0 } : \mathcal { X }  \mathcal { Y }$ . X is called feature space, such as $\mathbb { R } ^ { d }$ , and Y label space, in this case $\{ 1 , \cdots , n _ { c } \}$ . The goal of classification is finding, among a family of functions H, h<sup>ˆ</sup> such that,

$$
\hat {h} = \underset {h \in \mathcal {H}} {\mathrm{argmin}} \frac {1}{n} \sum_ {i = 1} ^ {n} \mathcal {L} (h (\mathbf {x} _ {i} ^ {(P)}), y _ {i} ^ {(P)}),\tag{1}
$$

where $\mathcal { L }$ is a loss function, such as the Cross-entropy (CCE), $\mathrm { C C E } ( \mathbf { y } , \hat { \mathbf { y } } ) \ = $ $\textstyle \sum _ { c = 1 } ^ { n _ { c } } y _ { c } \log \hat { y } _ { c }$ . This approach, known as empirical risk minimization, has the desirable property that $\hat { h }$ correctly predicts on unseen samples from P . This property is known as generalization. We refer readers to [27] for a review on the theory of generalization.

In this paper, we consider deep neural nets composed of 2 parts: an encoder network $\phi ,$ and a classifier h. The encoder maps data $\textbf { x } \in { \mathcal { X } }$ into a latent representation z $\in \mathcal { Z }$ , whereas the classifier maps the representation into a label space $\mathcal { V } .$ Hence, $\hat { y } _ { i } ^ { ( P ) } = h ( \phi ( \mathbf { x } _ { i } ^ { ( P ) } ) )$ ). As such, eq. 1 is minimized with respect the parameters of the encoder, $\theta _ { \phi }$ and classifier $\theta _ { h }$

![](images/e8185ad782800c1f3bd5c3e2ed1a95538504d2ef453aad9b67236f2a6c73649d.jpg)  
Fig. 1: Illustration of a deep neural net, where data $\mathbf { x } _ { i } ^ { ( P ) }$ are mapped into latent representation vectors $\mathbf { z } _ { i } ^ { ( P ) }$ through an encoder $\phi .$ The latent representation is then used to predict a class, i.e., $\hat { y } _ { i } ^ { ( P ) }$

The main challenge faced by ML models is generalizing beyond samples from $P .$ In this sense, it is desirable that h<sup>ˆ</sup> generalizes to diferent, but related distributions $Q ,$ , which is known as TL [23]. Within TL, DA is a popular framework where one seeks to improve performance on a target domain based on knowledge available in a source domain. Especially, a domain is a pair $\mathcal { D } = ( \mathcal { X } , P )$ , where $P$ is a distribution the feature space X . Likewise, a task is a pair $\mathcal { T } = ( \mathcal { V } , h _ { 0 } )$ , where $h _ { 0 } : \mathcal { X }  \mathcal { Y }$ is a ground-truth labeling function. Given a source domain and task $( \mathcal { D } _ { S } , \mathcal { T } _ { S } )$ , and a target domain and task, $( \mathcal { D } _ { T } , \mathcal { T } _ { T } )$ , in DA one has $\mathcal { T } _ { S } = \mathcal { T } _ { T }$ , but $P _ { S } \neq P _ { T }$ . As a consequence, $\mathcal { D } _ { S } \ne \mathcal { D } _ { T }$ . The goal of DA can be summarized as follows: given labeled samples from the source domain, and unlabeled samples from the target domain, find a classifier $\hat { h }$ that generalizes to samples from $P _ { T }$

In addition, one may have a scenario where source domain data is heterogeneous. In this case, one assumes that this domain is composed of several distributions, i.e., $P _ { S _ { 1 } } , \cdots , P _ { S _ { N } }$ , for $N > 1$ . This case is known in the literature as MSDA. Besides the challenge of having $P _ { S _ { \ell } } \neq P _ { T }$ , one has inter-domain shifts, i.e., $P _ { S _ { \ell } } \neq P _ { S _ { \ell ^ { \prime } } }$ , for $\ell \neq \ell ^ { \prime }$

Given our discussion so far, one needs a notion of closeness between $P _ { S }$ and $P _ { T }$ for having generalization to new distributions Q [27, Theorem 10]. We thus focus on DA methods that seek to reduce the distance between distributions $P _ { S }$ and $P _ { T }$ through data transformations. In a nutshell, these methods apply a mapping to $\mathbf { x } _ { i } ^ { ( \check { P _ { S } } ) }$ , so that $\{ T ( \mathbf { x } _ { i } ^ { ( P _ { S } ) } ) \} _ { i = 1 } ^ { n }$ is distributed in the same way as $\{ \mathbf { x } _ { j } ^ { ( P _ { T } ) } \}$ . This idea is illustrated in Fig. 2. This alignment supposes a criterion of dissimilarity between these objects. In this sense, one may use probability metrics, which are distances in the space of probability distributions. In our experiments, we consider three prominent metrics, namely, the H-distance, the Maximum Mean Discrepancy (MMD) and the Wasserstein distance.

![](images/7b44bcd7035ebb0b93d14a30455fbd6bf05c1969d46ca2b1b5b20aff3fcb4fd9.jpg)  
Fig. 2: Domain adaptation based on data transformation. In an ambient space, source and target data follow diferent probability distributions. As a result, a classifier learned on the source (blue straight line on the left) is not able to generalize on data from the target domain (orange elements). In this paper we consider methods that align the distributions through a data transformation T , which maps data into a latent space.

The H-distance has its roots on DA theory [2]. This distance measures how likely a classifier can separate samples from these distributions. Hence, let $h \in \mathcal H$ be a classifier,

$$
d _ {\mathcal {H}} (\hat {P} _ {S}, \hat {P} _ {T}) = 2 \left(1 - \min _ {h \in \mathcal {H}} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \log (1 - h (\mathbf {x} _ {i} ^ {(P _ {S})})) + \frac {1}{m} \sum_ {j = 1} ^ {m} \log h (\mathbf {x} _ {j} ^ {(P _ {T})})\right)\right).\tag{2}
$$

Note that the $d _ { \mathcal { H } }$ can be easily estimated from samples $\{ \mathbf { x } _ { i } ^ { ( P _ { S } ) } \} _ { i = } ^ { n }$ and $\{ \mathbf { x } _ { j } ^ { ( P _ { T } ) } \} _ { j = 1 } ^ { m }$ by learning a classifier that predicts the domain of a given sample (e.g., 0 for $P _ { S }$ , and 1 for $P _ { T } )$

The MMD has its roots on kernel theory [13], and was initially proposed to test if two samples come from the same distribution. Let $k : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R }$ be a kernel, the MMD can be defined as,

$$
\begin{array}{c} \mathrm{MMD} _ {k} (\hat {P} _ {S}, \hat {P} _ {T}) ^ {2} = \frac {1}{n ^ {2}} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {n} k (\mathbf {x} _ {i} ^ {(P _ {S})}, \mathbf {x} _ {j} ^ {(P _ {S})}) + \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {m} k (\mathbf {x} _ {i} ^ {(P _ {T})}, \mathbf {x} _ {j} ^ {(P _ {T})}) \\ - \frac {2}{n m} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} k (\mathbf {x} _ {i} ^ {(P _ {S})}, \mathbf {x} _ {j} ^ {(P _ {T})}), \end{array}\tag{3}
$$

examples of kernels include the linear kernel $k ( \mathbf { x } _ { i } ^ { ( P _ { S } ) } , \mathbf { x } _ { j } ^ { ( P _ { T } ) } ) = ( \mathbf { x } _ { i } ^ { ( P _ { S } ) } ) ^ { \top } \mathbf { x } _ { j } ^ { ( P _ { T } ) }$ and the Gaussian kernel, $k ( \mathbf { x } _ { i } ^ { ( P _ { S } ) } , \mathbf { x } _ { i } ^ { ( P _ { T } ) } ) = \exp { ( - \gamma \| \mathbf { x } _ { i } ^ { ( P _ { S } ) } - \mathbf { x } _ { j } ^ { ( P _ { T } ) } \| _ { 2 } ^ { 2 } ) }$ , for a parameter $\gamma > 0$ . Intuitively, the MMD is a distance between the means of distributions in an embedding space defined by the kernel k.

Finally, the Wasserstein distance $W _ { p }$ is rooted on the theory of Optimal Transport (OT). In its modern computational treatment [25,10], the OT problem can be phrased as,

$$
\gamma^ {\star} = \underset {\gamma \in \Gamma} {\operatorname{argmin}} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \gamma_ {i j} \| \mathbf {x} _ {i} ^ {(P _ {S})} - \mathbf {x} _ {j} ^ {(P _ {T})} \| _ {2} ^ {p},\tag{4}
$$

where $\gamma \in \mathbb { R } ^ { n \times m }$ is called OT plan, and $\varGamma$ is the set of mass preserving plans, i.e., matrices γ such that their row sum $\textstyle \sum _ { i = 1 } ^ { n } \gamma _ { i j } \ = \ m ^ { - 1 }$ , and column sum $\begin{array} { r } { \sum _ { j = 1 } ^ { m } \gamma _ { i j } = n ^ { - 1 } } \end{array}$ . Problem 4 is a linear program, which can be solved exactly through the Simplex method [6]. Based on $\gamma ^ { \star }$ , the Wasserstein distance is

$$
W _ {p} (\hat {P} _ {S}, \hat {P} _ {T}) ^ {p} = \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \gamma_ {i j} ^ {\star} \| \mathbf {x} _ {i} ^ {(P _ {S})} - \mathbf {x} _ {j} ^ {(P _ {T})} \| _ {2} ^ {p}.
$$

Let $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ denote the Gaussian distribution with mean $\mu \in \mathbb { R } ^ { d }$ , and covariance matrix $\Sigma \in \mathcal { S } _ { + } ^ { d }$ , i.e., a d × d symmetric and positive semi-definite matrix. For $p = 2 , P _ { S } = \mathcal { N } ( \mu _ { S } , \Sigma _ { S } )$ and $P _ { T } = \mathcal { N } ( \mu _ { T } , \varSigma _ { T } )$ , the Wasserstein distance is,

$$
W _ {2} (P _ {S}, P _ {T}) ^ {2} = \| \mu_ {S} - \mu_ {T} \| _ {2} ^ {2} + \mathcal {B} (\Sigma_ {S}, \Sigma_ {T}),\tag{5}
$$

where $\boldsymbol { B }$ is the Bures-metric between covariance matrices [29]. While OT-based DA methods use equation 4, equation 5 is commonly used for estimating the Wasserstein distance given samples. In this case, the parameters $( \mu _ { S } , \varSigma _ { S } , \mu _ { T } , \varSigma _ { T } )$ are the sample mean and covariance from each domain.

Table 1: Description of shallow and deep domain adaptation methods alongside the notion of distance they minimize during training.

<table><tr><td rowspan="2">Method</td><td colspan="3">Single Source</td><td colspan="4">Multi Source</td></tr><tr><td>Distance</td><td>Category</td><td>Reference</td><td>Method</td><td>Distance</td><td>Category</td><td>Reference</td></tr><tr><td>TCA</td><td>MMD</td><td>Shallow</td><td>[23]</td><td>M3SDA</td><td>MMD</td><td>Deep</td><td>[24]</td></tr><tr><td>OTDA</td><td> $W_2$ </td><td>Shallow</td><td>[4]</td><td> $M3SDA_\beta$ </td><td></td><td></td><td></td></tr><tr><td>JDOT</td><td> $W_2$ </td><td>Shallow</td><td>[3]</td><td>WJDOT</td><td> $W_2$ </td><td>Shallow</td><td>[30]</td></tr><tr><td>MMD</td><td>MMD</td><td>Deep</td><td>[12]</td><td> $WBT_{reg}$ </td><td> $W_2$ </td><td>Shallow</td><td>[9,8]</td></tr><tr><td>DANN</td><td> $d_H$ </td><td>Deep</td><td>[11]</td><td>DaDiL-R</td><td> $W_2$ </td><td>Shallow</td><td>[21]</td></tr><tr><td>DeepJDOT</td><td> $W_2$ </td><td>Deep</td><td>[5]</td><td>DaDiL-E</td><td></td><td></td><td></td></tr></table>

A final distinction between DA methods is with respect their strategy. First, we consider shallow DA methods. These strategies apply transformations to preextracted features, in the hope of aligning the data distributions. For instance, Transfer Component Analysis (TCA) [23] projects data into a lower dimensional space while minimizing the MMD. With respect the architecture shown in Fig. 1, these methods keep the parameters of the encoder network ϕ frozen during adaptation, and fine-tune the classifier h on the adapted data. Second, we consider deep DA methods, which rely on the encoder network ϕ for aligning the data. The principle is to minimize the distance in distribution between $\phi _ { \sharp } P _ { S }$ and $\phi _ { \sharp } P _ { T }$ (c.f., Fig. 2). This is the case of DeepJDOT [5], which minimizes the Wasserstein distance between the aforementioned distributions. In total, we consider 11 methods, as shown in Table 1. We refer readers to the original papers for further details on these algorithms.

## 3 Case Study: the Tennessee Eastman Process

In this section, we present our case study, the Tennessee Eastman Process (TEP). This chemical process was first introduced by [7], with the intent to serve as a realistic benchmark for the design of control and monitoring systems. From the perspective of fault detection and diagnosis [20], this system is widely used by the academic community. Henceforth, we follow the description of the TEP by [28]. The TEP consists on the production of two liquid product components, G and H, from 4 gaseous reactants, A, C, D and E, with an additional inert B and a byproduct F , which are related through 4 exothermic and irreversible reactions,

$$
\begin{array}{r l} A (\mathrm{g}) + C (\mathrm{g}) + D (\mathrm{g}) \to G (\mathrm{liq}) & \text {Product 1,} \\ A (\mathrm{g}) + C (\mathrm{g}) + E (\mathrm{g}) \to H (\mathrm{liq}) & \text {Product 2,} \\ A (\mathrm{g}) + E (\mathrm{g}) \to F (\mathrm{liq}) & \text {Byproduct,} \\ 3 D (\mathrm{g}) \to 2 F (\mathrm{liq}) & \text {Byproduct.} \end{array}\tag{6}
$$

The TEP system is composed by five major process units: reactor, product condenser, vapor-liquid separator, recycle compressor and product stripper, shown in Fig. 3. Based on the reactions in equation 6, there are 6 diferent modes of operation, which correspond to 3 diferent $G / H$ mass ratios, and a desired product rate. The diferent modes of operation are shown in Table 2.

From the perspective of DA, each mode of operation induces changes in the statistical properties of the data. As a result, a model learned with historical data from a set of operation modes (e.g., 1, · · · , 5) may not generalize to a new operation mode (e.g., 6). At the same time, collecting labeled data at the new operation mode is costly. MSDA is thus a natural solution, where one leverages historical data from previous modes to learn a better model on the new mode, only requiring unlabeled data on the new operation conditions. In section 3.1, we describe a methodology for building a MSDA benchmark on top of TEP simulations provided by [28].

To build an AFD system, we need to collect data from a set of sensors, then categorize the data into a set of faults. In this paper, we use the data provided by [28]. In their simulations, there are 53 sensors in the overall plant, corresponding to diferent physical and chemical quantities. We group these variables into measurements (denoted XME(i), for the i−th measurement) and manipulated (denoted $\mathrm { X M V } ( j )$ , for the j−th manipulation), as shown in Table 3.

![](images/98b7956c8a7cac6505b7b30d723fddeaef817474ec5ffb997ed18da9342bf965.jpg)  
Fig. 3: P&ID diagram for the TEP. Figure reproduced from [1], which shows the main components of the process. Measurements originally introduced by [7] are shown in gray, whereas the measurements introduced by [1] are shown in red. A simulation environment, based on this diagram, is described in [28].

Table 2: TEP operation modes, as described in [7]. In our experiments, each mode of operation corresponds to a diferent domain.

<table><tr><td>Mode</td><td>Mass Ratio</td><td>Production rate</td></tr><tr><td>1</td><td>50/50</td><td> $7038 \text{ kg h}^{-1} \text{ G and } 7038 \text{ kg h}^{-1} \text{ H}$ </td></tr><tr><td>2</td><td>10/90</td><td> $1408 \text{ kg h}^{-1} \text{ G and } 12,669 \text{ kg h}^{-1} \text{ H}$ </td></tr><tr><td>3</td><td>90/10</td><td> $10,000 \text{ kg h}^{-1} \text{ G and } 1111 \text{ kg h}^{-1} \text{ H}$ </td></tr><tr><td>4</td><td>50/50</td><td>maximum production rate</td></tr><tr><td>5</td><td>10/90</td><td>maximum production rate</td></tr><tr><td>6</td><td>90/10</td><td>maximum production rate</td></tr></table>

In this dataset, the TEP system is simulated for a 100 hours, with a sampling rate of 3 minutes. As such, we use each simulation as a sample in our MSDA benchmark. In each simulation, faults are introduced after 600 time steps (i.e., 30 hours). Concerning the type of faults, in their initial publication, [7] presents 20 types of process disturbances (faults 1 through 20 in Table 4), afecting diferent process variables. In addition to these initial faults, [1] proposed 8 additional faults under the type random variation, as shown in Table 4.

## 3.1 Benchmark preparation

Data Cleaning. For each mode, the simulations provided by [28] are divided into 3 groups: set-point variation, mode transitions and single fault. In the first case, the authors change the initial simulation set-point using a step or ramp function. In the second case, the simulation changes from one mode to another at an instant in time. In the third case, as previously mentioned, a fault is introduced at time step 600, i.e., after 30 hours of simulation. For each fault, there are multiple intensities available (e.g., 25%, 50%, 75% and 100% fault magnitude). For magnitudes 25%, 50% and 75%, the system is simulated 100 times, whereas for 100%, the system is simulated 200 times. As a result, for the single-fault scenario only, the data provided by [28] contains,

Table 3: Description of process variables of the TEP. Variables are divided into measurements (XME) and manipulated (XMV).

<table><tr><td>Variable</td><td>Description</td><td>Variable</td><td>Description</td><td>Variable</td><td>Description</td><td>Variable</td><td>Description</td></tr><tr><td>XME(1)</td><td>A Feed (kscmh)</td><td>XME(15)</td><td>Stripper Level (%)</td><td>XME(29)</td><td>Component A in Purge (mol %)</td><td>XMV(2)</td><td>E Feed (%)</td></tr><tr><td>XME(2)</td><td>D Feed (kg/h)</td><td>XME(16)</td><td>Stripper Pressure (kPa gauge)</td><td>XME(30)</td><td>Component B in Purge (mol %)</td><td>XMV(3)</td><td>A Feed (%)</td></tr><tr><td>XME(3)</td><td>E Feed (kg/h)</td><td>XME(17)</td><td>Stripper Underflow ( $m^{3}/h$ )</td><td>XME(31)</td><td>Component C in Purge (mol %)</td><td>XMV(4)</td><td>A &amp; C Feed (%)</td></tr><tr><td>XME(4)</td><td>A &amp; C Feed (kg/h)</td><td>XME(18)</td><td>Stripper Temp (°C)</td><td>XME(32)</td><td>Component D in Purge (mol %)</td><td>XMV(5)</td><td>Compressor recycle valve (%)</td></tr><tr><td>XME(5)</td><td>Recycle Flow (kscmh)</td><td>XME(19)</td><td>Stripper Steam Flow (kg/h)</td><td>XME(33)</td><td>Component E in Purge (mol %)</td><td>XMV(6)</td><td>Purge valve (%)</td></tr><tr><td>XME(6)</td><td>Reactor Feed rate (kscmh)</td><td>XME(20)</td><td>Compressor Work (kW)</td><td>XME(34)</td><td>Component F in Purge (mol %)</td><td>XMV(7)</td><td>Separator liquid flow (%)</td></tr><tr><td>XME(7)</td><td>Reactor Pressure (kscmh)</td><td>XME(21)</td><td>Reactor Coolant Temp (°C)</td><td>XME(35)</td><td>Component G in Purge (mol %)</td><td>XMV(8)</td><td>Stripper liquid flow (%)</td></tr><tr><td>XME(8)</td><td>Reactor Level (%)</td><td>XME(22)</td><td>Separator Coolant Temp (°C)</td><td>XME(36)</td><td>Component H in Purge (mol %)</td><td>XMV(9)</td><td>Stripper steam valve (%)</td></tr><tr><td>XME(9)</td><td>Reactor Temperature (°C)</td><td>XME(23)</td><td>Component A to Reactor (mol %)</td><td>XME(37)</td><td>Component D in Product (mol %)</td><td>XMV(10)</td><td>Reactor coolant (%)</td></tr><tr><td>XME(10)</td><td>Purge Rate (kscmh)</td><td>XME(24)</td><td>Component B to Reactor (mol %)</td><td>XME(38)</td><td>Component E in Product (mol %)</td><td>XMV(11)</td><td>Condenser Coolant (%)</td></tr><tr><td>XME(11)</td><td>Product Sep Temp (°C)</td><td>XME(25)</td><td>Component C to Reactor (mol %)</td><td>XME(39)</td><td>Component F in Product (mol %)</td><td>XMV(12)</td><td>Agitator Speed (%)</td></tr><tr><td>XME(12)</td><td>Product Sep Level (%)</td><td>XME(26)</td><td>Component D to Reactor (mol %)</td><td>XME(40)</td><td>Component G in Product (mol %)</td><td></td><td></td></tr><tr><td>XME(13)</td><td>Product Sep Pressure (kPa gauge)</td><td>XME(27)</td><td>Component E to Reactor (mol %)</td><td>XME(41)</td><td>Component H in Product (mol %)</td><td></td><td></td></tr><tr><td>XME(14)</td><td>Product Sep Underflow ( $m^{3}/h$ )</td><td>XME(28)</td><td>Component F to Reactor (mol %)</td><td>XMV(1)</td><td></td><td>D Feed (%)</td><td></td></tr></table>

Table 4: Description and types of faults for the TEP in the simulation environment of [28]. Faults are grouped into 4 types: step, random variation (RV), sticking and unknown.

<table><tr><td>Fault</td><td>Variable</td><td>Type</td><td>Fault Class</td><td>Variable</td><td>Type</td></tr><tr><td>1</td><td>A/C feed ratio, B composition constant</td><td>Step</td><td>15</td><td>Water outlet temperature (separator)</td><td>Sticking</td></tr><tr><td>2</td><td>B composition, A/C ratio constant</td><td>Step</td><td>16</td><td>Variation coefficient of the steam supply of the heat exchange of the stripper</td><td>RV</td></tr><tr><td>3</td><td>D feed temperature</td><td>Step</td><td>17</td><td>Variation coefficient of heat transfer (reactor)</td><td>RV</td></tr><tr><td>4</td><td>Water inlet temperature (reactor)</td><td>Step</td><td>18</td><td>Variation coefficient of heat transfer (condenser)</td><td>RV</td></tr><tr><td>5</td><td>Water inlet temperature (condenser)</td><td>Step</td><td>19</td><td>Unknown</td><td>Unknown</td></tr><tr><td>6</td><td>A feed loss</td><td>Step</td><td>20</td><td>Unknown</td><td>RV</td></tr><tr><td>7</td><td>C header pressure loss</td><td>Step</td><td>21</td><td>A feed temperature</td><td>RV</td></tr><tr><td>8</td><td>A/B/C composition of stream 4</td><td>RV</td><td>22</td><td>E feed temperature</td><td>RV</td></tr><tr><td>9</td><td>D feed temperature 4</td><td>RV</td><td>23</td><td>A feed flow</td><td>RV</td></tr><tr><td>10</td><td>C feed temperature</td><td>RV</td><td>24</td><td>D feed flow</td><td>RV</td></tr><tr><td>11</td><td>Water outlet temperature (reactor)</td><td>RV</td><td>25</td><td>E feed flow</td><td>RV</td></tr><tr><td>12</td><td>Water outlet temperature (separator)</td><td>RV</td><td>26</td><td>A &amp; C feed flow</td><td>RV</td></tr><tr><td>13</td><td>Reaction kinetics</td><td>RV</td><td>27</td><td>Water flow (reactor)</td><td>RV</td></tr><tr><td>14</td><td>Water outlet temperature (reactor)</td><td>Sticking</td><td>28</td><td>Water flow (condenser)</td><td>RV</td></tr></table>

## 28 faults × 6 modes × 500 simulations = 84000 simulations.

Nonetheless, one should note that some simulations terminate earlier than 100h, due to forced plant-shutdown. As a result, we adopt the following strategy: for each fault, we keep the first 100 simulations of highest magnitude that terminate successfully. For each selected simulation, we crop the signal into 2 parts. The first 30h correspond to the steady state, determined by the set point of the mode of operation. This first part of the signal characterizes the healthy state of the system (i.e., faultless state). We further sub-sample the number of faultless state signals to keep a balanced dataset (i.e., 100 per mode of operation). The second part consists on the next 30h of simulation. Since faults are introduced exactly at the 601th time step, the second part of the signal characterize each fault. This process generates a slightly imbalanced dataset of 17289 samples<sup>5</sup>. We summarize the division of samples among modes of operation in Table 5.

Table 5: Number and percentage of samples from each mode of operation.

<table><tr><td colspan="3">Mode of Operation # of Samples % of Samples</td></tr><tr><td>1</td><td>2900</td><td>16.77</td></tr><tr><td>2</td><td>2845</td><td>16.45</td></tr><tr><td>3</td><td>2899</td><td>16.76</td></tr><tr><td>4</td><td>2865</td><td>16.57</td></tr><tr><td>5</td><td>2883</td><td>16.67</td></tr><tr><td>6</td><td>2897</td><td>16.75</td></tr><tr><td>Total</td><td>17289</td><td>100</td></tr></table>

Variable selection and pre-processing. Out of the 53 variables presented in Table 3 some of these variables are not continuous (e.g., XME(23) through XME(41)). Given this remark, we follow [28], and consider a sub-set of 34 continuous signals as input to our neural nets. These are measurements XME(1) through XME(22), and manipulated variables XMV(1) through XMV(12). We thus have multi-variate time series of shape (34, 600), where 34 is the number of sensor readings (i.e., considered variables), and 600 corresponds to the number of time steps (T ). We further perform a standardization along each variable, within each mode, $x _ { i , j , t } ^ { ( P _ { \ell } ) } = ( x _ { i , j , t } ^ { ( P _ { \ell } ) } - \mu _ { j } ^ { ( P _ { \ell } ) } ) / \sigma _ { j } ^ { ( P _ { \ell } ) }$ , where,

$$
\mu_ {j} ^ {(P _ {\ell})} = \frac {1}{n _ {\ell} T} \sum_ {i = 1} ^ {n _ {\ell}} \sum_ {t = 1} ^ {T} x _ {i, j, t} ^ {(P _ {\ell})}, \text {and} \sigma_ {i, j, t} ^ {(P _ {\ell})} = \sqrt {\frac {1}{n _ {\ell} T} \sum_ {i = 1} ^ {n _ {\ell}} \sum_ {t = 1} ^ {T} (x _ {i , j , t} ^ {(P _ {\ell})} - \mu_ {j} ^ {(P _ {\ell})}) ^ {2}},
$$

where $n _ { \ell }$ is the number of samples for mode ℓ = 1, · · · , 6 (c.f., Table 5).

Neural Network Backbone. In DA, it is common to choose a backbone upon which methods will rely on. For instance, in image processing, residual networks [14] are widely used. In the context of time series, In our paper, we employ a Fully Convolutional Network (FCN) [19,32,17], which consists on three convolutional blocks followed by a Global Average Pooling (GAP) layer. Each convolutional block has a convolutional layer, and a normalization layer. In our experiments we verified that instance normalization [31] improves stability and performance over other normalization layers such as batch normalization [15].

## 3.2 Exploratory Data Analysis

Qualitative Analysis. We analyze the pairwise correlations of variables, conditioned on the type of fault, that is, IDV(1) through IDV(28), and the no-fault scenario. In Fig. 4, we illustrate a change in the pattern of correlations between variables, conditioned on the fault type, for modes 1 and 2. In comparison, these patterns drastically change for faults 15, 18 and 28, corresponding to a sticking fault on the water outlet temperate on the separator, a random variation on the heat transfer coeficient on the condenser and a random variation on the water flow on the condenser, respectively. Hence, the mode of production deeply impacts the dynamic of the system, which creates a shift in distribution between data from these modes.

![](images/68a31475a2c3c0e7cd96e2469cd7a6b3d6d0d7178d0bfd2e3e141e374b277bbc.jpg)  
(a) Mode 1

![](images/628ede14d4efa9bf6e41c9e8c33fbe0f01dc98697fa2320e16bd24b7ba1fc31c.jpg)  
(b) Mode 2  
Fig. 4: Qualitative analysis of distributional shift. In (a) and (b), we show the correlation between diferent variables in TEP, for modes 1 and 2, for each fault. On each correlation matrix, the coeficient $\rho _ { j j ^ { \prime } }$ corresponds to the Pearson correlation coeficient between $\{ x _ { j , t } \} _ { t = 1 } ^ { 6 0 0 }$ and $\{ x _ { j ^ { \prime } , t } \} _ { t = 1 } ^ { 6 0 0 }$ across simulations.

![](images/05a8972dfd308e4f21c59ac0bd96da4669169f74b039ab7925905fdf270f7b62.jpg)  
(a) Pairwise $W _ { 2 } .$

![](images/c167201680de5e2ca5ab1e0873cd3c085aec4e419f98f174c0446ee5fb463e9d.jpg)  
(b) Mode embeddings.  
Fig. 5: Quantitative analysis of distributional shift. Pairwise Wasserstein distance between modes (a). Mode embeddings based on MDS (b).

Quantitative Analysis. We quantify the shift between pairs of modes through the probability metrics introduced in section 2. We estimate the pairwise Wasserstein distances between modes using eq. 5. In Fig. 5 (a), we show the pairwise distance in probability distribution between diferent modes. On one hand, the most diferent mode with respect others is Mode 2, which is especially far from modes 1, 3 and 5. On the other hand, the most similar modes are 3 and 6. We can have a better picture about the level of similarity of these diferent domains by embedding them on the plane, as shown in Fig. 5 (b). We obtain these embeddings through Multi-Dimensional Scaling (MDS) [18], which defines the points in $\bar { \mathbb { R } } ^ { 2 }$ while preserving the pairwise distances between the embeddings.

From our qualitative and quantitative analysis, we expect lower performances with respect the adaptation towards mode 2, as it is the most dissimilar from other modes (c.f., Fig. 5 (a), average row). In contrast, adaptation between modes (3, 6), and (1, 4, 5) should work well as these modes share statistical characteristics. We verify these indications empirically in the next sections.

## 3.3 Single-Source Domain Adaptation

In this section, we explore single-source DA, i.e., when adaptation is done from a single source mode, to a single target mode. On the one hand, we refer to generalization, to the ability of a classifier to perform well on unseen data from an unseen domain. On the other hand, we refer to adaptation, when a classifier performs well on unseen data from the target domain. In this context, we have

![](images/fa287c65aa68824865a4c53d6fb881bd192b986452394003828365e6a67bc16a.jpg)  
(a) Baseline

![](images/130b370d0792d9ba977bffab06f4de3cebc9acd1a6b3ce9bc3559d264f464597.jpg)  
(b) DANN

![](images/21252c95caf076f4d12c685ff54c44bb79adc857214ec20716ea6c1b3722db37.jpg)  
(c) TCA

![](images/adb84542b3ccbc270c0266b704cb23c29bcac2d8bb61f86d6d6c07c643363c3b.jpg)  
(d) MMD

![](images/a81adc23bace9ff7dae1e785d95ba05bc72170cb5f792c804f93f19174e34979.jpg)  
(e) OTDA

![](images/01786b95becfb9863eda680bd2345e50363fa77b6b04b9222d4c565dd3bc5aa5.jpg)  
(f) DeepJDOT

![](images/f6eb007d3694d77c70f9de90cd1b6f94ddef491136a198b209fc4dee0c3112dc.jpg)  
(g) JDOT  
Fig. 6: Baseline (a) and single-source domain adaptation (b) algorithms.

2 baselines. The first, source-only, considers that a classifier is learned exclusively with source domain data (i.e., no adaptation). This corresponds to the of-diagonals of Figure 6 (a). Second, we have target-only, which trains and evaluates models on the target domain (i.e., no distributional shift). Note that the target-only scenario has an advantage over other methods, as it has access to labeled data in the target domain. This baseline can be seen as an upper bound in the adaptation performance.. With respect these scenarios, note that we verify our previous remarks, i.e., generalization towards mode 2 is much more dificult than other domains, and the clusters of similar domains (e.g., (3, 6) and (1, 4, 5)) generalize well.

We further compare the single-source DA methods presented in Table 1, which are shown in Fig. 6 (b) through (g). Overall, we find that OT-based methods have a higher performance than other metrics (e.g., the MMD). This is similar to previous findings on smaller scale problems, such as [22]. The best performing method is Optimal Transport Domain Adaptation (OTDA) [4], which maps source domain points to the target domain points through the OT plan (c.f., eq. 4). Nonetheless, one should be mindful of negative transfer [33] between similar modes (e.g., 3 → 5), which may result in performance degradation.

## 3.4 Multi-Source Domain Adaptation

In this section, we explore multi-source DA, i.e., when adaptation is done from multiple source domains towards a single target. Here, note that the models have much more labeled available data, as all domains are considered at once. We start our discussion by comparing the performance of single, source-only baselines, and the corresponding multi-source baseline for each target mode, which is shown in Fig. 7. Overall, the multi source-only baseline improve over single single-only for the same target. These baselines have similar performance when there are pairs of highly similar modes (e.g., modes 3 and 6), showing that extra data from additional modes is not as informative for generalization.

![](images/dcf13012bf42b3f28e159ae9f99cc65288237a277e2aac09ad650fc4228a835a.jpg)  
Fig. 7: Multi and single-source baseline comparison. On top, we show the target domain. In the abscissa, we show the corresponding baseline. The multisource scenario generally improves over the single source-only case.

We now consider the performance of DA algorithms in the multi-source setting. Besides native MSDA algorithms, i.e., algorithms that suppose the source as composed by diferent domains, we also consider single-source algorithms with access to the concatenation of all source domains. Our comparison is shown in Fig. 8. A first question is whether access to additional data is beneficial to adaptation. For instance, in single-source DA, methods exhibited negative transfer in the task $3  5$ . When provided access to data from all domains, all single-source adaptation method performance improved over the single-source baseline. As a result, even though data from multiple domains may not improve generalization, it does improves adaptation.

![](images/f3a64c5ddfc40e906a9a67edd6744c4281e3de9064bec427e28ef9f5ae8692b1.jpg)  
Fig. 8: Multi-source domain adaptation results. We compare all algorithms with access to labeled data from all source domains, except the target mode, from which we have access to unlabeled data. Methods in the abcisssa are ordered by average performance on all modes.

With respect Fig. 8, from the perspective of MSDA, methods that weight sources in a linear space, such as WJDOT, or in a Wasserstein space, such as WBT and DaDiL outperform the weighting of classifiers’ predictions, such as $\mathrm { M 3 S D A } _ { \beta }$ [24]. On the one hand, WJDOT can filter undesirable information during adaptation by assigning small weights to domains and samples. On the other hand, WBT and DaDiL combine the information in the sources non-linearly. These two strategies are efective in domain adaptation.

Finally, from Fig. 9, we can see that shallow DA methods (e.g., JDOT) generally improve over deep DA methods (e.g., DeepJDOT). Indeed, deep DA methods learn features that are invariant to the domain shift between diferent modes. As a result, these features may be less useful for classification. In a general note (both single, and multi-source methods), OT-based techniques outperform methods based on other distances, such as the MMD and $d _ { \mathcal { H } }$ . This remark agrees with previous studies on smaller scale systems [22].

![](images/079e0301c927b115dd6364e8bc3dae2e95ac81501f229b5ff25235b0a54c9d28.jpg)  
Fig. 9: Comparison of average adaptation performance of DA algorithms in the multi-source setting.

## 4 Conclusion

In this paper, we introduce a new benchmark for domain adaptation algorithms based on the Tenessee Eastman process [7]. The present benchmark is created by applying pre-processing steps on the simulations provided by [28] (c.f., section 3), thus creating a large scale dataset of time series. These time series are associated with diferent modes of production. Based on each mode of production, the statistical properties of the time series change (c.f., Fig. 4) creating a shift in the data probability distribution (c.f., Fig. 5). As a result, data trained on a specific mode may not generalize well to other modes of production, thus the need for domain adaptation. Through a series of experiments with singlesource and multi-source domain adaptation methods, we show that OT-based methods outperform methods that rely on the maximum mean discrepancy, and H−distances, which agrees with previous findings on smaller scale systems [22]. Besides providing the open source code for the reproduction of our benchmark, with this work we hope to encourage research on the intersection between domain adaptation and fault diagnosis [34].

## References

1. Bathelt, A., Ricker, N.L., Jelali, M.: Revision of the tennessee eastman process model. IFAC-PapersOnLine 48(8), 309–314 (2015)

2. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., Vaughan, J.W.: A theory of learning from diferent domains. Machine learning 79(1-2), 151–175 (2010)

3. Courty, N., Flamary, R., Habrard, A., Rakotomamonjy, A.: Joint distribution optimal transportation for domain adaptation. Advances in neural information processing systems 30 (2017)

4. Courty, N., Flamary, R., Tuia, D., Rakotomamonjy, A.: Optimal transport for domain adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence 39(9), 1853–1865 (2017). https://doi.org/10.1109/TPAMI.2016.2615921

5. Damodaran, B.B., Kellenberger, B., Flamary, R., Tuia, D., Courty, N.: Deepjdot: Deep joint distribution optimal transport for unsupervised domain adaptation. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 447– 463 (2018)

6. Dantzig, G.B., Orden, A., Wolfe, P., et al.: The generalized simplex method for minimizing a linear form under linear inequality restraints. Pacific Journal of Mathematics 5(2), 183–195 (1955)

7. Downs, J.J., Vogel, E.F.: A plant-wide industrial process control problem. Computers & chemical engineering 17(3), 245–255 (1993)

8. Fernandes Montesuma, E., Mboula, F.M.N.: Wasserstein barycenter for multisource domain adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16785–16793 (2021)

9. Fernandes Montesuma, E., Mboula, F.M.N.: Wasserstein barycenter transport for acoustic adaptation. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 3405–3409. IEEE (2021)

10. Fernandes Montesuma, E., Mboula, F.N., Souloumiac, A.: Recent advances in optimal transport for machine learning. arXiv preprint arXiv:2306.16156 (2023)

11. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., Lempitsky, V.: Domain-adversarial training of neural networks. The journal of machine learning research 17(1), 2096–2030 (2016)

12. Ghifary, M., Kleijn, W.B., Zhang, M.: Domain adaptive neural networks for object recognition. In: Pacific Rim international conference on artificial intelligence. pp. 898–904. Springer (2014)

13. Gretton, A., Borgwardt, K.M., Rasch, M., Schölkopf, B., Smola, A.J.: A kernel approach to comparing distributions. In: Proceedings of the National Conference on Artificial Intelligence. vol. 22, p. 1637. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999 (2007)

14. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

15. Iofe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International conference on machine learning. pp. 448–456. pmlr (2015)

16. Isermann, R.: Fault-diagnosis systems: an introduction from fault detection to fault tolerance. Springer Science & Business Media (2006)

17. Ismail Fawaz, H., Forestier, G., Weber, J., Idoumghar, L., Muller, P.A.: Deep learning for time series classification: a review. Data mining and knowledge discovery 33(4), 917–963 (2019)

18. Kruskal, J.B.: Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika 29(1), 1–27 (1964)

19. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3431–3440 (2015)

20. Melo, A., Câmara, M.M., Clavijo, N., Pinto, J.C.: Open benchmarks for assessment of process monitoring and fault diagnosis techniques: A review and critical analysis. Computers & Chemical Engineering 165, 107964 (2022)

21. Montesuma, E.F., Mboula, F., Souloumiac, A.: Multi-source domain adaptation through dataset dictionary learning in wasserstein space. In: European Conference on Artificial Intelligence. pp. 1739–1745 (09 2023). https://doi.org/10.3233/ FAIA230459

22. Montesuma, E.F., Mulas, M., Corona, F., Mboula, F.M.N.: Cross-domain fault diagnosis through optimal transport for a cstr process. IFAC-PapersOnLine 55(7), 946–951 (2022)

23. Pan, S.J., Tsang, I.W., Kwok, J.T., Yang, Q.: Domain adaptation via transfer component analysis. IEEE transactions on neural networks 22(2), 199–210 (2010)

24. Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., Wang, B.: Moment matching for multi-source domain adaptation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1406–1415 (2019)

25. Peyré, G., Cuturi, M.: Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning 11(5-6), 355–607 (2019)

26. Quiñonero-Candela, J., Sugiyama, M., Lawrence, N.D., Schwaighofer, A.: Dataset shift in machine learning. Mit Press (2009)

27. Redko, I., Morvant, E., Habrard, A., Sebban, M., Bennani, Y.: Advances in domain adaptation theory. Elsevier (2019)

28. Reinartz, C., Kulahci, M., Ravn, O.: An extended tennessee eastman simulation dataset for fault-detection and decision support systems. Computers & Chemical Engineering 149, 107281 (2021)

29. Takatsu, A.: Wasserstein geometry of Gaussian measures. Osaka Journal of Mathematics 48(4), 1005 – 1026 (2011)

30. Turrisi, R., Flamary, R., Rakotomamonjy, A., et al.: Multi-source domain adaptation via weighted joint distributions optimal transport. In: The 38th Conference on Uncertainty in Artificial Intelligence (2022)

31. Ulyanov, D., Vedaldi, A., Lempitsky, V.: Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022 (2016)

32. Wang, Z., Yan, W., Oates, T.: Time series classification from scratch with deep neural networks: A strong baseline. In: 2017 International joint conference on neural networks (IJCNN). pp. 1578–1585. IEEE (2017)

33. Zhang, W., Deng, L., Zhang, L., Wu, D.: A survey on negative transfer. IEEE/CAA Journal of Automatica Sinica 10(2), 305–329 (2022)

34. Zheng, H., Wang, R., Yang, Y., Yin, J., Li, Y., Li, Y., Xu, M.: Cross-domain fault diagnosis using knowledge transfer strategy: a review. IEEE Access 7, 129260– 129290 (2019)