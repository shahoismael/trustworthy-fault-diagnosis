# Explainable Artificial Intelligence for Fault Diagnosis of Industrial Processes

Kyojin Jang , Karl Ezra Salgado Pilario , Nayoung Lee, Il Moon , and Jonggeol Na

Abstract—Process monitoring is important for ensuring operational reliability and preventing occupational accidents. In recent years, data-driven methods such as machine learning and deep learning have been preferred for fault detection and diagnosis. In particular, unsupervised learning algorithms, such as auto-encoders, exhibit good detection performance, even for unlabeled data from complex processes. However, decisions generated from deepneural-network-based models are difficult to interpret and cannot provide explanatory insight to users. We address this issue by proposing a new fault diagnosis method using explainable artificial intelligence to break the traditional tradeoff between the accuracy and interpretability of deep learning model. First, an adversarial auto-encoder model for fault detection is built and then interpreted through the integration of Shapley additive explanations (SHAP) with a combined monitoring index. Using SHAP values, a diagnosis is conducted by allocating credit for detected faults, deviations from a normal state, among its input variables. The proposed diagnosis method can consider not only reconstruction space but also latent space, unlike conventional methods, which evaluate only reconstruction error. The proposed method was applied to two chemical process systems and compared with conventional diagnosis methods. The results highlight that the proposed method achieves the exact fault diagnosis for single and multiple faults and, also, distinguishes the global pattern of various fault types.

Manuscript received 4 October 2022; revised 25 November 2022 and 27 December 2022; accepted 30 December 2022. Date of publication 30 January 2023; date of current version 7 January 2025. This work was supported by the National Research Foundation of Korea (NRF) Grant funded by the Korean Government (MSIT) under Grant 2021R1C1C1012031 and Grant 2021R1A4A3025742. Paper no. TII-22- 4162. (Kyojin Jang and Karl Ezra Salgado contributed equally to this work.) (Corresponding authors: Il Moon; Jonggeol Na.)

Digital Object Identifier 10.1109/TII.2023.3240601

Index Terms—Auto-encoder (AE), continuous stirredtank reactor (CSTR), explainable AI, fault diagnosis, Tennessee Eastman (TE) process.

## I. INTRODUCTION

F <sup>AULT</sup> <sup>detection</sup> <sup>and</sup> <sup>diagnosis</sup> <sup>(FDD)</sup> <sup>is</sup> <sup>an</sup> <sup>important</sup> <sup>layer</sup>of safety in any industrial process [1]. It comprises methods that determine the presence (fault detection) and location (fault diagnosis) of process abnormalities [2]. Early detection and accurate diagnostics can help plant operators take corrective action before any further degradation occurs owing to the fault. Hence, FDD has an essential role in maintaining both safe and efficient process operations [3], [4].

FDD methods are commonly built using a data-driven approach [1], [2], where faults are detected statistically using models that are trained solely from the available plant datasets. These models typically use multivariate statistics [5], [6], machine learning (ML) [7], [8], and more recently, deep learning. Deep learning models are attractive because of their ability to extract deep hidden nonlinear features in plant data that are most sensitive to fault behaviors, without having to encode prior physical knowledge regarding the process. Therefore, deeplearning-based FDD methods are suitable for monitoring highly complex and integrated industrial plants.

A few existing deep architectures for FDD include convolutional neural networks (CNNs) [9], [10], deep belief networks (DBNs) [11], long short-term memory (LSTM) [12], stacked auto-encoders (SAE) [13], [14], variational auto-encoders (VAE) [15], and generative adversarial networks (GANs) [16]. By contrast, a recent study combined the advantages of two deep learning models to achieve a better detection performance using an adversarial auto-encoder (AAE) [17]. In [17], the results show that the AAE provides better fault detection than either a PCA or VAE for the well-known Tennessee Eastman (TE) process case study.

Although the above deep learning models are promising for fault detection, their results cannot be easily explained or interpreted, which makes it difficult for human users to trust them. To solve this problem, there have been an increasing number of studies on ML concerned with addressing this issue, which is known as explainable artificial intelligence (XAI) [18], [19]. One approach is the linear local interpretable model-agnostic explanation (LIME) [20], which fits a linear model to input perturbations and produces an explanation. Another approach is DeepLIFT [21], which calculates the contributions of all neurons in the network backpropagated to the input features. More recent work proposed a unified Shapley additive explanations (SHAP) [22] framework and shows that there is a unique solution in the class of additive feature attribution methods, such as LIME and DeepLIFT.

Related studies [23], [24] have proposed a local explainer for auto-encoders (AE) and CNN. In these studies, the goal was to explain supervised fault classification results rather than unsupervised fault detection. Moreover, previous studies have commonly reported variable contributions for fault diagnosis [25], [26]. However, this approach is derived specifically from linear models, such as PCA and CVA. In the case of unsupervised deep learning models, such as AE, fault diagnosis is derived indirectly from reconstruction error, which is the difference between the original input and its reconstruction. Hence, for deep learning models, variable contributions must be computed using more sophisticated explainers while capturing the global patterns of local explanations.

In this study, we combined a model-agnostic explainer with an unsupervised fault detection model to diagnose which variables contribute to the faulty state of the chemical process. The bestperforming AAE-based model is used as the fault detector and the SHAP, known as the state-of-the-art explanation technique, is used as the fault diagnosis model. The proposed method breaks the tradeoff relationship between fault detection and the diagnosability of deep learning models by explaining the inflation of the monitoring index for both latent and reconstruction space. To validate the proposed method, the fault diagnosis results were compared with those of a conventional method for single and multiple faults in two case studies. In addition, using hierarchical clustering on the SHAP-based contributions, we show that the faults are more distinguishable into clusters than with the same analysis conducted on reconstruction-based contributions.

## II. ADVERSARIAL AUTO-ENCODER FOR FAULT DETECTION

Our approach builds on a previous study [17] on detecting the fault state using the AAE model and achieves a state-of-the-art detection performance. The AAE architecture contains a typical AE and discriminator, as shown in Fig. 1(a). The encoder part of the $\mathbf { A E } , q _ { \pi } ( \mathbf { z } | \mathbf { x } )$ , where z is the latent vector and x is the input vector, simultaneously serves as a generator along with the discriminator $D _ { \lambda }$ within an adversarial framework. In addition to minimizing the reconstruction error in the AE, an adversarial training strategy used to distinguish the distribution of the latent vector z from the prior distribution is implemented as follows:

$$
\mathcal {V} (\pi , \lambda) = \mathbb {E} _ {\mathbf {z} ^ {\prime} \sim p (\mathbf {z} ^ {\prime})} \left[ \log D _ {\lambda} (\mathbf {z} ^ {\prime}) \right] + \mathbb {E} _ {\mathbf {z} \sim q (\mathbf {z})} [ \log (1 - D _ {\lambda} (\mathbf {z})) ]\tag{1}
$$

where $\mathbf { z } ^ { \prime }$ is the true sample vector generated from the prior distribution $p ( \mathbf { z } ^ { \prime } )$ . From the perspective of fault detection, a latent vector that provides an additional indication of the faults can be obtained using the AAE model.

To detect faults, a combined monitoring index using the existing multivariate statistics $T ^ { 2 }$ and SPE is proposed. Given the input vector x, latent vector z, and reconstructed vector x in the AAE model, $T ^ { 2 }$ , and SPE are defined as follows:

![](images/2110e95fbad4051383fc573f6d969c16e3cef1ff0ee1ba2ba7bed55c9d991dce.jpg)  
(a)

![](images/141bd580634c7d7a58a944653a21754b144a2325670d620c5b1e5e279de10a01.jpg)  
(b)  
Fig. 1. (a) AAE-based fault detection model. (b) Explainable fault detection framework.

$$
T ^ {2} = \mathbf {z} ^ {T} \Delta^ {- 1} \mathbf {z}\tag{2}
$$

$$
\mathrm{SPE} = (\mathbf {x} - \hat {\mathbf {x}}) ^ {T} (\mathbf {x} - \hat {\mathbf {x}})\tag{3}
$$

where $\Delta$ is the covariance matrix of the latent vector z. $T ^ { 2 }$ and SPE denote the deviation from the normal state in the latent space and reconstruction space, respectively.

Because $T ^ { 2 }$ and SPE behave in a complementary manner, it is possible to combine the two indices to simplify the fault diagnosis task. The combined monitoring index [27] that incorporates $T ^ { 2 }$ and SPE in a balanced manner, is defined as

$$
\psi = \frac {T ^ {2}}{T _ {\alpha} ^ {2}} + \frac {\mathrm{SPE}}{\mathrm{SPE} _ {\alpha}}\tag{4}
$$

where $T _ { \alpha } ^ { 2 }$ and $\mathrm { S P E } _ { \alpha }$ denote the control limits of $T ^ { 2 }$ and $\mathrm { S P E } ,$ <sup>T</sup>respectively. The control limit for $T ^ { 2 }$ <sup>T</sup>, SPE, and $\psi$ can be calculated using a kernel density estimation. If the combined index exceeds its control limit, a fault is detected.

## III. SHAP VALUES FOR FAULT DIAGNOSIS

SHAP is a representative XAI technique that can be applied to explain the prediction of a black-box model. Based on Shapley values from coalitional game theory, SHAP provides the feature importance of the local instance x and global model $f ( \mathbf { x } )$ . The feature importance is the contribution of each feature to the final prediction of the global model. In the process monitoring system, x is the sample, and $f ( \mathbf { x } )$ is the detection model. The combined monitoring index , which is calculated through the nonlinear <sup>ψ</sup>auto-encoder model $f$ is the target of the explanatory model $g .$ Therefore, we implemented SHAP as the fault diagnosis model to explain why the instance was not encoded and/or reconstructed well and, consequently, the monitoring index increased. Finally, the contribution of each process variable is calculated from the fault diagnosis model $^ { g , }$ as shown in Fig. 1(b). The fault diagnosis model is defined as

$$
g (\mathbf {x} ^ {\prime}) = \phi_ {0} + \sum_ {j = 1} ^ {n} \phi_ {j} \mathbf {x} _ {j} ^ {\prime}\tag{5}
$$

where $\mathbf { x } ^ { \prime } \subset \{ 0 , 1 \} ^ { n }$ is the coalition vector of $\mathbf { x } , n$ is the maximum coalition size, is the base value, and $\phi _ { j }$ is the importance of feature $j .$ The coalition vector represents the presence or absence of each feature in a binary format. The base value is the mean of the model output over the background set. For each sample, the sum of the feature importance plus the base value equals the monitoring index . A large value of $\phi _ { j }$ implies that the th variable contributes more to the fault detection index.

<sup>j</sup>According to game theory, classic Shapley values can be calculated from each feature as follows:

$$
\phi_ {j} = \sum_ {S \in j} \frac {| S | ! (| F | - | S | - 1) !}{| F | !} \left[ f _ {S \cup j} \left(\mathbf {x} _ {S \cup j}\right) - f _ {S} (\mathbf {x} _ {\mathbf {S}}) \right]\tag{6}
$$

where $F$ is the set of all input features, and $S$ is a subset of $F .$ <sup>F</sup>In addition, $f _ { S \cup j } ( \mathbf { x } _ { S \cup j } ) - f _ { S } ( \mathbf { x } _ { \mathbf { S } } )$ is the marginal contribution of the feature present in the coalition . Shapley value can be <sup>j S</sup>calculated as an average of this contribution over every possible permutation of the coalition. However, it is difficult to calculate $f _ { S } ( { \bf x } _ { \bf S } )$ owing to some missing features that are not included in . Therefore, to calculate $f _ { S } ( { \bf x } _ { \bf S } )$ , the following assumption is <sup>S</sup>made:

$$
f _ {S} (\mathbf {x _ {S}}) \approx \mathbb {E} [ f (\mathbf {x}) | \mathbf {x _ {S}} ]\tag{7}
$$

where $\mathbb { E } [ f ( \mathbf { x } ) | \mathbf { x } \mathbf { s } ]$ is the expected value of $f ( \mathbf { x } )$ conditioned on the features present in . By combining these conditional expectations with the Shapley values, we can simplify the computation of the SHAP value.

We computed the SHAP values of the input process variables to explain faults revealed by the detection model. SHAP values indicate the contribution of each feature to the difference between the actual and expected values of the monitoring index. For the proposed fault diagnosis model, it is meaningless to distinguish the sign of the SHAP values. Thus, the feature importance for each fault type can be calculated by averaging the absolute SHAP values per feature across the entire fault data as follows:

$$
I _ {j} = \sum_ {i = 1} ^ {n} \left| \phi_ {j} ^ {(i)} \right|\tag{8}
$$

where $\phi _ { j } ^ { ( i ) }$ is the contribution of th variable at th sample and is the total number of fault data.

![](images/d6f2e46eb83922137597f5b72b060619f6010277503f173f43f04451a70b7cde.jpg)  
Fig. 2. Flowchart of fault detection and diagnosis.

Because the number of possible coalitions increases exponentially when the number of features increases, and calculating <sup>M</sup>the SHAP values becomes computationally intractable, kernel SHAP, which is a model-agnostic approximation method, is used in this study. To build a local explanation model, kernel SHAP uses LIME along with the SHAP values. A local model uses a small background set from the data to build a local explanation model that considers the proximity to the sample to be explained. Here, the background set consisted of 100 normal samples, which were summarized using the k-means algorithm. Consequently, we can define a loss function $\mathcal { L }$ by adding the weighting kernel $\pi _ { \mathbf { x } ^ { \prime } }$ as follows:

$$
\mathcal {L} \left(f, g, \pi_ {\mathbf {x} ^ {\prime}}\right) = \sum_ {\mathbf {z} ^ {\prime} \subseteq \mathbf {x} ^ {\prime}} \left[ f _ {\mathbf {x}} \left(\mathbf {z} ^ {\prime}\right) - g \left(\mathbf {z} ^ {\prime}\right) \right] ^ {2} \pi_ {\mathbf {x} ^ {\prime}} \left(\mathbf {z} ^ {\prime}\right)\tag{9}
$$

$$
\pi_ {\mathbf {x} ^ {\prime}} (\mathbf {z} ^ {\prime}) = \frac {n - 1}{\binom {n} {| \mathbf {z} ^ {\prime} |} | \mathbf {z} ^ {\prime} | (n - | \mathbf {z} ^ {\prime} |)}\tag{10}
$$

where $\mathbf { z } ^ { \prime }$ is a random coalition vector near $\mathbf { x } ^ { \prime }$ , and $\left| \mathbf { z } ^ { \prime } \right|$ is the number of nonzero elements in $\mathbf { z } ^ { \prime }$

The overall flowchart of the proposed framework is described in Fig. 2, which is divided into two stages: offline modeling and online monitoring. In the offline modeling phase, we only use normal data both for the AAE model and the SHAP model. The combined monitoring index is calculated and the threshold is set for the normal data. The base value $\phi _ { 0 }$ for the SHAP model is also calculated by fitting the background set. In the online monitoring phase, the proposed framework determines whether the sample is faulty or not and which variable is responsible for this state. Online data are extracted through the trained AAE model and the combined monitoring index is calculated for the sample. If a fault is detected, i.e., the monitoring index rises above the threshold, the fault diagnosis model should explain why this sample was not encoded or reconstructed well. Finally, we can get the contribution of each variable for the local sample and also the fault map for the global pattern.

TABLE I  
FDR, FAR, AND F1 SCORE OF THE COMBINED INDEX FOR THE CSTR PROCESS

<table><tr><td rowspan="2">Fault No.</td><td colspan="3">PCA</td><td colspan="3">AE</td><td colspan="3">VAE</td><td colspan="3">AAE</td></tr><tr><td>FDR</td><td>FAR</td><td>F1 score</td><td>FDR</td><td>FAR</td><td>F1 score</td><td>FDR</td><td>FAR</td><td>F1 score</td><td>FDR</td><td>FAR</td><td>F1 score</td></tr><tr><td>1</td><td>0.8160</td><td>0.0500</td><td>0.8938</td><td>0.8960</td><td>0.1500</td><td>0.9304</td><td>0.7990</td><td>0.0900</td><td>0.8795</td><td>0.8530</td><td>0.0700</td><td>0.9138</td></tr><tr><td>2</td><td>0.9100</td><td>0.0050</td><td>0.9524</td><td>0.9220</td><td>0.0650</td><td>0.9530</td><td>0.8800</td><td>0.0250</td><td>0.9337</td><td>0.9120</td><td>0.0100</td><td>0.9530</td></tr><tr><td>3</td><td>0.9150</td><td>0.1500</td><td>0.9409</td><td>0.9200</td><td>0.1750</td><td>0.9412</td><td>0.8810</td><td>0.1650</td><td>0.9206</td><td>0.9060</td><td>0.0900</td><td>0.9418</td></tr><tr><td>4</td><td>0.9640</td><td>0.0950</td><td>0.9723</td><td>0.9670</td><td>0.3200</td><td>0.9522</td><td>0.9550</td><td>0.2300</td><td>0.9545</td><td>0.9670</td><td>0.1550</td><td>0.9680</td></tr><tr><td>5</td><td>0.9010</td><td>0.0250</td><td>0.9454</td><td>0.9100</td><td>0.2250</td><td>0.9309</td><td>0.9020</td><td>0.2550</td><td>0.9237</td><td>0.9040</td><td>0.2200</td><td>0.9281</td></tr><tr><td>6</td><td>0.9920</td><td>0.0350</td><td>0.9925</td><td>0.9920</td><td>0.1000</td><td>0.9861</td><td>0.9910</td><td>0.0600</td><td>0.9895</td><td>0.9920</td><td>0.0250</td><td>0.9935</td></tr><tr><td>7</td><td>0.9640</td><td>0.0500</td><td>0.9767</td><td>0.9820</td><td>0.1500</td><td>0.9761</td><td>0.9490</td><td>0.0550</td><td>0.9684</td><td>0.9750</td><td>0.0700</td><td>0.9804</td></tr><tr><td>8</td><td>0.8690</td><td>0.0850</td><td>0.9215</td><td>0.8910</td><td>0.2950</td><td>0.9138</td><td>0.8230</td><td>0.1850</td><td>0.8849</td><td>0.8730</td><td>0.2250</td><td>0.9103</td></tr><tr><td>9</td><td>0.9070</td><td>0.0100</td><td>0.9502</td><td>0.9520</td><td>0.3000</td><td>0.9463</td><td>0.9520</td><td>0.2100</td><td>0.9549</td><td>0.9620</td><td>0.1450</td><td>0.9663</td></tr><tr><td>10</td><td>0.9700</td><td>0.0550</td><td>0.9793</td><td>0.9810</td><td>0.2600</td><td>0.9651</td><td>0.9580</td><td>0.0750</td><td>0.9711</td><td>0.9710</td><td>0.0400</td><td>0.9813</td></tr><tr><td>11</td><td>0.9670</td><td>0.1300</td><td>0.9704</td><td>0.9610</td><td>0.0950</td><td>0.9707</td><td>0.9700</td><td>0.0600</td><td>0.9788</td><td>0.9660</td><td>0.0350</td><td>0.9792</td></tr></table>

## IV. CASE STUDIES

To demonstrate the effectiveness and feasibility of the proposed diagnosis method, a continuous stirred-tank reactor (CSTR) [28] and the TE [29] benchmark processes were considered. First, to validate the detection performance, an AAE-based model is compared with several other methods (PCA, AE, and VAE) by evaluating the fault detection rate (FDR) and false alarm rate (FAR). The contribution maps based on the SHAP values and reconstruction errors are then compared to demonstrate the effectiveness of the proposed method.

## A. CSTR Process

A schematic of the CSTR process is presented in [28]. Simulations of normal and faulty data were generated every 60 min for 20 h of operation under varying conditions. The sampling interval for all variables was 1 min. Ten faults are simulated referring to [28]. To further analyze the simultaneous faults, a case in which faults 4 and 5 occur together is added as the 11th fault. These faults were introduced from the 200th sample.

The hyperparameters of the network were determined through a grid search algorithm based on the reconstruction error of the validation set, which was 30% of the training set. In this case, the nodes of the AAE including the two encoding layers and the two decoding layers are [32, 16, 4, 16, 32]. In the discriminator, the network structure of two dense layers with a dropout and sigmoid activation was set as [16, 8, 1]. The batch size was set to 64 and the learning rates for the AE and discriminator were 0.0001 and 0.000 01, respectively. The structures of the AE and VAE models were constructed in the same manner as the AE part of the AAE. For comparison, four principal components in the PCA were applied.

Table I presents the detection results of the four methods on the 11 faults, where bold represents the best F1 score. The FDR and FAR were calculated based on the combined index, and the F1 score, which combines precision and recall into a single metric, was also calculated to compare the FDR and FAR simultaneously. It is clear that the AAE-based model shows the best fault detection performance of the four methods.

Fault isolation follows the detection of a fault and seeks to find the true faulty variable among all candidate variables. Here, sensor drift faults and their simultaneous faults are analyzed because the root causes are clearly identified solely from them.

First, we investigate fault 7, which is a sensor drift in variable $T .$ The fault isolation results using the traditional reconstructionand SHAP-based contributions are shown in Fig. 3(a). For each sample, the contribution of each variable was calculated and visualized. Owing to the smearing effect [30], normal variables (such as ) show large contribution values in the reconstructionbased contribution. Occasionally, the smearing effect leads to incorrect isolation, particularly for incipient faults. However, the SHAP-based method provides an improved fault isolation performance and eliminates the smearing effect. Variable , the true faulty variable, is correctly isolated for almost all of the faulty samples.

To demonstrate the effectiveness of the proposed method, fault 11, where faults 4 and 5 occur simultaneously, was used. The fault isolation results of two faults that occur at the same time are shown in Fig. 3(b). The reconstruction-based method designates $Q _ { c }$ as the faulty variable in addition to the root cause variables $C _ { i }$ and $T _ { i }$ . Traditional methods cannot guarantee the correct fault source backtracking for multiple sensor faults, and the reason for the smearing effect has therefore been analyzed. Interestingly, the SHAP-based method clearly identifies variables $C _ { i }$ and $T _ { i }$ as faulty sources. The proposed method is effective in the presence of multiple faults. It was found that the smearing effect was weakened in comparison to the existing fault diagnosis method.

Fig. 3(c) and (d) shows the distribution of the SHAP and feature values for the dataset of faults 7 and 11. A dot indicates a sample, and the color of the dot indicates the relative value of the feature. Fig. 3(c) shows that the variable has the largest SHAP value, which means that has the greatest influence on the high monitoring index. Similarly, in Fig. 3(d), the faulty variables <sub>i</sub> and $C _ { i }$ show extreme SHAP values with high feature values, which means that they have a significant impact on fault 11.

## B. TE Process

The TE process is additionally tested to verify the proposed method. The TE process is based on a simulation containing five major unit operations: reactor, condenser, separator, compressor, and stripper. There are 52 variables, including 22 process variables and 19 composition variables, i.e., X1–X41, and 11 manipulated variables, i.e., X42–X52. Please refer to [29] for the flowsheet of the TE process and the 21 fault types. The strong nonlinearity and complex fault patterns of this process are

![](images/a5b5b9cfb34cef0fb1d206f46e94132d299b997d8bd680908d12063adbbeec4e.jpg)  
(a)

![](images/ffcf1e7a6ce6133585ea60ae3b292dad37bc146ad76ced48769aae7a72b8b199.jpg)  
(b)

![](images/dc7f1dae41cf7e29f618aaa7aca4371d6e6a1d76f51c4bc98662d7b6dba6296b.jpg)  
(c)

![](images/7e34019fe2176b3c89492a842eff69dcc627a2214425b857bea4854c85047bfa.jpg)  
(d)  
Fig. 3. Fault diagnosis results for the CSTR process. Contribution map for faults (a) 7 and (b) 11. Distribution (points) and density (grey shade) of SHAP values for faults (c) 7 and (d) 11.

TABLE II  
FDR, FAR, AND F1 SCORE OF THE COMBINED INDEX FOR THE TE PROCESS

<table><tr><td rowspan="2">Fault No.</td><td colspan="3">PCA</td><td colspan="3">AE</td><td colspan="3">VAE</td><td colspan="3">AAE</td></tr><tr><td>FDR</td><td>FAR</td><td>F1 score</td><td>FDR</td><td>FAR</td><td>F1 score</td><td>FDR</td><td>FAR</td><td>F1 score</td><td>FDR</td><td>FAR</td><td>F1 score</td></tr><tr><td>1</td><td>0.9963</td><td>0.0625</td><td>0.9919</td><td>0.9963</td><td>0.0438</td><td>0.9938</td><td>0.9950</td><td>0.0313</td><td>0.9944</td><td>0.9975</td><td>0.0125</td><td>0.9975</td></tr><tr><td>2</td><td>0.9875</td><td>0.0313</td><td>0.9906</td><td>0.9863</td><td>0.0125</td><td>0.9918</td><td>0.9875</td><td>0.0125</td><td>0.9925</td><td>0.9888</td><td>0.0063</td><td>0.9937</td></tr><tr><td>3</td><td>0.3000</td><td>0.2938</td><td>0.4416</td><td>0.1613</td><td>0.0438</td><td>0.2756</td><td>0.1838</td><td>0.1500</td><td>0.3028</td><td>0.2113</td><td>0.1563</td><td>0.3400</td></tr><tr><td>4</td><td>0.9538</td><td>0.0313</td><td>0.9732</td><td>0.8275</td><td>0.0375</td><td>0.9019</td><td>0.9038</td><td>0.0313</td><td>0.9463</td><td>0.9888</td><td>0.0813</td><td>0.9863</td></tr><tr><td>5</td><td>0.4788</td><td>0.0813</td><td>0.6405</td><td>0.3538</td><td>0.0375</td><td>0.5197</td><td>0.4113</td><td>0.0438</td><td>0.5792</td><td>0.4163</td><td>0.0313</td><td>0.5852</td></tr><tr><td>6</td><td>1.0000</td><td>0.0188</td><td>0.9981</td><td>1.0000</td><td>0.0438</td><td>0.9956</td><td>1.0000</td><td>0.0250</td><td>0.9975</td><td>1.0000</td><td>0.0063</td><td>0.9994</td></tr><tr><td>7</td><td>1.0000</td><td>0.0313</td><td>0.9969</td><td>1.0000</td><td>0.0063</td><td>0.9994</td><td>1.0000</td><td>0.0250</td><td>0.9975</td><td>1.0000</td><td>0.0063</td><td>0.9994</td></tr><tr><td>8</td><td>0.9913</td><td>0.1688</td><td>0.9790</td><td>0.9925</td><td>0.0938</td><td>0.9869</td><td>0.9875</td><td>0.0750</td><td>0.9863</td><td>0.9913</td><td>0.0688</td><td>0.9888</td></tr><tr><td>9</td><td>0.2963</td><td>0.4063</td><td>0.4301</td><td>0.1388</td><td>0.1813</td><td>0.2362</td><td>0.1688</td><td>0.2563</td><td>0.2766</td><td>0.1663</td><td>0.2688</td><td>0.2725</td></tr><tr><td>10</td><td>0.6038</td><td>0.0813</td><td>0.7454</td><td>0.5913</td><td>0.0188</td><td>0.7414</td><td>0.6050</td><td>0.0250</td><td>0.7516</td><td>0.6875</td><td>0.0563</td><td>0.8094</td></tr><tr><td>11</td><td>0.7388</td><td>0.0938</td><td>0.8407</td><td>0.6600</td><td>0.0500</td><td>0.7904</td><td>0.7238</td><td>0.0813</td><td>0.8319</td><td>0.8125</td><td>0.1125</td><td>0.8856</td></tr><tr><td>12</td><td>0.9975</td><td>0.2375</td><td>0.9756</td><td>0.9900</td><td>0.0563</td><td>0.9894</td><td>0.9938</td><td>0.1625</td><td>0.9809</td><td>0.9938</td><td>0.1313</td><td>0.9839</td></tr><tr><td>13</td><td>0.9613</td><td>0.0250</td><td>0.9777</td><td>0.9488</td><td>0.0250</td><td>0.9712</td><td>0.9550</td><td>0.0375</td><td>0.9732</td><td>0.9538</td><td>0.0063</td><td>0.9757</td></tr><tr><td>14</td><td>1.0000</td><td>0.0625</td><td>0.9938</td><td>1.0000</td><td>0.0188</td><td>0.9981</td><td>1.0000</td><td>0.0375</td><td>0.9963</td><td>1.0000</td><td>0.0250</td><td>0.9975</td></tr><tr><td>15</td><td>0.1925</td><td>0.0375</td><td>0.3208</td><td>0.1600</td><td>0.0188</td><td>0.2750</td><td>0.2100</td><td>0.0188</td><td>0.3460</td><td>0.3150</td><td>0.0313</td><td>0.4768</td></tr><tr><td>16</td><td>0.4875</td><td>0.3938</td><td>0.6225</td><td>0.4613</td><td>0.3125</td><td>0.6054</td><td>0.4888</td><td>0.4125</td><td>0.6221</td><td>0.6313</td><td>0.5313</td><td>0.7266</td></tr><tr><td>17</td><td>0.9500</td><td>0.0813</td><td>0.9663</td><td>0.9038</td><td>0.0313</td><td>0.9463</td><td>0.9125</td><td>0.0375</td><td>0.9505</td><td>0.9338</td><td>0.0250</td><td>0.9632</td></tr><tr><td>18</td><td>0.9125</td><td>0.0375</td><td>0.9505</td><td>0.9200</td><td>0.0375</td><td>0.9546</td><td>0.9138</td><td>0.0500</td><td>0.9500</td><td>0.9313</td><td>0.0688</td><td>0.9576</td></tr><tr><td>19</td><td>0.3700</td><td>0.0813</td><td>0.5338</td><td>0.1763</td><td>0.0500</td><td>0.2972</td><td>0.1650</td><td>0.0625</td><td>0.2803</td><td>0.2063</td><td>0.0125</td><td>0.3413</td></tr><tr><td>20</td><td>0.6238</td><td>0.0125</td><td>0.7671</td><td>0.6138</td><td>0.0125</td><td>0.7595</td><td>0.6163</td><td>0.0125</td><td>0.7614</td><td>0.7275</td><td>0.0250</td><td>0.8398</td></tr><tr><td>21</td><td>0.5263</td><td>0.1500</td><td>0.6763</td><td>0.3975</td><td>0.1000</td><td>0.5608</td><td>0.4800</td><td>0.1125</td><td>0.6389</td><td>0.4788</td><td>0.1563</td><td>0.6341</td></tr></table>

suitable to demonstrate the advantages of the proposed method. All faults had 960 samples, and abnormal events occurred after the 160th sample. As the data characteristics changed, the structural parameters of AE were retuned and changed to [64, 32, 8, 32, 64] along with the leaky ReLU activation function. The batch size and learning rate were the same value as in the CSTR case.

The FDR, FAR, and F1 scores of the combined index for all faults were compared with those of the other methods in Table II. The highest F1 scores are indicated in bold. For easily detectable faults, such as 1, 2, 6, 7, 8, 12, 13, 17, and 18, which have large magnitudes, most of the methods yield FDRs of greater than 90%. For small faults, such as faults 3, 9, and 15, the FDRs achieved by all four methods were considerably low. Overall, the proposed method achieved a superior monitoring performance for the TE process.

![](images/e20d6eb76e83f832b85e06f78782b2d10ac2d8a3d61d7211ee2edbb7ec8c6129.jpg)  
Fig. 4. Contribution map of the TE process for fault 1.

For fault 1, the compositions of A and C suddenly increased. When fault 1 occurs, the MV values of the A feed flow (X44) are increased to maintain the composition of A. The flow rate of feed A (X1) increases owing to the action of the controller. Since C and E are involved in an equimolar reaction, variations in the C composition in the feed flow eventually affect the composition of E in the product flow. The excess component E is vaporized in the stripper by increasing the steam flow (X19) by adjusting the steam valve (X50). Therefore, the stripper temperature (X18) is higher.

Fig. 4 shows the diagnosis results for fault 1 using both reconstruction- and SHAP-based contributions. Since the process data were generated by implementing controllers, the process variables were stabilized after approximately the 400th sample and returned to normal operation conditions, except for the faulty variables. With the reconstruction-based method, the most significant variables after stabilization are X34 and X50, which are component F and the stripper steam valve, respectively. X50 is a relatively minor faulty variable, and X34 is not even relevant, indicating that the reconstruction-based method still suffers from a smearing effect. By contrast, in the SHAP-based method, after the process was stabilized, the faulty variables were reported as X1, X44, X18, X19, and X50. These variables are consistent with fault 1.

Fig. 5 shows that the variables were sorted by the normalized contributions. With the SHAP-based method, the first two variables relevant to the A feed rate account for more than 40% of the total contribution. Fewer variables can explain the same level of accumulated contribution when using the proposed method in comparison with the reconstruction-based approach. Overall, as shown in Table III, the cumulative contribution of the top 5 contributing variables under all fault types was higher with the SHAP-based method. It was easier to pinpoint the faulty variables using the proposed method than the reconstruction-based approach.

![](images/0ac1af24510f4b2025caf13820b788bf495cfcfa68d575271c411f5918edc455.jpg)  
Fig. 5. Contribution plot of the TE process for fault 1.

TABLE III  
MEAN AND STANDARD DEVIATION OF THE CUMULATIVE PERCENTAGES FOR THE TOP-5 CONTRIBUTING VARIABLES ACROSS ALL FAULT TYPES

<table><tr><td>Method</td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>5th</td></tr><tr><td>Reconstruction</td><td>0.2051(0.2039)</td><td>0.2934(0.2220)</td><td>0.3542(0.2392)</td><td>0.3979(0.2333)</td><td>0.4350(0.2286)</td></tr><tr><td>SHAP</td><td>0.2274(0.1751)</td><td>0.3311(0.1943)</td><td>0.4072(0.2108)</td><td>0.4597(0.2097)</td><td>0.5055(0.2076)</td></tr></table>

Hierarchical clustering was applied to both the reconstructionand SHAP-based contribution values. Here, seven fault cases, including faults 1, 2, 4, 5, 8, 11, and 15, were aggregated to determine the patterns of each fault type. To compare the two methods, the adjusted mutual information (AMI) score, which measures the agreement between the clustering result and the ground truth, was calculated. The AMI scores of the reconstruction- and SHAP-based methods were 0.285 and 0.308, respectively. Therefore, it is easier to distinguish the patterns of the fault state when using the proposed method. Distinguishing faults using clustering techniques means that not only can a fault be diagnosed using the SHAP value for a single case, but it also allows the distinction between several different fault cases.

A heatmap of SHAP values sorted based on the clustering results is shown in Fig. 6. The SHAP values of the features were normalized to show the patterns more clearly. The fault types in each cluster were identified and annotated. Local explanation embeddings reveal several subgroups of samples that share similar faulty patterns, where the samples are grouped based on their explanations. For example, faults 1, 2, and 8 were relevant to the feed composition and appeared adjacent to each other in the heatmap. Faults 4 and 11 were related to the reactor cooling water, and faults 5 and 15 were related to the condenser cooling water. They have very similar patterns, and thus, the two types are not differentiated and repeatedly appear. This grouping of samples can reveal high-level structures in fault datasets and insight into generalizing the characteristics of the fault types.

![](images/00f340e52bd5fb22bd7676e0ea364f7a7ab297260282e8a665c68f3e9d56687c.jpg)  
Fig. 6. Fault map of TE process: Hierarchical clustering of faulty samples based on their local SHAP values.

## V. CONCLUSION

In this article, a new fault diagnosis method for industrial processes was proposed based on the SHAP values that explained the black-box model. Despite their superior performance in fault detection, previous deep learning models had limitations in interpreting the model results and providing rich information about the faults. We proposed a framework that seamlessly integrated the existing unsupervised fault detection method and the SHAP value. SHAP assigned each variable a contribution value for a monitoring index calculated from the AAE model. Rather than simply using the reconstruction error, more accurate diagnostic results can be obtained by determining the input variables that influenced the output of the latent and reconstruction vectors. The diagnosis method was examined through the CSTR and TE processes. The results demonstrated that both single fault and multiple faults can be successfully isolated. The proposed method can be used by process operators to understand the prediction results of the black-box model in real-time and to determine the proper operation strategies. Despite the considerable results achieved in this study, the proposed method had limitations in estimating the causal effects of process variables. Accordingly, promising next steps involve developing an XAI-based fault propagation method that can examine causal inference between the fault status and the physical behavior.

## REFERENCES

[1] E. L. Russell, L. H. Chiang, and R. D. Braatz, Data-Driven Methods for Fault Detection and Diagnosis in Chemical Processes. New York, NY, USA: Springer, 2012.

[2] Z. Ge, Z. Song, and F. Gao, “Review of recent research on data-based process monitoring,” Ind. Eng. Chem. Res., vol. 52, pp. 3543–3562, 2013.

[3] K. E. Pilario, M. Shafiee, Y. Cao, L. Lao, and S.-H. Yang, “A review of kernel methods for feature extraction in nonlinear process monitoring,” Processes, vol. 8, 2020, Art. no. 24.

[4] O. AlShorman et al., “Sounds and acoustic emission-based early faul diagnosis of induction motor: A review study,” Adv. Mech. Eng., vol. 13, no. 2, 2021, Art. no. 1687814021996915.

[5] Y. Liu, J. Zeng, J. Bao, and L. Xie, “A unified probabilistic monitoring framework for multimode processes based on probabilistic linear discriminant analysis,” IEEE Trans. Ind. Inform., vol. 16, no. 10, pp. 6291–6300, Oct. 2020.

[6] A. Kumar et al., “VMD based trigonometric entropy measure: A simple and effective tool for dynamic degradation monitoring of rolling element bearing,” Meas. Sci. Technol., vol. 33, no. 1, 2021, Art. no. 014005.

[7] S. Yin, S. X. Ding, A. Haghani, H. Hao, and P. Zhang, “A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark tennessee eastman process,” J. Process Control, vol. 22, no. 9, pp. 1567–1581, 2012.

[8] O. AlShorman et al., “A review of artificial intelligence methods for condition monitoring and fault diagnosis of rolling element bearings for induction motor,” Shock Vib., vol. 2020, pp. 1–20, 2020.

[9] H. Wu and J. Zhao, “Deep convolutional neural network model based chemical process fault diagnosis,” Comput. Chem. Eng., vol. 115, pp. 185–197, 2018.

[10] J. Zhu, H. Shi, B. Song, Y. Tao, and S. Tan, “Convolutional neural network based feature learning for large-scale quality-related process monitoring,” IEEE Trans. Ind. Inform., vol. 18, no. 7, pp. 4555–4565, Jul. 2022.

[11] Z. Zhang and J. Zhao, “A deep belief network based fault diagnosis model for complex chemical processes,” Comput. Chem. Eng., vol. 107, pp. 395–407, 2017, doi: 10.1016/j.compchemeng.2017.02.041.

[12] J. Yuan and Y. Tian, “A multiscale feature learning scheme based on deep learning for industrial process monitoring and fault diagnosis,” IEEE Access, vol. 7, pp. 151189–151202, 2019. [Online]. Available: https://ieeexplore.ieee.org/document/8871174/

[13] J. Dong, R. Sun, K. Peng, Z. Shi, and L. Ma, “Quality monitoring and root cause diagnosis for industrial processes based on Lasso-SAE-CCA,” IEEE Access, vol 7, pp. 90230–90242, 2019. [Online]. Available: https: //ieeexplore.ieee.org/document/8752219/

[14] Z. Zhang, T. Jiang, S. Li, and Y. Yang, “Automated feature learning for nonlinear process monitoring–an approach using stacked denoising autoencoder and k-nearest neighbor rule,” J. Process Control, vol. 64, pp. 49–61, 2018.

[15] Z. Zhang, T. Jiang, C. Zhan, and Y. Yang, “Gaussian feature learning based on variational autoencoder for improving nonlinear process monitoring,” J. Process Control, vol. 75, pp. 136–155, 2019.

[16] Z. Chai and C. Zhao, “A fine-grained adversarial network method for crossdomain industrial fault diagnosis,” IEEE Trans. Autom. Sci. Eng., vol. 17, no. 3, pp. 1432–1442, Jul. 2020. [Online]. Available: https://ieeexplore. ieee.org/document/8950281/

[17] K. Jang, S. Hong, M. Kim, J. Na, and I. Moon, “Adversarial autoencoder based feature learning for fault detection in industrial processes,” IEEE Trans. Ind. Inform., vol. 18, no. 2, pp. 827–834, Feb. 2022. [Online]. Available: https://ieeexplore.ieee.org/document/9426453/

[18] P. Linardatos, V. Papastefanopoulos, and S. Kotsiantis, “Explainable AI: A review of machine learning interpretability methods,” Entropy, vol. 23, no. 1, pp. 1–45, 2021.

[19] W. Samek, G. Montavon, S. Lapuschkin, C. J. Anders, and K. R. Müller, “Explaining deep neural networks and beyond: A. review of methods and applications,” Proc. IEEE Proc. IRE, vol. 109, no. 3, pp. 247–278, 2021.

[20] M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should I trust you?’ Explaining the predictions of any classifier,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining, 2016, pp. 1135–1144.

[21] A. Shrikumar, P. Greenside, and A. Kundaje, “Learning important features through propagating activation differences,” in Proc. Int. Conf. Mach. Learn., 2017, pp. 3145–3153.

[22] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Adv. Neural Inf. Process. Syst., vol. 30, pp. 4765–4774, 2017.

[23] P. Agarwal, M. Tamer, and H. Budman, “Explainability: Relevance based dynamic deep learning algorithm for fault detection and diagnosis in chemical processes,” Comput. Chem. Eng., vol. 154, 2021, Art. no. 107467, doi: 10.1016/j.compchemeng.2021.107467.

[24] J. Lee, I. Noh, J. Lee, and S. W. Lee, “Development of an explainable fault diagnosis framework based on sensor data imagification: A case study of the robotic spot-welding process,” IEEE Trans. Ind. Inform., vol. 18, no. 10, pp. 6895–6904, Oct. 2022.

[25] J. Jiao, W. Zhen, W. Zhu, and G. Wang, “Quality-related root cause diagnosis based on orthogonal kernel principal component regression and transfer entropy,” IEEE Trans. Ind. Inform., vol. 17, no. 9, pp. 6347–6356, Sep. 2021.

[26] L. Yao and Z. Ge, “Industrial Big Data modeling and monitoring framework for plant-wide processes,” IEEE Trans. Ind. Inform., vol. 17, no. 9, pp. 6399–6408, Sep. 2021.

[27] H. H. Yue and S. J. Qin, “Reconstruction-based fault identification using a combined index,” Ind. Eng. Chem. Res., vol. 40, no. 20, pp. 4403–4414, 2001.

[28] K. E. S. Pilario and Y. Cao, “Canonical variate dissimilarity analysis for process incipient fault detection,” IEEE Trans. Ind. Inform., vol. 14, no. 12, pp. 5308–5315, Dec. 2018.

[29] J. J. Downs and E. F. Vogel, “A plant-wide industrial process control problem,” Comput. Chem. Eng., vol. 17, no. 3, pp. 245–255, 1993.

[30] J. A. Westerhuis, S. P. Gurden, and A. K. Smilde, “Generalized contribution plots in multivariate statistical process monitoring,” Chemometrics Intell. Lab. Syst., vol. 51, no. 1, pp. 95–114, 2000.

![](images/308f334072e3388f0a92bdc32c07671bce75b193dd68dcfb6e93391fe2d6311c.jpg)  
Kyojin Jang received the B.S. degree in chemical and biomolecular engineering from Yonse University, Seoul, South Korea, in 2018, where she is currently working toward the Ph.D. degree.  
Her research interest includes fault diagnosis based on machine learning.

![](images/a95e3f5e3d9f62d622bbe3371c29d168c72ed2792204ead7c07130d23cf1c856.jpg)

Karl Ezra Salgado Pilario received the B.Sc. (summa cum laude) and M.Sc. degrees in chemical engineering from the University of the Philippines, Quezon City, Philippines, in 2012 and 2015, respectively and the Ph.D. degree in energy and power from Cranfield University, Cranfield, U.K., in 2020.

He also participated in the Oxford Machine Learning Summer School in 2022. He is currently an Associate Professor with the Department of Chemical Engineering, University of the

Philippines, and also with the Artificial Intelligence Program at the same institution. His current research interests include industrial process data analytics and machine learning applications in energy, water, and environmental process systems.

![](images/1e38b0e0724a30d9bc6f5ad0d4e922f2ef5fca61a7b1db19693dd8545230aaf2.jpg)

Nayoung Lee received the B.S. degree in naval architecture and ocean engineering from Seoul National University, Seoul, South Korea, in 2018, where she is currently working toward the Ph.D. degree.

Her research interest includes application of machine learning in offshore process system and digital twin.

![](images/cd603317bf86d491e5292900f5162cabd800bf0caa9aae46d2a5b9a091e16697.jpg)  
cess design and modeling.

Il Moon received the B.S. degree in chemical engineering from Yonsei University, Seoul, South Korea, in 1983, the M.S. degree from Korea Advanced Institute of Science and Technology (KAIST), Daejeon, South Korea, in 1985, and the Ph.D. from Carnegie Mellon University, Pittsburgh, PA, USA, in 1992.

He was a Postdoctoral Research Fellow with Carnegie Mellon University. He is currently a full-time Professor with Yonsei University. His research interests include complex industrial pro-

![](images/84034047e64c17a39a2bd35745c1559aa02ccd18819e84481be3fdd1ee299d25.jpg)

Jonggeol Na received the B.S. degree in chemical and biomolecular engineering and the Ph.D. degree in chemical and biological engineering from Seoul National University, Seoul, South Korea, in 2013 and 2018, respectively.

He was a Postdoctoral Research Fellow with Carnegie Mellon University, USA, and the Korea Institute of Science and Technology (KIST), South Korea. He is currently an Assistant Professor with Ewha Womans University, Seoul, South Korea. His research with Seoul National

University focused on a computational science approach to the design and optimization of process systems. His research interests include the autonomous discovery of nonintuitive process system designs through artificial intelligence and multiscale simulations to accelerate the conceptual design of nontraditional electrical energy-based processes to improve the sustainability in the chemical industry.