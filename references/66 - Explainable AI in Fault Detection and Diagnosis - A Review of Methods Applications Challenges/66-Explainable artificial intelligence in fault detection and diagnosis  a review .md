# Explainable artificial intelligence in fault detection and diagnosis: a review of methods, applications, and implementation challenges

Ahmed Maged, Salah Haridy, Mohamed Hosny & Herman Shen

To cite this article: Ahmed Maged, Salah Haridy, Mohamed Hosny & Herman Shen (09 Jan 2026): Explainable artificial intelligence in fault detection and diagnosis: a review of methods, applications, and implementation challenges, Journal of Industrial and Production Engineering, DOI: 10.1080/21681015.2026.2613018

To link to this article: https://doi.org/10.1080/21681015.2026.2613018

![](images/ff3dbaaf76023f28e098993e08db96f573292224ccdd8e024e86ed241eba4b63.jpg)

Published online: 09 Jan 2026.

![](images/4f910fa7553bed31432daea9ab1316736769d031a1e71873659dc85700460565.jpg)

Submit your article to this journal

![](images/ada82909d8b5e1f57a880c7598d8a30976692f43b79594707931116027481815.jpg)

Article views: 220

![](images/0c3d2a41fec2ad33e88fabf3b3be54062d540d543c481871b3a023977147c688.jpg)

View related articles

![](images/0320aa932c87e847711694a62e573a617cfa8e2d4a3e816d30db5f6476b0ac8b.jpg)

View Crossmark data

![](images/133a7737e29f851dcde45302f206f8ae8569ad088da66b2c24cd2a2886ced8e5.jpg)

Citing articles: 3 View citing articles

Check for updates

# Explainable artificial intelligence in fault detection and diagnosis: a review of methods, applications, and implementation challenges

Ahmed Maged <sup>a,b</sup>, Salah Haridy <sup>c</sup>, Mohamed Hosny <sup>d</sup> and Herman Shen<sup>b,e</sup>

<sup>a</sup>Department of Industrial Engineering, American University of Sharjah, Sharjah, UAE; <sup>b</sup>Mechanical Engineering Department, University of North Texas, Denton, TX, USA; <sup>c</sup>Department of Industrial Engineering and Engineering Management, College of Engineering, University of Sharjah, Sharjah, United Arab Emirates; <sup>d</sup>IRC for Finance and Digital Economy, KFUPM Business School, King Fahd University of Petroleum and Minerals, Dhahran, Saudi Arabia; <sup>e</sup>Department of Mechanical and Aerospace Engineering, Ohio State University, Columbus, OH, USA

## ABSTRACT

Industry 4.0 increasingly relies on AI methods for fault detection and diagnosis (FDD). However, advanced machine learning models lack transparency, reducing trust in safetycritical settings. This review examines eXplainable AI (XAI) methods adapted for industrial FDD. It also proposes a taxonomy spanning model-agnostic methods, modelspecific approaches, and hybrid rule-based schemes. For each category, the paper explains how the methods reveal fault-related decision logic and examine their impact on diagnostic accuracy. The analysis shows that SHAP and feature-importance methods are the most widely used in FDD applications. Other methods (e.g., LIME) have seen limited adoption, partly due to scalability concerns. This study further examines limitations including high computational cost, restricted real-time performance, and scalability constraints. The findings indicate that although model-specific methods enhance interpretability, they continue to face challenges in scalability. The study also outlines key research questions related to evaluating explanation quality, integrating XAI into realtime FDD systems.

ARTICLE HISTORY

Received 12 August 2025

Revised 4 December 2025

Accepted 1 January 2026

KEYWORDS XAI; deep learning; machine learning; reliability; FDD

## 1. Introduction

Fault detection and diagnosis (FDD) has become a key concern in modern manufacturing systems where datadriven methods are central to maintaining reliability and safety. In recent years, Machine Learning (ML) algorithms, including Deep Learning (DL) models, have shown great promise in automating fault detection and diagnosis tasks [1]. However, these models are often viewed as black boxes. They operate in a way that is hard to interpret, which limits their acceptance in safety-critical settings. High accuracy alone is not enough, as practitioners need to know whether the model is using relevant features and whether its decisions can be trusted.

To address this, the field of eXplainable Artificial Intelligence (XAI) has emerged. XAI aims to provide insight into model behavior while keeping acceptable accuracy. As shown in Figure 1, XAI methods help users understand how predictions are formed, support fault tracing, and guide repair or maintenance actions. Interest in XAI has grown in recent years, although the idea dates back decades, with the term introduced by Van Lent et al. in 2004 [2], and the broader concept of explainability in ML has roots dating back to the 1970s, as referenced in Adadi and Berrada [3].

While plenty of review papers explore various XAI methods, detailing their mechanisms and general applications, this paper is a bit more focalized. It focuses on the application of XAI in fault detection and diagnosis, specifically within the manufacturing industry. This critical yet complex domain demands a targeted exploration of XAI’s capabilities and limitations. Review papers such as [4] and [5] are recommended for readers interested in a broader understanding of XAI methods.

This study aims to bridge the gap between advanced fault detection models and their practical adoption by providing a focused review of XAI techniques developed and applied for industrial fault detection and diagnosis. The objective is to clarify how these methods enhance interpretability, support trust in automated decision-making, and enable safer manufacturing operations.

![](images/a526f3385df2e8418f5e63b8d5e7421651b0ccd4369d7273450e8161759b349a.jpg)  
Figure 1. Schematic of XAI operational framework.

Accordingly, the main contributions of this study are as follows:

(1) It provides an in-depth review of XAI techniques applied specifically to industrial FDD, covering over a decade of research (2013–2024) across manufacturing, energy, and machinery domains.

(2) It proposes a structured taxonomy of XAI methods (intrinsic vs. post-hoc, local vs.

global, model-specific vs. agnostic) tailored to fault diagnosis and systematically compares how these approaches improve fault detection performance and interpretability.

(1) It presents a quantitative trend analysis that highlights the growing dominance of deep learningbased XAI methods after 2020.

(2) It critically evaluates current limitations such as computational cost, lack of explanation evaluation metrics, and challenges in real-time implementation, and outlines future research directions.

The literature review followed a structured process. Reputable databases, including IEEE Xplore, ScienceDirect, and SpringerLink, were selected for their strong coverage of fault detection, diagnosis, and XAI research. The search used keywords such as “Explainable AI,” “XAI,” “fault detection,” “fault diagnosis,” and “interpretable machine learning,” combined with Boolean operators (AND, OR). Papers published between 2013 and 2023 in peer-reviewed journals or conferences were included if they focused on FDD or XAI applications in industrial systems, described interpretability techniques, and provided experimental or case study validation. Studies without clear explainability focus or real-world application were excluded. The next section provides an overview of FDD followed by a general introduction to XAI. Section 3 then reviews existing XAI methods and algorithms applied in FDD and discusses key factors guiding their selection. Building on this structure, Section 3 also highlights how the main XAI families difer in scope, output, and suitability for industrial settings. Section 4 then summarizes the main insights from the reviewed studies and highlights the trends observed across domains and techniques. Section 5 presents the conclusion, outlines the main gaps identified in the field, and points to future research directions.

## 2. Background

## 2.1. Fault detection and diagnosis

Fault detection and diagnosis methods have been crucial characteristics of safety-critical applications. Nonetheless, due to the demands of higher productivity and dependable operation, fault detection and diagnosis are incorporated into almost all sophisticated systems and pieces of equipment. An efective FDD system should be able to monitor overall system health, and accurately identify and locate emerging faults, to enable the safe removal or correction of faulty components within the system [6].

Model-based fault detection and diagnosis require an accurate mathematical model of the process. Most initial work on fault detection and diagnosis was carried out model-based approaches utilizing the adaptive observers and system identification models of the processes [7]. They work well for small systems with clear dynamics. Their performance, however, drops when the system contains unmodeled disturbances or uncertainties [8]. In contrast, data-driven approaches do not rely on predefined physical rules. When efectively utilized, the large volumes of collected data can support accurate prediction of fault occurrence or system malfunction. These data-driven approaches usually involve statistical techniques such as control charts, signal processing, or Machine Learning (ML). ML has become increasingly popular due to its ability to handle large and complex datasets, its adaptability to changing conditions, and its ability to provide accurate and reliable predictions.

In general, ML-based fault detection methods can be described through three main approaches:

● The first approach relies on supervised learning, where the model is trained using labeled data that include both normal and faulty conditions. The model learns the boundary between these states and assigns new observations accordingly. This approach works well when suficient labeled data exist for the operating modes of interest.

● The second approach is based on anomaly detection. In this case, the model is trained only on data that represent normal operation. At deployment, samples that deviate from the learned normal behavior are flagged as potential faults. This is useful when faulty data are limited or not well labeled.

● The third approach uses residual analysis. A regression model predicts the expected output of the equipment, and the diference between the predicted and observed value is monitored over time. When this residual exceeds a set threshold, a fault is indicated. This method is often used when the system output is continuous and predictable.

For fault diagnosis, the supervised approach is extended so the model distinguishes among several faulty classes. This allows the system not only to detect the presence of a fault but also to identify its specific type. Interested readers can refer to other review papers that discuss fault detection and diagnosis methods in detail, such as [7] and [9], since the focus of the paper is on XAI, rather than on traditional ML-based FDD techniques.

## 2.2. Understanding explainable machine learning

The primary goal of XAI in fault detection and diagnosis is to provide interpretable and transparent models that can help humans understand the decision-making process of the ML algorithm [10]. In fault detection, the goal is to identify when a fault has occurred, or when a component is likely to fail. This is typically achieved by analyzing sensor data from the machine and detecting any anomalies or deviations from normal operating behavior.

In that context, XAI can be helpful in two ways. First is the Model-Based Explanation, which occurs when a fault is detected by an algorithm and the goal is to interpret this detection. Understanding the reasons behind a specific detected fault is straightforward for inherently interpretable models (like logistic regression or shallow decision trees). For less interpretable models (such as complex neural networks), post-hoc techniques like Local Interpretable Model-Agnostic Explanations (LIME) and Shapley Additive Explanations (SHAP) are employed to clarify why the model identifies certain instances as faults. This ensures consistency between the model’s detection and its explanation [11].

Second is the Data-Centric Explanation, which is encountered when a fault is identified by an expert without a model, the focus is on understanding why the data is flagged as faulty, independent of any detection models. If an expert uses an unknown algorithm, there’s a gap between the explanation and the detection method. This case emphasizes understanding faults in the data where no model is available, acting as surrogate methods for non-accessible fault detection models [12].

One might argue that XAI can only be used in the diagnosis phase rather than the detection phase. Nevertheless, it is important to recognize that the associations identified by ML algorithms do not inherently imply causality. There may be unobserved factors that are responsible for observed correlations among variables. In such cases, XAI can help understand the logic behind the predictions. For example, let us consider a computer numerical control (CNC) machine that operates within certain performance parameters monitored using multiple sensors, such as temperature, pressure, and vibration sensors. In the fault detection phase, the monitoring system may predict or detect that the machining head is overheating, while the XAI component identifies the feature or variable that contributed most to this output. In contrast, in fault diagnosis, one would search for the root cause of the heating problem (e.g. no coolant fluid)

## 3. Techniques for XAI in fault detection and diagnosis

XAI provides interpretable and transparent explanations for the decisions made by Machine Learning (ML) models. It aims to bridge the gap between the complex inner workings of these models and the need for human comprehension and trust in their outputs. ML interpretability methods can be classified based on various criteria as in Figure 2. A primary distinction diferentiates whether interpretability is achieved through model’s design (intrinsic analysis) or post-hoc analysis

Intrinsic methods embed interpretability into model construction by restricting complexity and utilizing intrinsically interpretable models like sparse linear models or short decision trees. Interpretability arises from the model’s transparent structure and inference process. In contrast, post-hoc methods are applied after a model has been trained to generate explanations. Techniques like permutation feature importance are model-agnostic and can be applied after training. However, post-hoc approaches may also be used with intrinsically interpretable models, such as computing permutation importance for decision trees.

One can also categorize XAI methods as model-specific and model-agnostic. The distinction between these two groups lies in their respective applicability to diferent types of ML models. Model-specific methods are specifically designed to work with a particular type or family of models. It usually uses the model’s internal structure, characteristics, or properties to provide explanations. These methods rely on the unique assumptions, features, or computations of the specific model, allowing for tailored explanations. For example, interpreting the regression weights in a linear model is a classic example of model-specific interpretation. Alternatively, model-agnostic methods are designed to operate independently of the underlying ML model. They can be applied to any type or family of models without requiring knowledge of the specific internal details. Model-agnostic methods aim to provide explanations by analyzing the model’s input-output behavior without exploiting any model-specific characteristics. These methods are particularly useful for addressing the black-box nature of complex models and providing insights into their decision-making processes. Common model-agnostic techniques include LIME and SHAP since they can be applied to a wide range of models.

![](images/79ca7474d7e2df8567dd0d7b7bbe3feda5133e7bcdb292bc7eabfbd1b0246bd5.jpg)  
Figure 2. XAI diferent categories.

Finally, XAI can be distinguished based on whether the interpretation method elucidates the rationale behind an individual prediction (Local methods) or comprehensively elucidates the behavior of the entire model (Global methods).

The choice between intrinsic and post-hoc explanation methods depends on factors such as domain, problem complexity, and the trade-of between interpretability and performance. In fault detection, posthoc methods are more common, reflecting a preference for accuracy. Intrinsic models embed interpretability in their design but may sacrifice some accuracy. To balance both aspects, several studies have explored hybrid approaches combining intrinsic models. The following section details XAI techniques for fault detection and diagnosis.

For illustration purposes, a dataset containing 1000 samples with 10 features is simulated. Of these features, 5 are directly informative, meaning they correlate with either the faulty or non-faulty class, and 2 are redundant, derived as linear combinations of the informative features to reflect real-world correlations. The dataset represents a practical yet simplified representation of an actual fault detection system where machinery failures are common. Each fault occurrence is recorded with sensor data readings at the time of the fault. Later, XAI tools are applied to the dataset to illustrate how diferent interpretability techniques explain the model’s behavior and highlight the features driving its decisions. Figure 3 shows a scatter plot of the data where it can be seen that input features exhibit diferent relations across each other. Notably, all features exhibit a high frequency of fluctuations without any visible trends or seasonal patterns. The histograms show that most features in the dataset display approximately Gaussian distributions while some of the variables are slightly skewed.

## 3.1. Traditional feature importance

Feature Importance is a common approach to measure how the input feature afects the output. It has been used multiple times to explain faults. Permutation importance is a model-agnostic technique that assesses feature importance by shufling (permuting) the values of a single feature and observing the impact on the model’s performance (e.g. accuracy, F1 score). If shufling a feature leads to a significant drop in performance, that feature is considered necessary. Then, for each feature, the values in the test dataset are systematically permuted to disrupt their relationship with the target, allowing the resulting change in prediction error to be measured. Finally, the new prediction error of the model on this perturbed dataset is calculated. The importance of feature $j$ is the increase in the prediction error as a result of the permuta tion, Importance $( j ) = { \mathsf { E r r o r } } _ { \mathsf { p e r m u t e d } } ( j ) - { \mathsf { E r r o r } } _ { \mathsf { o r i g i n a l } } .$

Aldrich and Auret [13] used permutation importance to explain the importance of input variables for fault detection and diagnosis of steady state faults.

Tree-based feature importance is specific model-specific to decision tree-based algorithms, such as Random Forests and eXtreme Gradient Boosting (XGBoost). It measures the contribution of each feature to the reduction in impurity (e.g. Gini impurity) or error at each node of the tree. First the Gini impurity is calculated at node t by $\begin{array} { r } { G ( t ) = 1 - \sum _ { i = 1 } ^ { k } p _ { i } ^ { 2 } } \end{array}$ where $p _ { j }$ is the proportion of the samples that belong to class i at node t.

$$
\text { Importance } (j) = \frac {\sum_ {t \in T : j \text { splitst }} (p (t) \times \Delta G (t , j))}{\sum_ {t \in T} p (t) \times \Delta G (t)}\tag{1}
$$

where $p ( t )$ is the proportion of samples reaching node $t , \Delta G ( t , j )$ is the decrease in Gini impurity from splitting node t on feature $j ,$ and T is the set of all nodes that split on feature j [14]. Using the simulated dataset, Random Forest is used as a classifications tool. The feature importance is presented in Figure 4. As can be seen, the feature importance plot was able to distinguish the important from non-important features.

In the context of fault detection and diagnosis, examples of explaining XGBoost, Random Forest, and Decision trees using Tree-based feature importance techniques can be found in [15–21].

![](images/6908268aacbed523b4101c50ed88717f8df89de7f5584ed0d4ca2039cf00eeb3.jpg)  
Figure 3. Scatter plot matrix of the simulated data with categories; faulty (orange) non-faulty (blue).

## 3.2. Local interpretable model-agnostic explanations (LIME)

LIME is a model-agnostic method that provides explanations for individual predictions by approximating the model locally with a simpler interpretable model.

Since LIME does not use a specific model (i.e. any model can be used as an interpretable model), it does not have specific equations. For sake of clarification and assuming a linear explanation model the algorithm should work as follows [22]:

1. Select an instance for explanation which is linear as assumed

$$
f (z ^ {\prime}) = \beta_ {0} + \sum_ {j = 1} ^ {n} \beta_ {j} z _ {j} ^ {\prime}\tag{2}
$$

where $\beta _ { j }$ are the coeficients explaining the contribution of each feature, and $z _ { j } ^ { ' }$ are the features of the perturbed sample.

(1) Generate a new dataset consisting of perturbed samples around the selected instance. This is typically done by adding noise to the features of the instance.

![](images/ad9df4711ede94e689a209131491c3090e99118b351817fcfa123093a471e9e3.jpg)  
Figure 4. Feature importance using random Forest classifier.

(2) Assign weights $\pi ( x )$ to these new samples based on their proximity to the original instance. The closer a perturbed sample is to the original instance, the higher its weight. The weights $\pi ( x )$ for the samples are computed using a kernel function that decreases with the distance from the original instance πðxÞ:

$$
\pi (x) = \exp \left(- \frac {d \left(x , z ^ {\prime}\right) ^ {2}}{\sigma^ {2}}\right)\tag{3}
$$

where $d ( x , z ^ { \prime } )$ is the distance between the original instance and the perturbed instance, and σ is a bandwidth parameter.

4. Use the coeficients of the simple model to explain the contribution of each feature to the prediction of the instance

Using the simulated dataset, the LIME values for one instance are shown in Figure 5. It illustrate how the method decomposes the prediction into feature-level contributions. The values in red and green reflect that this feature is afecting either the positive or negative class prediction, respectively. Features highlighted in red contribute toward predicting the instance as belonging to the faulty class, whereas features in green push the prediction toward the “non-faulty” class. The values here represent the contribution of each feature to the prediction. Hence higher values mean higher efect.

![](images/6fb925a5797dfdf8c5c7fe9c3926e01f253d7e667ba22eda0db570d812b82dd0.jpg)  
Figure 5. LIME values for explaining the simulated dataset.

The feature importance rankings obtained from Random Forest and LIME show clear diferences. Random Forest provides a global view of feature relevance, emphasizing variables that consistently enhance model accuracy across the entire dataset. In contrast, LIME generates local explanations that highlight feature importance for specific instances or contexts, which may difer from the overall trends identified by Random Forest.

Several studies have leveraged LIME for fault detection applications. For instance, Srinivasan, et al. [23] applied LIME to explain a model for fault detection in chillers. Lu, et al. [24] used a 1D convolutional neural network (CNN) classifier for rolling bearing health state identification, then employed LIME to interpret the model’s predictions. In Sairam, et al. [25], LIME was combined with XGBoost to build an explainable fault detection model for photovoltaic panels implementable on edge nodes with a real application on photovoltaic panels. Sundarrajan and Rajendran [26] also utilized LIME to interpret the predictions generated by pretrained deep networks such as ${ \mathsf { V G G } } ,$ Inception, and ResNet. Zhang, et al. [27] developed an attention-based interpretable prototypical network for smallsample damage identification with ultrasonic guided waves. Its channel attention module mitigated overfitting while extracting discriminative features. LIME then explained the network’s intrinsic mechanism for damage identification by determining critical input features contributing to its predictions.

## 3.3. Shapley additive explanations (SHAP)

Shapley Additive Explanations (SHAP) assigns an importance value to each feature based on its direct efect on a given prediction. It comes from cooperative game theory through the idea of Shapley values. A Shapley value measures how much a feature adds to the gap between the prediction for one case and the average prediction in the dataset. Shapley values can be expressed as [28]:

$$
\phi_ {i} (f, x) = \sum_ {S \subseteq N \backslash \{i \}} \frac {| S | ! (| N | - | S | - 1) !}{| N | !} \left(f _ {x} (S \cup \{i \}) - f _ {x} (S)\right)\tag{4}
$$

where N is the full set of features, S is any subset that does not include feature $i , | S |$ is the size of the subset, $f _ { x } ( S )$ is the model output when only the features in S are used, and $f _ { x } ( S \cup \{ i \} )$ is the output when feature i is added. The factorial term $\frac { | S | ! ( | N | - | S | - 1 ) ! } { | N | ! }$ gives the weight for each subset based on how many feature orderings can lead to it. SHAP uses these values to form an explanation:

$$
\text { explanation } (x) = \phi_ {0} (f) + \sum_ {i = 1} ^ {M} \phi_ {i} (f, x)\tag{5}
$$

where $\phi _ { 0 } ( f )$ is the average prediction for the dataset and $\textstyle \sum _ { i = 1 } ^ { M } \phi _ { i } ( f , x )$ sums the SHAP values for all features, quantifying their individual contributions to the diference between the model’s prediction for x and the average prediction. Thus, for a simple linear regression model, the SHAP values for each feature in a particular prediction instance would directly correspond to weights multiplied by the deviation of the input from its mean. This reflects each feature’s contribution toward the deviation from the average prediction.

SHAP is applied to the simulated dataset to generate explanations, as shown in Figure 6. Features with a wide spread of SHAP values, such as Feature 4 and 9, have variable impacts on the model’s predictions (i.e significantly afect the output), depending on their values in specific instances. Notably, SHAP produces a feature – importance pattern consistent with that obtained from tree-based methods

The eficacy and potential of SHAP have been highlighted by numerous studies that have utilized its interpretations in fault detection and diagnosis applications. This approach not only enhances the transparency of AI models but also strengthens decision-making by clearly identifying the factors that most significantly drive the model’s predictions.

Hwang and Lee [29] employed SHAP to analyze and interpret feature contributions in a bidirectiona long short-term memory (Bi-LSTM) anomaly detection model for industrial control systems, supporting timely operational responses. Jang, et al. [30] utilized SHAP values with an adversarial autoencoder to build fault maps distinguishing various fault types in chemical processes. The method involves creating an adversarial auto-encoder model and monitoring an index that is based on a weighted average of both the hoteling and squared prediction errors. Then, SHAP values are calculated and analyzed. Accordingly, a fault map is introduced. The proposed method is applied to two chemical process systems. The results demonstrated that the proposed method accurately diagnoses single and multiple faults and can distinguish the global pattern of various fault types. Chowdhury, et al. [31] used SHAP to diagnose faults in 3-D printers after using ensemble learning model of Random Forest and XGBoost for the training.

![](images/2afbd09379c4128319ac2e60569ebeb15768aeb08194ddc76b7a6cedd0404ce9.jpg)  
Figure 6. SHAP values for explaining the simulated dataset.

Brusa, et al. [32] showcased the efectiveness of SHAP in discerning and elucidating the most pivotal features in the context of two models: the Support Vector Machine (SVM) and the k-Nearest Neighbor (kNN). These models were employed to analyze vibration data originating from medium-sized bearings, which were collected and scrutinized for the purpose of classifying rotor faults. Moreover, Choi, et al. [33] incorporated SHAP explanations within an online framework for predictive maintenance in chemical plants.

Baptista, et al. [34] utilized the SHAP model to analyze the outcomes of three algorithms: Linear Regression, Multi-Layer Perceptron, and Echo State Network, with the goal of determining if there is a correlation between the prognostics metrics and the SHAP model’s explanations. They used a baseline dataset containing run-to-failure trajectories from jet engines to evaluate how well the SHAP explanations aligned with established prognostic indicators. The results demonstrated a close tracking between SHAP values and the metrics, with diferences observed among the models of course. Zhang, et al. [35] proposed a gamified framework called ENIGMA to address security challenges in cyber-physical systems. The approach employs Digital Twins as both a security assessment tool and a training environment, where game scenarios feature human and AI participants acting as attackers and defenders. SHAP values are then applied to explain AI decisions related to attack vectors. Similarly, Kumar and Hati [36] employed SHAP methodology to elucidate induction motor fault classification by CNNs

Onchis and Gillich [37] combined LIME and SHAP to improve the interpretability of deep learning models used for classifying accelerometer data. They introduced a compound stability-fit compensation index to reduce the instability of local explanations and enhance their reliability. By integrating LIME and SHAP, the study aimed to identify the location and depth of damaged beams in a transparent and trustworthy way.

Brito, et al. [38] utilized SHAP for fault diagnosis in rotating machinery. Furthermore, the authors conducted a comparison between SHAP and Local Depth-based Feature Importance for the Isolation Forest (Local-DIFFI) in terms of the efectiveness of these models in providing insights and explanations for fault diagnosis in rotating machinery.

## 3.4. Partial dependence plot (PDP) and individual conditional expectation (ICE)

Partial Dependence Plots (PDP) are model-agnostic tools that show how individual features afect predictions. They can help display the average efect of changing one feature at a time. On the other hand, Individual Conditional Expectation (ICE) visualizes how predictions shift for individual data points when a feature changes. While PDP captures general trends, ICE focuses more on the impact of a feature on specific point/instance. The math behind PDP and ICE is conceptually simple and relies on marginalizing or fixing specific features to examine their influence on the model’s predictions. For a feature $X _ { p }$ and a set of features ${ \tt X } _ { c } ,$ the partial dependence function for $X _ { p }$ is defined as:

$$
\mathrm{PD} _ {p} \left(x _ {p}\right) = E _ {X _ {c}} \left[ f \left(x _ {p}, X _ {c}\right) \right] = \int^ {f} \left(x _ {p}, x _ {c}\right) p \left(x _ {c}\right) d x _ {c}\tag{6}
$$

where $f ( x _ { p } , X _ { c } )$ represents the model prediction function, $x _ { p }$ is a specific value for the feature

$x _ { p } , X _ { c }$ are the other features in the dataset. $p ( x _ { c } )$ is the probability density function of $X _ { c } .$ . The expectation $E _ { X _ { c } }$ is calculated over the distribution of the other features, efectively marginalizing out these features. In simple terms, this represents the expected model output over the distribution of all other features while keeping $x _ { p }$ fixed at $x _ { s }$ .

For each individual i in the dataset, the ICE plot for feature $X _ { p }$ is defined as:

$$
\mathrm{ICE} _ {s} ^ {i} (x _ {s}) = f \left(x _ {s}, x _ {c} ^ {i}\right)\tag{7}
$$

where $f \left( \boldsymbol { x _ { p } } , \boldsymbol { x _ { c } ^ { i } } \right)$ is the prediction of the model when $X _ { p }$ is set to $x _ { s }$ and all other features $X _ { c }$ are fixed at their actual values $x _ { c } ^ { i } .$

PDP and ICE are applied to the simulated dataset in Figure 7. Each subplot corresponds to a diferent feature. PDP is the average efect of a feature on the prediction outcome across the dataset. The more nonlinear the line, the greater the feature's influence on the model’s predictive behavior. On the other hand, ICEs are the blue lines plotted for each instance in the dataset showing how the prediction changes with diferent values of the feature while other features are held constant. This provides a more granular view than PDP, highlighting individual variations and potential anomalies. In the simulated datasets, features like 9, 7, 6, 5, and 2 display a wide range of PDP and ICE values, have varying impacts on the model’s predictions based on their specific instance values.

Several studies adopted these techniques to gain deeper insight into fault-detection mechanisms, as shown. Prakash, et al. [39] proposed a methodology for detecting internal leakage in hydraulic pumps using an unbalanced dataset of electrical power signals from the pump’s drive motor. The methodology involves extracting refined composite multiscale dispersion and fuzzy entropies, along with three statistical indicators and second-order polynomial-based features. PDP and ICE are used to visualize and normalize these features.

In Mehdiyev and Fettke [40], ICE and SHAP were utilized to help in predictive process monitoring. By integrating top-floor and shop-floor data and applying a Deep Learning (DL) model, process outcomes were predicted. The authors showed that generated explanations using ICE and SHAP allowed domain experts to examine diferent perspectives and understand the factors influencing predictions.

Danesh, et al. [41] exploited PDP and ICE to visualize and explain the impact of inputs on predictions from a neural network used to predict electrical power output in a combined cycle power plant. Kang, et al. [42] used a decision tree to identify and extract key failure-related features of a train protection system. The input features were system type, operation mileage, and service time while the output parameter is the cumulative failure rate. For interpretability, they used various XAI tools including PDP, ICE, Feature Importance, and SHAP.

## 3.5. Layer-wise relevance propagation (LRP)

Layer-wise Relevance Propagation (LRP) is used to explain neural networks. LRP operates by propagating the prediction $f ( x )$ backward in the neural network, by means of purposely designed local propagation rules. Let j and k be neurons at two consecutive layers of the neural network. Propagating relevance scores $\left( R _ { k } \right) _ { k }$ at a given layer onto neurons of the lower layer is achieved by applying the rule:

![](images/b89763757dec63b9a2346d919a62a458110ce833fd9cbdb40fab47a67966cbc2.jpg)

![](images/928f081ac08598330e8d2df97957e8bcaa0b240005397f99e5673dfcd533719c.jpg)

![](images/1c541c7a8b386ddb1cf4e4fd7e90d64f7c048d428c3e3d7e84286f3db1d54ccf.jpg)

![](images/fbd5938394b4f15512e740f255b95dc88df243316731443f76d26a8274b9c438.jpg)

![](images/999047e6643ca47b01517fedb6c700be8064be24cfe0aa84373bbb1d8e75b729.jpg)

![](images/1be80961d470dfa5cf378d86a687bf85b8c7dff58906a4fc891c753e43eb8b1e.jpg)

![](images/44639fad8f63e9374a20a4e115b015f70a8fc80e4d1614dff0d2b15e74770513.jpg)

![](images/f5fd80a6695278d89bfac06f2e5ba819a6b3e0e61361efa9b8766be3736940d8.jpg)

![](images/da9c15a7527eed31dbc1d9ec27597aa4fe0756f4165aa8ce878bc244cc64eca8.jpg)

![](images/03e6e9a6033ee70e9b1ead4245c2a66435f4fd68df1903af987f3f9a1783826f.jpg)  
Figure 7. PDP and ICE curves for the simulated dataset.

$$
R _ {j} = \sum_ {k} \frac {z _ {j k}}{\sum_ {j} z _ {j k}} R _ {j}.\tag{8}
$$

The quantity $z _ { j k }$ models the extent to which neuron j has contributed to make neuron k relevant. The propagation procedure terminates once the input features have been reached. LRP was widely used in diferent applications. For instance, Agarwal, et al. [43] introduced a DL-based statistical monitoring methodology for Fault Detection and Diagnosis (FDD). It was found that the relevance scores generated through LRP can be used in an iterative way to identify and remove redundant input variables. This reduced overfitting on noisy data. It also improved the separation between classes and led to higher FDD test accuracy. In another study conducted by Grezmak, et al. [44], LRP was applied to scrutinize the performance of a CNN used for machine fault diagnosis, specifically through the analysis of time-frequency spectra images derived from vibration signals. LRP’s role here was to provide a pixel-level relevance maps that pinpoint the specific signal components most responsible for driving the model’s diagnostic decisions.

In a manufacturing context, Lee, et al. [45] utilized LRP to explore the predictive capacity of a defect image classification model. This allowed for the visualization and highlighting of the relevant regions within defect images, ofering domain experts valuable insights into the model’s decision-making process. Similarly, Han, et al. [46] employed LRP to explore the impact of data on a trained DL model for mechanical motor fault diagnosis.

## 3.6. Class activation mapping (CAM)

Class Activation Mapping (CAM) provides visual explanations for CNNs by producing heatmaps that highlight the regions in an input image that influence the model’s prediction. CAM uses global average pooling to weigh the feature maps, making it possible to identify the most important areas for a given decision [46].

Chen and Lee [47] embarked on a study focused on fault detection using vibration signals, which are converted into images through Short-Time Fourier Transform (STFT). Employing a CNN for classification, they utilized the Gradient-weighted CAM (Grad-CAM) technique to elucidate classification decisions. Grad-CAM extends CAM by relying on gradients to guide the explanation. It calculates the gradient of the predicted class with respect to the feature maps in a chosen convolutional layer. The resulting weights are combined to generate a heatmap. This heatmap highlights the image regions that have the strongest efect on the mode prediction. In the study by Chen and Lee [47], the authors further compared their approach against neural networks, Adaptive Network-based Fuzzy Inference Systems (ANFIS), and decision trees to validate the robustness of their findings. Similarly, Kim and Kim [48] integrated Grad-CAM into a CNN-based bearing fault diagnosis technique. They used the Normalized Bearing Characteristic Component (NBCC) as the CNN input, representing bearing failure symptoms efectively. Yoo and Jeong [49] also presented a vibration analysis process for bearing fault diagnosis using a fine-tuned VGG-19 model and Grad-CAM. The vibration data were collected from a motor using an accelerometer and Internet of Things (IoT) module. Spectrogram images were then generated for normal and diferent fault situations. The visualization model they proposed was compared to conventional defect frequency analysis methods.

Alike, Brito, et al. [50] applied Grad-CAM with a 1D CNN to enhance explainability in rotating-machinery fault diagnosis. Yu, et al. [51] used Grad-CAM and Eigenvector-based Class Activation Map (Eigen-CAM) to interpret ResNet06 in various databases, including bearing and gearbox datasets.

Kim, et al. [52] introduced an explainable model for fault diagnosis in linear motion guides using timedomain signals. The Frequency-domain-based Grad-CAM (FG-CAM) method was utilized to visualize the model’s classification criteria learned in the time domain. The authors highlighted the challenge of applying the model to multiaxis data, emphasizing the need to extract features from each axis individually. This led to the development of a grouped convolution approach, enabling the extraction of feature maps from each X-, Y-, and Z-axis and facilitating the visualization of the importance of different freguencies on model decisions Furthermore, multivariable data-based FG-CAM (mFG-CAM) was proposed to provide clearer visualization of the model’s decision criteria when dealing with multiaxis vibration signals.

Li, et al. [53] introduced Multilayer Grad-CAM (MLG-CAM) to tackle the issue of decreasing feature resolution in Grad-CAM. The MLG-CAM leverages gradients across multiple convolutional layers to obtain activation maps with varying resolutions, which are then combined through layer-weighted summation to generate a comprehensive activation map. Experiments conducted with MLG-CAM demonstrated its ability to emphasize cyclo-stationary impulses in the time domain and fault characteristic frequencies in the frequency domain.

Oh and Jeong [54] proposed a framework incorporating CNN for fault detection and CAM for fault diagnosis. To enhance the reliability of the diagnostic results, process monitoring was performed using Variational Autoencoders (VAE), which learned the CAM produced in the fault detection and diagnosis process, treating CAMs generated with misclassified label information as anomalies.

Lee, et al. [55] introduced a novel data imagification approach called Fuzzy-based Energy Pattern Image (FEPI) generation, which transformed sensor signals into FEPI data. A CNN-based fault diagnostic model was trained using FEPI data, and Grad-CAM was used to interpret the model’s predictions and identify critica regions for fault classification. This approach was presented in a case study of robotic spot welding, yielding promising results.

Yang, et al. [56] proposed a diferent fault detection and diagnosis method in rotating machinery. The main structure of the proposed method is based on the standard CNNs, but they added a penalty term to the loss function to penalize the model if it learned some insignificant fault features in the training process. They further utilized smoothed score-CAM, an upgraded version of CAM, to improve localization accuracy and heatmap visual quality. The smoothing process involved applying a Gaussian filter to the activation map to reduce over-activation and noise, resulting in more refined and reliable heatmaps.

Gwak, et al. [57] used a frequency-domain-based model for fault classification using vibration signals where they also used FG-CAM for interpretability. They also proposed a Power-Perturbation-Based Decision Boundary Analysis (POBA) framework to analyze changes in decision boundaries by modifying power spectral densities in input vibration signals.

## 3.7. Case based reasoning (CBR)

Case Based Reasoning (CBR) is method that explains a model’s output by retrieving, reusing, and revising similar past cases. CBR deals with very specific data from previous situations, and reuses results and experience to fit a new problem situation. A schematic of the process is shown in Figure 8.

CBR itself is less about complex mathematical formulas and more about a methodology or strategy for problem-solving that solves new problems by identifying and adapting solutions from the most similar previously encountered cases. However, the crucial part that can be mathematically defined is the similarity measure, which is fundamental for retrieving the most relevant cases. The general steps of CBR can be summarized as follows:

(1) Represent the case: Represent each case as a vector in feature space. If x denotes a new problem and $c _ { j }$ represents a stored case, both can be expressed as:

$$
x = (x _ {1}, x _ {2}, \ldots , x _ {n}), c _ {i} = (c _ {i 1}, c _ {i 2}, \ldots , c _ {i n})\tag{9}
$$

(2) Compute the similarity: Define a similarity measure $S i m ( x , c _ { i } )$ to quantify how close a stored case $c _ { j }$ is to the new problem x. A common choice is to use the inverse of the Euclidean distance or Cosine similarity giver respectively as

$$
\operatorname{Sim} \left(x, c _ {i}\right) = \frac {x \cdot c _ {i}}{\left| x \right| \left| c _ {i} \right|} = \frac {\sum_ {j = 1} ^ {n} x _ {j} c _ {i j}}{\sqrt {\sum_ {j = 1} ^ {n} x _ {j} ^ {2}} \sqrt {\sum_ {j = 1} ^ {n} c _ {i j} ^ {2}}}\tag{10}
$$

![](images/7ca2d564e0e8cc980698e80a1e827452ad578ab9bd9437707d6b9e8f0ebf3a9d.jpg)  
Figure 8. Schematic of case-based reasoning.

$$
\operatorname{Sim} \left(x, c _ {i}\right) = \frac {1}{\sqrt {\sum_ {j = 1} ^ {n} \left(x _ {j} - c _ {i j}\right) ^ {2}}},\tag{11}
$$

(3) Retrieve the most similar case: Retrieve the case $c ^ { * }$ from the case base C that maximizes the similarity measure:

$$
c ^ {*} = \arg \max _ {c _ {i} \in C} \operatorname{Sim} (x, c _ {i})\tag{12}
$$

(4) Reuse knowledge: Apply the solution associated with the retrieved case $c ^ { * }$ to the new problem x.

(5) Revise and retain: Evaluate the proposed solution. If necessary, adjust it based on new insights and store the updated case for future use.

For the simulated dataset, CBR would mean that when a new fault is detected, the system analyzes the current sensor readings, compares them to historical data, identifies the most similar previous incidents, and suggests a proven solution based on past outcomes.

Many authors have employed CBR. For example, Khosravani, et al. [58] used it in fault detection in injection molding of drippers. Zhao, et al. [59] utilized the case-based reasoning method for fault detection and diagnosis of the Tennessee Eastman process. The authors also proposed a case maintenance strategy to avoid redundant and noisy cases that are added to the case base. Similarly Boral, et al. [60] used CBR in a framework for fault detection and diagnosis and for suggested maintenance actions. An interesting study by Chen, et al. [61] proposed a CBR system based on 143 cases compiled from accurately diagnosed and successfully resolved aero-engine fault events.

## 3.8. Alternative and supplementary XAI techniques for fault detection and diagnosis

The earlier sections covered the widely used XAI methods in fault detection and diagnosis. Alongside these mainstream tools, there are alternative and supplementary approaches that support the same task. These methods are not part of the standard XAI family, and they do not follow a unified structure. They are often built for a specific case, and they rely on ideas from statistics, signal processing, or expert systems.

These approaches are useful because they add context and link older practices to current XAI work. Researchers use them to improve fault detection models, check the results of mainstream XAI tools, or handle situations where a more flexible or case-driven method is needed. This section introduces these alternative and supplementary approaches and outlines their use, strengths, and limits in fault detection and diagnosis.

Utama, et al. [62] used a model-agnostic method called Anchors to build an explainable fault detection model for photovoltaic panels implementable on edge nodes with a real application on photovoltaic panels. The purpose of Anchors is to find a decision rule that approximates the decision function of the model around that individual data point. These anchors are more human-interpretable explanations that can be mapped back to data features. Anchors can thus highlight the specific sensor-data patterns that the mode associates with fault conditions, providing a clearer rationale for its predictions.

Further advancing the field, Kim, et al. [63] proposed a visual XAI method that aims to provide explainability in fault diagnosis using a 1D vibration signal. It introduced a Frequency Activation Map (FAM) to visualize the classification criteria of a 1D CNN model. The methodology involves designing a CNN model with a norm constraint on the filters to ensure consistent filtering of frequency information. The model learns from vibration signals to classify normal and faulty states of equipment. The FAM is generated to visualize the specific frequencies that the model focuses on for classification. Jiang, et al. [64] proposed an interpretable DL model named Multi-Wavelet Kernel CNN (MWKCNN) for fault diagnosis. Features are extracted using the Multi-Wavelet Kernel Convolution (MWKC) layer which is constructed based on continuous wavelet transform (CWT) to detect the impulse signatures of faults. Then, a Kernel Wights Recalibration (KWR) module was used to assign diferent weights to diferent wavelet kernels dynamically. The interpretability of the proposed method was achieved by analyzing the distinctive behaviors of individual wavelet kernels, tracking how their weights evolved during training, and visualizing the corresponding learned feature patterns. The performance of the proposed method was validated using two gearbox datasets.

In the fault detection using monitoring, Lu and Yan [65] proposed a Deep Fisher Auto Encoder based Self-Organizing Map (DFAE-SOM) model for process monitoring. The method combines Fisher Discriminant Analysis (FDA) and auto encoders. To make the method interpretable, it was combined with a selforganizing map that project the high-dimensional data into a 2D space, where normal and fault classes are visualized in separate regions.

In the context of explainable rule based decision-making, Dorgo, et al. [66] proposed a decision treebased classifier for fault classification. The model utilized a sliding window-based data preprocessing approach and was designed to be able to detect and isolate faulty states. The proposed method in the paper used alarm thresholds to trigger alarms when the process variables exceed certain limits. The decision tree classifier was then employed to analyze the situation further and provide a set of rules and conditions that can be followed to interpret the alarm and make decisions. Similarly, Obregon, et al. [67] proposed a rule-based explanations (RBE) framework in combination with ML interpretation methods to understand the decision mechanisms of accurate and complex predictive models, specifically tree ensemble models, in the context of plastic injection molding quality control. The framework generated simple decision rules along with partial dependence plots and feature importance rankings to provide meaningful explanations and enhance the understanding of the main factors influencing manufacturing quality. The applicability of the RBE framework was demonstrated through two experiments using real industrial data from a plastic injection molding machine, showcasing its potential for improving production eficiency in this domain.

Harinarayan and Shalinie [68] took a diferent tack by leveraging SHAP for both local and global explanations in an XGBoost model. Then they generated diverse counterfactual explanations to be used as action recommendations to correct the fault scenario. Counterfactual explanations are a type of XAI technique. They involve generating alternative scenarios that could have led to a diferent outcome, given the same input data. It works by optimizing a distance function that provides the minimum change to the input instance to get a diferent target output. The decision maker can then choose the optimal change required in terms of cost.

In the field of probabilistic models, Maged and Xie [69] presented a systematic approach for utilizing the prediction uncertainty information generated by Bayesian Neural Networks (BNN) models along with the prediction values obtained from the output layer of the network in order to make optimal decisions. BNNs can provide a degree of interpretability by quantifying predictive uncertainty through their probabilistic framework. However, they are not inherently interpretable models like decision trees or rule-based systems. Their interpretability is limited compared to more explicitly interpretable models in the field of XAI. The proposed approach is applied to a real case study on vertical continuous plating of printed circuit boards.

Lastly, Conde, et al. [70] incorporated interpretability constraints into a boosting algorithm to produce accurate and easily interpretable classification rules. The authors introduced two methods for binary classification. These methods, Simple Isotonic LogitBoost (SILB) and Multiple Isotonic LogitBoost (MILB), aim to create classification rules that align with known monotonic relationships within the data. SILB operates by selecting the best-fitting variable in each boosting step while carefully considering isotonicity, ensuring that the relationships between selected variables and classification outcomes follow a consistent trend. MILB, on the other hand, takes a diferent approach, refitting the entire problem in each boosting step. This means that all predictors change their roles in the classification rule during the process while stil respecting isotonicity constraints. Their isotonic boosting approach was evaluated using simulations and real-world induction motor failure data.

## 3.9. Choosing the right XAI technique

As discussed, finding the optimal balance between model accuracy and explainability is not straightforward as increasing model complexity typically reduces explainability. The following discussion provides guidance to users seeking to incorporate XAI techniques into their analysis. Table 1 ofers an overview of genera questions and engineering-specific questions related to machine behavior and potential methodologies for interpretation. Each methodology is associated with suitable implementation processes, including relevant libraries in both R and Python

To illustrate the practical application of these concepts, consider a scenario on a production floor where “machine id 1” begins displaying erratic behavior. This sudden change raises concerns about the potential failure of a specific component. To diagnose the issue and facilitate informed decision-making, stakeholders resort to XAI techniques.

Table 1. Guide of the selection of the suitable XAI technique.

<table><tr><td>General Question</td><td>Engineering Question</td><td>Methodology to be Used</td><td>Implementation Process</td></tr><tr><td>What is the Influence of features on model output?</td><td>How do various features influence the model&#x27;s output?</td><td>SHAP</td><td>SHAP (Python library), SHAPR (R Package), IML (R Package), shapviz (R Package), dalex (Python and R Package)</td></tr><tr><td>How does the system perceive the impact of varying the value of a specific feature on the model output?</td><td>How does the system view the changes in feature values?</td><td>PDP/ICE</td><td>PDP (R Package), PDPbox (Python library), ICEbox (R Package)</td></tr><tr><td>How does the model draw upon past cases to justify the prediction?</td><td>Why does the system predict the failure of “machine id 1” based on past cases?</td><td>LIME</td><td>LIME (Python library, R Package), lime (Python Package), iml (R Package)</td></tr><tr><td>How does the system determine and articulate the most influential and reliable features guiding a specific prediction or decision?</td><td>What is the vibration feature threshold to ascertain correct machine functionality, and how accurate is the system in determining this threshold?</td><td>Anchors</td><td>Anchors (Python library), alibi (Python library)</td></tr><tr><td>What guiding rule or precedent does the system follow in making decisions?</td><td>What rule does the system adhere to when making decisions about the machinery?</td><td>Feature importance (random forest or XGBoost)</td><td>Scikit-learn (Python library), Xgboost (R Package), Lightgbm (R Package), ranger (R Package), h2o (Python and R Package)</td></tr><tr><td>How does the model output change with variations in a specific feature, considering past cases and their outcomes?</td><td>How does the model output fluctuate with variations in a specific feature considering historical cases?</td><td>Case-Based Reasoning</td><td>CBR (R Package), cbr (Python library)</td></tr><tr><td>What input features are most relevant to a prediction?</td><td>How does the system visualize and identify the most relevant component in “machine id 1” for a given prediction?</td><td>Class Activation Mapping</td><td>CAM (Python library), tf-keras-vis (Python library), keras-vis (Python library)</td></tr></table>

In addition to the specific XAI techniques outlined above, it is essential to recognize the users’ flexibility in employing alternative methods for model interpretation. Exploratory data analysis (EDA), Bayesian methods, or ad-hoc approaches can serve as valuable supplements to the suggested XAI techniques. These traditiona approaches enable users to utilize statistical and probabilistic reasoning, exploring relationships within the data and providing additional context to model predictions. The selection of a particular method ultimately depends on the nuances of the analysis and the desired level of interpretability goals.

## 4. Summary and discussion

Figure 9 provides an insightful chronological analysis of the research trends from 2016 to 2024, focusing on the application of XAI techniques with Machine Learning (ML) and Deep Learning (DL) techniques in fault detection and diagnosis. In the early phase (2016–2019), the use of ML and DL techniques remained balanced but modest. For instance, in 2016 and 2017, there was an equal distribution between ML and DL, with one study each for both methods [19,59]. The vears 2018 and 2019 show a clear dominance of ML methods. In 2018, only one study was conducted using ML with no notable DL research, while in 2019, two ML studies were recorded, marking the first instance where ML surpassed DL usage in fault detection and diagnosis studies. A significant shift in research trends is observed starting from 2020, with a sharp rise in DL methods. This trend continued to grow in subsequent years, with 2021 recording a substantial increase in both methods. However, DL outpaced ML in the following years, demonstrating its growing dominance. Interestingly, the year 2024 shows a significant drop in both ML and DL studies, with only one study recorded for ML and none for DL. This decline could signal a temporary shift in research priorities or reflect a maturing field in which scholarly eforts have begun to consolidate around a smaller set of well-established approaches. This chronological breakdown of ML and DL research in fault detection highlights the evolving landscape of AI applications in industrial settings, where DL is increasingly favored for its robust performance.

Figure 10 displays the percentage of XAI techniques used in the reviewed studies to understand the distribution of each technique across the body of work. The most frequently employed methods include SHAP (19%) and feature importance (17%) techniques. SHAP popularity stems from its robustness in providing both local and global interpretability, while feature importance is a straightforward yet efective approach. Class Activation Mapping (CAM) is another frequently used XAI tool. CAM highlights specific areas in an image that contribute to the model decision-making, ofering visual explanations of complex models. Less commonly employed techniques include LIME and PDP-ICE, which appear in 7 and 2 studies, respectively. Its relative infrequency may reflect limitations in scalability or a growing preference for techniques, such as SHAP, that deliver more stable and comprehensive interpretability. Case Based Reasoning (CBR) and Layer-wise Relevance Propagation (LRP) are utilized in 4 studies each. CBR, rooted in analogical reasoning, ofers explanations based on the closest matching cases from a database, making it well-suited for applications requiring human-friendly explanations [61]. LRP, meanwhile, is used to decompose neural network outputs by propagating relevance from the output layer back to the input features, ofering detailed insights into decision-making processes in neural networks.

![](images/0b467f99351f871dbd1a95d9113195b95c0a9c0ed5d14224483ade7021869418.jpg)  
Figure 9. Reviewed papers classified by the use of ML or DL methods, organized by publication year.

![](images/f8584c964d713aac9e44e36374ecee98d6797a0272381a5aec2a8838388d2026.jpg)  
Figure 10. XAI techniques involved in the reviewed studies.

Figure 11 provides a critical insight into the XAI techniques used in conjunction with ML and DL models for fault detection and diagnosis. By analyzing the distribution of XAI methods, one can infer how diferent XAI techniques are applied based on the nature of the underlying model. Feature importance techniques are widely used in both ML and DL. For ML, feature importance is more straightforward due to the structured nature of many traditional algorithms, such as decision trees and random forests, which naturally produce importance scores. In DL, feature importance is valuable for understanding complex model architectures by focusing on which features contribute most to the network output [18]. CAM is the most commonly used XA technique, with 12 studies applying it. It gained traction in DL because its heatmap-based visualizations ofer an intuitive way to explain complex CNNs. LIME was employed in ML and DL with slightly more DL models. For ML, LIME provides explanations but tends to be used less than feature importance or SHAP due to its complexity and computational cost, especially for large datasets. On the other hand, DL models can benefit more from local surrogate models provided by LIME. PDP-ICE techniques are used in 2 studies, exclusively in ML. They are mainly used with ML because of the relative simplicity and structured nature of the models, making it easier to create intuitive visualizations of how variables afect predictions. DL models, with their high dimensionality and non-linearity, make PDP-ICE less suitable. LRP is specifically designed to explain predictions from complex neural networks by backpropagating relevance scores, hence its exclusiveness in DL model.

Figure 12 displays the applications included in the reviewed papers, while Figure 13 shows XAI techniques integrated into the studies categorized by the application area. In the energy systems and renewable technologies domain, feature importance emerges as the most frequently used XAI technique. Feature importance is highly efective due to the complexity of energy generation processes, such as in wind turbines, photovoltaic panels, and electricity load forecasting [1,17,21]. These systems depend on multiple environmental and operational tabular variables, making it critical to understand which features, such as wind speed, solar irradiance, or system voltage most influence the predictions. The interpretability provided by XAI techniques supports decision-makers in optimizing energy output and enhancing the reliability of energy infrastructure. For rotating machinery and mechanical systems, CAM is the predominant explainability method. CAM is particularly useful in the analysis of rotating systems such as gearboxes, bearings,

![](images/446b609a44a4a64d2a97dd7e057490df56f30e88d331d41758e61ebb6bd5d7ab.jpg)  
Figure 11. Reviewed papers classified by the use of ML or DL methods, categorized by XAI technique.

![](images/89b2bd2683b8ca8a4b9f6e01b5226db051f1ec2e28c03e5a1f889777d548c9ed.jpg)  
Figure 12. Applications involved in the reviewed studies.

![](images/4a9bfc895cd819bf158f540c446d45e4d1f7552cb833d55f402e1d3b7e243fc5.jpg)  
Figure 13. XAI techniques integrated into the studies categorized by application area.

motors, and hydraulic pumps because it can visually identify which components of image or 1-dimensional - based data input contribute most to the model prediction [49,53,57].

Chemical processes often involve complex interactions between multiple variables, such as pressure, temperature, and chemical composition. Feature importance is especially valuable for identifying which of these parameters are the most critical in determining system behavior [33]. Understanding which features contribute most to the variability in product yield enables users to optimize process conditions, enhance eficiency, and minimize waste.

SHAP and LRP are the primary XAI techniques employed in industrial automation and control systems applications. SHAP is particularly beneficial for understanding how input features like machine parameters or control signals contribute to the outcome of predictive models. This is especially relevant in industrial contro systems where decisions on fault detection and system optimization need to be interpretable for operators [29]. While LRP helps pinpoint specific pixels or regions in image data that indicate defects, making it an essential tool for automated quality control [45]. LIME can be very efective for providing localized explanations in systems that monitor the health of structures, such as prismatic cantilever steel beams or other civi engineering structures [27]. LIME helps engineers understand which specific factors are influencing the risk of failure or structural degradation. By ofering interpretable and localized insights, LIME enables more precise maintenance planning and preventive interventions, ensuring the longevity and safety of critica infrastructure.

## 5. Conclusion and expanding horizons in XAI

This study set out to assess how XAI supports FDD in industrial settings and to provide guidance on method choice and use. Its main contributions include a synthesis of more than 60 studies published between 2013 and 2024, the development of a taxonomy for XAI in FDD, a trend analysis across domains such as energy and manufacturing, and practical guidelines for implementing suitable XAI methods. Numerous studies have shown that methods such as SHAP, LIME, CAM, and related tools can play an essential role in improving transparency, trust, and model interpretability, especially in settings where reliability and safety are critical. These XAI methods do not only enhance user understanding but also help bridge the gap between blackbox machine learning models and human decision-makers on the factory floor. This review highlights the strengths and limitations of existing approaches, identifies practical applications across diferent sectors, and emphasizes the need for customized solutions that address specific industrial requirements.

A key challenge in applying XAI to FDD lies in the level of explanation produced by diferent methods. The depth and form of explanation that a model provides often depend on the algorithm, and the appropriate level of detail varies across users and applications. What may be meaningful for a maintenance technician could be too complex for a plant manager, and vice versa. Explanation design, therefore, must account for user expertise and context. A promising direction is the use of natural language generation to create adaptive explanations that adjust their detail, terminology, and tone to match the user’s background, improving both clarity and practical value.

Another major gap that we have revealed through this review is the absence of standardized methods to evaluate the quality of explanations. Many existing studies evaluate explanations subjectively, without standardized or quantitative measures of quality. Developing cross-domain benchmarks and consistent metrics for explanation fidelity, stability, and human usefulness remains a priority. As noted by Doshi-Velez and Kim [71], explanation quality includes diverse elements such as fidelity to the model’s interna logic, sparsity, human usability, and decision-making support. Without standardized benchmarks and evaluation protocols, it is challenging to determine whether a given explanation is genuinely helpful or simply appears so. Addressing this issue requires the development of domain-specific metrics, shared datasets, and controlled evaluation environments that allow rigorous comparison of explanation methods under consistent conditions.

Looking forward, several future directions stand out as critical for moving XAI-FDD systems from research prototypes to robust, deployable solutions. One of the most pressing priorities is real-time deployment of XAI-enabled models on edge devices with limited computational resources. Many industrial applications require fast, localized decision-making under latency and resource constraints. However, to scale such systems broadly, further research is needed to optimize model architectures and explanation methods to work within tight time budgets and under restricted communication bandwidth. On-device processing must also account for intermittent connectivity, ensuring that XAI feedback remains available and secure even in unstable network environments. This requires lightweight, eficient, and secure solutions that preserve both model performance and interpretability in real-world settings. Moreover, addressing issues related to limited and heterogeneous data is essential, as highlighted by recent work from [72].

Another future direction involves developing unified metrics and benchmarking frameworks for evaluating explanation quality in FDD tasks. This will require the development of standardized datasets, protocols, and performance indicators that account for fidelity, robustness, comprehensibility, and human-centered usability. Such standards would enable fair cross-comparison of XAI methods and accelerate the transition from academic development to industry adoption. As in other fields of machine learning, challenge problems and benchmarking platforms have proven efective in driving progress. The same is needed for XAI in fault diagnosis.

The current landscape of XAI for fault diagnosis also shows that many models are narrowly focused on specific machines or processes. To maximize their utility, future work should aim for broader generalization. This includes building models that can transfer across machines, adapt to new fault scenarios, and perform reliably under diferent operational conditions. Methods such as transfer learning, domain adaptation, and meta-learning ofer promising pathways for enabling models to generalize across machines, fault types, and operating environments.

A further area for development is the integration of XAI with uncertainty quantification. It is not enough to explain a model’s prediction; users must also know how reliable that prediction is. Presenting confidence intervals alongside explanations allows users to make better-informed decisions and may help prevent overreliance on model outputs. Research in this area is still emerging, but methods such as Bayesian neura networks and deep ensembles show promise in producing prediction intervals that can be aligned with feature-based explanations. Future work should focus on building integrated pipelines that deliver both explanations and uncertainty estimates, ideally in ways that are interpretable themselves. Clear benchmarks for evaluating uncertainty in explanations are also needed to ensure that such outputs are trustworthy and actionable.

Finally, current research on XAI-FDD has largely focused on algorithmic development, with limited attention to the human role in the decision loop. Few studies examine how operators perceive, interpret, or act on model explanations, leaving uncertainty about which explanation formats best support decisionmaking in high-stakes settings. Future research should prioritize human-centered studies in real industria environments to evaluate explanation efectiveness through measurable outcomes such as fault detection speed, error reduction, and user accuracy. Incorporating operator feedback through active-learning frame works could enhance both system reliability and user engagement.

## Disclosure statement

No potential conflict of interest was reported by the author(s).

## Notes on contributors

Ahmed Maged received the Ph.D. degree in systems engineering from the City University of Hong Kong, Hong Kong, in 2023. His current research interests include quality engineering, anomaly detection, and machine learning.

Salah Haridy received his Ph.D. from Nanyang Technological University, Singapore in 2014. His research interests cover quality engineering, Six Sigma, SPC and design of experiments.

Mohamed Hosny received his Ph.D. (2022) in Biomedical Engineering from School of Life Science and Technology, Harbin Institute of Technology, China. His research interests include Computer Vision, Machine Learning and Image and Signal Processing.

Herman Shen received his M.S. and Ph.D. degrees from the Department of Aerospace Engineering, the University of Michigan (Michigan) in 1986 and 1989, respectively. He has pioneered developments in structural health management framework and fatigue life prediction schemes for gas turbine engines, airframes, additive manufacturing parts, power generation assets, wind turbines, ofshore platforms, pipelines, and auto bodies.

## ORCID

Ahmed Maged http://orcid.org/0000-0002-5071-5253 Salah Haridy http://orcid.org/0000-0002-8406-4647 Mohamed Hosny http://orcid.org/0009-0007-4039-4525

## Data availability statement

Data sharing is not applicable to this article as no new data were created or analyzed in this study.

## References

[1] Chiu MC, Lee YH, Chen TM. Integrating content-based image retrieval and deep learning to improve wafer bin map defect patterns classification. J Ind Production Eng. 2022;39(8):614–628. doi: 10.1080/21681015.2022.2074155

[2] Van Lent M, Fisher W, Mancuso M. An explainable artificial intelligence system for small-unit tactical behavior. In: Proceedings of the national conference on artificial intelligence, San Jose California. Citeseer; 2004. p. 900–907.

[3] Adadi A, Berrada M. Peeking inside the black-box: a survey on explainable artificial intelligence XAI. IEEE Access. 2018;6:52138–52160.

[4] Ali S, Abuhmed T, El-Sappagh S et al. Explainable artificial intelligence (XAI): what we know and what is left to attain trustworthy artificial intelligence. Inf Fusion. [2023 11 1];99:101805.

[5] Minh D, Wang HX, Li YF, et al. Explainable artificial intelligence: a comprehensive review. Artif Intell Rev. 2022 06 01;55(5):3503–3568. doi: 10.1007/s10462-021-10088-y

[6] Maged A, Zwetsloot I. Anomaly detection via real-time monitoring of high-dimensional event data. IEEE transactions on industrial informatics. 2023;20(2): 2856–64.

[7] Abid A, Khan MT, Iqbal J. A review on fault detection and diagnosis techniques: basics and beyond. Artif Intell Rev. 2021 06 01;54(5):3639–3664. doi: 10.1007/s10462-020-09934-2

[8] Zhang Y, Fan Y, Du W, et al. Nonlinear process monitoring using regression and reconstruction method. IEEE Trans Automat Sci Eng. 2016;13(3):1343–1354. doi: 10.1109/TASE.2016.2564442

[9] Park Y-J, Fan S-KS, Hsu C-Y. A review on fault detection and process diagnostics in industrial processes. Processes. 2020;8(9):1123. doi: 10.3390/pr8091123

[10] Iyer SV, Sangwan KS, Dhiraj, et al. A cognitive digital twin for process chain anomaly detection and bottleneck analysis. J Ind Production Eng. 2025;42(1):65–87. doi: 10.1080/21681015.2024.2381728

[11] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst. 2017;30:30.

[12] Arrieta AB, Díaz-Rodríguez N, Del Ser J, et al. Explainable artificial intelligence (XAI): concepts, taxonomies, opportunities and challenges toward responsible AI. Inf Fusion. 2020;58:82–115.

[13] Aldrich C, Auret L. Fault detection and diagnosis with random forest feature extraction and variable importance methods. Ifac Proc Volumes. 2010 01 01;43(9):79–86. doi: 10.3182/20100802-3-ZA-2014.00020

[14] Breiman L. Random forests. Mach Learn. 2001;45(1):5–32. doi: 10.1023/A:1010933404324

[15] Chakraborty D, Elzarka H. Early detection of faults in HVAC systems using an XGBoost model with a dynamic threshold. Energy Build. [2019 02 15];185:326–344. doi: 10.1016/j.enbuild.2018.12.032

[16] Zhang C, Wang D, Wang L, et al. Cause-aware failure detection using an interpretable XGBoost for optical networks. Opt Express. 27 September 2021; 29(20):31974–31992. doi: 10.1364/OE.436293

[17] Zheng H, Yuan J, Chen L. Short-term load forecasting using EMD-LSTM neural networks with a XGBoost algorithm for feature importance evaluation. Energies. 2017;10(8):1168. doi: 10.3390/en10081168

[18] Glaeser A, Selvaraj V, Lee S, et al. Applications of deep learning for fault detection in industrial cold forging. Int J Production Res. 2021;59(16):4826–4835. doi: 10.1080/00207543.2021.1891318

[19] Janssens O, Slavkovikj V, Vervisch B, et al. Convolutional neural network based fault detection for rotating machinery. J Sound Vibr. [1 September 2016];377:331–345. doi: 10.1016/j.jsv.2016.05.027

[20] Le V, Yao X, Miller C, et al. Series DC arc fault detection based on ensemble machine learning. IEEE Trans Power Electron. 2020;35(8):7826–7839. doi: 10.1109/TPEL.2020.2969561

[21] Zhang D, Qian L, Mao B, et al. A data-driven design for fault detection of wind turbines using random forests and XGBoost. IEEE Access. 2018;6:21020–21031. doi: 10.1109/ACCESS.2018.2818678

[22] Ribeiro MT, Singh S, Guestrin C. ”Why should I trust you?” Explaining the predictions of any classifier,”. In: Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, San Francisco, California, USA. 2016. p. 1135–1144.

[23] Srinivasan S, Arjunan P, Jin B, et al. Explainable AI for chiller fault-detection systems: gaining human trust. Computer. 2021;54(10):60–68. doi: 10.1109/MC.2021.3071551

[24] Lu F, Tong Q, Feng Z, et al. Explainable 1DCNN with demodulated frequency features method for fault diagnosis of rolling bearing under time-varying speed conditions. Meas Sci Technol. 2022;33(9):095022. doi: 10.1088/1361- 6501/ac78c5

[25] Sairam S, Seshadhri S, Marafioti G, et al. Edge-based explainable fault detection systems for photovoltaic panels on edge nodes. Renewable Energy. [1 February 2022];185:1425–1440. doi: 10.1016/j.renene.2021.10.063

[26] Sundarrajan K, Rajendran BK. Explainable eficient and optimized feature fusion network for surface defect detection, Int J Ady Manuf Technol. 2023 06 21, doi: 10.1007/s00170-023-11789-0

[27] Zhang H, Lin J, Hua J, et al. Attention-based interpretable prototypical network towards small-sample damage identification using ultrasonic guided waves. Mech Syst Signal Process. [1 April 2023];188:109990. doi: 10.1016/j ymssp.2022.109990

[28] Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst. 2017;30.

[29] Hwang C, Lee T. E-SFD: explainable sensor fault detection in the ICS anomaly detection system. IEEE Access 2021;9:140470–140486. doi: 10.1109/ACCESS.2021.3119573

[30] Jang K, Pilario KES, Lee N et al. Explainable artificial intelligence for fault diagnosis of industrial processes IEEE Transactions on Industrial Informatics. 2023;21(1):4–11.

[31] Chowdhury D, Sinha A, Das D. XAI-3DP: diagnosis and understanding faults of 3-D printer with explainable ensemble Al, IEEE Sens Lett, 2023:7(1):1–4, doi: 10.1109/L SENS.2022.3228327

[32] Brusa E, Cibrario L, Delprete C, et al. Explainable AI for machine fault diagnosis: understanding features’ contribution in machine learning models for industrial condition monitoring. Appl Sci. 2023;13(4):2038. doi: 10.3390 app13042038

[33] Choi H, Kim D, Kim J, et al. Explainable anomaly detection framework for predictive maintenance in manufacturing systems. Appl Soft Comput. 2022 08 01;125:109147. doi: 10.1016/j.asoc.2022.109147

[34] Baptista ML, Goebel K, Henriques EMP. Relation between prognostics predictor evaluation metrics and loca interpretability SHAP values. Artif Intel. 2022 05 01;306:103667. doi: 10.1016/j.artint.2022.103667

[35] Zhang D, Li C, Shahidehpour M, et al. A bi-level machine learning method for fault diagnosis of oil-immersed transformers with feature explainability. Int J Electr Power Energy Syst. 2022 01 01;134:107356. doi: 10.1016/j. ijepes.2021.107356

[36] Kumar P, Hati AS. Deep convolutional neural network based on adaptive gradient optimizer for fault detection in SCIM. ISA Trans. 01 May 2021; 111:350–359. doi: 10.1016/j.isatra.2020.10.052

[37] Onchis DM, Gillich G-R. Stable and explainable deep learning damage prediction for prismatic cantilever steel beam. Comput Ind. 2021 02 01;125:103359. doi: 10.1016/j.compind.2020.103359

[38] Brito LC, Susto GA, Brito JN, et al. An explainable artificial intelligence approach for unsupervised fault detection and diagnosis in rotating machinery. Mech Syst Signal Process. 2022 01 15;163:108105. doi: 10.1016/j.ymssp.2021. 108105

[39] Prakash J, Miglani A, Kankar PK. Internal leakage detection in hydraulic pump using model-agnostic feature ranking and ensemble classifiers. J Comput Inf Sci Eng. 2023;23(4). doi: 10.1115/1.4056365

[40] Mehdiyev N, Fettke P. Local post-hoc explanations for predictive process monitoring in manufacturing,” arXiv preprint arXiv:.10513, 2020.

[41] Danesh T, Ouaret R, Floquet P, et al. Hybridization of model-specific and model-agnostic methods for interpretability of neural network predictions: application to a power plant. Comput Chem Eng. [2023 08 1];176:108306. doi: 10.1016/j.compchemeng.2023.108306

[42] Kang R, Wang J, Chen J, et al. Analysis of failure features of high-speed automatic train protection system. IEEE Access 2021:9:128734-128746 doi: 10.1109/ACCESS 2021.3113381

[43] Agarwal P. Tamer M. Budman H. Explainability: relevance based dynamic deep learning algorithm for fault detection and diagnosis in chemical processes. Comput Chem Eng. [2021 11 1];154:107467. doi: 10.1016/j. compchemeng.2021.107467

[44] Grezmak J, Zhang J, Wang P, et al. Interpretable convolutional neural network through layer-wise relevance propagation for machine fault diagnosis. IEEE Sensors J. 2020;20(6):3172–3181. doi: 10.1109/JSEN.2019.2958787

[45] Lee M, Jeon J, Lee H. Explainable AI for domain experts: a post hoc analysis of deep learning for defect classification of TFT–LCD panels. J Intell Manuf. 2022 08 01;33(6):1747–1759. doi: 10.1007/s10845-021-01758-3

[46] Yang J, Zhao Y, Chen X, et al. Explainable deep learning method for power system stability evaluation with incomplete voltage data based on transfer learning. Measurement. 2025;247:116781.

[47] Chen H-Y, Lee C-H. Vibration signals analysis by explainable artificial intelligence (XAI) approach: application on bearing faults diagnosis. IEEE Access. 2020;8:134246–134256.

[48] Kim J, Kim J-M. Bearing fault diagnosis using Grad-CAM and acoustic emission signals. Appl Sci. 2020;10(6):2050. doi: 10.3390/app10062050

[49] Yoo Y, Jeong S. Vibration analysis process based on spectrogram using gradient class activation map with selection process of CNN model and feature layer. Displays. [2022 07 1];73:102233. doi: 10.1016/j.displa.2022. 102233

[50] Brito LC, Susto GA, Brito JN, et al. Fault diagnosis using explainable AI: a transfer learning-based approach for rotating machinery exploiting augmented synthetic data. Expert Syst Appl. 2023 12 01;232:120860. doi: 10.1016/j eswa.2023.120860

[51] Yu S, Wang M, Pang S, et al. Intelligent fault diagnosis and visual interpretability of rotating machinery based on residual neural network. Measurement. 2022 06 15;196:111228. doi: 10.1016/j.measurement.2022.111228

[52] Kim MS, Yun JP, Park P. An explainable convolutional neural network for fault diagnosis in linear motion guide. IEEE Trans Ind Inf. 2020;17(6):4036–4045. doi: 10.1109/TII.2020.3012989

[53] Li S, Li T, Sun C, et al. Multilayer Grad-CAM: an efective tool towards explainable deep neural networks for intelligent fault diagnosis. J Manuf Syst. 2023 08 01;69:20–30. doi: 10.1016/j.jmsy.2023.05.027

[54] Oh C, Jeong J. Vodca: verification of diagnosis using CAM-based approach for explainable process monitoring. Sensors. 2020;20(23):6858. doi: 10.3390/s20236858

[55] Lee J, Noh I, Lee J, et al. Development of an explainable fault diagnosis framework based on sensor data imagification: a case study of the robotic spot-welding process. IEEE Trans Ind Inf. 2022;18(10):6895–6904. doi: 10.1109/TII.2021.3134250

[56] Yang D, Karimi HR, Gelman L. An explainable intelligence fault diagnosis framework for rotating machinery. Neurocomputing. 2023 07 07;541:126257. doi: 10.1016/j.neucom.2023.126257

[57] Gwak M, Kim MS, Yun JP, et al. Robust and explainable fault diagnosis with power-perturbation-based decision boundary analysis of deep learning models. IEEE Trans Ind Inf. 2023;19(5):6982–6992. doi: 10.1109/TII.2022. 3207758

[58] Khosravani MR, Nasiri S, Weinberg K. Application of case-based reasoning in a fault detection system on production of drippers. Appl Soft Comput. 2019 02 01;75:227–232. doi: 10.1016/j.asoc.2018.11.017

[59] Zhao H, Liu J, Dong W, et al. An improved case-based reasoning method and its application on fault diagnosis of Tennessee Eastman process. Neurocomputing. 2017;249:266–276. doi: 10.1016/j.neucom.2017.04.022

[60] Boral S, Chaturvedi SK, Naikan VNA. A case-based reasoning system for fault detection and isolation: a case study on complex gearboxes. JQME. 2019;25(2):213–235. doi: 10.1108/JQME-05-2018-0039

[61] Chen M, Qu R, Fang W. Case-based reasoning system for fault diagnosis of aero-engines. Expert Syst Appl. 2022 09 15;202:117350. doi: 10.1016/j.eswa.2022.117350

[62] Utama C, Meske C, Schneider J, et al. Explainable artificial intelligence for photovoltaic fault detection: a comparison of instruments. Sol Energy. 2023 01 01;249:139–151. doi: 10.1016/j.solener.2022.11.018

[63] Kim MS, Yun JP, Park P. An explainable neural network for fault diagnosis with a frequency activation map. IEEE Access. 2021;9:98962–98972. doi: 10.1109/ACCESS.2021.3095565

[64] Jiang G, Wang J, Wang L, et al. An interpretable convolutional neural network with multi-wavelet kernel fusion for intelligent fault diagnosis. J Manuf Syst. 2023 10 01;70:18–30. doi: 10.1016/j.jmsy.2023.06.015

[65] Lu W, Yan X. Deep fisher autoencoder combined with self-organizing map for visual industrial process monitoring. J Manuf Syst. 2020;56:241–251. doi: 10.1016/j.jmsy.2020.05.005

[66] Dorgo G, Palazoglu A, Abonyi J. Decision trees for informative process alarm definition and alarm-based fault classification. Process Saf Environ Protect. 2021 05 01;149:312–324. doi: 10.1016/j.psep.2020.10.024

[67] Obregon J, Hong J, Jung J-Y. Rule-based explanations based on ensemble machine learning for detecting sink mark defects in the injection moulding process. J Manuf Syst. 2021 07 01;60:392–405. doi: 10.1016/j.jmsy.2021.07.001

[68] Harinarayan RRA, Shalinie SM. Xfddc: explainable fault detection diagnosis and correction framework for chemical process systems. Process Saf Environ Protect. 2022 09 01;165:463–474. doi: 10.1016/j.psep.2022.07.019

[69] Maged A, Xie M. Uncertainty utilization in fault detection using Bayesian deep learning. J Manuf Syst. 2022;64:316–329. doi: 10.1016/j.jmsy.2022.07.002

[70]. Conde D. Fernández MA, Rueda C, et al, Isotonic boosting classification rules, Ady Data Anal Classif, 2021 06 01:15 (2):289–313. doi: 10.1007/s11634-020-00404-9

[71] Doshi-Velez F, Kim B. Towards a rigorous science of interpretable machine learning. J Artif Intel Res. 2023 01 15;76 (1):897–950.

[72] Ramezankhani A, Harandi M, Seethaler M, et al. Smart manufacturing under limited and heterogeneous data: a simtoreal transfer learning with convolutional variational autoencoder in thermoforming. Int J Comput Integr Manuf. 2024:37(2):18-36