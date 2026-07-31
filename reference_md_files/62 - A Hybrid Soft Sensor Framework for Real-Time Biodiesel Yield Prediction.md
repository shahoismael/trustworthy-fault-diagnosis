# A hybrid soft sensor framework for real-time biodiesel yield prediction: Integrating mechanistic models and machine learning algorithms

![](images/469d6ed2d1c98abe38fe103ad80d5cd4b3c073b105468147797392c6adebe674.jpg)

Mustafa Kamal Pasha <sup>a</sup>, Lingmei Dai <sup>a</sup>, Dehua Liu <sup>a</sup>, Wei Du <sup>a,\*</sup>, Miao Guo

<sup>a</sup> Key Laboratory for Industrial Biocatalysis, Ministry of Education, Department of Chemical Engineering, Tsinghua University, Beijing 100084, China <sup>b</sup> Department of Engineering, Faculty of Natural & Mathematical Sciences, King’s College London, UK

## A R T I C L E I N F O

Keywords: Artificial intelligence Artificial neural network Biodiesel Machine learning Soft sensor

## A B S T R A C T

Biodiesel yield prediction is vital for optimizing process efficiency, minimizing costs, and maintaining product quality. Traditional methods are labor-intensive, costly. and lack real-time capabilities. leading to inefficiencies in operations. Data-driven soft sensors offer real-time prediction but require extensive, high-quality datasets, posing practical challenges. To address these limitations. this study proposes a hybrid soft sensor model that integrates mechanistic and data-driven approaches. Mechanistic models were utilized to generate computational data via MATLAB®, reducing the reliance on costly laboratory experiments. A comprehensive dataset (n = 1500) comprising seven input variables—catalyst type, feedstock type, temperature, reaction time, free fatty acid (FFA) content, water content, and methanol-to-oil ratio—along with one output variable (biodiesel yield) was developed. This dataset was used to train various machine learning algorithms, with the artificial neural network (ANN) model demonstrating the highest predictive accuracy, achieving an R<sup>2</sup> (goodness of fit) of 0.998 and root mean square error (RMSE) of 0.303. Hyperparameter tuning further enhanced the model’s performance, reducing RMSE and the mean absolute error (MAE) by 63 % and 61.7 %, respectively. By combining mechanistic and data-driven techniques, this hybrid model effectively overcomes the limitations of traditional and purely data-driven methods, providing a cost-effective and efficient solution for biodiesel yield prediction and data generation.

## 1. Introduction

Depleted oil reserves, rising crude oil prices, and environmental concerns have driven research into sustainable energy sources, particularly in the transportation sector. Biofuels, notably biodiesel, stand out due to their physiochemical similarity to diesel fuel [1]. Biodiesel’s appeal lies in its compatibility with existing infrastructure, making it a promising and environmentally friendly substitute for conventional diesel fuels [2]. Various biodiesel production technologies exist, including alkali-catalyzed [3], acid-catalyzed [4], heterogeneous acid-catalyzed [5], supercritical [3], and enzymatic processes [6]. The alkali-catalyzed process is commonly employed due to its rapid reaction time and high yield, but it has several drawbacks. In contrast, enzyme-catalyzed processes have gained attention for their efficiency and offer advantages such as mild reaction conditions, simplified separation, reduced waste generation, and flexibility in processing diverse feedstocks [7,8]. Du et al. [9], developed an innovative two-step enzymatic process that utilizes both free and immobilized lipase. This process, successfully industrialized in China using waste cooking oil, demonstrates remarkable efficiency with competitive unit prices, marking a significant advancement in biodiesel production. Our previous work covered various aspects of this technology, further highlighting its potential for widespread commercialization [10].

Biodiesel yield, a key performance indicator in the biodiesel production system, is significantly influenced by variations in catalyst and feedstock types. Precise and timely measurement of biodiesel yield during production is critically important to ensure optimal performance. The yield, representing the efficiency of converting raw materials into biodiesel, holds direct significance for operational costs, resource utilization, final product quality, and overall process efficacy [11]. Several factors intricately influence the yield, including the quality of raw materials such as Free Fatty Acid (FFA) content and impurity levels, as well as reaction conditions including temperature, pressure, catalyst type, and reaction time [12]. Thus, continued measurement of biodiesel yield is essential during process operation to ensure optimal production outcomes, mitigating wastage and operational inefficiencies. Biodiesel yield measurements during production process traditionally involve manual sampling, transport, and subsequent laboratory analysis using techniques such as Gas Chromatography (GC) [13], Mass Spectrometry (MS) [14], titration, and gravimetric analysis [15]. While these established analytical methods are accurate, they are labor-intensive and involve time-consuming off-line procedures, lacking the ability to offer real-time insights into the dynamic aspects of the production system.

![](images/fa80bfcabbea0b3e41c487f50bf744176e098c4534f3583a627561d385badf1f.jpg)  
Fig. 1. Schematic illustration of soft sensor development framework consists of the three main steps: (i) Data generation & pre-processing; (ii) Soft sensor development; (iii) Performance & accuracy assessment.

To ensure consistent estimation of biodiesel yield, it is crucial to develop online monitoring approaches that are easy to implement and robust. One effective solution is the incorporation of machine learning techniques through the use of soft sensors. Over the past three decades, soft sensors have been widely developed and applied in fields such as process control and optimization [16,17]. Soft sensors are inferential models that use easily measured variables to estimate process variables that are difficult to measure due to technological limitations, large measurement delays, or high investment costs [18]. Soft sensors can be categorized into model-driven (white-box-also called mechanistic model) and data-driven (black-box) models. Model-driven soft sensors are primarily used for planning and development of process plants [19]. These models are based on equations describing the chemical and physical principles underlying the process, such as mass-preservation principles, exothermic equations, energy balances, and reaction kinetics. However, developing these models requires extensive process expert knowledge, which is not always available. Additionally, model-driven soft sensors are difficult to integrate into existing process control systems and require extensive reprogramming and running full scale simulations in dynamic production environments where condi tions frequently change.

In contrast, data-driven soft sensors, known as black-box techniques, rely on empirical observations and historical data to make predictions [19]. These models have been extensively applied in biodiesel production processes to estimate offline parameters, including biodiesel yield [20,21], fuel quality [22,23], engine performance [24], cetane number [25], and emissions [26]. The application of data-driven soft sensors using machine learning (ML) techniques in biodiesel production has been comprehensively reviewed by Aghbashlo et al. (2021) [17]. Unlike model-driven soft sensors, data-driven models can be seamlessly integrated into existing process control systems, offering continuous and robust predictions without critical time delays. They also offer rapid adaptability and scalability to various production scenarios without the need for extensive reprogramming. However, data-driven models rely on expensive, laborious, and time-consuming collection of process data. Gathering a large amount of experimental or historical data under varied process conditions is not only costly and time-consuming but also environmentally detrimental. The limited availability of data, mostly ranging from 30 to 200 samples in previous research [17], restricts the model’s ability to capture intricate patterns and complexities of the biodiesel production process, leading to poor prediction accuracy when applied to new, unseen data. While there is no fixed requirement for dataset size in soft sensor development, larger datasets enhance the robustness of ML algorithms [27,28]. Therefore, careful selection of a comprehensive dataset encompassing diverse operational aspects and features is crucial. This practice allows algorithms to capture a broader range of patterns and relationships, extending beyond their standard parameters and improving prediction accuracy. Besides, biodiesel production processes involve diverse feedstocks, processing technologies, and catalyst types, all of which influence yield. To create a generalized model for yield predictions, extensive datasets covering all relevant features are necessary for ML algorithms to understand the wide spectrum of relationships between inputs and outputs [27]. However, previous research has often focused on singular feedstocks and catalysts in small-scale laboratory settings, leading to limited generalizability and susceptibility to overfitting or underfitting. issues which have been highlighted by previous studies [20,29–31]. Developing a comprehensive dataset encompassing multiple feedstocks and catalysts through experimentation remains challenging.

In view of the above context, this study employs a hybrid approach by combining model-driven and data-driven soft sensors, where the latter utilizes data generated by the mechanistic models. By integrating a mechanistic model (Aspen PLUS®) with MATLAB®, an industrially validated biodiesel production data is generated, encompassing a range of feedstocks and catalysts. This readily available and inexpensive data is then used to train and evaluate the performance of three ML regression models: Artificial Neural Network (ANN), Support Vector Regression (SVR), and Random Forest Regression (RFR). The effectiveness of each ML algorithm in predicting biodiesel yield was assessed using performance metrics such as the coefficient of determination (R<sup>2</sup>), root mean square error (RMSE) and mean absolute error (MAE). Hyperparameter tuning techniques are applied to optimize the algorithms and enhance their predictive capabilities. The dual approach, leveraging both data-driven and mechanistic models, ensures our predictive framework is robust and adaptable, meeting the demands of modern biodiesel production facilities. This work significantly advances process monitoring, automation, and control strategies in the biodiesel industry. Additionally, by reducing the laborious, expensive, and time-consuming nature of traditional laboratory trials required for data collection, it aligns with global efforts to promote a cleaner and more sustainable energy landscape.

a) Enzyme-catalyzed process using SBO and WCO  
![](images/2416f561a7136fee21ae622d3f8cd364d37ffb847affb2a9db42fa0f1c4b0aae.jpg)  
Fig. 2. Process flow diagram of biodiesel production process. (a) Enzymatic process using SBO and WCO, (b) Chemical catalyzed process using SBO

## 2. Methodology

Fig. 1 illustrates the methodological framework of soft sensor development, outlining the key stages of the process. The first stage involves data generation and pre-processing, where the dataset is generated, normalized, and prepared to ensure it is suitable for model training. In the second stage, soft sensor development, a range of machine learning models are trained, with a specific focus on employing optimization techniques to enhance their performance. The final stage focuses on performance and accuracy assessment, where evaluation metrics such as RMSE, $\mathrm { R } ^ { 2 } ,$ , and MAE are employed to assess the soft sensor’s effectiveness.

## 2.1. Mechanistic modeling and data generation

## 2.1.1. Process description of biodiesel production

Fig. 2 illustrates the biodiesel production processes using enzyme and chemical catalysts, utilizing soybean oil (SBO) and waste cooking oil (WCO) as feedstocks. In the enzymatic process (Fig. 2a), SBO/WCO is mixed with 1.5/1 wt% free lipase and processed through four reactors. Methanol is gradually added at a molar ratio of 4.5:1 in four steps to avoid enzyme deactivation and maintain stoichiometric deficiency. The reaction occurs at $4 0 ~ ^ { \circ } \mathrm { C }$ under atmospheric pressure, achieving about 90 % biodiesel conversion for SBO and 85 % for WCO. Side reactions cause hydrolysis, producing FFAs in the presence of water. The crude product undergoes centrifugation for phase separation; the light phase contains fatty acid methyl esters (FAME), unreacted oil, FFAs, methanol, water, and glycerol, while the heavy phase is treated with a ceramic membrane separator at 40 <sup>◦</sup>C and 0.2–0.3 MPa, recovering 97 % of the lipase. The permeate is distilled to recover methanol and water, yielding glycerol with 98 % purity. The biodiesel product, containing 3–4% FFAs, is further treated using immobilized lipase, converting FFAs into FAME (biodiesel), achieving over 98 % biodiesel yield within 2.5 h and reducing FFAs to less than 0.25 %.

The alkali-catalyzed process (Fig. 2b) operates at $6 0 ~ ^ { \circ } \mathrm { C }$ and 4 bar, with a methanol-to-oil ratio of 6:1 and 1 wt% NaOH. The reaction duration is 60 min, followed by methanol recovery through distillation. The crude product is washed to separate FAME from glycerol, NaOH, and methanol using a water-to-oil molar ratio of 2:67. NaOH is then neutralized with phosphoric acid at $6 0 \ { } ^ { \circ } \mathrm { C } ,$ converting it into $\mathrm { N a } _ { 3 } \mathrm { P O } _ { 4 } .$ Further purification yields 99.70 wt% FAME and 99.60 wt% glycerol. The operational aspects, design, and performance of these processes have been thoroughly detailed by the author in a previously published article [10].

Table 1  
Baseline conditions for the simulation of Aspen PLUS® process models [10].

<table><tr><td rowspan="2">Baseline Conditions</td><td>Process Model</td><td>Process Model</td><td>Process Model</td></tr><tr><td>1</td><td>2</td><td>3</td></tr><tr><td>Feedstock type</td><td>SBO</td><td>SBO</td><td>WCO</td></tr><tr><td>Catalyst type</td><td>NaOH</td><td>Liquid lipase</td><td>Liquid lipase</td></tr><tr><td>Impurities (%)</td><td>1</td><td>1</td><td>9</td></tr><tr><td>FFA content (%)</td><td>0</td><td>0</td><td>13</td></tr><tr><td>Methanol:Oil (molar)</td><td>6:1</td><td>4.5:1</td><td>4.5:1</td></tr><tr><td>Reaction time (hr)</td><td>1</td><td>8</td><td>18</td></tr><tr><td>Reactor temperature (°C)</td><td>60</td><td>40</td><td>40</td></tr><tr><td>Reactor pressure (bar)</td><td>4</td><td>1</td><td>1</td></tr><tr><td>Water content (%)</td><td>0</td><td>10</td><td>10</td></tr></table>

## 2.1.2. Mechanistic models development

For biodiesel yield prediction using soft sensors, the focus is solely on the reaction sections (highlighted in bold in Fig. 2) of the processes, as these are critical for developing mechanistic models. The yield of bio diesel at the reactor outlets is significantly influenced by the type and composition of the feedstock, as well as the catalyst used. These factors not only affect the reaction outcomes but also impact the subsequent separation and purification stages. Consequently, the operating conditions of the reactors, as detailed in Table 1, are tailored according to the catalyst type and feedstock composition.

Mechanistic models were developed using Aspen PLUS®, employing the UNIFAC property model (illustrated in Fig. 3). To accurately simulate catalytic behavior, the continuous stirred-tank reactor (CSTR) model was chosen instead of the stoichiometric reactor module. This choice allows for a more precise representation of the catalyst’s influence through reaction kinetics and process parameters specific to each catalyst type. Although the catalysts themselves are not explicitly modeled as components, their effects are incorporated via kinetics data, ensuring an accurate depiction of their behavior in the simulation. The CSTR model’s sensitivity to key operating conditions—temperature, pressure, residence time, and reactor volume—enables a realistic simulation of process dynamics. This sensitivity is crucial for generating data for further analysis. In the simulation, oil feedstock and methanol are mixed in the appropriate molar ratio, preheated, and then intro duced into the CSTR at the specified reaction temperature. The catalyst, added as a specific weight percentage of the oil, facilitates the reaction over a time period dependent on the feedstock and catalyst type. The reaction yields biodiesel along with by-products such as glycerol, unreacted methanol, FFA, water, and residual catalyst. The kinetic data for both enzymatic [32,33] and alkali-catalyzed [34] reactions, sourced from the literature, are used to simulate these reactions accurately within the CSTR framework.

## 2.1.3. Data generation and collection procedure

The computational procedure for data generation from the developed mechanistic model is illustrated in Fig. 4. MATLAB® was initially used to generate ‘n’ number of samples for each process variable, ensuring a comprehensive and representative dataset. A random sampling method, utilizing a normal distribution via the ‘randn’ function in MATLAB®, was employed (see Section S1of the Supplementary Information (SI) for details). Following sample generation, MATLAB® was integrated with Aspen PLUS® through an ActiveX server (a Microsoft Technology that allows software components to interact), enabling automated control of the Aspen PLUS® simulations (further details are provided in Section S2 of the SI). This integration allowed for systematic variations in the steady-state values of the mechanistic model while capturing the corresponding changes in input and output variables. This methodology of integrating MATLAB® with Aspen PLUS® is consistent with the approaches described by Ahmad et al. [28], and Nkulikiyinka et al. [18], aligning with established practices in the literature. By systematically varying input parameters in Aspen PLUS®, the simulations were run to convergence, producing output responses, particularly focusing on biodiesel yield.

![](images/df34d1da1d18a205d4f360dbc1162a114b063892154f9b9e0122390fe5c49f34.jpg)  
Fig. 3. Process flow diagram of biodiesel production process simulated in Aspen PLUS®.

![](images/d683f3a8c77c596aadea16b77837b6fc4e84f9be22073f7f48c805fbabf9e8fb.jpg)  
Fig. 4. Aspen PLUS®-MATLAB® integration for computational data generation

Table 2  
Set of input variables used to generate a dataset for biodiesel production process.

<table><tr><td>Parameter</td><td>Range</td><td>Interval</td><td>Mean</td></tr><tr><td>Catalyst type</td><td>Numerical attribute (1–2)</td><td>-</td><td>-</td></tr><tr><td>Feedstock type</td><td>Numerical attribute (1–2)</td><td>-</td><td>-</td></tr><tr><td>Temperature (°C)</td><td>30–80</td><td>1</td><td>40 and 60</td></tr><tr><td>Reaction time (hour)</td><td>1–30</td><td>1</td><td>9 and 18</td></tr><tr><td>FFA content (%)</td><td>0–25</td><td>0.5</td><td>13</td></tr><tr><td>Water content (%)</td><td>0–15</td><td>1</td><td>10</td></tr><tr><td>MeOH:Oil (molar)</td><td>1–10</td><td>0.5</td><td>4.5</td></tr></table>

The MATLAB® code was programmed to execute a specified number of iterations, generating a dataset that included seven critical input variables: feedstock type, catalyst type, reaction temperature, reaction time, FFA content, water content, and methanol-to-oil molar ratio. This structured approach ensured the creation of a reliable dataset for subsequent data-driven soft sensor modeling, enabling comprehensive variation in input conditions and recording output responses for each data sample. Table 2 provides details of the input parameters, including their respective ranges, intervals between samples, and the mean value around which the samples are symmetrically distributed.

## 2.2. Machine learning algorithms

Three machine learning algorithms, namely ANN, RFR, and SVR, were chosen for the development of the soft sensor (for details, see Section S4, S5 and S6 of the SI). These algorithms were specifically selected due to their proven success in biodiesel yield prediction applications, demonstrating a consistent track record of achieving high prediction accuracy. For instance, Sultana et al. (2022) [20] used ANN and SVR to predict biodiesel yield and compared their results with RSM, where SVR performed better in prediction accuracy. Nkulikiyinka et al. (2020) [18] adopted ANN and RF to build a prediction model using Aspen PLUS® simulation. Agrawal et al. (2024) [21] employed 14 different machine learning algorithms and evaluated their performance on biodiesel yield prediction. Out of 14 different algorithms, RF, Cat Boost and XGB-RF showed improved performance. These algorithm have exceptional performance across a diverse array of data scenarios. having the capacity to accommodate varying feature distributions, and adeptness in handling complex correlations within datasets [21,35]. These intrinsic characteristics render these algorithms potentially well-suited for application in the context of biodiesel yield predictions. Each of the selected model, underwent comprehensive training, evaluation, and validation procedures within the MATLAB® environment. This standardized approach ensures a rigorous and systematic assessment of their performance in the specific context of biodiesel yield prediction.

## 2.2.1. Artificial neural network

Artificial Neural Networks are well-established machine learning technique, garnering significant attention in various engineering appli cations $[ 3 6 , 3 7 ]$ . The network exhibits the capability to map input datasets to corresponding outputs. A prevalent architecture within ANN is the multilayer perceptron (MLP), employing a feed-forward back propagation network, consisting of input layers, hidden layer(s), and an output layer. The relationship between these layers is mathematically expressed as per Eq. (1) [20]:

$$
y _ {i} = f \left(\sum_ {i = 1} ^ {N} w _ {i} x _ {i} + b\right)\tag{1}
$$

Here, x represents input parameters, w signifies the weighted average of input parameters, f denotes the activation function generating output y, N stands for the number of inputs, and b represents the bias function facilitating optimal fitting of input data. Various transfer or activation functions, such as hyperbolic tangent sigmoid (tansig), linear (purelin), and log sigmoid (logsig), are available for data transfer [38]. In this study, tansig was chosen for transferring data from input to hidden layers, while purelin was selected for transferring data from hidden to output layers.

Among several backpropagation algorithms (e.g., Gradient Descent, Scaled Conjugate Gradient, Gradient Descent with Momentum, Levenberg-Marquardt, and Quasi-Newton), the Levenberg-Marquardt algorithm was employed for its recognized high performance [20]. The iterative process was pursued until achieving the minimum mean squared error (MSE), indicating optimization in the model’s performance.

## 2.2.2. Random Forest Regression

Random Forest, a supervised learning method introduced by Breiman, is adept at addressing classification and regression problems [21]. As an ensemble method, it utilizes individual models known as decision trees to collectively enhance prediction robustness and accuracy. The method’s versatility stems from its capability to effectively handle diverse problem types through the aggregation of predictions from multiple decision trees. Decision trees, which recursively partition data based on input attribute values, form the fundamental building blocks of the method.

## y = value of the leaf node

(2)

The predicted value (y) of a decision tree, as per Eq. (2), represents either a class label for classification or a continuous value for regression tasks. In the context of RFR, decision trees are constructed using randomly sampled subsets of the data, and their predictions are aggregated to yield the final prediction, as expressed in Eq. (3), where y is the result obtained by aggregating predictions from all decision trees divided by the total number of trees (n) [21].

$$
y = \frac {\sum (\text { predictions   of   the   decision   trees })}{n}\tag{3}
$$

Random Forest excels in handling multi-dimensional datasets with both categorical and numerical variables. To enhance prediction performance, a random subset of features is considered at each split during decision tree construction. This strategy improves model robustness by preventing overfitting and fostering generalized predictions. Despite computational expenses, the method’s resilience to outliers and ability to handle missing data contribute to its accuracy in various regression and classification problems.

## 2.2.3. Support Vector Regression

Support Vector Regression is a widely utilized machine learning approach that establishes a line or hyperplane in a higher-dimensional space to effectively fit the data [20,39]. To navigate the challenges associated with increased computational costs in higher dimensions, SVR employs a kernel function (φ) to find the hyperplane without incurring additional computational burdens. Boundary lines, forming a border on each side of the hyperplane, are crucial components, and support vectors are identified as data points closest to these boundary lines.

The connection between input (z) and output (y<sub>i</sub>) variables is rep resented by Eq. (4) [20]:

$$
y _ {i} = k (z) = \nu \phi (z) + c\tag{4}
$$

Her $\ , z = ( z 1 , z 2 , z 3 , z n )$ denotes the input value $, y _ { i } \in R$ is the output value, and $\nu \in R n , c \in R$ , and n represent the support vector weight vector, bias, and the number of training datasets, respectively. Additionally, ϕ(z) represents an irregular function for allocating input data. To elaborate on v and $c ,$ the mathematical expression described in Eq. (5) is used, which is subjected to the constraints defined by Eq. (6) [20].

$$
\text { Minimize }: \frac {1}{2} \| v ^ {2} \| + C \sum_ {i = 1} ^ {n} \left(\xi_ {i} - \xi_ {i} ^ {*}\right)\tag{5}
$$

$$
\text { Subject   to }: \left\{ \begin{array}{c} y _ {i} - (v \phi (z _ {i}) + c _ {i}) \leq \varepsilon + \xi_ {i} \\ (v \phi (z _ {i})) + c _ {i} - y _ {i} \leq \varepsilon + \xi_ {i} ^ {*} \\ \xi_ {i}, \xi_ {i} ^ {*} \geq 0 \end{array} \right.\tag{6}
$$

Here, C is the box constraint balancing empirical risk and model flatness, ξi and ξi<sup>\*</sup> are slack variables, and ε is the insensitive loss function defined in Eq. (7).

$$
L _ {\varepsilon} (y, k (z)) = \left\{ \begin{array}{c c} 0 & \text { if } | y - k (z) | \leq \varepsilon \\ | y - k (z) | - \varepsilon & \text { otherwise } \end{array} \right.\tag{7}
$$

The solution involves introducing Lagrangian multipliers $( \alpha _ { i }$ and $\boldsymbol { \alpha _ { i } ^ { * } } )$ and solving the optimization problem.

$$
\begin{array}{l} \text { Maximum }: \frac {1}{2} \sum_ {i = 1} ^ {n} \times \sum_ {j = 1} ^ {n} \left(a _ {i} - a _ {i} ^ {*}\right) \left(a _ {j} - a _ {j} ^ {*}\right) K (z _ {i}, z _ {j}) - \varepsilon \sum_ {i = 1} ^ {n} \left(a _ {i} + a _ {i} ^ {*}\right) \\ \quad + y _ {i} \sum_ {i = 1} ^ {n} \left(a _ {i} - a _ {i} ^ {*}\right) \end{array}
$$

$$
\text { Subject   to }: \sum_ {i = 1} ^ {n} \left(a _ {i} - a _ {i} ^ {*}\right) = 0 \text {   and   } a _ {i}, a _ {i} ^ {*} \in | 0, C |
$$

Where, $K ( z _ { i } , z _ { j } ) = \varphi ( z _ { i } ) .$ . φ(z ) is the kernel function. The final form of the SVR equation is expressed as Eq. (8):

$$
k (z) = \sum_ {i = 1} ^ {n} \left(a _ {i} - a _ {i} ^ {*}\right) K \left(z _ {i}, z _ {j}\right) + c\tag{8}
$$

The Radial Basis Function (RBF) used in this investigation is defined

![](images/bb489bc7c0386002ac9431788e47e72b5154f6a0b34d185aac1f39665035a4fb.jpg)  
Fig. 5. Validation of mechanistic models against industrial and literature data.

as Eq. (9):

$$
K \big (\mathbf {z} _ {i}, \mathbf {z} _ {j} \big) = e ^ {- \gamma \left\| \mathbf {z} _ {i} - \mathbf {z} _ {j} \right\| ^ {2}}\tag{9}
$$

## 2.3. Soft sensor model development

The dataset (n = 1500) with seven input parameters (feedstock type, catalyst type, temperature, time, FFA content, water content, methanol to oil ratio) and one output parameter (biodiesel yield), was utilized to train the soft sensor models. The framework for developing these models (Fig. 1) follows the main steps: (i) statistical analysis and dataset preprocessing; (ii) feature analysis using the Pearson’s Correlation; (iii) model development, tuning and calibration; (iv) testing of ML models to compare the models’ prediction accuracy; and (v) final soft sensor model with feature importance and additional tests for biodiesel prediction.

Given the diverse array of input and output variables within the dataset, a standardization procedure was undertaken before employing ML algorithms. This procedure aimed to diminish noise and enhance the comparability of input variables [40,41]. In preparation for analysis, multiple pre-processing steps were executed, including handling missing data, rescaling data, converting categorical attributes to numerical attributes, and standardizing the data. The input data instances were normalized within the range [0–100], with biodiesel feedstock type and catalyst type converted into numerical data using label encoding [0–2] before normalization and standardization.

During the variable selection step, Pearson’s Correlation Coefficient (PCC) was employed to identify correlations between variables, aiming to detect and eliminate redundancy, thereby mitigating overfitting issues before ML algorithm training. The PCC has been effectively utilized in previous studies, including those by Shyu et al. [39], Hikosaka et al. [42], and Zhang et al. [43], to illustrate the correlations between different variables.

Finally, after data analysis, 70 % of the data generated was randomly selected for model training, with the remaining 30 % reserved for testing the trained models. The optimal structure of ML models is heavily contingent on their hyperparameters. For hyperparameter tuning, a stratified 5-fold cross-validation resampling method was selected, considering the computational cost and the observed trend of RMSE with $\mathbf { \nabla } ^ { \cdot } k ^ { \prime } .$ . This method involves randomly partitioning the training dataset into five subsets, fitting the model to four of these subsets, and using the remaining one for validation to estimate errors and determine model performance when fitting the training dataset [21,39]. Although generalization performance typically increases with $\mathbf { \nabla } ^ { \cdot } k ^ { \prime }$ , it entails a higher computational cost, necessitating a trade-off. The ‘expand.grid function in MATLAB® facilitated grid search to determine the best hyperparameters for specific algorithms during the model training process [21].

To evaluate the performance of the regression models, several metrics were employed, including RMSE, $\mathrm { R } ^ { 2 } ,$ , and MAE. These metrics were selected to quantitatively analyze the predictive capabilities of the models [39]. The formulas for $\mathrm { R } ^ { \mathbf { \bar { 2 } } } \left( \mathrm { E q } . \mathrm { ( 1 0 ) } \right)$ , MAE (Eq. (11)), and RMSE (Eq. (12)) incorporate observed values (y ), predicted values $( x _ { i } ) _ { ; }$ , the mean of observed values $( y )$ , and the total number of samples (n).

$$
R ^ {2} = 1 - \frac {\sum_ {i = 1} ^ {n} (y _ {i} - x _ {i}) ^ {2}}{\sum_ {i = 1} ^ {n} (y _ {i} - \overline {{y _ {i}}}) ^ {2}}\tag{10}
$$

$$
M A E = \frac {\sum_ {i = 1} ^ {n} \left| \frac {x _ {i} - y _ {i}}{y _ {i}} \right|}{n}
$$

(11)

$$
R M S E = \sqrt {\frac {\sum_ {i = 1} ^ {n} (y _ {i} - x _ {i}) ^ {2}}{n}}\tag{12}
$$

![](images/afdc0ec48b530a26e0c0cd0c5a922e6dbfadfc91e426c1a5b6a8212c2f6f2f3a.jpg)

![](images/53b71bbe99d2cabea0a8c0eaf8bb33ccda191ff3e3f82315360f2e003257e5ee.jpg)

![](images/2480da51fe8da766b0534a14eedba132ad280cbf11cf6105759da66650860f40.jpg)

![](images/d4e1ec6ad54da6c38a8934c529d7faf6d7cc7d00e820434c681bda03ad0b8620.jpg)

![](images/6bf5ba699d7c762d2701a6af58ed88fe0970e59b858b2f650e1a3595b352a4c2.jpg)  
Fig. 6. Graphical descriptions of the inputs and output data layers versus number of samples for (a) temperature, (b) reaction time, (c) FFA content, (d) water content, (e) methanol-to-oil ratio, (f) biodiesel yield.

## 3. Results and discussion

3.1. Validation of mechanistic models and data collection for biodiesel production system

## 3.1.1. Validation of mechanistic models

The steady-state mechanistic models developed in Aspen PLUS®, under normal operating conditions, as outlined in Table 1, form the basis for generating training data essential for developing soft sensor models for biodiesel yield prediction. To ensure the reliability of these models for data generation, validation of the simulation results is critical.

In this study, three biodiesel production processes were simulated using Aspen PLUS®: two different feedstocks (soybean oil and waste cooking oil) processed using two different catalysts (an enzymatic catalyst and an alkali catalyst). The simulation results of the product stream (see Fig. 3) are validated through comparison with reference data as depicted in Fig. 5. Specifically, for enzymatic processes, the simulated results were validated against industrial data, whereas for the alkali catalyzed process [3], literature data were used for comparison. The results show that the simulated biodiesel yields for each combination of feedstock and catalyst closely match the reference values, with minimal relative errors, confirming that the models accurately replicate real plant operations.

This validation indicates that the mechanistic models accurately replicate industrial operations and are reliable for generating the data needed for soft sensor development. By accurately capturing key phe nomena in biodiesel production, these models provide a robust frame work for generating diverse datasets that account for various feedstocks, catalysts, and operating conditions. Such datasets are crucial for developing generalized machine learning models capable of predicting biodiesel yields under different scenarios. Additionally, leveraging these validated models minimizes the need for expensive and time-consuming laboratory experiments, improving the efficiency of the soft sensor development process.

## 3.1.2. Collection of training dataset

The integration of MATLAB® and Aspen PLUS® was crucial for automating the data generation process for biodiesel production modeling. MATLAB®, using its ActiveX interface, systematically varied key input variables such as temperature, reaction time, methanol-to-oil molar ratio, FFA content, and water content. This approach resulted in the generation of 1500 data samples, covering a range of operating conditions for both enzyme-catalyzed and alkali-catalyzed biodiesel production processes. The dataset is divided into three segments: the first 500 samples represent the alkali-catalyzed process using soybean oil, while the next 500 samples correspond to the enzyme-catalyzed process using soybean oil, and the final 500 samples pertain to the enzyme-catalyzed process using waste cooking oil. This distribution ensures a comprehensive exploration of the design space, reflecting the variability present in real-world biodiesel production systems.

Fig. 6 illustrates the normal distribution of the sampled data, providing an overview of the range and trends for each variable. The sequential arrangement of samples highlights the distinctions between feedstocks and catalyst types, illustrating how the dataset covers various conditions with values distributed around their respective optima. For example, Fig. 6a shows the normal distribution of temperature, ranging from 25 <sup>◦</sup>C to 80 <sup>◦</sup>C. The data clusters prominently at 40 <sup>◦</sup>C and 60 <sup>◦</sup>C, which correspond to the optimal temperatures for the enzyme-catalyzed and alkali-catalyzed processes, respectively. The enzyme-catalyzed process operates optimally at 40 <sup>◦</sup>C to maintain enzyme activity, while the alkali-catalyzed process typically requires higher temperatures (around 60 <sup>◦</sup>C) for efficient conversion rates [44]. The distribution of samples around these optimal points reflects realistic process variability.

Fig. 6b displays the distribution of reaction times, with samples clustering at durations of 1, 9, and 18 h, representing typical timeframes for both processes involving soybean oil and waste cooking oil. Fig. 6c shows the distribution of FFA content, where the first 1000 samples exhibit zero FFA, indicating the use of refined soybean oil with negligible acid value. In contrast, the FFA content in samples from 1001 to 1500, representing waste cooking oil, ranges from 5 % to 25 %, consistent with the higher FFA levels typically found in such feedstocks [45]. Similarly, Fig. 6d illustrates the distribution of water content. The alkali-catalyzed process samples contain no water, as the presence of water in this system leads to soap formation. In contrast, the enzyme-catalyzed process requires a certain amount of water for the liquid lipase catalyst to function effectively, with water content varying between 6 % and 15 %. Fig. 6e depicts the methanol-to-oil molar ratio, showing optimal values around 6:1 for the alkali-catalyzed process and approximately 4.5:1 for the enzyme-catalyzed process. The samples are distributed around these optima, representing the variability expected in industrial scenarios.

<table><tr><td></td><td>Feedstock type</td><td>Catalyst type</td><td>Temperature (°C)</td><td>Reaction Time (hr)</td><td>FFA Content (%)</td><td>Water Content (%)</td><td>MeOH:Oil (Molar)</td><td>Biodiesel Yield (%)</td></tr><tr><td>Feedstock type</td><td>1.00</td><td>0.50</td><td>0.42</td><td>-0.85</td><td>-0.97</td><td>-0.47</td><td>0.59</td><td>0.48</td></tr><tr><td>Catalyst type</td><td>0.50</td><td>1.00</td><td>0.84</td><td>-0.81</td><td>-0.48</td><td>-0.98</td><td>0.69</td><td>0.58</td></tr><tr><td>Temperature (°C)</td><td>0.42</td><td>0.84</td><td>1.00</td><td>-0.68</td><td>-0.40</td><td>-0.82</td><td>0.57</td><td>0.54</td></tr><tr><td>Reaction Time (hr)</td><td>-0.85</td><td>-0.81</td><td>-0.68</td><td>1.00</td><td>0.82</td><td>0.79</td><td>-0.72</td><td>-0.57</td></tr><tr><td>FFA Content (%)</td><td>-0.97</td><td>-0.48</td><td>-0.40</td><td>0.82</td><td>1.00</td><td>0.45</td><td>-0.60</td><td>-0.48</td></tr><tr><td>Water Content (%)</td><td>-0.47</td><td>-0.98</td><td>-0.82</td><td>0.79</td><td>0.45</td><td>1.00</td><td>-0.67</td><td>-0.57</td></tr><tr><td>MeOH:Oil (Molar)</td><td>0.59</td><td>0.69</td><td>0.57</td><td>-0.72</td><td>-0.60</td><td>-0.67</td><td>1.00</td><td>0.81</td></tr><tr><td>Biodiesel Yield (%)</td><td>0.48</td><td>0.58</td><td>0.54</td><td>-0.57</td><td>-0.48</td><td>-0.57</td><td>0.81</td><td>1.00</td></tr></table>

Fig. 7. Pearson’s correlation of biodiesel production dataset depicted as a heat map. Blue colors denote a negative correlation, whereas yellow color denotes a positive correlation.

Finally, Fig. 6f visualizes the output response—biodiesel yield—from the Aspen PLUS® model. A more detailed analysis of the probability distribution of the data samples is provided in Fig. S1 of the SI, offering insight into the sampling methodology and distribution of input vari ables for the biodiesel production processes. The sampling approach ensures comprehensive coverage of both optimal and suboptimal con ditions, providing valuable insights into the impact of reaction condi tions on biodiesel yield.

This systematic and randomized sampling strategy ensures efficient exploration of the design space, with input variables distributed within experimentally validated ranges. The resulting dataset, comprising 1500 samples, serves as a robust foundation for developing the soft sensor model, providing a thorough representation of biodiesel production processes under various conditions.

## 3.2. Data analysis

## 3.2.1. Data pre-processing and correlation analysis

Data pre-processing is a critical step in preparing the dataset for analysis and modeling, ensuring consistency and optimizing the performance of machine learning algorithms. The process included normalizing all input and output variables, such as FFA flowrate, water flowrate, methanol flowrate, and biodiesel yield, to a scale of 0–100. This normalization ensures that all variables are on a consistent scale, preventing features with larger numeric values from dominating the learning process, which is essential for the effective performance of machine learning models. Additionally, numerical attributes (0, 1, and 2) were assigned to differentiate between catalyst types and feedstock types. This categorization allows the model to distinguish between different feedstocks and catalysts, enhancing its ability to interpret and utilize categorical information effectively during the modeling process.

To examine the relationships between the input variables and iden tify potential overlap, Pearson’s correlation coefficient was applied. The PCC measures the linear correlation between pairs of variables, with values ranging from − 1 (perfect negative correlation) to 1 (perfect positive correlation), and 0 indicating no correlation [39]. This analysis is crucial for detecting multicollinearity, which occurs when independent variables are highly correlated, suggesting that they may be capturing the same underlying phenomenon. Multicollinearity can reduce model interpretability and performance, so identifying and possibly removing redundant variables can help improve model efficiency and reduce computational complexity.

The Pearson correlation heat map (Fig. 7) visualizes the relationships among input variables and biodiesel yield. The color intensity and numerical values in the heat map indicate the strength and direction of these correlations. Biodiesel yield shows positive correlations with feedstock type (0.48), catalyst type (0.58), temperature (0.54), and methanol-to-oil ratio (0.81). In contrast, it exhibits negative correlations with FFA content (-0.48), water content (-0.57), and reaction time (-0.57). The heat map also highlights that feedstock and catalyst types are highly correlated with other conditions like FFA and water content, suggesting potential multicollinearity due to these overlapping variables. Although feedstock and catalyst type variables are redundant and could be removed to simplify the model, they were retained because they play a crucial role as numerical attributes that categorize the dataset. Keeping these variables enhances the model’s ability to differentiate between distinct feedstock and catalyst types, which is essential for accurate yield prediction.

Overall, the Pearson correlation analysis provides a detailed understanding of the interrelationships between process variables, offering a solid basis for developing predictive models. This understanding helps inform decisions about which variables to include or exclude from the final model to optimize predictive accuracy and efficiency.

![](images/32f20f984593ad69eef9bae6bfb40d7aa520d592cfd8a2673921b08765603511.jpg)

![](images/f1da6afd0a46e89f619cfe786f65c63351bca9ff5a31bd2958f512044c4c2788.jpg)

![](images/7badcfc15dadc66cdc59ec501304d2f1e321145d8ae2948833754547608beb12.jpg)

![](images/a6022df6a10ab3598010b9aa8ee159988e308c53d14d5e83e0fc2c9f22fd94e5.jpg)

![](images/b91be92fd12cda510921ff5eda9f75f098d0ce64468fd9b1b580373a293b2316.jpg)

f  
![](images/cd2a5de8879ade991b7a08b6d55fe5a75659d4810b979f562f8e2e996f79c4ff.jpg)  
Fig. 8. Influence of data size on the performance (in terms of $\mathbb { R } ^ { 2 }$ and RMSE during training and testing) of machine learning algorithms.

## 3.2.2. Analysis of training data size

The analysis of training data size is essential in understanding its influence on machine learning algorithms, specifically for biodiesel yield prediction, where the relationships among input variables can be complex and nonlinear. Evaluating performance with different data sizes helps determine the model’s capability to handle, learn, and generalize effectively.

Fig. 8 presents a detailed analysis of the training and testing per formance, characterized by $\mathrm { R } ^ { 2 }$ (goodness of fit) and RMSE (prediction error), for three machine learning models—ANN, RFR, and SVR—as the dataset size varies. For the ANN model (Fig. 8a), at an initial dataset size of 100 samples, there is a notable disparity between the training $\mathtt { R } ^ { 2 }$ (approximately 0.998) and testing $\mathrm { R } ^ { 2 }$ (approximately 0.8), indicative of potential overfitting. The model fits the training data well but struggles with generalization to unseen data. As the dataset size increases, this disparity reduces, and both training and testing $\mathrm { R } ^ { 2 }$ converge toward higher values, around 0.9999, suggesting that a larger training dataset enhances ANN’s ability to generalize, reducing overfitting and improving robustness. The RMSE (Fig. 8b) trends mirror the behavior seen in $\bar { \mathsf { R } ^ { 2 } } .$ . With smaller datasets, the RMSE is higher in the testing phase, indicating poor generalization. However, as the dataset size grows, the RMSE decreases, reflecting improved predictive accuracy. This trend implies that increasing dataset diversity allows the ANN to capture complex input-output relationships more effectively. Overall, ANN consistently improves with a larger dataset, demonstrating its flexibility and capability to handle complexity, provided there is sufficient data. This makes ANN a strong candidate for biodiesel yield pre diction when enough data points are available.

For the RFR model (Fig. 8c), the training $\mathrm { R } ^ { 2 }$ is consistently high, suggesting effective learning from the training data. However, the testing $\mathtt { R } ^ { \overset { \triangledown } { 2 } }$ shows a fluctuating pattern, indicating some instability in performance as the dataset size changes. This variability could imply sensitivity to data diversity or the presence of outliers in certain subsets. The RMSE (Fig. 8d) for training shows a downward trend, reflecting that RFR effectively minimizes error on training data as the size increases. In contrast, the testing RMSE shows a zigzag pattern, correlating with the instability observed in testing $\mathrm { \mathrm { R } } ^ { 2 } .$ . This indicates that while RFR is capable of learning well from training data, it may struggle with generalization due to its sensitivity to the characteristics of the testing data. RFR’s fluctuating performance suggests that it is sensitive to the specific characteristics of the data, and this instability, particularly in the testing phase, highlights the need for careful hyperparameter tuning and potentially more robust feature engineering to enhance stability and generalizability.

The SVR model (Fig. 8e) initially shows good performance, but as the dataset size increases, there is a noticeable decline in training $\mathrm { { R } } ^ { 2 } .$ . This trend suggests that the SVR model struggles with the increasing complexity and diversity of larger datasets, which may not be captured effectively by the kernel function used (likely Gaussian in this case). Similar to $\mathrm { \tt R } ^ { \frac { \cdot } { 2 } } ,$ , the training RMSE (Fig. 8f) increases as the dataset size grows, which is unusual compared to typical expectations that more data should lead to better model performance. This behavior could be attributed to several factors: larger datasets often introduce more noise and diverse patterns, which can challenge SVR’s ability to generalize effectively; the SVR’s kernel might be unable to adequately capture the relationships as the data complexity increases, leading to overfitting or

d

f  
![](images/22ec8372202cc4f3d1042ea2a813cdd7eacdf7c4808ac8fd1a027af0d9562eaf.jpg)

![](images/d3c602a814fb7aa771d3fb89d4619b777958f839b9a12e400cf6b3f71329547f.jpg)

![](images/829c83cec616fbdcabe491fd2b51a59a31fc33681efb6c46acd8e30416e50373.jpg)

![](images/db3fa864f4c70abdf26c06411d0c2cf5b0525d8f83290b6a4a4131e1e4a6bebf.jpg)

![](images/71d36454e152c3028b1e259c56df01572f821f0ac182fe8f99cf65a46effdd6c.jpg)

![](images/9de58ad7c18e0e9500cce2df05bce4c223ddaf03d7356c38667a959017229ad0.jpg)  
Fig. 9. Regression plots of predicted vs. actual biodiesel yield for ANN, RFR, and SVR models, showing training and testing performance metrics $( \mathbb { R } ^ { 2 } ;$ , RMSE, and MAE).

underfitting issues; and the model’s performance degradation suggests that the hyperparameters might not be optimally tuned for the larger dataset, leading to poorer performance. The SVR model’s declining performance with increasing dataset size indicates challenges in generalization and potential overfitting/underfitting issues. This observation is consistent with Sultana et al. [20], who found that the SVR model performs well with smaller datasets, but its accuracy diminishes as the dataset size increases. Proper hyperparameter optimization and kernel function selection is crucial to ensure the SVR remain effective with larger datasets.

Overall. ANN displaved optimal performance across dataset sizes compared to other algorithms, with effective generalization to testing data, outperforming RFR and addressing overfitting/underfitting concerns observed in RFR and SVR. This analysis underscores the importance of careful model selection, hyperparameter tuning, and regularization, especially when scaling up the dataset size to ensure that the model remains effective and efficient across different data volumes.

## 3.3. Soft sensor modeling and prediction of biodiesel yield

## 3.3.1. Model development and performance comparison

Three machine learning models—ANN, RFR, and SVR—were employed to predict biodiesel yields, with the models trained on 70 % of the available samples (1050 samples) and evaluated using the remaining 30 % testing dataset (450 samples). The training data was used to estimate model parameters, while the testing data provided an unbiased assessment of the models’ predictive performance.

Fig. 9 presents scatter plots of the predicted versus actual vields. with fitting lines for the three models, alongside evaluation metrics including RMSE, MAE, and $\mathtt { R } ^ { 2 } .$ As shown in Fig. 9a and $\mathbf { b } ,$ the ANN model displayed superior predictive accuracy compared to RFR and SVR. The training RMSE for ANN (Fig. 9a) was 0.303, with an $\mathrm { R } ^ { 2 }$ of 0.99837, which indicates an excellent fit to the training data. The testing results (Fig. 9b) also demonstrated high performance, with an RMSE of 0.361 and an $\mathrm { R } ^ { 2 }$ of 0.99813, suggesting that the ANN model generalized well and did not overfit. The minimal gap between training and testing $\mathrm { R } ^ { 2 }$ values indicates consistent performance, demonstrating ANN’s strong ability to capture the underlying data patterns while avoiding overfitting or underfitting issues. Additionally, the low MAE value of 0.136 for training and 0.189 for testing highlights the ANN model’s precision in predicting biodiesel yield values.

The RFR model also showed acceptable predictive performance, though not at the same level as ANN. The training $\mathrm { R } ^ { 2 }$ for RFR (Fig. 9c) was 0.960, indicating that the model was able to explain most of the variance in the training data. However, the testing $\mathrm { R } ^ { 2 }$ dropped to 0.946, and the testing RMSE increased to 1.957, indicating that the model had some difficulty generalizing compared to ANN (Fig. 9d). While the RFR model was capable of learning effectively, the larger discrepancy between training and testing results points towards potential sensitivity to the data or limitations in its ability to capture more complex relationships in the dataset.

The SVR model exhibited the lowest predictive accuracy among the three models. The training $\mathrm { R } ^ { 2 }$ for SVR was 0.887, and the corresponding RMSE was 2.778, which implies that SVR struggled more with capturing the variance in the training dataset (see Fig. 9e). The testing $\mathrm { R } ^ { 2 }$ decreased to 0.831, and the RMSE for testing rose to 3.801, reflecting a significant drop in model performance when applied to unseen data (Fig. 9f). This suggests that SVR had difficulty generalizing and was less effective in predicting biodiesel yields accurately, potentially due to challenges in dealing with the complexity of the dataset, insufficient regularization, or suboptimal hyperparameter selection.

Table 3  
Hyperparameters and their optimal values of ML models.

<table><tr><td>Parameter</td><td>Optimal Value</td></tr><tr><td>ANN</td><td></td></tr><tr><td>Network type</td><td>Feed-Forward Back-Propagation</td></tr><tr><td>Training function</td><td>Levenberg-Marquardt (Trainlm)</td></tr><tr><td>Hidden layer</td><td>1</td></tr><tr><td>Activation function</td><td>Tangent-sigmoid</td></tr><tr><td>Learning rate</td><td>0.001</td></tr><tr><td>Number of neurons</td><td>21</td></tr><tr><td>Maximum Epochs</td><td>100</td></tr><tr><td>RFR</td><td></td></tr><tr><td>Number of trees</td><td>350</td></tr><tr><td>Maximum depth</td><td>100</td></tr><tr><td>Minimum leaf size</td><td>1</td></tr><tr><td>SVR</td><td></td></tr><tr><td>Box constraint</td><td>1000</td></tr><tr><td>Kernel function</td><td>Gaussian</td></tr><tr><td>Kernel Scale</td><td>10</td></tr><tr><td>Epsilon</td><td>0.0654</td></tr></table>

The black trend lines in each of the scatter plots represent the ideal relationship where predicted values match the actual values perfectly. In the ANN model plots, the predicted values are closely aligned with the trend line for both training and testing datasets, reflecting their high accuracy and consistency. For RFR and SVR, the scatter of data points deviates more from the trend line, especially in the testing phase, highlighting their lower predictive accuracy. These outcomes under score the suitability of ANN for biodiesel yield prediction, given its exceptional accuracy, generalization ability, and minimal error.

## 3.3.2. Hyperparameters optimization and tuning of machine learning models

The soft sensor models—ANN, RFR, and SVR—were optimized using grid search cross-validation to enhance predictive accuracy, with RMSE on the validation dataset as the primary evaluation metric. The final performance of each model was assessed using the remaining 30 % of the testing data to evaluate generalizability. A k-fold cross-validation approach ensured robust evaluation across multiple data partitions, helping identify the optimal hyperparameters for each model.

Table 3 outlines the key hyperparameters fine-tuned for the models. For the ANN, hyperparameters such as network type, training function, hidden layers, learning rate, neurons, and activation function were optimized. A feedforward back-propagation architecture with the Levenberg-Marquardt algorithm was used, and a learning rate of 0.001 was chosen to balance convergence speed and stability, reducing the risk of overshooting the optimum. The ANN model included one hidden layer with 21 neurons and a tangent-sigmoid activation function to handle the nonlinearity of the yield prediction problem, aligning with practices from Sultana et al. [20].

The RFR model was optimized for the number of trees, maximum depth, and minimum leaf size. The model utilized 350 trees to capture the complexity of biodiesel yield prediction without overfitting. A maximum depth of 100 was set to control tree growth, ensuring generalization. The minimum leaf size of 1 enabled learning detailed patterns while managing overfitting through ensemble averaging. For the SVR model, hyperparameters such as box constraint, kernel function, kernel scale, and epsilon were tuned. A box constraint of 1000 was set to balance error penalization and generalization. A Gaussian kernel function was used to capture nonlinear relationships, with a kernel scale of 10 to influence the model’s sensitivity to data points. An epsilon value of 0.0654 was chosen to define the margin of tolerance for prediction errors, allowing the model to accurately approximate biodiesel yield.

Table 4 presents the performance indicator values for the training phase of each model. Each metric demonstrated significant improvement after tuning the models, highlighting the importance of hyperparameter optimization in accuracy enhancement. Moreover, minimal variations were observed between training and testing, indicating the absence of overfitting or underfitting.

Fig. 10 provides a visual comparison of the predictive performance of each model before and after hyperparameter tuning, plotted against the unseen testing dataset. Each graph presents the actual and predicted biodiesel yields, allowing evaluation of the model’s predictive capability. For the ANN model (see Fig. 10a and b), the plot indicates a significant improvement in capturing the trends in biodiesel yield after hyperparameter tuning, with a reduction in RMSE from 0.356 to 0.182, demonstrating improved alignment between predicted and actual yields. This improvement highlights the ANN model’s ability to generalize well and reduce error after tuning. For the RFR model (Fig. 10c and d), the RMSE decreased from 1.957 to 1.084 after hyperparameter tuning, indicating enhanced prediction performance on unseen data. The optimized RFR model displayed a better fit to the actual yield values, with predictions more closely following the trends of the real data. Similarly, the SVR model (Fig. 10e and f) showed a noticeable reduction in RMSE from 3.801 to 1.906 after tuning, with an improved match between predicted and actual yields, indicating that the optimized SVR model better captured the underlying patterns in biodiesel yield data.

The model comparison shows that hyperparameter tuning significantly enhances the predictive performance of each model. The ANN model, in particular, achieved the highest accuracy, as indicated by the highest $\mathrm { R } ^ { 2 ^ { \circ } }$ value and the lowest RMSE and MAE among the three models. These results highlight the ANN model’s superiority in predicting biodiesel yield, making it the best choice for soft sensor development. This confirms that precise hyperparameter selection and tuning are essential for improving machine learning performance in complex regression tasks like biodiesel yield prediction.

## 3.3.3. Feature importance analysis

The feature importance score quantifies the contribution of each feature to the overall predictive performance of a machine learning model, encompassing both the linear and nonlinear dependencies of biodiesel yield on these features. This evaluation was conducted using the best-performing ANN model, employing the plot importance method. By ranking the relative importance of input variables, this analysis illustrates how each feature impacts biodiesel yield prediction, capturing the intricate relationships identified by the ANN model. Fig. 11 presents the relative importance of input variables in biodiesel yield prediction during training. The analysis discerned the water con tent feature as the most influential, contributing to 28 % of the variations in biodiesel yield. Subsequently, the type of feedstock emerged as the second most influential factor, with a feature importance score of 22 %. This aligns with the findings of Sukpancharoen et al. (2023) [31], underscoring the substantial impact of feedstock composition on yield prediction in comparison to other reaction conditions. Notably, the use of soybean and waste cooking oil as feedstocks introduced significant variations in process conditions.

Table 4  
Comparison of performance matrices (training) before and after tuning of the ML models.

<table><tr><td rowspan="2">Model</td><td colspan="3">ANN</td><td colspan="3">RFR</td><td colspan="3">SVR</td></tr><tr><td>RMSE</td><td>MAE</td><td> $R^2$ </td><td>RMSE</td><td>MAE</td><td> $R^2$ </td><td>RMSE</td><td>MAE</td><td> $R^2$ </td></tr><tr><td>Without tuning</td><td>0.303</td><td>0.136</td><td>0.998</td><td>1.52</td><td>0.949</td><td>0.96</td><td>2.77</td><td>1.039</td><td>0.887</td></tr><tr><td>After tuning</td><td>0.112</td><td>0.052</td><td>0.999</td><td>0.504</td><td>0.307</td><td>0.995</td><td>1.17</td><td>0.63</td><td>0.976</td></tr></table>

![](images/3ee6365a85b148c45fadfd3d1d450654cc2fca5161a90f2efedd225731356525.jpg)

![](images/c741fe8c8b9426f27ece141fd373b077478ad55d022e553a88e3617889fc9efc.jpg)

![](images/3f9e8a9b9ccf29f1a4cf419b1b0f22146ebd0882655b120cae1a13b0fe3c357f.jpg)

![](images/a66a16d92433d85a5817b7b7c5259b599263b1364dee90d65f10799e1d01fc15.jpg)

![](images/e1ce284dd53675eb9bf7cf6ee5f48e37f597378e04dbfb009342dfbaa8c4aa21.jpg)

![](images/f1a72d00cf51984d86008c56b59f5fd2aa78e5b537991e20bb90e600581fd4bd.jpg)  
Fig. 10. Models predictive performance on unseen testing data before and after hyperparameter optimization.

![](images/db9f746d83b38c2fd1b95fc1ebf542f92ce96b913221f95c736bb82af321aa58.jpg)  
Fig. 11. Feature importance score derived from ANN model in the prediction of biodiesel yield.

Additionally, the reaction temperature exhibited the third-greatest influence on biodiesel yield prediction, indicated by a feature impor tance score of 21 %. The contribution of reaction time to the model’ impact on biodiesel prediction was determined to be 14 %. In contrast, feature importance scores for other factors, including catalyst type, methanol to oil ratio, and FFA content, indicated relatively lower im pacts on the predictive capability of the machine learning model. This implies that the model can maintain accuracy even with the removal or introduction of noise or errors in these features. Identifying the most influential features enables the soft sensor model to make more focused and efficient predictions. This targeted approach allows for prioritizing critical parameters in process control, thereby directly enhancing the operational efficacy and economic feasibility of biodiesel production.

![](images/fa7cb39810a73b7ff3a9c3b1aa0939a30e7edfdfbc1769c412ab7d4e6909e9c1.jpg)

![](images/87bdb6b2d0bc2e16c1305dffe57cd957cd14cf4425eb0efb88f9a8e24ef9df3a.jpg)  
Fig. 12. Model predictions for extra literature data with the corresponding actual and predicted biodiesel yield.

## 3.4. Model validation and generalizability check

To further evaluate the predictive performance of the ANN-based soft sensor model, an additional set of 30 data points was incorporated into the analysis. These data points, obtained from a distinct source under separate experimental conditions [20], were explicitly excluded from the model’s training dataset. The dataset used for validation is based on microalgae oil and an acid catalyst. Despite this specific context, the reaction follows the same transesterification pathway and reaction conditions as those observed with chemical catalysts. This similarity enables a more thorough assessment of the model’s generalizability and its capacity to handle unseen data effectively. The assessment of the model’s prediction ability was conducted using the unseen literature data and illustrated in Fig. 12. Fig. 12a dis plays a scatter plot with a fitted line, presenting the literature experimental data versus the predicted data. Additionally, Fig. 12b illustrating the actual and predicted biodiesel yield for the literature dataset is provided. The $\bar { \mathsf { R } ^ { 2 } }$ was computed as 0.828, indicating a high level of predictive capability for the proposed ANN model. Furthermore, the MAE and RMSE were determined to be 1.361 and 1.515, respectively. These values are deemed acceptable within the context of data acquired under diverse experimental conditions.

The outcomes strongly suggest that the developed ANN model ex hibits a notable capacity to predict unseen data, showcasing robust generalizability beyond the conditions encountered during its training. This validation emphasizes the reliability and robustness of the ANN model, affirming its ability to provide accurate predictions across new and diverse datasets.

## 4. Conclusion

This study presents the development of a hybrid soft sensor model for biodiesel yield prediction, effectively addressing the limitations of traditional measurement techniques and purely data-driven soft sensors. By integrating mechanistic models with data-driven approaches, the model enables computational data generation using MATLAB®, signif icantly minimizing the need for costly and time-intensive laboratory experiments.

A comprehensive dataset was generated, consisting of seven input variables—catalyst type, feedstock type, temperature, reaction time, FFA content, water content, and methanol-to-oil ratio—and one output variable, biodiesel yield. This dataset was used to train various machine learning algorithms, including ANN, RFR, and SVR. Among these, the ANN model demonstrated the highest predictive accuracy, achieving an $\mathrm { R } ^ { 2 }$ of 0.998 and an RMSE of 0.303. Other algorithms like RFR and SVR were evaluated for comparison, but the ANN consistently outperformed them. Hyperparameter tuning further enhanced the ANN model’s performance, reducing RMSE and MAE by 63 % and 61.7 %, respectively, confirming its robustness and reliability. The hybrid model also showed strong generalizability, achieving an RMSE of 0.182 when tested on unseen data from various feedstocks and catalysts, validating its capability for accurate real-time yield predictions. Feature importance analysis indicated that feedstock type, water content, reaction time, and temperature accounted for 85 % of the impact score on biodiesel yield, emphasizing their critical role in predictive modeling.

The proposed model mitigates the laborious, expensive, and timeconsuming nature of traditional laboratory trials needed to generate experimental or historical data for data-driven models, offering a costeffective solution. This study lays the groundwork for future research on machine learning-based methods for monitoring and controlling biodiesel production systems, providing an efficient and scalable alternative to traditional analytical techniques. Future research should focus on expanding the dataset to include a broader range of feedstock sources and catalyst types, aiming to develop a more generalized model with enhanced predictive capabilities, ensuring broader applicability and reliability across various biodiesel production processes.

## CRediT authorship contribution statement

Mustafa Kamal Pasha: Writing – original draft, Visualization, Methodology, Investigation. Lingmei Dai: Methodology. Dehua Liu: Project administration. Wei Du: Writing – review & editing, Supervision, Project administration. Miao Guo: Writing – review & editing, Supervision, Project administration.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Appendix A. Supplementary data

Supplementary data to this article can be found online at https://doi. org/10.1016/j.renene.2024.121888.

## References

[1] L. Rocha-Meneses, A. Hari, A. Inayat, L.A. Yousef, S. Alarab, M. Abdallah, A. Shanableh, C. Ghenai, S. Shanmugam, T. Kikas, Recent advances on biodiese production from waste cooking oil (WCO): a review of reactors, catalysts, and optimization techniques impacting the production, Fuel 348 (2023), https://doi.

[2] H. Waudby, S.H. Zein, A circular economy approach for industrial scale biodiese production from palm oil mill effluent using microwave heating: design,

simulation, techno-economic analysis and location comparison, Process Saf Environ. Protect. 148 (2021) 1006–1018, https://doi.org/10.1016/j. psep.2021.02.011.

[3] S. Lee, D. Posarac, N. Ellis, Process simulation and economic analysis of biodiesel production processes using fresh and waste vegetable oil and supercritical methanol, Chem. Eng. Res. Des. 89 (2011) 2626–2642, https://doi.org/10.1016/j. cherd.2011.05.011.

[4] Y. Zhang, M. Dub´e, D. McLean, M.K.-B. technology, undefined, Biodiesel production from waste cooking oil: 2. Economic assessment and sensitivity analysis, ElsevierY Zhang, MA Dub´e, DD McLean, M KatesBioresource Technology, 2003•Elsevier (2003), https://doi.org/10.1016/S0960-8524(03)00150-0 (n.d.).

[5] A.H. West, D. Posarac, N. Ellis, Assessment of four biodiesel production processes using HYSYS, Plant, Bioresour Technol 99 (2008) 6587–6601, https://doi.org/ 10.1016/i,biortech.2007.11.046.

[6] L.F. Sotoft, B.G. Rong, K.V. Christensen, B. Norddahl, Process simulation and economical evaluation of enzymatic biodiesel production plant, Bioresour. Technol. 101 (2010) 5266–5274, https://doi.org/10.1016/j.biortech.2010.01.130.

[7] M.K. Pasha, L. Dai, D. Liu, W. Du, M. Guo, Biodiesel production with enzymatic technology: progress and perspectives, Biofuels, Bioproducts and Biorefining 15 (2021) 1526–1548, https://doi.org/10.1002/bbb.2236.

[8] M.K. Pasha, L. Dai, D. Liu, M. Guo, W. Du, An overview to process design, simulation and sustainability evaluation of biodiesel production, Biotechnol Biofuels 14 (2021), https://doi.org/10.1186/s13068-021-01977-z.

[9] G. Ma, L. Dai, D. Liu, W. Du, A robust two-step process for the efficient conversion of acidic soybean oil for biodiesel production, Catalysts 8 (2018), https://doi.org/ 10.3390/catal8110527.

[10] M. Kamal Pasha, M. Rahim, L. Dai, D. Liu, W. Du, M. Guo, Comparative study of a two-step enzymatic process and conventional chemical methods for biodiesel production: economic and environmental perspectives, Chem. Eng. J. 489 (2024), https://doi.org/10.1016/j.cej.2024.151254

[11] M.Y. Liow, W. Gourich, M.Y. Chang, J.M. Loh, E.S. Chan, C.P. Song, Towards rapid and sustainable synthesis of biodiesel: a review of effective parameters and scaleup potential of intensification technologies for enzymatic biodiesel production, J. Ind. Eng. Chem. 114 (2022) 1–18, https://doi.org/10.1016/j.jiec.2022.07.002.

[12] J.Z. Chen, S. Wang, B. Zhou, L. Dai, D. Liu, W. Du, A robust process for lipasemediated biodiesel production from microalgae lipid, RSC Adv. 6 (2016) 48515–48522, https://doi.org/10.1039/c6ra07144a.

[13] T.A. Andrade, M. Errico, K.V. Christensen, Evaluation of reaction mechanisms and kinetic parameters for the transesterification of Castor oil by liquid enzymes, Ind. Eng. Chem. Res. 56 (2017) 9478–9488, https://doi.org/10.1021/acs.iecr.7b02285.

[14] J.H.C. Wancura, M. Brondani, M.S.N. dos Santos, C.E.D. Oro, G.C. Wancura, M. V. Tres, J.V. Oliveira, Demystifying the enzymatic biodiesel: how lipases are contributing to its technological advances. Renew. Energy 216 (2023). https://doi org/10.1016/i.renene.2023.119085.

[15] D. Ly. W. Du. G. Zhang. D. Liu. Mechanism study on NS81006-mediated methanolysis of triglyceride in oil/water biphasic system for biodiesel production, Process Biochem. 45 (2010) 446–450, https://doi.org/10.1016/j. procbio.2009.10.017.

[16] J. Zhou, T. Shi, Q. Qian, C. He, J. Ren, Protocol for the design and accelerated optimization of a waste-to-energy system using AI tools, STAR Protoc 4 (2023), https://doi.org/10.1016/j.xpro.2023.102685.

[17] M. Aghbashlo, W. Peng, M. Tabatabaei, S.A. Kalogirou, S. Soltanian, H. Hosseinzadeh-Bandbafha, O. Mahian, S.S. Lam, Machine learning technology in biodiesel research: a review, Prog. Energy Combust. Sci. 85 (2021), https://doi. org/10.1016/i.pecs.2021.100904

[18] P. Nkulikivinka, Y. Yan, F. Gülec, V. Manovic, P.T. Clough, Prediction of sorption enhanced steam methane reforming products from machine learning based softsensor models, Energy and AI 2 (2020), https://doi.org/10.1016/j. egyai.2020.100037.

[19] P. Kadlec, B. Gabrys, S. Strandt, Data-driven soft sensors in the process industry, Comput. Chem. Eng. 33 (2009) 795–814, https://doi.org/10.1016/j. compchemeng.2008.12.012.

[20] N. Sultana, S.M.Z. Hossain, M. Abusaad, N. Alanbar, Y. Senan, S.A. Razzak, Prediction of biodiesel production from microalgal oil using Bayesian optimization algorithm-based machine learning approaches, Fuel 309 (2022), https://doi.org 10.1016/i.fuel2021.122184

[21] P. Agrawal, R. Gnanaprakash, S.H. Dhawane, Prediction of biodiesel yield emploving machine learning: interpretability analysis via shapley additive explanations, Fuel 359 (2024), https://doi.org/10.1016/j.fuel.2023.130516.

[22] W.K. Abdelbasset, S.M. Elkholi, M. Jade Catalan Opulencia, T. Diana, C.H. Su, M. Alashwal, M. Zwawi, M. Algarni, A. Abdelrahman, H. Chinh Nguyen, Development of multiple machine-learning computational techniques for optimization of heterogenous catalytic biodiesel production from waste vegetable oil: development of multiple machine-learning computational techniques for optimization, Arab. J. Chem. 15 (2022), https://doi.org/10.1016/j. arabic.2022.103843.

[23] H. Moayedi, B. Aghel, L.K. Foong, D.T. Bui, Feature validity during machine learning paradigms for predicting biodiesel purity, Fuel 262 (2020), https://doi. org/10.1016/i.fuel,2019.116498

[24] Z. Cui, S. Huang, M. Wang, K. Nie, Y. Fang, T. Tan, Improving the CFPP property of biodiesel via composition design: an intelligent raw material selection strategy

based on different machine learning algorithms, Renew. Energy 170 (2021) 354–363, https://doi.org/10.1016/j.renene.2021.02.008.

[25] M. Suvarna, M.I. Jahirul, W.H. Aaron-Yeap, C.V. Augustine, A. Umesh, M.G. Rasul, M.E. Günay, R. Yildirim, J. Janaun, Predicting biodiesel properties and its optimal fatty acid profile via explainable machine learning, Renew. Energy 189 (2022) 245–258, https://doi.org/10.1016/j.renene.2022.02.124.

[26] Y. Ying, P. Shao, S. Jiang, P. Sun, Artificial Neural Network Analysis of Immobilized Lipase Catalyzed Synthesis of Biodiesel from Rapeseed Soapstock., n. d.

[27] B. Siritanaratkul, Generalizability and limitations of machine learning for yield prediction of oxidative coupling of methane, Digital Chemical Engineering 2 (2022), https://doi.org/10.1016/j.dche.2022.100013.

[28] I. Ahmad, A. Ayub, U. Ibrahim, M.K. Khattak, M. Kano, Data-based sensing and stochastic analysis of biodiesel production process, Energies 12 (2019), https://doi. org/10.3390/en12010063.

[29] M.A.G. Nasim, O. Khan, M. Parvez, B.K. Bhatt, Optimizing ultrasonic reactor operating variables using intelligent soft computing models for increased biodiesel production, Green Technologies and Sustainability 1 (2023) 100033, https://doi. org/10.1016/i.grets.2023.100033.

[30] T.J. Rato, D.M.G. Neves, A. Antunes, M.S. Reis, A systematic PAT Soft Senso screening and development methodology applied to the prediction of free fatt acids in industrial biodiesel production, Fuel 282 (2020), https://doi.org/10.1016 j.fuel.2020.118800.

[31] S. Sukpancharoen, T. Katongtung, N. Rattanachoung, N. Tippayawong, Unlocking the potential of transesterification catalysts for biodiesel production through machine learning approach, Bioresour. Technol. 378 (2023), https://doi.org/ 10.1016/j.biortech.2023.128961.

[32] Y. Li, W. Du, L. Dai, D. Liu, Kinetic study on free lipase NS81006-catalyzed biodiesel production from soybean oil, J. Mol. Catal. B Enzym. 121 (2015) 22–27, https://doi.org/10.1016/j.molcatb.2015.07.013.

[33] A.M. Alsahly, E.M. Elkanzi, S.M.Z. Hossain, Lipase-catalyzed production of biodiesel: process simulation and economic analysis. Journal of Scientific Research and Studies 5 (2018) 142–152

[34] E. Sendzikiene, V. Makareviciene, P. Janulis, S. Kitrys, Kinetics of free fatty acid esterification with methanol in the production of biodiesel fuel, Eur. J. Lipid Sci. Technol, 106 (2004) 831–836, https://doi,org/10.1002/eilt.200401011

[35] M.S. Alhajeri, F. Abdullah, Z. Wu, P.D. Christofides, Physics-informed machine learning modeling for predictive control using noisy data, Chem. Eng. Res. Des. 186 (2022) 34–49, https://doi.org/10.1016/j.cherd.2022.07.035.

[36] A.I. Almohana, S.F. Almojil, M.A. Kamal, A.F. Alali, M. Kamal, S.E. Alkhatib, B F. Felemban, M. Algarni, Theoretical investigation on optimization of biodiesel production using waste cooking oil: machine learning modeling and experimental validation, Energy Rep. 8 (2022) 11938–11951, https://doi.org/10.1016/j. egyr.2022.08.265.

[37] S.D. Shelare. P.N. Belkhode. K.C. Nikam. LD. Jathar. K. Shahapurkar, M.E. M. Soudagar, I. Veza, T.M.Y. Khan, M.A. Kalam, A.S. Nizami, M. Rehan, Biofuels for a sustainable future: examining the role of nano-additives, economics, policy, internet of things, artificial intelligence and machine learning technology in biodiesel production, Energy 282 (2023), https://doi.org/10.1016/j. energy.2023.128874.

[38] M. Hajar, F. Vahabzadeh, Artificial neural network modeling of biolubrican production using Novozym 435 and castor oil substrate, Ind. Crops Prod. 52 (2014) 430–438, https://doi.org/10.1016/j.indcrop.2013.11.020.

[39] H.Y. Shyu, C.J. Castro, R.A. Bair, Q. Lu, D.H. Yeh, Development of a soft sensor using machine learning algorithms for predicting the water quality of an onsite wastewater treatment system, ACS Environmental Au 3 (2023) 308–318, https:// doi.org/10.1021/acsenvironau.2c00072.

[40] M.-R. Pourramezan, A. Rohani, M.H. Abbaspour-Fard, Comparative analysis of soft computing models for predicting viscosity in diesel engine lubricants: an alternative approach to condition monitoring, ACS Omega (2023), https://doi.org 10.1021/acsomega.3c07780.

[41] O. Khan, M.Z. Khan, M.T. Alam, A. Ullah, M. Abbas, C.A. Saleel, S. Shaik, A. Afzal, Comparative study of soft computing and metaheuristic models in developing reduced exhaust emission characteristics for diesel engine fueled with various blends of biodiesel and metallic nanoadditive mixtures: an ANFIS-GA-HSA approach, ACS Omega 8 (2023) 7344–7367, https://doi.org/10.1021/ acsomega.2c05246.

[42] T. Hikosaka, S. Aoshima, T. Miyao, K. Funatsu, Soft sensor modeling for identifying significant process variables with time delays, Ind. Eng. Chem. Res. 59 (2020) 12156–12163, https://doi.org/10.1021/acs.iecr.0c01655

[43] W. Zhang, S.P. Wei, M.K.H. Winkler, A.V. Mueller, Design of a soft sensor for monitoring phosphorous uptake in an EBPR process, ACS ES and T Engineering 2 (2022) 1847–1856, https://doi.org/10.1021/acsestengg.2c00090.

[44] S. Morais, T.M. Mata, A.A. Martins, G.A. Pinto, C.A.V. Costa, Simulation and life cycle assessment of process design alternatives for biodiesel production from waste vegetable oils, J. Clean. Prod. 18 (2010) 1251–1259, https://doi.org/10.1016/j. iclepro.2010.04.014.

[45] I.A. Penarrubia˜ Fernandez, D.H. Liu, J. Zhao, LCA studies comparing alkaline and immobilized enzyme catalyst processes for biodiesel production under Braziliar conditions, Resour, Consery, Recycl, 119 (2017) 117–127, https://doi,org 10.1016/i.resconrec.2016.05.009.