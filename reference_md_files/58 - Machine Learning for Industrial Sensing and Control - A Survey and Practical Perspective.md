# Machine learning for industrial sensing and control: A survey and practical perspective<sup>⋆</sup>

Nathan P. Lawrence<sup>a</sup>, Seshu Kumar Damarla<sup>c</sup>, Jong Woo Kim<sup>f</sup>, Aditya Tulsyan<sup>c</sup>, Faraz Amjad<sup>c</sup>, Kai Wang<sup>g</sup>, Benoit Chachuat<sup>d</sup>, Jong Min Lee<sup>e</sup>, Biao Huang<sup>c</sup>, R. Bhushan Gopaluni<sup>b</sup>

<sup>a</sup>Department of Mathematics, University of British Columbia, Canada <sup>b</sup>Department of Chemical and Biological Engineering, University of British Columbia, Canada

<sup>c</sup>Department of Chemical and Materials Engineering, University of Alberta, Canada <sup>d</sup>The Sargent Centre for Process Systems Engineering, Department of Chemical Engineering, Imperial College London, London SW7 2AZ, UK

<sup>e</sup>School of Chemical and Biological Engineering, Institute of Chemical Processes, Seoul National University, 1, Gwanak-ro, Gwanak-gu, Seoul 08826, Republic of Korea <sup>f</sup>Department of Energy and Chemical Engineering, Incheon National University, Incheon 22012, Republic of Korea

<sup>g</sup>School of Automation, Central South University, Changsha, 410083, China

## Abstract

With the rise of deep learning, there has been renewed interest within the process industries to utilize data on large-scale nonlinear sensing and control problems. We identify key statistical and machine learning techniques that have seen practical success in the process industries. To do so, we start with hybrid modeling to provide a methodological framework underlying core application areas: soft sensing, process optimization, and control. Soft sensing contains a wealth of industrial applications of statistical and machine learning methods. We quantitatively identify research trends, allowing insight into the most successful techniques in practice. We consider two distinct flavors for data-driven optimization and control: hybrid modeling in conjunction with mathematical programming techniques and reinforcement learning. Throughout these application areas, we discuss their respective industrial requirements and challenges. A common challenge is the interpretability and eficiency of purely data-driven methods. This suggests a need to carefully balance deep learning techniques with domain knowledge. As a result, we highlight ways prior knowledge may be integrated into industrial machine learning applications. The treatment of methods, problems, and applications presented here is poised to inform and inspire practitioners and researchers to develop impactful data-driven sensing, optimization, and control solutions in the process industries.

Keywords: statistical machine learning, deep learning, hybrid modeling, soft sensing, reinforcement learning, control

## 1. Motivation

Data analytics and machine learning (ML) ideas are not new to the process industries<sup>1</sup>. The review paper by Venkatasubramanian [1] provides an excellent overview of the history, successes, and failures of various attempts over more than three decades to use ideas from artificial intelligence (AI) in the industry. In particular, statistical techniques such as principal component analysis, partial least squares, canonical correlation analysis, and time series methods for modeling, such as maximum likelihood estimation and prediction error methods, have been extensively used in industry [2]. Several classification and clustering algorithms, such as k-means, support vector machines, and Fisher discriminant analysis, are also widely used in industry [3, 4]. And several nonlinear approaches, such as kernel methods, Gaussian processes, and adaptive control algorithms, such as reinforcement learning, have been applied in some niche applications [5, 6, 7].

Despite the longstanding success of many statistical techniques in industry, there is also considerable interest in developing sensing and control technologies based on more recent ML architectures [1, 8, 9]. Broadly speaking, these aspirations are driven by the promises of increased autonomy: increased operational eficiency, consistency, and safety; improved scalability beyond linear methods; upskilling of plant personnel [10]. Consequently, this paper addresses the need to dissect and organize the general use of modern ML techniques in industrial applications. In doing so, such a treatment will inform practitioners of the latest research trends and their potential practical impact. Conversely, researchers in core areas will benefit from a holistic view of successful ML techniques and the industrial requirements they satisfy.

## 1.1. Overview and scope

This paper is a significant extension of Gopaluni et al. [11]: in addition to a more detailed and expansive treatment of the literature, we discuss the practical success of various methods. Note that this is primarily a problemdriven survey; however, we have provided suficient references for interested readers on the underlying methods discussed here. Moreover, we have included additional exposition on some of these methods in the supplementary material.

Hybrid modeling is first introduced to provide a conceptual framework underlying core application areas, namely:<sup>2</sup>

## 1. Soft sensing

## 2. Process control

Process control also includes process optimization. In our survey, we identify several methodological areas of research: statistical learning and machine learning, deep learning and its variants, and reinforcement learning. Algorithms from each of these methodological areas are used to varying degrees among the core applications. Soft sensing encompasses more statistical and machine learning methods, with some discussion of deep learning. On the optimization and control side, we discuss hybrid modeling in tandem with mathematical programming and reinforcement learning.

This is by no means an exhaustive survey of the recent research on these topics. However, we have tried our best to include some of the most critical developments of ML tools in the process industries. In that vein, we only discuss methods that have seen industrial use or have received considerable research attention within process systems engineering, either in real life or in simulations. Therefore, speculation about the potential use of very recent developments in the broader ML community, such as ChatGPT or other large language models, is beyond the scope of this paper. However, we provide insight into the practical deployment of ML techniques in the process industries.

This paper surveys a large number of algorithms. Table 1 gives a convenient list to reference across all sections. Throughout this paper, artificial intelligence is the broadest term for classifying machines that aim to mimic human intelligence. It is intended to predict, automate, and optimize the tasks humans have traditionally performed, such as speech recognition, image recognition, decision-making, and translation. Machine learning is an area of artificial intelligence and computer science where algorithms are developed to extract patterns from data and make predictions. Supervised learning is a branch of machine learning comprised of algorithms for determining a predictive model based on labeled data with known outcomes. On the other hand, unsupervised learning is a branch of machine learning devoted to learning patterns from unlabeled data.

## 2. Mathematical modeling approaches

The core applications of this paper are soft sensing and process optimization and control. These areas rely on dynamic mathematical models to infer measurements, make decisions, and synthesize controllers. Therefore, before describing the prominent machine learning (ML) techniques in these areas, it is useful to introduce the foundational assumptions and architectures underlying such models.

## 2.1. Knowledge-driven, data-driven, and hybrid modeling

Knowledge-driven (mechanistic or white box) modeling based on first principles and data-driven (or black box) modeling constitute two opposite strategies. Developing mechanistic models requires a deep understanding of the processes at play. It is often labor-intensive, but embodying first principles may enable extrapolation beyond the conditions under which these models are trained. By construction, mechanistic models have a fixed structure and comprise a fixed number of parameters, often with a physical or empirical interpretation. For this reason, they may also be classified as parametric models.

By contrast, data-driven models require little physical knowledge and are fast to deploy or maintain. But a larger dataset is also typically needed for their construction, and their validity may not extend far beyond the conditions under which they are trained. The structure of a data-driven model does not need to be dictated by a priori knowledge but may be tailored to the training data at hand instead. A further distinction is whether a data-driven model tries to describe data with a set of parameters of fixed size, regardless of the size of the training dataset, in which case it is categorized as parametric, or whether its structure and number of parameters may evolve with the size of the dataset, commonly referred to as nonparametric [13, 14]. The socalled nonparametric regression models fall in the second category, whereby the predictor does not take a predetermined form, using techniques such as nearest-neighbor interpolation, local regression, and Gaussian process (GP) regression. However, the distinction between parametric and nonparametric models in statistical and ML is not without intricacies. For instance, a linear SVM is a typical example of a parametric model, having a fixed number of parameters—a weight for each input dimension. In contrast, RBF-kernel SVM may be considered nonparametric since the number of parameters grows with the size of the training set—a weight for each training point.

Table 1: Full forms for acronyms. Divided into three sections, top to bottom: 1) statistical learning, 2) machine learning & deep learning, and 3) reinforcement learning & control methods.

<table><tr><td>CCA</td><td>Canonical correlation analysis</td><td>LASSO</td><td>Least absolute shrinkage and selection operator</td></tr><tr><td>FA</td><td>Factorial analysis</td><td>LR</td><td>Logistic regression</td></tr><tr><td>GMM</td><td>Gaussian mixture model</td><td>PCA</td><td>Principal component analysis</td></tr><tr><td>ICA</td><td>Independent component analysis</td><td>PLS</td><td>Partial least squares</td></tr><tr><td>LARS</td><td>Least-angle regression</td><td>RBC</td><td>Reconstruction-based contribution</td></tr><tr><td>ANFIS</td><td>Adaptive network fuzzy inference system</td><td>GRNN</td><td>General regression neural network</td></tr><tr><td>ANN</td><td>Artificial neural network</td><td>MLP</td><td>Multilayer perceptron</td></tr><tr><td>BN</td><td>Bayesian network</td><td>RBFNN</td><td>Radial basis function neural network</td></tr><tr><td>CNN</td><td>Convolutional neural network</td><td>RNN</td><td>Recurrent neural network</td></tr><tr><td>DNNE</td><td>Decorrelated neural network ensemble</td><td>RT</td><td>Regression tree</td></tr><tr><td>DNN</td><td>Deep neural network</td><td>RVM</td><td>Relevance vector machine</td></tr><tr><td>ELM</td><td>Extreme learning machine</td><td>SFA</td><td>Slow feature analysis</td></tr><tr><td>ESN</td><td>Echo state network</td><td>SVM</td><td>Support vector machine</td></tr><tr><td>ENN</td><td>Elman neural network</td><td>TL</td><td>Transfer learning</td></tr><tr><td>GPR</td><td>Gaussian process regression</td><td>WNN</td><td>Wavelet neural network</td></tr><tr><td>A3C</td><td>Asynchronous advantage actor-critic</td><td> $PI^2$ </td><td>Policy improvement with path integrals</td></tr><tr><td>ADP</td><td>Approximate dynamic programming</td><td>PID</td><td>Proportional-integral-derivative</td></tr><tr><td>DDPG</td><td>Deep deterministic policy gradient</td><td>PPO</td><td>Proximal policy optimization</td></tr><tr><td>DQN</td><td>Deep Q-network</td><td>RTO</td><td>Real-time optimization</td></tr><tr><td>HJB</td><td>Hamilton-Jacobi-Bellman</td><td>SAC</td><td>Soft actor-critic</td></tr><tr><td>MPC</td><td>Model predictive control</td><td>TD3</td><td>Twin-delayed DDPG</td></tr></table>

The basic idea behind hybrid models is to combine knowledge-driven and data-driven models in such a way as to overcome their respective limitations. This strategy is also frequently referred to as gray box or block-oriented modeling in the literature. At the same time, the term hybrid semi-parametric modeling is coined to describe those hybrid models where the data-driven component is nonparametric [15]. Multi-fidelity modeling has also developed fast in recent years and is akin to hybrid modeling. The idea is to use a (possibly inaccurate) knowledge-driven model as low-fidelity and correct it with (noisy) process data, considered to be higher fidelity [16]. In particular, this strategy has been applied in uncertainty propagation, inference, and optimization and is also instrumental in small data problems (see supplementary material).

It is worth noting that hybrid modeling has been investigated for over 25 years in chemical and biological process engineering [17, 18, 19, 20]. The claimed benefits of hybrid modeling in these application domains include faster prediction capability, better extrapolation capability, better calibration properties, easier model life-cycle management, and higher benefit/cost ratio to solve complex problems; see recent survey papers on the development and applications of hybrid models by von Stosch et al. [15], Solle et al. [21], Schuppert and Mrziglod [22], Zendehboudi et al. [23], Ahmad et al. [24], Bradley et al. [25]. Hybrid models may be used to enable soft sensors (see Section 3) or model-based optimization and control (see Section 4) in a first principles approach.

![](images/0f21e4377f6178a3d32eec6049d147e96ba9100f3c211d3dc130ee72e5d475a4.jpg)  
Figure 1: Typology of hybrid models (see von Stosch et al. [15]). A and C represent serial structures: under A, a data-driven model is used as input to a knowledge-driven model; C is the reverse. B represents a parallel structure in which knowledge-driven predictions are corrected by data-driven predictions.

## 2.2. Hybrid modeling paradigms

## 2.2.1. Traditional serial and parallel hybrid models

The usual classification of hybrid model structures is either as serial or parallel [26]. In the serial approach, the data-driven model is most commonly used as an input to the mechanistic model (see Figure 1A), for instance, a material balance equation with a kinetic rate expressed using a data-driven model. This structure is especially suited to situations where precise knowledge about specific underlying mechanisms is lacking, yet suficient process data exists to infer the corresponding relationship [17, 20]. However, when the mechanistic part of the model presents a structural mismatch, one should not expect the serial approach to perform better than a purely mechanistic approach. In the parallel approach, by contrast, the output of the data-driven model is used to correct the predictions of the mechanistic model [18, 19], most often in the form of an additive correction (see Figure 1B). This structure can significantly improve the prediction accuracy of a mechanistic model when the data-driven component is trained on the residuals between process observations and mechanistic model predictions. However, this accuracy may not be better than the sole mechanistic model when the process conditions difer drastically from those in the training set.

Historically, the most common data-driven modeling techniques embedded in hybrid models have been multilayer perceptron (MLP) and RBF-based regression [15]. Recent representative applications include the development of a serial hybrid model to predict hydraulic fractures created by injecting fluid into a reservoir that accounts for the leak-of rate of the fracturing fluid using an MLP [27] and the development of a serial hybrid model of the thin film growth process coupling a macroscopic gas phase model described by partial diferential equations to a microscopic thin-film model described by stochastic partial diferential equations via an MLP [28]. Naturally, many other statistical and ML techniques have also been investigated in this context. For instance, Ghosh et al. [29] used subspace identification to construct the data-driven component in a parallel hybrid model and demonstrated the approach on a batch polymerization reactor. Lopez et al. [30] developed a serial hybrid model of a lignocellulosic fermentation process, whereby the glucose concentration is estimated from spectroscopic data using a partial least squares regression model. GP regression has also attracted attention due to its ability to estimate the predictor’s variance, for example, in bioprocess engineering applications [31].

Parallel hybrid models can significantly alleviate the issue of maintaining a complex mechanistic model since the data-driven component is trained to capture model mismatch in the first place, possibly in a nonparametric manner. For dynamic systems in particular, a popular approach entails training the data-driven model on the residuals between the predicted and observed states at given time instants [32]. Notice that such a data-driven model could either comprise algebraic or diferential equations. By contrast, serial hybrid models can prove more challenging to design, especially when the outputs of the data-driven component cannot be observed directly [33]. In such a case, training and assessing the performance of the data-driven component requires one to simulate the full serial hybrid model and compare its outputs to the available observations. Identifying the unknown model parameters within such hybrid models has relied on regularized regression techniques, such as LASSO and LARS [34].

Another challenge shared by serial and parallel hybrid modeling paradigms is automatically detecting the best structure for the data-driven component. Generally speaking, minimizing the number of parameters needed to capture the underlying mechanisms is desirable, that is, to neither underfit nor overfit the data. Classical approaches to help discriminate among multiple nonparametric model structures include the Akaike Information Criterion and Bayesian Information Criteria. Willis and von Stosch [35] proposed an approach based on sparse regression and mixed-integer programming to simultaneously decide the structure and identify the parameters for a class of rational functions embedded into a serial hybrid model. Recently, Zhang et al. [36] applied hybrid modeling in combination with sparse identification of nonlinear dynamics [SINDy; 37] to a photo-production bioprocess, whereby a sparse quadratic correction of the kinetic model is identified using mixed-integer nonlinear programming techniques. More generally, there is significant scope for extending sparse and symbolic regression techniques to enable the construction of hybrid models. Notably, the platform ALAMO [38] can enforce constraints on the response variables to incorporate first principles knowledge, thereby revealing hidden relationships between regression parameters that may not be directly available to the modeler. One approach to incorporating such constraints is via semi-infinite programming [39]. Another promising direction entails using sum-of-squares optimization techniques to tackle this problem [40, 41].

## 2.2.2. Emerging trends

The traditional hybrid modeling approach has put a mechanistic model at its core. It uses data-driven elements to either describe specific unknown or poorly understood mechanisms or correct the predictions of the mechanistic model. Another way of incorporating domain knowledge and mechanistic models is feature engineering, where the inputs to the data-driven elements are augmented by terms that would also appear in mechanistic models; for instance, think of enthalpy, which is not a measurement but a useful term in energy balances. Hybrid models whereby the mechanistic model is now used as an input to the data-driven component have become increasingly popular in recent years (see Figure 1C). This approach includes physics-informed neural networks where the underlying conservation equations are imposed as extra constraints on the MLP’s parameters [42], like the classical orthogonal collocation theory on finite elements using piecewise polynomials [43, 44]. Co-Kriging techniques have also been developed where a GP trained using data from a mechanistic model is combined with a second GP trained using process data (or a high-fidelity model) [45]. Such an approach also enables multi-fidelity modeling using linear or nonlinear autoregressive techniques [46, 47] and deep GPs [48], and finding applications, for instance, in the optimization of complex black box simulators and legacy codes. Another body of research has been concerned with learning a dynamic system by accounting for prior information, for instance, the regression of polynomial dynamic systems with prior information using sum-of-squares optimization methods [49].

Since there is no universal framework, a recurring challenge with hybrid modeling is selecting the appropriate paradigm—for example, physics-driven against data-driven backbone, or serial against parallel structure—for a particular application, such as small vs. large datasets or noisy vs. high-quality data. This selection process still lacks a solid theoretical basis, although systematic computational comparisons of various hybridization techniques have emerged in recent years [25]. Finally, looking beyond current hybrid models, Venkatasubramanian [1] argued for the development of hybrid artificial intelligence systems that would combine not only mechanistic with data-driven models but also causal models-based explanatory systems or domain-specific knowledge engines. Likewise, the mechanistic model could be replaced by a graph-theoretical model, such as signed digraphs, or a production system model, creating entirely new research fields.

## 3. Soft sensors in process industries

Soft sensing represents the most fundamental application of machine learning (ML) techniques in the process industries. By extension, optimization and control add complexity to a soft sensing core. As a result, based on our analysis and own experience, soft sensing contains the most industrial penetration of ML applications. We quantitatively analyze which ML methods have seen practical success and which are currently being researched. We ofer practical considerations and insights for implementing soft sensors in practice to balance the apparent industrial-academic disconnect.

## 3.1. Motivation for soft sensing

In the process industries, some variables are dificult to measure online due to technological limitations or the high cost of sensors. These variables indicate a product’s intermediate or final quality and must be continuously monitored and controlled. In such circumstances, mathematical models are developed using easy-to-measure variables. These models provide a continuous estimate for quality variables in real time. The mathematical models devoted to the estimation of plant variables are called soft sensors [50, 51]. The process industries, such as refineries, steel plants, polymer industries, or cement industries, remain the dominant users of soft sensors (see Figure 2).

Similar to hybrid modeling, soft sensors can be categorized as knowledgedriven and data-driven. Knowledge-driven soft sensors (or white box models), such as Kalman filters, are based on first principles models that describe the physical and chemical laws that govern the process, such as mass and energy balance equations. In contrast, data-driven soft sensors (or black box models) have no information about the process and are based on empirical observations (historical process data). A third type of soft sensor, called hybrid models (or gray box models), uses a data-driven method to estimate the parameters of a knowledge-driven model. This special combination is closely related to the general concept of hybrid modeling, as discussed in Section 2.

![](images/febd71788c9affb111cdd44734069daa7d8ba99e47d9135122558826338de9d9.jpg)  
Figure 2: Distribution of soft sensor applications.

For instance, a model may incorporate physics-based simulations and process measurements.

## 3.2. A quantitative overview of soft sensing

Literature was collected by gathering articles published between 2015 and 2023 in relevant journals from publishing houses like Elsevier, Springer, Wiley, Taylor and Francis, MDPI, World Scientific, Hindawi, De Gruyter, AMSE, and IEEE. For the publication search, keywords such as “soft sensor”, “virtual sensor” or “inferential model” were used. The statistics shown in Figure 3 were computed based on the collected literature.

These statistics indicate that the research conducted in soft sensing between 2015 and 2023 was primarily focused on data-driven models. This is unsurprising, as data-driven soft sensors can often capture complex and unexplained process dynamics more succinctly. In contrast, knowledge-driven soft sensors require much expert process knowledge, which is not always available. In addition, knowledge-driven soft sensors are dificult to calibrate, especially for complex nonlinear processes. Note that hybrid model-based soft sensors received the least research attention. Data-driven soft sensors can be further categorized based on the learning technique used for modeling.

Tables 2 to 3 show the current trends in the data-driven soft sensing. Table 1 contains the full forms for the acronyms used in Tables 2 to 3. The research in soft sensing has dramatically shifted from statistical to ML methods. Artificial neural networks (ANNs) received the greatest attention among ML methods. The class of feedforward single hidden layer neural networks (shallow networks)—encompassing multilayer perceptron (MLP),

![](images/e5f24b2f6b8c6c4eb6c25ce5b69d4edbdf15284d4bd60f65970a7ef8b2503bf9.jpg)  
Figure 3: Research publication in soft sensors from 2015 to 2023.

GRNN, ELM, radial basis function neural network (RBFNN), wavelet neural network (WNN) in Table 1—have more applications in soft sensing than recurrent neural networks (RNNs) and deep learning. Aside from ANNs, support vector machine (SVM) is the second most widely used ML method for developing inferential models.

Transfer learning is slowly gaining applications in inferential measurements. Transfer learning alludes to the scenario where knowledge gained while performing one specific task is exploited to carry out a diferent but related task. Especially when data collection becomes dificult in the task of interest, transfer learning still works by sharing information on relevant data in other domains [52]. Transfer learning has yet to be applied to the online prediction of process variables.

Static (time-invariant) soft sensors are developed using data from a single operating mode. However, their prediction accuracy degrades over time as the process shifts to a new operating region. Adaptive soft sensors tackle this issue by updating their parameters based on new samples.<sup>3</sup> Less than onethird of soft sensors are adaptive, most of which use a just-in-time strategy to update model parameters in response to samples arriving in real time (see Figure 4 and Table 4). Therefore, computationally feasible methods are required. In particular, partial least squares (PLS) is the preferred algorithm for local modeling. The training of global soft sensor models is performed ofline. Then, trained soft sensor models are deployed online to obtain realtime estimates for key process or quality variables. Although training time is comparatively very high, most global soft sensors produce estimates quickly when used online [51, 53]

Table 2: Distribution of data-driven methods for soft sensors, split between statistical and ML methods.

<table><tr><td>Statistical methods</td><td>% of publications</td><td>ML methods</td><td>% of publications</td></tr><tr><td>PLS</td><td>11.38</td><td>ANN</td><td>47.72</td></tr><tr><td>PCA</td><td>4.54</td><td>TL</td><td>2.02</td></tr><tr><td>FA</td><td>0.95</td><td>RT</td><td>4.59</td></tr><tr><td>ICA</td><td>1.70</td><td>SFA</td><td>2.75</td></tr><tr><td>LASSO</td><td>1.51</td><td>RVM</td><td>2.57</td></tr><tr><td>GMM</td><td>2.64</td><td>SVM</td><td>7.53</td></tr><tr><td>-</td><td>-</td><td>ANFIS</td><td>1.65</td></tr><tr><td>-</td><td>-</td><td>GPR</td><td>5.87</td></tr><tr><td>-</td><td>-</td><td>BN</td><td>2.57</td></tr><tr><td>Total</td><td>22.73</td><td>Total</td><td>77.27</td></tr></table>

Table 3: Distribution of various types of ANNs for soft sensors.

<table><tr><td>Method</td><td>% of publications</td></tr><tr><td>MLP</td><td>14.23</td></tr><tr><td>DNNE</td><td>0.75</td></tr><tr><td>DNN</td><td>34.64</td></tr><tr><td>ELM</td><td>11.92</td></tr><tr><td>GRNN</td><td>6.92</td></tr><tr><td>WNN</td><td>2.31</td></tr><tr><td>RBFNN</td><td>2.68</td></tr><tr><td>RNN</td><td>21.94</td></tr><tr><td>ENN</td><td>0.38</td></tr><tr><td>ESN</td><td>4.23</td></tr></table>

![](images/169838cbda34f4e7ff26103c9c82990f9611cdff6f8756f7eb0250beb688d840.jpg)  
Figure 4: Distribution of global and adaptive soft sensors.

In Table 5, publications on each data-driven technique have been grouped into three categories: publications based on simulation data, publications based on industrial data, and publications that reported industrial implementation. Notice that most soft sensors have been developed and tested on industrial data. Still, only some of them—PLS, MLP, WNN, SVM, relevance vector machine (RVM), Gaussian process regression (GPR) and regression tree (RT)—have made it into actual industrial implementation. Of course, there may be a publication bias for academic examples, as not all real-world industrial applications may be reported on.

## 3.3. Computational cost of soft sensors

The training time refers to the time taken to determine optimal values for the parameters of a soft sensor model. Once the developed soft sensor is implemented online in a distributed control system, it is used to estimate key process or quality variables at regular sampling intervals. The time required to get the estimates is called soft sensing time.

Table 4: Distribution of statistical and ML methods in local modeling of adaptive soft sensors.

<table><tr><td>Method</td><td>% of publications</td></tr><tr><td>PCA</td><td>7.05</td></tr><tr><td>MLP</td><td>3.52</td></tr><tr><td>SVR</td><td>11.77</td></tr><tr><td>GMM</td><td>2.35</td></tr><tr><td>GPR</td><td>15.30</td></tr><tr><td>BN</td><td>4.70</td></tr><tr><td>RVM</td><td>2.35</td></tr><tr><td>ELM</td><td>9.42</td></tr><tr><td>FA</td><td>3.52</td></tr><tr><td>PLS</td><td>31.78</td></tr><tr><td>LASSO</td><td>8.24</td></tr></table>

Diferent techniques have various levels of computational complexity, that is, model training time. Since principal component analysis (PCA) [54], slow feature analysis [55], independent component analysis [56], and factorial analysis [57] can be developed in a single iteration, they require relatively low computational time compared to LASSO [58], and GMM [59] techniques, which involve using iterative optimization algorithms to determine the model parameters. In general, ML methods need more computational time than statistical methods [53]. Further, the computational complexity of ML methods is influenced by the factors listed below [60] :

• Amount of training data.

• Number of features or input variables.

• Type of training algorithm employed.

• Number of layers.

• Number of neurons (size) in layers.

• Type of device used (such as CPU or GPU).

Table 5: Breakdown of methods for soft sensors according to the level of industrial applications.

<table><tr><td>Method</td><td>Simulation data</td><td>Industrial data</td><td>Industrial use</td><td>Number of publications</td></tr><tr><td>PCA</td><td>8</td><td>16</td><td>0</td><td>24</td></tr><tr><td>PLS</td><td>12</td><td>45</td><td>3</td><td>60</td></tr><tr><td>SFA</td><td>4</td><td>11</td><td>0</td><td>15</td></tr><tr><td>ICA</td><td>7</td><td>2</td><td>0</td><td>9</td></tr><tr><td>LR</td><td>0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>LASSO</td><td>1</td><td>6</td><td>0</td><td>7</td></tr><tr><td>FA</td><td>0</td><td>5</td><td>0</td><td>5</td></tr><tr><td>GMM</td><td>3</td><td>11</td><td>0</td><td>14</td></tr><tr><td>MLP</td><td>4</td><td>32</td><td>1</td><td>37</td></tr><tr><td>RBFNN</td><td>2</td><td>5</td><td>0</td><td>7</td></tr><tr><td>WNN</td><td>1</td><td>4</td><td>1</td><td>6</td></tr><tr><td>RNN</td><td>4</td><td>53</td><td>0</td><td>57</td></tr><tr><td>GRNN</td><td>5</td><td>13</td><td>0</td><td>18</td></tr><tr><td>ELM</td><td>6</td><td>25</td><td>0</td><td>31</td></tr><tr><td>ANFIS</td><td>3</td><td>6</td><td>0</td><td>9</td></tr><tr><td>DNNE</td><td>1</td><td>1</td><td>0</td><td>2</td></tr><tr><td>DNN</td><td>7</td><td>80</td><td>0</td><td>87</td></tr><tr><td>SVM</td><td>12</td><td>21</td><td>4</td><td>37</td></tr><tr><td>RVM</td><td>4</td><td>9</td><td>1</td><td>14</td></tr><tr><td>GPR</td><td>12</td><td>18</td><td>2</td><td>32</td></tr><tr><td>RT</td><td>5</td><td>19</td><td>1</td><td>25</td></tr><tr><td>BN</td><td>0</td><td>14</td><td>0</td><td>14</td></tr><tr><td>TL</td><td>4</td><td>7</td><td>0</td><td>11</td></tr><tr><td>ESN</td><td>2</td><td>9</td><td>0</td><td>11</td></tr><tr><td>ENN</td><td>0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>Total</td><td>107</td><td>414</td><td>13</td><td>534</td></tr></table>

The ELM is considered the fastest ML algorithm because it does not have parameters that need to be learned. The second fastest ML method is the GRNN, which has a single learnable parameter (spread or width of a radial basis function).<sup>4</sup> Then come RTs and decorrelated neural network ensembles, which can be constructed more easily than shallow neural networks like RBFNN, MLP, WNN, and adaptive network fuzzy inference system (ANFIS). As RBFNN uses hybrid learning (not hybrid modeling)— unsupervised learning for the middle layer and supervised learning (linear regression) for the last layer—it is usually faster than MLP, ANFIS, and WNN, which use iterative gradient descent algorithms. SVM is the slowest of the kernel-based ML methods (SVM, GPR, and RVM). Bayesian networks rely on the expectation-maximization algorithm to optimize their parameters, which takes a little more training time than RTs. Dynamic ML methods, such as RNNs, involve more operations than their static ML counterparts, so they require more memory and computational power [61, 14]. Similarly, deep neural networks (DNNs) often include several layers and hence, contain many parameters. A large amount of training data is necessary to train DNNs. Therefore, DNNs are recognized as the most computationally expensive methods of all the data-driven techniques.

## 3.4. Industry implementation of soft sensors

In industries, soft sensors are developed by in-house control engineers or third-party contractors (service engineers) from service providers such as Honeywell or Yokogawa. These service providers use their own software to build the soft sensors. When the existing technology used by service providers is inadequate to handle a problem or in-house control engineers have no knowledge of other soft sensing algorithms, the industries provide research funding to universities, research organizations, or startups to develop sophisticated soft sensors to model complex nonlinear processes. The following steps outline how soft sensors are developed and implemented in industries.

1. After recognizing a need for a soft sensor application, a team consisting of a panel operator, process engineer, control engineer, and project manager is formed. The process engineer prepares a charter to define the core objectives, scope, responsibilities, and timeline of the project. This outlines the benefits that the soft sensor project can ofer. All the benefits are usually quantified in terms of how much money can be saved. For example, this cost–benefit analysis typically involves weighing the upfront costs—hardware, software, consultants—and continued costs—software licenses, in-house domain experts to handle support and maintenance—against anticipated improved revenue and throughput, as well as reduced cost of the soft sensor. Once the team is satisfied with the benefits, the soft sensor project launches.

2. The next step in executing a soft sensor project involves obtaining process knowledge or expert experience knowledge to identify input variables that have a noteworthy influence on output variables [62]. The use of process knowledge or expert experience avoids the inclusion of redundant input variables in soft sensor modeling, leading to reduced model complexity and improved accuracy. In the absence of such knowledge, ML algorithms such as LASSO, hybrid LASSO, and ridge regression can be used to identify and remove input variables that have negligible impact on the output variable.

3. The third step entails process data collection and preprocessing. The process data are often abundant but poor in information. This is due to significant disturbances, outliers, and missing values. Soft sensors developed using these data may provide incorrect estimates for quality variables. The outliers and missing values from the raw industrial data should be removed to obtain clean data for developing the soft sensor. Although it may not be theoretically rigorous, the usual practice is to detect and delete samples with outliers [63]. Missing values are treated in the same fashion. This approach ensures that the clean data are free of outliers and missing values.

4. The data collection in industrial settings is often associated with multirate sampling. If the sampling frequency of the input variables is higher than that of the output variable, then it is necessary to synchronize the variables. Down-sampling may be used to deal with the multi-rate sampling problem. In the down-sampling approach, samples of the input variables that do not have the respective measurements of the output variable are removed [53].

5. After the process data are preprocessed, they are split into training and validation subsets. The training subset is used to construct a soft sensor model whereas the validation subset is used to evaluate the prediction performance of the soft sensor model. This is called ofline validation. The usual practice is to develop a linear model first. If the linear model cannot produce accurate estimates, then more complex statistical or ML algorithms are used.

6. If the soft sensor model delivers satisfactory performance in the ofline validation, it is implemented in a distributed control system. Then the performance of the soft sensor is monitored for some time period. If the soft sensor exhibits poor performance, then modifications are made. This is online validation. For ofline and online validation, metrics such as the correlation coeficient and root mean squared error are used to quantify the performance of soft sensors [53]. In addition, qualitative analysis is considered to see if soft sensor estimates follow the lab data trend. If the soft sensor estimates are poor, the input data are first examined for possible reasons, such as sensor failures, data transmission problems, outliers, plant shutdowns, and plant upsets. Poor estimates can be characterized by low correlation to lab data, estimates out of the operational range, or significant deviation from lab data. If the input data are good, the following strategies are used to get accurate and reliable estimates:

• Retraining of the soft sensor using the latest data.

• Changing the soft sensor modeling algorithm.

• Using a diferent training algorithm.

• Changing the parameter initialization method.

• Using approaches that can avoid or reduce overfitting.

Regardless of the type of soft sensor, practicing engineers usually follow the above approach to assess the performance of soft sensors.

7. If the online soft sensor consistently provides reasonable results, the soft sensor is used as a measuring device in a control loop. After successfully implementing the soft sensor-based control application, the soft sensor application is handed over to the panel operator. The human-in-theloop aspect described above is crucial in translating research results into practical applications.

## 3.5. Challenges in soft sensor development

Challenges that are often encountered in soft sensor developments are discussed below.

• Lack of labeled data is the main challenge that must be dealt with in order to build good soft sensor models. Quality variables are less frequently measured than easily measurable process variables, such as temperature, pressure, flow rate, and level. A sample of a quality variable is collected once every shift (that is, 8 hours) or 24 hours. Because of the long sampling interval, an insuficient amount of practical labeled data is available. A soft sensor trained with a limited amount of labeled data may not be able to capture the underlying relationship between the input variables and the output variable. To deal with this problem, a virtual sample generation method may be used to obtain estimated output values for the corresponding input data [64]. As an alternative, semi-supervised learning may be used to construct the soft sensor. Unsupervised learning algorithms like PCA, autoencoders, stacked autoencoders, or deep belief networks can extract features from unlabeled input data. These features are related to the output variable by any data-driven linear or nonlinear model [14].

• Operating conditions of the industrial process may change depending on the demand for products, prices of raw materials, and so on. A soft sensor developed using data from one operating condition may not perform well when the operating condition changes. In this situation, multimode soft sensors can be used to get accurate estimates [65].

• Soft sensor maintenance is crucial to continuously attain reasonable estimates, as the performance of an online soft sensor may degrade over time. As a result, estimates obtained by a poorly performing soft sensor do not follow lab data trends. To circumvent this hurdle, the soft sensor is retrained with recent data, and deployed online. A more popular approach to maintain the accuracy of the soft sensor is to adopt a bias updating strategy. In the bias updating strategy, the soft sensor outputs are brought closer to the lab data [66].

## 4. Data-driven and hybrid modeling approaches for optimization and control

We revisit data-driven and hybrid modeling in the context of solving optimization and control problems. We further introduce reinforcement learning as an emerging paradigm for solving challenging control tasks. In the same way hybrid modeling represents a spectrum between knowledge-based and data-based modeling, model-based optimization, model predictive control, and reinforcement learning all encompass model-based and model-free methodologies. Naturally, these techniques are also compatible with hybrid modeling approaches, ofering new challenges and research opportunities.

## 4.1. Model-based optimization

A large number of hybrid modeling applications have been geared towards ofline process optimization. Here, a hybrid model is appealing because key operational variables in terms of process performance may be included in the mechanistic part of the model. This is to retain suficient extrapolation while capturing other parts of the process using data-driven techniques, for example, to reduce the computational burden. Local (gradient-based) or stochastic search techniques have traditionally been applied to solve the resulting model-based optimization problems. But a recent trend has been using complete search techniques to overcome convergence to a local optimum and guarantee global optimality in problems with trained machine learning models embedded, such as multilayer perceptron (MLP) [67, 68, 69], Gaussian process (GP) [70], or gradient-boosted trees [71]. Applications in chemical engineering include the optimization of simple reactor operations and process flowsheets [67] and optimal catalyst selection [71].

It should be noted that developing a data-driven or hybrid model to speed up the optimization of a more fundamental model is akin to conducting a surrogate-based optimization. The latter constitutes an active research area in process flowsheeting, computational fluid dynamics, and molecular dynamics [72]. They can be broadly classified into local and global approaches. Global approaches proceed by constructing a surrogate model based on an ensemble of mechanistic simulations before optimizing it, often within an iteration where the surrogate is progressively refined. Several successful implementations rely on MLPs [73], GPs [74, 75, 76], or a combination of various basis functions [39, 77] for the surrogate modeling. Practical applications have been for rigorous design of distillation columns [75, 76] and flowsheet or superstructure optimization of chemical processes [74, 73]. By contrast, local approaches maintain an accurate surrogate of the mechanistic model within a trust region, whose position and size are adapted iteratively. This procedure entails reconstructing the surrogate model as the trust region moves around. Still, it can ofer global convergence guarantees, for example, when the surrogates meet the full linearity property [78]. Applications of this approach to chemical process optimization include solved-based $\mathrm { C O _ { 2 } }$ capture [79] and integrated carbon capture and conversion [80].

## 4.2. Model predictive control and real-time optimization

The real-time optimization (RTO) and nonlinear/economic model predictive control (MPC) methodologies use a process model at their core. So far, most successful implementations of RTO and MPC have relied on mechanistic models [81, 82, 83]. But there has been interest in data-driven approaches, which use surrogate models trained on historical data or mechanistic model simulations to drive the optimization. The type of surrogate models used in such data-driven MPC includes MLPs [84, 85] and GPs [86, 87]. However, comparatively little work has been published on embedding hybrid models into MPC to reduce data dependency and infuse physical knowledge for better extrapolation capability [88, 89]. Teixeira et al. [90] applied batchto-batch optimization to bioprocesses by relying on hybrid models where an adjustable mixture of nonparametric and parametric models represented the cell population subsystem. In the RTO area, Cubillos et al. [91] investigated the use of parallel hybrid models with MLP embedded on the Williams benchmark plant, but then they had to use stochastic search methods to solve the resulting optimization problems. Recently, Zhang et al. [89] took the extra step of using the same hybrid model simultaneously in the RTO and MPC layers and demonstrated the benefits for a simulated CSTR and distillation column. Notice that most of these applications consider serial hybrid models with embedded MLPs to approximate complex nonlinearities in the system. Nevertheless, there is a dearth of industrial or experimental implementations of such technologies to date.

An RTO methodology that exploits the parallel approach of hybrid semiparametric modeling at its core is modifier adaptation [92]. Unlike classical RTO, modifier adaptation does not adapt the mechanistic model but adds correction terms—the modifiers—to the cost and constraint functions in the optimization model. The original work used process measurements to estimate linear (gradient-based) corrections [93]. Gao et al. [94] proposed combining quadratic regression models trained on available plant data with a nominal mechanistic model to account for curvature information and filter out the process noise. Likewise, Singhal et al. [95] investigated datadriven approaches based on quadratic surrogates as modifiers for the predicted cost and constraint functions and devised an online adaptation strategy for the surrogates inspired by trust-region ideas. Implementations of this RTO methodology for industrial systems include load sharing for gas compressors [96] and solid-oxide fuel cells [97].

More recently, Ferreira et al. [98] were the first to consider GPs, trained from past measurement information, as the cost and constraint modifiers. Using nonparametric regression models to describe the plant-model mismatch in RTO applications makes sense insofar as the mismatch is generally structural. Del Rio Chanona et al. [99, 100] developed this strategy further by introducing modifier-adaptation schemes that rely on trust regions to capture the GPs’ ability to capture the cost and constraint mismatch. Recently, Petsagkourakis et al. [101] proposed to use co-Kriging to drive the surrogate modeling, where a first (low-fidelity) GP emulating the mechanistic process model is integrated within a second (high-fidelity) GP that is trained using the process measurements. The benefits of using GPs in this context lie in their ability to perform real-time uncertainty quantification and allow chance constraints to be satisfied with high confidence. By and large, these developments share many common grounds with surrogate-based optimization techniques (see Section 4.1), with the added complexity that the process data are noisy and the process optimum might change over time. Finally, it is worth noting that the potential benefits of this RTO technology have been mostly investigated through numerical simulation, which cannot substitute for both experimental and industrial validations and should be the subject of future research.

## 4.3. Reinforcement learning

Reinforcement learning (RL) is a class of numerical methods for the datadriven sequential decision-making problem [102]. The RL agent (algorithm) aims to find an optimal policy, or controller, based on industrial process data collected through interactions with its environment.

Note that RL represents a more general class of techniques from hybrid modeling-based optimization. Briefly, RL includes algorithms for synthesizing control policies without explicit reliance on a model of the process dynamics. The supplementary material contains a more precise background on RL; readers are also referred to Sutton and Barto [102].

Finding such a policy requires solving the Bellman equation based on the principle of optimality. However, the equation is often intractable as it ends up with a high-dimensional optimization problem [103]. Recent advances in machine learning (ML) enable feature analysis of raw sensory-level using deep neural networks (DNNs). The aid of DNNs facilitates eficient numerical methods for approximately solving the Bellman equation. Therefore, the scalability of RL algorithms has been significantly improved. As a result, so-called deep RL is an emerging technology that has shown remarkable performance in real-world and simulated applications such as robotics, autonomous driving, and board games [104, 105, 106].

Deep RL has naturally gained attention from the process control community. In this section, we survey applications of RL in process control, and we discuss advances and challenges in RL as they potentially pertain to process control applications.

## 4.3.1. Reinforcement learning for process control

With high demands on the performance of process systems, eficient optimization is becoming increasingly essential. The ultimate dream goal of any process control system is to develop a controller capable of attaining optimality in large-scale, nonlinear, and hybrid models with constraints, fast online calculation, and adaptation. This ideal controller should be amenable to a closed-loop solution and robust to online disturbances.

Mathematical programming-based control, such as MPC and direct optimization, are popular because they adequately address many of these requirements. Sections 4.1 and 4.2 discuss the mathematical programming paradigm in more detail. RL has been studied in parallel because it has contrasting features compared to mathematical programming methods [107]. According to the review and perspective studies of Shin et al. [5], Nian et al. [8], Spielberg et al. [6], Yoo et al. [108], the advantages of RL are that: First, a closed-loop state feedback policy can be obtained for generic stochastic control problems, while an open-loop solution is obtained through mathematical programming approaches. Most of the computation is done ofline by learning the policy through ofline data or simulation. Assuming that the environment used for ofline training is identical to that of the online implementation, the policy is optimal. Second, the mathematical programming formulation for stochastic control problems often becomes prohibitively large to be solved within a decision interval. On the other hand, uncertainties are implicitly or explicitly quantified by the value or policy functions in RL approaches. The trained RL policy can be implemented with minimal online computation required. Third, RL is flexible to varying levels of system knowledge, including model-free, partial model-free, and model-based RL. Table 6 summarizes the comparison between RL and mathematical programming methods.

Several pioneering pieces of work due to Wilson and Martinez [109], Kaisare et al. [110], Peroni et al. [111] proposed applying model-free RL to process control problems over discretized state and action spaces. Qlearning was implemented for the tracking control of a fed-batch bioreactor [109] and free-end maximization problem of a fed-batch bioreactor [110, 111]. Lee et al. [112], Lee and Lee [113] extended the concept of applying modelfree RL to dual adaptive control and scheduling problems. It was shown that the approximation of the value function could provide robust control despite the presence of process noise and model changes. RL methods that guarantee robustness in dynamic optimization were later studied in Nosair et al. [114], Yang and Lee [115].

Some recent applications of RL rely on a linear approximator to solve optimal control problems with a continuous state space model [116, 117, 118, 119]. Especially, Zhu et al. [116] applied a model-free RL variant called factorial fast-food dynamic policy programming to a Vinyl Acetate monomer process. The algorithm improves scalability by breaking down the exponential size of the action space by action space factorization. In the meantime, model-free deep RL applications have become increasingly studied in the process control field. Table 7 summarizes some recent work in this area. In the remaining sections, we elaborate on the use of deep RL in process control.

## 4.3.2. Practical implementation of reinforcement learning

One promising application of RL is the synthesis of existing control structures [141, 142, 143, 144, 145]. For example, proportional-integral-derivative (PID) controllers constitute the lowest level of control structures, and augmenting these with RL methods immediately gives practical results. PID tuning is a suitable testbed for RL applications, as there exists a suite of tuning methods and industrial autotuners to benchmark against [136]. PID controllers are also standard in practice, meaning the base layer control is not substituted for a more complex strategy, for example, based on DNNs (see Figure 5).

Model-free RL was applied to schedule a set of PID gains obtained a priori [146] or from internal model control [147]. Berger and da Fonseca Neto [148] used a model-based RL method, called dual heuristic dynamic programming, to compute PID gains. Nian et al. [8] applied deep Q-network (DQN) to determine the gains of PID controllers and compared the performance with MPC. Lawrence et al. [136] conducted an experimental study on the autotuning of PID controllers using the twin-delayed DDPG (TD3) algorithm. Figure 5 depicts a feedback diagram in the RL setting: the actor is formulated as a PID controller for the flow rate to a two-tank system, while the agent processes data on a PC to update the actor-critic parameters.

Table 6: A comparison of RL and mathematical programming.

<table><tr><td></td><td>Reinforcement learning</td><td>Mathematical programming approaches</td></tr><tr><td>Model knowledge</td><td>Flexible</td><td>Full model</td></tr><tr><td>Feedback</td><td>Trained policy function</td><td>Solution of optimization problems</td></tr><tr><td>Online computation</td><td>Negligible</td><td>High</td></tr><tr><td>Offline computation</td><td>High</td><td>Not required</td></tr><tr><td>Robustness</td><td>Backward propagation of uncertain scenario (value-based methods)</td><td>Forward propagation of uncertain scenario</td></tr><tr><td>Constraint handling</td><td>Immature (especially, state variable constraints)</td><td>Straightforward</td></tr><tr><td>Asymptotic stability</td><td>Ultimate upper-boundedness</td><td>Asymptotically stable</td></tr><tr><td>Scalability</td><td>High</td><td>Medium</td></tr><tr><td>Adaptation</td><td>Exploitation and exploration can be controlled. However, slow.</td><td>Fast. However, performance depends on estimators.</td></tr></table>

Table 7: Model-free deep RL applications in process control. Asterisk (\*) indicates a model-based modification to the nominal algorithm. Highlighted rows indicate validation on a physical system.

<table><tr><td></td><td>RL algorithm</td><td>Application/process</td></tr><tr><td>Pandian and Noel [120]</td><td>DQN*[121]</td><td>Quadruple tank system</td></tr><tr><td>Wang et al. [122]</td><td>PPO[123]</td><td>HVAC control</td></tr><tr><td>Ma et al. [124]</td><td>DDPG[125]</td><td>Polymerization system</td></tr><tr><td>Spielberg et al. [6]</td><td>DDPG[125]</td><td>HVAC control</td></tr><tr><td>Oh et al. [126]</td><td>DQN[121]</td><td>Moving bed process</td></tr><tr><td>Petsagkourakis et al. [127]</td><td>REINFORCE*[128]</td><td>Fed-batch bioreactor</td></tr><tr><td>Bao et al. [129]</td><td>TD3*[130]</td><td>Setpoint tracking</td></tr><tr><td>Dogru et al. [131]</td><td>A3C[132]</td><td>Hybrid three-tank system</td></tr><tr><td>Joshi et al. [133]</td><td>TD3[130]</td><td>Transesterification process</td></tr><tr><td>Mowbray et al. [134]</td><td>REINFORCE[128]</td><td>Setpoint tracking</td></tr><tr><td>Yoo et al. [135]</td><td>DDPG[125]</td><td>Polymerization process</td></tr><tr><td>Lawrence et al. [136]</td><td>TD3[130]</td><td>PID tuning</td></tr><tr><td>Zhu et al. [137]</td><td>SAC[138]</td><td>Polyol process</td></tr><tr><td>Janjua et al. [139]</td><td>GVF[140]</td><td>Water treatment</td></tr></table>

Another application is to construct hierarchical control structures with RL methods. Shafi et al. [149] introduced a two-layer structure for optimizing the bitumen recovery rate of a primary separation vessel. A supervisory RL agent optimizes the recovery rate, while a low-level RL agent computes the interface level actuation. Kim et al. [150] proposed a diferent type of twolayer structure for a product maximization problem of a fed-batch bioreactor. A model-based RL agent solves the high-level optimization problem, and an MPC tracks the trajectory of the high-level optimizer, rejecting real-time disturbances.

Several studies make a comparison between RL methods based on practical performance criteria. Wang et al. [151] compared 14 model-free and model-based RL algorithms based on the following criteria: nominal performance, sample eficiency (total training time, training time per step), robustness against noise, and asymptotic performance. Lawrence et al. [136] proposed nominal performance, stability, perturbation to the system, initialization, hyperparameters, training duration, practicality, and specialization as key criteria for evaluating RL methods for process control problems. In addition, Dogru et al. [131] used the extent of exploration: the ratio of the visited over the total operational state and action spaces.

It is worth noting that RL implementations on physical systems are sparse. Some works in process control applications are validated on physical systems [147, 152, 120, 8, 136, 131]. These references tend to focus on PID tuning or low-dimensional state/action spaces. A cascaded tank system is also the most common environment. There are several plausible reasons for the lack of real-world RL applications: The added engineering and software development is not always feasible to accommodate; the algorithmic complexity of RL algorithms exacerbates the issue; practical and theoretical problems, such as sample eficiency, convergence, and closed-loop stability, are pressing concerns. Indeed, most deep RL algorithms can achieve impressive final performance on complex tasks, but at the cost of extensive hyperparameter tuning and significant variation between implementations [153]. In the following section, we highlight a few methods that are geared towards making RL more reliable and scalable: Synthesis between model-based and model-free learning; transfer learning and meta-RL; ofline RL.

![](images/42d0504b71f6e7667206c311e31f622c10ab6bc12db2ddb80e08008b4b1ce4ec.jpg)  
Figure 5: Application of RL for tuning PI controllers in a lab setting. The policy plays the role of a PI controller and receives updates towards improved performance. J is a general long-term cost function and $k _ { p } , k _ { i }$ are controller gains. Adapted from [136].

## 4.3.3. Challenges and advances in deep reinforcement learning

Applying RL to industrial settings has many practical, technological, and theoretical challenges. We refer to Shin et al. [5], Nian et al. [8] for further reading. Here, we mainly focus on the sample eficiency of RL algorithms. Sample eficiency refers to the amount of data needed to train an RL agent. The supplementary material contains a more general discussion about ML with limited data.

Classical algorithms for value-based methods, such as Q-learning, and policy-based methods, such as REINFORCE<sup>5</sup>, enjoy theoretical convergence. However, convergence can be slow due to high variance in value estimates or limited to the tabular setting or linear function approximation [102]. Nonetheless, these methods provide the foundation for deep RL algorithms. Deep RL attempts to scale up RL methods to high-dimensional problems as a synthesis with the deep learning framework. The first notable result is an extension of Q-learning, named DQNs, introduced by Mnih et al. [121]. DQNs are limited to discrete action spaces but showed impressive results in tasks with high-dimensional sensory input data, such as Atari games.

More recent algorithms, such as the deep deterministic policy gradient (DDPG) algorithm [125], allow for continuous action spaces. Despite the advances made by DDPG, it is notoriously dificult to use, for example, due to sensitivity to hyperparameters and overestimation of Q-function values [153]. This limits the viability of DDPG for real-world applications such as process control, as a physical system cannot be extensively probed. However, the concurrent algorithms, TD3 [130] and soft actor-critic [138], built of DDPG to improve the overall training robustness and sample eficiency. Despite these advances, model-free RL algorithms alone are not suficiently dataeficient and, therefore, not yet useful in real industrial applications [154]. In the rest of this section, we identify several areas of RL research aimed at this issue.

Although formulating a dynamic model can be a bottleneck in the RL algorithm, model-based methods require much fewer interactions with the plant [154]. Several model-based RL algorithms have been developed, focusing on solving the continuous-time counterpart of the Bellman equation called the Hamilton-Jacobi-Bellman (HJB) equation. Since they aim to solve the HJB equation adaptively, the methods are called approximate dynamic programming (ADP) [155, 156, 157]. ADP algorithms vary with their levels of model utilization, ranging from heuristic dynamic programming, dual heuristic programming, and globalized dual heuristic programming [158, 159]. Stochastic optimal control is an extension for handling stochastic diferential equations, a continuous-time description for uncertainty. Policy improvement with path integrals $( \mathrm { P I } ^ { 2 } )$ is a sampling approach to solving the stochastic HJB equation [160]. PI<sup>2</sup> has shown remarkable data eficiency and performance for robot learning.

Another line of work has focused on unifying model-free and model-based approaches [161, 162]. The main motivation is that model-free algorithms often achieve superior final (asymptotic) performance over model-based approaches but sufer from relatively weak sample complexity. Bao et al. [129] utilized ideas from $\mathrm { D ' O r o }$ and Jaśkowski [161] wherein a dynamics model is used to improve the action gradient estimation of the critic network. While integrating dynamic models into traditionally model-free algorithms has proved promising, these algorithms are designed to train an agent using online interactions on a system-by-system basis. More general strategies aim to reduce the cost of calibrating RL agents to novel environments by utilizing historical datasets, training over many related systems, or transferring previously trained agents to new ones.

Ofline RL (sometimes called batch RL) aims to learn an optimal policy from historical data alone [163]. Although of-policy algorithms like DDPG can theoretically learn from historical data, online exploration is critical unless constraints are imposed on the learned policy [164]. An ofline strategy for pre-training RL agents with historical process data, followed by online fine-tuning of the policy, is proposed by Mowbray et al. [134]. On the other hand, transfer learning is a framework for speeding up the training of RL agents. By pre-training a policy, such as in a simulation environment, one can use this as the initial policy on the true system of interest. This idea is demonstrated for batch bioprocess optimization [127]. One can eficiently mitigate plant-model mismatch by fine-tuning the initial policy on the real system.

Meta-learning, or learning to learn, is a ML strategy for leveraging prior training experience to learn a new “task” quickly [165]. Meta-RL is a strategy for training a “meta agent” to synthesize experience from many related systems to adapt its policy to novel systems rapidly. For example, Finn et al. [166] develop a simple and highly influential algorithm for any neural network architecture that directly optimizes for initial parameters such that they can quickly be adapted to new tasks with a small amount of data, showing superior performance over standard transfer learning in classification and RL tasks. Duan et al. [167] propose strategies for learning a latent context variable as part of the meta-policy architecture, thereby capturing the “task” structure and enabling the meta-RL agent to adapt its policy with new process data. This framework is appealing in process control applications because many systems may have a known structure, making training over a distribution of related systems feasible. Consequently, this end-to-end framework removes a model identification step during the online implementation of the RL agent by leveraging prior training experience. Meta-RL has also seen recent applications to process control [168].

While significant strides have been made to make these algorithms more sample-eficient, they are not yet practical. Motivated by this challenge, we have outlined diferent ways in which models can be integrated into otherwise model-free algorithms. Moreover, meta-RL, ofline RL, and transfer learning, while still emerging, are promising avenues for MPC applications. These areas have tremendous potential for applications that can redefine automation in the process industries.

## 5. Discussion

Soft sensing and process control encompass statistical learning, machine learning, deep learning, and reinforcement learning to varying degrees. Table 8 shows the respective high-level prominence in these two application areas. Although Table 2 indicates significant interest in the soft sensing literature around deep learning, Table 5 shows methods like PLS and SVM have received the most industrial use. However, the prominent use of industrial data is still promising. Meanwhile, our survey of process control indicates a more significant emphasis on deep learning and reinforcement learning in the literature. Simulation-based studies are commonplace in this context, as discussed in Section 4.3.2.<sup>6</sup>

Table 8 and the above discussion show a duality between sensing and control in the context of machine learning methods. To fully capture the benefits of modern machine learning methods, a unified framework that encompasses modeling, sensing, and control is required. Reinforcement learning is well-suited to bridge the gap between sensing and control through a global reward-based objective (rather than treating prediction and control performance as independent goals). Applications in sensing do not necessarily contradict the model-free nature of reinforcement learning, which is most appealing. Rather, this characteristic makes it versatile for processing and optimizing real system data. To illustrate this point, Xie et al. [171] propose using reinforcement learning for sensing, even though it has typically been described in the context of control. Moreover, Esfahani et al. [172] utilize reinforcement learning for both state estimation and control under a single closed-loop performance objective.

Table 8: Method-application pairs covered in this survey. <sup>✓</sup>: significant emphasis, <sup>✗</sup>: sparse emphasis.

<table><tr><td></td><td>Soft sensing</td><td>Process control</td></tr><tr><td>Statistical learning</td><td>✓</td><td>✕</td></tr><tr><td>Machine learning</td><td>✓</td><td>✓</td></tr><tr><td>Deep learning</td><td>✓</td><td>✓</td></tr><tr><td>Reinforcement learning</td><td>✕</td><td>✓</td></tr></table>

On the other hand, Section 4.3 discussed the complexity of reinforcement learning algorithms. More broadly, deep learning and reinforcement learning algorithms are rife with complexity and hyperparameters, making it dificult to parse their fundamental inner workings [153, 173]. A promising avenue toward unifying sensing and control is distilling reinforcement learning pipelines and reimagining techniques from other branches of machine learning. Truly robust and powerful methods will follow from such a critical rapprochement of the longstanding statistical learning methods in Table 1 and newer concepts in deep learning and reinforcement learning. An instance of this aspiration in action is by Eysenbach et al. [174], where they show a novel use of binary classification and policy iteration is capable of achieving state-of-the-art performance.

## 5.1. Conclusions

Recent advances in machine learning give us renewed optimism for achieving higher levels of automation in the process industries. To distill this general goal, we have surveyed soft sensing and process control through a practical lens. Soft sensing represents the most dominant area regarding industrial applications of statistical and machine learning techniques. On the other hand, considerable research attention has been given to deep learning applications, but with limited industrial successes. Through synthesizing research trends and industrial requirements, we have strived to enable academics and practitioners alike to develop sophisticated yet practical methods for building better models and controllers.

## Acknowledgements

We are grateful to the anonymous reviewers for their detailed and constructive feedback; their comments significantly improved the quality of this paper. NPL & RBG gratefully acknowledge the financial support of the Natural Sciences and Engineering Research Council of Canada (NSERC) and Honeywell Process Solutions. JML gratefully acknowledges the research facilities for this work provided by the Institute of Engineering Research at Seoul National University. BC gratefully acknowledges funding by the Engineering and Physical Sciences Research Council (EPSRC) under grants EP/T000414/1 and EP/W003317/1. BH, FA and SKD gratefully acknowledge financial supports from the Natural Sciences and Engineering Research Council of Canada (NSERC) under grants IRCPJ 417793-15 and ALLRP 561080-20.

## References

[1] Venkat Venkatasubramanian. The promise of artificial intelligence in chemical engineering: Is it here, finally? AIChE Journal, 65(2):466–478, 2019.

[2] Leo H Chiang, Evan L Russell, and Richard D Braatz. Fault detection and diagnosis in industrial systems. Springer Science & Business Media, 2000.

[3] S Joe Qin and Leo H Chiang. Advances and opportunities in machine learning for process data analytics. Computers & Chemical Engineering, 126:465–473, 2019.

[4] Zhiqiang Ge, Zhihuan Song, Steven X Ding, and Biao Huang. Data mining and analytics in the process industry: The role of machine learning. IEEE Access, 5: 20590–20616, 2017.

[5] Joohyun Shin, Thomas A Badgwell, Kuang-Hung Liu, and Jay H Lee. Reinforcement Learning–Overview of recent progress and implications for process control. Computers & Chemical Engineering, 127:282–294, 2019.

[6] Steven Spielberg, Aditya Tulsyan, Nathan P. Lawrence, Philip D. Loewen, and R. Bhushan Gopaluni. Toward self-driving processes: A deep reinforcement learning approach to control. AIChE Journal, 65, 2019.

[7] Aditya Tulsyan, Tony Wang, Gregg Schorner, Hamid Khodabandehlou, Myra Coufal, and Cenk Undey. Automatic real-time calibration, assessment, and maintenance of generic Raman models for online monitoring of cell culture processes. Biotechnology and Bioengineering, 117(2):404–416, 2020.

[8] Rui Nian, Jinfeng Liu, and Biao Huang. A review on reinforcement learning: Introduction and applications in industrial process control. Computers & Chemical Engineering, page 106886, 2020.

[9] Xiaotian Bi, Ruoshi Qin, Deyang Wu, Shaodong Zheng, and Jinsong Zhao. One step forward for smart chemical process fault detection and diagnosis. Computers & Chemical Engineering, 164:107884, 2022.

[10] Thomas Gamer, Mario Hoernicke, Benjamin Kloepper, Reinhard Bauer, and Alf J Isaksson. The autonomous industrial plant–future of process engineering, operations and maintenance. Journal of Process Control, 88:101–110, 2020.

[11] R. Bhushan Gopaluni, Aditya Tulsyan, Benoit Chachuat, Biao Huang, Jong Min Lee, Faraz Amjad, Seshu Kumar Damarla, Jong Woo Kim, and Nathan P. Lawrence. Modern machine learning tools for monitoring and control of industrial processes: A survey. IFAC-PapersOnLine, 53(2):218–229, 2020.

[12] Joel Sansana, Mark N Joswiak, Ivan Castillo, Zhenyu Wang, Ricardo Rendall, Leo H Chiang, and Marco S Reis. Recent trends on hybrid modeling for industry 4.0. Computers & Chemical Engineering, 151:107365, 2021.

[13] Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning: Data Mining, Inference and Prediction. Springer Series in Statistics, New York, 2nd edition, 2009.

[14] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.

[15] Moritz von Stosch, Rui Oliveira, Joana Peres, and Sebastião Feyo de Azevedo. Hybrid semi-parametric modeling in process systems engineering: Past, present and future. Computers & Chemical Engineering, 60:86 – 101, 2014.

[16] Benjamin Peherstorfer, Karen Willcox, and Max Gunzburger. Survey of multifidelity methods in uncertainty propagation, inference, and optimization. SIAM Review, 60 (3):550–591, 2018.

[17] Dimitris C. Psichogios and Lyle H. Ungar. A hybrid neural network-first principles approach to process modeling. AIChE Journal, 38(10):1499–1511, 1992.

[18] Hong-Te Su, N. Bhat, P. A. Minderman, and T. J. McAvoy. Integrating neural networks with first principles models for dynamic modeling. IFAC Proceedings Volumes, 25(5):327–332, 1992.

[19] Michael L. Thompson and Mark A. Kramer. Modeling chemical processes using prior knowledge and neural networks. AIChE Journal, 40(8):1328–1340, 1994.

[20] L. Chen, O. Bernard, G. Bastin, and P. Angelov. Hybrid modelling of biotechnological processes using neural networks. Control Engineering Practice, 8(7):821–827, 2000.

[21] Dörte Solle, Bernd Hitzmann, Christoph Herwig, Manuel Pereira Remelhe, Sophia Ulonska, Lynn Wuerth, Adrian Prata, and Thomas Steckenreiter. Between the poles of data-driven and mechanistic modeling for process operation. Chemie Ingenieur Technik, 89(5):542–561, 2017.

[22] A. Schuppert and T. Mrziglod. Hybrid model identification and discrimination with practical examples from the chemical industry. In J. Glassey and M. von Stosch, editors, Hybrid Modeling in Process Industries, pages 63–88, Boca Raton, 2018. CRC Press.

[23] Sohrab Zendehboudi, Nima Rezaei, and Ali Lohi. Applications of hybrid models in chemical, petroleum, and energy systems: A systematic review. Applied Energy, 228:2539–2566, 2018.

[24] I. Ahmad, A. Ayub, M. Kano, and I.I. Cheema. Gray-box soft sensors in process industry: Current practice, and future prospects in era of big data. Processes, 8(2): 243, 2020.

[25] William Bradley, Jinhyeun Kim, Zachary Kilwein, Logan Blakely, Michael Eydenberg, Jordan Jalvin, Carl Laird, and Fani Boukouvala. Perspectives on the integration between first-principles and data-driven modeling. Computers & Chemical Engineering, 166:107898, 2022.

[26] Mukul Agarwal. Combining neural and conventional paradigms for modelling, prediction and control. International Journal of Systems Science, 28(1):65–81, 1997.

[27] Mohammed Saad Faizan Bangi and Joseph Sang-Il Kwon. Deep hybrid modeling of chemical process: Application to hydraulic fracturing. Computers & Chemical Engineering, 134:106696, 2020.

[28] Donovan Chafart and Luis A. Ricardez-Sandoval. Optimization and control of a thin film growth process: A hybrid first principles/artificial neural network based multiscale modelling approach. Computers & Chemical Engineering, 119:465–479, 2018.

[29] D. Ghosh, E. Hermonat, P. Mhaskar, S. Snowling, and R. Goel. Hybrid modeling approach integrating first-principles models with subspace identification. Industrial & Engineering Chemistry Research, 58(30):13533–13543, 2019.

[30] Pau Cabaneros Lopez, Isuru A. Udugama, Sune T. Thomsen, Christian Roslander, Helena Junicke, Miguel Mauricio-Iglesias, and Krist V. Gernaey. Towards a digital twin: a hybrid data-driven and mechanistic digital shadow to forecast the evolution of lignocellulosic fermentation. Biofuels, Bioproducts & Biorefining, 14(5):1046– 1060, 2020.

[31] Dongda Zhang, Ehecatl Antonio Del Rio-Chanona, Panagiotis Petsagkourakis, and Jonathan Wagner. Hybrid physics-based and data-driven modeling for bioprocess

online simulation and optimization. Biotechnology & Bioengineering, 116(11):2919– 2930, 2019.

[32] Belmiro Duarte, P. M. Saraiva, and C. C. Pantelides. Combined mechanistic and empirical modelling. International Journal of Chemical Reactor Engineering, 2(1), 2004.

[33] C. de Prada, D. Hose, G. Gutierrez, and J. L. Pitarch. Developing grey-box dynamic process models. IFAC-PapersOnLine, 51(2):523–528, 2018.

[34] Tim Hesterberg, Nam Hee Choi, Lukas Meier, and Chris Fraley. Least angle and $\ell _ { 1 }$ penalized regression: A review. Statistics Surveys, 2:61–93, 2008.

[35] Mark J. Willis and Moritz von Stosch. Simultaneous parameter identification and discrimination of the nonparametric structure of hybrid semi-parametric models. Computers & Chemical Engineering, 104:366–376, 2017.

[36] Dongda Zhang, Thomas R. Savage, and Bovinille A. Cho. Combining model structure identification and hybrid modelling for photo-production process predictive simulation and optimisation. Biotechnology & Bioengineering, 117(11):3356–3367, 2020.

[37] Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–3937, 2016.

[38] Zachary T. Wilson and Nikolaos V. Sahinidis. The ALAMO approach to machine learning. Computers & Chemical Engineering, 106:785–795, 2017.

[39] Alison Cozad, Nikolaos V. Sahinidis, and David C. Miller. A combined firstprinciples and data-driven approach to model building. Computers & Chemical Engineering, 73:116–127, 2015.

[40] K. M. Nauta, S. Weiland, A. C. Backx, and A. Jokic. Approximation of fast dynamics in kinetic networks using non-negative polynomials. In 2007 IEEE International Conference on Control Applications, pages 1144–1149, 2007.

[41] J. L. Pitarch, A. Sala, and C. de Prada. A systematic grey-box modeling methodology via data reconciliation and SOS constrained regression. Processes, 7(3):170, 2016.

[42] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686 – 707, 2019.

[43] G. F. Carey and B. A. Finlayson. Orthogonal collocation on finite elements. Chemical Engineering Science, 30(5):587–596, 1975.

[44] Lorenz T Biegler. An overview of simultaneous strategies for dynamic optimization. Chemical Engineering and Processing: Process Intensification, 46(11):1043–1053, 2007.

[45] H. Liu, J. Cai, and Y. S. Ong. Remarks on multi-output Gaussian process regression. Knowledge-Based Systems, 144:102–121, 2018.

[46] L. Le Gratiet and J. Garnier. Recursive co-kriging model for design of computer experiments with multiple levels of fidelity. International Journal for Uncertainty Quantification, 4(5):365–386, 2014.

[47] P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence, and G. E. Karniadakis. Nonlinear information fusion algorithms for data-eficient multi-fidelity modelling. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2198), 2017.

[48] K. Cutajar, M. Pullin, A. Damianou, N. Lawrence, and J. González. Deep Gaussian processes for multi-fidelity modeling. Advances in Neural Information Processing Systems, 32, 2018.

[49] Amir Ali Ahmadi and Bachir El Khadir. Learning dynamical systems with side information, 2020.

[50] Shima Khatibisepehr, Biao Huang, and Swanand Khare. Design of inferential sensors in the process industry: A review of Bayesian methods. Journal of Process Control, 23(10):1575–1596, 2013.

[51] Luigi Fortuna, Salvatore Graziani, Alessandro Rizzo, and Maria Gabriella Xibilia. Soft sensors for monitoring and control of industrial processes. Springer Science & Business Media, 2007.

[52] Fei Chu, Xu Zhao, Yuan Yao, Tao Chen, and Fuli Wang. Transfer learning for batch process optimal control using LV-PTM and adaptive control strategy. Journal of Process Control, 81:197–208, 2019.

[53] Petr Kadlec, Bogdan Gabrys, and Sibylle Strandt. Data-driven soft sensors in the process industry. Computers & Chemical Engineering, 33(4):795–814, 2009.

[54] Ian T Jollife and Jorge Cadima. Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 374(2065):20150202, 2016.

[55] Laurenz Wiskott and Terrence J Sejnowski. Slow feature analysis: Unsupervised learning of invariances. Neural computation, 14(4):715–770, 2002.

[56] Aapo Hyvärinen and Erkki Oja. Independent component analysis: algorithms and applications. Neural networks, 13(4-5):411–430, 2000.

[57] Zhiqiang Ge. Supervised latent factor analysis for process data regression modeling and soft sensor application. IEEE Transactions on Control Systems Technology, 24 (3):1004–1011, 2015.

[58] Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1):267–288, 1996.

[59] Le Yao and Zhiqiang Ge. Scalable semisupervised GMM for big data quality prediction in multimode processes. IEEE Transactions on Industrial Electronics, 66(5): 3681–3692, 2018.

[60] John D Kelleher, Brian Mac Namee, and Aoife D’arcy. Fundamentals of machine learning for predictive data analytics: algorithms, worked examples, and case studies. MIT press, 2020.

[61] Kai Wang, Bhushan Gopaluni, Junghui Chen, and Zhihuan Song. Deep learning of complex batch process data and its application on quality prediction. IEEE Transactions on Industrial Informatics, 2018.

[62] S Joe Qin, Siyi Guo, Zheyu Li, Leo H Chiang, Ivan Castillo, Birgit Braun, and Zhenyu Wang. Integration of process knowledge and statistical learning for the Dow data challenge problem. Computers & Chemical Engineering, 153:107451, 2021.

[63] Christophe Leys, Christophe Ley, Olivier Klein, Philippe Bernard, and Laurent Licata. Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. Journal of experimental social psychology, 49(4):764–766, 2013.

[64] Yalin Wang Ling Li, Seshu Kumar Damarla and Biao Huang. A Gaussian mixture model based virtual sample generation approach for small datasets in industrial processes. Information Sciences, 581:262–277, 2021.

[65] Zhihuan Song Weiming Shao and Le Yao. Soft sensor development for multimode processes based on semisupervised Gaussian mixture models. IFAC PapersOnline, 51:614–619, 2018.

[66] Kuilin Chen, Ivan Castillo, Leo H Chiang, and Jie Yu. Soft sensor model maintenance: A case study in industrial processes. IFAC-PapersOnLine, 48(8):427–432, 2015.

[67] Artur M. Schweidtmann and Alexander Mitsos. Deterministic global optimization with artificial neural networks embedded. Journal of Optimization Theory & Applications, 180(3):925–948, 2019.

[68] R. Anderson, J. Huchette, W. Ma, C. Tjandraatmadja, and J. P. Vielma. Strong mixed-integer programming formulations for trained neural networks. Mathematical Programming, 183:3–39, 2020.

[69] C. Tsay, J. Kronqvist, A. Thebelt, and R. Misener. Partition-based formulations for mixed-integer optimization of trained ReLU neural networks. Advances in Neural Information Processing Systems, 34:3068–3080, 2021.

[70] Artur M. Schweidtmann, Dominik Bongartz, Daniel Grothe, Tim Kerkenhof, Xiaopeng Lin, Jaromil Najman, and Alexander Mitsos. Deterministic global optimization with Gaussian processes embedded. Mathematical Programming Computation, 13:553–581, 2021.

[71] M. Mistry, D. Letsios, G. Krennrich, R. M. Lee, and R. Misener. Mixed-integer convex nonlinear optimization with gradient-boosted trees embedded. INFORMS Journal on Computing, 33(3):1103–1119, 2021.

[72] Lorenz T. Biegler, Yi-dong Lang, and Weijie Lin. Multi-scale optimization for process systems engineering. Computers & Chemical Engineering, 60:17–30, 2014.

[73] Carlos A. Henao and Christos T. Maravelias. Surrogate-based superstructure optimization framework. AIChE Journal, 57(5):1216–1232, 2011.

[74] José A. Caballero and Ignacio E. Grossmann. An algorithm for the use of surrogate models in modular flowsheet optimization. AIChE Journal, 54(10):2633–2650, 2008.

[75] Natalia Quirante, Juan Javaloyes, and José. Caballero. Rigorous design of distillation columns using surrogate models based on kriging interpolation. AIChE Journal, 61 (7):2169–2187, 2015.

[76] Tobias Keßler, Christian Kunde, Kevin McBride, Nick Mertens, Dennis Michaels, Kai Sundmacher, and Achim Kienle. Global optimization of distillation columns using explicit and implicit surrogate models. Chemical Engineering Science, 197: 235–245, 2019.

[77] Fani Boukouvala and Christodoulos A. Floudas. ARGONAUT: AlgoRithms for Global Optimization of coNstrAined grey-box compUTational problems. Optimization Letters, 11(5):895–913, 2017.

[78] Andrew R. Conn, Katya Scheinberg, and Luis N. Vicente. Introduction to Derivative-Free Optimization. MOS-SIAM Series on Optimization, 2009.

[79] John P. Eason and Lorenz T. Biegler. Advanced trust region optimization strategies for glass box/black box models. AIChE Journal, 64(11):3934–3943, 2018.

[80] Ishan Bajaj, Shachit S. Iyer, and M. M. Faruque Hasan. A trust region-based two phase algorithm for constrained black-box and grey-box optimization with infeasible initial point. Computers & Chemical Engineering, 116:306–321, 2018.

[81] Max Schwenzer, Muzafer Ay, Thomas Bergs, and Dirk Abel. Review on model predictive control: An engineering perspective. The International Journal of Advanced Manufacturing Technology, 117(5-6):1327–1349, 2021.

[82] S Joe Qin and Thomas A Badgwell. An overview of nonlinear model predictive control applications. Nonlinear model predictive control, pages 369–392, 2000.

[83] Michael G Forbes, Rohit S Patwardhan, Hamza Hamadah, and R Bhushan Gopaluni. Model predictive control in industry: Challenges and opportunities. IFAC-PapersOnLine, 48(8):531–538, 2015.

[84] Stephen Piche, Bijan Sayyar-Rodsari, Doug Johnson, and Mark Gerules. Nonlinear model predictive control using neural networks. IEEE Control Systems Magazine, 20(3):53–62, 2000.

[85] Zhe Wu, Anh Tran, David Rincon, and Panagiotis D. Christofides. Machine learningbased predictive control of nonlinear processes. Part I: Theory. AIChE Journal, 65 (11):e16729, 2019.

[86] Juš Kocijan, Roderick Murray-Smith, Carl Edward Rasmussen, and Agathe Girard. Gaussian process model based predictive control. In Proceeding of American Control Conference, volume 3, pages 2214–2219, 2004.

[87] L. Hewing, J. Kabzan, and M. N. Zeilinger. Cautious model predictive control using Gaussian process regression. IEEE Transactions on Control Systems Technology, 28 (6):2736–2743, 2020.

[88] Casimir C. Klimasauskas. Hybrid modeling for robust nonlinear multivariable control. ISA Transactions, 37(4):291–297, 1998.

[89] Zhihao Zhang, Zhe Wu, David Rincon, and Panagiotis D. Christofides. Real-time optimization and control of nonlinear processes using machine learning. Mathematics, 7(10):890, 2019. doi: 10.3390/math7100890.

[90] Ana P. Teixeira, João J. Clemente, António E. Cunha, Manuel J. T. Carrondo, and Rui Oliveira. Bioprocess iterative batch-to-batch optimization based on hybrid parametric/nonparametric models. Biotechnology Progress, 22(1):247–258, 2006.

[91] F. A. Cubillos, G. Acuña, and E.L. Lima. Real-time process optimization based on grey-box neural models. Brazilian Journal of Chemical Engineering, 24:433–443, 2007.

[92] B. Chachuat, B. Srinivasan, and D. Bonvin. Adaptation strategies for real-time optimization. Computers & Chemical Engineering, 33(10):1557–1567, 2009.

[93] A. Marchetti, B. Chachuat, and D. Bonvin. Modifier-adaptation methodology for real-time optimization. Industrial & Engineering Chemistry Research, 48(13):6022– 6033, 2009.

[94] Weihua Gao, Simon Wenzel, and Sebastian Engell. A reliable modifier-adaptation strategy for real-time optimization. Computers & Chemical Engineering, 91:318– 328, 2016.

[95] Martand Singhal, Alejandro G. Marchetti, Timm Faulwasser, and Dominique Bonvin. Real-time optimization based on adaptation of surrogate models. IFAC-PapersOnLine, 49(7):412–417, 2016.

[96] P. Milosavljevic, A. G. Marchetti, A. Cortinovis, T. Faulwasser, M. Mercangöz, and D. Bonvin. Real-time optimization of load sharing for gas compressors in the presence of uncertainty. Applied Energy, 272:114883, 2020.

[97] T. de Avila Ferreira, Z. Wuillemin, A.G. Marchetti, C. Salzmann, J. Van Herle, and D. Bonvin. Real-time optimization of an experimental solid-oxide fuel-cell system. Journal of Power Sources, 429:168–179, 2019.

[98] T. d. A. Ferreira, H. A. Shukla, T. Faulwasser, C. N. Jones, and D. Bonvin. Realtime optimization of uncertain process systems via modifier adaptation and Gaussian processes. In 2018 European Control Conference (ECC), pages 465–470, 2018.

[99] Ehecatl Antonio del Rio Chanona, JE Alves Graciano, Eric Bradford, and Benoit Chachuat. Modifier-adaptation schemes employing Gaussian processes and trust regions for real-time optimization. IFAC-PapersOnLine, 52(1):52–57, 2019.

[100] Ehecatl Antonio del Rio Chanona, Panagiotis Petsagkourakis, Eric Bradford, JE Alves Graciano, and Benoît Chachuat. Real-time optimization meets Bayesian optimization and derivative-free optimization: A tale of modifier adaptation. Computers & Chemical Engineering, 147:107249, 2021.

[101] P. Petsagkourakis, B. Chachuat, and E. A. del Rio-Chanona. Safe real-time optimization using multi-fidelity Gaussian processes. In 60th IEEE Conference on Decision and Control (CDC), pages 6734–6741, 2021.

[102] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.

[103] Dimitri Bertsekas. Dynamic programming and optimal control: Volume I, volume 1. Athena scientific, 2012.

[104] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1):1334– 1373, 2016.

[105] Grady Williams, Paul Drews, Brian Goldfain, James M Rehg, and Evangelos A Theodorou. Aggressive driving with model predictive path integral control. In IEEE International Conference on Robotics and Automation (ICRA), pages 1433– 1440, 2016.

[106] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of Go without human knowledge. Nature, 550(7676):354, 2017.

[107] Lucian Buşoniu, Tim de Bruin, Domagoj Tolić, Jens Kober, and Ivana Palunko. Reinforcement learning for control: Performance, stability, and deep approximators. Annual Reviews in Control, pages 8–28, 2018.

[108] Haeun Yoo, Ha Eun Byun, Dongho Han, and Jay H Lee. Reinforcement learning for batch process control: Review and perspectives. Annual Reviews in Control, 52: 108–119, 2021.

[109] JA Wilson and EC Martinez. Neuro-fuzzy modeling and control of a batch process involving simultaneous reaction and distillation. Computers & Chemical Engineering, 21:S1233–S1238, 1997.

[110] Niket S Kaisare, Jong Min Lee, and Jay H Lee. Simulation based strategy for nonlinear optimal control: Application to a microbial cell reactor. International Journal of Robust and Nonlinear Control: IFAC-Afiliated Journal, 13(3-4):347–363, 2003.

[111] Catalina Valencia Peroni, Niket S Kaisare, and Jay H Lee. Optimal control of a fedbatch bioreactor using simulation-based approximate dynamic programming. IEEE Transactions on Control Systems Technology, 13(5):786–790, 2005.

[112] Jong Min Lee, Niket S Kaisare, and Jay H Lee. Choice of approximator and design of penalty function for an approximate dynamic programming based control approach. Journal of Process Control, 16(2):135–156, 2006.

[113] Jong Min Lee and Jay H Lee. An approximate dynamic programming based approach to dual adaptive control. Journal of process control, 19(5):859–864, 2009.

[114] Hussam Nosair, Yu Yang, and Jong Min Lee. Min–max control using parametric approximate dynamic programming. Control Engineering Practice, 18(2):190–197, 2010.

[115] Yu Yang and Jong Min Lee. A switching robust model predictive control approach for nonlinear systems. Journal of Process Control, 23(6):852–860, 2013.

[116] Lingwei Zhu, Yunduan Cui, Go Takami, Hiroaki Kanokogi, and Takamitsu Matsubara. Scalable reinforcement learning for plant-wide control of vinyl acetate monomer process. Control Engineering Practice, 97:104331, 2020.

[117] Bei Sun, Mingfang He, Yalin Wang, Weihua Gui, Chunhua Yang, and Quanmin Zhu. A data-driven optimal control approach for solution purification process. Journal of Process Control, 68:171–185, 2018.

[118] Yulei Ge, Shurong Li, and Peng Chang. An approximate dynamic programming method for the optimal control of Alkai-Surfactant-Polymer flooding. Journal of Process Control, 64:15–26, 2018.

[119] Jong Woo Kim, Go Bong Choi, and Jong Min Lee. A POMDP framework for integrated scheduling of infrastructure maintenance and inspection. Computers & Chemical Engineering, 112:239–252, 2018.

[120] B Jaganatha Pandian and Mathew Mithra Noel. Control of a bioreactor using a new partially supervised reinforcement learning algorithm. Journal of Process Control, 69:16–29, 2018.

[121] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, and Georg Ostrovski. Human–level control through deep reinforcement learning. Nature, 518 (7540):529, 2015.

[122] Yuan Wang, Kirubakaran Velswamy, and Biao Huang. A novel approach to feedback control with deep reinforcement learning. IFAC-PapersOnLine, 51(18):31–36, 2018.

[123] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[124] Yan Ma, Wenbo Zhu, Michael G Benton, and José Romagnoli. Continuous control of a polymerization system with deep reinforcement learning. Journal of Process Control, 75:40–47, 2019.

[125] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

[126] Tae Hoon Oh, Jong Woo Kim, Sang Hwan Son, Hosoo Kim, Kyungmoo Lee, and Jong Min Lee. Automatic control of simulated moving bed process with deep Qnetwork. Journal of Chromatography A, 1647:462073, 2021.

[127] Panagiotis Petsagkourakis, Ilya Orson Sandoval, Eric Bradford, Dongda Zhang, and Ehecatl Antonio del Rio-Chanona. Reinforcement learning for batch bioprocess optimization. Computers & Chemical Engineering, 133:106649, 2020.

[128] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

[129] Yaoyao Bao, Yuanming Zhu, and Feng Qian. A Deep Reinforcement Learning Approach to Improve the Learning Performance in Process Control. Industrial & Engineering Chemistry Research, page acs.iecr.0c05678, 2021.

[130] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1587–1596. PMLR, 10–15 Jul 2018.

[131] Oguzhan Dogru, Nathan Wieczorek, Kirubakaran Velswamy, Fadi Ibrahim, and Biao Huang. Online reinforcement learning for a continuous space system with experimental validation. Journal of Process Control, 104:86–100, 2021.

[132] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pages 1928–1937. PMLR, 2016.

[133] Tanuja Joshi, Shikhar Makker, Hariprasad Kodamana, and Harikumar Kandath. Twin actor twin delayed deep deterministic policy gradient (TATD3) learning for batch process control. Computers & Chemical Engineering, 155:107527, 2021.

[134] Max Mowbray, Robin Smith, Ehecatl A Del Rio-Chanona, and Dongda Zhang. Using process data to generate an optimal control policy via apprenticeship and reinforcement learning. AIChE Journal, page e17306, 2021.

[135] Haeun Yoo, Boeun Kim, Jong Woo Kim, and Jay H Lee. Reinforcement learning based optimal control of batch processes using Monte-Carlo deep deterministic policy gradient with phase segmentation. Computers & Chemical Engineering, 144:107133, 2021.

[136] Nathan P. Lawrence, Michael G. Forbes, Philip D. Loewen, Daniel G. McClement, Johan U. Backström, and R. Bhushan Gopaluni. Deep reinforcement learning with shallow controllers: An experimental application to PID tuning. Control Engineering Practice, 121:105046, 2022.

[137] Wenbo Zhu, Ivan Castillo, Zhenyu Wang, Ricardo Rendall, Leo H Chiang, Philippe Hayot, and Jose A Romagnoli. Benchmark study of reinforcement learning in controlling and optimizing batch processes. Journal of Advanced Manufacturing and Processing, 4(2):e10113, 2022.

[138] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Of-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1861–1870. PMLR, 10–15 Jul 2018.

[139] Muhammad Kamran Janjua, Haseeb Shah, Martha White, Erfan Miahi, Marlos C Machado, and Adam White. Gvfs in the real world: making predictions online for water treatment. Machine Learning, pages 1–31, 2023.

[140] Richard S Sutton, Joseph Modayil, Michael Delp, Thomas Degris, Patrick M Pilarski, Adam White, and Doina Precup. Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In The 10th International Conference on Autonomous Agents and Multiagent Systems-Volume 2, pages 761–768, 2011.

[141] M Sedighizadeh and A Rezazadeh. Adaptive PID controller based on reinforcement learning for wind turbine control. In Proceedings of World Academy of Science, Engineering and Technology, volume 27, pages 257–262. Citeseer, 2008.

[142] Ignacio Carlucho, Mariano De Paula, Sebastian A. Villar, and Gerardo G. Acosta. Incremental Q-learning strategy for adaptive PID control of mobile robots. Expert Systems with Applications, 80:183–199, 2017.

[143] William J. Shipman and Loutjie C. Coetzee. Reinforcement learning and deep neural networks for PI controller tuning. IFAC-PapersOnLine, 52(14):111–116, 2019.

[144] Athindran Ramesh Kumar and Peter J. Ramadge. DifLoop: Tuning PID controllers by diferentiating through the feedback loop. In 2021 55th Annual Conference on Information Sciences and Systems (CISS), pages 1–6, 2021.

[145] Ayub I. Lakhani, Myisha A. Chowdhury, and Qiugang Lu. Stability-preserving automatic tuning of PID control with reinforcement learning. Complex Engineering Systems, 2(1):3, 2022.

[146] Jay H Lee and Jong Min Lee. Approximate dynamic programming based approach to process control and scheduling. Computers & Chemical Engineering, 30(10-12): 1603–1618, 2006.

[147] Lena Abbasi Brujeni, Jong Min Lee, and Sirish L Shah. Dynamic tuning of PIcontrollers based on model-free reinforcement learning methods. IEEE, 2010.

[148] Marcus AR Berger and JoÃo Viana da Fonseca Neto. Neurodynamic programming approach for the PID controller adaptation. IFAC Proceedings Volumes, 46(11): 534–539, 2013.

[149] Hareem Shafi, Kirubakaran Velswamy, Fadi Ibrahim, and Biao Huang. A Hierarchical Constrained Reinforcement Learning for Optimization of Bitumen Recovery Rate in a Primary Separation Vessel. Computers & Chemical Engineering, page 106939, 2020.

[150] Jong Woo Kim, Byung Jun Park, Tae Hoon Oh, and Jong Min Lee. Model-based reinforcement learning and predictive control for two-stage optimal control of fedbatch bioreactor. Computers & Chemical Engineering, 154:107465, 2021.

[151] Tingwu Wang, Xuchan Bao, Ignasi Clavera, Jerrick Hoang, Yeming Wen, Eric Langlois, Shunshi Zhang, Guodong Zhang, Pieter Abbeel, and Jimmy Ba. Benchmarking model-based reinforcement learning. arXiv preprint arXiv:1907.02057, 2019.

[152] S. Syafiie, F. Tadeo, E. Martinez, and T. Alvarez. Model-free control based on reinforcement learning for a wastewater treatment problem. Applied Soft Computing, 11(1):73–82, 2011.

[153] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, and David Meger. Deep reinforcement learning that matters. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[154] Benjamin Recht. A tour of reinforcement learning: The view from continuous control. Annual Review of Control, Robotics, and Autonomous Systems, 2:253–279, 2019.

[155] Danil V Prokhorov and Donald C Wunsch. Adaptive critic designs. IEEE transactions on Neural Networks, 8(5):997–1007, 1997.

[156] Frank L Lewis and Draguna Vrabie. Reinforcement learning and adaptive dynamic programming for feedback control. IEEE Circuits and Systems Magazine, 9(3), 2009.

[157] Yu Jiang and Zhong-Ping Jiang. Robust adaptive dynamic programming and feedback stabilization of nonlinear systems. IEEE Transactions on Neural Networks and Learning Systems, 25(5):882–893, 2014.

[158] Jong Woo Kim, Byung Jun Park, Haeun Yoo, Tae Hoon Oh, Jay H Lee, and Jong Min Lee. A model-based deep reinforcement learning method applied to finitehorizon optimal control of nonlinear control-afine system. Journal of Process Control, 87:166–178, 2020.

[159] Jong Woo Kim, Tae Hoon Oh, Sang Hwan Son, Dong Hwi Jeong, and Jong Min Lee. Convergence analysis of the deep neural networks based globalized dual heuristic programming. Automatica, 122:109222, 2020.

[160] Evangelos Theodorou, Jonas Buchli, and Stefan Schaal. A generalized path integral control approach to reinforcement learning. Journal of machine learning research, 11(Nov):3137–3181, 2010.

[161] Pierluca D’Oro and Wojciech Jaśkowski. How to learn a useful critic? Model-based action-gradient-estimator policy optimization. Advances in Neural Information Processing Systems, 33, 2020.

[162] Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. In Advances in Neural Information Processing Systems, pages 12498–12509, 2019.

[163] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Ofline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

[164] Scott Fujimoto, David Meger, and Doina Precup. Of-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, pages 2052–2062, 2019.

[165] Mike Huisman, Jan N. van Rijn, and Aske Plaat. A survey of deep meta-learning. Artificial Intelligence Review, 2021.

[166] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1126–1135, 2017.

[167] Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. $\mathrm { { R L } } ^ { 2 } \colon$ : Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.

[168] Daniel G. McClement, Nathan P. Lawrence, Johan U. Backström, Philip D. Loewen, Michael G. Forbes, and R. Bhushan Gopaluni. Meta-reinforcement learning for the tuning of PI controllers: An ofline approach. Journal of Process Control, 118: 139–152, 2022.

[169] Daniele Masti and Alberto Bemporad. Learning nonlinear state–space models using autoencoders. Automatica, 129:109666, 2021.

[170] Matthew Kyle Schlegel, Volodymyr Tkachuk, Adam M White, and Martha White. Investigating action encodings in recurrent neural networks in reinforcement learning. Transactions on Machine Learning Research, 2022.

[171] Junyao Xie, Oguzhan Dogru, Biao Huang, Chris Godwaldt, and Brett Willms. Reinforcement learning for soft sensor design through autonomous cross-domain data selection. Computers & Chemical Engineering, 173:108209, 2023.

[172] Hossein Nejatbakhsh Esfahani, Arash Bahari Kordabad, Wenqi Cai, and Sebastien Gros. Learning-based state estimation and control using mhe and mpc schemes with imperfect models. European Journal of Control, page 100880, 2023.

[173] Edward H. Bras, Tobias M. Louw, and Steven M. Bradshaw. Classical actor-critic applied to the control of a self-regulatory process. IFAC-PapersOnLine, 56(2):7172– 7177, 2023. 22nd IFAC World Congress.

[174] Benjamin Eysenbach, Tianjun Zhang, Sergey Levine, and Russ R Salakhutdinov. Contrastive learning as goal-conditioned reinforcement learning. Advances in Neural Information Processing Systems, 35:35603–35620, 2022.