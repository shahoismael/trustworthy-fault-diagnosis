ORIGINAL PAPER

![](images/cac10475d95ba74eabd68c36dd980a0928bfd072f170f4552eed607a43ac8006.jpg)

# Enhancing fault detection and diagnosis systems for a chemical process: a study on convolutional neural networks and transfer learning

Ana Cláudia Oliveira e Souza<sup>1</sup> · Maurício Bezerra de Souza Jr.<sup>2</sup> · Flávio Vasconcelos da Silva<sup>1</sup>

Received: 28 July 2022 / Accepted: 8 July 2023 / Published online: 26 July 2023

© The Author(s), under exclusive licence to Springer-Verlag GmbH Germany, part of Springer Nature 2023

## Abstract

The study and development of fault detection and diagnosis (FDD) systems are relevant tasks for industrial processes. Another prominent feld is applying deep learning (DL) models to solve engineering problems, such as FDD systems’ design. Often, the preliminary tests are conducted using simulated datasets to verify the chosen methodology and avoid unnecessarily disturbing the real process. Even if the data used come from a computer simulation, it must remain as realistic as possible. In several studies, researchers have used the Tennessee Eastman Process (TEP) benchmark for addressing the application of DL models to build efective FDD frameworks. However, most of them use preexisting datasets, and this presents some drawbacks that can negatively impact the DL model’s training stage. In addition, none of them have evaluated how to adjust the existing FDD model when the process control strategy is changed. This paper presents various topologies of convolutional neural networks (CNNs) to model a FDD system for the TEP benchmark using new datasets. For the frst time, we investigate the performance of fully convolutional networks (FCNs) in the TEP study case. Additionally, we apply transfer learning (TL) to surpass the model inadequacy when the data distribution changes due to an alteration in the process’ closed-loop system.

Keywords Fault detection and diagnosis · Convolutional neural network · Transfer learning · Tennessee Eastman process

## 1 Introduction

In the context of industrial processes, a fault is a deviation of some process variable (feature) or calculated parameter from its standard operating condition (Isermann 2006). Therefore, the occurrence of a fault is an undesired situation that dis turbs the process functioning. The presence of a persisting fault for an extended period can initiate a failure, which is an event in which the system can no longer perform its function properly. Given the processes’ characteristics, chemical industries present hazardous environments in which fres, explosions, and toxic releases can result from malfunctioning sensors, actuators, mechanical equipment, or control systems (Hussin et al. 2015). If a coolant pump fails or a pressure relief valve sticks, lives can be lost, environmental damage can be caused, and the costs to replace the faulty equipment can be quite elevated.

Avoiding any kind of equipment or instrumentation failure is a vital task for engineers in charge of monitoring the operation of chemical process plants. For this reason, the study and development of fault detection and diagnosis (FDD) systems are essential. Various approaches can be used to obtain an efcient FDD system. There are qualitative model-based methods, quantitative model-based methods, and process history-based methods (i.e., data-driven techniques) (Venkatasubramanian et al. 2003a, b, c). The latter have received a great deal of attention in recent years because they allow managers to make decisions based on data analytics evidence (Torrecilla and Romo 2018).

Over the past few years, several machine learning (ML) models with traditional architectures have been used for building data-driven FDD systems, such as artifcial neural networks (Venkatasubramanian et al. 1990: Behbahani et al. 2009; Zhu et al. 2014; Rostek et al. 2015; Heo and Lee 2018; Xie et al. 2019), support vector machines (Shin et al. 2005; Liang and Du 2007; Mahadevan and Shah 2009; Gao et al. 2012), and fuzzy/neuro-fuzzy models (Lau et al. 2010; Karimi and Salahshoor 2012; Subbaraj and Kannapiran 2014; Abdelkrim et al. 2019). Despite the promising results observed with the application of these conventional ML methods, some drawbacks must be highlighted. First, most of them need to be integrated with some statistical techniques, such as principal component analysis (PCA), to reduce data dimensionality and perform data feature extraction because they are unable to work with raw data (LeCun et al. 2015). These are time-consuming tasks that require the availability of engineers with specifc domain knowledge, creating one more obstacle to obtaining a robust and intelligent FDD system. Finally, the traditional ML models do not scale properly when the amount of data available starts to increase, which occurs often due to the digitalization of the industries.

These problems are solved when deep learning (DL) methods are applied. In DL, the data features and patterns are automatically extracted by linear and nonlinear operations carried out in each hidden processing layer of the DL model (LeCun et al. 2015). In addition, the learning stage of deep architectures naturally demands a signifcant amount of data. Regarding the development of FDD systems, applying DL techniques is an ingenious way of taking advantage of the large volume of digital data currently available in process industries.

The Tennessee Eastman Process (TEP) is a well-known benchmarking problem proposed by Downs and Vogel (1993) at the American Institute of Chemical Engineers (AIChE) Conference of 1990. The simulator is based on a real chemical process with 20 process disturbances implemented. It has been extensively used to study plant-wide control strategies, multivariable control, optimization, predictive and nonlinear control, and fault detection and diagnosis. Recently, in some studies, researchers have utilized the data from TEP to develop FDD systems using DL models, such as hierarchical deep neural networks (Xie and Bai 2015), deep belief neural networks (Zhang and Zhao 2017), convolutional neural networks (CNNs) (Wu and Zhao 2018), long short-term memory recurrent neural networks (Park et al. 2019), and bidirectional recurrent neural networks (Zhang et al. 2020). Although they have contributed to a frst performance evaluation of the deep models tested, they all have one drawback: the normal and faulty datasets are obtained independently. Each fault is simulated separately, the disturbance is inserted in the simulation in the same instant for all faults, and each fault's duration is the same. All of these factors combined help distance the simulation from what actually happens in a chemical industry’s routine, where failures can occur at any time, and various stationary states can be observed. It is essential to keep the transition period between one state and another to make the database as realistic as possible. One viable solution is to run one unique continuous simulation in which the faults’ occurrences and durations are randomly chosen. This data acquisition methodology for the TEP was initially proposed by Xavier and Seixas (2018) but has never been applied to train CNNs or conventional ML models.

Deep learning models have indeed stood out in the pattern-recognition feld, but like any model, they also have some shortcomings that must be overcome. The training process and the hyperparameter tuning are two time-consuming stages and therefore should only be performed when necessary. On the other hand, the training and testing data must have the same distributions and belong to the same feature space (Pan and Yang 2009). Therefore, in theory, every time an operational change occurs in the process unit, such as an adjustment to the control structure, the current FDD model should be discarded and a new model should be trained and tuned from scratch. That is costly and impractical. One way of overcoming this obstacle is applying transfer learning (TL). The use of TL in FDD tasks has been reported (Liu and Huang 2019; Chen et al. 2019; Wu and Zhao 2020; Li et al. 2020), but none of the research has focused on studying the application of TL when the entire control strategy of a chemical process is changed.

In regard specifcally to CNNs, another compelling feature of this model is the possibility of converting its conventional topology into a simpler one, where fully connected (FC) layers or pooling layers are no longer necessary. Among the existing approaches, Shelhamer et al. (2017) suggested a network in which convolutional layers replace the last FC layers but pooling layers are still applied. Another possibility consists of not only removing the FC layers but also eliminating the intermediate pooling layers. In this case, the downsampling operation is performed by the convolution layers themselves using an increased stride (Springenberg et al. 2015). For the sake of simplicity in this paper, we reference both schemes as fully convolutional networks (FCNs). FCNs have been extensively applied in semantic segmentation tasks and have shown promising results (Renton et al. 2017; Shelhamer et al. 2017; Tian et al. 2021). However, FCNs have not yet been tested on the TEP case study to model FDD systems. We will compare the performance of traditional CNNs and FCNs in this study to fll this research gap.

The present work aimed to develop a FDD system based on a CNN model and compare its performance to traditional ML architectures to determine whether the deep model has an outstanding behavior. We used the benchmark TEP, generated new datasets, and made them available online. Unlike in previous studies, we obtained normal and faulty data by running continuous simulations. Additionally, we investigated FCNs’ performance for the frst time regarding the TEP problem. Furthermore, we applied TL as a strategy to overcome the necessity of training from scratch the previ ous FDD model obtained when the control strategy used to control the TEP functioning is changed.

## 1.1 Background and related work

In the big data era, the increasing availability of historical process data combined with the recent advances in computer technology, such as software (e.g., friendly programming environments) and hardware (e.g., GPUs, efcient data storage systems), open a promising scenario for the application of data-driven approaches based on artifcial intelligence (AI) algorithms (Shu et al. 2016; Venkatasubramanian 2019). In the research feld of fault detection and diagnosis systems, the evolving nature of AI-based frameworks is another great advantage in modeling over modelbased techniques. As long as representative historical data is available, the developed models can be adjusted to improve their performance. Therefore, it is possible to adapt the AIbased FDD systems when new types of faults are recorded, when the process measurements present a diferent dynamic behavior, or when the process operating mode changes.

In the context of evolving systems, Majdani et al. (2018) proposed an adaptive framework based on artifcial neural networks to model a cyber-physical system for data acquisition and preprocessing purposes. The developed system also has an anomaly detection step. To evaluate the model’s performance, the authors used historical sensor data of a gas turbine from an oil and gas installation. The system was capable of detecting anomalies before their occurrence and the multilayer perceptron (MLP) network presented the best performance results against the other machine learning algorithms tested, such as support vector machines, decision tree random forest and k-nearest neighbors.

Research involving other data-driven approaches has also been conducted. In Santos et al. (2022), the authors developed an evolving fault detection model for dynamic system based on an online learning algorithm, named AutoCloud. The proposed framework was evaluated using data from a pilot plant of level control composed of two connected pressurized tanks, a centrifugal pump and two control valves. As a clustering technique, AutoCloud was responsible for labeling the data stream collected and identifying the diferent types of faults (data corresponding to 12 diferent faults in sensors and actuators was generated). F1-scores above 80% were achieved. Hartert et al. (2010) proposed a semi-supervised framework to monitor a dynamic system where the faulty classes were evolving. The methodology was based on the application of a fuzzy k-nearest neighbours algorithm. The authors evaluated their approach using a database from a switching dynamic system with three functioning modes.

Toubakh and Sayed-Mouchaweh (2015) investigated the application of a clustering algorithm named AuDyC to model a monitoring system to detect drift-like faults in wind turbines. The input data came from a benchmark model. Eighteen types of faults in sensors and actuators were considered. The authors used two drift indicators to detect the moment when the system operating conditions started to diverge from the expected normal behavior. AuDyC is an unsupervised technique based on the calculation of clusters statistical properties. Once defned the properties (such as the mean and the variance-covariance matrix) of the norma operating condition, faulty classes can be continuously identifed in online operation.

Concerning other approaches to model FDD systems, one of the main advantages of applying model-based techniques (Dalton and Patton 1998; McKenzie et al. 1998; Simani and Fantuzzi 2006; Baniardalani et al. 2010; Tidriri et al. 2018; Oliveira et al. 2021; Li et al. 2022) is the inclusion of physical knowledge about the asset being monitored in the fault detection model. However, these techniques require the development of representative models (based on frstprinciples models or state-space models) and this is not a trivial task, because the majority of industrial processes are complex and non-linear. Discrepancies between the real process/equipment and the model can strongly afect the fault detection accuracy. In addition, the computational cost of using model-based approaches in real-time FDD systems is quite elevated (Venkatasubramanian et al. 2003a).

Regarding the application of the Tennessee Eastman Process benchmark to study fault detection and diagnosis systems, Andonovski et al. (2018) applied evolving fuzzy models to develop a process monitoring tool to predict the occurrence of undesirable events. According to the authors approach, fuzzy rules were automatically added or removed from the cloud-based fuzzy models based on the nonlinear process dynamics. The authors evaluated the proposed methodology using simulated TEP datasets and data from a real water-chiller plant (composed of four water chillers and four cooling towers). Online learning (Wang et al. 2006; Pu and Li 2021) and transfer learning (Wang et al. 2019, 2022; Cheng et al. 2020; Wu and Zhao 2020) are other research felds that have been investigated based on the Tennessee Eastman Process benchmark.

As discussed before, these works use datasets already available in public repositories which sufer from the same problem: the fault occurrences are independently simulated. In the present work, we overcome these downsides by generating new continuous datasets, where the occurrence of each fault and its duration are randomly defned by the simulator. Besides that, the proposed framework is entirely datadriven and does not require any phenomenological model. In addition, we investigated the application of transfer learning. Usually, databases from diferent process mode operations (TEP can operate under six distinct modes) have been used to study the application of TL on the TEP benchmark.

However, in this work, we tackled this problem from another perspective. We used two diferent TEP simulators, each with a diferent control strategy implemented, to simulate a change in the distribution of the input variables. Therefore, TL was applied to model a new FDD system when the process’ control strategy changes.

The remainder of the paper is arranged as follows: In Sect. 2, we discuss the basic theory of CNNs and TL. Section 3 introduces the TEP, and Sect. 4 details the proposed FDD framework. In Sect. 5, we present and discuss the results. Finally, Sect. 6 summarizes the discussion, and we discuss possible future works.

## 2 Convolutional neural networks

CNNs (or ConvNets) are among the best-known DL architectures (Lecun et al. 1998; LeCun et al. 2015). Their development was inspired by the experiments conducted by Hubel and Wiesel (1959); they analyzed the activation of neurons in diferent regions of a cat’s visual cortex. Figure 1 shows a traditional CNN’s generic topology and its principal elements. The frst set of layers is responsible for the feature extraction stage. The CNN learns and automatically selects the attributes present in the input matrices during the training process. In this process, the feature maps are generated. They are the outputs of the internal operations (intermediate layers) of the CNN (Goodfellow et al. 2016). The second arrangement of layers classifes the processed information and provides the target classifcation.

The following three operations are responsible for the feature learning stage: convolutions, poolings, and activations. The convolution operation (typically denoted with an asterisk) is a dot-product linear operation (Eq. 1) in which flters (also known as kernels) composed of weights are multiplied over the entire volume of the input data. The parameter responsible for controlling the amount of flter movement over the data volume is called the stride. Through this process, the neural network identifes and isolates the data attributes.

$$
a = \sum_ {i = 1} ^ {M} x _ {i} ^ {l - 1} * k _ {i j} ^ {l} j = 1, \ldots , N\tag{1}
$$

where a is the output of a convolution, x is the feature map the previous layer generates, k represents the flter, and M and N are the numbers of input feature maps and flters applied at the convolutional layer, respectively. After each convolution, an activation function is applied to its output. Among the possibilities, the rectifed linear unit function (ReLU) is the most indicated for use in deep networks (Aggarwal 2018). The ReLU function is given by Eq. 2. Equation 3 corresponds to the calculations that generate the feature maps. Usually, a bias term is also added before applying the activation function (Goodfellow et al. 2016).

$$
f (a) = m a x (0, a)\tag{2}
$$

$$
o _ {j} ^ {l} = f (a) = f \Big (\sum_ {i = 1} ^ {M} x _ {i} ^ {l - 1} * k _ {i j} ^ {l} + b _ {j} ^ {l} \Big) j = 1, \ldots , N\tag{3}
$$

where f is the ReLU activation function, a is the convolution operation output, k is the convolution kernel, x is the input data, b is the bias, M is the number of input matrices, N is the number of kernels applied in each convolution layer and o is the resulting feature map.

Then, a downsampling operation is usually applied. The pooling layers reduce the dimensionality of the feature maps, consequently avoiding overftting and decreasing the calculations’ computational complexity. The approach known as max pooling uses kernels to “walk” through the feature maps’ grid regions, taking these regions’ maximum value.

![](images/3c97e2217e5fb815cb0a36684c81f87bd9a16d0223eda9794a023f2adb4d4853.jpg)  
Fig. 1 Generic topology of a convolutional neural network

Max pooling is the pooling method applied in this work. Therefore, defning the best feature extraction structure is a design task. There are several possibilities: C-R-P, C-R-C-R-P, C-R-P-C-R-P, C-R-C-R-C-R-P, and so on (where C represents a convolution layer, R indicates the ReLU activation function, and P represents a pooling layer). Figures 2 and 3 represent the convolution and the max pooling operations, respectively.

Before the fnal fully connected layers, a fatten layer is applied (Fig. 1). Flatten transforms the last pooled feature maps into a one-dimensional array for feeding the FC layers. This is necessary to adequate the dimension of the information resulting from the feature extraction stage. The FC layers are responsible for classifying the information the convolutional layers process into the desired predicted outputs. These layers also demand the use of activation functions. When using CNNs for multiclass classifcation, the softmax function (Eq. 4) is prevalent. Softmax returns the occurrence probability of each class given the input sample provided. Therefore, the class with the highest probability correspond to the actual process status.

$$
\operatorname{softmax} \left(x _ {i}\right) = \frac {\exp \left(x _ {i}\right)}{\sum_ {j = 1} ^ {n} \exp \left(x _ {j}\right)}\tag{4}
$$

On the other hand, some FCN architectures do not use FC layers for the fnal classifcation of the features learned. Instead, FC layers are converted to Conv layers. This conversion does not change the number of total trainable parameters of the model, and it is possible simply by adjusting the flter kernel size. According to Springenberg et al. (2015), another interesting architectural change involves getting rid of the pooling operations. In this scenario, the intermediate feature maps are downsampled by increasing the stride of some of the convolutional layers. This strided convolution does not increase or decrease the total number of weights and biases to optimize either. Nevertheless, an expressive reduction of the convolution overlapping is an inevitable consequence that must be evaluated.

![](images/e82bb6fc302b52fd4e044e190d94d275916d0981ce3568e264d25730186fe979.jpg)  
Fig. 3 An example of a max pooling operation with stride = 2 between a (4 × 4) feature map and a (2 × 2) flter

The CNN architecture was chosen due to its outstanding results in recent engineering applications (Wu and Zhao 2018; Chen et al. 2019; Liu and Huang 2019). In addition, CNNs do not require knowledge-based feature extraction because they can identify and select the essential features present in the input samples automatically. This property is quite useful because it reduces the total time spent building the FDD system. In this work, the CNNs were modeled using flters of size (3 × 3) and stride = 1. We applied the max pooling operation with stride = 2 and $( 2 \times 2 )$ flters. Regarding the activation functions, ReLU and softmax were used in the intermediate layers and the fnal FC layer, respectively. The other hyperparameters (e.g., number of flters, number of layers, etc.) were defned by an automatic optimization process. Detailed information about the design of the CNNs is given in Sect. 4.2

Fig. 2 Representation of the convolution operation with stride = 1 between a (7 × 7) input and a (3 × 3) flter. (Adapted from Aggarwal 2018)  
![](images/20ddc88cadadfffa61287f00cd65529f48a73f22fba0a74cf5d7c710a38682ab.jpg)

## 2.1 Transfer learning

The application of DL models to any task has three main drawbacks: an extensive database must be available, the model topology design (hyperparameter tuning) is complex and involves several trials, and the model training is timeconsuming. If the input data distribution changes for any reason, the learning model will not be able to predict the outputs correctly (Pan and Yang 2009). However, building a new model from scratch is not the best choice. Fortunately, TL is a technique that can help overcome these issues when a new model is needed.

In TL, the model built (weights and biases) from a source dataset (with distribution ${ \sf P } _ { 1 } )$ is used to generalize the representations observed in another dataset (with distribution ${ \bf P } _ { 2 } )$ . One of the most well-known TL methods is called fnetuning. Fine-tuning uses some pre-trained layers from a source model (which was trained with a large dataset) to improve the generalization of a second (and usually smaller) dataset. Therefore, it is not necessary to retrain the entire model. A generic framework for fne-tuning is represented in Fig. 4. Speaking of a CNN for a classifcation task, one possible fne-tuning choice is to retrain only the FC layers weights. This implies the assumption that the patterns and representations the convolutional layers learn are useful and apply to both settings.

In the research field of FDD, Liu and Huang (2019) applied fne-tuning in a model based on the famous LeNet-5 CNN. They used two datasets – one of motor bearing fault diagnosis and the other of self-priming centrifugal pump fault diagnosis. The authors proved that the representations the model learned from one set were generic enough to allow for fne-tuning using the second databank. Wu and Zhao (2020) used fne-tuning and domain adaptation (another TL approach) to evaluate the application of TL for an FDD task when a chemical process (TEP benchmark) switches its running modes. Li et al. (2020) used two benchmarks (a continuous stirred tank reactor and a plant-wide pulpmill process) to evaluate the performance of domain adaptation when simulated and real data are used to build an FDD system. The goal was to overcome the usual lack of faulty labeled data. Zhang et al. (2017) evaluated the performance of TL for bearing fault diagnosis when the working conditions change (such as the bearing fault diameters and the motor load and speed). The results showed that TL was efective and enabled good fault detection in the new operating conditions tested

![](images/84f2e6d6c7265891eb5778753b4f849af4a867d07ad3ce700d6519f890ecd0a4.jpg)  
Fig. 4 Fine-tuning approach for a CNN classifer

In none of these works did the researchers assess TL’s performance when the process control strategy changed. In chemical plants, it is common to modify closed-loop systems to achieve better control results. Therefore, we address the novel aspect of applying TL when the data distribution changes are caused by using two control strategies in the TEP benchmark.

## 3 The benchmark: Tennessee Eastman process

Based on a real chemical process, the TEP is an industrial plant benchmark Downs and Vogel (1993) proposed at the American Institute of Chemical Engineers (AIChE) Conference of 1990. It comprises fve main unit operations: a reactor, a condenser, a compressor, a liquid-vapor separator, and a stripper (Fig. 5). The process goal is to obtain two main products, G and H, from four reactants. However, it is necessary to separate G and H from the excess reagents, a byproduct, and an inert compound. Four irreversible and exothermic reactions occur:

$$
A _ {(g)} + C _ {(g)} + D _ {(g)} \rightarrow G _ {(l)}\tag{5}
$$

$$
A _ {(g)} + C _ {(g)} + E _ {(g)} \rightarrow H _ {(l)}\tag{6}
$$

$$
A _ {(g)} + E _ {(g)} \rightarrow F _ {(l)}\tag{7}
$$

$$
3 D _ {(g)} \rightarrow 2 F _ {(l)}\tag{8}
$$

In its original form (an open-loop FORTRAN code Downs and Vogel created in 1993), the process involves 53 variables: 22 continuous process measurements (such as temperatures, pressures, and levels), 19 sampled variables (i.e., the compositions of reagents and products in diferent process streams), and 12 manipulated variables. Twenty load changes are also implemented, which are used to simulate the occurrence of faulty periods at any time in the simulation. Table 1 lists these process disturbances.

The present work uses two TEP simulator models for normal and faulty data acquisition. Professor Lawrence Ricker from Washington University revised the frst one (Ricker 2005; Bathelt et al. 2014). This version is implemented in MATLAB/Simulink with the closed-loop strategy Laarson et al. (2001) presented. Despite the code modifcations implemented to increase the number of process disturbances and measured variables, we did not use them in the present work. We only considered the standard 20 possible faults and 53 process measurements. The second version uses the FORTRAN codes Braatz’s group (Russell et al. 2000; Chiang et al. 2000) proposed and Xavier and Seixas (2018) modifed. The second simulator applies the control structure Lyman and Georgakis (1995) designed (Fig. 5), and it runs in Python using a wrapper Câmara (2019) developed.

Laarson et al. (2001) proposed a closed-loop strategy based on PI controllers with cascade systems. The highlights are the presence of a loop dedicated to controlling the reactor pressure and the maintenance of some variables at their constrained values, such as the compressor recycle valve closed, the stripper steam valve closed, and the agitator speed at maximum. On the other hand, Lyman and Georgakis (1995) (Fig. 5) used P and PI controllers with a few cascade systems. The authors left the reactor pressure uncontrolled, and there are no constrained variables (e.g., the compressor recycle valve is manipulated to control the recycle fow rate). Because the closed-loop schemes are distinct, some variables’ behaviors are quite diferent for the two models. Figure 6 compares the diferences between the behavior of the same process measurements under the same conditions but obtained from each simulator described above. This scenario is evaluated with the application of TL.

The chemical plant can operate in 6 distinct modes, but we only simulated mode 1 (G/H mass ratio = 50/50) in this study. Downs and Vogel (1993) fully listed and explained mass and energy balances and the physical properties of reactants and products. We discuss the detailed methodology for data acquisition in the next section.

## 4 FDD framework and data acquisition

## 4.1 Data acquisition and preprocessing

The TEP datasets available online (Ricker 2005; Braatz 2020; Rieth et al. 2017) were not used in this work because they present some disadvantages: They are balanced for normal and faulty data, each fault is simulated separately, the disturbance is introduced in the simulation in the same instant for all faults, and each fault’s duration is the same. Therefore, we adopted the following steps to generate a more realistic dataset in accordance with a chemical process plant’s real daily routine:

Table 1 Tennessee Eastman process disturbances (Downs and Vogel 1993)

<table><tr><td>Disturbance</td><td>Description</td><td>Type</td></tr><tr><td>Fault 1</td><td>A/C feed ratio, B composition constant (stream 4)</td><td>Step</td></tr><tr><td>Fault 2</td><td>B composition, A/C feed ratio constant (stream 4)</td><td>Step</td></tr><tr><td>Fault 3</td><td>D feed temperature (stream 2)</td><td>Step</td></tr><tr><td>Fault 4</td><td>Reactor cooling water inlet temperature</td><td>Step</td></tr><tr><td>Fault 5</td><td>Condenser cooling water inlet temperature</td><td>Step</td></tr><tr><td>Fault 6</td><td>A feed loss (stream 1)</td><td>Step</td></tr><tr><td>Fault 7</td><td>C header pressure loss (stream 4)</td><td>Step</td></tr><tr><td>Fault 8</td><td>A, B, C feed composition (stream 4)</td><td>Random variation</td></tr><tr><td>Fault 9</td><td>D feed temperature (stream 2)</td><td>Random variation</td></tr><tr><td>Fault 10</td><td>C feed temperature (stream 4)</td><td>Random variation</td></tr><tr><td>Fault 11</td><td>Reactor cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>Fault 12</td><td>Condenser cooling water inlet temperature</td><td>Random variation</td></tr><tr><td>Fault 13</td><td>Reaction kinetics</td><td>Slow drift</td></tr><tr><td>Fault 14</td><td>Reactor cooling water valve</td><td>Sticking</td></tr><tr><td>Fault 15</td><td>Condenser cooling water valve</td><td>Sticking</td></tr><tr><td>Fault 16</td><td>Unknown</td><td>Unknown</td></tr><tr><td>Fault 17</td><td>Unknown</td><td>Unknown</td></tr><tr><td>Fault 18</td><td>Unknown</td><td>Unknown</td></tr><tr><td>Fault 19</td><td>Unknown</td><td>Unknown</td></tr><tr><td>Fault 20</td><td>Unknown</td><td>Unknown</td></tr></table>

![](images/1e43f13291a2c0e29f0bcef0fd8e790db1fff60ddfaabe06b96ebf6cf6f29a6a.jpg)  
Fig. 5 Tennessee Eastman Process diagram with the control strategy proposed by Lyman and Georgakis (1995). (Font: Xavier and Seixas 2018; Lyman and Georgakis,1995)

a. We only conducted two continuous simulations, one for training and the other for testing.

b. We chose faults and their durations randomly.

c. Between two periods of fault, there is always one period of normal operation (with its duration also randomly chosen).

d. It is possible to generate a highly imbalanced dataset.

We used MATLAB/Simulink® (MathWorks, Inc., Natwick, MA) to run the Ricker (2005) model and simulated the modifed Braatz (2000) model in a Jupyter Notebook (Anaconda Distribution). For MATLAB and Python simulations, we applied the methodology Xavier and Seixas (2018) proposed, where a binary disturbance matrix (1 – there is an activate fault; 0 – normal operation) is the input of the simulators. Only one disturbance is introduced in the simulation at a time, but the same fault can appear multiple times. We randomly chose a sampling period of 3 min and the state durations (faulty or normal) between 24 and 48 h. From the Python simulator, one set with data corresponding to three years of operation (525,600 samples) and another set with data corresponding to one vear of operation (175,200 samples) were generated for training and testing, respectively. Additionally, we obtained data samples corresponding to one year of operation from the MATLAB model. All the simulated databanks used in this research are available online and can be found in the new\_tep\_datasets repository (Souza et al. 2021).

Figure 7 compares the stripper temperature’s dynamic behavior in two scenarios. In the frst one, the time series comes from a continuous simulation in which normal and faulty operation periods are alternated (sequence of events: normal – fault 10 – normal – fault 11 – normal). In the second scenario, we conducted three separated simulations (a. normal – fault 10; b. normal – fault 11; c. normal), and then the time series were unifed. The diference between running one simulation and separated simulations for each fault is clear. The variables’ dynamic behaviors are afected. In addition, there is a transition period at the beginning of normal operations, which cannot be observed in the individual simulation datasets available on the internet. This kind of nuance must be present in the training dataset of any ML model to be used for the FDD task: otherwise. the model will not be able to classify these periods correctly when put into production. This simple comparison reinforces the importance of running continuous simulations for studying FDD problems.

Because CNNs work better with grid-structured inputs, we transformed the datasets into 2-dimensional matrices. Therefore, the CNN will map the data features and patterns along with time and space domains. We created matrices of size m × n, where m is the time span of data contained in each matrix and n corresponds to the number of process features. In this work, the agitator speed (one of the manipulated variables recorded) was discarded because it remained constant in both simulator models used here. Figure 8 shows the steps of the data preprocessing method.

Fig. 6 Python and MATLAB simulators variables behavior for reactant A feed stream and reactor pressure  
![](images/5da39aebb5f28d9c9f66a7f3bc4fbfa500bff94c16fc8472b55d0fbc6ec06a80.jpg)

![](images/ab4153c588d3aff3efd2b2ac66625feacd4df15ae4ce9ad617e965052b7375f4.jpg)

![](images/074a98c59dacf79c44cf642b58f1efa65c85ceb014cca963342072c3410787eb.jpg)  
Fig. 7 Diferences in the dynamic behavior of the stripper temperature depending on the simulation mode

There are 52 process variables, which is the number of columns in each matrix. Working with square inputs ensures more efcient processing for CNNs (Aggarwal 2018); therefore, the number of rows in the matrices was also 52. We used a sliding window with a step size of 5 to scan the datasets and generate the matrices. Before the matrix’s creation, standardization ensured that every feature presented zero mean and unit variance.

## 4.2 FDD system

Figure 9 shows the framework of the FDD system based on a CNN. After the preprocessing step, we divided the matrices into three sets for training, validation, and testing. We applied the early stopping technique using the validation set to avoid overftting during the learning stage. Adam (Kingma and Ba 2015) is the gradient descendent optimization algorithm used to minimize a categorical crossentropy loss function during the supervised training step. We employed The OPTUNA optimization framework (Akiba et al. 2019) for hyperparameter tuning. The batch size, learning rate, bias and kernel methods of initialization, number of flters of each convolution layer, and network structure (i.e., number of convolution layers and its combination with pooling layers) were the parameters OPTUNA optimized. The chosen objective function was to minimize the loss of the validation dataset. We selected the best parameters to test in each trial using the Tree-structured Parzen Estimator algorithm enabled by the TPESampler on OPTUNA.

![](images/70da6a44a36abcd8eaa3b79f2ed3fcc808db26cd768246d365f4b669ca64f579.jpg)  
Fig. 9 FDD system based on CNN model framework

We organized the classifed instances in normalized confusion matrix heatmaps for assessing the training and testing results. Normalization was necessary due to the highly unbalanced datasets generated. We calculated recall (also known as fault detection rate), precision, and the f1-score for each target class according to Eq. (9), Eq. (10), and Eq. (11), respectively. The counting of true negatives (TN), true positives (TP), false negatives (FN), and false positives (FP) came from the confusion matrices built.

Fig. 8 Data preprocessing methodology scheme  
![](images/2ba394826f642f93009bf1cac01a94f5b17e86c0df46ea04891420e549502251.jpg)

$$
r e c a l l = \frac {T P}{T P + F N}\tag{9}
$$

$$
p r e c i s i o n = \frac {T P}{T P + F P}\tag{10}
$$

$$
F 1 s c o r e = \frac {2 \times p r e c i s i o n \times r e c a l l}{p r e c i s i o n + r e c a l l}\tag{11}
$$

Recall answers the following question: of all the real faulty instances, how many of those did the FDD system correctly classify? Therefore, it is a way to verify the rate of missed detections. On the other hand, the precision gives information about the false alarms, i.e., normal instances classifed as faults. Because both metrics are important for FDD tasks, their combination is also calculated and is given by the f1-score. These are the key performance indicators (KPIs) used to evaluate the modeled FDD system.

In this investigation, we tested the application of TL in a scenario in which the source domain is changed, i.e., the input data distribution used to train the current model in production is no longer the same as the real-time input data being collected during the chemical plant operation. We simulated this situation by changing the control structure used to keep the process stable. Therefore, we used the fnetuning method according to the following steps:

## 5 Transfer learning experiment

Step 1. Train the CNN with data from the Python simulator, which applies the control structure Lyman and Georgakis (1995) designed.

Step 2. Keep all the weights of the convolutional and pooling layers adjusted by the training step as well as the topology and the hyperparameters tuned.

Step 3. Apply fne-tuning for training only the CNN FC layer, using the data from the MATLAB simulator with the closed-loop strategy Laarson et al. (2001) proposed.

Step 4. Test the new model.

We used the Python 3.7.4 programming language with the open-source libraries Keras and TensorFlow to develop the FDD system. We conducted the simulations for data acquisition as well as the training and testing stages of the machine and deep learning models on a computer with an Intel i7-9700 CPU (9th gen) 3.00 GHz 12 MB, 32 GB RAM, and Ubuntu 20.04.1 LTS (8 cores and 8 threads). In the stages involving the training of the AI models and the optimization of their hyperparameters, the processing was parallelized among the eight available threads. In the next section, we will present and discuss the elapsed real time demanded by each investigated model in the steps of training and validation.

## 6 Results and discussion

The Python simulator generated four years of data samples (700,800 instances totalizing 673 MB of data). Table 2 lists the number of matrices obtained after the preprocessing stage for each possible process status. It is a highly unbalanced dataset with many more examples of normal operation than examples of each of the 20 simulated disturbances. Additionally, it is important to highlight that we performed no cleaning operations on the data: we did not remove outliers, we did not attenuate the variables’ noise, and we maintained the transition periods between the faulty operation and the normal ones. We made these choices to keep the datasets realistic and mimic the daily reality of a process unit as closely as possible. Because their simulations had a limited time range, faults 6, 12, and 18 have substantially fewer matrices. The process simulation shuts down if these faults remain active for periods longer than 5 h.

To design the CNN model, we kept some hyperparameters constant according to Aggarwal’s (2018) recommendation. Therefore, the convolutional layers use kernels of size (3 × 3) with stride set to 1, and the kernels of pooling layers have a size of (2 × 2) and stride set to 2. ReLU is

Table 2 Number of matrices for each process status obtained from the Python simulator

<table><tr><td>Process status</td><td>Number of training matrices</td><td>Number of testing matrices</td></tr><tr><td>Normal</td><td>49,058</td><td>16,011</td></tr><tr><td>Fault 1</td><td>2359</td><td>923</td></tr><tr><td>Fault 2</td><td>3573</td><td>247</td></tr><tr><td>Fault 3</td><td>2638</td><td>639</td></tr><tr><td>Fault 4</td><td>2314</td><td>1557</td></tr><tr><td>Fault 5</td><td>2628</td><td>1191</td></tr><tr><td>Fault 6</td><td>90</td><td>50</td></tr><tr><td>Fault 7</td><td>3036</td><td>854</td></tr><tr><td>Fault 8</td><td>3013</td><td>983</td></tr><tr><td>Fault 9</td><td>3261</td><td>1090</td></tr><tr><td>Fault 10</td><td>3062</td><td>1992</td></tr><tr><td>Fault 11</td><td>2293</td><td>938</td></tr><tr><td>Fault 12</td><td>66</td><td>26</td></tr><tr><td>Fault 13</td><td>2229</td><td>1157</td></tr><tr><td>Fault 14</td><td>3310</td><td>829</td></tr><tr><td>Fault 15</td><td>2624</td><td>517</td></tr><tr><td>Fault 16</td><td>2682</td><td>701</td></tr><tr><td>Fault 17</td><td>2629</td><td>850</td></tr><tr><td>Fault 18</td><td>73</td><td>39</td></tr><tr><td>Fault 19</td><td>2735</td><td>902</td></tr><tr><td>Fault 20</td><td>3603</td><td>740</td></tr><tr><td>Total</td><td>97,276</td><td>32,236</td></tr></table>

the activation function applied after each convolution layer, and the last FC layer uses the softmax function. We also used padding. We selected the hyperparameters not to keep constant through the dynamic search space OPTUNA built. During optimization, the number of flters in the third Conv layer (as well as its existence or nonexistence), the number of neurons in the frst FC layer (as well as its existence or nonexistence), and the optimizer learning rate were the three hyperparameters that most infuenced the minimization of the validation loss (Fig. 10).

The relationship between the objective function value and the order in which each set of hyperparameters was tested is represented in a slice plot for the three most relevant parameters (Fig. 11). It is noteworthy that the frst trials OPTUNA evaluated comprised models with fewer flters in the frst FC layer and higher learning rates (0.01 to 0.1). However, this choice of hyperparameters led to greater validation losses, which is not desirable. Therefore, as the optimization proceeded, the application of lower learning rates and the improvement of the fnal classifcation power by increasing the number of neurons in the frst FC layer helped improve the model’s performance, resulting in a decrease in the validation loss. Table 3 shows some of the topologies OPTUNA automatically tested and each one’s average metrics evaluated (C refers to convolutional layers, P represents the pooling layers, and FC indicates the FC layers). The training learning rates (LRs) applied are also listed. We conditionally formatted some tables in this paper so that light green markers indicate the best results and red markers indicate the worst results.

Table 3. Averageresults for the CNN model’s design stage

<table><tr><td>Model</td><td>Topology</td><td>Training time (min)*</td><td>Valid. precision (%)</td><td>Valid. recall (%)</td></tr><tr><td>M1</td><td>C(20)-C(30)-P-FC(21)-LR(0.001)</td><td>79</td><td>80.4</td><td>77.7</td></tr><tr><td>M2</td><td>C(20)-C(40)-C(40)-P-FC(21)-LR(0.0003)</td><td>112</td><td>81.6</td><td>82.9</td></tr><tr><td>M3</td><td>C(10)-C(30)-C(40)-P-FC(21)-LR(0.0017)</td><td>91</td><td>84.1</td><td>83.3</td></tr><tr><td>M4</td><td>C(10)-C(20)-P-FC(21)-LR(0.004)</td><td>20</td><td>79.9</td><td>81.6</td></tr><tr><td>M5</td><td>C(20)-C(30)-P-C(50)-P-FC(30)-FC(21)-LR(0.0005)</td><td>108</td><td>81.9</td><td>85.2</td></tr><tr><td>M6</td><td>C(20)-C(20)-C(40)-P-FC(21)-LR(0.018)</td><td>63</td><td>75.1</td><td>82.7</td></tr><tr><td>M7</td><td>C(20)-C(30)-C(50)-P-FC(21)-LR(0.001)</td><td>73</td><td>80.2</td><td>77.8</td></tr><tr><td>M8</td><td>C(20)-C(30)-C(40)-P-FC(21)-LR(0.021)</td><td>156</td><td>73.7</td><td>82.5</td></tr><tr><td>M9</td><td>C(20)-C(30)-C(50)-P-FC(50)-FC(21)-LR(0.001)</td><td>99</td><td>80.9</td><td>80.8</td></tr><tr><td>M10</td><td>C(20)-C(30)-C(50)-P-FC(30)-FC(21)-LR(0.0005)</td><td>155</td><td>79.6</td><td>82.7</td></tr></table>

\* Elapsed real time.  
\* Elapsedreal time

Fig. 10 OPTUNA’s importance plot  
![](images/4edf8ba46985eedfcaa22ffeff3e1f461aa0bff886d64ce07c9089d03636607e.jpg)

Fig. 11 OPTUNA’s slice plot for the most important hyperparameters regarding the optimization  
![](images/f17c285ca8563a03583f2f2df679f7adf01fccf5a901000f2f714d8fce363f3b.jpg)

From all the topologies tested, we chose Model 5 (M5) based on its validation performance. Although Model 3 showed greater precision. M5 had a lower rate of missed detections, i.e., higher recall, which is desirable in FDD systems. Higher learning rates (such as those used to train M6 and M8) led to worse results, especially concerning the classifcation of the normal operation instances. Models with lower learning rates (i.e., < 0.01) performed better on the validation dataset and achieved the most promising KPIs. M5 comprised 273,361 trainable parameters, and it took 1 h and 48 min (elapsed real-time) to adjust them during the learning step. Table 4 lists the test performance metrics for each process status, and Fig. 12 shows the normalized confusion matrix of the Model 5 ofine test.

\* Elapsed real time.

Model 5 correctly classifed (with f1-score > 80%) 15 of the 21 existing process statuses. Its average KPIs for the test set were 78.3% precision, 80.5% recall, and 78.9% f1-score. The problematic faults were Fault 3 (step in D feed temperature), Fault 9 (random variation in D feed temperature), Fault 15 (condenser cooling water sticking valve), Fault 16 (unknown), and Fault 18 (unknown). Faults 3, 9, and 15 are already known in the literature as incipient faults because they present responses very similar to the noise applied in every variable of the TEP benchmark. Therefore, when these disturbances are active, the input variables’ dynamic behavior does not change enough to be distinguished from the normal operation. The same applies to Fault 16. Other works have also shown difculties in detecting and diagnosing these faults (Zhang and Zhao 2017: Wu and Zhao 2018: Xavier and Seixas 2018). Fault 18 is not known as a problematic fault to classify. The authors believe that the CNN model could not correctly detect it due to the small number of input matrices of this fault used in the training stage. Although Faults 6 and 12 also had restricted time ranges to avoid the simulation shutdown, these two faults greatly afect the TEP variables’ dynamic behavior, which is not true for Fault 18.

Table 4 Detailed test results with Model 5

<table><tr><td>Process status</td><td>Precision (%)</td><td>Recall (%)</td><td>f1-score (%)</td><td>TP</td><td>TN</td><td>FP</td><td>FN</td></tr><tr><td>Normal</td><td>85.3</td><td>72.5</td><td>78.4</td><td>11,613</td><td>14,217</td><td>2008</td><td>4398</td></tr><tr><td>Fault 1</td><td>99.8</td><td>100.0</td><td>99.9</td><td>923</td><td>31,311</td><td>2</td><td>0</td></tr><tr><td>Fault 2</td><td>97.6</td><td>100.0</td><td>98.8</td><td>247</td><td>31,983</td><td>6</td><td>0</td></tr><tr><td>Fault 3</td><td>16.4</td><td>11.7</td><td>13.7</td><td>75</td><td>31,215</td><td>382</td><td>564</td></tr><tr><td>Fault 4</td><td>99.3</td><td>98.9</td><td>99.1</td><td>1540</td><td>30,668</td><td>11</td><td>17</td></tr><tr><td>Fault 5</td><td>100.0</td><td>99.9</td><td>100.0</td><td>1190</td><td>31,045</td><td>0</td><td>1</td></tr><tr><td>Fault 6</td><td>100.0</td><td>100.0</td><td>100.0</td><td>50</td><td>32,186</td><td>0</td><td>0</td></tr><tr><td>Fault 7</td><td>100.0</td><td>100.0</td><td>100.0</td><td>854</td><td>31,382</td><td>0</td><td>0</td></tr><tr><td>Fault 8</td><td>92.8</td><td>92.4</td><td>92.6</td><td>908</td><td>31,183</td><td>70</td><td>75</td></tr><tr><td>Fault 9</td><td>7.6</td><td>11.1</td><td>9.1</td><td>121</td><td>29,685</td><td>1461</td><td>969</td></tr><tr><td>Fault 10</td><td>93.3</td><td>85.6</td><td>89.3</td><td>1705</td><td>30,121</td><td>123</td><td>287</td></tr><tr><td>Fault 11</td><td>99.7</td><td>98.9</td><td>99.3</td><td>928</td><td>31,295</td><td>3</td><td>10</td></tr><tr><td>Fault 12</td><td>81.3</td><td>100.0</td><td>89.7</td><td>26</td><td>32,204</td><td>6</td><td>0</td></tr><tr><td>Fault 13</td><td>87.5</td><td>93.5</td><td>90.4</td><td>1082</td><td>30,924</td><td>155</td><td>75</td></tr><tr><td>Fault 14</td><td>100.0</td><td>100.0</td><td>100.0</td><td>829</td><td>31,407</td><td>0</td><td>0</td></tr><tr><td>Fault 15</td><td>4.6</td><td>21.1</td><td>7.6</td><td>109</td><td>29,474</td><td>2245</td><td>408</td></tr><tr><td>Fault 16</td><td>39.5</td><td>56.8</td><td>46.6</td><td>398</td><td>30,925</td><td>610</td><td>303</td></tr><tr><td>Fault 17</td><td>97.1</td><td>98.9</td><td>98.0</td><td>841</td><td>31,361</td><td>25</td><td>9</td></tr><tr><td>Fault 18</td><td>48.8</td><td>53.8</td><td>51.2</td><td>21</td><td>32,175</td><td>22</td><td>18</td></tr><tr><td>Fault 19</td><td>99.9</td><td>97.0</td><td>98.4</td><td>875</td><td>31,333</td><td>1</td><td>27</td></tr><tr><td>Fault 20</td><td>94.7</td><td>98.6</td><td>96.6</td><td>730</td><td>31,455</td><td>41</td><td>10</td></tr></table>

![](images/5edc697876044d2d01c0b198933f352ecffac7fc9ad60c89b3dc5bcb4ab5d5f0.jpg)  
Fig. 12 Normalized confusion matrix for Model 5 with the test set

It is worth noting that despite Model 5’s promising performance, its detection of the normal operation periods could be better. The recall for the normal status was 74.21%, meaning that M5 raises approximately 25% of false alarms (i.e., among the total number of normal instances in the test set, M5 misclassifed 25% of them). The confusion matrix (Fig. 12) indicates that the normal matrices are being con fused with instances of Faults 9 and 15, both incipient faults. To tackle this weakness of Model 5, we investigated FCNs. We provide the results in the following subsection.

Because the method for running the simulations in this study was diferent and the datasets used in the training stage are not the same, it is not fair to compare our results directly to those presented in other works. Therefore, we implemented and tested three conventional ML models. The goal is to determine whether the FDD system based on a CNN model (a DL model) outperforms some traditional ML methods. The chosen architectures were random forest (RF), multilayer perceptron (MLP), and support vector machine (SVM). Unlike CNNs, these models require a step of feature extraction and dimensionality reduction because they cannot automatically select which features are important for the training process. Therefore, we used principal component analysis (PCA). The frst 10 PCs could explain approximately 80% of the system variance. Therefore, we reduced the number of input variables from 52 to 10.

The procedure for building the models, from training to testing, is the same as that which Fig. 9 presents. Table 5 lists the best topologies tuned for each model (RF, SVM, and MLP) and the average testing results. Some of each model’s hyperparameters are also highlighted. For the MLP, the number of hidden layers and the number of neurons in each layer were evaluated; for the RF, we considered the number of trees (i.e., number of estimators) and the max depth allowed for each tree. Finally, for the SVM, we tested two types of kernels. Table 6 shows some of the results of the best models (MLP1, RF1, and SVM1) for each process status.

Table 5. Averageresults for the traditional machine learning models evaluated

<table><tr><td>Model</td><td>Topology</td><td>Training time (min)*</td><td>Valid. precision (%)</td><td>Valid. recall (%)</td></tr><tr><td>MLP1</td><td>FC(40)-FC(30)-FC(21)-LR(0.0001)</td><td>44</td><td>75.1</td><td>71.6</td></tr><tr><td>MLP2</td><td>FC(50)-FC(40)-FC(40)-FC(21)-LR(0.0001)</td><td>15</td><td>75.0</td><td>71.2</td></tr><tr><td>RF1</td><td>n_estimators(100)-max_depth(50) w/ Gini</td><td>16</td><td>77.7</td><td>64.0</td></tr><tr><td>RF2</td><td>n_estimators(200)-max_depth(50) w/ Gini</td><td>28</td><td>76.4</td><td>60.9</td></tr><tr><td>SVM1</td><td>Kernel(RBF)</td><td>431</td><td>76.4</td><td>54.8</td></tr><tr><td>SVM2</td><td>Kernel(Linear)</td><td>749</td><td>70.6</td><td>49.3</td></tr></table>

\* Elapsed real time.  
\* Elapsed real time

None of the architectures evaluated returned better results than the best CNN model trained (Model 5). However, some interesting outcomes must be highlighted. Despite the previous dimensionality reduction, the SVM takes a long time in the training stage (approximately 7 h). Even then, it could not correctly classify the faulty test samples. Therefore, the SVM was the model with the lowest recall (54.8%), i.e., the highest missed detection rate. Although the MLP and RF models took less time to train, they also presented lower recall than the CNN model. This is a highly undesirable aspect because the FDD model must be capable of distinguishing the normal operation from the disturbed operation. Increasing the model complexity did not pay of for the MLP and RF models. Even though MLP2 contains more hidden layers and neurons, its validation results were the same as those of MLP1. The same observation applies for RF2, which despite having been trained with more estimators, does not outperform RF1.

Faults 3, 9, and 15 remain with low detection rates for the three models, as Table 6 shows, but the MLP could correctly classify Fault 16 with 91.7% and 81.9% precision and recall, respectively. On the other hand, some faults that the CNN model correctly classifed with great precision and recall presented worse classifcation results, such as Faults 8, 10, 11, 12, 13, 19, and 20. Figure 13 shows the confusion matrix for the MLP1 model, the best among those listed in Table 6.

Table 6 for the best traditional machine learning models built

<table><tr><td rowspan="2">Process status</td><td colspan="2">Random Forest</td><td colspan="2">Multilayer Perceptron</td><td colspan="2">Support Vector Machine</td></tr><tr><td>Precision (%)</td><td>Recall (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>Precision (%)</td><td>Recall (%)</td></tr><tr><td>Normal</td><td>73.6</td><td>66.9</td><td>81.6</td><td>96.5</td><td>70.5</td><td>98.2</td></tr><tr><td>Fault 1</td><td>98.4</td><td>98.0</td><td>99.2</td><td>95.5</td><td>99.2</td><td>97.6</td></tr><tr><td>Fault 2</td><td>97.0</td><td>97.9</td><td>83.9</td><td>97.2</td><td>74.6</td><td>10.1</td></tr><tr><td>Fault 3</td><td>3.7</td><td>19.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Fault 4</td><td>96.5</td><td>97.1</td><td>96.8</td><td>96.2</td><td>92.2</td><td>98.6</td></tr><tr><td>Fault 5</td><td>98.5</td><td>92.4</td><td>98.7</td><td>99.1</td><td>98.9</td><td>99.6</td></tr><tr><td>Fault 6</td><td>98.7</td><td>90.8</td><td>93.6</td><td>97.7</td><td>98.7</td><td>89.6</td></tr><tr><td>Fault 7</td><td>99.8</td><td>99.2</td><td>98.6</td><td>99.6</td><td>99.9</td><td>99.4</td></tr><tr><td>Fault 8</td><td>90.3</td><td>74.6</td><td>77.1</td><td>78.7</td><td>89.4</td><td>84.7</td></tr><tr><td>Fault 9</td><td>6.9</td><td>4.8</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Fault 10</td><td>93.7</td><td>51.1</td><td>93.6</td><td>80.2</td><td>96.8</td><td>12.6</td></tr><tr><td>Fault 11</td><td>90.1</td><td>74.7</td><td>88.5</td><td>78.3</td><td>96.4</td><td>54.7</td></tr><tr><td>Fault 12</td><td>72.6</td><td>21.8</td><td>60.2</td><td>43.8</td><td>86.4</td><td>37.1</td></tr><tr><td>Fault 13</td><td>87.7</td><td>72.0</td><td>86.4</td><td>80.9</td><td>89.2</td><td>76.5</td></tr><tr><td>Fault 14</td><td>98.6</td><td>97.5</td><td>96.2</td><td>98.7</td><td>99.0</td><td>93.9</td></tr><tr><td>Fault 15</td><td>3.1</td><td>12.1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Fault 16</td><td>97.1</td><td>49.8</td><td>91.7</td><td>82.0</td><td>98.2</td><td>10.2</td></tr><tr><td>Fault 17</td><td>89.8</td><td>88.4</td><td>91.2</td><td>91.8</td><td>87.7</td><td>88.2</td></tr><tr><td>Fault 18</td><td>58.2</td><td>7.9</td><td>49.7</td><td>12.9</td><td>35.1</td><td>3.4</td></tr><tr><td>Fault 19</td><td>85.5</td><td>66.0</td><td>96.6</td><td>88.2</td><td>97.0</td><td>33.9</td></tr><tr><td>Fault 20</td><td>92.3</td><td>62.4</td><td>93.9</td><td>86.3</td><td>95.6</td><td>61.7</td></tr></table>

The comparison of the FDD system based on a CNN model with traditional ML architectures (MLP, RF and SVM) confirmed the DL model’s outstanding performance. Concerning the results presented by CNNs and MLPs, despite both being deep learning models, they have distinct architectures and characteristics, which could be related to the performance diferences observed. First of all, the input data structure provided to each model was diferent. In the present work, 2D CNNs were investigated; therefore, the datasets were transformed into 2-dimensional matrices of sizes m × n (with m = n = 52). On the other hand, this transformation was not applied to train the MLPs (as well as the RFs and SVMs models). Consequently, each data instance fed into the CNN contains information from multiple timestamps, as the sensor measurements were stacked into 52-row matrices. As faults also propagate spatially, not just temporally, we believe that the CNN benefted from the structure of the provided data.

In addition, the convolution and pooling operations of CNNs are primarily based on the understanding of spatial relationships in the input data (i.e., their learning procedure is based on fnding spatial dependencies in local regions of the input instances) (Aggarwal 2018). Therefore, the second point related to the better performances of CNNs is their automatic feature extraction procedure. Even though PCA was applied to the data before training MLPs, SVMs, and RFs, feature selection is an intrinsic process carried out by the internal layers of the CNN, which considers the temporal and spatial dimensions of the problem.

![](images/e7e4993479132278ed11225f7055514da6345b64353fa0f7c692aaf2a1561d68.jpg)  
Fig. 13 Normalized confusion matrix for the MLP1 model with the test set

Finally, according to LeCun et al. (2015), CNNs present better generalization results compared to networks with fully connected layers (e.g., MLPs). They are less prone to overftting due to the convolution operation and the use of flters and strides, which enable parameter sharing since the states of internal layers are not fully connected to the ones of previous layers. This idea of sparse connections also reduces the computational complexity of the CNNs’ training stages. Other studies have applied CNNs and MLPs to model regression and classifcation tasks, and CNNs have demonstrated advantageous aspects (Medina et al. 2017; Botalb et al. 2018; Khalifani et al. 2022).

It is essential to emphasize that the presented results are based on training, validation, and testing datasets with different characteristics from those commonly used in other published works. We conducted the simulations to obtain datasets that were as realistic as possible. Therefore, we kept the transition period at the beginning of normal operation, and the previous process states afected the variables dynamic behaviors. All these changes make the training process difcult because the distinction between normal and fault operating patterns becomes more complex, yet good results were achieved.

## 6.1 Investigation of fully convolutional networks

As mentioned earlier, Model 5, the best CNN topology (with optimal hyperparameters that OPTUNA defned), despite the good initial results, could still be improved in some respects, such as normal operation detection. To address this issue, we investigated FCNs. We evaluated two strategies. The frst model (FCN1) keeps the pooling operations, but the two FC layers are converted into convolutional layers with stride 1 and kernel sizes of (13 × 13) and (1 × 1), respectively. In the second model (FCN2), in addition to the removal of FC layers, the pooling layers are eliminated. Therefore, FCN2 has some Conv layers with an increased stride (equal to 2) to downsample the intermediate feature maps. Every hyperparameter previously defned was kept the same. Therefore, a straightforward comparison between the models is possible because they all have the same total trainable parameters (273,361 weights and biases). Table 7 presents the average results of the training and validation stages.

Only the conversion of the FC layers in the convolutional layers did not return good results. In fact, deterioration occurred in the model performance, refected in the KPIs evaluated. On the other hand, we achieved promising results in the validation set with the use of a “strided convolution” (Springenberg et al. 2015). The increase in precision observed with FCN2 means that the numbers of false positives decreased, i.e., fewer normal instances were incorrectly classifed as any of the 20 existing faults. In addition, the training stage of FCN2 was surprisingly faster than that of M5 regarding the elapsed real time. Despite a decrease of 1.6% in the validation recall, FCN2 was selected and evaluated with the ofine test dataset. Figure 14 is the normalized confusion matrix for FCN2 in the out-of-sample instances.

In the test set, the average KPIs for FCN2 were 80.6% precision, 83.7% recall, and 81.6% f1-score. Compared to M5’s performance, the detection of Faults 3, 8, 9, 10, 15, 16, and 18 improved. It is worth noting that the faults that benefted most from the conversion of M5 into its fully convolutional architecture were Faults 3 and 16, both recognized as challenging faults in the literature. The good results M5 achieved for Faults 1, 2, 4, 5, 6, 7, 11, 12, 14, 17, and 20 remained unchanged in FCN2. The only classes in which FCN2 could not provide better detection and diagnosis than M5 were Faults 13 and 15. Regarding the normal operation, the improvement was not expressive (M5 f1-score of 78.4% versus FCN2 f1-score of 80.8%), but we achieved it after an elapsed real training time 80% more quickly than the M5 one.

Table 8 summarizes the performance indicators of the best AI-based FDD models obtained in this work. Among the artifcial intelligence algorithms investigated, the performance of the convolutional neural networks was superior to model the FDD system. Even the multilayer perceptron networks (which are also a deep learning architecture) did not show satisfactory results. As discussed in the previous sections, certain architectural characteristics of CNNs may have contributed to this model achieving better results, such as the application of input data with spatial and temporal dimensions (2D matrices) in the training stage, the automatic feature selection, and the parameter sharing observed in C and P layers, which reduces the tendency to overft. In

Table 7. Comparisonbetween the best traditional CNN model and its FCN forms

<table><tr><td>Model</td><td>Topology</td><td>Training time (min)*</td><td>Valid. precision (%)</td><td>Valid. recall (%)</td></tr><tr><td>M5</td><td>C(20)-C(30)-P-C(50)-P-FC(30)-FC(21)-LR(0.0005)</td><td>108</td><td>81.9</td><td>85.2</td></tr><tr><td>FCN1</td><td>C(20)-C(30)-P-C(50)-P-C(30)-C(21) – LR(0.0005)</td><td>112</td><td>81.0</td><td>83.8</td></tr><tr><td>FCN2</td><td>C(20)-C(30)(2)-C(50)(2)-C(30)-C(21) - LR(0.0005)</td><td>20</td><td>83.8</td><td>84.9</td></tr></table>

\* Elapsed real time.

![](images/b25fdafb554abc0635f012439d5c3ed8fc065d0a1a19959c3c6d9ee259a4615c.jpg)  
Fig. 14 Normalized confusion matrix for the FCN2 model with the test set

Table 8 Summary of results for the best FDD models using the test dataset

<table><tr><td>Model</td><td>Test precision (%)</td><td>Test recall (%)</td><td>Test f1-score (%)</td></tr><tr><td>M5</td><td>78.3</td><td>80.5</td><td>78.9</td></tr><tr><td>FCN2</td><td>80.6</td><td>83.7</td><td>81.6</td></tr><tr><td>MLP1</td><td>74.1</td><td>71.1</td><td>72.0</td></tr><tr><td>RF1</td><td>70.4</td><td>63.4</td><td>67.3</td></tr><tr><td>SVM1</td><td>61.8</td><td>54.8</td><td>57.7</td></tr></table>

addition, it is important to highlight the promising results shown by the fully convolutional networks. In FCN2. FC layers were not employed, and the pooling operation was performed by the convolution layers themselves.

Regarding the computational complexity of the modeling procedure, the neural network architectures (i.e., CNNs and MLPs) investigated did not require more than four intermediate layers to achieve the best observed results (see Tables 3 and 5). The champion model, the convolutional neural network identifed as model M5, took less than 2 h to complete its training and validation stages, and its fully convolutional topology, model FCN2, took only 20 min to train (elapsed real time). Therefore, given the size of the training dataset and the hardware used, the computational cost for training the AI-based FDD system proposed in this work was low.

## 6.2 Application of transfer learning

To evaluate the fne-tuning performance when the TEP closed-loop system is changed, we used data corresponding to one year of operation from the MATLAB simulator. We used the frst trimester of this dataset for training and validation (approximately 50,900 samples), and we retained the next six months (approximately 85,700 samples) as an outof-sample set for testing. As a result of the preprocessing stage, the training and validation sets comprised 8073 and 1425 matrices, respectively. Therefore, the number of input matrices used to fne-tune the fnal layers is much lower than the number of input samples used to train the entire source model with the Python dataset. We intentionally designed this scenario to simulate the reality of a lack of data when the input data distribution is changed (in this case, represented by the alteration of the closed-loop strategy).

We chose the source model with the best results achieved thus far, i.e., the fully convolutional topology with increased stride in some layers, FCN2. Table 9 presents a comparison between the source model results and the fine-tuned models results. The only hyperparameter change was the learning rate (LR). It is a good practice to run the fne-tuning process with smaller LRs to avoid losing any important representation in the CNN’s classifcation section. We kept everything else the same as in the FCN2. The best fne-tuned model (FCN2-TF2) used a learning rate equal to 0.0005 and returned 79.8% and 76.6% precision and recall, respectively. Although the FCN2-TF2 metrics were not superior to those of those of the original model (FCN2), the results are still satisfying for a surrogate model while more data are acquired.

To investigate the performance of the TL approach over time, when more data became available, we fne-tuned two more models. FCN2-TF4 uses 11,300 matrices for training, which corresponds to approximately fve months of operation, and we trained FCN2-TF5 with data corresponding to seven months of operation. These experiments are related to a subfeld of machine learning methods called online learning. In online learning, the models are incrementally trained using new data instances that arrive in a sequential way (Hoi et al. 2021). Application of online learning is suitable for modeling dynamic systems, where the behavior of the data employed is always evolving.

Table 9 Average results for the model fne-tuning experiment

<table><tr><td>Model</td><td>Number of input matrices</td><td>Learning rate</td><td>Test precision (%)</td><td>Test recall (%)</td></tr><tr><td>FCN2*</td><td>70,038</td><td>0.0005</td><td>80.6</td><td>83.7</td></tr><tr><td>FCN2-TF1</td><td>8073</td><td>0.0005</td><td>79.7</td><td>75.3</td></tr><tr><td>FCN2-TF2</td><td>8073</td><td>0.0001</td><td>79.8</td><td>76.6</td></tr><tr><td>FCN2-TF3</td><td>8073</td><td>0.00005</td><td>75.2</td><td>73.1</td></tr><tr><td>FCN2-TF4</td><td>11,300</td><td>0.0001</td><td>81.4</td><td>80.6</td></tr><tr><td>FCN2-TF5</td><td>15,900</td><td>0.0001</td><td>82.2</td><td>81.5</td></tr></table>

\* Source model trained with 3 years of the Python simulator

According to the results presented in Table 9, with the availability of more examples to learn, the fne-tuning of the models’ classifcation section signifcantly improves. Of course, a tradeof occurs regarding the moment sufcient data is available; then, the application of fne-tuning no longer makes sense, and instead, the model can be entirely retrained. The operators and engineers involved must evaluate this choice. However, it is clear how useful TL can be in a scenario with scarce data. These initial results are promising regarding the application of online learning in fault detection and diagnosis tasks. Further research could focus on investigating the models’ behavior when new training data samples arrives at a higher frequency. In addition, it is important to evaluate the complexity of the algorithms applied to avoid computationally expensive training stages.

Table 10 shows the results for each process status when we applied the test set from the MATLAB simulator. We performed the testing stage in the source model (i.e., FCN2 without fne-tuning) and in the fne-tuned model with the most adequate learning rate (FCN2-TF2). The poor results from testing FCN2 with the MATLAB dataset prove that the data distribution certainly changes with the alteration of the control strategies. We could still detect and diagnose some faults that possess specifc signal signatures, such as Faults 2, 4, 7, 11, and 17. However, most faults were entirely misunderstood with the other process statuses.

The application of TL solved this problem. Fine-tuning allowed for an increase in the precision and/or recall for all the process statuses, except for the incipient faults that remained undetectable (Faults 9, 15, and 16). The fne-tuned model did not entirely neglect Fault 3, which is known for being difcult to detect. In addition, the total training times of the fne-tuned models were almost equivalent (a little bit shorter – approximately 15 min) to that of the source model. We skipped the entire model design step because the CNN topology and the hyperparameters previously chosen did not change (except for the learning rate). These are advantages of applying transfer learning strategies because they allow for the quick generation of an adequate substitute model while more historical data are obtained and archived.

These results confrm that building a FDD system based on a CNN is an efcient methodology. The use of fully convolutional architectures and the application of TL when changes occur in the variable data distributions are promising strategies to keep in mind in work with CNN models.

Table 10 Detailed test results with the MATLAB dataset for FCN2 with and without fnetuning

<table><tr><td rowspan="2">Process status</td><td colspan="3">FCN2 without fine-tuning (LR=0.0005)</td><td colspan="3">FCN2 with fine-tuning (LR=0.0001)</td></tr><tr><td>Precision (%)</td><td>Recall (%)</td><td>f1-score (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>f1-score (%)</td></tr><tr><td>Normal</td><td>77.2</td><td>2.9</td><td>5.7</td><td>77.4</td><td>94.0</td><td>84.9</td></tr><tr><td>Fault 1</td><td>99.3</td><td>71.1</td><td>82.8</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Fault 2</td><td>98.2</td><td>100.0</td><td>99.1</td><td>100.0</td><td>99.6</td><td>99.8</td></tr><tr><td>Fault 3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>55.5</td><td>60.9</td><td>58.1</td></tr><tr><td>Fault 4</td><td>98.8</td><td>100.0</td><td>99.4</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Fault 5</td><td>6.0</td><td>10.3</td><td>7.6</td><td>95.8</td><td>69.1</td><td>80.3</td></tr><tr><td>Fault 6</td><td>57.1</td><td>9.5</td><td>16.3</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Fault 7</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Fault 8</td><td>26.4</td><td>17.1</td><td>20.8</td><td>88.4</td><td>65.7</td><td>75.4</td></tr><tr><td>Fault 9</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Fault 10</td><td>5.2</td><td>100.0</td><td>9.8</td><td>99.0</td><td>89.9</td><td>94.2</td></tr><tr><td>Fault 11</td><td>79.9</td><td>100.0</td><td>88.8</td><td>99.9</td><td>99.9</td><td>99.9</td></tr><tr><td>Fault 12</td><td>0.0</td><td>0.0</td><td>0.0</td><td>83.8</td><td>89.9</td><td>86.7</td></tr><tr><td>Fault 13</td><td>62.2</td><td>12.3</td><td>20.6</td><td>84.5</td><td>91.3</td><td>87.8</td></tr><tr><td>Fault 14</td><td>100.0</td><td>9.0</td><td>16.5</td><td>100.0</td><td>75.1</td><td>85.8</td></tr><tr><td>Fault 15</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1.1</td><td>0.2</td><td>0.4</td></tr><tr><td>Fault 16</td><td>4.2</td><td>35.1</td><td>7.5</td><td>1.5</td><td>0.7</td><td>1.0</td></tr><tr><td>Fault 17</td><td>94.7</td><td>98.8</td><td>96.7</td><td>97.0</td><td>98.8</td><td>97.9</td></tr><tr><td>Fault 18</td><td>93.5</td><td>58.9</td><td>72.3</td><td>93.3</td><td>76.7</td><td>84.2</td></tr><tr><td>Fault 19</td><td>0.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>99.5</td><td>99.8</td></tr><tr><td>Fault 20</td><td>0.0</td><td>0.0</td><td>0.0</td><td>99.5</td><td>96.5</td><td>98.0</td></tr></table>

## 7 Conclusions

In this study, we acquired new datasets from continuous simulations to develop an FDD system based on a CNN for the TEP benchmark. Running continuous simulations allowed us to process variables’ essential characteristics, such as the transition range at the beginning of every normal operation period and how past events (active faults) alter the current process state’s dynamic behavior. The frst CNN model, M5, achieved 78.3% precision and 80.5% recall using three convolution lavers for feature extraction and two FC laver for classifcation (C(20)-C(30)-P-C(50)-P-FC(30)-FC(21)). Compared to other traditional ML models, the CNN proved its outstanding performance; the best ML model, a multilayer perceptron with three FC layers, presented only 75.11% and 71.59% precision and recall, respectively. Moreover, the conversion of M5 into a fully convolutional topology improved the detection and diagnosis of several faults, including some of the classes known for their incipient dynamic behaviors. FCN2 (with two strided convolutions and the fnal FC layers converted into Conv ones) achieved 80.6% precision and 83.7% recall, taking only 20 min to train (elapsed real time).

Finally, applying TL to avoid building another CNN model from scratch was efective. In the case study, we evaluated the change in the data distribution due to an alteration in the process control strategy. We used two TEP simulators, each with diferent closed-loop schemes. With the fnetuning application, only the fnal layers were retrained; it was not necessary to perform hyperparameter tuning again or redesign the network topology. The fne-tuned model achieved 79.8% precision and 76.6% recall using only data corresponding to three months of operation for the learning procedure. When changes in the data distribution are observed, fne-tuning proved to be a good alternative to obtain a surrogate model when more data are acquired.

Given the promising initial results, a future work possibility is the expansion of the transfer learning application to address the scenario where there are only a few labeled examples for a specifc fault (e.g., fault 18). In addition, complementary research is being developed to enable the comparison of the CNN results’ not only against other deep learning algorithms, such as recurrent neural networks (RNN), long short-term memory networks (LSTM), deep belief networks (DBN), and autoencoders (AE), but also against the combination of diferent DL architectures. Recent research has shown promising results by applying combined frameworks (e.g., CNN-RNN, CNN-LSTM) in fault detection and diagnosis tasks (Yong and Nugroho 2022; Shi et al. 2022; Saxena et al. 2023). Regarding the detection and diagnosis of incipient faults (which do not present a clearly defned fault signature, such as faults 3, 9 and

15), we have been considering the application of physicsinformed neural networks (PINNs). Our hypothesis is that the inclusion of some physical knowledge to the FDD system could improve the detection of challenging faults.

Acknowledgements Authors appreciate the fnancial support provided by Conselho Nacional de Desenvolvimento Científco e Tecnológico – CNPq (Grant number: 140913/2019-0). Professor Maurício B. de Souza Jr. is grateful to fnancial support from CNPq (Grant number: 311153/2021-6).

Data Availability The datasets generated during and/or analysed during the current study are available in the new\_tep\_datasets repository, https://github.com/anasouzac/new\_tep\_datasets.

## Declarations

Conflict of interest The authors declare that they have no known competing fnancial interests or personal relationships that could have appeared to infuence the work reported in this paper.

## References

Abdelkrim C. Meridiet MS. Boutasseta N. Boulanouar L. (2019) Detec tion and classifcation of bearing faults in industrial geared motors using temporal features and adaptive neuro-fuzzy inference system. https://doi.org/10.1016/i.helivon.2019.e02046. Helivon

Aggarwal CC (2018) Neural networks and deep learning – a Textbook, vol 1. Springer Nature, ed. Switzerland

Akiba T, Sano S, Yanase T, Ohta T, Koyama M (2019) Optuna: a nextgeneration hyperparameter optimization framework. KDD Appl Data Sci Track 19:4–8. https://doi.org/10.1145/3292500.333070

Andonovski G, Mušič G, Blažič S, Škrjanc I (2018) Evolving model identifcation for process monitoring and prediction of non-linear systems. Eng Appl Artif Intell 68:214–221. https://doi.org/10. 1016/j.engappai.2017.10.020

Baniardalani S, Askari J, Lunze J (2010) Qualitative model based fault diagnosis using a threshold level. Int J Control Autom Syst 8(3):683–694. https://doi.org/10.1007/s12555-010-0323-4

Bathelt A, Ricker NL, Jelali M (2014) Revision of the Tennessee Eastman process model. IFAC-PapersOnLine 48:309–314. https://doi. org/10.1016/j.ifacol.2015.08.199

Behbahani RM, Jazayeri-Rad H, Hajmirzaee S (2009) Fault detection and diagnosis in a sour gas absorption column using neural networks. Chem Eng Technol 32:840–845. https://doi.org/10.1002/ ceat.200800486

Botalb A, Moinuddin M, Al-Saggaf UM, Ali SSA (2018) Contrasting Convolutional Neural Network (CNN) with Multi-Layer Perceptron (MLP) for Big Data Analysis. International Conference on Intelligent and Advanced System (ICIAS) 1–5. https://doi.org/10. 1109/ICIAS.2018.8540626

Braatz RD (2020) Tennessee Eastman problem simulation data. Massachusetts Institute of Technology. http://web.mit.edu/braatzgroup/ links.html. Accessed 20 December 2021

Câmara MM (2019) GitHub. tep2py. https://github.com/camaramm/ tep2py. Accessed 20 December 2021

Chen Z, Gryllias K, Li W (2019) Intelligent fault diagnosis for rotary machinery using transferable convolutional neural network. IEEE Trans Industr Inf 16:339–349. https://doi.org/10.1109/TIL2019 2917233

Cheng H, Liu Y, Huang D, Xu C, Wu J (2020) A novel ensemble adaptive sparse bavesian transfer learning machine for nonlinear

large-scale process monitoring. Sensors 20:6139. https://doi.org 10.3390/s20216139

Chiang LH, Russell EL, Braatz RD (2000) Fault diagnosis in chemica processes using Fisher discriminant analysis, discriminant partial least squares, and principal component analysis. Chemometr Intell Lab Syst 50:243–252. https://doi.org/10.1016/S0169-7439(99) 00061-1

Dalton T, Patton R (1998) Model-based fault diagnosis of a two-pump system. Trans Inst Meas Control 20(3):115–124. https://doi.org 10.1177/014233129802000302

Downs JJ, Vogel EF (1993) A plant-wide industrial process control problem. Comput Chem Eng 17:245–255. https://doi.org/10.1016/ 0098-1354(93)80018-I

Gao Y, Yang T, Xing N, Xu M Fault Detection and Diagnosis for Spacecraft using Principal Component Analysis and Support Vector Machines. 2012 7th IEEE Conference on Industrial, Electronics (2012) and Applications (ICIEA). https://doi.org/10.1109/ ICIEA.2012.6361054

Goodfellow I, Bengio Y, Courville A (2016) Deep learning. 1 ed. MIT Press, Cambridge

Hartert L, Mouchaweh MS, Billaudel P (2010) A semi-supervised dynamic version of fuzzy K-Nearest neighbours to monitor evolving systems. Evol Syst 1:3–15. https://doi.org/10.1007/ s12530-010-9001-2

Heo S, Lee JH (2018) Fault detection and classifcation using artifcial neural networks. IFAC Papers Online 51:470–475. https://doi.org 10.1016/j.ifacol.2018.09.380

Hoi SCH, Sahoo D, Lu J, Zhao P (2021) Online learning: a comprehensive survey. Neurocomputing 459:249–289. https://doi.org/10. 1016/j.neucom.2021.04.112

Hubel DH, Wiesel T (1959) Receptive felds of single neurones in the cat's striate cortex. J Physiol 124(3):574–591. https://doi.org/10. 1113/jphysiol.1959.sp006308

Hussin NE, Johari A, Kidam K, Hashim H (2015) Major hazards of process equipment failures in the chemical process industry. Appl Mech Mater 735:75–79. https://doi.org/10.4028/www.scientifc. net/AMM.735.75

Isermann R (2006) Fault-Diagnosis Systems: an introduction from fault detection to fault tolerance, 1 edn. Springer, Germany

Karimi I, Salahshoor K (2012) A new fault detection and diagnosis approach for a distillation column based on a combined PCA and ANFIS scheme. 2012 24th Chinese Control and Decision Conference (CCDC). https://doi.org/10.1109/CCDC.2012.6244542

Khalifani S. Darvishzadeh R. Azad N. Rahmani RS (2022) Prediction of sunfower grain yield under normal and salinity stress by RBF, MLP and, CNN models. Ind Crops Prod 189(115762). https://doi. org/10.1016/i.indcrop.2022.115762

Kingma DP, Ba JL (2015) Adam: a method for stochastic optimization. 3rd International Conference for Learning Representations (ICLR 2015)

Knowledge transfer from simulation to physical processes. Comput Chem Eng. https://doi.org/10.1016/i.compchemeng.2020.106904

Larsson T et al (2001) Self-optimizing control of a large-scale plant: the Tennessee Eastman process. Ind Eng Chem Res 40:4889– 4901. https://doi.org/10.1021/ie000586y

Lau CK, Heng YS, Hussain MA, Mohamad Nor MI (2010) Fault diagnosis of the polypropylene production process (UNIPOL PP) using ANFIS. ISA Trans 49:559–566. https://doi.org/10.1016/j. isatra.2010.06.007

Lecun Y, Bottou L, Bengio Y, Hafner P (1998) Gradient-based learning applied to document recognition. Proceedings of the IEEE 86(11):2278–2324. https://doi.org/10.1109/5.726791

Lecun Y, Bengio Y, Hinton G (2015) Deep learning. Nature 521:436– 444. https://doi.org/10.1038/nature14539

Li W, Gu S, Zhang X, Chen T (2020) Transfer learning for process fault diagnosis

Li T, Zhao Y, Zhang C, Zhou K, Zhang X (2022) A semantic modelbased fault detection approach for building energy systems. Build Environ 207:108548. https://doi.org/10.1016/j.buildenv. 2021.108548

Liang J, Du R (2007) Model-based Fault Detection and diagnosis of HVAC systems using support Vector Machine method. Int J Refrig 30:1104–1114. https://doi.org/10.1016/j.ijrefrig.2006. 12.012

Liu Q, Huang C (2019) Fault diagnosis method based on transfer convolutional neural networks. IEEE Access 7:171423–171430

Lyman PR, Georgakis M (1995) Plant-wide control of the Tennessee Eastman process. Comput Chem Eng 19:321–331. https://doi.org/ 10.1016/0098-1354(94)00057-U

Mahadevan S, Shah SL (2009) Fault detection and diagnosis in process data using one-class support vector machines. J Process Control 19:1627–1639. https://doi.org/10.1016/j.jprocont.2009.07.011

Majdani F, Petrovski A, Doolan D (2018) Evolving ANN–based sensors for a context–aware cyber physical system of an ofshore gas turbine. Evol Syst 9:119–133. https://doi.org/10.1007/ s12530-017-9206-8

McKenzie FD, Gonzalez AJ, Morris R (1998) An integrated modelbased approach for real-time on-line diagnosis of complex systems. Eng Appl Artif Intell 11:279–291. https://doi.org/10.1016/ S0952-1976(97)00054-7

Medina E, Petraglia MR, Gomes JGRC, Petraglia A (2017) Comparison of CNN and MLP classifiers for algae detection in underwater pipelines. Seventh International Conference on Image Processing Theory, Tools and Applications (IPTA). https://doi.org/10.1109/ IPTA.2017.8310098

Oliveira MVM. Cunha BZ, Daniel GB (2021) A model-based technique to identify lubrication condition of hydrodynamic bearings using the rotor vibrational response. Tribol Int 160:107038. https://doi.org/10.1016/i.triboint.2021.107038

Pan SJ, Yang Q (2009) A survey on transfer learning. IEEE Trans Knowl Data Eng 22:1345–1359. https://doi.org/10.1109/TKDE. 2009.191

Park P, Di Marco P, Shin H, Bang J (2019) Fault detection and diagnosis using combined uutoencoder and long short-term memory network. Sensors. https://doi.org/10.3390/s19214612

Pu X, Li C (2021) Online semisupervised broad learning system for industrial fault diagnosis. IEEE Trans Industr Inf 17(10). https:// doi.org/10.1109/TII.2020.3048990

Renton G, Chatelain C, Adam S, Kermorvant C, Paquet T (2017) Handwritten text line segmentation using fully convolutional network 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), pp 5–9. https://doi.org/10.1109/ICDAR. 2017.321

Ricker NL (2005) Tennessee Eastman Challenge Archive. http://depts. washington.edu/control/LARRY/TE/download.html. Accessed 20 December 2021

Rieth CA, Amsel BD, Tran R, Cook MB (2017) Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation. Harvard Dataverse, V1. https://doi.org/10.7910/DVN 6C3JR1

Rostek K, Morytko L, Jankowska A (2015) Early detection and prediction of leaks in fluidized-bed boilers using artificial neural networks. Energy 89:914–923. https://doi.org/10.1016/j.energy. 2015.06.042

Russell EL, Chiang LH. Braatz RD (2000) Fault detection in industrial processes using canonical variate analysis and dynamic principal component analysis Chemometr Intell Lab Syst 51:81–93, https:/ doi.org/10.1016/S0169-7439(00)00058-7

Santos MR, Costa BSJ, Bezerra CG, Andonovski G, Guedes LA (2022) An evolving approach for fault diagnosis of dynamic systems. Expert Syst Appl 189:115983. https://doi.org/10.1016/j.eswa. 2021.115983

Saxena A, Kumar R, Rawat AK, Majid M, Singh J, Devakirubakaran S, Singh GK (2023) Abnormal health monitoring and assessment of a three-phase induction motor using a supervised CNN-RNN-based machine learning algorithm. Math Probl Eng. https://doi.org/10.1155/2023/1264345

Shelhamer E, Long J. Darrell T (2017) Fully convolutional networks for semantic segmentation. IEEE Trans Pattern Anal Mach Intell 39:640–651. https://doi.org/10.1109/TPAMI.2016.2572683

Shi J, Peng D, Peng Z, Zhang Z, Goebel K, Wu D (2022) Planetary gearbox fault diagnosis using bidirectional-convolutional LSTM networks. Mech Syst Signal Process 162(107996). https://doi. org/10.1016/j.ymssp.2021.107996

Shin HJ, Eom D, Kim S (2005) One-class support vector machines— an application in machine fault detection and classifcation. Comput Ind Eng 48:395–408. https://doi.org/10.1016/j.cie. 2005.01.009

Shu Y et al (2016) Abnormal situation management: Challenges and opportunities in the big data era. Comput Chem Eng 91:104– 113. https://doi.org/10.1016/j.compchemeng.2016.04.011

Simani S, Fantuzzi C (2006) Dynamic system identifcation and model-based fault diagnosis of an industrial gas turbine prototype. Mechatronics 16:341–363. https://doi.org/10.1016/j. mechatronics.2006.01.002

Souza ACO (2021) new-tep-datasets. v1. https://github.com/anaso uza26/new\_tep\_datasets. Accessed 20 December 2021

Springenberg JT, Dosovitskiy A, Brox T, Riedmiller M (2015) Striving for simplicity: the all convolutional net. International Conference on Learning Representations (ICLR), pp 1–14

Subbaraj P, Kannapiran B (2014) Fault detection and diagnosis of pneumatic valve using adaptive neuro-fuzzy inference system approach. Appl Soft Comput 19:362–371. https://doi.org/10. 1016/j.asoc.2014.02.008

Tian T, Chu Z, Hu Q, Ma L (2021) Class-wise fully convolutional network for semantic segmentation of remote sensing images. Remote Sens. https://doi.org/10.3390/rs13163211

Tidriri K, Chatti N, Verron S, Tiplica T (2018) Model-based fault detection and diagnosis of complex chemical processes: A case study of the Tennessee Eastman process. Proceeding of the Institution of Mechanical Engineers Part I - Journal of Systems and Control Engineering 232(6):742–760. https://doi.org/10. 1177/0959651818764510

Torrecilla JL, Romo J (2018) Stat Probab Lett 136:15–19. https:// doi.org/10.1016/j.spl.2018.02.038. Data learning from big data

Toubakh H, Sayed-Mouchaweh M (2015) Hybrid dynamic data-driven approach for drift-like fault detection in wind turbines. Evol Syst 6:115–129. https://doi.org/10.1007 s12530-014-9119-8

Venkatasubramanian V (2019) The promise of artifcial intelligence in chemical engineering: is it here. finally? AlChE Journal 65(2):466–478. https://doi.org/10.1002/aic.16489

Venkatasubramanian V, Vaidyanathan R, Yamamoto Y (1990) Process fault detection and diagnosis using neural networks—I. steadystate processes. Comput Chem Eng 14:699–712. https://doi.org 10.1016/0098-1354(90)87081-Y

Venkatasubramanian V et al (2003a) A review of process fault detection and diagnosis, part I: quantitative model-based methods. Comput Chem Eng 27:293–311. https://doi.org/10.1016/S0098- 1354(02)00160-6

Venkatasubramanian V et al (2003b) A review of process fault detection and diagnosis, part II: qualitative models and search strategies. Comput Chem Eng 27:313–326. https://doi.org/10.1016 S0098-1354(02)00161-8

Venkatasubramanian V et al (2003c) A review of process fault detection and diagnosis, part III: process history based methods. Comput Chem Eng 27:327–346. https://doi.org/10.1016/S0098- 1354(02)00162-X

Wang H, Li P, Gao F, Song Z, Ding SX (2006) Kernel classifer with adaptive structure and fxed memory for process diagnosis. AIChE J 52:3515–3531. https://doi.org/10.1002/aic.10982

Wang X, Liu X, Li Y (2019) An incremental model transfer method for complex process fault diagnosis. IEEE/CAA J Automatica Sinica 6(5):1268–1280. https://doi.org/10.1109/JAS.2019.1911618

Wang K, Zhou W, Mo Y, Yuan X, Wang Y, Yang C (2022) New mode cold start monitoring in industrial processes: a solution of spatial– temporal feature transfer. Knowl Based Syst 248:108851. https:/ doi.org/10.1016/j.knosys.2022.108851

Wu H, Zhao J (2018) Deep convolutional neural network model based chemical process fault diagnosis. Computers & Chemical Engineering 115:185–197. https://doi.org/10.1016/j.compchemeng. 2018.04.009

Wu H, Zhao J (2020) Fault detection and diagnosis based on transfer learning for multimode chemical processes. Comput Chem Eng https://doi.org/10.1016/j.compchemeng.2020.106731

Xavier GM, Seixas JM (2018) Fault detection and diagnosis in a chemical process using long short-term memory recurrent neural network. 2018 International Joint Conference on Neural Networks (IJCNN). https://doi.org/10.1109/IJCNN.2018.8489385

Xie D, Bai L (2015) A hierarchical deep neural network for fault diagnosis on Tennessee-Eastman process. 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA). https://doi.org/10.1109/ICMLA.2015.208

Xie Z, Yang X. Li A. Ji Z (2019) Fault diagnosis in Industrial Chemical processes using optimal probabilistic neural network. Can J Chem Eng 97:2453–2464. https://doi.org/10.1002/cjce.23491

Yong LZ, Nugroho H (2022) Acoustic anomaly detection of mechanical failure: time-distributed CNN-RNN deep learning models.

Control, instrumentation and mechatronics: theory and practice. Lecture Notes in Electrical Engineering 921:662–672. https://doi. org/10.1007/978-981-19-3923-5\_57

Zhang Z, Zhao J (2017) A deep belief network based fault diagnosis model for complex chemical processes. Comput Chem Eng 107:395–407. https://doi.org/10.1016/j.compchemeng.2017.02. 041

Zhang R, Tao H, Wu L, Guan Y (2017) Transfer learning with neural networks for bearing fault diagnosis in changing working conditions. IEEE Access 5:14347–14357. https://doi.org/10.1109/ ACCESS.2017.2720965

Zhang S. Bi K, Oiu T (2020) Bidirectional recurrent neural networkbased chemical process fault diagnosis. Ind Eng Chem Res 59:824–834. https://doi.org/10.1021/acs.iecr.9b05885

Zhu Q, Jia Y, Peng D, Xu Y (2014) Study and application of fault prediction methods with improved reservoir neural networks. Chin J Chem Eng 22:812–819. https://doi.org/10.1016/j.cjche.2014.05. 016

Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional afliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.