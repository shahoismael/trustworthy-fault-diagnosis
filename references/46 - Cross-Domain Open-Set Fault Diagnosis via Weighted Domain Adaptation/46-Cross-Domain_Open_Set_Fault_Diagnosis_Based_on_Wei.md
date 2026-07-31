Article

# Cross-Domain Open Set Fault Diagnosis Based on Weighted Domain Adaptation with Double Classifiers

Huaqing Wang <sup>1</sup>, Zhitao Xu <sup>1</sup>, Xingwei Tong <sup>1</sup> and Liuyang Song <sup>2,</sup>\*

College of Mechanical Electrical Engineering, Beijing University of Chemical Technology, Beijing 100029, China 2 Key Laboratory of Health Monitoring and Self-recovery for High-end Mechanical Equipment, Beijing University of Chemical Technology, Beijing 100029, China

Correspondence: songliuyang@mail.buct.edu.cn

Abstract: The application of transfer learning in fault diagnosis has been developed in recent years. It can use existing data to solve the problem of fault recognition under different working conditions. Due to the complexity of the equipment and the openness of the working environment in industrial production, the status of the equipment is changeable, and the collected signals can have new fault classes. Therefore, the open set recognition ability of the transfer learning method is an urgent research direction. The existing transfer learning model can have a severe negative transfer problem when solving the open set problem, resulting in the aliasing of samples in the feature space and the inability to separate the unknown classes. To solve this problem, we propose a Weighted Domain Adaptation with Double Classifiers (WDADC) method. Specifically, WDADC designs the weighting module based on Jensen–Shannon divergence, which can evaluate the similarity between each sample in the target domain and each class in the source domain. Based on this similarity, a weighted loss is constructed to promote the positive transfer between shared classes in the two domains to realize the recognition of shared classes and the separation of unknown classes. In addition, the structure of double classifiers in WDADC can mitigate the overfitting of the model by maximizing the discrepancy, which helps extract the domain-invariant and class-separable features of the samples when the discrepancy between the two domains is large. The model’s performance is verified in several fault datasets of rotating machinery. The results show that the method is effective in open set fault diagnosis and superior to the common domain adaptation methods.

![](images/71e26a510d81de5dbad2ca40e6bd2fc537efad3455120ffb1319e38d66f3b98e.jpg)

Citation: Wang, H.; Xu, Z.; Tong, X.; Song, L. Cross-Domain Open Set Fault Diagnosis Based on Weighted Domain Adaptation with Double Classifiers. Sensors 2023, 23, 2137. https://doi.org/10.3390/ s23042137

Academic Editor: Giovanni Betta

Received: 19 January 2023 Revised: 4 February 2023 Accepted: 11 February 2023 Published: 14 February 2023

![](images/943fadfe9669468b6e829a8b9490ba581dcdd99efea4c170f67f2ed2d9194241.jpg)

Copyright: © 2023 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https:// creativecommons.org/licenses/by/ 4.0/).

Keywords: fault diagnosis; open set domain adaptation; transfer learning; rotating machinery; deep learning

## 1. Introduction

In the modern industry, it is critical to keep the equipment running safely and stably [1]. As the integral parts of equipment, rotating components such as bearings and gears are prone to fault due to the harsh working environment, which will affect the stable operation of the equipment. Therefore, efficient fault diagnosis methods play an important role in early fault warning and maintenance [2], which can effectively reduce property losses and casualties caused by mechanical faults [3].

Traditional signal processing knowledge combined with machine learning methods, such as the BP neural network and Empirical Mode Decomposition (EMD), can effectively identify fault types and predict the operation status of components [4–9]. However, the feature extraction process of these methods seriously relies on professional knowledge and can easily consume many resources in the era of big data [10]. The emergence of Deep Learning (DL) can solve the problem of relying on the workforce [11]. The DL model represented by the Convolution Neural Network (CNN) and Long Short-Term Memory (LSTM) neural network is used in fault diagnosis, with promising results because of the powerful feature extraction capabilities [12–18]. Among most research, applying DL to a fault diagnosis requires two preconditions: (1) Test samples and samples participatingfault diagnosis requires two preconditions: (1) Test samples and samples participating in in model training have the same label space. (2) There are enough labels in the trainingmodel training have the same label space. (2) There are enough labels in the training sam samples. However, due to the changes in the working environment of the equipment,ples. However, due to the changes in the working environment of the equipment, the dis the distribution of samples collected each time is different, and there are few samplestribution of samples collected each time is different, and there are few samples with labels with labels under the same working conditions. Therefore, it is important to realize faultunder the same working conditions. Therefore, it is important to realize fault diagnosis diagnosis under different operating environments using labeled samples.under different operating environments using labeled samples.

As a branch of the Transfer Learning (TL) method emerging in recent years, domainAs a branch of the Transfer Learning (TL) method emerging in recent years, domain adaptation provides a new idea for cross-domain fault diagnosis using the concept ofadaptation provides a new idea for cross-domain fault diagnosis using the concept of ad adversarial learning. It can use the existing and available label information to achieve aversarial learning. It can use the existing and available label information to achieve a cross cross-domain fault diagnosis by reducing the distribution differences of the features [19,20].domain fault diagnosis by reducing the distribution differences of the features [19,20]. In In the current research on cross-domain fault diagnosis, it is mostly assumed that the twothe current research on cross-domain fault diagnosis, it is mostly assumed that the two domains contain the same fault category. Under this assumption, the training model candomains contain the same fault category. Under this assumption, the training model can extract the domain invariant features of the samples to achieve the purpose of cross-domain fault diagnosis.<sub>main</sub> <sub>fault</sub> <sub>diag</sub>

However, the equipment status collected in a different environment is unknown, usually including classes outside the source domain, as shown in Figure 1. The existing domain adaptation model used in this case can lead to errors in sample alignment in the feature space and even affect the alignment between known classes under different conditions. In recent years, scholars have made initial achievements in the research of open set fault diagnosis, most of which are based on different theories to establish models that reject unknown samples [21]. However, these methods are prone to the problem of sample aliasing in the feature space when the discrepancy between the two domains is significant, resulting in low diagnostic accuracy.

![](images/64065c67fc6440292b2bf5ebc1bd899311be853711c73ffe1d1e481ebe8ed21c.jpg)

(a)Closed set domain adaptation setting  
![](images/badfc07f4f8932c6ab93d3abd7ab6b8639a417fcd17d9295766233e02fba04ca.jpg)  
(b)Open set domain adaptation setting  
Figure 1. Difference between closed set and open set settings. (a) Closed set domain adaptation setting, (b) Open set domain adaptation setting.

To solve the above problems and improve the accuracy of open set fault diagnosis, a method based on WDADC is proposed. The basic framework of the method is shown in Figure 2. The role of the feature extractor module is to extract the high-dimensional features of the processed data, the weighting module assigns weights to samples by calculating their similarity, and the classification module plays the role of correctly classifying the samples. Through the interaction between modules, we can accurately identify samples of the shared classes while separating samples of the non-shared class, thus achieving the goal of open set fault diagnosis.

![](images/f5749b9e9b1fa6dab01d83d94df0c023e20ce552aa38bae513358ec081b6da86.jpg)  
<sup>igure</sup> <sup>2.</sup> <sup>The</sup> <sup>framework</sup> <sup>of</sup> <sup>our</sup> <sup>method.</sup>Figure 2. The framework of our method.

The main contributions of this paper are as follows:The main contributions of this paper are as follows:

(1) For the problem of open set fault diagnosis, we design a weighting module based (1) For the problem of open set fault diagnosis, we design a weighting module based on on Jensen–Shannon divergence in the adversarial model to evaluate the similarity of sam-Jensen–Shannon divergence in the adversarial model to evaluate the similarity of samples ples between the two domains. The target domain samples are assigned weights accord-between the two domains. The target domain samples are assigned weights according ng to the similarity to facilitate alignment between shared class samples and separate the to the similarity to facilitate alignment between shared class samples and separate the unknown samples.unknown samples.

(2) Considering the negative transfer problem in open set fault diagnosis, a weighted (2) Considering the negative transfer problem in open set fault diagnosis, a weighted loss function is constructed to update the model in the direction of extracting domain in-loss function is constructed to update the model in the direction of extracting domain variant and class separable features of samples. When the distribution of samples in the invariant and class separable features of samples. When the distribution of samples in the two domains is quite different, the discrepancy between the double classifiers is used to two domains is quite different, the discrepancy between the double classifiers is used to improve the generalization of the model, and cross-domain open set fault diagnosis can improve the generalization of the model, and cross-domain open set fault diagnosis can be realized. be realized.

(3) The experiment on several mechanical fault datasets shows that the proposed (3) The experiment on several mechanical fault datasets shows that the proposed method performs better than other domain adaptation methods. method performs better than other domain adaptation methods.

The structural arrangement of this paper is shown as follows. Section 2 outlines the basic theory of the TL methods. Section 3 presents the proposed method in detail. Section 4 presents the designed validation experiments and analyzes the experimental results. Section 5 summarizes the entire paper and plans the future work.

## 2. Related Work

This section describes the application of TL in fault diagnosis. According to the different problems to be solved, we divided the application of TL in fault diagnosis into the closed set and open set. Additionally, according to the different modeling principles, we further distinguished between the methods. We reviewed each method mentioned, as shown in Table 1.

Table 1. The application of TL in fault diagnosis.

<table><tr><td></td><td>Problems</td><td>Types</td><td>Papers</td></tr><tr><td rowspan="6">Transfer Learning for fault diagnosis</td><td rowspan="4">Closed set</td><td>Instance-based</td><td>[22,23]</td></tr><tr><td>Mapping-based</td><td>[24,25]</td></tr><tr><td>Model-based</td><td>[26,27]</td></tr><tr><td>Adversary-based</td><td>[28–31]</td></tr><tr><td rowspan="2">Open set</td><td>Discriminate-based</td><td>[32–35]</td></tr><tr><td>Generation-based</td><td>[36,37]</td></tr></table>

## 2.1. TL Methods Applied to Closed Set Fault Diagnosis

TL broadens the conditions for neural networks to be used in fault diagnosis. Its goal is to solve the problem that the characteristic information in signals under different working conditions is different and challenging to be recognized. According to the different technologies used, the development of TL in fault diagnosis can be divided into the following types: instance-based, mapping-based, model-based, and adversary-based.

The instance-based TL method assumes many overlaps in the features of the signal under different states. By weighting the source domain samples to construct a feature distribution similar to the target domain, the model has a good effect when testing the target domain samples after training [22,23].

The mapping-based TL method believes there will be some discrepancy between the two domains. Still, the discrepancy can be eliminated through different mapping methods in the feature space so that similar samples can be gathered together. Different samples can be separated from each other to achieve the goal of cross-domain fault diagnosis [24,25].

The model-based TL considers that the shallow part of the network model is less relevant to the final classification task, and it usually extracts the macro features of the fault. The labeled samples can be used to train the shallow part of the network model. The output layer parameters are adjusted by fine-tuning to accurately classify the target domain samples [26,27].

The idea of adversary-based TL comes from adversarial learning. Through the training objectives of each module in the model, the features of the samples extracted in the model can be separated by the classifier without being distinguished by the domain discriminator [28]. As the representative of the adversarial model, the domain adaptation model expects that the domain invariant and class separable features can be extracted after training. This method has been widely studied in cross-domain fault diagnosis [29–31].

TL has made good research progress in cross-domain fault diagnosis. Still, the above methods often default that the data in two domains have the same class, which limits the practical development of TL in fault diagnosis.

## 2.2. TL Methods Applied to Open Set Fault Diagnosis

In industrial practice, acquiring fault labels will consume a lot of financial and human resources. Due to changes in operation and environment, the classes of signals collected in the equipment are unknown. The use of common TL methods will cause serious aliasing between non-shared class and shared class samples during feature alignment. Therefore, cross-domain open set fault diagnosis has become an urgent development direction. The existing open set fault diagnosis methods are mainly considered from the perspective of modeling, which is primarily divided into discriminative and generative models [38].

The discriminative models in open set fault diagnosis include the traditional machine learning-based and deep neural network-based models. The machine learning-based model establishes a mechanism to reject non-shared samples by setting an empirical threshold of the machine learning model or analyzing the distribution of abnormal data in combination with extreme value theory (EVT) to achieve open set fault diagnosis [32,33]. However, this method has the problem that, once samples are recognized as unknown classes, they cannot be correctly classified again through training iterations. The deep neural network-based model mainly uses the powerful feature extraction capability of DL and solves the inherent closed set problem caused by the normalization of the Softmax layer in the network [34,35] to achieve the goal of open set fault diagnosis by identifying samples of non-shared classes. However, the negative transfer problem caused by the misjudgment of these methods is the urgent research direction.

The application of generative models in open set fault diagnosis is achieved through adversarial learning. The generation model generates samples with similar characteristics to the actual samples by learning the sample characteristics to facilitate the model’s training and increase the robustness of the model, which can also solve the data imbalance and small sample problems. Non-instance-based generative models combined with EVT or the empirical threshold setting can achieve open set fault diagnosis through the adversarial training of modules [36,37]. This method relies heavily on the validity of the generated samples, especially the generation of unknown samples needed to strengthen the stability and reliability further.

## 3. Proposed Method

Unlike models such as the Domain Adaptation Neural Network (DANN) applied to a closed set domain adaptation problem, the Open Set Domain Adaptation by Backpropaga tion (OSBP) [39] aims to correctly identify the shared class samples in the target domain and classify the non-shared class samples as unknown classes.

## 3.1. Problem Description

In open set fault diagnosis, the datasets with label information are defined as the source domain, while the target domain are the datasets without a label. $\left\{ x _ { i } ^ { s } , y _ { i } ^ { s } \right\}$ and $\left\{ x _ { i } ^ { t } , y _ { i } ^ { t } \right\}$ are the i-th sample and its label in the source and target domains. In the open set problem, the labels of the source domain are a subset of the labels of the target domain, the intersection of the two is called the shared class, and the labels are set as M classes, according to the number of sample classes. In the target domain, the complementary set of the source domain is called the non-shared class, and the label is set to $\bar { M } + 1$

## 3.2. OSBP

In OSBP, the model mainly includes a feature extractor and a classifier. The feature extractor maps the samples to the same feature space. The classifier receives the features output from the feature extractor and outputs the (M + 1)-dimensional probability through the Softmax function. The OSBP forms an adversarial relationship between the feature extractor and the classifier by the Gradient Reversal Layer (GRL) to train the model. Through continuous training, the feature extractor is optimized to maximize the loss of the classification, which helps to align the samples between the two domains. The probability $p _ { x ^ { t } } ^ { M + 1 }$ that the sample is in the non-shared class is compared with the set threshold t by the output of the classifier. When $p < t ,$ the samples are classified as shared classes, and otherwise, they are classified as non-shared classes. This method aims to construct a decision boundary between shared class samples and non-shared class samples when the label of the target domain sample is unknown. The training objectives of the method are shown as follows:

$$
\begin{array}{c} \min _ {G} L _ {s} - L _ {t} \\ \min _ {C} L _ {s} + L _ {t} \end{array}\tag{1}
$$

where $L _ { s }$ is the classification loss, and $L _ { t }$ is the binary cross-entropy loss of the sample classification result concerning the threshold $t ,$ with t standing at 0.5. From the training objective, it can be seen that the feature extractor expects to maximize the loss of the classification so that the probability of a sample being classified as a non-shared class is far from t. On the other hand, the classifier expects the probability of a sample being classified as a non-shared class to converge to t. The model parameters are continuously updated by the backpropagation of loss.

## 3.3. Proposed Method<sup>3.3.</sup> <sup>Proposed</sup> <sup>Metho</sup>

<sub>In</sub> <sub>OSBP,</sub> <sub>due</sub> <sub>to</sub> <sub>non-shared</sub> <sub>class</sub> <sub>samples,</sub> <sub>the</sub> <sub>corresponding</sub> <sub>samples</sub> <sub>in</sub> <sub>the</sub> <sub>two</sub>In OSBP, due to non-shared class samples, the corresponding samples in the two do domains are prone to aliasing during feature alignment, resulting in a negative transfer.<sup>mains</sup> <sup>are</sup> <sup>prone</sup> <sup>to</sup> <sup>aliasing</sup> <sup>during</sup> <sup>feature</sup> <sup>alignment,</sup> <sup>resulting</sup> <sup>in</sup> <sup>a</sup> <sup>negative</sup> <sup>transfer</sup> Therefore, in WDADC, a weighting module is designed to improve the OSBP. By measuring<sup>Therefore,</sup> <sup>in</sup> <sup>WDADC,</sup> <sup>a</sup> <sup>weighting</sup> <sup>module</sup> <sup>is</sup> <sup>designed</sup> <sup>to</sup> <sup>improve</sup> <sup>the</sup> <sup>OSBP.</sup> <sup>By</sup> <sup>meas</sup> the distance of the sample in the feature space, the samples are assigned different weights,<sup>uring</sup> <sup>the</sup> <sup>distance</sup> <sup>of</sup> <sup>the</sup> <sup>sample</sup> <sup>in</sup> <sup>the</sup> <sup>feature</sup> <sup>space,</sup> <sup>the</sup> <sup>samples</sup> <sup>are</sup> <sup>assigned</sup> <sup>differen</sup> <sub>thus</sub> <sub>promoting</sub> <sub>the</sub> <sub>positive</sub> <sub>transfer</sub> <sub>between</sub> <sub>shared</sub> <sub>classes</sub> <sub>in</sub> <sub>the</sub> <sub>two</sub> <sub>domains.</sub> <sub>In</sub>weights, thus promoting the positive transfer between shared classes in the two domain addition, using the discrepancy between different classifiers, the generalization of theIn addition, using the discrepancy between different classifiers, the generalization of th proposed method is improved to achieve an excellent cross-domain open set diagnosticproposed method is improved to achieve an excellent cross-domain open set diagnosti performance of the model. The training process and model structure of this method areperformance of the model. The training process and model structure of this method ar shown in Figure 3.shown in Figure

![](images/4442b2656d66655e2a6267d21886549cb63bd28ab7cdf665e4959096e5de452b.jpg)  
Figure 3. The training process and model structure of the proposed method.

(1) Feature extraction module: the module is denoted by G and consists of four layers of convolution and two layers of full connection, where each convolutional is connected with a batch normalization (BN) layer afterward. This module maps the samples to the same feature space and extracts the high-dimensional features. The features of the samples are output when going through the feature extraction module, as shown in Formula (2):

$$
\begin{array}{r} F _ {i} ^ {t} = G (x _ {i} ^ {t}) \\ F _ {i} ^ {s} = G (x _ {i} ^ {s}) \end{array}\tag{2}
$$

where $F _ { i } ^ { s }$ and $F _ { i } ^ { t }$ <sup>(</sup> <sup>)</sup> i i<sup>F</sup> <sup>G</sup> <sup>x=</sup> are the high-dimensional features of the i-th source domain and target domain samples.

For the labeled samples, the cross-entropy loss training model is calculated by the label information to correctly diagnose the fault, and the loss is shown in Formula (3):

$$
L _ {C} = \frac {1}{n _ {s}} \sum_ {i = 1} ^ {n _ {s}} L _ {C E} (F _ {i} ^ {s}, y _ {i} ^ {s})\tag{3}
$$

where $n _ { s }$ is the number of labeled samples, and $L _ { C E }$ is the cross-entropy loss.

(2) Weighting module: The module is denoted by W, and the novelty of this paper is that we design a weight calculation method based on Jensen–Shannon (JS) divergence, which assigns different weights to the samples by calculating the distances between the samples of the two domains in the feature space to promote a positive transfer between the shared classes. First, the class centers of the labeled samples in the feature space are calculated as follows:

$$
C _ {a} = \frac {1}{n _ {a}} \sum_ {i = 1} ^ {n _ {a}} F _ {a, i} ^ {s}\tag{4}
$$

where $C _ { a }$ is the class center of the a-th class, $a \in \{ 1 , 2 , \ldots , \mathsf { M } \} , n _ { a }$ is the number of a-th class samples, and $F _ { a , i } ^ { s }$ is the feature of the a-th class samples in the source domain.

Due to the existence of non-shared classes and the differences in working conditions, the feature distribution between the two domains is not precisely the same. The JS di vergence is highly sensitive to the differences between two distributions and has a good diagnostic performance in different degrees of fault [40], and as a variation of Kullback– Leibler (KL) divergence, the JS divergence makes improvements over the symmetry and value domain range, making it more accurate in similarity discriminations. Therefore, to calculate the similarity between the sample and the class center in the two domains, JS divergence is used to calculate the method of their differences. In this paper, the dis tance between the sample features and the class centers in the two domains is calculated as follows:

$$
D _ {J S, a} (C _ {a} | F _ {i} ^ {t}) = \frac {1}{2} D _ {K L} (C _ {a} | \frac {C _ {a} + F _ {i} ^ {t}}{2}) + \frac {1}{2} D _ {K L} (F _ {i} ^ {t} | \frac {C _ {a} + F _ {i} ^ {t}}{2})\tag{5}
$$

where $D _ { J S , a } ( C _ { a } | F _ { i } ^ { t } )$ ) is the JS divergence between the i-th sample feature in the target domain and the a-th class sample center source domain. $D _ { K L }$ represents the KL divergence, which is calculated as shown in Formula (6):

$$
D _ {K L} (M | N) = \sum_ {i = 1} ^ {v} M _ {i} \ln (\frac {M _ {i}}{N _ {i}})\tag{6}
$$

where v denotes the dimension of the output of the G module, and $M _ { i }$ and $N _ { i }$ represent the i-th elements of the vector

The distance between the sample and each class center in the feature space is calculated, and the sum of the distances is used as the similarity judging index between the i-th samples in the two domains, as shown in Formula (7):

$$
D _ {i} = \sum_ {a = 1} ^ {M} D _ {J S, a} (C _ {a} | F _ {i} ^ {t})\tag{7}
$$

When $D _ { i }$ is smaller, we consider that this target domain sample is more similar to the source domain. Therefore, $D _ { i }$ is normalized to [0, 1], and the difference between 1 and it is taken as the weights of this target domain sample in the model, as shown in Formula (8):

$$
w _ {i} = 1 - \frac {D _ {i} - D _ {\mathrm{max}}}{D _ {\mathrm{max}} - D _ {\mathrm{min}}}\tag{8}
$$

where $w _ { i }$ is the weight generated by the weighting module for the i-th sample in the target domain.

(4) Classification module: The classification module is denoted by C, including two independent classifiers C1 and C2, both consisting of a fully connected layer and a BN layer. This module receives high-dimensional features from the feature extraction module and classifies them into M + 1 classes. The loss for each target domain sample is calculated by binary cross-entropy, as shown in Formula (9):

$$
L _ {t} ^ {i} = - t \ln (p _ {x _ {i} ^ {t}} ^ {M + 1}) - (1 - t) \ln (1 - p _ {x _ {i} ^ {t}} ^ {M + 1})\tag{9}
$$

where $L _ { t } ^ { i }$ is the binary cross-entropy loss of the i-th target domain sample, and $p _ { x _ { i } ^ { t } } ^ { M + 1 }$ is the probability that the i-th sample is recognized as a non-shared class.

The weights obtained from the calculations are added to the optimization process to obtain the weighted binary cross-entropy loss:

$$
L _ {d} = (\sum_ {i = 1} ^ {n _ {t}} L _ {t} ^ {i} w _ {i}) / (n _ {t} \sum_ {i = 1} ^ {n _ {t}} w _ {i})\tag{10}
$$

where $n _ { t }$ is the number of samples in the target domain, and $L _ { d }$ is the total loss.

There is only one classifier in most models, which is prone to overfitting, resulting in low efficiency in the cross-domain fault diagnosis. In this paper, we use the discrepancy between double classifiers, generated by the complexity of the model and the initialization of the parameters, to promote the alignment between cross-domain samples by maximizing this discrepancy. It can improve the generalization and stability of the model. The calculation process is as shown in Formula (11):

$$
L _ {s} = - \sum_ {i = 1} ^ {n _ {s}} (p _ {c 1, i} \ln (p _ {c 2, i}) + (1 - p _ {c 1, i}) \ln (1 - p _ {c 2, i}))\tag{11}
$$

The training objectives of this method are shown as follows. The adversarial relationship between classification and feature extraction modules is formed through GRL to achieve a continuous and stable diagnosis.

$$
\min _ {G} L _ {c} - L _ {d} + L _ {s} \min _ {C} L _ {c} + L _ {d} - L _ {s}\tag{12}
$$

## 4. Experimental Methods

In this section, we designed experiments on multiple datasets of a mechanical fault to verify the proposed method.

## 4.1. Datasets Description

## (1) Laboratory Gearbox Dataset

As shown in Figure 4a, the laboratory gearbox test rig consists of acceleration sensors, a braking system, and a gearbox. The state of the gear is divided into the chipped tooth, root wear, and healthy conditions, and the state of the bearing is divided into the inner race fault and outer race fault. The sampling frequency is set to 100 kHz, and the different working conditions are set by adjusting the rotational speed. Six types of status data are collected under 1200 rpm, 1500 rpm, and 1800 rpm bearing an inner race and gear root wear compound fault (IR), bearing an inner race and gear-chipped tooth compound fault (IT), bearing an inner race fault (I), bearing an outer race and gear root wear compound fault (OR), bearing an outer race and gear chipped tooth compound fault (OT), and bearing an outer race fault (O).

## (2) The CWRU Dataset

As shown in Figure 4b, the Case Western Reserve University (CWRU) test rig consists of sensors, a motor, and an electronic controller. The bearing damage is a single-point damage by EDM. The bearing data of its drive end is used for testing. Set the sampling frequency to 12 kHz and different working conditions by adjusting the load. Four types of status data are collected under 0 hp, 1 hp, and 2 hp loads: inner race fault, rolling fault, outer race fault, and healthy conditions.

## (3) The IMS dataset

As shown in Figure 4c, the Intelligent Maintenance Systems (IMS) test rig consists of accelerometers, a motor, bearings, and thermocouples. By applying a longitudinal load to the bearing, the whole process of bearing from healthy conditions to a fault is recorded. Set the sampling frequency to 20 kHz, and the speed is 2000 rpm. Four status data are collected, including the inner race fault, rolling fault, outer race fault, and healthy conditions.

## (4) The centrifugal pump dataset

As shown in Figure 4d, the overhung impeller centrifugal pump dataset is collected in<sub>As</sub> <sub>shown</sub> <sub>in</sub> <sub>Figure</sub> <sub>4d,</sub> <sub>the</sub> <sub>overhung</sub> <sub>impeller</sub> <sub>centrifugal</sub> <sub>pump</sub> <sub>dataset</sub> <sub>is</sub> <sub>collected</sub> the industrial scene. The sampling frequency is set to 32.8 kHz, and the data of bearing healthy conditions, inner race fault, outer race fault, and rolling fault are collected under 745 rpm and 1485 rpm. The signal contains more interference components than the data collected in the laboratory and the public dataset.

![](images/ad709e3330e9b25adaf36cd72173e3f10763f7db911722feb97728917ae29fa3.jpg)  
(a) The lab gearbox test rig

![](images/7e717388d82ae014856bcaa67ca3aab56399544c7c1477a6d2d39b4e4d7cfe09.jpg)  
(b) The CWRU bearing test rig

Acceleronmeter Radial Load Thermocouples  
![](images/250a2b8b43d7c2c2b335fcaca28db2bc3502d582cf3a63e7325ab3d48e1e9570.jpg)  
(c) The IMS bearing test rig

![](images/c53f4974d42d958b4c9ebae87a8b737d5cf225faa9d2f55d3714992f86e8964d.jpg)  
(d) The overhung impeller centrifugal pump  
Figure 4. The test rigs. (a) The lab gearbox test rig, (b) The CWRU bearing test rig, (c) The IMS bearing test rig, (d) The overhung impeller centrifugal pumpbearing test rig, (d) The overhung impeller centrifugal pump.

## 4.2. Experiment Settings for Transfer Tasks

## 4.2.1. The Transfer Tasks between the Same Equipment

For the laboratory gearbox dataset, to avoid the deviation of the experimental results <sup>For</sup> <sup>the</sup> <sup>laboratory</sup> <sup>gearbox</sup> <sup>dataset,</sup> <sup>to</sup> <sup>avoid</sup> <sup>the</sup> <sup>deviation</sup> <sup>of</sup> <sup>the</sup> <sup>experimental</sup> <sup>result</sup>caused by the fault types, the shared and non-shared classes are switched when different <sup>caused</sup> <sup>by</sup> <sup>the</sup> <sup>fault</sup> <sup>types,</sup> <sup>the</sup> <sup>shared</sup> <sup>and</sup> <sup>non-shared</sup> <sup>classes</sup> <sup>are</sup> <sup>switched</sup> <sup>when</sup> <sup>differen</sup>transfer tasks are set according to the rotational speed, as shown in Table 2. The labels are set to 0, 1, and 2, according to the order in the shared class in the table. The remaining data are the non-shared part, and the label is set to 3.

Table 2. The transfer task settings in the lab gearbox dataset.

<table><tr><td>Task</td><td>Source</td><td>Target</td><td>Shared Class</td></tr><tr><td>a1</td><td>1800</td><td>1500</td><td>IR, IT, I</td></tr><tr><td>a2</td><td>1500</td><td>1800</td><td>IR, IT, I</td></tr><tr><td>a3</td><td>1800</td><td>1200</td><td>IR, OT, I</td></tr><tr><td>a4</td><td>1200</td><td>1800</td><td>IR, OT, I</td></tr><tr><td>a5</td><td>1500</td><td>1200</td><td>OT, IT, I</td></tr><tr><td>a6</td><td>1200</td><td>1500</td><td>OT, IT, I</td></tr></table>

For the bearing dataset of CWRU, the inner race fault, rolling fault, and healthy conditions data are set as the shared class, and the labels are set as 0, 1, and 2 in turn. The outer race fault data is the non-shared part, and the label is set as 3. The tasks according to the load settings are shown in Table 3.

Table 3. The transfer task settings in the CWRU dataset.

<table><tr><td>Task</td><td>Source</td><td>Target</td><td>Task</td><td>Source</td><td>Target</td></tr><tr><td>b1</td><td>0</td><td>1</td><td>b4</td><td>2</td><td>1</td></tr><tr><td>b2</td><td>1</td><td>0</td><td>b5</td><td>0</td><td>2</td></tr><tr><td>b3</td><td>1</td><td>2</td><td>b6</td><td>2</td><td>0</td></tr></table>

In the laboratory gearbox dataset experiments, the source domain contains 4500 samples, 1500 of each class. In the target domain, the number of samples per class in the shared part is 1000 and 500 samples per class in the non-shared part, totaling 4500, and the samples in the shared class in the target domain account for 66.67% of the total samples. In the experiments on the bearing dataset of CWRU, the source domain contains 3000 samples, 750 of each class, and 750 samples per class in the target domain, totaling 3000, and the samples in the shared class account for 75% of the total samples.

## 4.2.2. The Transfer Tasks between the Different Equipment

For the conditions of different equipment, the CWRU and the IMS datasets are used to verify the diagnostic effect of the proposed model. It can be found from the introduction that the test rigs for collecting the two datasets are different in terms of load, speed, etc. Select the IMS dataset and the CWRU dataset with a 3 hp load to set the transfer tasks. The shared classes are the rolling fault, outer race fault, and health conditions. Set the labels to 0, 1, and 2 in order. Set the inner race fault to only exist in the target domain, and set the label to 3. The two domains, respectively, contain 3000 samples, including 1000 samples of each class in the source domain and 750 samples of each class in the target domain.

## 4.3. Data Preprocessing

At present, most of the samples for the fault diagnosis of equipment use time domain vibration signals, or arrange time domain one-dimensional vibration signals into matrices and convert them into two-dimensional images. However, due to the complex work environment of the equipment, the collected vibration signal is usually greatly interfered with by external vibration sources, which is easy to show as nonstationary, and a single time domain cannot fully express the relationship between the collected signal and the fault. Therefore, in this paper, the time–frequency image jointly represented by the time domain and frequency domain is selected as the input of the model, which can contain more fault information.

The common time–frequency analysis includes short-time Fourier transform (STFT) and a wavelet analysis. One of the distinctions between the two methods is the basis function. The basis function of STFT is a sine signal. The original signal is constructed by the superposition of sine signals of different frequencies. In the wavelet analysis, the basis function has a lot of selectivity and can perform scale transformation, which can effectively avoid the problem of time domain resolution in STFT.

In all the transfer tasks in this paper, we transform the time domain vibration signal into time–frequency images through a wavelet analysis. First, the collected vibration signals are normalized to [0, 1] to reduce the impact of the abnormal values. Then, the overlapping sampling is carried out when the overlapping amount is 600, and the length of each sample is 1024. After the time–frequency image of the one-dimensional vibration signal is generated by the wavelet analysis, the time–frequency image is grayed to reduce the redundant information in the sample and speed up the calculation efficiency of the model.

For the model parameters, according to the experience of intelligent diagnosis, the number of epochs is set to 400, the batch size is 16, the model is optimized using Stochastic Gradient Descent (SGD), the momentum is 0.9, the learning rate is 0.001, and the sample input size is 3 ∗ 32.

## 4.4. Competitors

To prove the performance of the proposed model, some domain adaptation methods are used for experimental comparison.

(1) Deep correlation alignment (CORAL): This method reduces the difference in feature distribution by aligning the statistical features of the two domains in the feature space.

(2) DANN: As a typical domain adaptation neural network, this method correctly recognizes test samples through continuous adversaries between the feature extractor and domain classifier in training.

(3) OSBP: As the base model before improvement, this method has the function of separating the non-shared samples while the shared samples are recognized in the feature sample space.

(4) OpenMax: As a typical method to solve the open set problem, it uses the EVT to fit the Weibull distribution with known samples to build a model that can reject unknown samples.

(5) A new adversarial network with multiple auxiliary classifiers (ANMAC): This method uses multiple auxiliary classifiers to evaluate the weight of each sample and sets a soft threshold to establish a model for open set recognition [21].

## 4.5. Experimental Results and Analysis of the Same Equipment

## 4.5.1. Experimental Results

The diagnostic results of the laboratory gearbox dataset diagnostic task are shown in Table 4 and Figure 5. The average accuracy of DANN, CORAL, OpenMax, ANMAC, OSBP, and the proposed method is 65.81%, 64.34%, 82.49%, 92.34%, 85.62%, and 96.01%, respectively. The proposed method is superior to the comparison methods in all diagnostic tasks. Specifically, the average accuracy of the traditional domain adaptation models DANN and CORAL is about 65%, which is close to the proportion of shared class samples in the target domain. It can be predicted that the two methods cannot separate unknown class samples. For OpenMax, using EVT can separate some unknown class samples, but the domain adaptation problem cannot be solved well, resulting in a low accuracy. ANMAC achieved more than 90% accuracy in all diagnostic tasks, reducing the side effects of the fixed threshold in OSBP, and dramatically improved the accuracy compared with OSBP. However, the volatility of the soft threshold can reduce the model’s performance, making the accuracy unable to be further enhanced. The proposed method assigns weights to the samples, promoting a positive transfer between shared class samples and keeping non-shared class samples away by calculating the similarity, solving the problem of cross domain open set fault diagnosis.

Table 4. Diagnostic results in the lab gearbox dataset (accuracy (%) ± standard deviation).

<table><tr><td>Task</td><td>DANN</td><td>CORAL</td><td>OpenMax</td><td>ANMAC</td><td>OSBP</td><td>Proposed</td></tr><tr><td>a1</td><td>66.31 ± 0.17</td><td>64.93 ± 0.33</td><td>85.67 ± 0.79</td><td>92.87 ± 0.35</td><td>85.62 ± 0.75</td><td>97.02 ± 0.34</td></tr><tr><td>a2</td><td>64.93 ± 0.51</td><td>61.60 ± 0.79</td><td>83.27 ± 0.83</td><td>90.13 ± 0.34</td><td>86.67 ± 0.31</td><td>92.33 ± 0.53</td></tr><tr><td>a3</td><td>66.18 ± 0.33</td><td>65.76 ± 0.42</td><td>80.80 ± 0.34</td><td>91.73 ± 0.26</td><td>86.76 ± 0.58</td><td>96.49 ± 0.21</td></tr><tr><td>a4</td><td>65.31 ± 0.38</td><td>63.44 ± 0.67</td><td>81.87 ± 0.51</td><td>94.07 ± 0.87</td><td>88.07 ± 0.69</td><td>97.16 ± 0.07</td></tr><tr><td>a5</td><td>66.35 ± 0.25</td><td>65.36 ± 0.43</td><td>83.27 ± 0.87</td><td>93.31 ± 0.09</td><td>82.71 ± 0.52</td><td>96.20 ± 0.13</td></tr><tr><td>a6</td><td>65.76 ± 0.23</td><td>64.96 ± 0.57</td><td>80.07 ± 0.21</td><td>91.93 ± 0.87</td><td>83.91 ± 0.39</td><td>96.87 ± 0.30</td></tr><tr><td>AVG</td><td>65.81</td><td>64.34</td><td>82.49</td><td>92.34</td><td>85.62</td><td>96.01</td></tr></table>

The classification results of the diagnostic tasks in the CWRU dataset are shown in Table 5 and Figure 6. The average accuracy of DANN, CORAL, OpenMax, ANMAC, OSBP, and the proposed method is 75.00%, 74.51%, 91.35%, 97.58%, 89.94%, and 98.53%, respectively. Similar to the diagnosis results in the laboratory gearbox dataset, the accuracy of DANN and CORAL is close to the proportion of shared samples in the target domain, and they cannot play a role in the open set fault diagnosis. Due to the apparent fault features of the signals and the slight differences under different working conditions in this dataset, the accuracy of OpenMax was dramatically improved, reaching more than 90%.

Similarly, the reduction of the fluctuation of the soft threshold also improved the accuracyweights to the samples, promoting a positive transfer between shared class samples and of ANMAC. The proposed method is superior to the comparison methods in all diagnostickeeping non-shared class samples away by calculating the similarity, solving the problem tasks, and the low standard deviation proves that it has good diagnostic stability.of cross-domain open set fault diagnosis.

![](images/23bebc08ce211f4d172bb45fbddf4c16ccf553b540cd28b1165a7e6c8de4551c.jpg)  
Figure 5. Diagnosis performance comparisons in the lab gearbox dataset.

Table 5. Diagnosis results in the CWRU dataset (accuracy (%) ± standard deviation).

<table><tr><td>Task</td><td>DANN</td><td>CORAL</td><td>OpenMax</td><td>ANMAC</td><td>OSBP</td><td>Proposed</td></tr><tr><td>b1</td><td>75.00 ± 0</td><td>74.87 ± 0.32</td><td>90.15 ± 0.25</td><td>98.63 ± 0.23</td><td>89.80 ± 0.45</td><td>98.67 ± 0.26</td></tr><tr><td>b2</td><td>75.00 ± 0</td><td>74.33 ± 0.27</td><td>92.34 ± 0.18</td><td>98.13 ± 0.31</td><td>91.77 ± 0.37</td><td>98.67 ± 0.17</td></tr><tr><td>b3</td><td>75.00 ± 0</td><td>74.50 ± 0.08</td><td>89.57 ± 0.49</td><td>95.03 ± 0.77</td><td>89.43 ± 0.74</td><td>98.30 ± 0.22</td></tr><tr><td>b4</td><td>75.00 ± 0</td><td>75.00 ± 0</td><td>90.73 ± 0.39</td><td>98.07 ± 0.34</td><td>90.93 ± 0.23</td><td>98.83 ± 0.24</td></tr><tr><td>b5</td><td>75.00 ± 0</td><td>74.17 ± 0.41</td><td>93.70 ± 0.09</td><td>97.97 ± 0.37</td><td>88.93 ± 0.59</td><td>98.53 ± 0.11</td></tr><tr><td>b6</td><td>75.00 ± 0</td><td>74.17 ± 0.21</td><td>91.63 ± 0.12</td><td>97.63 ± 0.65</td><td>88.77 ± 0.47</td><td>98.20 ± 0.34</td></tr><tr><td>AVG</td><td>75.00</td><td>74.51</td><td>91.35</td><td>97.58</td><td>89.94</td><td>98.53</td></tr></table>

![](images/1f8d789e2176af54a36a188389b94170f5202dd694011ed1d9c790bbdae4d8dd.jpg)  
Figure 6. The diagnosis performance comparisons in the CWRU datasetFigure 6. The diagnosis performance comparisons in the CWRU dataset.

## 4.5.2. Feature Visualization and Confusion Matrix

Take the diagnosis task a1 in the laboratory gearbox dataset as an example, and randomly select some samples for feature visualization through the t-SNE algorithm, as shown in Figure 7. The confusion matrix of the diagnosis results is shown in Figure 8.

![](images/8b44e0217bb918a08d42368b29f7f485df9b8cdc8f139b35eba4e6559a9dbfcd.jpg)  
(a) Deep CORAL

![](images/584733199ab7fd4fb451ae5f5cba118894f624951853d601556d869e3c7672ec.jpg)  
(b) DANN

![](images/ee30280a1e2f90708a3c133ef0594c5f4ab715f8b5911ad9c2b6f2a92c979afb.jpg)  
(c) OpenMax

![](images/fdad59c8bbc36c484850e02f9c7a4fd29a28d146b548d53323eb1a6bd3fe64aa.jpg)

![](images/5db3e228e60041f85c64fc126f8f8207f0f5b14d5ec0732a1f3c44edd5c213a8.jpg)  
(e) OSBP

![](images/33e5e420cb70223ad116d4d16db163f7f38454c22bc57880f032e638535f0fca.jpg)

Figure 7. Feature visualization for transfer task a1. (a) Deep CORAL, (b) DANN, (c) OpenMax, (d) ANMAC, (e) OSBP, (f) Proposed.  
![](images/0a9f6cffa8b74abbc4a48e4166d2ce4402f11218c9bab95e902592c5daba41a0.jpg)

![](images/3fda5c11e7e6a7276109abd856138e0f94437e998e560c3bae9182444e162903.jpg)

![](images/8cb284574c88104fea3cb0a6a3007beefc04f994e5cb5bb251fd28727c70e880.jpg)  
(c) OpenMax

![](images/ccf937e6a965b99ca1d78b20a3ad2ebae276686b60a70283f955e600efddf885.jpg)  
(d) ANMAC

![](images/24af8b29d2ce52df4055aeb18049da807450a35d767926b9d29996293dffa073.jpg)  
(e) OSBP

![](images/34bdbf94bc2d0589f91ffb61f8ac6efeccea1623ae78c2dbd69e2650b21d3de1.jpg)  
(f) Proposed  
<sup>Figure</sup> <sup>8.</sup> <sup>The</sup> <sup>confusion</sup> <sup>matrix</sup> <sup>of</sup> <sup>the</sup> <sup>results</sup> <sup>for</sup> <sup>transfer</sup> <sup>task</sup> <sup>a1.</sup> <sup>(a)</sup> <sup>Deep</sup> <sup>CORAL,</sup> <sup>(b)</sup> <sup>DANN,</sup> <sup>(c)</sup> Figure 8. The confusion matrix of the results for transfer task a1. (a) Deep CORAL, (b) DANN, <sup>OpenMax,</sup> <sup>(d)</sup> <sup>ANMAC,</sup> <sup>(e)</sup> <sup>OSBP,</sup> <sup>(f)</sup> <sup>Proposed</sup> (c) OpenMax, (d) ANMAC, (e) OSBP, (f) Proposed.

It can be found that the traditional domain adaptation methods DANN and Deep CORAL can well align each shared class in the feature extractor’s mapping space, and the recognition accuracy is close to 100%. However, the non-shared part in the target domain is confused with other classes, and the recognition accuracy is 0, as the models cannot recognize non-shared class samples. In OpenMax, the shared class samples can be well aligned, the diagnostic accuracy reaches more than 90%, and the samples can be well aligned in the feature space. However, the non-shared class samples have severe aliasing, reducing the overall accuracy. In OSBP, there is a tendency for non-shared samples to be separated. However, due to the settings of the fixed threshold, the model has a negative transfer problem, resulting in the wrong domain alignment of the shared class samples. ANMAC alleviates the problems in OSBP. The recognition accuracy of the shared samples is close to 90%, but there is still room for improvement. In the proposed method, the positive transfer between the shared classes is promoted by the weighting module. The results show that the shared classes can be aligned in the feature space, and the recognition accuracy is over 96%. Additionally, the non-shared samples can be separated well, the confusion with the shared class samples is significantly reduced, and the recognition accuracy is close to 100%, proving the proposed method’s advantages in cross-domain open set fault diagnosis.

## 4.6. Experimental Results and Analysis of the Different Equipment

The diagnostic accuracy of this method and the comparison method in different equipment is shown in Table 6. The average accuracy rates of DANN, CORAL, OpenMax, ANMAC, OSBP, and the proposed method in the two diagnostic tasks were 43%, 37.6%, 43.4%, 67%, 60.5% and 80.2%, respectively. Specifically, due to the significant differences of the signals collected by different equipment, the performances of traditional domain adaptation methods DANN and CORAL in open set fault diagnosis are abysmal, and due to the existence of non-shared samples, many samples are aliased in the feature space alignment, resulting in a low diagnostic accuracy. As a typical method for open set recognition, OpenMax has a significant decrease in recognition accuracy due to the large differences between the two domains. OSBP and ANMAC have some effects on the open set diagnosis of different equipment, but they cannot provide stable and efficient diagnosis results. The proposed method not only promotes the alignment of each class in the shared class and the separation of non-shared samples but also uses the differences between the two classifiers to improve the model’s generalization effectively. The average diagnostic accuracy of the task can reach about 80%, which is superior to all comparable models, providing the possibility of feature transfer between different equipment.

Table 6. Diagnosis results in the different equipment (accuracy (%) ± standard deviation).

<table><tr><td>Task</td><td>DANN</td><td>CORAL</td><td>OpenMax</td><td>ANMAC</td><td>OSBP</td><td>Proposed</td></tr><tr><td>IMS → CWRU</td><td>45.93 ± 2.30</td><td>38.33 ± 1.66</td><td>47.54 ± 3.03</td><td>69.87 ± 1.91</td><td>65.50 ± 1.30</td><td>80.53 ± 1.02</td></tr><tr><td>CWRU → IMS</td><td>40.03 ± 2.35</td><td>36.83 ± 1.85</td><td>39.33 ± 1.54</td><td>64.15 ± 1.33</td><td>58.40 ± 1.74</td><td>79.93 ± 0.91</td></tr><tr><td>AVG</td><td>42.98</td><td>37.58</td><td>43.44</td><td>67.01</td><td>60.45</td><td>80.23</td></tr></table>

The feature visualization of the random sample and the confusion matrix of the diagnostic results in the transfer task IMS → CWRU are shown in Figure 9. Since the IMS is a life cycle dataset, the samples contain rich feature information, which makes the diagnosis accuracy of the IMS as the source domain relatively high. In the sharing part, most samples can accurately align the corresponding classes in the source domain, with an accuracy rate of about 85%. Only the sample in the health condition and the inner race fault sample as a non-shared class have a significant overlap, which provides feasibility for open set fault diagnosis between different equipment.

![](images/5b9f15c7da65f7c02c58985dfdcb0b275586d5f3b07e42ca24bf2a8a8cd403d5.jpg)  
(a) Feature visualization

![](images/55420e9c8cf61b10d7e5dffa49598602011a772a6abe4bb3c4f52bf3079b749b.jpg)  
(b) Confusion matrix  
<sub>→</sub>Figure 9. Results analysis of transfer task: IMS→CWRU. (a) Feature visualization, (b) Confusion matrix.

## <sup>matrix</sup> 4.7. Test on Adaptability

All of the above experimental data are from laboratories or public datasets, the inter ference in the signal is minor, and the fault features are significant. However, in the actual industry, the signal collected by sensors often has large interference, such as noise and other sound sources. The dataset of the overhung impeller centrifugal pump is used to verify the robustness and generalization of the proposed method. The dataset is collected under industrial scenes, and the signal contains large interference. The equipment shown in Figure 4d selects the rolling fault as the non-shared class in the two domains. The transfer task c1 takes the data under 743 rpm working conditions as the source domain and theEW 16 of 19 743 rpm data in the transfer task c2 as the target domain.

The diagnostic results of all the methods in the two diagnosis tasks are shown in Table 7. The average accuracy of DANN, CORAL, OpenMax, ANMAC, OSBP, and the <sup>proposed</sup> <sup>method</sup> <sup>are</sup> <sup>74.93%,</sup> <sup>72.87%,</sup> <sup>76.24%,</sup> <sup>91.13%,</sup> <sup>83.07%,</sup> <sup>and</sup> <sup>96.07%,</sup> <sup>respectively.</sup>Obviously, due to the interference in the signal, the diagnostic accuracy of all the methods Obviously, due to the interference in the signal, the diagnostic accuracy of all the methods decreased to a certain extent, and from the standard deviation, the stability of the diagnosis fluctuated. The method proposed in this paper still achieved the highest diagnostic accuracyaccuracy among all the diagnostic models, and the standard deviation was low, indicating among all the diagnostic models, and the standard deviation was low, indicating that thethat the method had good robustness and generalization. The feature visualization of the method had good robustness and generalization. The feature visualization of the randomrandom sample and the confusion matrix of the diagnostic results in the transfer task c1 sample and the confusion matrix of the diagnostic results in the transfer task c1 are shownare shown in Figure 10. It can be seen that, in the feature space, a small number of samples in Figure 10. It can be seen that, in the feature space, a small number of samples have severehave severe feature aliasing. Still, most of the samples can align well with the correspondfeature aliasing. Still, most of the samples can align well with the corresponding classes,ing classes, indicating that the method can realize the transfer of fault features during indicating that the method can realize the transfer of fault features during large interference.<sup>large</sup> <sup>interference.</sup>

<sub>Table</sub> <sub>7. Diagnosis</sub> <sub>results</sub> <sub>in</sub> <sub>the</sub> <sub>centrifugal</sub> <sub>pump</sub> <sub>dataset</sub> <sub>(accuracy</sub> <sub>(%)</sub> ± <sub>standard</sub> <sub>deviation).</sub>Table 7. Diagnosis results in the centrifugal pump dataset (accuracy (%) ± standard deviation).

<table><tr><td>Task</td><td>DANN</td><td>CORAL</td><td>OpenMax</td><td>ANMAC</td><td>OSBP</td><td>Proposed</td></tr><tr><td>c1</td><td>75.00 ± 0</td><td>72.53 ± 1.08</td><td>72.87 ± 1.68</td><td>90.13 ± 1.79</td><td>82.67 ± 1.41</td><td>95.13 ± 0.88</td></tr><tr><td>c2</td><td>74.86 ± 0.32</td><td>73.2 ± 1.17</td><td>79.61 ± 3.94</td><td>92.13 ± 1.07</td><td>83.46 ± 0.81</td><td>97.00 ± 0.54</td></tr><tr><td>AVG</td><td>74.93</td><td>72.87</td><td>76.24</td><td>91.13</td><td>83.07</td><td>96.07</td></tr></table>

![](images/3c7444d9de7f2eb85d2c70cd2c02b60a1bbe55c59787f6daead43b4366507b1c.jpg)  
(a) Feature visualization

![](images/c8a05ce4bb15a0e661d4411a34bfcc0d4da5c9fc8ca23d85b1b2661a0abcfc51.jpg)  
(b) Confusion matrix  
Figure 10. Results analysis of transfer task c1. (a) Feature visualization, (b) Confusion matrix.

## 4.8. The Limitations and Scope

From the above experimental results, the proposed method performs well in the open set fault diagnosis of rotating machinery. To evaluate the efficiency of the proposed method, Table 8 lists the average training times and model parameters of all the methods under the same conditions. The training time is the average time required for each epoch, and the parameter count is the parameter size required for building the model. As can be seen from the results in the table, since the main network model of all the methods is the same, the difference between the parameter count and the parameter size is small. In terms of training time, the CORAL and OpenMax algorithms have advantages. Other algorithms spend more time in training due to the use of adversarial learning. For the proposed method, due to the calculation of the similarity between the class center and the samples in the feature space, the training takes the most time, which is the limitation of the proposed method and one of the future research directions.

Table 8. The analysis of the complexity and computation time.

<table><tr><td>Method</td><td>Training Time (s)</td><td>Parameter Count</td><td>Params Size (MB)</td></tr><tr><td>DANN</td><td>3.49</td><td>660,742</td><td>2.52</td></tr><tr><td>CORAL</td><td>2.11</td><td>660,136</td><td>2.52</td></tr><tr><td>OpenMax</td><td>2.27</td><td>660,136</td><td>2.52</td></tr><tr><td>ANMAC</td><td>4.28</td><td>669,183</td><td>2.55</td></tr><tr><td>OSBP</td><td>2.79</td><td>660,540</td><td>2.52</td></tr><tr><td>Proposed</td><td>4.52</td><td>660,944</td><td>2.52</td></tr></table>

Regarding the application scope, the CORAL and DANN algorithms play an excellent role in closed set fault diagnosis. However, neither of these two methods can recognize unknown samples. OpenMax, ANMAC, and OSBP can realize open set fault diagnosis, but the accuracy needs to be improved. The proposed method improves the recognition accuracy in open set fault diagnosis, and the time spent is worthwhile from the results.

## 5. Conclusions and Prospects

In this paper, a cross-domain open set fault diagnosis method based on WDADC was proposed. This method makes the time-frequency image of the signal as the input of model and extracts the features of the sample. Then, a weighting module is designed to assign larger weights to samples with a higher similarity, and a weighted cross-entropy loss function is constructed to promote a positive transfer between the classes of shared samples to achieve a continuous fault diagnosis. In addition, to improve the diagnostic effect of the model on different equipment, double classifiers are designed to enhance the generalization of the model. The experimental results showed that WDADC has advantages in open set fault diagnosis. Under different working conditions of the gearbox, the average accuracy of the proposed method is more than 95%, about 30% higher than the traditional domain adaptation method and over 10% higher than the typical method for open set recognition. More importantly, the experiment has verified the feasibility of the proposed method in feature transfer between different equipment. As shown in the results, the proposed method still has an average accuracy of more than 80% in the transfer tasks between the different equipment, which is 35%, 40%, 35%, 10%, and 15% higher than that of DANN, CORAL, OpenMax, ANMAC, and OSBP. This shows that the proposed method can be extended to different equipment fault diagnoses.

The proposed method achieves the open set fault diagnosis performance well in these datasets. However, in industrial practice, the collected datasets may have the problem of imbalance, which will significantly affect the diagnosis results of the model. This will be a future research direction.

Author Contributions: All the authors contributed to the theoretical introduction and experimental design in this manuscript. All authors have read and agreed to the published version of the manuscript.

Funding: This work was supported by the Beijing Natural Science Foundation (Grant NO.L211009), the Joint Project of BRC-BC (XK2020-04), and the Open Fund Project of Key Laboratory Hunan Province of Health Maintenance Mechanical Equipment (2021JXKFJJ01).

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: Not applicable.

Conflicts of Interest: The authors have no competing interest to declare that are relevant to the content of this article

## Nomenclature

$C$ Classification module $C _ { a }$ Class center of the a-th class $D _ { i }$ Distance between the sample and class center $D _ { J S }$ Jensen–Shannon divergence $D _ { K L }$ Kullback–Leibler divergence $F _ { i } ^ { t }$ Features of the i-th target domain sample $F _ { i } ^ { s }$ Features of the i-th source domain sample $G$ Feature extraction module $L _ { C E }$ the cross-entropy loss $L _ { d }$ the total loss $L _ { s }$ Discrepancy loss of classifiers $L _ { t } ^ { i }$ the binary cross-entropy loss of the i-th target domain sample $M$ Number of fault classes in the source domain $n _ { s }$ Number of labeled samples $n _ { t }$ Number of unlabeled samples $n _ { a }$ Number of a-th class samples $p _ { x _ { i } ^ { t } } ^ { M + 1 }$ Probability that the i-th sample is recognized as a non-shared cl $p _ { c 1 }$ Probability distribution of classifier 1’s output $p _ { c 2 }$ Probability distribution of classifier $2 ^ { \prime } \mathrm { s }$ output $t$ Threshold $v$ Dimension of the Feature extraction module’s output $W$ Weighting module $w _ { i }$ Weight of the i-th sample $x _ { i } ^ { s }$ The i-th sample in the source domain $x _ { i } ^ { t }$ The i-th sample in the target domain $y _ { i } ^ { s }$ Label of the i-th sample in the source domain $y _ { i } ^ { t }$ Label of the i-th sample in the target domain

## References

1. Han, C.; Lu, W.; Wang, H.; Song, L.; Cui, L. Multistate fault diagnosis strategy for bearings based on an improved convolutional sparse coding with priori periodic filter group. Mech. Syst. Signal Process 2023, 188, 109995. [CrossRef]

2. Zhang, C.; Zhao, S.; Yang, Z.; Chen, Y. A reliable data-driven state-of-health estimation model for lithium-ion batteries in electric vehicles. Front. Energy Res. 2022, 10, 1013800. [CrossRef]

3. Zhuang, D.; Liu, H.; Zheng, H.; Xu, L.; Gu, Z.; Cheng, G.; Qiu, J. The IBA-ISMO Method for Rolling Bearing Fault Diagnosis Based on VMD-Sample Entropy. Sensors 2023, 23, 991. [CrossRef] [PubMed]

Peyrano, O.G.; Vignolo, J.; Mayer, R.; Marticorena, M. Online Unbalance Detection and Diagnosis on Large Flexible Rotors by SVR and ANN trained by Dynamic Multibody Simulations. J. Dyn. Monit. Diagn. 2022, 1, 139–147. [CrossRef]

5. Li, J.; Wang, H.; Song, L. A novel sparse feature extraction method based on sparse signal via dual-channel self-adaptive TQWT. Chin. J. Aeronaut. 2021, 34, 157–169. [CrossRef]

6. Ye, X.; Hu, Y.; Shen, J.; Chen, C.; Zhai, G. An Adaptive Optimized TVF-EMD Based on a Sparsity-Impact Measure Index for Bearing Incipient Fault Diagnosis. IEEE Trans. Instrum. Meas. 2021, 70, 1–11. [CrossRef]

7. Han, C.; Lu, W.; Wang, P.; Song, L.; Wang, H. A recursive sparse representation strategy for bearing fault diagnosis. Measurement 2022, 187, 110360. [CrossRef]

8. Wang, H.; Jing, W.; Li, Y.; Yang, H. Fault Diagnosis of Fuel System Based on Improved Extreme Learning Machine. Neural Process. Lett. 2021, 53, 2553–2565. [CrossRef]

9. Zhang, C.; Zhao, S.; He, Y. An Integrated Method of the Future Capacity and RUL Prediction for Lithium-Ion Battery Pack. IEEE Trans. Veh. Technol. 2022, 71, 2601–2613. [CrossRef]

10. Lin, L.; Pang, X.; Zhang, J.; Sun, X.; Zhang, L. System Performance and Empathetic Design Enhance User Experience for Fault Diagnosis Expert System. In Proceedings of the Engineering Psychology and Cognitive Ergonomics: 18th International Conference, EPCE 2021, Held as Part of the 23rd HCI International Conference, HCII 2021, Virtual Event, 24–29 July 2021; Springer: Berlin/Heidelberg, Germany, 2021; pp. 357–367. [CrossRef]

11. Hinton, G.E.; Salakhutdinov, R.R. Reducing the Dimensionality of Data with Neural Networks. Science 2006, 313, 504–507. [CrossRef] [PubMed]

12. Wang, H.; Lin, T.; Cui, L.; Ma, B.; Dong, Z.; Song, L. Multitask learning-based self-attention encoding atrous convolutional neural network for remaining useful life prediction. IEEE Trans. Instrum. Meas. 2022, 71, 1–8. [CrossRef]

13. Qian, C.; Zhu, J.; Shen, Y.; Jiang, Q.; Zhang, Q. Deep Transfer Learning in Mechanical Intelligent Fault Diagnosis: Application and Challenge. Neural Process. Lett. 2022, 54, 2509–2531. [CrossRef]

14. Lin, T.; Wang, H.; Guo, X.; Wang, P.; Song, L. A novel prediction network for remaining useful life of rotating machinery. Int. J. Adv. Manuf. Technol. 2023, 124, 4009–4018. [CrossRef]

15. Wang, P.; Song, L.; Guo, X.; Wang, H.; Cui, L. A High-Stability Diagnosis Model Based on a Multiscale Feature Fusion Convolutional Neural Network. IEEE Trans. Instrum. Meas. 2021, 70, 1–9. [CrossRef]

16. Zhao, S.; Zhang, C.; Wang, Y. Lithium-ion battery capacity and remaining useful life prediction using board learning system and long short-term memory neural network. J. Energy Storage 2022, 52, 104901. [CrossRef]

17. Lei, Y.; Wang, W.; Yan, T.; Li, N.; Nandi, A. Residual Convolution LSTM Network for Machines Remaining Useful Life Prediction and Uncertainty Quantification. J. Dyn. Monit. Diagn. 2021, 1, 2–8. [CrossRef]

18. Sun, J.; Gu, X.; He, J.; Yang, S.; Tu, Y.; Wu, C. A Robust Approach of Multi-sensor Fusion for Fault Diagnosis Using Convolution Neural Network. J. Dyn. Monit. Diagn. 2022, 1, 103–110. [CrossRef]

19. Zhang, Y.; Yu, K.; Ren, Z.; Zhou, S. Joint Domain Alignment and Class Alignment Method for Cross-Domain Fault Diagnosis of Rotating Machinery. IEEE Trans. Instrum. Meas. 2021, 70, 1–12. [CrossRef]

20. Zhang, G.; Li, Y.; Jiang, W.; Shu, L. A fault diagnosis method for wind turbines with limited labeled data based on balanced joint adaptive network. Neurocomputing 2022, 481, 133–153. [CrossRef]

21. Zhu, J.; Huang, C.-G.; Shen, C.; Shen, Y. Cross-Domain Open-Set Machinery Fault Diagnosis Based on Adversarial Network With Multiple Auxiliary Classifiers. IEEE Trans. Ind. Informatics 2022, 18, 8077–8086. [CrossRef]

22. Lee, K.; Han, S.; Pham, V.; Cho, S.; Choi, H.-J.; Lee, J.; Noh, I.; Lee, S. Multi-Objective Instance Weighting-Based Deep Transfer Learning Network for Intelligent Fault Diagnosis. Appl. Sci. 2021, 11, 2370. [CrossRef]

23. Zhao, B.; Cheng, C.; Zhang, G.; Lin, M.; Peng, Z.; Meng, G. An Instance and Feature-Based Hybrid Transfer Model for Fault Diagnosis of Rotating Machinery With Different Speeds. IEEE Trans. Instrum. Meas. 2022, 71, 1–12. [CrossRef]

24. Li, Q.; Shen, C.; Chen, L.; Zhu, Z. Knowledge mapping-based adversarial domain adaptation: A novel fault diagnosis method with high generalizability under variable working conditions. Mech. Syst. Signal Process. 2021, 147, 107095. [CrossRef]

25. Shao, X.; Kim, C.-S. Unsupervised Domain Adaptive 1D-CNN for Fault Diagnosis of Bearing. Sensors 2022, 22, 4156. [CrossRef]

26. Zhong, H.; Lv, Y.; Yuan, R.; Yang, D. Bearing fault diagnosis using transfer learning and self-attention ensemble lightweight convolutional neural network. Neurocomputing 2022, 501, 765–777. [CrossRef]

27. Wang, J.; Zhang, W.; Wu, H.; Zhou, J. Improved bilayer convolution transfer learning neural network for industrial fault detection. Can. J. Chem. Eng. 2022, 100, 1814–1825. [CrossRef]

28. Li, J.; Huang, R.; He, G.; Liao, Y.; Wang, Z.; Li, W. A Two-Stage Transfer Adversarial Network for Intelligent Fault Diagnosis of Rotating Machinery With Multiple New Faults. IEEE/ASME Trans. Mechatronics 2021, 26, 1591–1601. [CrossRef]

29. Kuang, J.; Xu, G.; Tao, T.; Wu, Q. Class-Imbalance Adversarial Transfer Learning Network for Cross-Domain Fault Diagnosis With Imbalanced Data. IEEE Trans. Instrum. Meas. 2022, 71, 1–11. [CrossRef]

30. He, W.; Chen, J.; Zhou, Y.; Liu, X.; Chen, B.; Guo, B. An Intelligent Machinery Fault Diagnosis Method Based on GAN and Transfer Learning under Variable Working Conditions. Sensors 2022, 22, 9175. [CrossRef]

31. Zhao, X.; Shao, F.; Zhang, Y. A Novel Joint Adversarial Domain Adaptation Method for Rotary Machine Fault Diagnosis under Different Working Conditions. Sensors 2022, 22, 9007. [CrossRef]

32. Jung, D. Data-Driven Open-Set Fault Classification of Residual Data Using Bayesian Filtering. IEEE Trans. Control. Syst. Technol. 2020, 28, 2045–2052. [CrossRef]

33. Schmidt, S.; Heyns, P.S. An open set recognition methodology utilising discrepancy analysis for gear diagnostics under varying operating conditions. Mech. Syst. Signal Process. 2019, 119, 1–22. [CrossRef]

34. Sun, X.; Ling, K.-V.; Sin, K.-K.; Liu, Y. Air Leakage Detection of Pneumatic Train Door Subsystems Using Open Set Recognition. IEEE Trans. Instrum. Meas. 2021, 70, 1–9. [CrossRef]

35. Wang, C.; Xin, C.; Xu, Z. A novel deep metric learning model for imbalanced fault diagnosis and toward open-set classification. Knowledge-Based Syst. 2021, 220, 106925. [CrossRef]

36. Zhao, C.; Shen, W. Dual adversarial network for cross-domain open set fault diagnosis. Reliab. Eng. Syst. Saf. 2022, 221, 108358. [CrossRef]

37. Yu, X.; Zhao, Z.; Zhang, X.; Zhang, Q.; Liu, Y.; Sun, C.; Chen, X. Deep-Learning-Based Open Set Fault Diagnosis by Extreme Value Theory. IEEE Trans. Ind. Informatics 2021, 18, 185–196. [CrossRef]

38. Geng, C.; Huang, S.-J.; Chen, S. Recent Advances in Open Set Recognition: A Survey. IEEE Trans. Pattern Anal. Mach. Intell. 2021, 43, 3614–3631. [CrossRef]

39. Saito, K.; Yamamoto, S.; Ushiku, Y.; Harada, T. Open Set Domain Adaptation by Backpropagation. Lect. Notes Comput. Sci. 2018, 11209, 156–171. [CrossRef]

40. Zhang, X.; Delpha, C.; Diallo, D. Incipient fault detection and estimation based on Jensen–Shannon divergence in a data-driven approach. Signal Process. 2020, 169, 107410. [CrossRef]

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.