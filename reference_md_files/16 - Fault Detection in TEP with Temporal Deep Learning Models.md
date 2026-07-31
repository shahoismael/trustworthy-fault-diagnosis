Full length article

# Fault detection in Tennessee Eastman process with temporal deep learning models

![](images/5b7a2ac028c580b3de14584f9a1ae0b412f0fdb42029b48e17f6ec86976c8da2.jpg)

Ildar Lomov <sup>a</sup>, Mark Lyubimov <sup>a</sup>, Ilya Makarov <sup>a,∗</sup>, Leonid E. Zhukov <sup>a,b,∗</sup>

<sup>a</sup> HSE University, Moscow, Russia

<sup>b</sup> Sberbank AI Lab, Moscow, Russia

## A R T I C L E I N F O

Keywords: Industrial machine learning Fault detection Tennessee Eastman process Chemical processes Deep learning Generative adversarial network Industrial data integration Industrial data management

## A B S T R A C T

Automated early process fault detection and prediction remains a challenging problem in industrial processes. Traditionally it has been done by multivariate statistical analysis of sensor readings and, more recently, with the help of machine learning methods. The quality of machine learning models strongly depends on feature engineering, that in turn heavily relies on expertise of the process engineers and model developers. With the recent advent of deep learning neural network methods and abundance of available sensor data, it became possible to develop advanced approaches to early fault detection and prediction that do not require feature engineering and provide more accurate and timely results.

In this paper we investigate a wide range of recurrent and convolutional architectures on the publicly available simulated Tennessee Eastman Process extended TEP dataset for the fault detection in chemical processes. We have selected the best architecture for the task and proposed a novel temporal CNN1D2D architecture that achieves overall better performance on the dataset than any referenced method. We have also proposed to use Generative Adversarial Network GAN to extend and enrich data used in training.

## 1. Introduction

Rapid transitioning to Industry 4.0 has significantly increased the amount of information being collected from industrial processes. Some of the incoming data is being monitored in real time by process opera tors, but most of it is being collected and stored in data base for future use. In spite of the large number of sensors and data sources, process fault prediction and prevention remains challenging problem.

Traditionally, process data is analyzed using multivariate statisti cal process monitoring (MSPM) approaches. More recently, various machine learning approaches has been introduced for fault detection problem. These approaches rely on feature engineering, manual process of windowing, combining and manipulating of raw data to help extract meaningful signal from it. Feature engineering requires deep under stating of both processes technology and machine learning methods. Carefully crafted features often allow to compensate for insufficient amount of data and rely heavily on process engineers expertise and experience [1,2].

More recently deep neural network gain increased attention and popularity in machine learning as a preferred method for image, sound and text processing, demonstrating best in class results. In essence, these approaches allows to bypass tedious feature engineering process and use raw sensor data as in input. During the training process, deep neural network finds the best signal representations (embedding) and uses it instead of manually generated features, thus training the entire learning pipeline.

At the same time, plenty of modeling challenges remain. The data often comes ‘‘dirty’’ with part of the signal corrupted by noise or malfunctioning sensors; each sensor located in a different part of the process pipeline and hence has different lag time, not obvious from the data itself, sometimes obfuscated by signal transmission delays. In addition, the available amount of labeled data is often insufficient for training the deep learning approaches [3,4]; different augmentation and simulation strategies are applied, among which Generative Adver sarial Network (GAN) are an efficient way to extend the training set that leads to overall performance improvement. From the modeling perspective, many deep learning models are not well suited for long sequence memorization and thus cannot easily handle signal lag or delays [5,6]; time series from different sensors are often well correlated following the process physics which creates additional challenges for model training.

In this paper we developed and trained deep learning architectures (DL) for Tennessee Eastman Process (TEP) fault detection problem using modified data set from [7].

The main contribution of our work is threefold:

![](images/568570e161ecccb3f5aba806484dcad40451ab28371fb3151110120cc1cdc7de.jpg)  
Fig. 1. Tennessee Eastman Process. Image credit to J.J. Downs, Computers & Chemical Engineering [8] by Elsevier, 1993.

• We selected the best RNN architecture for fault detection problem among the well known and widely used architectures

• We develop a novel temporal CNN architecture combining 1D and 2D convolutions, allowing to detect various fault patterns, handle internal data fluctuations and correlation between sensors

• We proposed a way to use generative adversarial network (GAN) to enrich and extend training data and to embed into overall solution

The paper is organized as follows. TEP Data description section describes the data set and the generating process of the TEP data set. We also discuss main challenges working with this dataset. In Related work section , a review of the recent methods is presented. We describe the motivation behind our machine learning approach in Models section. We also present detailed architecture of our solution, including Temporal CNN 1D2D and GAN dataset pipeline and baseline approach. We provide evaluation details in Experiments and Models comparison and discussion sections followed by the discussion of the limitations of our approach and some ideas for the further improvements. In Conclusion, we summarize the work we have done and outline the suggestions for further research.

## 2. TEP data description

The Tennessee Eastman Process (TEP) dataset is a commonly used benchmark for comparing different anomaly detection approaches. The process schema is shown in Fig. 1. The TEP dataset consists of two parts: standard and extended.

## 2.1. TEP flow

The TEP model is a simulation model, introduced by [8] for develop ing and testing process control and monitoring techniques. it consist of five main components (Fig. 1): reactor, condenser, compressor, separator and stripper, and eight components labeled from A to H. The gaseous reactants A, C, D, E and the inert B are fed to the reactor, where liquid products G and H are formed, and F is a byproduct.

## 2.2. Standard TEP dataset

The TEP dataset is a simulated multivariate time series correspond ing to a normal state of the system or one of multiple possible system faults. The TEP simulation can produce multiple time series example for each fault type, but typically only single example is used in the literature. We will refer to this dataset as a ‘‘standard TEP dataset’’. It contains two time series for each fault type and a normal state: shorter time series for training and longer time series for testing. Every example consists of simulated time series recordings from multiple sensors. Unfortunately, the data available per fault in this dataset is not sufficient for training deep learning models The standard TEP dataset is available by the link at MIT repository.

## 2.3. Extended TEP dataset

To enrich the data, the authors of [7] generated more examples per fault/normal state, which are identical to the standard dataset except for the random seeds used. Thus, the extended dataset has 500 examples per fault/normal state in training subset and 960 example per fault/normal state in testing subset. Every time series example contains 52 features (41 process measurements and 11 manipulated variables) that were sampled every three minutes from the running process. There are $3 * 5 0 0 / 6 0 = 2 5$ h of training data and 3 ∗ 960∕60 = 48 h of testing data. It is assumed in the model, that faults states starts after one hour in training data and after eight hours in testing data. This dataset is available at Harvard repository.

Table 1  
![](images/13b2e736b0914b0409618916e38ba2ca691d13b9c2cb2ed2cedce910d92a49ab.jpg)

![](images/5732b8646b93d718bf628168174ac3fa653917685496cdef071aa67cfe4ca57a.jpg)

![](images/c8b6a50f271bfcb28fcd087363ab8705210515d40df3de4669d5ca38a97bbfbe.jpg)

![](images/62ef5b23959b73bce856bec30412aa0e2aa142714b3bb98b94d5eb37ec501541.jpg)  
Fig. 2. All 52 features of training dataset for fault type #1, green stands for normal state, red appear in faulty state

Description of fault types in TEP. Fault ID 0 means normal work, while the other represent faults in the process.

<table><tr><td>Fault ID</td><td>Description</td><td>Type</td></tr><tr><td>0</td><td>Normal behavior</td><td>None</td></tr><tr><td>1</td><td>A/C feed ratio, B composition constant</td><td>Step</td></tr><tr><td>2</td><td>B composition, A/C ratio constant</td><td>Step</td></tr><tr><td>3</td><td>D feed temperature</td><td>Step</td></tr><tr><td>4</td><td>Reactor cooling water inlet temperature</td><td>Step</td></tr><tr><td>5</td><td>Condenser cooling water inlet temperature</td><td>Step</td></tr><tr><td>6</td><td>A feed loss</td><td>Step</td></tr><tr><td>7</td><td>C header pressure loss</td><td>Step</td></tr><tr><td>8</td><td>A, B, C feed composition</td><td>Random Variation</td></tr><tr><td>9</td><td>D feed temperature</td><td>Random Variation</td></tr><tr><td>10</td><td>C feed temperature</td><td>Random Variation</td></tr><tr><td>11</td><td>Reactor cooling water inlet temperature</td><td>Random Variation</td></tr><tr><td>12</td><td>Condenser cooling water inlet temperature</td><td>Random Variation</td></tr><tr><td>13</td><td>Reaction kinetics</td><td>Slow Drift</td></tr><tr><td>14</td><td>Reactor cooling water valve</td><td>Sticking</td></tr><tr><td>15</td><td>Condenser cooling water valve</td><td>Sticking</td></tr><tr><td>16–20</td><td>Unknown</td><td>Unknown</td></tr></table>

We formulated fault detection problem as machine learning multi class classification with 21 balanced classes: one class for normal state and twenty classes for types of fault described in Table 1. We kept all the classes, including those difficult to detect (in contrast with [9]).

An example of time series for all 52 features of simulation run #1 with fault number #1 is shown in Fig. 2,

## 3. Related work

Recent advances in process control optimization and industrial data integration and management [10–12] provide a solid platform for data-driven decision making. In [13–15] the authors showed how integration of industrial information can be utilized for industrial processes control and monitoring. For example, information integration in chemical process engineering based on semantic technologies allows to determine noteworthy potential in the flow of chemical process [16].

During the most recent decade, data-driven methodologies have brought a lot of useful insights, designing manually engineered metrics to measure deviations from normal behavior [17–19]. Unfortunately, such methods require a lot of domain specific information as a prior knowledge, since the data varies significantly among the problems in the underlying systems [20–22].

There are exist two machine learning approaches to data-driven fault detection — unsupervised and supervised. In unsupervised ap proach, a model learns the normal state of the process and then detects deviations from the normal through anomaly detection. In order to use this approach, one only needs examples of normal process behavior an no additional fault labeling is required [23]. The most well known unsupervised fault identification methods are Principal Component

Analysis (PCA) [24], Independent Component Analysis (ICA) [25], and Partial Least Squares (PLS) [26]. Multiple examples of using the above methods for TEP are given in there Refs. [27–36].

Supervised learning approach requires training of a classifier to separate normal and faulty process state based on the examples of both. It can then be used to monitor and predict known process faults in real-time manner. The main methods used in supervised learning are Support Vector Classifier (SVC) [37] and neural networks [2,38,39]. TEP fault detection via variations of SVC [40] was improved in several modifications, using PCA [41], kernel PCA [42], non-linear SVM kernels [43], and Extreme Learning Machines (ELM) [44]. In [45], combination of Hidden Markov and Bayesian networks was applied to solve the problem.

Since most of the time processes function properly, the amount of available normal process samples far exceeds the number of abnormal faulty states. The small number of fault examples and class imbalances makes prediction challenging for supervised models. A semi-supervised learning approach using can improve the model quality by taking into the account both the labeled and unlabeled data [46].

However, both supervised and semi-supervised methods have a sig nificant limitation. They can only predict types of faults available in the training data, i.e. the faults that have been observed previously [2,39] and cannot detect previously unseen process faults. Unfortunately, in real-world applications examples of all possible system faults are never available [3].

Approaches based on deep neural network architectures for chem ical process fault diagnosis has been widely studied over the last five years [4,22,39,47,48]. For example, in [3], the authors present a Deep Neural Network that is trained with the help of active learning to enable the model generalize the large amount of sensor data in order to fore cast process mishaps. In [49], a labeled autoencoder combined with k nearest neighbor was applied for industrial process monitoring, extend ing another pattern recognition method based on autoencoders [50]. In [51], authors used deep residual network with principal component analysis, improved by extended deep belief network [52]. Supervised dual stacked auto-encoders were suggested in [53,54]. Several studies have used various RNN-based models to deal with fault detection in industrial, chemical and other sorts of multivariate time-series data [55– 57]. Variational recurrent autoencoders were suggested to benefit TEP in [58]. Bi-directional RNN were applied in [59]. An overview of deep learning based models for TEP fault detection problem can be found in [60].

The lack of widely available industrial data and imbalanced classes (few process fault examples) still remains a major issue for developing of fault prediction models. This leads researchers to use industrial pro cess simulation models for data generation to train their ML algorithms [4,57,61].

More recently, a new promising approach was proposed in [62], that uses generative adversarial network GAN for data enrichment and extension.

## 4. Models

In this section we describe and discuss our models. First, we describe recurrent architectures, their construction and limitations. Next, we de scribe the most recent convolutional models, followed by the proposed Temporal CNN1D2D architecture. We finally discuss the GAN-based pipeline for the data augmentation. In all the cases, sequences of ??th sensor are denoted as $x _ { 0 } ^ { i } , \ldots , x _ { N _ { i } } ^ { i }$

Long time sequences present several challenges in modeling: they can lead to gradient vanishing or so called gradient explosion during model training and require processing of almost entire sequence be fore the sequence class (process state) is determined. Both problems can be solved by working on subsequences of data. We propose two approaches for doing it using recurrent models.

![](images/357f23d34a55324809aedb1e5b4f5dab24a7e36ee0414b84a9b59bc40828097d.jpg)  
Fig. 3. Proposed models type 1 configuration.

![](images/7dad0e7dc1c4247b05a7d256b9f24488cbfad06198a5538b33192975da176568.jpg)  
Fig. 4. Proposed models type 1 with attention configuration.

## 4.1. Recurrent models: type 1

We generated subsequences of the varying length (5, 25, 50, 100 and 250) starting from sequences’ origin and trained on them to predict corresponding target class. This allows the model to predict faults on shorter sequences and hence on early inputs.

Our approach was verified with GRU [63] and LSTM [64] models separately. Below, we provide exact models’ architectures after fine tuning. We will refer that kind of models as ‘‘type 1’’ or ‘‘GRU: type 1’’ and ‘‘LSTM: type 1’’, respectively (Fig. 3).

The described above approach could be used with attention mechanism introduced by [65]. For the output, we calculate context vector ?? that is exactly weighted sum of all hidden states $h _ { k }$ from the encoder. The attention weights are obtained from softmax function of dot product of the last hidden state ?? and the full set of hidden states $h _ { k } .$ . We will refer that kind of models as ‘‘type 1’’ or ‘‘GRU + att: type $1 ^ { \dag \dag }$ and ‘‘LSTM + att: type $^ { 1 \ ' } ,$ respectively (Fig. 4).

$$
c = \sum_ {k = 1} ^ {n} a _ {k} h _ {k}\tag{1}
$$

![](images/d2f13b1e51445deec20e0ca457d238752526657a58b08d09a2dd132ad401ac82.jpg)  
Fig. 5. Proposed Transformer model consisting of six similar layers.

![](images/4be085065bdba0092249d58dc020abcea7937d51044d60f29ba1ab8274ca08c0.jpg)  
Fig. 6. Proposed models type 2 configuration.

where alignment weight between ?? and $h _ { k }$ is

$$
a _ {k} = \frac {\exp (s c o r e (s , h _ {k}))}{\sum_ {j = 1} ^ {n} \exp (s c o r e (s , h _ {j}))}\tag{2}
$$

and score is

$$
s c o r e (s, h _ {k}) = s ^ {T} h _ {k}\tag{3}
$$

With recurrent model ‘‘type 1’’ we can apply transformer on variable length sequences to predict the only target value. Usually, transformer consists of encoder and decoder as described by [66], but here we take only encoder part and then stack fully connected layers as a class predictor. Based on the fine-tuning, six-layer transformer was selected. 1D convolutional layer with groups parameter equals to number of features means weighted sum along each feature. We will refer that kind of model as ‘‘Transformer: type 1’’ or ‘‘Transformer’’ (Fig. 5).

## 4.2. Recurrent models: type 2

Alternatively, we could address the long sequence problem by using sequence-to-sequence approach to predict target value on every time step [67]. In this approach, we can use previous sensor information as a context for predicting next process state. Obviously, we cannot apply bidirectional recurrent models here since it might leaks the future samples. We also applied the proposed model to subsequences, but unlike previous models, we used subsequences with fixed lengths of 200 elements. All subsequences were obtained by splitting source sequences into two parts. We will refer to that kind of models as ‘‘type 2’’ or ‘‘GRU: type 2’’ and ‘‘LSTM: type 2’’, respectively (see Fig. 6).

## 4.3. Temporal CNN1D and temporal CNN1D2D

It is well known, that convolutional neural networks are the best architectures for image processing tasks. At the same time, recurrent models are the most suitable for sequential data processing, includ ing multivariate time series analysis and fault detection problems. It would be advantageous to combine those approaches and use convolution to detect auto-correlation patterns within single sensor data and cross-correlation between sensors.

Temporal Convolutional Network (TCN) [68] demonstrates good results and fits our needs well. It was taken as a base line in our work after certain modifications to handle industrial sensor information.

In [68], authors propose a Temporal CNN network that combines a lot of tricks starting from AlexNet [69], where the normalization and dropout have been used to benefit from the larger datasets, to [70], where the residual connections were invented to provide an ability to avoid vanishing gradients problem for a 152-layers deep neural network, and even more. The architecture of the Temporal CNN is shown in Fig. 7 where the most interesting part is Dilated Casual Convolution, which gives the neural network ability to capture the context of a long sequence similar to the ‘‘memory’’ module in recurrent architectures.

In this work, we suggest a slightly different version of the original Temporal Convolutional Network called Temporal CNN1D. We use LeakyReLU instead of RelU because it gives increased training stability which is important for the rest of this work related to GAN-based archi tectures. Our basic temporal CNN model consists of 4 temporal blocks of 64 filters each as shown in Fig. 7. The dilation parameter increased from 1 to 4 compare to the original paper, while kernel remains the same size of 5, which gives the best result during the preliminary research. The rest of the network consists of 64 and 21 Dense layers followed by Softmax like in the default multi-class classification setting.

Furthermore, we propose combined architecture of TCN called Temporal CNN1D2D that consists of 1d separated convolutions followed by 2d regular convolutions.

The motivation for the proposed Temporal CNN1D2D neural archi tecture comes from the following discussion. First of all, the Temporal CNN model described in [68] has been well tested on a uni-variate time series data. The effectiveness of this model is due to dilated convolutions designed to capture long running patterns in the signal data (similarly to the memory module in the recurrent architectures). When Temporal CNN model is applied to the multi-variate TEP data, it creates independent representations for each signal. To overcome this difficulty we follow an idea from image processing, where 2D convolutions are efficient due to correlation between neighboring pixels. Similarly, since signals from separate sensors are correlated (due to the continuity of the process) 1 we can use 2D convolution simultaneously on the set of sensors [69]. Implementation wise, we use stacking of 2D convolution blocks with the original Temporal CNN blocks to achieve this effect.

This idea is explained by the fact that we have a problem that relates to multivariate time series data, as opposite to uni-variate set up of the [68,71]. Additional 2D convolutions aims to capture relations between individual highly correlated features of the same sample. We expect, that this idea might be more efficient than learning just the same dilated filter for each feature.

In simple words, having highly correlated industry sensors, one may predict future events based on signals from other sensors with certain delay. For example, rise of the temperature may lately broke the pressure in the process. Thus, having signal from one sensor we may make a decision combining observations of previous sensor measurements and current situation in meaningful and unified framework.

The detailed architecture of Temporal CNN1D2D is presented in Fig. 8. In particular, we use two o Temporal CNN2D blocks on top of the two explained earlier Temporal CNN1D 7 blocks to benefit from correlated features. We also use separate filters for every feature in the first 2 Temporal CNN1D blocks to take care of inbound patterns instead of generalizing all features at the same time. As the result, it makes our model more dependent on the features ordering coming from the causal dependency of sensor observations along the process.

![](images/24c374322532a7836f7ca324d78e8fe4d452ca2edd62b61ae94d43754bc47844.jpg)  
Fig. 7. Temporal CNN1D. On the green background, the CNN1d-block is presented. On the right sight, the proposed Temporal CNN1D architecture is shown where green rectangles correspond to the left-side CNN1d-block complemented with the necessary parameters.

![](images/1379d0a68f0732bd4c98e43471654de871a190319fbaf674c872c382d2957e61.jpg)  
Fig. 8. Temporal CNN1D2D. On the blue background, the CNN2d-block is presented. On the right sight, the proposed Temporal CNN1D2D architecture is shown where green rectangles correspond to the CNN1d-block from 7 and blue rectangles correspond to the left-side CNN2d-block complemented with the necessary parameters.

As for the remaining details, we use LeakyReLU as an activation function to increase training stability, which is important for the GAN based setup. The dilation parameter does not vary much, so the model cannot capture long-running dependencies as opposite to Temporal CNN1D. The lack of model flexibility is compensated by the remainder, which consists of two CNN2D blocks with increasing channels and stride, followed by 64 and 21 Dense layers with a Softmax layer on top like in the standard setting for multi-class classification setting.

## 4.4. Generative adversarial network

The very basic generator–discriminator architecture was introduced in [72]. The idea of Generative Adversarial Network (GAN) is that there are Generator ?? that produces new data samples, and Discriminator ?? that tries to distinguish real data and the data produced by Generator. The adversarial game is given by

$$
\begin{array}{l} \min _ {G} \max _ {D} F (D, G) = E _ {x \sim p _ {d} (x)} [ \ln D (x | y) ] + \\ \qquad + E _ {x \sim p _ {z} (z)} [ \ln (1 - D (G (z | y))) ], \end{array}\tag{4}
$$

where $y$ includes original data and conditions for input data. Joint learning of such a model provides new ways to saturate data with samples of the same distribution, which is indistinguishable from the real one.

$$
l _ {D}
$$

$$
l _ {D} = \lambda_ {1} l _ {C r o s s E n t r o p y L o s s} + \lambda_ {2} l _ {B i n a r y C r o s s E n t r o p y}\tag{5}
$$

Here we use a multitask set up for the discriminator. The Cross-Entropy term benefits discriminator from the fact that we know the labels of the data. This approach of multitask learning is described in [73] and improves GAN game in most cases. We set up the same value of 1.0 to $\lambda _ { 1 }$ and $\lambda _ { 2 } ,$ but minor changes are allowed and do not lead to a loss of learning stability.

The generator loss function $l _ { G }$ (G-loss) can be defined as follows

$$
l _ {G} = \lambda_ {1} l _ {\text { Similarity }} + \lambda_ {2} l _ {\text { BinaryCrossEntropy }}\tag{6}
$$

$\lambda _ { 1 }$ parameter must be many times smaller than $\lambda _ { 2 } ,$ since we want the model to generate series that are different from the original dataset. In our setting, we use $\lambda _ { 1 } = 0 . 0 0 1 \lambda _ { 2 }$ . This term is needed only to capture the scale and the distribution of the original time series values, not for its shape. We have tried both $l _ { 1 }$ and $l _ { 2 }$ distances for the similarity with $l _ { 2 }$ providing better performance.

The entire adversarial pipeline is shown in Fig. 9. We use the proposed Temporal CNN1D2D for the discriminator and a basic LSTM architecture as a generator, since CNN-based generator is very unstable and frequently leads to a ‘‘mode collapse’’ problem in GANs. We used many tricks to keep the training process stable including label smoothing and label inversion for the generator, as well as multitask setting for the discriminator. Many tips and hints on this were taken from [71.73] and described later in the Experiments.

As for the remainder details, the Fault Type Dense Head consists of 3 Dense layers with LeakyReLU(0.1) in between, to avoid sparse gradients in order to keep the training process stable [73]. The numbers of hidden units are 128, 64, 21, respectively. The Head ends with a Softmax layer like in the default multi-class classification setting. Each layer of the LSTM-based generator consists of 128 hidden units (this value was varied from 64 to 256 across experiments).

![](images/b8fa20d97359cd76841b929078f539fa09326868170edef52145a75d1d728401.jpg)  
Fig. 9. Generative Adversarial Network with Proposed Temporal CNN1D2D discriminator 8. The model enriches training data by generating samples using the GAN approach and use these samples to improve the classification solution. The LSTM-Generator gets random noise along with labels and generates a batch of fake data. Discriminator aims to distinguish reals and fakes using its Real/Fake Dense Head. At the same time, the Discriminator learns to classify given data into 21 different fault types using its second Fault Type Dense Head in a multitasking manner. The optimization is done with respect to combined G- and D-losses as shown by incoming arrows.

## 5. Experiments

Our Experiments section consists of two parts. In the first part we describe a series of experiments on training and evaluation of proposed solutions for the fault detection problem. We also describe metrics that were used to compare model performance.

The second part describes two proposed convolutional architectures and suggested GAN model. In also contains the details of the training process, together with challenges and limitations, and the proposed GAN-based data augmentation pipeline. The trained GAN model is compared to other models in terms of TPR and FPR measures.

## 5.1. Recurrent models training and comparison

We did not apply any data saturation or omitting techniques since there were not missing values in the data. Data was scaled with standard scaling (also known as normalization) approach on each dimension separately:

$$
\hat {x} _ {i} ^ {j} = \frac {x _ {i} ^ {j} - \bar {x} _ {i}}{\sigma_ {i}}\tag{7}
$$

Here $\bar { x } _ { i }$ is the mean and $\sigma _ { i }$ is the standard deviation for ??th dimension. Every proposed model has been fine-tuned by comparing mul tiple architectures and hyper-parameters. Models were trained with following configuration:

• during 100 epochs;

• with Adam optimizer [74];

• with reduce on plateau scheduler;

• initial learning rate equals 0.001

• each batch contains 64 sequences

• the best epoch states were chosen for test evaluation.

In this section we provide experimental results for the models. All models are compared by TPR, FPR, and detection delay per class. All values are rounded to two decimal places.

The three most important metrics in fault detection are TPR (true positive rate), FPR (false positive rate) and detection delay. TPR and FPR metrics are calculated as ‘‘one vs others’’, which means that we can evaluate TPR and FPR for every target class separately. We alternately fix base class and compute confusion matrix, followed by TPR and FPR metrics computation from it.<sup>1</sup>

In [75], authors use alternative metrics for model comparison: False Detection Rate (FDR) and Detection Rate (DR). FPR is the probability of wrongly detected normal condition. False alarm rate is FPR for binary problem of the fault detection vs. normal behavior.

$$
F P R = \frac {\text { number   of   normal   data   detected   as   fault }}{\text { number   of   normal   data }}\tag{8}
$$

DR is the probability of correctly detected abnormal conditions.

$$
T P R = \frac {\text { number   fault   data   that   have   been   detected   as   fault }}{\text { number   fault   data }}\tag{9}
$$

Detection delay (DD) is the time delay between the process state change and the detection of this change by the algorithm. Small detection delay is critical for the algorithm’s applicability in the real world settings.

Simulation runs which were not detected with true class are not considered in detection delay, so we measured delay only among correctly predicted instances.

![](images/c9236d45e70738b3860d247b7c7729bbbd01966a7fdec123a87ceb9ccc6e1b44.jpg)  
Fig. 10. Average detection delay per class along proposed models.

As for detection delay results (Fig. 10 and Table 2), we can see that all the models work in similar ways. 15th class is predicted later than the others, whereas the 4th, 5th, 6th, 7th classes are predicted much earlier. We cannot say that one model performs consistently better, because the situation is different from class to class. Type 2 models detect classes 1, 4, 5, 6, 7 and 14 earlier than the competitors, but need much more time steps to detect class 15. Also, model ‘‘LSTM + att type: 1’’ outperforms any other model on class 15 with about 8 time steps needed.

Detection metrics depend on the length of the sequence of data, on which they are verified. As we presented above, the worst detection delay is about 56.47 time steps to detect the 15th class. Thus, the metrics on the 60 length sequences of data should be checked as the results could differ. As it was described in TEP Data description, there is a 3 min interval between two sequential time steps, therefore we consider 3 h of data for prediction. So, we will also consider metrics for the two cases: evaluation on the full sequences, i.e. 800 time steps, and on 60 time steps after faulty state starts. The idea behind this lie in fact, that during short period sequences the fault may be captured, but there may be not enough data to correctly detect fault ID from the data.

First, we well consider performance on 60 time steps. In Fig. 11 and Table 3 we can see that most classes are detected with similar metrics except two classes: normal state and fault type #15. This problem was also mentioned in [2], where classes 3, 9 and 15 were removed from training data due to their complexity to detect.

‘‘Type 1’’ models perform similarly on most classes in terms of TPR and FPR. TPR is about 61%–71%, which is lower than type 2 models with 79%–86% on the normal state class. Also, there is a significant difference between models on class 15 in terms of TPR: from 27% of ‘‘LSTM type: 1’’ to 70% of ‘‘LSTM + att type: 1’’.

Transformer has underperformed any other model in terms of TPR and FPR. In Fig. 11 type 2 models show a higher TPR, but also a higher FPR than the competitors suggesting that these models are less conservative. The trade-off between TPR and FPR should be selected based on the costs of both TPR and FPR errors. The model reaches close to 100% accuracy, except the 0th, 9th, 15th and 20th classes. But as we mentioned before, classes #9 and #15 are the most difficult detect according to the other studies in the field.

Second, we well consider performance for all time steps. The TPR and FPR Fig. 12 and Table 4 show that type 2 models have a lower TPR on class 15, but a higher TPR on class 0 (normal state) that type 1 models. Discriminating normal state from all faulty states is a more important problem than between faulty classes. In this case type 2 model performed better than type 1 models. Moreover, it is worth noting that the Transformer model had the lowest TPR among all proposed models on classes 0, 9, 12, 18.

In the FPR figure we can see that type 2 models demonstrates 0 FPR on every class, whereas type 1 models show about 1%–2% of FPR and Transformer performs badly on classes 0, 6, 9, 12, 15 and 18.

It is worth mentioning that Transformer works worse that any other proposed model on almost every class. Most likely Transformer’s architecture is not suitable for sequences of different lengths.

For practical usage in industrial fault detection management, it is important to consider a binary detection problem between a non-faulty and any faulty state.<sup>2</sup>

Hence, we evaluated binary metrics on short and long sequences as was described above. As it is shown in Tables 5 and 6 type 2 models outperform other models in terms TPR and FPR on both sequence’s lengths: 60 and 800.

## 5.2. Convolutional and GAN-based models training and comparison

In order to compare all solutions, we have trained three models: Baseline 3-layer LSTM, a proposed Temporal CNN1D and a Temporal CNN1D2D. We also trained one more Temporal CNN1D2D with a fine tuned parameters and the same amount of training steps as a baseline. For the regular setting, we use Adam with a learning rate of 0.001 and reduce on plateau learning rate scheduler.

For the recurrent model reference, we used a 3-layer LSTM model that predicts the fault at each timestamp as described in Models, with 128 hidden units each, followed by 128, 64 and 21-dense layers with ReLU in between and a softmax as usual.

A few words regarding hyper-parameters tuning of the suggested models. The architecture of Temporal CNN1D was taken exactly as is from [68]. We only changed the activation function ReLU, since ReLU leads to unstable convergence of the discriminator part of the GAN 9. The Temporal CNN1D consists of 4 Temporal CNN blocks where each dilated convolution has 64 channels and 1, 2, 4, 8 dilation width consequently. Other parameters such as p = 0.2 of dropout, weight norm and weights initialization were taken from the best mode described in [68].

The hyper-parameters of Temporal CNN1D2D network were aligned with the Temporal CNN1D network and tuned experimentally. We preserve the same number temporal blocks 7 on the same level - 4, in order to preserve comparable amount of parameters utilized in the model. In both architectures 7 and 8 the LeakyReLU with ?? = 0.1 was used for the GAN training stability. In the Temporal CNN1D2D block at the left side of 8 the single stride and 32 and 64 channels were used as described in [69]. In addition, using architecture as a GAN discriminator imposes many restrictions on the final hyper-parameters including the Temporal CNN blocks order and size. The obtained our final architecture after multiple GAN experiments. Especially, the final network is sensitive to the number of channels in convolutional layer and, generally, to the number of parameters of the final network. The consequences of incorrect selection of hyper-parameters are described in the Discussion section later in this work

The training set up is organized as follows. Each epoch of the training refers to a single pass over the dataset, where each sample is divided into 500 training samples of a fixed length, that ends up at a certain point of a simulation run. For the majority of samples we set 60 timestamps as a sequence length. Batch is a set of 64 data samples.

For each architecture, training lasted for 40 epochs. In all the cases, this was enough for the network to converge to certain level of accuracy.

In the case of TPR, Fig. 13 shows that the Temporal CNN1D model outperforms the Baseline LSTM and the proposed Temporal CNN1D2D model in the standard training set up. The Baseline LSTM and Temporal CNN1D2D models show quite similar results in detecting classes 9, 10, 12, 15, and 0. Temporal CNN1D outperforms the described models in identifying classes 3, 4, 18, 20, but loses even Baseline LSTM in classifying types 0, 9, 15, 17. Discriminating normal state from al the faulty states is a much more significant problem than between faulty classes. Among the presented models, the best results on average are shown by the Temporal CNN1D mode, the LSTM model shows the worst results , while the Temporal CNN1D2D model demonstrates average results. The Temporal CNN1D2D has an increased number of parameters compare to Temporal CNN1, which can compensated by using more training data generated by GAN model.

Table 2  
Average detection delay per class along proposed models.

<table><tr><td>Fault</td><td>GRU type: 1</td><td>LSTM type: 1</td><td>GRU + att type: 1</td><td>LSTM + att type: 1</td><td>Transformer</td><td>GRU type: 2</td><td>LSTM type: 2</td></tr><tr><td>0</td><td>16.14</td><td>12.27</td><td>11.64</td><td>17.24</td><td>26.33</td><td>9.08</td><td>5.30</td></tr><tr><td>1</td><td>2.28</td><td>2.31</td><td>2.91</td><td>2.92</td><td>1.92</td><td>1.75</td><td>1.68</td></tr><tr><td>2</td><td>9.68</td><td>11.40</td><td>8.38</td><td>10.08</td><td>8.08</td><td>5.31</td><td>6.85</td></tr><tr><td>3</td><td>6.86</td><td>6.70</td><td>5.03</td><td>4.66</td><td>3.47</td><td>5.84</td><td>5.92</td></tr><tr><td>4</td><td>2.20</td><td>3.73</td><td>2.05</td><td>2.95</td><td>2.06</td><td>1.00</td><td>1.00</td></tr><tr><td>5</td><td>3.46</td><td>3.53</td><td>3.30</td><td>2.59</td><td>1.62</td><td>1.02</td><td>1.00</td></tr><tr><td>6</td><td>1.78</td><td>1.69</td><td>1.56</td><td>2.11</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>7</td><td>1.28</td><td>1.34</td><td>2.34</td><td>1.90</td><td>1.34</td><td>1.00</td><td>1.00</td></tr><tr><td>8</td><td>15.58</td><td>17.90</td><td>17.36</td><td>17.13</td><td>17.30</td><td>17.04</td><td>17.68</td></tr><tr><td>9</td><td>10.48</td><td>10.94</td><td>14.59</td><td>11.06</td><td>20.17</td><td>17.65</td><td>22.90</td></tr><tr><td>10</td><td>22.60</td><td>23.34</td><td>22.20</td><td>22.42</td><td>21.50</td><td>24.54</td><td>25.56</td></tr><tr><td>11</td><td>5.16</td><td>5.39</td><td>6.32</td><td>6.22</td><td>6.71</td><td>6.83</td><td>7.06</td></tr><tr><td>12</td><td>6.21</td><td>7.03</td><td>6.88</td><td>6.95</td><td>6.74</td><td>7.33</td><td>7.40</td></tr><tr><td>13</td><td>33.10</td><td>34.30</td><td>32.54</td><td>33.67</td><td>31.07</td><td>34.14</td><td>36.06</td></tr><tr><td>14</td><td>2.48</td><td>2.65</td><td>2.36</td><td>2.68</td><td>2.74</td><td>1.59</td><td>1.89</td></tr><tr><td>15</td><td>22.83</td><td>20.75</td><td>29.15</td><td>8.00</td><td>21.41</td><td>48.82</td><td>56.47</td></tr><tr><td>16</td><td>13.19</td><td>13.06</td><td>11.94</td><td>12.68</td><td>11.86</td><td>12.71</td><td>13.63</td></tr><tr><td>17</td><td>27.97</td><td>28.98</td><td>29.92</td><td>29.58</td><td>30.04</td><td>30.06</td><td>30.49</td></tr><tr><td>18</td><td>43.46</td><td>43.69</td><td>45.12</td><td>45.29</td><td>37.60</td><td>44.04</td><td>45.75</td></tr><tr><td>19</td><td>3.95</td><td>4.51</td><td>3.25</td><td>4.11</td><td>3.06</td><td>6.20</td><td>6.43</td></tr><tr><td>20</td><td>29.85</td><td>31.67</td><td>32.56</td><td>34.18</td><td>35.93</td><td>32.65</td><td>38.02</td></tr></table>

Table 3  
FPR/TPR per class along proposed models, evaluated on 60 time steps.

<table><tr><td>Fault</td><td>GRU type: 1</td><td>LSTM type: 1</td><td>GRU + att type: 1</td><td>LSTM + att type: 1</td><td>Transformer</td><td>GRU type: 2</td><td>LSTM type: 2</td></tr><tr><td>0</td><td>0.03/0.61</td><td>0.05/0.71</td><td>0.03/0.64</td><td>0.03/0.66</td><td>0.03/0.47</td><td>0.04/0.79</td><td>0.06/0.86</td></tr><tr><td>1</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>2</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>3</td><td>0.00/0.94</td><td>0.00/0.92</td><td>0.00/0.95</td><td>0.00/0.95</td><td>0.00/0.95</td><td>0.00/0.96</td><td>0.00/0.96</td></tr><tr><td>4</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>5</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>6</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>7</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>8</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>9</td><td>0.01/0.86</td><td>0.02/0.90</td><td>0.01/0.86</td><td>0.01/0.88</td><td>0.03/0.55</td><td>0.01/0.92</td><td>0.01/0.81</td></tr><tr><td>10</td><td>0.00/0.98</td><td>0.00/0.98</td><td>0.00/0.98</td><td>0.00/0.98</td><td>0.00/0.99</td><td>0.00/0.98</td><td>0.00/0.97</td></tr><tr><td>11</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>12</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>13</td><td>0.00/0.89</td><td>0.00/0.91</td><td>0.00/0.92</td><td>0.00/0.93</td><td>0.00/0.87</td><td>0.00/0.92</td><td>0.00/0.93</td></tr><tr><td>14</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>15</td><td>0.03/0.67</td><td>0.01/0.27</td><td>0.03/0.68</td><td>0.03/0.70</td><td>0.03/0.61</td><td>0.01/0.58</td><td>0.01/0.41</td></tr><tr><td>16</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>17</td><td>0.00/0.95</td><td>0.00/0.95</td><td>0.00/0.95</td><td>0.00/0.95</td><td>0.00/0.93</td><td>0.00/0.95</td><td>0.00/0.94</td></tr><tr><td>18</td><td>0.00/0.73</td><td>0.00/0.74</td><td>0.00/0.73</td><td>0.00/0.73</td><td>0.00/0.71</td><td>0.00/0.75</td><td>0.00/0.74</td></tr><tr><td>19</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>20</td><td>0.00/0.89</td><td>0.00/0.89</td><td>0.00/0.90</td><td>0.00/0.91</td><td>0.00/0.87</td><td>0.00/0.89</td><td>0.00/0.88</td></tr></table>

![](images/d5c8069d1107e6b945356cc3a6d734afac5fbb6e7ebf965f329673486577719e.jpg)

![](images/6cf878fc86f232c1bec433b812e0a13b85060c7ee20a92f3554ea361a14f03dc.jpg)  
Fig. 11. (a) TPR and (b) FPR per class along proposed models, evaluated on 60 time steps.

We have trained a Generative Adversarial Neural network as shown in Fig. 9 to generate new samples and extend the training set, so CNNs can benefit from this additional data. The training process of such a system has many pitfalls, in particular, the stability of the adversarial game is highly dependent on hyperparameters. Here is some guidance on how we improve the stability, most of them are taken from the papers [71,73]

• Use Adam(lr = 0.0002) without any schedule.

• Avoid sparse gradients. Use LeakyReLU activation, Average Pooling

![](images/6468d8a62f155430bd779a97869fcac719b086c05b9edd905e041cfcf5d4257d.jpg)

![](images/036be8e760ad51718c483c935c95a23a10f8b76cd9d75f8f93354759e3ed3411.jpg)  
Fig. 12. (a) TPR and (b) FPR per class along proposed models, evaluated on all time steps.

![](images/dfb3efef57cb150e9d20c52723a515975d208f9b03127eda91465a505b7cda32.jpg)

![](images/0c32f807f5d94bd688902c1bd0e22fcdf8c387ec9c32716f2e9dbcf231131339.jpg)  
Fig. 13. (a) TPR and (b) FPR per class along proposed models. The value for each label is computed in a ‘‘one vs all’’ manner. Inference uses the first 60 time steps of the input sequence. The red for (a) TPR line shows that the Proposed Temporal CNN1D2D model with GAN enrichment outperforms the rest in most cases. The Proposed Temporal CNN1D2D model with GAN enrichment is presented as a red line at (b) FPR. It shows competitive results compared to the rest of the models while achieving the precise targeting of the normal scenario is still a big issue for all models.

Table 4  
FPR/TPR per class along proposed models, evaluated on all time steps.

<table><tr><td>Fault</td><td>GRU type: 1</td><td>LSTM type: 1</td><td>GRU + att type: 1</td><td>LSTM + att type: 1</td><td>Transformer</td><td>GRU type: 2</td><td>LSTM type: 2</td></tr><tr><td>0</td><td>0.00/0.83</td><td>0.03/0.81</td><td>0.00/0.75</td><td>0.00/0.94</td><td>0.02/0.67</td><td>0.00/1.00</td><td>0.01/0.99</td></tr><tr><td>1</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>2</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>3</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>4</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>5</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>6</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/0.94</td></tr><tr><td>7</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>8</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>9</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.01/0.70</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>10</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>11</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>12</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.01/0.89</td><td>0.00/1.00</td><td>0.00/0.99</td></tr><tr><td>13</td><td>0.00/1.00</td><td>0.00/0.98</td><td>0.00/0.99</td><td>0.00/0.97</td><td>0.00/0.95</td><td>0.00/0.99</td><td>0.00/0.96</td></tr><tr><td>14</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>15</td><td>0.01/0.93</td><td>0.01/0.40</td><td>0.01/1.00</td><td>0.00/0.98</td><td>0.02/0.81</td><td>0.00/0.94</td><td>0.00/0.89</td></tr><tr><td>16</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>17</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>18</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/0.79</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>19</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>20</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr></table>

Table 5  
FPR/TPR per class along proposed models, evaluated on 60 time steps.

<table><tr><td>Fault</td><td>GRU type: 1</td><td>LSTM type: 1</td><td>GRU + att type: 1</td><td>LSTM + att type: 1</td><td>Transformer</td><td>GRU type: 2</td><td>LSTM type: 2</td></tr><tr><td>0</td><td>0.03/0.61</td><td>0.05/0.71</td><td>0.03/0.64</td><td>0.03/0.66</td><td>0.03/0.47</td><td>0.04/0.79</td><td>0.06/0.86</td></tr><tr><td>1</td><td>0.39/0.97</td><td>0.29/0.95</td><td>0.36/0.97</td><td>0.34/0.97</td><td>0.53/0.97</td><td>0.21/0.96</td><td>0.14/0.94</td></tr></table>

Table 6  
FPR/TPR per class along proposed models, evaluated on all time steps.

<table><tr><td>Fault</td><td>GRU type: 1</td><td>LSTM type: 1</td><td>GRU + att type: 1</td><td>LSTM + att type: 1</td><td>Transformer</td><td>GRU type: 2</td><td>LSTM type: 2</td></tr><tr><td>0</td><td>0.00/0.83</td><td>0.03/0.81</td><td>0.00/0.75</td><td>0.00/0.94</td><td>0.02/0.67</td><td>0.00/1.00</td><td>0.01/0.99</td></tr><tr><td>1</td><td>0.17/1.00</td><td>0.19/0.97</td><td>0.25/1.00</td><td>0.06/1.00</td><td>0.33/0.98</td><td>0.00/1.00</td><td>0.01/0.99</td></tr></table>

![](images/04f43c11e269634c38084ffbd8de4ecda689e65e43bde777c121f312a5817f4b.jpg)

![](images/c05c5ef9e013aeedc755473c6fd81c53401b6747be3cb50dc16db9f916ae604b.jpg)

![](images/dc900e102bd5389af1a8cac08b91c3b7d00d9d5be6cd2b770ef2d80dcb8c0c93.jpg)

![](images/47abb7171b40ea7b5dd3d8428320d2f7231d23af3662df2cf6eb58617fff329f.jpg)

![](images/eb2291583250eace06ab8a382d83a53c8282bf2023ba90ed70ca4d72042a507e.jpg)

![](images/aab423b4aae848a88e4c83172e30d61c2bdfcf9d4af7f8833de9996a6d30e07d.jpg)

![](images/73fa2f2298524ad7066898340bb4cec4465d94200ab0f11a78a997ad2cf82d9a.jpg)

![](images/7491255e7298ea4338f109a08705b78ddefc7f828b656da9688d6f30934fe106.jpg)  
Fig. 14. Four pairs of Real (each first row) and Fake (each second row) time series obtained in each case for the same sensor and the same fault type. It can be seen that all the pairs have comparable scales and similar shapes. It is the desired result for the task of the dataset enrichment, where the variety of a similar scale and shape provides better data augmentation to the training process.

• Add some noise to training: apply label smoothing [73], inverse labels for Generator with some prob, add noise to Discriminator inputs.

• Use LSTM for Generator and CNN for Discriminator. LSTM Dis criminator cannot give a stable training process.

• Train Discriminator more iterations than Generator. In particular, we trained the Generator once for every 20 steps of the Discriminator.

• Use combined loss functions and multitask learning: added MSE and CrossEntropyLoss in our case.

The proposed Temporal CNN1D2D was chosen as a Discriminator and all the parameters remain the same as it was described in Models. The only difference in Discriminator network compared to Temporal CNN1D2D architecture is in an extra head in Discriminator for the binary classification task, which consists of 64 and 21 Dense layers with a LeakyReLU(0.1) in between.

The Discriminator of the system was given batches of [64, 500, 52] size, where 500 corresponds to the sequence lengths and 52 to the number of features. Real and Fake data comes in different batches one by one. The samples in the training set were randomly shuffled at the start of each epoch. The Generator was given batches of [64, 500, 100 + 1], where 100 corresponds to a random uniform noise and 1 is a condition of a particular fault type, that is going to be generated.

In order to improve the Temporal CNN1D2D overall performance, we have increased training data in the following manner. First, we have used the weights taken from the discriminator of the GAN network as the initial state and applied the fine-tuning approach to the same amount of steps for all three models above, but with a reduced learning rate of 0.0001. Second, we have fine-tuned the model with the same amount of training steps as for the rest of the models, replacing 20% of the real data with the fake ones on which the score of the Fault Type classifier is above 0.8 for one of the suggested fault types. This approach is called an ‘‘active learning’’, but here we use it in a completely automated manner instead of hand labeling.

The idea behind additionally generated data is illustrated in Fig. 14. It is seen that the fake data has a similar shape and scale, so the classifier can profit from that distortions and learn more general rules for identifying a particular kind of issue.

After that, we have performed inference on the test set and compare the results with the rest of the models in terms of FPR and TPR. TPR and FPR for four CNN-based models are presented at Fig. 13. The experiment shows interesting results in terms of TPR and FPR metrics for Temporal CNN1D2D and the extended training data as well. The proposed Temporal CNN1D2D with the GAN enrichment model works a bit better in defining the normal condition in comparison with the referenced Temporal CNN1D model. As it was mentioned, this is one of the most important indicators, because each of these cases will lead to process interruption. In terms of identifying the hardest types of faults, the model fails in the case of 10, 20, while it overcomes the competitors in cases of 9, 12, 18, 17, and 20. Which is a more significant result, in terms of FPR, as shown at Fig. 13(b), the model overcomes the rest of models in almost all cases, excluding the fault type 19.

Summarizing the results obtained in the Experiments section, we concluded that

• our results were found to be on par or better than the previously reported results on the same data, in particular [6];

• the proposed recurrent architectures along with the Temporal CNN1D can be a go-to approach for the fault detection task due to its simplicity and accuracy;

• the normal operating state is difficult to detect, and it is often miss-classified with some other errors; this problem may be tackled using anomaly detection methods in cooperation with industrial operation cost prioritization, so that false fault detection will be treated according to the real process management, rather than from ML point of view;

• the GAN-involved data augmentation may significantly increase the proposed Temporal CNN1D2D model and make it decent in terms of TPR and even the best in terms of FPR.

The presented experiments showed the superior performance of modern deep learning architectures versus standard machine learning models reported in the Related work. Our models have been successfully applied to the synthetic TEP dataset as an open research case, however, they could be easily integrated within the industrial process management with low cost of support and minimal manual tuning.

## 6. Models comparison and discussion

In Tables 7 and 8 we compare our models performance using FPR (FDR)/TPR (DR) and FAR metrics to the state-of-the art research on TEP fault detection. Both of our models demonstrated 100% TPR in several classes: 1, 2, 4–6, 8, 19. And 0% FPR on 1–8, 10, 11, 14 , 15–20 classes. FAR metrics for recent deep learning models show performance with less than 1%, which of great importance for industrial process monitoring and control.

Table 8  
FPR/TPR per class comparison. PCA, DBN-2, DBN-SVDD, WDBN-SVDD, LE-DBN and UN-DBN models were measured on standard TEP dataset [8]. ANN, IPCA, GAN, MSDAE-TP models and our best models (‘‘GRU type: 2’’ and TCNN1D2D-GAN) were trained and tested on extended TEP dataset [7]. Not all the metrics are available for comparison.

<table><tr><td>Fault</td><td>PCA [53]</td><td>DBN-2 [52]</td><td>DBN-SVDD [75]</td><td>WDBN-SVDD [75]</td><td>LE-DBN [75]</td><td>UN-DBN [76]</td><td>ANN [6]</td><td>IPCA [6]</td><td>GAN [77]</td><td>MSDAE-TP [77]</td><td>GRU type:2 ours</td><td>TCNN1D2D-GAN ours</td></tr><tr><td>1</td><td>-/0.99</td><td>0.00/1.00</td><td>0.02/0.99</td><td>0.01/0.99</td><td>0.03/1.00</td><td>0.03/1.00</td><td>-/1.00</td><td>-/1.00</td><td>0.03/1.00</td><td>0.01/1.00</td><td>/1.00</td><td>0.00/1.00</td></tr><tr><td>2</td><td>-/0.98</td><td>0.08/1.00</td><td>0.02/0.99</td><td>0.02/0.98</td><td>0.02/0.98</td><td>0.01/0.99</td><td>-/0.99</td><td>-/0.99</td><td>0.00/0.99</td><td>0.01/0.99</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>3</td><td>-/0.002</td><td>-/-</td><td>0.01/0.07</td><td>0.01/0.01</td><td>0.01/0.02</td><td>0.02/0.11</td><td>-/-</td><td>-/-</td><td>0.1/0.1</td><td>0.00/0.22</td><td>0.00/1.00</td><td>0.00/0.96</td></tr><tr><td>4</td><td>-/0.54</td><td>0.01/1.00</td><td>0.01/0.99</td><td>0.03/0.99</td><td>0.03/1.00</td><td>0.01/1.00</td><td>-/1.00</td><td>-/1.00</td><td>0.06/0.56</td><td>0.01/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>5</td><td>-/ 0.22</td><td>0.00/0.01</td><td>0.01/1.00</td><td>0.03/0.93</td><td>0.03/1.00</td><td>0.01/1.00</td><td>-/1.00</td><td>-/1.00</td><td>0.06/0.32</td><td>0.01/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>6</td><td>-/0.99</td><td>0.00/0.01</td><td>0.03/1.00</td><td>0.02/1.00</td><td>0.03/1.00</td><td>0.01/1.00</td><td>-/1.00</td><td>-/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>7</td><td>-/1.00</td><td>0.00/0.01</td><td>0.03/1.00</td><td>0.02/1.00</td><td>0.01/1.00</td><td>0.03/1.00</td><td>-/-</td><td>-/-</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td><td>0.00/0.99</td></tr><tr><td>8</td><td>-/0.96</td><td>0.02/0.98</td><td>0.02/0.98</td><td>0.01/0.97</td><td>0.03/0.97</td><td>0.02/0.98</td><td>-/0.98</td><td>-/0.98</td><td>0.00/0.98</td><td>0.02/0.99</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>9</td><td>-/0.001</td><td>-/-</td><td>0.01/0.01</td><td>0.01/0.01</td><td>0.01/0.01</td><td>0.01/0.02</td><td>-/-</td><td>-/-</td><td>0.2/0.08</td><td>0.03/0.04</td><td>0.00/1.00</td><td>0.01/0.94</td></tr><tr><td>10</td><td>-/0.33</td><td>0.07/0.78</td><td>0.03/0.74</td><td>0.01/0.7</td><td>0.03/0.76</td><td>0.02/0.82</td><td>-/-/0.94</td><td>-/0.92</td><td>0.00/0.51</td><td>0.02/0.95</td><td>0.00/1.00</td><td>0.00/0.75</td></tr><tr><td>11</td><td>-/0.2</td><td>0.00/0.99</td><td>0.01/0.75</td><td>0.03/0.79</td><td>0.02/0.89</td><td>0.00/0.91</td><td>-/0.97</td><td>-/0.96</td><td>0.06/0.58</td><td>0.03/0.97</td><td>0.00/1.00</td><td>0.00/0.95</td></tr><tr><td>12</td><td>-/0.97</td><td>0.00/1.00</td><td>0.01/0.99</td><td>0.03/0.96</td><td>0.03/0.99</td><td>0.01/1.00</td><td>-/0.99</td><td>-/0.99</td><td>0.13/0.99</td><td>0.02/1.00</td><td>0.00/0.99</td><td>0.00/0.97</td></tr><tr><td>13</td><td>-/0.94</td><td>0.00/0.9</td><td>0.03/0.95</td><td>0.02/0.96</td><td>0.02/0.96</td><td>0.01/0.95</td><td>-/0.96</td><td>-/0.96</td><td>0.02/0.95</td><td>0.02/0.97</td><td>0.00/0.99</td><td>0.00/0.90</td></tr><tr><td>14</td><td>-/1.00</td><td>0.00/1.00</td><td>0.02/1.00</td><td>0.02/1.00</td><td>0.02/1.00</td><td>0.01/1.00</td><td>-/0.99</td><td>-/1.00</td><td>0.02/1.00</td><td>0.01/1.00</td><td>0.00/1.00</td><td>0.00/0.99</td></tr><tr><td>15</td><td>-/0.001</td><td>-/-</td><td>0.02/0.14</td><td>0.01/0.13</td><td>0.01/0.17</td><td>0.02/0.13</td><td>-/-</td><td>-/-</td><td>0.03/0.13</td><td>0.01/0.39</td><td>0.00/0.94</td><td>0.01/0.72</td></tr><tr><td>16</td><td>-/0.15</td><td>0.06/0.79</td><td>0.01/0.56</td><td>0.01/0.68</td><td>0.02/0.69</td><td>0.02/0.64</td><td>-/0.95</td><td>-/0.96</td><td>0.2/0.34</td><td>01/0.92</td><td>0.00/1.00</td><td>0.00/0.98</td></tr><tr><td>17</td><td>-/0.74</td><td>0.00/1.00</td><td>0.01/0.96</td><td>0.01/0.97</td><td>0.02/0.98</td><td>0.02/0.98</td><td>-/0.96</td><td>-/0.97</td><td>0.02/0.91</td><td>0.02/0.98</td><td>0.00/1.00</td><td>0.00/0.97</td></tr><tr><td>18</td><td>-/0.88</td><td>0.00/0.93</td><td>0.03/0.90</td><td>0.00/0.90</td><td>0.01/0.91</td><td>0.01/0.89</td><td>-/0.94</td><td>-/0.95</td><td>0.02/0.90</td><td>0.01/0.94</td><td>0.00/1.00</td><td>0.00/0.85</td></tr><tr><td>19</td><td>-/0.14</td><td>0.03/0.97</td><td>0.02/0.59</td><td>0.01/0.80</td><td>0.01/0.91</td><td>0.03/0.98</td><td>-/0.99</td><td>-/0.99</td><td>0.01/0.12</td><td>0.02/1.00</td><td>0.00/1.00</td><td>0.00/1.00</td></tr><tr><td>20</td><td>-/0.31</td><td>0.00/0.93</td><td>0.01/0.82</td><td>0.03/0.68</td><td>0.03/0.74</td><td>0.02/0.87</td><td>-/0.94</td><td>-/0.96</td><td>0.00/0.58</td><td>0.01/0.92</td><td>0.00/1.00</td><td>0.00/0.88</td></tr><tr><td>21</td><td>-/0.26</td><td>0.01/0.85</td><td>0.02/0.52</td><td>0.01/0.44</td><td>0.03/0.34</td><td>0.03/0.50</td><td>-/-</td><td>-/-</td><td>0.06/0.5</td><td>0.02/0.60</td><td>-/-</td><td>-/-</td></tr></table>

FAR per class comparison. LE-DBN model was measured on standard TEP dataset [8]. SVM, ANN, IPCA, QC, TSSAE models and our best models (‘‘GRU type: 2’’ and TCNN1D2D-GAN) were trained and tested on extended TEP dataset [7]. Not all the metrics are available for comparison.

<table><tr><td>Fault</td><td>LE-DBN [75]</td><td>SVM [43]</td><td>ANN [6]</td><td>IPCA [6]</td><td>QC [22]</td><td>TSSAE [53]</td><td>GRU type: 2 ours</td></tr><tr><td>1</td><td>0.01</td><td>0.03</td><td>0.00</td><td>0.00</td><td>0.05</td><td>0.01</td><td>0.00</td></tr></table>

In Table 9 we show comparison of the models in terms of reported detection delay, with on-par performance in certain classes with LE-DBN [75], while outperforming other models and LE-DBN on most of the faults.

We believe that the reason for high performance of our models on both FRD/DR and Detection Delay metrics are due not only to the advanced deep learning architectures, but also to the process of GAN data enrichment stabilizing training process. IT is especially seen on the ‘hard’ classes of faults ‘3’, ‘9’, ‘15’, which are usually omitted in studies or exhibit poor model performance.

The training process of the Generative Adversarial Neural Network is extremely challenging, especially when using conditional injection to the generator. It occasionally encounters ‘‘mode collapse’’ problem described in [73,78]. During the experiments, we have found two basic shapes of data where the discriminator tends to converge. See Fig. 15 for details.

Many tips and tricks were described in [73] and tested in our experiments as we described in the previous section. We also want to mention a Wasserstein’s loss [79] and the use of Adaptive Instance Normalization [80] in layers of Generator and Discriminator networks. These two methods have a high potential to provide the system stability from run to run and can be selected for the further improvement of the GAN enrichment pipeline.

Our model especially shines on the faults that are notoriously difficult to detect and where most of other methods demonstrate poor performance. We can associate good performance on these errors with a combination of factors. First, we use more complex architectures for our neural networks with a large number of parameters. We can do so this thanks to the proposed data processing using the GAN enrichment pipeline which generates more data for training. Second, errors ‘3’ and ‘9’ are associated with the temperature of different refining nodes. Temperature changes affect many system nodes and with a strong delay. In our work, we take an advantage from simultaneous flow of 52 data sensors instead of working with each sensor data separately like in [5]. Moreover, the feature of capturing the error patterns from the correlated adjacent sensors is in design of our GPU-based and Temporal CNN1D2D models by default.

## 7. Conclusion

In this work we focused on developing and testing approaches for fault detection on extended TEP dataset. To establish the baseline, we have tested a wide range of recurrent and convolutional architectures, such as LSTM, GRU, RNN with attention and Transformer, as well as Temporal CNN1D and selected the optimal architecture

We have proposed a novel Temporal CNN1D2D architecture and compared it to the baseline using TPR/PFR metrics. We have shown the superior performance of our algorithm compare to the baseline and peers, especially on the most challenging to detect faults 3, 9, 15. We have also proposed to use Generative Adversarial Neural Network along with the multitask objective function to extend and enrich data samples used in training.

Our results show that ‘‘GRU: type2’’ model and a GAN based data enrichment pipeline along with Temporal CNN1D2D generator provides the best overall results on extended TEP dataset. We plan to release the code from our experiments along with the paper publica tion.

## CRediT authorship contribution statement

Ildar Lomov: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing- origina draft. Mark Lyubimov: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing- orig inal draft. Ilya Makarov: Conceptualization, Methodology, Valida tion, Formal analysis, Investigation, Resources, Writing- original draft, Writing - review & editing, Visualization, Supervision, Project admin istration. Leonid E. Zhukov: Conceptualization, Validation, Formal analysis, Investigation, Writing- original draft, Writing - review & editing, Supervision, Project administration.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Table 9  
Detection delay per class comparison. DBN-SVDD, WDBN-SVDD, LE-DBN, PCA, SA models were measured on standard TEP dataset [8]. Our models were trained and tested on extended TEP dataset [7]. Not all the metrics are available for comparison.

<table><tr><td>Fault</td><td>DBN-SVDD [75]</td><td>WDBN-SVDD [75]</td><td>LE-DBN [75]</td><td>PCA [58]</td><td>SA [58]</td><td>GRU type: 2 ours</td><td>TCNN1D2D-GAN ours</td></tr><tr><td>1</td><td>7</td><td>7</td><td>0</td><td>9</td><td>6</td><td>1</td><td>1</td></tr><tr><td>2</td><td>5</td><td>5</td><td>13</td><td>36</td><td>21</td><td>4</td><td>3</td></tr><tr><td>3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>5</td><td>42</td></tr><tr><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>1</td></tr><tr><td>5</td><td>1</td><td>1</td><td>0</td><td>0</td><td>6</td><td>0</td><td>2</td></tr><tr><td>6</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>1</td></tr><tr><td>7</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>1</td></tr><tr><td>8</td><td>17</td><td>17</td><td>57</td><td>57</td><td>48</td><td>16</td><td>28</td></tr><tr><td>9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>17</td><td>25</td></tr><tr><td>10</td><td>7</td><td>7</td><td>19</td><td>120</td><td>18</td><td>24</td><td>20</td></tr><tr><td>11</td><td>7</td><td>7</td><td>3</td><td>15</td><td>21</td><td>6</td><td>6</td></tr><tr><td>12</td><td>2</td><td>2</td><td>2</td><td>6</td><td>0</td><td>6</td><td>14</td></tr><tr><td>13</td><td>40</td><td>40</td><td>10</td><td>108</td><td>117</td><td>33</td><td>48</td></tr><tr><td>14</td><td>1</td><td>1</td><td>0</td><td>0</td><td>6</td><td>1</td><td>4</td></tr><tr><td>15</td><td>237</td><td>237</td><td>22</td><td>-</td><td>-</td><td>48</td><td>39</td></tr><tr><td>16</td><td>18</td><td>18</td><td>11</td><td>33</td><td>39</td><td>12</td><td>6</td></tr><tr><td>17</td><td>20</td><td>20</td><td>19</td><td>72</td><td>24</td><td>29</td><td>25</td></tr><tr><td>18</td><td>14</td><td>14</td><td>16</td><td>180</td><td>102</td><td>43</td><td>40</td></tr><tr><td>19</td><td>10</td><td>10</td><td>10</td><td>30</td><td>15</td><td>5</td><td>6</td></tr><tr><td>20</td><td>8</td><td>8</td><td>15</td><td>246</td><td>54</td><td>32</td><td>15</td></tr><tr><td>21</td><td>250</td><td>250</td><td>261</td><td>39</td><td>18</td><td>-</td><td>-</td></tr></table>

![](images/e1314ed027ea324e2354d1f5ac21c7a0664794bdf2ec66a1a81563d603485e9b.jpg)

![](images/ce2144a5e4b7ebe64a5a6c7d0021ef0d154868f3d85b6d9bc014be328492ef4f.jpg)

![](images/0d64615fa54ab80913c863a176d56bc6709814ab4c9d5c17908b9a7e27afc377.jpg)  
Fig. 15. A pair of the most common examples for the ‘‘mode collapse’’ scenario described in [73,78]. Left side values fluctuate around the same value with a huge amplitude. This case is common for recurrent generators. Right side values go up and fluctuate around the same value with a small amplitude. This case is common for convolutional generators, which can be. for example. the Proposed Temporal CNN1D2D architecture after a small modification.

## Funding description

The research is supported by the Russian Science Foundation under grant 21-11-00045. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

## References

[1] I. Yélamos, G. Escudero, M. Graells, L. Puigjaner, Performance assess ment of a novel fault diagnosis system based on support vector machines, Comput. Chem. Eng. 33 (1) (2009) 244–255, http://dx.doi.org/10.1016/j. compchemeng.2008.08.008, URL: http://www.sciencedirect.com/science/article/ pji/S0098135408001671

[2] M. Grbovic, W. Li, N.A. Subrahmanya, A.K. Usadi, S. Vucetic, Cold start approach for data-driven fault detection, JEEE Trans, Ind. Inf, 9 (4) (2013) 2264–2273

[3] P. Jiang, Z. Hu, J. Liu, S. Yu. F. Wu, Fault diagnosis based on chemical sensor data with an active deep neural network, Sensors 16 (2016) 1695, http://dx.doi.org/10.3390/s16101695.

[4] P. Filonov, F. Kitashov, A. Lavrentyev, RNN-based early cyber-attack detection for the Tennessee Eastman process, 2017, arXiv:1709.02232

[5] S. Heo, J.H. Lee, Fault detection and classification using artificial neural networks. JFAC-PapersOnLine 51 (18) (2018) 470–475.

[6] N. Basha, M.Z. Sheriff, C. Kravaris, H. Nounou, M. Nounou, Multiclass data classification using fault-detection-based techniques, Comput. Chem. Eng. (2020) 106786.

[7] C.A. Rieth, B.D. Amsel, R. Tran, M.B. Cook, Issues and advances in anomaly detection evaluation for joint human-automated systems, in: Internationa Conference on Applied Human Factors and Ergonomics, Springer, 2017. pp 52–63

[8] J.J. Downs, E.F. Vogel, A plant-wide industrial process control problem, Comput. Chem. Eng. 17 (3) (1993) 245–255

[9] M. Grbovic, W. Li, N.A. Subrahmanya, A.K. Usadi, S. Vucetic, Cold start approach for data-driven fault detection, JEEE Trans. Ind. Inf, 9 (4) (2013) 2264–2273

[10] E. Hämäläinen, T. Inkinen, Industrial applications of big data in disruptive innovations supporting environmental reporting, J. Ind. Inf. Integr. 16 (2019) 100105.

[11] G. Shao, H. Latif, C. Martin-Villalba, P. Denno, Standards-based integration of advanced process control and optimization, J. Ind. Inf. Integr. 13 (2019) 1–12

[12] S. Yin, J.J. Rodriguez-Andina, Y. Jiang, Real-time monitoring and control of industrial cyberphysical systems: With integrated plant-wide monitoring and control framework, IEEE Ind. Electron. Mag. 13 (4) (2019) 38–47.

[13] Y. Chen. Industrial information integration—A literature review 2006-2015. J Ind. Inf, Integr, 2 (2016) 30–64

[14] Y. Chen, A survey on industrial information integration 2016–2019, J. Ind. Integr. Manag. 5 (01) (2020) 33–163.

[15] L.D. Xu, Industrial information integration – An emerging subject in industrializa tion and informatization process, J. Ind. Inf. Integr. 17 (2020) 100128, http://dx. doi.org/10.1016/j.jii.2020.100128, URL: http://www.sciencedirect.com/science article/pii/S2452414X20300042

[16] A. Wiesner, J. Morbach, W. Marquardt, Information integration in chemical process engineering based on semantic technologies, Comput. Chem. Eng. 35 (4) (2011) 692–708.

[17] C. Bedoya, C. Uribe, C. Isaza, Unsupervised feature selection based on fuzzy clus tering for fault detection of the Tennessee Eastman process, in: Ibero-American Conference on Artificial Intelligence, Springer, 2012, pp. 350–360.

[18] M.A.A. Rad, M.J. Yazdanpanah, Designing supervised local neural network classifiers based on EM clustering for fault diagnosis of Tennessee Eastman process, Chemometr, Intell, Lab. Syst, 146 (2015) 149–157.

[19] R. Fezai, M. Mansouri, K. Abodayeh, H. Nounou, M. Nounou, Online reduced gaussian process regression based generalized likelihood ratio test for fault detection, J. Process Control 85 (2020) 30–40.

[20] C. Liu, K.G. Lore, Z. Jiang, S. Sarkar, Root-cause analysis for time-serie anomalies via spatiotemporal graphical modeling in distributed complex systems, 2018, arXiv:1805.12296.

[21] Y. Jiang, S. Yin, O. Kaynak, Optimized design of parity relation based residua generator for fault detection: Data-driven approaches, IEEE Trans. Ind. Inf. (2020).

[22] A. Ajagekar, F. You, Quantum computing assisted deep learning for fault detection and diagnosis in industrial process systems, 2020, arXiv:2003.00264.

[23] S. Yin, S.X. Ding, A. Haghani, H. Hao, P. Zhang, A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process, J. Process Control 22 (9) (2012) 1567–1581.

[24] S. Wold, K. Esbensen, P. Geladi, Principal component analysis, Chemometr. Intell. Lab. Syst. 2 (1–3) (1987) 37–52.

[25] A. Hyvärinen, E. Oja, Independent component analysis: algorithms and applications, Neural Netw. 13 (4–5) (2000) 411–430.

[26] T. Mehmood, K.H. Liland, L. Snipen, S. Sæbø, A review of variable selection methods in partial least squares regression, Chemometr. Intell. Lab. Syst. 118 (2012) 62–69.

[27] G. Lee, C. Han, E.S. Yoon, Multiple-fault diagnosis of the Tennessee Eastman process based on system decomposition and dynamic PLS, Ind. Eng. Chem. Res. 43 (25) (2004) 8037–8048.

[28] G. Li, C.F. Alcala, S.J. Qin, D. Zhou, Generalized reconstruction-based contri butions for output-relevant fault diagnosis with application to the Tennessee Eastman process, IEEE Trans. Control Syst. Technol. 19 (5) (2010) 1114–1127.

[29] M.N. Nashalji, M.A. Shoorehdeli, M. Teshnehlab, Fault detection of the Tennessee Eastman process using improved PCA and neural classifier, in: Soft Computing in Industrial Applications, Springer, 2010, pp. 41–50.

[30] C. Lau, K. Ghosh, M.A. Hussain, C.C. Hassan, Fault diagnosis of Tennessee Eastman process with multi-scale PCA and ANFIS, Chemometr. Intell. Lab. Syst. 120 (2013) 1–14.

[31] T.J. Rato, M.S. Reis, Fault detection in the Tennessee Eastman benchmark process using dynamic principal components analysis based on decorrelated residuals (DPCA-DR), Chemometr. Intell. Lab. Syst. 125 (2013) 101–108.

[32] J. Dong, K. Zhang, Y. Huang, G. Li. K. Peng, Adaptive total PLS based quality: relevant process monitoring with application to the Tennessee Eastman process, Neurocomputing 154 (2015) 77–85.

[33] S. Yin, C. Yang, J. Zhang, Y. Jiang, A data-driven learning approach for nonlinea process monitoring based on available sensing measurements, IEEE Trans. Ind. Electron, 64 (1) (2016) 643–653

[34] C. Hu, Z. Xu, X. Kong, J. Luo, Recursive-CPLS-based quality-relevant and process relevant fault monitoring with application to the Tennessee Eastman process, JEEE Access 7 (2019) 128746–128757

[35] D. Sun, X. Gong, Y. Chen, Integrating canonical variate analysis and kerne independent component analysis for Tennessee Eastman process monitoring, J. Chem. Eng, Japan 53 (3) (2020) 126–133.

[36] C. Zhang, Q. Guo, Y. Li, Fault detection in the Tennessee Eastman benchmark process using principal component difference based on K-nearest neighbors, IEEE Access 8 (2020) 49999–50009

[37] D. Lee. J. Lee. Domain described support vector classifier for multi-classification problems, Pattern Recognit. 40 (1) (2007) 41–51.

[38] D. Xie, L. Bai, A hierarchical deep neural network for fault diagnosis on Tennessee-Fastman process in: 2015 JFFF 14th International Conference or Machine Learning and Applications, ICMLA, IEEE, 2015, pp. 745–748.

[39] W. Sun, A.R.C. Paiva, P. Xu, A. Sundaram, R.D. Braatz, Fault detection and identification using Bavesian recurrent neural networks. 2019. arXiv:1911.04386.

[40] A. Kulkarni, V.K. Jayaraman, B.D. Kulkarni, Knowledge incorporated support vector machines to detect faults in Tennessee Eastman process, Comput. Chem. Eng. 29 (10) (2005) 2128–2133.

[41] X. Gao, J. Hou, An improved SVM integrated GS-PCA fault diagnosis approach of Tennessee Eastman process, Neurocomputing 174 (2016) 906–911.

[42] K. Zhang, K. Qian, Y. Chai, Y. Li, J. Liu, Research on fault diagnosis of tennessee eastman process based on kpca and SVM, in: 2014 Seventh Internationa Symposium on Computational Intelligence and Design, Vol. 1, IEEE, 2014, pp. 490–495.

[43] M. Onel, C.A. Kieslich, E.N. Pistikopoulos, A nonlinear support vector machine based feature selection approach for fault detection and diagnosis: Application to the Tennessee Eastman process, AIChE J. 65 (3) (2019) 992–1005.

[44] F. de Assis Boldt, T.W. Rauber, F.M. Varejão, Cascade feature selection and elm for automatic fault diagnosis of the tennessee eastman process, Neurocomputing 239 (2017) 238–248.

[45] M.G. Don, F. Khan, Dynamic process fault detection and diagnosis based on a combined approach of hidden Markov and Bayesian network model, Chem. Eng. Sci. 201 (2019) 82–96.

[46] R. He, G. Chen, C. Dong, S. Sun, X. Shen, Data-driven digital twin technology for optimized control in process systems. ISA Trans, 95 (2019) 221–234.

[47] A. Sheta, M. Braik, H. Al-Hiary, Modeling the Tennessee Eastman chemica process reactor using bio-inspired feedforward neural network (BI-FF-NN). Int. J. Adv, Manuf Technol 103 (1–4) (2019) 1359–1380

[48] M. Adeli. A. Mazinan. High efficiency fault-detection and fault-tolerant con: trol approach in Tennessee Eastman process via fuzzy-based neural network representation, Complex Intell. Syst. 6 (1) (2020) 199–212.

[49] S. Yan, X. Yan, Using labeled autoencoder to supervise neural network combined with k-nearest neighbor for visual industrial process monitoring, Ind. Eng. Chem. Res, 58 (23) (2019) 9952–9958

[50] J. Yu, X. Zheng, S. Wang, A deep autoencoder feature learning method for process pattern recognition, J. Process Control 79 (2019) 1–15.

[51] Y. Wang, H. Li, Complex chemical process operation evaluations using a novel analytic hierarchy process model integrating deep residual network with principal component analysis, Chemometr. Intell. Lab. Syst. 191 (2019) 118–128.

[52] Y. Wang, Z. Pan, X. Yuan, C. Yang, W. Gui, A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network, ISA Trans. 96 (2020) 457–467.

[53] S. Yan, X. Yan, Design teacher and supervised dual stacked auto-encoders for quality-relevant fault detection in industrial process, Appl. Soft Comput. 81 (2019) 105526.

[54] J. Yu, X. Yan, Multiscale intelligent fault detection system based on agglomer ative hierarchical clustering using stacked denoising autoencoder with tempora information, Appl. Soft Comput. 95 (2020) 106525.

[55] P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, G.M. Shroff, LSTM-based encoder-decoder for multi-sensor anomaly detection, 2016, CoRR abs/1607.00148. URL: http://arxiv.org/abs/1607.00148, arXiv:1607.00148.

[56] M. Yadav, P. Malhotra, L. Vig, K. Sriram, G.M. Shroff, ODE - Augmented training improves anomaly detection in sensor data from machines, 2016, CoRR abs/1605.01534. URL: http://arxiv.org/abs/1605.01534, arXiv:1605.01534.

[57] P. Filonov, A. Lavrentyev, A. Vorontsov, Multivariate industrial time series with cyber-attack simulation: Fault detection using an LSTM-based predictive data model, 2016, CoRR abs/1612.06676. URL: http://arxiv.org/abs/1612.06676, arXiv:1612.06676.

[58] E. Cheng. O.P. He. J. Zhao. A novel process monitoring approach based on variational recurrent autoencoder, Comput. Chem. Eng. 129 (2019) 106515.

[59] G.S. Chadha, A. Panambilly, A. Schwung, S.X. Ding, Bidirectional deep recurrent neural networks for process fault classification, ISA Trans. (2020).

[60] G.S. Chadha, A. Schwung, Comparison of deep neural network architectures for fault detection in Tennessee Eastman process, in: 2017 22nd IEEE International Conference on Emerging Technologies and Factory Automation, ETFA, IEEE, 2017, pp. 1–8.

[61] C.A. Rieth, B.D. Amsel, R. Tran, M.B. Cook, Additional Tennessee Eastman pro cess simulation data for anomaly detection evaluation, 2017, Harvard Dataverse URL: https://doi.org/10.7910/DVN/6C3JR1

[62] X. Yang, D. Feng, Generative adversarial network based anomaly detection on the benchmark Tennessee Eastman process, in: 2019 5th International Conference on Control, Automation and Robotics, ICCAR, IEEE, 2019, pp. 644–648.

[63] K. Yao, T. Cohn, K. Vylomova, K. Duh, C. Dyer, Depth-gated LSTM, 2015, arXiv preprint arXiv:1508.03790.

[64] S. Hochreiter, J. Schmidhuber, Long short-term memory, Neural Comput. 9 (8) (1997).1735–1780.

[65] D. Bahdanau, K. Cho, Y. Bengio, Neural machine translation by jointly learning to align and translate, 2014, arXiv preprint arXiv:1409.0473.

[66] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems, 2017, pp. 5998–6008.

[67] Y. Belinkov, Y. Bisk, Synthetic and natural noise both break neural machine translation. 2017. arXiv preprint arXiv:1711.02173.

[68] S. Bai, J.Z. Kolter, V. Koltun, An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, 2018, CoRR abs/1803.01271. URL: http://arxiv.org/abs/1803.01271, arXiv:1803.01271.

[69] A. Krizhevsky, I. Sutskever, G.E. Hinton, ImageNet classification with deep convolutional neural networks, in: NIPS, 2012, pp. 1097–1105.

[70] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2016, pp. 770–778.

[71] C. Esteban, S.L. Hyland, G. Rätsch, Real-valued (medical) time series generation with recurrent conditional GANs, 2017, arXiv:1706.02633

[72] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial nets, in: Advances in Neura Information Processing Systems, 2014, pp. 2672–2680.

[73] T. Salimans, I.J. Goodfellow, W. Zaremba, V. Cheung, A. Radford, X. Chen, Improved techniques for training GANs, 2016, CoRR abs/1606.03498. URL: http://arxiv.org/abs/1606.03498. arXiv:1606.03498.

[74] D.P. Kingma, J. Ba, Adam: A method for stochastic optimization, 2014, arXiv: 1412.6980.

[75] J. Yu, X. Yan, Layer-by-layer enhancement strategy of favorable features of the deep belief network for industrial process monitoring. Ind. Eng. Chem. Res. 57 (2018) http://dx.doi.org/10.1021/acs.iecr.8b04689.

[76] J. Yu, X. Yan, Whole process monitoring based on unstable neuron output information in hidden lavers of deep belief network, IEEE Trans. Cybern. (2019).

[77] J. Yu, X. Yan, Multiscale intelligent fault detection system based on ag glomerative hierarchical clustering using stacked denoising autoencoder with temporal information, Appl. Soft Comput. 95 (2020) 106525, http://dx.doi. org/10.1016/j.asoc.2020.106525, URL: http://www.sciencedirect.com/science/ article/pii/S1568494620304646.

[78] M. Mirza, S. Osindero, Conditional generative adversarial nets, 2014, arXiv: 1411.1784.

[80] X. Huang, S.J. Belongie, Arbitrary style transfer in real-time with adaptive instance normalization, 2017, CoRR abs/1703.06868. URL: http://arxiv.org/abs/ 1703.06868, arXiv:1703.06868

[79] M. Arjovsky, S. Chintala, L. Bottou, Wasserstein GAN, 2017, arXiv:1701.07875.