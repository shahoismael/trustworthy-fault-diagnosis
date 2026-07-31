# Accepted Manuscript

Deep convolutional neural network model based chemical process fault diagnosis

Hao Wu , Jinsong Zhao

Computers & Chemical Engineering

![](images/743a2b06ffc729d560f2fd44c1e743987bf8207ab4c3ab16bed5311ef87b1569.jpg)

PII: S0098-1354(18)30299-0 DOI: 10.1016/j.compchemeng.2018.04.009 Reference: CACE 6077

To appear in: Computers and Chemical Engineering

Received date: 17 November 2017

Revised date: 4 April 2018

Accepted date: 9 April 2018

Please cite this article as: Hao Wu , Jinsong Zhao , Deep convolutional neural network model based chemical process fault diagnosis, Computers and Chemical Engineering (2018), doi: 10.1016/j.compchemeng.2018.04.009

This is a PDF file of an unedited manuscript that has been accepted for publication. As a service to our customers we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and review of the resulting proof before it is published in its final form. Please note that during the production process errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.

## Highlights

 A deep convolutional neural network model based fault diagnosis method is proposed for chemical processes.

 A deep convolutional neural network model is constructed and applied in the Tennessee Eastman process.

 An average fault diagnosis rate of 88.2% is achieved.

 The model tuning and the dynamic diagnostic performance are explored.

# Deep convolutional neural network model based chemical process fault diagnosis

Hao Wu <sup>a</sup>, Jinsong Zhao <sup>a,b,\*</sup>

<sup>a</sup> State Key Laboratory of Chemical Engineering, Department of Chemical Engineering,

Tsinghua University, Beijing, China

<sup>b</sup> Beijing Key Laboratory of Industrial Big Data System and Application, Tsinghua University, Beijing, China

## Abstract

Numerous accidents in chemical processes have caused emergency shutdowns, property losses, casualties and/or environmental disruptions in the chemical process industry. Fault detection and diagnosis (FDD) can help operators timely detect and diagnose abnormal situations, and take right actions to avoid adverse consequences. However, FDD is still far from widely practical applications. Over the past few years, deep convolutional neural network (DCNN) has shown excellent performance on machine-learning tasks. In this paper, a fault diagnosis method based on a DCNN model consisting of convolutional layers, pooling layers, dropout, fully connected layers is proposed for chemical process fault diagnosis. The benchmark Tennessee Eastman (TE) process is utilized to verify the outstanding performance of the fault diagnosis method.

## Keywords

Fault diagnosis; Deep convolutional neural network; Alarm management; Tennessee Eastman process

## 1. Introduction

With the wide application of distributed control systems (DCS) during the past three decades, chemical processes have become more and more automated. However, there have been many tragic chemical process accidents, which have resulted in fatalities as well as asset, environmental damages. It seems that we have not tackled the process safety issues head on. One critical protection layer is still missing. This layer currently is relying upon operators who need to be timely aware of abnormal situations and make corrective decisions. However, it requires safety intelligence of operators. Different operators may have different levels of safety intelligence. For an operator who is lack of enough safety intelligence, it is almost impossible for him/her to play this critical role. Therefore, there have been an industrial need to develop an intelligent fault detection and diagnosis (FDD) system to assist operators in handling abnormal situations.

Various FDD methods have been proposed in literature so far. Generally, the FDD methods can be classified into three categories: knowledge based, model based and data based methods. The process model-based methods can be further classified into qualitative model based and quantitative model based methods (Eslamloueyan et al., 2003; Venkatasubramanian et al., 2003a, 2003b, 2003c). Due to the insurmountable drawbacks of knowledge based and model based methods, data based methods usually stand out among the three categories, especially in the era of internet of things.

Data based FDD methods can be classified into statistical methods, shallow learning methods and deep learning methods. Statistical methods include principle component analysis (PCA) (Wise et al., 1990; Russell et al., 2000; Cho et al., 2005; Rato et al., 2016), independent component analysis (ICA) (Kano et al., 2003; Lee et al., 2007; Hsu et al., 2010; Ge et al. 2012; Fan and Wang, 2014), partial least squares (PLS) (MacGregor et al., 1994; Plovoso and Kosanovich, 1994; Zhang and Hu, 2011), fisher discriminant analysis (FDA) (Chiang et al., 2000; Zhu and Song, 2011), qualitative trend analysis (QTA) (Maurya et al., 2005, 2007, 2010) and their derivative methods. Based on the industrial benchmark of Tennessee Eastman (TE) process, a comparison study on these statistical methods was conducted for process monitoring and fault diagnosis (Yin et al., 2012).

Shallow learning methods include support vector machine (SVM) (Chiang et al., 2004; Kulkarni et al., 2005; Zhang, 2008; Mahadevan and Shah, 2009; Yélamos et al., 2009;), artificial immune system (AIS) (Dai and Zhao, 2011; Ghosh and Srinivasan, 2011; Shu and Zhao, 2016), k-nearest neighbor (KNN) (He and Wang, 2007), Gaussian mixture model (GMM) (Choi et al., 2004; Yu and Qin, 2008) and artificial neural network (ANN) (Venkatasubramanian and Chan, 1989; Watanabe et al., 1989; Fan et al., 1993). Shallow learning methods have been successfully utilized to treat fault diagnosis as a classification problem. Furthermore, some neural networks derived from ANN were developed for FDD, such as hierarchical ANN (HANN) (Watanabe et al., 1994; Eslamloueyan et al., 2003), Duty-Oriented HANN (DOHANN) (Eslamloueyan, 2011) and supervised local multilayer perceptron (SLMLP) (Ayubi and Yazdanpanah, 2015).

Despite of the advantages of the above two categories of data driven methods, FDD is still far from widely practical applications due to two major obstacles. One is that these two categories of FDD methods often require considerable amount of domain expertise to determine the fault features in both spatial and temporal domains. The other one is that the fault diagnosis rate is still not high enough.

Over the past few years, deep learning has become an outstanding technology and has shown better performance than the aforementioned methods in many fields. A deep learning architecture is a multilayer stack of several simple but non-linear layers, such as restricted boltzmann machines (RBM), convolutional and pooling layers. The difference between “shallow” and “deep” is that, deep learning emphasizes the importance of feature extraction layer by layer. Each layer can transform the input representation at a low level into a representation at a higher and more abstract level (LeCun et al., 2015). With the composition of enough transformations, complex functions can be learned and the output representation of the last layer is easier for pattern recognition tasks. With the rapid development of deep learning, some deep learning based methods have been proposed for chemical process fault diagnosis. A hierarchical deep neural network (HDNN) was proposed for diagnosing the faults on the TE process (Xie and Li, 2015). The average correct classification rate reached 80.5% (except fault 03, 09, 15), which was higher than DOHANN. Lv et al. utilized stacked sparse auto encoder neural networks and a softmax classifier for FDD (Lv et al., 2016). In the latest study, an extensible deep belief network (DBN) based fault diagnosis model was proposed by the research group of the corresponding author (Zhang and Zhao, 2017). Features of fault data in spatial and temporal domains were extracted by DBN sub-networks, then a global back-propagation network was used for fault classification. Even though the average fault diagnosis rate for all of the 20 fault types in the TE process reached 82.1%, an all-time high record, it is still far from real applications. Therefore, further researches are still needed.

The above three deep learning based methods are all based on DBN, which is a stack of RBMs. DBN was first proposed in 2006 (Hinton and Salakhutdinov, 2016a; Hinton et al., 2016b), which is widely regarded as the start of deep learning. However, in the famous ImageNet Large Scale Visual Recognition Challenge (ILSVRC) competition in 2012, AlexNet (Krizhevsky et al., 2012), the first model that used a deep convolutional neural network (DCNN) won the champion. Since then, DCNN has brought about a revolution in computer vision and pattern recognition. Now DCNN is the dominant approach for almost all recognition and detection tasks (LeCun et al., 2015). However, DCNN model based fault diagnosis method for chemical processes has been paid much less attention.

The basic architectures of typical convolutional neural networks mainly contain two types of layers for feature extraction: convolutional layers and pooling layers. In addition, other layers such as dropout and fully connected (FC) layers, are also significant for DCNN. Compared with DBN, DCNN has three advantages: the first one is that convolutional and pooling layers are local-connected with filters, which can help extract local patterns or features better; the second one is that DCNN with similarly sized layers has fewer parameters and requires less computation time; the third one is that “overfitting” can be avoided by using dropout and pooling layers.

This paper presents a DCNN model based fault diagnosis method for complex chemical processes. The rest of the paper is organized as follows: section 2 introduces the basic theory of DCNN, including convolutional layers, pooling layers, dropout and FC layers. Section 3 presents the DCNN based chemical process fault diagnosis method in detail. Section 4 shows the experiment result of the fault diagnosis on the TE process. Finally, section 5 summarizes this paper.

## 2. Deep convolutional neural network

Convolutional neural network (CNN) was first proposed in the late 1980s, to process data that comes in the form of multiple arrays (LeCun et al., 1989). Since then, it has made great success in object detection and recognition in the computer-vision domain. Over the past few years, several outstanding DCNN architectures have been reported, including AlexNet (Krizhevsky et al., 2012), Network in Network (Lin et al., 2013), VGG (Simonyan and Zisserman, 2014), GoogLeNet (Szegedy et al.,2014), ResNet (He et al., 2016), etc. The general function of DCNN includes feature extraction and classification. For feature extraction, convolutional layers and pooling layers are stacked to transform the raw data into a representation at a higher level. Then fully connected layers are utilized to classify the transformed representation into certain class. Labelled data are required at the training phase which is based on the backpropagation procedure. At the beginning of the training phase, data samples are divided into mini-batches by providing a parameter named “batch size”. The basic method for training neural networks is stochastic gradient descent (SGD). The idea of SGD is to update the weights and biases of neurons after each batch computation, the aim of which is to minimize the training error. The training of the multilayer architecture is a supervised learning process.

## 2.1. Convolutional layer

In a convolutional layer, the representation of the output is composed of feature maps, within which each unit is connected to a local patch in the input feature map through a filter composed of a set of weights. All units in an output feature map share the same filter (shared weights). Different feature maps in a layer use different filters. A typical convolutional layer is shown in Fig. 1. There are two reasons for using convolutional layers. First, in array data such as process data, local groups of values are often highly correlated, and local patterns can be formed to make pattern detection or recognition easier. Second, the distinctive local patterns can appear anywhere in the input feature maps, hence units at different locations sharing the same weights can help detect the same patterns regardless of their locations (LeCun et al., 2015).

![](images/f99a91d2ebe4cea4d7c3d7cbf038e8c1ef906685196c98f7a34a404ae93d9484.jpg)  
Fig. 1. Convolutional layer.

In a convolutional layer, assume that there are ?? feature maps as the input and ?? filters. Generally, we can use Eq. (1) to calculate the output feature maps of the ??th layer (Bouvrie, 2006):

$$
x _ {j} ^ {l} = f \big (\sum_ {i = 1, \dots , M} x _ {i} ^ {l - 1} * k _ {i j} ^ {l} + b _ {j} ^ {l} \big), j = 1, \dots , N,\tag{1}
$$

where $k _ { i j } ^ { l }$ represents the kernel of the ??th filter connected to the ??th input map,

$x _ { i } ^ { l - 1 }$ represents the ?? th input map and $x _ { j } ^ { l }$ represents the ?? th output map, $b _ { j } ^ { l }$ represents the bias corresponding to ??th filter, ?? represents the activation function, and ∗ represents the convolutional operation. In this way, we can obtain ?? feature maps as the output. Assuming that the kernel size is ?? × ??, we can use Eq. (2) to compute the number of all the parameters of a convolutional layer:

$$
P = N \times (s \times s \times M + 1)\tag{2}
$$

![](images/b5d73a0bb55534668daf57457fd20ac75a545853ad2db338ef2af593cf23ba24.jpg)  
Fig. 2. Convolutional operation.

The convolutional operation is shown in Fig. 2, where the size of the input map is 4×4 ,the kernel size is 2×2, and the stride is 1. After the convolutional operation and the addition of the corresponding bias, an activation function is applied for computing the output feature maps. The common activation functions for neural networks include logistic function, hyperbolic tangent function and rectified linear unit (ReLU) function as shown in Eq. (3) \~ (5):

$$
f (x) = (1 + e ^ {- x}) ^ {- 1}\tag{3}
$$

$$
f (x) = \tanh (x)\tag{4}
$$

$$
f (x) = \max (0, x)\tag{5}
$$

However, it has been proved that DCNN with ReLUs can be trained several times faster than their equivalents with the other functions (Krizhevsky et al., 2012). Since then, ReLU has become the first choice for designing a DCNN architecture.

## 2.2. Pooling layer

A pooling layer is also known as a sub-sampling layer which follows a convolutional layer and produces down-sampled versions of the input feature maps. The objective of a pooling layer is to merge similar local features into one. There are three advantages for the use of pooling layers. Firstly, because the relative positions of the features forming a local pattern may vary slightly, it is more reliable for detection to merge similar features in local positions (LeCun et al., 2015). Secondly, pooling layers usually do not have parameters and can reduce the dimension of the feature representation. The use of pooling layers can also greatly reduce the computation time and the parameters of the whole network. Thirdly, pooling layers are beneficial for preventing the “overfitting” (see section 2.3).

![](images/d4334dd932b006bf7de11bd829f01e7bbd64fd273e674194ef111f97f3062354.jpg)

There are two types of pooling operations: max pooling and average pooling. A max pooling unit computes the maximum of a local patch of units in a feature map, and an average pooling unit computes the average. The computation procedure in a pooling layer is similar with that in a convolutional layer. In a pooling layer, assuming that there are ?? feature maps as the input, there must be ?? output feature maps. Generally, we can use Eq. (6) to calculate the output feature maps of the ??th layer (Bouvrie, 2006):

$$
x _ {j} ^ {l} = f \big (\beta_ {j} ^ {l} d o w n \big (x _ {j} ^ {l - 1} \big) + b _ {j} ^ {l} \big), j = 1, \dots , M,\tag{6}
$$

where $x _ { j } ^ { l - 1 }$ represents the ??th input map and $x _ { j } ^ { l }$ represents the ??th output map, $b _ { j } ^ { l }$ and $\beta _ { j } ^ { l }$ represent the additive bias and the multiplicative bias corresponding to the ??th filter respectively, ?? represents the activation function, and ???????? represents the sub-sampling function. Generally, $\beta _ { j } ^ { l } , b _ { j } ^ { l }$ can be negligible and there is no parameter in pooling layers. In this way, we can obtain M feature maps as the output. The max pooling and average pooling operations are shown in Fig. 3, where the size of the input map is 4×4 ,the kernel size is 2×2, and the stride is 2.

<table><tr><td>2</td><td>2</td><td>2</td><td>3</td></tr><tr><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>1</td><td>3</td><td>1</td><td>4</td></tr><tr><td>4</td><td>2</td><td>1</td><td>2</td></tr></table>

Average pooling

<table><tr><td>7/4</td><td>7/4</td></tr><tr><td>5/2</td><td>2</td></tr></table>

Fig. 3. Max pooling and average pooling operations.

## 2.3. Dropout

![](images/b64f2cf04ebe1957ffd6374ea34beacd64bcbc4c0c4e4e4c6fad499f9953ee27.jpg)  
(a) without dropout.

![](images/a63b4e818df3d396cad3a358351ef2758583d1ab16f44e31f8e045525ff67181.jpg)  
(b) with dropout.

Fig. 4. The training error and the testing error.

If a large neural network is trained on a small training dataset, it may have poor performance on the testing dataset. This phenomenon is called “overfitting”. Generally, at the whole training phase, the training error will decrease with the number of iterations increasing. The testing error will decrease at the beginning of the training phase, however, will increase later. Fig. 4(a) shows the phenomenon of “overfitting”. By using the dropout mechanism, the trend of the testing error will be similar to the trend of the training error (see Fig. 4(b)). It is very helpful for overcoming the “overfitting” problem.

![](images/709eddfc76aaaf27d14f0f5f268bcf2815790bfc7c8c90c80065c59a92b08078.jpg)  
(b) without dropout.

![](images/c1f44584acdf5d3ba1be1816796f249a4044ddae1476602103f37d82fc18355e.jpg)  
(b) with dropout.  
Fig. 5. Dropout neural net model.

Dropout avoids “overfitting” by randomly omitting some feature detectors in each iteration of the training stage (Hinton et al., 2012; Srivastava et al., 2014). Generally, in the dropout, we need to set the retaining probability ?? to a certain value (for example, $p = 0 . 5 )$ (see Fig. 5). It will set the output of each neuron to zero with the probability of 0.5 (Krizhevsky et al., 2012). This method can destroy complex co-adaptations of hidden neurons and prevent that a neuron is only useful in the context of several other specific neurons. The omitted half of the neurons will not be able to make contribution for the forward computation and the back propagation.

## 2.4. Fully connected layer

Convolutional layers and pooling layers constitute the feature extractor of the whole DCNN. Following the feature extractor, the objective of FC layers is to classify the features extracted from the raw data. FC layers are essentially backpropagation neural networks and their input must be one-dimensional vectors. Assume that the lengths of the input and output vectors are ?? and ?? respectively. Each value of the input vector is connected to each value of the output vector through one neuron (see Fig. 6). Then we can use Eq. (7) to calculate the output vector of the ??th layer (Bouvrie, 2006):

$$
x _ {j} ^ {l} = f \big (\sum_ {i = 1, \dots , M} x _ {i} ^ {l - 1} \times w _ {i j} ^ {l} + b _ {j} ^ {l} \big), j = 1, \dots , N,\tag{7}
$$

where $w _ { i j } ^ { l }$ represents the weight of the ??th output value connected to the ??th input value, $x _ { i } ^ { l - 1 }$ represents the ??th input value and $x _ { j } ^ { l }$ represents the ??th output value, $b _ { j } ^ { l }$ represents the bias corresponding to the ?? th output value, and ?? represents the activation function. The computation for the number of all the parameters of a FC layer is as follows:

$$
P = M \times N + N\tag{8}
$$

A typical DCNN architecture contains several FC layers in the end. Due to the design of full connectivity, the parameters of FC layers are redundant, compared with other types of layers. The parameters of FC layers constitute more than 80% of parameters of the whole DCNN.

![](images/8ae584bb17fc2b9525167455a3b1bed4e00dda3d522d15c62bd22a6cb6b40cd7.jpg)  
Fig. 6. Fully connected layer.

## 3. DCNN based fault diagnosis method

![](images/5da02963bea6dc19becfcb6b55bb478ee043539ad993f5a8865bafbc493787df.jpg)

Fig. 7. The framework of the DCNN based fault diagnosis method.

Faults in chemical processes are essentially states that process variables deviate from their normal states. Data from different state deviations can be used to diagnose the fault types. Most of the published FDD methods only studied the features of fault data in the spatial domain, while the time-varying features in the temporal domain are relatively less studied. It has been proved that the time-varying features of fault data are also critical for distinguishing fault types (Maurya et al., 2005, 2007, 2010; Zhang and Zhao, 2017), which is also true even for human experts in performing various diagnosis tasks. Although DCNN is mainly used for images in computer vision, it can be used for spectrograms in speech recognition as well. DCNN extracts features from local patches, therefore, a simple data preprocessing is done in the proposed method to set the time-series of process variables of each equipment next to each other. In this way, there exists spatial relationship along the variable dimension. Similar with spectrograms including time and frequency domains, the data preprocessing also transforms the process data into two-dimensional matrices (temporal and spatial domains) with the size of $m \times n$ , where ?? represents the length of a certain period of time (sample time length) and ?? represents the number of variables. For example, with $n = 5 0 , m = 2 0$ (sampling period is set to 3 min), this means that the data of 50 variables from time t-1h to time t is used as a sample matrix for diagnosing the status of time t.

The framework of the DCNN based fault diagnosis method is shown in Fig. 7. Its diagnosis procedures including offline and online stages are described as follows:

## Offline stage:

Step 1: Historical data is collected and preprocessed from the chemical process.

Step 2: Through the data preprocessing, historical data is transformed into sample matrices with the size of ?? × ?? and labelled with their corresponding classes, including “normal” and their fault types (see Fig. 8(a)).

Step 3: The sample matrices including their corresponding labels are divided into the training set and the testing set.

![](images/ce9c9048ad5b244891473fa2d5dd6f11fb2a2b030152a14e09fc469e1fa02e00.jpg)

Step 4: The DCNN model is designed for the chemical process (see Fig. 8(b)).

Step 5: The DCNN model is trained.

Step 6: The DCNN model is tested.

Step 7: The fault diagnosis result is outputted and visualized (see Fig. 8(c)).

Step 8: If the fault diagnosis rate in testing is satisfactory, the model will be used for online fault diagnosis; if unsatisfactory, the DCNN model needs to be redesigned (Step 4).

## Online stage:

Step 1: Online data is collected and preprocessed from the chemical process.

Step 2: Through the same preprocessing, online data is transformed into sample matrices with the size of $m \times n$

Step 3: Online sample matrices are input to the DCNN model. The model can give a predicted diagnosis result for each sample matrix. The diagnosis result is either “normal” or one specific fault type.

Step 4: If there is a discrepancy between the predicted diagnosis result and the judgement of the human experts, the DCNN model needs to be retrained with the new data.

![](images/ffa1f419e0d5e093eac7dd8ef7f96b8e8db7ca26084afb898cd6ea6948fd9d63.jpg)  
(a) Process data collection and preprocessing.

![](images/739dea038cba8a97160d180704407272ed9df0c09586b66835519fcc77577054.jpg)  
(b) DCNN model construction.

![](images/960c288c8c1c0ed3d2ac5a270c40ffe54259954d53291a311a02ff59d4f153f8.jpg)  
(c) Fault diagnosis visualization using t-SNE.

Fig. 8. The DCNN model for fault diagnosis.

## 4. Experiment result

## 4.1. Tennessee Eastman process

![](images/e544951129ceeed2cf9b25e8b6eb1f71d108ec8c4d6d331b3e9e724725c187eb.jpg)  
Fig. 9. P&ID of TE process (Bathelt et al., 2015).

The benchmark Tennessee Eastman (TE) process (Downs and Vogel, 1993) is used for showing the advantages of the proposed DCNN model in this paper. The simulator is based on the revised version (see Fig. 9) which is available at http://depts.washington.edu/control/LARRY/TE/download.html (Bathelt et al., 2015). The process variables include 12 process manipulated variables, 22 continuous process measurements and 19 component analysis measurements. Even though there are 28 process disturbances (fault types) in the revised version, we only select IDV(1) – IDV(20) for comparison with other algorithms. Normal data and fault data of the TE process are collected from the simulations on MATLAB 2016a (see Table 1 and Table 2). The method of the simulations refers to Zhang’s research (Zhang and Zhao, 2017). The sampling period is set to 3 min (20 samples/h). The simulator runs for 500 h in the normal state. 10000 normal sample matrices are then collected and each contains one-hour data. In each simulation of the 20 faults, the simulator runs for 10 h in the normal state at the beginning. Then the corresponding fault disturbance is introduced and the simulator continue to run for 40 h. In this way, 40 h fault data (800 fault sample matrices) are collected in each simulation. It must be noted that the simulations of fault 06 shut down after 7 h in the fault state, hence each simulation of fault 06 only has 7 h fault data. The simulation of each fault type repeats 10 times with 10 different initial states. We randomly select 80% normal sample matrices (8000) as training normal sample matrices and the remaining normal sample matrices (2000) as testing normal sample matrices. In the same way, for each fault, we randomly select 122720 training fault sample matrices from eight simulations and 30680 testing fault sample matrices from the other two simulations.

In order to extract the features of process data in both spatial and temporal domains, data samples are transformed into two-dimensional matrices with ?? × ??. In the TE process, XMV(5) (Compressor recycle valve), XMV(9) (Stripper steam valve) and XMV(12) (Agitator speed) are constant during the simulations. These three variables are excluded and therefore each sample matrix includes the information of the rest 50 variables for 1 h (?? = 20, ?? = 50).

Table 1 The fault sample matrices collected from the TE process simulator.

<table><tr><td>Status index</td><td>Training Time length/h</td><td>Number of training sample matrices</td><td>Testing Time length/h</td><td>Number of testing sample matrices</td></tr><tr><td>IDV 01-05</td><td rowspan="2">40×19×8</td><td rowspan="2">800×19×8</td><td rowspan="2">40×19×2</td><td rowspan="2">800×19×2</td></tr><tr><td>07-20</td></tr><tr><td>IDV 06</td><td>7×8</td><td>140×8</td><td>7×2</td><td>140×2</td></tr><tr><td>Fault</td><td>6136</td><td>122720</td><td>1534</td><td>30680</td></tr></table>

Table 2 The normal sample matrices collected from the TE process simulator.

<table><tr><td>Status index</td><td>Training Time length/h</td><td>Number of training sample matrices</td><td>Testing Time length/h</td><td>Number of testing sample matrices</td></tr><tr><td>Normal</td><td>400</td><td>8000</td><td>100</td><td>2000</td></tr></table>

## 4.2. DCNN model for the TE process

It is a common issue that there is no scientific guidance for designing an optimal DCNN architecture. In order to find a proper model, we tried several DCNN models, the architectures of which are shown in Table 3. Then an architecture with the most outstanding fault diagnosis performance was selected from Table 3. Parameters of the selected architecture were studied through experiments. The parameters mainly contain the number of filters of each convolutional layer and the output length of the first FC layer.

Table 3 DCNN model candidates for fault diagnosis of the TE process.

<table><tr><td>Model</td><td>Architecture</td></tr><tr><td>Model 1</td><td>Conv(128)-Pool-FC(300)*-FC (21)</td></tr><tr><td>Model 2</td><td>Conv(128)-Conv(128)-Pool-FC(300)*-FC(21)</td></tr><tr><td>Model 3</td><td>Conv(128)-Conv(128)-Conv(128)-Pool-FC(300)*-FC(21)</td></tr><tr><td>Model 4</td><td>Conv(128)-Conv(128)-Conv(128)-Conv(128)-Pool-FC(300)*-FC(21)</td></tr><tr><td>Model 5</td><td>Conv(64)-Pool-Conv(128)-Pool-FC(300)*-FC(21)</td></tr><tr><td>Model 6</td><td>Conv(64)-Conv(64)-Pool-Conv(128)-Pool-FC(300)*-FC(21)</td></tr><tr><td>Model 7</td><td>Conv(64)-Conv(64)-Pool-Conv(128)-Pool(2×1)-FC(300)*-FC(21)</td></tr><tr><td>Model 8</td><td>Conv(64)-Conv(64)-Pool-Conv(128)-Pool(1×2)-FC(300)*-FC(21)</td></tr><tr><td>Model 9</td><td>Conv(64)-Pool-Conv(128)-Conv(128)-Pool-FC(300)*-FC(21)</td></tr><tr><td>Model 10</td><td>Conv(64)-Conv(64)-Pool-Conv(128)-Conv(128)-Pool-FC(300)*-FC(21)</td></tr><tr><td>Model 11</td><td>Conv(64)-Conv(64)-Conv(64)-Pool- -Conv(128)-Conv(128)-Pool(2×1)-FC(300)*-FC(21)</td></tr><tr><td>Model 12</td><td>Conv(64)-Conv(64)-Pool- -Conv(128)-Conv(128)-Conv(128)-Pool(2×1)-FC(300)*-FC(21)</td></tr></table>

FC\*: Dropout (?? = 0.5) is utilized for this FC layer.  
Pool: The default of kernel size in pooling layers is 2×2 and all the strides are set to 2.

In the following, Model 7 is explained as an example. The input size of one sample matrix is 20×50, where “20” represents sample time length and “50” represents the number of process variables. Here we use 3 convolutional layers, 2 max pooling layers and 2 FC layers. In the three convolutional layers, the kernel sizes are all set to 3×3 and the strides are set to 1. The first and second convolutional layers both contain 64 filters and the third contains 128 filters. The two max pooling layers are behind the second and third convolutional layers. The kernel size is set to 2×2 in the first pooling layer and set to 2×1 in the second pooling layer. Here the output of one sample matrix is a three-dimensional array (3×21×128). As mentioned in section 2.4, the input of FC layers must be a one-dimensional vector, hence we utilize a “Flatten” layer to reshape three-dimensional arrays into one-dimensional vectors with the size of 8064 (3×21×128). The output length of the first FC layer is set to 300 and “Dropout” is used for this layer. The last FC layer outputs the classes of sample matrices with a “softmax” function. The softmax function also named normalized exponential function, is a generalization of the logistic function that transforms a K-dimensional vector Z of arbitrary real values into a K-dimensional vector σ(Z) of real values in the range of (0, 1) that add up to 1. The function is shown as:

$$
\sigma (\mathrm{Z}) _ {j} = \frac {e ^ {Z _ {j}}}{\sum_ {k = 1} ^ {K} e ^ {Z _ {k}}}, j = 1, 2, \ldots , K,\tag{9}
$$

After the transformation of the softmax function, the model outputs a vector of length 21, each value of which represents the possibility of the corresponding class. For testing a sample matrix, the predicted diagnosis result is the class with the highest possibility.

## 4.3. Fault diagnosis result

## 4.3.1 FDR and FPR

After the data collection and the DCNN model construction are completed, the training and testing procedures of classification are implemented in a server computer. In order to show the experiment result of the fault diagnosis, define the confusion matrix, fault diagnosis rate (FDR) and false positive rate (FPR) for the ??th class in Table 4, Eq. (10) and (11), respectively:

Table 4 Confusion matrix for the ??th class.

<table><tr><td></td><td>Number of samples in the ith class (Predicted)</td><td>Number of samples in the other classes (Predicted)</td></tr><tr><td>Number of samples in the ith class (Actual)</td><td>p</td><td>b</td></tr><tr><td>Number of samples in the other classes (Actual)</td><td>q</td><td>d</td></tr><tr><td></td><td>FDR =  $\frac{p}{p+b}$ </td><td>(10)</td></tr><tr><td></td><td>FPR =  $\frac{q}{q+d}$ </td><td>(11)</td></tr></table>

Here the batch size for training is set to 128 and the number of epochs is set to 50. “Batch size” means the number of sample matrices in one forward/backward pass of each iteration. “One Epoch” means all the training sample matrices are passed forward and backward through the network only once. The fault diagnosis testing is on the basis of one sample matrix each time. Each sample matrix contains the time-series data of the 50 variables from time t-1h to time t to diagnose the status of time t. The testing average FDR and the training/testing time of the models in Table 3 are listed in Table 5. Model 3 has the highest testing average FDR (88.4%) and takes 56s×50=46.7min for training. With a little decrease of the testing average FDR (88.2%), Model 7 only takes about half the training time of Model 3. In the following discussion, Model 7 is chosen as the best architecture. Then through a lot of experiments, the parameters of Model 7 are set to {conv(64), conv(64), conv(128), FC(300)}.

Table 5 The testing average FDR and the training and testing time.

<table><tr><td>Model</td><td>Testing average FDR (%)</td><td>Training time for one epoch (s)</td><td>Testing time for one sample matrix (ms)</td></tr><tr><td>Model 1</td><td>85.0</td><td>60</td><td>1.9</td></tr><tr><td>Model 2</td><td>87.9</td><td>57</td><td>1.8</td></tr><tr><td>Model 3</td><td>88.4</td><td>56</td><td>1.8</td></tr><tr><td>Model 4</td><td>88.0</td><td>56</td><td>1.8</td></tr><tr><td>Model 5</td><td>86.4</td><td>24</td><td>1.5</td></tr><tr><td>Model 6</td><td>87.5</td><td>24</td><td>1.5</td></tr><tr><td>Model 7</td><td>88.2</td><td>30</td><td>1.5</td></tr><tr><td>Model 8</td><td>87.5</td><td>31</td><td>1.5</td></tr><tr><td>Model 9</td><td>86.5</td><td>24</td><td>1.5</td></tr><tr><td>Model 10</td><td>87.5</td><td>24</td><td>1.5</td></tr><tr><td>Model 11</td><td>87.5</td><td>31</td><td>1.5</td></tr><tr><td>Model 12</td><td>87.8</td><td>27</td><td>1.5</td></tr></table>

Fig. 10 illustrates the curves of the accuracy at the training and testing phases. The DCNN based fault diagnosis results are listed in Table 6. Among the 122720 training sample matrices, the average FDR of training dataset is 98.6% and the corresponding average FPR is 0.1%. As for the testing dataset including 30680 sample matrices, the average FDR reaches 88.2% and the corresponding average FPR is 0.5%. These results show that except “Fault 09” (D feed temperature in stream 2 – random variation), “Fault $1 5 ^ { \circ }$ (Condenser cooling water valve - sticking) and “Fault $1 6 ^ { \circ }$ (Unknown), the other fault types can be diagnosed with more than 91% FDRs by the DCNN model. The details of the testing result for fault diagnosis are illustrated in Fig. 11. It shows the confusion matrix of all the 21 classes. “Size” and “Color” both represent the value of the confusion matrix.

![](images/b7c014451a5b0a97cead48eaa762178a7b1bae864b3fd115027dfe6c8cee7aeb.jpg)  
Fig. 10. The training accuracy and testing accuracy for the iteration process.

![](images/4527c45d0aff7ddbff8fa0a7b8c313f25d2fe25510e9702484e128789a89bb1b.jpg)  
Fig. 11. The details of the testing result for fault diagnosis.

Table 6 The comparison of fault diagnosis results between the DBN based model (Zhang and Zhao, 2017) and the DCNN based model.

<table><tr><td>Status index</td><td>FDR-Train (DBN)</td><td>FDR-Train (DCNN)</td><td>FPR-Train (DCNN)</td><td>FDR-Test (DBN)</td><td>FDR-Test (DCNN)</td><td>FPR-Test (DCNN)</td></tr><tr><td>Normal</td><td>-</td><td>0.916</td><td>0.001</td><td>-</td><td>0.978</td><td>0.015</td></tr><tr><td>Fault 01</td><td>1.00</td><td>0.998</td><td>0.000</td><td>1.00</td><td>0.986</td><td>0.003</td></tr><tr><td>Fault 02</td><td>1.00</td><td>0.996</td><td>0.000</td><td>0.99</td><td>0.985</td><td>0.000</td></tr><tr><td>Fault 03</td><td>0.99</td><td>0.996</td><td>0.000</td><td>0.95</td><td>0.917</td><td>0.008</td></tr><tr><td>Fault 04</td><td>0.98</td><td>0.999</td><td>0.000</td><td>0.98</td><td>0.976</td><td>0.000</td></tr><tr><td>Fault 05</td><td>0.90</td><td>0.998</td><td>0.000</td><td>0.86</td><td>0.915</td><td>0.006</td></tr><tr><td>Fault 06</td><td>1.00</td><td>0.998</td><td>0.000</td><td>1.00</td><td>0.975</td><td>0.000</td></tr><tr><td>Fault 07</td><td>1.00</td><td>0.999</td><td>0.000</td><td>1.00</td><td>0.999</td><td>0.000</td></tr><tr><td>Fault 08</td><td>0.96</td><td>0.985</td><td>0.001</td><td>0.78</td><td>0.922</td><td>0.001</td></tr><tr><td>Fault 09</td><td>0.655</td><td>0.973</td><td>0.000</td><td>0.57</td><td>0.584</td><td>0.020</td></tr><tr><td>Fault 10</td><td>0.975</td><td>0.977</td><td>0.000</td><td>0.98</td><td>0.964</td><td>0.000</td></tr><tr><td>Fault 11</td><td>0.975</td><td>0.995</td><td>0.000</td><td>0.87</td><td>0.984</td><td>0.001</td></tr><tr><td>Fault 12</td><td>0.855</td><td>0.992</td><td>0.000</td><td>0.85</td><td>0.956</td><td>0.001</td></tr><tr><td>Fault 13</td><td>0.965</td><td>0.978</td><td>0.001</td><td>0.88</td><td>0.957</td><td>0.000</td></tr><tr><td>Fault 14</td><td>0.96</td><td>0.998</td><td>0.000</td><td>0.87</td><td>0.987</td><td>0.000</td></tr><tr><td>Fault 15</td><td>0</td><td>0.997</td><td>0.001</td><td>0</td><td>0.28</td><td>0.028</td></tr><tr><td>Fault 16</td><td>0</td><td>0.912</td><td>0.009</td><td>0</td><td>0.442</td><td>0.038</td></tr><tr><td>Fault 17</td><td>1.00</td><td>0.988</td><td>0.004</td><td>1.00</td><td>0.945</td><td>0.000</td></tr><tr><td>Fault 18</td><td>1.00</td><td>0.970</td><td>0.002</td><td>0.98</td><td>0.939</td><td>0.001</td></tr><tr><td>Fault 19</td><td>0.97</td><td>0.996</td><td>0.000</td><td>0.93</td><td>0.986</td><td>0.000</td></tr><tr><td>Fault 20</td><td>0.987</td><td>0.971</td><td>0.001</td><td>0.93</td><td>0.933</td><td>0.000</td></tr><tr><td>Average*</td><td>0.859</td><td>0.986</td><td>0.001</td><td>0.821</td><td>0.882</td><td>0.005</td></tr></table>

\*The average excludes the FDR of normal status because it is not available in the research of DBN based model.

In the previous researches about the TE process fault diagnosis, the best reported results were achieved by the DBN based model (Zhang and Zhao, 2017). In order to show the performance of the proposed method, the results of the DBN based model and the DCNN based model are compared in Table 6. For the training dataset, the DBN based model cannot diagnose the three fault “Fault 09”, “Fault 15” and “Fault $1 6 ^ { \circ }$ and the FDRs of “Fault 15” and “Fault $1 6 ^ { \circ }$ are even 0%. By contrast, the DCNN based model shows excellent performance on all the fault types in the training dataset, with 98.6% average FDR. For the testing dataset, the FDRs of “Fault $0 5 ^ { \prime \prime } { , } ^ { \mathrm { ~ \tiny ~ \mathfrak ~ { ~ F ~ a u l t ~ } 0 8 ^ { \circ } ~ } }$ “Fault 11”, “Fault 12”, “Fault 13”, “Fault 14” are less than 88% in the DBN based model. By contrast, the proposed DCNN model achieves more than 91% FDRs of the six fault types. Additionally, the DCNN based model can partially diagnose “Fault 15” and “Fault 16”, despite that their FDRs are still low. Overall, the average of testing FDRs through the DCNN based model is 88.2%, which is 6.1% higher than that of the DBN based model.

## 4.3.2 Hierarchical feature learning visualization

To facilitate the understanding of the feature learning process of the DCNN model, it is vital to learn about its hierarchical feature learning process. However, it is difficult for the output representation of each layer to visualize the diagnosis because the learned features are high-dimensional. To solve this problem, we adopt the t-distributed stochastic neighbor embedding (t-SNE) (Maaten and Hinton, 2008) method, which is usually used to visualize the hierarchical feature learning process of the DCNN model.

The t-SNE method is a variation of Stochastic Neighbor Embedding (SNE) (Hinton and Roweis, 2002), and is better at revealing the distribution of high-dimensional data. The t-SNE can embed high-dimensional features of each layer into a space of two or three dimensions, which can be visualized in a scatter map. With the 2D or 3D map corresponding to each layer, we can visualize the feature learning process easily. Through the experiment, we find that 3D maps are not suitable for visualization of the DCNN based fault diagnosis model. Therefore, the high-dimensional output features of each layer are embedded into 2D maps, which are then plotted in the subfigures in Fig. 12.

800 sample matrices of the 21 classes (one normal class and 20 fault classes) are randomly selected from the testing set for visualization. The size of the input data is 800×20×50. Then the t-SNE method is used to transform these 20×50 sample matrices into 800 vectors of length 2. In Fig. 12(a), each point represents a sample matrix, plotted by the first value of each vector on the horizontal axis and the second value on the vertical axis. The points are marked with their actual class labels, “Normal” labelled with “0”, “Fault 01” labelled with “1”, etc. Additionally, in order to distinguish the clusters for viewing, different colors are used to represent their class labels as well. Similarly, the output of each layer is transformed by the t-SNE method into a vector of length 2 so that it can be visualized in Fig. 12(b)\~(h). It should be noted that the output of the “Dropout” and “Flatten” layers are not visualized because these layers are useless for feature learning.

(a) t-SNE of the input  
![](images/a6bb8829fe48c33055ec94373b7d527c9484289531d79048ecf41683ef094f42.jpg)

(b) t-SNE of the output of Conv-1  
![](images/a775f8ab0ba27a6670f74142d9d6da7f080123d60cae9f2e9862f25f43be3f64.jpg)  
(d) t-SNE of the output of MaxPool-1

(c) t-SNE of the outpt of Conv-2  
![](images/f655bbcaed9293734d55f9da845d2482c10f4029d6df247b8bb49c96ce52313c.jpg)

![](images/c6050e98abb8a2a5d5f87ef204d371e6e3f5cd3c7dd18d5552bfb4689e31b42b.jpg)

(e) t-SNE of the output of Conv-3  
![](images/789063cd2562e2d31c565c58b42f4f3bab973476495828f94256a9be3ad86ed7.jpg)  
(g) t-SNE of the output of FC-1

(f) t-SNE of the output of MaxPool-2  
![](images/581071f65334c7018fa22749fda9abac03c01ec307d80cb47f4911de77e402fc.jpg)  
(h) t-SNE of the output of FC-2

![](images/3d39ec97e02a87989a1803d22d95f0d5c2ebabe3a1b1af92050d4f8d53d1f8c1.jpg)

![](images/7c646219cc948a21c4b7e53c05f9b3a46f493886685c03a78e5ce1c4fc5917f8.jpg)  
Fig. 12. DCNN model visualization using t-SNE.

As illustrated in Fig. 12(a), the raw process data samples of all the classes are mixed up. Through three convolutional layers and two max pooling layers, we can find that the samples are gradually clustered by the class labels in the t-SNE maps (see Fig. 12(b)\~(f)). Theoretically, clearer clusters mean better performance for classification. Finally, Fig. 12(g) and (h) are the t-SNE maps of the last two fully connected layers. The last t-SNE map illustrates the result of classification, and shows rather clear clustering of samples. These subfigures powerfully prove that the DCNN model is effective for the fault diagnosis task. Additionally, in Fig. 12(h), we can also find that the points with labels “9”, “15” and “16” are mixed up in the t-SNE map. It is well known that the FDRs of these three fault types in the previously reported researches were also quite low (Zhang and Zhao, 2017, Rato and Reis, 2017). Since the fluctuation ranges of the root cause variables of these three faults are unknown in the 50 variables of the TE simulator (Bathelt et al., 2015), it is hard to explain why their FDRs are low.

Model 7 is retrained by using the data of normal and the other 17 fault types

(without “Fault $0 9 ^ { \mathfrak { n } }$ , “Fault $1 5 ^ { \circ }$ and “Fault 16”). The result is listed in Table 7.

Table 7 The fault diagnosis result without “Fault 09”, “Fault 15” and “Fault $1 6 ^ { \circ } .$

<table><tr><td>Status index</td><td>FDR(Test)</td><td>FPR(Test)</td><td>Status index</td><td>FDR(Test)</td><td>FPR(Test)</td></tr><tr><td>Normal</td><td>0.996</td><td>0.022</td><td>Fault 10</td><td>0.959</td><td>0.000</td></tr><tr><td>Fault 01</td><td>0.995</td><td>0.002</td><td>Fault 11</td><td>0.984</td><td>0.000</td></tr><tr><td>Fault 02</td><td>0.990</td><td>0.000</td><td>Fault 12</td><td>0.968</td><td>0.002</td></tr><tr><td>Fault 03</td><td>0.961</td><td>0.001</td><td>Fault 13</td><td>0.939</td><td>0.000</td></tr><tr><td>Fault 04</td><td>0.999</td><td>0.001</td><td>Fault 14</td><td>0.992</td><td>0.000</td></tr><tr><td>Fault 05</td><td>0.951</td><td>0.001</td><td>Fault 17</td><td>0.950</td><td>0.000</td></tr><tr><td>Fault 06</td><td>0.989</td><td>0.000</td><td>Fault 18</td><td>0.946</td><td>0.000</td></tr><tr><td>Fault 07</td><td>1.000</td><td>0.000</td><td>Fault 19</td><td>0.985</td><td>0.000</td></tr><tr><td>Fault 08</td><td>0.938</td><td>0.002</td><td>Fault 20</td><td>0.940</td><td>0.000</td></tr><tr><td></td><td></td><td></td><td>Average</td><td>0.970</td><td>0.001</td></tr></table>

## 4.3.3 Model tuning

Table 7 shows that the FDR for normal operation is 99.6%. This means that 0.4% of normal data is misdiagnosed as fault data, which is usually regarded as “false alarm”. If the false alarm rate is large, operators would be bombarded with lots of false alarms during normal operation. In order to minimize the false alarm rate, the output of the DCNN model can be tuned. After the softmax function, the model outputs a vector of length 18, $\{ r _ { i } | i = 0 { \sim } 8 , 1 0 { \sim } 1 4 , 1 7 { \sim } 2 0 \}$ , each value of which represents the possibility of the corresponding class. The diagnosis result (R) is the class with the highest possibility, $R \equiv \mathrm { a r g m a x } _ { i } ( r _ { i } )$ . To reduce the normal data misdiagnosed as fault data, the following procedure is used for diagnosing the class of data:

$$
\begin{array}{l} \text {set n_{0} ;} \\ \text {if r_{0} >n_{0} :} \\ \qquad \text {return R = 0;} \\ \text {else:} \\ \qquad \text {return R = argmax_{i} (r_{i})} \end{array}
$$

Table 8 lists the false alarm rates of the normal state and the average FDRs of all the fault types with several values of $n _ { 0 }$ . With the decrease of $n _ { 0 }$ , the false alarm rate decreases but the average FDR of faults decreases as well. This brings out the trade-off between the needs of false alarm rate and fault diagnosis rate.

Table 8 The result of tuning the model output.

<table><tr><td> $n_0$ </td><td>False alarm rate</td><td>Average FDR</td></tr><tr><td>0.5</td><td>0.004</td><td>0.970</td></tr><tr><td>0.4</td><td>0.003</td><td>0.969</td></tr><tr><td>0.3</td><td>0.002</td><td>0.969</td></tr><tr><td>0.2</td><td>0.002</td><td>0.968</td></tr><tr><td>0.1</td><td>0.002</td><td>0.967</td></tr><tr><td>0.005</td><td>0.001</td><td>0.961</td></tr><tr><td>0.004</td><td>0.0004</td><td>0.961</td></tr></table>

## 4.3.4 Diagnostic performance

The fault development is a dynamic process. Therefore, it is necessary to explore the diagnostic performance of the proposed method as time progresses after the fault introduction. Table 9 lists the average FDR during different time periods after the fault introduction. It can be found that the average FDR increases as time progresses.

Table 9 The average FDR during different time periods after the fault introduction.

<table><tr><td>Time period</td><td>0~1h</td><td>1~2h</td><td>2~3h</td><td>3~4h</td><td>4~5h</td><td>5~6h</td><td>6~7h</td><td>7~8h</td></tr><tr><td>Average FDR</td><td>0.432</td><td>0.766</td><td>0.941</td><td>1.000</td><td>0.985</td><td>0.999</td><td>1.000</td><td>1.000</td></tr></table>

Furthermore, the diagnosis time is explored to show when the fault can be diagnosed correctly after the fault occurrence. In the aforementioned discussion, sampling period $( \mathbf { A } ^ { * } )$ is set to 3 min, fault simulating time $( \mathrm { B ^ { * } } )$ is set to 40 h, and sample time length (C\*) is set to 20 (20×3min=1h). For comparison, the diagnosis time in different simulation conditions is listed in Table 10. If a fault is diagnosed consecutively for 5 (when $\mathbf { A } ^ { * } { = } 3 \mathrm { m i n } )$ evaluations, or 9 (when $\mathbf { A } ^ { * } { = } 1 \mathrm { m i n } )$ evaluations, or 30 (when $A ^ { * } { \boldsymbol { \Xi } } 1 5 { \mathrm { s } } )$ evaluations, a diagnostic result is indicated. It is important to note that before the diagnosis time, one or several faults in the 17 fault types may be misdiagnosed. For example, in the condition of $( \mathbf { A } ^ { * } { = } 3 \mathrm { m i n }$ $B ^ { * } { = } 4 0 \mathrm { h } .$ , C\*=20), 16 fault types can be correctly diagnosed when the first fault is detected. However, “Fault $1 3 ^ { \circ }$ will be misdiagnosed as “Fault $0 2 ^ { \circ }$ at 85min after the fault occurrence. From Table 10, it can be found that as the sampling period decreases (from 3min to 15s), the faults can be correctly diagnosed early.

Table 10 The diagnosis time (min) in different simulation conditions.

<table><tr><td>A*</td><td colspan="4">3min</td><td colspan="4">1min</td><td colspan="2">15s</td></tr><tr><td>B*</td><td colspan="2">40 h</td><td colspan="2">20 h</td><td colspan="2">20 h</td><td colspan="2">10 h</td><td colspan="2">20 h</td></tr><tr><td>C*</td><td>20</td><td>10</td><td>20</td><td>10</td><td>20</td><td>10</td><td>20</td><td>10</td><td>20</td><td>10</td></tr><tr><td>Fault 01</td><td>28</td><td>36</td><td>30</td><td>31</td><td>16</td><td>18</td><td>18</td><td>18</td><td>15</td><td>13</td></tr><tr><td>Fault 02</td><td>39</td><td>50</td><td>60</td><td>47</td><td>17</td><td>19</td><td>17</td><td>15</td><td>17</td><td>18</td></tr><tr><td>Fault 03</td><td>35</td><td>26</td><td>34</td><td>26</td><td>14</td><td>18</td><td>19</td><td>17</td><td>9</td><td>7</td></tr><tr><td>Fault 04</td><td>15</td><td>34</td><td>16</td><td>38</td><td>9</td><td>9</td><td>9</td><td>10</td><td>8</td><td>8</td></tr><tr><td>Fault 05</td><td>33</td><td>12</td><td>28</td><td>13</td><td>10</td><td>9</td><td>10</td><td>9</td><td>26</td><td>22</td></tr><tr><td>Fault 06</td><td>18</td><td>25</td><td>23</td><td>27</td><td>11</td><td>12</td><td>11</td><td>12</td><td>9</td><td>8</td></tr><tr><td>Fault 07</td><td>15</td><td>15</td><td>15</td><td>15</td><td>9</td><td>9</td><td>9</td><td>9</td><td>7</td><td>7</td></tr><tr><td>Fault 08</td><td>94</td><td>89</td><td>97</td><td>92</td><td>44</td><td>38</td><td>13</td><td>44</td><td>81</td><td>71</td></tr><tr><td>Fault 10</td><td>82</td><td>57</td><td>91</td><td>61</td><td>31</td><td>39</td><td>40</td><td>39</td><td>24</td><td>52</td></tr><tr><td>Fault 11</td><td>45</td><td>45</td><td>56</td><td>43</td><td>12</td><td>17</td><td>16</td><td>16</td><td>19</td><td>18</td></tr><tr><td>Fault 12</td><td>52</td><td>33</td><td>66</td><td>44</td><td>13</td><td>18</td><td>27</td><td>20</td><td>13</td><td>9</td></tr><tr><td>Fault 13</td><td>100</td><td>82</td><td>107</td><td>101</td><td>131</td><td>105</td><td>77</td><td>86</td><td>72</td><td>62</td></tr><tr><td>Fault 14</td><td>24</td><td>24</td><td>27</td><td>23</td><td>15</td><td>16</td><td>15</td><td>17</td><td>12</td><td>12</td></tr><tr><td>Fault 17</td><td>124</td><td>123</td><td>123</td><td>115</td><td>27</td><td>34</td><td>35</td><td>42</td><td>43</td><td>41</td></tr><tr><td>Fault 18</td><td>45</td><td>56</td><td>107</td><td>96</td><td>36</td><td>72</td><td>43</td><td>28</td><td>19</td><td>20</td></tr><tr><td>Fault 19</td><td>64</td><td>53</td><td>59</td><td>51</td><td>17</td><td>18</td><td>17</td><td>18</td><td>17</td><td>15</td></tr><tr><td>Fault 20</td><td>146</td><td>163</td><td>134</td><td>154</td><td>79</td><td>60</td><td>66</td><td>29</td><td>44</td><td>62</td></tr><tr><td>Average</td><td>56</td><td>54</td><td>63</td><td>57</td><td>29</td><td>30</td><td>26</td><td>25</td><td>26</td><td>26</td></tr></table>

A\*: sampling period; B\*: fault simulating time; C\*: sample time length.

## 5. Conclusion

In this paper, a DCNN model based chemical process fault diagnosis method is proposed. In order to extract the features in both spatial and temporal domains, a DCNN model is built with convolutional layers, pooling layers, dropout and FC layers. Raw process data is transformed into a ?? × ?? dimensional feature map, where ?? represents the length of a certain period of time (sample time length) and ?? represents the number of variables.

The experiment result shows that the proposed DCNN based fault diagnosis method has excellent performance on the benchmark TE process. The average FDR of all the 20 fault types reaches 88.2%, which is higher than other fault diagnosis methods published in literatures. Except three fault types (“Fault 09”, “Fault 15” and “Fault 16”) which are notoriously difficult to diagnose, the FDRs of the other fault types are all over 91%. Additionally, the t-SNE method is utilized to visualize the hierarchical learning feature process. Most data sample matrices of are clearly and correctly clustered by the DCNN in the t-SNE map. Then the model is tuned to reduce the false alarms. This brings out the trade-off between the needs of false alarm rate and fault diagnosis rate. Finally, the dynamic diagnostic performance and the diagnosis time are explored.

This method is prospective for industrial applications due to its outstanding fault diagnosis rates and false positive rates. However, since the model still relies on historical fault data samples, it is not applicable to diagnosis of faults without historical data or with little historical data. For a super complex chemical process which generally has more than thousands of variables, input data dimension ?? will become a large number while the other dimension ?? will be relatively very small. Under this situation, how to design a DCNN architecture will become a serious problem. The future research work will be focused on designing a DCNN model for super complex chemical processes with thousands of process variables.

## Acknowledge

The authors gratefully acknowledge support from the National Natural Science Foundation of China (No. 61433001).

## Reference

Ayubi Rad, M.A., Yazdanpanah, M.J., 2015. Designing supervised local neural network classifiers based on EM clustering for fault diagnosis of Tennessee Eastman process. Chemom. Intell. Lab. Syst. 146, 149–157. doi:10.1016/j.chemolab.2015.05.013

Bathelt, A., Ricker, N.L., Jelali, M., 2015. Revision of the Tennessee eastman process model, in: IFAC-PapersOnLine. pp. 309–314. doi:10.1016/j.ifacol.2015.08.199

Bouvrie, J., 2006. Notes on convolutional neural networks. In Pract. 47–60. doi:http://dx.doi.org/10.1016/j.protcy.2014.09.007

Chiang, L.H., Russell, E.L., Braatz, R.D., 2000. Fault diagnosis in chemical processes using Fisher discriminant analysis, discriminant partial least squares, and principal component analysis. Chemom. Intell. Lab. Syst. 50, 243–252. doi:10.1016/S0169-7439(99)00061-1

Chiang, L.H., Kotanchek, M.E., Kordon, A.K., 2004. Fault diagnosis based on Fisher discriminant analysis and support vector machines. Comput. Chem. Eng. 28, 1389–1401. doi:10.1016/j.compchemeng.2003.10.002

Cho, J.-H., Lee, J.-M., Wook Choi, S., Lee, D., Lee, I.-B., 2005. Fault identification for process monitoring using kernel principal component analysis. Chem. Eng. Sci. 60, 279–288. doi:10.1016/j.ces.2004.08.007

Choi, S.W., Park, J.H., Lee, I.-B., 2004. Process monitoring using a Gaussian mixture model via principal component analysis and discriminant analysis. Comput.

Dai, Y., Zhao, J., 2011. Fault Diagnosis of Batch Chemical Processes Using a Dynamic Time Warping (DTW)-Based Artificial Immune System. Ind. Eng. Chem. Res. 50, 4534–4544. doi:10.1021/ie101465b

Downs, J.J., Vogel, E.F., 1993. A plant-wide industrial process control problem. Comput. Chem. Eng. 17, 245–255. doi:10.1016/0098-1354(93)80018-I

Eslamloueyan, R., Shahrokhi, M., Bozorgmehri, R., 2003. Multiple simultaneous fault diagnosis via hierarchical and single artificial neural networks. Sci. Iran. 10.

Eslamloueyan, R., 2011. Designing a hierarchical neural network based on fuzzy clustering for fault diagnosis of the Tennessee-Eastman process. Appl. Soft Comput. 11, 1407–1415. doi:10.1016/j.asoc.2010.04.012

Fan, J.Y., Nikolaou, M., White, R.E., 1993. An approach to fault diagnosis of chemical processes via neural networks. AIChE J. 39, 82–88. doi:10.1002/aic.690390109

Fan, J., Wang, Y., 2014. Fault detection and diagnosis of non-linear non-Gaussian dynamic processes using kernel dynamic independent component analysis. Inf. Sci. (Ny). 259, 369–379. doi:10.1016/j.ins.2013.06.021

Ge, Z., Xie, L., Kruger, U., Song, Z., 2012. Local ICA for multivariate statistical fault diagnosis in systems with unknown signal and error distributions. AIChE J. 58, 2357–2372. doi:10.1002/aic.12760

Ghosh, K., Srinivasan, R., 2011. Immune-System-Inspired Approach to Process doi:10.1021/ie100767c

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep Residual Learning for Image Recognition, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778. doi:10.1109/CVPR.2016.90

He, Q.P., Wang, J., 2007. Fault detection using the k-nearest neighbor rule for semiconductor manufacturing processes. IEEE Trans. Semicond. Manuf. 20, 345–354. doi:10.1109/TSM.2007.907607

Hinton, G.E., Roweis, S.T., 2002. Stochastic neighbor embedding. Adv. Neural Inf. Process. Syst. 833–840. doi:http://books.nips.cc/papers/files/nips15/AA45.pdf

Hinton, G.E., Salakhutdinov, R., 2006a. Reducing the Dimensionality of Data using Neural Networks. Science (80-. ). 313, 504–507.

Hinton, G.E., Osindero, S., Teh, Y.-W., 2006b. A Fast Learning Algorithm for Deep Belief Nets. Neural Comput. 18, 1527–1554. doi:10.1162/neco.2006.18.7.1527

Hinton, G.E., Srivastava, N., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.R., 2012. Improving neural networks by preventing co-adaptation of feature detectors. arXiv Prepr. arXiv:1207.0580

Hsu, C.-C., Chen, M.-C., Chen, L.-S., 2010. A novel process monitoring approach with dynamic independent component analysis. Control Eng. Pract. 18, 242–253. doi:10.1016/j.conengprac.2009.11.002

Kano, M., Tanaka, S., Hasebe, S., Hashimoto, I., Ohno, H., 2003. Monitoring independent components for fault detection. AIChE J. 49, 969–976. doi:10.1002/aic.690490414

Krizhevsky, A., Sutskever, I., Hinton, G.E., 2012. ImageNet Classification with Deep Convolutional Neural Networks. Adv. Neural Inf. Process. Syst. 1–9. doi:http://dx.doi.org/10.1016/j.protcy.2014.09.007

Kulkarni, A., Jayaraman, V.K., Kulkarni, B.D., 2005. Knowledge incorporated support vector machines to detect faults in Tennessee Eastman Process. Comput. Chem. Eng. 29, 2128–2133. doi:10.1016/j.compchemeng.2005.06.006

LeCun, Y., Jackel, L.D., Boser, B., Denker, J.S., Graf, H.P., Guyon, I., Henderson, D., Howard, R.E., Hubbard, W., 1989. Handwritten digit recognition: applications of

neural network chips and automatic learning. IEEE Commun. Mag. 27, 41–46. doi:10.1109/35.41400

LeCun, Y., Bengio, Y., Hinton, G., 2015. Deep learning. Nature 521, 436–444. doi:10.1038/nature14539

Lee, J.-M., Qin, S.J., Lee, I.-B., 2007. Fault detection of non-linear processes using kernel independent component analysis. Can. J. Chem. Eng. 85, 526–536.

Lin, M., Chen, Q., Yan, S., 2013. Network In Network. arXiv Prepr. 10. doi:10.1109/ASRU.2015.7404828

Lv, F., Wen, C., Bao, Z., Liu, M., 2016. Fault Diagnosis Based on Deep Learning. 2016 Am. Control Conf. 6851–6856, 6851–6856. doi:10.1109/ACC.2016.7526751

Maaten, L. Van Der, Hinton, G., 2008. Visualizing Data using t-SNE. J. Mach. Learn. Res. 1 620, 267–84. doi:10.1007/s10479-011-0841-3

Maurya, M.R., Rengaswamy, R., Venkatasubramanian, V., 2005. Fault Diagnosis by Qualitative Trend Analysis of the Principal Components. Chem. Eng. Res. Des. 83, 1122–1132. https://doi.org/10.1205/cherd.04280

Maurya, M.R., Rengaswamy, R., Venkatasubramanian, V., 2007. A signed directed graph and qualitative trend analysis-based framework for incipient fault diagnosis. Chem. Eng. Res. Des. 85, 1407–1422. https://doi.org/10.1016/S0263-8762(07)73181-7

Maurya, M.R., Paritosh, P.K., Rengaswamy, R., Venkatasubramanian, V., 2010. A framework for on-line trend extraction and fault diagnosis. Eng. Appl. Artif. Intell. 23, 950–960. https://doi.org/10.1016/j.engappai.2010.01.027

MacGregor, J.F., Jaeckle, C., Kiparissides, C., Koutoudi, M., 1994. Process monitoring and diagnosis by multiblock PLS methods. AIChE J. 40, 826–838. doi:10.1002/aic.690400509

Mahadevan, S., Shah, S.L., 2009. Fault detection and diagnosis in process data using one-class support vector machines. J. Process Control 19, 1627–1639. doi:10.1016/j.jprocont.2009.07.011

Plovoso, M.J., Kosanovich, K.A., 1994. Applications of multivariate statistical methods to process monitoring and controller design. Int. J. Control 59, 743–765. doi:10.1080/00207179408923103

Rato, T., Reis, M., Schmitt, E., Hubert, M., De Ketelaere, B., 2016. A systematic comparison of PCA-based Statistical Process Monitoring methods for high-dimensional, time-dependent Processes. AIChE J. 62, 1478–1493. doi:10.1002/aic.15062

Rato, T.J., Reis, M.S., 2017. Markovian and Non-Markovian sensitivity enhancing transformations for process monitoring. Chem. Eng. Sci. 163, 223–233. https://doi.org/10.1016/j.ces.2017.01.047

Russell, E.L., Chiang, L.H., Braatz, R.D., 2000. Fault detection in industrial processes using canonical variate analysis and dynamic principal component analysis. Chemom. Intell. Lab. Syst. 51, 81–93. doi:10.1016/S0169-7439(00)00058-7

Shu, Y., Zhao, J., 2016. Fault Diagnosis of Chemical Processes Using Artificial Immune System with Vaccine Transplant. Ind. Eng. Chem. Res. 55, 3360–3371. doi:10.1021/acs.iecr.5b02646

Simonyan, K., Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. CoRR abs/1409.1.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R., 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. J. Mach. Learn. Res. 15, 1929–1958. doi:10.1214/12-AOS1000

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., 2014. Going deeper with convolutions. arXiv Prepr. arXiv 1409.4842 1–9. doi:10.1109/CVPR.2015.7298594

Venkatasubramanian, V., Chan, K., 1989. A neural network methodology for process fault diagnosis. AIChE J. 35, 1993–2002. doi:10.1002/aic.690351210

Venkatasubramanian, V., Rengaswamy, R., Ka, S.N., Kavuri, S.N., Ka, S.N., 2003a. A review of process fault detection and diagnosis Part II : Qualitative models and search strategies. Comput. Chem. 313–326. doi:10.1016/S0098-1354(02)00161-8

Venkatasubramanian, V., Rengaswamy, R., Kavuri, S.N., Yin, K., 2003b. A review of process fault detection and diagnosis part III: Process history based methods. Comput. Chem. Eng. doi:10.1016/S0098-1354(02)00162-X

Venkatasubramanian, V., Rengaswamy, R., Kavuri, S.N., Yin, K., 2003c. A review of process fault detection and diagnosis Part I: Quantitative model based methods. Comput. Chem. Eng. 27, 293–311. doi:10.1016/S0098-1354(02)00160-6

Watanabe, K., Matsuura, I., Abe, M., Kubota, M., Himmelblau, D.M., 1989. Incipient fault diagnosis of chemical processes via artificial neural networks. AIChE J. 35, 1803–1812. doi:10.1002/aic.690351106

Watanabe, K., Hirota, S., Hou, L., Himmelblau, D.M., 1994. Diagnosis of multiple simultaneous fault via hierarchical artificial neural networks. AIChE J. 40, 839– 848. doi:10.1002/aic.690400510

Wise, B.M., Ricker, N.L., Veltkamp, D.F., Kowalski, B.R., 1990. Theoretical basis for the use of principal component models for monitoring multivariate processes. Process Control Qual. 1, 41–51.

Xie, D.,Li, B., 2015. A hierarchical deep neural network for fault diagnosis on Tennessee-Eastman process. ICMLA, 2015 IEEE $1 4 ^ { \mathrm { t h } }$ International Conference. doi:10.1109/ICMLA.2015.208

Yélamos, I., Escudero, G., Graells, M., Puigjaner, L., 2009. Performance assessment of a novel fault diagnosis system based on support vector machines. Comput.

Yin, S., Ding, S.X., Haghani, A., Hao, H., Zhang, P., 2012. A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process. J. Process Control 22, 1567–1581. doi:10.1016/j.jprocont.2012.06.009

Yu, J., Qin, S.J., 2008. Multimode process monitoring with bayesian inference-based finite Gaussian mixture models. AIChE J. 54, 1811–1829. doi:10.1002/aic.11515

Zhang, Y.W., 2008. Fault detection and diagnosis of nonlinear processes using improved kernel independent component analysis (KICA) and Support Vector Machine (SVM). Ind. Eng. Chem. Res. 47, 6961–6971. doi:Doi 10.1021/Ie071496x

Zhang, Y., Hu, Z., 2011. Multivariate process monitoring and analysis based on multi-scale KPLS. Chem. Eng. Res. Des. 89, 2667–2678. doi:10.1016/j.cherd.2011.05.005

Zhang, Z., Zhao, J., 2017. A deep belief network based fault diagnosis model for complex chemical processes. Comput. Chem. Eng. doi:10.1016/j.compchemeng.2017.02.041

Zhu, Z.B., Song, Z.H., 2011. A novel fault diagnosis system using pattern classification on kernel FDA subspace. Expert Syst. Appl. 38, 6895–6905. doi:10.1016/j.eswa.2010.12.034