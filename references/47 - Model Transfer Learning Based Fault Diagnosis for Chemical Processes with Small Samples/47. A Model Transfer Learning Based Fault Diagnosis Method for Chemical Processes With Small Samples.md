# A Model Transfer Learning Based Fault Diagnosis Method for Chemical Processes With Small Samples

Jun-Wei Zhu\* <sup></sup> , Bo Wang, and Xin Wang

Abstract: Traditional fault diagnosis methods relies on sufficient fault samples, but it is unrealistic since the fault is a low possibility event in real industrial scenes. To address the above issue, this paper proposed a fault diagnosis method for chemical processes with small samples. First, a data self-generating-based transfer learning (DSGTL) method is presented to expand the fault samples. The characteristic of fault data is learned by adversarial relation and transferred to the generated data. Moreover, a model-based transfer learning strategy is adopted to improve the robustness of the proposed method to the quality of generated data. Second, the sample reconstruction-based convolutional neural network (SR-CNN) is proposed which adaptively extracts features from both spatial domain and time domain and identifies the fault type of industrial process with small samples. Finally, the experimental result of the Tennessee Eastman (TE) process proves the validity and the feasibility of the proposed method.

Keywords: Convolutional neural network, fault diagnosis, generative adversarial network, small samples, transfer learning.

## 1. INTRODUCTION

Nowadays, the safety and reliability of the plant in chemical processes is vital importance. It is necessary to develop an intelligent fault detection and diagnosis (FDD) system in the industry to detect the abnormal situation and identify the fault type. The existing FDD techniques can be divided into three categories: knowledge-based [1], model-based [2] and data-based methods [3]. Nowadays, because of the widely used distributed control systems, plenty of process data is provided to build a data-based FDD model. In the wake of the developments in computer hardware, deep learning methods have made great progress [4]. Especially, the deep convolutional neural network (DCNN) [5] achieved high fault diagnosis accuracy considering 20 fault types with sufficient 2-D sensortime samples in Tennessee Eastman (TE) process.

Despite the reliable diagnostic results of deep learning methods, these methods rely on plenty of fault data which is unrealistic. In practical application, there are few effective methods because the fault data is limited and the system is always in normal operation. It causes the classifier is likely to have poor generalization ability. Therefore, the problem of the small sample gets lots of attention, whose main purpose is to realize the fault diagnosis with only a few fault data [6].

Data augmentation methods are always used to increase the fault samples. An oversampling method such as Syn thetic Minority Over-sampling Technique is used to gen erate the new data by linearly combining the existing sam ples [7]. But it may cause the overfitting problem because of the excessive similarity between row data and gener ated data [8]. An auxiliary classifier generative adversarial network (GAN) [9] method is proposed to generate realistic 1-D raw data for data augmentation and the quality of generated data is evaluated by statistical features and experiments [10]. The time-frequency transformation technology is used to construct the 2-D representation of signals [11], and the deep convolutional generative adversarial network (DCGAN) [12] is employed to get a balanced dataset.

Transfer learning (TL) is a popular tool for fault diagnosis, which also has been involved in the case of small samples. A transfer framework under the small samples condition is proposed which can expand the fault samples and make the shared distribution smoother [13]. The maximum mean discrepancy item and deep neural network [14] are integrated to minimize the discrepancy penalty between the features in training data and testing data. In [15], the fault diagnosis model of the gas turbine was constructed using a convolutional neural network. The lack of training data was solved to some extent by migrating the fault knowledge among different gas turbines. In [16,17], the fault diagnosis model of the related equipment was constructed by using the powerful feature extraction ability of the self-encoder with the limited story samples. Based on a unified 1D convolution network, the authors of [18] build seven few-shot transfer learning methods for few-shot diagnosis of three datasets.

In the above results, the performance of fault diagnosis has been greatly improved. However, the fault types that need to be considered in the real process are generally more than the faults considered in previous methods. This article considers two aspects to overcome the abovementioned problem. On the one hand, the traditional convolutional neural network (CNN) methods didn’t take the spatial correlation of sensors into account [5,19]. In the real industrial process, the related sensors are first affected after the fault occurs. Subsequently, due to the feedback effect of the loop such as PID, the fault gradually spreads to the adjacent sensors and ultimately affects the whole system. Based on the above phenomenon, the propagation of faults in the process has time lag and spatial correlation. Therefore, only considering either the time domain or the spatial domain cannot extract deep fault information well. On the other hand, the previous data augmentation method used to mix the generated data with the real data to train a fault classifier which is very dependent on the quality of the generated data. However, the distribution of the generated data cannot be consistent with the distribution of the real data [20].

In this article, a sample reconstruction-based transfer learning (SR-TL) method is proposed for chemical processes fault diagnosis with small samples. The detailed contributions of this paper include

1) The data self-generating-based transfer learning (DS-GTL) method is presented. Through the adversarial learning process, the characteristics of real samples are transferred to generate fake samples. Then, the model-based transfer learning strategy is applied to train the fault classifier through the real samples and the fake samples successively, which can reduce the dependence of the proposed method on the quality of generated data.

2) Inspired by the application of 3D-CNN [21] in human action recognition, a sample reconstructionbased CNN (SR-CNN) method is first proposed to extract both time and spatial information to train a classifier with better generalization ability with small samples. Thus, when considering more than 10 types of fault, the proposed method still shows good diagnostic performance.

## 2. SR-BASED FAULT DIAGNOSIS METHOD

## 2.1. Sample reconstruction

The proposed sample reconstruction method considers the propagation of fault effect in both time and spatial domains, and the detailed steps are shown as follows:

Step 1: Normalize the scale of each sensor ¯v to [−1, 1] by

$$
v = \frac {\bar {v} - \min (\bar {v})}{\max (\bar {v}) - \min (\bar {v})} \times 2 - 1.\tag{1}
$$

Step 2: With the help of the literature and system flow chart, the complex large-scale industrial process can be divided into multiple subsystem units such as input unit $u _ { 1 }$ , reactor unit $u _ { 2 } ,$ , condenser unit $u _ { 3 } ,$ , output unit $u _ { 4 }$ and so on. Each sensor belonging to different subsystem unit is also assigned accordingly.

Step 3: For the whole system, subsystem units should be arranged according to the material transfer relation ship between subsystems. The material enters the system from the input unit and then reacts in the reactor. After the chemical reaction, the product is separated by the condensing unit and then reaches the output unit.

Based on the above process, the spatial relationship between the units is obtained that $u _ { 1 }$ is adjacent to $u _ { 2 } , u _ { 2 }$ is adjacent to u and $u _ { 3 }$ is adjacent to $u _ { 4 }$

Step 4: In each unit, the position of the variable in the sample matrix is determined by the positional relation ship of the sensor in the real industrial process. In sensor networks, there is great redundancy and high correlation between the data collected, while multiple sensor nodes sense consenting event information and the closer the ge ographical location, the higher the correlation between the nodes, due to the high density of the network topology. Therefore, the more relevant the two sensors are in the ac tual process, the closer they are in the sample matrix.

For example, suppose that each sample contains Ψ sensors and t sampling time. The traditional 2-D sample matrix with the size of $\Psi \times t$ is reconstructed to 3-D form V with the size of $\xi \times \phi \times t$ by the proposed sample reconstruction method, where $\Psi = \xi \times \phi$ . The reconstructed sample matrix is shown as follows:

$$
V = \left[ V _ {1}, \dots , V _ {\tau}, \dots , V _ {t} \right], 1 \leq \tau \leq t,\tag{2}
$$

$$
V _ {\tau} = \left[ \begin{array}{c c c c} v _ {1 1 \tau} ^ {1} & v _ {1 2 \tau} ^ {2} & \dots & v _ {1 \varphi \tau} ^ {R} \\ v _ {2 1 \tau} ^ {1} & v _ {2 2 \tau} ^ {2} & & \vdots \\ \vdots & & \ddots & \vdots \\ v _ {\xi 1 \tau} ^ {1} & v _ {\xi 2 \tau} ^ {2} & \dots & v _ {\xi \phi \tau} ^ {R} \end{array} \right],   1 \leq r \leq R,\tag{3}
$$

where $V _ { \tau }$ means the value of sensors at time τ, and r denotes that the sensor belongs to r-th subsystem unit.

The features of time and spatial domain can be easily extracted by convolutional operation, which helps reduc the depth of the classifier and data generator. Thus, in th case of small samples, the proposed sample reconstruction method can effectively avoid the poor generalization ability of the model.

## 2.2. Data self-generating-based transfer learning

The proposed sample reconstruction-based convolutional generative adversarial network (SR-CGAN) transfers the characteristic of fault samples to a new domain through adversarial learning based on DCGAN [12]. The SR-CGAN consists of two networks, one is generator (G) that generates fake data similar to the real data, and the other is discriminator (D), which evaluates whether the data is real or generated by the generator. The two networks challenge each other in a zero-sum game, as shown in Fig. 1. In this way, G and D constitute a dynamic “game process” with the final equilibrium point.

During the SR-CGAN training process, G learns the distribution of the training data $p _ { r }$ to confuse D that mapping the random noise samples z from a predefined latent space $p _ { g }$ to $p _ { r }$ . By contrast, D attempts to provide a high probability for real data and provide a low probability for the generated data. To specifically demonstrate SR-CGAN mathematically, the main objective of the minmax game between generator and discriminator is

$$
\begin{array}{l} \underset {G} {\min} \underset {D} {\max} (D, G) \\ = E _ {x \setminus p _ {r}} [ \log D (x) ] + E _ {\tilde {x} \sim p _ {g}} [ \log (1 - D (\tilde {x})) ], \end{array}\tag{4}
$$

where x is real training data, ˜x is the fake data generated by G, D(·) denotes the probability that the input comes from the real data.

Different from the standard DCGAN, considering that only a few training samples are available, the network depth is reduced to minimize the number of parameters. The modified generator consists of FC layer, batch normalization (BN) layers, and fractional-strided step convolution (FSC) layers [22], and the discriminator consists of convolutional layers, BN layers, and FC layer. Moreover, kernels in FSC layers and convolutional layers are also adjusted to 3-D form to match the 3-D input samples.

In the network optimization stage, adversarial training is implemented. The stochastic gradient descent (SGD) algorithm is adopted to find the optimal values of model parameters. Specifically, a two-step optimization is applied in each iteration. The parameters $\theta _ { g }$ and $\theta _ { d }$ represents the network parameters in each of the two networks, respectively, i represents the number of iterations. $\theta _ { g }$ in $G$ are optimized as (5) when parameters $\theta _ { d }$ in $D$ are fixed.

![](images/d818f8479506f6fa3e5d502420fe0b041fb836b635f46ea8a72d4059f181d6cd.jpg)  
Fig. 1. Framework of SR-CGAN.

$$
\begin{array}{l} L _ {g} ^ {i} = \frac {1}{\lambda} \sum_ {i = 1} ^ {\lambda} [ \log (1 - D (\tilde {x} _ {i})) ], \\ \theta_ {g} ^ {i + 1} = \theta_ {g} ^ {i} + \delta \frac {\partial L _ {g} ^ {i}}{\partial \theta_ {g} ^ {i}}, \end{array}\tag{5}
$$

where $L _ { g } ^ { i }$ denotes the objective for the $\lambda$ generated sam ples, δ represents the learning rate. Then, $\theta _ { d }$ is updated while $\theta _ { g }$ remains constant as

$$
\begin{array}{l} L _ {d} ^ {i} = L _ {g} ^ {i} + \frac {1}{\lambda} \sum_ {i = 1} ^ {\lambda} [ \log (D (\tilde {x} _ {i})) ], \\ \theta_ {d} ^ {i + 1} = \theta_ {d} ^ {i} + \delta \frac {\partial L _ {d} ^ {i}}{\partial \theta_ {d} ^ {i}}, \end{array}\tag{6}
$$

where $L _ { d } ^ { i }$ denotes the objective for the λ real samples. Through iterations of the two-step optimization in (5)-(6), the global optimum can be achieved when $p _ { g } = p _ { r }$

Assuming that γ types of fault is considered, the samples of η -th fault $X _ { \eta }$ is adopted to train a SR-CGAN model $G _ { \eta }$ . The trained $G _ { \eta }$ generates sufficent fake samples as ${ \tilde { X } } _ { \eta } ,$ , and the fake samples set is

$$
\tilde {X} = \left[ \tilde {X} _ {1}, \dots , \tilde {X} _ {\eta}, \dots , \tilde {X} _ {\gamma} \right], 1 \leq \eta \leq \gamma .\tag{7}
$$

Since the number of generated data is much larger than the real data, the weight of the fake sample set X<sup>˜</sup> must be much larger than the real sample set X, which leads to th wrong classification boundary.

Therefore, different from the traditional GAN-based FDD methods that always mix the generated data with the real data to train the classifier, the proposed DSGTL method takes the generated data X<sup>˜</sup> as the middle domain to pre-train the fault classifier. Subsequently, the parameters of the pre-trained model are transferred to be fine-tuned by the available real samples X to further improve the classi fication performance. Compared with direct training with a small amount of real samples, sufficient generated samples can help the classifier avoid falling into the local optimum, and the real samples are used to find the global optimum of the fault diagnosis model through fine-tune operation.

## 2.3. SR-CNN model training

The SR-CNN is adopted to extract the deep features from time and spatial domain for multiple fault diagnosis. Comparing with another CNN method, the proposed SR-CNN consists of much fewer layers, and the model structure is shown in Fig. 2. N , $N _ { 2 }$ is the number of filters, γ is the number of fault types.

![](images/b938e0c22c120858254db663dc0b4d9790ceaaf98996e0e75389005375d7aeca.jpg)  
Fig. 2. SR-CNN structure.

The output representation of the convolutional layer is composed of feature maps. Assuming that there are M feature maps as the input and N filters, the calculation process of convolutional layer is represented as follows:

$$
x _ {j} ^ {l} = f \left(\sum_ {i = 1, \dots , M} x _ {i} ^ {l - 1} * k _ {i j} ^ {l} + b _ {j} ^ {l}\right), j = 1, 2, \dots , N,\tag{8}
$$

where $k _ { i j }$ is the kernel of the j-th filter connectting to the i-th input map, $x _ { j } ^ { l }$ means the j-th output map in layer l and $x _ { i } ^ { l - 1 }$ represents the j-th input map, $b _ { j } ^ { l }$ is the bias corresponding to j-th filter, “∗” represents the convolutional operation and $f ( \cdot )$ represents the rectified linear unit (ReLU) activation function to improve the convergence rate of the model, written as follows:

$$
f (x) = \max (0, x).\tag{9}
$$

The pooling layer is also called sub-sampling layer, which reduces the size of the input feature map and combines similar local features into one. In this paper, Max pooling is adopted, which takes the maximum value of each block as the fusion feature. In a pooling layer, if there exist N feature maps as the input, there must be N output feature maps. Equation (10) calculates the output of the l-th pooling layer.

$$
S _ {j} ^ {l} = d o w n (x _ {j} ^ {l - 1}), j = 1, 2, \dots , N,\tag{10}
$$

where down represents the sub-sampling function. $x _ { j } ^ { l - 1 }$ is the j-th input map.

Different from the feature extraction function of the above two kinds of layers, the FC layer classifies features into a specific label. The output of the last pooling layer $S _ { j } ^ { l }$ should be reshaped to 1-D array S. Assume that the size of input S and output F are $r \times 1$ and h×1, each input neuron is connected to each output neuron. The output matrix F of FC layer is calculated as

$$
F = f (w ^ {T} \cdot S + b),\tag{11}
$$

where w is a $r \times h$ weight matrix, b is the $h \times 1$ bias matrix and $f ( \cdot )$ represents softmax function, which transform a $\gamma -$ dimensional vector $F$ to within (0, 1) range. The function is shown as

$$
y _ {j} ^ {\alpha} = \frac {e ^ {F _ {j}}}{\sum_ {k = 1} ^ {\gamma} e ^ {F _ {k}}}, j = 1, 2, \dots , \gamma ,\tag{12}
$$

![](images/7e519f50d3ab48b10ab34fde7800a4c8c0e6bc12216850d1e0dee7959612b23c.jpg)  
Fig. 3. Flowchart of the SR-TL method.

where $y ^ { \alpha } = [ y _ { 1 } ^ { \alpha } , y _ { 2 } ^ { \alpha } , . . . , y _ { \gamma } ^ { \alpha } ] , y ^ { \alpha }$ is the predicted label of the α-th sample.

The training set is divided into small batches according to the hyperparameter named “batch size” which means the number of sample matrices in one forward/backward pass of each iteration. The cross-entropy loss is calculated according to (13) and minimized by Adam optimization algorithm [23].

$$
E = - \frac {1}{\sigma} \sum_ {\alpha = 1} ^ {\sigma} \left[ T ^ {\alpha} \ln y ^ {\alpha} + (1 - T ^ {\alpha}) \ln (1 - y ^ {\alpha}) \right],\tag{13}
$$

where σ represents the batch size, $T ^ { \alpha }$ is the true label of the α-th sample. The parameters of SR-CNN model is modified by backpropagation algorithm until model con verges.

The whole framework of the SR-TL method is shown in Fig. 3. In the SR-TL training process, the limited training data is preprocessed and reconstructed to 3-D form. Every type of fault samples are used to train a SR-CGAN model which can generate high-quality fault samples. Therefore, the SR-CNN model is pre-trained by the generated samples and fine-tuned by the real samples, and the fault diagnosis model is obtained and stored, which can be called online. In the model test process, the system’s real-time data is preprocessed, and it is detected whether the system is under normal conditions. If the system condition is abnormal, the collected data will be reconstructed to 3- D form, and the trained SR-CNN model will identify the fault type.

## 3. CASE STUDY

## 3.1. Tennessee Eastman process

Nowadays, TE process is widely used for researche about process monitoring, FDD problem, and optimiza tion control [24]. Hence, it is adopted to verify the per formance of the proposed FDD method. The simulator is based on the revised version [25]. TE process includs 12 manipulated variables, 22 continuous process measurements, and 19 component analysis measurements. In this work, we selected 31 variables, which are listed in Table 1. The main operating units of this process are the reactor, the condenser, the cycle compressor, the gas/liquid separator, and the stripper. Although there are including 28 fault types, we only take the IDV1- IDV20 into consideration for algorithm verification. IDV6 is not taken into account because the system is forced to shut down about 7 hours after the fault happens. The detail of the fault description is listed in Table 2.

Table 1. Blocks and the corresponding variables.

<table><tr><td>No.</td><td>Measured variable</td><td>No.</td><td>Measured variable</td></tr><tr><td>1</td><td>A feed</td><td>17</td><td>Stripper underflow</td></tr><tr><td>2</td><td>D feed</td><td>18</td><td>Stripper temperature</td></tr><tr><td>3</td><td>E feed</td><td>19</td><td>Stripper steam flow</td></tr><tr><td>4</td><td>Total feed</td><td>20</td><td>Compressor work</td></tr><tr><td>5</td><td>Recycle flow</td><td>21</td><td>Reactor cooling water outlet temperature</td></tr><tr><td>6</td><td>Reactor feed rate</td><td>22</td><td>Separator cooling water outlet temperature</td></tr><tr><td>7</td><td>Reactor pressure</td><td>23</td><td>D feed flow valve</td></tr><tr><td>8</td><td>Reactor level</td><td>24</td><td>E feed flow valve</td></tr><tr><td>9</td><td>Reactor temperature</td><td>25</td><td>A feed flow valve</td></tr><tr><td>10</td><td>Purge rate</td><td>26</td><td>Total feed flow valve</td></tr><tr><td>11</td><td>Product separator temperature</td><td>27</td><td>Purge valve</td></tr><tr><td>12</td><td>Product separator level</td><td>28</td><td>Separator pot liquid flow valve</td></tr><tr><td>13</td><td>Product separator pressure</td><td>29</td><td>Stripper liquid product flow valve</td></tr><tr><td>14</td><td>Product separator underflow</td><td>30</td><td>Reactor cooling water flow</td></tr><tr><td>15</td><td>Stripper level</td><td rowspan="2">31</td><td rowspan="2">Condenser cooling water flow</td></tr><tr><td>16</td><td>Stripper pressure</td></tr></table>

First of all, the simulator runs about 100 hours for each kind of system condition. The measured variables are collected and normalized to −1-1 to eliminate the difference caused by different dimensions. The sampling period is set to 3 min, so that we get 2000 sampling points for each system condition. Secondly, considering the information of time domain, each sample contains 10 sampling points. Thus, every working condition contains 200 samples. For each fault type, 20 samples are randomly selected for training and the rest 180 samples are taken for test.

## 3.2. Fault diagnosis

To establish the SR-TL based fault diagnosis model, the 31 selected variables are assigned to 4 subsystem units according to their spatial position in the real system and the causal relationship between variables, as shown in Table 3. Fig. 4 shows the projection of the reconstructed sample matrix which is corresponding to (3). ‘0’ means the valu is a constant of zero, and the background color represents different subsystem units. The variable number in Table 3 and Fig. 4 corresponds to the number in Table 1.

Table 2. Fault types descriptions in TE.

<table><tr><td>IDV</td><td>Description</td></tr><tr><td>0</td><td>Normal operation</td></tr><tr><td>1</td><td>A/C feed ratio, B composition constant</td></tr><tr><td>2</td><td>B composition. A/C ratio constant</td></tr><tr><td>3</td><td>D feed temperature</td></tr><tr><td>4</td><td>Reactor cooling water inlet temperature</td></tr><tr><td>5</td><td>Condenser cooling water inlet temperature</td></tr><tr><td>7</td><td>C header pressure loss-reduced availability</td></tr><tr><td>8</td><td>A, B, C feed composition</td></tr><tr><td>9</td><td>D feed temperature</td></tr><tr><td>10</td><td>C feed temperature</td></tr><tr><td>11</td><td>Reactor cooling water inlet temperature</td></tr><tr><td>12</td><td>Condenser cooling water inlet temperature</td></tr><tr><td>13</td><td>Reaction kinetics</td></tr><tr><td>14</td><td>Reactor cooling water valve</td></tr><tr><td>15</td><td>Condenser cooling water valve</td></tr><tr><td>16-20</td><td>Unknown faults</td></tr></table>

Table 3. Blocks and the corresponding variables.

<table><tr><td>Blocks</td><td>Variables</td><td>Description</td></tr><tr><td>1</td><td>1, 2, 3, 5, 6, 23, 24, 25</td><td>Input</td></tr><tr><td>2</td><td>7, 8, 9, 21, 30</td><td>Reactor</td></tr><tr><td>3</td><td>10, 11, 12, 13, 14, 20, 22, 27, 28, 31</td><td>Separator, compressor</td></tr><tr><td>4</td><td>4, 15, 16, 17, 18, 19, 26, 29</td><td>Condenser, stripper</td></tr></table>

<table><tr><td>1</td><td>2</td><td>3</td><td>0</td><td>0</td><td>10</td><td>11</td><td>12</td><td>4</td><td>15</td><td>16</td></tr><tr><td>5</td><td>6</td><td>23</td><td>7</td><td>8</td><td>13</td><td>14</td><td>20</td><td>17</td><td>18</td><td>19</td></tr><tr><td>24</td><td>25</td><td>9</td><td>21</td><td>30</td><td>22</td><td>27</td><td>28</td><td>31</td><td>26</td><td>29</td></tr></table>

Fig. 4. Projection of the reconstructed sample matrix.

For each type of fault, the preprocessed training samples are used to train a SR-CGAN model. Then, the SR-CGAN model transfers the characteristics of fault samples to 300 generated samples for each type of fault. Take IDV1 as an example for visualization, the t-distributed stochastic neighbor embedding (t-SNE) [26] technique is utilized to map the real samples and generated samples into 3- D space, as shown in Fig. 5. The distribution of sample proved that although $\tilde { X } _ { 1 }$ is distributed around $X _ { 1 }$ , the cores of the two distributions are still different.

Second, use the fake sample set $\tilde { X }$ to pre-train a SR CNN fault classifier. When the pre-trained model converges, the model is transferred to be fine-tuned by th real sample set X.

![](images/3f593b8c668ca282d747d4378feb0d2dd2d08d468d2f01f57e41dec1ed923680.jpg)  
Fig. 5. t-SNE visualization of SR-CGAN performance (IDV1).

Table 4. Model structure.

<table><tr><td>Network</td><td>Model structure</td></tr><tr><td>SR-CNN</td><td>Conv layer - Conv layer- MaxPooling - FC layer</td></tr><tr><td>SR-CGAN (G)</td><td>FC layer - BN - FSC layer- BN - FSC layer</td></tr><tr><td>SR-CGAN (D)</td><td>Conv layer - Conv layer- BN - FC layer</td></tr></table>

Finally, the diagnostic performance of the trained SR-CNN model is tested by the test samples, and the test accuracy of the proposed method is obtained.

The fault diagnosis experiments are carried out in two cases: Case 1 for IDV1-11 and Case 2 for IDV1-20. IDV6 is not taken into consideration for both two cases. The detail of the network structure for experiments is shown in Table 4.

For SR-CNN model, the number of filters $N _ { 1 }$ is set to 128, N is set to 256, the number of fault type is 10 and 19, filter size in convolutional layers is $2 \times 2 \times 2$ , strides are set to $1 \times 1 \times 1$ , padding is ‘SAME’, the filter size of pooling layer is set to $2 \times 2 \times 2$ , strides are set to $2 \times 2 \times 2$ When pre-training the SR-CNN, the batch size and epoch are set to 128 and 20. In the fine-tuning process, the batch size and epoch are set to 32 and 100.

For generator, the size of noise input is set to $1 \times 1 0 0 ,$ the first FSC layer’s filter size is set to $2 \times 2 \times 2 .$ , strides are set to $2 \times 2 \times 2$ , the second FSC layer’s filter size is set to $2 \times 2 \times 2$ , strides are set to $2 \times 2 \times 1$ . For discriminator, the filter size and strides of two convolutional layers are both set to $2 \times 2 \times 2$ . And the batch size and epoch for SR-CGAN are set to 8 and 200.

To show the detail of diagnostic performance and the effect of DSGTL strategy, the curves of training loss and testing accuracy (Case 1) are illustrated in Fig. 6. In the pre-train stage, the training loss decreased and the test accuracy increased to about 35%. Once the fine-tuning operation is applied (epoch 20), the loss suddenly increased a lot, which means that the pre-trained model can not fully fit the real sample distribution because of the differenc between fake and real distributions. Thus, X adjusts the parameters of the model to minimize the training loss. At about the 80th epoch, the SR-CNN model converges and the test accuracy reaches the maximum of about 81%.

![](images/946b8e29e00f8d6b160ff73425d20c74ac9d7e745bc6006c47b6d3d27da15c01.jpg)  
Fig. 6. The training loss and testing accuracy for the SR-CNN iteration process.

Table 5. The average accuracy between different methods.

<table><tr><td>Fault diagnosis accuracy</td><td>Case 1</td><td>Case 2</td></tr><tr><td>SR-TL (our method)</td><td>80.9%</td><td>75.6%</td></tr><tr><td>MLP</td><td>65.3%</td><td>52.5%</td></tr><tr><td>GAN-MLP</td><td>46.6%</td><td>33.7%</td></tr><tr><td>DCNN</td><td>42.3%</td><td>35.8%</td></tr><tr><td>Wasserstein-TL</td><td>76.3%</td><td>69.5%</td></tr></table>

To assess the capacity of SR-TL based fault diagnosis method, comparative experiments are also conducted in this article, and various methods are involved, i.e., MLP, GAN-MLP, DCNN [5] and Wasserstein-TL method [13]. MLP and GAN-MLP are used as comparison methods in [27]. The DCNN network contains three convolutional layers, two pooling layers and one FC layer, which takes much time for training. For Wasserstein-TL, the generated samples are weighted by Wasserstein distance that the weight is inversely proportional to distance.

Table 5 shows the average accuracy of ten experiments, the performance of GAN and DCNN is limited by the number of training samples. Although GAN-MLP generates many samples for training, the accuracy is much lower than MLP. The reason for this phenomenon may be that the unrealistic data generated by GAN leads to the confusion of classification. The Wasserstein-TL method has achieved good performance, but its accuracy still lower than the proposed SR-TL method. The experimental result proved that the proposed method has better generalization ability, and can diagnose 19 types of fault with very limited fault samples. The details of the fault diagno sis result are illustrated in Fig. 7 which shows the confusion matrix of Case 1. The principal diagonal shows the classification accuracy of each fault type.

![](images/8fad3a95430f4ddd22fa2e2b5bac5f7a2624cc9713eaae5429eedc2bec8cad56.jpg)  
Fig. 7. The confusion matrix of testing samples.

## 4. CONCLUSION

In this paper, the SR-TL based fault diagnosis method for chemical processes is proposed to overcome the small sample problem. The experimental results demonstrate that the proposed method has an excellent performance in TE process. Even when considering 10 and 19 fault types, the test accuracy can reach 80.9% and 75.6%, which is much better than other compared methods.

This method is appropriate for large-scale industrial processes which include many manipulate variables and measurement variables. The expert knowledge for sample reconstruction could be easily obtained from expert and system literature. Moreover, the proposed method has low requirements for the number of training data which limited many traditional fault diagnosis methods in real industrial scenes. In our future work, how to apply the SR-TL method to real industrial processes will be discussed.

## CONFLICTS OF INTERESTS

The authors declare that there is no competing financial interest or personal relationship that could have appeared to influence the work reported in this paper.

## REFERENCES

[1] Y. Chi, Y. Dong, and J. Wang, “Knowledge-based fault diagnosis in industrial internet of things: A survey,” IEEE Internet of Things Journal, vol. 9, no. 15, pp. 12886-12900, 2022.

[2] J. Zhu, C. Gu, S. Ding, and W. Zhang, “A new observerbased cooperative fault-tolerant tracking control method with application to networked multiaxis motion control system,” IEEE Transactions on Industrial Electronics, vol. 68, no. 8, pp. 7422-7432, 2021.

[3] S. X. Ding, S. Yin, K. Peng, H. Hao, and B. Shen, “A novel scheme for key performance indicator prediction and diag nosis with application to an industrial hot strip mill,” IEEE Transactions on Industrial Informatics, vol. 9, no. 4, pp. 2239-2247, 2012.

[4] Y. Liu, Z. Chen, and Y. Li, “Robot search path plannin method based on prioritized deep reinforcement learning,” International Journal of Control, Automation, and Systems, vol. 20, no. 8, pp. 2669-2680, 2022.

[5] H. Wu and J. Zhao, “Deep convolutional neural network model based chemical process fault diagnosis,” Computers & Chemical Engineering, vol. 115, pp. 185-197, 2018.

[6] J. Liu, F. Qu, X. Hong, and H. Zhang, “A small-sample wind turbine fault detection method with synthetic fault data using generative adversarial nets,” IEEE Transactions on Industrial Informatics, vol. 15, no. 7, pp. 3877-3888, 2018.

[7] H. Han, W.-Y. Wang, and B.-H. Mao, “Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning,” Proc. of International Conference on Intelligent Computing, Springer, pp. 878-887, 2005.

[8] S. Lu, J. Feng, H. Zhang, J. Liu, and Z. Wu, “An estimation method of defect size from MFL image using visual transformation convolutional neural network,” IEEE Transactions on Industrial Informatics, vol. 15, no. 1, pp. 213-224 2018.

[9] M. Sung, J. Kim, and M. Lee, “Realistic sonar image simulation using deep learning for underwater object detection,” International Journal of Control, Automation, and Systems, vol. 18, no. 3, pp. 523-534, 2020.

[10] S. Shao, P. Wang, and R. Yan, “Generative adversarial net works for data augmentation in machine fault diagnosis,” Computers in Industry, vol. 106, pp. 85-93, 2019.

[11] J. Viola, Y. Chen, and J. Wang, “Faultface: Deep convolutional generative adversarial network (DCGAN) based ball-bearing failure detection method,” Information Sci ences, vol. 542, pp. 195-211, 2021.

[12] A. Radford, L. Metz, and S. Chintala, “Unsupervised rep resentation learning with deep convolutional generative adversarial networks,” arXiv preprint arXiv:1511.06434, 2015.

[13] J. Liu and Y. Ren, “A general transfer framework based on industrial process fault diagnosis under small samples,” IEEE Transactions on Industrial Informatics, vol. 17, no 9, pp. 6073-6083, 2021.

[14] L. Wen, L. Gao, and X. Li, “A new deep transfer learn ing based on sparse auto-encoder for fault diagnosis,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 49, no. 1, pp. 136-144, 2017.

[15] M. Liang, X. Yang, and F. Jin, “Convolutional neural network-based deep transfer learning for fault detection of gas turbine combustion chambers,” Applied Energy, vol 302, 117509, 2021.

[16] D. Yang and Karimi, “Residual wide-kernel deep convolu tional auto-encoder for intelligent rotating machinery fault diagnosis with limited samples,” Neural Networks, vol. 141, pp. 133-144, 2021.

[17] K. Zhao and H. Jiang, “A new data generation approach with modified wasserstein auto-encoder for rotating machinery fault diagnosis with limited fault data,” Knowledge-Based Systems, vol. 238, 107892, 2022.

[18] J. Wu, Z. Zhao, C. Sun, R. Yan, and X. Chen, “Few-shot transfer learning for intelligent fault diagnosis of machine,” Measurement, vol. 166, 108202, 2020.

[19] W. Yu and C. Zhao, “Broad convolutional neural network based industrial process fault diagnosis with incremental learning capability,” IEEE Transactions on Industrial Electronics, vol. 67, no. 6, pp. 5081-5091, 2019.

[20] X. Gao, F. Deng, and X. Yue, “Data augmentation in fault diagnosis based on the wasserstein generative adversarial network with gradient penalty,” Neurocomputing, vol. 396, pp. 487-494, 2020.

[21] S. Ji, M. Yang, and K. Yu, “3D convolutional neural networks for human action recognition,” IEEE Transactions on pattern analysis and machine intelligence, vol. 35, no. 1, pp. 221-231, 2012.

[22] T. Pan, J. Chen, J. Xie, Y. Chang, and Z. Zhou, “Intelligent fault identification for industrial automation system via multi-scale convolutional generative adversarial network with partially labeled samples,” ISA Transactions, vol. 101, pp. 379-389, 2020.

[23] D. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[24] J. J. Downs and E. F. Vogel, “A plant-wide industrial process control problem,” Computers & Chemical Engineering, vol. 17, no. 3, pp. 245-255, 1993.

[25] A. Bathelt, N. L. Ricker, and M. Jelali, “Revision of the Tennessee Eastman process model,” IFAC-PapersOnLine, vol. 48, no. 8, pp. 309-314, 2015.

[26] L. Maaten, “Accelerating t-SNE using tree-based algorithms,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3221-3245, 2014.

[27] X. Jiang and Z. Ge, “Data augmentation classifier for imbalanced fault classification,” IEEE Transactions on Automation Science and Engineering, vol. 18, no. 3, pp. 1206- 1217, 2021.

![](images/07e7335dca0a6c32371188f0a4ba3b047fb0452e7f082e12c17d592cdd76c66a.jpg)

Jun-Wei Zhu received his B.S. degree in control theory and engineering from Northeastern University, China, in 2008, an M.S. degree in control theory and engineering from Shenyang University, China, in 2011, and a Ph.D. degree in control theory and engineering from Northeastern University, China, in 2016. He is currently a special-termed Associate Profes-

sor at the College of Information Engineering, Zhejiang Univer sity of Technology. He is also a visiting professor of the Institute for Automatic Control and Complex Systems (AKS), Universit of Duisburg-Essen, Germany, from September 2019 to Septem ber 2020. His research interests include cyber-physical systems fault diagnosis, and fault tolerant control.

![](images/ae154992fd0a44aa69c8f2c05c17c646508aeef13c6e6bf10e0de55373501b19.jpg)

Bo Wang received her B.S. degree in automation from the Anhui Normal University, China, in 2020. She is currently working toward a master’s degree in control science and engineering from the Zhejiang University of Technology, Hangzhou, China. Her research interests include Faul classification, fault diagnosis, and faulttolerant control.

![](images/8673be919eedc4156faeb71b6c313e2d59e05ec761693997a814206853d4fad9.jpg)

Xin Wang received his B.S. degree in information and computing science and an M.S. degree in operational research and cybernetics from Heilongjiang University Harbin, China, in 2008 and 2011, respec tively, and a Ph.D. degree in navigation guidance and control from Northeastern University, Shenyang, China, in 2016. He is currently a Lecturer with the School of

Mathematical Science, Heilongjiang University, Harbin, China, and also a Post-Doctoral Fellow with the Department of Electrical Engineering, Yeungnam University, Gyeongsan, Korea From November 2017 to November 2018, he was a Visiting Professor in the Department of Mechanical Engineering, Universit of Victoria, Victoria, British Columbia, Canada. His research interests include fault diagnosis, fault-tolerant control, multiagent coordination, and time-delay systems.

Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affil iations.