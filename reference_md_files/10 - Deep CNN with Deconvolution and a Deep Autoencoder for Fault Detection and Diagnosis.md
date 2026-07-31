http://pubs.acs.org/journal/acsodf

Article

# Deep Convolutional Neural Network with Deconvolution and a Deep Autoencoder for Fault Detection and Diagnosis

Yasuhiro Kanno and Hiromasa Kaneko<sub>\*</sub>

![](images/3de6e0ceaf8d13a1313038248b7fc467c5c53e5bc39d43f58fd4c74ab9d50dd8.jpg)

Cite This: ACS Omega 2022, 7, 2458−2466

![](images/3b1df9eafede938d91fc77e437b6d00f694f53b42e074d365868f159973dc2f1.jpg)

Read Online

ACCESS

Metrics & More

Article Recommendations

\*sı Supporting Information

![](images/ce4e612af6f445acf8073bb42fcfd972f2a0480ee7bb0b56d43deaffabae52dc.jpg)  
ABSTRACT: In chemical plants and other industrial facilities, the rapid and accurate detection of the root causes of process faults is essential for the prevention of unknown accidents. This study focused on deep learning while considering the di<sup>f</sup>erent phenomena that can occur in industrial facilities. A deep convolutional neural network with deconvolution and a deep autoencoder (DDD) is proposed. DDD assesses the process dynamics and the nonlinearity between process variables. During the operation of DDD, fault detection is carried out using the reconstruction error between the data reconstructed through the model and the input data. After a process fault is detected, the magnitude of the contribution of each process variable to the detected process fault is calculated by applying gradient-weighted class activation mapping to the established network. The e<sup>f</sup>ectiveness of DDD in fault detection and diagnosis was veri<sup>fi</sup>ed through experiments on the Tennessee Eastman process dataset, demonstrating that it can achieve improved performance compared to the conventional fault detection and diagnosis.

## 1. INTRODUCTION

In chemical plants, accidents and mechanical failures result in signi<sup>fi</sup>cant economic losses. In such environments, process variables, such as temperature, <sup>fl</sup>ow rates, and pressure, are measured constantly to monitor and control chemical plant operations while ensuring the safety of the equipment as well as that of the human resources.<sup>1</sup> It is also possible to monitor chemical plant processes by applying techniques based on statistical process control (SPC), which rely on the variable data of measured processes. The number of process variables required to account for all the physical and chemical phenomena increases as the processes become increasingly complex. Therefore, monitoring multiple process variables collectively using multivariate SPC (MSPC) is considered a highly e<sup>fi</sup>cient approach. As a result, various MSPC-based methodologies<sup>2</sup> have been proposed. Examples of statistical methods that rely on MSPC include principal component analysis (PCA),<sup>3</sup> independent component analysis,<sup>4</sup> partial least squares analysis,<sup>5</sup> arti<sup>fi</sup>cial neural networks,<sup>6</sup> and support vector machines.<sup>7</sup> In addition, deep neural networks have been receiving increased research attention because the networks can be used to express complex relationships between various process variables.

Using neural networks with multiple layers, it is possible to realize the deeper learning of features contained in the data in a stepwise manner. Deep learning-based approaches, such as the deep autoencoder (DAE)<sup>8</sup> and convolutional neural networks (CNNs),<sup>9</sup> are used to detect and classify faults in chemical processes<sup>10</sup> and motor bearing.<sup>11</sup> Such approaches are also employed in other various <sup>fi</sup>elds and applications such as the diagnosis of malfunctions, including bearing failures<sup>12</sup> and turbine failures.<sup>13</sup> However, the interpretation of constructed deep neural networks can be di<sup>fi</sup>cult. This is despite their necessity in identifying the root causes of process faults in chemical processes following the detection of such faults.

Therefore, in the <sup>fi</sup>eld of arti<sup>fi</sup>cial intelligence, several methods for clarifying the bases estimated using deep neural networks have been proposed. For example, CNNs are currently considered e<sup>f</sup>ective approaches in the <sup>fi</sup>eld of image processing. CNNs incorporate a visualization method known as gradient-weighted class activation mapping (Grad CAM)<sup>14</sup> whose main function involves clarifying the basis for a judgment. Grad-CAM is a method for calculating the part of a neural network that contributes the most to a speci<sup>fi</sup>c output classi<sup>fi</sup>cation using the gradient of the convolutional layer and the probability score. This method has been used in the classi<sup>fi</sup>cation of pig models<sup>15</sup> and MRI brain images to establish a basis<sup>16</sup> for the classi<sup>fi</sup>cation of Alzheimer’s disease. However, although conventional methods can be applied in supervised learning applications, such as those for image classi<sup>fi</sup>cation, the methods cannot be applied in unsupervised learning applications, such as those for detecting process faults in chemical plants. Therefore, it is crucial to detect process faults and diagnose such faults.

![](images/e1cd35d5d2633f87da6f58e055e5826bc41f6c802ba6064d030bf57a39d4066d.jpg)

The aim of this study was to develop a method for detecting process faults using a deep neural network. In this study, a method that combines a CNN with a DAE is proposed to consider the nonlinearity between process variables and process dynamics in process variables. Because CNNs can consider the pixel intensity as well as the spatial relationship between pixels, it is possible to extract the temporal characteristics of each process variable. Subsequent dimen sional reduction was performed using a DAE, after which the latent variables considering the nonlinearity of the variables were extracted. Process faults were detected and diagnosed using the data transformed through the model (similar to $T ^ { 2 }$ in PCA-based MSPC) and the reconstruction error of the input data (similar to Q in PCA-based MSPC). This methodology is referred to as a deep convolutional neural network with deconvolution and a deep autoencoder (DDD). By applying Grad-CAM in the constructed neural network, it is possible to detect and diagnose process faults in latent variables by visualizing the high-weight input variables for each latent variable. To verify the e<sup>f</sup>ectiveness of DDD, the performances of existing MSPC-based methods, i.e., DAE, CNN, and DDD, in detecting process faults were compared on the Tennessee Eastman process (TEP) dataset. Furthermore, the process variables related to a process fault are diagnosed using Grad-CAM for the DDD.

## 2. METHODS

The proposed method, DDD, combines a DAE with a CNN, and fault diagnosis using DDD is based on the incorporation of Grad-CAM. First, DAE, CNN, and Grad-CAM are explained, and then DDD and fault diagnosis with DDD are discussed.

2.1. Deep Autoencoder. A basic autoencoder (AE) is a neural network comprising three layers, namely, an input layer, a single hidden layer, and an output layer. A network model comprising multiple hidden layers is referred to as a DAE. Figure 1 shows a schematic diagram of an AE. Given the input data $\mathbf { X } \epsilon R ^ { n \times { u } }$ , where n represents the number of samples and u represents the number of process variables $\mathrm { X } ,$ the input samples ${ \bf x } _ { i } \epsilon R ^ { u } , i = 1 , 2 , . . . ,$ n are encoded into neurons ${ \bf h } _ { i } \bar { \epsilon } R ^ { \nu } ,$ where v represents the number of neurons in the hidden layer, using the following formula:

![](images/420ad1324e7bc85680e24b44079ec9498d341846f6600b4d049ee7f1454b68e2.jpg)  
Figure 1. Basic concept of an autoencoder.

$$
\mathbf {h} _ {i} = f (\mathbf {W} _ {1} \mathbf {x} _ {i} + \mathbf {b} _ {1})\tag{1}
$$

where ${ \bf W } _ { 1 } \epsilon R ^ { \nu \times u }$ and ${ \bf b } _ { 1 } \epsilon R ^ { \nu }$ represent the weight and bias in the encoding process, respectively. Subsequently, the input sample is decoded from the neurons ${ \bf h } _ { i } \epsilon R ^ { \nu }$ into the reconstructed sample $\hat { \mathbf { x } } _ { i } \epsilon R ^ { u }$ using the following formula:

$$
\hat {\mathbf {x}} _ {i} = f (\mathbf {W} _ {2} \mathbf {h} _ {i} + \mathbf {b} _ {2})\tag{2}
$$

where ${ \bf W } _ { 2 } \epsilon R ^ { u \times \nu }$ and $ { \mathbf { b } } _ { 2 } \epsilon R ^ { \nu }$ represent the weight and bias in the decoding process, respectively. Therefore, the reconstructed data $\mathbf { \hat { X } } \mathbf { \epsilon } \mathbf { \check { R } } ^ { n \mathbf { \hat { \times } } u }$ of X are obtained from the AE. f represents an activation function that extracts the input features, and the sigmoid, tanh, and recti<sup>fi</sup>ed linear unit (ReLU) functions are used as general activation functions. The AE is trained so that the reconstruction error between X and X̂ diminishes, and $\theta =$ $\{ \mathbf { W } _ { 1 } , \mathbf { b } _ { 1 } , \mathbf { W } _ { 2 } , \mathbf { b } _ { 2 } \}$ is updated using the backpropagation method, as shown in the following equation:

$$
\min J _ {\mathrm{AE}} = \frac {1}{2 n} \sum_ {i = 1} ^ {n} | | \hat {x} _ {i} - x _ {i} | | ^ {2}\tag{3}
$$

2.2. Convolutional Neural Network. A CNN is a neural network that comprises an input layer, a convolutional layer, a pooling layer, a deconvolution layer, and an output layer. The convolutional layer applies a de<sup>fi</sup>ned number of <sup>fi</sup>lters to obtain a feature map of the input image, and the pooling layer reduces the number of input features. The convolutional and pooling layers are alternately repeated several times to extract the <sup>fi</sup>nal number of features. The image reconstructed through the deconvolution layer is then the output.

2.2.1. Convolutional Layer. The output of the convolutional layer comprises feature maps in which each unit is connected to a local patch of the input feature map via a weighted <sup>fi</sup>lter. All the units in the output feature map share the same <sup>fi</sup>lter, and within a layer, di<sup>f</sup>erent feature maps use di<sup>f</sup>erent <sup>fi</sup>lters. The convolutional layer can be used to facilitate the detection or recognition of patterns present in the process data. Assuming that there are $\mathbf { \bar { \rho } } _ { M \times \mathbf { \rho } 1 }$ input feature maps x<sup>l</sup> in the l layer and N <sup>fi</sup>lters, the output feature map $\mathbf { x } _ { i } ^ { l + 1 }$ at the jth position in the l + 1 layer is calculated as follows:

$$
\mathbf {x} _ {j} ^ {l + 1} = f \Bigg (\sum_ {i = 1} ^ {M} \mathbf {x} _ {i} ^ {l *} k _ {i j} ^ {l} + b _ {j} ^ {l} \Bigg), j = 1, \dots , N\tag{4}
$$

where k<sup>l</sup> represents the kernel of the jth <sup>fi</sup>lter connected to the ith, x<sup>l</sup>represents the ith input’s feature map, $\mathbf { x } _ { j } ^ { l + 1 }$ represents the jth input’s feature map, b<sub>j</sub><sup>l</sup>represents the bias corresponding to the jth <sup>fi</sup>lter, f represents the activation function, and the asterisk symbol (∗) represents the convolution operation. Common activation functions for neural networks include the sigmoid, tanh, and ReLU functions. Assuming a kernel size of s $\times \ : s ,$ the number of all parameters in the convolutional layer is calculated as follows:

$$
P = N \times (s \times s \times M + 1)\tag{5}
$$

The output feature map obtained from the convolutional layer is transferred to the pooling layer.

2.2.2. Pooling Layer. The pooling layer follows the convolutional layer and downsamples the input feature map. The purpose of the pooling layer is to compress information and transform the input data into a more manageable form. The use of pooling layers has two advantages. First, because the relative positions of the features forming the local pattern may di<sup>f</sup>er slightly, detecting the features with similar local positions o<sup>f</sup>ers enhanced reliability. Second, the dimensionality of feature representation can be reduced without setting parameters, thereby signi<sup>fi</sup>cantly reducing the computation time and the parameters of the entire network.

![](images/11805a5ae31feb22e46281f67bc4456ac67e88d4ff48bbf1f02263c495f51d66.jpg)  
Figure 2. Basic concept of DDD.

Primarily, there are two modes of pooling: maximum pooling and average pooling. The maximum pooling mode calculates the maximum value among the units in the feature map, and the average pooling mode calculates the average value of the units. In the pooling layer, when M feature maps of the l layer are the input, M feature maps are the output, as shown in the following formula:

$$
\mathbf {x} _ {j} ^ {l + 1} = f (\beta_ {j} ^ {l *} \mathrm{down} (\mathbf {x} _ {j} ^ {l}) + b _ {j} ^ {l})\tag{6}
$$

where x<sup>l</sup>represents the jth input feature map, $\mathbf { x } _ { j } ^ { l + \mathrm { ~ l ~ } }$ represents the $j \mathrm { t h }$ output feature map, β<sup>l</sup>and b<sup>l</sup>represent the multiplicative and additive biases corresponding to the jth <sup>fi</sup>lter, respectively, f represents the activation function, and “down” represents the subsampling function.

2.2.3. Deconvolution Layer. The deconvolution layer recovers the number of features extracted through the convolution and pooling layers to the resolution of the original feature. It is advantageous in that it can be used to perform upsampling at the same time as the training process, and it can be used to minimize the loss resulting from feature resampling.<sup>17</sup>

2.2.4. Training. After obtaining the reconstructed $\hat { \mathbf { X } }$ from the output layer, learning is performed so that the reconstruction errors for X and X̂ diminish, while the network parameters are refreshed through the backpropagation method, as shown in the following equation:

$$
\min J _ {\mathrm{CNN}} = \frac {1}{2 n} \sum_ {i = 1} ^ {n} | | \hat {\mathbf {x}} _ {i} - \mathbf {x} _ {i} | | ^ {2}\tag{7}
$$

2.2.5. Gradient-Weighted Class Activation Mapping. Grad-CAM is a method that enables the visual explanation of CNN-based prediction results using the gradient information that <sup>fl</sup>ows into the last convolutional layer of a CNN. The CNNs used in image analysis comprise a feature extraction part that stacks convolutional and pooling layers over multiple layers and an identi<sup>fi</sup>cation part that receives the feature quantity output and matches it with a class label to perform supervised learning. The identi<sup>fi</sup>cation component typically comprises a fully connected multilayer neural network, and the <sup>fi</sup>nal layer is used to convert the feature quantity into a probability score for each class.

Grad-CAM is used to identify image locations with a signi<sup>fi</sup>cant e<sup>f</sup>ect on the probability score for each class by averaging the changes (derivative coe<sup>fi</sup>cients) that occur in the probability scores when an insigni<sup>fi</sup>cant change is applied to an image location.

First, by applying the formula presented below, the gradient ${ \partial y _ { c } } / { \partial A _ { i j } ^ { k } } { \circ \dot { \mathrm { f } } }$ the intensity $A _ { i j } ^ { k } \mathsf { a t }$ the $( i , \ j )$ pixel of the kth convolutional feature map is calculated using the probability score $y _ { c }$ of class c. By averaging these values for all pixels, the weighting factor α<sup>c</sup>for the kth <sup>fi</sup>lter of class c can be computed. $\mathbf { A }$ larger α<sup>c</sup>value indicates the increased importance of the feature map $A ^ { k }$ for class $c ,$ as follows:

$$
\alpha_ {k} ^ {c} = \frac {1}{Z} \sum_ {i} \sum_ {j} \frac {\partial y _ {c}}{\partial A _ {i j} ^ {k}}\tag{8}
$$

where $Z$ indicates the number of channels. A heat map of a size that is similar to that of the convolutional feature map is generated by calculating the weighted average of k <sup>fi</sup>lters using the calculated value of $\alpha _ { k } ^ { c }$ after which the output of the ReLU function is obtained.

$$
\mathbf {L} _ {\mathrm{Grad-CAM}} ^ {c} = \operatorname{ReLU} \left(\sum_ {k} \alpha_ {k} ^ {c} \mathbf {A} ^ {k}\right)\tag{9}
$$

Overlaying onto the input data is possible by resizing $\mathbf { L } _ { \mathrm { G r a d - C A M } } ^ { c } .$

2.3. Proposed DDD. A CNN can be used to assess the correlation between adjacent elements of the input tensor. However, accurate feature extraction is not possible for unordered process data, even when convolution is performed using a general $3 \times 3$ <sup>fi</sup>lter. Accordingly, in this study, only the temporal characteristics of each process variable are initially extracted through the CNN, meaning that the order of process variables does not matter. The DAE is then connected to extract the nonlinearity between process variables. Figure 2 shows an outline of DDD.

To evaluate the process dynamics, the components of the input sample are converted into m $\times \left( n + 1 \right)$ (m represents the number of input variables, and n represents the number of time delay variables). For temporal feature extraction, the sample is <sup>fi</sup>rst input into a hidden layer comprising a convolutional layer and pooling layer. Afterward, multidimensional data are converted into one-dimensional data via a fully connected layer. The one-dimensional data are then input into the hidden layer of the DAE to realize the connection between the CNN and the DAE. The number of neurons in the middle layer of the DAE is compressed such that it is at least smaller than m in the input layer. The data are then reconstructed through the decoder and the deconvolution layer. For the loss function $L ,$ the model is trained; therefore, the reconstruction errors at the input layer level are insigni<sup>fi</sup>cant.

$$
L = \frac {1}{m} \sum_ {i = 1} ^ {m} | | \hat {x} _ {i} - x _ {i} | | ^ {2}\tag{10}
$$

In DDD, it is necessary to select the number of convolutional layers, the number of <sup>fi</sup>lters, the number of hidden layers in the $\mathrm { A E } ,$ and the number of neurons in each hidden layer. The temporal midpoints of the training data used in this study are regarded as the validation data, and the <sup>fi</sup>nal model was constructed using a combination of hyperparameters with the smallest L of the temporal midpoints.

DDD detects process faults using two statistics, namely, as $T ^ { 2 }$ statistics and Q statistics, similar to the PCA-based MSPC method. The $T ^ { 2 }$ statistic is calculated using the square of the distance obtained from the origin of the standardized number of the hidden layer neurons, as follows:

$$
T ^ {2} = \sum_ {c = 1} ^ {d} \left(\frac {t _ {c}}{\sigma_ {c}}\right) ^ {2}\tag{11}
$$

where $d$ represents the number of neurons in the middle layer, $t _ { c }$ represents the value of the cth neuron, and $\sigma _ { c }$ represents the standard deviation of the cth neuron. The Q statistic is obtained from the reconstruction error between the input and output layers, as follows:

$$
Q = \| x _ {i} - \hat {x} _ {i} \| ^ {2}\tag{12}
$$

Each threshold $\tau _ { T } { } ^ { 2 }$ or $\tau _ { Q }$ was set to a value containing 99.7% of the total $T ^ { 2 }$ and $Q$ values calculated using the training data. 99.7% is a value that is based on the 3σ method. The resulting model was used to determine whether the new data were abnormal. If the $T ^ { 2 } \left( T _ { \mathrm { t e s t } } ^ { 2 } \right)$ and Q statistics obtained by inputting the new data into the model $( Q _ { \mathrm { t e s t } } )$ satisfy either $T _ { \mathrm { t e s t } } ^ { \mathrm { ~ \tiny ~ { ~ 2 ~ } ~ } } > \bar { \tau } _ { T ^ { \mathrm { ~ \tiny ~ { ~ 2 ~ } ~ } } }$ or $Q _ { \mathrm { t e s t } } > \tau _ { Q }$ the new data are considered abnormal, and the other data are considered normal. The process variables related to such abnormalities are searched when an abnormal condition is detected through the monitoring process. For the $T ^ { 2 }$ statistic, the weight of the ith input variable in the cth neuron of the hidden layer is represented using $w _ { i } ^ { c } ,$ and the contribution of the ith input variable to the $T ^ { 2 }$ statistic is de<sup>fi</sup>ned as follows:

$$
C _ {T ^ {2}, i} = \sum_ {c = 1} ^ {d} \left(\frac {t _ {c}}{\sigma_ {c}}\right) ^ {2 ^ {*}} w _ {i} ^ {c}\tag{13}
$$

The contribution of the ith input variable of the Q statistic is de<sup>fi</sup>ned as follows:

$$
C _ {Q, i} = (x _ {i} - \hat {x} _ {i}) ^ {2}\tag{14}
$$

## 3. RESULTS AND DISCUSSION

The TEP dataset 18 was used to verify the e<sup>f</sup>ectiveness of DDD. Eastman Chemical Company developed the TEP dataset to mimic an actual industrial process, and this dataset has been used to evaluate the performance of various methods for process control and monitoring. The TEP dataset comprises <sup>fi</sup>ve main units, namely, a reactor, stripper, condenser, recycle compressor, and separator, with a total of eight components $( \mathrm { A }$ through H). The liquid products, G and ${ \mathrm { H } } ,$ and the by-product, $\mathrm { F } ,$ are generated from the gaseous reactants $\begin{array} { r } { \mathbf { A } , \mathbf { C } , \mathbf { D } , } \end{array}$ and E through chemical reactions. The process is described in detail in the study by Downs and Vogel.<sup>18</sup> The TEP dataset incorporates a total of 52 variables, which include 22 process variables, 11 instrumental variables, and 19 component analysis result variables. In this study, only 22 process measurement variables were used because the process variables are a<sup>f</sup>ected because of manipulation. Each process variable employed herein is listed in Table S1. The values for these process variables were measured every 3 min. The training data comprised 1500 min of normal data (500 samples), and the test data comprised 2880 min of data (960 samples) in which 21 types of process faults listed in Table S2 occurred. These datasets and control structures are similar to those previously reported in the literature.<sup>19</sup> In each of 21 types of test data, a process fault occurred after 480 min (160 samples). The models are constructed using the training data, which include only the normal data.

To consider the process dynamics, the samples inputted into DDD were transformed into an m × (n + 1) matrix, where m represents the number of input variables, and n represents the number of time delay variables. In this study, m was set as 22 for the number of input variables, and n was set as 22, based on the study by Krizhevsky et $a l . ^ { 2 \acute { 0 } }$ Data were preprocessed by range-scaling at a range of $_ { 0 - 1 }$ for each process variable.

The false negative rate (FNR) (%) and false alarm rate (FAR) (%) were used to evaluate the performance of process fault detection using DDD, as shown below:

$$
\mathrm{FNR} (\%) = \frac {\mathrm{FN}}{\mathrm{TP} + \mathrm{FN}} \times 1 0 0\tag{15}
$$

$$
\mathrm{FAR} (\%) = \frac {\mathrm{FP}}{\mathrm{FP+TN}} \times 1 0 0\tag{16}
$$

where TP represents the number of samples that are normal when the model is also normal, FN represents the number of samples that are normal when the model is abnormal, and TN represents samples that are actually abnormal when the model is abnormal. FP represents the number of samples considered normal by the model when the samples are actually abnormal, FNR represents the proportion of classes that are considered abnormal among the actual normal samples, and FAR represents the proportion of classes that are considered normal among the samples that are actually abnormal. Performance improves with a decrease in either FNR or FAR. The capability to detect process faults was determined to be inferior to random estimation if both the FNR and FAR were ≥50%.

The DAE and CNN were used as comparison methods. Table 1 shows the hyperparameters required for each method.

Table 1. Hyperparameters for Each Method

<table><tr><td>method</td><td>hyperparameter</td></tr><tr><td>DAE</td><td> $l_{\text{AE}}$ , s</td></tr><tr><td>CNN</td><td> $l_{\text{conv}}$ , f</td></tr><tr><td>DDD</td><td> $l_{\text{conv}}$ , f,  $l_{\text{AE}}$ , s</td></tr></table>

$l _ { \mathrm { A E } }$ represents the number of hidden layers, $\mathbf { s } \epsilon R ^ { l _ { \mathrm { A E } } }$ represents the rate of reduction from the number of neurons in the previous layer, and s is required only for determining the $l _ { \mathrm { A E } }$ quantity. $l _ { \mathrm { c o n v } }$ represents the number of convolutional layers, and $\mathbf { f } \epsilon R ^ { l _ { \mathrm { { c o n v } } } }$ represents the number of <sup>fi</sup>lters. f is required only for determining the $l _ { \mathrm { c o n v } }$ quantity. Table 2 displays the candidates for each hyperparameter, and Table 3 shows the combination of hyperparameters with the minimum loss function using the median data from the training dataset.

Table 2. Candidates for Hyperparameters

<table><tr><td>hyperparameter</td><td>candidates</td></tr><tr><td> $l_{\text{AE}}$ </td><td>1, 2, 3, 4</td></tr><tr><td>s</td><td>1/2, 1/3</td></tr><tr><td> $l_{\text{conv}}$ </td><td>1, 2, 3, 4</td></tr><tr><td>f</td><td>8, 16, 32, 64</td></tr></table>

![](images/8c32b02444f0e24f20703fa28111ca9ee7337a1d0d0349444930a394d4dca242.jpg)

Table 3. Optimized Hyperparameter Values for Each Method

<table><tr><td>method</td><td> $l_{AE}$ </td><td>s</td><td> $l_{conv}$ </td><td>f</td></tr><tr><td>DAE</td><td>4</td><td>1/2, 1/3, 1/3, 1/3</td><td></td><td></td></tr><tr><td>CNN</td><td></td><td></td><td>4</td><td>64, 64, 64, 64</td></tr><tr><td>DDD</td><td>3</td><td>1/3, 1/3, 1/3</td><td>4</td><td>8, 8, 16, 16</td></tr></table>

A model using the hyperparameter values listed in Table 3 was constructed, and 21 process faults were detected. Table 4

Table 4. FNR and FAR Results for Each Method in 21 Process Faults

<table><tr><td rowspan="2"></td><td colspan="2">DAE</td><td colspan="2">CNN</td><td colspan="2">DDD</td></tr><tr><td>FNR</td><td>FAR</td><td>FNR</td><td>FAR</td><td>FNR</td><td>FAR</td></tr><tr><td>1</td><td>15.6</td><td>0.4</td><td>4.4</td><td>0.4</td><td>24.4</td><td>0.4</td></tr><tr><td>2</td><td>8.1</td><td>1.4</td><td>4.4</td><td>1.8</td><td>4.4</td><td>1.5</td></tr><tr><td>3</td><td>44.4</td><td>51.9</td><td>26.9</td><td>54.8</td><td>10.0</td><td>72.5</td></tr><tr><td>4</td><td>11.3</td><td>73.6</td><td>4.4</td><td>76.8</td><td>9.4</td><td>72.8</td></tr><tr><td>5</td><td>11.3</td><td>48.9</td><td>4.4</td><td>56.5</td><td>9.4</td><td>54.3</td></tr><tr><td>6</td><td>6.3</td><td>0.1</td><td>1.3</td><td>0.0</td><td>7.5</td><td>0.1</td></tr><tr><td>7</td><td>12.5</td><td>32.4</td><td>8.1</td><td>37.4</td><td>6.3</td><td>35.9</td></tr><tr><td>8</td><td>19.4</td><td>0.9</td><td>3.8</td><td>0.0</td><td>2.5</td><td>1.3</td></tr><tr><td>9</td><td>48.1</td><td>52.9</td><td>33.1</td><td>57.1</td><td>31.3</td><td>63.5</td></tr><tr><td>10</td><td>11.9</td><td>14.8</td><td>16.3</td><td>14.1</td><td>9.4</td><td>11.8</td></tr><tr><td>11</td><td>17.5</td><td>1.4</td><td>20.0</td><td>0.6</td><td>24.4</td><td>0.4</td></tr><tr><td>12</td><td>25.0</td><td>0.0</td><td>17.5</td><td>0.0</td><td>7.5</td><td>0.0</td></tr><tr><td>13</td><td>8.1</td><td>4.8</td><td>1.3</td><td>5.3</td><td>6.9</td><td>3.9</td></tr><tr><td>14</td><td>18.8</td><td>0.1</td><td>8.1</td><td>0.0</td><td>21.3</td><td>0.1</td></tr><tr><td>15</td><td>6.9</td><td>61.6</td><td>1.3</td><td>60.9</td><td>9.4</td><td>65.4</td></tr><tr><td>16</td><td>71.3</td><td>10.5</td><td>61.3</td><td>11.5</td><td>63.8</td><td>13.8</td></tr><tr><td>17</td><td>24.4</td><td>2.3</td><td>19.4</td><td>2.5</td><td>25.6</td><td>1.1</td></tr><tr><td>18</td><td>11.9</td><td>6.4</td><td>13.8</td><td>7.4</td><td>8.1</td><td>6.0</td></tr><tr><td>19</td><td>8.1</td><td>3.8</td><td>3.1</td><td>1.8</td><td>1.9</td><td>1.5</td></tr><tr><td>20</td><td>10.6</td><td>12.4</td><td>0.0</td><td>13.6</td><td>2.5</td><td>13.9</td></tr><tr><td>21</td><td>26.9</td><td>27.8</td><td>27.5</td><td>41.5</td><td>10.6</td><td>29.9</td></tr></table>

shows the results of abnormality detection using DAE, CNN, and DDD. In this study, methods that exceeded 50% in terms of either FNR or FAR were not used in performance comparison. Regarding process faults $3 , 4 , 9 , 1 5 ,$ and $^ { 1 6 , }$ it was con<sup>fi</sup>rmed that no process fault was detected because $\mathrm { F A R } ,$ FNR, or both exceeded 50% using all methods. DDD exhibited the most favorable FAR among seven process faults, followed by the DAE and the CNN with six and four process faults, respectively. The CNN demonstrated the most favorable FNR among nine process faults, followed by DDD with eight process faults.

Figure 3 shows the delay time from 480 min after an abnormality occurs until the point the abnormality is detected. The delay time was con<sup>fi</sup>rmed to be insigni<sup>fi</sup>cant for DDD entirely. The average delay times for DAE, CNN, and DDD were 22.9, 41.7, and 22.4 min, respectively, and DDD exhibited the highest average speed in the detection of process faults.

As examples of detected process faults, Figures 4−7 show the time plots for each statistic corresponding to process faults 2, 6, 13, and 19, respectively. The black horizontal line represents the threshold value, which signals an abnormality when the threshold value is exceeded. The time presented on the horizontal axis represents the time at which an abnormality is detected using each statistic. According to Figures 4−7, the time plots of the $Q$ statistics for DAE, CNN, and DDD are similar. By contrast, the CNN cannot calculate the $T ^ { 2 }$ statistic, the DAE indicated subthreshold values in all cases, and neither could detect process faults. Despite this result, it was con<sup>fi</sup>rmed (see Figures 4−6) that DDD could accurately detect process faults with respect to the $T ^ { 2 }$ statistic. For process fault 13, as shown in Figure $^ { 6 , }$ process faults could be detected earlier compared with using the Q statistic. For process fault $^ { 1 9 , }$ as shown in Figure 7, DDD could detect process faults accurately based on the Q statistic. Therefore, it was con<sup>fi</sup>rmed that DDD can be used to detect abnormalities accurately.

![](images/4185126d2a7441aa45e32a6c39968a170099fc0a76509167b6087fe83ca4a8ac.jpg)  
(a)

![](images/0423f3f91df68ae72027054c78b946c8f36cb6d3fda96d1cf28fbf2a933c9e9a.jpg)  
(b)

![](images/6a6d69c8eed0ef755dfc9f1f1fe96c2e74cc75fbb053a8a7334c8e68ef1bd0e2.jpg)  
(c)

Figure 3. Expected fault detection delay: (a) DAE, (b) CNN, and (c) DDD.  
![](images/4ed60ae15aefe749141838a8dafe4c68d4f90b8d49236ff64d46cab2629c1f15.jpg)  
(a)

![](images/726a3273b86e3dc26a2ef44ce5663f1d3f0c927c9b4373d13bc9b9e0537ce06d.jpg)  
(b)  
(c)  
Figure 4. Time plot of each statistic for each method in process fault 2. Horizontal straight lines indicate the thresholds, and values in the x-axis indicate the fault detection times. (a) DAE, (b) CNN, and (c) DDD.

![](images/bacc7e4d8ca046c466d63eea333798f2e9d8bc44dd04a921eb46ffb59ad3bc2c.jpg)

![](images/34695485f36de7ad1662910bea5b8445959fe2d1a830c40e08c1ead00d450db1.jpg)  
(a)

![](images/201f77d32d909ecb5e0d84c8da44df2bbbaecff206138b91ab74f4fa829b4e0d.jpg)  
(b)

![](images/08210c574c9a346fa31b96685c23355478ffdda4115321f88d7437090c57c5e6.jpg)

![](images/9650cd8027c94df31723f2b22963aa366a598c4b34d536a35e6455b03bcdcbe7.jpg)  
(c)

Figure 5. Time plot of each statistic for each method in process fault 6. Horizontal straight lines indicate the thresholds, and values in the x-axis indicate the fault detection times. (a) DAE, (b) CNN, and (c) DDD.  
![](images/61bc474052012412df130ae6556a3f4e6bc5d0e39337f75e1a9377d1882899bc.jpg)

![](images/8c951df5eba42cad09ac4dad124652bcacf36850fb9e95531df5e2a27c2107e0.jpg)  
(a)

![](images/a6f4b64885ddb91755de2114e0bcbc78246d704ce050b1bc4daefc8575751f6c.jpg)  
(b)

![](images/409e8ce1bea280ae5ce88d5d98ed7de8451ceb26701ac4cb3b7c35d4a59c77a0.jpg)

![](images/34201ac62d48b012493803e15aec5a7f8287109a981ffab76fa0c929bbf1879a.jpg)  
(c)  
Figure 6. Time plot of each statistic for each method in process fault 13. Horizontal straight lines indicate the thresholds, and values in the x-axis indicate the fault detection times. (a) DAE, (b) CNN, and (c) DDD.

Subsequently, the process variables related to the abnormalities were identi<sup>fi</sup>ed. For the DDD, the diagnoses of process faults $1 , ~ 2 , ~ 6 ,$ and 13 are outlined in Figures $^ { 8 - 1 1 , }$ respectively. The contribution of each process variable to the process fault can be calculated using the $T ^ { 2 }$ and Q statistics. It was con<sup>fi</sup>rmed that DDD can be used to diagnose abnormalities of process variables using the $T ^ { 2 }$ and $Q$ statistics, which cannot otherwise be conducted using conventional deep learning-based techniques. Process variable 1 showed that DDD contributed highly to the Q statistic in process fault $^ { 1 , }$ which is presented in Figure $^ { 8 , }$ and an abnormality in the supply <sup>fl</sup>ow rate of raw material A was identi<sup>fi</sup>ed. DDD’s $T ^ { 2 }$ statistic further contributed to the process fault detection of process variable 20, thereby suggesting that the compressor

![](images/4fbdde1b47a995fe838da88ce5139a720d97650432e5853dff8b736bdd64ef64.jpg)

![](images/f70d68cde59be555ed560aa6ca438102928b304c8ccc5334603d426b12a653de.jpg)  
(a)

![](images/6d795847f8103ca117596a34412f63aa71942525bf9523a40c8171aaf38181c4.jpg)  
(b)

![](images/2905b6c6964c14da41d6a1cb154e1df8c5abab8dac758a685473b0197a5e99c0.jpg)

![](images/a0821152e9a1139860be1103dfa1561df536a5dce0bf6d5da97051d3af42d6d4.jpg)  
(c)

Figure 7. Time plot of each statistic for each method in process fault 19. Horizontal straight lines indicate the thresholds, and values in the x-axis indicate the fault detection times. (a) DAE, (b) CNN, and (c) DDD.  
![](images/cca335cce1ae353ee76e2eecc145117f39e10833115d8ef645b384a0dcd98bc9.jpg)

![](images/4d039149d5053e6f856e4b8a3c23fa34ae607de2f03c3be431761222466f1736.jpg)

![](images/cf5654a184c74f739e264d3297f60614665a3efa391e894ca1d76f99f55b34ba.jpg)  
Figure 8. Process fault diagnosis results of DDD in process fault 1.

![](images/462a72113622c0e10dcf5bb7c51e0735ce9a3bf9f2a462b1e8034a9ad42a4441.jpg)

![](images/d2d8734b18e85ff35d7bffff9d6e76baf116b0298ab936b6ae11c79887c25139.jpg)  
Figure 9. Process fault diagnosis results of DDD in process fault 2.

![](images/62aec6abd4250aaeb6f62419c9356308c7cc4cbe58dea554dd190f48693c2ef8.jpg)

failure is related to the feed rate of raw material A. Based on process fault $^ { 2 , }$ which is shown in Figure 9, DDD was used to successfully diagnose the presence of an error in the purging of process variable 10. This can be expected to cause abnormalities in the composition of the product. According to process fault $^ { 6 , }$ as shown in Figure 10, process variables $^ { 7 , }$ $^ { 1 3 , }$ and 16 have larger contributions than other process variables, and abnormalities occur in the reactor, separator, and stripper. However, according to Table ${ \mathrm { S } } 2 ,$ this diagnostic result is considered di<sup>f</sup>erent from the actual cause of the abnormality. It can be concluded that only the $T ^ { 2 }$ statistic of DDD indicates that the supply <sup>fl</sup>ow rate of raw material A with respect to process variable 1 is abnormal, and thus, it can contribute to the identi<sup>fi</sup>cation of the root cause of process fault 6. Regarding process fault $^ { 1 3 , }$ which is presented in Figure 11, the abnormal pressure levels in the reactor, separator, and stripper were diagnosed because of the drift in the reaction rate constant. Moreover, DDD was used to successfully determine that there was a substantial contribution by the <sup>fl</sup>ow rate of raw material D. Therefore, DDD can be used to increase the information used to identify the causes of process faults by digitizing the degree of in<sup>fl</sup>uence on the process faults resulting from the high-dimensional feature quantities expressed through deep learning, and it can contribute to the detection of the causes of process faults that cannot be con<sup>fi</sup>rmed using conventional methods.

Figure 10. Process fault diagnosis results of DDD in process fault 6.  
![](images/9d70216deffe507097163b6721f076d811468774957feb05600082caea8c9b6a.jpg)

![](images/5b6736dfdaaf33dddd42bed95b9142e8263dc55629e092655d70ef27063421cf.jpg)  
Figure 11. Process fault diagnosis results of DDD in process fault 13.

## 4. CONCLUSIONS

In this study, a deep convolutional neural network with deconvolution and a deep autoencoder (DDD) was proposed for the construction of an MSPC-based deep neural network that assesses the process dynamics and the nonlinearity between process variables. DDD can be used to detect and diagnose process faults through the constructed neural network. Based on the CNN and DAE, DDD can be used to e<sup>f</sup>ectively represent the relationship between process variables hidden in process data while simultaneously accounting for the dynamic characteristics and nonlinearity of process variables. By calculating the Q and $T ^ { 2 }$ statistics using DDD, it is possible to detect process faults based on these factors, and the $T ^ { 2 }$ and Q statistics can be used to digitize the information related to the process variables that contribute to a speci<sup>fi</sup>c abnormality.

A case study using the TEP dataset was conducted to verify the e<sup>f</sup>ectiveness of DDD. DDD can be used to determine the contributions of various process variables to each process fault quantitatively. Overall, compared with conventional process fault detection methods, DDD demonstrates enhanced performance, and its implementation successfully increases the number of determining factors used for identifying the causes of process faults through its ability to e<sup>f</sup>ectively present the process variables involved in process faults. Because tensorial data can be analyzed in chemical and biological manufacturing processes,<sup>21</sup> the tensorial data that consider both the process variables and the process dynamics can be analyzed e<sup>f</sup>ectively using DDD as future research. However, it should be noted that DDD has limitations in that process data in normal states are required to construct process fault detection and diagnosis models. It is expected that the proposed approach can improve the e<sup>fi</sup>ciency of process control and management in chemical plants and industrial facilities through the detection and diagnosis of process faults.

## ASSOCIATED CONTENT

## \*sı Supporting Information

The Supporting Information is available free of charge at https://pubs.acs.org/doi/10.1021/acsomega.1c06607.

Process variables and types of process faults in the TEP dataset (PDF)

## AUTHOR INFORMATION

## Corresponding Author

Hiromasa Kaneko − Department of Applied Chemistry, School of Science and Technology, Meiji University, Kawasaki, Kanagawa 214-8571, Japan; orcid.org/0000- 0001-8367-6476; Email: hkaneko@meiji.ac.jp

## Author

Yasuhiro Kanno − Department of Applied Chemistry, School of Science and Technology, Meiji University, Kawasaki, Kanagawa 214-8571, Japan

Complete contact information is available at: https://pubs.acs.org/10.1021/acsomega.1c06607

## Notes

The authors declare no competing <sup>fi</sup>nancial interest. The data that support the <sup>fi</sup>ndings of this study are available in the work of ref 19.

## ACKNOWLEDGMENTS

This study was supported by the Grant-in-Aid for Scienti<sup>fi</sup>c Research (KAKENHI) (grant number 19K15352) from the Japan Society for the Promotion of Science.

## REFERENCES

(1) Venkatasubramanian, V.; Rengaswamy, R.; Yin, K.; Kavuri, S. N. A review of process fault detection and diagnosis: Part I: Quantitative model-based methods. Comput. Chem. Eng. 2013, 27, 293−311.

(2) Deng, X.; Tian, X.; Chen, S.; Harris, C. J. Deep learning based nonlinear principal component analysis for industrial process fault detection. In 2017 International Joint Conference on Neural Networks 2017, 1237−1243.

(3) Wise, B. M.; Ricker, N. L.; Veltkamp, D. F.; Kowalski, B. R. A theoretical basis for the use of principle component models for monitoring multivariate processes. Process Control Qual. 1990, 1, 41− 51.

(4) Kano, M.; Tanaka, S.; Hasebe, S.; Hashimoto, I.; Ohno, H. Monitoring independent components for fault detection. AIChE J. 2003, 49, 969−976.

(5) MacGregor, J. F.; Jaeckle, C.; Kiparissides, C.; Koutoudi, M. Process monitoring and diagnosis by multiblock PLS methods. AIChE J. 1994, 40, 826−838.

(6) Watanabe, K.; Matsuura, I.; Abe, M.; Kubota, M.; Himmelblau, D. M. Incipient fault diagnosis of chemical processes via artificial neural networks. AIChE J. 1989, 35, 1803−1812.

(7) Peng, Y.; Chen, X.; Ye, Q.; Jiao, J. Fault detection and classification in chemical processes using NMFSC and structural SVMs. Can. J. Chem. Eng. 2014, 92, 1016−1023.

(8) Hinton, G. E.; Salakhutdinov, R. R. Reducing the dimensionality of data with neural networks. Science 2006, 313, 504−507.

(9) Dong, C.; Loy, C. C.; He, K.; Tang, X. Image Super-Resolution Using Deep Convolutional Networks. IEEE Trans. Pattern Anal. Mach. Int. 2016, 38, 295−307.

(10) Wu, H.; Zhao, J. Deep convolutional neural network model based chemical process fault diagnosis. Comput. Chem. Eng. 2018, 115, 185−197.

(11) Wen, L.; Li, X.; Gao, L.; Zhang, Y. A new convolutional neural network-based data-driven fault diagnosis method. IEEE Trans. Ind. Electron. 2018, 65, 5990−5998.

(12) Xu, G.; Liu, M.; Jiang, Z.; Söffker, D.; Shen, W. Bearing fault diagnosis method based on deep convolutional neural network and random forest ensemble learning. Sensors 2019, 19, 1018.

(13) Jiang, G.; Xie, P.; He, H.; Yan, J. Wind turbine fault detection using a denoising autoencoder with temporal information. IEEE/ ASME Trans. Mechatron. 2018, 23, 89−100.

(14) Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; Batra, D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. IEEE Int. Conf. Comput. Vision 2017, 2017, 618−626.

(15) Hansen, M. F.; Smith, M. L.; Smith, L. N.; Smith, N.; Salter, M. G.; Baxter, E. M.; Farish, M.; Grieve, B. Towards on-farm pig face recognition using convolutional neural networks. Comput. Ind. 2018, 98, 145−152.

(16) Yang, C.; Rangarajan, A.; Ranka, S. Visual explanations from deep 3D convolutional neural networks for Alzheimer’s disease classification. AMIA2018 2018, 1571−1580.

(17) Zhao, H.; Liu, H.; Hu, W.; Yan, X. Anomaly detection and fault analysis of wind turbine components based on deep learning network. Renewable Energy 2018, 127, 825−834.

(18) Downs, J. J.; Vogel, E. F. A plant-wide industrial process control problem. Comput. Chem. Eng. 1993, 17, 245−255.

(19) Russell, E. L.; Chiang, L. H.; Braatz, R. D. Fault detection in industrial processes using canonical variate analysis and dynamic principal component analysis. Chemom. Intell. Lab. Syst. 2000, 51, 81− 93.

(20) Krizhevsky, A.; Sutskever, I.; Hinton, G. E. ImageNet Classification with Deep Convolutional Neural Networks. Commun. ACM 2017, 60, 84−90.

(21) Sun, W.; Braatz, R. D. Opportunities in tensorial data analytics for chemical and biological manufacturing processes. Comput. Chem. Eng. 2020, 143, 107099.