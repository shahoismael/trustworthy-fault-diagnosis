# Deep learning with spatiotemporal attention-based LSTM for industrial soft sensor model development

Xiaofeng Yuan, Lin Li, Yuri A. W. Shardt, Yalin Wang, Chunhua Yang

Abstract—Industrial process data are naturally complex time series with high nonlinearities and dynamics. To model nonlinear dynamic processes, a long short-term memory (LSTM) network is very suitable for soft sensor model development. However, the original LSTM does not consider variable and sample relevance for quality prediction. In order to overcome this problem, a spatiotemporal attention-based LSTM network is proposed for soft sensor modeling, which can, not only identify important input variables that are related to the quality variable at each time step, but also adaptively discover quality-related hidden states across all time steps. By taking the spatiotemporal quality-relevant interactions into consideration, the prediction performance can be improved for the soft sensor model. The effectiveness and flexibility of the proposed model is demonstrated on an industrial hydrocracking process to predict the initial boiling points of heavy naphtha and aviation kerosene.

Index Terms—Soft sensor; Quality prediction; Deep learning; Attention mechanism; Spatiotemporal attention-based LSTM (STA-LSTM).

## I. INTRODUCTION

N industrial processes, product properties are important to ensure process safety and production quality [1-4]. In many situations, offline laboratory analysis is used for some process quality variables that are difficult to obtain in real time. Although offline laboratory analysis can provide accurate measurements, the sampling cycle is often very long, and it usually leads to low sampling rates and large measurement delay. Some plants may be installed with online analytical instruments to obtain measurement data for key quality variables. However, online quality instruments are often expensive and difficult to maintain. Neither offline tests nor online sensors can meet the requirements for real-time process monitoring, control, and optimization. Recently, soft sensors have been widely used for online estimation of key product qualities thanks to their rapid response, low maintenance costs, and accurate prediction results. Soft sensors can predict the difficult-to-measure quality variables by building predictive mathematical models based on secondary process variables that are easy to measure, such as temperatures, pressures, and flowrate [5-8].

Usually, soft sensors are divided into two main categories: first principle, or white-box models, and data-driven, or black-box models. First-principle models are developed using the laws of nature, such as mass and energy balances, force balances, or reaction mechanisms [9]. On the other hand, data-driven models are developed solely using the available data without necessarily considering the physical meaning of the resulting models [9]. Given the common use of distributed control systems (DCS) and the relative availability of historical process data, data-driven soft sensors have become popular. Typical data-driven modeling methods are principal component analysis (PCA) [10, 11], partial least squares (PLS) [12, 13] and artificial neural network (ANN) [14, 15]. Of these, ANN is a common method that is widely used in soft sensor development. For example, Dam et al. [16] proposed a soft sensor based on artificial neural networks and introduced a genetic algorithm to optimize the network structure and weights. Napoli et al. [17] developed a soft sensor for predicting the atmospheric pylon aviation kerosene condensation point using self-introduction sampling, noise injection, and neural network stacking. However, shallow ANNs have limited ability to express complex functions, and its generalizability is restricted to large systems. Multilayer networks are easily affected by gradient vanishing and exploding problems. In 2006, Hinton et al. [18] proposed a deep learning technique to resolve this problem. They showed that an artificial neural network with multiple hidden layers can accurately extract features using deep learning. Also, it is possible for deep neural networks to effectively overcome the difficulty of network training through layer-wise unsupervised pretraining and supervised fine tuning.

Deep neural networks have also been introduced for soft sensor modeling because of their better performance [19-27]. However, these soft sensors are mainly based on static deep networks, such as deep belief networks (DBN) [28] and stacked auto-encoders (SAE) [29, 30], in which data samples are assumed to be independent and identically distributed (the i.i.d. assumption). Nevertheless, industrial processes have intrinsically complex nonlinear dynamic behavior due to such factors as complex physiochemical reaction mechanisms, feedback control, and dynamic noise. The data sequences are sampled from real-valued and continuous processes over time. Hence, they are naturally time sequences with highly nonlinear temporal correlations. To model such data sequences, the models have to include more data from the past steps or have memory unit of the past inputs. Recurrent neural networks (RNNs) [31], a type of dynamic neural networks, have been widely used to capture temporal dynamic behavior in time series data. However, it is difficult for standard RNNs to model long sequences, since they also suffer from the gradient vanishing problem [32]. To handle this issue, a long short-term memory (LSTM) network was proposed by Hochreiter and Schmidhuber [33], in which memory cells and three nonlinear gates are used to replace the basic activation unit in RNN. LSTM can, not only forget the useless information in the past, but also judge the current information and store useful information in the memory cell [34, 35]. Thus, LSTM networks are more effective in learning long-term temporal dependencies. This leads to its successful applications in many fields such as language modeling [36], time series prediction [37], and automatic speech recognition [38].

Although LSTM is very helpful in capturing long-term dependencies, it cannot focus on different variables at different time steps. To overcome this problem, an attention-based encoder-decoder network was proposed in [39]. Based on LSTM units, an encoder-decoder network can be constructed to resolve the sequence-to-sequence problem. With the encoder part, the input sequence can be converted into a fixed-length vector, and then the generated fixed vector is converted into an output sequence by the decoder. However, the performance of this model will decrease rapidly with the increase of the length of the input sequence. This is a common problem in industrial data sequences, since the quality variables are often predicted based on very long lagged input series. The attention-based encoder-decoder architecture seeks to differentiate between hidden states with different attention weights across all time steps in a prediction window. In recent years, attention mechanisms have performed well for many different tasks like machine translation, image classification, and natural language processing [40, 41]. The attention phenomenon is also very common in industrial processes, since data samples at previous instants always have a different impact on prediction of the current data. However, this is rarely considered in the previous process data modeling approaches. In this paper, the attention mechanism is used for dynamic modeling of the industrial processes. Moreover, most existing attention-based models mainly consider the temporal dynamics and correlations between data samples. The impact of the input variables on the quality prediction is not considered in the existing attention mechanism. For industrial processes, the secondary process variables often have different impact on the quality prediction due to the physiochemical properties of the system.

Thus, this paper proposes a new spatiotemporal, attention-based LSTM (STA-LSTM) method for application to industrial soft sensor modeling. In this approach, the attention mechanism seeks to obtain the spatial correlation between the input and target quality variables. Then, the temporal attention mechanism seeks to model the time dynamic behavior for the final prediction. This allows STA-LSTM to not only adaptively identify the input features, but also handle dynamic behavior. Finally, the proposed STA-LSTM method will be applied to an industrial hydrocracking process to predict the initial boiling points of heavy naphtha and aviation kerosene.

## II. ENCODER-DECODER ARCHITECTURE AND THE ATTENTION MECHANISM

An encoder-decoder is a common framework in deep learning. It was proposed by Sutskever [42] to solve the problem of sequence-to-sequence modeling. For the encoder, the input sequence is first transformed to the hidden state sequence. Then, the hidden state sequence is converted into a fixed-length vector. After that, the previously generated fixed vector is used to predict a target output sequence by the decoder. The structure of the encoder-decoder architecture is shown in Fig. 1. The basic encoder and decoder units can be any models, such as SAE, RNN or LSTM. Moreover, the encoder and decoder units can be different from each other.

![](images/0963fa8ca0c644648e9b0964938930156a58b117466b2eec5568e13aef45765d.jpg)  
Fig. 1. The structure of an encoder-decoder

The encoder-decoder network is often carried out in a sliding window manner. Assume the input and output sequences are respectively $\left\{ x _ { ( 1 ) } , x _ { ( 2 ) } , . . . , x _ { ( T ) } \right\}$ and $\left\{ y _ { ( 1 ) } , y _ { ( 2 ) } , . . . , y _ { ( l ) } \right\}$ in a given window. For soft sensor application, we often have T $= l .$ However, they can be different for other modeling tasks like machine translation, in which the length of the output sequences is not known a priori.

For the encoder-decoder architecture, the input sequences are first encoded to the hidden series as $\left\{ h _ { _ { ( 1 ) } } , h _ { _ { ( 2 ) } } , . . . , h _ { _ { ( T ) } } \right\}$ . The feature representation $h _ { ( t ) }$ is progressively learned from $x _ { ( t ) }$ and $h _ { ( t - 1 ) }$ as

$$
h _ {(t)} = f (x _ {(t)}, h _ {(t - 1)})\tag{1}
$$

where $f ( \cdot )$ could be any nonlinear activation function like the sigmoid or tanh function. As can be seen, the hidden state series $\left\{ h _ { _ { ( 1 ) } } , h _ { _ { ( 2 ) } } , . . . , h _ { _ { ( T ) } } \right\}$ are dynamic features extracted from the input sequence. Then, they may be directly used to serve as the inputs for the decoder to estimate the output sequence of $y _ { ( t ) }$ with $h _ { ( t ) }$ if the lengths are the same for the input and output sequences. However, it is difficult to directly apply this method when the input and output sequences have different lengths with complicated and nonmonotonic relationships. In this way, a simple strategy is to further transfer the hidden state sequence into a fixed-sized context vector. In this way, the context vector can be generated from the hidden state sequence $\left\{ h _ { _ { ( 1 ) } } , h _ { _ { ( 2 ) } } , . . . , h _ { _ { ( T ) } } \right\}$ as

$$
c = f _ {1} (h _ {(1)}, h _ {(2)},..., h _ {(T)})\tag{2}
$$

where $f _ { 1 } ( \cdot )$ represents a mapping function. In this way, the context vector includes the information across the whole input sequence, which can be regarded as a deeper abstract of the input sequence. Then, the content vector can be used as the input for the decoder to predict the output sequence. Once the context vector is obtained, the output at time t can be progressively predicted by its previous output sequence $\left\{ y _ { ( 1 ) } , y _ { ( 2 ) } , . . . , y _ { ( t - 1 ) } \right\}$ and the vector c as

$$
\hat {y} _ {(t)} = f _ {2} (c, y _ {(1)},..., y _ {(t - 1)}) \quad 1 \leq t \leq l\tag{3}
$$

where $f _ { 2 } ( \cdot )$ is also a nonlinear activation function.

However, with the increase in the length of the input sequence, the context vector will lose long-term input information about the past, which leads to a decrease in the prediction performance [43]. To avoid this problem, the attention mechanism is introduced to the encoder-decoder framework. The structure of the attention-based encoder-decoder model is shown in Fig. 2. For the attention mechanism, a distinct vector $c _ { ( t ) }$ rather than a fixed c is designed to predict each output sample $y _ { ( t ) }$ at all instants. For prediction at sampling step t, the attention mechanism assigns an individual attention weight $w _ { i }$ to each encoded hidden feature state $h _ { ( i ) }$ according to its relationship with the previous hidden state in the decoder. Thus, the context vector changes for different predicted output instants. The context vector at time t can be computed as

$$
c _ {(t)} = \sum_ {i = 1} ^ {T} w _ {i} h _ {(i)}\tag{4}
$$

where $w _ { i }$ is the attention weight of the $i ^ { \mathrm { { t h } } }$ hidden state $h _ { ( i ) }$ at time i.

![](images/a598a8cf891c1c47224095c8952eaf977e0a16db5beb05228de8ab8f663ddc78.jpg)  
Fig. 2. The structure of an attention-based encoder-decoder

## III. SPATIOTEMPORAL ATTENTION-BASED LONG SHORT-TERM MEMORY NETWORK (STA-LSTM)

In this section, the spatiotemporal attention-based LSTM network is developed for soft sensor modeling. Following the encoder-decoder architecture, two kinds of attention mechanisms are used in the proposed STA-LSTM model. In the encoder, a spatial variable attention mechanism is introduced to selectively distinguish input variables related to quality prediction at each time step. Moreover, different spatial attention values are assigned to the input variables. Then, the variable attention-weighted sample data becomes the new input to the encoder LSTM. After that, the encoder LSTM network is used to learn the hidden states of the new weighted inputs. At the second step, a temporal attention mechanism is introduced to adaptively find out the encoder hidden states related to the quality prediction across different time steps. The adaptive context vector is the weighted sum over the products of the encoder hidden states and their corresponding temporal attention values. Finally, with the adaptive context vector as the input, the output is predicted by another decoder LSTM. Fig. 3 illustrates the proposed model. Detailed steps are described below.

## A. Spatial attention

Since it is necessary to capture the long-term dependencies in the industrial data time series, LSTM units are used as the basic activation function units in the proposed STA-LSTM model. For the given input sequence $\left\{ x _ { ( 1 ) } , x _ { ( 2 ) } , \cdots , x _ { ( T ) } \right\}$ in a subwindow, assume each sample has input variables with $\boldsymbol { x } _ { ( t ) } = ( x _ { ( t ) } ^ { 1 } , x _ { ( t ) } ^ { 2 } , \cdots , x _ { ( t ) } ^ { n } )$ . We can obtain the relationship between each input variable and the quality variable by referring to the previous decoder hidden state, which can be calculated by a metric similarity between the current original input $x _ { ( t ) }$ and the previous hidden state $s _ { ( t - 1 ) }$ in the decoder LSTM. The spatial attention architecture can be a multilayer perceptron network. Usually, a two-layer network is used to obtain the variable spatial attention as

$$
e _ {(t)} ^ {i} = V _ {1} ^ {i} \tanh (W _ {1} ^ {i} s _ {(t - 1)} + U _ {1} ^ {i} x _ {(t)} ^ {i} + b _ {1} ^ {i}) \quad 1 \leq i \leq n\tag{5}
$$

$$
\alpha_ {(t)} ^ {i} = \frac {\left| e _ {(t)} ^ {i} \right|}{\sum_ {j = 1} ^ {n} \left| e _ {(t)} ^ {j} \right|}, \quad 1 \leq i \leq n\tag{6}
$$

where $V _ { 1 } ^ { i } , W _ { 1 } ^ { i } , U _ { 1 } ^ { i }$ , and $b _ { 1 } ^ { i }$ are the parameters to be learnt, tanh is the hyperbolic activation function, and $e _ { ( t ) } ^ { i }$ is the attention value that represents the importance of the $i ^ { \mathrm { { t h } } }$ input variable at time t for quality prediction. Eq. (6) is mainly used to ensure that the attention values for all the input variables add up to 1 at each sampling instant t. The normalized value $\boldsymbol { \alpha } _ { ( t ) } ^ { i }$ is called the spatial attention weight. represents taking the absolute value. Since $\boldsymbol { S } _ { ( t - 1 ) }$ is the hidden state of the decoder LSTM, it contains information related to the quality variable. As can be seen from Fig. 3, the spatial attention architecture is usually a perceptron network, the parameters in Eq. (5) are learnt through back propagation through time (BPTT) in the training procedure of the whole network. Once the parameters of the spatial attention network are determined, the correlation between input and output can be obtained by computing a similarity calculation between the decoder hidden state $ { \boldsymbol { S } } _ { ( t - 1 ) }$ and input data $x _ { ( t ) }$ with Eq. (5). Correspondingly, the attention weights can be directly calculated with Eq. (5) and (6).

When the spatial attentions are obtained for the input variables of sample $x _ { ( t ) }$ , the spatial attention weighted sample can be expressed as

$$
\tilde {x} _ {(t)} = (\alpha_ {(t)} ^ {1} x _ {(t)} ^ {1}, \alpha_ {(t)} ^ {2} x _ {(t)} ^ {2}, \dots , \alpha_ {(t)} ^ {n} x _ {(t)} ^ {n})\tag{7}
$$

A LSTM network is then used to learn the hidden states of the encoder from the variable attention weighted input $\tilde { x } _ { ( t ) }$ . In

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS

LSTM, three gate controllers and a memory cell are placed into the basic LSTM unit, namely the input, forget, and output gates. The three gates are used to determine what information should be remembered from the weighted time series. The memory cell is used to store the input information for all time steps. The LSTM network implements temporal memory through the switch of these gates and a memory cell to prevent the gradient vanishing problem. Then, the hidden state can be written during a forward pass as

$$
f _ {(t)} = \sigma (W _ {f x} \tilde {x} _ {(t)} + W _ {f h} h _ {(t - 1)} + b _ {f})
$$

$$
i _ {(t)} = \sigma (W _ {i x} \tilde {x} _ {(t)} + W _ {i h} h _ {(t - 1)} + b _ {i})\tag{8}
$$

$$
o _ {(t)} = \sigma (W _ {o x} \tilde {x} _ {(t)} + W _ {o h} h _ {(t - 1)} + b _ {o})\tag{9}
$$

(10)

$$
\tilde {c} _ {(t)} = \tanh (W _ {c x} \tilde {x} _ {(t)} + W _ {c h} h _ {(t - 1)} + b _ {c})\tag{11}
$$

$$
m _ {(t)} = f _ {(t)} \odot m _ {(t - 1)} + i _ {(t)} \odot \tilde {c} _ {(t)}
$$

$$
h _ {(t)} = o _ {(t)} \odot \tanh (m _ {(t)})\tag{12}
$$

(13)

where $f _ { ( t ) } , i _ { ( t ) } , o _ { ( t ) } , \tilde { c } _ { ( t ) }$ represent the forget gate, input gate, output gate and intermediate state of the encoder LSTM unit; $m _ { ( t ) }$ is the corresponding memory cell; is pointwise multiplication of two vectors; $\sigma$ is the nonlinear sigmoid activation functions; and $W _ { f \ast } , W _ { o \ast } , W _ { i \ast } , W _ { c \ast }$ and $b _ { f } , b _ { i } , b _ { o } , b _ { c }$ are the parameters to be learnt. By introducing the spatial variable attention mechanism, the encoder can adaptively identify the input variables that are more related to the quality variable. After the spatial-attention-based LSTM, the hidden states are used as inputs to the temporal-attention-based LSTM.

![](images/5932d30cae1534c4c861e943377c1403f15485ddd5f05f5054cb01a8f2fa5244.jpg)  
Fig. 3. Framework of the spatiotemporal attention-based LSTM (TA: temporal attention; SA: spatial attention)

## B. Temporal attention

After the encoder, another LSTM neural network-based decoder is used to predict the quality variable $\hat { y } _ { ( t ) }$ . With the increase of the input sequence, it is difficult to retain all the necessary information for the decoder. Thus, the performance of the encoder-decoder architecture will degrade rapidly as the input sequence increases. To solve this problem, the temporal sample attention mechanism is introduced to the decoder LSTM to adaptively determine relevant hidden states generated from the encoder LSTM across all time instants, which can be measured by referring to the previous decoder hidden state. Each encoder hidden state is assigned a temporal attention value. Then, an adaptively weighted content vector is obtained as the input for the decoder LSTM. In this way, the attention mechanism breaks the limitation of the traditional encoder-decoder structure that internally relies on a fixed-length vector during encoding and decoding. As can be seen from Fig. 3, the temporal attention value of hidden state at time t can be computed as

$$
g _ {(t)} ^ {k} = V _ {2} ^ {k} \tanh (W _ {2} ^ {k} s _ {(t - 1)} + U _ {2} ^ {k} h _ {(t - T + k)} + b _ {2} ^ {k}) \quad 1 \leq k \leq T\tag{14}
$$

$$
\beta_ {(t)} ^ {k} = \frac {\left| g _ {(t)} ^ {k} \right|}{\sum_ {m = 1} ^ {T} \left| g _ {(t)} ^ {m} \right|}, \quad 1 \leq k \leq T\tag{15}
$$

where $s _ { ( t - 1 ) }$ is the previous decoder hidden state; $V _ { 2 } ^ { k } , W _ { 2 } ^ { k } , U _ { 2 } ^ { k }$ and $b _ { 2 } ^ { k }$ are the parameters to be learnt with regard to the $k ^ { \mathrm { { t h } } }$ sample in the window; $g _ { ( t ) } ^ { k }$ represents the attention value of the $k ^ { \mathrm { { t h } } }$ encoder hidden state of the subwindow for time step t; and T is the subwindow size. Eq. 15 is mainly used to ensure that the attention values for all the hidden states add up to 1. The normalized value $\boldsymbol { \beta } _ { ( t ) } ^ { k }$ is called the temporal attention weight. Then, a temporal weighted sum of all the encoder hidden state can be calculated to get the adaptive context vector

$$
c _ {(t)} = \sum_ {k = 1} ^ {T} h _ {(t - T + k)} \beta_ {(t)} ^ {k}\tag{16}
$$

Once we obtain the context vector at time t, it is combined with the given target series $\left\{ y _ { ( 1 ) } , y _ { ( 2 ) } , . . . , y _ { ( t - 1 ) } \right\}$ to update the decoder hidden state

$$
\tilde {s} _ {(t - 1)} = W _ {3} y _ {(t - 1)} + V _ {3} s _ {(t - 1)} + b _ {3}\tag{17}
$$

$$
s _ {(t)} = f _ {l} (c _ {(t)}, \tilde {s} _ {(t - 1)})\tag{18}
$$

where $W _ { 3 }$ and $V _ { 3 }$ are weight matrices; $b _ { 3 }$ is the bias vector; and $f _ { l } ( \cdot )$ is an LSTM unit. Finally, the prediction output $\hat { y } _ { ( T ) }$ is computed using

$$
\begin{array}{l} \hat {y} _ {(T)} = F (y _ {(1)}, y _ {(2)},..., y _ {(T - 1)}, x _ {(1)}, x _ {(2)},..., x _ {(T)}) \\ = V (f _ {l} ([ \tilde {s} _ {(T - 1)}; c _ {(T)} ])) + b _ {v} \end{array}\tag{19}
$$

where $\left[ \tilde { s } _ { ( T - 1 ) } ; c _ { ( T ) } \right]$ is a concatenation of the decoder hidden state at time T − 1 and the context vector $c _ { ( T ) }$ at time T; and and $b _ { \nu }$ are weight matrix and bias vectors.

For model training, the Adam algorithm is used because it is superior to the momentum gradient descent method and the root mean square back propagation (RMSProp) algorithm [44] in terms of computing time and memory requirements. The parameters of the model can be learnt by minimizing the mean squared error (MSE) using standard back propagation through time. MSE can be calculated as

$$
M S E = \frac {1}{T _ {\text { training }}} \sum_ {t = 1} ^ {T _ {\text { training }}} \left(y _ {(t)} - \hat {y} _ {(t)}\right) ^ {2}\tag{20}
$$

where $y _ { ( t ) }$ and $\hat { y } _ { ( t ) }$ are the actual and predicted quality values at sampling time t, and $T _ { t r a i n i n g }$ is the total number of training data.

Algorithm 1 gives the details of the proposed STA-LSTM method. Backpropagation through time (BPTT) is used to compute the gradient of parameters. Then, the Adam algorithm is used to update the network parameters.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Spatiotemporal Attention-based LSTM

Input: Dataset  $D = (x_{(t)}, y_{(t)}), t = 1, 2, \ldots, k$  with
    $x_{(t)} = \left\{ x_{(t)}^{1}, x_{(t)}^{2}, \ldots, x_{(t)}^{n} \right\}$ ; Learning rate  $\eta$ ; Hidden neurons N;
    Batch size B; Training epochs E; Window size T

Output: Predicted quality variable  $\hat{y}_{(t)}, t = 1, 2, \ldots, k$ 

Steps:
Standardized Dataset D;
Random Initialization of Network Parameters;
for  $t \leftarrow 1$  to k do

Encoder:
for  $m \leftarrow 1$  to T do

calculate spatial variable attention
 $e_{(t-m+1)} \leftarrow V_{1} \tanh(W_{1}s_{(t-m)} + U_{1}x_{(t-m+1)})$ ;

attention normalization  $\alpha_{(t-m+1)}^{i} \leftarrow \frac{\|e_{(t)}^{i}\|}{\sum_{j=1}^{n} \|e_{(t)}^{j}\|}, i = 1, 2, \ldots, n$ ;

spatial attention weighted sample
 $\tilde{x}_{(t-m+1)} \leftarrow \alpha_{(t-m+1)} \cdot x_{(t-m+1)}$ ;

encoder hidden states  $h_{(t-m+1)} \leftarrow f_{LSTM1}(\tilde{x}_{(t-m+1)}, h_{(t-m)})$ ;

Decoder:
for  $r \leftarrow 1$  to T do

calculate temporal attention
 $g_{(t)}^{r} \leftarrow V_{2} \tanh(W_{2}s_{(t-1)} + U_{2}h_{(t-r+1)})$ ;

attention normalization  $\beta_{(t)}^{r} \leftarrow \frac{\|g_{(t)}^{r}\|}{\sum_{m=1}^{T} \|g_{(t)}^{m}\|}$ ;

context vector  $c_{(t)} \leftarrow \beta_{(t)}^{1}h_{(t)} + \beta_{(t)}^{2}h_{(t-1)} + \ldots + \beta_{(t)}^{T}h_{(t-T+1)}$ ;

decoder hidden states  $s_{(t)} \leftarrow f_{LSTM2}(c_{(t)}, s_{(t-1)}, y_{(t-1)})$ ;

predicted quality variable  $\hat{y}_{(t)} \leftarrow f_{linear}(s_{(t)})$ ;
</div>

## IV. CASE STUDIES

To validate the effectiveness of STA-LSTM for soft sensor prediction, it is applied to an industrial hydrocracking process to predict the initial boiling points of the heavy naphtha and aviation kerosene. Fig. 4 shows the STA-LSTM-based soft sensor modeling framework. The configurations of the simulation computer are: the operating system is Windows $7 ;$ the CPU is an Intel i5-4460 (3.20 GHz); the RAM is 8 GB; and the code software is Python 3.5.

![](images/5a52aa876cd9e34688c7ad2b7a69938998b157d1139560615801a646080bc198.jpg)  
Fig. 4. Flowchart for the STA-LSTM-based soft sensor model framework

For performance comparison, the autoregressive integrated moving average model with external input (ARIMAX), static DBN, RNN, the standard LSTM, the attention-based LSTM without spatial variable attention, and the proposed STA-LSTM are developed for soft sensors of quality prediction. The root mean squared error (RMSE) and correlation coefficient, $R ^ { 2 } ,$ , are used to compare the accuracy of different models. These are defined as

$$
R M S E = \sqrt {\sum_ {t = 1} ^ {T _ {\text { testing }}} \left(y _ {(t)} - \hat {y} _ {(t)}\right) ^ {2} / \left(T _ {\text { testing }} - 1\right)}\tag{21}
$$

$$
R ^ {2} = 1 - \sum_ {t = 1} ^ {T _ {\text { testing }}} (y _ {(t)} - \hat {y} _ {(t)}) ^ {2} / \sum_ {t = 1} ^ {T _ {\text { testing }}} (y _ {(t)} - \tilde {y}) ^ {2}\tag{22}
$$

where $T _ { t e s t i n g }$ is the number of testing samples; $\tilde { y }$ is the mean of the real quality values in the testing data; and $y _ { ( t ) }$ and $\hat { y } _ { ( t ) }$ are respectively the labeled and predicted values for the quality variable at t.

## A. Description of the hydrocracking process

Hydrocracking is an important process in the petrochemical industry that converts the raw oil into its valuable constituents. Fig. 5 shows a diagram of a typical hydrocracking process. As can be seen in Fig. 5, the main devices in this process are the heating furnace, the hydrotreater, the hydrocracker, the high/low-pressure separators (HPS/LPS) and the distillation columns in the downstream separation part. The heavy gas oil and vacuum gas oil are hydrogenated, cracked, and isomerized with a hydrogen-rich gas stream at high pressure and temperature. In this way, the heavy compounds are converted into light oil products such as gasoline, kerosene, and light diesel [45]. Concretely, the make-up hydrogen and heavy vacuum gas oil are first mixed as the two main raw materials to carry out hydrocracking reaction under high temperature and pressure in the hydrocracking reactor. Then, the light oil products such as gasoline, kerosene, and light diesel oil can be obtained through a series of heat transfer, cooling, and heating in the HPS/LPS and fractionation unit. As can be seen, the hydrocracking process is a large time-delay system with complex physicochemical reactions and long-processing technology in a series of devices or apparatuses. It often costs hours or more time to obtain the product from the feedstocks. Usually, quality attributes, such as the initial boiling points of the heavy naphtha and aviation kerosene, are key performance indicators (KPIs) that can reflect the sufficiency and efficiency of the process. The timely measurement of these KPIs can provide real-time feedback for process monitoring, control, and optimization. However, these key variables are often very difficult to measure online. Moreover, since the reaction kinetics are very complex, a first-principles model cannot meet the industrial estimation requirements due to process variations like the changes in feed compositions, operating conditions, and catalyst de-activation. Hence, most of the KPIs are obtained using offline laboratory tests, which results in large time delay for process control and monitoring.

To deal with this problem, soft sensors have been adopted to predict the KPIs in this process based on historical process data. The initial boiling points of the heavy naphtha and aviation kerosene are selected as the quality variables to be predicted in this study. For this purpose, 43 process variables like temperature and pressure were chosen as secondary variables for the soft sensors [46].

![](images/f3394b1e6070445a2556a8de72fe2f0f492d16aacd7f44171c8c2d37a259a353.jpg)  
Fig. 5 Schematic of the hydrocracking process

## B. Prediction for the initial boiling point of heavy naphtha

The initial boiling point of heavy naphtha is the temperature at the end of the condenser when the first drop is obtained. The data were collected from a real industrial refinery in China, for a period of about two years from December 20, 2016 to September 29, 2018 with the sampling frequency of one quality sample every day. For confidentiality reasons, all the variables are normalized to the range 0 to 1. To build the soft sensor model, 650 labelled samples are collected from this process. The first 350 samples are selected as the training data, the middle 100 samples are used for model parameter validation, and the remaining 200 are used for the testing dataset.

For STA-LSTM in this study, the forecasting length is one sample for each subwindow. That is to say, once one sample is predicted, the window is moved forward by one step to predict the next sample. First, some of the main parameters that need to be optimized in the model are the number of time steps in the subwindow T; the number of neurons p in the hidden layer for the encoder LSTM; and the number of neurons q in the hidden layer for the decoder LSTM. The window size T is selected with a grid search from the candidate set {2, 3, 4, 5, 6}. Detailed RMSE for the window size is shown in Fig. 6. As can be seen from Fig. $^ { 6 , }$ T is selected to be 3 in this experiment since this achieves the best performance on the validation set for the model. For simplicity, p is set to be equal to q. As well, these two parameters are determined using a grid search on the set {15, 30, 60, 90, 120}. When $p = q = 6 0$ , the model achieves the best performance for the validation set. Thus, the number of neurons is selected as 60 in this study. Next, the prediction performance of RMSE as a function of different window size is investigated. As well, the minibatch size, which is the number of samples in each minibatch, should be determined for model training. By changing the minibatch size from the set {10, 20, 30, 40, 50}, the corresponding prediction performance of RMSE is obtained using the validation dataset. Fig. 7 shows the results. It can be seen that the optimal minibatch size is 30. Similarly, the learning rate is set as 0.001 and the epoch number is determined as 100 during the training process in this case. In addition, the neural network in spatial attention and temporal attention are both two-layer neural networks. Thus, the number of input and output neurons in the SA is equal to the dimension of the input variables. For the TA, the number of input and output are the window size T and the dimension of predicted quality variable, respectively.

![](images/ecbc4c312f76223887ac9f0027a35269c45031094ba116188c6d91eddb39ed29.jpg)

Fig 6. The relationship between RMSE and window size, T, for STA-LSTM for the initial boiling point of heavy naphtha dataset  
![](images/45359e56e85b4435be85f49e9817989bcad7fc656157cf95942e48acbbab0ba0.jpg)  
Fig. 7. The relationship between RMSE and minibatch size for STA-LSTM for the initial boiling point of heavy naphtha dataset

Table I Prediction RMSE of the three methods for the initial boiling point of heavy naphtha on the testing dataset

<table><tr><td>Method</td><td>RMSEtesting</td><td> $R^{2}_{testing}$ </td></tr><tr><td>ARIMAX</td><td>0.0767</td><td>0.3031</td></tr><tr><td>DBN</td><td>0.0754</td><td>0.3262</td></tr><tr><td>RNN</td><td>0.0521</td><td>0.4162</td></tr><tr><td>LSTM</td><td>0.0574</td><td>0.6092</td></tr><tr><td>Attention LSTM</td><td>0.0397</td><td>0.8082</td></tr><tr><td>STA-LSTM</td><td>0.0184</td><td>0.9548</td></tr></table>

![](images/e20f18427936c411462a53936706730414458565992c76a617d24185ffe0f4ec.jpg)  
(a)

![](images/b83568c7ff81cabb4e25ec468b0cd63cc20409d42a3db270dd03ee7e0e27021c.jpg)

![](images/078e97cc95ccc6afecf172daf3f3e343ba4062f4460cb7d1f0c0cdfe4052db44.jpg)  
(c)

(b)  
![](images/557e4b263b5a94cc5b6728cde55c3e30f8a45fd3dabc9f7aa08cf0f395116d67.jpg)  
(d)  
Fig. 8. Model validity tests for STA-LSTM for the prediction of the initial boiling point of heavy naphtha. (a) model residuals; (b) histogram of the residuals; (c) quantile-quantile plot; (d) residual autocorrelation

It takes 31.99 s to train the proposed STA-LSTM-based soft sensor model and 0.14 s for testing of the initial boiling point of heavy naphtha. It is essential to validate the structure and parameters for the proposed model [47, 48]. Hence, some statistical tests, like the detailed prediction residuals, the histogram, the quantile-quantile plot and residual autocorrelation, are carried on the model residuals, which are shown in Fig 8. Fig. 8(a) shows the detailed prediction errors for each testing sample. Fig. 8(b) and Fig. 8(c) show that the prediction errors are normally distributed. Fig. 8(d) gives the residual autocorrelations, which shows that the residuals are as required independent of each other. Hence, the prediction errors are normally distributed and independent, which indicates that the proposed model is good.

Table I compares the prediction performance for the six soft sensors using the testing dataset. The moving average order is optimized as 3 and the autoregressive order is determined as 4 for ARIMAX. From Table I, ARIMAX has the worst prediction performance on the testing dataset. Since it is a linear model, it is difficult to capture the nonlinear relationships of this process. For the deep belief network (DBN), it can model the nonlinear relationship in process data. Thus, it performs better than ARIMAX. However, it is a static method that the data temporal relationship is not taken into consideration for modeling. The recurrent neural network (RNN) has the ability of capturing the dynamic nature in process data. Thus, RNN outperforms DBN. However, it is difficult for RNN to learn long-term dependencies. For the LSTM network, it can partially extract the nonlinear characteristics of the data through the nonlinear activation function and use a memory cell to store long-term information for quality variable prediction. However, it does not give different attention to the subwindow data at different time steps. Hence, it may lose important hidden state information from the past. On the other hand, attention-based LSTM model uses an attention mechanism to take the previous decoder hidden state as a reference. Then, different attention weights are assigned to the encoder hidden states across the time steps in the subwindow for prediction of each query data.

In this way, attention-based LSTM can get much better prediction performance than LSTM. For the proposed STA-LSTM model, the spatiotemporal mechanism can, not only adaptively discover the relevant samples across the time steps in the window, but also adaptively identify the input variables related to the quality variable at the current time step t. Hence, the proposed method has better prediction accuracy since the spatial attention mechanism and temporal attention mechanism are simultaneously introduced for adaptive modeling. Furthermore, the detailed predictions on the testing dataset are shown in Fig. 9 for ARIMAX, DBN, RNN, LSTM, attention-based LSTM and STA-LSTM. As can be seen from Fig. 9, the prediction performances of the ARIMAX, DBN, RNN and LSTM-based soft sensors are the worst, since their prediction curves do follow the measured values. For attention-based LSTM, the prediction curve tracks the changes in measured values much better, but there are still large deviations between the predicted and measured output values. However, by introducing the spatial and temporal attention mechanism into the attention-based LSTM, the prediction curves can track the measured output curve very well. Fig. 10 shows the spatial variable attention weights of five samples for the STA-LSTM model in the testing set. It can be seen that the input variables have different importance for the prediction and the quality variable.

![](images/f2582c5e269797ed2f1f7e8d0ce50087f5162a99b9c518aac5fc2bbbd6ecffc4.jpg)  
(a)

![](images/05c04589dab9692a7dda03f9c3e849510fbbf11cba1da9f35340930e901c579a.jpg)  
(b)

![](images/ed101353fbc26dae3b542336a0924ba0904033e15752cc3eae473fb7000bbbb3.jpg)  
(c)

![](images/7b2aeabb4d9a8eab5f669462aaa89743e9993d75fbdd25ec3b8bdbfa1cfc4407.jpg)  
(d)

![](images/78c56ea2434a4ec1e7b3b5d2d1132cbed08e885f8ab41c6fdd71640a7b18c60e.jpg)  
(e)

![](images/aa290393b94d6bc2cc876c6f45d2869fa4edc37ce84ee11b87199af7e94df4ba.jpg)  
(f)  
Fig. 9. Detailed prediction results for the initial boiling point of heavy naphtha: (a) ARIMAX; (b) DBN; (c) RNN; (d) LSTM; (e) attention-based LSTM; and (f) STA-LSTM

![](images/10bf6c38092e4024ef37874bb75a11ea20be683469be83bec0edb3a0ff37abdb.jpg)  
Fig.10. The spatial variable attention weights of five samples in STA-LSTM for the initial boiling point of heavy naphtha

## C. Prediction for the initial boiling point of aviation kerosene

Furthermore, the proposed STA-LSTM model is applied to predicting the initial boiling point of the aviation kerosene. The dataset was collected from January 19, 2016 to November 30, 2018 with a sampling frequency of 12 h per sample. A total of 1689 samples were collected. To build the soft sensor model, the first 800 samples are selected as the training data. The next 200 samples are used for model parameter validation and the remaining 689 are for testing purposes. As before, the process and quality variables are standardized before modeling.

Also, the important parameters should be determined for the models. The first parameter is the window size T. Similarly, by changing the window size from the candidate set {2, 3, 4, 5, 6}, the prediction RMSE on the validation set is obtained after model training. Detailed results are given in Fig. 11. As can be seen from Fig. 11, the model achieves the best performance using the validation set when the window size is T = 3. Thus, the window size T is set to be 3 for STA-LSTM. Similarly, the number of neurons is selected as 60 for the encoder and decoder. Then, the minibatch size is investigated again in this case, which is selected from the set {32, 64, 96, 128, 160}. Here, the prediction performance of RMSE with THE minibatch is calculated for the validation set. The results are shown in Fig. 12. As can be seen, the validation prediction error reaches a minimum when the minibatch size is 128. Hence, the minibatch size is set to be 128. In addition, the learning rate is determined as 0.01. Also, the epoch number is set to 120 since the value of loss function converges after 120 training iterations.

![](images/3d7b542ed6e4b6f6cc82b5373aaca5decd8b37665baa606e161766944fa8f6c4.jpg)  
Fig 11. The relationship between RMSE and window size, T, for STA-LSTM using the initial boiling point of aviation kerosene dataset

![](images/f89119e61f928be1e66d7a4254c3c159170d7851007457ba35f4f31d5b76dcde.jpg)  
Fig. 12. The relationship between RMSE and minibatch size for STA-LSTM for the initial boiling point of aviation kerosene dataset

Table II Prediction RMSE of three methods for the initial boiling point of aviation kerosene using the testing dataset

<table><tr><td>Method</td><td> $RMSE_{testing}$ </td><td> $R^{2}_{testing}$ </td></tr><tr><td>ARIMAX</td><td>0.6420</td><td>0.0508</td></tr><tr><td>DBN</td><td>0.6326</td><td>0.0565</td></tr><tr><td>RNN</td><td>0.6011</td><td>0.1086</td></tr><tr><td>LSTM</td><td>0.5969</td><td>0.1820</td></tr><tr><td>Attention LSTM</td><td>0.4624</td><td>0.5091</td></tr><tr><td>STA-LSTM</td><td>0.2620</td><td>0.7613</td></tr></table>

![](images/c54015cfe05274b2a00ab9e5a564a3d2abd99c7f27950789fe77b7a1d2f04a37.jpg)  
(a)

![](images/aa8ea177b22733c12278cc724fd1179493dcfd1a8f5b32e77f9521f4e05cb228.jpg)  
(b)

![](images/c5d7d256b9c5a7c8fff9ea73df02b5c0a3d05daf0ddf34f9a616ebe659ba94e0.jpg)  
(c)

![](images/1077030fb8d6db2466f3723c3fe34b332c981b58a987a38836a00674c6e8cd83.jpg)  
(d)  
Fig. 13. Model validity tests for STA-LSTM for the prediction of the initial boiling point of aviation kerosene. (a) model residuals; (b) histogram of the residuals; (c) quantile-quantile plot; (d) residual autocorrelation.

It takes 56.49 s to train the proposed STA-LSTM-based soft sensor model and 0.47 s for testing. Also, some tests are performed on the model residuals, which are shown in Fig 13. It can be seen from Fig. 13(a) that the prediction errors are randomly distributed. Fig. 13(b) and Fig. 13(c) show that the prediction errors are approximately normally distributed. Fig 13. (d) shows the residual autocorrelations, which suggest that the residuals are independent of each other. Thus, STA-LSTM has a good model structure and parameters with accurate prediction performance.

Similarly, ARIMAX, DBN, RNN, LSTM, attention-based LSTM and STA-LSTM are used to predict the initial boiling point of aviation kerosene. Table II gives the prediction results of the six methods on the testing dataset. As before, the ARIMAX, DBN, RNN and LSTM networks give the worst prediction accuracy, while the attention-based LSTM provides a better prediction by using an attention mechanism to take the previous hidden states into consideration and assign different attention to them across the time steps. However, STA-LSTM provides the best prediction performance, since it can incorporate the most relevant information into its prediction. Fig. 14 further shows the detailed prediction values for the testing dataset with ARIMAX, DBN, RNN, LSTM, attention-based LSTM and STA-LSTM. From Fig. 14, the prediction of the proposed STA-LSTM method is in good match with the actual trajectory of the initial boiling point of aviation kerosene, and thus has a much smaller deviation. It can easily be seen that STA-LSTM provides the best prediction curve among the six methods. Also, Fig. 15 shows the spatial attention weight of each input variable for five samples, from which it can be seen that some variables are more important in predicting the quality of this sample.

![](images/51a3e6c690f95190b686ff1a7a027961ea9586028a32754356e3042d19cf1c2f.jpg)  
(a)

![](images/e5f6387e255d37725a45533decc83fc8212c19461df2c91243e53db0d92caad8.jpg)  
(b)

![](images/c87f8d2b56f2db7b349fca245cbec816cd4d2aed1c63e7f7da27d9f53551b1a2.jpg)  
(c)

![](images/2a9b5c190c35546daa126e08464e94adde711a5ea86a5cbc2fb55df2be3f85ae.jpg)

![](images/2867624cfb79593702f5d5d0f774572c851e49882399faf4613397aa83f18632.jpg)  
(e)

(d)  
![](images/bfd921293030fba5cd6d63e0d2c5fd4521633a6823273d392584f1b26cd9c162.jpg)  
(f)

Fig. 14. Prediction performance for the initial boiling point of aviation kerosene: (a) ARIMAX; (b) DBN; (c) RNN; (d) LSTM; (e) attention-based LSTM; (f) STA-LSTM  
![](images/11cc68c06857040378268dd3b1a77311b8b57a321e68de907b1a360c371c561f.jpg)  
Fig. 15. The spatial variable attention weights of five samples in STALATM for the initial boiling point of aviation kerosene

## V. CONCLUSION

In this paper, an attention-based framework is introduced for data-driven soft sensor modeling of industrial data time series. Since traditional attention-based LSTM networks only focus on the adaptive selection of the hidden states at different time steps, the importance of input variables with quality prediction is not considered for attention modeling. Hence, a new spatiotemporal attention-based LSTM model is proposed to obtain both the spatial variable and temporal sample attention for accurate modeling. A spatial attention mechanism is used to adaptively discover the input variables that are related to the quality variable and the attention weights are given for each input variable. Then, the temporal attention mechanism is used to model the temporal relevance of the hidden states at different time steps. Finally, the proposed STA-LSTM model is applied to an industrial hydrocracking process for quality prediction. The results show that the proposed model outperforms the ARIMAX, DBN, RNN, LSTM and attention-based LSTM models. As can be seen, these models require the process data to be uniformly sampled with a fixed frequency. Future work will focus on modeling with an irregular sampling rate and providing multistep predictions for industrial processes.

## REFERENCES

[1] X. Yuan, Y. Wang, C. Yang, Z. Ge, Z. Song, and W. Gui, “Weighted linear dynamic system for feature representation and soft sensor application in nonlinear dynamic industrial processes,” IEEE. T. Ind. Electron., vol. 65, no. 2, pp. 1508-1517, 2018.

[2] S. Khatibisepehr, B. Huang, and S. Khare, “Design of inferential sensors in the process industry: A review of Bayesian methods,” J. Process. Contr., vol. 23, no. 10, pp. 1575-1596, 2013.

[3] X. Yuan, C. Ou, Y. Wang, C. Yang, and W. Gui, “A layer-wise data augmentation strategy for deep learning networks and its soft sensor application in an industrial hydrocracking process,” IEEE T. Neur. Net. Lear. Syst., pp. DOI: 10.1109/TNNLS.2019.2951708, 2019.

[4] M. Järvisalo, T. Ahonen, J. Ahola, A. Kosonen, and M. Niemelä, “Soft-sensor-based flow rate and specific energy estimation of industrial variable-speed-driven twin rotary screw compressor,” IEEE. T. Ind. Electron., vol. 63, no. 5, pp. 3282-3289, 2016.

[5] N. Chen, J. Dai, X. Yuan, W. Gui, W. Ren, and H. N. Koivo, “Temperature Prediction Model for Roller Kiln by ALD-Based Double Locally Weighted Kernel Principal Component Regression,” IEEE T. Instrum. Meas., vol. 67, no. 8, pp. 2001-2010, 2018.

[6] X. Yuan, Z. Ge, B. Huang, and Z. Song, “A probabilistic just-in-time learning framework for soft sensor development with missing data,” IEEE. T. Contr. Syst. T., vol. 25, no. 3, pp. 1124-1132, 2017.

[7] W. Shao, and X. Tian, “Adaptive soft sensor for quality prediction of chemical processes based on selective ensemble of local partial least squares models,” Chem. Eng. Res. Des., vol. 95, pp. 113-132, 2015.

[8] J. Dai, N. Chen, X. Yuan, W. Gui, and L. Luo, “Temperature prediction for roller kiln based on hybrid first-principle model and data-driven MW-DLWKPCR model,” ISA T., vol. 98, pp. 403-417, 2020.

[9] Y. A. Shardt, Statistics for chemical and process engineers: a modern approach, Cham, Switzerland: Springer International Publishing, 2015.

[10] X. Yuan, Z. Ge, B. Huang, Z. Song, and Y. Wang, “Semisupervised JITL Framework for Nonlinear Industrial Soft Sensing Based on Locally Semisupervised Weighted PCR,” IEEE Trans. Ind. Informat., vol. 13, no. 2, pp. 532-541, 2017.

[11] Z. Ge, “Mixture Bayesian regularization of PCR model and soft sensing application,” IEEE. T. Ind. Electron., vol. 62, no. 7, pp. 4336-4343, 2015.

[12] J. Zheng, Z. Song, and Z. Ge, “Probabilistic learning of partial least squares regression model: Theory and industrial applications,” Chemometr. Intell. Lab. Syst., vol. 158, pp. 80-90, 2016.

[13] X. Yuan, J. Zhou, and Y. Wang, “A spatial-temporal LWPLS for adaptive soft sensor modeling and its application for an industrial hydrocracking process,” Chemometr. Intell. Lab. Syst, vol. 197, pp. 103921, 2020.

[14] Y. Wang, D. Wu, and X. Yuan, “A two-layer ensemble learning framework for data-driven soft sensor of the diesel attributes in an industrial hydrocracking process,” J. Chemometr., vol. 33, no. 12, pp. e3185, 2019.

[15] X. Yuan, S. Qi, and Y. Wang, “Stacked Enhanced Auto-encoder for Data-driven Soft Sensing of Quality Variable,” IEEE T. Instrum. Meas., to be published, 2020.

[16] M. Dam, and D. N. Saraf, “Design of neural networks using genetic algorithm for on-line property estimation of crude fractionator products,” Comput. Chem. Eng., vol. 30, no. 4, pp. 722-729, 2006.

[17] N. M. Ramli, M. A. Hussain, B. M. Jan, and B. J. N. Abdullah, “Composition Prediction of a Debutanizer Column using Equation Based Artificial Neural Network Model,” Neurocomputing, vol. 131, no. 12, pp. 59-76, 2014.

[18] G. E. Hinton, “Learning multiple a layers of representation,” Trends. Cogn. Sci., vol. 11, no. 10, pp. 428-434, 2007.

[19] X. Yuan, B. Huang, Y. Wang, C. Yang, and W. Gui, “Deep Learning-Based Feature Representation and Its Application for Soft Sensor Modeling With Variable-Wise Weighted SAE,” IEEE Trans. Ind. Informat., vol. 14, no. 7, pp. 3235-3243, 2018.

[20] S. Chao, Y. Fan, D. Huang, and W. Lyu, “Data-driven soft sensor development based on deep learning technique,” J. Process. Contr., vol. 24, no. 3, pp. 223-233, 2014.

[21] X. Yuan, C. Ou, Y. Wang, C. Yang, and W. Gui, “Deep quality-related feature extraction for soft sensing modeling: A deep learning approach with hybrid VW-SAE,” Neurocomputing, DOI: 10.1016/j.neucom.2018.11.107, 2019.

[22] S. Graziani, and M. G. Xibilia, "A deep learning based soft sensor for a sour water stripping plant." pp. 1-6.

[23] X. Yuan, Y. Gu, Y. Wang, C. Yang, and W. Gui, “A deep supervised learning framework for data-driven soft sensor modeling of industrial processes,” IEEE T. Neur. Net. Lear. Syst., DOI: 10.1109/TNNLS.2019.2957366, 2019.

[24] L. Yao, and Z. Ge, “Deep Learning of Semisupervised Process Data With Hierarchical Extreme Learning Machine and Soft Sensor Application,” IEEE. T. Ind. Electron., vol. 65, no. 2, pp. 1490-1498, 2018.

[25] X. Yuan, C. Ou, Y. Wang, C. Yang, and W. Gui, “A novel semi-supervised pre-training strategy for deep networks and its application for quality variable prediction in industrial processes,” Chem. Eng. Sci., vol. 217, pp. 115509, 2020.

[26] K. Wang, B. Gopaluni, J. Chen, and Z. Song, “Deep Learning of Complex Batch Process Data and Its Application on Quality Prediction,” IEEE. T. Ind. Inf., 2018.

[27] X. Yuan, Y. Wang, C. Yang, and W. Gui, “Stacked isomorphic autoencoder based soft analyzer and its application to sulfur recovery unit,” Inform. Sci., to be published, 2020.

[28] Y. Wang, Z. Pan, X. Yuan, C. Yang, and W. Gui, “A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network,” ISA T., vol. 96, pp. 457-467, 2020.

[29] W. Yan, D. Tang, and Y. Lin, “A Data-Driven Soft Sensor Modeling Method Based on Deep Learning and its Application,” IEEE. T. Ind. Electron., vol. 64, no. 5, pp. 4237-4245, 2017.

[30] X. Yuan, J. Zhou, B. Huang, Y. Wang, C. Yang, and W. Gui, “Hierarchical quality-relevant feature representation for soft sensor modeling: a novel deep learning strategy,” IEEE. T. Ind. Inf., vol. 16, no. 6, pp. 3721-3730, 2020.

[31] A. Graves, “Supervised Sequence Labelling with Recurrent Neural Networks,” Studies in Computational Intelligence, vol. 385, 2008.

[32] Y. Bengio, P. Simard, and P. Frasconi, “Learning long-term dependencies with gradient descent is difficult,” IEEE trans. neural. netw., vol. 5, no. 2, pp. 157-166, 1994.

[33] S. Hochreiter, and J. Schmidhuber, “Long Short-Term Memory,” Neural. Comput., vol. 9, no. 8, pp. 1735-1780, 1997.

[34] X. Yuan, L. Li, and Y. Wang, “Nonlinear dynamic soft sensor modeling with supervised long short-term memory network,” IEEE. T. Ind. Inf., vol. 16, no. 5, pp. 3168-3176, 2020.

[35] X. Yuan, L. Li, Y. Wang, C. Yang, and W. Gui, “Deep learning for quality prediction of nonlinear dynamic process with variable attention-based long short-term memory network,” Can. J. Chem. Eng., DOI: 10.1002/cjce.23665, 2019.

[36] G. Kurata, B. Ramabhadran, G. Saon, and A. Sethy, "Language Modeling with Highway LSTM." pp. 244-251.

[37] F. A. Gers, D. Eck, and J. Schmidhuber, "Applying LSTM to time series predictable through time-window approaches," Lecture Notes in Computer Science. pp. 669-676.

[38] A. Graves, N. Jaitly, and A.-r. Mohamed, "Hybrid speech recognition with deep bidirectional LSTM." pp. 273-278.

[39] D. Bahdanau, K. Cho, and Y. J. a. p. a. Bengio, “Neural machine translation by jointly learning to align and translate,” Compute. Sci., vol. 1409, 2014.

[40] Y. Qin, D. Song, H. Chen, W. Cheng, G. Jiang, and G. Cottrell, “A dual-stage attention-based recurrent neural network for time series prediction,” IJAI., 2017.

[41] V. Mnih, N. Heess, and A. Graves, "Recurrent models of visual attention." pp. 2204-2212.

[42] I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to sequence learning with neural networks." pp. 3104-3112.

[43] K. Cho, B. V. Merrienboer, D. Bahdanau, and Y. J. C. S. Bengio, "On the Properties of Neural Machine Translation: Encoder-Decoder Approaches."

[44] D. Kingma, and J. Ba, “Adam: A Method for Stochastic Optimization,” Compute. Sci., 2014.

[45] X. Yuan, J. Zhou, Y. Wang, and C. Yang, “Multi-similarity measurement driven ensemble just-in-time learning for soft sensing of industrial processes,” J. Chemometr., vol. 32, no. 9, pp. e3040, 2018.

[46] X. Yuan, J. Zhou, and Y. Wang, "A Comparative Study of Adaptive Soft Sensors for Quality Prediction in an Industrial Refining Hydrocracking Process." pp. 1064-1068.

[47] S. A. BlLlings, and W. S. F. Voon, “Correlation based model validity tests for non-linear models,” Int. J. Control, vol. 44, no. 1, pp. 235-244, 1986.

[48] S. A. Billings, and W. S. F. Voon, “A prediction-error and stepwise-regression estimation algorithm for non-linear systems,” Int. J. Control, vol. 44, no. 3, pp. 803-822, 1986.

![](images/b9f0adeed0bfbb3874679cabb147926bfc2331f458f912c44dfe85b619805aac.jpg)

Xiaofeng Yuan (M’17) received the B.Eng. and Ph.D. degrees from the Department of Control Science and Engineering, Zhejiang University, Hangzhou, China, in 2011 and 2016, respectively.

He was a visiting scholar with the Department of Chemical and Materials Engineering, University of Alberta,

Edmonton, AB, Canada, from November 2014 to May 2015. He is currently an Associate Professor with the School of Automation, Central south University. His research interests include deep learning and artificial intelligence, machine learning and pattern recognition, industrial process soft sensor modeling, process data analysis, etc.

![](images/2203e6af29e66e5fb0bb49ce486773fb3a9947ba6bd67b0618074a7a13eafc0a.jpg)

Lin Li is a postgraduate student at the School of Automation, Central South University, Changsha, China. She received her B.Eng. degree in School of Information Engineering from Xiangtan University, Xiangtan, China, in 2018. Her research interests include deep learning, machine learning, soft sensor modeling, process data mining, etc.

![](images/fd2c1cb3a0d15a41a6d6248ebebae3e42dc5750262f2566899da1f9d01f37311.jpg)

Yuri A. W. Shardt is currently a professor and chair of the Department of Automation Engineering at the Technical University of Ilmenau. His areas of interest focus on big data, system identification, data quality assessment, holistic control, and the smart world. Previously, he was an assistant

professor at the University of Waterloo in the Department of Chemical Engineering and a holder of the prestigious Alexander von Humboldt Scholarship at the University of Duisburg-Essen in the Institute of Control and Complex Systems. He has written a book, entitled Statistics for Chemical and Process Engineers: A Modern Approach, that focuses on the required mathematical background in order to implement advanced statistical methods using Excel<sup>®</sup> and MATLAB<sup>®</sup>.

This book has been translated into German and is scheduled to be published in late 2020 by Springer as Methoden der Statistik und Prozessanalyse. Prof. Shardt has written numerous papers that have appeared in such journals as Automatica, Journal of Process Control, IEEE Transactions on Industrial Electronics, and Industrial and Engineering Chemistry Research and presented at multiple conferences. He has taught various courses in the intersection between statistics, chemical engineering, process control, EXCEL<sup>®</sup>, and MATLAB<sup>®</sup>. Prof. Dr. Shardt completed his doctoral degree under the supervision of Prof. Dr. Biao Huang at the University of Alberta. His thesis examined the methods for extracting valuable data for system identification from data historians for application to soft sensor design. In addition to his academic work, he has spent considerable time in industry working on implementing various process control solutions. He also has interests in linguistics, as well as software internationalisation and localisation.

![](images/552d9ce7f9e7a0a4b5f7b10d5534ebb9e20ca163f8fb77f5c5c368d19a6cf90d.jpg)

Yalin Wang (M’17) received the B.Eng. and Ph.D. degrees from the Department of Control Science and Engineering, Central South University, Changsha, China, in 1995 and 2001, respectively.

Since 2003, she has been with the School of Information Science and Engineering, Central south University, where she was at

first an Associate Professor and then a Professor. She is currently a Professor with the School of Automation, Central South University. Her research interests include the modeling, optimization and control for complex industrial processes, intelligent control, and process simulation.

![](images/9ef2bdecc83c925a8bdd26ee4595b445e9ae4d017d7b29eff136bda7ecc95b36.jpg)

Chunhua Yang (M’09) received the M.Eng. degree in automatic control engineering and the Ph.D. degree in control science and engineering from Central South University, Changsha, China, in 1988 and 2002, respectively. She was with the Department of Electrical Engineering, Katholieke Universiteit Leuven, Leuven, Belgium, from 1999 to 2001. She is currently a Full

Professor with Central South University. Her current research interests include modeling and optimal control of complex industrial process, intelligent control system, and fault-tolerant computing of real-time systems.