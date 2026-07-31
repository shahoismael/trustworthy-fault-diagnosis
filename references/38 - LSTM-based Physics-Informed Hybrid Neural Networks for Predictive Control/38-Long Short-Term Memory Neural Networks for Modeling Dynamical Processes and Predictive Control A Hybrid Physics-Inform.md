Article

# Long Short-Term Memory Neural Networks for Modeling Dynamical Processes and Predictive Control: A Hybrid Physics-Informed Approach

Krzysztof Zarzycki \* and Maciej Ławry ´nczuk

Institute of Control and Computation Engineering, Faculty of Electronics and Information Technology, Warsaw University of Technology, ul. Nowowiejska 15/19, 00-665 Warsaw, Poland; maciej.lawrynczuk@pw.edu.pl \* Correspondence: krzysztof.zarzycki@pw.edu.pl

Abstract: This work has two objectives. Firstly, it describes a novel physics-informed hybrid neural network (PIHNN) model based on the long short-term memory (LSTM) neural network. The presented model structure combines the first-principle process description and data-driven neural sub-models using a specialized data fusion block that relies on fuzzy logic. The second objective of this work is to detail a computationally efficient model predictive control (MPC) algorithm that employs the PIHNN model. The validity of the presented modeling and MPC approaches is demonstrated for a simulated polymerization reactor. It is shown that the PIHNN structure gives very good modeling results, while the MPC controller results in excellent control quality.

Keywords: dynamical systems; LSTM neural networks; physics-informed neural networks; model predictive control

![](images/062705f34c5bc61ab4c725d4cd91d4505560f9bbd0125e4a745f054a0f058516.jpg)

Citation: Zarzycki, K.; Ławry ´nczuk, M. Long Short-Term Memory Neural Networks for Modeling Dynamical Processes and Predictive Control: A Hybrid Physics-Informed Approach. Sensors 2023, 23, 8898. https:// doi.org/10.3390/s23218898

Academic Editors: Ching-Hung Lee and Lian-Wang Lee

Received: 4 October 2023 Revised: 26 October 2023 Accepted: 30 October 2023 Published: 1 November 2023

![](images/8320bebd7364c9ae849900da016dabd4571a52bc5eb5e243ff0afafb30712cbf.jpg)

Copyright: © 2023 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https:// creativecommons.org/licenses/by/ 4.0/).

## 1. Introduction

Model predictive control (MPC) algorithms, as highlighted in [1,2], find their primary applications in managing processes that classical control methods struggle to handle effectively. These processes often involve multiple-input, multiple-output (MIMO) systems or exhibit strong nonlinearity. MPC, renowned for its flexibility in accommodating various constraints, excels in ensuring high-quality control, even in the face of challeng ing processes. Real-world instances of successful MPC applications include control of chemical reactors [3,4] and distillation towers [5], as well as the integration of MPC in embedded systems controlling heating ventilation and air conditioning systems (HVAC) [6], quadrotors [7], fuel cells [8], autonomous vehicles [9], and underwater vehicles [10].

As emphasized in [11–13], accurate sensor measurements of essential process variables play a critical role in MPC. It is widely acknowledged that the absence of these measurements inevitably leads to a significant loss in control performance. To address this challenge, when the necessary measurements are not readily available, engineers commonly employ online estimation techniques, such as Kalman or extended Kalman filters ([14]). Furthermore, specialized methods and strategies have been developed to tackle this issue in specific applications. In the domain of vehicles, innovative solutions have emerged. The authors of [15] introduce a real-world example where a vehicle employs an external camera to detect obstacles and lane positions on the road. Additionally, it utilizes external rear-corner radars to identify objects approaching from the rear. An intriguing application of sensors is presented in [16], where an anemometer measures external factors such as wind force and direction. Beyond the automotive sector, there are applications like sea ship depth measurement. In [10], a depth sensor is installed for precisely measuring sea ship depth, with heave speed derived from the depth sensor data. Finally, MPC is also used to manage fault-tolerant control. This application addresses issues like stiction in control valves, as discussed in [17].

The cornerstone of any effective MPC algorithm is the precision of its process model. Broadly, two general model classes are usually considered: first-principle (FP) models rooted in the fundamental understanding of the process; black-box approximations. Both model classes have their distinct strengths and limitations:

FP models demand meticulous process descriptions and accurate parameter values but offer unparalleled modeling precision across a wide operating range, even in abnormal situations. In practice, however, the values of some model parameters may be imprecise or unknown.

In contrast, data-driven black-box models, including support-vector machines (SVM) [18], multi-layer perceptron (MLP) neural networks [19,20], radial basis function (RBF) networks [21], and recurrent long short-term memory networks (LSTM) [22,23], require no prior domain expertise. Among these, gated recurrent unit (GRU) has gained traction in modeling dynamic systems [24,25] and integrating with MPC algorithms [23,26,27]. Neural networks models have proven to be very useful, especially when dealing with complex dynamical processes, such as predator–prey systems [28–30]. However, black-box models may struggle when the available dataset lacks coverage for certain process variables, particularly those operating at infrequent points.

Physics-informed neural network models (PINNs) offer a compelling fusion of both modeling approaches. These models combine the foundational principles governing the process with the data-driven power of machine learning. The result is a versatile model that adheres to fundamental laws while approximating the behavior of real-world processes. The literature showcases PINN applications in scenarios where parameters of ordinary differential equations (ODE) models are either imprecisely known [31] or immeasurable [32]. Furthermore, PINNs can approximate parameters of partial differential equations (PDE) [33]. These PINN models find utility in replacing numerical solvers for ODEs [34] and even serve as models within MPC frameworks [35]. Additionally, one can find several hybrid models aiming to combine a data-driven modeling approach with knowledge of physics. The hybrid physical guided neural network [36] is a feed-forward neural network integrated with a first-principles model. The entire hybrid model is trained jointly. This training process involves incorporating a fusion output layer that utilizes a straightforward interpolation technique. Other examples include using deep neural networks in a physically guided modeling approach [37], in modeling lithium batteries [38], and in modeling a traffic state [39]. One can also find examples of introducing physics directly in the forward pass of the neural network to model the lake temperature [40].

This study addresses a common modeling challenge characterized by two specific limitations. Firstly, process-variable measurements are typically feasible but confined to a limited vicinity of certain operating points. Consequently, the resulting models exhibit localized validity, restricted to the regions where data have been collected for identification purposes. Secondly, although fundamentally sound, the existing first-principle models describing the process often lack precision due to imprecise parameters. In response to these limitations, this work introduces an innovative physics-informed hybrid neural network (PIHNN) model structure, leveraging LSTM neural networks. This approach combines elements from both first-principle and black-box data-driven methodologies, offering robust modeling capabilities in scenarios characterized by the aforementioned issues. Within this research, we delve into two data fusion techniques, drawing from the principles of the first-principle process description and the LSTM network, both employing a fuzzy-logic-based approach. The initial method employs a simplified data fusion block, while the subsequent method harnesses machine learning techniques to minimize overall model errors. To assess the effectiveness of the proposed model structure and data fusion techniques, we apply them to a benchmark polymerization reactor process.

Additionally, we integrate the developed PIHNN model into the MPC framework. Our analysis encompasses a straightforward MPC algorithm with a nonlinear optimization MPC (MPC-NO) and a more intricate linearization-based MPC scheme named the MPC algorithm with nonlinear prediction and linearization around the predicted trajectory (MPC-NPLPT), which relies on computationally uncomplicated quadratic optimization tasks. Our findings demonstrate that the linearization-based MPC approach can yield commendable control performance while significantly reducing computational demands compared to nonlinear counterparts. An initial iteration of the PIHNN model was introduced in conference proceedings [41], where a basic GRU neural network was employed. This current study represents a substantial expansion of previous research efforts. Here, we consider more general LSTM-based PIHNN models, comprehensively examine the model’s structure, explore various potential variants and present details. Furthermore, we introduce an efficient model predictive control (MPC) algorithm for the PIHNN models considered in this study.

This work is organized as follows. Firstly, Section 2 presents the general structure and the details of the hybrid PIHNN model structure utilizing LSTM neural networks. The state–space modeling approach is employed. Secondly, Section 3 briefly describes the general MPC scheme with nonlinear optimization and presents general formulation, necessary implementation details, and the resulting quadratic optimization task of the linearization-based MPC method. Section 4 thoroughly studies the validity of as many as sic PIHNN model variants applied to approximate the behavior of a chemical reactor benchmark. Furthermore, the control efficiency and computational speed of the recommenced linearization-based MPC algorithm is shown. Finally, Section 5 concludes the article.

## 2. Hybrid Physics-Informed Models Using LSTM Neural Networks

We introduce an innovative PIHNN model that blends a data-driven approach with expert knowledge of the underlying physics of the process. To effectively apply the PIHNN model, the following conditions must be satisfied:

The process input and output variables, i.e., the manipulated and controlled variables, respectively, must be measurable. State variables may be measured or observed using a state estimation, e.g., in the form of an extended Kalman filter (EKF).

The FP process should exist in the form of a set of differential equations and, when necessary, additional algebraic relations based on the fundamental laws of physics governing the process.

However, we assume that the measurements and the FP model may exhibit imperfections. Specifically, the measurements may originate from a limited range within the entire spectrum of process variable variability. Furthermore, the FP model may also contain inaccuracies and be susceptible to errors arising from factors such as incorrect estimation of specific process parameters or measurement inaccuracies.

## 2.1. Model Structure

This paper primarily focuses on single-input single-output (SISO) process modeling. The process input and output are denoted as u and y, respectively. Additionally, the process has $n _ { x }$ state variables, represented as $\boldsymbol { x } = [ x _ { 1 } \dots x _ { n _ { \mathrm { x } } } ] ^ { \mathrm { T } }$

Figure 1 illustrates the model’s overall structure. The PIHNN model is divided into three distinct components. The first model component, highlighted in blue, is entirely data-driven. It comprises $n _ { \mathrm { L S T M } }$ neural sub-models, each trained on available data. The number of data-driven sub-models corresponds to the number of distinct operational areas of the process from which measurement data can be collected. Each sub-model takes the vector $\mathbf { \boldsymbol { x } } _ { \mathrm { L S T M } } ^ { i }$ as the input and generates the scalar $y _ { \mathrm { L S T M } } ^ { i }$ as the output. LSTM networks are employed in this study, as earlier research has demonstrated their exceptional ability to model dynamical processes [23,27]. However, it is important to note that alternative data-driven models could also be applied in this context. The second component of the PIHNN structure, highlighted in green, is rooted in expert knowledge about the underlying physics of the process. It consists of an FP sub-model formulated using ordinary differential equations. The input to this sub-model is the vector $\pmb { \mathrm { X } } _ { \mathrm { F P } }$ , while the output is denoted by the scalar $y _ { \mathrm { F P } }$ . The third component of the PIHNN structure, highlighted in orange, represents the data fusion block (DF). In general, many decision models can be used here, such as neural networks of various architectures. However, we recommend using the fuzzy data fusion block (Fuzzy DF) because it directly incorporates the sub-models. There is no need to train Fuzzy DF on data, which is particularly useful when training data is lacking across specific ranges of process variable variability. By selecting membership function shapes, one can determine which areas and to what extent we should consider the sub-models when calculating the overall PIHNN model output. The DF block takes output calculated by all LSTM sub-models and the FP models as inputs. Based on the current operating state of the process, represented by the vector $\mathbf { X } _ { \mathrm { D F } } .$ , it makes decisions regarding the combination of outputs from all sub-models. The primary goal of this fusion process is to minimize the overall error of the entire PIHNN model.

![](images/403849e7d75e01c2c13ac53c65e396daccb15b5c7d78074251bcf9643f6caf0b.jpg)  
Figure 1. General structure of the PIHNN model.

## 2.2. First-Principle Sub-Model

Typically, the FP model utilizes fundamental physical laws formulated in the continuous-time domain, i.e., a set of differential equations must be considered. The state equations have the classical form

$$
\dot {x} _ {1} (t) = f _ {1} (x _ {1} (t), \ldots , x _ {n _ {\mathrm{x}}} (t), u (t))\tag{1}
$$

$$
\dot {x} _ {n _ {\mathrm{x}}} (t) = f _ {n _ {\mathrm{x}}} (x _ {1} (t), \ldots , x _ {n _ {\mathrm{x}}} (t), u (t))\tag{2}
$$

while the output equation is

$$
y (t) = g (x _ {1} (t), \ldots , x _ {n _ {\mathrm{x}}} (t))\tag{3}
$$

where $f _ { 1 } , \ldots , f _ { n _ { \mathrm { v } } } \colon \mathbf { R } ^ { n _ { \mathrm { x } } + 1 }  \mathbf { R }$ and $g \colon \mathbf { R }  \mathbf { R }$ are nonlinear functions. Since we will next use the PIHNN model relying on the FP model in the MPC algorithm with online linearization, we require the functions $f _ { 1 } , \dots , f _ { n _ { \mathrm { x } } } , g$ to be differentiable. From Equations (1)–(3), we can find a corresponding discrete-time FP model

$$
x _ {1} (k) = f _ {1} ^ {\mathrm{d}} (x _ {1} (k - 1), \ldots , x _ {n _ {\mathrm{x}}} (k - 1), u (k - 1))\tag{4}
$$

$$
x _ {n _ {\mathrm{x}}} (k) = f _ {n _ {\mathrm{x}}} ^ {\mathrm{d}} (x _ {1} (k - 1), \ldots , x _ {n _ {\mathrm{x}}} (k - 1), u (k - 1))\tag{5}
$$

$$
y (k) = g ^ {\mathrm{d}} (x _ {1} (k - 1), \ldots , x _ {n _ {\mathrm{x}}} (k - 1))\tag{6}
$$

where $f _ { 1 } ^ { \mathrm { d } } , \ldots , f _ { n _ { \mathrm { x } } } ^ { \mathrm { d } } \colon \mathbf { R } ^ { n _ { \mathrm { x } } + 1 }  \mathbf { R }$ and $g ^ { \mathrm { d } } \colon { \mathbf { R } }  { \mathbf { R } }$ are nonlinear mapping functions. The input vector to the FP model can be expressed as

$$
\mathbf {X} _ {\mathrm{FP}} (k) = \left[ x ^ {\mathrm{T}} (k - 1) u (k - 1) \right] ^ {\mathrm{T}}\tag{7}
$$

## 2.3. LSTM Sub-Model

LSTM networks were developed in response to the vanishing gradient problem that impacts traditional recurrent neural networks [42]. Each LSTM neuron is referred to as a “cell” (Figure 2) and encompasses gates responsible for governing the flow of information within the network. The LSTM cell comprises four distinct gates:

The forget gate f determines which values from the previous cell state should be retained and which should be discarded.

The input gate i selects values from both the previous hidden state and the current input for updating purposes.

The cell state candidate gate $g$ initially regulates the flow of information within the network and subsequently computes the candidate value for the current cell state.

• The output gate o is responsible for calculating the new hidden state $' h ^ { \prime }$

![](images/0293ff0bab745b1cb7171208084a923b5670f9586993f0abc15979701b4f7ef0.jpg)  
Figure 2. Structure of the LSTM cell.

Each cell in the network has its input vector expressed as

$$
\mathbf {X} _ {\mathrm{LSTM}} (k) = [ x _ {\mathrm{LSTM}} ^ {1}, \ldots , x _ {\mathrm{LSTM}} ^ {m} ] ^ {T} = [ u (k - 1), \ldots , u (k - n _ {\mathrm{B}}), y (k - 1), \ldots , y (k - n _ {\mathrm{A}}) ] ^ {\mathrm{T}}\tag{8}
$$

where parameters $n _ { \mathrm { A } }$ and $n _ { \mathrm { B } }$ define the order of dynamics of the model. The LSTM network has $n _ { \mathrm { N } }$ cells. The weights in the network can be written in a matrix form

$$
\boldsymbol {W} = \left[ \begin{array}{c} \boldsymbol {W} _ {\mathrm{i}} \\ \boldsymbol {W} _ {\mathrm{f}} \\ \boldsymbol {W} _ {\mathrm{g}} \\ \boldsymbol {W} _ {\mathrm{o}} \end{array} \right], \boldsymbol {R} = \left[ \begin{array}{c} \boldsymbol {R} _ {\mathrm{i}} \\ \boldsymbol {R} _ {\mathrm{f}} \\ \boldsymbol {R} _ {\mathrm{g}} \\ \boldsymbol {R} _ {\mathrm{o}} \end{array} \right], \boldsymbol {b} = \left[ \begin{array}{c} \boldsymbol {b} _ {\mathrm{i}} \\ \boldsymbol {b} _ {\mathrm{f}} \\ \boldsymbol {b} _ {\mathrm{g}} \\ \boldsymbol {b} _ {\mathrm{o}} \end{array} \right]\tag{9}
$$

The input weight matrices $W _ { \mathrm { i } } , W _ { \mathrm { f } } , W _ { \mathrm { g } }$ and $W _ { \mathrm { o } }$ have dimensionality $n _ { \mathrm { N } } \times n _ { \mathrm { f } } ;$ the recurrent weight matrices $R _ { \mathrm { i } } , R _ { \mathrm { f } } , R _ { \mathrm { g } }$ and $R _ { \mathrm { o } }$ have dimensionality $n _ { \mathrm { N } } \times n _ { \mathrm { N } } ;$ and the bias vectors $\begin{array} { r } { \pmb { b } _ { \mathrm { i } } , } \end{array}$ $b _ { \mathrm { f } } , b _ { \mathrm { g } }$ and $b _ { \mathrm { o } }$ have dimensionality $n _ { \mathrm { N } } \times 1$ , respectively. At time instant $k ,$ the LSTM model initially calculates the output value of each gate

$$
\boldsymbol {i} (k) = \sigma \left(\boldsymbol {W} _ {\mathrm{i}} \boldsymbol {X} _ {\text { LSTM }} + \boldsymbol {R} _ {\mathrm{i}} \boldsymbol {h} (k - 1) + \boldsymbol {b} _ {\mathrm{i}}\right)\tag{10}
$$

$$
\boldsymbol {f} (k) = \sigma \left(\boldsymbol {W} _ {\mathrm{f}} \boldsymbol {X} _ {\text { LSTM }} + \boldsymbol {R} _ {\mathrm{f}} \boldsymbol {h} (k - 1) + \boldsymbol {b} _ {\mathrm{f}}\right)\tag{11}
$$

$$
\boldsymbol {g} (k) = \tanh \left(\boldsymbol {W} _ {\mathrm{g}} \boldsymbol {X} _ {\text { LSTM }} + \boldsymbol {R} _ {\mathrm{g}} \boldsymbol {h} (k - 1) + \boldsymbol {b} _ {\mathrm{g}}\right)\tag{12}
$$

$$
\boldsymbol {o} (k) = \sigma \left(\boldsymbol {W} _ {\mathrm{o}} \boldsymbol {X} _ {\text { LSTM }} + \boldsymbol {R} _ {\mathrm{o}} \boldsymbol {h} (k - 1) + \boldsymbol {b} _ {\mathrm{o}}\right)\tag{13}
$$

Subsequently, the cell state of the network can be computed

$$
\boldsymbol {c} (k) = \boldsymbol {f} (k) \circ \boldsymbol {c} (k - 1) + \boldsymbol {i} (k) \circ \boldsymbol {g} (k)\tag{14}
$$

where the symbol ◦ represents the Hadamard product of vectors. Finally, the hidden state can calculated

$$
\boldsymbol {h} (k) = \boldsymbol {o} (k) \circ \tanh (\boldsymbol {c} (k))\tag{15}
$$

The LSTM layer of the network is typically added to a fully connected layer (Figure 3), with weight matrix $W _ { \mathrm { y } }$ with a dimensionality of $1 \times n _ { \mathrm { N } }$ and bias $b _ { \mathrm { y } }$ . Finally, the computation of the network’s output at time instant k can be expressed as

$$
y _ {(i)} ^ {\mathrm{LSTM}} (k) = \boldsymbol {W} _ {\mathrm{y}} \boldsymbol {h} (k) + b _ {\mathrm{y}}\tag{16}
$$

![](images/afff9a098ba6671ef8f857fff239e84198d90b358063b226508cf111e641186d.jpg)  
Figure 3. Structure of the whole LSTM network.

One can represent Equations (10)–(15) in scalar form, which will prove useful for the derivation of the MPC algorithm considered in Section 3. The scalar form expressions for the n-th elements of the gate and state vectors are

$$
i _ {n} (k) = \sigma \bigg (\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{i}} x _ {\mathrm{LSTM}} ^ {m} (k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{i}} h _ {m} (k - 1)\right) + b _ {n} ^ {\mathrm{i}} \bigg)\tag{17}
$$

$$
f _ {n} (k) = \sigma \bigg (\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{f}} x _ {\mathrm{LSTM}} ^ {m} (k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{f}} h _ {m} (k - 1)\right) + b _ {n} ^ {\mathrm{f}} \bigg)\tag{18}
$$

$$
g _ {n} (k) = \tanh \left(\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{g}} x _ {\mathrm{LSTM}} ^ {m} (k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{g}} h _ {m} (k - 1)\right) + b _ {n} ^ {\mathrm{g}}\right)\tag{19}
$$

$$
o _ {n} (k) = \sigma \bigg (\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{o}} x _ {\mathrm{LSTM}} ^ {m} (k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{o}} h _ {m} (k - 1)\right) + b _ {n} ^ {\mathrm{o}} \bigg)\tag{20}
$$

$$
c _ {n} (k) = f _ {n} (k) c _ {n} (k - 1) + i _ {n} (k) g _ {n} (k)\tag{21}
$$

$$
h _ {n} (k) = o _ {n} (k) \tanh \left(c _ {n} (k)\right)\tag{22}
$$

$$
y _ {(i)} ^ {\mathrm{LSTM}} (k) = \sum_ {n = 1} ^ {n _ {\mathrm{N}}} w _ {\mathrm{y}} ^ {n} h _ {n} (k) + b _ {\mathrm{y}}\tag{23}
$$

Equations (22) and (23) could be used to find the output of the network in the form of one equation

$$
y _ {(i)} ^ {\mathrm{LSTM}} (k) = \sum_ {n = 1} ^ {n _ {\mathrm{N}}} w _ {\mathrm{y}} ^ {n} \Big (o _ {n} (k) \tanh \big (f _ {n} (k) c _ {n} (k - 1) + i _ {n} (k) g _ {n} (k) \big) \Big) + b _ {\mathrm{y}}\tag{24}
$$

## 2.4. Fuzzy Data Fusion Block

Considering Figure 1, the output of the whole PIHNN model is

$$
y _ {\mathrm{PIHNN}} (k) = \frac { \sum_ {n = 1} ^ {n _ {\mathrm{LSTM}}} y _ {n} ^ {\mathrm{LSTM}} (k) \mu_ {n} ^ {\mathrm{LSTM}} (k) + y _ {\mathrm{FP}} (k) \left(\sum_ {n = 1} ^ {n _ {\mathrm{FP}}} \mu_ {n} ^ {\mathrm{FP}} (k)\right)}{ \sum_ {n = 1} ^ {n _ {\mathrm{LSTM}}} \mu_ {n} ^ {\mathrm{LSTM}} (k) + \sum_ {n = 1} ^ {n _ {\mathrm{FP}}} \mu_ {n} ^ {\mathrm{FP}} (k)}\tag{25}
$$

In this study, we use trapezoidal, sigmoidal, and Gaussian membership functions. For trapezoidal functions, we have

$$
\mu_ {n} ^ {\mathrm{LSTM}} (k) = \mu^ {\mathrm{LSTM}} (\mathbf {X} _ {\mathrm{DF}} (k)) = \max \biggl (\min \biggl (\frac {\mathbf {X} _ {\mathrm{DF}} (k) - a _ {n}}{b _ {n} - a _ {n}}, 1, \frac {d _ {n} - \mathbf {X} _ {\mathrm{DF}} (k)}{d _ {n} - c _ {n}} \biggr), 0 \biggr)\tag{26}
$$

for sigmoidal ones, we write

$$
\mu_ {n} ^ {\mathrm{LSTM}} (k) = \mu^ {\mathrm{LSTM}} (\mathbf {X} _ {\mathrm{DF}} (k)) = \frac {1}{1 + e ^ {- a _ {n} (\mathbf {X} _ {\mathrm{DF}} (k) - b _ {n})}} \frac {1}{1 + e ^ {- c _ {n} (\mathbf {X} _ {\mathrm{DF}} (k) - d _ {n})}}\tag{27}
$$

and for Gaussian ones, we define

$$
\mu_ {n} ^ {\mathrm{LSTM}} (k) = \mu^ {\mathrm{LSTM}} (\mathbf {X} _ {\mathrm{DF}} (k)) = \exp \left(\frac {- (\mathbf {X} _ {\mathrm{DF}} (k) - a _ {n})}{2 b _ {n} ^ {2}}\right)\tag{28}
$$

The signal $\mathbf { \boldsymbol { x } } _ { \mathrm { D F } } ( \boldsymbol { k } ) = \boldsymbol { y } ( \boldsymbol { k } )$ or $\mathbf { \boldsymbol { x } } _ { \mathrm { D F } } ( \boldsymbol { k } ) = \boldsymbol { u } ( \boldsymbol { k } - 1 )$ defines the current operating point of the process. Parameters $a _ { n } , b _ { n } , c _ { n } , d _ { n }$ define the shape of membership functions used.

## 2.5. Model Development Procedure

The process for establishing the PIHNN model unfolds as follows:

1. We determine the number of distinct training datasets that can be derived from the process measurements.

2. We conduct training of the LSTM network for each training dataset.

3. We implement a discrete FP model of the process.

4. We select the initial shape and range of the membership function within the DF block.

5. We deliver the outputs of the LSTM sub-models and the output of the FP model as inputs of the DF block, where their fusion is carried out based on the current operational state of the process. This fusion process determines the output of the PIHNN model.

6. We assess the quality of PIHNN modeling. If it proves unsatisfactory, then it becomes necessary to modify the shape of the membership function.

7. We adjust the membership function’s shape, which can be executed manually, drawing upon expert knowledge, or using an optimization procedure.

The flow chart of the model development procedure is also presented in Figure 4.

![](images/288f50c17fdc87c0b77991a7ae6f65f28d8f0f30b46c91aee25d68ab71b57bad.jpg)  
Figure 4. Flow chart for development of PIHNN model.

## 3. LSTM PIHNN Models in Predictive Control

## 3.1. Basic Predictive Control Problem Formulation

This work utilizes the general MPC formulation [1,2]. Namely, at each discrete-time sampling instant $k ,$ where $k = 0 , 1 , 2 , . . . ,$ the MPC controller performs real-time calculations to determine the vector of decision variables. It is defined as the following current and future increments of the input variable

$$
\triangle \boldsymbol {u} (k) = [ \triangle u (k | k) \triangle u (k + 1 | k) \dots \triangle u (k + N _ {\mathrm{u}} - 1 | k) ] ^ {\mathrm{T}}\tag{29}
$$

The symbol $\triangle u ( k | k )$ represents the increment of the manipulated variable at time instant $k ,$ computed at the same time instant k. Similarly, the symbol $\triangle u ( k + 1 | k )$ corresponds to the increment of the manipulated variable at the future time instant k + 1, computed at the current time instant k. This notation extends to subsequent time instants as well. $N _ { \mathrm { u } }$ represents the control horizon, which determines the length of the MPC decision variable vector. The fundamental MPC optimization problem aims to minimize the predicted control error, minimize excessive increments of the manipulated variable, and satisfy constraints. Let us denote the set-point of the controlled variable for the future sampling instant $k + p$ known at the current instant k by $y ^ { \mathsf { s p } } ( k + p | k )$ and the corresponding prediction determined from the process model by ${ \hat { y } } ( k + p | k )$ . We consider the predictions and control errors on the prediction horizon N. As far as the magnitude constraints on the manipulated variable and the predicted controlled variable are concerned, they are represented by $u ^ { \mathrm { m i n } } , u ^ { \mathrm { m a x } }$ and $y ^ { \mathrm { m i n } } , y ^ { \mathrm { m a x } }$ , respectively. The fundamental MPC optimization task can be formulated as follows:

$$
\min _ {\triangle u (k)} \left\{J (k) = \sum_ {p = 1} ^ {N} \left(y ^ {\mathrm{sp}} (k + p | k) ^ {2} - \hat {y} (k + p | k)\right) + \lambda \sum_ {p = 0} ^ {N _ {\mathrm{u}} - 1} \left(\triangle u (k + p | k)\right) ^ {2} \right\}
$$

subject to

$$
\begin{array}{l} u ^ {\min} \leq u (k + p | k) \leq u ^ {\max},   p = 0, \ldots , N _ {\mathrm{u}} - 1 \\ \triangle u ^ {\min} \leq \triangle u (k + p | k) \leq \triangle u ^ {\max},   p = 0, \ldots , N _ {\mathrm{u}} - 1 \\ y ^ {\min} \leq \hat {y} (k + p | k) \leq y ^ {\max},   p = 1, \ldots , N. \end{array}\tag{30}
$$

In general, the predictions over the prediction horizon are obtained as

$$
\hat {y} (k + p | k) = y (k + p | k) + d (k)\tag{31}
$$

where the model output for the future discrete time $k + p ,$ , determined at the current time k, is denoted as $y ( k + p | k )$ . The unmeasured disturbance, that covers the model error and real disturbances that act on the controlled process, is computed as the difference between the measured value of the process controlled variable and its estimation obtained from the model. The MPC optimization problem (30) is solved online at each sampling instant, yielding the solution vector (29). According to the principle of repetitive control, the first element of the obtained solution vector is sent to the process and the whole procedure is repeated in the subsequent sampling instants.

## 3.2. Nonlinear MPC Optimization for PIHNN Models

Suppose a nonlinear model, e.g., an LSTM structure or the PIHNN model described in this work, is directly used to determine the predictions ${ \hat { y } } ( k + p | k )$ . The general MPC optimization problem (30) becomes nonlinear in that case. We will refer to such a control method as MPC-NO.

## 3.3. Quadratic MPC Optimization for PIHNN Models

In order to derive a computationally attractive alternative to the MPC-NO method, we derive an MPC with successive linearization of the predicted trajectory. Such an approach will make it possible to derive a quadratic optimization MPC task. We use the general approach to predicted trajectory linearization known as the MPC-NPLPT method, introduced in [19,20]. However, the application of an original PIHNN model structure requires careful derivation of the algorithm. Firstly, let us define the predicted trajectory of the controlled variable over the entire prediction horizon, i.e., the following vector:

$$
\hat {\boldsymbol {y}} (k) = [ \hat {y} (k + 1 | k) \dots \hat {y} (k + N | k) ] ^ {\mathrm{T}}\tag{32}
$$

In the MPC-NPLPT approach, linearization is performed along a trajectory of the manipulated variable defined over the control horizon. It has the following form:

$$
\boldsymbol {u} ^ {\text { traj }} (k) = \left[ u ^ {\text { traj }} (k | k) \dots u ^ {\text { traj }} (k + N _ {\mathbf {u}} - 1 | k) \right] ^ {\mathrm{T}}\tag{33}
$$

From the definition of the control horizon, it follows that $u ^ { \mathrm { t r a j } } ( k + p | k ) = u ^ { \mathrm { t r a j } } ( k + N _ { \mathrm { u } } - 1 | k )$ for $p = N _ { \mathrm { u } } , \ldots , N$ . The input trajectory (33) is utilized to determine the predicted trajectory of the controlled variable over the prediction horizon

$$
\hat {\boldsymbol {y}} ^ {\text { traj }} (k) = \left[ \hat {y} ^ {\text { traj }} (k + 1 | k) \dots \hat {y} ^ {\text { traj }} (k + N | k) \right] ^ {\mathrm{T}}\tag{34}
$$

For linearization, we use Taylor’s approach. Let us define the vector comprising the current and future values of the manipulated variable that correspond to the MPC decision variable vector (29)

$$
\boldsymbol {u} (k) = [ u (k | k) \dots u (k + N _ {\mathrm{u}} - 1 | k) ] ^ {\mathrm{T}}\tag{35}
$$

Taking advantage of the compact vector–matrix notation, the predicted trajectory, ${ \hat { y } } ( k )$ , is expressed as the following linear function of the vector (35):

$$
\hat {\boldsymbol {y}} (k) = \hat {\boldsymbol {y}} ^ {\mathrm{traj}} (k) + \boldsymbol {H} (k) (\boldsymbol {u} (k) - \boldsymbol {u} ^ {\mathrm{traj}} (k))\tag{36}
$$

The $N \times N _ { \mathrm { u } }$ matrix

$$
\boldsymbol {H} (k) = \frac {\mathrm{d} \hat {\boldsymbol {y}} ^ {\text { traj }} (k)}{\mathrm{d} \boldsymbol {u} ^ {\text { traj }} (k)}\tag{37}
$$

defines partial derivatives of the predicted controlled variable’s trajectory with respect to the future manipulated variable’s trajectory; both trajectories take into account the linearization conditions, so we have to utilize the trajectories $\hat { \boldsymbol y } ^ { \mathrm { t r a j } } ( \boldsymbol k )$ and ${ \pmb u } ^ { \mathrm { t r a j } } ( k )$ , respectively. The entries of the matrix $H ( k )$ are

$$
H _ {r + 1, p} (k) = \frac {\partial \hat {y} ^ {\mathrm{traj}} (k + p | k)}{\partial u ^ {\mathrm{traj}} (k + r | k)}\tag{38}
$$

for all predictions over the prediction horizon, i.e., $p = 1 , \ldots , N ,$ and all computed values of the manipulated variable over the entire control horizon, $\mathrm { i . e . , } r = 0 , \ldots , N _ { \mathrm { u } }$ . The link between the vectors ${ \pmb u } ( k )$ and $\triangle u ( k )$ is

$$
\boldsymbol {u} (k) = \boldsymbol {J} \triangle \boldsymbol {u} (k) + \boldsymbol {u} (k - 1)\tag{39}
$$

when the entries of the $N _ { \mathrm { u } } \times N _ { \mathrm { u } }$ auxiliary matrix J are defined as

$$
J _ {i, j} = \left\{ \begin{array}{l l} 0 & \text { if } i <   j \\ 1 & \text { if } i \geq j \end{array} \right.\tag{40}
$$

and the vector of length $N _ { \mathbf { u } }$ is

$$
\boldsymbol {u} (k - 1) = [ u (k - 1) \dots u (k - 1) ] ^ {\mathrm{T}}\tag{41}
$$

Using the linearized trajectory (36) and the rule (39), the general predictive control optimization task (30) is transformed to the subsequent quadratic optimization problem, as follows:

$$
\min _ {\triangle \boldsymbol {u} (k)} \left\{\left\| \boldsymbol {y} ^ {\mathrm{sp}} (k) - \boldsymbol {H} (k) \boldsymbol {J} \triangle \boldsymbol {u} (k) - \hat {\boldsymbol {y}} (k) - \boldsymbol {H} (k) (\boldsymbol {u} (k - 1) - \boldsymbol {u} (k)) \right\| ^ {2} + \| \triangle \boldsymbol {u} (k) \| _ {\Lambda} ^ {2} \right\}
$$

subject to

$$
\boldsymbol {u} ^ {\min} \leq J \triangle \boldsymbol {u} (k) + \boldsymbol {u} (k - 1) \leq \boldsymbol {u} ^ {\max}\tag{42}
$$

$$
\triangle \boldsymbol {u} ^ {\min} \leq \triangle \boldsymbol {u} (k) \leq \triangle \boldsymbol {u} ^ {\max}
$$

$$
\boldsymbol {y} ^ {\min} \leq \boldsymbol {H} (k) \boldsymbol {J} \triangle \boldsymbol {u} (k) + \hat {\boldsymbol {y}} (k) + \boldsymbol {H} (k) (\boldsymbol {u} (k - 1) - \boldsymbol {u} (k)) \leq \boldsymbol {y} ^ {\max}
$$

The definitions for all necessary symbols used in the above problem are

• Λ: a diagonal $N _ { \mathbf { u } } \times N _ { \mathbf { u } }$ matrix with diagonal entries equal to the weighting coefficient $\lambda ;$

$\pmb { u } ^ { \mathrm { m i n . } }$ : a vector of length $N _ { \mathbf { u } } ,$ where all elements are equal to $u ^ { \mathrm { m i n } } .$ ;

$u ^ { \mathrm { m a x . } }$ a vector of length $N _ { \mathbf { u } } ,$ where all elements are equal to $u ^ { \mathrm { m a x . } }$

$\triangle { u } ^ { \mathrm { m i n } }$ : a vector of length $N _ { \mathrm { u } } ,$ , where all elements are equal to $\triangle u ^ { \mathrm { { m i n } } }$

$\triangle u ^ { \mathrm { m a x } . }$ : a vector of length $N _ { \mathrm { u } } ,$ , where all elements are equal to $\triangle u ^ { \mathrm { m a x } } ;$

$y ^ { \mathrm { m i n . } }$ : a vector of length $N ,$ where all elements are equal to $y ^ { \mathrm { m i n } } .$ ;

$y ^ { \mathrm { m a x . } }$ a vector of length $N ,$ where all elements are equal to $y ^ { \mathrm { m a x } }$

## 3.4. PIHNN Prediction

Let us now discuss how the PIHNN model discussed in this work is utilized for MPC prediction, i.e., to calculate the predicted trajectory of the controlled variable defined by Equation (34). We use Equation (25) for the future time instant $k + p$ which gives

$$
y ^ {\mathrm{PIHNN}} (k + p | k) = \frac {\sum_ {n = 1} ^ {n _ {\mathrm{LSTM}}} y _ {n} ^ {\mathrm{LSTM}} (k + p | k) \mu_ {n} ^ {\mathrm{LSTM}} (k) + y _ {\mathrm{FP}} (k + p | k) \left(\sum_ {n = 1} ^ {n _ {\mathrm{FP}}} \mu_ {n} ^ {\mathrm{FP}} (k)\right)}{\sum_ {n = 1} ^ {n _ {\mathrm{LSTM}}} \mu_ {n} ^ {\mathrm{LSTM}} (k) + \sum_ {n = 1} ^ {n _ {\mathrm{FP}}} \mu_ {n} ^ {\mathrm{FP}} (k)}\tag{43}
$$

Taking advantage of Equation (31), the predictions are, therefore, expressed as

$$
\hat {y} ^ {\mathrm{PIHNN}} (k + p | k) = \frac {\sum_ {n = 1} ^ {n _ {\mathrm{LSTM}}} y _ {n} ^ {\mathrm{LSTM}} (k + p | k) \mu_ {n} ^ {\mathrm{LSTM}} (k) + y _ {\mathrm{FP}} (k + p | k) \left(\sum_ {n = 1} ^ {n _ {\mathrm{FP}}} \mu_ {n} ^ {\mathrm{FP}} (k)\right)}{\sum_ {n = 1} ^ {n _ {\mathrm{LSTM}}} \mu_ {n} ^ {\mathrm{LSTM}} (k) + \sum_ {n = 1} ^ {n _ {\mathrm{FP}}} \mu_ {n} ^ {\mathrm{FP}} (k)} + d (k)\tag{44}
$$

where the membership functions are defined by Equations (26), (27) or (28). Let us note that the predicted trajectory from the PIHNN model depends on the trajectories generated by both LSTM and FP sub-models. The disturbance (the prediction error) is determined as the difference between the measured process output and its estimation obtained from the model

$$
d (k) = y (k) - y ^ {\mathrm{PIHNN}} (k)\tag{45}
$$

where the signal $y ^ { \mathrm { P I H N N } } ( k )$ is found from Equation (31).

## 3.5. LSTM Model Prediction

For each LSTM sub-model, the calculations start with computing the predicted output of the gates. For this purpose, we use Equations (17)–(20) which yield the following

$$
i _ {n} (k + p | k) = \sigma \bigg (\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{i}} x _ {\mathrm{LSTM}} ^ {m} (k + p | k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{i}} h _ {m} (k - 1 + p | k)\right) + b _ {n} ^ {\mathrm{i}} \bigg)\tag{46}
$$

$$
f _ {n} (k + p | k) = \sigma \bigg (\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{f}} x _ {\mathrm{LSTM}} ^ {m} (k + p | k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{f}} h _ {m} (k - 1 + p | k)\right) + b _ {n} ^ {\mathrm{f}} \bigg)\tag{47}
$$

$$
g _ {n} (k + p | k) = \tanh \left(\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{g}} x _ {\text {LSTM}} ^ {m} (k + p | k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{g}} h _ {m} (k - 1 + p | k)\right) + b _ {n} ^ {\mathrm{g}}\right)\tag{48}
$$

$$
o _ {n} (k + p | k) = \sigma \bigg (\sum_ {m = 1} ^ {n _ {\mathrm{A}} + n _ {\mathrm{B}}} \left(w _ {n, m} ^ {\mathrm{o}} x _ {\mathrm{LSTM}} ^ {m} (k + p | k)\right) + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{o}} h _ {m} (k - 1 + p | k)\right) + b _ {n} ^ {\mathrm{o}} \bigg)\tag{49}
$$

Let us introduce auxiliary integer variables as follows: $I _ { \mathrm { u f } } ( p ) \ = \ \mathrm { m a x } ( \mathrm { m i n } ( p , n _ { \mathrm { B } } ) , 0 )$ $I _ { \mathrm { v f } } ( p ) = \mathrm { m i n } ( p - 1 , n _ { \mathrm { A } } )$ . We can represent gate predictions as

$$
\begin{array}{r l} & i _ {n} (k + p | k) = \sigma \bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{i}} u (k - m + p | k) + \sum_ {m = I _ {\mathrm{uf}} (p) + 1} ^ {n _ {\mathrm{B}}} w _ {n, I _ {\mathrm{uf}} (p) + m} ^ {\mathrm{i}} u (k - m + p) \\ & \qquad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{i}} \hat {y} ^ {\mathrm{LSTM}} (k - m + p | k) + \sum_ {m = I _ {\mathrm{yf}} (p) + 1} ^ {n _ {\mathrm{A}}} w _ {n, I _ {\mathrm{yf}} (p) + m} ^ {\mathrm{i}} y (k - m + p) \\ & \qquad + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} r _ {n, m} ^ {\mathrm{i}} h _ {m} (k + p - 1 | k) + b _ {n} ^ {\mathrm{i}} \bigg) \end{array}\tag{50}
$$

$$
\begin{array}{l} f _ {n} (k + p | k) = \sigma \bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{f}} u (k - m + p | k) + \sum_ {m = I _ {\mathrm{uf}} (p) + 1} ^ {n _ {\mathrm{B}}} w _ {n, I _ {\mathrm{uf}} (p) + m} ^ {\mathrm{f}} u (k - m + p) \\ \qquad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{f}} \hat {y} ^ {\text {LSTM}} (k - m + p | k) + \sum_ {m = I _ {\mathrm{yf}} (p) + 1} ^ {n _ {\mathrm{A}}} w _ {n, I _ {\mathrm{yf}} (p) + m} ^ {\mathrm{f}} y (k - m + p) \\ \qquad + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} r _ {n, m} ^ {\mathrm{f}} h _ {m} (k + p - 1 | k) + b _ {n} ^ {\mathrm{f}} \bigg) \end{array}\tag{51}
$$

$$
\begin{array}{l} g _ {n} (k + p | k) = \tanh \bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{g}} u (k - m + p | k) + \sum_ {m = I _ {\mathrm{uf}} (p) + 1} ^ {n _ {\mathrm{B}}} w _ {n, I _ {\mathrm{uf}} (p) + m} ^ {\mathrm{g}} u (k - m + p) \\ \qquad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{g}} \hat {y} ^ {\text {LSTM}} (k - m + p | k) + \sum_ {m = I _ {\mathrm{yf}} (p) + 1} ^ {n _ {\mathrm{A}}} w _ {n, I _ {\mathrm{yf}} (p) + m} ^ {\mathrm{g}} y (k - m + p) \\ \qquad + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} \left(r _ {n, m} ^ {\mathrm{g}} h _ {m} (k + p - 1 | k)\right) + b _ {n} ^ {\mathrm{g}} \bigg) \\ \text {and} \end{array}\tag{52}
$$

$$
\begin{array}{l} o _ {n} (k + p | k) = \sigma \bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{o}} u (k - m + p | k) + \sum_ {m = I _ {\mathrm{uf}} (p) + 1} ^ {n _ {\mathrm{B}}} w _ {n, I _ {\mathrm{uf}} (p) + m} ^ {\mathrm{o}} u (k - m + p) \\ \qquad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{o}} \hat {y} ^ {\text {LSTM}} (k - m + p | k) + \sum_ {m = I _ {\mathrm{yf}} (p) + 1} ^ {n _ {\mathrm{A}}} w _ {n, I _ {\mathrm{yf}} (p) + m} ^ {\mathrm{o}} y (k - m + p) \\ \qquad + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} r _ {n, m} ^ {\mathrm{o}} h _ {m} (k + p - 1 | k) + b _ {n} ^ {\mathrm{o}} \bigg) \end{array}\tag{53}
$$

Then, the predicted cell and hidden states can be determined from Equations (21) and (22)

$$
c _ {n} (k + p | k) = f _ {n} (k + p | k) c _ {n} (k - 1 + p | k) + i _ {n} (k + p | k) g _ {n} (k + p | k)\tag{54}
$$

$$
h _ {n} (k + p | k) = o _ {n} (k + p | k) \tanh \left(c _ {n} (k + p | k)\right)\tag{55}
$$

Let us stress that the above equations have to be used recurrently for $p = 1 , \ldots , N .$ . Finally, the predicted output of the i-th LSTM sub-model can be computed from Equation (23)

$$
\hat {y} _ {(i)} ^ {\mathrm{LSTM}} (k + p | k) = \sum_ {n = 1} ^ {n _ {\mathrm{N}}} w _ {\mathrm{y}} ^ {n} \Bigl (h _ {n} (k + p | k) \Bigr) + b _ {\mathrm{y}}\tag{56}
$$

which can be also expressed as

$$
\begin{array}{c} \hat {y} _ {(i)} ^ {\mathrm{LSTM}} (k + p | k) = \sum_ {n = 1} ^ {n _ {\mathrm{N}}} w _ {\mathrm{y}} ^ {n} \Big (o _ {n} (k + p | k) \tanh \big (f _ {n} (k + p | k) c _ {n} (k + p - 1 | k) \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad + i _ {n} (k + p | k) g _ {n} (k + p | k) \big) \Big) + b _ {\mathrm{y}} + d (k) \end{array}\tag{57}
$$

## 3.6. FP Model Prediction

Using Equations (4) and (5), we find model states and the output for the future time instant k + 1

$$
x _ {1} (k + p | k) = f _ {1} \left(x _ {1} (k + p - 1 | k), \dots , x _ {n _ {\mathrm{x}}} (k + p - 1 | k), u (k + p - 1 | k)\right)\tag{58}
$$

$$
x _ {n _ {\mathrm{x}}} (k + p | k) = f _ {n _ {\mathrm{x}}} \left(x _ {1} (k + p - 1 | k), \dots , x _ {n _ {\mathrm{x}}} (k + p | k), u (k + p | k)\right)\tag{59}
$$

$$
y ^ {\mathrm{FP}} (k + p | k) = g (x _ {1} (k + p - 1 | k), \ldots , x _ {n _ {\mathrm{x}}} (k + p - 1 | k))\tag{60}
$$

To simplify the following calculations, let us start with computing the prediction of the states for the time instant $k + p$

$$
\hat {x} _ {1} (k + 1 | k) = f _ {1} (k + 1 | k) = f _ {1} \left(x _ {1} (k), \dots , x _ {n _ {\mathrm{x}}} (k), u (k | k)\right) + \nu_ {1} (k)\tag{61}
$$

$$
\hat {x} _ {n _ {\mathrm{x}}} (k + 1 | k) = f _ {n _ {\mathrm{x}}} (k + 1 | k) = f _ {n _ {\mathrm{x}}} \left(x _ {1} (k), \dots , x _ {n _ {\mathrm{x}}} (k), u (k | k)\right) + \nu_ {n} (k)\tag{62}
$$

From Equation (6), we find the corresponding predicted controlled variable:

$$
\hat {y} ^ {\mathrm{FP}} (k + 1 | k) = g (k + 1 | k) + d (k) = g (\hat {x} _ {1} (k + 1 | k), \dots , \hat {x} _ {n _ {\mathrm{x}}} (k + 1 | k)) + d (k)\tag{63}
$$

Next, we can determine the predictions for the subsequent sampling instants:

$$
\hat {x} _ {1} (k + p | k) = f _ {1} (k + p | k) = f _ {1} (\hat {x} _ {1} (k + p - 1 | k), \ldots , \hat {x} _ {n _ {\mathrm{x}}} (k + p - 1 | k), u (k + p - 1 | k)) + \nu_ {1} (k)\tag{64}
$$

$$
\hat {x} _ {n _ {\mathrm{x}}} (k + p | k) = f _ {n _ {\mathrm{x}}} (k + p | k) = f _ {n _ {\mathrm{x}}} (\hat {x} _ {1} (k + p - 1 | k), \dots , \hat {x} _ {n _ {\mathrm{x}}} (k + p - 1 | k), u (k + p - 1 | k)) + \nu_ {n} (k)\tag{65}
$$

$$
\hat {y} ^ {\mathrm{FP}} (k + p | k) = g (k + p | k) + d (k) = g (\hat {x} _ {1} (k + p | k), \ldots , \hat {x} _ {n _ {\mathrm{x}}} (k + p | k)) + d (k)\tag{66}
$$

where $p = 2 , \ldots , N$ . The state and output disturbances (prediction errors), respectively, are computed as the measurements compared with the outputs of the corresponding model equations

$$
\nu_ {1} (k) = x _ {1} (k) - f _ {1} \left(x _ {1} (k - 1), \dots , x _ {n _ {\mathrm{x}}} (k - 1), u (k - 1)\right)\tag{67}
$$

$$
\nu_ {n} (k) = x _ {n} (k) - f _ {n} \left(x _ {1} (k - 1), \dots , x _ {n _ {\mathrm{x}}} (k - 1), u (k - 1)\right)\tag{68}
$$

$$
d (k) = y (k) - g (x _ {1} (k - 1), \dots , x _ {n} (k - 1))\tag{69}
$$

## 3.7. PIHNN Model Derivatives

The entries of the matrix $H ( k )$ (Equation (37)) are computed from Equation (38). Differentiation of Equation (44) yields

$$
\frac {\partial \hat {y} ^ {\text { PIHNN }} (k + p | k)}{\partial u (k + r | k)} = \frac {\sum_ {n = 1} ^ {n _ {\text { LSTM }}} \frac {\partial \hat {y} _ {n} ^ {\text { LSTM }} (k + p | k)}{\partial u (k + r | k)} \mu_ {n} ^ {\text { LSTM }} (k) + \frac {\partial \hat {y} ^ {\text { FP }} (k + p | k)}{\partial u (k + r | k)} \left(\sum_ {n = 1} ^ {n _ {\text { FP }}} \mu_ {n} ^ {\text { FP }} (k)\right)}{\sum_ {n = 1} ^ {n _ {\text { LSTM }}} \mu_ {n} ^ {\text { LSTM }} (k) + \sum_ {n = 1} ^ {n _ {\text { FP }}} \mu_ {n} ^ {\text { FP }} (k)}\tag{70}
$$

Let us note that the derivatives of the whole PIHNN model depend on the LSTM and FP sub-model derivatives.

## 3.8. LSTM Model Derivatives

Derivatives for LSTM sub-models are calculated by differentiating Equation (57)

$$
\frac {\partial \hat {y} _ {(i)} ^ {\mathrm{LSTM}} (k + p | k)}{\partial u (k + r | k)} = \sum_ {n = 1} ^ {n _ {\mathrm{N}}} w _ {\mathrm{n}} ^ {y} \frac {\partial h _ {n} (k + p | k)}{\partial u (k + r | k)}\tag{71}
$$

For all $p = 1 , \ldots , N$ and $r = 0 , \ldots , N _ { \mathrm { u } } - 1$ , the subsequent step involves the application of the chain rule of differentiation. Initially, it is imperative to determine the derivatives of gates $i , f , g$ and o. We proceed to differentiate Equation (50):

$$
\begin{array}{r l} \frac {\partial i _ {n} (k + p | k)}{\partial u (k + r | k)} & = i _ {n} (k + p | k) (1 - i _ {n} (k + p | k)) \Bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{i}} \frac {\partial u (k - m + p | k)}{\partial u (k + r | k)} \\ & \quad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{i}} \frac {\partial \hat {y} ^ {\mathrm{LSTM}} (k - m + p | k)}{\partial u (k + r | k)} + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} r _ {n, m} ^ {\mathrm{i}} \frac {\partial h _ {m} (k + p - 1 | k)}{\partial u (k + r | k)} \Bigg) \end{array}\tag{72}
$$

Equation (51) gives

$$
\begin{array}{r l} \frac {\partial f _ {n} (k + p | k)}{\partial u (k + r | k)} & = f _ {n} (k + p | k) (1 - f _ {n} (k + p | k)) \Bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{f}} \frac {\partial u (k - m + p | k)}{\partial u (k + r | k)} \\ & \quad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{f}} \frac {\partial \hat {y} ^ {\mathrm{LSTM}} (k - m + p | k)}{\partial u (k + r | k)} + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} r _ {n, m} ^ {\mathrm{f}} \frac {\partial h _ {m} (k + p - 1 | k)}{\partial u (k + r | k)} \Bigg) \end{array}\tag{73}
$$

from Equation (52), we obtain

$$
\begin{array}{r l} \frac {\partial g _ {n} (k + p | k)}{\partial u (k + r | k)} & = (1 - g _ {n} ^ {2} (k + 1 | k)) \Bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{g}} \frac {\partial u (k - m + p | k)}{\partial u (k + r | k)} \\ & \quad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{g}} \frac {\partial \hat {y} ^ {\mathrm{LSTM}} (k - m + p | k)}{\partial u (k + r | k)} + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} r _ {n, m} ^ {\mathrm{g}} \frac {\partial h _ {m} (k + p - 1 | k)}{\partial u (k + r | k)} \Bigg) \end{array}\tag{74}
$$

Finally, using Equation (53), we derive

$$
\begin{array}{r l} \frac {\partial o _ {n} (k + p | k)}{\partial u (k + r | k)} & = o _ {n} (k + p | k) (1 - o _ {n} (k + p | k)) \Bigg (\sum_ {m = 1} ^ {I _ {\mathrm{uf}} (p)} w _ {n, m} ^ {\mathrm{o}} \frac {\partial u (k - m + p | k)}{\partial u (k + r | k)} \\ & \quad + \sum_ {m = 1} ^ {I _ {\mathrm{yf}} (p)} w _ {n, n _ {\mathrm{B}} + m} ^ {\mathrm{o}} \frac {\partial \hat {y} ^ {\mathrm{LSTM}} (k - m + p | k)}{\partial u (k + r | k)} + \sum_ {m = 1} ^ {n _ {\mathrm{N}}} r _ {n, m} ^ {\mathrm{o}} \frac {\partial h _ {m} (k + p - 1 | k)}{\partial u (k + r | k)} \Bigg) \end{array}\tag{75}
$$

The following step involves computing the derivative of the cell state c using Equation (54)

$$
\begin{array}{c} \frac {\partial c _ {n} (k + p | k)}{\partial u (k + r | k)} = \frac {\partial f _ {n} (k + p | k)}{\partial u (k + r | k)} c _ {n} (k + p - 1 | k) + f _ {n} (k + p | k) \frac {\partial c _ {n} (k + p - 1 | k)}{\partial u (k + r | k)} \\ + \frac {\partial i _ {n} (k + p | k)}{\partial u (k + r | k)} g _ {n} (k + p | k) + i _ {n} (k + p | k) \frac {\partial g _ {n} (k + p | k)}{\partial u (k + r | k)} \end{array}\tag{76}
$$

and from Equation (54), we can derive the derivatives of the hidden state h

$$
\begin{array}{l} \frac {\partial h _ {n} (k + p | k)}{\partial u (k + r | k)} = \frac {\partial o _ {n} (k + p | k)}{\partial u (k + r | k)} \tanh \left(c _ {n} (k + p | k)\right) \\ \qquad + o _ {n} (k + p | k) \Big (1 - \tanh ^ {2} \left(c _ {n} (k + p | k)\right) \Big) \frac {\partial c _ {n} (k + p | k)}{\partial u (k + r | k)} \end{array}\tag{77}
$$

## 3.9. FP Model Derivatives

We start by finding derivatives of the predicted state variables for the sampling instant $k + 1$ . Differentiating Equations (61) and (62), we obtain

$$
\begin{array}{l} \frac {\partial \hat {x} _ {1} (k + 1 | k)}{\partial u (k + r | k)} = \sum_ {i = 1} ^ {n _ {\mathrm{x}}} \frac {\partial f _ {1} (x _ {1} (k) , \ldots , x _ {n _ {\mathrm{x}}} (k) , u (k | k))}{\partial x _ {i} (k)} \frac {\partial x _ {i} (k)}{\partial u (k + r | k)} \\ \qquad + \frac {\partial f _ {1} (x _ {1} (k) , \ldots , x _ {n _ {\mathrm{x}}} (k) , u (k | k))}{\partial u (k | k)} \frac {\partial u (k | k)}{\partial u (k + r | k)} \\ \vdots \\ \frac {\partial \hat {x} _ {n _ {\mathrm{x}}} (k + 1 | k)}{\partial u (k + r | k)} = \sum_ {i = 1} ^ {n _ {\mathrm{x}}} \frac {\partial f _ {n _ {\mathrm{x}}} (x _ {1} (k) , \ldots , x _ {n _ {\mathrm{x}}} (k) , u (k | k))}{\partial x _ {i} (k)} \frac {\partial x _ {i} (k)}{\partial u (k + r | k)} \\ \qquad + \frac {\partial f _ {n _ {\mathrm{x}}} (x _ {1} (k) , \ldots , x _ {n _ {\mathrm{x}}} (k) , u (k | k))}{\partial u (k | k)} \frac {\partial u (k | k)}{\partial u (k + r | k)} \end{array}\tag{78}
$$

(79)

Knowing that

$$
\frac {\partial u ^ {\text { traj }} (k + p | k)}{\partial u ^ {\text { traj }} (k + r | k)} = \left\{ \begin{array}{l l} 1 & \text { if   } p = r \text {   or   } (p > r \text {   and   } r = N _ {\mathrm{u}} - 1) \\ 0 & \text { otherwise } \end{array} \right.\tag{80}
$$

we can simplify Equations (78) and (79) to

$$
\begin{array}{c} \frac {\partial \hat {x} _ {1} (k + 1 | k)}{\partial u (k + r | k)} = \frac {\partial f _ {1} (x _ {1} (k) , \ldots , x _ {n _ {\mathrm{x}}} (k) , u (k | k))}{\partial u (k | k)} \frac {\partial u (k | k)}{\partial u (k + r | k)} \\ \vdots \\ \frac {\partial \hat {x} _ {n _ {\mathrm{x}}} (k + 1 | k)}{\partial u (k + r | k)} = \frac {\partial f _ {n _ {\mathrm{x}}} (x _ {1} (k) , \ldots , x _ {n _ {\mathrm{x}}} (k) , u (k | k))}{\partial u (k | k)} \frac {\partial u (k | k)}{\partial u (k + r | k)} \end{array}\tag{81}
$$

(82)

The next step is to find the derivative for the FP model states and the controlled variable for prediction at the sampling instant $k + p ,$ where $p = 2 , \ldots , N .$ . From Equation (63), we have

$$
\frac {\partial \hat {y} ^ {\mathrm{FP}} (k + 1 | k)}{\partial u (k + r | k)} = \sum_ {i = 1} ^ {n _ {\mathrm{x}}} \frac {\partial g (\hat {x} _ {1} (k + 1 | k) , \dots , \hat {x} _ {n _ {\mathrm{x}}} (k + 1 | k))}{\partial \hat {x} _ {i} (k + 1 | k)} \frac {\partial \hat {x} _ {i} (k + 1 | k)}{\partial u (k + r | k)}\tag{83}
$$

Next, we can determine the derivatives when $p = 2 , \ldots , N$ . We start with the state variables. From Equations (64) and (65), we obtain

$$
\begin{array}{l} \frac {\partial \hat {x} _ {1} (k + p | k)}{\partial u (k + r | k)} = \sum_ {i = 1} ^ {n _ {\mathrm{x}}} \frac {\partial f _ {1} (\hat {x} _ {1} (k + p - 1 | k) , \ldots , \hat {x} _ {n _ {\mathrm{x}}} (k + p - 1 | k) , u (k + p - 1 | k))}{\partial \hat {x} _ {i} (k + p - 1 | k)} \\ \qquad \times \frac {\partial \hat {x} _ {i} (k + p - 1 | k)}{\partial u (k + r | k)} \\ \qquad + \frac {\partial f _ {1} (\hat {x} _ {1} (k + p - 1 | k) , \ldots , \hat {x} _ {n _ {\mathrm{x}}} (k + p - 1 | k) , u (k + p - 1 | k))}{\partial u (k + p - 1 | k)} \\ \qquad \times \frac {\partial u (k + p - 1 | k)}{\partial u (k + r | k)} \end{array}\tag{84}
$$

$$
\begin{array}{l} \frac {\partial \hat {x} _ {n _ {\mathrm{x}}} (k + p | k)}{\partial u (k + r | k)} = \sum_ {i = 1} ^ {n _ {\mathrm{x}}} \frac {\partial f _ {n _ {\mathrm{x}}} (\hat {x} _ {1} (k + p - 1 | k) , \ldots , \hat {x} _ {n _ {\mathrm{x}}} (k + p - 1 | k) , u (k + p - 1 | k))}{\partial \hat {x} _ {i} (k + p - 1 | k)} \\ \qquad \qquad \times \frac {\partial \hat {x} _ {i} (k + p - 1 | k)}{\partial u (k + r | k)} \\ \qquad + \frac {\partial f _ {n _ {\mathrm{x}}} (\hat {x} _ {1} (k + p - 1 | k) , \ldots , \hat {x} _ {n _ {\mathrm{x}}} (k + p - 1 | k) , u (k + p - 1 | k))}{\partial u (k + p - 1 | k)} \\ \qquad \times \frac {\partial u (k + p - 1 | k)}{\partial u (k + r | k)} \end{array}\tag{85}
$$

Finally, we can find the predictions of the FP sub-model output using Equation (66)

$$
\frac {\partial \hat {y} ^ {\mathrm{FP}} (k + p | k)}{\partial u (k + r | k)} = \sum_ {i = 1} ^ {n _ {\mathrm{x}}} \frac {\partial g (\hat {x} _ {1} (k + p | k) , \dots , \hat {x} _ {n _ {\mathrm{x}}} (k + p | k))}{\partial \hat {x} _ {i} (k + p | k)} \frac {\partial \hat {x} _ {i} (k + p | k)}{\partial u (k + r | k)}\tag{86}
$$

## 4. Results

## 4.1. Polymerization Process Description

The process under study is a polymerization reactor [43] that is frequently used as a benchmark to assess the usefulness of models and control methods, e.g., [20,27]. This process is characterized by a single input, representing the initiator’s flow rate, denoted as $F _ { \mathrm { I } } ( \mathbf { m } ^ { 3 } \mathbf { h } ^ { - 1 } )$ ). Likewise, it has a single output, the number average molecular weight (NAMW) (kg kmol<sup>−1</sup>). Both input and output signals have been appropriately normalized to facilitate the training of neural networks. The scaling is defined as follows: $u = 1 0 0 \big ( F _ { \mathrm { I } } - \overline { { F } } _ { \mathrm { I } } \big )$ and $y = 0 . 0 0 0 1 \mathrm { ( N A M W - N A M W ) }$ ). The values at the nominal operating point are $\overline { { F } } _ { \mathrm { I } } = 0 . 0 1 6 7 8 3$ and $\overline { { \mathrm { N A M W } } } = 2 0 { , } 0 0 0$ . The polymerization process operates with a sampling time $T = 1$ .8 seconds.

Let us note that the predictions determined from the LSTM sub-models are universal, as derived in Section 3.5. Similarly, let us note that the derivatives matrix determined from the LSTM sub-models are universal, as derived in Section 3.8. Hence, it is only necessary to derive specific equations for prediction using the specific first-principle model of the process. Next, we have to derive equations for the derivatives matrix.

## 4.2. First-Principle Model for Polymerization Process and Its Use in MPC

The continuous-time first-principle model of the polymerization process [43] is discreticized using the Euler method. The discrete-time model has the following form:

$$
x _ {1} (k) = T \left(6 0 - 1 0 x _ {1} (k - 1) \sqrt {x _ {2} (k - 1)}\right) + x _ {1} (k - 1)\tag{87}
$$

$$
x _ {2} (k) = T \left(8 0 u (k - 1) - 1 0. 1 0 2 2 x _ {2} (k - 1)\right) + x _ {2} (k - 1)\tag{88}
$$

$$
x _ {3} (k) = T \left(0. 0 0 2 4 1 2 1 x _ {1} (k - 1) \sqrt {x _ {2} (k - 1) + 0 . 1 1 2 1 9 1 x _ {2} (k - 1) - 1 0 x _ {3} (k - 1)}\right)
$$

$$
+ x _ {3} (k - 1)\tag{89}
$$

$$
x _ {4} (k) = T \left(2 4 5. 9 7 8 x _ {1} (k - 1) \sqrt {x _ {2} (k - 1)} - 1 0 x _ {4} (k - 1)\right) + x _ {4} (k - 1)\tag{90}
$$

$$
y ^ {\mathrm{FP}} (k) = \frac {x _ {4} (k)}{x _ {3} (k)}\tag{91}
$$

where the model parameters are: $p _ { 1 } = 6 0 T , p _ { 2 } = - 1 0 T , p _ { 3 } = 8 0 T , p _ { 4 } = - 1 0 . 1 0 2 2 T + 1$ $p _ { 5 } = 0 . 0 0 2 4 1 2 1 T , p _ { 6 } = 0 . 1 1 2 1 9 1 T , p _ { 7 } = - 1 0 T + 1 , p _ { 8 } = 2 4 5 . 9 7 8 T , p _ { 9 } = - 1 0 T + 1 . 7 7 7 T ,$

It is important to note that to emulate the imperfections and inaccuracies of the FP model, we introduced a 20 percent increase to the gain of the model during the simulation experiments, i.e.,

$$
y _ {\mathrm{disturbed}} ^ {\mathrm{FP}} (k) = 1. 2 y ^ {\mathrm{FP}} (k) = 1. 2 \frac {x _ {4} (k)}{x _ {3} (k)}\tag{92}
$$

For the PIHNN model used in our MPC algorithm, we have to derive equations for the prediction using the specific FP model of the considered benchmark system and the general rules formulated in Section 3.6. They will allow to calculate the predicted trajectory $\bar { \hat { y } } ^ { \mathrm { t r a j } } ( k )$ , as defined by Equation (34). We start with determining prediction equations when $p = 1$ . From Equations (87)–(91), we obtain

$$
\hat {x} _ {1} (k + 1 | k) = p _ {1} + p _ {2} x _ {1} (k) \sqrt {x _ {2} (k) + x _ {1} (k) + \nu_ {1} (k)}\tag{93}
$$

$$
\hat {x} _ {2} (k + 1 | k) = p _ {3} u (k | k) + p _ {4} x _ {2} (k) + \nu_ {2} (k)\tag{94}
$$

$$
\hat {x} _ {3} (k + 1 | k) = p _ {5} x _ {1} (k) \sqrt {x _ {2} (k) + p _ {6} x _ {2} (k) + p _ {7} x _ {3} (k) + \nu_ {3} (k)}\tag{95}
$$

$$
\hat {x} _ {4} (k + 1 | k) = p _ {8} x _ {1} (k) \sqrt {x _ {2} (k) + p _ {9} x _ {4} (k) + \nu_ {4} (k)}\tag{96}
$$

$$
\hat {y} ^ {\mathrm{FP}} (k + 1 | k) = \frac {\hat {x} _ {4} (k + 1 | k)}{\hat {x} _ {3} (k + 1 | k)} + d (k)\tag{97}
$$

The state and output disturbances are derived from the general Equations (67)–(69), respectively, which gives

$$
\nu_ {1} (k) = x _ {1} (k) - p _ {1} + p _ {2} x _ {1} (k - 1) \sqrt {x _ {2} (k - 1)} + x _ {1} (k - 1)\tag{98}
$$

$$
\nu_ {2} (k) = x _ {2} (k) - p _ {3} u (k - 1) + p _ {4} x _ {2} (k - 1)\tag{99}
$$

$$
\nu_ {3} (k) = x _ {3} (k) - p _ {5} x _ {1} (k - 1) \sqrt {x _ {2} (k - 1)} + p _ {6} x _ {2} (k - 1) + p _ {7} x _ {3} (k - 1)\tag{100}
$$

$$
\nu_ {4} (k) = x _ {4} (k) - p _ {8} x _ {1} (k - 1) \sqrt {x _ {2} (k - 1)} + p _ {9} x _ {4} (k - 1)\tag{101}
$$

$$
d (k) = y (k) - \frac {x _ {4} (k)}{x _ {3} (k)}\tag{102}
$$

Next, we find the equations for state and output predictions for $p = 2 , \ldots , N$

$$
\hat {x} _ {1} (k + p | k) = p _ {1} + p _ {2} \hat {x} _ {1} (k + p - 1 | k) \sqrt {\hat {x} _ {2} (k + p - 1 | k)} + \hat {x} _ {1} (k + p - 1 | k) + \nu_ {1} (k)\tag{103}
$$

$$
\hat {x} _ {2} (k + p | k) = p _ {3} u (k + p - 1 | k) + p _ {4} \hat {x} _ {2} (k + p - 1 | k) + \nu_ {2} (k)\tag{104}
$$

$$
\hat {x} _ {3} (k + p | k) = p _ {5} \hat {x} _ {1} (k + p - 1 | k) \sqrt {x _ {2} (k + p - 1 | k)} + p _ {6} x _ {2} (k + p - 1 | k)
$$

$$
+ p _ {7} \hat {x} _ {3} (k + p - 1 | k) + \nu_ {3} (k)\tag{105}
$$

$$
\hat {x} _ {4} (k + p | k) = p _ {8} \hat {x} _ {1} (k + p - 1 | k) \sqrt {\hat {x} _ {2} (k + p - 1 | k)} + p _ {9} \hat {x} _ {4} (k + p - 1 | k) + \nu_ {4} (k)\tag{106}
$$

$$
\hat {y} ^ {\mathrm{FP}} (k + p | k) = \frac {\hat {x} _ {4} (k + p | k)}{\hat {x} _ {3} (k + p | k)} + d (k)\tag{107}
$$

Using the above predictions generated by the FP model, we have to determine derivatives of the predicted trajectory of the controlled variable with respect to the trajectory of the manip ulated variable, i.e., the derivative matrix $H ( k )$ , as defined by Equation (38). For this purpose, we use the general rules formulated in Section 3.9. We consider Equations (81) and ( 82) and we obtain

$$
\frac {\partial \hat {x} _ {1} (k + 1 | k)}{\partial u (k + r | k)} = 0\tag{108}
$$

$$
\frac {\partial \hat {x} _ {2} (k + 1 | k)}{\partial u (k + r | k)} = 8 0 T \frac {\partial u (k | k)}{\partial u (k + r | k)}\tag{109}
$$

$$
\frac {\partial \hat {x} _ {3} (k + 1 | k)}{\partial u (k + r | k)} = 0\tag{110}
$$

$$
\frac {\partial \hat {x} _ {4} (k + 1 | k)}{\partial u (k + r | k)} = 0\tag{111}
$$

Equation (83) allows us to express the output derivatives as

$$
\frac {\partial \hat {y} ^ {\mathrm{FP}} (k + p | k)}{\partial u (k + r | k)} = 0\tag{112}
$$

Finally, we use Equations (84)–(86) to determine the state variable and output derivatives, respectively

$$
\frac {\partial \hat {x} _ {1} (k + p | k)}{\partial u (k + r | k)} = p _ {2} \frac {\partial \hat {x} _ {1} (k + p - 1 | k)}{\partial u (k + r | k)} \sqrt {\hat {x} _ {2} (k + p - 1 | k)} - 0. 5 p _ {2} \hat {x} _ {1} (k + p - 1 | k)\tag{113}
$$

$$
\times (\hat {x} _ {2} (k + p - 1 | k)) ^ {- 2} \frac {\partial \hat {x} _ {2} (k + p - 1 | k)}{\partial u (k + r | k)} + \frac {\hat {x} _ {1} (k + p - 1 | k)}{\partial u (k + r | k)}
$$

$$
\frac {\partial \hat {x} _ {2} (k + p | k)}{\partial u (k + r | k)} = p _ {3} \frac {\partial u (k + p - 1 | k)}{\partial u (k + r | k)} + p _ {4} \frac {\hat {x} _ {2} (k + p - 1 | k)}{\partial u (k + r | k)}\tag{114}
$$

$$
\begin{array}{l} \frac {\partial \hat {x} _ {3} (k + p | k)}{\partial u (k + r | k)} = p _ {5} \frac {\partial \hat {x} _ {1} (k + p - 1 | k)}{\partial u (k + r | k)} \sqrt {\hat {x} _ {2} (k + p - 1 | k)} - 0. 5 p _ {5} \hat {x} _ {1} (k + p - 1 | k) \\ \qquad \times (\hat {x} _ {2} (k + p - 1 | k)) ^ {- 2} \frac {\partial \hat {x} _ {2} (k + p - 1 | k)}{\partial u (k + r | k)} + p _ {6} \frac {\hat {x} _ {2} (k + p - 1 | k)}{\partial u (k + r | k)} \\ \qquad + p _ {7} \frac {\hat {x} _ {3} (k + p - 1 | k)}{\partial u (k + r | k)} \end{array}\tag{115}
$$

$$
\frac {\partial \hat {x} _ {4} (k + p | k)}{\partial u (k + r | k)} = p _ {8} \frac {\partial \hat {x} _ {1} (k + p - 1 | k)}{\partial u (k + r | k)} \sqrt {\hat {x} _ {2} (k + p - 1 | k)} - 0. 5 p _ {8} \hat {x} _ {1} (k + p - 1 | k)
$$

$$
\times (\hat {x} _ {2} (k + p - 1 | k)) ^ {- 2} \frac {\partial \hat {x} _ {2} (k + p - 1 | k)}{\partial u (k + r | k)} + p _ {9} \frac {\hat {x} _ {4} (k + p - 1 | k)}{\partial u (k + r | k)}\tag{116}
$$

$$
\frac {\partial \hat {y} ^ {\mathrm{FP}} (k + p | k)}{\partial u (k + r | k)} = \frac {1}{\hat {x} _ {3} (k + p | k) ^ {2}} \bigg (\frac {\partial \hat {x} _ {4} (k + p | k)}{\partial u (k + r | k)} \hat {x} _ {3} (k + p | k) - \hat {x} _ {4} (k + p | k) \frac {\partial \hat {x} _ {3} (k + p | k)}{\partial u (k + r | k)} \bigg)\tag{117}
$$

for all $p = 2 , \ldots , N$ and $r = 0 , \ldots , N _ { \mathrm { u } } - 1$

## 4.3. LSTM Model for Polymerization Process

Two separate training datasets have been collected from the simulated process (i.e., from simulation of the continuous-time first-principle models), for different operating conditions, as follows:

1. dataset 1 has been collected for the range of the manipulated variable $0 . 0 0 3 < F _ { \mathrm { I } } <$ 0.0129, which results in the controlled variable $2 . 7 8 \times 1 0 ^ { 4 } < \mathrm { N A M W } < 4 . 5 5 \times 1 0 ^ { 4 }$

2. dataset 2 has been collected for $0 . 0 5 < F _ { \mathrm { I } } < 0 . 0 6$ which results in $1 . 4 1 \times 1 0 ^ { 4 } <$ $\mathrm { N A M W } < 1 . 5 4 \times 1 0 ^ { 4 }$

The datasets were then used to train two LSTM models, denoted thereafter as LSTM1 and LSTM2. Both models have been trained with the same parameters:

• the number of neurons $n _ { \mathrm { N } } = 7 ;$

• the order of dynamics $n _ { \mathrm { A } } = 0 , n _ { \mathrm { B } } = 1$

LSTM models have been trained in MATLAB on a PC equipped with an Nvidia GeForce 970 GTX GPU, an Intel i5-3450 CPU and 16 GB of RAM. We have employed the Adam optimization algorithm with a learning rate of 0.001 and a maximum of 1000 training epochs.

## 4.4. Modeling Quality of LSTM and FP Models

The modeling quality of all sub-models developed for the polymerization process can be compared in Figure 5. In this comparison, we can see the individual outputs of all sub-models when operating independently with the test dataset. LSTM1, trained predominantly with data featuring large NAMW values, unsurprisingly demonstrates exceptional performance when dealing with such high NAMW values. However, the model’s capability to provide correct outputs diminishes when it encounters data not present in the training dataset. Conversely, LSTM2, trained with low NAMW values, excels when the NAMW values are indeed low. However, it exhibits subpar performance when attempting to model high NAMW values. Notably, in the FP model with the increased gain performs poorly across the entire range of NAMW values.

![](images/3a08263054b82367aa27bd94fce8afca38e38c1cc1af92ba62ddc24c80c0bc7e.jpg)  
Figure 5. A total of 1000 samples of the validation dataset vs. outputs of two local LSTM sub-models and the FP model with an incorrect gain.

## 4.5. Development of PIHNN Models

Once all the sub-models have been prepared, the next step to design the PIHNN model is to develop the DF block. Various membership function shapes have been tested, i.e.:

PIHNN model ver. 1—initial trapezoidal functions;

• PIHNN model ver. 2—optimized trapezoidal functions;

• PIHNN model ver. 3—initial trapezoidal functions;

• PIHNN model ver. 4—optimized trapezoidal functions;

• PIHNN model ver. 5—initial trapezoidal functions;

• PIHNN model ver. 6—optimized trapezoidal functions.

The membership functions are depicted in Figure 6. Our understanding of the sub models has guided the initial choices of these shapes. The plots display fuzzified variable values along the horizontal axis, specifically representing the NAMW output of the poly merization reactor. Along the vertical axis, one can find the membership function values. Each membership function corresponds to a particular model. LSTM1, which was trained on data with large NAMW values, is most effective when dealing with large NAMW values. The blue membership functions on the plot indicate the range of NAMW values for which prioritizing the use of the LSTM1 model is recommended. LSTM2, characterized by yellow membership functions, is best suited for NAMW values close to the data in its training set, which primarily includes small values of NAMW. In scenarios where NAMW values fall outside the data ranges of both training sets, the most reliable choice is to utilize the FP model, represented by orange membership functions. Once the initial shapes have been determined, the subsequent step involves utilizing an optimization procedure to fine-tune these shapes. The procedure starts with initial membership function shapes, using Levenberg–Marquardt to minimize the overall error of the PIHNN model.

![](images/0b4da1b26118b8df7875eec7f0077970922fd8e377ffbe6ccedc036e15466269.jpg)

![](images/8e291f5e56c860657f6823350fb3ba5be051d05659e7d1997b39fea4e5f86831.jpg)

![](images/cea059f8d61db85278b9b57a761224ef8fa4e2b9516ea7ff3a51134e902004fe.jpg)

(a)  
![](images/bcedac30a09d93e9f94a1086de3e5f0a57845a59911985e4052feabf1d74bfeb.jpg)

![](images/dba0e946d99c962f8d6f6487ffcc37484b3875424235a52e0f7cd110b4377c8a.jpg)

(b)  
![](images/51795a1ba1fd2e2338c78b070643da75232b50da254088788cd8cf9032ec0614.jpg)  
(c)  
Figure 6. Membership functions for considered fuzzy PIHNN models: fuzzy set 1 (blue), fuzzy set 2 (orange), fuzzy set 3 (yellow). (a) Initial (left) and optimized (right) trapezoidal membership functions; (b) initial (left) and optimized (right) sigmoidal membership functions; (c) initial (left) and optimized (right) Gauss membership functions.

## 4.6. PIHNN Modeling Quality

The results of the polymerization reactor modeling experiments are presented in Figures 7–9. These figures illustrate the initial 1500 steps of the simulation. Each figure showcases the outputs of two PIHNN models: one with the initial membership function shapes (orange) and the other with optimized (yellow) membership function shapes. These results are compared to the data from the test set. Figure 7 presents the use of the most straightforward decision blocks with trapezoidal membership functions. Even this simplest approach enables the PIHNN model to outperform individual sub-models. The initial shape of the membership function allows the PIHNN structure to represent the data effectively for both small and large values of NAMW. In cases with intermediate values of NAMW, the PIHNN model averages the outputs of the sub-models, while model output still exhibits some deviation from the test data, there is a clear improvement over the FP model. The model with a tuned shape has lower error overall; however, it tends to have poorer modeling quality for both large and small values of NAMW in comparison to the LSTM sub-models.

Figure 8 illustrates the utilization of sigmoidal membership functions in the DF block of the PIHNN model. Here, the sigmoidal shape allows the PIHNN to excel in modeling small, large, and intermediate NAMW values. Importantly, the tendency to average out intermediate NAMW values, as previously observed with the trapezoidal DF model, has been eliminated with the sigmoidal DF PIHNN model. Adopting sigmoidal functions has resulted in highly accurate modeling of medium NAMW values. When comparing the output signals of the models with the initial and tuned shapes of the membership functions, they exhibit minimal differences, with only slight variations noticeable for intermediate values of NAMW.

![](images/18aa3b46ddcc10e0013ae304997921e595678b80d403101d8a89f38e078fb732.jpg)

Figure 7. A total of 1000 samples of the validation dataset vs. the output of initial and optimized fuzzy PIHNN structures with trapezoidal MFs (PIHNN models ver. 1 and ver. 2).  
![](images/95298be49185007ad17b66c6c54df3ab8ff8f82548cff4fd374e102fd6e7528c.jpg)  
Figure 8. A total of 1000 samples of the validation dataset vs. the output of initial and optimized fuzzy PIHNN structures with sigmoid MFs (PIHNN models ver. 3 and ver. 4).

Finally, Figure 9 presents the utilization of Gaussian membership functions in a DF block of PIHNN model. Here, one can observe that the Gaussian decision model tends to average the values of the three sub-models across the entire spectrum of NAMW variability. This effect is particularly evident in the model with the initial shape of the membership function, where, for large values of NAMW, the model noticeably diverges from the data. As a result, for large NAMW values, PIHNN gives worse results than the independent LSTM1 submodel. Low and intermediate NAMW values are subject to much lower modeling errors. Although optimizing the shape mitigated this averaging effect somewhat, the model’s output still exhibits relatively large errors.

![](images/0e156deac4e12473175807acc438079b342b6f28acfb122c743225a169a00eed.jpg)  
Figure 9. A total of 1000 samples of the validation dataset vs. the output of initial and optimized fuzzy PIHNN structures with Gauss MFs (PIHNN models ver. 5 and ver. 6).

## 4.7. Validation of MPC Algorithms Using PIHNN Models

The PIHNN model, in six different versions, has been implemented in MPC algorithms. We compare the results obtained from two types of controllers: one with nonlinear optimization (MPC-NO) and the second one recommended in this work, involving linearization around the prediction trajectory (MPC-NPLPT). Table 1 compares the control errors determined for these controllers. First, it is worth noting that the best control quality is achieved for models utilizing DF with Gaussian function shapes. Models employing trapezoidal functions exhibited slightly higher errors, while the poorest performance was observed in models with sigmoidal-shaped functions. This observation may seem counterintuitive, considering that models with sigmoidal membership functions have smaller modeling errors compared to models with Gaussian ones. It is important to stress that the shape of the closed-loop output trajectory with the MPC controller is affected not only by the quality of the model used but also by the feedback mechanism. Even though Gaussian models exhibit a higher error rate, their inherent averaging characteristic enhances the performance of the MPC controller when coupled with feedback.

Table 1. Control errors of MPC algorithms with different PIHNN models.

<table><tr><td>Model Type</td><td>MPC-NO</td><td>MPC-NPLPT</td></tr><tr><td>PIHNN ver. 1</td><td>3.051</td><td>3.116</td></tr><tr><td>PIHNN ver. 2</td><td>3.031</td><td>3.095</td></tr><tr><td>PIHNN ver. 3</td><td>3.205</td><td>3.263</td></tr><tr><td>PIHNN ver. 4</td><td>3.220</td><td>3.290</td></tr><tr><td>PIHNN ver. 5</td><td>2.935</td><td>2.741</td></tr><tr><td>PIHNN ver. 6</td><td>2.965</td><td>3.020</td></tr></table>

Secondly, Table 1 demonstrates that the MPC-NPLPT controller generally yields slightly higher error values than the MPC-NO one when utilizing the same PIHNN model for prediction. This result is not surprising, as MPC-NPLPT employs a linearized model. During linearization, some of the information present in the nonlinear model is simplified or lost. The exception here is PIHNN model ver. 5, where MPC-NPLPT algorithm provides better controller performance. This may be attributed to chance where the simplifications happened to benefit the controller’s performance in this specific case. However, it is worth noting that the error differences between MPC-NO and MPC-NPLT controllers are minimal for each type of PIHNN model, and both types of controllers work very well.

Table 2 compares the average time required by each MPC controller for control calculations. The computations have been conducted on a PC, and since it is not a real-time system, results may vary on different PCs. Therefore, the results are presented as percentages. The longest time recorded for MPC-NO with PIHNN model ver. 3, which amounted to 140 ms, is considered as 100%. The table reveals that the implementation of the online linearization-based MPC controller significantly reduced the calculation time required, resulting in a 4–5 times decrease compared to nonlinear controllers.

Table 2. Average execution time of MPC algorithms with different PIHNN models.

<table><tr><td>Model Type</td><td>MPC-NO</td><td>MPC-NPLPT</td></tr><tr><td>PIHNN ver. 1</td><td>94.4%</td><td>22.4%</td></tr><tr><td>PIHNN ver. 2</td><td>97.2%</td><td>23.1%</td></tr><tr><td>PIHNN ver. 3</td><td>100.0%</td><td>21.0%</td></tr><tr><td>PIHNN ver. 4</td><td>97.2%</td><td>23.1%</td></tr><tr><td>PIHNN ver. 5</td><td>89.3%</td><td>24.5%</td></tr><tr><td>PIHNN ver. 6</td><td>93.7%</td><td>23.1%</td></tr></table>

The results are also visually presented. In Figure 10, one can observe the performance of the MPC algorithm with a DF employing trapezoidal functions. The output signals for the PIHNN model with the initial function shape are swift without overshoot for both low and high values of NAMW. However, for intermediate NAMW values, there is a slightly larger overshoot, and the settling time is extended. The signals are quite similar in the case of DF with a tuned function shape, but there is a greater overshoot for intermediate NAMW values. Additionally, it is worth noting that the results obtained for the MPC-NO controller are practically indistinguishable from those for the MPC-NPLT one.

Figure 11 illustrates the results for sigmoidal membership functions. Here, we can observe that the overshoot becomes more pronounced, particularly for intermediate NAMW values, especially when considering the set-point $\mathrm { N \hat { A } M W ^ { s p } } = \dot { 2 } . 5 \times 1 0 ^ { 4 }$

![](images/e8de7f0dc89b1c5f4d3d2b9c1c8b69655a11888eb271c500fe729b7ac9e4f6be.jpg)  
Figure 10. Cont.  
(a)

![](images/8552c36e1653d16dfa9eb0cf87b09a569cad45f876215de344f4d48486f030c9.jpg)

![](images/9232620ea1c5cc258c281031c552adcc12a5f66d04586df09fdd7474d84ad6fc.jpg)  
(b)

Figure 10. MPC with trapezoidal membership function shapes. (a) MPC-NO and MPC-NPLPT controllers using PIHNN model ver. 1; (b) MPC-NO and MPC-NPLPT controllers using PIHNN model ver. 2.  
![](images/8dcbc60fcee041395d471b4e7876a86b7e5bd73b722809084e0f1cda80d1139c.jpg)

![](images/b9be9127d576248d46455b94a519d5570e2c979e9054ff12139fc15534c6730d.jpg)

(a)  
![](images/e31125cdcca4223b982209472a0e351f7485b482b4e6f728f7d113eae5bf854c.jpg)

![](images/9cc3cab1f99d0cf9d911e56695da418a8b8fefe154e7128a7ccb85c3434d9589.jpg)  
(b)  
Figure 11. MPC controllers with sigmoidal membership function shapes. (a) MPC-NO and MPC-NPLPT controllers using PIHNN model ver. 3; (b) MPC-NO and MPC-NPLPT controllers using PIHNN model ver. 4.

![](images/4e0549da221cb3dc5701d75ec5612d17281c98d75a273bf3e7464d5f5bebfdcb.jpg)

The final Figure 12 displays the results of applying Gaussian membership functions. These results are characterized by the shortest settling time and the smallest overshoot. Notably, the controller exhibits excellent performance for average values of NAMW. This observation leads to the conclusion that the averaging nature of Gaussian functions, as seen earlier in the modeling phase (Figure 9, positively impacts the controller’s performance when using the model in the MPC scheme. For NAMW values within the range of $2 \times 1 0 ^ { 4 }$ to $3 \times 1 0 ^ { 4 }$ , the FP model significantly impacts PIHNN performance. As mentioned, the FP model is imperfect, featuring an increased gain of 20%.

![](images/5140a9ea109f354da3d406240c37e05ee5c04da164dbe19916e7c6bc75cc3573.jpg)

![](images/3374ae9bd2b0a6d27d6c5eae60591a1907ad0304f708226c2aa95170bcc9598f.jpg)  
(a)  
(b)  
Figure 12. MPC controllers with Gaussian membership function shapes. (a) MPC-NO and MPC-NPLPT controllers using PIHNN model ver. 5; (b) MPC-NO and MPC-NPLPT controllers using PIHNN model ver. 6.

## 5. Conclusions

This work defines a new PIHNN model structure that combines the first-principle process description and data-driven neural sub-models using a specialized data fusion block that relies on fuzzy logic. We consider a very practical case when the available first-principle model is imperfect and the data cannot be measured in the complete range of process operation. By combining an imperfect physical model with data obtained from an incomplete range of operations, we have developed a hybrid model that significantly improves performance across the entire range of signal variability. Secondly, this work develops a computationally efficient MPC controller for the PIHNN model. We show the efficacy of the PIHNN model and the resulting MPC controller for a simulated polymerization benchmark. We study the efficiency of different data fusion fuzzy blocks and their impact on model accuracy. We recommend tuning, i.e., optimizing the fuzzy membership functions, greatly improving model accuracy. Finally, we show that the described MPC controller based on the PIHNN model gives excellent results. Namely, the obtained control quality is very similar to that possible in MPC relying on nonlinear optimization while its calculation time is a few times shorter. In our future work, we plan to develop a methodology for designing PIHNN structures tailored to processes with multiple inputs and outputs. Additionally, it is interesting to check the impact of employing various decision model types within the data fusion block on PIHNN modeling quality.

Author Contributions: Conceptualisation K.Z. and M.Ł.; methodology, K.Z. and M.Ł.; software, K.Z. and M.Ł.; validation, K.Z. and M.Ł.; formal analysis, K.Z. and M.Ł.; investigation, K.Z.; writing— original draft preparation, K.Z. and M.Ł.; writing—review and editing, K.Z. and M.Ł.; visualization, K.Z.; supervision, M.Ł. All authors have read and agreed to the published version of the manuscript.

Funding: This research was financed by the Warsaw University of Technology in the framework of the project for the scientific discipline automatic control, electronics and electrical engineering.

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: On request from the authors.

Conflicts of Interest: The authors declare no conflict of interest.

## References

1. Camacho, E.F.; Bordons, C. Model Predictive Control; Springer: London, UK, 1999.

2. Tatjewski, P. Advanced Control of Industrial Processes, Structures and Algorithms; Springer: London, UK, 2007.

3. Hosen, M.A.; Hussain, M.A.; Mjalli, F.S. Control of polystyrene batch reactors using neural network based model predictive control (NNMPC): An experimental investigation. Control. Eng. Pract. 2011, 19, 454–467. [CrossRef]

4. Wang, B.; Shahzad, M.; Zhu, X.; Rehman, K.U.; Uddin, S. A Non-linear Model Predictive Control Based on Grey-Wolf Optimization Using Least-Square Support Vector Machine for Product Concentration Control in l-Lysine Fermentation. Sensors 2020, 20, 3335. [CrossRef]

5. Assandri, A.D.; de Prada, C.; Rueda, A.; Martínez, J.S. Nonlinear parametric predictive temperature control of a distillation column. Control. Eng. Pract. 2013, 21, 1795–1806. [CrossRef]

Carli, R.: Cavone, G.; Ben Othman, S.; Dotoli, M. IoT Based Architecture for Model Predictive Control of HVAC Systems in Smart Buildings. Sensors 2020. 20. 781. [CrossRefl

7. Alexis, K.; Nikolakopoulos, G.; Tzes, A. Switching model predictive attitude control for a quadrotor helicopter subject to atmospheric disturbances. ISA Trans. 2011, 19, 1195–1207. [CrossRef]

8. Gruber, J.K.; Doll, M.; Bordons, C. Design and experimental validation of a constrained MPC for the air feed of a fuel cell. Control. Eng. Pract. 2009, 17, 874–885. [CrossRef]

9. Lima, P.F.; Pereira, G.C.; Mårtensson, J.; Wahlberg, B. Experimental validation of model predictive control stability for autonomous driving. Control. Eng. Pract. 2018, 81, 244–255. [CrossRef]

10. Yao, F.; Yang, C.; Liu, X.; Zhang, M. Experimental Evaluation on Depth Control Using Improved Model Predictive Control for Autonomous Underwater Vehicle (AUVs). Sensors 2018, 18, 2321. [CrossRef]

11. Ding, Z.; Sun, C.; Zhou, M.; Liu, Z.; Wu, C. Intersection Vehicle Turning Control for Fully Autonomous Driving Scenarios. Sensors 2021, 21, 3995. [CrossRef]

12. Bassolillo, S.R.; D’Amato, E.; Notaro, I.; Blasi, L.; Mattei, M. Decentralized Mesh-Based Model Predictive Control for Swarms of UAVs. Sensors 2020, 20, 4324. [CrossRef]

13. Xiong, L.; Fu, Z.; Zeng, D.; Leng, B. An Optimized Trajectory Planner and Motion Controller Framework for Autonomous Driving in Unstructured Environments. Sensors 2021, 21, 4409. [CrossRef]

14. Simon, D. Optimal State Estimation: Kalman, H, and Nonlinear Approaches; John Wiley and Sons: Hoboken, NJ, USA, 2006.

15. Karimshoushtari, M.; Novara, C.; Tango, F. How Imitation Learning and Human Factors Can Be Combined in a Model Predictive Control Algorithm for Adaptive Motion Planning and Control. Sensors 2021, 21, 4012. [CrossRef]

16. Miller, A.; Rybczak, M.; Rak, A. Towards the Autonomy: Control Systems for the Ship in Confined and Open Waters. Sensors 2021, 21, 2286. [CrossRef]

17. Bacci di Capaci, R.; Vaccari, M.; Pannocchia, G. Model predictive control design for multivariable processes in the presence of valve stiction. J. Process. Control. 2018, 71, 25–34. [CrossRef]

18. Ławry ´nczuk, M. Modelling and predictive control of a neutralisation reactor using sparse Support Vector Machine Wiener models. Neurocomputing 2016, 205, 311–328. [CrossRef]

19. Ławry ´nczuk, M. Nonlinear Predictive Control Using Wiener Models: Computationally Efficient Approaches for Polynomial and Neural Structures; Studies in Systems, Decision and Control; Springer: Cham, Switzerland, 2022; Volume 389.

20. Ławry ´nczuk, M. Computationally Efficient Model Predictive Control Algorithms: A Neural Network Approach; Studies in Systems, Decision and Control; Springer: Cham, Switzerland, 2014; Volume 3.

21. Balla, K.M.; Nørgaard, J.T.; Bendtsen, J.D.; Kallesøe, C.S. Model Predictive Control using linearized Radial Basis Function Neural Models for Water Distribution Networks. In Proceedings of the 2019 IEEE Conference on Control Technology and Applications (CCTA), Hong Kong, China, 19–21 August 2019; pp. 368–373.

22. Schwedersky, B.B.; Flesch, R.C.C.; Dangui, H.A.S. Practical nonlinear model predictive control algorithm for Long Short-Term Memory networks. IFAC-PapersOnLine 2019, 52, 468–473. [CrossRef]

23. Zarzycki, K.; Ławry ´nczuk, M. Advanced predictive control for GRU and LSTM networks. Inf. Sci. 2022, 616, 229–254. [CrossRef]

24. Wang, Y. A new concept using LSTM Neural Networks for dynamic system identification. In Proceedings of the 2017 American Control Conference (ACC), Seattle, WA, USA, 24–26 May 2017; pp. 5324–5329.

25. Jordan, I.D.; Sokół, P.A.; Park, I.M. Gated Recurrent Units Viewed Through the Lens of Continuous Time Dynamical Systems. Front. Comput. Neurosci. 2021, 15, 678158. [CrossRef]

26. Bonassi, F.; da Silva, C.F.O.; Scattolini, R. Nonlinear MPC for Offset-Free Tracking of systems learned by GRU Neural Networks. IFAC-PapersOnLine 2021, 54, 54–59. [CrossRef]

27. Zarzycki, K.; Ławry ´nczuk, M. LSTM and GRU Neural Networks as Models of Dynamical Processes Used in Predictive Control: A Comparison for Two Chemical Reactors. Sensors 2021, 21, 5625. [CrossRef]

28. Li Ping, Z.; Min, X.; Hui-Nan, W. Hybrid control of bifurcation in a predator-prey system with three delays. Acta Phys. Sin. 2011, 60, 010506. [CrossRef]

29. Lu, L.; Huang, C.; Song, X. Bifurcation control of a fractional-order PD control strategy for a delayed fractional-order prey–predator system. Eur. Phys. J. Plus 2023, 138, 77. [CrossRef]

30. Xu, C.; Cui, X.; Li, P.; Yan, J.; Yao, L. Exploration on dynamics in a discrete predator–prey competitive model involving feedback J. Biol. Dyn. 2023 17

31. Alhajeri, M.S.; Luo, J.; Wu, Z.; Albalawi, F.; Christofides, P.D. Process structure-based recurrent neural network modeling for predictive control: A comparative study. Chem. Eng. Res. Des. 2022, 179, 77–89. [CrossRef]

32. Roehrl, M.A.; Runkler, T.A.; Brandtstetter, V.; Tokic, M.; Obermayer, S. Modeling System Dynamics with Physics-Informed Neural Networks Based on Lagrangian Mechanics. IFAC-PapersOnLine 2020, 53, 9195–9200. [CrossRef]

33. Yang, L.; Meng, X.; Karniadakis, G.E. B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data. J. Comput. Phys. 2021, 425, 109913. [CrossRef]

34. Nascimento, R.G.; Fricke, K.; Viana, F.A. A tutorial on solving ordinary differential equations using Python and hybrid physics-informed neural network. Eng. Appl. Artif. Intell. 2020, 96, 103996. [CrossRef]

35. Antonelo, E.A.; Camponogara, E.; Seman, L.O.; de Souza, E.R.; Jordanou, J.P.; Hübner, J.F. Physics-Informed Neural Nets-based Control. arXiv 2021, arXiv:2104.02556.

36. Bolderman, M.; Lazar, M.; Butler, H. Physics–Guided Neural Networks for Inversion–based Feedforward Control applied to Linear Motors. In Proceedings of the 2021 IEEE Conference on Control Technology and Applications (CCTA), San Diego, CA, USA, 9–11 August 2021; pp. 1115–1120.

37. Wang, R.; Yu, R. Physics-Guided Deep Learning for Dynamical Systems: A Survey. arXiv 2021, arXiv:2107.01272.

38. Nascimento, R.G.; Corbetta, M.; Kulkarni, C.S.; Viana, F.A. Hybrid physics-informed neural networks for lithium-ion battery modeling and prognosis. J. Power Sources 2021, 513, 230526.

39. Shi, R.; Mo, Z.; Di, X. Physics-Informed Deep Learning for Traffic State Estimation: A Hybrid Paradigm Informed By Second Order Traffic Models. In Proceedings of the AAAI Conference on Artificial Intelligence, Virtual, 2–9 February 2021; Volume 35, pp. 540–547.

40. Daw, A.; Karpatne, A.; Watkins, W.; Read, J.; Kumar, V. Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling. arXiv 2017, arXiv:1710.11431.

41. Zarzycki, K.; Ławry ´nczuk, M. Physics-Informed Hybrid Neural Network Model for MPC: A Fuzzy Approach. Lecture Notes in Networks and Systems; Pawełczyk M., Bismor D., Ogonowski S., Kacprzyk J., Eds.; Springer Nature Switzerland: Cham, Switzerland, 2023; Volume 708, pp. 183–192.

42. Hochreiter, S. Untersuchungen zu dynamischen neuronalen Netzen. Master’s Thesis, Technical University Munich, Munich, Germany, 1991.

43. Doyle, F.J.; Ogunnaike, B.A.; Pearson, R. Nonlinear model-based control using second-order Volterra models. Automatica 1995, 31.697-714.[CrossRefl