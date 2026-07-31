# Physics-Informed Neural Networks for Process Systems: Handling Plant-Model Mismatch

Farshad Moayedi, Aswin Chandrasekar, Sarah Rasmussen, Samardeep Sarna, Brandon Corbett, and Prashant Mhaskar\*

![](images/55e68d8ff7f53d01168ba87fed5649bcb2a8925b8bc4fec189da5839c709711a.jpg)

Cite This: Ind. Eng. Chem. Res. 2024, 63, 13650−13659

Read Online

ACCESS

Metrics & More

Article Recommendations

![](images/f75d085d85fffc3762eb5b0a878c6788ec13e13840302f6d4aa25a7a1981c6dc.jpg)  
ABSTRACT: This work addresses the problem of leveraging first-principles knowledge with data-driven techniques in the Physics-Informed/Inspired Neural Network (PINN) framework to handle plant−model mismatch. To this end, a PINN is developed utilizing the first-principles model of the system and plant data and demonstrated to handle plant−model mismatch. The PINN is compared with another dynamic modeling technique, a Recurrent Neural Network (RNN), and for the illustrative simulation example, is shown to improve the predictive capabilities of the model compared to the other techniques. In particular, purely datadriven approaches often encounter challenges when applied to complex systems. This can lead to compromised predictive performance in situations where the model fails to capture the actual relationships among system variables. In contrast, the PINN respects the physical characteristics of the problem, while yielding a good dynamic model, based on process data. These results indicate the benefit of utilizing hybrid modeling techniques and their potential application to more complex systems.

## 1. INTRODUCTION

Mathematical modeling plays a crucial role in chemical engineering, facilitating the operation and control of chemical processes. Among various modeling methods, the firstprinciples approach, which employs fundamental physical laws and chemical kinetics to develop models, is often sought after. However, these models can be dificult to develop and maintain. Consequently, data-driven models are gaining popularity as a complement or alternative to first-principles models.<sup>1</sup> neural networks (ANNs) have recently gained popularity. ANNs are machine learning modeling techniques (universal function approximators) aimed at capturing complex nonlinear relationships between input and output variables without requiring a thorough understanding of the system’s underlying physics. This feature makes them suitable for modeling nonlinear and complex chemical systems where traditional first-principles models may be limited by incomplete or uncertain system knowledge. ANNs are being extensively explored for potential use in chemical engineering, including process optimization, control, and forecasting of process 2,3 outcomes.

One challenge of using nonlinear neural networks in practice is their susceptibility to overfitting, a phenomenon where the model becomes too intricate and fits the random fluctuations in the training data rather than the underlying patterns. Applying new data can result in inadequate performance, leading to inaccurate predictions.<sup>4</sup> By leveraging the advantages of first-principles and data-driven approaches, hybrid models can provide an improved model.<sup>5</sup> This Gray box modeling is a technique that combines first-principles modeling with data-driven methods to strike a balance between model accuracy and computational eficiency while reducing the risk of overfitting. Two novel Knowledge-Informed Lasso− Lasso (KILL) and Lasso-Ridge (KILR) algorithms are proposed, to retain physical variables in inferential models by integrating process knowledge into sparse learning methods. Incorporating first-principles into the neural network (NN) architecture can reduce the need for extensive training data and provide a strong foundation of physical laws and relationships to guide the modeling process.<sup>7−9</sup> A Neural Ordinary Diferential Equation (NODE) model is introduced for continuous-time approximation of dynamic systems to address the limitations of discrete-time models like recurrent neural networks (RNNs).<sup>10</sup>

![](images/b2150650b8d89d017e9e08225279d3d112438253a07a025e187f78ef7e3e2bd9.jpg)

Outside of using NNs as part of a hybrid modeling structure, extensive research has been done on hybrid modeling approaches. First-principles equations have been incorporated into a Support Vector Regression (SVR) to estimate the unknown parameters of the equation.<sup>11</sup> A hybrid technique has been proposed that uses a Hammerstein model estimated using Least Squares Support Vector Machine (LSSVM), in conjunction with the subspace identification method.<sup>12</sup> Various other ways of integrating first-principles models with data-driven models in hybrid structures have been proposed,<sup>13,14</sup> including contributions aimed at handling the missing data problem.<sup>15</sup>

Recent advances have also proposed the use of various types of Shallow Neural Networks (SNNs) and Deep Neural part of hybrid models. For example, a DNN was employed to model the flocculation process as part of a hybrid modeling framework, integrating with first-principles models and used in the mass balance model.<sup>18</sup> Also, a hybrid model is formed by merging a modified kinetic model with a DNN trained using time-series process data to predict sensitive and uncertain parameters in a fermentation system.<sup>19</sup> In another work, the complex behavior of a biological process was efectively captured using parameter function neural networks. The neural network model accurately modeled process dynamics and aided in discerning between process and measurement uncertainties and underlying biochemical phenomena.<sup>20</sup> DNNs excel in managing nonlinearities and uncertainties compared to SNNs, but their eficacy encounters some challenges including needing a large dataset, the complexity of optimizing architecture and parameters, a lack of straightforward physical interpretation, and susceptibility to overfitting and underfitting risks,<sup>21</sup>.<sup>22</sup> The combination of physical phenomena in NNs, known as Physics-Informed Neural Networks (PINNs), presents an excellent opportunity to develop good NN models with limited data.<sup>23</sup>

PINNs have been used to solve diferential equations such as PDEs,<sup>24</sup> fractional, and integro-diferential equations (IDEs),<sup>25</sup>.<sup>26</sup> Moreover, PINNs are used for parameter estimations in cases where the system dynamics are not fully understood. A PINN is introduced to estimate parameters and unknown physics in PDE models.<sup>27</sup> A PINN-based framework is proposed to solve ODEs for parameter identification<sup>28</sup> as well. Another PINN is used in deep embedding-based hybrid models to capture unknown dynamics using data.<sup>29</sup> PINNs encode model equations as a part of the modeling structure in which an NN must fit observed data while reducing a residual.<sup>30</sup> Novel online learning Physics-Informed Recurrent Neural Network (PIRNN)-enhanced modeling is incorporated into model predictive control, which involves training PIRNN for the forward problem and estimating uncertain parameters for the inverse problem, enhancing the model’s ability to handle plant−model mismatches.<sup>31</sup> Previous research has made significant strides in elucidating PINN applications in dynamic systems, yet certain complexities warrant further exploration.

Notably in the previous study,<sup>31</sup> the PIRNN is trained on collocation points (no empirical data) without using actual data which may be limited, and complete physics knowledge (without plant-model mismatch) as the forward problem. On the other hand, the inverse problem addresses incomplete knowledge of physics by incorporating uncertain parameters and estimating these parameters using observational data. This allows for the updating of the PIRNN using error-triggering mechanisms, which helps the model conform to the system dynamics in the presence of the plant−model mismatch. An alternative is to leverage the (relatively) small amount of data to directly improve an existing first-principles model (with possibly incorrect parameters). In fact, rather than creating two distinct phases�one for training the PINN model without considering uncertain parameters and actual data, and another for utilizing actual data to refine the trained PINN and address plant-model mismatch�we can streamline the process by developing a unified PINN framework.

This manuscript demonstrates the key benefits of PINNs enable building good models by using relatively small datasets with modest measurement noise by leveraging the first principles models, while handling plant−model mismatch. The rest of the manuscript is organized as follows: Section 2 describes the illustrative example, a series of two CSTRs. This section also provides an overview of the RRN modeling and presents the RNN-based model application to illustrate its limitations, motivating the necessity of physics-based model ing. Section 3 presents the proposed PINN model. Section 4 presents the simulation results for the illustrative example. Finally, section 5 provides the concluding remarks.

## 2. PRELIMINARIES

An overview of the illustrative example is presented in this section, followed by a review of current modeling approaches and a discussion of PINNs.

2.1. Illustrative Example. Consider a series of two continuous stirred-tank reactors (CSTRs) shown in Figure 1.

The CSTRs in the series are chosen as an illustrative example in the paper to illustrate the ability of PINN models to capture the physical characteristics of the problem that a purely data-driven modeling technique may mischaracterize. In particular, the CSTR in series provides instances of physical characteristics that can be easily verified, such as the fact that changes to the inlet to the second CSTR should not impact the outlet from the first CSTR. The simulation results use this specific scenario as a key test case to compare various modeling techniques against the PINN. The process inputs are each tank’s inlet concentration and heat added to/removed from the reactor and the measured outputs are the concentration and temperature of each tank. Mathematically, the process can be described using ordinary diferential equations (ODEs) as follows:

![](images/b267093f7bc9946510a65782e9bc4a51fc1d74889aef834b82b43738e91dcf89.jpg)  
Figure 1. Schematic of two CSTRs in series.

Table 1. Process Parameter Values for the Illustrative Example

<table><tr><td>parameter</td><td> $V$ </td><td> $R$ </td><td> $T_{\text{in}}$ </td><td> $\Delta H$ </td><td> $K_0$ </td><td> $E$ </td><td> $C_p$ </td><td> $ρ$ </td><td> $F$ </td></tr><tr><td>value</td><td>20</td><td>8.314</td><td>320</td><td> $-4.78 \times 10^{3}$ </td><td> $7.2 \times 10^{6}$ </td><td> $4.157 \times 10^{4}$ </td><td>0.4</td><td>1000</td><td>50</td></tr><tr><td>unit</td><td> $\text{m}^{3}$ </td><td> $\text{J mol}^{-1} \text{K}^{-1}$ </td><td>K</td><td> $\text{J mol}^{-1}$ </td><td> $\text{h}^{-1}$ </td><td> $\text{J mol}^{-1}$ </td><td> $\text{J kg}^{-1} \text{K}^{-1}$ </td><td> $\text{kg m}^{3}$ </td><td> $\text{m}^{3} \text{h}^{-1}$ </td></tr></table>

![](images/21243b8451b489fc020097510313b3082643074002104e26b3777b69115bb5c1.jpg)  
Figure 2. Training scaled inputs (red dots) and scaled outputs (blue dots).

$$
\frac {\mathrm{d} C _ {\mathrm{A1}}}{\mathrm{d} t} = \frac {F}{V} (C _ {\mathrm{Ain,1}} - C _ {\mathrm{A1}}) - K _ {0} \mathrm{e} ^ {- E / R T _ {1}} C _ {\mathrm{A1}}\tag{1}
$$

$$
\frac {\mathrm{d} T _ {1}}{\mathrm{d} t} = \frac {F}{V} (T _ {\mathrm{in}} - T _ {1}) - \frac {\Delta H}{\rho C _ {\mathrm{p}}} K _ {0} \mathrm{e} ^ {- E / R T _ {1}} C _ {\mathrm{A1}} + \frac {Q _ {\mathrm{in,1}}}{\rho C _ {\mathrm{p}} V}\tag{2}
$$

$$
\frac {\mathrm{d} C _ {\mathrm{A2}}}{\mathrm{d} t} = \frac {F}{V} (C _ {\mathrm{Ain,2}} + C _ {\mathrm{A1}} - 2 C _ {\mathrm{A2}}) - K _ {0} \mathrm{e} ^ {- E / R T _ {2}} C _ {\mathrm{A2}}\tag{3}
$$

$$
\frac {\mathrm{d} T _ {2}}{\mathrm{d} t} = \frac {F}{V} (T _ {\mathrm{in}} + T _ {1} - 2 T _ {2}) - \frac {\Delta H}{\rho C _ {\mathrm{p}}} K _ {0} \mathrm{e} ^ {- E / R T _ {2}} C _ {\mathrm{A2}} + \frac {Q _ {\mathrm{in,2}}}{\rho C _ {\mathrm{p}} V}\tag{4}
$$

where $C _ { \mathrm { A l } }$ is the concentration of species A in the first reactor, $T _ { 1 }$ is the temperature of the first reactor, $C _ { \mathrm { A } 2 }$ is the concentration of species A in the second reactor, $T _ { 2 }$ is the temperature of the second reactor, $C _ { \mathrm { A i n , 1 } }$ is the inlet concentration of species A into the first reactor, $C _ { \mathrm { A i n } , 2 }$ is the inlet concentration of species A into the second reactor, $Q _ { \mathrm { i n , 1 } }$ is the heat added to/removed from the first reactor, $Q _ { \mathrm { i n } , 2 }$ is the heat added to/removed from the second reactor, F is the inlet volumetric flow rate, V is the volume of reactors, $K _ { 0 }$ is the Arrhenius constant, E is the activation energy, R is the universal gas constant, $T _ { \mathrm { i n } }$ is the inlet temperature, ΔH is the heat of reaction, $\rho$ is the density of the solution, and $C _ { p }$ is the heat capacity of the solution.

To mimic the typically low availability of open-loop data, a step-ramp random input sequence is employed to generate the data over 10 h for each training, testing, and validation. This profile involves alternating between a constant input value (for each input) held for multiple time steps followed by a ramp over a single time step. This pattern is repeated, with the constant value maintained for a series of time steps between each ramp. Then, the next step involves solving the ODEs using the ode.int solver and related parameters, tabulated in Table 1 to compute corresponding outputs. Then, all datasets are re-scaled using min-max normalization. Gaussian noise is introduced to emulate measurement noise, using a standard normal distribution (mean = 0, standard deviation = 1) and scaled by 0.01.

Finally, as shown in Figure 2, A dataset of 50 points in the noisy training dataset is selected as available data for modeling. The selection of these 50 data points is guided by Latin Hypercube Sampling (LHS), a method designed to ensure a more uniform distribution of samples across the input space than simple random sampling.

2.2. Existing Neural Network-Based Models. A Recurrent Neural Network (RNN), a type of ANN frequently applied in modeling nonlinear systems, is used as an example of a data-driven model that can be utilized as part of PINNs. Unlike feed-forward NNs, which process data strictly in a forward direction, RNNs have connections that loop back on themselves, allowing modeling process dynamics.

Figure 3 illustrates that the current hidden states of the network are obtained from current and past information and

![](images/ecb03e9772de52828e908f2816b8f95f8622616ec2e8d1c715fe03652e5e65e2.jpg)  
Figure 3. Schematic of an RNN.

they are reintroduced into the network, facilitating the keeping of information from previous states. Subsequently, this information is employed to compute the current outputs. The mathematical representation of the RNN model is defined as follows:

$$
h _ {k} = a _ {h} (W _ {u} u _ {k} + W _ {h} h _ {k - 1})\tag{5}
$$

CA,1  
![](images/0ab2ae9bc63d52c117da2fc61e27e4a9f4e855027dff0cfa035d4654bee21339.jpg)

T1  
![](images/0179a6242ed7149715c1c1793231e154c347d418d19c5b861e79a21097cf7d08.jpg)

$$
y _ {k} = a _ {y} (W _ {y} h _ {k})\tag{6}
$$

where $a _ { h }$ and $a _ { y }$ are activation functions of hidden and output layers, respectively. $W _ { w } \ W _ { h } ,$ and $W _ { y }$ are weight matrices of input, hidden, and output layers, respectively. The parameters $h _ { k } , u _ { k } ,$ and $y _ { k }$ are vectors of the current hidden state, input, and output, respectively.

In this manuscript, the proposed RNN architecture utilizes a lag of 1. However, in this specific architecture, the lag component will be manually incorporated into a neural network, attempting to predict the next output using the current input and the current output. By incorporating this lag, a standard ANN transforms into an RNN. As a result, based on the chosen structure of the NN, the data matrices must be organized as follows:

$$
X = \left[ \begin{array}{c c c} u 1 _ {0} & \dots & y 4 _ {0} \\ \vdots & \ddots & \vdots \\ u 1 _ {n - 1} & \dots & y 4 _ {n - 1} \end{array} \right] Y = \left[ \begin{array}{c c c} y 1 _ {1} & \dots & y 4 _ {1} \\ \vdots & \ddots & \vdots \\ y 1 _ {n} & \dots & y 4 _ {n} \end{array} \right]\tag{7}
$$

where the matrix X comprises all input and output parameters from time zero to $n - 1 .$ . Conversely, the matrix Y includes all the output parameters from time 1 to n. For the present example, the RNN’s input and output layers consist of 8 input nodes (4 inputs to the CSTR and 4 outputs of the previous

CA,2  
![](images/bb3387f7971a09830bdbcba4bd163d9a60138ac4d1ae9063d9123c42300545c4.jpg)

![](images/b7eadf4f969b5cc604335f695431331ec349e816022a96ae7e3703f8e5cee8c9.jpg)  
Figure 4. Scaled actual output value (blue curve) vs RNN’s scaled predictions (black curve) for the first test dataset.

T\_1  
![](images/db67236a38f6719c28b1d07020d1c8e506ca263cd555a8ae5db1a11a60996445.jpg)

![](images/b29dfb22b7ede4b021b72c656f7619b6e1230838e2c12484cf2503b1e6895d1d.jpg)

![](images/5d97d4931b07a712f123e0d78244030223044d7915c7fa590fe81bd1082284c3.jpg)

T2  
![](images/27a308ef30247997095bd4ab8519c7b3c6a23d08361477a3fc578b0c0183ed1a.jpg)  
Figure 5. Scaled actual output value (blue curve) vs RNN’s scaled predictions (black curve) for the second test dataset.

time step) and 4 output nodes (4 outputs of the current time step), respectively. Additionally, the RNN incorporates one hidden layer with 4 nodes. Figure $^ { 4 , }$ shows the results of the RNN in predicting the first test dataset, demonstrating the ability of the RNN to capture the behavior of the process reasonably well while being limited by the data and presence of measurement noise. As seen in Figure $^ { 4 , }$ RNN is able to predict the $C _ { \mathrm { A } 2 }$ and $T _ { 1 }$ adequately, with some visible error in predicting $T _ { 2 }$ and more for $C _ { \mathrm { A l } } .$ . Note that both the actual and predicted values are shown in the scaled version.

Remark 1. While traditional RNNs, such as the one utilized in this work, represent a purely data-driven modeling approach, it is important to note that alternative RNN architectures like Long Short-Term Memory (LSTM) networks could also be considered. However, the primary focus of this manuscript is to demonstrate the incorporation of physical knowledge into a neural network based model. The selection of a traditional RNN is simply one choice to highlight the contrast between purely data-driven models and hybrid models (PINNs), where the incorporation of physical knowledge signif icantly enhances predictive performance.

The challenge dataset is used to evaluate the ability of the RNN to capture the physical characteristics of the problem. Notably, the second test dataset maintains three input variables $C _ { \mathrm { A i n , 1 } } , \mathrm { Q } _ { \mathrm { i n , 1 } } ,$ and $Q _ { \mathrm { i n } , 2 }$ at constant values while only varying the $C _ { \mathrm { A i n } , 2 } .$ As seen in Figure 5, the RNN predicts changes in variables associated with the first reactor, violating the actual dynamics of the system, and motivating the use of PINN.

Remark 2. The choice of lag 1 in the RNN model is based on the use of training and testing datasets, and a result of experimenting with dif ferent lag values. In particular, the NN was tested with up to and including 10 lags, with the resultant RNN (as expected) not able to predict constant values accurately in the challenge test dataset.

## 3. PROPOSED MODELING APPROACH

PINNs consist of two key elements, the so-called Data and Physics parts. The data part consists of a typical NN model, trained based on a cost function that evaluates the disparity between actual and predicted values. Concurrently, the physics part means to incorporate first-principles process knowledge into the PINN. In summary, the primary objective of the data part is to achieve predictive accuracy compared to actual values, and simultaneously, the physics part’s main task is to direct the predicted values to apply to the physical equations governing the system.

3.1. Data Part. The Data Part of the PINN is designed to predict the system’s output vector based on the provided input vector. This part works completely similarly to the typical NN models. To train the model, and assess predictive performance, a Data Loss term is defined according to eq 8:

$$
\mathrm{DataLoss} = \frac {1}{m} \sum_ {i = 1} ^ {m} \left(y _ {i} - \hat {y _ {i}}\right) ^ {2}\tag{8}
$$

where $y _ { i }$ is the measured value of the output and $\hat { y } _ { i }$ is the corresponding predicted value, and m is the number of actual data points, which is 50.

3.2. Physics Part. The Physics Part of the PINN operates concurrently to guide modeling by incorporating the process knowledge, and in the present instance, through a model comprising of ODEs. Notably, this part does not require any process data and is only predicated on making the dynamics captured by the NN match the dynamics of the first-principles model. The next subsection describes the computation of the derivatives of the NN using the Euler Method.

3.2.1. Computing Derivatives of the NN Using the Euler Method. As shown in eq $^ { 9 , }$ the Euler Method is employed to estimate derivatives of the NN model at each time point:

$$
f (t _ {k}, \hat {y} _ {k}) = \frac {(\hat {y} _ {k + 1} - \hat {y} _ {k})}{h}\tag{9}
$$

where f is the estimate of the derivative of $\hat { y }$ and h represents the step size or time step used to discretize the continuous time diferential equation and determines how far forward in time we move from $t _ { k }$ to $t _ { k + 1 }$ to obtain $y _ { k + 1 } .$ To employ the Euler method, a crucial consideration is the selection of an appropriate time step, denoted as h in eq 9. The time step should be suficiently small to accurately capture the dynamics of the system. Moreover, in this study, during the data generation process, a time step of 0.002 over a 10-h period is utilized, resulting in measurements obtained every 7.2 s. The choice of this time step in data generation ensures that the time interval between each two predicted outputs (Ts) corresponds to the time step (h) used to calculate the derivatives using Euler’s method. As a result, the first intermediate matrix (El) is formed based on the value of derivatives as shown in eq 10 below:

$$
E l = \left(\left[ \begin{array}{c c c} \widehat {y 1} _ {1} & \dots & \widehat {y 4} _ {1} \\ \widehat {y 1} _ {2} & \dots & \widehat {y 4} _ {2} \\ \vdots & \ddots & \vdots \\ \widehat {y 1} _ {k} & \dots & \widehat {y 4} _ {k} \end{array} \right] - \left[ \begin{array}{c c c} y 1 _ {0} & \dots & y 4 _ {0} \\ \widehat {y 1} _ {1} & \dots & \widehat {y 4} _ {1} \\ \vdots & \ddots & \vdots \\ \widehat {y 1} _ {k - 1} & \dots & \widehat {y 4} _ {k - 1} \end{array} \right]\right) / T s\tag{10}
$$

where $\hat { y }$ represents predicted output values by the neural network and Ts is the time interval between predicted outputs.

It is important to note that, for the first row of the El matrix, the actual output’s values are used because, at each step, the RNN predicts the next outputs using the current outputs and inputs; thus, the predictions only start from the time step after the initial.

3.2.2. Handling Plant−Model Mismatch. One of the key benefits of PINN that is demonstrated in the present manuscript is the ability to handle plant−model mismatch. Thus, the expectation is that the PINN is able to model the process better compared to the first-principles equations alone (due to the first-principles models having plant−model mismatch), and better than an NN model using the data alone (due to lack of first-principles knowledge). To illustrate this, the first-principles model utilized has a parametric mismatch compared to the process, via a scaling factor. The value of 0.7 is selected as the scaling factor, which is multiplied by three parameters: volume, the Arrhenius constant, and heat capacity. Thus, the model available to PINN is the one with these ”incorrect” values of the parameters.

3.2.3. Computing Derivatives using the First-Principles Model. The time derivatives are computed using the model equations described by eqs 11.

$$
\frac {\mathrm{d} \hat {C} _ {\mathrm{A1}}}{\mathrm{d} t} = \frac {F}{V ^ {*}} (C _ {\mathrm{Ain,1}} - \hat {C} _ {\mathrm{A1}}) - K _ {0} ^ {*} \mathrm{e} ^ {- E / \hat {T} _ {1}} \hat {C} _ {\mathrm{A1}}\tag{11a}
$$

$$
\frac {\mathrm{d} \hat {T _ {1}}}{\mathrm{d} t} = \frac {F}{V ^ {*}} (T _ {\mathrm{in}} - \hat {T _ {1}}) - \frac {\Delta H}{\rho C _ {p} ^ {*}} K _ {0} ^ {*} \mathrm{e} ^ {- E / R \hat {T _ {1}}} \hat {C} _ {\mathrm{A1}} + \frac {Q _ {\mathrm{in,1}}}{\rho C _ {p} ^ {*} V ^ {*}}\tag{11b}
$$

$$
\frac {\mathrm{d} \hat {C} _ {\mathrm{A2}}}{\mathrm{d} t} = \frac {F}{V ^ {*}} (C _ {\mathrm{Ain,2}} + \hat {C} _ {\mathrm{A1}} - 2 \hat {C} _ {\mathrm{A2}}) - K _ {0} ^ {*} \mathrm{e} ^ {- E / R \hat {T} _ {2}} \hat {C} _ {\mathrm{A2}}\tag{11c}
$$

$$
\frac {\mathrm{d} \hat {T} _ {2}}{\mathrm{d} t} = \frac {F}{V ^ {*}} (T _ {\mathrm{in}} + \hat {T} _ {1} - 2 \hat {T} _ {2}) - \frac {\Delta H}{\rho C _ {p} ^ {*}} K _ {0} ^ {*} \mathrm{e} ^ {- E / R \hat {T} _ {2}} \hat {C} _ {\mathrm{A2}} + \frac {Q _ {\mathrm{in,2}}}{\rho C _ {p} ^ {*} V ^ {*}}\tag{11d}
$$

where $V ^ { * } , ~ C _ { p } ^ { * } ,$ , and $K _ { 0 } ^ { * }$ are, respectively, the model volume, heat capacity, and Arrhenius constant made by the scaling factor, and $\hat { C } _ { \mathrm { A 1 } } , ~ \hat { T } _ { 1 } , ~ \hat { C } _ { \mathrm { A 2 } } ,$ and $\hat { T } _ { 2 }$ are the predicted values of outputs. This means that all calculations are based on inputs and predicted outputs (except the first row which uses inputs and theoretical outputs, similar to the first matrix) by the NN. As a result, a second intermediate matrix $\left( E q \right)$ is formed as shown by eq 12:

$$
E q = \left[ \begin{array}{c c c c} \mathrm{d} \hat {C} _ {\mathrm{A1}} / \mathrm{d} t (0) & \mathrm{d} \hat {T} _ {1} / \mathrm{d} t (0) & \mathrm{d} \hat {C} _ {\mathrm{A2}} / \mathrm{d} t (0) & \mathrm{d} \hat {T} _ {2} / \mathrm{d} t (0) \\ \vdots & \vdots & \vdots & \vdots \\ \mathrm{d} \hat {C} _ {\mathrm{A1}} / \mathrm{d} t (k - 1) & \mathrm{d} \hat {T} _ {1} / \mathrm{d} t (k - 1) & \mathrm{d} \hat {C} _ {\mathrm{A2}} / \mathrm{d} t (k - 1) & \mathrm{d} \hat {T} _ {2} / \mathrm{d} t (k - 1) \end{array} \right]\tag{12}
$$

It is essential to note that aligning the rows in these two matrices is a crucial point that needs careful attention to understand what the first and the last rows of each matrix are. Suppose there are $k + 1$ values (as inputs for NN to get predicted outputs) from time zero to time k for the physics part. This implies that according to eq 7, the X matrix used by the NN has k rows from time zero to time k − 1 (because the NN is using a lag of 1). In the first matrix (El), the initial row which is related to time zero, represents the diference between the predicted outputs for time 1 (which is the first prediction by the NN) and the corresponding output for the time zero (which should be actual output), divided by Ts. Similarly, the final row of this matrix which is related to time $k \mathrm { ~ - ~ } 1 ,$ illustrates the diference between the predicted outputs for time k and the predicted output for time $k - 1 ,$ , also divided by Ts. Consequently, the first matrix will consist of k rows. On the other hand, for the second matrix (Eq), the first row which is related to time zero, corresponds to the derivative values for the time zero using inputs and corresponding outputs (which should be actual output) for the time zero. Finally, the last row of this matrix which is related to the timek − 1, is the derivative values using inputs and predicted outputs for the time k − 1. Likewise, the second matrix will also contain k rows. As a result, the Physics Loss is computed by comparing these two derivative (intermediate) matrices, as shown in eq 13.

![](images/5de472a6945a9cd2ea4a7fe7257622a6803e1e7b64092d973031b2841e7c0563.jpg)

![](images/fbe575c55f239b76cc79f1ce8e47575f63645b79d31ac210338f0efffacbebac.jpg)

![](images/337d86282b87ae421cb6c72aeac093d7664881e381143ec4d87b9c154c1014f5.jpg)

![](images/16101d851770ff96f3a48ff6e8ecc8ee2f4a70f73dd30a214904ae4484cc548e.jpg)

![](images/194773fe8f72a6f9b16131aebe0a1a2a3a4b83237b358ba243728af1a6ddb13e.jpg)

![](images/f4eea96845a70d6a51d1fff1b8913cdef204098068805f1e00008a7ef81073fd.jpg)

![](images/1d379051e9e628c1aed00df8f4d04ccfc82392b6cd8578fbaa1bff6757eea80f.jpg)

![](images/cc83d38e23f5cffe4110242fb74876921aba5aeb644c57d0e703e508ec0ac76f.jpg)  
Figure 6. Scaled actual output (blue curve) vs the first-principles model’s scaled predictions (black curve) for the first (four left) and the second (four right) test datasets without plant−model mismatch.

$$
\mathrm{PhysicsLoss} = \frac {1}{k} \sum_ {i = 1} ^ {k} (E l - E q) ^ {2}\tag{13}
$$

where k is the number of rows in each matrix. Finally, the Overall Loss value of the PINN is defined as eq 14 by assigning a specific weight for each part.

$$
\text { Overall   Loss } = C _ {1} (\text { Data   Loss }) + C _ {2} (\text { Physics   Loss })\tag{14}
$$

The choice of the weights $C _ { 1 }$ and $C _ { 2 }$ enables specifying the relative importance of diferent components in the overall loss function. Increasing $C _ { 1 }$ leads to an increase in the reliance on the data loss which is the diferences between scaled actual and scaled predicted outputs, and increasing $C _ { 2 } ,$ leads to an emphasis on the physics loss, which is the diferences between derivatives.

Remark 3. It is important to note that, although sometimes NNs can work with unscaled data, it is usually a good idea to scale data before using it in NNs to facilitate faster convergence and prevent exploding gradients. If, however, the unscaled form of the equations is used in the physics part of the PINN where ODEs are involved, using weights in the overall loss f unction, enables balancing the two terms f rom the “data loss” and the “physics loss”.

Remark 4. An alternative approach in the physics part of the PINN would be that instead of calculating the value of derivatives and getting MSE between the derivatives matrices, we can integrate the ODEs and then calculate the value of outputs using them (predicted by the f irst-principles model). Then the physics loss could be described as the dif ference between the predicted values by the NN and those predicted by the f irst-principles model. However, integrating ODEs can be computationally expensive, especially for complex systems, while working directly with derivatives (using methods like the Euler method) is computationally more ef f icient and easier to implement, compared to integrating ODEs. The key is to recognize that in the physics part, there are no “data” involved, and thus there isn’t the associated measurement noise to contend with, enabling the use of derivatives.

Remark 5. Another alternative approach for physics-informed machine learning could be using f irst-principles-based data in the training of NNs as proposed by ref 32. In this work, an RNNbased model is developed based on a combination of noisy (process) and noise-f ree (f irst-principles) training data with separate cost f unctions and it is shown that employing the co teaching technique, as well as using f irst-principles-based data can mitigate the overf itting in NN-based models. The present manuscript is a more direct incorporation of the f irst-principles equations into the NN model.

## 4. APPLICATION TO AN ILLUSTRATIVE EXAMPLE

In this section, the proposed PINN model is applied to an illustrative example: a series of two CSTRs presented by Sartorius. In section 4.1, the results of the first-principles mode without consideration of parametric mismatch are demon strated on the first and the second test datasets. In section 4.2, the first-principles model’s performance with parametric mismatch is presented on both test datasets. PINN’s results are also shown on the test datasets in section 4.3. These results can show the superiority of PINN over the first-principles model in the presence of plant−model mismatch and the result of RNN, which was presented previously, in predicting outputs and simultaneously handling the plant−model mismatch.

4.1. First-Principles Model without Parametric Mismatch Results. Based on definitions, first-principles models rooted in fundamental physical laws, describe systems based on known principles and relationships. This study defines the firstprinciples model as the physical component of the PINN (without enabling the data part) that uses the system’s available ODEs. Essentially, when the PINN operates only based on its physics part, it functions as a first-principles model. When the mentioned parametric mismatch (scaling factor) is not assumed, the first-principles model performs without plant−model mismatch. Figure 6 shows that as expected, the first-principles model performs accurately on both test datasets when it works based on real parameters of ODEs (without parametric mismatch).

4.2. First-Principles Model with Parametric Mismatch Results. As mentioned in the previous section, when the scaling factor is utilized, the first-principles model works in the presence of the plant−model mismatch. Figure 7 illustrates the performance results of this first-principles model, which is somewhat poorer, compared to the model without parametric mismatch, but it still matches the right physics of the problem.

![](images/2beeb8b2e1e168bb3e16034f0d19c1f36e2eff9c09e6e9ab3deff51f83e6ad06.jpg)  
Figure 7. Scaled actual output (blue curve) vs the first-principles model’s scaled predictions (black curve) for the first (four left) and the second (four right) test datasets in the presence of the plant−model mismatch.

![](images/d4833d3a4437dcb54b0ca98c43e03eba3c4a4a6b92405dc3cf3a97a3f6c560fc.jpg)  
Figure 8. Actual output (blue curve) vs the PINN’s predictions (black curve) for the first (four left) and the second (four right) test datasets in the presence of the plant−model mismatch.

Table 2. MSE (on Scaled Values) for the First Test Dataset

<table><tr><td>model</td><td> $C_{A1}$ </td><td> $T_1$ </td><td> $C_{A2}$ </td><td> $T_2$ </td></tr><tr><td>RNN</td><td> $3.196 \times 10^{-3}$ </td><td> $1.480 \times 10^{-3}$ </td><td> $1.213 \times 10^{-3}$ </td><td> $2.215 \times 10^{-3}$ </td></tr><tr><td>FPM</td><td> $2.249 \times 10^{-4}$ </td><td> $2.176 \times 10^{-4}$ </td><td> $1.822 \times 10^{-4}$ </td><td> $1.957 \times 10^{-4}$ </td></tr><tr><td>FPM (with PMM)</td><td> $8.731 \times 10^{-3}$ </td><td> $2.135 \times 10^{-4}$ </td><td> $6.498 \times 10^{-3}$ </td><td> $2.263 \times 10^{-4}$ </td></tr><tr><td>PINN</td><td> $2.305 \times 10^{-3}$ </td><td> $2.252 \times 10^{-4}$ </td><td> $1.808 \times 10^{-3}$ </td><td> $2.421 \times 10^{-4}$ </td></tr></table>

Table 3. MSE (on Scaled Values) for the Second (Challenging) Test Dataset

<table><tr><td>model</td><td> $C_{A1}$ </td><td> $T_1$ </td><td> $C_{A2}$ </td><td> $T_2$ </td></tr><tr><td>RNN</td><td> $5.080 \times 10^{-4}$ </td><td> $8.324 \times 10^{-4}$ </td><td> $1.559 \times 10^{-3}$ </td><td> $1.017 \times 10^{-3}$ </td></tr><tr><td>FPM</td><td> $2.061 \times 10^{-4}$ </td><td> $2.018 \times 10^{-4}$ </td><td> $1.738 \times 10^{-4}$ </td><td> $1.878 \times 10^{-4}$ </td></tr><tr><td>FPM (with PMM)</td><td> $2.387 \times 10^{-3}$ </td><td> $2.005 \times 10^{-4}$ </td><td> $1.104 \times 10^{-2}$ </td><td> $2.037 \times 10^{-4}$ </td></tr><tr><td>PINN</td><td> $7.401 \times 10^{-4}$ </td><td> $2.231 \times 10^{-4}$ </td><td> $3.361 \times 10^{-3}$ </td><td> $2.177 \times 10^{-4}$ </td></tr></table>

4.3. PINN’s Results. The results obtained using the PINN are presented in Figure 8. Comparing these results with the RNN and the completely first-principles model in the presence of the plant−model mismatch shows that the PINN performs better than both, especially the RNN model.

Table 2 and 3 show the mean squared errors (MSEs) across various modeling approaches, including RNN, FPM (without parametric mismatch), FPM in the presence of plant−model mismatch, and the proposed PINN for both test datasets.

As seen, the PINN is able to improve upon the purely datadriven RNN, because of the ability to incorporate the first principles equations. Just as importantly, it is able to use the data to correct for the plant−model mismatch present in the first-principles model, but not get too ”thrown” of due to the measurement noise.

Remark 6. Note that the utilization of the f irst-principles model in data-driven techniques has been done in the context of subspace identif ication models as well. The results<sup>33</sup> emphasize leveraging the physical constraints of the system to build a constrained subspace identif ication model, which addresses the limitations of standard (unconstrained) subspace identif ication. By incorporating f irst-principles knowledge, the hybrid model avoids identif ying fake relationships between simulated inputs and output and works better to predict outputs, particularly those that are signif icantly mispredicted by the unconstrained subspace model. The resultant model, however, is still a linear time-invariant model. In contrast, the present manuscript demonstrates the ability to incorporate f irst-principles knowledge/constraints into a nonlinear modeling technique.

While this paper has shown that PINNs exhibit improved performance in modeling and predictions compared to both first-principles and data-driven models, their utility should be qualified. The potential for overfitting poses challenges, particularly in scenarios with high signal-to-noise ratios. For the present simulation scenario, as the measurement noise increases or data excitation decreases, PINN performance diminishes, converging towards FPM accuracy in the presence of plant model mismatch. Furthermore, beyond a certain threshold, PINN performance deteriorates even further compared to FPM. The key observation is that PINN can outperform FPMs under conditions where the measurement noise (or signal-to-noise ratio to be more precise) is “reasonable”. Exploring this topic further is a focus for future research endeavors.

## 5. CONCLUSION

In this study, a PINN is proposed by leveraging first-principles knowledge alongside data-driven techniques, using limited process data, to address plant−model mismatch. Using both first-principles models and plant data, PINN’s ability to handle such mismatches has been shown and its performance is compared with a purely data-driven approach, an RNN, for an illustrative example. The illustrative example is chosen to have a very simple explainable first-principles behavior�with two CSTRs in series, where changes to the inlet of the second CSTR should not impact the output of the first CSTR. Traditional data-driven methods fail to adequately predict the outputs of Tank 1 in a series system of two CSTRs, as the inputs to Tank 2 erroneously influence Tank 1’s outputs. This leads to poorer predictions compared to the PINN. Conversely, the PINN uses the process data and the physical characteristics of the system and is able to build a good enough dynamic model. In the following, the performance of PINN is also assessed against a completely first-principles model, considering the presence of plant−model mismatch inherent in first-principles models, and subject to modest measurement noise. The comparison demonstrates that PINN outperforms the completely first-principles model as it leverages informative process data as well.

## AUTHOR INFORMATION

## Corresponding Author

Prashant Mhaskar − Department of Chemical Engineering, McMaster University, Hamilton, Ontario L8S 4L8, Canada;

orcid.org/0000-0001-5866-211X; Email: mhaskar@ mcmaster.ca

## Authors

Farshad Moayedi − Department of Chemical Engineering, McMaster University, Hamilton, Ontario L8S 4L8, Canada Aswin Chandrasekar − Department of Chemical Engineering, McMaster University, Hamilton, Ontario L8S 4L8, Canada Sarah Rasmussen − Department of Chemical Engineering, McMaster University, Hamilton, Ontario L8S 4L8, Canada Samardeep Sarna − Department of Chemical Engineering, McMaster University, Hamilton, Ontario L8S 4L8, Canada Brandon Corbett − Sartorius Canada, Inc., Oakville, Ontario L6M 2 V9, Canada

Complete contact information is available at: https://pubs.acs.org/10.1021/acs.iecr.4c00690

## Author Contributions

Farshad Moayedi: Conceptualization; formal analysis; meth odology; coding; writing−original draft; writing−review and editing. Aswin Chandrasekar: Formal analysis; coding. Sarah Rasmussen: Formal analysis; methodology; coding. Samardeep Sarna: Conceptualization; Brandon Corbett: Conceptu alization; methodology; Prashant Mhaskar: Conceptualiza tion; methodology; writing−review and editing; supervision.

## Notes

The authors declare no competing financial interest.

## ■ ACKNOWLEDGMENTS

Financial support from the McMaster Advanced Control Consortium is gratefully acknowledged.

## REFERENCES

(1) Bradley, W.; Kim, J.; Kilwein, Z.; Blakely, L.; Eydenberg, M.; Jalvin, J.; Laird, C.; Boukouvala, F. Perspectives on the integration between first-principles and data-driven modeling. Comput. Chem. Eng. 2022, 166, 107898.

(2) Venkatasubramanian, V. The promise of artificial intelligence in chemical engineering: Is it here, finally? AIChE J. 2019, 65, 466−478.

(3) Kasmuri, N.; Kamarudin, S.; Abdullah, S.; Hasan, H.; Som, A. M. Integrated advanced nonlinear neural network-simulink control system for production of bio-methanol from sugar cane bagasse via pyrolysis. Energy 2019, 168, 261−272.

(4) Bejani, M. M.; Ghatee, M. A systematic review on overfitting control in shallow and deep neural networks. Artif. Intell. Rev. 2021, 54. 6391-6438

(5) Hassanpour, H.; Mhaskar, P.; Risbeck, M. J. A hybrid machine learning approach integrating recurrent neural networks with subspace identification for modelling HVAC systems. Can. J. Chem. Eng. 2022, 100, 3620−3634.

(6) Liu, Y.; Qin, S. J. Knowledge-informed Sparse Learning for Relevant Feature Selection and Optimal Quality Prediction. IEEE Trans. Ind. Inform. 2023, 19, 11499−11507.

(7) Menesklou, P.; Sinn, T.; Nirschl, H.; Gleiss, M. Grey box modelling of decanter centrifuges by coupling a numerical process model with a neural network. Minerals 2021, 11, 755.

(8) Asprion, N.; Böttcher, R.; Pack, R.; Stavrou, M. E.; Höller, J.; Schwientek, J.; Bortz, M. Gray-box modeling for the optimization of chemical processes. Chem. Ing. Techn. 2019, 91, 305−313.

(9) Yerramilli, S.; Tangirala, A. K. Detection and diagnosis of modelplant mismatch in MIMO systems using plant-model ratio. IFAC PapersOnLine 2016, 49, 266−271.

(10) Luo, J.; Abdullah, F.; Christofides, P. D. Model predictive control of nonlinear processes using neural ordinary differential equation models. Comput. Chem. Eng. 2023, 178, 108367.

(11) Wang, X.; Yan, C.; Qin, B. SVR-Based Hybrid Model for a Rotary Dryer. In 2009 International Conference on Computational Intelligence and Natural Computing. IEEE, 2009, Vol. 2, pp 537−540, DOI: 10.1109/CINC.2009.260.

(12) Zhou, P.; Song, H.; Wang, H.; Chai, T. Data-driven nonlinear subspace modeling for prediction and control of molten iron quality indices in blast furnace ironmaking. IEEE Trans. Control Syst. Technol. 2017, 25, 1761−1774.

(13) Ghosh, D.; Hermonat, E.; Mhaskar, P.; Snowling, S.; Goel, R. Hybrid modeling approach integrating first-principles models with subspace identification. Ind. Eng. Chem. Res. 2019, 58, 13533−13543.

(14) Sun, B.; Yang, C.; Wang, Y.; Gui, W.; Craig, I.; Olivier, L. A comprehensive hybrid first principles/machine learning modeling framework for complex industrial processes. J. Process Control 2020, 86, 30−43.

(15) Patel, N.; Corbett, B.; Trygg, J.; McCready, C.; Mhaskar, P. Subspace based model identification for an industrial bioreactor: Handling infrequent sampling using missing data algorithms. Processes 2020, 8, 1686.

(16) Luo, J.; Canuso, V.; Jang, J.B.; Wu, Z.; Morales-Guio, C. G.; Christofides, P. D. Machine learning-based operational modeling of an electrochemical reactor: Handling data variability and improving empirical models. Ind. Eng. Chem. Res. 2022, 61, 8399−8410.

(17) Chandrasekar, A.; Zhang, S.; Mhaskar, P. A Hybrid Hubspace-RNN based approach for Modelling of Non-Linear Batch Processes. Chem. Eng. Sci. 2023, 281, 119118.

(18) Nazemzadeh, N.; Malanca, A. A.; Nielsen, R. F.; Gernaey, K. V.; Andersson, M. P.; Mansouri, S. S. Integration of first-principle models and machine learning in a modeling framework: An application to flocculation. Chem. Eng. Sci. 2021, 245, 116864.

(19) Shah, P.; Sheriff, M. Z.; Bangi, M. S. F.; Kravaris, C.; Kwon, J. S. I.; Botre, C.; Hirota, J. Deep neural network-based hybrid modeling and experimental validation for an industry-scale fermentation process: Identification of time-varying dependencies among parameters. Chem. Eng. J. 2022, 441, 135643.

(20) Laursen, S. Ö .; Webb, D.; Ramirez, W. F. Dynamic hybrid neural network model of an industrial fed-batch fermentation process to produce foreign protein. Comput. Chem. Eng. 2007, 31, 163−170.

(21) Hussain, H.; Tamizharasan, P.; Rahul, C. Design possibilities and challenges of DNN models: A review on the perspective of end devices. Artif. Intell. Rev. 2022, 1−59.

(22) Hamid, A.; Hasan, A. H.; Azhari, S. N.; Harun, Z.; Putra, Z. A. Hybrid modelling for remote process monitoring and optimisation. Dig. Chem. Eng. 2022, 4, 100044.

(23) Karniadakis, G. E.; Kevrekidis, I. G.; Lu, L.; Perdikaris, P.; Wang, S.; Yang, L. Physics-informed machine learning. Nat. Rev. Phys. 2021, 3, 422−440.

(24) Guo, Y.; Cao, X.; Liu, B.; Gao, M. Solving partial differential equations using deep learning and physical constraints. Appl. Sci. 2020, 10, 5917.

(25) Yuan, L.; Ni, Y. Q.; Deng, X. Y.; Hao, S. A-PINN: Auxiliary physics informed neural networks for forward and inverse problems of nonlinear integro-differential equations. J. Comput. Phys. 2022, 462, 111260.

(26) Buzaev, F.; Gao, J.; Chuprov, I.; Kazakov, E. Hybrid acceleration techniques for the physics-informed neural networks: a comparative analysis. Mach. Learn. 2024, 113, 3675−3692.

(27) Tartakovsky, A. M.; Marrero, C. O.; Perdikaris, P.; Tartakovsky, G. D.; Barajas-Solano, D. Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems. Water Resour. Res. 2020, 56, No. e2019WR026731.

(28) Li, W.; Lee, K. M. Physics informed neural network for parameter identification and boundary force estimation of compliant and biomechanical systems. Int. J. Intell. Robotics Appl. 2021, 5, 313− 325.

(30) Cuomo, S.; Di Cola, V. S.; Giampaolo, F.; Rozza, G.; Raissi, M.; Piccialli, F. Scientific machine learning through physics−informed neural networks: Where we are and what’s next. J. Sci. Comput. 2022, 92, 88.

(31) Zheng, Y.; Wu, Z. Physics-informed online machine learning and predictive control of nonlinear processes with parameter uncertainty. Ind. Eng. Chem. Res. 2023, 62, 2804−2818.

(32) Alhajeri, M. S.; Abdullah, F.; Wu, Z.; Christofides, P. D. Physics-informed machine learning modeling for predictive control using noisy data. Chem. Eng. Res. Des. 2022, 186, 34−49.

(33) Patel, N.; Nease, J.; Aumi, S.; Ewaschuk, C.; Luo, J.; Mhaskar, P. Integrating Data-Driven Modeling with First-Principles Knowledge. Ind. Eng. Chem. Res. 2020, 59, 5103−5113.

![](images/4e950098616d777e0e3f36e81ab74db87e0a6a5729a8ec029ad5eaf23d5a9910.jpg)

CAS BIOFINDER DISCOVERY PLATFORMTM PRECISION DATA FOR FASTER DRUG DISCOVERY

CAS BioFinder helps you identify targets, biomarkers, and pathways

Unlock insights