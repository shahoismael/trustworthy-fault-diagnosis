# Physics-Informed Neural Networks for Dynamic Process Operations with Limited Physical Knowledge and Data

Mehmet Velioglu<sup>a,b</sup>, Song Zhai<sup>e</sup>, Sophia Rupprecht<sup>a,f</sup> , Alexander Mitsos<sup>c,a,d</sup>, Andreas Jupke<sup>e</sup>, Manuel Dahmen<sup>a,</sup>∗

<sup>a</sup> Institute of Climate and Energy Systems, Energy Systems Engineering (ICE-1), Forschungszentrum Jülich GmbH, Jülich 52425, Germany

<sup>b</sup> RWTH Aachen University, Aachen 52062, Germany

<sup>c</sup> JARA-ENERGY, Jülich 52425, Germany

<sup>d</sup> RWTH Aachen University, Process Systems Engineering (AVT.SVT), Aachen 52074, Germany

e RWTH Aachen University, Fluid Process Engineering (AVT.FVT), Aachen 52074, Germany

<sup>f</sup> Delft University of Technology, 2629 HZ, Delft, The Netherlands

Abstract: In chemical engineering, process data are expensive to acquire, and complex phenomena are dificult to fully model. We explore the use of physics-informed neural networks (PINNs) for modeling dynamic processes with incomplete mechanistic semi-explicit diferential-algebraic equation systems and scarce process data. In particular, we focus on estimating states for which neither direct observational data nor constitutive equations are available. We propose an easy-to-apply heuristic to assess whether estimation of such states may be possible. As numerical examples, we consider a continuously stirred tank reactor and a liquid-liquid separator. We find that PINNs can infer immeasurable states with reasonable accuracy, even if respective constitutive equations are unknown. We thus show that PINNs are capable of modeling processes when relatively few experimental data and only partially known mechanistic descriptions are available, and conclude that they constitute a promising avenue that warrants further investigation.

Keywords: Physics-informed neural networks, Chemical engineering, Dynamic process modeling, State estimation, Van de Vusse reaction, Liquid-liquid separator

## 1. Introduction

Dynamic operation and control of chemical and biotechnological processes are essential for eficient and sustainable production. Mathematical models describing the behavior of such processes are often classified concerning their degree of reliance on physical/chemical knowledge or data into three categories: (1) white-box or first-principle or mechanistic models, (2) black-box or data-driven models, and (3) gray box or hybrid models (Zendehboudi et al., 2018; Marquardt, 1996).

Black-box modeling relies on (measurement) data to establish a predictive relation between process inputs and outputs, thus avoiding the need for a mechanistic process description. In recent years, approaches involving deep neural networks (DNNs) have become particularly prominent data-driven models for process operations. DNNs can model nonlinear dependencies between multiple inputs and outputs (Goodfellow et al., 2016) but require extensive training data and often fail to make physically consistent predictions in scientific or engineering applications (Zendehboudi et al., 2018). In contrast, mechanistic process models are based on the governing physical and chemical laws of a system and suitable constitutive equations and comprise relatively few parameters that need to be estimated from data (von Stosch et al., 2014). They typically allow for physically consistent predictions. However, in chemical and biotechnological processes, complex phenomena such as reaction kinetics, coalescence, or sedimentation often lack a rigorous mathematical description, hindering the mechanistic modeling of such processes (Kahrs and Marquardt, 2008). Hybrid modeling combines mechanistic and data-driven modeling and aims to take advantage of the respective strengths and mitigate the respective weaknesses of the two ap proaches. Compared to purely data-driven models, suitably-designed hybrid models require less training data, make physically more consistent predictions, and (thus) extrapolate to a higher extent (Kahrs and Marquardt, 2007).

Hybrid models have been used extensively to model dynamic process operation problems if complete system knowledge is unavailable (Rofel and Betlem, 2006) and thus have become a crucial modeling tool for numerous tasks related to chemical process control (Asprion et al., 2019). Various types of hybrid model structures have been proposed over the years in the process systems engineering (PSE) community, with the sequential approach and the parallel approach being the most prominent structures. For instance, Psichogios and Ungar (1992) studied incorporating an artificial neural network to predict states lacking a constitutive description inside an otherwise mechanistic model for a fed-batch bioreactor (sequential approach). Su et al. (1992) proposed to correct the mismatch between a white-box model and process data from a polymer reaction system by a neural network (parallel approach). The parallel approach can also be combined with the sequential approach, i.e., a second mechanistic model is added after the parallel hybrid model to enforce physically consistent predictions, see, e.g., (Thompson and Kramer, 1994). Recently, the popularity of hybrid modeling in chemical engineering has been increasing again due to advancements in machine learning and the rise of digital twins in smart manufacturing (Yang et al., 2020). Some notable contemporary works on hybrid modeling are dedicated to the estimation of (spatio-)temporally varying parameters, which is related to the estimation of states with missing constitutive equations, the main topic of our article. Specifically, Shah et al. (2022) estimate time-varying parameters in fermentation processes, Pahari et al. (2024) estimate spatio-temporally varying difusivity in a reaction-difusion model, and Sitapure and Sang-Il Kwon (2023) estimate kinetic parameters in a batch crystallization process with a transformer architecture. For further applications of hybrid modeling in chemical engineering, we refer the reader to review papers by Sansana et al. (2021); Yang et al. (2020); Sharma and Liu (2022); Schweidtmann et al. (2021).

Physics-based regularization of DNNs gives rise to so-called physics-informed neural networks (PINNs), which have some similarities to hybrid models but are better regarded as a special variant of a data-driven model that is trained with available physical laws as constraints (Bradley et al., 2022). Specifically, in a PINN, the DNN acts as the sole prediction model, but it is informed about governing physical laws during training through additional terms in the loss function (Nabian and Meidani, 2019; Karniadakis et al., 2021). In contrast, hybrid models have distinct mechanistic and data-driven sub-models which jointly produce a prediction (Bradley et al., 2022; Schweidtmann et al., 2024).

The origins of physics-based regularization date back to (at least) the works of Lagaris et al. (1998) on solving ordinary and partial diferential equations using neural networks (NNs) as universal function approximators. This approach was originally not taken up widely, likely due to the general limitations of NN training at that time. However, Raissi et al. (2019) recently revisited the physics-based regularization approach using modern algorithms and tools for training and introduced the term PINN.

The original PINN architectures (Raissi et al., 2019; Nascimento et al., 2020) did not account for varying initial/boundary conditions or control inputs. However, Antonelo et al. (2021) showed that adding control inputs and initial conditions to the NN makes the PINN approach suitable for control applications. Another application of PINNs for control purposes was proposed by Arnold and King (2021), who pursued a state-space modeling approach based on PINNs, including initial conditions as inputs to the NN. However, separate networks are trained for each discretized control actuation instead of adding control inputs to the network.

Recently, PINNs have also seen a surge in chemical engineering applications, mainly in the form of physics-informed recurrent neural networks (Zheng et al., 2023). For instance, they have been applied in conjunction with model predictive control (MPC) to a continuously stirred tank reactor (CSTR) (Zheng et al., 2023) and a batch crystallization process (Wu et al., 2023), to control systems with noisy data (Alhajeri et al., 2022) and parametric uncertainty (Zheng and Wu, 2023), and to fluid flow problems, most notably flow field prediction in cyclone separators (Queiroz et al., 2021) and a Van de Vusse CSTR (Choi et al., 2022). Ji et al. (2021) developed PINNs that can address stif chemical kinetic problems.

While studies have shown that PINNs are promising model candidates for chemical engineering applications, open questions remain about their utility for state estimation. In general, state estimation is concerned with estimating the state of a given process utilizing measurement data and a mathematical process model (Barfoot, 2017; Gelb et al., 1974). State estimation is often performed with filtering techniques, e.g., the Kalman filter (Kalman, 1960b), which have recently also been combined with PINNs, see, e.g., (Tan et al., 2023; Arnold and King, 2021). PINNs have also been used to estimate unmeasured states directly, i.e., without the use of a state estimation technique. For instance, Raissi et al. (2020) estimated velocity and pressure fields from the concentration data of a passive scalar from flow field visualizations, using Navier Stokes equations as the physics knowledge. Recently, (Wu et al., 2023) showed that PINN with partial physics knowledge can estimate immeasurable states in a batch crystallization process by using the known governing equations of these states. The question, however, remains whether PINNs can estimate states for which neither direct observational data nor constitutive equations are available.

In the present work, we thus set out to answer the following two questions: (i) Can PINNs estimate immeasurable process states for which constitutive equations are not known? (ii) Under which conditions can we expect this to work? To this end, we will first conceptualize PINN-based dynamic process models in a setting of partially known mechanistic equations as well as measured and unmeasured process states. Specifically, we consider systems that (i) can be described by diferential-algebraic equations (DAEs) in principle, (ii) for which only partial mechanistic knowledge in the form of some known equations is available, and (iii) for which process data for some states is available. Regarding the PINN modeling, we follow the standard approach, as it was first introduced by Raissi et al. (2019), but with the extensions to initial states and control inputs by Antonelo et al. (2021). We propose the use of an incidence matrix as an easy-to-apply heuristic to a priori evaluate whether estimation of unmeasured states with a PINN may be possible. We then perform extensive numerical studies by using two fully-known mechanistic models to emulate situations where some, but not full, mechanistic knowledge is available for modeling purposes. Specifically, we study a CSTR model with Van de Vusse reaction from the literature (van de Vusse, 1964) and a liquid-liquid separator for which we develop a model by extending the model from Backi et al. (2018, 2019). We follow an in-silico approach to generate process data, i.e., we use the full-order mechanistic model, which in a real situation would not be available, to generate synthetic observational data. Controlling the amount and diversity of training data allows us to run extensive numerical experiments on the fitting and generalization capabilities of PINNs as well as vanilla neural network benchmark models, i.e., multilayer perceptrons. Following the taxonomy of process quantities and model equations by Marquardt (1996), we distinguish balance equations and constitutive equations and emulate situations with diferent degrees of mechanistic knowledge available for PINN model development.

The paper is structured as follows: Section 2 presents the proposed approach for PINN-based dynamic process modeling with incomplete physical knowledge, and our heuristic for assessing the state estimation capabilities of a PINN. Section 3 provides numerical examples and results for the CSTR, focusing on the physics-informed part of the PINN by varying the amount of physical knowledge provided. Section 4 provides numerical examples and results for the liquid-liquid separator, focusing on the data-driven part of the PINN by varying the number of measured properties provided as NN inputs. In all examples, the empirical findings are related to the results from the heuristic. Section 5 discusses the conclusion and future work.

## 2. Methods

## 2.1. Preliminaries

Raissi et al. (2019) introduced PINNs to find data-driven solutions to partial diferential equations (PDEs) utilizing DNNs. In their approach, they employ the NN to approximate the solution of a PDE problem. The inputs to the DNN are the spatio-temporal coordinates, and the DNN outputs are the states of the dynamic system. The DNN is trained in a semi-supervised manner, e.g., with small amounts of labeled data, i.e., process data with corresponding input/output relations, and large amounts of unlabeled data, i.e., collocation points in time and space where residuals of governing equations, i.e., the PDEs, are computed. These residuals constitute a loss term that penalizes the deviations of the DNN outputs from the governing equations. Thus, PINNs can learn to obey the physical laws of the system.

In their original form, PINNs do not account for control variables. The extension to control applica tions is, however, straightforward: Antonelo et al. (2021) added the control variable(s) and initial states as NN inputs. Considering initial states as network inputs means that the PINN model can be trained for various samples of initial states and control variables, facilitating extensive coverage of the state and control action spaces. The time domain of the PINN can be chosen according to the needs of the control scheme, e.g., in MPC applications, step-wise constant control inputs are often used. Thus, if the PINN time domain [0, T ] corresponds to the length of a step-wise constant control input, the control inputs from the perspective of the NN are not functions of time but constants. It is therefore, in general, necessary to distinguish PINN time t from process time τ and to chain the PINN predictions in order to simulate longer periods involving changing control inputs (cf. Figure 1). Note that in the numerical examples in Sections 3 and 4, for the sake of a simple implementation, we study varying control inputs which are however kept constant throughout the entire process duration, thus implying t = τ . For further details on including control actions into PINNs, we refer the reader to Antonelo et al. (2021) for integrating PINNs into MPC.

![](images/d2c2756b2986ff9c15ef802c52d668f2752270250e1b16d29d7393c13152d730.jpg)  
Fig. 1. Relationship between PINN time t and process time τ: The PINN time domain [0, T ] corresponds to the length of a step-wise constant control input. In general, PINN time t difers from process time τ and chaining of model predictions is required to simulate longer periods of time. Only if the control input is constant over the entire process duration, t and τ coincide. Measurements can come from an irregular grid.

## 2.2. PINN-based dynamic process modeling with partial physical knowledge

We consider the scenario where a partial mechanistic process model is available that can be used for physics-based regularization of a NN. We assume that this partial process model comes in the form of a semi-explicit diferential-algebraic equation (DAE) system (Brenan et al., 1996):

$$
\dot {\mathbf {x}} (t) = \boldsymbol {f} (\mathbf {x} (t), \mathbf {y} (t), \mathbf {u}),\tag{1a}
$$

$$
\mathbf {0} = \boldsymbol {g} (\mathbf {x} (t), \mathbf {y} (t), \mathbf {u})\tag{1b}
$$

Here, $\mathbf { x } ( t ) \in \mathbb { R } ^ { n _ { x } }$ is the diferential states vector, $\mathbf { y } ( t ) \in \mathbb { R } ^ { n _ { y } }$ is the algebraic states vector, and $\mathbf { u } \in \mathbb { R } ^ { n _ { u } }$ is the control inputs vector. The dot symbol ( ˙ ) denotes a time derivative. f denotes the right-hand side (RHS) of the ordinary diferential Equations 1a, and g is the RHS of the algebraic Equations 1b.

In a practical setting, some states might be impossible to measure (immeasurable), e.g., reaction rate constants, or some states might be impractical/expensive to measure, e.g., concentrations. The term unmeasured states covers both of these types and will be used throughout this work. We aim to estimate unmeasured process states with the available partial mechanistic knowledge and measurement data on other measured states. To this end, we sub-categorize the diferential and algebraic states into measured and unmeasured states, using superscripts m and $u ,$ respectively. This is a special case of the more general output equations used in observability analysis and control, see, $\mathrm { e . g . }$ , Lee and Markus (1967).

To predict the measured states $\mathbf { x } ^ { m } ( t ) \in \mathbb { R } ^ { n _ { x ^ { m } } } , \mathbf { y } ^ { m } ( t ) \in \mathbb { R } ^ { n _ { y ^ { m } } }$ and to estimate the unmeasured states $\mathbf { x } ^ { u } ( t ) \in \mathbb { R } ^ { n _ { x ^ { u } } } , \mathbf { y } ^ { u } ( t ) \in \mathbb { R } ^ { n _ { y ^ { u } } }$ , we use the neural network ${ \bf N N } _ { { \bf w } , { \bf b } }$ with weights w and biases $\mathbf { b } ,$ i.e., $[ \hat { \mathbf { x } } ( t ) , \hat { \mathbf { y } } ( t ) ] = \mathbf { N } \mathbf { N } _ { \mathbf { w } , \mathbf { b } } ( t , \mathbf { x } ^ { m } ( t _ { 0 } ) , \mathbf { u } )$ , where $\hat { { \mathbf x } } ( t )$ and ${ \hat { \mathbf { y } } } ( t )$ denote the NN predictions of the diferential and algebraic states, respectively. The network inputs are the time t, the initial values of the measured diferential states $\mathbf { x } ^ { m } ( t _ { 0 } )$ , and the control inputs u. The NN parameters w and b can be learned by minimizing the mean squared error loss, similar to Raissi et al. (2019); Antonelo et al. (2021):

$$
M S E _ {t o t a l} = M S E _ {d a t a} + \lambda_ {1} M S E _ {p h y s i c s} + \lambda_ {2} M S E _ {i n i t},
$$

$$
M S E _ {d a t a} = \frac {1}{n _ {x ^ {m}} N _ {d}} \sum_ {j = 1} ^ {N _ {d}} (\hat {\mathbf {x}} ^ {m} (t _ {j}) - \mathbf {x} ^ {m} (t _ {j})) ^ {2} + \frac {1}{n _ {y ^ {m}} N _ {d}} \sum_ {j = 1} ^ {N _ {d}} (\hat {\mathbf {y}} ^ {m} (t _ {j}) - \mathbf {y} ^ {m} (t _ {j})) ^ {2},\tag{2a}
$$

$$
M S E _ {p h y s i c s} = \frac {1}{n _ {x} N _ {e}} \sum_ {j = 1} ^ {N _ {e}} \left(\dot {\hat {\mathbf {x}}} (t _ {j}) - \pmb {f} (\hat {\mathbf {x}} (t _ {j}), \hat {\mathbf {y}} (t _ {j}), \mathbf {u} _ {j})\right) ^ {2} +\tag{2b}
$$

$$
\frac {\lambda_ {g}}{n _ {y} N _ {e}} \sum_ {j = 1} ^ {N _ {e}} \left(\pmb {g} (\hat {\mathbf {x}} (t _ {j}), \hat {\mathbf {y}} (t _ {j}), \mathbf {u} _ {j})\right) ^ {2},\tag{2c}
$$

$$
M S E _ {i n i t} = \frac {1}{n _ {x ^ {m}} N _ {i}} \sum_ {j = 1} ^ {N _ {i}} (\hat {\mathbf {x}} _ {j} ^ {m} (t _ {0}) - \mathbf {x} _ {j} ^ {m} (t _ {0})) ^ {2}\tag{2d}
$$

Here, $M S E _ { d a t a }$ corresponds to the loss term accounting for the measurement data, $M S E _ { p h y s i c s }$ corresponds to the loss term that is computed with the available physics knowledge (Equations 1a and 1b), and $M S E _ { i n i t }$ corresponds to a loss term that describes the mismatch between the NN predictions at $t = t _ { 0 }$ and the initial values $\mathbf { x } _ { j } ^ { m } ( t _ { 0 } )$ . N denotes the number of data points. Note that the subscript $j$ refers to finitely many samples taken at times $t _ { j }$ , with corresponding initial values $\mathbf { x } _ { j } ^ { m } ( t _ { 0 } )$ and control actions $\mathbf { u } _ { j }$ . We omit the latter two from the notation for simplicity. The subscripts $d , e ,$ , and i correspond to data points associated with $M S E _ { d a t a } , M S E _ { p h y s i c s }$ , and $M S E _ { i n i t }$ , respectively.

$\lambda _ { 1 }$ and $\lambda _ { 2 }$ denote the weights of the physics and initial condition loss terms, respectively, and $\lambda _ { g }$ establishes a weighting between the algebraic and the diferential equations in the physics loss term.

Note that for the calculation of $M S E _ { p h y s i c s }$ and $M S E _ { i n i t }$ no measurement data are needed. For $M S E _ { p h y s i c s }$ , we calculate the physics residuals using Equations 1a and 1b at randomly sampled time points $t = t _ { j }$ . For $M S E _ { i n i t }$ , we train the NN predictions $\hat { \mathbf { x } } ^ { m } ( t = t _ { 0 } )$ to comply with the initial values $\mathbf { x } ^ { m } ( t _ { 0 } )$ , again for randomly sampled values in a given range.

![](images/502472353d2e92fccfaba8fb78c1312e0e6cbe3a4f6eede4f81c898aea706762.jpg)  
Fig. 2. PINN-based dynamic process model with semi-explicit DAE physics model

## 2.3. Heuristic for assessing PINN state estimation capabilities

We propose a heuristic to a priori assess whether a PINN may be capable of estimating unmeasured process states by drawing inspirations from DAE solvability analysis, see, e.g., (Brenan et al., 1996). Our conjecture is that the PINN can leverage training data, i.e., samples for $\mathbf { x } _ { m } ( t _ { j } )$ and $\mathbf { y } _ { m } ( t _ { j } )$ , to “solve” the known Equations (1a) and (1b) for the unknown states $\mathbf { x } _ { u } ( t _ { j } )$ and ${ \bf y } _ { u } ( t _ { j } )$ at a point $t _ { j }$ . Specifically, our heuristic mimics structural index analysis by means of an incidence matrix (Duf and Gear, 1986; Gani and Cameron, 1992; Unger et al., 1995). In our PINN incidence matrix, the rows represent the RHSs of the known physics equations, i.e., f and g (see Equations (1a) and (1b)), and the columns represent the unmeasured process states $\mathbf { x } _ { u }$ and ${ \bf y } _ { u } .$ . Each occurrence of an unmeasured state in f and g is indicated by drawing a cross ( ) in the corresponding entry of the matrix. Note that the PINN uses AD to compute <sup>˙</sup>xˆ, i.e., the derivative of the NN outputs xˆ(t) with respect to the NN input t. Moreover, the NN learns to assemble state trajectories from the data provided at distinct time points $t _ { j }$ , and thus, it implicitly learns time-derivatives of the states. Consequently, we do not consider x˙ , i.e., the left-hand side (LHS) of Equations 1a, as unknowns but restrict our analysis to the RHSs f and g where no time-derivatives appear (see Equations (1a) and (1b)). This implies that we do not consider ${ \dot { x } } _ { j }$ as an occurrence of $x _ { j }$ when we assemble the incidence matrix.

We conjecture that the incidence matrix having a full-column rank, i.e., if exactly one cross in each column can be marked with a circle without marking more than one cross in a single row, constitutes an indicator for possible state estimation. A simple example of an incidence matrix for a PINN is given in Table 1. Note that an incidence matrix having more equations than unmeasured states, i.e., more rows than columns, is not a concern in itself. In fact, each additional equation may provide additional regularization to the NN and thus may be regarded as beneficial. We stress that the incidence matrix is a heuristic, i.e., it represents neither a necessary nor a suficient condition for state estimation with a PINN (see Sections SM5 and SM6 of the Supplementary Materials), and thus, it can give wrong results. Note that for fully-specified dynamic systems, necessary and suficient criteria for observability analysis exist, see, e.g., (Lee and Markus, 1967; Kou et al., 1973), based on trajectory information. Since we have an incomplete physics model, we instead construct the heuristic with a point-wise analysis, similar to the solvability analysis of equation systems (Brenan et al., 1996; Duf and Gear, 1986; Gani and Cameron, 1992; Unger et al., 1995). The practical construction and interpretation of the incidence matrix are demonstrated extensively in Sections 3 and 4.

Tab. 1. Incidence matrix for a PINN with a semi-explicit DAE physics model: Measurement data for training is available for $x _ { 1 } ^ { m }$ only. The unmeasured states $x _ { 2 } ^ { u }$ and $y ^ { u }$ shall be estimated from the data on $x _ { 1 } ^ { m }$ . The cross $( \times )$ denotes the occurrence of an unmeasured state in a physics equation. The incidence matrix has full-column rank, as it is possible to mark exactly one cross in each column without marking more than one cross in a single row.

<table><tr><td colspan="3">Known physics model (semi-explicit DAE)</td><td colspan="3">Incidence matrix</td></tr><tr><td>(a):</td><td> $\dot{x}_{1}^{m}$ </td><td>=  $x_{1}^{m} + x_{2}^{u}$ </td><td> $[f,g] \downarrow$ </td><td> $[x^{u},y^{u}] \rightarrow$ </td><td> $x_{2}^{u}$  |  $y^{u}$ </td></tr><tr><td>(b):</td><td> $\dot{x}_{2}^{u}$ </td><td>=  $3x_{1}^{m}$ </td><td>(a)</td><td>⊗</td><td></td></tr><tr><td>(c):</td><td>0</td><td>=  $x_{1}^{m}x_{2}^{u} + y^{u}$ </td><td>(b)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>(c)</td><td>×</td><td>⊗</td></tr></table>

## 2.4. Vanilla NN benchmark models

To compare the predictions of a PINN model with a purely data-driven benchmark, we choose a feed-forward artificial neural network (ANN), as ANNs are widely used and can have a similar network architecture as the PINN model, thus allowing us to study the efects of the physics-based regularization. To make the comparison as meaningful as possible, we use the same hyperparameters and training scheme for the PINN model and the vanilla ANN model. Still, the network architecture for the vanilla ANN is slightly diferent from that of the PINN in the sense that only the measured states can be network outputs, as no process data is available for the unmeasured states. We use the following loss function to train the vanilla ANN, omitting the physics-based regularization term in Equation (2a) but keeping the loss term for the initial conditions:

$$
\begin{array}{c} {M S E = M S E _ {d a t a} + \lambda_ {1} M S E _ {i n i t},} \\ {M S E _ {d a t a} = \frac {1}{n _ {x ^ {m}} N _ {d}} \sum_ {j = 1} ^ {N _ {d}} (\hat {\mathbf {x}} ^ {m} (t _ {j}) - \mathbf {x} ^ {m} (t _ {j})) ^ {2} + \frac {1}{n _ {y ^ {m}} N _ {d}} \sum_ {j = 1} ^ {N _ {d}} (\hat {\mathbf {y}} ^ {m} (t _ {j}) - \mathbf {y} ^ {m} (t _ {j})) ^ {2},} \\ {M S E _ {i n i t} = \frac {1}{n _ {x ^ {m}} N _ {i}} \sum_ {j = 1} ^ {N _ {i}} (\hat {\mathbf {x}} _ {j} ^ {m} (t _ {0}) - \mathbf {x} _ {j} ^ {m} (t _ {0})) ^ {2}} \end{array}
$$

![](images/0d8fcfb3f6ce19de7a4545c355f228a3be3e9e5c487112d2eb2282be49991701.jpg)  
Fig. 3. Schematic representation of the van de Vusse CSTR

## 3. Numerical example 1: Van de Vusse Reactor

We use the Van de Vusse (van de Vusse, 1964) CSTR, a common benchmark problem in the literature on nonlinear control applications (Chen et al., 1995), to investigate generalization, state estimation, and extrapolation capabilities of the PINN models under varying amounts of physical knowledge provided through physics equations. Thus, we focus on the physics regularization aspect of the PINN.

The van de Vusse reaction scheme reads:

$$
\begin{array}{l} \text {A} \xrightarrow {\mathrm{k} _ {1}} \text {B} \xrightarrow {\mathrm{k} _ {2}} \text {C}, \\ 2 \text {A} \xrightarrow {\mathrm{k} _ {3}} \text {D}. \end{array}
$$

Substance A is fed to the reactor with concentration $c _ { A , i n }$ and temperature $T _ { i n }$ . Substance B is the desired product, whereas substances C and D are unwanted byproducts. Heat is removed from the cooling jacket fluid with rate $\dot { Q } _ { K }$ by an external heat exchanger. The schematic of the CSTR is given in Figure 3. The dynamics of the reactor are given by the following nonlinear equations derived from component balances for substances A and B and energy balances for the reactor and the cooling jacket

(Chen et al., 1995):

$$
\dot {c} _ {A} (t) = \frac {\dot {V} (t)}{V _ {R}} (c _ {A, i n} - c _ {A} (t)) - k _ {1} (T) c _ {A} (t) - k _ {3} (T) c _ {A} (t) ^ {2},\tag{3a}
$$

$$
\dot {c} _ {B} (t) = - \frac {\dot {V} (t)}{V _ {R}} c _ {B} (t) + k _ {1} (T) c _ {A} (t) - k _ {2} (T) c _ {B} (t),\tag{3b}
$$

$$
\dot {T} (t) = \frac {\dot {V} (t)}{V _ {R}} (T _ {i n} - T (t)) - \frac {1}{\rho C _ {p}} [ k _ {1} (T) c _ {A} (t) \Delta H _ {A B} + k _ {2} (T) c _ {B} (t) \Delta H _ {B C}\tag{3c}
$$

$$
\left. + k _ {3} (T) c _ {A} (t) ^ {2} \Delta H _ {A D} \right] + \frac {k _ {w} A _ {R}}{\rho C _ {p} V _ {R}} (T _ {K} (t) - T (t)),
$$

$$
\dot {T} _ {K} (t) = \frac {1}{m _ {K} C _ {p K}} [ \dot {Q} _ {K} (t) + k _ {w} A _ {R} (T (t) - T _ {K} (t)) ]\tag{3d}
$$

Here, $c _ { A } ( t )$ and $c _ { B } ( t )$ denote the concentrations of substances A and $\mathrm { B } , T ( t )$ is the reactor temperature, and $T _ { K } ( t )$ is the cooling jacket temperature, assumed to be uniform in space. The aforementioned quantities correspond to the diferential states x of the Van de Vusse CSTR, i.e., $\mathbf { x } = [ c _ { A } , c _ { B } , T , T _ { K } ] ^ { T }$ . The flow rate $\dot { V } ( t )$ , and the heat transfer rate by the coolant $\dot { Q } _ { K } ( t )$ (heat removal) are the manipulated variables. Note that the dot notation in $\dot { V } ( t )$ and $\dot { Q } _ { K } ( t )$ indicates flow rates (as opposed to time derivatives). The reaction rate constants $k _ { i } ( T )$ correspond to the algebraic states y and are calculated using the Arrhenius equation:

$$
k _ {i} (T) = k _ {i 0} \exp \biggl (\frac {E _ {a , i}}{T} \biggr), \qquad i = 1, 2, 3\tag{4}
$$

All parameters listed in Equations (3) and (4) are given in Table 2.

During our preliminary tests, we observed that having values in a similar order of magnitude for the diferent PINN inputs and outputs improves the training stability and performance. However, when normalizing the outputs, the PINN physics equations must be scaled accordingly. Thus, we decided to make the time, states, and manipulated variables dimensionless and use dimensionless equations to calculate the physics loss. We give the dimensionless variables and equations in the Supplementary Materials. In addition, we normalize the PINN inputs, i.e., we scale the input features to values between -1 and 1.

To investigate the efects of varying physical knowledge, we create three diferent PINN models with increasing physics knowledge. Moreover, we investigate the performance of a vanilla ANN model to facilitate a comparison between the PINN model and a purely data-driven model. We list these models, the physics equations, the knowledge supplied to the PINN model, and the measured and unmeasured states in Table 3. Moreover, we give the network schematic of each model in the Supplementary Materials.

Tab. 2. Parameters for the van de Vusse CSTR, taken from (Chen et $\mathrm { a l . , }$ 1995).

<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>inlet molar flow rate of substance A</td><td> $c_{A,in}$ </td><td>5.10 mol/L</td></tr><tr><td>inlet temperature</td><td> $T_{in}$ </td><td>378.1 K</td></tr><tr><td>collision factor for reaction 1</td><td> $k_{10}$ </td><td> $1.287 \times 10^{12}$  1/h</td></tr><tr><td>collision factor for reaction 2</td><td> $k_{20}$ </td><td> $1.287 \times 10^{12}$  1/h</td></tr><tr><td>collision factor for reaction 3</td><td> $k_{30}$ </td><td> $9.043 \times 10^{9}$  L/(mol h)</td></tr><tr><td>activation energy for reaction 1</td><td> $E_{a,1}$ </td><td>-9758.3 K</td></tr><tr><td>activation energy for reaction 2</td><td> $E_{a,2}$ </td><td>-9758.3 K</td></tr><tr><td>activation energy for reaction 3</td><td> $E_{a,3}$ </td><td>-8560 K</td></tr><tr><td>enthalpy of reaction 1</td><td> $\Delta H_{AB}$ </td><td>4.2 kJ/molA</td></tr><tr><td>enthalpy of reaction 2</td><td> $\Delta H_{BC}$ </td><td>-11.0 kJ/molB</td></tr><tr><td>enthalpy of reaction 3</td><td> $\Delta H_{AD}$ </td><td>-41.85 kJ/molA</td></tr><tr><td>density</td><td> $\rho$ </td><td>0.9342 kg/L</td></tr><tr><td>heat capacity</td><td> $C_P$ </td><td>3.01 kJ/(kg K)</td></tr><tr><td>heat capacity of coolant</td><td> $C_{PK}$ </td><td>2.00 kJ/(kg K)</td></tr><tr><td>heat transfer coefficient of cooling jacket</td><td> $k_w$ </td><td>4032 kJ/(h m2K)</td></tr><tr><td>surface area of cooling jacket</td><td> $A_R$ </td><td>0.215 m2</td></tr><tr><td>reactor volume</td><td> $V_R$ </td><td>0.01 m3</td></tr><tr><td>coolant mass</td><td> $m_K$ </td><td>5.0 kg</td></tr></table>

<sub>t</sub> <sub>controls</sub> <sub>which</sub> <sub>we</sub>, <sub>for</sub> <sub>the</sub> <sub>sake</sub> <sub>of</sub> <sub>a</sub> <sub>simple</sub> <sub>i</sub>m<sup>plementation</sup>, <sup>keep</sup> <sup>constant</sup> <sup>throughout</sup> <sup>the</sup> <sup>invest</sup> be unmeasured depending on the case study (cf. Section 3.4). The time dependence of the states is not shown explicitly for brevity. The manipulated variables
˙VVR and ˙Q
K Tab. 3. Physical knowledge and output configuration (measured and unmeasured process states) for the Van de Vusse CSTR PINN models. In PINN-C,
T and
TK can also

<table><tr><td>Model name</td><td>Physics knowledge</td><td>Physics equations</td><td>Measured process states</td><td>Unmeasured process states</td></tr><tr><td>Vanilla ANN</td><td>None</td><td>None</td><td> $c_A, c_B, T, T_K$ </td><td>None</td></tr><tr><td>PINN-A</td><td>Mole balances with net reaction rates</td><td> $\dot{c}_A = \frac{\dot{V}}{V_R}(c_{A,in} - c_A) + r_A$  $\dot{c}_B = -\frac{\dot{V}}{V_R}c_B + r_B$ </td><td> $c_A, c_B, T, T_K$ </td><td> $r_A, r_B$ </td></tr><tr><td>PINN-B</td><td>Mole and energy balances with individual reaction rates</td><td> $\dot{c}_A = \frac{\dot{V}}{V_R}(c_{A,in} - c_A) - r_1 - r_3$  $\dot{c}_B = -\frac{\dot{V}}{V_R}c_B + r_1 - r_2$  $\dot{T} = \frac{\dot{V}}{V_R}(T_{in} - T) + \frac{k_w A_R}{\rho C_p V_R}(T_K - T)$  $-\frac{1}{\rho C_p}(r_1 \Delta H_{AB} + r_2 \Delta H_{BC} + r_3 \Delta H_{AD})$  $\dot{T}_K = \frac{1}{m_K C_{pK}}(\dot{Q}_K + k_w A_R(T - T_K))$ </td><td> $c_A, c_B, T, T_K$ </td><td> $r_1, r_2, r_3$ </td></tr><tr><td>PINN-C</td><td>Mole and energy balances with reaction rate expressions (without Arrhenius&#x27; law)</td><td> $\dot{c}_A = \frac{\dot{V}}{V_R}(c_{A,in} - c_A) - k_1 c_A - k_3 c_A^2$  $\dot{c}_B = -\frac{\dot{V}}{V_R}c_B + k_1 c_A - k_2 c_B$  $\dot{T} = \frac{\dot{V}}{V_R}(T_{in} - T) + \frac{k_w A_R}{\rho C_p V_R}(T_K - T)$  $-\frac{1}{\rho C_p}(k_1 c_A \Delta H_{AB} + k_2 c_B \Delta H_{BC} + k_3 c_A^2 \Delta H_{AD})$  $\dot{T}_K = \frac{1}{m_K C_{pK}}(\dot{Q}_K + k_w A_R(T - T_K))$ </td><td> $c_A, c_B, T, T_K$ </td><td> $k_1, k_2, k_3$ </td></tr></table>

## 3.1. Data set generation, training, and hyperparameter selection

We assume the operating ranges presented in Table 4, with a selected time interval for step-wise control changes of $T = 6 0 { \mathrm { s } } , { \mathrm { i . e . , } } t \in [ 0 , 6 0 ] { \mathrm { s } }$ . Data generation to calculate the physics loss term $M S E _ { p h y s i c s }$ and the initial condition loss term $M S E _ { i n i t }$ in Equations (2) are done by selecting $N _ { e } = 1 0 , 0 0 0$ collocation and $N _ { i } = 1 0 0$ initial value points. This selection is done using Latin Hypercube sampling (Iman et al., 1981).

For the process data generation, we use the explicit Runge-Kutta method of order 5, utilizing solve\_ivp solver from scipy.integrate module in Python (Virtanen et al., 2020; Dormand and Prince, 1980). We solve the full-order process model (Equations (3) and (4)) for time $t \in [ 0 , 6 0 ]$ s with random inputs for $\begin{array} { r } { c _ { A } ( t _ { 0 } ) , \ c _ { B } ( t _ { 0 } ) , \ T ( t _ { 0 } ) , \ T _ { K } ( t _ { 0 } ) , \ \frac { \dot { V } } { V _ { R } } , \ \dot { Q } _ { K } } \end{array}$ in the given ranges and keeping the manipulated variables $\frac { \dot { V } } { V _ { R } }$ and $\dot { Q } _ { K }$ constant throughout the investigated process duration of $6 0 \mathrm { s } ,$ with relative and absolute error of $1 \times 1 0 ^ { - 1 3 }$ and $1 \times 1 0 ^ { - 1 6 }$ respectively. We output each process trajectory on an equidistant time grid with step-size $\Delta t = 0 . 6 \mathrm { s }$ . Note that $\Delta t = 0 . 6 \mathfrak { s }$ s pertains to the granularity of the training/testing trajectories; the PINN at the prediction phase can make one-shot predictions for any time $t \in [ 0 , 6 0 ]$ s. Moreover, the training/testing trajectories could also be obtained from an irregular, i.e., non-equidistant, time grid. We create $N _ { t o t a l } = 1 0 0$ trajectories from which we select $N _ { t e s t } = 2 0$ trajectories for testing. For training, we use $N _ { t r a i n }$ trajectories, each one having $N _ { m } = 1 0 1$ data points. The total number of measurement points is thus $N _ { d } = N _ { t r a i n } N _ { m }$ . Specifically, we create two training sets from the 80 trajectories that are not used for the testing: First, we create a training set representing a low-data regime consisting of only $N _ { t r a i n } = 2 0$ training trajectories. Second, we create a training set representing a high-data regime consisting of $N _ { t r a i n } = 8 0$ training trajectories.

Tab. 4. Operating ranges for states and inputs in the Van de Vusse CSTR example. The lower bound is denoted by lb, and the upper bound is denoted by ub. These values are chosen to remain in the vicinity of a steady state. Extreme values refer to the minimum and maximum values appearing in a generated trajectory.

<table><tr><td rowspan="2">Variable</td><td rowspan="2">Unit</td><td colspan="2">Initial value</td><td colspan="2">Extreme value</td></tr><tr><td>lb</td><td>ub</td><td>min</td><td>max</td></tr><tr><td> $c_A$ </td><td>mol/L</td><td>2.14</td><td>2.57</td><td>1.74</td><td>2.74</td></tr><tr><td> $c_B$ </td><td>mol/L</td><td>0.87</td><td>1.09</td><td>0.87</td><td>1.28</td></tr><tr><td>T</td><td>K</td><td>387</td><td>403</td><td>385</td><td>403</td></tr><tr><td> $T_K$ </td><td>K</td><td>371</td><td>386</td><td>371</td><td>395</td></tr><tr><td> $\frac{\dot{V}}{V_R}$ </td><td>(1/h)</td><td>5</td><td>28.4</td><td>5</td><td>28.4</td></tr><tr><td> $Q_K$ </td><td>kJ/h</td><td>-2227</td><td>0</td><td>-2227</td><td>0</td></tr></table>

For the training, we use a hybrid strategy; we first start with the Adam optimizer (Kingma and Ba, 2017) and then switch to the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm (Liu and Nocedal, 1989). L-BFGS typically provides more accurate results for PINNs (Markidis, 2021);

however, it tends to get stuck in a local minimum if used directly (Markidis, 2021). Thus, Adam is first used to avoid local minima, and then L-BFGS is used for fine-tuning following the approach presented by (Markidis, 2021; Jin et al., 2021) since we could confirm their observation during our preliminary studies. We use a dynamic weighting scheme to decide on the weights $\lambda _ { i }$ in Equation (2), called inverse Dirichlet weighting (IDW) (Maddu et al., 2022). For this purpose, we used code snippets from the GitHub repository of Maddu et al. (2022). In preliminary studies, we found IDW to yield decent results but did not perform a systematic comparison of diferent weighting schemes. As evidenced by the results stated below, the PINNs consistently outperform the corresponding vanilla NN benchmark models. Thus, we refrained from further investigations into diferent weighting schemes. As IDW only works with firstorder optimizers, we apply it only during the Adam optimization step and then keep the final weights constant for the L-BFGS optimization step. We start the training process with the Adam optimizer fo 1000 epochs with a learning rate of 0.001. After that, we utilize the L-BFGS optimizer for 300 epochs. Mean squared error (MSE) is the metric used for minimization.

To determine the architecture parameters of the PINN, we utilize a grid search varying the following hyperparameters: activation function $\in \ \{ \operatorname { t a n h } , \operatorname { s i g m o i d } \}$ , depth of the hidden layers $\in \ \{ 1 , 2 , 3 , 4 \}$ and width of hidden layers (number of nodes) $\in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ . We investigate all four models for both data regimes. The tanh activation function performs best in all cases. Moreover, we find that the bestperforming width and depth of the hidden layers do not change across models but with the amount of training data. For the low-data regime, we find that a network with 2 hidden layers and 32 nodes performs the best. A network with 2 hidden layers and 64 nodes performs best for the high-data regime. The grid search is done with 5 randomly drawn data sets and 5 runs for each data set to account for variations in training/test split and weight initialization. Moreover, all the upcoming studies are also done using 5 data sets and 5 runs for each data set. The result of a run is reported as the average error over $N _ { t e s t } = 2 0$ trajectories.

## 3.2. Prediction of measured states

In this subsection, we investigate the generalization capabilities of the diferent PINN models and the vanilla ANN model listed in Table 3.

As can be seen from Figure 4, the prediction error for all states decreases with increasing physical knowledge supplied to the models, except for the reactant concentration $c _ { A }$ in the low-data regime. Moreover, all PINN models perform better than the vanilla ANN model in predicting measured states for both data regimes. A particularly interesting result is that PINN-A performs better at estimating the measured states T and $T _ { K }$ than the vanilla ANN, even though both models predict these states only based on data, i.e., PINN-A does not have energy balances, and $T$ and $T _ { K }$ do not appear in the mole balances. A possible explanation could be that, since PINN-A has physics knowledge on $c _ { A }$ and $c _ { B } ,$ , it reaches a lower loss value on $c _ { A }$ and $c _ { B }$ than the vanilla ANN and thus has more room to optimize for $T$ and $T _ { K }$

We conclude that the PINN models show strong generalization capabilities, better than the purely data-driven model, especially in the low-data regime.

![](images/6c72b0daec283e3fc847ce120399a1304c496c57988b7b5398178d6fe6e954d8.jpg)  
(a) Concentration of reactant $c _ { A } .$

![](images/3837aedb7bf2da3a070ba8123a3d3e6d74fa60ebbb1d41f860338cfc49c31c33.jpg)  
(b) Concentration of product $c _ { B } .$

![](images/c69d9f3cc778be0c7e52ad4f5bc0a5ebe92817da902bdff9f6a1e49450c983bf.jpg)  
(c) Temperature of the tank T .

![](images/91218e57cb92a98ab983d2e68dbbd65da46016308fa29339cd79b7542543793c.jpg)  
(d) Cooling jacket temperature $T _ { K }$  
Fig. 4. Test set error for the measured states for all models and data regimes. Boxplots show the results of 25 models (5 runs each for 5 data sets), averaged over the test set of each model. The error metric is the mean absolute percentage error (MAPE).

## 3.3. Algebraic state estimation

We now investigate if the PINN models can predict unmeasured algebraic states $\mathbf { y } ^ { u }$ with reasonable accuracy. First, we conduct an incidence matrix analysis for each PINN model. Table 5 shows that all PINN models have a full-column rank incidence matrix, suggesting that state estimation is possible in all cases.

Table 6 reports test errors for the unmeasured algebraic states. Since the compared quantities are

Tab. 5. Incidence matrices of PINN-A, PINN-B, and PINN-C for the Van de Vusse reactor example. If an unmeasured state appears in an equation, it is marked with a cross. Encircled crosses show feasible assignments of states to equations.

(a) Incidence matrix of PINN-A. Matrix has full-column rank.

<table><tr><td>$ [f,g]\downarrow\quad[x^{u},y^{u}]\rightarrow $</td><td>$ r_{A} $</td><td>$ r_{B} $</td></tr><tr><td>Eqn. for $ \dot{c}_{A} $</td><td>⊗</td><td></td></tr><tr><td>Eqn. for $ \dot{c}_{B} $</td><td></td><td>⊗</td></tr></table>

(b) Incidence matrix of PINN-B. Matrix has full-column rank.

<table><tr><td> $[f,g] \downarrow$   $[x^{u},y^{u}] \rightarrow$ </td><td> $r_{1}$ </td><td> $r_{2}$ </td><td> $r_{3}$ </td></tr><tr><td>Eqn. for  $\dot{c}_{A}$ </td><td>⊗</td><td></td><td>×</td></tr><tr><td>Eqn. for  $\dot{c}_{B}$ </td><td>×</td><td>⊗</td><td></td></tr><tr><td>Eqn. for  $\dot{T}$ </td><td>×</td><td>×</td><td>⊗</td></tr><tr><td>Eqn. for  $\dot{T}_{K}$ </td><td></td><td></td><td></td></tr></table>

(c) Incidence matrix of PINN-C. Matrix has full-column rank.

<table><tr><td> $[f,g] \downarrow$   $[x^{u},y^{u}] \rightarrow$ </td><td> $k_{1}$ </td><td> $k_{2}$ </td><td> $k_{3}$ </td></tr><tr><td>Eqn. for  $\dot{c}_{A}$ </td><td>⊗</td><td></td><td>×</td></tr><tr><td>Eqn. for  $\dot{c}_{B}$ </td><td>×</td><td>⊗</td><td></td></tr><tr><td>Eqn. for  $\dot{T}$ </td><td>×</td><td>×</td><td>⊗</td></tr><tr><td>Eqn. for  $\dot{T}_{K}$ </td><td></td><td></td><td></td></tr></table>

diferent for the diferent models, a direct comparison between the models is not justified. However, we can conclude that all models can predict the unmeasured algebraic states with acceptable accuracy (less than 10 % mean absolute percentage error), except $r _ { 3 }$ in PINN-B. We also observe that the accuracy gap between the low and high data regimes decreases as the provided physics knowledge increases.

Note that the estimated algebraic states, i.e., the net reaction rates (PINN-A), the individual reaction rates (PINN-B), and the reaction rate constants (PINN-C), were not only unmeasured, i.e., no process data was used for training, but also their corresponding constitutive equations were not provided. This example thus shows that PINNs can, in certain situations, infer immeasurable states, even if respective constitutive equations are unknown.

Tab. 6. Estimation accuracy for the unmeasured algebraic states $\mathbf { y } ^ { u }$ on the test set for all PINN models and data regimes. Results are averaged over 25 models (5 runs each for 5 data sets). The error metric is the mean absolute percentage error (MAPE ).

<table><tr><td>Model</td><td>Unmeasured algebraic state</td><td>Low data regime</td><td>High data regime</td></tr><tr><td rowspan="2">PINN-A</td><td> $r_A$ </td><td>4.71%</td><td>2.61%</td></tr><tr><td> $r_B$ </td><td>9.31%</td><td>5.12%</td></tr><tr><td rowspan="3">PINN-B</td><td> $r_1$ </td><td>4.33%</td><td>3.43%</td></tr><tr><td> $r_2$ </td><td>9.15%</td><td>7.27%</td></tr><tr><td> $r_3$ </td><td>11.99%</td><td>10.42%</td></tr><tr><td rowspan="3">PINN-C</td><td> $k_1$ </td><td>3.59%</td><td>2.90%</td></tr><tr><td> $k_2$ </td><td>6.84%</td><td>6.13%</td></tr><tr><td> $k_3$ </td><td>7.14%</td><td>6.98%</td></tr></table>

## 3.4. Diferential state estimation

We study PINN-C and create three diferent settings to empirically gauge the diferential state estimation capabilities. In the first setting, we assume that state $c _ { A }$ is unmeasured i.e., $\mathbf { x } ^ { u } = [ c _ { A } ] ^ { T }$ . In the second setting, T is unmeasured, i.e., $\mathbf { x } ^ { u } = [ T ] ^ { T }$ . In the third setting, $T _ { K }$ is unmeasured, i.e., $\mathbf { x } ^ { u } = [ T _ { K } ] ^ { T }$ For all settings, the algebraic states $k _ { 1 } , k _ { 2 } ,$ and k are also unmeasured, i.e., $\mathbf { y } ^ { u } = [ k _ { 1 } , k _ { 2 } , k _ { 3 } ] ^ { T }$

In the first setting, $\mathbf { x } ^ { u } = [ c _ { A } ] ^ { T }$ , we do not obtain a full-column rank incidence matrix, as can be seen from Table $\mathrm { 7 a } .$ , whereas in the other two settings we do (Tables 7b and 7c).

Tab. 7. Incidence matrices of PINN-C with $\mathbf { x } ^ { u } = [ c _ { A } ] ^ { T } , \mathbf { x } ^ { u } = [ T ] ^ { T }$ and $\mathbf { x } ^ { u } = [ T _ { K } ] ^ { T }$ for Van de Vusse reactor example. If an unmeasured state appears in an equation, it is marked with a cross. Encircled crosses show feasible assignments of states to equations.

(a) Incidence matrix of PINN-C with $\mathbf { x } ^ { u } = \left[ c _ { A } \right] ^ { T }$ (setting 1). Matrix does not have full-column rank.

<table><tr><td> $[f,g] \downarrow$   $[x^{u},y^{u}] \rightarrow$ </td><td> $c_{A}$ </td><td> $k_{1}$ </td><td> $k_{2}$ </td><td> $k_{3}$ </td></tr><tr><td>Eqn. for  $\dot{c}_{A}$ </td><td>×</td><td>×</td><td></td><td>×</td></tr><tr><td>Eqn. for  $\dot{c}_{B}$ </td><td>×</td><td>×</td><td>×</td><td></td></tr><tr><td>Eqn. for  $\dot{T}$ </td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Eqn. for  $\dot{T}_{K}$ </td><td></td><td></td><td></td><td></td></tr></table>

(b) Incidence matrix of PINN-C with $\mathbf { x } ^ { u } = [ T ] ^ { T }$ (setting 2). Matrix has full-column rank.

<table><tr><td> $[f,g] \downarrow$ </td><td> $[x^{u},y^{u}] \rightarrow$ </td><td>T</td><td> $k_{1}$ </td><td> $k_{2}$ </td><td> $k_{3}$ </td></tr><tr><td>Eqn. for  $\dot{c}_{A}$ </td><td></td><td></td><td>⊗</td><td></td><td>×</td></tr><tr><td>Eqn. for  $\dot{c}_{B}$ </td><td></td><td></td><td>×</td><td>⊗</td><td></td></tr><tr><td>Eqn. for  $\dot{T}$ </td><td></td><td>×</td><td>×</td><td>×</td><td>⊗</td></tr><tr><td>Eqn. for  $\dot{T}_{K}$ </td><td></td><td>⊗</td><td></td><td></td><td></td></tr></table>

(c) Incidence matrix of PINN-C with $\mathbf { x } ^ { u } = [ T _ { K } ] ^ { T }$ (setting 3). Matrix has full-column rank.

<table><tr><td> $[f,g] \downarrow$   $[x^{u},y^{u}] \rightarrow$ </td><td> $T_{K}$ </td><td> $k_{1}$ </td><td> $k_{2}$ </td><td> $k_{3}$ </td></tr><tr><td>Eqn. for  $\dot{c}_{A}$ </td><td></td><td>⊗</td><td></td><td>×</td></tr><tr><td>Eqn. for  $\dot{c}_{B}$ </td><td></td><td>×</td><td>⊗</td><td></td></tr><tr><td>Eqn. for  $\dot{T}$ </td><td>×</td><td>×</td><td>×</td><td>⊗</td></tr><tr><td>Eqn. for  $\dot{T}_{K}$ </td><td>⊗</td><td></td><td></td><td></td></tr></table>

In Figure 5, we see that the PINN model with $\mathbf { x } ^ { u } = [ c _ { A } ] ^ { T }$ (setting 1) indeed fails to estimate $c _ { A }$ as indicated by the incidence matrix (Table 7a). In contrast, the MAPE values suggest that the PINN models with $\mathbf { x } ^ { u } = [ T ] ^ { T }$ (setting 2) and $\mathbf { x } ^ { u } = [ T _ { K } ] ^ { T }$ (setting 3) yield good results for the estimation of the respective unmeasured diferential states T and $T _ { K }$ . However, when we compare the results to the case where all diferential states were measured (cf. Figure 4), we see that the MAPE values are about 20 times higher in case of $T _ { K }$ , and around 5 times higher in case of T . More importantly, as the ranges of $T$ and $T _ { K }$ are quite low compared to the actual values (cf. Table 4), the MAPE values can be deceptively low. Thus, as a more reliable measure of goodness of fit, we evaluate the coeficient of determination $\textstyle ( \mathrm { R } ^ { 2 } )$ . As can be seen from Figure $6 ,$ the PINN-C model with $\mathbf { x } ^ { u } = [ T ] ^ { T }$ can successfully predict the unmeasured diferential state $T ,$ with $\mathrm { R ^ { 2 } }$ scores above 0.90. However, the PINN-C model with $\mathbf { x } ^ { u } = [ T _ { K } ] ^ { T }$ essentially fails to estimate $T _ { K }$ , with $\mathrm { R ^ { 2 } }$ scores ranging between 0.15 and 0.85.

In state estimation theory (Kalman, 1960a; Lee and Markus, 1967), a system is called observable if the initial values of unmeasured states can be estimated uniquely using the information on measured states and a mathematical process model. In the particular example considered here, the initial state (and thus the trajectory) of T can be estimated uniquely by the PINN using the data on the measured states and the built-in physical knowledge, whereas this is not the case for $T _ { K }$ . Transferring observability conditions for nonlinear dynamic models, see, e.g., Lee and Markus (1967); Kou et al. (1973), to PINNs is not straightforward and thus considered beyond the scope of this paper.

![](images/62f471713b38a646250e1d92806f7ff18aa2dedc92b08211c2d6a469b4074dc4.jpg)

![](images/c62b3f0d6db0c65906006ced1b305d4b10b87774e511e20a3b4fcf599673143d.jpg)  
Fig. 5. Test set errors for the unmeasured diferential states of PINN-C with $\mathbf { x } ^ { u } = [ c _ { A } ] ^ { T } , \mathbf { x } ^ { u } = [ T ] ^ { T }$ and $\mathbf { x } ^ { u } = [ T _ { K } ] ^ { T }$ for the Van de Vusse reactor example. All error values correspond to the respective unmeasured diferential state, $\mathbf { e . g . }$ , the value for the model with $\mathbf { x } ^ { u } = [ c _ { A } ] ^ { \mathsf { T } }$ shows the error of $c _ { A }$ . Boxplots show the results of 25 models (5 runs each for 5 data sets), averaged over the test set of each model. The error metric is the mean absolute percentage error (MAPE).

![](images/59d3bcf847e50655a7fe2254a316e3b44b1b8aaf110c83e42d63e79eca6658f8.jpg)  
Fig. 6. $R ^ { 2 }$ values for PINN-C with $\mathbf { x } ^ { u } = [ T _ { K } ] ^ { T }$ and $\mathbf { x } ^ { u } = [ T ] ^ { T }$ (test set goodness of fit).

## 3.5. Extrapolation capabilities

We now explore if the PINN can extrapolate beyond the bounds of the process data supplied for training. For this purpose, we create a set of test trajectories with the initial value of $c _ { A 0 }$ out of the bounds of $c _ { A 0 }$ in the training trajectories. We term this set extrapolation set. In Table 8, respective ranges for the inputs $c _ { A 0 }$ can be seen for training, test, and extrapolation sets. In contrast to purely data-driven models, the PINNs may also learn the system dynamics from the physics residuals. Nevertheless, we still expect a lower accuracy in the extrapolation regime since we do not provide measurement data about that regime during training.

As can be seen from Figure 7, the test errors on both the test and the extrapolation sets are much lower for the PINN models compared to the vanilla ANN model. We also observe that PINN-C, the model with the most physics knowledge, has the lowest diference in accuracy between test and extrapolation sets. Thus, we conclude that the PINN models can extrapolate better than the non-informed NN, and the extrapolation accuracies tend to increase when more physics knowledge is incorporated into the PINN.

Tab. 8. Ranges of the initial state $c _ { A 0 }$ for the trajectories used in the training, test, and extrapolation sets.

<table><tr><td>Set</td><td>Initial State</td><td>Lower bound</td><td>Upper bound</td></tr><tr><td>Training set</td><td> $c_{A0}$ </td><td> $2.14\frac{mol}{L}$ </td><td> $2.57\frac{mol}{L}$ </td></tr><tr><td>Test set</td><td> $c_{A0}$ </td><td> $2.14\frac{mol}{L}$ </td><td> $2.57\frac{mol}{L}$ </td></tr><tr><td>Extrapolation set</td><td> $c_{A0}$ </td><td> $1.71\frac{mol}{L}$ </td><td> $2.14\frac{mol}{L}$ </td></tr></table>

![](images/800974dbc5cf55739845a6c514ae395b804b13fa6dd1f316693f6740c73a74b0.jpg)  
Fig. 7. Test and extrapolation set errors of the reactant concentration $c _ { A }$ for all models. The results are on the low-data regime. The error metric is the mean absolute percentage error (MAPE).

## 4. Numerical example 2: Liquid-liquid separator

With this second example, we aim to investigate the generalization and state estimation capabilities of the PINN models under varying amounts of measured physical property data supplied as additional inputs to the NN. Thus, we now focus on the data-driven part of the PINN.

The dynamic liquid-liquid separator model shown below is based on the work of Backi et al. (2018, 2019). We included extensions for swarm sedimentation in the aqueous phase, convection terms for the drop size distribution (DSD) in the dense-packed zone (DPZ) analogously to Backi et al. (2018), and a state-of-the-art coalescence model (Henschke, 1995). The chosen swarm model (Mersmann, 1980) was also used to model liquid-liquid columns (Kampwerth et al., 2020) and takes the form of Stokes’ law (Stokes, 2009) for diminishing hold-ups. Stokes’ law was experimentally confirmed to model the outlet hold-up of liquid-liquid separator accurately (Ye et al., 2023).

The considered liquid-liquid separator shown in Fig. 8 is divided into three subsystems: light (organic) phase, dense-packed zone (DPZ), and heavy (aqueous) phase. The light phase is assumed to be free of the dispersed phase; the DPZ is assumed to have a constant hold-up $\bar { \epsilon } _ { p }$ (volume phase fraction of dispersed phase) of 0.9, and the heavy phase contains dispersed organic droplets but mostly water. The total volume flow $\dot { V } _ { \mathrm { i n } }$ enters the separator in the heavy phase with the dispersed light phase described by the Sauter mean diameter $d _ { 3 2 }$ and phase fraction of organic phase $\epsilon _ { \mathrm { i n } } .$ . In the heavy phase, the dispersed droplets sediment upwards as a droplet swarm, resulting in the volume flow of organic droplets to the DPZ $\dot { V _ { s } }$ . In the DPZ, drop-drop coalescence is assumed to be negligible, and only droplet-interface coalescence occurs, giving the volume flow of coalesced drops $\dot { V } _ { c }$ to the light phase. The volume flow of water $\dot { V } _ { w }$ from the aqueous phase to the DPZ stems from trapped water between the sedimented droplets and coalesced drops at the interface of the organic phase. By applying a volume balance to the DPZ and assuming a constant hold-up, the volume flow of water can be expressed by the sedimentation and coalescence rate. The outlet volume flow of the aqueous $\dot { V } _ { \mathrm { a q , o u t } }$ and organic phase $\dot { V } _ { \mathrm { o r g , o u t } }$ are the manipulated variables of the settler, $\mathbf { u } = [ \dot { V } _ { \mathrm { a q , o u t } } , \dot { V } _ { \mathrm { o r g , o u t } } ] ^ { T }$

The following volume balance equations are obtained after transforming the volume of a cylindrical segment to the height of each segment (Backi et al., 2018):

$$
\dot {h} _ {L} (t) = \frac {\dot {V} _ {i n} (t) - \dot {V} _ {a q , o u t} (t) - \dot {V} _ {o r g , o u t} (t)}{2 L \sqrt {h _ {L} (t) (2 r - h _ {L} (t))}},\tag{5a}
$$

$$
\dot {h} _ {\mathrm{DPZ}} (t) = \frac {\dot {V} _ {i n} (t) - \dot {V} _ {a q , o u t} (t) - \dot {V} _ {c} (t)}{2 L \sqrt {h _ {\mathrm{DPZ}} (t) (2 r - h _ {\mathrm{DPZ}} (t))}},\tag{5b}
$$

$$
\dot {h} _ {\mathrm{aq}} (t) = \frac {\dot {V} _ {i n} (t) - \dot {V} _ {a q , o u t} (t) - \dot {V} _ {s} (t) \frac {1}{\bar {\epsilon} _ {p}} + \dot {V} _ {c} (t) \frac {1 - \bar {\epsilon} _ {p}}{\bar {\epsilon} _ {p}}}{2 L \sqrt {h _ {\mathrm{aq}} (t) (2 r - h _ {\mathrm{aq}} (t))}}\tag{5c}
$$

Here, $h _ { L }$ , h<sub>DPZ</sub>, and $h _ { \mathrm { a q } }$ are the heights of the total liquid, the DPZ, and the aqueous phase, respectively, each measured from the bottom of the separator. They constitute the diferential states x of the system.

![](images/bdc2942a7df8195fcac2e778ed71319a8ffc6eef8e52a75c00a3f47337652d7c.jpg)  
Fig. 8. Separator with the light phase (top), dense-packed zone (center), heavy phase (bottom), and flows. The dispersion with the properties phase fraction of dispersed phase $\epsilon _ { \mathrm { i n } } ,$ Sauter mean diameter $d _ { 3 2 } ,$ , and total volume flow rate $V _ { \mathrm { i n } }$ enters the heavy phase from the left. The heavy phase has the following outgoing flows: sedimentation rate $\dot { V } _ { s } ,$ , water flow rate $\dot { V } _ { w }$ and outlet flow $\dot { V } _ { \mathrm { a q , o u t } }$ . The dense-packed zone is modeled with a constant hold-up $\bar { \epsilon } _ { p } = 0 . 9$ and a coalescence rate $\dot { V } _ { c }$ The light phase has the outlet $\dot { V } _ { \mathrm { o r g , o u t } }$

Note that the volume flow rates $\dot { V } _ { i n } , \dot { V } _ { a q , o u t } , \dot { V } _ { o r g , o u t } , \dot { V } _ { c }$ , and $\dot { V } _ { s }$ are algebraic quantities. Similar to the CSTR case (Section 3), use of the dot notation to indicate flow rates is motivated by standard practice in engineering. In contrast, the dot symbols on the LHS of Equations $\mathrm { ( 5 a ) } - \mathrm { ( 5 c ) }$ denote derivatives with respect to time. In the full-order mechanistic model (see Section SM4 of the Supporting Materials), the coalescence and sedimentation rates $\dot { V } _ { c }$ and $\dot { V _ { s } }$ are functions of $h _ { \mathrm { a q } }$ and $h _ { \mathrm { { D P Z } } }$ , boundary conditions at the entrance and physical properties such as the Sauter mean diameter $d _ { 3 2 }$ and the coalescence parameter $r _ { \mathrm { v } } .$ . As they cannot be measured, we aim to estimate $\dot { V } _ { c }$ and $\dot { V } _ { s }$ with a PINN model that uses only Equations $\mathrm { ( 5 a ) } - \mathrm { ( 5 c ) }$ as available physical knowledge, i.e., the constitutive equations for the coalescence and sedimentation rates $\dot { V } _ { c }$ and $\dot { V _ { s } }$ are assumed to be unknown.

We assume that the total liquid height in the separator is constant, as this is the usual mode of operation. Then, the diferential balance Equation (5a) becomes an algebraic relation, serving as a closure condition for the flows in and out of the separator:

$$
\dot {V} _ {\mathrm{in}} (t) - \dot {V} _ {\mathrm{aq,out}} (t) - \dot {V} _ {\mathrm{org,out}} (t) = 0
$$

We also aim to investigate whether the PINN can take advantage of measurement data on $d _ { 3 2 }$ and $r _ { \mathrm { v } }$ that are provided as input to the NN although these quantities do not appear in the physics Equations $\mathrm { ( 5 a ) } - \mathrm { ( 5 c ) }$ . Thus, we create three diferent PINN models with an increasing number of physical properties added as inputs to the NN, along with a vanilla NN for comparison (cf. Table 9). Moreover, we show the network structure of the models in the Supplementary Materials. As in Section 3, we make the time, states, and manipulated variables dimensionless for better performance and stability during NN training.

We give the dimensionless variables and equations in the Supplementary Materials.

Tab. 9. Inputs of the models for the liquid-liquid separator.

<table><tr><td>Model name</td><td>Network inputs</td></tr><tr><td>Vanilla ANN</td><td> $t, h_{\text{aq}}(t_0), h_{\text{DPZ}}(t_0), \dot{V}_{\text{aq,out}}, \dot{V}_{\text{org,out}}$ </td></tr><tr><td>Base PINN</td><td> $t, h_{\text{aq}}(t_0), h_{\text{DPZ}}(t_0), \dot{V}_{\text{aq,out}}, \dot{V}_{\text{org,out}}$ </td></tr><tr><td>PINN-d32</td><td> $t, h_{\text{aq}}(t_0), h_{\text{DPZ}}(t_0), \dot{V}_{\text{aq,out}}, \dot{V}_{\text{org,out}}, d_{32}$ </td></tr><tr><td>PINN-d32-rv</td><td> $t, h_{\text{aq}}(t_0), h_{\text{DPZ}}(t_0), \dot{V}_{\text{aq,out}}, \dot{V}_{\text{org,out}}, d_{32}, r_v$ </td></tr></table>

## 4.1. Data set generation, training, and hyperparameter selection

We investigate the phase separation of n-butyl acetate dispersed in water in a pilot-scale separator. The radius and length of the separator are $R = 0 . 1 \mathrm { m }$ and $L = 1 . 8 \mathrm { m }$ . We take the operating ranges presented in Table 10, with a selected time interval for step-wise control changes and thus process time of 20 s, i.e., $t \in [ 0 , 2 0 ]$ s. We keep the manipulated variables constant throughout the process time for implementation reasons, as done in Section 3. Data generation to calculate the physics loss term $M S E _ { p h y s i c s }$ and the initial condition loss term $M S E _ { i n i t }$ in Equations (2) are done by selecting $N _ { e } = 1 0 0 0 0$ collocation and $N _ { i } ~ = ~ 1 0 0$ initial value points. Again, the selection is done using Latin Hypercube sampling. We choose the bounds for the initial states $\mathbf { x } ( t _ { 0 } )$ corresponding to the minimum and maximum values of the states x in the operating range of the process (see Table 10), and perform similarly for the control variables u. We use the explicit Runge-Kutta method of order 5 for the process data generation, utilizing solve\_ivp solver from scipy.integrate module in Python (Virtanen et al., 2020; Dormand and Prince, 1980), with a relative and absolute error of $1 \times 1 0 ^ { - 1 2 }$ . We output the trajectories on a time grid $t \in [ 0 , 2 0 ]$ s with $\Delta t = 0 . 1 \mathrm { s }$ . Nonphysical states, such as flooding of the separator with the DPZ, are addressed by early termination. The resulting shorter trajectories are kept in the data set; however, the step size $\Delta t$ is adjusted to keep a constant number of grid points among all trajectories. We create $N _ { t o t a l } = 2 0 0$ trajectories. From these, we select $N _ { t e s t } = 4 0$ trajectories for testing. For training, we use $N _ { t r a i n }$ trajectories, each having $N _ { m } = 2 0 1$ data points. The total number of measurement points are $N _ { d } = N _ { t r a i n } N _ { m }$ . Again, we create two training sets from the remaining 160 trajectories not used for testing: a training set representing a low-data regime consisting of only $N _ { t r a i n } = 2 0$ training trajectories, and a training set representing a high-data regime consisting of $N _ { t r a i n } = 1 6 0$ training trajectories.

We use the strategy described in Section 3.1 for the training and hyperparameter optimization. For the low-data regime, we find that a network with two hidden layers and 32 nodes performs the best. A network with two hidden layers and 128 nodes performs best for the high-data regime. The tanh activation function performs best in all cases. The grid search is done with 5 data sets and 5 runs for each data set to account for variations in training/test split and weight initialization. Moreover, we use a sigmoid activation function for the output layer to bound the output values between 0 and 1 to prevent the square root in the denominator of Equations (5b) and (5c) from attaining negative values during PINN training. The following numerical studies are done with 5 data sets and 5 runs for each data set. The results of the runs are reported as the average error over $N _ { t e s t } = 2 0$ trajectories.

Tab. 10. Ranges for initial states and inputs for the liquid-liquid separator example.

<table><tr><td>Variable</td><td>Lower bound</td><td>Upper bound</td></tr><tr><td> $h_{\text{aq,0}}$ </td><td>0.090 m</td><td>0.110 m</td></tr><tr><td> $h_{\text{DPZ,0}}$ </td><td>0.108 m</td><td>0.132 m</td></tr><tr><td> $\dot{V}_{\text{aq,out}}$ </td><td> $4.5 \times 10^{-4} \text{m}^{3}/\text{s}$ </td><td> $5.5 \times 10^{-4} \text{m}^{3}/\text{s}$ </td></tr><tr><td> $\dot{V}_{\text{org,out}}$ </td><td> $2.0 \times 10^{-4} \text{m}^{3}/\text{s}$ </td><td> $5.0 \times 10^{-4} \text{m}^{3}/\text{s}$ </td></tr><tr><td> $d_{32}$ </td><td> $9.0 \times 10^{-4} \text{m}$ </td><td> $1.1 \times 10^{-3} \text{m}$ </td></tr><tr><td> $r_{\text{v}}$ </td><td>0.033</td><td>0.043</td></tr></table>

## 4.2. Results

We show the incidence matrix of the liquid-liquid separator PINN model in Table 11. The unmeasured NN outputs are the algebraic states $\mathbf { y } ^ { u } = [ \dot { V } _ { c } , \dot { V } _ { s } ]$ . The total liquid height is known and constant, i.e., $\dot { h } _ { L } = 0$ . The incidence matrix shows a feasible assignment and thus indicates possible state estimation.

Tab. 11. Incidence matrix of the PINN models for the liquid-liquid separator. The matrix is identical for all three PINN models. If an unmeasured state appears in an equation, it is marked with a cross. Encircled crosses show the feasible assignment of states to equations. The matrix has full column rank.

<table><tr><td> $[f,g] \downarrow$   $[x^{u},y^{u}] \rightarrow$ </td><td> $\dot{V}_{c}$ </td><td> $\dot{V}_{s}$ </td></tr><tr><td>Eqn. (5a)</td><td></td><td></td></tr><tr><td>Eqn. (5b)</td><td>⊗</td><td></td></tr><tr><td>Eqn. (5c)</td><td>×</td><td>⊗</td></tr></table>

We compare the prediction error of the states of the liquid-liquid separator model based on test set data. As can be seen from Figure 9a, the prediction accuracy of the DPZ height h<sub>DPZ</sub> increases slightly with the addition of the Sauter mean diameter at the inlet $d _ { 3 2 }$ as a NN input. However, a more drastic increase can be noted if the coalescence parameter $r _ { \mathrm { v } }$ is added as NN input. We see a similar trend with the estimation of the coalescence rate $\dot { V } _ { c }$ in Figure 9b, although the accuracy increase is more apparent in the high-data regime. As explained in the Supplementary Materials, the coalescence parameter $r _ { \mathrm { v } }$ plays a more direct role in the determination of the coalescence rate $\dot { V } _ { c } ,$ which in turn has a high impact on h<sub>DPZ</sub> (Equation (5b)). The Sauter mean diameter at the inlet $d _ { 3 2 }$ has only an indirect role since the sub-model for coalescence and sedimentation in the full-order mechanistic model (see Section SM4 of the Supporting Materials) divides the separator into segments through the axial length. Thus, the Sauter mean diameter at each segment $d _ { 3 2 , }$ determines the coalescence rate rather than the value at the inlet. Moreover, since the PINN models are not trained with the data of the coalescence rate $\dot { V } _ { c } ,$ , and the sub-model for the coalescence rate is not provided as physics knowledge, the estimation accuracy of the coalescence rate $\dot { V } _ { c }$ highly depends on the prediction accuracy of h<sub>DPZ</sub>.

In Figure 10a, we observe that the prediction accuracy of the water height $h _ { \mathrm { a q } }$ does not change notably with the addition of $d _ { 3 2 }$ and $r _ { \mathrm { v } }$ as further NN inputs. We note a similar trend for the prediction of the sedimentation rate $\dot { V _ { s } }$ in Figure 10b. These findings are not unexpected since the added physical properties play a negligible role in the sub-model for sedimentation rate $\dot { V _ { s } }$ in the full-order mechanistic model and consequently for the water height $h _ { \mathrm { a q } }$

The vanilla ANN performs considerably worse: For the DPZ height h<sub>DPZ</sub>, the mean error of 25 models is 2.06 % (MAPE) for the low-data regime and 1.78 % (MAPE) for the high-data regime. For the water phase height $h _ { \mathrm { a q } } ,$ the mean error of 25 models is 1.33 % (MAPE) for the low-data regime and 1.13 % (MAPE) for the high-data regime. Note that the vanilla ANN cannot estimate the coalescence and sedimentation rates, $\dot { V } _ { c }$ and $\dot { V } _ { s } ,$ , as no measurement data were available for training.

Overall, all PINN models show great generalization capabilities in the low-data regime for the prediction of $h _ { \mathrm { { D P Z } } }$ which is a significant performance indicator for separation eficiency, with a maximum value of 0.46 % for the mean absolute percentage error (MAPE). Moreover, the PINN models can estimate the unmeasured states, for which constitutive equations were assumed to be unknown, with a maximum error value of 8.28 % for the coalescence rate $\dot { V } _ { c } ,$ and with a maximum error value of 1.62 % for the sedimentation rate $\dot { V _ { s } }$ . As a final remark, we observe that adding $d _ { 3 2 }$ and $r _ { \mathrm { v } }$ as inputs to the PINN significantly improves the prediction of $h _ { \mathrm { D P Z } }$ and the estimation of $\dot { V } _ { c }$

![](images/552374c4af1a389622f3007d01bdd6ad1d0a4f4533a0ee1697d3f6726d4e2aff.jpg)  
(a) Dense-packed zone height h<sub>DPZ</sub>.

![](images/a25dccf4ebcd118faa38857a37d8d0ea160bc74e492d4da9eb1de83460b559b1.jpg)  
(b) Coalescence rate $\dot { V } _ { c } .$  
Fig. 9. Test set error for the DPZ height $h _ { \mathrm { D P Z } }$ and the coalescence rate $\dot { V } _ { c }$ for all PINN models and data regimes. The error metric is the mean absolute percentage error (MAPE).

![](images/2f3a9813f7ea47efa8a0cddac0f5091f70dcc56d638a173000cad1ba99723788.jpg)  
(a) Water height $h _ { \mathrm { a q } }$

![](images/d7007ce2b7fe6bac2e89cd649225b841a4aa5cc1dc3d617bcbf9f72de89c78dd.jpg)  
(b) Sedimentation rate $\dot { V } _ { s }$  
Fig. 10. Test set error for the water height $h _ { \mathrm { a q } }$ and the sedimentation rate $\dot { V _ { s } }$ for all PINN models and data regimes. The error metric is the mean absolute percentage error (MAPE).

## 5. Conclusion and Outlook

This paper investigates the PINN-based dynamic modeling of chemical engineering processes that are characterized by limited physical knowledge and limited data availability. Recognizing that certain process states, e.g., reaction rates or coalescence rates, often lack descriptive constitutive equations and cannot be measured, we set out to see if PINNs can infer such unmeasured states by leveraging known physical equations and data on measured states. To this end, we conducted numerical studies using two fullyknown mechanistic process models and mimicking real-world modeling situations that are characterized by limited physical knowledge and data availability. Specifically, we assumed that certain equations would be unknown and thus unavailable for PINN development and that only a subset of process states would be measurable.

In both the Van de Vusse continuously stirred tank reactor (CSTR) example and the liquid-liquid separator example, we found that PINN models vastly outperform vanilla NNs of equal size, show superior generalization with respect to diferent initial states and control actions as well as superior extrapolation capabilities in regions of the state space without training data. Importantly, we observed that PINN models indeed may be capable of estimating unmeasured states, even if the corresponding constitutive relations are unknown. We provided a heuristic for when the estimation of such unmeasured states migh be successful. Although representing neither a necessary nor a suficient condition for state estimation, the heuristic is easy to use and can be applied even before data collection is initiated.

Future work should concern the investigation of implicit DAE models in PINNs and whether the heuristic can be improved based on theory for observability of nonlinear dynamic systems, see, e.g.,

Lee and Markus (1967); Kou et al. (1973). The feed-forward PINNs used in our work could also be compared to physics-informed recurrent neural networks that rely on time discretization, see, e.g., Zheng et al. (2023), or transformer-based PINN architectures, see, e.g., Zhao et al. (2023). Furthermore, PINN modeling and control of actual plant operations should be considered.

We conclude that PINN models with partial physical knowledge constitute a promising alternative to hybrid mechanistic/data-driven models in chemical engineering applications and warrant further investigation by the PSE community due to their potential to estimate states for which neither constitutive equations nor training data are available. Such further investigation should include performance comparisons between PINNs and hybrid models on identical tasks. For instance, for the estimation of immeasurable states y that lack constitutive equations, a DNN predicting measurable diferential state x followed by a mechanistic model that uses (i) known balance equations, (ii) the predictions of x, and (iii) estimates of x˙ obtained through automatic diferentiation of the DNN to compute immeasurable algebraic states y would constitute a sequential hybrid model that could be compared to a PINN. Similarly, hybrid models recently proposed by Pahari et al. (2024) and Sitapure and Sang-Il Kwon (2023) use DNNs and time-series transformers, respectively, to estimate (spatio-)temporally varying quantitie as inputs to mechanistic sub-models.

## Declaration of Competing Interest

We have no conflict of interest.

## Acknowledgements

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 466656378 – within the Priority Programme “SPP 2331:Machine Learning in Chemical Engineering”. This work was performed as part of the Helmholtz School for Data Science in Life, Earth and Energy (HDS-LEE). We acknowledge financial support by the Helmholtz Association of German Research Centers through program-oriented funding. We would like to give special thanks to Lukas Polte, Fabian Mausbeck and Lukas Thiel (Aachener Verfahrenstechnik, Fluid Process Engineering, RWTH Aachen University) for their valuable suggestions and dedicated help on the separator model. We would like to give special thanks to Adel Mhamdi (Aachener Verfahrenstechnik, Process Systems Engineering, RWTH Aachen University) for providing valuable insight into the concepts of state estimation and observability.

## Author contributions

• MV developed the PINN-based dynamic models, implemented the PINN model for the Van de Vusse CSTR example, analyzed the results, and wrote the draft of all sections except the system description in Section 4 and the separator related parts of the Supplementary Materials.

• MV implemented the PINN model for the liquid-liquid separator example in close collaboration with SZ.

• SZ implemented and extended the mechanistic model of the liquid-liquid separator from the literature (Backi et al., 2018, 2019), wrote the system description in Section 4 and the separator-related parts of the Supplementary Materials.

• SZ and MV jointly analyzed the PINN results for the separator example and wrote Section 4.2.

• SR implemented the mechanistic model of the Van de Vusse CSTR, developed a preliminary PINN model for the Van de Vusse CSTR, and investigated scaling and dynamic weighting of PINNs in close collaboration with MV.

• MD conceptualized and supervised the work with the exception of the derivation of the mechanistic separator model, and provided help and guidance on the methodology.

• AM provided conceptual input on the theory, methods, and case studies and provided further supervision.

• AJ provided conceptual input on the separator model and provided further supervision.

• All authors have reviewed and edited the manuscript.

## CRediT authorship contribution statement

Mehmet Velioglu: Conceptualization, Methodology, Software, Investigation, Writing - original draft, review & editing, Visualization, Supervision. Song Zhai: Conceptualization, Methodology, Software, Investigation, Writing - original draft, review & editing, Visualization. Sophia Rupprecht: Methodology, Software, Investigation, Writing - review & editing. Andreas Jupke: Conceptualization, Writing - review & editing, Supervision, Funding acquisition. Alexander Mitsos: Conceptualization, Methodology, Writing - review & editing, Supervision. Manuel Dahmen: Conceptualization, Methodology, Writing - review & editing, Supervision, Funding acquisition.

## Bibliography

Alhajeri, M.S., Abdullah, F., Wu, Z., Christofides, P.D., 2022. Physics-informed machine learning mod eling for predictive control using noisy data. Chemical Engineering Research and Design 186, 34–49. doi:https://doi.org/10.1016/j.cherd.2022.07.035.

Antonelo, E.A., Camponogara, E., Seman, L.O., de Souza, E.R., Jordanou, J.P., Hübner, J.F., 2021. Physics-informed neural nets-based control. arXiv preprint arXiv:2104.02556.

Arnold, F., King, R., 2021. State–space modeling for control based on physics-informed neural networks. Engineering Applications of Artificial Intelligence 101. doi:10.1016/j.engappai.2021.104195.

Asprion, N., Böttcher, R., Pack, R., Stavrou, M.E., Höller, J., Schwientek, J., Bortz, M., 2019. Gray Box Modeling for the Optimization of Chemical Processes. Chemie-Ingenieur-Technik 91, 305–313. doi:10.1002/CITE.201800086.

Backi, C.J., Emebu, S., Skogestad, S., Grimes, B.A., 2019. A simple modeling approach to control emulsion layers in gravity separators, in: 29th European Symposium on Computer Aided Process Engineering. Elsevier. volume 46 of Computer Aided Chemical Engineering, pp. 1159–1164. doi:10. 1016/B978-0-12-818634-3.50194-6.

Backi, C.J., Grimes, B.A., Skogestad, S., 2018. A Control- and Estimation-Oriented Gravity Separator Model for Oil and Gas Applications Based upon First-Principles. Industrial and Engineering Chemistry Research 57, 7201–7217. doi:10.1021/acs.iecr.7b04297.

Barfoot, T.D., 2017. State Estimation for Robotics. Cambridge University Press.

Bradley, W., Kim, J., Kilwein, Z., Blakely, L., Eydenberg, M., Jalvin, J., Laird, C., Boukouvala, F., 2022. Perspectives on the integration between first-principles and data-driven modeling. Computers & Chemical Engineering 166, 107898. doi:https://doi.org/10.1016/j.compchemeng.2022.107898.

Brenan, K., Campbell, S., Petzold, L., 1996. Numerical Solution of Initial-Value Problems in Diferential-Algebraic Equations. volume 14. SIAM.

Chen, H., Kremling, H., Allgöwer, F., 1995. Nonlinear Predictive Control of a Benchmark CSTR. Proceedings of the 3rd European Control Conference, Rome-Italy. , 3247–3252.

Choi, S., Jung, I., Kim, H., Na, J., Lee, J.M., 2022. Physics-informed deep learning for data-driven solutions of computational fluid dynamics. Korean Journal of Chemical Engineering 39, 515–528. doi:https://doi.org/10.1007/s11814-021-0979-x.

Dormand, J., Prince, P., 1980. A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics 6, 19–26. doi:https://doi.org/10.1016/0771-050X(80)90013-3.

Duf, I., Gear, C., 1986. Computing the structural index. SIAM Journal on Algebraic Discrete Methods 7, 594–603.

Gani, R., Cameron, I.T., 1992. Modelling for dynamic simulation of chemical processes: the index problem. Chemical Engineering Science 47, 1311–1315. doi:https://doi.org/10.1016/0009-2509(92) 80252-8.

Gelb, A., et al., 1974. Applied Optimal Estimation. MIT press.

Goodfellow, I., Bengio, Y., Courville, A., 2016. Deep Learning. MIT Press.

Henschke, M., 1995. Dimensionierung liegender Flüssig-flüssig-Abscheider anhand diskontinuierlicher Absetzversuche. Ph.D. thesis. RWTH Aachen University.

Iman, R.L., Helton, J.C., Campbell, J.E., 1981. An Approach to Sensitivity Analysis of Computer Models: Part I—Introduction, Input Variable Selection and Preliminary Variable Assessment. Journal of Quality Technology 13, 174–183. doi:10.1080/00224065.1981.11978748.

Ji, W., Qiu, W., Shi, Z., Pan, S., Deng, S., 2021. Stif-pinn: Physics-informed neural network for stif chemical kinetics. The Journal of Physical Chemistry A 125, 8098–8106. doi:10.1021/acs.jpca. 1c05102.

Jin, X., Cai, S., Li, H., Karniadakis, G.E., 2021. NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations. Journal of Computational Physics 426, 109951. doi:10.1016/J.JCP.2020.109951.

Kahrs, O., Marquardt, W., 2007. The validity domain of hybrid models and its application in process optimization. Chemical Engineering and Processing: Process Intensification 46, 1054–1066. doi:10. 1016/J.CEP.2007.02.031.

Kahrs, O., Marquardt, W., 2008. Incremental identification of hybrid process models. Computers & Chemical Engineering 32, 694–705. doi:10.1016/J.COMPCHEMENG.2007.02.014.

Kalman, R., 1960a. On the general theory of control systems. IFAC Proceedings Volumes 1, 491–502.

Kalman, R.E., 1960b. A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering 82, 35–45. doi:10.1115/1.3662552.

Kampwerth, J., Weber, B., Rußkamp, J., Kaminski, S., Jupke, A., 2020. Towards a holistic solvent screening: On the importance of fluid dynamics in a rate-based extraction model. Chemical Engineering Science 227. doi:10.1016/j.ces.2020.115905.

Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., Yang, L., 2021. Physics-informed machine learning. Nature Reviews Physics 3, 422–440. doi:10.1038/s42254-021-00314-5.

Kingma, D.P., Ba, J., 2017. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Kou, S.R., Elliott, D.L., Tarn, T.J., 1973. Observability of nonlinear systems. Information and Control 22, 89–99.

Lagaris, I.E., Likas, A., Fotiadis, D.I., 1998. Artificial neural networks for solving ordinary and partial diferential equations. IEEE Transactions on Neural Networks 9, 987–1000. doi:10.1109/72.712178.

Lee, E., Markus, L., 1967. Foundations of Optimal Control Theory. SIAM series in applied mathematics, Wiley.

Liu, D.C., Nocedal, J., 1989. On the limited memory BFGS method for large scale optimization. Mathematical Programming 45, 503–528. doi:10.1007/BF01589116/METRICS.

Maddu, S., Sturm, D., Müller, C.L., Sbalzarini, I.F., 2022. Inverse Dirichlet weighting enables reliable training of physics informed neural networks. Machine Learning: Science and Technology 3, 015026. doi:10.1088/2632-2153/ac3712.

Markidis, S., 2021. The Old and the New: Can Physics-Informed Deep-Learning Replace Traditional Linear Solvers? Frontiers in Big Data 4. doi:10.3389/fdata.2021.669097.

Marquardt, W., 1996. Trends in computer-aided process modeling. Computers & Chemical Engineering 20, 591–609. doi:https://doi.org/10.1016/0098-1354(95)00195-6. fifth International Symposium on Process Systems Engineering.

Mersmann, A., 1980. Zum Flutpunkt in flüssig/flüssig–Gegenstromkolonnen. Chemie Ingenieur Technik 52, 933–942. doi:10.1002/cite.330521203.

Nabian, M.A., Meidani, H., 2019. Physics-Driven Regularization of Deep Neural Networks for Enhanced Engineering Design and Analysis. Journal of Computing and Information Science in Engineering 20. doi:10.1115/1.4044507.

Nascimento, R.G., Fricke, K., Viana, F.A., 2020. A tutorial on solving ordinary diferential equations using Python and hybrid physics-informed neural network. Engineering Applications of Artificial Intelligence 96. doi:10.1016/j.engappai.2020.103996.

Pahari, S., Shah, P., Sang-Il Kwon, J., 2024. Unveiling latent chemical mechanisms: Hybrid modeling for estimating spatiotemporally varying parameters in moving boundary problems. Industrial & Engineering Chemistry Research 63, 1501–1514. doi:10.1021/acs.iecr.3c03531.

Psichogios, D.C., Ungar, L.H., 1992. A hybrid neural network-first principles approach to process mod eling. AIChE Journal 38, 1499–1511. doi:10.1002/aic.690381003.

Queiroz, L., Santos, F., Oliveira, J., Souza, M., 2021. Physics-informed deep learning to predict flow fields in cyclone separators. Digital Chemical Engineering 1, 100002. doi:https://doi.org/10.1016/ j.dche.2021.100002.

Raissi, M., Perdikaris, P., Karniadakis, G.E., 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics 378, 686–707. doi:10.1016/j.jcp.2018.10.045.

Raissi, M., Yazdani, A., Karniadakis, G.E., 2020. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. Science 367, 1026–1030. doi:10.1126/science.aaw4741.

Rofel, B., Betlem, B., 2006. Process Dynamics and Control: Modeling for Control and Prediction. Process Dynamics and Control: Modeling for Control and Prediction, Wiley.

Sansana, J., Joswiak, M.N., Castillo, I., Wang, Z., Rendall, R., Chiang, L.H., Reis, M.S., 2021. Recent trends on hybrid modeling for industry 4.0. Computers & Chemical Engineering 151, 107365. doi:https://doi.org/10.1016/j.compchemeng.2021.107365.

Schweidtmann, A.M., Esche, E., Fischer, A., Kloft, M., Repke, J.U., Sager, S., Mitsos, A., 2021. Machine learning in chemical engineering: A perspective. Chemie Ingenieur Technik 93, 2029–2039. doi:https: //doi.org/10.1002/cite.202100083.

Schweidtmann, A.M., Zhang, D., von Stosch, M., 2024. A review and perspective on hybrid modeling methodologies. Digital Chemical Engineering 10, 100136. doi:https://doi.org/10.1016/j.dche. 2023.100136.

Shah, P., Sherif, M.Z., Bangi, M.S.F., Kravaris, C., Kwon, J.S.I., Botre, C., Hirota, J., 2022. Deep neural network-based hybrid modeling and experimental validation for an industry-scale fermentation

process: Identification of time-varying dependencies among parameters. Chemical Engineering Journal 441, 135643. doi:https://doi.org/10.1016/j.cej.2022.135643.

Sharma, N., Liu, Y.A., 2022. A hybrid science-guided machine learning approach for modeling chemical processes: A review. AIChE Journal 68. doi:10.1002/aic.17609.

Sitapure, N., Sang-Il Kwon, J., 2023. Introducing hybrid modeling with time-series-transformers: A comparative study of series and parallel approach in batch crystallization. Industrial & Engineering Chemistry Research 62, 21278–21291. doi:10.1021/acs.iecr.3c02624.

Stokes, G.G., 2009. On the Efect of the Internal Friction of Fluids on the Motion of Pendulums. Cambridge University Press. Cambridge Library Collection - Mathematics, p. 1–10.

von Stosch, M., Oliveira, R., Peres, J., Feyo de Azevedo, S., 2014. Hybrid semi-parametric modeling in process systems engineering: Past, present and future. Computers and Chemical Engineering 60, 86–101. doi:10.1016/j.compchemeng.2013.08.008.

Su, H.T., Bhat, N., Minderman, P., McAvoy, T., 1992. Integrating Neural Networks with First Principles Models for Dynamic Modeling. IFAC Proceedings Volumes 25, 327–332. doi:10.1016/S1474-6670(17) 51013-7.

Tan, C., Cai, Y., Wang, H., Sun, X., Chen, L., 2023. Vehicle state estimation combining physics-informed neural network and unscented kalman filtering on manifolds. Sensors 23, 6665. doi:10.3390/s23156665.

Thompson, M.L., Kramer, M.A., 1994. Modeling chemical processes using prior knowledge and neural networks. AIChE Journal 40, 1328–1340. doi:10.1002/aic.690400806.

Unger, J., Kröner, A., Marquardt, W., 1995. Structural analysis of diferential-algebraic equation systems—theory and applications. Computers & Chemical Engineering 19, 867–882. doi:https: //doi.org/10.1016/0098-1354(94)00094-5.

Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S.J., Brett, M., Wilson, J., Millman, K.J., Mayorov, N., Nelson, A.R.J., Jones, E., Kern, R., Larson, E., Carey, C.J., Polat, I., Feng, Y., Moore, E.W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E.A., Harris, C.R., Archibald, A.M., Ribeiro, A.H., Pedregosa, F., van Mulbregt, P., Vijaykumar, A., Bardelli, A.P., Rothberg, A., Hilboll, A., Kloeckner, A., Scopatz, A., Lee, A., Rokem, A., Woods, C.N., Fulton, C., Masson, C., Häggström, C., Fitzgerald, C., Nicholson, D.A., Hagen, D.R., Pasechnik, D.V., Olivetti, E.,

Martin, E., Wieser, E., Silva, F., Lenders, F., Wilhelm, F., Young, G., Price, G.A., Ingold, G.L., Allen, G.E., Lee, G.R., Audren, H., Probst, I., Dietrich, J.P., Silterra, J., Webber, J.T., Slavič, J., Nothman, J., Buchner, J., Kulick, J., Schönberger, J.L., de Miranda Cardoso, J.V., Reimer, J., Harrington, J., Rodríguez, J.L.C., Nunez-Iglesias, J., Kuczynski, J., Tritz, K., Thoma, M., Newville, M., Kümmerer, M., Bolingbroke, M., Tartre, M., Pak, M., Smith, N.J., Nowaczyk, N., Shebanov, N., Pavlyk, O., Brodtkorb, P.A., Lee, P., McGibbon, R.T., Feldbauer, R., Lewis, S., Tygier, S., Sievert, S., Vigna, S., Peterson, S., More, S., Pudlik, T., Oshima, T., Pingel, T.J., Robitaille, T.P., Spura, T., Jones, T.R., Cera, T., Leslie, T., Zito, T., Krauss, T., Upadhyay, U., Halchenko, Y.O., Vázquez-Baeza, Y., Contributors, S.., 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17, 261–272. doi:10.1038/s41592-019-0686-2.

van de Vusse, J.G., 1964. Plug-flow type reactor versus tank reactor. Chemical Engineering Science 19, 994–996. doi:10.1016/0009-2509(64)85109-5.

Wu, G., Yion, W.T.G., Dang, K.L.N.Q., Wu, Z., 2023. Physics-informed machine learning for mpc: Application to a batch crystallization process. Chemical Engineering Research and Design 192, 556– 569. doi:https://doi.org/10.1016/j.cherd.2023.02.048.

Yang, S., Navarathna, P., Ghosh, S., Bequette, B.W., 2020. Hybrid modeling in the era of smart manufacturing. Computers & Chemical Engineering 140, 106874. doi:https://doi.org/10.1016/ j.compchemeng.2020.106874.

Ye, S., Hohl, L., Kraume, M., 2023. Impact of feeding conditions on continuous liquid-liquid gravity separation, part I: Inlet and outlet drop size, dense-packed zone and separation eficiency. Chemical Engineering Science 282. doi:10.1016/j.ces.2023.119237.

Zendehboudi, S., Rezaei, N., Lohi, A., 2018. Applications of hybrid models in chemical, petroleum, and energy systems: A systematic review. Applied Energy 228, 2539–2566. doi:10.1016/J.APENERGY. 2018.06.051.

Zhao, Z., Ding, X., Prakash, B.A., 2023. Pinnsformer: A transformer-based framework for physicsinformed neural networks. arXiv preprint arXiv:2307.11833.

Zheng, Y., Hu, C., Wang, X., Wu, Z., 2023. Physics-informed recurrent neural network modeling for predictive control of nonlinear processes. Journal of Process Control 128, 103005. doi:https://doi. org/10.1016/j.jprocont.2023.103005.

Zheng, Y., Wu, Z., 2023. Physics-informed online machine learning and predictive control of nonlinear processes with parameter uncertainty. Industrial & Engineering Chemistry Research 62, 2804–2818. doi:10.1021/acs.iecr.2c03691.

# Supplementary Materials - Physics-Informed Neural Networks for Dynamic Process Operations with Limited Physical Knowledge and Data

Mehmet Velioglu<sup>a,b</sup>, Song Zhai<sup>e</sup>, Sophia Rupprecht<sup>a,f</sup> , Alexander Mitsos<sup>c,a,d</sup>, Andreas Jupke<sup>e</sup>, Manue

Dahmen<sup>a,</sup>∗

<sup>a</sup> Institute of Climate and Energy Systems, Energy Systems Engineering (ICE-1), Forschungszentrum Jülich GmbH, Jülich 52425, Germany

<sup>b</sup> RWTH Aachen University Aachen 52062, Germany

<sup>c</sup> JARA-ENERGY, Jülich 52425, Germany

<sup>d</sup> RWTH Aachen University, Process Systems Engineering (AVT.SVT), Aachen 52074, Germany

<sup>e</sup> RWTH Aachen University, Fluid Process Engineering (AVT.FVT), Aachen 52074, Germany

<sup>f</sup> Delft University of Technology, 2629 HZ, Delft, The Netherlands

SM1: Figures of the PINN models used in the numerical examples

The network schematics of the models for the Van de Vusse CSTR are shown in Figure 1.

The schematic of the PINN models for the liquid-liquid separator is shown in Figure 2.

![](images/af346c76d00b0aa6d7e7db30222f319b42b56c279f0805b75251eb1d02771109.jpg)  
(a) Network schematic of the vanilla ANN

![](images/1aa53308229d26e45ef88a90c0359b175a5d5b2b78a02515e12125235c1e2534.jpg)  
(b) Network schematic of PINN-A

![](images/dc01be8f080fb096790d9d96fa875f27814199f4c0fe6df4c6a8bf712505177d.jpg)  
(c) Network schematic of PINN-B

![](images/d17ea54a7dc94581b13472a0a0d0e47e5fb63c35f1dbc85003efec417f47685b.jpg)  
(d) Network schematic of PINN-C  
Fig. 1. Network schematics of models used in the Van de Vusse CSTR example. Note that the figure does not show the actual depth and width of the hidden layers.

![](images/949c0f9990691d1cb4756d5660808d34ef0d209f26b4639cfb106f0ca802ed06.jpg)  
Fig. 2. General schematic of the PINN models for the liquid-liquid separator. $\dot { V } _ { \mathrm { a q , o u t } }$ is shown as $\dot { V } _ { \mathbf { a q } }$ and $\dot { V } _ { \mathrm { o r g , o u t } }$ is shown as $\dot { V } _ { \mathrm { o r g } }$ . Model $" \mathrm { P I N N } "$ has no physical property as input; model $" \mathrm { P I N N - d _ { 3 2 } } "$ has the Sauter mean diameter $d _ { 3 2 }$ as a NN input; model $" \mathrm { P I N N - d _ { 3 2 } - r _ { v } } " $ has both the Sauter mean diameter $d _ { 3 2 }$ and the coalescence parameter $r _ { \mathrm { v } }$ as NN inputs. Note that the figure does not show the actual depth and width of the hidden layers.

## SM2: Dimensionless variables and equations of the Van de Vusse Reactor

The equations for the Van de Vusse CSTR are given in Equations (3) in the main text. To make the problem dimensionless, we introduce the following dimensionless quantities, inspired by the work of Gamboa-Torres and Flores-Tlacuahuac (2000):

$$
\begin{array}{l} t ^ {*} = \frac {t}{\tau}, \qquad c _ {A} ^ {*} = \frac {c _ {A}}{c _ {A , i n}}, \quad c _ {B} ^ {*} = \frac {c _ {B}}{c _ {A , i n}}, \quad T ^ {*} = \frac {T}{T _ {i n}}, \quad T _ {K} ^ {*} = \frac {T _ {K}}{T _ {i n}}, \\ \left(\frac {\dot {V}}{V _ {R}}\right) ^ {*} = \frac {\dot {V}}{V _ {R}} \frac {1}{q _ {f}}, \quad \dot {Q} _ {K} ^ {*} = \frac {\dot {Q} _ {K}}{\dot {Q} _ {K , f}}, \quad k _ {1} ^ {*} = \frac {k _ {1}}{k _ {f}}, \quad k _ {2} ^ {*} = \frac {k _ {2}}{k _ {f}}, \quad k _ {3} ^ {*} = c _ {A, i n} \frac {k _ {3}}{k _ {f}} \end{array}
$$

The values of the normalization parameters $\tau , q _ { f } , \dot { Q } _ { K , f }$ and $k _ { f }$ are given in Table 1. For better readability, we introduce the following symbols:

$$
P = - \frac {\Delta H _ {A B} k _ {f} c _ {A , i n}}{\rho C _ {p} T _ {i n}}, \qquad M = \frac {k _ {w} A _ {R}}{\rho C _ {p} V _ {R}}, \qquad L = \frac {k _ {w} A _ {R}}{m _ {K} C _ {p K}}, \qquad R = \frac {\dot {Q} _ {K , f}}{m _ {K} C _ {p K} T _ {i n}}
$$

Then, the dimensionless versions of the Equations (3) in the main text become:

$$
\begin{array}{r l} & {\frac {1}{\tau} \left(\frac {\mathrm{d} c _ {A} ^ {*}}{\mathrm{d} t ^ {*}}\right) = (q _ {f}) \left(\frac {\dot {V}}{V _ {R}}\right) ^ {*} (1 - c _ {A} ^ {*}) - k _ {f} k _ {1} ^ {*} (T) c _ {A} ^ {*} - k _ {f} k _ {3} ^ {*} (T) c _ {A} ^ {* 2},} \\ & {\frac {1}{\tau} \left(\frac {\mathrm{d} c _ {B} ^ {*}}{\mathrm{d} t ^ {*}}\right) = - (q _ {f}) \left(\frac {\dot {V}}{V _ {R}}\right) ^ {*} c _ {B} ^ {*} + k _ {f} k _ {1} ^ {*} (T) c _ {A} ^ {*} - k _ {f} k _ {2} ^ {*} (T) c _ {B} ^ {*},} \\ & {\frac {1}{\tau} \left(\frac {\mathrm{d} T ^ {*}}{\mathrm{d} t ^ {*}}\right) = (q _ {f}) \left(\frac {\dot {V}}{V _ {R}}\right) ^ {*} (1 - T ^ {*}) + P \left[ k _ {1} ^ {*} (T) c _ {A} ^ {*} + k _ {2} ^ {*} (T) c _ {B} ^ {*} \frac {\Delta H _ {B C}}{\Delta H _ {A B}} + k _ {3} ^ {*} (T) c _ {A} ^ {* 2} \frac {\Delta H _ {A D}}{\Delta H _ {A B}} \right]} \\ & {\qquad + M (T _ {K} ^ {*} - T ^ {*}),} \\ & {\frac {1}{\tau} \left(\frac {\mathrm{d} T _ {K} ^ {*}}{\mathrm{d} t ^ {*}}\right) = L (T ^ {*} - T _ {K} ^ {*}) + \dot {Q} _ {K} ^ {*} R} \end{array}
$$

with

$$
k _ {i} ^ {*} (T ^ {*}) = \frac {k _ {i 0} \exp \left(\frac {E _ {i} / T _ {i n}}{T ^ {*}}\right)}{k _ {f}}, \qquad i = 1, 2, 3
$$

Tab. 1. Parameters for the dimensionless van de Vusse CSTR equations. The values are chosen to bound time and contro actions between 0 and 1

$$
\begin{array}{c c} \text {Symbol} & \text {Value} \\ \hline \tau & 6 0   \text {s} \\ q _ {f} & 2 8. 4   (1 / \text {h}) \\ \dot {Q} _ {K, f} & - 2 2 2 7   \text {kJ / h} \\ k _ {f} & 3 6   (1 / \text {h}) \end{array}
$$

## SM3: Dimensionless variables and equations of the liquid-liquid separator case study

The equations for the liquid-liquid separator are given in Equations (5) in the main text. To make the problem dimensionless, we introduce the following dimensionless quantities:

$$
t ^ {*} = \frac {t}{\tau}, h _ {L} ^ {*} = \frac {h _ {L}}{2 r}, h _ {D P Z} ^ {*} = \frac {h _ {D P Z}}{2 r}, h _ {\mathrm{aq}} ^ {*} = \frac {h _ {\mathrm{aq}}}{2 r}, \dot {V} _ {i} ^ {*} = \frac {\dot {V} _ {i}}{q _ {f}},
$$

The values of the normalization parameters above are given in Table 2. Then the dimensionless version becomes:

$$
\begin{array}{r l} & {\frac {1}{\tau} \left(\frac {\mathrm{d} h _ {L} ^ {*}}{\mathrm{d} t ^ {*}}\right) = \frac {q _ {f}}{2 r} \frac {\dot {V} _ {i n} ^ {*} - \dot {V} _ {a q , o u t} ^ {*} - \dot {V} _ {o r g , o u t} ^ {*}}{2 L \sqrt {2 r h _ {L} ^ {*} (2 r - 2 r h _ {L} ^ {*})}},} \\ & {\frac {1}{\tau} \left(\frac {\mathrm{d} h _ {D P Z} ^ {*}}{\mathrm{d} t ^ {*}}\right) = \frac {q _ {f}}{2 r} \frac {\dot {V} _ {i n} ^ {*} - \dot {V} _ {a q , o u t} ^ {*} - \dot {V} _ {c} ^ {*}}{2 L \sqrt {2 r h _ {D P Z} ^ {*} (2 r - 2 r h _ {D P Z} ^ {*})}}} \\ & {\frac {1}{\tau} \left(\frac {\mathrm{d} h _ {\mathrm{aq}} ^ {*}}{\mathrm{d} t ^ {*}}\right) = \frac {q _ {f}}{2 r} \frac {\dot {V} _ {i n} ^ {*} - \dot {V} _ {a q , o u t} ^ {*} - \dot {V} _ {s} ^ {*} \frac {1}{\bar {\epsilon} _ {p}} + \dot {V} _ {c} ^ {*} \frac {1 - \bar {\epsilon} _ {p}}{\bar {\epsilon} _ {p}}}{2 L \sqrt {2 r h _ {\mathrm{aq}} ^ {*} (2 r - 2 r h _ {\mathrm{aq}} ^ {*})}}} \end{array}
$$

Tab. 2. Parameters for the dimensionless liquid-liquid separator equations. The values are chosen to bound time, contro actions, and states between 0 and 1.

<table><tr><td>Symbol</td><td>Value</td></tr><tr><td> $\tau$ </td><td>20 s</td></tr><tr><td> $q_{f}$ </td><td> $1 \times 10^{-3} \text{ m}^{3}/\text{s}$ </td></tr></table>

## SM4: Lumped model of a liquid-liquid separator

The sedimentation and coalescence rate, $\dot { V _ { s } }$ and $\dot { V } _ { c } ,$ , respectively, are determined by lumping a detailed 1D model with discretized drop population balance shown in Fig. 3. The dynamic liquid-liquid separator model shown below is based on the work of Backi et al. (2018, 2019). We included extensions for swarm sedimentation in the aqueous phase, convection terms for the drop size distribution (DSD) in the dense-packed zone (DPZ) analogously to Backi et al. (2018), and a state-of-the-art coalescence model (Henschke, 1995). The chosen swarm model (Mersmann, 1980) was also used to model liquid-liquid columns (Kampwerth et al., 2020) and takes the form of Stokes’ law (Stokes, 2009) for diminishing hold ups. Stokes’ law was experimentally confirmed to model the outlet hold-up of liquid-liquid separator accurately (Ye et al., 2023b).

Three liquid phases are modeled, aqueous, DPZ, and organic. The organic phase is dispersed, and the continuous organic phase is free from aqueous droplets. The lumped model assumes given constant heights for each phase, an instantaneous development of sedimenting droplets, and a DPZ starting from the entrance with given boundary conditions from the 0D settler model (cf. Section 4 in the main text). In Addition, no coalescence in the aqueous phase is assumed.

In the following, expressions for the water volume flow rate $\dot { V } _ { w }$ , sedimentation rate $\dot { V _ { s } }$ , and coalescence rate $\dot { V } _ { c }$ are derived. The water volume flow $\dot { V } _ { w }$ from the aqueous phase can be written as a function of the coalescence and sedimentation rates resulting from a volume balance for the DPZ by assuming a

![](images/b9c33f736c34ec3cf7e4cc24f6f78c081fa9fe3198a6c0de618bc92f3cfe8f72.jpg)  
Fig. 3. Detailed 1D model of the heavy phase and DPZ at an axial element i and for a droplet class j. $\dot { V _ { i } }$ is the convective flow of the aqueous phase, $\epsilon _ { i }$ the hold-up in the aqueous phase, $n _ { i , j }$ the number of droplets. ∆x is the discretization length in axial direction and $\Delta y _ { \mathrm { p o s } , i j }$ is the sedimented distance of droplets. $\dot { V } _ { s , i j } , \dot { V } _ { w , i }$ and $\dot { V } _ { c , i }$ are the volume flow of sedimenting organic droplets, trapped water, and coalescing organic, respectively $V _ { d p z , i }$ and $n _ { d p z , i }$ are the convective flows of the DPZ and number distribution in the DPZ.

constant hold-up $\bar { \epsilon } _ { p }$ in the DPZ:

$$
\frac {\mathrm{d} V _ {\mathrm{DPZ}}}{\mathrm{d} t} = \dot {V} _ {s} + \dot {V} _ {w} - \dot {V} _ {c}\tag{1}
$$

$$
\bar {\epsilon} _ {p} \frac {\mathrm{d} V _ {\mathrm{DPZ}}}{\mathrm{d} t} = \dot {V} _ {s} - \dot {V} _ {c}\tag{2}
$$

$$
\Rightarrow \dot {V} _ {w} = (\dot {V} _ {s} - \dot {V} _ {c}) \frac {1 - \bar {\epsilon} _ {p}}{\bar {\epsilon} _ {p}}\tag{3}
$$

The sedimentation rates $\dot { V _ { s } }$ result from droplets sedimenting with the swarm sedimentation velocity to the interface and moving in a plug flow in the horizontal direction. The droplets are assumed to move with the same velocity as the heavy phase in x-direction and are homogeneously distributed in the heavy phase at the entrance. The sedimentation rate is calculated as

$$
\begin{array}{r l} & {\dot {V} _ {s} = \sum_ {i} ^ {N _ {s}} \dot {V} _ {s, i},} \\ & {\dot {V} _ {s, i} = \dot {V} _ {i} \sum_ {j} ^ {N _ {d}} \frac {n _ {s , i , j}}{n _ {i}},} \end{array}
$$

where $\dot { V } _ { s , i }$ and $\dot { V _ { i } }$ is the volume flow by sedimentation and convective volume flow in segment i, $n _ { s , i , j }$ is the number of droplets in class $j$ and in axial segment i reaching the $\mathrm { D P Z , }$ and $n _ { i }$ is the total number of droplets in segment i. $N _ { s }$ and $N _ { d }$ are the number of axial discretization elements and drop diameter classes, respectively. The convective volume flow $\dot { V _ { i } }$ entering segment i is calculated starting from the entrance using the following equations:

$$
\begin{array}{l} \dot {V} _ {i + 1} = \dot {V} _ {i} - \dot {V} _ {s, i} - \dot {V} _ {w, i} \qquad \mathrm{for} \quad i = 0, \dots , N _ {s} \\ \dot {V} _ {i = 0} = \dot {V} _ {\mathrm{in}} \end{array} ,
$$

Similarly, the total number of drops $n _ { i }$ remaining in a segment i is determined as

$$
\begin{array}{c} n _ {i} = \sum_ {j} ^ {N _ {d}} n _ {i, j}, \\ n _ {i + 1, j} = n _ {i, j} - n _ {s, i, j} \qquad \text {for} \quad i = 0,..., N _ {s} \quad \text {and} \quad j = 1,..., N _ {d}, \\ n _ {i = 0, j} = n _ {0, j}, \end{array}
$$

where $n _ { 0 , j }$ results from the droplet number distribution at the inlet. The Sauter mean diameter at the inlet and number distribution are related by assuming a self-similar volume-based log-normal drop size distribution with a normalized standard deviation of $\sigma / d _ { 3 2 } = 0 . 3 2$ (Kraume et al., 2004; Ye et al., 2023a). The last missing part is the determination of sedimented droplets $n _ { s , i , j }$ , which is calculated as

$$
\begin{array}{l} {n _ {s, i, j} = n _ {i, j} \frac {\tau_ {x , i} v _ {s , j , i}}{h _ {\mathrm{aq}} - y _ {i , j}}} \\ {n _ {s, i, j} = n _ {i, j}} \end{array}
$$

$$
\begin{array}{l l} \text {if} & \tau_ {x, i} <   \tau_ {y, i, j} \quad , \\ \text {if} & \tau_ {x, i} \geq \tau_ {y, i, j} \quad , \end{array}
$$

where $\tau _ { x , i }$ and $\tau _ { y , i , j }$ are the residence time in x- and $\mathrm { y } \mathrm { . }$ -direction. $v _ { s , j , i }$ is the swarm sedimentation of droplet class $j$ in segment $i , \ h _ { \mathrm { a q } }$ is the height of the aqueous phase, and $y _ { i , j }$ is the vertical position of droplet class j in segment i. The residence times are determined as

$$
\tau_ {x, i} = \frac {V _ {\mathrm{aq}} / N _ {s}}{\dot {V} _ {i}},\tag{4}
$$

$$
\tau_ {y, i, j} = \frac {h _ {\mathrm{aq}} - y _ {i , j}}{v _ {s , i , j}},\tag{5}
$$

$$
v _ {s, i, j} = \frac {g d _ {j} ^ {2} \Delta \rho}{1 8 \eta_ {c}} (1 - \epsilon_ {i}) ^ {(n - 1)} \quad ,\tag{6}
$$

where $v _ { s , i , j }$ is the swarm sedimentation velocity calculated with the swarm exponent $( n = 2 )$ (Mersmann, 1980; Kampwerth et al., 2020), g the gravitational constant, $\Delta \rho$ the density diference between the aqueous and organic phase, $\eta _ { c }$ the viscosity of the continuous phase, and $\epsilon _ { i }$ the hold-up in the aqueous phase at segment i. For hold-ups approaching 0, Equation (6) takes the form of Stokes’ law (Stokes, 2009) that was experimentally confirmed to model the outlet hold-up accurately (Ye et al., 2023b). $h _ { \mathrm { a q } }$ is the height of the aqueous phase and a function of the volume of the aqueous phase, radius R, and length L of the separator. The geometric equations are given as follows:

$$
\begin{array}{c} {V _ {\mathrm{aq}} = A _ {x} (h _ {\mathrm{aq}}) L \quad ,} \\ {A _ {x} (h) = R ^ {2} \arccos (1 - h / R) - (R - h) \sqrt {2 R h - h ^ {2}}} \end{array}
$$

The hold-up and vertical position are calculated similarly to the convective volume flow starting from the entrance using the following equations:

$$
\begin{array}{r l r} \epsilon_ {i + 1} = \frac {\epsilon_ {i} \dot {V} _ {i} - \dot {V} _ {s , i}}{\dot {V} _ {s , i + 1}} & & \mathrm{for} i = 0, \dots , N _ {s}, \\ \epsilon_ {i = 0} = \epsilon_ {\mathrm{in}}, \\ y _ {i + 1, j} = y _ {i, j} + v _ {s, i, j} \min (\tau_ {x, i}, \tau_ {y, i, j}) & & \mathrm{for} i = 0, \dots , N _ {s}; j = 1, \dots , N _ {d}, \\ y _ {i = 0, j} = 0 & & \mathrm{for} j = 1, \dots , N _ {d} \end{array}
$$

Thus, the sedimentation rate can be calculated from a given height and the boundary conditions. The coalescence rate influences the sedimentation rate indirectly, as the water volume flow rate is a function of the coalescence rate (see Eq. (3)). Therefore, the coalescence rate influences the convective flow and the residence time for droplets to sediment.

The coalescence rate $\dot { V } _ { c }$ is a function of the Sauter mean diameter, height of the DPZ, and physical properties. The coalescence rate is adopted from (Henschke, 1995) and is calculated as

$$
\begin{array}{l} \dot {V} _ {c} = \sum_ {i} ^ {N _ {s}} \dot {V} _ {c, i} = \sum_ {i} ^ {N _ {s}} \frac {2 A _ {y} d _ {3 2 , \mathrm{DPZ} , i}}{3 \tau_ {d i , i}} \\ A _ {y} = 2 \Delta x \sqrt {2 R h _ {\mathrm{DPZ}} - h _ {\mathrm{DPZ}} ^ {2}} \end{array} ,
$$

where $A _ { y }$ is the area in the y-direction between the organic phase and DPZ and is a function of the height of the DPZ, h<sub>DPZ</sub>. d<sub>32,DPZ,i</sub> is the Sauter mean diameter in the DPZ at segment $i ,$ and $\tau _ { d i , i }$ is the coalescence time at segment i. The coalescence time depends on physical properties and the height

of the DPZ (Henschke, 1995), $\mathrm { i . e . }$ 2

$$
\begin{array}{r l r} \tau_ {d i, i} = \frac {(6 \pi) ^ {7 / 6} \eta_ {c} r _ {\mathrm{a}} ^ {7 / 3}}{4 \sigma^ {5 / 6} H _ {c} ^ {1 / 6} r _ {\mathrm{f,i}} r _ {\mathrm{v}}} & , \\ r _ {\mathrm{f,i}} = 0. 5 2 3 9 d _ {3 2, \mathrm{DPZ}, i} \sqrt {1 - \frac {4 . 7}{L a _ {\mathrm{mod} , i} + 4 . 7}} & , \\ r _ {\mathrm{a}} = 0. 5 d _ {3 2, \mathrm{DPZ}, i} \left(1 - \sqrt {1 - \frac {4 . 7}{L a _ {\mathrm{mod} , i} + 4 . 7}}\right) & , \\ L a _ {\mathrm{mod}, i} = \left(\frac {\Delta \rho g}{\sigma}\right) ^ {0. 6} (h _ {\mathrm{DPZ}} - h _ {\mathrm{aq}}) ^ {0. 2} d _ {3 2, \mathrm{DPZ}, i} & , \end{array}
$$

where σ is the interfacial tension, $H _ { c }$ is the Hamaker constant fixed to $1 0 \times 1 0 ^ { - 2 0 } \mathrm { N m } , r _ { \mathrm { f , i } }$ and $r _ { \mathrm { a } }$ are radii resulting from deformed droplets between the drop-interface. The drop deformation is characterized by the modified Laplace number $L a _ { \mathrm { m o d } }$ representing the ratio between hydrostatic pressure and interfacial tension. $r _ { \mathrm { v } }$ is the coalescence parameter specific to a liquid-liquid system describing the coalescence afinity and can be determined by batch settling experiment with the liquid-liquid system. The last missing variable is $d _ { 3 2 , \mathrm { D P Z } , i }$ that is a function of the number distribution of the drops $n _ { \mathrm { D P Z } , i , j }$ in the segment i, previous segment $i - 1$ , and the convective flow $\dot { V } _ { \mathrm { D P Z , i } }$ <sub>i</sub> in the DPZ. The number distribution of drops in the segment i is calculated from the sedimenting drops $n _ { s , i }$ as:

$$
n _ {\mathrm{DPZ}, i, j} = n _ {\mathrm{DPZ}, i - 1, j} + n _ {s, i, j}
$$

$$
\mathrm{if} \dot {V} _ {\mathrm{DPZ}, i} > 0,
$$

$$
n _ {\mathrm{DPZ}, i, j} = n _ {s, i, j}
$$

$$
\mathrm{if} \dot {V} _ {\mathrm{DPZ}, i} = 0,
$$

$$
\begin{array}{r l} & {\dot {V} _ {\mathrm{DPZ}, i + 1} = \max \left(\dot {V} _ {\mathrm{DPZ}, i} + \frac {\dot {V} _ {s , i} - \dot {V} _ {c , i}}{\bar {\epsilon} _ {p}}, 0\right) \quad ,} \\ & {d _ {3 2, \mathrm{DPZ}, i} = \frac {\sum_ {j} n _ {\mathrm{DPZ} , i , j} d _ {j} ^ {3}}{\sum_ {j} n _ {\mathrm{DPZ} , i , j} d _ {j} ^ {2}}} \end{array}
$$

The convective flow in the DPZ results from a volume balance and is ensured by the maximum function to be nonnegative. In case of a greater coalescence rate than the sum of convective flow and sedimentation rate, the Sauter mean diameter consists only of sedimenting droplets.

The mechanistic model is solved by 200 discretization elements for the separator length and 50 discretization elements for the drop size distribution. The number of discretization points was determined by sensitivity studies. Model parameter, physical properties and geometry data are listed in Table 3.

Tab. 3. Parameters for the liquid-liquid separator with n-butyl acetate dispersed in water.

<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td><td>Source</td></tr><tr><td>Radius of separator</td><td> $R$ </td><td>0.1 m</td><td>own lab</td></tr><tr><td>Length of separator</td><td> $L$ </td><td>1.8 m</td><td>own lab</td></tr><tr><td>Gravity constant</td><td> $g$ </td><td>9.81 m/s2</td><td>-</td></tr><tr><td>Density difference</td><td> $\Delta\rho$ </td><td>115 kg/m3</td><td>(Henschke, 1995)</td></tr><tr><td>Viscosity of organic phase</td><td> $\eta_{\text{org}}$ </td><td>0.775 mPa s</td><td>(Henschke, 1995)</td></tr><tr><td>Viscosity of aqueous phase</td><td> $\eta_{\text{aq}}$ </td><td>1.012 mPa s</td><td>(Henschke, 1995)</td></tr><tr><td>Interfacial tension</td><td> $\sigma$ </td><td>0.013 N/m</td><td>(Henschke, 1995)</td></tr><tr><td>Coalescence parameter</td><td> $r_v$ </td><td>0.0383</td><td>(Henschke, 1995)</td></tr><tr><td>Hamacker constant</td><td> $H_c$ </td><td> $1 \times 10^{-20}$  N m</td><td>(Henschke, 1995)</td></tr><tr><td>Hold-up in DPZ</td><td> $\bar{\epsilon}_p$ </td><td>0.9</td><td>(Henschke, 1995)</td></tr></table>

## SM5: Counter-example showing that the heuristic is not a necessary condition for the estimation of states

We present a counter-example showing that our heuristic (Section 2.3 in the main manuscript) is not a necessary condition for the estimation of unmeasured states. Consider the following ordinary diferential equation (ODE) system:

$$
\dot {x} _ {1} ^ {m} (t) = x _ {1} ^ {m} (t) x _ {2} ^ {u} (t) + x _ {3} ^ {u} (t),
$$

$$
\dot {x} _ {2} ^ {u} (t) = 0,\tag{7}
$$

(8)

$$
\dot {x} _ {3} ^ {u} (t) = 0\tag{9}
$$

Here, $x _ { 1 } ^ { m } , x _ { 2 } ^ { u }$ and x<sup>u</sup> denote the diferential states and t denotes time. Note that this ODE system is a complete system, as it has three equations and three variables. We thus consider the extreme case where we can integrate a full physical model into the PINN. In the following, we assume that $x _ { 1 } ^ { m }$ is measured while $x _ { 2 } ^ { u }$ and $x _ { 3 } ^ { u }$ are unmeasured.

Table 4 shows the incidence matrix for the counter-example given by Equations (7)-(9). Obviously, the incidence matrix does not have full-column rank, suggesting that state estimation would not work. Tab. 4. Incidence matrix of the counter-example represented by Equations (7)-(9). If an unmeasured state appears in an equation, it is marked with a cross. The matrix does not have full-column rank.

<table><tr><td>$ [\boldsymbol{f},\boldsymbol{g}] \downarrow $</td><td>$ [\mathbf{x}^{u},\mathbf{y}^{u}] \rightarrow $</td><td>$ x_{2}^{u} $</td><td>$ x_{3}^{u} $</td></tr><tr><td colspan="2">Eqn. (7)</td><td>×</td><td>×</td></tr><tr><td colspan="2">Eqn. (8)</td><td></td><td></td></tr><tr><td colspan="2">Eqn. (9)</td><td></td><td></td></tr></table>

We will show that the initial values $x _ { 2 , 0 } ^ { u } = x _ { 2 } ^ { u } ( t = 0 )$ and $x _ { 3 , 0 } ^ { u } = x _ { 3 } ^ { u } ( t = 0 )$ can be determined using measurement data $x _ { 1 } ^ { m } ( t _ { j } )$ that provide trajectory information about $x _ { 1 } ^ { m }$ . If the initial states $x _ { 2 , 0 } ^ { u }$ and $x _ { 3 , 0 } ^ { u }$ can be uniquely determined from such measurement data $x _ { 1 } ^ { m } ( t _ { j } )$ , the states $x _ { 2 } ^ { u }$ and $x _ { 3 } ^ { u }$ are said to be observable (Kalman, 1960; Lee and Markus, 1967).

The analytical solution to the ODE system reads:

$$
\begin{array}{l} x _ {1} ^ {m} (t) = \left(x _ {1, 0} ^ {m} + \frac {x _ {3 , 0} ^ {u}}{x _ {2 , 0} ^ {u}}\right) e ^ {(x _ {2, 0} ^ {u} t)} - \frac {x _ {3 , 0} ^ {u}}{x _ {2 , 0} ^ {u}}, \\ x _ {2} ^ {u} (t) = x _ {2, 0} ^ {u}, \\ x _ {3} ^ {u} (t) = x _ {3, 0} ^ {u} \end{array}
$$

Here, $x _ { 1 , 0 } ^ { m } = x _ { 1 } ^ { m } ( t = 0 )$ denotes the initial value for the diferential state $x _ { 1 } ^ { m }$ . Since we have measurement data $x _ { 1 } ^ { m } ( t _ { j } )$ , we assume that $x _ { 1 , 0 } ^ { m }$ is known. A key concept of observability analysis is the exploitation of derivative information, see, $\mathrm { e . g . }$ , Lee and Markus (1967); Kou et al. (1973). Accordingly, we consider the first and second order derivatives of the analytical solution for the measured state $x _ { 1 } ^ { m }$ , i.e.,

$$
\dot {x} _ {1} ^ {m} (t) = \left(x _ {1, 0} ^ {m} x _ {2, 0} ^ {u} + x _ {3, 0} ^ {u}\right) e ^ {(x _ {2, 0} ^ {u} t)},\tag{10}
$$

$$
\ddot {x} _ {1} ^ {m} (t) = \left(x _ {1, 0} ^ {m} x _ {2, 0} ^ {u} ^ {2} + x _ {3, 0} ^ {u} x _ {2, 0} ^ {u}\right) e ^ {(x _ {2, 0} ^ {u} t)}.\tag{11}
$$

Since the PINN is given many samples $x _ { 1 } ^ { m } ( t _ { j } )$ for diferent $t _ { j }$ during training, it is provided with trajectory data on $x _ { 1 } ^ { m }$ , which, in principle, allows deriving derivative information. Thus, we assume the derivatives of $x _ { 1 } ^ { m }$ to be known. With the assumption of known derivatives ${ \dot { x } } _ { 1 } ^ { m } ( t )$ and $\ddot { x } _ { 1 } ^ { m } ( t )$ , Equations (10) and (11) can be solved for the unknown initial states $x _ { 2 , 0 } ^ { u }$ and $x _ { 3 , 0 } ^ { u } { \mathrm { : } }$

$$
\begin{array}{l} x _ {2, 0} ^ {u} = \frac {\ddot {x} _ {1} ^ {m} (t)}{\dot {x} _ {1} ^ {m} (t)}, \\ x _ {3, 0} ^ {u} = \frac {\dot {x} _ {1} ^ {m} (t)}{e ^ {\left(\frac {\ddot {x} _ {1} ^ {m} (t)}{\dot {x} _ {1} ^ {m} (t)} t\right)}} - x _ {1, 0} ^ {m} \frac {\ddot {x} _ {1} ^ {m} (t)}{\dot {x} _ {1} ^ {m} (t)} \end{array}
$$

Thus, the states x<sup>u</sup> and $x _ { 3 } ^ { u }$ are observable if $\dot { x } _ { 1 } ^ { m } ( t ) \neq 0$ , although the incidence matrix suggests that state estimation should not work.

We provide empirical evidence showing that a PINN for the counter-example is indeed capable of estimating the unmeasured states $x _ { 2 } ^ { u }$ and $x _ { 3 } ^ { u }$ . To this end, we implement a corresponding PINN model that takes t as input and has the outputs ${ \mathrm { { \Omega } } } _ { 1 } m ( t ) , x _ { 2 } ^ { u } ( t )$ and $x _ { 3 } ^ { u } ( t )$ . We omit an input $x _ { 1 , 0 } ^ { m }$ , since we test the PINN for a single choice of initial values. We provide synthetic measurement data, i.e., samples $x _ { 1 } ^ { m } ( t _ { j } )$ , as training data, and integrate Equations $( 7 ) \ - ( 9 )$ as physics knowledge into the PINN. Specifically, we create training data using the explicit Runge-Kutta method of order 5, utilizing solve\_ivp solver from scipy.integrate module in Python (Virtanen et al., 2020; Dormand and Prince, 1980) and using $x _ { 1 , 0 } ^ { m } = 1$ $x _ { 2 , 0 } ^ { u } = 1$ and $x _ { 3 , 0 } ^ { u } = 2 $ . The time domain is chosen as $t \in [ 0 , 1 ]$

As can be seen from Figure 4, the PINN correctly estimates $x _ { 2 } ^ { u } ( t )$ and $x _ { 3 } ^ { u } ( t )$ . This finding supports the results obtained from the observability analysis and clarifies that the heuristic based on the incidence matrix does not constitute a necessary condition for state estimation.

![](images/6380d74c74e58cf53d33f93a360b41e9b4de60a7f106ad27a36a58e732569a68.jpg)  
Fig. 4. Predicted trajectory of $x _ { 1 } ^ { m } ( t )$ and estimated trajectories of x<sup>u</sup>(t) and $x _ { 3 } ^ { u } ( t )$ by the PINN (lines), along with the ground truth trajectory data obtained from the ODE solver (markers).

## SM6: Counter-example showing that the heuristic is not a suficient condition for the estimation of states

The following counter-example demonstrates that our heuristic (Section 2.3 in the main manuscript) is not a suficient condition for the estimation of states:

$$
\dot {x} _ {1} ^ {u} = x _ {1} ^ {u} + x _ {2} ^ {u} + y ^ {m}\tag{12}
$$

$$
\dot {x} _ {1} ^ {u} = \left((x _ {1} ^ {u}) ^ {2} + (x _ {2} ^ {u}) ^ {2} + (y ^ {m}) ^ {2} + 2 x _ {1} ^ {u} x _ {2} ^ {u} + 2 x _ {1} ^ {u} y ^ {m} + 2 x _ {2} ^ {u} y ^ {m}\right) ^ {\frac {1}{2}}\tag{13}
$$

Here, $y ^ { m }$ is the measured state, whereas the diferential states $x _ { 1 } ^ { u }$ and $x _ { 2 } ^ { u }$ are unmeasured.

It is easy to see that Equations (12) and (13) are dependent:

$$
\begin{array}{r l} & {\dot {x} _ {1} ^ {u} = \left((x _ {1} ^ {u}) ^ {2} + (x _ {2} ^ {u}) ^ {2} + (y ^ {m}) ^ {2} + 2 x _ {1} ^ {u} x _ {2} ^ {u} + 2 x _ {1} ^ {u} y ^ {m} + 2 x _ {2} ^ {u} y ^ {m}\right) ^ {\frac {1}{2}}} \\ & {\qquad = \left((x _ {1} ^ {u} + x _ {2} ^ {u} + y ^ {m}) (x _ {1} ^ {u} + x _ {2} ^ {u} + y ^ {m})\right) ^ {\frac {1}{2}}} \\ & {\qquad = x _ {1} ^ {u} + x _ {2} ^ {u} + y ^ {m}} \end{array}
$$

As the ODE system has infinitely many solutions, state estimation is impossible, although the incidence matrix has full-column rank (see Table 5).

Tab. 5. Incidence matrix of the counter-example represented by Equations (12)-(13). If an unmeasured state appears in an equation, it is marked with a cross. The matrix does have full-column rank.

<table><tr><td>$ [f,g] \downarrow $</td><td>$ [x^{u},y^{u}] \rightarrow $</td><td>$ x_{1}^{u} $</td><td>$ x_{2}^{u} $</td></tr><tr><td colspan="2">Eqn. (12)</td><td>⊗</td><td>×</td></tr><tr><td colspan="2">Eqn. (13)</td><td>×</td><td>⊗</td></tr></table>

## Bibliography

Backi, C.J., Emebu, S., Skogestad, S., Grimes, B.A., 2019. A simple modeling approach to control emulsion layers in gravity separators, in: 29th European Symposium on Computer Aided Process Engineering. Elsevier. volume 46 of Computer Aided Chemical Engineering, pp. 1159–1164. doi:10. 1016/B978-0-12-818634-3.50194-6.

Backi, C.J., Grimes, B.A., Skogestad, S., 2018. A Control- and Estimation-Oriented Gravity Separator Model for Oil and Gas Applications Based upon First-Principles. Industrial and Engineering Chemistry Research 57, 7201–7217. doi:10.1021/acs.iecr.7b04297.

Dormand, J., Prince, P., 1980. A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics 6, 19–26. doi:https://doi.org/10.1016/0771-050X(80)90013-3.

Gamboa-Torres, A., Flores-Tlacuahuac, A., 2000. Efect of process design/operation on the steady-state operability of a cstr: Reactions a→b→c. Chemical Engineering Research and Design 78, 481–491. doi:https://doi.org/10.1205/026387600527392.

Henschke, M., 1995. Dimensionierung liegender Flüssig-flüssig-Abscheider anhand diskontinuierlicher Absetzversuche. Ph.D. thesis. RWTH Aachen University.

Kalman, R., 1960. On the general theory of control systems. IFAC Proceedings Volumes 1, 491–502.

Kampwerth, J., Weber, B., Rußkamp, J., Kaminski, S., Jupke, A., 2020. Towards a holistic solvent screening: On the importance of fluid dynamics in a rate-based extraction model. Chemical Engineering Science 227. doi:10.1016/j.ces.2020.115905.

Kou, S.R., Elliott, D.L., Tarn, T.J., 1973. Observability of nonlinear systems. Information and Control 22, 89–99.

Kraume, M., Gäbler, A., Schulze, K., 2004. Influence of physical properties on drop size distributions of stirred liquid-liquid dispersions. Chemical Engineering and Technology 27, 330–334. doi:10.1002/ ceat.200402006.

Lee, E., Markus, L., 1967. Foundations of Optimal Control Theory. SIAM series in applied mathematics, Wiley.

Mersmann, A., 1980. Zum Flutpunkt in flüssig/flüssig–Gegenstromkolonnen. Chemie Ingenieur Technik 52, 933–942. doi:10.1002/cite.330521203.

Stokes, G.G., 2009. On the Efect of the Internal Friction of Fluids on the Motion of Pendulums. Cambridge University Press. Cambridge Library Collection - Mathematics, p. 1–10.

Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S.J., Brett, M., Wilson, J., Millman, K.J., Mayorov, N., Nelson, A.R.J., Jones, E., Kern, R., Larson, E., Carey, C.J., Polat, I., Feng, Y., Moore, E.W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E.A., Harris, C.R., Archibald, A.M., Ribeiro, A.H., Pedregosa, F., van Mulbregt, P., Vijaykumar, A., Bardelli, A.P., Rothberg, A., Hilboll, A., Kloeckner, A., Scopatz, A., Lee, A., Rokem, A., Woods, C.N., Fulton, C., Masson, C., Häggström, C., Fitzgerald, C., Nicholson, D.A., Hagen, D.R., Pasechnik, D.V., Olivetti, E., Martin, E., Wieser, E., Silva, F., Lenders, F., Wilhelm, F., Young, G., Price, G.A., Ingold, G.L., Allen, G.E., Lee, G.R., Audren, H., Probst, I., Dietrich, J.P., Silterra, J., Webber, J.T., Slavič, J., Nothman, J., Buchner, J., Kulick, J., Schönberger, J.L., de Miranda Cardoso, J.V., Reimer, J., Harrington, J., Rodríguez, J.L.C., Nunez-Iglesias, J., Kuczynski, J., Tritz, K., Thoma, M., Newville, M., Kümmerer, M., Bolingbroke, M., Tartre, M., Pak, M., Smith, N.J., Nowaczyk, N., Shebanov, N., Pavlyk, O., Brodtkorb, P.A., Lee, P., McGibbon, R.T., Feldbauer, R., Lewis, S., Tygier, S., Sievert, S., Vigna, S., Peterson, S., More, S., Pudlik, T., Oshima, T., Pingel, T.J., Robitaille, T.P., Spura, T., Jones, T.R., Cera, T., Leslie, T., Zito, T., Krauss, T., Upadhyay, U., Halchenko, Y.O., Vázquez-Baeza, Y.,

Contributors, S.., 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17, 261–272. doi:10.1038/s41592-019-0686-2.

Ye, S., Hohl, L., Charlafti, E., Jin, Z., Kraume, M., 2023a. Efect of temperature on mixing and separation of stirred liquid/liquid dispersions over a wide range of dispersed phase fractions. Chemical Engineering Science 274. doi:10.1016/j.ces.2023.118676.

Ye, S., Hohl, L., Kraume, M., 2023b. Impact of feeding conditions on continuous liquid-liquid gravity separation, part I: Inlet and outlet drop size, dense-packed zone and separation eficiency. Chemical Engineering Science 282. doi:10.1016/j.ces.2023.119237.