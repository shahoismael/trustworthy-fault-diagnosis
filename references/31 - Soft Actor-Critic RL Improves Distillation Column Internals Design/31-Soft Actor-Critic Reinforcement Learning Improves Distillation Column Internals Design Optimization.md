Article

# Soft Actor-Critic Reinforcement Learning Improves Distillation Column Internals Design Optimization

Dhan Lord B. Fortela <sup>1,2,</sup>\* , Holden Broussard <sup>1</sup>, Renee Ward <sup>1</sup>, Carly Broussard <sup>1</sup>, Ashley P. Mikolajczyk <sup>1,2</sup>, Magdy A. Bayoumi <sup>3</sup> and Mark E. Zappi <sup>1,2</sup>

Department of Chemical Engineering, University of Louisiana at Lafayette, Lafayette, LA 70504, USA; holden.broussard1@louisiana.edu (H.B.); renee.ward1@louisiana.edu (R.W.); carly.broussard1@louisiana.edu (C.B.); ashley.mikolajczyk@louisiana.edu (A.P.M.); mark.zappi@louisiana.edu (M.E.Z.)

2 The Energy Institute of Louisiana, University of Louisiana at Lafayette, Lafayette, LA 70504, USA 3 Department of Electrical and Computer Engineering, University of Louisiana at Lafayette, Lafayette, LA 70504, USA; magdy.bayoumi@louisiana.edu

米 Correspondence: dhanlord.fortela@louisiana.edu

![](images/e336e6feea8f27c42fd4834836e22689f2f03cc3b0fae90b471ce1ce417386be.jpg)

Academic Editors: Iuliana Deleanu and Gabriela Olimpia Isopencu

Received: 22 January 2025 Revised: 7 March 2025 Accepted: 12 March 2025 Published: 18 March 2025

Citation: Fortela, D.L.B.; Broussard, H.; Ward, R.; Broussard, C.; Mikolajczyk, A.P.; Bayoumi, M.A.; Zappi, M.E. Soft Actor-Critic Reinforcement Learning Improves Distillation Column Internals Design Optimization. ChemEngineering 2025, 9, 34. https://doi.org/10.3390/ chemengineering9020034

Copyright: © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/).

Abstract: Amid the advancements in computer-based chemical process modeling and simulation packages used in commercial applications aimed at accelerating chemical process design and analysis, there are still certain tasks in design optimization, such as distillation column internals design, that become bottlenecks due to inherent limitations in such soft ware packages. This work demonstrates the use of soft actor-critic (SAC) reinforcement learning (RL) in automating the task of determining the optimal design of trayed multistage distillation columns. The design environment was created using the AspenPlus<sup>®</sup> software (version 12, Aspen Technology Inc., Bedford, Massachusetts, USA) with its RadFrac module for the required rigorous modeling of the column internals. The RL computational work was achieved by developing a Python package that allows interfacing with AspenPlus<sup>®</sup> and by implementing in OpenAI’s Gymnasium module (version 1.0.0, OpenAI Inc., San Francisco, California, USA) the learning space for the state and action variables. The results evidently show that (1) SAC RL works as an automation approach for the design of distillation column internals, (2) the reward scheme in the SAC model significantly affects SAC performance, (3) column diameter is a significant constraint in achieving column internals design specifications in flooding, and (4) SAC hyperparameters have varying effects on SAC performance. SAC RL can be implemented as a one-shot learning model that can significantly improve the design of multistage distillation column internals by automating the optimization process.

Keywords: machine learning; reinforcement learning; chemical process design

## 1. Introduction

Computer-aided chemical process simulation has been an important tool not just in chemical engineering education, but also in commercial platforms for the design and analysis of industrial systems that process bulk chemicals, specialty chemicals, pharmaceuticals, and many more [1,2]. Among several software packages, AspenPlus<sup>®</sup>, which is part of Aspen Tech’s suite of software packages [3–5], has been the state of the art due to its wide array of functional capabilities (e.g., property analysis, rigorous models, process flowsheeting, API for external software such as Fortran and Python, etc.) and its extensive database of chemical and physical properties [3,6–10]. Despite being the most advanced chemical process modeling software, this software has its inherent limitations [11]. One such limitation is that it still requires users to perform manual iterative optimization of<sup>uch</sup> <sup>limitation</sup> <sup>is</sup> <sup>that</sup> <sup>it</sup> <sup>still</sup> <sup>requires</sup> <sup>users</sup> <sup>to</sup> <sup>perform</sup> <sup>manual</sup> <sup>iterative</sup> <sup>optimization</sup> certain designs, which can sometimes result in extended periods spent on iterations by<sup>ertain</sup> <sup>designs,</sup> <sup>which</sup> <sup>can</sup> <sup>sometimes</sup> <sup>result</sup> <sup>in</sup> <sup>extended</sup> <sup>periods</sup> <sup>spent</sup> <sup>on</sup> <sup>iterations</sup> the user or in a sub-optimal final design [12]. These challenges can be exacerbated when<sup>he</sup> <sup>user</sup> <sup>or</sup> <sup>in</sup> <sup>a</sup> <sup>sub-optimal</sup> <sup>final</sup> <sup>design</sup> <sup>[12].</sup> <sup>These</sup> <sup>challenges</sup> <sup>can</sup> <sup>be</sup> <sup>exacerbated</sup> <sup>wh</sup> dealing with a high-dimensional space in the design parameters being explored [11,12].<sup>ealing</sup> <sup>with</sup> <sup>a</sup> <sup>high-dimensional</sup> <sup>space</sup> <sup>in</sup> <sup>the</sup> <sup>design</sup> <sup>parameters</sup> <sup>being</sup> <sup>explored</sup> <sup>[11,1</sup> Addressing these challenges and establishing solutions can make the design process via thisddressing these challenges and establishing solutions can make the design process software tool (and other similar software packages) significantly more efficient. This workhis software tool (and other similar software packages) significantly more eficient. T aimed to demonstrate the use of reinforcement learning (RL), specifically the soft actorork aimed to demonstrate the use of reinforcement learning (RL), specifically the s critic (SAC) algorithm [13,14], to improve the design optimization task for the internals ofctor-critic (SAC) algorithm [13,14], to improve the design optimization task for the int multistage distillation columns (Figure 1).<sub>als</sub> <sub>of</sub> <sub>multistage</sub> <sub>distillation</sub> <sub>columns</sub> <sub>(F</sub>

![](images/0318676a87735f037d28d7dcb1b1ad84e124ee310d9c39da92aeb941ea3e47a8.jpg)  
<sup>igure</sup> <sup>1.</sup> <sup>Schematic</sup> <sup>of</sup> <sup>the</sup> <sup>implementation</sup> <sup>of</sup> <sup>the</sup> <sup>soft</sup> <sup>actor-critic</sup> <sup>(SAC)</sup> <sup>RL</sup> <sup>algorithm</sup> <sup>in</sup> <sup>the</sup> <sup>opti</sup>Figure 1. Schematic of the implementation of the soft actor-critic (SAC) RL algorithm in the optimizaation of column internals design in a multistage distillation column. The Environment box is stion of column internals design in a multistage distillation column. The Environment box is separate from the SAC RL model box, but there exists a mechanism for the environment to be observed by the SAC RL to collect the environment’s state and reward and a mechanism for the SAC RL model to apply actions to the environment.

## 1.1. The Challenge: Distillation Column Internals Design

The specification of column internals in a multistage distillation column significantly <sup>The</sup> <sup>specification</sup> <sup>of</sup> <sup>column</sup> <sup>internals</sup> <sup>in</sup> <sup>a</sup> <sup>multistage</sup> <sup>distillation</sup> <sup>column</sup> <sup>significan</sup>dictates the hydraulics involved in the interactions of the vapor and liquid streams through ictates the hydraulics involved in the interactions of the vapor and liquid streathe various stages inside the column (Figure 1). This consequently creates a feedback loop hrough the various stages inside the column (Figure 1). This consequently creates a fein the usual design process—the hydraulics in the column become metrics of the goodness ack loop in the usual design process—the hydraulics in the column become metricsof the column internals design specifications [15]. In this study, the hydraulics parameter he goodness of the column internals design specifications [15]. In this study, the called the percent (%) approach to flooding, which is denoted here using the notation $f _ { \% } { } _ { , }$ dr, is used to measure the goodness of the distillation column internals design. This parameter $f _ { \% }$ is not the only metric of good column design [2,15], but it is a significant indicator [16]. The numerous nonlinear relations and models involved in computing the column internals parameters that produce allowable values of $f \%$ make this design task very difficult to automate using traditional programming techniques. Even the AspenPlus<sup>®</sup> software does not currently have a built-in automated optimization function for this task [3,10]. Hence, this study demonstrates an SAC RL approach to provide this valuable automation. For a seamless integration between the distillation column internals parameters and the SAC RL model, this study uses the RadFrac module available in AspenPlus<sup>®</sup>. The RadFrac module is the most rigorous module in AspenPlus<sup>®</sup> fitting to be used in column internals design because of the stage-by-stage vapor–liquid calculations, hydraulics analysis, and its support for trayed and packed column designs [17,18]. A Python (version 3.9.21, Python Software Foundation, Wilmington, Delaware, USA) module called ‘pywin32’ [19] serves as the API to read data from and send data to AspenPlus<sup>®</sup>.

## 1.2. The Solution: SAC Algorithm-Based Design Optimization

This work evaluates the capability of SAC to automatically learn the best design parameter settings in the RadFrac column. SAC is an algorithm that optimizes a stochastic policy in an off-policy manner, which implements the combined capabilities of stochastic policy optimization and deep deterministic policy gradient (DDPG) approaches [13,14,20]. Because of the blend of stochastic and deterministic capabilities, the prediction of the AC model for the next action steps are well within the bounds of the defined action space [13,14]. This is a feature that can be fitting to implement in distillation column design because the column parameters are allowed to be sampled from only a finite range of values due to limitations in fluid dynamics, vapor–liquid equilibrium, energy balance, and other constraints imposed by natural law.

Note that a DDPG model was also evaluated during the preliminary works for this project, and the DDPG model predictions failed to stay within the bounds of the action variables (column internals design parameters), resulting in input value errors of the column design parameters when implemented in the AspenPlus<sup>®</sup> model file. This prompted the researchers to eliminate DDPG and focus only on SAC as the RL to evaluate for column design optimization in this work.

## 2. Methodology

## 2.1. Environment: Multistage Distillation Column as RadFrac in AspenPlus<sup>®</sup>

A model of tray-type multistage distillation for the separation of ethanol $( C _ { 2 } \mathrm { { H } _ { 5 } \mathrm { { O H ) } } }$ and water $\left( \mathrm { H } _ { 2 } \mathrm { O } \right)$ was chosen because this binary chemical system is commonly used in many design analysis tasks in chemical engineering. Nonetheless, the methodology of this study should naturally be applied to other chemical systems undergoing distillation in a trayed column. The well-established AspenPlus<sup>®</sup> software was chosen as the modeling platform for the tray-type column using its RadFrac module. Please refer to the accompanying AspenPlus<sup>®</sup> file (.apw and bkp) for all other details of the model used (see Data Availability section).

The details of the binary mixture to be separated via distillation are based on the literature materials [2,15] and are as follows. A mixture of ethanol and water must be distilled in a sieve tray-type distillation column. The feed contains 35% ethanol and 65% water, and the distillate product stream must contain at least 80% ethanol. Note that 87% ethanol is the azeotropic point of the binary system under atmospheric pressure [15], which is the reason for the target distillate concentration of ethanol. The feed is flowing at 20,000 kmo $/ \mathrm { h } ,$ is a saturated liquid, and is coming to the column at pressure of 1 bar. The distillate rate must be at 104 kmol/h and the external reflux ratio must be 2.5. After some preliminary graphical methods for estimating the number of stages (using the McCabe– Thiele method) [2,15,21], it was determined that the total number of stages must be 20 and the feed tray must be on stage 10. The column must have a total condenser and a partial reboiler. The task is to design a sieve tray type multistage distillation column (see Figure 2) using the RadFrac module in AspenPlus<sup>®</sup>. Specifically, the column internals must be optimized with the objective of achieving the approach to flooding $f _ { \% }$ levels to within the acceptable range, e.g., 80–90% (see Section 2.2).

(A)  
![](images/6e746c6ccfa94088e9253c8d63e78b503070946742ceb82c4a1d1290f920f7fd.jpg)

![](images/de89a071fd3c844be46b532998cfe6e792dc104616aa47845061e859f92809bb.jpg)  
Figure 2. Distillation column internals design parameters studied for the application of SAC.Figure 2. Distillation column internals design parameters studied for the application of SAC. (A) crossross-section view of a vertical section of the column indicating the parameters of column diamsection view of a vertical section of the column indicating the parameters of column diameter and tray spacing, (B) top view of a tray indicating the parameters of weir side length and column diameter, (C) view indicating the parameters of downcomer clearance and weir height, and (D) view of a sieve tray indicating parameter hole diameter.

To simplify the modeling in AspenPlus<sup>®</sup> and the implementation of SAC RL, the feed <sup>.2.</sup> <sup>Distillation</sup> <sup>Column</sup> <sup>Flooding</sup> tray was assigned to the TOP section of the column. So, the TOP section covers Stage 1 to The state variable for this study is the column flooStage 10 while the BOT section covers Stage 11 to Stage 20.

## 2.2. Distillation Column Flooding

The state variable for this study is the column flooding measured as % approach to flooding (denoted here as $f _ { \% } ) _ { - }$ , which is one of the main performance metrics for sizing distillation column internals [16]. Flooding is the excessive accumulation of liquid in the column, resulting in poor equilibrium contact between the vapor and liquid streams. The % approach to flooding must be minimized, and a typical range of % approach to flooding is 65% to 90% according to Wankat [15], while Jones and Mellborn [22] suggest a 75% <sup>ccurs</sup> <sup>when</sup> <sup>the</sup> <sup>liquid</sup> <sup>is</sup> <sup>backed</sup> <sup>up</sup> <sup>into</sup> <sup>the</sup> <sup>downcomer</sup> <sup>due</sup> <sup>to</sup> <sup>tray</sup> <sup>pressure</sup> <sup>drop,</sup> <sup>w</sup>approach to flooding is a good value for various chemical systems. There are four main <sup>s</sup> <sup>usually</sup> <sup>caused</sup> <sup>by</sup> <sup>short</sup> <sup>tray</sup> <sup>spacing</sup> <sup>[16,22].</sup> <sup>Downcomer</sup> <sup>choke</sup> <sup>flooding</sup> <sup>(Figure</sup>mechanisms of flooding, as shown in Figure 3. Downcomer backup flooding (Figure 3A) <sup>occurs</sup> <sup>when</sup> <sup>the</sup> <sup>entrance</sup> <sup>to</sup> <sup>the</sup> <sup>downcomer</sup> <sup>is</sup> <sup>too</sup> <sup>narrow,</sup> <sup>resulting</sup> <sup>in</sup> <sup>build-up</sup> <sup>of</sup> <sup>ex</sup>occurs when the liquid is backed up into the downcomer due to tray pressure drop, which is <sup>riction</sup> <sup>losses</sup> <sup>for</sup> <sup>the</sup> <sup>liquid</sup> <sup>to</sup> <sup>overcome</sup> <sup>[16].</sup> <sup>Spray</sup> <sup>entrainment</sup> <sup>flooding</sup> <sup>(Figure</sup> <sup>3C)</sup>usually caused by short tray spacing [16,22]. Downcomer choke flooding (Figure 3B) occurs curs when there is a relatively low amount of liquid on the tray because of relatively swhen the entrance to the downcomer is too narrow, resulting in build-up of excess friction weir height and narrow downcomer clearance [16]. Froth entrainment flooding (Filosses for the liquid to overcome [16]. Spray entrainment flooding (Figure 3C) occurs when D) occurs when significant height of froth reaches the tray above, resulting in comthere is a relatively low amount of liquid on the tray because of relatively short weir height and narrow downcomer clearance [16]. Froth entrainment flooding (Figure 3D) occurs when significant height of froth reaches the tray above, resulting in compromise of the stream concentrations, and this can be a significant effect of settings in the tray spacing, sieve hole diameter, and fraction of the tray allocated for vapor–liquid exchange [15,16].

![](images/fd722465a85456ab843bd6ed41dcf69a6c5eb2ba5e7fa88440ce8a17815a9a84.jpg)  
Figure 3. Distillation tray flooding mechanisms: (A) downcomer backup flooding, (B) downcomerFigure 3. Distillation tray flooding mechanisms: (A) downcomer backup flooding, (B) downcomer choke flooding, (C) spray entrainment flooding, and (D) froth entrainment floodingchoke flooding, (C) spray entrainment flooding, and (D) froth entrainment flooding.

## 2.3. Notation

We now present notations to formally define the various components of the task in relation to the established notations in RL. The RL problem at hand is a policy search in a Markov Decision Process (MDP) involving the tuple $( S , A , p , r )$ . The state space S is a continuous space that consists of the following column internals design parameters (Figures 1 and 2): top section (TOP) % approach to flooding and bottom section (BOT) % approach to flooding. Note that the state % approach to flooding can be greater than 100% according to distillation column design principles, so $S ~  ~ [ 0 , \infty )$ . The action space A is a continuous space that consists of the following column internals design parameters (Figures 1 and 2): tray spacing, weir height, downcomer clearance, weir side length, and hole diameter. The action space is bounded by the allowable values of column design parameters in the $\mathrm { \sf A s p e n P l u s } ^ { \mathrm { \tiny \textregistered } }$ software, and the actual values also scale with the units (i.e., ft, cm, or mm; kg or lb; etc.) they are assigned in the software by the user, as explained above. The state transition probability p represents the probability density of the next state $s _ { t + 1 } \in S$ given the current state $s _ { t } \in S$ and action $a _ { t } \in A$ . The environment (i.e., column internals design) emits a reward $r \in R  [ r _ { m i n } , r _ { m a x } ]$ from each transition (i.e., process simulation). Note that this study also evaluated how various forms of the reward system R may affect the SAC performance.

Table 1 summarizes the variables for the action space and state space with their corresponding descriptions according to the actual settings of the AspenPlus<sup>®</sup> model used in the study. Also refer to Figure 2 for visualization of the action variables defined in Table 1. The action space A is multidimensional and consists of the column design parameters, denoted as $A _ { 1 } , A _ { 2 } , \ldots A _ { 1 1 }$ , as shown in Table 1. The state space S is two-dimensional and consists of $S _ { 1 }$ and $S _ { 2 }$ (Table 1). Note that each section of the column, i.e., TOP and BOT, is represented in the action and state spaces even though the design parameters between the sections are similar.

Table 1. Summary of action and state variables used in the study.

<table><tr><td>Action/State Variable</td><td>Description</td></tr><tr><td>Action 1 ( $A_1$ ) = TOP Downcomer clearance</td><td>Distance between the bottom edge of the downcomer and the tray below; value range: [30, 150]; units: mm</td></tr><tr><td>Action 2 ( $A_2$ ) = TOP Tray spacing</td><td>Distance between two consecutive trays; value range: [1, 7]; units: ft</td></tr><tr><td>Action 3 ( $A_3$ ) = TOP Weir height</td><td>Height of a tray outlet weir, which regulates the amount of liquid build-up on the plate surface; value range: [10, 150]; units: mm</td></tr></table>

Table 1. Cont.

<table><tr><td>Action/State Variable</td><td>Description</td></tr><tr><td>Action 4 ( $A_4$ ) = TOP Sieve hole diameter</td><td>Diameter of the holes on the sieve tray; value range: [5, 15]; units: mm</td></tr><tr><td>Action 5 ( $A_5$ ) = TOP Weir side length</td><td>Length of the tray outlet weir; value range: [0.1, 1]; units: ft</td></tr><tr><td>Action 6 ( $A_6$ ) = BOT Downcomer clearance</td><td>Distance between the bottom edge of the downcomer and the tray below; value range: [30, 150]; units: mm</td></tr><tr><td>Action 7 ( $A_7$ ) = BOT Tray spacing</td><td>Distance between two consecutive trays; value range: [1, 7]; units: ft</td></tr><tr><td>Action 8 ( $A_8$ ) = BOT Weir height</td><td>Height of a tray outlet weir, which regulates the amount of liquid build-up on the plate surface; value range: [10, 150]; units: mm</td></tr><tr><td>Action 9 ( $A_9$ ) = BOT Sieve hole diameter</td><td>Diameter of the holes on the sieve tray; value range: [5, 15]; units: mm</td></tr><tr><td>Action 10 ( $A_{10}$ ) = BOT Weir side length</td><td>Length of the tray outlet weir; value range: [0.1, 1]; units: ft</td></tr><tr><td>Action 11 ( $A_{11}$ ) = Column diameter **</td><td>Diameter of the column; value range: [4, 9]; units: ft</td></tr><tr><td>State 1 ( $S_1$ ) = TOP  $f_\%$ </td><td>Flooding  $f_\%$  of the column top section; value range : [0,∞); units: %</td></tr><tr><td>State 2 ( $S_2$ ) = BOT  $f_\%$ </td><td>Flooding  $f_\%$  of the column bottom section; value range : [0,∞); units: %</td></tr></table>

\*\* Note: column diameter $A _ { 1 1 }$ was included in the SACR RL action space (i.e., varied) only when testing the effect of column diameter (see Section 3.3). Otherwise, A was fixed when SAC RL was learning to optimize the other column design parameters.

## 2.4. SAC RL

We present in this section the key equations of the SAC RL algorithm, but the reader should consult the original works [13,14,20] that established this algorithm if more details are needed.

The goal in SAC RL is to determine a policy π, which maximizes two components: (1) the expected return $R ( \tau )$ from the rewards $\displaystyle r _ { t } ,$ where ${ \boldsymbol \tau } = ( s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , \dots )$ is the trajectory, when the agent acts according to the policy and (2) the entropy of the policy H(π), i.e., entropy of the predicted next actions. To achieve this, the policy model π must be estimated by an optimal policy $\pi ^ { * }$ that is calibrated (via its model parameters) using data collected about the environment [14]. Policy, in this context, is a rule used to decide what next actions to take based on the current state of the environment. Specifically, π is a probability distribution over the possible actions $a _ { t }$ given the current state $s _ { t }$ of the environment, i.e., $a _ { t } \ \sim \ \pi ( \cdot | s _ { t } )$ . The optimal policy $\pi _ { \theta } ^ { * }$ is modeled using deep neural network with network weights $\theta ,$ and this becomes the actor network. In essence, the main goal in training an SAC RL model can then be expressed generally as follows:

$$
\pi^ {*} = \pi^ {*} = \operatorname * {a r g m a x} _ {\pi} \mathrm{E} _ {\tau \sim \pi} \Big [ \sum_ {t = 0} ^ {\infty} \gamma^ {t} (R (s _ {t}, a _ {t}, s _ {t + 1}) + \alpha H (\pi (\cdot | s _ {t}))) \Big ]\tag{1}
$$

where E is the expectation, α is the trade-off coefficient for the entropy H term, and $\gamma$ is the discount factor in the value function [20].

In general, the reward function R depends on the current state $s _ { t } ,$ , the action just taken ${ { a } _ { t } } ,$ and the next state $s _ { t + 1 } ,$ such that $r _ { t } = R ( s _ { t } , a _ { t } , s _ { t + 1 } )$ . The dependence in the function R is frequently simplified to have dependence on the state only, i.e., $r _ { t } = R ( s _ { t } )$ ), or on the state action only, i.e., $r _ { t } = R ( s _ { t } , a _ { t } )$ . The SAC model used in this study used the former model $r _ { t } = R ( s _ { t } )$ . One of the two components of the objective function in the SAC RL agent is to maximize the cumulative reward over a trajectory $\tau ,$ and this cumulative reward is commonly called the return $R ( \tau )$ . It has been shown that a discounted return helps in the convergence of RL models, so this is also adopted in the SAC learning model such that $R ( \tau ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { t }$ , where γ is the discount factor [14].

The maximum entropy term $H ( \pi )$ (second term in Equation (1)) is the unique feature of the SAC RL. By design, this term encourages the RL model to still explore as it learns to prioritize actions with projected high rewards [14]. This prevents the model from being focused on sub-optimal actions [23].

The determination of the best actor network $\pi _ { \theta } ^ { * }$ is achieved by using value feedback from a system of critic networks (see Figure 1), which are also deep neural networks and that implement double Q-learning [14]. SAC concurrently learns the policy $\pi ^ { * }$ and two Q-functions $Q _ { \phi _ { 1 } }$ and $Q _ { \phi _ { 2 } }$ of the critic, where $\phi _ { 1 }$ and $\phi _ { 2 }$ are the network weights of the critic networks.

The smoothing constant, $\tau \in \ [ 0 , 1 ]$ , is a hyperparameter that controls updating the weights of the target networks for the Q-functions $\boldsymbol { Q } _ { \phi _ { 1 } ^ { \prime } }$ ′ and $Q _ { \phi _ { 2 } ^ { \prime } } \ [ 2 4 ]$ . In notation: $\phi _ { 1 } { } ^ { \prime }  \tau \phi _ { 1 } + ( 1 - \tau ) \phi _ { 1 } { } ^ { \prime }$ and $\phi _ { 2 } { ' }  \tau \phi _ { 2 } + ( 1 - \tau ) \phi _ { 2 } { ' }$

The fourth hyperparameter to be evaluated is the “replay buffer length”, which originates from using replay buffer as a solution to extract more information from the history sequential data for RL training [24]. Replay buffer is a component of the computation of the target Q-functions $Q _ { \phi _ { 1 } ^ { \prime } }$ and $Q _ { \phi _ { 2 } }$ ′ presented above. The replay buffer is a finitesized (replay buffer length) memory datum. The buffer is created by sampling transitions from the environment according to the exploration policy, and the tuple $\left( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } \right)$ is stored in the replay buffer. The buffer eventually is filled as more tuple entries are added. When the buffer length is reached, the oldest samples are discarded while maintaining the buffer length. At each timestep, the actor and critic are updated by sampling a minibatch uniformly from the buffer.

## 2.5. Implementation: OpenAI Gymnasium, PyTorch (Version 2.6.0+cu118)

To eliminate the need to create our custom RL environment, we used the prebuilt RL module called “Gymnasium” by OpenAI [25,26]. This module implements the necessary algorithms and functions to define variable spaces and efficient action space-sampling mechanisms to ensure that RL models are exposed to a wide variety of scenarios, resulting in improved learning performance [25,27]. The Python codes are based on prior work with code implementation of RL policy gradient methods [28].

## 2.5.1. Action Variables

Since all action variables in A are continuous, these were defined using the ‘Gymnasium.Box()’ function, which is designed to handle a tensor of continuous variables. The lower and upper bounds of these action variables were required arguments in the ‘Gymnasium.Box()’ functions.

## 2.5.2. State Variables

Defining the state space was very similar to how the action space (Section 2.5.1) was defined using the ‘Gymnasium.Box()’ function.

## 2.5.3. Reward Scheme

The reward schemes studied are summarized in Table 2. Reward is based on the flooding level $f \%$ of each column section and the scheme for awarding the reward values. Previous works have demonstrated that the reward scheme can significantly affect the performance of RL algorithms [29–32]. The unique aspect of the reward schemes adopted for this study is the separate computation of reward for each column section $r _ { k }$ and then the aggregation of these by summation (weighted and unweighted) into a single reward value r (Table 2). Schemes 1 and 2 are built on a model that computes the penalty for deviating from the target state by taking the difference of the target state value(s) and the current value of the state [32]. Schemes 3 and 4 are based on the concept of dividing the state values according to ranges and assigning the highest score on the target range while assigning decreasing reward scores as the binning range moves away from the target range. Scheme 5 is built on top of a binary reward system; hence, it represents a binary scheme. The coding of these reward schemes was performed in the environment definition via Python scripting.

Table 2. Summary of reward schemes used in this study.

<table><tr><td>Reward Model</td><td>Definition</td></tr><tr><td>Scheme 1</td><td> $r = \sum_{k \in \{TOP, BOT\}}^{K} r_k = r_{TOP} + r_{BOT}$ where  $r_k$  is the reward in section  $k$  of the column. For each  $k$ : $r_k = \begin{cases} 100, & 80 \leq f_\% < 90 \\ 100 - 2d_{err}, & f_\% < 80 \\ 100 - 2d_{err}, & f_\% \geq 90 \end{cases}$ where  $d_{err} = |f_\% - 80|$  or  $d_{err} = |f_\% - 90|$ </td></tr><tr><td>Scheme 2</td><td> $r = \sum_{k \in \{TOP, BOT\}}^{K} \frac{1}{K} r_k = 0.5 r_{TOP} + 0.5 r_{BOT}$ where  $r_k$  is the reward in section  $k$  of the column. For each  $k$ : $r_k = \begin{cases} 100, & f_\% \leq 85 \\ -d_{err}, & f_\% > 85 \end{cases}$ where  $d_{err} = |f_\% - 85|$ </td></tr><tr><td>Scheme 3</td><td>Reward  $r_k$  is based on the intervals of  $f_\%$  with highest score for the target interval [80, 90). $r = \sum_{k \in \{TOP, BOT\}}^{K} \frac{1}{K} r_k = 0.5 r_{TOP} + 0.5 r_{BOT}$ where  $r_k$  is the reward in section  $k$  of the column. For each  $k$ : $r_k = -100 \text{ if } f_\% \in [200, \infty); r_k = -80 \text{ if } f_\% \in [180, 200); r_k = -60 \text{ if } f_\% \in [160, 180); r_k = -40 \text{ if } f_\% \in [140, 160); r_k = -30 \text{ if } f_\% \in [120, 140); r_k = -20 \text{ if } f_\% \in [100, 120); r_k = -10 \text{ if } f_\% \in [90, 100); r_k = 0$ if  $f_\% \in [0, 400); r_k = 20 \text{ if } f_\% \in [40, 50); r_k = 40 \text{ if } f_\% \in [50, 60); r_k = 60$ if  $f_\% \in [60, 70); r_k = 80 \text{ if } f_\% \in [70, 80); r_k = 100 \text{ if } f_\% \in [80, 90)$ </td></tr><tr><td>Scheme 4</td><td>Reward  $r_k$  is similar to Scheme 3 but with reward values lower by a factor of 10: $r = \sum_{k \in \{TOP, BOT\}}^{K} \frac{1}{K} r_k = 0.5 r_{TOP} + 0.5 r_{BOT}$ where  $r_k$  is the reward in section  $k$  of the column. For each  $k$ : $r_k = -10 \text{ if } f_\% \in [200, \infty); r_k = -8 \text{ if } f_\% \in [180, 200); r_k = -6 \text{ if } f_\% \in [160, 180); r_k = -4 \text{ if } f_\% \in [140, 160); r_k = -3 \text{ if } f_\% \in [120, 140); r_k = -2 \text{ if } f_\% \in [100, 120); r_k = -1 \text{ if } f_\% \in [90, 100); r_k = 0 \text{ if } f_\% \in [0, 40); r_k = 2 \text{ if } f_\% \in [40, 50); r_k = 4$ if  $f_\% \in [50, 60); r_k = 6 \text{ if } f_\% \in [60, 70); r_k = 8 \text{ if } f_\% \in [70, 80); r_k = 10 \text{ if } f_\% \in [80, 90)$ </td></tr><tr><td>Scheme 5</td><td>The reward is built on a binary baseline scheme: $r = \sum_{k \in \{TOP, BOT\}}^{K} \frac{1}{K} r_k = 0.5 r_{TOP} + 0.5 r_{BOT}$ where  $r_k$  is the reward in section  $k$  of the column. For each  $k$ : $r_k = \begin{cases} 1, & f_\% \leq 85 \\ -1, & f_\% > 85 \end{cases}$ </td></tr></table>

## 2.5.4. Hardware Setup

The computer specifications are the following. Unit: Dell Precision 7670 Mobile Workstation; CPU: Intel Core i7-12850HX, 25 MB cache, 24 threads, 16 core, 2.1 GHz–4.8 GHz; memory: 64 GB DDR5; GPU: NVIDIA GeForce RTX 3080Ti with 16 GB GDDR6. All computational work was implemented in this laptop computer running the AspenTech version 12 software suite containing the AspenPlus<sup>®</sup> and Anaconda Navigator version 2.5 software [33], containing Python version 3.10 and all the necessary Python packages, such as PyTorch (CUDA-enabled), Gymnasium by OpenAI, and the pywin32 package [19] that interfaces with Windows to access the AspenPlus<sup>®</sup> file. Note that even though this study used a GPU-enabled computer to accelerate PyTorch computations, a CPU-only implemen tation of PyTorch can still accomplish the computations, but at slower computation speeds. Nonetheless, the computations in this work do not require specialized computer hardware beyond a typical hardware requirement for running the AspenPlus<sup>®</sup> software.

## 2.5.5. Code and Documentation of the Work Performed

The necessary codes used in this work have been organized and stored in an online repository via GitHub, as detailed in the Data Availability section. There are two main code files: (1) a Python class file that defines all functions used to read data from (output data) and write data to (input data) AspenPlus<sup>®</sup> via Python, and (2) Jupyter Notebook (version 6.29.5) files that document the implementation of the SAC RL.

## 2.6. SAC RL Runs

Since RL training is very sensitive to randomization of the initial weight in the deep neural networks in the actor and in the critic components of the SAC model, a set of ten runs was implemented for each learning setting. Each run had a unique random number generator (RNG) index value used for all PyTorch computations, which consequently fixed the random number used in initializing the weights of the neural networks. The ten unique RNG indices can be seen in the accompanying Jupyter Notebook files in the project online repository (see Data Availability section). This deliberate randomization using known RNG indices accomplishes two crucial tasks: (1) testing the robustness in SACR RL models and (2) allowing for the repeatability of the randomized runs and results.

All runs were set to a maximum of 500 iterations per SAC learning run, where each iteration consisted of one cycle of the SAC model reading the current state and reward values of the environment (column model in AspenPlus<sup>®</sup>), SAC model making a prediction for the next actions, and sending the predicted actions to the environment for simulation (to reach the next state and reward for the next cycle). Even though the traditional fields implementing RL (e.g., computer science, robotics, etc.) would usually set maximum iterations up to the range of millions [13], distillation column design iteration becomes impractical if the maximum iterations become very high, so our setting of a maximum of 500 iterations per SAC learning run was based on a practical basis. The data collected from each run as discussed above would not require extensive downstream processing, as these data by themselves are sufficient to evaluate the performance of SAC RL.

Furthermore, each run started with an untrained SAC RL model. This design of the study aimed to demonstrate that even an untrained SAC RL model can learn to optimize within a reasonable length of time (max of 500 iterations).

## 2.7. Data Collection and Analysis

There were numerous data that could be collected in this study, because the nature of the environment is highly data intensive, inherent in computer-based design of distillation columns in AspenPlus<sup>®</sup>, and because the SAC RL training is data intensive. This prompted the research team to focus on collecting only key data that would enable evaluation of the performance of SAC RL in column internals design. The data for the following variables were saved and used for discussion: levels of the action variables, levels of the state variables, levels of the reward variables, and runtime per iteration. The saving of these datasets was coded in the run codes to eliminate user errors and streamline the analysis.

## 3. Results and Discussion

The results are presented in the following order: Section 3.1 covers the results to demonstrate the feasibility of using SAC RL for distillation column design; Section 3.2 shows the effect of the reward scheme on the performance of SAC RL; Section 3.3 discusses the effect of column diameter on the performance of SAC; Section 3.4 shows the effect of the SAC’s hyperparameters setting on its performance; and Section 3.5 covers the comparison of SAC RL with established optimization algorithms.

## 3.1. Feasibility of Using SAC RL for Distillation Column Internals Design

The first question that must be answered is, “Does SAC RL work as an automation method in optimizing distillation column internals design?” The answer is yes, it works, but there are pertinent trends in its performance that must be addressed, which we now cover as we proceed. The evidence for this claim can be seen in Figures 4 and 5, which were the results when SAC RL was implemented on the column with a fixed uniform (in both TOP and BOT sections) diameter of 6 feet. Using the reward Scheme 1, the SAC model learned how to optimize the column design as the training progressed, converging to predicting column designs that maximize the reward (max of 200), as seen in Figure 4A, and selecting actions (column design parameter levels) that resulted in % flooding within the target range of 80–90% (Table 2, Scheme 1), as seen in Figure 4B,C. Initially, the SAC model did not have any optimal policy in determining the best next actions, because the deep neural networks in the critic and actor components were randomly initialized. This is evident in the very large standard deviation of the reward values and state variable values at the beginning of the training (Figure 4A–C). However, as the cycle of training–prediction–implementation steps progressed, the model learned the best actions that result in high rewards and the bad actions that result in low rewards, and it eventually converges to optimal policies (actor and critic) that favor the prediction of actions (Figure 5) that result in favorable $f _ { \% }$ values resulting in the highest rewards (Figure 4). This trend of convergence is very evident in the last 100 iterations, i.e., 400–500 iteration steps accompanied by smaller (narrower) standard deviations of reward and state variable values (Figure 4).

Another crucial aspect of the design task is the duration of the SAC RL automated optimization, because chemical process design can be time sensitive [3]. The summary of runtime (in seconds) as shown in Figure 4D indicates that the average time per design iteration by the SAC model is 1 s (on the laptop computer used in this study—see Section 2.5.4, Hardware Setup, for details on the computer). This means that a single run with a maximum of 500 iterations (Figure 4) takes only around 10 min, which may be significantly lower than the time it takes to iterate the design manually.

With the numerous iterations that resulted in $f \%$ values within the target range [80%, 90%) as shown in Figure 4B,C, the SAC-based optimization essentially exhaustively discovered numerous ways in which the column internals parameters can be set while achieving target design specification, i.e., $f _ { \% } \in [ 8 0 \% , 9 0 \% )$ . The action space trends shown in Figure 5 show the corresponding column design parameters for these design iterations that meet the design specification.

(A)  
![](images/f23e07357808f6e7c4ba8949bac9a647c098afad14643fc4896facbf0e4ca1e4.jpg)  
(B)

![](images/4f48ce3daeaa3df618fa31d55934a8d6b0890bd71222ebee9c47733143610930.jpg)  
(C)

![](images/e923063f38e2c17089959019dc9509dd5444bb4a3b1958c23680ea2ddc1528af.jpg)  
(D)

![](images/352d0317974155dc2d24e7b0251dada6462e715a8e2666b8ab43846cb6d25310.jpg)  
Figure 4. Performance of implementing SAC RL to optimize the distillation column internals design by using the reward Scheme 1 for a maximum of 500 iterations. The distillation column diameter was fixed at 6 feet. Ten runs were implemented, with each run having its unique RNG index. The following SAC RL model settings were used: $\tau = 0 . 0 5 , \gamma = 0 . 9 9 , \alpha = 0 . 2 ,$ , and replay buffer length = 50. <sup>owing</sup> <sup>SAC</sup> <sup>RL</sup> <sup>model</sup> <sup>setings</sup> <sup>were</sup> <sup>used:</sup> <sup>??</sup> <sup>=</sup> <sup>0.05</sup> <sup>,</sup> <sup>??</sup> <sup>=</sup> <sup>0.99</sup> <sup>,</sup> <sup>??</sup> <sup>=</sup> <sup>0.2</sup> <sup>,</sup> (A) SAC RL reward value, (B) column TOP section % flooding level, (C) column BOT section % lay buffer length = 50. (A) SAC RL reward value, (B) column TOP section % flooding levflooding level, and (D) runtime in seconds per SAC RL iteration. The associated actions (column internals design levels) predicted by the SAC RL model are shown in the next figure (Figure 5).

![](images/3fba18904ca3d21d62da76d7004f2ea49ea8dd51755a64aab273d8238d5d621d.jpg)  
(B)Action 2: TOP Tray Spacing, ft

(F) Action 6: BOT Downcomer Clearance, mm  
![](images/ab8e9bac0e45e304187f3ff03223911de3c5a36ea3fa9e6075d328d269f84ef2.jpg)

![](images/8944414911380c9115871905bc1849652faddce1304c82741e215341b50485dd.jpg)  
(C)Action 3: TOP Weir Height, mm

(G) Action 7: BOT Tray Spacing, ft  
![](images/d4abd57b7d5483eeaf1f00184fa8747f80a80bcadb956972dd7c8a6d97b38b43.jpg)

![](images/bb88a09587eff4e8ea7e09185b0d5fc0f484127f3f39efb8032558ab94e654a0.jpg)  
(D)Action 4: TOP Sieve Hole Diameter, mm

(H)Action 8: BOT Weir Height, mm  
![](images/c37f1d574aff7da1b4880de90cf7cbd0e15106d80cdf3daf86c9b040253cda1b.jpg)

![](images/4eb5125ab74a2cdf9a69c58c601d04e49d4ad9f94e02123b43448efa9fee4ea4.jpg)  
(E) Action5: TOP Weir Side Length, ft

(I) Action 9: BOT Sieve Hole Diameter, mm  
![](images/90b876c7bba81a79f68ffd2f14e878a4a0671fbc956db81208fad15a0df55876.jpg)

![](images/6984a88993881103a27bf9ca11a5359347cc9b94869e542fa3c5c837d771de71.jpg)

(J)Action 10: BOT Weir Side Length, ft  
![](images/57563e457a0741c07f86aad5e887b707d0b819e52d055ca57c05529e9fdc7318.jpg)  
Figure 5. Actions (column internals design levels) predicted by the SAC RL model as it learned the Figure 5. Actions (column internals design levels) predicted by the SAC RL model as it learned the best policies to optimize the reward. Ten runs were implemented, with each run having its unique best policies to optimize the reward. Ten runs were implemented, with each run having its unique RNG index. The following SAC RL model settings were used: $\tau = 0 . 0 5 , \gamma = 0 . 9 9 , \alpha = 0 . 2 ,$ and replay buffer length = 50. (A) Action 1 $( A _ { 1 } ) = \mathrm { T O P }$ downcomer clearance, (B) Action $2 \left( A _ { 2 } \right) = \mathrm { T O P }$ tray spacing, (C) Action $3 \left( A _ { 3 } \right) = \mathrm { T O P }$ weir height, (D) Action 4 $( A _ { 4 } ) = \mathrm { T O P }$ sieve hole diameter, (E) Action $5 \left( A _ { 5 } \right) = \mathrm { T O P }$ weir side length, (F) Action 6 $( A _ { 6 } ) = \mathrm { B O T }$ downcomer clearance, (G) Action 7 $( A _ { 7 } ) = \mathrm { B O T }$ tray spacing, (H) Action 8 $( A _ { 8 } ) = \mathrm { B O T }$ weir height, (I) Action 9 $( A _ { 9 } ) = \mathrm { B O T }$ sieve hole diameter, and (J) Action $1 0 \left( A _ { 1 0 } \right) = \mathrm { B O T }$ weir side length.

## 3.2. Effect of Reward Scheme on the Performance of SAC

There is evidence in the literature of RL algorithms that show the impact of the reward <sup>There</sup> <sup>is</sup> <sup>evidence</sup> <sup>in</sup> <sup>the</sup> <sup>literature</sup> <sup>of</sup> <sup>RL</sup> <sup>algorithms</sup> <sup>that</sup> <sup>show</sup> <sup>the</sup> <sup>impact</sup> <sup>of</sup> <sup>the</sup> <sup>re-</sup>scheme on the performance of RL models [29–31], and this was evaluated in the study. The <sup>ward</sup> <sup>scheme</sup> <sup>on</sup> <sup>the</sup> <sup>performance</sup> <sup>of</sup> <sup>RL</sup> <sup>models</sup> <sup>[29–31],</sup> <sup>and</sup> <sup>this</sup> <sup>was</sup> <sup>evaluated</sup> <sup>in</sup> <sup>the</sup> results of testing the effects of the reward schemes (Table 2) are summarized in Table 3. The <sup>study.</sup> <sup>The</sup> <sup>results</sup> <sup>of</sup> <sup>testing</sup> <sup>the</sup> <sup>efects</sup> <sup>of</sup> <sup>the</sup> <sup>reward</sup> <sup>schemes</sup> <sup>(Table</sup> <sup>2)</sup> <sup>are</sup> <sup>summarized</sup> data in Table 3 were calculated using the results of the last 100 iteration steps (iteration in Table 3. The data in Table 3 were calculated using the results of the last 100 iteration <sub>steps</sub> <sub>400</sub> <sub>to</sub> <sub>500)</sub> <sub>in</sub> <sub>each</sub> <sub>of</sub> <sub>the</sub> <sub>10</sub> <sub>runs.</sub> <sub>The</sub> <sub>idea</sub> <sub>behind</sub> <sub>this</sub> <sub>data</sub> <sub>analysis</sub> <sub>is</sub> <sub>that</sub> <sub>the</sub> steps (iteration steps 400 to 500) in each of the 10 runs. The idea behind this data analysis <sub>learning</sub> <sub>of</sub> <sub>the</sub> <sub>SAC</sub> <sub>RL</sub> <sub>model</sub> <sub>should</sub> <sub>be</sub> <sub>improving</sub> <sub>as</sub> <sub>it</sub> <sub>progresses</sub> <sub>in</sub> <sub>the</sub> <sub>sequence</sub> <sub>of</sub> <sup>is</sup> <sup>that</sup> <sup>the</sup> <sup>learning</sup> <sup>of</sup> <sup>the</sup> <sup>SAC</sup> <sup>RL</sup> <sup>model</sup> <sup>should</sup> <sup>be</sup> <sup>improving</sup> <sup>as</sup> <sup>it</sup> <sup>progresses</sup> <sup>in</sup> <sup>the</sup> iterations and that the best version of the model is achieved toward the end of the learning process. Expectation is based on the proven convergence of SAC RL as more training data are introduced to the model [13,14]. Therefore, computing the fraction of these last steps of iterations that meet the specific $f \%$ cut-off would be warranted. Furthermore, the number of steps considered was 100 because the replay buffer length for the SAC model used in these runs is 100.

Table 3. Summary of fractions of column design iterations that are below the cut-off values for % flooding values $f \%$ . The last 100 iteration steps of each of the 10 runs was used, i.e., 1000 samples per reward model. The following SAC RL model settings were used: τ = 0.05, γ = 0.99, α = 0.2, and replay buffer length = 100. Column diameter was fixed at 6 feet.

<table><tr><td>Reward Model</td><td>Column Section</td><td>Mean  $f_{\%}$ </td><td>Std. Dev. $f_{\%}$ </td><td>Fraction of  $f_{\%}<100$ </td><td>Fraction of  $f_{\%}<90$ </td><td>Fraction of  $f_{\%}<85$ </td><td>Fraction of  $f_{\%}<80$ </td></tr><tr><td rowspan="2">Scheme 1</td><td>TOP</td><td>88.5</td><td>8.7</td><td>0.955</td><td>0.826</td><td>0.251</td><td>0</td></tr><tr><td>BOT</td><td>88.8</td><td>8.6</td><td>0.945</td><td>0.796</td><td>0.216</td><td>0</td></tr><tr><td rowspan="2">Scheme 2</td><td>TOP</td><td>87.9</td><td>7.1</td><td>0.963</td><td>0.837</td><td>0.293</td><td>0</td></tr><tr><td>BOT</td><td>88.2</td><td>6.1</td><td>0.974</td><td>0.835</td><td>0.146</td><td>0</td></tr><tr><td rowspan="2">Scheme 3</td><td>TOP</td><td>87.7</td><td>6.0</td><td>0.971</td><td>0.841</td><td>0.29</td><td>0</td></tr><tr><td>BOT</td><td>88.0</td><td>5.6</td><td>0.976</td><td>0.823</td><td>0.214</td><td>0</td></tr><tr><td rowspan="2">Scheme 4</td><td>TOP</td><td>93.8</td><td>14.6</td><td>0.851</td><td>0.587</td><td>0.111</td><td>0</td></tr><tr><td>BOT</td><td>94.0</td><td>14.6</td><td>0.845</td><td>0.57</td><td>0.102</td><td>0</td></tr><tr><td rowspan="2">Scheme 5</td><td>TOP</td><td>101.9</td><td>21.8</td><td>0.686</td><td>0.371</td><td>0.067</td><td>0</td></tr><tr><td>BOT</td><td>102.4</td><td>22.0</td><td>0.686</td><td>0.359</td><td>0.052</td><td>0</td></tr></table>

Looking at the mean and standard deviation results (Table 3), it can be observed that only Schemes 1, 2, and 3 achieved $f _ { \% }$ values within the target interval [80%, 90%), while Scheme 4 and Scheme 5 produced $f \%$ values above this target range. This supports prior observations in RL models that the reward scheme affects the performance of the RL model, even the SAC RL model. An interesting comparison is between Scheme 3 and Scheme 4, because these two schemes are similar in terms of the binning intervals of the $f _ { \% }$ values but with Scheme 4 reward values 10 times lower than the Scheme 3 reward values. This means that the scaling of the reward values significantly affects the performance of the SAC RL model.

The consequence of the effects of the reward scheme is also evident in the fraction of the iterations that meet cut-off values of the design metric $f \%$ (Table 3). The best-performing reward schemes, Schemes 1, 2, and 3, predicted column design specifications with $f _ { \% } < 9 0$ around at least 0.80 fraction of the iterations. The poor-performing schemes predicted column design specifications with $f _ { \% } < 9 0$ at low fractions: \~0.50 for Scheme 4 and \~0.35 for Scheme 5. Scheme 5, which represents a “binary scheme”, is the worst reward scheme.

## 3.3. Effect of Column Diameter on the Performance of SAC

With the positive results of implementing SAC RL, as shown in the previous section, it was imperative to evaluate the limitation of the approach, and a direct way of accomplishing this is by varying the diameter of the column. This is because column flooding is inversely proportional to column diameter, as shown in the theory of distillation column design $[ 2 , 1 5 , 1 6 ]$ i.e., $f _ { \% } \sim { \frac { 1 } { \left( \mathrm { C o l u m n ~ D i a m e t e r } \right) ^ { 2 } } }$ . One set of experiments used fixed column diameter values of 5 ft, 6 ft, and 7 ft; another set included the column diameter $( A _ { 1 1 } )$ in the action space used by the SAC RL model. The results of implementing SAC RL in these experimental settings are shown in Figure 6.

(A) Column Diameter=5 feet: TOP Section  
![](images/c01ff434f1109b0d3e7467863ead85826944aec393679ecf02ad0635f07df6d9.jpg)  
(C) Column Diameter=6 feet: TOP Section

(B) Column Diameter=5 feet: BOT Section  
![](images/e240f1eaef93afa6a8305f5e72576e261f1af34f7fa547700d7de7dc5b8669c4.jpg)  
(D) Column Diameter=6 feet: BOT Section

![](images/64b3d8aff514ab98493177e889312638284582f210f02fb0e856b003ebb70d1e.jpg)  
(E) Column Diameter=7 feet: TOP Section

![](images/244538e81f099d8516f99ba6c87fcf4428f24d46c54fbf4f53fd0899acf79d8f.jpg)

![](images/098a1532f7b7666f69a198cd24863cafbe161b593e67bce05474e26f1288ef64.jpg)  
(G) Column Diameter=Action $( A _ { 1 1 } ) \colon$ TOP Section

(F) Column Diameter=7 feet: BOT Section  
![](images/a516bcf3f5966021beaf63db445065f4fbeb4238591ad9768cc9a100cfea75e4.jpg)

![](images/74e5edb7c28c74cf612f9cd6a0e4317a56ab199803153c4ee4fbc7da9ed978d2.jpg)

(H) Column Diameter= Action $( A _ { 1 1 } ) \colon$ BOT Section  
![](images/93d09ded9e61ca3add089a59c44914364aa7b541038664b5276bb4e1875f62d3.jpg)

(I) Action $\underline { { ( A _ { 1 1 } ) } } \colon$ Column Diameter, ft  
![](images/d569a2d327606172e5012b35c83dd8c49535415c6bc95b23fe2605e4e0215a72.jpg)  
Figure 6. Flooding results $f \%$ when SAC RL was implemented at varied column diameter levels. (A) TOP at 5 ft diameter, (B) BOT at 5 ft diameter, (C) TOP at 6 ft diameter, (D) BOT at 6 ft diameter, (E) TOP at 7 ft diameter, and (F) BOT at 7 ft diameter. When column diameter is in action space as $A _ { 1 1 }$ of SAC RL: (G) TOP flooding, (H) BOT flooding, and (I) column diameter predictions by SAC RL.

It can be seen in Figure 6 that the SAC RL converges in terms of its reward values and the resulting state variable values as model training progresses across all column diameter settings. In general, the flooding levels $f \%$ decrease as the column diameter increases. This is consistent with the theoretical relation that $f \%$ is inversely proportional to column diameter (relation shown above). When the column diameter is lowest, at 5 feet, the converged SAC RL model is just approaching 100% flooding from above (Figure $\mathsf { 6 A } , \mathsf { B } ) .$ with actual values reaching only \~110% flooding as the lower limit. This means that it is impossible to have a 5 ft column that can operate at the target [80%, 90%) range of $f _ { \% }$ This impossibility can be quickly checked by implementing the SAC RL, as was done here, instead of the user spending a very long time manually iterating for something that cannot happen (if column diameter is fixed at 5 ft).

At this point, it is imperative to ask the question, “If the flooding $f _ { \% }$ is inversely proportional to column diameter, then why not just keep on increasing the diamete $? ? ^ { \prime \prime }$ The answer comes from the limitations of fluid dynamics in the column when diameter is set too large. The issue of “weeping” can occur when there is a large pool of liquid on the tray (due to large column diameter) while vapor pressure is too low due to a larger active area of vapor–liquid contact resulting from a larger diameter [2,15]. This study does not cover solving the issue of weeping (it can be covered in a future extension of SAC RL in a similar approach). This then begs the question, “How was this trade-off in column diameter and flooding included in the SAC RL model?” This trade-off was included by implementing a specialized reward scheme by modifying reward Scheme 1 (Table 2) as follows (and let us call this Scheme 6 for quick referencing):

$$
r = \frac {1}{3} r _ {T O P} + \frac {1}{3} r _ {\mathrm{BOT}} + \frac {1}{3} d _ {d i a}\tag{2}
$$

where $d _ { d i a } = \frac { 1 0 0 } { \mathrm { m a x } ( \mathrm { C o l . D i a m a t e r } ) - \mathrm { m i n } ( \mathrm { C o l . D i a m a t e r } ) } [ \mathrm { m i n } ( \mathrm { C o l . D i a m e t e r } ) - A _ { 1 1 } ]$ , and $r _ { k }$ is the reward in section k of the column. For each k:

$$
r _ {k} = \left\{ \begin{array}{c} 1 0 0, 8 0 \leq f _ {\%} <   9 0 \\ 1 0 0 - 2 d _ {e r r}, f _ {\%} <   8 0 \\ 1 0 0 - 2 d _ {e r r}, f _ {\%} \geq 9 0 \end{array} \right.\tag{3}
$$

where $d _ { e r r } = | f _ { \% } - 8 5 |$

This reward Scheme 6 uses the maximum and minimum possible values of the column diameter, i.e., [4 ft, 9 ft] to compute a scaling factor adjusted to a magnitude of 100 as the maximum but assigned a negative value, i.e., [min(Col. Diameter) $- A _ { 1 1 } \big ] \in [ 0 , - \infty )$ , to penalize the reward r when the predicted action value for column diameter $A _ { 1 1 }$ becomes too large. In essence, $d _ { d i a } \in [ - 1 0 0 , 0 ]$ is a penalty value that becomes more negative as column diameter increases. Scheme 6 bounds reward r, resulting in the convergence of the SAC RL model, as shown in Figure 6G–I.

## 3.4. Effect of SAC Hyperparameters on Performance

Inherent in training RL models is the tuning of hyperparameters in order to further refine the models. This section covers the results of evaluating the effect of four hyperparameters: α, γ, τ, and replay buffer length. Of the numerous hyperparameters, these four have direct effects on SAC RL because of their roles in the fundamental equations of SAC. Table 4 summarizes the flooding results for the top section (TOP) of the column set at a fixed diameter of 6 feet. Even though data were also collected for the bottom section (BOT) of the column, space limitations in the manuscript prompted reducing the presented data to half; hence, Table 4 shows data for the TOP section.

The effect of hyperparameter α can be seen by first comparing the results between Settings 10 and 11 and between Settings 7 and 8 (Table 4). The general trend is that a lower α value of 0.1 will produce higher fractions of runs that fall within the target condition $f _ { \% } < 9 0$ compared with when α is set higher, at 0.5. However, when comparing between Settings 8 and $^ { 9 , }$ it can be seen that $\alpha = 0 . 1$ has a lower fraction $f _ { \% } < 9 0$ compared with $\alpha = 0 . 2$ . This means that there is an optimal value for α in the range 0.1 to 0.5, and this value may be close to 0.2. Since the value of α is a measure of the fraction of entropy contribution in the loss function (see Equation (1)), this observed trend means that the term in the loss function for maximizing entropy should not be zero, but it also should not be a very large part of the loss function. This is just a fitting observation because this entropy term is a unique feature of the SAC in comparison with other RL algorithms implementing actor-critic networks [13,23].

Table 4. Summary of fractions of column design iterations that are below the cut-off value $f _ { \% } = 9 0$ for flooding values $f _ { \% }$ . The last 100 iteration steps of each of the 10 runs were used, i.e., 1000 samples per reward model. The runs implemented were for a column diameter = 6 feet and reward model Scheme 1.

<table><tr><td rowspan="2">SAC Setting</td><td colspan="4">Hyperparameter</td><td rowspan="2">Fraction of  $f_{\%}<90$ </td><td rowspan="2">Mean  $f_{\%}$ </td><td rowspan="2">Std. Dev.  $f_{\%}$ </td></tr><tr><td>α</td><td>τ</td><td>γ</td><td>Replay Buffer Length</td></tr><tr><td>1</td><td>0.2</td><td>0.05</td><td>0.99</td><td>50</td><td>0.945</td><td>85.3</td><td>4.0</td></tr><tr><td>2</td><td>0.2</td><td>0.05</td><td>0.9</td><td>50</td><td>0.918</td><td>85.9</td><td>3.6</td></tr><tr><td>3</td><td>0.5</td><td>0.05</td><td>0.9</td><td>100</td><td>0.881</td><td>86.8</td><td>5.3</td></tr><tr><td>4</td><td>0.2</td><td>0.05</td><td>0.9</td><td>100</td><td>0.886</td><td>86.8</td><td>6.3</td></tr><tr><td>5</td><td>0.2</td><td>0.05</td><td>0.99</td><td>100</td><td>0.826</td><td>88.5</td><td>8.7</td></tr><tr><td>6</td><td>0.2</td><td>0.01</td><td>0.99</td><td>50</td><td>0.978</td><td>84.8</td><td>2.1</td></tr><tr><td>7</td><td>0.5</td><td>0.01</td><td>0.9</td><td>50</td><td>0.865</td><td>86.8</td><td>5.2</td></tr><tr><td>8</td><td>0.1</td><td>0.01</td><td>0.9</td><td>50</td><td>0.881</td><td>86.7</td><td>5.1</td></tr><tr><td>9</td><td>0.2</td><td>0.01</td><td>0.9</td><td>50</td><td>0.938</td><td>85.4</td><td>3.2</td></tr><tr><td>10</td><td>0.5</td><td>0.01</td><td>0.9</td><td>100</td><td>0.916</td><td>86.5</td><td>5.3</td></tr><tr><td>11</td><td>0.1</td><td>0.01</td><td>0.9</td><td>100</td><td>0.947</td><td>85.8</td><td>3.2</td></tr><tr><td>12</td><td>0.2</td><td>0.01</td><td>0.9</td><td>100</td><td>0.932</td><td>86.4</td><td>6.1</td></tr><tr><td>13</td><td>0.2</td><td>0.01</td><td>0.99</td><td>100</td><td>0.870</td><td>87.5</td><td>7.7</td></tr></table>

The effect of hyperparameter τ can be seen by first comparing the results between Settings 3 and 10 and between Settings 4 and 12 (Table 4). Both pairs show that a lower value of τ results in a higher fraction $f _ { \% } < 9 0$ . If there is any optimum value somewhere, these data pairs cannot support such a possibility—perhaps a more extensive scan of the hyperparameter space can discover an optimum, but this is beyond the scope of this study. Since hyperparameter τ represents the smoothing coefficient for the target Q-functions, setting its value to zero may not be a good idea, because this number controls the update of the Q-function deep neural network weights.

The effect of hyperparameter $\gamma$ can be seen by first comparing the results between Settings 1 and 2, between Settings 4 and 5, and between Settings 12 and 13 (Table 4). It can be seen in these pairs that the trends for fraction $f _ { \% } < 9 0$ do not follow a clear, consistent trend. It is possible that there are confounding effects in the SAC RL training that cannot be easily isolated to be the sole effect of $\gamma .$ . Therefore, there is no conclusive trend for the effect of this hyperparameter based on the data.

The effect of the hyperparameter ‘replay buffer length’ can be seen by first comparing results between Settings 2 and $^ { 4 , }$ between Settings 6 and $^ { 1 3 , }$ and between Settings 8 and 11 (Table 4). It can be seen in these pairs that the trends for fraction $f _ { \% } < 9 0$ do not follow a clear, consistent trend. Similar to hyperparameter $\gamma ,$ it is possible that there are confounding effects in the SAC RL training that cannot be easily isolated to be the sole effect of ‘replay buffer length’. Therefore, there is no conclusive trend for the effect of this hyperparameter based on the data.

## 3.5. Comparison of SAC RL Algorithm with Established Optimization Algorithms

A pertinent question at this point is, “How does SAC RL performance compare with established optimization algorithms?” This question is necessary to answer, because this study is proposing to use SAC RL in an optimization task that has been heavily studied [34]. The tested algorithms are the following: Nelder–Mead [35], Broyden–Fletcher– Goldfarb–Shanno (BFGS) [36], Sequential Least Squares Programming (SLSQP) [37], dual annealing [38], and simplicial homology global optimization (SHGO) [39], which were implemented using their pre-built functions in the Python module SciPy [40]. To simplify the comparative analysis, the target flooding of $f _ { \% } = 8 5$ was used, which was based on reward Scheme 1 used for the SAC RL. Each algorithm was implemented in ten runtimes, with each run assigned a unique RNG index. Figure 7 summarizes the results from experiments grouped into (1) at a fixed column diameter of 6 ft, as shown in Figure 7A−F; and (2) at variable column diameters with a penalty on the diameter the same as that of18 of 21 Scheme 6, as shown in Figure 7G,H.

Fixed Column Diameter = 6 ft  
![](images/32f6dc0798bbac9eebdf4017141a665ec6a5e6e799f07332fefc7efa29f7a175.jpg)  
(C) BFGS Algorithm

![](images/a602dfaee76aa2409cbc5d28536f6294172928f73c5bf08c5e2df5d17ce6d995.jpg)

![](images/f91df7636e490f6f27b96b853b5825af5b98894255d35aa5d1d52ab51774a98a.jpg)

(D)SLSQP Algorithm  
![](images/a85df3bad073f37fb9061926928b76f907f15cc0fb76b3f30e2723ecc6a91cc5.jpg)

(E) Dual Annealing Algorithm  
![](images/3c4e1d4b5d5c4e00a8a1c2987e3eb6e8b9a788886e411c2e0d205034af3d01d7.jpg)

(F) SHGO Algorithm  
![](images/e257a4a75161db7eacd772fbb359315f15e920bad683d8fe6690a5471479e068.jpg)

Variable Column Diameter  
![](images/623e4c078bdf0195b83e8dd5d345f01e2214c06b021460742674859d9cb1281e.jpg)

![](images/58df4a9d97255870a7b457c78aeefc60f23f879e0ae93b8ff4651a9fb466e68e.jpg)  
Figure 7. Comparing the performance of the SAC-RL algorithm with that of established optimiza-Figure 7. Comparing the performance of the SAC-RL algorithm with that of established optimization tion algorithms using reward Scheme 1 as the basis for the target of optimization to, at most, 85% algorithms using reward Scheme 1 as the basis for the target of optimization to, at most, 85% approach to flooding. Ten runs were implemented for each algorithm, with each run having a unique RNG index. With a fixed distillation column diameter of 6 ft: (A) via SAC-RL, (B) via Nelder–Mead, (C) via BFGS, (D) via SLSQP, (E) via dual annealing, and (F) via SHGO. With variable distillation column diameter of in the range 4 ft to 9 ft: (G) via SAC-RL and (H) via Nelder–Mead.

It can be seen that each of the algorithms tested exhibited unique performance in terms of optimizing to 85% flooding. Poor performance can be seen for BFGS and SLSQP, which exhibited early stops (Figure 7C,D), while dual annealing and SHGO completed the runs but with erratic behavior in various stages of the iterations (Figure 7E,F). The Nelder–Mead exhibited a smooth convergence to the target when the column diameter was fixed (Figure 7B). However, Nelder–Mead failed to converge to the target when the column diameter was included in the varied parameters (Figure 7H). We also want to note that dual annealing and SHGO were also tested when the column diameter was varied, but these algorithms exhibited erratic performance and failed to converge to the target. The SAC-RL algorithm was able to converge to the target in both the fixed column diameter (Figure 7A) and variable column diameter (Figure 7G) design scenarios. These trends show the advantage of SAC RL over established optimization algorithms for the task of optimizing distillation column internals design. Due to SAC RL’s maximum entropy term, it continues to explore as it learns, resulting in the prevention of exploitation of actions that may be sub-optimal.

## 4. Conclusions

This study focused on developing an automated method using soft actor-critic (SAC) reinforcement leaning for optimizing the design of distillation columns, and this aim was achieved, as supported by the results. The integration of SAC RL into AspenPlus<sup>®</sup> model optimization was achieved by using Python codes written to allow a cyclic interaction between the distillation column model AspenPlus<sup>®</sup> and the SAC model in PyTorch. The specific chemical system used as a case study was a binary mixture of ethanol and water, and the RadFrac module in AspenPlus<sup>®</sup> was used to enable rigorous column internals design. The results clearly support the following findings: (1) SAC RL works as an automation approach for the design of distillation column internals, (2) the reward scheme in the SAC model significantly affects SAC performance, (3) column diameter is a significant constraint in achieving column internals design specifications in flooding, and (4) SAC hyperparameters have varying effects on SAC performance.

This study also demonstrated that an untrained SAC RL model can quickly learn how to optimize the design of a distillation column. This means that the algorithm can be implemented as a one-shot RL for column internals design. This has significant implications for possible future implementation of the technique as an integral part of computer-based chemical process modeling, e.g., if this technique becomes part of AspenPlus<sup>®</sup> software (and other similar computer packages). That is, there is no need to install a pre-trained SAC RL model into a computer-based chemical process simulation package that would utilize this technique.

Author Contributions: Conceptualization, D.L.B.F., H.B., R.W., C.B. and A.P.M.; methodology, D.L.B.F. and H.B.; software, D.L.B.F. and H.B.; formal analysis, DLBF, A.P.M., M.A.B. and M.E.Z.; investigation, D.L.B.F., H.B., R.W. and C.B.; resources, M.A.B. and M.E.Z.; data curation, D.L.B.F. and H.B.; writing, D.L.B.F., H.B. and A.P.M.; project administration, D.L.B.F.; funding acquisition, D.L.B.F. and M.E.Z. All authors have read and agreed to the published version of the manuscript.

Funding: This research was funded by two grants: (1) LURA Grant by the Louisiana Space Grant Consortium (LaSPACE) with subaward number PO-0000277305 under primary NASA grant number 80NSSC20M0110, and (2) LaSSO Grant by the Louisiana Space Grant Consortium (LaSPACE) and Louisiana Sea Grant (LSG) with subaward number PO-0000277330 under primary NASA grant number 80NSSC20M0110.

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: All Python codes, Jupyter Notebook files, AspenPlus<sup>®</sup> model files, sample raw data in spreadsheet, and sample graphics used in the paper are archived online in the project GitHub repository [41]: https://github.com/dhanfort/aspenRL.git (Access date: 16 February 2025). This is an open-access repository under MIT License. The reader is encouraged to contact the corresponding author if there is a need for more information beyond the open-access materials.

Acknowledgments: We acknowledge the support of the staff of the Energy Institute of Louisiana (EIL): Sheila Holmes and Bill Holmes. EIL facilities were instrumental in completing this work.

Conflicts of Interest: The authors declare no conflicts of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.

## References

1. Seidel, T.; Biegler, L.T. Distillation column optimization: A formal method using stage-to stage computations and distributed streams. Chem. Eng. Sci. 2025, 302, 120875. [CrossRef]

2. Seader, J.D.; Henley, E.J.; Roper, D.K. Separation Process Principles: With Applications Using Process Simulators, 4th ed.; Wiley: New York, NY, USA, 2016.

3. Al-Malah, K.I.M. Aspen Plus: Chemical Engineering Applications, 2nd ed.; Wiley: New York, NY, USA, 2022.

4. Haydary, J. Chemical Process Design and Simulation: Aspen Plus and Aspen Hysys Applications; Wiley: New York, NY, USA, 2019.

5. AspenTech. AspenPlus. Available online: https://www.aspentech.com/en (accessed on 19 January 2025).

Bao, J.; Gao, B.; Wu, X.; Yoshimoto, M.; Nakao, K. Simulation of industrial catalytic-distillation process for production of methyl tert-butyl ether by developing user’s model on Aspen plus platform. Chem. Eng. J. 2002, 90, 253–266. [CrossRef]

7. Kamkeng, A.D.N.; Wang, M. Technical analysis of the modified Fischer-Tropsch synthesis process for direct CO2 conversion into gasoline fuel: Performance improvement via ex-situ water removal. Chem. Eng. J. 2023, 462, 142048. [CrossRef]

8. Syauqi, A.; Kim, H.; Lim, H. Optimizing olefin purification: An artificial intelligence-based process-conscious PI controller tuning for double dividing wall column distillation. Chem. Eng. J. 2024, 500, 156645. [CrossRef]

9. Byun, M.; Lee, H.; Choe, C.; Cheon, S.; Lim, H. Machine learning based predictive model for methanol steam reforming with technical, environmental, and economic perspectives. Chem. Eng. J. 2021, 426, 131639. [CrossRef]

10. Schefflan, R. Teach Yourself the Basics of Aspen Plus, 2nd ed.; Wiley: New York, NY, USA, 2016.

11. Agarwal, R.K.; Shao, Y. Process Simulations and Techno-Economic Analysis with Aspen Plus. In Modeling and Simulation of Fluidized Bed Reactors for Chemical Looping Combustion; Agarwal, R.K., Shao, Y., Eds.; Springer International Publishing: Cham, Switzerland, 2024; pp. 17–73. [CrossRef]

12. Chen, Q. The Application of Process Simulation Software of Aspen Plus Chemical Engineering in the Design of Distillation Column. In Proceedings of the Cyber Security Intelligence and Analytics, Haikou, China, 28–29 February 2020; pp. 618–622.

13. Haarnoja, T.; Zhou, A.; Abbeel, P.; Levine, S. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. arXiv 2018, arXiv:1801.01290. [CrossRef]

14. Haarnoja, T.; Zhou, A.; Hartikainen, K.; Tucker, G.; Ha, S.; Tan, J.; Kumar, V.; Zhu, H.; Gupta, A.; Abbeel, P.; et al. Soft Actor-Critic Algorithms and Applications. arXiv 2018, arXiv:1812.05905.

15. Wankat, P. Separation Process Engineering—Includes Mass Transfer Analysis, 3rd ed.; Prentice Hall: New York, NY, USA, 2012.

16. Kister, H. Distillation Design; McGraw-Hill: Boston, MA, USA, 1992.

17. Taqvi, S.A.; Tufa, L.D.; Muhadizir, S. Optimization and Dynamics of Distillation Column Using Aspen Plus<sup>®</sup>. Procedia Eng. 2016, 148, 978–984. [CrossRef]

18. Tapia, J.F.D. Chapter 16—Basics of process simulation with Aspen Plus\*. In Chemical Engineering Process Simulation, 2nd ed.; Foo, D.C.Y., Ed.; Elsevier: Amsterdam, The Netherlands, 2023; pp. 343–360. [CrossRef]

19. Hammond, M. pywin32. Available online: https://pypi.org/project/pywin32/ (accessed on 9 January 2025).

20. OpenAI. Soft Actor-Critic. Available online: https://spinningup.openai.com/en/latest/algorithms/sac.html (accessed on 9 January 2025).

21. McCabe, W.L.; Thiele, E.W. Graphical Design of Fractionating Columns. Ind. Eng. Chem. 1925, 17, 605–611. [CrossRef]

22. Jones, E.; Mellborn, M. Fractionating column economics. In Chemical Engineering Progress (CEP); AIChE: New York, NY, USA, 1982; pp. 52–55.

23. Liu, J.; Guo, Q.; Zhang, J.; Diao, R.; Xu, G. Perspectives on Soft Actor–Critic (SAC)-Aided Operational Control Strategies for Modern Power Systems with Growing Stochastics and Dynamics. Appl. Sci. 2025, 15, 900. [CrossRef]

24. Lillicrap, T.P.; Hunt, J.J.; Pritzel, A.; Heess, N.; Erez, T.; Tassa, Y.; Silver, D.; Wierstra, D. Continuous control with deep reinforcement learning. arXiv 2015, arXiv:1509.02971. [CrossRef]

25. Towers, M.; Kwiatkowski, A.; Terry, J.; Balis, J.U.; De Cola, G.; Deleu, T.; Goulão, M.; Kallinteris, A.; Krimmel, M.; Kg, A.; et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv 2024, arXiv:2407.17032. [CrossRef]

26. Brockman, G.; Cheung, V.; Pettersson, L.; Schneider, J.; Schulman, J.; Tang, J.; Zaremba, W. OpenAI Gym. arXiv 2016, arXiv:1606.01540.

27. Zhang, X.; Mao, W.; Mowlavi, S.; Benosman, M.; Ba¸sar, T. Controlgym: Large-Scale Control Environments for Benchmarking Reinforcement Learning Algorithms. arXiv 2023, arXiv:2311.18736. [CrossRef]

28. Yoon, C. GitHub Repo: Policy-Gradient-Methods. Available online: https://github.com/cyoon1729/Policy-Gradient-Methods. git (accessed on 19 January 2025).

29. Nath, A.; Oveisi, A.; Pal, A.K.; Nestorovi´c, T. Exploring reward shaping in discrete and continuous action spaces: A deep reinforcement learning study on Turtlebot3. PAMM 2024, 24, e202400169. [CrossRef]

30. Viswanadhapalli, J.K.; Elumalai, V.K.; Shivram, S.; Shah, S.; Mahajan, D. Deep reinforcement learning with reward shaping for tracking control and vibration suppression of flexible link manipulator. Appl. Soft Comput. 2024, 152, 110756. [CrossRef]

31. Veviurko, G.; Böhmer, W.; de Weerdt, M. To the Max: Reinventing Reward in Reinforcement Learning. arXiv 2024, arXiv:2402.01361. [CrossRef]

32. Dayal, A.; Cenkeramaddi, L.R.; Jha, A. Reward criteria impact on the performance of reinforcement learning agent for autonomous navigation. Appl. Soft Comput. 2022, 126, 109241. [CrossRef]

33. Anaconda. Anaconda: The Operating System for AI. Available online: https://www.anaconda.com/ (accessed on 5 February 2025).

34. Kochenderfer, M.J.; Wheeler, T.A. Algorithms for Optimization; The MIT Press: Boston, MA, USA, 2019.

35. Gao, F.; Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters. Comput. Optim. Appl. 2012, 51, 259–277. [CrossRef]

36. Fletcher, R. Practical Methods of Optimization; John Wiley & Sons, Ltd.: New York, NY, USA, 1987.

37. Kraft, D.; Dfvlr, F.B. A Software Package for Sequential Quadratic Programming; DFVLR: Berlin, Germany, 1988.

38. Xiang, Y.; Sun, D.Y.; Fan, W.; Gong, X.G. Generalized simulated annealing algorithm and its application to the Thomson model. Phys. Lett. A 1997, 233, 216–220. [CrossRef]

39. Endres, S.C.; Sandrock, C.; Focke, W.W. A simplicial homology algorithm for Lipschitz optimisation. J. Glob. Optim. 2018, 72, 181–217. [CrossRef]

40. Virtanen, P.; Gommers, R.; Oliphant, T.E.; Haberland, M.; Reddy, T.; Cournapeau, D.; Burovski, E.; Peterson, P.; Weckesser, W.; Bright, J.; et al. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nat. Methods 2020, 17, 261–272. [CrossRef] [PubMed]

41. Fortela, D.L.B. aspenRL: Enhanced AspenPlus-based Multi-stage Distillation Design Using SAC Reinforcement Learning. Available online: https://github.com/dhanfort/aspenRL (accessed on 19 January 2025).

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.