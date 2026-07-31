# Computing operation procedures for chemical plants using whole-plant simulation models

![](images/6683531f16ede1c94152e796013d0e677e777f87c9adfce0ee6e8f0345244846.jpg)

Shumpei Kubosawa <sup>a,b,∗</sup>, Takashi Onishi <sup>a,b</sup>, Yoshimasa Tsuruoka <sup>a,c</sup>

<sup>a</sup> NEC-AIST AI Cooperative Research Laboratory, AIST, Aomi 2-4-7, Koto-ku, Tokyo, 135-0064, Japan

<sup>b</sup> Data Science Research Laboratories, NEC Corporation, Shimonumabe 1753, Nakahara-ku, Kawasaki, Kanagawa 216-8666, Japan

<sup>c</sup> Department of Information and Communication Engineering, The University of Tokyo, Hongo 7-3-1, Bunkyo-ku, Tokyo 113-8656, Japan

## A R T I C L E I N F O

Keywords: Chemical plant operation Procedure optimisation Reinforcement learning Dynamic simulation System identification

## A B S T R A C T

Chemical plants are complex dynamical systems. Optimising plant operation for non-stationary scenarios, such as changing the output product and recovering from abrupt disturbances, is challenging because a chemical plant has many operation points and complex responses. A plant simulator can be used to compute the optimal procedures. However, because of modelling errors or contingent changes in the external conditions, such as weather and feed purity, there exist gaps between the behaviour of a simulator and that of a real plant. This poses another challenge in a simulator-based approach, which adds to the computational complexity of the problem. In this study, we propose a simulator-based approach for optimising chemical plant operations using deep reinforcement learning and knowledge-based automated reasoning. Specifically, a reinforcement learning agent is trained on a whole-plant simulator with a policy gradient algorithm, using automated reasoning to narrow down the action space of the agent. To maintain the optimality of the procedures in a real plant, a simple method for the state and parameter estimation of the system at run time is introduced. This method can improve the accuracy of the response prediction model (i.e. the plant simulator) on which the agent depends. The presented method is evaluated on a real chemical distillation plant. The experimental results indicate that the proposed approach consumed only half the time and steam (heat energy) in comparison with that in the case of human-emulated procedures.

## 1. Introduction

Chemical plants are complex dynamical systems comprising several components whose state transitions are often nonlinear and dependent on various factors such as past states, disturbances and operation procedures. Plants are typically operated by highly experienced human operators, based on standard chemical engineering knowledge, a plant structure representation known as a piping and instrumentation dia gram (P&ID), and manuals that describe standard operation procedures. The development of intelligent systems that can support these human operators is becoming increasingly important in the chemical industry because the implementation can significantly improve the safety and efficiency of plant operations.

In this study, as a step towards building such an intelligent system, we developed a system that can compute operation procedures for non stationary operations, such as varying the load (amount of production) or product specification (e.g. purity), and starting up the plant. The plant operator provides the operation objective (e.g. change the product purity from 99.5% to 95%) to the system, and the system outputs the corresponding procedure, which consists of a time series of the actual set-point values of each proportional–integral–derivative (PID)

controller. The proposed system is based on a whole-plant simulator and on deep reinforcement learning, which has recently been applied to control problems in various domains (Gu et al., 2017; Lillicrap et al., 2016). Although deep reinforcement learning has proven to be successful in solving many control problems, establishing appropriate operation procedures for chemical plants is a highly challenging because of the vast search space of the possible operation procedures that results from continuous manipulation of the numerous variables in a plant. To narrow down this search space, the proposed system uses the qualitative knowledge employed by skilled operators and focuses the search on potential operation procedures. In addition, it provides human operators with a procedure and the predicted plant response using the simulator to guide them in deciding whether or not to adopt that procedure.

The proposed approach differs from existing studies on the automatic generation of plant operation procedures in certain important aspects. Some previous studies (Gabbar et al., 2004; Gofuku et al., 2004) focused on identifying the instruments to manipulate and the order in which they should be manipulated; therefore, they employed functional or qualitative models. To generate procedures with nu merical control values, the proposed system employs a quantitative and dynamic model for simulation in addition to qualitative models. Dynamic simulation models for entire chemical plants have been used for training human operators (Kano & Ogawa, 2010; Patle et al., 2014); and not for identifying appropriate operation procedures automatically. Model predictive control (MPC) is a well-known method for process control and has long been studied in the field of chemical engineering (Eaton & Rawlings, 1991). It has proven to be an effec tive approach for optimising control sequences for particular chemica processes (Forbes et al., 2015; Qin & Badgwell, 2003). Dynamic matrix control (DMC) (Cutler & Ramaker, 1980) is one of the most common and practical MPC approaches. DMC systems are based on linear models (i.e. step response models), which are developed and tuned using the recoded step response data of an operational plant. Owing to its ‘linear’ nature (utilising a dynamic ‘matrix’), DMC cannot utilise elaborated nonlinear models, such as plant simulators, which are based on chemical engineering knowledge and the detailed information of a plant structure (e.g. P&IDs). In addition, testing the automatically generated operation procedures is not preferred because of the risk of unstable production that may be caused during the testing. Moreover, DMC cannot optimise procedures in situations that are significantly different from the situations employed to collect step response data, i.e. unexperienced and unmodelled conditions such as unusual steady states and nonstationary states. Therefore, DMC is unsuitable for sup porting transition operations that involve nonstationary scenarios. To improve robustness, i.e. to maintain the optimality on a real plant even in unideal situations for the simulator (e.g. existence of disturbances, modelling errors, and ageing), the proposed approach employs a simple search method for online identification of the simulation parameters and states. Using this method, the proposed system repeats the normal forward simulations for the search without considering the simulation mechanisms (e.g. differential equations) on which conventional MPC methods depend.

The proposed system generates an operation procedure using automated reasoning, deep reinforcement learning, and a dynamic simu lator that is synchronised with a plant online. The effectiveness of the proposed approach is evaluated on a real plant for chemical distillation, and it is demonstrated that it can generate a procedure that achieves a significantly faster transition from one steady state to another target steady state in comparison with human-generated procedures.

The contributions of this study are three-fold: (i) it introduces a novel piecewise linear neural network suitable for the chemical engineering domain as the policy function in reinforcement learning, (ii) it proposes an architecture that connects an actual plant and the learning agent, and finally, (iii) it presents the experiments conducted on a real chemical plant for the performance evaluation of the proposed system.

## 2. Background

## 2.1. Reinforcement learning

Reinforcement learning (Sutton & Barto, 1998) is a branch of ma chine learning that studies how an agent can acquire action sequences that maximise the long-term reward given by an environment. Rein forcement learning can optimise procedures in an environment defined as a partially observable Markov decision process wherein some in ternal hidden states may be unobservable by the learning algorithm. Unlike classical logic-based planning, reinforcement learning covers continuous control problems and has been utilised for parameter optimisation in PID controllers (Sedighizadeh & Rezazadeh, 2008). In addition, in recent years, reinforcement learning has gained significant interest by defeating human experts in many domains, such as the game of Go (Silver et al., 2017). This can be attributed to its ability to deal with delayed reward settings, e.g. while training a game-playing agent, the final outcome is unknown until the game ends. This characteristic is suitable for chemical plant control because chemical plants are commonly modelled as combinations of first-order systems with dead-time processes. Specifically, it may take a long time to change a plant state to the desired final state (e.g. steady state) from a distant initial state (e.g. shut-down state). Utilising this feature, chemical plant control systems can be developed by simply specifying the desired final state as the rewarding state in reinforcement learning without considering the detailed transient states.

Proximal policy optimisation (PPO) (Schulman et al., 2017) is a relatively recent deep reinforcement learning algorithm and has achieved state-of-the-art performance on a virtual robot control problem on physical simulators. PPO is a variant of policy gradient methods. A policy is a parametrised mapping from an observation space (e.g. a set of states or sensor vectors) to an action space (e.g. a set of vectors consisting of the set-point values (SVs) of PID controllers) and typically expressed as a conditional probability,

$$
P (a _ {t} | s _ {t}) = \pi_ {\theta} (a _ {t} | s _ {t}),\tag{1}
$$

where $\theta$ is the set of parameters, $\pi$ is the policy distribution function, and $a _ { t }$ and $s _ { t }$ are the action and the state at time $t ,$ respectively. If the action, $a _ { t } ,$ which is sampled from the probability distribution of Eq. (1) at state $s _ { t }$ earns a high reward from the environment, the policy gradient methods increase the probability of action $a _ { t }$ by changing the parameter set, $\theta ,$ in the direction of either

$$
g = \Psi \nabla \pi_ {\theta} (a _ {t} | s _ {t}),\tag{2}
$$

where $\psi \in \mathbb R$ may be the total reward at each time $r _ { t } ,$

$$
R = \sum_ {t = 0} ^ {\infty} r _ {t},\tag{3}
$$

or other reward-related values (Schulman et ${ \mathrm { a l . , } }$ 2018). The gradient vector, $\nabla \pi _ { \theta } ( a _ { t } | s _ { t } )$ , points to the steepest direction of the probability value in the parameter space. Consequently, if action $a _ { t }$ at state $s _ { t }$ earns a high reward, and the policy parameters are updated in the direction of Eq. (2), the probability value in Eq. (1) can be increased. As a result, the sampling probability of the action increases. PPO is based on this policy gradient method and has additional constraints on the extent of parameter changes in each update to avoid the instability problem, which refers to the significant changes in the behaviour while updating the parameters.

## 2.2. Dynamic simulation

## Plant simulator

Dynamic simulation is conducted for reproducing the quantitative responses of a dynamical system utilising mathematical representations of physical models. It has been employed for training human operators (Kano & Ogawa, 2010; Patle et al., 2014) and optimising operation procedures of industrial process plants (Kvamsdal et al., 1999; Shirakawa et al., 2005). Examples of such simulators include the Tennessee Eastman process (TEP) model (Downs & Vogel, 1993) and the vinyl acetate monomer (VAM) plant model (Machida et al., 2016).<sup>1</sup> Both models were developed to study and evaluate technologies that help in solving plant-wide and multivariate control problems.

The TEP model is based on an actual industrial process of the Eastman Chemical Company. To protect the proprietary information, the physical properties of the components and the model parameters are virtual. The model involves eight ‘virtual’ chemical components (four materials, two products, one byproduct, and one impurity of the feeds) and three major processes (exothermic reaction, separation, and recycling). The separation process comprises two units (a vapour–liquid separator and a stripper). The process is measured using 41 sensors (19 analysers with time delay and a low sampling rate, and 22 continuous measurements) and is manipulated by 12 variables. The control objectives are (1) maintaining process variables at the given values, (2) maintaining operating conditions based on the equipment constraints, (3) stabilising the product rate and quality during disturbances, (4) removing fluctuation in the process values, which may affect other processes and (5) rapidly and smoothly performing transition operations, including recovery from disturbances. The model considers 15 disturbance scenarios, including change in feed composition, change in cooling water temperature, and reaction kinetics. Furthermore, depend ing on the scenario, the disturbances are implemented as step changes, random changes, slow drifts, and sticking.

The VAM plant model is also developed as a benchmark problem for the study of process control and process data analysis. The model involves six chemical components (three materials, one product, and two byproducts) and three major processes as in the case of the TEP model. Additionally, plant scale is larger than that of the TEP model. The VAM plant model is equipped with a total of 10 heat exchangers, and the separation process involves four units (a liquid–vapour separator, an absorber, a distillation column, and a decanter). Thus, the fluctuation of a process variable can easily trigger a deviation from the operation conditions. The process is measured by 107 sensors including 31 analysers and controlled by 45 PID controllers, 31 valves, four pumps, and one compressor. The model includes 19 malfunction scenarios, which include reduction in the reaction activity, change in the heat transmission of a unit, stopping of pumps, and failure of sensors and 11 disturbance scenarios similar to the TEP model. In addition to the abundance of irregular scenarios, the VAM plant model covers starting-up and shutting-down operations.

## Online simulator

To predict the response of a real plant, the initial state of a simulation should be the current real plant state. Therefore, a method is required for online state estimation to generate the initial state of the simulator, including unobservable states (e.g. amount of substance in each pipe) based on observable states (e.g. process values of the PID controllers, such as the flowrate, pressure, and temperature) of the rea plant. In this study, an online simulation method proposed by Nakaya et al. (2006) was employed for this purpose.

Using this method, the SVs (i.e. target values of the PID controllers) of the real plant are input to the simulator simultaneously, and the response is calculated in real time. Even if the initial state of the online simulator is different from the real state, the simulation states approach the real state with time because if the simulator is appropriately modelled, both the real and the simulated systems (or system states) respond similarly. Moreover, if the similarity between the observable states increases, the similarity between the unobservable states also increases.

## 3. Proposed method

In this study, we developed a system that can automatically com pute optimal operation procedures for chemical plants. This system is designed to assist human operators who are controlling a plant to ensure it reaches a desirable state. The proposed system uses both qualitative and quantitative (dynamic simulation) models of a plant and outputs operation procedures that include actual manipulation of values obtained from automatic optimisation of specified objective functions, also called reward functions.

This developed system comprises four key components: qualitative reasoning, reinforcement learning, controller agent, and system identi fication. In the preparation phase, the qualitative reasoning component specifies the instruments to be manipulated, and the reinforcement learning agent determines the extent of manipulation at each step of the sequence through interaction with the simulator. After the interactive training, the agent is used as the controller of the plant in the control phase. The human operator inputs the desired state of the plant to the controller, which then calculates the manipulation values based on the current plant state, which is estimated online by the system identification component. The system identifier continuously estimates the plant state to minimise the difference between the simulation and real plant states.

The process flow is depicted in Fig. 1 and is summarised as follows:

1. (Preparation phase) First, the user of the system prepares a P&ID of the target plant, target qualitative state (e.g. ‘increase product flow’) and reward function that defines the numerical operating conditions.

2. From the knowledge base converted from the P&ID and the qualitative state, the reasoning engine (a modified version of Phillip (Yamamoto et al., 2015)) specifies the set of instruments (e.g. valves) to be manipulated.

3. The reinforcement learning agent is configured to use the set of instruments as action points in addition to the prepared reward function. To implement the agent, we utilised the ChainerRL (Fujita et al., 2019) deep reinforcement learning library, which is coded in Python. To connect the agent and the dynamic simulator (e.g. the VAM plant model (Machida et al., 2016)), we developed the custom environment, which is compliant with OpenAI Gym (Brockman et al., 2016) interfaces.

4. At this stage, the agent starts training. It hypothesises an oper ation procedure, tests it on the simulator, and finally, receives sensor values as its response. Furthermore, it considers a plan to be better than other plans if it yields greater rewards calculated from the observed sensor values. The agent repeats this process until a certain termination condition is satisfied, e.g. threshold of the cumulative reward. While training, the simulator and the real plant are not required to be connected.

5. (Control phase) The trained agent is copied to the controller of the real plant. Subsequently, the developed system begins to synchronise the states of the real plant and the states of the simulator by periodical searches for the optimal simulation parameters.

6. The user inputs a desired quantitative state (e.g. target product flow) to the controller, and subsequently, the agent outputs an optimal procedure using the synchronised current real plant state on the simulator.

7. Finally, the user decides whether or not to adopt the procedure output by the system based on the simulated plant response.

## 3.1. Specifying manipulation points by reasoning

A chemical plant has many manipulation points (e.g. PID con trollers). The VAM plant simulator (Machida et al., 2016), which simulates common processes in chemical engineering, has 45 controllers and 107 sensors. However, reinforcement learning performance often deteriorates when the number of possible actions (the number of dimensions of the search space) is increased because of dimensionality. Note that, to achieve better control performance than that of univariate controllers (e.g. PIDs), multivariate control methods (e.g. MPC) have been developed. Increasing the number of control points is necessary to improve the potential control performance. However, reinforcement learning optimises the procedure by trial and error using a simulator, and many control points and responses of a chemical plant may be interdependent (e.g. opening valve A and closing valve B may cause similar responses). Therefore, the existence of control points irrelevant to the main objective increases the difficulty of the optimisation problem, i.e., that of training the agent.

Fig. 2 plots the cumulative rewards of agents with different number of actions in a recovery task when a built-in malfunction (MAL03) is applied in the VAM plant model while training it during the preliminary experiments of this study. A reward function is based on the difference between the current and target states. The results indicate that increasing the number of possible actions degrades the performance from the start of the updates because, in this case, the agent tends to test harmful operations. To explore optimal policies in a practical computation time, the knowledge of the plant structure and standard chemical engineering is utilised to narrow down the space of possible procedures.

![](images/62dfe259a0de483cc9141a2380dcd4b00136c662c7ccc5f8645ea065a504e080.jpg)  
Fig. 1. Architecture of proposed system.

![](images/b7bf2348c6693eea53bf924d40ad715aa6b3a6ec5f59bec1599b180330236594.jpg)

Fig. 2. Progress of cumulative rewards while training reinforcement learning (PPO) agents with different number of actions using VAM plant simulator.  
![](images/20d36a5bd9ebcd9de67edc92cebbb76ee748c7c6ec6805711a5f67d673d643ec.jpg)  
Fig. 3. Piping diagram of a heat exchanger.

## Qualitative reasoning

To specify the control points to manipulate in a given case by tracing back the causality of plant behaviour, a qualitative reasoning framework (De Kleer & Seely Brown, 1984; Forbus, 1997) was employed to describe the plant structure, i.e. causal relationships between the instruments connected by pipes. The plant structure is commonly described as a diagram known as a P&ID. Fig. 3 illustrates the structure of a simple heat exchanger. The input feed (e.g. liquid) at location ?? passes through valve $v _ { 1 } ,$ , which adjusts the flowrate. If the feed flow $A _ { \mathrm { f l o w } }$ is stable (unchanged) and $v _ { 1 }$ aperture size is decreased, the passed amount at ?? also decreases. Valve $v _ { 2 }$ works as well as $v _ { 1 }$ . The output of $v _ { 1 }$ is heated by the hot steam originating from location $D ,$ i.e. the temperature of location ?? increases if the flowrate of the output of $v _ { 1 }$ is unchanged and the flowrate of ?? is unchanged or increased. This diagram can be converted into the following expression in predicate logic:

$$
\begin{array}{l} \text {structure} \quad \text {bin} (A _ {\text {flow}}, v _ {1}, B _ {\text {flow}}) \wedge \text {bin} (C _ {\text {flow}}, v _ {2}, D _ {\text {flow}}) \\ \quad \wedge \text {inv} (B _ {\text {flow}}, T) \wedge \text {bin} (T, D _ {\text {flow}}, B _ {\text {temp}}), \\ \text {state} \quad \text {unchanged} (A _ {\text {flow}}) \wedge \text {unchanged} (B _ {\text {flow}}) \wedge \text {unchanged} (C _ {\text {flow}}) \\ \quad \wedge \text {increase} (B _ {\text {temp}}), \end{array}
$$

where $" \wedge '$ is the operator of logical conjunction (i.e.‘and’) and a pred icate (e.g. unchanged) assigns the characteristics of the state to its argument of a plant part $( \mathrm { e } . g . \ A _ { \mathrm { f l o w } } )$ . Each variable denotes a state in the diagram separated by instruments, such as valves or a heat exchanger, except ?? . which is introduced for expanding the nested arithmetic ‘??=bin(inv(??), ??)’ as ‘inv(??, $t ) \ \wedge \ \mathsf { b i n } ( t , \ y , \ z ) ^ { , }$ . Each state adopts one of the three discrete attribute values: ‘increase’, ‘decrease’, or ‘unchanged’, indicating the signs of the velocity: $^ { \cdot } + ^ { \prime } , ~ { \stackrel { \cdot } { - } } , ~ 0 \mathrm { r } ~ ^ { \cdot } 0 ^ { \prime }$ respectively. Predicates ‘bin’ and ‘inv’ are logical expressions of binary and unary arithmetic operators for addition and inversion, respectively. The formula, $\mathrm { \cdot b i n } ( A _ { \mathrm { f l o w } } , v _ { 1 } , B _ { \mathrm { f l o w } } ) ^ { \prime }$ , represents the dependence of the flow velocity of location ?? (e.g. increase) on that of ?? (e.g. unchanged) and the aperture opening velocity of $v _ { 1 }$ (e.g. increase). Note that if the flowrate of ?? is decreased and that of the $v _ { 1 }$ output is increased, the velocity of changing the temperature of ?? becomes ‘unknown’, i.e. bin $( ^ {  } , ^ { \cdot } + ^ { \cdot } ) { = } ? ^ { \prime }$ , because this qualitative model does not deal with numerical differences.

## Abductive reasoning

Abductive reasoning is a method of logical inference that can esti mate the chain of inference rules between observations and unobserved hypotheses. In predicate logic, e.g. using observation ??(??) ∧ ??(??) and inference rules $A ( x ) \Rightarrow B ( x )$ and $B ( x ) \Rightarrow C ( x )$ , abductive reasoning can induce hypothesis $B ( x )$ because it is connected by the two inference rules and observation. If ??(??) is considered as the start state and ??(??) as the goal state, then abductive reasoning is used as an automated planner in which the states and transitions are expressed in predicate logic. A modified version of the reasoning engine, Phillip (Yamamoto et al., 2015), was employed in this study.

## 3.2. Piecewise linear policy function

Although the response of a chemical plant as a whole is complex, each controller may deal with a relatively simple feedback mechanism. In addition, in chemical engineering, linear models and exponential functions are commonly adopted for parameter fitting because they have been empirically shown to exhibit good extrapolation perfor mance. To leverage these characteristics, a piecewise linear (PL) ap proximator was employed as the policy function of the reinforcement learning agent, i.e.

$$
f (\mathbf {x}) = \sum_ {i = 1} ^ {L} \sigma_ {i} (s _ {i} (\mathbf {x})) g _ {i} (\mathbf {x}),\tag{4}
$$

where $\textbf { x } \in \mathbb { R } ^ { N }$ is an observation vector, $N \in \mathbb { N }$ is the number of sensors, and $L \in \mathbb { N }$ is the number of linear models $( i \in \{ 1 , 2 , 3 , \ldots , L \} )$ Moreover,

$$
\sigma_ {i} (\mathbf {x}) = \frac {\exp (x _ {i})}{\sum_ {j} \exp (x _ {j})},\tag{5}
$$

$$
s _ {i} (\mathbf {x}) = \mathbf {u} _ {i} ^ {\top} \mathbf {x} + a _ {i}, \mathrm{and}\tag{6}
$$

$$
g _ {i} (\mathbf {x}) = \mathbf {w} _ {i} ^ {\top} \mathbf {x} + b _ {i},\tag{7}
$$

where ${ \bf u } _ { i } \in { \mathbb { R } } ^ { N }$ and ${ \bf w } _ { i } \in { \mathbb { R } } ^ { N }$ . This approximator contains the selection and linear models. The selection model, $\begin{array} { r } { 0 \leq \sigma _ { i } ( s _ { i } ( \mathbf { x } ) ) \leq 1 \ ( \sum _ { i } \sigma _ { i } ( s _ { i } ( \mathbf { x } ) ) = } \end{array}$ 1), continuously (softly) selects linear models from ?? candidates $( g _ { i } ( \mathbf { x } ) )$ If $\sigma _ { k } ( s ( { \bf x } ) ) = 1 _ { \mathrm { : } }$ , then $\sigma _ { i } ( s _ { i } ( \mathbf { x } ) ) = 0 ~ ( i \neq k ) ;$ therefore, it can select only one model from ?? linear models. In addition, the selection model can use all models equally if $\begin{array} { r } { \sigma _ { i } ( s _ { i } ( { \bf x } ) ) = \frac { 1 } { { \cal I } } } \end{array}$ for all $i ,$ in which case, the approximator becomes an ordinary linear model. The surface of the function is ‘smoothed’, and thus, it is trainable by standard gradient descent, as in neural networks. In addition, the ‘linear’ complexity of the approximator can be controlled by adjusting the hyperparameter, ??, which directly specifies the number of possible linear regions.

## Preliminary evaluation

The performance of the PL policy function was evaluated on a reinforcement learning task for operating the plant simulator. This task was varying the load of a chemical distillation plant, with 16 sensors comprising a state vector and 3 actions. PPO was employed for learning the policy of the agent, and the policy was represented by a standard fully connected (FC) network and the PL approximator of this study. The FC policy contained two layers, each of which comprised 64 elements activated by the rectified linear unit; this was the best configuration found in the preliminary experiments of this study. The PL policy contained 10 linear regions (i.e. ?? = 10) for each action. Note that the hyperparameters were searched, i.e. ?? for the PL policy, the number of layers (two and three layers) and the number of elements in a layer (32, 64, 96 and 128 elements) for the FC policy, and the best configuration was selected for this evaluation. Fig. 4 shows the cumulative rewards achieved using the policies while training. Both agents finally achieved similar values of the cumulative reward (performance of training episodes); however, the PL policy achieves the highest reward while also being significantly faster than the FC policy.

Because the PL policy has a simple structure and its function ap proximation ability is limited in comparison with that of the FC policy, the PL policy is robust against outlier samples that originate from the exploration of the state space during training. In addition, owing to the shallow structure of the PL policy, the change in a parameter does not affect the global shape of the function and only changes the local portion. This characteristic is advantageous for avoiding the ‘catastrophic forgetting’ problem. The input distribution of the policy function in the state space may vary with time; specifically, at the start of training, it is difficult for the agent to achieve the desired states, whereas after the training, the agent frequently visits the desired states. Therefore, if a parameter update affects the overall shape of the function, the ‘lessons’ learned at the beginning of the training are easily lost, and the training progress become gradual owing to the frequent retraining. The PL policy can avoid these cases.

Fig. 5 shows the performance comparison of the extrapolation abil ity of the policies. Both agents are trained for target loads of 90%, 100%, and 110% using the eight initial states of the simulator. They are evaluated for both the three trained targets and the two untrained targets (80% and 120%). The PL policy achieved higher performance overall, whereas the FC policy failed with the untrained target of 80%.

![](images/d7a9f215d99e87654bc4dd515f37e4988a3db3306c4707ab16383c792d8fae03.jpg)  
Fig. 4. Cumulative rewards of agents (PPO) with PL and FC policy for load down task on distillation plant.

In addition, the action values of the FC policy may include oscil lations, whereas the PL policy is stable owing to its simple structure. Fig. 6 shows the difference between these policies using the plant simulator. The procedure generated by the PL policy (top-left figure) is more stable than that of the FC policy (top-right figure). Moreover, the fluctuations caused by the FC policy affects the main steam pressure (middle-right figure). The behaviour of all steam-driven equipment (e.g. heater) depends on the main steam pressure; therefore, the fluctuation of the procedure may induce malfunctions in the entire plant. These unstable procedures are unpreferable from the operation safety perspective. One solution for stabilising action values is to introduce instability penalty in the reward function. This method was tested for training FC policies; however, the adjustment of the penalty configuration was complex, and a significant improvement could not be achieved.

## 3.3. Online system identification

Because of modelling errors or disturbances, the optimal values of some simulation parameters (e.g. heat loss to atmosphere) may change over time in real scenarios (e.g. change in air temperature). This affects the simulation performance, particularly when a plant is in an unsteady state. To find the optimal parameters online, a simple maximum likelihood estimation method was employed. This method can be explained as follows:

First, a set of candidate parameters is sampled as follows:

$$
\Theta = \{\theta | \theta \sim \mathcal {N} (\theta_ {\mathrm{old}}, \sigma^ {2}) \},\tag{8}
$$

where $\theta _ { \mathrm { o l d } }$ is the latest parameter and $\sigma ^ { 2 }$ is a nonnegative hyperparameter. Second, the loss is calculated for each $\theta \in \Theta$ as

$$
\mathcal {L} (\theta) = \sum_ {k} \sum_ {\tau} (y _ {k} (x ^ {(\tau)}) - \hat {y} _ {k} (x ^ {(\tau)}; \theta)) ^ {2},\tag{9}
$$

where $y _ { k }$ is the ??th sensor value of the real plant, $\hat { y } _ { k }$ is the ??th sensor value of the corresponding simulator, and $x ^ { ( \tau ) }$ is the control input at time ??. Finally, we select

$$
\theta_ {\text { new }} = \underset {\theta \in \Theta} {\operatorname{argmin}} \mathcal {L} (\theta).\tag{10}
$$

This process is repeated every 10 min using the last 10 min of the real plant data.

## 4. Plant experiment

The proposed method was evaluated on a real chemical distillation plant for the separation of water and methanol from their mixture.

![](images/bc0cb7230ac9dc4fb006966dbcfd55e490d02a0f99d06ab9fbcbace3add78d3e.jpg)

![](images/f5a24c5c71817738b6f5bcbcf7faeae70bfbfc716b8f5bae8597b94a58f0e691.jpg)

![](images/a01f9d3c2a332f83f01bc511d9af67649f891dea950a95195e12c4917b7800f2.jpg)

![](images/ab8bf179dabc0137fc71c3957fceb40931055838afd05ca8c776bcdb03e67d31.jpg)

![](images/8981e511214bd66bd20eb4cc790d0017eaa16880aa278281dc814f72f91a7783.jpg)

Fig. 5. Extrapolation performance of agents (PPO) with PL (blue) and FC (orange) policies for load change task on distillation plant. Targets of 80% and 120% are untrained situations. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)  
![](images/0c08fd239e0a10b3877c790f595beb243c9ab77ab821875e3bd2414331f3f016.jpg)  
(a) PL policy.

![](images/3c21e52ec2a32b810b2526d64ec2c7102308be66cedba465aee88a6eb1bf9063.jpg)  
(b) FC policy.

Fig. 6. Procedure and response of PL (left) and FC (right) policies  
![](images/926886b7c781f817967280240ea39ae485a4b4ae6038060e063f2614012265a1.jpg)  
Fig. 7. Chemical plant used in evaluation experiment.

![](images/253f0f1a7ca421befe560645ed82fcbd2825d7611bd079d551490001210dfbec.jpg)  
Fig. 8. Real distillation plant P&ID that was used for the experiment.

## 4.1. Target plant

Fig. 7 depicts the target real distillation plant that separates a binary liquid mixture by chemical distillation. The key component of the plant is a distillation column (tower). The P&ID of the plant is shown in Fig. 8. The plant comprises one distillation tower, two heaters, two coolers, and one drum (a tank as a buffer). Moreover, it is equipped with a distributed control system, and the PID controllers on the system maintain the plant at the set-point state, which is typically the case in standard chemical plants. This is a commonly used minimal archi tecture design for a continuous binary distillation plant. The dynamic simulator of this plant was also utilised.

Based on the P&ID, this plant has seven PID controllers (circles) and six valves (butterfly marks). Note that there are six independent controllers because the level controller (LC) of the reflux drum is cascaded to the flow controller of the top product valve. In addition, the LCs should not be handled after starting up the plant owing to safety reasons; therefore, only four of the seven controllers are generally available. The SVs of these four PID controllers were employed as candidate action points. The two control targets (controlled variables from the perspective of MPC) are product flow and product purity.

![](images/f4983b2c64011c694caa4a31be4bf1e8baafe10b5e93a5726c7bf7a33748d360.jpg)  
(a) Load down.

![](images/5d64d6611c4d04b07b4a4d8a5aec216c02ad665750f0d16215ddb5cddc2ce2ab.jpg)  
(b) Load up.

Fig. 9. Top product flows during load down (left) and load up (right) procedures.  
![](images/724070e4fcfd642f21b0fab90339f72cecfde1a02429d4a4eb5a0dabfdea7824.jpg)  
(a) Load down.

![](images/58a9760cb511a8a337962773da4c3f948f6b839fcc18bed29d2a69ce76b6294f.jpg)  
(b) Load up.  
Fig. 10. Time series of generated optimal load down (left) and load up (right) procedures

## 4.2. Experimental setting

First, a knowledge base was created and the reinforcement learning agents were configured. The agents were trained offline using the target plant simulator, with various prepared initial states, instead of online states. The agents received an observation vector consisting of 17 sensors, 2 soft sensors (top and bottom product purity), 7 SVs, and 1 simulation parameter (of heat balance) on the simulator. One training episode consists of 36 steps with a step interval of 5 min; therefore, each episode is implemented for 3 h in the simulation world. PPO was employed for training the agent. The hyperparameters of the PPO were as follows<sup>2</sup>: nsteps = 7200, minibatch size = 360, epochs = 10, ?? = 0.2, ?? = 1.0 (default 0.99), ?? = 0.99 (default 0.95), value function coefficient = 0.5 (default 1.0), entropy coefficient = 0 (default 0.01), Adam stepsize = 3e − 4, and Adam epsilon = 1e − 5. This is a typical PPO configuration. The hyperparameter of the PL policy, ??, was set as 10. A domain randomisation technique (Tobin et al., 2017) was also employed for adapting to the abrupt changes in the heat balance parameter due to disturbances (e.g. weather change), which may occur in the real plant. During the training, the value of the heat balance parameter was randomly changed.

When the proposed system was applied to the real plant, the pro posed SVs were set manually by human operators who, for operational safety, reviewed and authorised the time series of the SVs (the procedure) and the simulated plant response. The setting of the SVs were performed every 5 min

## 4.3. Load change task

The first experimental task was ‘load change’, i.e. a change in the amount of production while maintaining the purity of both the top and bottom products. Load change is a common operation in chemical plants, and includes both ‘load down’ and ‘load up’. Two experiments were conducted.<sup>3</sup> The agent was trained for various target loads ranging from 70% to 125%, which were randomly selected while training.

The target state of the ‘load down’ task can be expressed as the logical formula, ‘decrease(‘top product flow’) ∧ decrease(‘bottom product flow’) ∧ decrease(‘feed flow’) ∧ unchanged(‘top product purity’) ∧ unchanged(‘bottom product purity’).’ To achieve this state, based on the knowledge base, the reasoning engine hypothesises ‘decrease the feed flow SV’, ‘reduce the reflux flow SV’, ‘reduce the reboiler steam flow SV’, and ‘maintain the preheater temperature SV’. Therefore, the first three controllers were selected as the action points of the reinforcement learning agent (and the last one was omitted). It should be noted that these three action points were also employed in the load up case.

Fig. 9 shows the top product flow of each method in the real plant (red line) and the simulator. The target flow is 75% for the load down and 125% for the load up. The coloured band in the graph depicts the range that is deemed acceptable as the target state of this plant. ‘PID (5 steps)’ shows the emulated procedure of an expert human operator that changes each SV to the final value optimised by the proposed method gradually (and evenly) in five steps. ‘PID (1 step)’ is a sharp and unsafe procedure that runs in only one step. The PID procedure was evaluated on the simulator because the operation schedule of the real plant is tight and the gaps in the responses between the real plant and the simulator are reasonably small in this experiment. The proposed method achieved over 40% faster transition than the other methods. From the perspective of stability, the proposed procedure caused overshoots. Speedy procedures may cause overshoot in general; however, these can be avoided by adjusting the reward settings (e.g. by introducing penalties for overshooting).

Fig. 10 shows the time series of the proposed action values. In a typical operation, owing to the difficulties in predicting accurate time delay responses, which depend on each detailed scenario, human operators change the SVs gradually to the target final values. In the load down (load up) case, they simply decrease (or increase) each SV.

![](images/ce4068719ab310f82bb624189efdeb7bb91ca8974c2d629b4796d48222760d44.jpg)  
(a) Load down.

![](images/978eda46521625780a5d704f0d135a7945361bff9cc7be3dbe87ad203e1fba27.jpg)  
(b) Load up.

Fig. 11. Relative economical performance (consumed time and steam) of load down (left) and load up (right) procedures.  
![](images/a0c831200988f4e6814b1152ce85249e6c78677d6a03497a0c19a3afd8bfcacd.jpg)  
Fig. 12. Sampled and measured purity of top product on the real plant in the degrading procedure.

![](images/1c5d6afcce812ac543bd8d415d7aa63a6ea3114fe87453c7c8798f5ac8ce1462.jpg)  
Fig. 13. Time series of generated optimal degrading procedure.

In contrast, our agent adjusts each SV more systematically. This is the main reason for the increase in the optimality. In addition, the agent also optimises the final state including SVs.

Fig. 11 shows a comparison of the economical performance of the load change procedures using the simulator. This evaluation was conducted by the expert operators of the plant. The proposed method consumed only half the time and steam (heat energy) in comparison with that in the case of human-emulated procedures.

## 4.4. Grade change task

The second task was ‘grade change’, i.e. degrading or upgrading the product quality. The quality is typically evaluated based on the purity of the product. It was assumed that the plant maintains its load, i.e. the feed flow is not changed. For this task, the hypothesised action points were ‘the reboiler steam controller’ and ‘the reflux flow controller’.

Fig. 12 shows the time series of the purity of the top product. The target purity was set as 95%. The coloured band depicts the acceptable range. The purity continuously decreases, and does not achieve the target range even after 2 h from the start.

![](images/5a6c1ebd63158f129650230a9131c97720ea588cc659c842710226137835667d.jpg)  
Fig. 14. Fluctuation of optimal heat balance parameter in [W/m<sup>2</sup> K] in one day.

Apart from the product flow of the load change cases, the product purity is sensitive to the dynamic response of the tower temperature, which is represented by the heat balance parameter in the simulator. Therefore, the value of this parameter was changed to an estimated value at that time (after 2 h had elapsed), and the agent was relaunched to output a new procedure based on the new parameter. After 3 h had elapsed, the new SV (final value of the re-planned procedure) was set in the plant. The step changes of the action values, as plotted in Fig. 13, after 3 h had elapsed, reflect this manipulation. We set a limit value for the difference in the action values, and thus, the possibility of obtaining a sudden change was suppressed. Finally, the purity successfully reached the target state after 4 h had elapsed, as shown in Fig. 12.

The time series of the optimal parameter value every 10 min is plotted in Fig. 14. This value changes considerably after the commence ment of the purity change procedure, and the average value increases after the surge. This parameter change led to a higher purity than expected after 2 h had elapsed. Note that the surge and the steplike increase after the surge reflect disturbances and modelling errors, i.e. difference in behaviours (e.g. air temperature change, non-ideality, imperfect mixing). Therefore, the change indicates the total change in the scenario, including the change in the original ‘heat balance’.

## 5. Discussion

## 5.1. Sim-to-Real gap

Simulation models typically have modelling errors from unmodelled phenomena, and thus, there are differences between the responses of a simulator and an actual plant. One approach for decreasing this gap when a difference is caused by an unmodelled change in the parameters over time is to optimise the unobservable parameters online from observable states. Particle filters (Doucet, 2000) are well-known for such system identification.

Another approach to reduce this gap is to consider modelling errors as disturbances (Pinto et al., 2017). Training is performed on an agent for optimising operation and another agent for destabilising the system by applying disturbances in parallel.

## 5.2. Generation of plant operation procedures

Automatic generation of operation procedures is a challenging plant control task and has long been studied (Gabbar et al., 2004). Logicbased classical planning techniques can be used to explore action sequences from one state to another based on a discrete state representation and a discrete action (state transition rule) set. Classical planning techniques can derive a sequence of elements to be manipulated using qualitative behavioural models of a plant, and have been utilised for operation procedure generation in chemical plants (Aylett et al., 1998). Gofuku et al. (2004) used a qualitative functional model based on mul tilevel flow modelling for a plant and derived the procedures for leading a plant to a desired state by tracing back the influence propagation rules between plant elements such as valves and pipes. Because this type of method focuses on discrete qualitative models, it cannot deal with continuous manipulation and optimisation of operation element values.

## 5.3. MPC

MPC is a method for establishing optimal quantitative operation procedures (Kano & Ogawa, 2010; Qin & Badgwell, 2003). Some MPC methods based on state–space models assumes the models to be accessi ble, continuous, and differentiable (Griffith, 2018); however, real plants can have discrete state transitions (e.g. on/off states). The models are also required to be representable in mathematical frameworks, such as state–space models, and thus, many commercial plant sim ulators, commonly installed in large-scale plants, are unsuitable for use with MPC without any modification. DMC, one of the practical MPC methods, is limited to linear models, whereas a plant response may have nonlinearities. The development of DMC requires conducting numerous step response tests on operational plants because achieving good performance requires an excessive number of step response coefficients (Lundström et al., 1995). In addition, use of DMC is unsuitable for nonstationary scenarios, such as varying the plant load, because the response or the dynamic characteristics (linear model of DMC) of the plant depends on the balancing point (i.e. material balance and energy balance). For instance, the response (e.g. increase in the tower top temperature on increasing the hot steam for the reboiler by 10 [kg/h]) is different for the 100% and 75% load scenarios. The main purpose of a dynamic simulation is to reproduce such nonsteady changes over time; therefore, the combination of a dynamic simulation and reinforcement learning can achieve improved performance in such nonsteady cases.

In contrast to process-model-based methods (e.g. state–space MPC and DMC), the method developed in this study can consider whole plant states, such as recycling of materials, reutilisation of heat, and the interactions between different processes; thus, it can contribute to plant-wide optimisation.

To consider and optimise the economic performance of a plant, as described in Section 4., real-time optimisation (RTO) (Trierweiler, 2014) is typically employed with MPC. RTO calculates a desired state from the economical perspective by solving a nonlinear optimisation problem at the plant site, and subsequently, MPC calculates the optimal manipulation values by solving the difficult optimisation problem. These methods require a large computational resource during the entire implementation of the control system. In contrast with MPC and RTO, the proposed method does not require such a large computational resource at the plant site because the optimisation process is conducted in the preparation (learning) phase before starting the actual control. Note that the reinforcement learning agent optimises the desired final state as well as the transient changing states.

Real plants have several uncertainties and dealing with these is crucial; however, it has been highlighted that MPC needs further development for industrial applications in this aspect (Bemporad & Morari, 1999). To deal with the problem of system uncertainty and stochastic disturbances, stochastic MPC (SMPC) (Heirung et al., 2018; Paulson et al., 2020) is proposed. SMPC extends MPC by introducing stochastic optimal control and improves the performance when the system can be sufficiently expressed as a linear model. SMPC can ensure optimality of the output for linear systems because it solves the optimisation problem for each state. In comparison, reinforcement learning cannot typically ensure its optimality. However, it does have the potential for increasing the optimality of nonlinear systems.

## 6. Conclusion

In this study, we proposed a method for generating operation proce dures for chemical plants using automated reasoning and reinforcement learning. The candidate set of instruments for manipulation is specified based on plant knowledge and utilises an automated reasoning engine. Subsequently, the reinforcement learning agent determines the opti mal time series procedure through interaction with a plant simulator. In particular, it uses a PL policy function, which can improve the extrapolation performance, i.e. adds robustness to the varying target states, as demonstrated in a simulation experiment. Furthermore, to synchronise the states of a real plant and the simulator connected to the agent, online simulation and a simple method for simulation parameter estimation were used. By conducting experiments on a real distillation plant, the proposed method was demonstrated to assist human operators in performing economical operation.

In future studies, we plan to focus on the following three major topics: (i) improvement in the system identification, (ii) real-time adjustments of the action values, and (iii) improvement in the interaction between the proposed system and the plant operators.

In fluctuating scenarios, to ensure robust control of a real plant, quick recovery from an unexpected real plant state to the agent predicted state by faster and more precise system identification meth ods than that used in this study, followed by re-planning of the agent, are required.

The identification can be time-consuming if we solely depend on the iterative computation of the simulator; moreover, the gaps between responses of real and simulated plants will increase with time without any counteraction during the identification process. Therefore, reflex ive and rapid adjustments of the pre-planned action sequence based on the latest gap information will also be required.

We also plan to improve the methods of interactions between the proposed system and the plant operators for reliable and long-lasting optimal operations. For this purpose, it is important to increase the information for decision-making on adoption of the agent plan. One direction is to improve the explainability of the system, e.g. the inter pretability of the agent behaviour. Another direction is to provide a quantitative confidence of the plan and its prediction to the operators by introducing uncertainty quantification (Swiler & Giunta, 2007).

List of abbreviations

DMC Dynamic Matrix Control

FC Fully Connected

LC Level Controller

MPC Model Predictive Control

P&ID Piping and Instrumentation Diagram

PID Proportional–Integral–Differential controller

PL Piecewise Linear

PPO Proximal Policy Optimisation

RTO Real-Time Optimisation

SMPC Stochastic Model Predictive Control

SV Set-point Value

TEP Tennessee Eastman Process

VAM Vynil Acetate Monomer

## Declaration of competing interest

The authors declare that they have no known competing finan cial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This study was a result of a collaborative research of AIST, Mitsui Chemicals Inc., Omega Simulation Co., Ltd., and NEC Corporation. The authors are grateful to Yasuo Fujisawa, Toshihide Kihara, Masahiko Tatsumi, Masanori Endo, Atsushi Uchimura and Norio Esak (Mitsui Chemicals) for the helpful discussion on the experiments and the needs for supporting human plant operators. They also acknowl edge the useful advice regarding the design of the system architecture for the experiments and modelling of the chemical process provided by Gentaro Fukano, Tsutomu Kimura, Akihiko Imagawa, Takayasu Ikeda, and Yasuhiro Kamata (Omega Simulation).

## References

Aylett, R., Soutter, J. K., Petley, G. J., & Chung, P. W. H. (1998). AI planning in a chemical plant domain. In Proceedings of European conference on artificial intelligence (pp. 622–626).

Bemporad, A., & Morari, M. (1999). Robust model predictive control: A survey. In Robustness in identification and control (pp. 207–226). Springer.

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). Openai gym. arXiv preprint arXiv:1606.01540.

Cutler, C. R., & Ramaker, B. L. (1980). Dynamic matrix control - a computer contro algorithm. In Joint automatic control conference, 17. San Francisco, CA.

De Kleer, J., & Seely Brown, J. (1984). A qualitative physics based on confluences. Artificial Intelligence, 24(1–3), 7–83.

Doucet, A. (2000). On sequential Monte Carlo sampling methods for Bayesian filtering. Statistics and Computing, 10, 197–208.

Downs. J. J.. & Vogel. E.. E. (1993). A plant-wide industrial process control problem Computers and Chemical Engineering, 17(3), 245–255.

Eaton, J. W., & Rawlings, J. B. (1991). Model predictive control of chemical processes. In 1991 American control conference (pp. 1790–1795). IEEE.

Forbes, M. G., Patwardhan, R. S., Hamadah, H., & Gopaluni, R. B. (2015). Mode predictive control in industry: Challenges and opportunities. IFAC-PapersOnLine, 48(8), 531–538.

Forbus, K. D. (1997). Qualitative reasoning. In CRC handbook of computer science and engineering (pp. 715–733). CRC Press

Fujita, Y., Kataoka, T., Nagarajan, P., & Ishikawa, T. (2019). ChainerRL: A deep reinforcement learning library. In Workshop on deep reinforcement learning at th 33rd conference on neural information processing systems.

Gabbar, H. A., Aoyama, A., & Naka, Y. (2004). AOPS: automated operating procedure synthesis for chemical batch plants. Transaction of the Society of Instrument and Control Engineers, 40(9), 968–977.

Gofuku, A., Inoue, T., & Sugihara, T. (2004). A technique to generate plausible counter operation procedures for an emergency situation based on a model expressing Journal of Nuclear Science and Technology 54

Griffith, D. W. (2018). Advances in nonlinear model predictive control for large-scale chemical process systems (Ph.D. thesis), Carnegie Mellon University.

Gu, S., Holly, E., Lillicrap, T., & Levine, S. (2017). Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. In 201Z JEEE international conference on robotics and automation (pp. 3389–3396). IEEE

Heirung, T. A. N., Paulson, J. A., O’Leary, J., & Mesbah, A. (2018). Stochastic model predictive control - how does it work? Computers and Chemical Engineering, [ISSN: 0098-1354] 114, 158–170.

Kano, M., & Ogawa, M. (2010). The state of the art in chemical process control in Japan: Good practice and questionnaire survey. Journal of Process Control, 20(9), 969–982.

Kvamsdal, H., Svendsen, H., Olsvik, O., & Hertzberg, T. (1999). Dynamic simulation and optimization of a catalytic steam reformer. Chemical Engineering Science, 54(13–14), 2697–2706.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. In Bengio, Y. & LeCun, Y. (Eds.), 4th international conference on learning representations, Conference track proceedings.

Lundström, P., Lee, J., Morari, M., & Skogestad, S. (1995). Limitations of dynamic matrix control. Computers and Chemical Engineering, [ISSN: 0098-1354] 19(4), 409–421.

Machida, Y., Ootakara, S., Seki, H., Hashimoto, Y., Kano, M., Miyake, Y., Anzai, N., Sawai, M., Katsuno, T., & Omata, T. (2016). Vinyl acetate monomer (VAM) plant model: A new benchmark problem for control and operation study. IFAC-PapersOnLine, 49, 533–538.

Nakaya, M., Fukano, G., Onoe, Y., & Ohtani, T. (2006). On-line simulator for plant operation. In 2006 6th World Congress on Intelligent Control and Automation, vol. 2 (pp. 7882–7885.

Patle, D. S., Ahmad, Z., & Rangaiah, G. P. (2014). Operator training simulators in the chemical industry: review, issues, and future directions. Reviews in Chemical Engineering, 30(2), 199–216.

Paulson, J. A., Buehler, E. A., Braatz, R. D., & Mesbah, A. (2020). Stochastic model predictive control with joint chance constraints. International Journal of Control, 93(1), 126–139.

Pinto, L., Davidson, J., Sukthankar, R., & Gupta, A. (2017). Robust adversarial reinforcement learning. In Precup, D. & Teh, Y. W. (Eds.), Proceedings of 34th international conference on machine learning (pp. 2817–2826).

Qin, S. J., & Badgwell, T. A. (2003). A survey of industrial model predictive control technology. Control Engineering Practice, 11(7), 733–764

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2018). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv: 1506.02438.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. Computing Research Repository, abs/1707.06347, arXiv:1707.06347.

Sedighizadeh, M., & Rezazadeh, A. (2008). Adaptive PID controller based on reinforcement learning for wind turbine control. In Proceedings of world academy of science, engineering and technology, vol. 27 .

Shirakawa, M., Nakamoto, M., & Hosaka, S. (2005). Dynamic simulation and opti mization of start-up processes in combined cycle power plants. JSME International Journal Series B Fluids and Thermal Engineering, 48(1), 122–128.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., & Hassabis, D. (2017). Mastering the game of go without human knowledge. Nature, 550(7676), 354–359.

Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT Press. Swiler, L. P., & Giunta. A. A. (2007). Aleatory and epistemic uncertainty quantification Swiler. L. P.. & Giunta. A. A. (2007). Aleatory and epistemic uncertainty quantificatior

for engineering applications. In Proceedings of joint statistical meetings. Washington, DC: Sandia National Laboratories.

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (pp. 23–30).

Trierweiler, J. (2014). Real-time optimization of industrial processes. In Encyclopedia of systems and control. London: Springer-Verlag, ISBN: 978-1-4471-5102-9.

Yamamoto, K., Inoue, N., Inui, K., Arase, Y., & Tsujii, J. (2015). Boosting the effi ciency of first-order abductive reasoning using pre-estimated relatedness between predicates. International Journal of Machine Learning and Computing, 5, 114–120.