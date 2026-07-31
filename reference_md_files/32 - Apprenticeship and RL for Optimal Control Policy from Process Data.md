P R O C E S S S Y S T E M S E N G I N E E R I N G

AICOEURNAL

# Using process data to generate an optimal control policy via apprenticeship and reinforcement learning

Max Mowbray<sup>1</sup> | Robin Smith<sup>1</sup> | Ehecatl A. Del Rio-Chanona<sup>2</sup> | Dongda Zhang<sup>1</sup>

<sup>1</sup>Department of Chemical Engineering and Analytical Science, The University of Manchester, Manchester, UK <sup>2</sup>Department of Chemical Engineering, Imperial College London, London, UK

## Correspondence

Ehecatl A. Del Rio-Chanona, Department of Chemical Engineering, Imperial College London, South Kensington, London SW7 2AZ, UK.

Email: a.del-rio-chanona@imperial.ac.uk

Dongda Zhang, Department of Chemical Engineering and Analytical Science, The University of Manchester, Oxford Road, Manchester M1 3BU, UK. Email: dongda.zhang@manchester.ac.uk

## Abstract

Reinforcement learning (RL) is a data-driven approach to synthesizing an optimal control policy. A barrier to wide implementation of RL-based controllers is its datahungry nature during online training and its inability to extract useful information from human operator and historical process operation data. Here, we present a twostep framework to resolve this challenge. First, we employ apprenticeship learning via inverse RL to analyze historical process data for synchronous identification of a reward function and parameterization of the control policy. This is conducted offline. Second, the parameterization is improved online efficiently under the ongoing process via RL within only a few iterations. Significant advantages of this framework include to allow for the hot-start of RL algorithms for process optimal control, and robust abstraction of existing controllers and control knowledge from data. The framework is demonstrated on three case studies, showing its potential for chemical process control.

## K E Y W O R D S

apprenticeship learning, inverse reinforcement learning, machine learning, optimal control, reinforcement learning

## 1 | INTRODUCTION

Recent initiatives for efficiency improvements in industrial process operation has driven interest in the development of high performance, advanced process control (APC) schemes. Reinforcement learning (RL) has achieved impressive results on benchmark game-based control tasks,<sup>1,2</sup> providing an avenue for research in translation to APC. In spite of its high potential, RL has yet to produce any meaningful impact in the (bio)chemical process industry. This work presents a two-step approach to RL-based policy learning, which leverages process data to parameterize an existing control law and then improves the performance of such control further. Additionally, the approach promises to increase the learning efficiency of RL-based control policies, reducing computational and technical investment, as well as data demand.

RL constitutes a subfield of machine learning (ML), which aims to learn optimal control policies. Here, the control problem is formulated as a Markov decision process (MDP), which describes decision-making as a value maximization problem. MDPs construct a probabilistic framework for the discrete-time evolution of a stochastic decision process, with the cost (or value) associated with a control policy, and ultimately process trajectory, evaluated by a reward function. Explicitly, MDPs provide a mathematical basis for sequential decisionmaking in stochastic environments, which is a description common to process control.<sup>3</sup> Figure 1 details the interpretation of process control as an MDP. The structure of MDPs provides natural closed-loop feedback control.

Solution to an MDP provides a policy π( ), which minimizes the expected cost or equivalently maximizes the expected value associated with the evolution of process state. Such a policy satisfies the

![](images/07b805ff13f1a54b6c1d6e826dee5a7a450094f1eea38873296eb1ed3c32e613.jpg)  
F I G U R E 1 Translation of the framework provided by Markov decision process (MDPs) to process control, where the process is analogous to an environment, and the controller to an agent. $\pmb { x } _ { t }$ is representative of the true system state at discrete time t; $\pmb { u } _ { t }$ is the control action computed by the control law at discrete time t; and $R _ { t + 1 }$ is the scalar feedback signal (reward) indicative of the quality of process evolution at time t + 1

Bellman optimality equation, which is a discrete-time analogue to the continuous-time Hamilton–Jacobi–Bellman equation.<sup>3</sup> Dynamic programming (DP) methods provide exact solution to the Bellman optimality equation. However, such an approach assumes knowledge of the exact process dynamics. DP becomes additionally impractical in the highly dimensional continuous state and action spaces often observed in the process industries.<sup>4</sup> In contrast, RL methods do not require knowledge of the exact process dynamics to learn a solution policy. Instead, RL learns from experience of the process, allowing for π( ) to be recalibrated as the process evolves through time via process data.<sup>5</sup> Furthermore, RL has shown significant industrial potential as demonstrated in a number of research works, which have explored application to the calibration of PID controllers;<sup>6</sup> set point tracking;<sup>7</sup> dynamic optimization of nonlinear, stochastic systems;<sup>5,8,9</sup> de novo drug<sup>10</sup> and protein design;<sup>11</sup> and in augmentation of the performance of various model predictive control (MPC) approaches.<sup>12,13</sup> Indeed, the potential use of RL draws discussion of its relation to MPC in the development of APC schemes. MPC schemes require periodic recalibration, which demands expense in technical expertise and often process downtime. The data-driven nature of RL could well mitigate this. Further, the framework provided by MDPs accounts for process stochasticity in a closed-loop manner, converse to MPC where decisions are based on open-loop simulation of the process model, with the loop only “closed” upon observation of the system state at the next discrete time index. Hence, inputs from an RL controller will account for disturbance whereas MPC may not. This provides a theoretical basis for the benefit of RL over MPC controllers.

One set of RL algorithms are known generally as policy optimization methods. Policy optimization methods aim to learn a policy by implicitly learning the value or cost over the decision space<sup>14–16</sup> and directly parameterizing a policy. There are a number of approaches to policy optimization as underpinned by evolutionary strategies, finite difference and policy gradient methods.<sup>17,18</sup> Policy optimization methods have been deployed for tasks including dynamic optimization of nonlinear stochastic processes<sup>19</sup> and tracking problems<sup>6</sup>. For further review of RL methods and their application within the process industries, we direct the reader to the following works.<sup>7,20</sup>

The learning process encapsulated by RL demands both time and technical investment in policy training. This is highlighted further given that RL-based controllers are currently unable to generalize well across control tasks, for example, different changes of set point, meaning policy training is typically undertaken for each task.<sup>21</sup> As a result, implementation of RL control policies is computation and expertise expensive. To solve this problem, this work proposes a method to reduce the time and resource investment demanded by RL, through leverage of process data to learn from demonstration provided by an existing (but unknown) con trol policy. Then, the initialized RL is improved by learning from the real process over a short time period, thus outperforming the existing control policy. This two-step strategy has been recently deployed in domains including autonomous helicopter flight<sup>22</sup> and self-driving cars.<sup>23,24</sup> To demonstrate this approach, Section 2 will introduce the preliminaries and motivation, Section 3 will outline the methodology, with Section 4 exhibiting different case studies.

## 2 | PRELIMINARIES

## 2.1 | Policy gradients and reinforce

Policy gradient methods directly learn a policy. Through the use of artificial neural networks (ANNs) as parameterization, the policy may be deployed naturally in either discrete or continuous action spaces through appropriate network construction.<sup>25</sup> Policy gradient methods do not explicitly learn the value of the policy. Instead, under the policy gradient theorem, acting with respect to the policy and gaining experience of the process dynamics provides approximation of the direction in which value increases fastest in parameter space. Hence, learning proceeds through gradient ascent to update the parameters of the policy to ensure control policies of high value (or low cost) are more probable.<sup>18</sup>

One policy gradient algorithm, reinforce with baseline, approxi mates the direction in which the policy observes increased perfor mance through Monte Carlo realizations of the process dynamics under the current policy parameterization. This algorithm has several advantages such as convergence to locally optimal solutions in policy space<sup>26</sup> and efficient exploration of the decision space without requirement for a bandit strategy or further optimization routine for action selection—as is the case in many pure action-value methods.<sup>27</sup> Demonstration of the method is also available.<sup>19</sup> Therefore, it is used in this work to learn an RL parameterization of an existing control policy from process data. Despite favor of the reinforce with baseline algorithm, other RL methods capable of operating in continuous action and state spaces could be implemented, such as entropy regularized policy optimization methods,<sup>16</sup> trust region policy optimization,<sup>14</sup> and proximal policy optimization (PPO) methods.<sup>15</sup>

## 2.2 | Learning from demonstrations via apprenticeship

Learning from demonstrations encompasses an increasingly prevalent and established group of methods, which leverage data generated from an existing but unknown control policy to aid learning-based control systems. This concept is generally termed as apprenticeship learning (AL). AL has been adopted in a number of complex control domains,<sup>22,24</sup> but to our knowledge, this work is the first to propose use of the method to leverage plant data directly, and this is one of the primary contributions of this work. The concepts of AL are expressed in three main subfields including behavioral cloning (i.e., supervised learning), inverse optimal control, and inverse reinforcement learning (IRL).

This study exploited IRL built upon the framework provided by MDPs.<sup>28</sup> MDPs express process objectives mathematically as a reward function. The reward function provides a scalar feedback signal indicative of the optimality of process evolution. IRL is concerned with the task of mathematically abstracting the reward function given process knowledge and demonstrations from an existing control policy. The IRL problem is formalized as: given observations of an existing policy over time, sensory inputs available for determination of the originally demonstrated control law and a model of the process; determine the reward function that can mostly justify the demonstrated behavior.<sup>24,29,30</sup> IRL proceeds on the assumption that demonstrated control action is noisily optimal under the reward function derived.<sup>30,31</sup> However, it should be noted that this does not necessarily imply that the policy is optimal in view of the true objectives for process control and optimization.

As such, IRL leverages process data to learn a reward function that encodes the control objectives of an existing scheme into a feedback signal. A control policy that maximizes the utility of this reward function within the MDP framework provides a parameterization of the existing control scheme. Hence the pairing of IRL with RL as an MDP solver, allows for synchronously learning the parameterization of an existing but unknown control policy as described in process data. The generated reward function can be used to compare against the process objective (if known) and suggest if the extracted control policy is suitable for online learning. Moreover, manual modifications are always implemented during process control even if the process objective is known. These manual modifications cannot be quantified by human operators, but can be retrieved from historical data by IRL. Therefore, using IRL to generate a reward function is advantageous for parameterization of the optimal control policy.

## 2.3 | Motivation

In the following work, we demonstrate a framework for learning and optimization of chemical processes. The framework consists of two steps: offline learning, and online learning and improvement. Here the use of terminology is converse to that common in the ML commu nity. In this work, offline learning indicates a process of AL (via IRL) to infer control objectives from process data and the learning of a corresponding parameterization of the control policy described by data; online improvement then indicates the transfer of the learned parameterization to the real system for the purpose of further policy improvement under the true process objective. The framework enables the learning of an RL-based control policy, by leveraging process data from existing control schemes (offline) and subsequently improves the learned policy parameterization via further RL (online). The automation of offline learning and the policy tuning process that is associated, provides a significant contribution given the technical computational and data demands of RL-based policy learning.

Offline learning produces a parameterization of the existing control policy, which could be deployed directly for control. The parameterization will achieve similar performance to that expressed by the original control scheme. If necessary, the parameterization may then be transferred to the second stage of online learning for further policy improvement. It should be emphasized that the leveraging of process data is significant given the practical difficulties in learning an RL based policy “from scratch”.<sup>19,32</sup> The framework also lends itself to the improvement and recalibration of the control scheme temporally. Figure 2 provides further description of the framework proposed.

## 3 | METHODOLOGY

## 3.1 | Problem statement

The following work proceeds on the formulation of the underlying problem of process control as an MDP. The true dynamics of an MDP are described as follows:

$$
\pmb {x} _ {t + 1} \sim p (\pmb {x} _ {t + 1} | \pmb {x} _ {t}, \pmb {u} _ {t})\tag{3.1.1}
$$

![](images/3d4732e7bb65991f0c6f456eaf5558c709805ea82a69ea512f71e664a50a4025.jpg)  
F I G U R E 2 The offline–online framework proposed for the learning and optimization of processes. Offline learning utilizes process data to learn a reward function R(α<sup>\*</sup>) and a parameterization of the demonstrated policy $\pi _ { p o } \big ( \pmb { \theta } _ { ( k _ { 0 } ) } \big )$ . Online learning utilizes the learned parameterization as initialization for further policy optimization under a reward function $R _ { p o } ( \cdot )$ ) descriptive of the true process objective [Color figure can be viewed at wileyonlinelibrary.com]

$$
\mathbf {y} _ {t + 1} \sim p (\mathbf {y} _ {t + 1} | \mathbf {x} _ {t + 1})\tag{3.1.2}
$$

where $\mathbf { x } { \in } \mathbb { R } ^ { n _ { x } }$ is a vector of continuous variables representative of the true system state, $\pmb { u } \in \mathbb { R } ^ { n _ { u } }$ the manipulated variables (MVs), $\pmb { \gamma } \in \mathbb { R } ^ { n _ { y } }$ the observed control variables and t is indicative of the discrete time index.<sup>33</sup> The process evolution between discrete time indices t and $t + 1$ is governed by the conditional density function p(<sup>x</sup><sub>t + 1</sub> <sup>x</sup><sub>t</sub>, <sup>u</sup><sub>t</sub>). Similarly, the observation $\pmb { \gamma _ { \mathrm { t } } }$ of the true state of the system $\pmb { x } _ { t }$ is governed by the conditional density $p ( { \pmb y } _ { t } | { \pmb x } _ { t } ) .$ . To facilitate learning of a policy prior to transfer to the real system, approximation of the true dynamics proceeds based on state-space models and assumptions regarding process stochasticity, hence:

$$
\boldsymbol {x} _ {t + 1} = f (\boldsymbol {x} _ {t}, \boldsymbol {u} _ {t}, \boldsymbol {d} _ {t})\tag{3.1.3}
$$

$$
\pmb {y} _ {t + 1} = g (\pmb {x} _ {t + 1})\tag{3.1.4}
$$

where $\pmb { f } ( \cdot ) : \mathbb { R } ^ { n _ { x } \times n _ { u } \times n _ { d } } \longrightarrow \mathbb { R } ^ { n _ { x } }$ is representative of the process dynamics and $\pmb { d } _ { t } \in \mathbb { R } ^ { n _ { d } }$ is representative of the process disturbance. The mapping $g ( \cdot ) : \mathbb { R } ^ { n _ { x } } \longrightarrow \mathbb { R } ^ { n _ { y } }$ is the state observation associated with measurement noise<sup>2</sup>.

The following work deploys RL to learn a control policy from process data. The objective of RL is to minimize the expected cost of a dynamic process (or equivalently to maximize its value). In the following, a process trajectory, $\pmb { \tau } = ( \pmb { x } _ { 0 } , \pmb { y } _ { 0 } , \pmb { u } _ { 0 } , . . . \pmb { u } _ { T - 1 } , \pmb { x } _ { T } , \pmb { y } _ { T } )$ , describes the manner in which a process evolves over a given discrete time horizon of length T. The cost or value $G ( \tau )$ of the process trajectory over a finite horizon is denoted:

$$
G (\pmb {\tau}) = \sum_ {t = 1} ^ {T} \gamma^ {t - 1} R _ {t}\tag{3.1.5}
$$

where $\gamma \in ( 0 ,$ , 1] is a discount factor, which provides a net present value interpretation of future value; and $R _ { t }$ is the credit (reward) assigned to the process' evolution between time indices t 1 and t. However, in view of process stochasticity, the probability of observing τ adheres to a conditional density p(τ θ) based on the control policy and process dynamics:

$$
p \Big (\boldsymbol {\tau} | \boldsymbol {\theta}) = \bar {\rho} (\mathbf {x} _ {0}) p (\mathbf {y} _ {0} | \mathbf {x} _ {0}) \prod_ {t = 0} ^ {T - 1} \pi (\mathbf {u} _ {t} | \mathbf {y} _ {t}, \boldsymbol {\theta}) p (\mathbf {x} _ {t + 1} | \mathbf {x} _ {t}, \mathbf {u} _ {t}) p (\mathbf {y} _ {t + 1} | \mathbf {x} _ {t + 1})\tag{3.1.6}
$$

where $\bar { \rho } ( { \pmb x } _ { 0 } )$ is the probability density of the initial system state; $\pi ( \boldsymbol { u } _ { t } | \boldsymbol { \mathsf { y } } _ { t } , \cdot )$ is the conditional density function descriptive of the learned policy, which is parameterized by $\pmb \theta \in \mathbb { R } ^ { n _ { \theta } } ;$ and $p ( \pmb { x } _ { t + 1 } | \pmb { x } _ { t } , \pmb { u } _ { t } )$ is the conditional density function representative of the process dynamics.

Note that the definition of a policy as a conditional density function implies it is stochastic. This is important in the scope of the learning process associated with RL but does not necessarily assert the use of a stochastic policy upon deployment for control of the real system (only the mode might be used in practice). The objective of the RL problem and learning process is to find a policy π( $\mathbf { \nabla } _ { \cdot , \pmb { \theta } ^ { \mp } ) }$ that maximizes the objective $J ( \tau ) ,$ such that

$$
\pi (\cdot , \boldsymbol {\theta} ^ {*}) = \operatorname{argmin} _ {\pi (\cdot , \boldsymbol {\theta})} - J (\boldsymbol {\tau})\tag{3.1.7}
$$

$$
J (\boldsymbol {\tau}) = \int p (\boldsymbol {\tau} | \boldsymbol {\theta}) G (\boldsymbol {\tau}) d \boldsymbol {\tau}\tag{3.1.8}
$$

Equation (3.1.8) describes the probability-weighted average of trajec tory value and hence reformulation may utilize equivalence of $J ( \tau )$ as the expectation of trajectory value under the policy parameters θ, such that

$$
J (\pmb {\tau}) = \mathbb {E} _ {\tau \sim p (\pmb {\tau} | \pmb {\theta})} [ G (\pmb {\tau}) ]\tag{3.1.9}
$$

The description provided in this section formalizes the problem of optimal control under the framework provided by MDPs. One approach to finding approximate solution to the problem described by Equations (3.1.7)–(3.1.9) is encompassed by policy optimization RL methods.

## 3.2 | Policy gradient and reinforce

Policy gradient methods are a subset of policy optimization methods, which estimate the gradient of the objective detailed by Equation (3.1.8) with respect to the parameters of the current policy. Mathematically, this is described by the policy gradient theorem.<sup>18</sup> The Supporting Information (SI) provides full derivation and explana tion of the policy gradient theorem. Given an estimate of the true pol icy gradient, gradient ascent methods facilitate policy improvement to make trajectories of higher reward more probable. In this manner, the policy parameterization is updated (via Equation (3.2.2)) in the direc tion provided by the policy gradient (Equation (3.2.1)):

$$
\nabla_ {\boldsymbol {\theta} _ {(j)}} J (\boldsymbol {\tau}) = \nabla_ {\boldsymbol {\theta}} \int p (\boldsymbol {\tau} | \boldsymbol {\theta}) G (\boldsymbol {\tau}) d \boldsymbol {\tau}
$$

$$
= \mathbb {E} _ {\tau \sim p (\boldsymbol {\tau} | \boldsymbol {\theta})} [ G (\boldsymbol {\tau}) \nabla_ {\boldsymbol {\theta}} \log p (\boldsymbol {\tau} | \boldsymbol {\theta}) ]\tag{3.2.1}
$$

$$
\boldsymbol {\theta} _ {(j + 1)} = \boldsymbol {\theta} _ {(j)} + \omega \nabla_ {\boldsymbol {\theta} _ {(j)}} J (\boldsymbol {\tau})\tag{3.2.2}
$$

where j is the iteration of policy optimization, and ω is the step size in the direction of the policy gradient, $\nabla _ { \pmb { \theta } _ { ( j ) } } J ( \pmb { \tau } )$ . The derivation of Equation (3.2.1) leverages the use of a logarithmic identity (see SI). This enables mathematical separation of the conditional probability functions descriptive of the process dynamics and policy (see Equation (3.1.6)). Given the process dynamics are independent of the parameterization, θ<sup>,</sup> of the policy, $\pi ( \pmb \theta , \cdot ) ,$ examination of Equation (3.1.6) provides:

$$
\nabla_ {\boldsymbol {\theta} (j)} \log p (\boldsymbol {\tau} | \boldsymbol {\theta}) = \sum_ {t = 0} ^ {T - 1} \nabla_ {\boldsymbol {\theta} (j)} \log \pi \left(\boldsymbol {u} _ {t} | \boldsymbol {y} _ {t}, \boldsymbol {\theta} _ {(j)}\right)\tag{3.2.3}
$$

Consequently, the policy gradient described by Equation (3.2.1) is reformulated as:

$$
\nabla_ {\boldsymbol {\theta} (j)} J (\boldsymbol {\tau}) = \mathbb {E} _ {\tau} \left[ G (\boldsymbol {\tau}) \sum_ {t = 0} ^ {T - 1} \nabla_ {\boldsymbol {\theta} (j)} \log \pi \big (\boldsymbol {u} _ {t} | \boldsymbol {y} _ {t}, \boldsymbol {\theta} _ {(j)} \big) \right]\tag{3.2.4}
$$

Exact computation of the true policy gradient requires full knowledge of the conditional density functions descriptive of process dynamics. Given such knowledge of the process dynamics are unavailable, the policy gradient is approximated by directly sampling the process under the current policy parameterization over a given time horizon via a Monte Carlo method.<sup>5</sup> This is encapsulated by the reinforce with baseline algorithm, which is detailed by Algorithm 1.

## Reinforce with baseline

Input: Initialize: a policy π with initial parameters $\pmb { \theta } _ { 0 } ;$ learning rate ω; episode length T; K episodes for Monte Carlo rollouts of the policy; and, N training epochs. Early stopping conditions may also be implemented.

Output: A policy π(<sup>u</sup>| <sup>y</sup>, θ)

$\mathsf { f o r j } = 1 , . . . , N$ do

1. Perform Monte Carlo realizations of the policy for T timesteps and K trajectories. Store all state action pairs observed $\left( { \pmb u } _ { t } ^ { k } , { \pmb y } _ { t } ^ { k } \right)$ , as well as the total return from the episode $G _ { t } ^ { k }$ (see Equation (3.1.5))

2. Estimate the policy gradient and update the parameters of the policy such that $\begin{array} { r } { \pmb { \theta } _ { ( j + 1 ) } = \pmb { \theta } _ { ( j ) } + \omega _ { ( j ) } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } } \end{array}$ $\begin{array} { r } { \left[ \left( G ^ { k } - b \right) \nabla _ { \pmb { \theta } } \sum _ { t = 0 } ^ { T - 1 } \mathsf { I n } ~ \pi \big ( \pmb { u } _ { t } ^ { k } | \pmb { y } _ { t } ^ { k } , \pmb { \theta } _ { ( j ) } \big ) \right] } \end{array}$ , where $\textstyle b = { \frac { 1 } { K } } \sum _ { k = 1 } ^ { K } G ^ { k }$

Through utilization of the Monte Carlo method, an unbiased approximation of the true policy gradient is obtained. However, due to the stochastic nature of both the policy and process dynamics, the gradient may observe high variance. In order to reduce the variance of approximation, a baseline b is introduced.<sup>5</sup> This baseline is formulated directly as the expectation of cost associated with the realizations of the policy. In this manner, the update balances the cost of an action against the expected cost from the current policy.

It is of important note that the parameterization of the policy must be continuously differentiable as prescribed by the policy gradient theorem. Naturally, this lends to application of ANNs for function approximation in this work. Specifically, a recurrent long short-term memory (LSTM) neural network was used for parameterization of the control policy. Recurrent LSTM neural networks have demonstrated utility in dynamic stochastic control problems with extension to systems characterized by partial observability.<sup>2</sup> General detail of the mathematical operations specific to LSTMs can be found in the following works,<sup>34,35</sup> with figurative description of the network used in this application provided by Section SI.2 of the SI. The investigation utilized the Pytorch 1.3.1 framework and first-order gradient ascent method Adam to train the LSTM network proposed. The network structure was composed of two hidden layers, each with 20 LSTM cells. A leaky rectified linear unit (ReLU) activation function was applied across both hidden layers and a ReLU6 activation function was applied across the output layer, naturally bounding the output prediction. For a random variable z, the ReLU6 transformation is described as:

$$
\operatorname{ReLU6} (z) = \min (\max (0, z), 6)\tag{3.2.5}
$$

The network designed in the context of this work, predicts the mean $\left( \mu _ { t } \right)$ and standard deviation $\mathbf { \Pi } ( \pmb { \sigma } _ { t } )$ of a unimodal multivariate normal distribution. This distribution describes the conditional density function representative of the control policy, such that: $u _ { t } \sim$ $\pi ( \boldsymbol { u } _ { t } | \boldsymbol { y } _ { t } , \boldsymbol { H } _ { t } , \pmb { \theta } ) = \mathcal { N } \big ( \pmb { \mu } _ { t } , \pmb { \sigma } _ { t } ^ { 2 } \big )$ , where $\textstyle H _ { t }$ is a learned parameterization of the history of process states provided by the LSTM cells, and $\sigma _ { t } ^ { 2 }$ is the variance. Here, we formally construct the control policy as stochastic. However, upon deployment of the policy to the real system, the policy may be assumed deterministic through selection of the actions corresponding to the mode (equivalently, the mean) of the multivari ate normal distribution, such that $\pmb { u } _ { t } = \pmb { \mu } _ { t }$

In this section, we have presented an approach to solving the MDP characteristic of a control problem through use of the policy gradient method, reinforce with baseline, in combination with an LSTM network for parameterization of the learned policy. In the following we introduce an approach to policy learning, namely maximum entropy IRL (MaxEnt IRL), which utilizes existing process data to learn from demonstration. Conceptually, this approach is commonly known as AL.

## 3.3 | AL via IRL

AL via IRL is a general approach to policy learning from demonstration (i.e., process data). The benefits to such an approach are twofold. First, AL via IRL provides a parameterization of the existing control policy expressed in the process data. Second, it facilitates RL-based policy learning under the “real” process objective as it provides an initial policy to hot-start the RL procedure. Otherwise, initially, the agent (or controller) will explore the control action space randomly, which results in a data hungry and time-consuming approach. These benefits are exploited by the framework proposed in Section 2.3 as detailed by Figure 2.

The foundational IRL algorithms construct the reward function R : <sup>Y</sup> ℝ as a linear combination of state features representative of the system state, $\varphi \in \mathcal { R } ^ { d \times 1 }$ , such that:

$$
R = \alpha_ {1} \varphi_ {1} + \alpha_ {2} \varphi_ {2} + \dots + \alpha_ {\mathrm{d}} \varphi_ {d}
$$

<sub>ð</sub>3:3:1<sub>Þ</sub>

where $\alpha _ { \mathrm { i } }$ are feature weightings and $\varphi _ { i } { : } \mathsf { Y } \to \mathbb { R }$ explicitly represent the system state (<sup>y</sup>), but also implicitly encode control objectives.

Typically, φ are hand designed based on process and control task knowledge.<sup>29</sup> Knowledge of process objectives can also be applied to place bounds on the weights α in the reward function; however, this may not always be desired as one could assert technical bias on the problem and reduce the feasible region. From this definition of the reward function R(α, <sup>y</sup>), consequent reformulation of the policy optimization objective J(τ) in Equation (3.1.9) yields

$$
J (\pmb {\tau}) = \mathbb {E} _ {\tau \sim p (\pmb {\tau} | \theta)} \left[ \sum_ {t = 1} ^ {T} \gamma^ {t - 1} R (\pmb {\alpha}, \pmb {y} _ {t}) \right]\tag{3.3.2}
$$

$$
J (\boldsymbol {\tau}) = \sum_ {i = 1} ^ {d} \alpha_ {i} \mathbb {E} _ {\tau \sim p (\boldsymbol {\tau} | \theta)} \left[ \sum_ {t = 1} ^ {T} \gamma^ {t - 1} \varphi_ {i} (\mathbf {y} _ {t}) \right]\tag{3.3.3}
$$

This may be further decomposed through definition of trajectory features, $\upsilon _ { 1 } ,$ such that for the discounted case:

$$
v _ {i} ^ {\gamma} = \sum_ {t = 1} ^ {T} \gamma^ {t - 1} \varphi_ {i} (\mathbf {y} _ {t})\tag{3.3.4}
$$

$$
J (\pmb {\tau}) = \sum_ {i = 1} ^ {d} \alpha_ {i} \mathbb {E} _ {\tau \sim p (\pmb {\tau} | \theta) [ v _ {i} ^ {\gamma} ]}\tag{3.3.5}
$$

$$
J (\pmb {\tau}) = \pmb {\alpha} ^ {T} \mathbb {E} _ {\tau \sim p (\pmb {\tau} | \theta) [ \pmb {v} ^ {\gamma} ]}\tag{3.3.6}
$$

where $\pmb { \alpha } \in \mathbb { R } ^ { d \times 1 }$ and $\pmb { \nu } ^ { \gamma } \in \mathbb { R } ^ { d \times 1 }$ . Equivalently, undiscounted trajectory features υ may be recovered by setting $\gamma = 1$ . The characterization of a policy and process trajectory in terms of υ enables RL to learn from multiple, distributed trajectories and reduces the problem to learning feature weights α<sup>\*</sup>.<sup>29,30</sup> Conceivably, a number of different reward functions exist that recover the desired behavior. The current study uses the MaxEnt IRL framework proposed by Ziebart et al.,<sup>30,36</sup> which proceeds in identification of α via a probabilistic approach as underpinned by the principle of maximum entropy.

## 3.4 | Maximum entropy IRL

In AL, we are interested in learning a policy as described by a conditional probability density function $\pi ( \boldsymbol { u } _ { t } | \boldsymbol { y } _ { t } , \cdot \dot$ ), such that upon deployment of the policy to the real system, the process observes the same evolution as that described by process data (see Equation (3.1.6)). Explicitly, the investigation learns the expert's policy expressed by process trajectories $\mathbf { T } = \left[ \pmb { \tau } _ { 1 } ^ { E } , . . . , \pmb { \tau } _ { K } ^ { E } \right]$ as characterized by trajectory features, $\left\{ \pmb { \upsilon } _ { k } ^ { E } \right\}$ , where $k = 1 , . . . , K$ . MaxEnt ${ \mathsf { I R L } } ^ { 3 0 }$ is an established method and poses solution to the problem of learning such an approximate policy. It learns a reward function that maximizes the likelihood of observing the demonstrated trajectories Τ given an accurate model of the process dynamics. Further discussion is provided in SI.3. It follows that the log-probability of observing a given trajectory τ is proportional to the cumulative undiscounted reward observed between a start and terminal state,<sup>36</sup> such that:

$$
p (\boldsymbol {\tau} \mid \boldsymbol {\alpha}) = \frac {\exp \left\{\boldsymbol {\alpha} ^ {T} \boldsymbol {v} (\boldsymbol {\tau}) \right\}}{Z (\boldsymbol {\alpha} , \cdot)}\tag{3.4.1}
$$

where $\pmb { \upsilon } = [ \upsilon _ { 1 } , \upsilon _ { 2 } , . . . , \upsilon _ { d } ] ,$ , and $\begin{array} { r } { Z ( \alpha , \cdot ) = \sum _ { \tau \in \mathbf { T } } \mathrm { e } \times \mathrm { p } \{ \alpha ^ { T } \upsilon ( \tau ) \} } \end{array}$ is the partition function, which enforces normalization of the distribution. Formally, the approach prescribes that each of the demonstrations, ${ \pmb { \tau } } ^ { E } \in { \bf T } ,$ are independently and identically distributed such that the likelihood of observing the set of trajectories, Τ, expressed in process data is:

$$
p (\mathbf {T} | \boldsymbol {\alpha}) = \prod_ {k = 1} ^ {K} p \left(\boldsymbol {\tau} _ {k} ^ {E} | \boldsymbol {\alpha}\right) = \prod_ {k = 1} ^ {K} \frac {1}{Z (\boldsymbol {\alpha} , \cdot)} \exp \left\{\boldsymbol {\alpha} ^ {T} \boldsymbol {v} _ {k} ^ {E} \right\}\tag{3.4.2}
$$

where $Z ( \pmb { \alpha } , \cdot )$ is assumed constant for all $\pmb { \tau } ^ { E } \in \mathbf { T } ; ^ { 3 0 }$ and p( α) is the like lihood of observing the set of demonstrations. Under the maximum entropy formulation ,<sup>30,31,36,39</sup> optimal solution of the feature weights, $\pmb { \alpha } ^ { * }$ is:

$$
\boldsymbol {\alpha} ^ {*} = \operatorname{argmax} _ {\boldsymbol {\alpha}} p (\mathbf {T} | \boldsymbol {\alpha}) = \operatorname{argmax} _ {\boldsymbol {\alpha}} \prod_ {k = 1} ^ {K} p \left(\tau_ {k} ^ {E} | \boldsymbol {\alpha}\right)\tag{3.4.3}
$$

The gradient of the log-likelihood objective (Equation (3.4.3)) with respect to feature weights, α, is formulated as:

$$
\nabla_ {\boldsymbol {\alpha} _ {(i)}} \sum_ {k = 1} ^ {K} \log p \left(\boldsymbol {\tau} _ {k} ^ {E} | \boldsymbol {\alpha} _ {(i)}\right) = \frac {1}{K} \sum_ {k = 1} ^ {K} \boldsymbol {v} _ {k} ^ {E} - \nabla_ {\boldsymbol {\alpha} _ {(i)}} \log Z (\boldsymbol {\alpha} _ {(i)}, \cdot)\tag{3.4.4}
$$

$$
\nabla_ {\boldsymbol {\alpha} _ {(i)}} \log Z (\boldsymbol {\alpha} _ {(i)}, \cdot) = \mathbb {E} _ {\boldsymbol {\tau} ^ {\pi} \sim p (\boldsymbol {\tau} ^ {\pi} | \boldsymbol {\alpha} _ {(i)}, \boldsymbol {\theta} ^ {*})} [ \boldsymbol {v} ^ {\pi} ]\tag{3.4.5}
$$

where $\nabla _ { \pmb { \alpha } _ { ( i ) } } \log Z ( \pmb { \alpha } _ { ( i ) } \mathrm { \ , \ } \cdot )$ is estimated via policy optimization in the underlying MDP to find a policy, $\pi ( \cdot , \pmb \theta ^ { \ast } ) ,$ , that maximizes the following modified objective, and then subsequently performing Monte Carlo realizations of the solution policy under the process dynamics to provide sample trajectories, $\pmb { \xi } = \left[ \pmb { \tau } _ { 1 } ^ { \pi } , . . , \pmb { \tau } _ { N } ^ { \pi } \right]$ characterized by υ<sup>π</sup> 	 , where $\pmb { \mathscr { n } } = 1 , . . . , N .$ This is also discussed further in Section SI.3. Equations (3.4.4) and (3.4.5) suggest that the MaxEnt IRL problem finds a weight vector, $\pmb { \alpha } ^ { * } ,$ which minimizes the differences between the expected trajectory features of the learned policy and that which is demonstrated. Gradient-based optimization methods may be deployed to find solution, ${ \pmb { \alpha } } ^ { * } ,$ , by stepping parameter values, α, in the direction of the gradient.<sup>30,36</sup> This work utilizes the first-order gradient ascent method (Equation (3.4.6)).

$$
\boldsymbol {\alpha} _ {(i + 1)} = \boldsymbol {\alpha} _ {(i)} + \kappa \nabla_ {\boldsymbol {\alpha} _ {(i)}} \log p (\mathbf {T} | \boldsymbol {\alpha} _ {(i)})\tag{3.4.6}
$$

where κ is a learning rate. The problem formulated here constitutes a bi-level optimization, with the upper level task approached by MaxEnt IRL and the lower level task handled by the policy gradient method reinforce. In each iteration i of the upper MaxEnt IRL problem, a new reward function, $R ( \alpha _ { ( i ) } , \cdot ) ,$ is abstracted. The underlying MDP is subsequently solved by policy optimization and estimation of the partition function and $\mathbb { E } [ \pmb { \upsilon } ^ { \pi } ]$ provided. The reinforce method and the approach to solving the lower level optimization task is detailed by Algorithm 1.

It should be noted that the approaches to policy optimization provided by PPO and entropy regularization could provide further stability in learning and accuracy in estimation of the partition function, respectively. In view of the length of the horizon specific to many control tasks, discounted trajectory features $\pmb { \upsilon } ^ { \gamma } ,$ , as described by Equation (3.3.4), should be used rather than the undiscounted features. This establishes the upper MaxEnt IRL task as a nonconvex optimization<sup>37</sup> but provides performance improvements in the lower level policy optimization task. Algorithm 2 details the MaxEnt IRL algorithm further.

## MaxEnt inverse reinforcement learning

Input: Initialize: a policy $\pi _ { ( 0 ) } ^ { A }$ with initial parameters $\theta _ { ( 0 ) } ;$ a weight vector α; state feature functions φ(<sup>x</sup>); trajectory features representative of the demonstrated trajectories $\pmb { \upsilon } ^ { E } ;$ maximum iterations $N _ { m a x } ;$ learning rate κ;

Output: optimal weights $\pmb { \alpha } ^ { * }$ and agent parameterization of the demonstrated policy $\pi _ { \mathfrak { p o } } ( \mathfrak { g } _ { ( \mathtt { k } _ { 0 } ) } ) ,$ for further policy improvement in online learning.

for $\pmb { n } = 1 , . . . , N _ { m a x } \mathbf { d } \pmb { \mathrm { o } }$

1. Perform policy optimization of $\pi _ { ( n - 1 ) } ^ { A }$ under the current reward function $R ( \pmb { \alpha } _ { ( n ) } )$ via Algorithm 1. Return $\pi _ { ( n ) } ^ { A }$ as solution to the MDP defined.

2. Perform Monte Carlo realization of $\pi _ { ( n ) } ^ { A }$ (via Algorithm S1) to evaluate the policy. Return the trajectory features characteristic of the expected process evolution under the policy $\mathbb { E } \big [ { \pmb { \upsilon } } ^ { \pi _ { ( n ) } } \big ]$

3. Approximate the gradient of the likelihood of observing the demonstrated trajectories with respect to the weights $\begin{array} { r } { \nabla _ { \alpha } \mathsf { l o g } p ( \mathbf { T } | \alpha ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \pmb { \upsilon } _ { k } ^ { E } - \mathbb { E } [ \pmb { \upsilon } ^ { \pi _ { ( n ) } } ] } \end{array}$

4. Perform gradient ascent such that $\alpha _ { ( n + 1 ) } ~ = ~ \alpha _ { ( n ) } -$ + κ logp (Τ| α)

end

## 3.5 | Overview of the proposed methodology

The methodology proposed leverages the large amount of process control data available to industry to learn an RL-based parameterization of a previously implemented control scheme through AL via IRL. This parameterization should express the existing control law as well as the process knowledge of operators provided the available data is sufficiently rich. Once a parameterization is constructed offline, it is deployed as initialization for further RL-based policy improvement (online). This online learning proceeds under a reward function descriptive of the real process objectives. Through this approach, we significantly reduce the computational and technical investment associated with training an RL-based control policy. Specifically, the improvements noted are drawn from the offline section of the framework. Here, we combine simulation with the use of IRL to automate analysis of historical process data. This enables us to directly abstract a reward function, which provides clear preference (discrimination) over controls from: (i) knowledge of the process control task we are concerned with (represented by the basis features, φ, in the reward function); and (ii) empirical observations of the system and its behavior in response to controls (by optimizing the feature weight α). Learning under this reward function provides a parameterization of the existing control scheme expressed in process data. Section 4 presents a number of computational case studies for empirical demonstration of the framework described.

## 4 | COMPUTATIONAL CASE STUDIES

## 4.1 | Introduction to the case studies

The optimization objective of the following studies is set point tracking in a multiple-input, multiple-output (MIMO) control scheme. Specifically, the process is a nonisothermal continuous stirred tank reactor under operation of an endothermic isomerism reaction of the form: A B. The reaction rate temperature dependence is described by the Arrhenius kinetics. Demonstration is provided in the form of process data generated by the action of a PID control scheme, produced via a discrete time Python 3.7.3 implementation. The controlled variables (<sup>y</sup>) are concentration of reagent, $C _ { A } ^ { o b s }$ and temperature of the reactor, $T ^ { o b s }$ . The MVs (<sup>u</sup>) are the temperature of a heating jacket, $T _ { E }$ and concentration of the reagent in the input stream, $C _ { A 0 } .$ Bounds are placed upon the absolute values of the control space. Definition of process variable follows:

$$
\boldsymbol {y} = \left[ C _ {A} ^ {o b s}, T ^ {o b s} \right] ^ {T}\tag{4.1.1a}
$$

$$
\boldsymbol {x} = \left[ C _ {A}, T \right] ^ {T}\tag{4.1.1b}
$$

$$
\boldsymbol {u} = \left[ C _ {A 0}, T _ {E} \right] ^ {T}\tag{4.1.1c}
$$

In the case studies presented, the process model is of deviation variable form and was derived from first principles. The deviation vari able $, z ^ { * }$ of random variable, z is expressed as:

$$
Z ^ {*} = Z - Z _ {s s}\tag{4.1.2}
$$

where $z _ { s s }$ is the previous steady-state value of z. Process stochasticity (disturbance) is assumed zero mean Gaussian, as is the nature of sys tem observation. Therefore, approximation of the true underlying process dynamics takes the form of a system of stochastic differential equations, such that

$$
\boldsymbol {x} _ {t + 1} ^ {*} = \boldsymbol {x} _ {t} ^ {*} + h (\boldsymbol {x} _ {t} ^ {*}, \boldsymbol {u} _ {t} ^ {*}) d t + \delta (\boldsymbol {x} _ {t} ^ {*}) d W _ {t}\tag{4.1.3a}
$$

$$
\mathbf {y} _ {t + 1} ^ {*} = g \left(\mathbf {x} _ {t + 1} ^ {*}\right)\tag{4.1.4a}
$$

where function $h ( \cdot )$ is descriptive of the underlying process dynamics; δ( ) the magnitude of disturbance, as described by the Wiener process, ${ W _ { t } } ^ { 3 8 } ;$ and, g( ) describes the nature of system observation. In the following studies,

$$
h (\boldsymbol {x} _ {t} ^ {*}, \boldsymbol {u} _ {t} ^ {*}) = \left[ \begin{array}{c c} - 3. 9 9 7 & - 0. 4 4 6 \\ - 6. 0 9 2 & - 1. 5 8 1 \end{array} \right] \boldsymbol {x} _ {t} ^ {*} + \left[ \begin{array}{c c} 0. 5 0 0 & 0 \\ 0 & 0. 3 0 5 \end{array} \right] \boldsymbol {u} _ {t} ^ {*}\tag{4.1.3b}
$$

$$
\delta (\mathbf {x} _ {t} ^ {*}) = \left[ \begin{array}{c c} 0. 5 0 0 & 0 \\ 0 & 0. 3 0 0 \end{array} \right] \mathbf {x} _ {t} ^ {*}\tag{4.1.3c}
$$

$$
g \big (\boldsymbol {x} _ {t + 1} ^ {*} \big) = \left[ \begin{array}{c c} 1 + \mathcal {N} (0, 0. 0 2 5) & 0 \\ 0 & 1 + \mathcal {N} (0, 0. 0 2 5) \end{array} \right] \boldsymbol {x} _ {t + 1} ^ {*}
$$

<sub>ð</sub>4:1:4b<sub>Þ</sub>

and the Euler Maryuama method was utilized for system integration.<sup>38</sup> The SI provides formal derivation and parameter values. Given the formulation of the MIMO problem, the investigation is concerned with controlling the evolution of error, ε within both the temperature, $T ^ { o b s }$ and reagent concentration, $C _ { A } ^ { o b }$ control loops.

## 4.2 | Design of state features for AL

The introduction provided in Section 3.4 outlines a framework for learning the weight vector ${ \pmb { \alpha } } ^ { * } ,$ which provides a linear mapping from state representations, φ, to scalar cost. Further, for a given representation, a set of possible process trajectories exist, which match the counts of state features (trajectory features) of the existing policy. Therefore, design of $\pmb { \varphi }$ should consider both the process, optimization objectives and restriction of the possible set of trajectories. As a result, this work proposes the use of three types of state features, all of which provide consistent control objectives temporally and utilize knowledge of the underlying process control task.

## 4.2.1 | Type I

The first state feature proposed is encapsulated by the radial basis function (RBF). The RBF provides a similarity measure and allocates exponentially lower cost or greater value for those control policies which achieve set point tracking. The feature is formulated as:

$$
\hat {\varepsilon} = \frac {y _ {s p} - y}{y _ {s p} - y _ {s s}}\tag{4.2.1}
$$

$$
\varphi_ {l} (\hat {\varepsilon}) = e ^ {- (\beta \hat {\varepsilon}) ^ {2}}\tag{4.2.2}
$$

where $\pmb { \gamma _ { s s } }$ is the previous observed steady state of the system, $\gamma _ { s p }$ is the desired set point, $\beta$ is the shape parameter and $\varphi _ { I } ( \hat { \varepsilon } ) = [ 0 , 1 ]$ . The closer the value of $\beta$ to zero, the greater the offset tolerated and the denser the reward landscape. Conversely, higher values of $\beta$ provide exponentially greater rewards for trajectories closer to the set point, but a sparser reward landscape. In the following case studies, the investigation utilized $\beta = 1 0$

## 4.2.2 | Type II

Although the Type I feature is an absolute measure of control performance, alone it does not fully characterize the evolution of system response. Furthermore, the set of possible process trajectories, which could match the representation of the demonstrated policy $\boldsymbol { v } ^ { E }$ is large. To restrict the possible set, Type II and III features take inspiration from the PID control law, which at a given time is a linear combination of the error, $\varepsilon = \boldsymbol { y } _ { s p }$ y, in the control loop at the current time point (proportional), the manner in which the error has evolved over time (integral) and the projected evolution of error in the future (deriva tive). Hence, the Type II state feature proposed intends to quantify how the absolute error in a control loop evolves temporally. As such, Type II state features are described as:

$$
\varphi_ {I I} (\hat {\varepsilon}) = \int_ {0} ^ {t} | \hat {\varepsilon} | d t \approx \sum_ {j = 1} ^ {t c} | \hat {\varepsilon} | \Delta t\tag{4.2.3}
$$

where Δt is equivalent to the sampling time or times at which con trol is provided (in this work, the two are synonymous), refers to the absolute value; j the discrete time index and tc the current time point. The absolute magnitude of the error provides clear control objective regardless of whether the error ^ε is positive or negative in value. If this was not taken, actions that decrease error in the control loop may be penalized or rewarded in an RL setting depending upon whether the integral of the error becomes positive or negative as a result.

## 4.2.3 | Type III

The design of Type III state features aims to quantify how the error in the control loop may evolve into the future. As a result, the feature approximates the derivative of the error in the control loop at the sampled time:

$$
\varphi_ {I I I} (\hat {\varepsilon}) = \frac {d | \hat {\varepsilon} |}{d t} \approx \frac {| \hat {\varepsilon} _ {t c} | - | \hat {\varepsilon} _ {t c - 1} |}{\Delta t}\tag{4.2.4}
$$

where tc 1 is the previous discrete time index. In view of the proposed state features, the investigation is able to characterize control trajectories and provide direct and consistent control objective. As a result, the reward function R of the MDP described is specified as

$$
\begin{array}{r l} & R = \alpha_ {1} \varphi_ {I} \Big (\hat {\varepsilon} _ {C _ {A} ^ {*}} \Big) + \alpha_ {2} \varphi_ {I} (\hat {\varepsilon} _ {T ^ {*}}) + \alpha_ {3} \varphi_ {I I} \Big (\hat {\varepsilon} _ {C _ {A} ^ {*}} \Big) + \alpha_ {4} \varphi_ {I I} (\hat {\varepsilon} _ {T ^ {*}}) + \alpha_ {5} \varphi_ {I I I} \Big (\hat {\varepsilon} _ {C _ {A} ^ {*}} \Big) \\ & \quad + \alpha_ {6} \varphi_ {I I I} (\hat {\varepsilon} _ {T ^ {*}}) \end{array}\tag{4.2.5}
$$

## 4.3 | Case study definitions

Three case studies demonstrate the use of the framework in different contexts and control tasks. Table 1 details the specific experimental setup. Case Study I demonstrates the framework proposed for deployment when subjectively near optimal control is provided by an existing control scheme. Case Study II demonstrates the framework is still effective when the control demonstrated by an existing scheme is subjectively suboptimal. Case Study III explores the potential to transfer knowledge within the framework in order to aid efficiency in learning on different control tasks.

control policy is that of a well-tuned PID controller (PID1 as detailed by the SI).

Utilizing 500 Monte Carlo realizations of the PID1 policy, the methodology was able to generate an informative dataset and subsequently characterize the policy using the six basis features presented in Equation (4.2.5), with $\gamma = 0 . 9 9$ and $\tau = 5 0$ indicates the length of the discrete-time finite horizon. The trajectory feature expectations of PID1 are outlined in Table 2.

## 5.1.1 | Results of AL via MaxEnt IRL

## 5 | RESULTS AND DISCUSSION

The purpose of this case study is to construct an RL controller which learns from demonstration provided by a near optimal control policy and then to improve it further. As such, we demonstrate the full utility of the offline-online framework proposed. First, offline learning under MaxEnt IRL is deployed to find a linear combination $\pmb { \alpha } ^ { * }$ of state features, which infers and encodes control objectives into a feedback signal or reward function. Under this reward function, a parameterization of the control policy expressed in process data is learned in order to match the demonstrated process behavior as characterized through expected trajectory features. The learned parameterization is then improved under the real process objective, which in this case is pure tracking. Here the demonstrated

## 5.1 | Case Study I—Learning from near optimal demonstrations

From Table 2, it is concluded that under the characterization of the PID1 policy $\pmb { \upsilon } ^ { \gamma , E }$ , Algorithm 2 was able to learn an agent parameterization of the demonstrated policy (i.e., PID controller). This was achieved after just four iterations of the algorithm. Each iteration is composed of solving an MDP via RL (detailed by Algorithm 1) and then updating the weight vector α via Equation (3.4.4). The hyper parameters for Algorithm 2 and each iteration are detailed by the SI. It is worth reiterating that there is a set of possible policies, which observe the same expected trajectory feature counts $\mathbb { E } \left[ \pmb { \upsilon } ^ { \gamma , E } \right]$ as that of the demonstrated policy. In the context of this work, further restricting the possible set is not necessary; however, introduction of further state features φ would facilitate such. Given that φ compose the reward function and all express inherent set point tracking objectives, intuitively, any of the policies from the possible set, which match the trajectory features of the demonstrated policy should provide good initialization for further policy improvement. The learned weight vector α may also be interpreted and provide insight into the dynamics of the respective control loops.

T A B L E 1 Conditions of design for the case studies detailed. The real initial state of the controlled variables $\pmb { x } _ { 0 }$ is drawn from the respective distributions. The set point $\gamma _ { \mathsf { s p } } ^ { \ast }$ details the new setpoint of the respective control variables as set at t = 0

<table><tr><td>Case study</td><td>System parameter</td><td>Concentration ( $C_A^*$ ) control loop</td><td>Temperature ( $T^*$ ) control loop</td></tr><tr><td rowspan="2">I</td><td>Initial state distribution  $\bar{\rho}(\mathbf{x}_0)$ </td><td> $\mathcal{N}$  (0, 0.25)</td><td> $\mathcal{N}$  (0, 0.75)</td></tr><tr><td>Set point  $Y_{sp}^*$ </td><td>-1</td><td>4</td></tr><tr><td rowspan="2">II</td><td>Initial state distribution  $\bar{\rho}(\mathbf{x}_0)$ </td><td> $\mathcal{N}$  (0, 0.25)</td><td> $\mathcal{N}$  (0, 0.75)</td></tr><tr><td>Set point  $Y_{sp}^*$ </td><td>1</td><td>4</td></tr><tr><td rowspan="2">III</td><td>Initial state distribution  $\bar{\rho}(\mathbf{x}_0)$ </td><td> $\mathcal{N}$  (0, 0.25)</td><td> $\mathcal{N}$  (0, 0.75)</td></tr><tr><td>Set point  $Y_{sp}^*$ </td><td>-2.5</td><td>3</td></tr></table>

T A B L E 2 The expected discounted trajectory features of PID1 $( \pmb { \upsilon } ^ { \gamma , E } )$ and the policy learned through AL $( \pmb { \upsilon } ^ { \gamma , \pi } )$ , and IRL's feature weight (α<sup>\*</sup>) generated in CS I. $\boldsymbol { \Upsilon } ^ { \ast } -$ Type indicates the type of trajectory feature and the respective control loop error

<table><tr><td rowspan="2"></td><td colspan="6">Trajectory features</td></tr><tr><td> $C_A^* - I$ </td><td> $T^* - I$ </td><td> $C_A^* - II$ </td><td> $T^* - II$ </td><td> $C_A^* - III$ </td><td> $T^* - III$ </td></tr><tr><td> $\mathbb{E}[v^{\gamma,E}]$ </td><td>21.63</td><td>20.68</td><td>4.08</td><td>7.93</td><td>-22.87</td><td>-22.43</td></tr><tr><td> $\mathbb{E}[v^{\gamma,\pi}]$ </td><td>21.41</td><td>20.76</td><td>4.31</td><td>7.03</td><td>-22.28</td><td>-22.71</td></tr><tr><td> $\alpha^*$ </td><td>0.137</td><td>0.652</td><td>-0.067</td><td>-0.630</td><td>-0.194</td><td>-0.343</td></tr></table>

Abbreviations: AL, apprenticeship learning; IRL, inverse reinforcement learning.

(B)  
![](images/f95520ca94b232a1ff8de617af50647431f5826d8c8ef9d85e1dd6522424177c.jpg)

(C)  
![](images/d33545faf5e86d6d27941f757b157d5df9c597ea15e4b9e56a614307858bf74d.jpg)

(D)  
![](images/5700ba09f357d56ba9601c0cf97d7f2864ebd65eb180ebecc64954e1231a2340.jpg)

![](images/f6d4c1978dd6a1057cb2dd413cbe5668c5d5cb5b4d73da2eb6acc11ae157cdef.jpg)  
F I G U R E 3 Optimal policy of the agent in Case Study I. (A,B) Control and system response of the concentration control loop and of the temperature control loop, respectively. (C,D) Zoomed system response in the concentration control loop and in the temperature control loop, respectively. $\pi ^ { A }$ and $\pi ^ { E }$ indicate the policy of the agent (after online learning) and the PID, respectively. Solid line represents the mean control response and the shaded regions indicate the standard deviation. Line colors of manipulated variables: blue— $- \pi ^ { A } ;$ light $\mathsf { g r e e n { - } } \pi ^ { E } .$ Line colors of control variables: red— $- \pi ^ { A } ;$ dark $\mathrm { g r e e n } - \pi ^ { E } .$ . Line color of set points: orange [Color figure can be viewed at wileyonlinelibrary.com]

The state features that are specific to the temperature control loop receive a greater weight than the concentration control loop. This is likely reflective of the endothermic nature of reaction and the relative changes of set point in the temperature loop and concentration loop. Compared to changing reactant concentration, an increase in reactor temperature T will likely shift reaction equilibrium more significantly in a manner to increase consumption of reagent. As a result, the system dynamics act in a way to aid the set point change in the concentration control loop. Hence, greater weighting is allocated to control of the temperature control loop.

In this section, we show the utility of the offline learning method proposed in the context of learning by demonstration (or AL). Subsequently, we demonstrate how online learning may be deployed for further policy improvement.

## 5.1.2 | Online learning and optimal control

Further improvement of the initial policy (Section 5.1.1) utilizes Algorithm 1 and a real process reward function shown as Equation (5.1.2.1), which expresses pure set point tracking objective

$$
R = \varphi_ {I} \left(\hat {\varepsilon} _ {C _ {A} ^ {*}}\right) + \varphi_ {I} \left(\hat {\varepsilon} _ {T ^ {*}}\right)\tag{5.1.2.1}
$$

Here, the parameter $\beta$ in $\varphi _ { I }$ (Equation (4.2.2)) is retuned to ensure that high performance set-point tracking is achieved $( \beta = 3 0 )$ . The final result of the policy obtained is displayed in Figure 3.

Examination of Figure 3(A) describes the control policies of the agent and PID1 within the concentration control loop. Given the initialization provided by IRL, further online RL-based policy improvement learns a control observably similar but relatively smoother, to that demonstrated by the PID controller. Explicitly, the policy improvement was provided by two rounds of online learning, with 10 training iterations (epochs) per round. As a result, the agent is able to facilitate a system response, which meets set point faster with less overshoot observed than using the PID controller (shown in Figure 3 (C)). Similar observations are made in analysis of Figure 3(B,D), which demonstrate the response of the temperature control loop. In this case, the online updated RL yields a better temperature response characterized by a fast rise time with no observable overshoot.

## 5.2 | Case Study II—Learning from suboptimal demonstrations

In Case Study II, the demonstrations (process data) are derived from a second PID controller (PID2 detailed by the SI). Compared to Case Study I, the demonstrations provided by the PID controller here are of an overdamped control response, which subjectively appears suboptimal.

T A B L E 3 The expected discounted trajectory features of the PID2 $( \pmb { \upsilon } ^ { \gamma , E } )$ and the policy learned through $\mathsf { A L } \left( \pmb { \upsilon } ^ { \gamma , \pi } \right) ,$ , and IRL's feature weight $( { \pmb { \alpha } } ^ { * } )$ generated in CS I. $\boldsymbol { \Upsilon } ^ { \ast } -$ Type indicates the type of trajectory feature and the respective control loop error

<table><tr><td rowspan="2"></td><td colspan="6">Trajectory features  $v$ </td></tr><tr><td> $C_A^* - I$ </td><td> $T^* - I$ </td><td> $C_A^* - II$ </td><td> $T^* - II$ </td><td> $C_A^* - III$ </td><td> $T^* - III$ </td></tr><tr><td> $\mathbb{E}[v^{y,E}]$ </td><td>13.76</td><td>8.52</td><td>8.02</td><td>15.53</td><td>-22.49</td><td>-20.71</td></tr><tr><td> $\mathbb{E}[v^{y,\pi}]$ </td><td>16.41</td><td>7.10</td><td>6.46</td><td>13.29</td><td>-21.82</td><td>-18.79</td></tr><tr><td> $\alpha^*$ </td><td>-0.259</td><td>-0.182</td><td>-0.545</td><td>-0.093</td><td>-0.545</td><td>-0.545</td></tr></table>

Abbreviations: AL, apprenticeship learning; IRL, inverse reinforcement learning.

![](images/4feb83fc0f0fccd8ad6f5a481e1da20c41b25776787c15dbed3a3b179bb04f82.jpg)

(B)  
![](images/b4d4ba3a9233bf5139b6738a12323cb25f8485d317f64a22af391a807907ab58.jpg)  
F I G U R E 4 System response over the first 30 control interactions from the policy learned from demonstration during apprenticeship learning (AL) in Case Study II. (A,B) System response in the concentration control loop and the temperature control loop, respectively. $\pi ^ { A }$ and $\pi ^ { E }$ indicate the response associated with the policy of the agent (after offline learning) and that demonstrated, respectively. Solid line represents the mean control response and the shaded regions indicate the standard deviation. Line colors of control variables: red— $- \pi ^ { A } ;$ dark $\mathrm { g r e e n } - \pi ^ { E } .$ . Line color of set points: orange [Color figure can be viewed at wileyonlinelibrary.com]

(A)  
![](images/9bf13ed77d71e3cbce04ca21f14ce74d0e890250a3e557437a49a886a944e6ac.jpg)

(B)  
![](images/34f2253c80932a440df97055738209c3ef0dec637a9364b07b1534a1098a7557.jpg)

(C)  
![](images/23baa2761464dc70731045f0e0d0c1115ac139d9ab7f3547dba6238acf8f8f36.jpg)

(D)  
![](images/96981811c090b5cc6ca494c29e079b323842ca20be9c2f22c55ba094eebd5590.jpg)  
F I G U R E 5 Optimal policy of the agent in CS II over the full simulated horizon. (A,B) Control and system response of the concentration control loop and the temperature control loop, respectively. (C,D) Zoom of the system response in the concentration control loop and in the temperature control loop, respectively. $\pi ^ { A }$ and $\pi ^ { E }$ indicate the policy of the agent (after online learning) and the PID, respectively. Solid line represents the mean control response and the shaded regions indicate the standard deviation. Line colors of manipulated variables: $\mathsf { b l u e { \mathrm { - } } } { \boldsymbol { \pi } } ^ { A } ;$ ; lig $\mathrm { g r e e n } - \pi ^ { E } .$ Line colors of control variables: red— $- \pi ^ { A } ;$ dark $\mathsf { g r e e n } - \pi ^ { E } .$ . Line color of set points: orange [Color figure can be viewed at wileyonlinelibrary.com]

T A B L E 4 The expected discounted trajectory features of the PID1 generated in CS III. $\curlyvee ^ { * } .$ -Type indicates the type of trajectory feature and the respective control loop error

<table><tr><td rowspan="2"></td><td colspan="6">Trajectory features  $v$ </td></tr><tr><td> $C_A^* - I$ </td><td> $T^* - I$ </td><td> $C_A^* - II$ </td><td> $T^* - II$ </td><td> $C_A^* - III$ </td><td> $T^* - III$ </td></tr><tr><td> $\mathbb{E}[v^{γ,E}]$ </td><td>16.07</td><td>18.36</td><td>8.08</td><td>8.35</td><td>-21.83</td><td>-22.78</td></tr><tr><td> $\mathbb{E}[v^{γ,π}]$ </td><td>14.00</td><td>18.04</td><td>9.37</td><td>6.50</td><td>-19.94</td><td>-21.06</td></tr><tr><td> $α$ </td><td>0.664</td><td>0.052</td><td>-0.223</td><td>-0.226</td><td>-0.403</td><td>-0.541</td></tr></table>

## 5.2.1 | Results of AL via MaxEnt IRL

In similar fashion to Section 5.1.1, Algorithm S1 was used to characterize the demonstrations from PID2. Table 3 details the resultant trajectory feature expectations $\mathbb { E } \left[ { \pmb { \upsilon } } ^ { \gamma , E } \right]$

Once again, Algorithm 2 facilitates the learning of an agent parameterization of the demonstrated policy in three iterations. It is of note, however, that the methodology was unable to match the trajectory features exactly. Instead, a good approximation of the demonstrated policy was produced. There are two points of discussion here. First, it is likely that the reward function itself is underspecified and further state features, $\varphi ,$ should be proposed. Second, it is possible that the objectives of the demonstrated control policy cannot be described purely as a linear combination of the state features<sup>31</sup>—although the linear approximation in this case is reasonable, given the similarity of the trajectory features.

(B)  
![](images/3055047872006aef0af6c566718d82cf30a9756731e90082dc881c939693f4e6.jpg)

(C)  
![](images/ab10a99a389072d1238ce0628d8ea619e73d4aab11c223af15324dbecb5cb43a.jpg)  
(D)

In this case study, state features relevant to the concentration control loop are allocated the greatest weighting. This is because the set points are changed in the same direction (as detailed by Table 1). Naturally, a rise in reagent concentration will cause a decrease in temperature (endothermic reaction), whilst a rise in temperature will facili tate the conversion of reagent concentration. As the reaction equilibrium is more sensitive to the temperature change, greater weightings must be added to the concentration control loop to reach the new set point.

![](images/0cbbdb8d8f2945c275609dc1a62cfcec5fc5b245c09b2b95f522745e920af2ea.jpg)  
F I GU RE 6 Policy $\pi ^ { A }$ generated as a result of knowledge transfer through apprenticeship learning (AL) and online policy optimization. (A,B) Control and system response of the concentration control loop and the temperature control loop, respectively. (C,D) Zoom of the system response in the concentration control loop and the temperature control loop, respectively. $\pi ^ { A }$ and $\pi ^ { E }$ indicate the policy of the agent (after online learning) and the PID, respectively. Solid line represents the mean control response and the shaded regions indicate the standard deviation [Color figure can be viewed at wileyonlinelibrary.com]

![](images/9eadfac5290053d7ac40174a8ceb48f93c2453925099cbbbcfd52dfcde25e145.jpg)

Furthermore, Type I state features are allocated negative weights, which is unusual. Intuitively, Type I features represent a similarity measure between the current state of the system and the desired set point. Given that the feature value is non-negative $( \varphi _ { I } = [ 0 , 1 ] )$ , a negative reward weighting means that the IRL learnt objective function will prevent the process from reaching the new set point. This is the primarily attributed to the fact that a large proportion of the demonstrations never reached the new set point (Figures 4 and 5) due to the overdamped control response. As AL considers the expert's (i.e., PID controller) actions as a noisily optimal control policy, it will find the optimal solution of weight vector, $\pmb { \alpha } ^ { * } ,$ to reproduce this overdamped control response. Therefore, the current result indicates that if the demonstration data does not contain a good control policy, it is essential to further improve the AL generated policy through online learning.

## 5.2.2 | Online learning and optimal control

As in Section 5.1.2, online learning is performed to improve the AL policy (initialized for RL). Given that a degree of offset was present in both control loops as detailed by Figure 4, two short rounds of RL policy improvement, again consisting of 10 training epochs, proceeded with hand tuning of the parameter β in each round. Figure 5 details the final results of the update RL model. From Figure 5, it is found that the improved policy of the agent $\pi ^ { A } ,$ observes a faster rise time, no overshoot and subjectively better set point tracking than the demonstrated policy (PID). In this way, the methodology shows ability to learn from suboptimal demonstrations and then efficiently improve the learned parameterization of the demonstrated policy through online learning (in this work, 24 min spent online to update the RL).

## 5.3 | Case Study III—Knowledge transfer in learning from demonstration

Finally, Case Study III demonstrates how knowledge transfer from one task improves the efficiency of offline AL for further set points. Here, we again assume the availability of existing demonstrations as described by process data. The control task (set point change) in this study is described by Table 1 and is different to both tasks examined in Case Studies I and II. Again, we would like to learn a parameterization of the control policy (offline) expressed in the process data and then improve it further (online), but we wish to reduce the computational budget associated with offline AL. Thus, we propose to transfer knowledge from a previous study to improve computational and learning efficiency.

Knowledge transfer is in the form of the offline learned policy parameterization, $\pi _ { \mathfrak { p o } } \big ( \mathfrak { \bullet } _ { ( \mathtt { k 0 } ) } \big )$ and weight vector, ${ \pmb { \alpha } } ^ { * } ,$ , from a previous task. Here, knowledge is transferred from Case Study I, given its better PID performance than Case Study II. Both $\pmb { \alpha } ^ { * }$ and $\pi _ { \mathfrak { p o } } \big ( \bullet _ { ( \mathsf { k } _ { 0 } ) } \big )$ from Case Study I are provided as initialization for AL of the new task in Case Study III. Update of this initialization only takes 80 epochs.

Previously, the two studies recovered demonstrated behavior within a total of 300 and 250 epochs of policy optimization, respectively. This reduction in the computational intensity of policy learning demonstrates that the computational burden of AL via IRL—under the current methodology—may be significantly reduced through knowledge transfer. In this study, process data were generated using PID1. Table 4 details the corresponding trajectory feature expectations, $\pmb { \upsilon } ^ { \gamma , E } .$

Given the parameterization as learned via IRL, a further two rounds of 10 epochs of RL enabled further policy improvement online. The results are presented in Figure 6. Figure 6(A,B) highlights how the policy learned under knowledge transfer achieves pure set point tracking with a smoother control policy than that demonstrated by PID1. Once again, Figure 6(C,D) shows that this control policy successfully facilitates a system response with fast rise time, but no overshoot or oscillatory behavior around the set point, as is present in the demonstrations.

## 6 | CONCLUSIONS

In this article, we propose a framework based on AL to learn a control law based on process data, this approach allows us to synthesize a neural network control policy from a previous controller (e.g., PID, MPC, o human controllers) more robustly than with supervised learning. Having learned a parameterization of the control law, subsequent deployment of RL enables further policy improvement by directly interacting with the real process, thus outperforming the existing control law. Here, AL is implemented through IRL. Given the data-driven nature of IRL, the RL-based policy parameterization promises to express the action of the control scheme and process knowledge of the operators. RL is constructed using a policy optimization algorithm, although other methods could be also applied in the future. Based on the case studies, it is con cluded that the proposed framework can effectively extract control information from available process data, transfer knowledge between different cases, and can result in a better optimal control policy effi ciently. It should be noted that we assume the availability of rich informative datasets. If the data is not informative, the framework is unlikely to be effective. Future work will explore implementation of various data augmentation strategies, based on physical knowledge or statisti cal analyses, to artificially synthesize informative datasets.

## DATA AVAILABILITY STATEMENT

Data sharing not applicable to this article as no experimental datasets were generated or analysed during the current study.

## ORCID

Max Mowbray https://orcid.org/0000-0003-1398-0469 Dongda Zhang https://orcid.org/0000-0001-5956-4618

## REFERENCES

1. Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature. 2015:518(7540):529-52933.

2. Heess N, Hunt JJ, Lillicrap TP, Silver D. Memory-based control with recurrent neural networks

3. Kirk DE. Optimal Control Theory: An Introduction. New York: Dover Publications; 1998.

4. Liu D, Wei Q, Wang D, Yang X, Li H. Overview of adaptive dynamic programming. Adaptive Dynamic Programming with Applications in Optimal Control. Basel: Springer International Publishing; 2017:1-33.

5. Petsagkourakis P, Sandoval IO, Bradford E, Zhang D, del Rio-Chanona EA. Reinforcement learning for batch bioprocess optimization. Comput Chem Eng. 2020;133:106649.

6. Lawrence NP, Stewart GE, Loewen PD, Forbes MG, Backstrom JU, Gopaluni RB. Optimal PID and antiwindup control design as a rein forcement learning problem. arXiv:200504539 [cs, eess, math].

7. Spielberg S, Tulsyan A, Lawrence NP, Loewen PD, Bhushan GR. Toward self-driving processes: a deep reinforcement learning approach to control. AIChE J. 2019;65(10):e16689. https://doi.org/ 10.1002/aic.16689.

8. Kim JW, Park BJ, Yoo H, Oh TH, Lee JH, Lee JM. A model-based deep reinforcement learning method applied to finite-horizon optimal control of nonlinear control-affine system. J Process Control. 2020;87: 166-178.

9. Kim Y, Lee JM. Model-based reinforcement learning for nonlinear optimal control with practical asymptotic stability guarantees. AIChE J. 2020;n/a(n/a):e16544. https://doi.org/10.1002/aic.16544.

10. Gottipati SK, Sattarov B, Niu S, et al. Learning to navigate the synthetically accessible chemical space using reinforcement learning. arXiv:200412485 [cs].

11. Angermueller C, Dohan D, Belanger D, Deshpande R, Murphy K, Colwell L. Model-based reinforcement learning for biological sequence design. International Conference on Learning Representations.

12. Gros S, Zanon M. Data-driven economic NMPC using reinforcement learning. arXiv:190404152 [cs].

13. Zanon M, Kungurtsev V, Gros S. Reinforcement learning based on real-time iteration NMPC. arXiv:200505225 [cs, eess].

14. Schulman J, Levine S, Moritz P, Jordan MI, Abbeel P. Trust region pol icy optimization. arXiv:150205477 [cs].

15. Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal pol icy optimization algorithms. arXiv:170706347 [cs].

16. Schulman J, Chen X, Abbeel P. Equivalence between policy gradients and soft Q-learning. arXiv:170406440 [cs].

17. Lehman J, Chen J, Clune J, Stanley KO. ES is more than just a tradi tional finite-difference approximator

18. Sutton RS, McAllester D, Singh S, Mansour Y. Policy gradient methods for reinforcement learning with function approximation. Advances in Neural Information Processing Systems. Neural information processing systems foundation; 2000:1057-1063.

19. Petsagkourakis P, Sandoval IO, Bradford E, Galvanin F, Zhang D, del Rio-Chanona EA. Chance constrained policy optimization for process control and optimization. arXiv:200800030 [cs, eess].

20. Shin J, Badgwell TA, Liu K-H, Lee JH. Reinforcement learning— overview of recent progress and implications for process control. Comput Chem Eng. 2019;127:282-294. https://doi.org/10.1016/j. compchemeng.2019.05.029.

21. Beaulieu S, Frati L, Miconi T, et al. Learning to continually learn. arXiv: 200209571 [cs, stat].

22. Coates A, Abbeel P, Ng A. Apprenticeship learning for helicopter con trol. Commun ACM. 2009;52(7):97-105.

23. Wu Z, Sun L, Zhan W, Yang C, Tomizuka M. Efficient sampling based maximum entropy inverse reinforcement learning with

application to autonomous driving. IEEE Robot Autom Lett. 2020;5 (4):5355-5362.

24. Silver D, Bagnell JA, Stentz A. Learning from demonstration for autonomous navigation in complex unstructured terrain. Int J Robot Res. 2010;29(12):1565-1592.

25. Sutton RS. Reinforcement Learning: An Introduction. 2nd ed: The MIT Press; 2018.

26. Zhang K, Koppel A, Zhu H, Bas¸ar T. Global convergence of policy gradient methods to (almost) locally optimal policies. arXiv:190608383 [cs, eess, math, stat].

27. Simmons-Edler R, Eisner B, Mitchell E, Seung S, Lee D. Q-learning for continuous actions with cross-entropy guided policies. arXiv: 190310605 [cs].

28. Azar NA, Shahmansoorian A, Davoudi M. From inverse optimal control to inverse reinforcement learning: a historical review. Annu Rev Control. 2020;50:119-138

29. Abbeel P, Ng AY. Apprenticeship learning via inverse reinforcement learning. Twenty-First International Conference on Machine Learning - ICML ‘04. ACM Press; 2004:1. https://doi.org/10.1145/1015330. 1015430.

30. Ziebart B, Maas A, Bagnell JA, Dey AK. Maximum entropy inverse reinforcement learning. In: Proceedings of the 23rd National Conference on Artificial Intelligence - Volume 3. AAAI'08. AAAI Press; 2008.

31. Wulfmeier M, Ondruska P, Posner I. Maximum entropy deep inverse reinforcement learning. arXiv:150704888 [cs].

32. Karg B, Alamo T, Lucia S. Probabilistic performance validation of deep learning-based robust NMPC controllers. arXiv:191013906 [cs, eess math].

33. Rohani S. Coulson and Richardson's Chemical Engineering. Volume 3B Process Control. 4th ed. Oxford: Butterworth-Heinemann; 2017.

34. Hochreiter S, Schmidhuber J. Long short-term memory. Neural Com put. 1997;9(8):1735-17380.

35. Colah C. Understanding LSTM Networks; 2015. https://colah.github io/posts/2015-08-Understanding-LSTMs

36. Ziebart B. Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy.

37. Zhou Z, Bloem M, Bambos N. Infinite time horizon maximum causal entropy inverse reinforcement learning. IEEE Trans Automat Contr. 2018;63(9):2787-2802.

38. Mao X. The truncated Euler–Maruyama method for stochastic differ ential equations. J Comput Appl Math. 2015;290(C):370-384.

39. Jaynes ET. Information theory and statistical mechanics. Phys Rev. 1957;106(4):620-630. https://doi.org/10.1103/PhysRev.106.620.

## SUPPORTING INFORMATION

Additional supporting information may be found online in the Supporting Information section at the end of this article.

How to cite this article: Mowbray M, Smith R, Del Rio-Chanona EA, Zhang D. Using process data to generate an optimal control policy via apprenticeship and reinforcement learning. AIChE J. 2021;67(9):e17306. https://doi.org/10. 1002/aic.17306