Review

# Recent Advances in Reinforcement Learning for Chemical Process Control

Venkata Srikar Devarakonda <sup>1</sup>, Wei Sun <sup>2</sup>, Xun Tang <sup>2,</sup>\* and Yuhe Tian <sup>1,</sup>\*

1 Department of Chemical and Biomedical Engineering, West Virginia University, Morgantown, WV 26506, USA

Cain Department of Chemical Engineering, Louisiana State University, Baton Rouge, LA 70803, USA

Correspondence: xuntang@lsu.edu (X.T.); yuhe.tian@mail.wvu.edu (Y.T.)

Abstract: This paper reviews the recent advancements of reinforcement learning (RL) for chemical process control. RL presents a systematic strategy in which the machine learning agent learns a policy of actions based on interactions with the environment. We first provide a brief overview of RL theoretic basis built on Markov decision processes (MDPs) and then move onto its application to process control. With particular interest in chemical processes, we review state-of-the-art research developments on RL for controller tuning and direct control policy learning. This work highlights the importance of safe RL control to incorporate deterministic or probabilistic safety constraints such as constrained MDPs, control barrier functions, etc. We conclude the review with a discussion on some of the outstanding challenges such as sampling efficiency, generalizability, uncertainty, and observability, as well as the emergent and future directions to address these limitations.

Keywords: chemical process control; reinforcement learning; safe reinforcement learning; controller tuning; optimal control

![](images/8f26567b460b0a0cde4713ba0c1b45206526e32876fbf99e857074b94c77c93c.jpg)

Academic Editor: Maria Jose Martin deVidales

Received: 14 April 2025 Revised: 22 May 2025 Accepted: 28 May 2025 Published: 5 June 2025

Citation: Devarakonda, V.S.; Sun, W.; Tang, X.; Tian, Y. Recent Advances in Reinforcement Learning for Chemical Process Control. Processes 2025, 13, 1791. https://doi.org/10.3390/ pr13061791

Copyright: © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/).

## 1. Introduction

Process control is an integral part of the chemical process industry, which investigates the manipulation of process inputs to ensure the desired output behavior under disturbances while addressing potential constraints and regulations, such as safety requirements and environmental specifications [1,2]. Arguably, the most popular feedback control strategies in chemical processes are proportional-integral-derivative (PID) control and model predictive control (MPC). PID controllers calculate the input values based on the error between the measured output values and their setpoint values for setpoint tracking and disturbance rejection. PID control has been widely adopted in industrial practice, under pinned by extensive theoretical analysis on controller tuning to improve its stability and operational performance [3]. However, optimality is normally not guaranteed for PID con trol. On the other hand, MPC advances process control by utilizing a mathematical model to predict the process dynamics over a specified time window [4,5] and couples with optimization algorithms to determine the optimal control actions that satisfy both an objective function and possible constraints. Model development (mechanistic/data-driven), online optimization algorithms, and control under uncertainty remain the core areas for MPC research while yet presenting unresolved challenges for practical implementations [6,7].

Belonging to the same optimal control realm, reinforcement learning (RL) offers another promising alternative for process control, given its capability of data-driven control formulation and superior performance in handling system stochasticity. RL is typically deployed for sequential decision-making problems to optimize a defined cumulative reward, especially in scenarios where limited or no prior knowledge about the system exists. As shown in Figure 1, a generic RL setting involves two key entities: an agent and an environment. The environment is the system of interest such as a chemical production process. The state of the system changes in response to the actions taken by the agent. The agent is the decision-making entity, which is trained to take an action based on the current state of the environment. A reward is then received as feedback to the agent after taking a certain action. The objective of the agent is to maximize the cumulative reward, which is termed as the return. Readers of interest are referred to Sutton and Barto [8] for a comprehensive review on RL algorithmic fundamentals.

![](images/67491c5ead1224263e91b57e51ace6e2edeadce0d57db25d7c2a90836303968a.jpg)  
Figure 1. A schematic of reinforcement learning.

In the context of process control, an RL algorithm can learn the process dynamics through direct interactions with the process system. The reward to the RL agent can be based on cost, setpoint deviation, safety, etc. By iteratively interacting with the environment and analyzing the reward, RL strives to converge to a policy that can decide the optimal control actions based on the pre-specified optimization objective. Since the observed state affects the actions of the agent, RL problems are intrinsically closed-loop. Using the example of control under uncertainty, Petsagkourakis et al. [9] noted that a deep reinforcement learning (DRL)-based control policy would determine the control actions by considering the future impacts of disturbances in a closed-loop manner. This leads to more robust control actions, which are resilient to disturbances while maximizing the reward function.

Several excellent review papers have been presented on the application of RL in process control. Shin et al. [10] conducted a review on the applications of RL for process control and process decision making. The review covered various RL algorithms together with the major challenges for RL implementation. RL was also compared with conventional mathematical programming-based methods (e.g., MPC) for process control and multi-level decision making. Nian et al. [11] reviewed the fundamental Markov decision process, RL solution algorithms (e.g., dynamic programming, Monte Carlo methods, temporal difference learning) and DRL methods. The authors also discussed the correlation of RL with process control and the comparison with traditional control methods. An illustrating example was presented for the control of an industrial pumping system. Faria et al. [12] discussed the detailed RL implementations in terms of simulation-based offline training, transfer learning for policy deployment, and process demonstrations. A review of RL for process control and fault detection was offered by Dogru et al. [13]. Wang et al. [14] presented a tutorial review using a reactor control example to showcase the efficacy of RL, physics-informed RL, transfer RL, and safe RL. Nievas et al. [15] provided a broader perspective of RL applications in various manufacturing sectors including the power industry, robot automation, etc.

This work differentiates from the aforementioned review papers in the following aspects: (i) we focus on RL for the control of chemical process systems, with a particular interest in the advances over the latest five years; (ii) we emphasize the importance of safe RL for process control and review the recent advances; and (iii) we discuss the potential of emerging algorithms (e.g., large language model-enhanced RL) in overcoming key challenges such as sampling efficiency and generalizability. The remainder of this article is structured as follows: Section 2 introduces the basics of RL including the Markov decision processes and representative DRL methods. Section 3 reviews RL-based process control, i.e., RL for autonomous PID tuning and RL for control policy learning. Section 4 focuses on safe RL using deterministic and probabilistic safety constraints and the applications in process control. Section 5 discusses the limitations, challenges, emergent and future directions on RL development. Section 6 presents concluding remarks.

## 2. Reinforcement Learning Basics

For the continuity of this paper, this section briefly reviews Markov decision processes and representative RL methods to solve such problems. These methods will be used as the foundation for RL-based control, to be discussed in Section 3. Readers of interest are referred to [10,11] for comprehensive discussions on RL mathematical fundamentals and solution strategies.

## 2.1. Markov Decision Processes and Useful Definitions

Markov decision processes (MDPs) are used to mathematically model the optimal control problem that the RL algorithms aim to solve [16]. An MDP has four main components: (i) a set of possible states S that characterize the system or environment; (ii) a set of possible actions A taken to manipulate the system; (iii) a probabilistic state transition function $P ( s , a , s ^ { \prime } )$ that defines the probability for the system to progress to future state $s ^ { \prime }$ from current state s, with action $a ;$ and (iv) a reward function $R ( s , a )$ that represents how desirable it is to take a certain action given a particular state. The state transition function in MDPs is considered to follow the Markov property, $\mathrm { i . e . , }$ the evolution of the system state only depends on the current state $s _ { t }$ and the action being performed $a _ { t }$ . Mathematically, the Markov property is defined in Equation (1).

$$
P (s _ {t + 1} | s _ {t}, a _ {t}, s _ {t - 1}, a _ {t - 1}, \dots , s _ {0}, a _ {0}) = P (s _ {t + 1} | s _ {t}, a _ {t})\tag{1}
$$

where t is discrete time, s is the system state, and a is the action. A conceptual example of the problem framed by an MDP is given in discrete time for an infinite horizon below (Equation (2)).

$$
\begin{array}{r l} {J (\pi) = \underset {\pi} {\max}} & {E [ \sum_ {t = 0} ^ {\infty} \gamma^ {t} R (s _ {t}, a _ {t}) | s _ {0} ]} \\ {\mathrm{s.t.}} & {a _ {t} \sim \pi (\cdot | s _ {t})} \\ & {s _ {t + 1} \sim P (\cdot | s _ {t}, a _ {t})} \end{array}\tag{2}
$$

where $\pi$ is a stochastic policy that maps states to actions, and $a _ { t } \sim \pi ( \cdot | s _ { t } )$ means action $a _ { t }$ is sampled from the probability distribution defined by $\pi ( \cdot | s _ { t } ) . \gamma \in [ 0 , 1 )$ is the discount factor, used to ensure that the sum of future rewards is bounded in infinite time. The solution to this problem gives an optimal policy, $\pi ^ { * } = \arg \operatorname* { m a x } J ( \pi )$ , that maximizes the expected sum of discounted future rewards for all time. To solve this problem with RL, it is desirable to define additional functions and parameters. For a given policy $\pi ,$ a trajectory τ of system states and actions can be realized from time t to infinity (Equation (3)). The sum of all future rewards based on this trajectory is the return $G _ { t }$ defined as in Equation (4).

$$
\tau = \left[ s _ {t}, a _ {t}, s _ {t + 1}, a _ {t + 1}, s _ {t + 2}, a _ {t + 2}, \dots \right]\tag{3}
$$

$$
G _ {t} (\tau) = R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} R _ {t + 3} + \dots = \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1}\tag{4}
$$

A state value function $V _ { \pi } ( s )$ is defined by taking the expectation of the return, as given in Equation (5). The state-value function represents the expected return from starting in a state, $s ,$ following a policy, π.

$$
V _ {\pi} (s) = E _ {\pi} [ G _ {t} (\tau) | s _ {t} = s ] = E _ {\pi} [ \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1} | s _ {t} = s ]\tag{5}
$$

Similarly, an action-value function (Q-function) is defined by Equation (6), which represents the expected value of the return from starting in a state s, performing an action $a ,$ and following a policy π. The finite-time horizon formulations can be derived by bounding the upper limit of k.

$$
Q _ {\pi} (a, s) = E _ {\pi} [ G _ {t} (\tau) | s _ {t} = s, a _ {t} = a ] = E _ {\pi} [ \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1} | s _ {t} = s, a _ {t} = a ]\tag{6}
$$

In certain algorithms, it is useful to use an alternative function called the advantage function, $A _ { \pi } ( s , a ) = Q _ { \pi } ( s , a ) - V _ { \pi } ( s )$ , which represents how much better one action is over another. Note that the return function can be defined recursively, $\mathbf { e . g . } , G _ { t } = R _ { t + 1 } + G _ { t + 1 } ,$ and that an optimal state value function can be achieved by following the optimal policy $V ^ { * } ( s ) = \mathrm { m a x } _ { \pi } V _ { \pi } ( s )$ , which can be derived with the Bellman equation (Equation (7)). Following similar steps, the Bellman optimality equation for the Q-function can also be derived (Equation (8)).

$$
V ^ {*} (s) = \max _ {a} E [ R _ {t} + \gamma V ^ {*} (s _ {t + 1}) ]\tag{7}
$$

$$
Q ^ {*} (s, a) = E \left[ R _ {t} + \gamma \max _ {a _ {t + 1}} Q ^ {*} \left(s _ {t + 1}, a _ {t + 1}\right) \right]\tag{8}
$$

The optimal advantage function follows as $A ^ { * } ( s , a ) = Q ^ { * } ( s , a ) - V ^ { * } ( s )$ and provides critical insight, since if $A ^ { * } ( s , a ) = 0$ , the action is then optimal for the current state. These definitions provide the basis for modern RL algorithms, which often estimate the functions of interest $( \mathrm { e . g . , } V ( s )$ or $Q ( s , a ) )$ using neural networks (NNs) with the goal to satisfy the optimality equations through time series updates.

The above formulations assume full observability of the system states, and such MDPs are referred to as fully observable Markov decision processes (FOMDPs). In practice, however, the states of the system or processes might not always be fully observable, leading to partially observable Markov decision processes (POMDPs) or hidden Markov decision processes (HMDPs). In these scenarios, state estimation techniques can be incorporated into the RL algorithm for policy training. For example, in POMDP-based RL problems, observations from the system are used to infer the unmeasurable states, typically in the form of probabilistic distributions. Representative techniques to infer the states for the POMDPs include Bayesian methods, Kalman filter approaches, and neural-network-based state estimators. In Bayesian methods, the posterior belief states are computed using Bayes’ rule [17]. Chenna et al. [18] evaluated the performance of the Kalman filter and recurrent neural networks (RNNs) in state estimation. RNNs have also been used in deep reinforcement learning algorithms for solving POMDPs [19–21]. The use of these techniques allows one to treat a POMDP as an FOMDP since full state estimates become available. Extended discussions on the observability in RL are provided in Section 5.

## 2.2. State-of-the-Art RL Algorithms

A summary of the representative RL algorithms is given in Table 1, from the classic deep Q-learning algorithm to more recent algorithms such as offline-to-online RL and parallelized Q-network, with a brief overview of each algorithm presented in the following text. Readers of interest are referred to the original research publications for more details. We also refer the readers to review articles on the general topic of RL [22–27] for further reading.

Table 1. RL algorithms—an indicative list.

<table><tr><td>Algorithm</td><td>Reference</td><td>Key Features</td><td>Year</td></tr><tr><td>DQN</td><td>Mnih et al. [28]</td><td>Addressed issues of bias and variance in Q-learning algorithms by introducing target Q-networks and exploiting experience replay</td><td>2013</td></tr><tr><td>DPG</td><td>Silver et al. [29]</td><td>A representative departure from stochastic policies that provides the theoretical foundation for deterministic policies in modern RL</td><td>2014</td></tr><tr><td>DDPG</td><td>Lillicrap et al. [30]</td><td>Leveraged the tools from DQN to provide a stable and efficient deterministic policy algorithm for continuous action spaces</td><td>2015</td></tr><tr><td>TRPO</td><td>Schulman et al. [31]</td><td>Policy update by minimizing the KL-divergence between new and old policies</td><td>2015</td></tr><tr><td>PPO</td><td>Schulman et al. [32]</td><td>Clipped surrogate objective to avoid the expensive computation of KL-divergence</td><td>2017</td></tr><tr><td>TD3</td><td>Fujimoto et al. [33]</td><td>Twin independent Q-networks, target policy smoothing with clipped noise, delayed policy update</td><td>2018</td></tr><tr><td>SAC</td><td>Haarnoja et al. [34]</td><td>Entropy term added for soft Bellman equation, stochastic policy</td><td>2018</td></tr><tr><td>RND</td><td>Burda et al. [35]</td><td>Introduces an exploration bonus using the prediction error of a fixed initial neural network for policy improvement</td><td>2018</td></tr><tr><td>CQL</td><td>Kumar et al. [36]</td><td>Learns a conservative Q-function to address limitations in standard off-policy RL methods</td><td>2020</td></tr><tr><td>Distributional SAC</td><td>Duan et al. [37], Ma et al. [38]</td><td>Leverages distribution function to mitigate Q-value overestimation</td><td>2020, 2020</td></tr><tr><td>Diffusion-QL</td><td>Wang et al. [39]</td><td>Represents the policy as a diffusion model to address challenges associated with function approximation on out-of-distribution actions</td><td>2023</td></tr><tr><td>Offline-to-online RL</td><td>Zheng et al. [40]</td><td>Considers the difference between offline and online data for adaptive policy learning</td><td>2023</td></tr><tr><td>PQN</td><td>Gallici et al. [41]</td><td>Uses LayerNorm to accelerate and simplify temporal difference policy training</td><td>2025</td></tr></table>

## • Deep Q-Learning

Q-learning is a classic off-policy method based on temporal difference (TD) learning. The traditional form of Q-learning constructs a table of Q-values for each state–action pair in an exhaustive manner. As the size of the state space increases, this algorithm may become computationally intensive or even intractable. In order to address this issue, deep Q-learning (DQN) is introduced to approximate the Q-function using a deep neural network with weights (θ), i.e., $Q ( s , a ) \approx Q ( s , a , \theta )$

An important feature of DQN is the use of a second neural network as a target network that computes the target Q-values and is updated less frequently to ensure algorithmic stability. DQN also employs replay buffers (or experience replay), which is a set of all previous experiences that the algorithm has explored. It usually stores these previous experiences in the form of state–action–reward–state tuples. Replay buffers can be very large, storing up to millions of experiences, and are used to train the neural network parameters. The update of the DQN is typically achieved by minimizing the Bellman loss function defined as the mean-squared error between the predicted Q-value of the current state–action pair and the current Q-value (Equation (9)). Gradient descent can be utilized to optimize the loss function and update the weights of the neural network. Additionally, the DQN algorithm adopts an ϵ-greedy exploration policy to search the action space. More details can be found in Mnih et al. [28] and Wang et al. [42].

$$
L = E [ (R + \gamma \max _ {a ^ {\prime}} Q (s ^ {\prime}, a ^ {\prime}, \theta^ {-}) - Q (s, a, \theta)) ^ {2} ]\tag{9}
$$

where s and a are the current state and action, $s ^ { \prime }$ and $a ^ { \prime }$ are the next state and action, θ are the weights of the main Q-network, and θ<sup>−</sup> are the weights of the target Q-network. The optimal action can be computed by a = argmax $Q \big ( s , a , \theta \big )$ , since this algorithm is for discrete action spaces.

## • Deterministic Policy Gradient

Deterministic Policy Gradient (DPG) is a policy iteration method where a deterministic policy is developed based on the expected gradient of the action–value function [29]. DPG can be used to map continuous states to continuous actions. It has a Monte Carlostyle of training paradigm. Let $\mu _ { \theta }$ denote the deterministic policy parametrized with θ. $\rho ^ { \mu }$ defines the discounted state distribution in the deterministic case in analogy to the probabilistic distribution in the stochastic case. The objective is to maximize the expected total discounted reward $J ( \mu _ { \theta } )$ , as given in Equation (10). The deterministic policy gradient theorem is formally presented by Equation (11), and can be applied to actor–critic (AC) methods for both on-policy and off-policy learning.

$$
J (\mu_ {\theta}) = E _ {s \sim \rho^ {\mu}} [ r (s, \mu_ {\theta} (s)) ]\tag{10}
$$

$$
\nabla_ {\theta} J (\mu_ {\theta}) = E _ {s \sim \rho^ {\mu}} [ \nabla_ {\theta} \mu_ {\theta} (s) \nabla_ {a} Q ^ {\mu} (s, a) | _ {a = \mu_ {\theta} (s)} ]\tag{11}
$$

## • Deep Deterministic Policy Gradient

Deep Deterministic Policy Gradient (DDPG) [30] presents a seminal actor-critic (AC) algorithm featuring a combination of DQN and DPG. It utilizes the techniques presented in the DQN algorithm (e.g., target networks and experience replay) to provide a more computationally stable algorithm for continuous action space. A schematic of DDPG is illustrated in Figure 2. The algorithm employs the following: (i) a DQN-based critic to estimate the Q-function and (ii) a deterministic policy actor to choose an action based on the current state. Both the actor and critic utilize target networks that are updated using Polyak averaging to avoid aggressive and unstable updates. An actor network is used to estimate the policy for continuous action space, since there is no computationally efficient method to repeatedly compute $a = \arg \operatorname* { m a x } Q ( s , a , \theta )$ , when a is continuous. In this algorithm, the actor network thrives to maximize Q-value approximations and the critic network aims to minimize the Bellman loss.

The DDPG method samples actions in an off-policy manner during training. Instead of adopting the ϵ-greedy exploration from DQN, the algorithm uses Ornstein–Ulhenbeck (OU) exploratory noise to explore the continuous action space [43]. Similar to DQN, the use of an experience replay buffer allows the networks to update based on both the current and the past experiences, for a more sampling-efficient learning.

![](images/18da472203e250ad931adaf7dfa3dc16672f9b994e688d99961c433e63a91ce6.jpg)  
Figure 2. Illustration of DDPG.

## • Trust Region Policy Optimization

Trust Region Policy Optimization (TRPO) is a policy gradient-based RL method introduced in Schulman et al. [31]. It can be characterized as an on-policy method for stochastic policy optimization. TRPO can be applied to both discrete and continuous action spaces. One advantage of TRPO is the guarantee that the updated policies are monotonically improved over the old ones for general classes of stochastic policies. As shown in Equation (12), if a non-negative expected advantage can be achieved for all the states $s ,$ namely, $\begin{array} { r } { \sum _ { a } \tilde { \pi } ( a | s ) A _ { \pi _ { i } } ( s , a ) \geq 0 . } \end{array}$ , the expected return (η) of the new policy π˜ will be no worse than the old policy π. Note that $\rho _ { \pi } ( s )$ is the state visitation density. To relax the calculation of $\rho _ { \tilde { \pi } }$ as a function of the new policy π˜ , a local approximation L(π˜ ) is normally used in place of η(π˜ ) (Equation (13)).

$$
\eta (\tilde {\pi}) = \eta (\pi) + \sum_ {s} \rho_ {\tilde {\pi}} (s) \sum_ {a} \tilde {\pi} (a | s) A _ {\pi_ {i}} (s, a)\tag{12}
$$

$$
L (\tilde {\pi}) = \eta (\pi) + \sum_ {s} \rho_ {\pi} (s) \sum_ {a} \tilde {\pi} (a | s) A _ {\pi_ {i}} (s, a)\tag{13}
$$

A key feature of the practical implementation of TRPO is to take large yet robust steps in the optimization. This is carried out by solving a constrained optimization problem to determine a new set of weights and biases for a parameterized stochastic policy $( \pi _ { \theta } ( \cdot | s ) )$ , as given in Equation (14).

$$
\begin{array}{r l} \underset {\theta_ {n e w}} {\max} & L _ {\theta_ {o l d}} (\theta_ {n e w}) \\ s. t. & \bar {D} _ {K L} ^ {\rho \theta_ {o l d}} (\theta_ {o l d}, \theta_ {n e w}) \leq \delta \end{array}\tag{14}
$$

where $\bar { D } _ { K L } ^ { \rho \theta _ { o l d } } ( \theta _ { o l d } , \theta _ { n e w } )$ is the average Kullback–Leibler (KL) divergence between the two stochastic policies parameterized by $\theta _ { o l d }$ and $\theta _ { n e w }$ , which are bounded by a hyperparameter δ. The constraint on the average KL divergence does not hold the strong theoretical guarantees across the full state space, but it addresses the solvability issue with imposing constraints on each individual point. This average-based constraint provides heuristic confidence that, on average, the policy is improving or at worst, would stay the same, at each update. In practice, this optimization problem can still be complex to implement as it requires a second-order optimization solver to repeatedly compute the Hessian of the KL divergence constraint.

## • Proximal Policy Optimization

Proximal Policy Optimization (PPO) is built on the same motivation as TRPO, i.e., to make large and robust improvements in the policy while ensuring a stable convergence. Built upon TRPO, Schulman et al. [32] introduced PPO as a more generalized AC-based approach with first-order optimization to simplify the implementation and reduce sampling complexity while retaining the same stability benefits. A schematic of PPO is illustrated in Figure 3. While TRPO implements the KL divergence criterion as a hard constraint, PPO uses a clipped surrogate objective to avoid the expensive Hessian computation of the KL divergence altogether. The clipped surrogate objective is represented in Equation (15). A hyperparameter ϵ is used to develop a special clipping in the objective function that effectively eliminates aggressive and unstable policy updates. This clipped objective removes the incentive for the ratio of the new policy to the old policy $( \mathrm { i . e . , ~ } \frac { \hat { \pi } _ { \boldsymbol { \theta } } ( a \vert s ) } { \pi _ { \boldsymbol { \theta } _ { k } } ( a \vert s ) } )$ to move out of the range of $[ 1 - \epsilon , 1 + \epsilon ]$ . The objective can be maximized using minibatch stochastic gradient descent.

$$
\mathcal {L} ^ {C L I P} (\theta) = \mathbb {E} _ {t} \left[ \min \left(\frac {\pi_ {\theta} (a | s)}{\pi_ {\theta_ {k}} (a | s)} A ^ {\pi_ {\theta_ {k}}} (s, a), \operatorname{clip} \left(\frac {\pi_ {\theta} (a | s)}{\pi_ {\theta_ {k}} (a | s)}, 1 - \epsilon , 1 + \epsilon\right) A ^ {\pi_ {\theta_ {k}}} (s, a)\right) \right]\tag{15}
$$

![](images/c94f7ff7f7ecd9d45d37011da6d0492d5bd749b0c9a49422e754db10e69e999a.jpg)  
Figure 3. Illustration of PPO.

## • Twin Delayed Deep Deterministic Policy Gradient

Twin Delayed Deep Deterministic Policy Gradient (TD3) is an extension of the DDPG proposed by Fujimoto et al. [33]. This algorithm aims to reduce the Q-function overestimation bias caused by DQN and DDPG. Like DDPG, TD3 is also an off-policy algorithm for continuous action spaces utilizing replay buffers. TD3 has three key modifications that have contributed to improved performances over DDPG: clipped double Q-learning, delayed policy updates, and target policy smoothing.

For clipped double Q-learning, two independent critic networks are used with their Q-functions (denoted by $Q _ { \theta _ { 1 } }$ and $Q _ { \theta _ { 2 } }$ in Equation (16)). The algorithm uses the smaller value of the two to make the actor updates. By choosing this smaller, or more “pessimistic” Q-value, TD3 could achieve reduced overestimation of the Q-function.

$$
y _ {1} (r, s ^ {\prime}) = r + \gamma \min _ {i = 1, 2} Q _ {\theta_ {i}} (s ^ {\prime}, \pi_ {\phi_ {1}} (s ^ {\prime}))\tag{16}
$$

where $\theta _ { 1 }$ and $\theta _ { 2 }$ are the weights of the twin critic networks representing the independent value functions, and $\phi _ { 1 }$ is the weight of the actor network.

The delayed policy update feature ensures that the policy is not updated as frequently as the critic, usually at a ratio of two critic updates for every one policy update. The update of the policy network is performed at least until the Q-values have converged, $\mathrm { i . e . , }$ the value error is as small as possible. As for target policy smoothing, a clipped zero-mean noise ϵ with a variance $\sigma$ is added to the target actions (Equations (17) and (18)). This addresses the shortcoming of DDPG that, due to Q-function overestimation in certain areas of the action and state spaces, random peaks may develop, and these local maxima are quickly exploited by the policy. Using the noise within the target policy ensures that similar states and actions have similar Q-values, essentially smoothing the peaks within the Q-estimates.

$$
y = r + \gamma \min _ {i = 1, 2} Q _ {\theta_ {i}} (s ^ {\prime}, \pi_ {\phi_ {1}} (s ^ {\prime}) + \epsilon)\tag{17}
$$

$$
\epsilon \sim \operatorname{clip} (\mathcal {N} (0, \delta), - c, c)\tag{18}
$$

## • Soft Actor–Critic

Soft Actor–Critic (SAC), originally proposed by Haarnoja et al. [34], aims to improve sampling efficiency and to learn a stochastic policy. SAC is an AC-based off-policy method based on maximum entropy learning. Similar to TD3, SAC utilizes the mean squared Bellman error as the loss function to train the critic networks. SAC also trains two independent critics using a shared target value and then uses the clipped double-Q methodology of choosing the smaller Q-value to make updates. In contrast to TD3, the target includes an entropy term, making use of a modified version called a soft Bellman equation.

Another major difference is the use of the current policy instead of the target policy to choose actions for the future states. The loss function of critic networks is formulated as Equation (19). The target value (y) can be described as Equation (20). The objective for the actor network training is to maximize the expected future return and the expected entropy, as shown in Equation (21).

$$
L (\theta_ {i}, \mathcal {D}) = \mathbb {E} _ {(s, a, r, s ^ {\prime}) \sim \mathcal {D}} [ (Q _ {\theta_ {i}} (s, a) - y (r, s ^ {\prime}) ^ {2} ]\tag{19}
$$

$$
y (r, s ^ {\prime}) = r + \gamma (\min _ {j = 1, 2} Q _ {\theta_ {t a r g e t, j}} (s ^ {\prime}, a ^ {\prime}) - \alpha \log \pi_ {\theta} (a ^ {\prime} | s ^ {\prime}))\tag{20}
$$

$$
V ^ {\pi} (s) = \underset {a \sim \pi} {\mathbb {E}} [ Q ^ {\pi} (s, a) ] - \alpha \log \pi (a | s)\tag{21}
$$

where $i = \{ 1 , 2 \}$ indicates the independent critic networks, $\theta _ { i }$ is the weight of the networks, $y$ is the target value, $a ^ { \prime }$ is the future action corresponding to the future state $s ^ { \prime }$ sampled from the policy for the target calculation.

## • Random Network Distillation

The random network distillation (RND) algorithm is an exploration method introduced to deal with the parallel processing of large numbers of samples in RL agent training [35]. The central idea of the RND method is to use prediction errors of networks trained on the agent’s past experience to quantify the novelty of the new experience. The algorithm is composed of two neural networks: a fixed and randomly initialized target network that converts an observation into an embedding, and a predictor neural network that is trained by gradient descent to minimize the expected prediction mean squared error. The RND prediction error can be viewed as a special case of a previously proposed uncertainty quantification method in Osband et al. [44] that quantifies the uncertainty in predicting the constant zero function in the following minimization, with regression targets $y _ { i } = 0 ;$

$$
\theta = \arg \min _ {\theta} \mathbb {E} _ {(x _ {i}, y _ {i}) \sim \mathcal {D}} \| f _ {\theta} (x _ {i}) + f _ {\theta^ {*}} (x _ {i}) - y _ {i} \| ^ {2} + \mathcal {R} (\theta)\tag{22}
$$

where $D \ = \ \{ x _ { i } , y _ { i } \}$ is the data distribution, $f _ { \theta }$ is the mapping of the neural network parameters $\theta ,$ and R is a regularization term from the prior. For detailed description and demonstration on study cases, we refer the readers to Burda et al. [35].

## • Conservative Q-Learning

Introduced by Kumar et al. [36], the Conservative Q-learning (CQL) algorithm aims to address the overestimation issue associated with standard value-based off-policy RL algorithms on out-of-distribution actions, with a conservative estimate of the value function. The algorithm targets estimating the value $V ( s )$ of a target policy π given a dataset ${ \mathcal { D } } ,$ generated from a behavior policy $\pi _ { \beta } .$ . The penalty is formulated to minimize the expected Q-function (Q<sup>ˆ</sup> ) under a particular distribution of state–action pairs, $\mu ( s , a )$ , as in:

$$
\hat {Q} ^ {k + 1} \leftarrow \arg \min _ {Q} \alpha \mathbb {E} _ {s \sim \mathcal {D}, a \sim \mu (a | s)} [ Q (s, a) ] + \frac {1}{2} \mathbb {E} _ {s, a \sim \mathcal {D}} \bigg [ \Big (Q (s, a) - \mathcal {B} ^ {\pi} \hat {Q} ^ {k} (s, a) \Big) ^ {2} \bigg ]\tag{23}
$$

where $\boldsymbol { B }$ is the Bellman operator. By setting $\mu ( s , a ) = d ^ { \pi _ { \beta } } ( s ) \mu ( a | s )$ to match the statemarginal in the dataset, it can be proven that the Q-function can provide a lower bound at all $( s , a )$ pairs. To tighten the bound, the authors introduced an additional Q-value maximization term to formulate a new iterative update as:

$$
\begin{array}{r l} \hat {Q} ^ {k + 1} \leftarrow \arg \min _ {Q} \alpha \cdot \Big (\mathbb {E} _ {s \sim \mathcal {D}, a \sim \mu (a | s)} [ Q (s, a) ] - \mathbb {E} _ {s \sim \mathcal {D}, a \sim \hat {\pi} _ {\beta} (a | s)} [ Q (s, a) ] \Big) \\ & + \frac {1}{2} \mathbb {E} _ {s, a, s ^ {\prime} \sim \mathcal {D}} \bigg [ \Big (Q (s, a) - \mathcal {B} ^ {\pi} \hat {Q} ^ {k} (s, a) \Big) ^ {2} \bigg ] \end{array}\tag{24}
$$

To utilize the Q-value lower bound in policy optimization, the authors then formulated a family of optimization problems over $\mu ( a | s )$ to approximate the policy that maximizes the current Q-function iteration, with a particular choice of regularizer $R ( \mu )$ , as in the following [36]:

$$
\begin{array}{r l} & {\underset {Q} {\min} \underset {\mu} {\max} \alpha \Big (\mathbb {E} _ {s \sim \mathcal {D}, a \sim \mu (a | s)} [ Q (s, a) ] - \mathbb {E} _ {s \sim \mathcal {D}, a \sim \hat {\pi} _ {\beta} (a | s)} [ Q (s, a) ] \Big)} \\ & {\qquad + \frac {1}{2} \mathbb {E} _ {s, a, s ^ {\prime} \sim \mathcal {D}} \bigg [ \Big (Q (s, a) - \mathcal {B} ^ {\pi_ {k}} \hat {Q} ^ {k} (s, a) \Big) ^ {2} \bigg ] + R (\mu)} \end{array}\tag{25}
$$

## • Distributional SAC

To mitigate Q-value overestimations, Duan et al. presented a distributional soft actor– critic algorithm for continuous control settings in [37]. Specifically, the algorithm first learns a distribution function of state–action results and then embeds the return distribution function into maximum entropy RL for mitigated Q-value overestimation. The algorithm was developed with a general entropy-augmented objective with a policy entropy term H to augment the reward J.

$$
J _ {\pi} = \mathbb {E} _ {(s _ {i} \geq t, a _ {i} \geq t) \sim \rho_ {\pi}, r _ {i} \geq t \sim R (\cdot | s _ {i}, a _ {i})} \left[ \sum_ {i = t} ^ {\infty} \gamma^ {i - t} [ r _ {i} + \alpha \mathcal {H} (\pi (\cdot | s _ {i})) ] \right]\tag{26}
$$

where

$$
\begin{array}{c} \mathcal {H} (\pi (\cdot | s)) = - \int_ {a \in \mathcal {A}} \pi (a | s) \log \pi (a | s) d a \\ = \mathbb {E} _ {a \sim \pi (\cdot | s)} [ - \log \pi (a | s) ] \end{array}\tag{27}
$$

The expected soft return for selecting action $a _ { t }$ in state $s _ { t }$ with policy π with the soft Q-value of policy π is then defined as:

$$
Q^{\pi}(s_{t},a_{t}) = \mathbb{E}_{r\sim R(\cdot |s_{t},a_{t})}[r] + \gamma   \mathbb{E}_{\substack{(s_{i} > t,a_{i}\geq t)\sim \rho_{\pi},\\ r_{i} > t\sim R(\cdot |s_{i},a_{i})}}[G_{t + 1}]\tag{28}
$$

where the return $G _ { t }$ is calculated as $\begin{array} { r } { G _ { t } = \sum _ { i = t } ^ { \infty } \gamma ^ { i - t } [ r _ { i } - \alpha \log \pi ( a _ { i } | s _ { i } ) ] } \end{array}$ , with α being the discount factor. The maximum entropy RL is then extended to distributional learning by directly modeling the distribution of the soft returns $Z ^ { \pi ( s , a ) } , \tilde { Z } ( Z ^ { \pi ( s , a ) } | s , a )$ , where $Z ^ { \pi ( s , a ) }$ is defined as: $Z ^ { \pi } ( s _ { t } , a _ { t } ) = r _ { t } + \gamma G _ { t + 1 } { \Big | } _ { ( s _ { i } > t , a _ { i } > t ) \sim \rho _ { \pi } , r _ { i } \geq t \sim R ( \cdot | s _ { i } , a _ { i } ) }$

The distributional variant of the Bellman operator $( \mathcal { T } _ { D } ^ { \pi } )$ in the maximum entropy framework is defined as:

$$
\mathcal {T} _ {D} ^ {\pi} Z ^ {\pi} (s, a) \stackrel {{D}} {{=}} r + \gamma \big (Z ^ {\pi} (s ^ {\prime}, a ^ {\prime}) - \alpha \log \pi (a ^ {\prime} | s ^ {\prime}) \big)\tag{29}
$$

and the Bellman operator can be implemented as a loss function defined as the following:

$$
\tilde {Z} _ {\text { new }} = \arg \min _ {Z} \mathbb {E} _ {(s, a) \sim \rho_ {\pi}} \left[ d \left(\mathcal {T} _ {D} ^ {\pi} \tilde {Z} _ {\text { old }} (\cdot | s, a), \tilde {Z} (\cdot | s, a)\right) \right]\tag{30}
$$

where $\tilde { Z }$ is the soft state–action return distribution or the distributional value function. Theoretical analysis and case study demonstration are given in [37]. Other notable examples with distributional SAC algorithms include [38], for example.

## • Diffusion Q-Learning

Diffusion Q-Learning was introduced by Wang et al. [39] to use diffusion models for policy regularization to address the inaccurate function approximation issue with standard offline RL algorithms with out-of-distribution actions. The RL policy is represented as the reverse process of a conditional diffusion model as:

$$
\pi_ {\theta} (\boldsymbol {a} \mid s) = p _ {\theta} (a ^ {0: N} \mid s) = \mathcal {N} (a ^ {N}; 0, \mathbf {I}) \prod_ {i = 1} ^ {N} p _ {\theta} (a ^ {i - 1} \mid a ^ {i}, s)\tag{31}
$$

where $P _ { \theta }$ is parameterized as a noise prediction model with covariance matrix $\begin{array} { r l } { \Sigma _ { \theta } ( \pmb { a } ^ { i } , \pmb { s } , i ) = } \end{array}$ $\beta _ { i } I$ and mean constructed as $\begin{array} { r } { \mu _ { \theta } ( { \pmb a } ^ { i } , { \pmb s } , i ) = \frac { 1 } { \sqrt { \bar { \alpha } _ { i } } } \Big ( { \pmb a } ^ { i } - \frac { \beta _ { i } } { \sqrt { 1 - \bar { \alpha } _ { i } } } \epsilon _ { \theta } ( { \pmb a } ^ { i } , { \pmb s } , i ) \Big ) } \end{array}$ , with $a ^ { N }$ sampled from $\mathcal { N } ( 0 , I )$ to form the reverse diffusion chain parameterized by θ:

$$
\boldsymbol {a} ^ {i - 1} \mid \boldsymbol {a} ^ {i} = \frac {\boldsymbol {a} ^ {i}}{\sqrt {\alpha_ {i}}} - \frac {\beta_ {i}}{\sqrt {\alpha_ {i} (1 - \bar {\alpha} _ {i})}} \epsilon_ {\theta} (\boldsymbol {a} ^ {i}, \boldsymbol {s}, i) + \sqrt {\beta_ {i}} \boldsymbol {\epsilon}, \quad \boldsymbol {\epsilon} \sim \mathcal {N} (0, I), \quad \text { for } i = N, \dots , 1.\tag{32}
$$

The conditional model is then trained with a simplified objective function $\mathcal { L } _ { d } ( \theta )$

$$
\mathcal {L} _ {d} (\theta) = \mathbb {E} _ {i \sim \mathcal {U}, \epsilon \sim \mathcal {N} (0, I), (s, \boldsymbol {a}) \sim \mathcal {D}} \bigg [ \left\| \epsilon - \epsilon_ {\theta} \Big (\sqrt {\bar {\alpha} _ {i}} \boldsymbol {a} + \sqrt {1 - \bar {\alpha} _ {i}} \boldsymbol {\epsilon}, \boldsymbol {s}, i \Big) \right\| ^ {2} \bigg ]\tag{33}
$$

and the final policy-learning objective is given as a combination of policy regularization and policy improvement, as:

$$
\pi = \arg \min _ {\pi_ {\theta}} \mathcal {L} (\theta) = \mathcal {L} _ {d} (\theta) + \mathcal {L} _ {q} (\theta) = \mathcal {L} _ {d} (\theta) - \alpha \cdot \mathbb {E} _ {\boldsymbol {s} \sim \mathcal {D}, \boldsymbol {a} ^ {0} \sim \pi_ {\theta}} \Big [ Q _ {\phi} (\boldsymbol {s}, \boldsymbol {a} ^ {0}) \Big ]\tag{34}
$$

where the Q-value can be solved by minimizing the Bellman operator.

## • Offline-to-Online RL

In [40], Zheng et al. introduced an offline-to-online algorithm to address the challenge of online learning in obtaining large amounts of data for policy training, and the issue with offline learning where the performance can be poor if the offline data do not represent well the online interactions. The underlying idea of the algorithm is to use an optimistic online

RL update strategy to emphasize more the real situation of the policy, and a pessimistic offline RL update strategy to also leverage offline data for policy training. The idea can be represented in the following general formalization:

$$
C ^ {k + 1} \leftarrow \mathcal {F} \Big (\mathbb {A} (C ^ {k}) + \mathcal {W} (\boldsymbol {s}, \boldsymbol {a}) \mathbb {B} (C ^ {k}) \Big)\tag{35}
$$

where C represents a policy or value function for policy-based or value-based methods. $\mathbb { A } ( C )$ is the optimistic update strategy, such as the Bellman error for value-based methods, and <sup>B</sup>(C) is the pessimistic update strategy as a penalty to make the learned agent take actions close to the dataset or take conservative actions. ${ \mathscr W } ( s , { \pmb a } )$ is the weight function to the penalty term $\mathbb { B } ( C )$ that enables adaptive learning of the policy, and $\mathcal { F }$ is the general updating operator, such as argmin or argmax. Demonstration with value-based and policybased methods is provided in [40].

## • Parallelized Q-network

Proposed in Gallici et al. [41], the parallelized Q-learning (PQN) algorithm seeks to address the stabilization issue in training with TD algorithms with off-policy data, with a simplified version of DQN. The basis of the algorithm is to use LayerNorm and $L ^ { 2 }$ regularization for TD stabilization with the following Q-function approximator:

$$
Q _ {\phi} ^ {k} (x) = w ^ {\top} \sigma_ {\text { Post }} \circ \text { LayerNorm } ^ {k} [ \sigma_ {\text { Pre }} \circ M x ]\tag{36}
$$

where $\phi = [ w ^ { \top } , \mathrm { V e c } ( M ) ^ { \top } ]$ is the parameter vector, $M \in \mathbb { R } ^ { k \times d }$ is a $k \times d$ matrix, $w \in \mathbb { R } ^ { k }$ is a vector of the final layer weights, and $\sigma _ { P r e }$ and $\sigma _ { P o s t }$ are element-wise $C ^ { 2 }$ continuous activations with bounded 2nd order derivatives [41]. The author further assumes $\sigma _ { P o s t }$ is L<sub>Psot</sub>–Lipschitz with $\sigma _ { P o s t } ( 0 ) = 0$ . The LayerNorm is defined as:

$$
\operatorname{LayerNorm} _ {i} ^ {k} [ f (x) ] := \frac {1}{\sqrt {k}} \cdot \frac {f _ {i} (x) - \frac {1}{k} \sum_ {j = 0} ^ {k - 1} f _ {j} (x)}{\sqrt {\frac {1}{k} \sum_ {i = 0} ^ {k - 1} \left(f _ {i} (x) - \frac {1}{k} \sum_ {j = 0} ^ {k - 1} f _ {j} (x)\right) ^ {2} + \epsilon}}\tag{37}
$$

where is a small constant for numerical stability. The authors then demonstrated that with λ-returns, which is a parallelized variant of a previously proposed approach in Daley and Amato [45], an exploration policy can then be rolled out: starting with $\begin{array} { r } { R _ { i + T } ^ { \lambda } = \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \phi } \left( s _ { i + T } , a ^ { \prime } \right) } \end{array}$ , and computed recursively back in time from $R _ { i + T - 1 } ^ { \lambda }$ to $R _ { i } ^ { \lambda }$ for the targets, using $\begin{array} { r } { R _ { t } ^ { \lambda } = r _ { t } + \gamma \big [ \lambda R _ { t + 1 } ^ { \lambda } + ( 1 - \lambda ) \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \phi } ( s _ { t + 1 } , a ^ { \prime } ) \big ] } \end{array}$

## 3. Implementations for Process Control

Since its advent, RL has received increasing applications for chemical process control. In terms of applications on control strategy development, RL has been mainly used for controller tuning and direct policy learning, while in terms of processes, RL has been widely applied from molecular design, to materials, to chemical reaction optimization, and to process design and optimization etc. In this section, we review the latest advances on these research fronts. While RL-based controller tuning can generally be carried out on fixed-parameter controllers, such as PID, lead-lag, linear quadratic regulator (LQR), and gain scheduling controllers, etc., here, we will primarily focus on PID tuning, given its popularity, and give a brief summary on some of the other types of controllers. While most of the references in this review are chemical process-related, a brief summary on some of the specific fields in chemical processes that have benefited from and are seeing increasing applications with RL is also given at the end of this section.

## 3.1. RL-Based PID Tuning

A generic proportional-integral-derivative (PID) formulation is given in Equation (38):

$$
u (t) = K _ {p} \epsilon (t) + K _ {I} \int_ {0} ^ {T} \epsilon (t) d t + K _ {D} \dot {\epsilon} (t)\tag{38}
$$

ϵ(t) is the deviation from the set-point of the controlled variable at any given time t. $K _ { p } ,$ $K _ { I } ,$ and $K _ { D }$ are the proportional gain, integral gain, and derivative gain as the tunable parameters, respectively. PID tuning is conventionally performed offline under a certain set of operating conditions [46]. These approaches can be categorized as (i) rule-based, such as Ziegler–Nichols methods and Cohen–Coon etc., and (ii) model-based, such as internal model control (IMC). However, these tuning methods might require knowledge on the system dynamics, time-consuming fine tuning, and might not offer satisfactory performance for complex systems with nonlinearity, stochasticity, and unknown dynamics due to the lack of real-time response to time-varying uncertainties [47]. To improve the applicability of PID controllers, researchers have referred to RL for (1) automation, where tuning the parameters can be efficiently performed by interacting with the environment in an autonomous manner; (2) real-time adaptivity, where the controller dynamically tunes the parameters to adapt to changes or disturbances in the process; and (3) optimality, where multiple optimization objectives and constraints can be incorporated in designing the controller parameters for optimal performance.

One of the first efforts on this front is credited to Howell and Best [48], where PID tuning is automated using RL for a Ford Motors Zetec engine. The parameters were initialized with the Ziegler–Nichols tuning rule, and the authors applied a continuous action reinforcement learning automata (CARLA) algorithm to tune the controller, and achieved reduced process operating cost as compared to using the baseline PID parameters. As depicted in Figure 4, the RL agent was designed with several CARLA modules, each of which represents a variable for one PID parameter. In this way, three CARLA modules were used to tune one single PID controller. The performance metric was evaluated each time, based on which action the agent chose to optimize the controller tuning parameter. This approach, however, would come at a very high computational cost for the training process, as the process scale increases with more controllers, a large number of CARLA modules may be required.

![](images/457ddc03390f94378d4ac0b53478038843de78b761cfff006176631c30be4ee8.jpg)  
Figure 4. A schematic of the CARLA algorithm (reproduced from [48]).

The advent of DRL has further transformed the use of RL for autonomous PID tuning. Nian et al. [11] provided an excellent literature review on the early advances in the implementation of DRL algorithms for PID tuning (before 2020). Over the last few years, AC-algorithms have been increasingly used to design the RL agents, as they overcome the shortcomings of the value-based and policy-based methods. Table 2 presents a list of recent methods for RL-based PID tuning, followed by detailed discussions of the representative works.

Table 2. Methods for RL-based PID tuning — an indicative list.

<table><tr><td>Publication</td><td>Algorithm</td><td>Key features</td><td>Year</td></tr><tr><td>Dogru et al. [49]</td><td>Contextual bandit</td><td>Constrained-RL; Offline training using approximate step-response model; Online fine tuning</td><td>2022</td></tr><tr><td>Lakhani et al. [50]</td><td>DPG</td><td>PID tuning with stability consideration; Episodic tuning; Layer normalization</td><td>2022</td></tr><tr><td>Fujii et al. [51]</td><td>Policy-gradient-type AC</td><td>MIMO process; Complex process with input coupling and input-output lag</td><td>2021</td></tr><tr><td>Mate et al. [52]</td><td>DDPG</td><td>Single DRL agent used to tune multiple PID controllers; Inverting gradients to integrate interval constraints on tuning parameters</td><td>2023</td></tr><tr><td>Wang and Ricardez-Sandoval [47]</td><td>Modified IDDPG</td><td>Nonlinear MIMO; Time-varying uncertainty</td><td>2024</td></tr><tr><td>Li and Yu [53]</td><td>TGSL-TD3PG</td><td>Multiple agents trained via imitation learning and curriculum learning</td><td>2021</td></tr><tr><td>Li and Yu [54]</td><td>FSSL-TD3</td><td>Improved exploration using fittest survival strategy</td><td>2021</td></tr><tr><td>Lawrence et al. [55]</td><td>Modified TD3</td><td>Implementation in a real physical process; Software-hardware interplay; Interpretability</td><td>2022</td></tr><tr><td>Chowdhury et al. [56]</td><td>EMTD3</td><td>Entropy-maximizing stochastic actor for environment exploration, followed by deterministic actor for local exploitation</td><td>2023</td></tr><tr><td>Veerasamy et al. [57]</td><td>TD3 and SAC</td><td>Comparison of discrete and Gaussian reward functions; Nonlinear process</td><td>2024</td></tr><tr><td>Shuprajhaa et al. [58]</td><td>Modified PPO</td><td>Unstable processes with unbounded output; Action repeat; Early stopping</td><td>2022</td></tr></table>

Mate et al. [52] presented the PID tuning of a plant-wide control framework involving multiple single-input single-output (SISO) PID controllers. A DDPG algorithm was used to simultaneously tune these PID controllers to account for interactions between different control loops. A challenge for the proposed algorithm was that it required a massive amount of real-time operating data to train the RL agent, which required the development of, e.g., a high-fidelity digital twin to enable online deployment. Interval constraints were enforced on the tuning parameters to ensure safe plant operations. These constraints were integrated within the RL algorithm via the inverting gradients approach [59], as illustrated in Figure 5a. Namely, if a parameter reached the constraint bound, the sign of its gradient would be inverted. This inverting gradients approach is widely applied in many RL-based PID tuning methods, as can be noted from the following discussions.

![](images/1b5bbe846142b0d969d1bcb124748c8e380f2bff4edd87214c271dbfe06a809e.jpg)  
Figure 5. Examples of RL-based PID tuning methods: (a) Inverting gradients approach (reproduced from [59]), (b) Training and deployment strategy (reproduced from [49]), (c) TGSL-TD3PG for SOFC (reproduced from [53]), (d) Entropy-maximizing TD3 approach (reproduced from [56]).

For multi-input multi-output (MIMO) systems, Fujii et al. [51] developed a self-tuning PI control system for a thin film production process using an AC-based DRL algorithm. One challenge was the spatial mechanical coupling in this production process, where there existed inter-dependencies among the various inputs. This was addressed by introducing a spatial coefficient that can be tuned to compensate the degree of interaction between inputs. A better performance was observed in terms of film thickness control as compared to a conventional PI controller tuned with Ziegler–Nichols method. However, transient oscillation was also observed in the spatial root mean squared error, suggesting further improvement was needed. Wang and Ricardez-Sandoval [47] modified the inverted DDPG algorithm to further account for time-varying uncertainty in nonlinear MIMO systems. The widely used PID tuning criterion, integral of the time-weighted absolute errors (ITAEs), was used as the reward function. The algorithm was tested for a MIMO CSTR system by simulating time-varying uncertainty in two different ways: catalyst deactivation and regeneration. While conventionally-tuned controllers failed to address the uncertainty and led to notable offsets and ITAEs, the RL-based ITAE optimization algorithm provided significantly better performance with >90% reduction in ITAE mean and standard deviation.

Dogru et al. [49] introduced a constrained RL algorithm using a contextual bandit agent that does not require the assumption of Markovian state transitions. The authors also developed a training and deployment strategy to minimize online tuning. As shown in Figure 5b, The agent was initially trained offline on an approximate step-response model of the process to determine an initial set of tuning parameters. On this basis, the agent would be fine tuned when implemented online to interact with the real process under sensory noise. The online computational complexity was significantly reduced as part of the training shifted offline. The approach was successfully showcased on a pilot-scale experimental process involving a multi-model tank system.

Li and Yu [53] proposed the two-stage training strategy twin delayed deep determination policy gradient (TGSL-TD3PG) approach as a large-scale DRL algorithm, as shown in

Figure 5c. This algorithm leveraged imitation and curriculum learning to train the agent. The resultant algorithm was reported to improve the robustness, adaptability, and constraint consideration compared to conventional PID control. It was applied to optimize the PID controller parameters to regulate the output voltage of a solid oxide fuel cell (SOFC). The optimal tuning parameters led to a 45.2% reduction of output voltage setting time and 30% less overshoot in the output voltage. Li and Yu [54] further addressed the consideration of fuel utilization constraints in addition to output voltage control. The authors investigated the tuning of a fractional order PID controller in SOFC using a modified fittest survival strategy large-scale twin delayed deep deterministic policy gradient (FSSL-TD3) algorithm. FSSL-TD3 showed improved exploration efficiency as the use of fittest survival strategy and imitation learning provided higher robustness and adaptivity for tuning.

Also addressing the implementation of RL in actual physical systems, Lawrence et al. [55] proposed an algorithm where the policy was directly parametrized by PID parameters. This also led to enhanced interpretability of the RL controller. A modified TD3 algorithm was designed using the OpenAI Gym environment. Herein, a human machine interface (HMI) was developed in Matlab to connect with the tank system. HMI also recorded process data and sent them to the RL agent periodically. The RL agent would convert the data into state–action–reward–transition series (s, a, r, s<sup>′</sup>) and construct a replay buffer. The reward function was formulated based on the tracking error for output variables as well as the change in manipulating variables at two successive time steps. To ensure safe operations, input constraints were enforced using the inverse gradient approach (Figure 5a). The authors compared the results with the Skogstad internal model control (SIMC) tuning method and the Honeywell Accutune III relay auto-tuning algorithm. The RL algorithm demonstrated better performance in terms of nominal performance, robustness, disturbance rejection, and satisfaction of input constraints. However, the RL algorithm used 30–50 min for training while Accutune III used around 10 min.

Closed-loop stability under noisy system measurements remains an open research question for RL-based control. The parameters chosen for autonomous RL-based PID tuning should be ensured to maintain stability of the system. To this purpose, Lakhani et al. [50] presented a stability-preserving RL-based algorithm for autonomous PID tuning based on DPG. Layer normalization was implemented for the actor and critic networks to prevent policy saturation and to ensure the convergence to the optimal PID parameters. An episodic tuning method was developed in which the parameters in the DPG networks were updated at the end of each episode. In this way, the PID parameters stayed the same for each episode to gain a thorough evaluation of the entire closed-loop performance. The reward function was set based on tracking error. A supervisor mechanism was applied to update the tuning parameters if better tracking performance was achieved or, otherwise, switch back to the baseline values to ensure stable closed-loop operation. The efficacy of the proposed method was demonstrated using a second-order plus dead-time (SOPDT) process.

Sampling efficiency presents another outstanding challenge for RL. As mentioned earlier, a large amount of data is required for RL agent training over many episodes. The low sample efficiency caused by the inefficient exploration strategies also could lead to sub-optimal policies and sub-optimal parameters. To address this challenge, Chowdhury et al. [56] developed an entropy-maximizing TD3 (EMTD3) approach for autonomous PID tuning (Figure 5d). To improve exploration, the algorithm started with an entropy-maximizing stochastic actor to obtain the knowledge on the environment, followed by a deterministic actor for local exploitation to accelerate convergence. This algorithm was showcased using a SOPDT process and a nonlinear CSTR system. The EMTD3 algorithm was shown to provide superior performance compared to the traditional TD3 and SAC algorithms in terms of sample efficiency, convergence speed, and stability. However, this algorithm required a good selection of hyperparameters to ensure satisfactory control performance.

## 3.2. Application of RL on Other Controllers

RL algorithms are also used to facilitate gain scheduling controller and LQR design. In [60], the authors presented a gain-scheduled PI controller design with a proximal pol icy optimization approach to simultaneously target three design objectives: quick initial response, minimized overshoot, and fast settling and minimal steady state error. Specif ically, the control task is divided into three segments, with one RL-based controller for each segment to target one of the three specific objectives. Each RL-based controller will output a set of PI controller design parameters, and the scheduling of the switch among different parameters is dictated by the error band between the target and the process output. Simulation results with a second-order system and a steam turbine system demonstrated significant improvement over other popular approaches considered in the work [60]. Other related works include [61,62]. In [63], Rizvi et al. incorporated RL for a model-free solution to a continuous-time LQR problem. In their work, a filter-based observer was first imple mented to parameterize the state vector for output feedback, and they investigated both policy and value iterations in searching for the optimal solution to the time-continuous LQR problem. Theoretical analysis and simulation results demonstrated that the proposed RL-based solution to the LQR problem can overcome limitations from exploration bias, with improved stability. Interested readers are also referred to the classic work by Bradtke [64] on this topic.

The application of RL on model predictive control (MPC) has also received wide attention over the years. MPC stands as one of the most popular controllers in industrial application and enjoys advantages such as safety constraint handling and real-time adaptability, etc. However, the performance of MPC is heavily dependent on the underlying model, which can be challenging to identify. RL can then be used to learn and adapt to a new environment for predictions to be used in the MPC setup, potentially improving the stability of the closed-loop MPC control. For example, in [65], the authors combined economic MPC (EMPC) with RL to form an optimal control framework, where RL is used to estimate the bounded parameters of the underlying nonlinear model for an accurate prediction of the system dynamics, while the optimal control actions are calculated with EMPC. Theoretical analysis suggests the achievement of practical stability and recursive feasibility under certain assumptions, and implementation on a catalytic oxidation of ethylene to ethylene oxide process in a continuous stirred-tank reactor demonstrated improved accuracy in model parameter estimation, as well as in the resulting control performance. In [66], Oh et al. proposed a double-deep Q-network to learn the action–value function in an off-policy manner to assign the terminal cost used in the MPC objective function. It was claimed by the authors that the proposed control framework could improve the control pol icy with smaller amounts of data, with less sensitivity to the Q-network hyperparameters, and can enable explicit incorporation of state constraints. Simulated testing results on an industrial-scale penicillin product bioreactor showed that, the proposed method improved the control policy with smaller amounts of data compared to DDQN and DDPG, and it also outperformed DDP by overcoming the model–plant mismatch. Other related research works and review articles on this topic include [67–71], to name a few.

## 3.3. RL for Control Policy Learning

In addition to tuning the control parameters, RL has been applied to directly learn the control policy. Table 3 presents a list of recent methods for RL control, followed by detailed discussions of representative works.

Table 3. RL methods for control policy learning—an indicative list.

<table><tr><td>Publication</td><td>Algorithm</td><td>Key Features</td><td>Year</td></tr><tr><td>Spielberg et al. [72]</td><td>AC</td><td>Discrete-time SISO, MIMO, and nonlinear processes; Set-point tracking and disturbance rejection</td><td>2019</td></tr><tr><td>Goulart &amp; Pereira [73]</td><td>AC</td><td>Particle swarm optimization for RL hyperparameter tuning</td><td>2020</td></tr><tr><td>Petsagkourakis et al. [9]</td><td>REINFORCE</td><td>Batch process with plant-model mismatch; Offline training followed by transfer learning and online deployment</td><td>2020</td></tr><tr><td>Bangi and Kwon [74]</td><td>DDPG</td><td>Principal component analysis for dimensionality reduction; Transfer learning for online deployment</td><td>2021</td></tr><tr><td>Siraskar [75]</td><td>DDPG</td><td>Graded learning technique; MATLAB (R2019a) RL Toolbox and Simulink</td><td>2021</td></tr><tr><td>Panjapornpon et al. [76]</td><td>DDPG</td><td>Multi-agent RL with gated recurrent unit layer; Grid search for hyperparameter tuning</td><td>2022</td></tr><tr><td>Patel [77]</td><td>DDPG</td><td>Problem formulation with domain knowledge</td><td>2023</td></tr><tr><td>Bao et al. [78]</td><td>DDACP</td><td>Immediate reward to provide gradient information to actor; Expectation form of policy gradient</td><td>2021</td></tr><tr><td>Elmaz et al. [79]</td><td>PPO</td><td>Combining two separate process phases; Constraints implemented via logarithmic barrier functions</td><td>2023</td></tr><tr><td>Hong et al. [80]</td><td>DDPG</td><td>DRL-PID cascade control</td><td>2024</td></tr><tr><td>Beahr et al. [81]</td><td>TD3</td><td>RL-PID in parallel</td><td>2024</td></tr><tr><td>Wang and Wu [82]</td><td>Physics-informed RL</td><td>Integrating Lyapunov stability and policy iteration convergence conditions</td><td>2024</td></tr><tr><td>Faridi et al. [83]</td><td>Model-based DRL</td><td>Deep NN to learn process dynamics</td><td>2024</td></tr><tr><td>Croll et al. [84]</td><td>-</td><td>Comparison of DQN; ppO, TD3, etc.</td><td>2023</td></tr><tr><td>Oh [85]</td><td>-</td><td>Comparison of data-driven MPC using N4SID, NNARX, LSTM vs. model-free RL control using DDPG, TD3, SAC</td><td>2024</td></tr></table>

The motivation of Spielberg et al. [72] was to develop model-free control that could learn a dynamic data-driven control policy as a replacement to the interruptive model re-identification procedures used in the current practice. The authors successfully imple mented an AC-style algorithm to develop a DRL-based controller (Figure 6a) for set-point tracking of discrete-time SISO, MIMO, and nonlinear processes with dynamic disturbances. Case studies were presented for a paper-making machine, a distillation column, and a heating, ventilation, and air conditioning process. Alves Goulart and Dutra Pereira [73] developed an AC-based pH controller to handle the wastewater from the electroplating industry. A key aspect of the controller was that the tuning of the RL hyperparameters was automated by adapting the particle swarm optimization (PSO) algorithm. Using a conventional PID controller, oscillatory responses were observed due to the highly nonlinear process dynamics, with an offset of at least 12.6% for the servo operation. Testing with the RL controller demonstrated a smooth control performance with no offsets, in contrast. As hyperparameter tuning was key to RL, the authors compared the algorithm with or without the PSO-based tuning, and found that the superior control performance was only achieved when the hyperparameters were well-tuned.

![](images/fa9728d2a10a205665b981268b4e19ec1bb399254b370518b9018c1dc61b7a6b.jpg)  
Figure 6. Examples of RL control algorithms: (a) AC-based DRL control (reproduced from [72]), (b) DPG-based RL control with transfer learning (reproduced from [9]), (c) Physics-informed RL control (reproduced from [82]).

As briefly discussed in the Introduction section, Petsagkourakis et al. [9] implemented a policy-gradient-based REINFORCE algorithm for batch bioprocesses with plant–model mismatch. The batch process was characterized with complex nonlinear dynamics and stochastic behaviors. As depicted in Figure 6b, the agent was initially trained offline using a state-space representation of the process model. Transfer learning was then implemented for online deployment. Part of the policy network would be kept frozen to retain the knowledge learned offline, while the last hidden layers or the newly added layers would be trained using the real-time data. The offline training time was reported to be 3 h on CPU while the online implementation required merely 0.002 s. The performance was compared with that of a nonlinear MPC. Transfer learning was also leveraged by Bangi and Kwon [74] for online learning using real-time data. In this work, a DDPG-based RL control strategy was developed for a hydraulic fracturing process. A key feature of this algorithm was using principal component analysis to reduce the dimensionality of the RL state before actor and critic learning, thus achieving faster learning. A reduced-order model was developed using the multivariate output error state space algorithm, based on which the agent was trained offline. The use of transfer learning was also observed to reduce the online computational time. A cumulative reward function was designed that comprised three separate terms to, respectively, minimize tracking error, enforce constraints on control inputs, and incorporate prior process knowledge.

Panjapornpon et al. [76] discussed a multi-agent DDPG-based RL strategy for the coupled control of pH and liquid level in an acid–base reaction process. A gated recurrent unit (GRU) layer was used for the actor and critic networks. A grid search technique was also implemented for the efficient tuning of hyperparameters such as the number of training episodes and the number of hidden layers. The reward function was designed separately for the pH and the liquid level control, accounting for tracking error and output constraints. The algorithm was compared to the performance of SISO PI controllers, showcasing superior set-point tracking speed with less oscillation and lower error metrics.

Bao et al. [78] developed the deep deterministic actor critic predictor (DDACP) architecture to enhance the learning process for dynamic control. The process dynamics were assumed to follow Gaussian distribution. The action–value function was then separated into (i) immediate reward that provided the actor with gradient at the early stages and (ii) next state’s value function for policy gradient estimation in the expectation form. The control performance of the DDACP algorithm was compared with that of DDPG, TD3, and model-based action gradient estimation methods (e.g., MAGE-DDPG and MAGE-TD3). The DDACP method achieved more stable convergence and higher total reward after learning 10,000 episodes, and also showed superior control performance in comparison with PID control for SISO systems and MPC for MIMO systems.

Several works have also explored the integrated use of RL control and conventional process control techniques. Hong et al. [80] presented a PID-DRL cascade control scheme. A DDPG-based architecture was used to model the DRL controller. The states of the RL environment were designed as a PID-control inspired observation vector containing three values: (i) the current value of the output variable, (ii) the error value that denotes the current deviation of the output variable from its set point, and (iii) an error integral to capture the accumulated error accounting for historical behavior. The reward function was designed to both reward the smaller deviations using the reciprocal of their absolute values, and penalize the larger deviations using the square root of their absolute values. This PID-DRL cascade control was then compared with PID-PID cascade control, and demonstrated a >50% reduction in terms of the integral absolute errors. Beahr et al. [81] developed a control scheme which operated RL control and PID in parallel. The RL agent could learn from the PID control actions to enable safe and efficient online training. By continuously evaluating the online performance of both control strategies, the process system could transition from PID to RL if the agent decision making was improved, or vice versa if stability should be attained via the PID controller. Even when paired with suboptimal PID, the RL agent derived near-optimal control policies without significant performance degradation.

In the context of process control, data-driven methods (e.g., systems identification, neural networks) can be applied to “learn the model”, while reinforcement learning can be applied to directly “learn the value”. To quantitatively compare their efficacy, Oh [85] studied six methods: (i) three data-driven MPC approaches using N4SID, NNARX, long short term memory (LSTM) to learn the model, and (ii) three model-free RL approaches based on DDPG, TD3, and SAC to learn the value. Three dynamic process case studies were investigated, including continuous stirred tank reactor, slow moving bed, and a semi-batch Penicillin bioreactor. The general conclusion from this work was that data-driven MPC presented lower cost variances and better data efficiency, while RL-based control performed better at capturing complex process dynamics (e.g., the periodic SMB process). The recent impetus of physics-informed machine learning also opens up new opportunities for process modeling and control [86,87]. As RL is essentially a black-box approach, the infusion of physics-based knowledge can also address the challenge of interpretability. As an example toward this direction, Wang and Wu [82] introduced an AC-based physics-informed RL algorithm to provide optimal control with input constraints. The knowledge of Lyapunov stability and policy iteration convergence conditions were incorporated to the value and policy networks, as shown in Figure 6c.

Reinforcement learning has benefited a variety of research fields in chemical process control. While it is impossible to cover all the related applications, here, we provide a brief summary on some of the notable areas with recent original research publications and/or review articles: drug design [88–90], crystallization [91–97], polymerization [98,99], material self-assembly [100–105], water treatment [106–112], chemical process monitoring and fault detection [113,114], autonomous operation [12,65,115], chemical reaction optimization [116–122], real-time chemical process design and optimization [123–129]. Other notable review articles include: [10,11,13,77,130] .

## 4. Safe Reinforcement Learning

For chemical systems, it is imperative to satisfy process constraints to ensure operational safety. While input constraints can be relatively easier to be integrated to RL problems by defining the action space accordingly, it is not trivial to enforce path constraints (e.g., on state variables). The low interpretability of RL as a black box also presents formidable challenges, hindering the practical implementations of RL-based control. Hence, safe RL has emerged as a relatively new field of study where the agents are designed to incorporate safety considerations. This section provides a discussion on the general theoretic algorithms for safe RL and their applications to chemical process control.

## 4.1. General Basics

A key review paper is offered by Garcıa and Fernández [131] on the advances of safe RL before 2015. The authors formally defined safe RL as the process to learn policies that can maximize the expectation of the return while ensuring system performance and/or safety constraints during the learning/deployment processes. They also classified the existing safe RL approaches into the two categories: (i) modifying the optimization crite rion to integrate the risk concept (e.g., worst case criterion, risk-sensitive criterion) and (ii) modifying the exploration process to avoid unsafe situations (e.g., incorporating ex ternal knowledge, risk-direction exploration). Brunke et al. [132] also discussed the three levels of safety that can be posed via different constraint formulation: (a) Safety Level I—which encourages constraint satisfaction (or in other words, allows possible minimal violations). This can be posed via soft constraints such as adding penalty terms and con straining the expected value. (b) Safety Level II—which requires no violations to happen at high probability. This can be posed via probabilistic constraints, and (c) Safety Level III—which requires no violations to be posed via hard constraints. These safety constraints can be applied to assure safe operating states, input actions, stability guarantee, etc.

A number of safe RL approaches have been developed, such as safe exploration, safe optimization, safety critic, constrained MDP-based RL, etc. A brief overview is presented in the following, and comprehensive discussions can be found in [131,132]. In safe exploration, the safety layer presents an example that can be utilized to transform an optimal but potentially unsafe action to the nearest action satisfying the safety constraints. Safe optimization refers to the “safe” sampling of actions to optimize the cost function (i.e., to eliminate actions that exceed safety thresholds or violate constraints). These approaches typically leverage Gaussian process (GP) models. Safety critic is another technique within the broader category of Safe RL algorithms. Similar to conventional critics, a safety critic is learned based on a safe action–value function $( Q _ { s a f e } ^ { \pi } )$ that determines whether the given action leads to unsafe states. Several works have used safety critics to determine safe alternative inputs in case the agent encounters uncertainties.

Constrained MDPs (CMDPs) are also widely used for safe RL. CMDPs are an extension of the conventional MDPs subject to a set of constraints $\mathcal { C } = \{ C _ { i } , b _ { i } \} _ { i = 1 } ^ { m } ,$ , where $c _ { i }$ are the costs and $b _ { i }$ are the bounds for safety constraints. Incorporating this to RL, the cost value functions $V _ { \pi } ^ { c _ { i } } ( s )$ can be defined as in Equation (39). The corresponding action–value function and advantage function can be formulated in a similar way. On this basis, the cost function can be calculated via Equation (40).

$$
V _ {\pi} ^ {c _ {i}} (s) = \mathbb {E} _ {\pi} [ \sum_ {t = 0} ^ {\infty} \gamma^ {t} c _ {i} (s _ {t}, a _ {t}) | s _ {0} = s ]\tag{39}
$$

$$
C _ {i} (\pi) = \mathbb {E} _ {s \sim \rho_ {0} (\cdot)} [ V _ {\pi} ^ {c _ {i}} (s) ]\tag{40}
$$

The goal is thus to maximize the reward while ensuring safety (Equation (41)), and this can be achieved with the Constrained policy optimization (CPO) method, proposed by Achiam et al. [133].

$$
\max _ {\pi \in \Pi_ {S}} J (\pi) \quad \text { s.t. } \quad C _ {i} (\pi) \preceq b _ {i}\tag{41}
$$

On the basis of safe RL theory, Gu et al. [134] summarized five fundamental questions for RL safety:

Safety Policy: How to find an optimal policy that is also compliant with the safety constraints (i.e., to avoid adversary attacks, undesirable situations, reduce risks).

Safety Complexity: How many training samples are required to arrive at the safe policy? Sample complexity is a key issue, as pointed out in many works [135,136].

Safety Applications: What are the latest advances for the applications?

Safety Benchmarks: What benchmarks can be used to examine safe RL performance in a fair and holistic manner?

• Safety Challenges: What are the challenges for the research of safe RL?

## 4.2. Safe RL Approaches in Chemical Process Control

Safety-aware chemical process control more often refers to the integration between chemical process safety and control [137–139]. A formal correlation between process safety and safe RL control presents a key yet untapped research question. Safe RL also presents a relatively under-explored research area for chemical processes, and the aforementioned approaches (e.g., safe exploration, CMDPs) are oriented from the perspective of safety reinforcement learning. In the specific context of safe RL for process control, the approaches can be broadly classified as (i) constraint tightening with backoffs, (ii) control barrier functions, and (iii) control invariant sets. A list of works is summarized in Table $^ { 4 , }$ with detailed discussions following.

Table 4. Safe RL methods for process control—an indicative list.

<table><tr><td>Publication</td><td>Safety Additions</td><td>Key Features</td><td>Year</td></tr><tr><td>Savage et al. [140]</td><td>Constraint tightening with backoffs</td><td>GPs to approximate action-value function; Analytical uncertainty; GP-based backoffs</td><td>2021</td></tr><tr><td>Pan et al. [141]</td><td>Constraint tightening with backoffs</td><td>“Oracle”-assisted constrained Q-learning; Self-tuning backoffs using Broyden&#x27;s method</td><td>2021</td></tr><tr><td>Petsagkourakis et al. [142]</td><td>Constraint tightening with Backoffs</td><td>Chance constrained policy optimization; GP-based backoffs for policy gradient algorithm</td><td>2022</td></tr></table>

Table 4. Cont.

<table><tr><td>Publication</td><td>Safety additions</td><td>Key features</td><td>Year</td></tr><tr><td>Kim and Oh [143]</td><td>Backoffs and control barrier function</td><td>GP-based backoffs; Plant-model mismatches and stochastic disturbances</td><td>2024</td></tr><tr><td>Kim and Lee [144]</td><td>Control Lyapunov functions</td><td>Value function restricted to control Lyapunov function; Asymptotic stability</td><td>2022</td></tr><tr><td>Wang and Wu [145]</td><td>Control Lyapunov-barrier functions</td><td>Value network trained to approximate the control Lyapunov-barrier function; Input constraints</td><td>2024</td></tr><tr><td>Bo et al. [146]</td><td>Controlled invariant sets</td><td>CIS-based initial state sampling, reward design, state reset; Safety supervisor algorithm for online training</td><td>2023</td></tr><tr><td>Wang et al. [147]</td><td>Controlled invariant sets &amp; Lyapunov functions</td><td>Lyapunov neural network trained using data from controlled invariant sets; Transfer learning</td><td>2024</td></tr></table>

## • Constraint Tightening with Backoffs

In the constraint tightening with backoffs approaches, the constraints are tightened by adding backoffs, which would reduce the perceived size of the feasible region to im prove chance constraint satisfaction [143]. These methods have been successfully applied for integrated design and control problems, e.g., Ricardez-Sandoval et al. [148,149]. For process control, Bradford et al. [150] applied the technique to a stochastic data-driven nonlinear MPC. Therein, Gaussian-process-based sampling was utilized to derive state space representations based on input/output data. Explicit backoffs were used to tighten the constraints based on closed-loop simulations. Petsagkourakis et al. [142] developed a chance constrained policy optimization approach that used adaptive GP-based backoffs for constraint tightening to the policy-gradient-based REINFORCE algorithm. The goal of the algorithm was to find the least conservative backoffs that could satisfy the joint chance constraints at the desired probability and confidence interval. Kim and Oh [143] presented a model-based safe RL approach that used determined GP-based adaptive backoffs for constraint tightening under plant–model mismatches and stochastic disturbances. This work was built on the authors’ prior work utilizing control barrier function to enforce state and input constraints, as well as Sontag’s formula to improve system stability. Savage et al. [140] developed another strategy utilizing GPs in a model-free safe RL algorithm for discrete actions in time. Instead of using neural networks as in many other works, the action–value function was designed via GPs. The analytical uncertainty was determined from the standard deviation calculated from the GP posterior probability distribution. On this basis, Bayesian optimization was then leveraged to improve the balance of exploration and exploitation. The probabilistic constraint violation modeled via GP was adopted as a constraint tightening backoff to encourage safe exploration.

## • Control Barrier Functions

Barrier functions are commonly used in constrained optimization problems, which are augmented to the objective function to account for inequality constraints and to ensure constraint satisfaction. The barrier functions are designed to approach infinity at the boundaries of the feasible region, thus ensuring the system remains within the feasible region. Control barrier functions (CBFs) are the counterpart in the control theory [151]. CBFs use the concepts of barrier certificates in a control-theoretic perspective to guarantee safety along the manipulated dynamic system trajectory [152,153]. In their simplest definition,

CBFs restrict the control actions within a safe set. It takes the current state as an input and, in most cases, outputs a finite real number. However, if the process progresses toward a boundary of the safe set, the CBF usually approaches an infinite value. A CBF constraint typically has the form $\dot { h } ( x , u ) \leq \alpha h ( x )$ , where $h ( x )$ is the CBF defined on states x, which must be satisfied when determining the control action (u). In this way, CBFs can be used to design controllers with safety guarantees by guiding the system away from the unsafe boundaries [154].

Kim and Lee [144] presented a model-based RL control strategy for a nonlinear optimal control problem with asymptotic stability guarantees. The value function was restricted to CLF, and a Lyapunov neural network was adopted for this purpose. The policy was updated based on a modified Sontag formula. Kim and Kim [155] extended this approach to include input and state constraints by introducing a barrier function. The applicability was also extended to general nonlinear systems beyond control-affine systems. Wang and Wu [145] developed a safe RL algorithm for the optimal control of nonlinear systems with input constraints, which featured a performance index function based on control Lyapunov-barrier functions (CLBFs) to achieve closed-loop stability and safety, with the CLBF-based value function approximated with neural networks. Demonstrated by the authors, the CLBFs can simultaneously provide the theoretical properties of both CLFs and CBFs.

## • Control Invariant Sets

Positive invariant set [156] for dynamic systems refers to a subset of the state space where once the system enters the set it will stay in it. Using the concept of “energy” from Lyapunov theory for dynamic systems, the boundaries of positive invariant sets can be formed by the level surfaces of a Lyapunov function. Considering a dynamic system, the initial states belong to a certain admissible set defined by the state constraints. The dynamic trajectories might take the system out of the admissible set of states. However, if the admissible states are included within the positive variant set, safe trajectories are guaranteed without constraint violations. Extending this concept to control theory, a control invariant set (CIS) can be formally defined as: a set $\mathcal { X } \subseteq \mathbb { R } ^ { n }$ is control invariant for a dynamic system $x ^ { + } = f ( x , u )$ , if for all states in the set $x \in \mathcal { X }$ , there exists a control action u that guarantees the system states would remain in the set $x ^ { + } \in \mathcal { X }$ . Note, n is the dimension of the state space. The same definition can be extended to a robust CIS in the presence of disturbances and/or uncertainties.

Bo et al. [146] developed a CIS-enhanced safe RL algorithm that used an explicit form of the CIS for RL training. The training was performed both offline and online. Offline training leveraged CIS for (i) the design of RL reward function—to penalize the actions that led the system states to move outside the CIS, (ii) sampling of the initial state—as starting from an initial state within the CIS ensured the RL to determine control actions which could stabilize the system, and (iii) reset of the state—in the cases when the trajectory caused the current state to be outside the CIS. The controller was then deployed online with additional training to account for possible new unexplored operating conditions. The trained agent was allowed to operate on policy until it encountered a new state detected by the safety supervisor algorithm. The system was then reset and retrained. The agent was expected to keep retraining during the safety supervisor algorithm until an action was found that could keep the system within the CIS, or until the maximum iterations were reached. In the latter case, a backup table was used to provide safe actions.

Wang et al. [147] presented a safe transfer RL framework that combined the use of CIS and Lyapunov functions. The transfer RL algorithm leveraged knowledge from pre-trained source RL tasks to learn new, related tasks faster with better data efficiency. The critic network was developed as a Lyapunov NN. However, instead of using barrier functions, safety guarantees were provided during training by using data from the CIS. Analogous to the discussion of transfer RL in Section 3, the weights of the critic network of the source task were then transferred identically to the target network with an additional layer with weights initialized to the identity. Then the weights of the target critic network were fine tuned by training in a similar fashion as the source network training. This would allow new models with data scarcity to be developed for new tasks by using knowledge from the source datasets. A theoretical error bound was also derived to quantify the difference between the source and the target tasks.

## 5. Limitations, Emerging and Future Explorations

As discussed in the previous sections, RL provides a promising strategy for process control with reduced errors, enhanced adaptability to stochastic systems, and better handling of complex non-linear dynamics. However, key open challenges remain on algorithmic interpretability, sampling efficiency, performance evaluation, etc. In this section, we discuss some of the limitations and challenges in RL and chemical process control, as well as the emerging and future directions of research in this field.

## 5.1. Environments for RL Design and Benchmark

Many online environments have been developed that provide attractive tools to design and benchmark RL algorithms such as Gymnasium [157] and OpenAI’s Gym [158]. However, OpenAI Gym lacks parallelization on GPUs. Some vectorized environments use JAX to implement parallelization. These include Gymnax [159], Brax [160], Jumanji [161], and PGX [162]. For a specific environment for process control implementations, PC-Gym has been developed by Bloor et al. [163], which is an open-source tool to develop and evaluate RL algorithms for chemical process control. PC-Gym can be used as the environment within the general RL framework, and agents can be trained using DDPG, ppO, or SAC frameworks. PC-Gym is also designed to include constraint handling abilities. Tools are included for reward function design, disturbance generation, visualization, etc. Benchmark with nonlinear MPC-based “oracle” is also supported to evaluate the performance of RL based control. PC-Gym is built in Python with discrete-time representations using CASADi or JAX vectorization for numerical integration.

## 5.2. Data Efficiency

Data efficiency is a widely recognized challenge for RL. A large amount of data is required for RL agent training over many episodes. Model-free RL algorithms typically start with the action–values initialized as zero, and hence, even the most trivial tasks would take a lot of samples to be learned. To give a more intuitive idea: OpenAI Five [164], a scaled-up PPO-based agent for game playing, was trained on approximately 45,000 years worth of real-time experience over the course of 10 months. OpenAI Rubik’s cube agent [165] required 13,000 years worth of compressed real-time simulation for training. Swan et al. [166] commented that the reliance on sampled rewards may be intrinsically not safe for RL train ing and application in real-world processes. The limitation in sampling efficiency is also related to other key challenges such as RL scalability [11] and generalizability [167].

Extensive efforts have been made to improve the sampling efficiency of RL and DRL frameworks. Experience replay has been used to improve the sampling efficiency of DQN [168] and DDPG [29]. As discussed in Section 2.2, experience replay stores previous interactions in the form of tuples ${ < s , a , r , s ^ { \prime } > }$ collected via exploration. Prioritized experience replay Schaul et al. [169] presents an enhanced version that prioritized the transitions causing high temporal difference error instead of uniformly sampling all experiences. This leads to a higher expected progress during agent learning. However, to prevent the loss of diversity in experiences, a stochastic prioritization is implemented, along with importance sampling to reach a balance between prioritization and uniform sampling. Prioritized experience replay was shown to improve the DQN learning speeds at 41 out of 49 Atari 2600 games.

Transfer learning is considered a promising method to improve data efficiency and generalizability of RL algorithms. The underlying hypothesis of using transfer learning to address data efficiency issues in RL is that, knowledge learned from different yet related environments can compensate information scarcity of the target environment, that using both limited information of the target system and knowledge from related environments can lead to efficient and reliable policy learning. Different approaches can facilitate such knowledge transfer, such as with regularization techniques to avoid overfitting on the source system, with domain randomization to introduce variations in the source system in training the RL algorithm such that it could treat information in the target system as a variant of the source system [170], or to partially retrain the RL algorithm with information from the new system, etc. Using the Deep RL as an example, the knowledge about the envi ronment is embedded in the weights of the neural network, learned from interacting with the environment. When transferring to a new environment with data deficiency, instead of learning the large number of the NN weights for the new environment from scratch, trans fer learning algorithms seek to adapt the NN previously trained from a related system by further training with data collected from the new system for weights tailored for the new en vironment. Taylor and Stone [171], Lazaric [172], Zhu et al. [173], Ada et al. [174] provide a comprehensive survey on transfer learning for RL for the past several decades. Notable applications of transfer learning for data efficiency span from process design to control and to autonomous manufacturing. For example, in [175], the authors deployed transfer learning into a RL framework to achieve process design with an 8% higher revenue and a reduced learning time by a factor of 2. In [176], the authors demonstrated the application of transfer learning in learning the manufacturing process parameters to autonomously fabricate photonic crystals, which can function as an acoustic filter. The authors first trained a source RL algorithm with a first-principle-based model and then learned a probabilistic model based on the source RL algorithm to approximate the reward function in the target system. The authors demonstrated that with transfer learning, the agent can identify the op timal state with as few as 25 artifacts, whereas traditional design approaches would require 144 artifacts, significantly reducing the amount of training data needed. For applications on process control, transfer learning can be used in different ways, such as approximating the process model for model-based RL control, approximating the reward function, or learning the new control policy directly, such as the previously mentioned work in [9]. Other no table examples include [177], where the authors used offline apprenticeship learning for preliminary policy parameterization and then fine-tuned the policy with online training. We refer the readers to review articles [12,178,179] for a more comprehensive survey on how TL can benefit RL in terms of process control.

Meta-reinforcement learning is another attractive alternative for data efficiency and generalizability of RL. Similar, yet different from TL, meta-learning aims at leveraging knowledge previously learned from a wide range of tasks (meta-data), to learn quickly the solution of an unseen system. In meta-RL, the agent learns “how to learn” based on experiences with different systems, thus gaining the capability of adapting to changing and/or unseen environment, such as learning the update rules for the RL agent parameters, as shown in [180]. In [180], the authors proposed a meta-learning architecture, named the learned policy gradient (LPG), to learn both what to predict and how to improve the policy. The LPG is a backward LSTM network that produces as the output how to update the policy and the prediction from the trajectory of the agent. The architecture is trained by considering a comprehensive improvement of a population of agents interacting with different environments for improved generalization to a specific environment. The authors reported that by training the architecture only on toy environments, it gained satisfactory generalizability to complex Atari games, showing the potential of meta-learning in improving RL generalizability. In [181], the authors presented a context-based metalearning RL method for offline PI controller tuning to improve the online tuning sample efficiency. Specifically, the authors developed an RNN to capture the system context and to enable the agent to adapt to changes in the process dynamics. The controller was then trained offline by exposing the actor–critic agent to a broad distribution of different dynamics generated from a first-order plus time delay system model (FOPTD), where the model parameters were uniformly sampled from a range to generate a distribution of different environments. Application of the meta-learning tuned PI controller on a nonlinear two-tank system, which can be approximated with the FOPTD models, demonstrated its capability of rapidly finding the reasonable PI parameters for the new system with noise added to the measurement, further justifying the efficacy of meta-learning in addressing the sample efficiency issue in process control. Other notable works on generalization improvement include [182,183], and review article [184].

## 5.3. Uncertainty-Aware RL Algorithms

One direction for continued efforts can be the development of uncertainty-aware RL algorithms. Uncertainty is ubiquitous. In reinforcement learning, especially modelbased reinforcement learning, uncertainties can come from training the model, inheriting uncertainties native to model development, and also from interacting with the environ ment. Neglecting these uncertainties can result in ineffective performance for practical implementation. Being aware of this, studies on uncertainty measuring, assessing, and uncertainty-aware RL development have started to gain attentions. In [185], the author classified uncertainties as either aleatoric, which are inherent to the system and typically unavoidable, or epistemic, which are due to modeling and can be mitigated. In RL, aleatoric uncertainties could come from stochastic rewards, stochastic observations, and stochastic ac tions. Although they cannot be reduced, measuring and quantifying these uncertainties can help improve the performance of the control. Addressing epistemic uncertainty has been prominently considered in exploration, either implicitly, such as with the count method, where the underlying rationale is to increase training samples to reduce uncertainty, or explicitly. Common methods on explicitly estimating and handling epistemic uncertainties in RL include bootstrapping Q-networks, Monte Carlo Dropout, and variants based off these two fundamental concepts. In [185], Lockwood provided an excellent review on contemporary development in uncertainty estimate and handling in RL, with references to representative works [185]. Other notable works include [186], where the authors proposed an opinion inference algorithm using subjective logic to selectively emphasize more on the critical uncertainties in training the RL algorithm; [187], where the authors developed an action-conditioned ensemble model capable of assessing uncertainty to represent the environment and then proposed an adaptive truncation approach-based RL method for learning efficiency and improved performance; and [188], where an action-conditioned model was developed to assess the uncertainty in an application of an autonomous driving scenario. In addition to the aforementioned works, other methods also include Bayesian neural networks [189–193], or the Dempster–Shafer theory [194–197], etc.

## 5.4. State Observability

Observability is essential in ensuring an effective control. While most of the RLalgorithms have assumed observability of the system with fully observable MDPs, partially observable processes (POMDPs) are prevalent in real applications, limiting the application of RL. Dealing with partially observable processes is notoriously challenging, as training the algorithm would require an exponentially large number of samples. Therefore, RLalgorithms that account for state observability present another invaluable research direction. Current efforts on this topic include (1) state estimation-based RL, such as observable op erator models [198], belief state estimation, and world models [199–201] for compressed or abstracted representation of the environment, to facilitate decision making. Examples include [202], where the authors proposed an object-centric belief model with sequential Monte Carlo inference to provide multiple object-centric scene hypotheses, to improve the performance of RL in an application to a POMDP. In [201], the authors proposed a transformer-based world model agent, named TransDreamer, to use a transformer state space model to predict process dynamics to train the RL agent. Testing on visual RL tasks that required long-range memory access showed outstanding performance. Other notable works include [187,203–205], to name a few. (2) RL with function approximation. In [206], the authors introduced a partially observable bilinear framework and proposed an actor– critic style algorithm to perform agnostic policy learning. Leveraging both memory-based policies and a value function class that incorporates both memory and future observations as the inputs, the authors achieved satisfactory performance. In [207], Uehara et al. devel oped a model-free off-policy algorithm by introducing future-dependent value functions as conditional moment equations to use history proxies as instrumental variables. With a minimax learning method, the authors presented close to the true policy value with the proposed approach. Other important works include [208,208–211], to name a few.

## 5.5. RL Enhanced by Large Language Models

There has been a recent impetus in the development of large language models (LLMs). LLMs are trained on terabytes worth of data, containing billions of parameters [212]. Notable examples of these include GPT-3 [213], which was trained with 175 billion parameters and demonstrated that scaling up the language model can greatly improve task-agnostic, few-shot performance; LLaMA [214], which showed the possibility of training foundation language models with parameters ranging from 7 B to 65 B with publicly available datasets, without resorting to proprietary and inaccessible datasets, for competitive performance compared to benchmark models; and GPT-4 [215], which possesses the capability of solving a wide range of novel and difficult problems with strikingly close to human-level performance that often vastly outperforms prior LLMs such as ChatGPT. These models possess excellent natural language (NL) and computer vision (CV) parsing abilities and can master a wide range of tasks such as language generation and knowledge correlation. Emergent capabilities of LLMs have also been developed such as in-context learning, reasoning, and generalization [216]. These capabilities are defined as abilities that are not present in smaller-scale models but are present in large-scale models that cannot be predicted by simply extrapolating the performance improvements on smaller-scale models, due to the scaling up of the language models. However, how the LLMs gained such capabilities is yet to be explored. Additionally, while step-by-step tasks are complex for smaller models, LLMs can use sophisticated prompting strategies to structure the algorithm in a much more articulate manner, enabling LLMs to output sequences that reflect the progression of thoughts or actions. Such prompting strategies include: Chain of Thought (CoT) [217], which allows the model to decompose multi-step problems into intermediate natural language reasoning to formulate solutions to the overall problem; Tree of Thought (ToT) [218], which builds upon CoT by incorporating multiple different reasoning paths and self-evaluating choices at each intermediate steps, for deliberate decision making; and Graph of Thought (GoT) [219], which is further built upon ToT, and models the information as an arbitrary graph, where the “thoughts” are the vertices, and the edges represent the dependencies between the vertices, to form synergistic outcomes, distill the essence of the thoughts, and/or to enhance the thoughts with feedback loops.

Incorporation of the multimodal data (e.g., NL- and CV-based image data) into the RL structure has also been a challenge [220,221]. It increases the sampling inefficiency as the agent not only has to learn a policy but also needs to parse the multimodal data. The advances in LLMs provide instrumental capabilities not only to handle the multimodal data, but also to handle the inherent challenges posed by the conventional RL paradigm. Pre-trained LLMs such as generative pre-trained transformer (GPT) models posses a vast amount of world knowledge. Cao et al. [222] presented a survey in which they introduced the term LLM-enhanced RL, which was defined as “the methods that utilize the multimodal information processing, generating, reasoning, and other high-level cognitive capabilities of pre-trained, knowledge-inherent LLM models to assist the RL paradigm”. The sam pling inefficiency of RL can be improved by providing contextually valid predictions or suggestions by using a framework of LLMs, thereby reducing the required number of interactions between the agent and the environment. LLMs have also been used for a more informed reward design, as showcased in Li et al. [223]. Notable works on this topic include: Chakraborty et al. [224], which presented a framework for language-based feedback to improve RL generalization; Pang et al. [225], which used LLMs to translate NL-based instructions to task-specific language within RL; and Vyas and Mercangöz [226], which proposed the use of an agentic system comprising LLM-agents for autonomous control with adaptivity to unforeseen disturbances. Detailed discussions on the potential and challenges of LLM-enhanced RL can be found in Cao et al. [222].

In addition to the aforementioned topics, we also note other directions such as convergence, reachability, stability, interpretability [227], and multi-agent control [228,229], etc., and refer the readers to review articles [25,230–233] on the limitations, challenges, and opportunities of RL for further reading.

## 6. Concluding Remarks

This article provides a literature review on the recent advances in RL for chemical process control. The article first introduces the concept of RL and how it intersects with process control. We then review RL basics including the fundamentals of MDPs and stateof-art RL algorithms (e.g., DQN, DDPG, TD3, SAC). The article also surveys the recent implementations of RL within process control from the aspects of RL-based controller tuning and RL for control policy learning. Various methods of safe RL are presented with examples on how they have been implemented for process control. The article further discusses the limitations, challenges, and opportunities offered by the emerging computing algorithms (e.g., transfer learning, meta learning, and large language models) to enhance RL-based control toward practical implementations.

Author Contributions: Conceptualization, W.S., X.T. and Y.T.; Investigation, V.S.D., X.T. and Y.T.; writing—original draft preparation, V.S.D., X.T. and Y.T.; writing—review and editing, V.S.D., X.T. and Y.T.; visualization, V.S.D., X.T. and Y.T.; supervision, W.S., X.T. and Y.T.; project administration, Y.T.; funding acquisition, Y.T. All authors have read and agreed to the published version of the manuscript.

Funding: V.S.D. and Y.T. were supported by Department of Energy grant DE-EE0011195.

Data Availability Statement: No new data were created or analyzed in this study.

Acknowledgments: The authors would like to acknowledge the support from Department of Chem ical and Biomedical Engineering at West Virginia University, and Cain Department of Chemical Engineering at Louisiana State University.

Conflicts of Interest: The authors declare no conflicts of interest.

## References

1. Bequette , B.W. Process Control: Modeling, Design, and Simulation; Prentice Hall Professional: Hoboken, NJ, USA, 2003.

2. Kravaris, C.; Kookos, I.K. Understanding Process Dynamics and Control; Cambridge University Press: Cambridge, UK, 2021.

3. Borase, R.P.; Maghade, D.; Sondkar, S.; Pawar, S. A review of PID control, tuning methods and applications. Int. J. Dyn. Control 2021, 9, 818–827. [CrossRef]

4. Lee, J.H. Model predictive control: Review of the three decades of development. Int. J. Control Autom. Syst. 2011, 9, 415–424. [CrossRef]

5. Schwenzer, M.; Ay, M.; Bergs, T.; Abel, D. Review on model predictive control: An engineering perspective. Int. J. Adv. Manuf. Technol. 2021, 117, 1327–1349. [CrossRef]

6. Daoutidis, P.; Megan, L.; Tang, W. The future of control of process systems. Comput. Chem. Eng. 2023, 178, 108365. [CrossRef]

7. Biegler, L.T. A perspective on nonlinear model predictive control. Korean J. Chem. Eng. 2021, 38, 1317–1332. [CrossRef]

8. Sutton, R.S.; Barto, A.G. Reinforcement Learning: An Introduction, 2nd ed.; The MIT Press: Cambridge, MA, USA, 2018.

9. Petsagkourakis, P.; Sandoval, I.; Bradford, E.; Zhang, D.; Del Rio-Chanona, E. Reinforcement learning for batch bioprocess optimization. Comput. Chem. Eng. 2020, 133, 106649. [CrossRef]

10. Shin, J.; Badgwell, T.A.; Liu, K.H.; Lee, J.H. Reinforcement Learning—Overview of recent progress and implications for process control. Comput. Chem. Eng. 2019, 127, 282–294. [CrossRef]

11. Nian, R.; Liu, J.; Huang, B. A review On reinforcement learning: Introduction and applications in industrial process control. Comput. Chem. Eng. 2020, 139, 106886. [CrossRef]

12. Faria, R.D.R.; Capron, B.D.O.; Secchi, A.R.; De Souza, M.B. Where Reinforcement Learning Meets Process Control: Review and Guidelines. Processes 2022, 10, 2311. [CrossRef]

13. Dogru, O.; Xie, J.; Prakash, O.; Chiplunkar, R.; Soesanto, J.; Chen, H.; Velswamy, K.; Ibrahim, F.; Huang, B. Reinforcement Learning in Process Industries: Review and Perspective. IEEE/CAA J. Autom. Sin. 2024, 11, 283–300. [CrossRef]

14. Wang, Y.; Zhu, X.; Wu, Z. A tutorial review of policy iteration methods in reinforcement learning for nonlinear optimal control. Digit. Chem. Eng. 2025, 15, 100231. [CrossRef]

15. Nievas, N.; Pagès-Bernaus, A.; Bonada, F.; Echeverria, L.; Domingo, X. Reinforcement Learning for Autonomous Process Control in Industry 4.0: Advantages and Challenges. Appl. Artif. Intell. 2024, 38, 2383101. [CrossRef]

16. Bellman, R. A Markovian Decision Process. J. Math. Mech. 1957, 6, 679–684. [CrossRef]

17. Vlassis, N.; Ghavamzadeh, M.; Mannor, S.; Poupart, P. Bayesian Reinforcement Learning. In Reinforcement Learning; Wiering, M., Van Otterlo, M., Eds.; Adaptation, Learning, and Optimization; Springer: Berlin/Heidelberg, Germany, 2012; Volume 12, pp. 359–386.

18. Chenna, S.K.; Jain, Y.K.; Kapoor, H.; Bapi, R.S.; Yadaiah, N.; Negi, A.; Rao, V.S.; Deekshatulu, B.L. State Estimation and Tracking Problems: A Comparison Between Kalman Filter and Recurrent Neural Networks. In Neural Information Processing; Hutchison, D., Kanade, T., Kittler, J., Kleinberg, J.M., Mattern, F., Mitchell, J.C., Naor, M., Nierstrasz, O., Pandu Rangan, C., Steffen, B., Eds.; Lecture Notes in Computer Science; Springer: Berlin/Heidelberg, Germany, 2004; Volume 3316, pp. 275–281.

19. Meng, L.; Gorbet, R.; Kuli´c, D. Memory-based Deep Reinforcement Learning for POMDPs. arXiv 2021. [CrossRef]

20. Chadès, I.; Pascal, L.V.; Nicol, S.; Fletcher, C.S.; Ferrer-Mestres, J. A primer on partially observable Markov decision processes (POMDPs). Methods Ecol. Evol. 2021, 12, 2058–2072. [CrossRef]

21. Xiang, X.; Foo, S. Recent Advances in Deep Reinforcement Learning Applications for Solving Partially Observable Markov Decision Processes (POMDP) Problems: Part 1—Fundamentals and Applications in Games, Robotics and Natural Language Processing. Mach. Learn. Knowl. Extr. 2021, 3, 554–581. [CrossRef]

22. AlMahamid, F.; Grolinger, K. Reinforcement Learning Algorithms: An Overview and Classification. In Proceedings of the 2021 IEEE Canadian Conference on Electrical and Computer Engineering (CCECE), Virtual Conference, 12–17 September 2021; pp. 1–7. [CrossRef]

23. Padakandla, S. A Survey of Reinforcement Learning Algorithms for Dynamically Varying Environments. Acm Comput. Surv. 2021, 54, 1–25. [CrossRef]

24. Han, D.; Mulyana, B.; Stankovic, V.; Cheng, S. A Survey on Deep Reinforcement Learning Algorithms for Robotic Manipulation. Sensors 2023, 23, 3762. [CrossRef]

25. Shakya, A.K.; Pillai, G.; Chakrabarty, S. Reinforcement learning algorithms: A brief survey. Expert Syst. Appl. 2023, 231, 120495. [CrossRef]

26. Rolf, B.; Jackson, I.; Müller, M.; Lang, S.; Reggelin, T.; Ivanov, D. A review on reinforcement learning algorithms and applications in supply chain management. Int. J. Prod. Res. 2023, 61, 7151–7179. [CrossRef]

27. Hu, K.; Li, M.; Song, Z.; Xu, K.; Xia, Q.; Sun, N.; Zhou, P.; Xia, M. A review of research on reinforcement learning algorithms for multi-agents. Neurocomputing 2024, 599, 128068. [CrossRef]

28. Mnih, V.; Kavukcuoglu, K.; Silver, D.; Graves, A.; Antonoglou, I.; Wierstra, D.; Riedmiller, M. Playing Atari with Deep Reinforcement Learning. arXiv 2013, arXiv:1312.5602. [CrossRef]

29. Silver, D.; Lever, G.; Heess, N.; Degris, T.; Wierstra, D.; Riedmiller, M. Deterministic policy gradient algorithms. In Proceedings of the 31st International Conference on Machine Learning, Beijing, China, 21–26 June 2014; pp. 387–395.

30. Lillicrap, T.P.; Hunt, J.J.; Pritzel, A.; Heess, N.; Erez, T.; Tassa, Y.; Silver, D.; Wierstra, D. Continuous control with deep reinforcement learning. arXiv 2015. Version Number: 6. [CrossRef]

31. Schulman, J.; Levine, S.; Abbeel, P.; Jordan, M.; Moritz, P. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning, Lille, France, 7–9 July 2015.

32. Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; Klimov, O. Proximal policy optimization algorithms. arXiv 2017, arXiv:1707.06347.

33. Fujimoto, S.; van Hoof, H.; Meger, D. Addressing Function Approximation Error in Actor-Critic Methods. arXiv 2018. Version Number: 3. [CrossRef]

34. Haarnoja, T.; Zhou, A.; Abbeel, P.; Levine, S. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. arXiv 2018. Version Number: 2. [CrossRef]

35. Burda, Y.; Edwards, H.; Storkey, A.; Klimov, O. Exploration by Random Network Distillation. arXiv 2018, arXiv:1810.12894.

36. Kumar, A.; Zhou, A.; Tucker, G.; Levine, S. Conservative Q-Learning for Offline Reinforcement Learning. arXiv 2020, arXiv:2006.04779.

37. Duan, J.; Guan, Y.; Li, S.E.; Ren, Y.; Cheng, B. Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors. arXiv 2020, arXiv:2001.02811. [CrossRef]

38. Ma, X.; Xia, L.; Zhou, Z.; Yang, J.; Zhao, Q. DSAC: Distributional Soft Actor Critic for Risk-Sensitive Reinforcement Learning. arXiv 2020, arXiv:2004.14547.

39. Wang, Z.; Hunt, J.J.; Zhou, M. Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning. arXiv 2023, arXiv:2208.06193.

40. Zheng, H.; Luo, X.; Wei, P.; Song, X.; Li, D.; Jiang, J. Adaptive Policy Learning for Offline-to-Online Reinforcement Learning. arXiv 2023, arXiv:2303.07693. [CrossRef]

41. Gallici, M.; Fellows, M.; Ellis, B.; Pou, B.; Masmitja, I.; Foerster, J.N.; Martin, M. Simplifying Deep Temporal Difference Learning. arXiv 2025, arXiv:2407.04811.

42. Wang, X.; Wang, S.; Liang, X.; Zhao, D.; Huang, J.; Xu, X.; Dai, B.; Miao, Q. Deep Reinforcement Learning: A Survey. IEEE Trans. Neural Netw. Learn. Syst. 2024, 35, 5064–5078. [CrossRef]

43. Uhlenbeck, G.E.; Ornstein, L.S. On the theory of the Brownian motion. Phys. Rev. 1930, 36, 823. [CrossRef]

44. Osband, I.; Aslanides, J.; Cassirer, A. Randomized prior functions for deep reinforcement learning. arXiv 2018, arXiv:1806.03335.

45. Daley, B.; Amato, C. Reconciling -Returns with Experience Replay. In Proceedings of the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, BC, Canada, 8–14 December 2019; Volume 32.

46. Seborg, D.E.; Mellichamp, D.A.; Edgar, T.F. Process Dynamics and Control, 3rd ed.; Wylie Series in Chemical Engineering; John Wiley & Sons: Hoboken, NJ, USA, 2011.

47. Wang, H.; Ricardez-Sandoval, L.A. A Deep Reinforcement Learning-Based PID Tuning Strategy for Nonlinear MIMO Systems with Time-varying Uncertainty. IFAC-PapersOnLine 2024, 58, 887–892. [CrossRef]

48. Howell, M.; Best, M. On-line PID tuning for engine idle-speed control using continuous action reinforcement learning automata. Control Eng. Pract. 2000, 8, 147–154. [CrossRef]

49. Dogru, O.; Velswamy, K.; Ibrahim, F.; Wu, Y.; Sundaramoorthy, A.S.; Huang, B.; Xu, S.; Nixon, M.; Bell, N. Reinforcement learning approach to autonomous PID tuning. Comput. Chem. Eng. 2022, 161, 107760. [CrossRef]

50. Lakhani, A.I.; Chowdhury, M.A.; Lu, Q. Stability-Preserving Automatic Tuning of PID Control with Reinforcement Learning. arXiv 2022, arXiv:2112.15187. [CrossRef]

51. Fujii, F.; Kaneishi, A.; Nii, T.; Maenishi, R.; Tanaka, S. Self-Tuning Two Degree-of-Freedom Proportional–Integral Control System Based on Reinforcement Learning for a Multiple-Input Multiple-Output Industrial Process That Suffers from Spatial Input Coupling. Processes 2021, 9, 487. [CrossRef]

52. Mate, S.; Pal, P.; Jaiswal, A.; Bhartiya, S. Simultaneous tuning of multiple PID controllers for multivariable systems using deep reinforcement learning. Digit. Chem. Eng. 2023, 9, 100131. [CrossRef]

53. Li, J.; Yu, T. A novel data-driven controller for solid oxide fuel cell via deep reinforcement learning. J. Clean. Prod. 2021, 321, 128929. [CrossRef]

54. Li, J.; Yu, T. Optimal adaptive control for solid oxide fuel cell with operating constraints via large-scale deep reinforcement learning. Control Eng. Pract. 2021, 117, 104951. [CrossRef]

55. Lawrence, N.P.; Forbes, M.G.; Loewen, P.D.; McClement, D.G.; Backström, J.U.; Gopaluni, R.B. Deep reinforcement learning with shallow controllers: An experimental application to PID tuning. Control Eng. Pract. 2022, 121, 105046. [CrossRef]

56. Chowdhury, M.A.; Al-Wahaibi, S.S.; Lu, Q. Entropy-maximizing TD3-based reinforcement learning for adaptive PID control of dynamical systems. Comput. Chem. Eng. 2023, 178, 108393. [CrossRef]

57. Veerasamy, G.; Balaji, S.; Kadirvelu, T.; Ramasamy, V. Reinforcement Learning Based Adaptive PID Controller for a Continuous Stirred Tank Heater Process. Iran. J. Chem. Chem. Eng. 2025, 44, 265–282. [CrossRef]

58. Shuprajhaa, T.; Sujit, S.K.; Srinivasan, K. Reinforcement learning based adaptive PID controller design for control of lin ear/nonlinear unstable processes. Appl. Soft Comput. 2022, 128, 109450. [CrossRef]

59. Hausknecht, M.; Stone, P. Deep reinforcement learning in parameterized action space. arXiv 2015, arXiv:1511.04143.

60. Kumar, K.P.; Detroja, K.P. Gain Scheduled PI controller design using Multi-Objective Reinforcement Learning. IFAC-PapersOnLine 2024, 58, 132–137. [CrossRef]

61. Makumi, W.; Greene, M.L.; Bell, Z.; Bialy, B.; Kamalapurkar, R.; Dixon, W. Hierarchical Reinforcement Learning and Gain Scheduling-based Control of a Hypersonic Vehicle. In Proceedings of the Aiaa Scitech 2023 Forum, National Harbor, MD, USA & Online, 23–27 January 2023.

62. Timmerman, M.; Patel, A.; Reinhart, T. Adaptive Gain Scheduling using Reinforcement Learning for Quadcopter Control. arXiv 2024, arXiv:2403.07216

63. Rizvi, S.A.A.; Lin, Z. Reinforcement Learning-Based Linear Quadratic Regulation of Continuous-Time Systems Using Dynamic Output Feedback. IEEE Trans. Cybern. 2020, 50, 4670–4679. [CrossRef] [PubMed]

64. Bradtke, S. Reinforcement Learning Applied to Linear Quadratic Regulation. In Proceedings of the Advances in Neural Information Processing Systems, Denver, CO, USA, 30 November–3 December 1992; Hanson, S., Cowan, J., Giles, C., Eds.; Morgan-Kaufmann: Burlington, MA, USA, 1992; Volume 5.

65. Alhazmi, K.; Albalawi, F.; Sarathy, M.S. A reinforcement learning-based economic model predictive control framework for autonomous operation of chemical reactors. Chem. Eng. J. 2022, 428, 130993. [CrossRef]

66. Oh, T.H.; Park, H.M.; Kim, J.W.; Lee, J.M. Integration of reinforcement learning and model predictive control to optimize semi-batch bioreactor. AIChE J. 2022, 68, e17658.

67. Kim, J.W.; Park, B.J.; Oh, T.H.; Lee, J.M. Model-based reinforcement learning and predictive control for two-stage optimal control of fed-batch bioreactor. Comput. Chem. Eng. 2021, 154, 107465. [CrossRef]

68. Zhang, Z.; Li, S. Enhanced reinforcement learning in two-layer economic model predictive control for operation optimization in dynamic environment. Chem. Eng. Res. Des. 2023, 196, 133–143. [CrossRef]

69. Kordabad, A.B.; Reinhardt, D.; Anand, A.S.; Gros, S. Reinforcement Learning for MPC: Fundamentals and Current Challenges. IFAC-PapersOnLine 2023, 56, 5773–5780. . [CrossRef]

70. Mesbah, A.; Wabersich, K.P.; Schoellig, A.P.; Zeilinger, M.N.; Lucia, S.; Badgwell, T.A.; Paulson, J.A. Fusion of Machine Learning and MPC under Uncertainty: What Advances Are on the Horizon? In Proceedings of the 2022 American Control Conference (ACC), Atlanta, GA, USA, 8–10 June 2022; pp. 342–357. [CrossRef]

71. Hedrick, E.; Hedrick, K.; Bhattacharyya, D.; Zitney, S.E.; Omell, B. Reinforcement learning for online adaptation of model predictive controllers: Application to a selective catalytic reduction unit. Comput. Chem. Eng. 2022, 160, 107727. [CrossRef]

72. Spielberg, S.; Tulsyan, A.; Lawrence, N.P.; Loewen, P.D.; Bhushan Gopaluni, R. Toward self-driving processes: A deep reinforcement learning approach to control. AIChE J. 2019, 65, e16689. [CrossRef]

73. Alves Goulart, D.; Dutra Pereira, R. Autonomous pH control by reinforcement learning for electroplating industry wastewater. Comput. Chem. Eng. 2020, 140, 106909. [CrossRef]

74. Bangi, M.S.F.; Kwon, J.S.I. Deep reinforcement learning control of hydraulic fracturing. Comput. Chem. Eng. 2021, 154, 107489. [CrossRef]

75. Siraskar, R. Reinforcement learning for control of valves. Mach. Learn. Appl. 2021, 4, 100030. [CrossRef]

76. Panjapornpon, C.; Chinchalongporn, P.; Bardeeniz, S.; Makkayatorn, R.; Wongpunnawat, W. Reinforcement Learning Control with Deep Deterministic Policy Gradient Algorithm for Multivariable pH Process. Processes 2022, 10, 2514. [CrossRef]

77. Patel, K.M. A practical Reinforcement Learning implementation approach for continuous process control. Comput. Chem. Eng. 2023, 174, 108232. [CrossRef]

78. Bao, Y.; Zhu, Y.; Qian, F. A Deep Reinforcement Learning Approach to Improve the Learning Performance in Process Control. Ind. Eng. Chem. Res. 2021, 60, 5504–5515. [CrossRef]

79. Elmaz, F.; Di Caprio, U.; Wu, M.; Wouters, Y.; Van Der Vorst, G.; Vandervoort, N.; Anwar, A.; Leblebici, M.E.; Hellinckx, P.; Mercelis, S. Reinforcement Learning-Based Approach for Optimizing Solvent-Switch Processes. Comput. Chem. Eng. 2023, 176, 108310. [CrossRef]

80. Hong, X.; Shou, Z.; Chen, W.; Liao, Z.; Sun, J.; Yang, Y.; Wang, J.; Yang, Y. A reinforcement learning-based temperature control of fluidized bed reactor in gas-phase polyethylene process. Comput. Chem. Eng. 2024, 183, 108588. [CrossRef]

81. Beahr, D.; Bhattacharyya, D.; Allan, D.A.; Zitney, S.E. Development of algorithms for augmenting and replacing conventional process control using reinforcement learning. Comput. Chem. Eng. 2024, 190, 108826. [CrossRef]

82. Wang, Y.; Wu, Z. Physics-informed reinforcement learning for optimal control of nonlinear systems. AIChE J. 2024, 70, e18542. [CrossRef]

83. Faridi, I.K.; Tsotsas, E.; Kharaghani, A. Advancing Process Control in Fluidized Bed Biomass Gasification Using Model-Based Deep Reinforcement Learning. Processes 2024, 12, 254. [CrossRef]

84. Croll, H.C.; Ikuma, K.; Ong, S.K.; Sarkar, S. Systematic Performance Evaluation of Reinforcement Learning Algorithms Applied to Wastewater Treatment Control Optimization. Environ. Sci. Technol. 2023, 57, 18382–18390. [CrossRef] [PubMed]

85. Oh, T.H. Quantitative comparison of reinforcement learning and data-driven model predictive control for chemical and biological processes. Comput. Chem. Eng. 2024, 181, 108558. [CrossRef]

86. Bradley, W.; Kim, J.; Kilwein, Z.; Blakely, L.; Eydenberg, M.; Jalvin, J.; Laird, C.; Boukouvala, F. Perspectives on the integration between first-principles and data-driven modeling. Comput. Chem. Eng. 2022, 166, 107898. [CrossRef]

87. Pistikopoulos, E.N.; Tian, Y. Advanced modeling and optimization strategies for process synthesis. Annu. Rev. Chem. Biomol. Eng. 2024, 15. [CrossRef] [PubMed]

88. Popova, M.; Isayev, O.; Tropsha, A. Deep reinforcement learning for de novo drug design. Sci. Adv. 2018, 4, eaap7885. [CrossRef] [PubMed]

89. Ståhl, N.; Falkman, G.; Karlsson, A.; Mathiason, G.; Bostrom, J. Deep reinforcement learning for multiparameter optimization in de novo drug design. J. Chem. Inf. Model. 2019, 59, 3166–3176. [CrossRef]

90. Nikita, S.; Tiwari, A.; Sonawat, D.; Kodamana, H.; Rathore, A.S. Reinforcement learning based optimization of process chromatography for continuous processing of biopharmaceuticals. Chem. Eng. Sci. 2021, 230, 116171. [CrossRef]

91. Benyahia, B.; Anandan, P.D.; Rielly, C. Control of batch and continuous crystallization processes using reinforcement learning. In Computer Aided Chemical Engineering; Elsevier: Amsterdam, The Netherlands, 2021; Volume 50, pp. 1371–1376.

92. Manee, V.; Baratti, R.; Romagnoli, J.A. Learning to navigate a crystallization model with deep reinforcement learning. Chem. Eng. Res. Des. 2022, 178, 111–123.

93. Anandan, P.D.; Rielly, C.D.; Benyahia, B. Optimal control policies of a crystallization process using inverse reinforcement learning. In Computer Aided Chemical Engineering; Elsevier: Amsterdam, The Netherlands, 2022; Volume 51, pp. 1093–1098.

94. Lu, M.; Rao, S.; Yue, H.; Han, J.; Wang, J. Recent advances in the application of machine learning to crystal behavior and crystallization process control. Cryst. Growth Des. 2024, 24, 5374–5396. [CrossRef]

95. Xiouras, C.; Cameli, F.; Quilló, G.L.; Kavousanakis, M.E.; Vlachos, D.G.; Stefanidis, G.D. Applications of artificial intelligence and machine learning algorithms to crystallization. Chem. Rev. 2022, 122, 13006–13042. [CrossRef]

96. Yoo, H.; Byun, H.E.; Han, D.; Lee, J.H. Reinforcement learning for batch process control: Review and perspectives. Annu. Rev. Control 2021, 52, 108–119. [CrossRef]

97. Ma, Y.; Zhu, W.; Benton, M.G.; Romagnoli, J. Continuous control of a polymerization system with deep reinforcement learning. J. Process Control 2019, 75, 40–47. [CrossRef]

98. Li, H.; Collins, C.R.; Ribelli, T.G.; Matyjaszewski, K.; Gordon, G.J.; Kowalewski, T.; Yaron, D.J. Tuning the molecular weight distribution from atom transfer radical polymerization using deep reinforcement learning. Mol. Syst. Des. Eng. 2018, 3, 496–508. [CrossRef]

99. Cassola, S.; Duhovic, M.; Schmidt, T.; May, D. Machine learning for polymer composites process simulation—A review. Compos. Part B Eng. 2022, 246, 110208. [CrossRef]

100. Zhang, J.; Yang, J.; Zhang, Y.; Bevan, M.A. Controlling colloidal crystals via morphing energy landscapes and reinforcement learning. Sci. Adv. 2020, 6, eabd6716. [CrossRef]

101. Whitelam, S.; Tamblyn, I. Learning to grow: Control of material self-assembly using evolutionary reinforcement learning. Phys. Rev. E 2020, 101, 052604. [CrossRef]

102. Li, R.; Zhang, C.; Xie, W.; Gong, Y.; Ding, F.; Dai, H.; Chen, Z.; Yin, F.; Zhang, Z. Deep reinforcement learning empowers automated inverse design and optimization of photonic crystals for nanoscale laser cavities. Nanophotonics 2023, 12, 319–334. [CrossRef] [PubMed]

103. Tovey, S.; Zimmer, D.; Lohrmann, C.; Merkt, T.; Koppenhoefer, S.; Heuthe, V.L.; Bechinger, C.; Holm, C. Environmental effects on emergent strategy in micro-scale multi-agent reinforcement learning. arXiv 2023, arXiv:2307.00994.

104. Chatterjee, S.; Jacobs, W.M. Multiobjective Optimization for Targeted Self-Assembly among Competing Polymorphs. Phys. Rev. X 2025, 15, 011075. [CrossRef]

105. Lieu, U.T.; Yoshinaga, N. Dynamic control of self-assembly of quasicrystalline structures through reinforcement learning. Soft Matter 2025, 21, 514–525. [CrossRef]

106. Yang, Q.; Cao, W.; Meng, W.; Si, J. Reinforcement-learning-based tracking control of waste water treatment process under realistic system conditions and control performance requirements. IEEE Trans. Syst. Man Cybern. Syst. 2021, 52, 5284–5294. [CrossRef]

107. Chen, M.; Cui, Y.; Wang, X.; Xie, H.; Liu, F.; Luo, T.; Zheng, S.; Luo, Y. A reinforcement learning approach to irrigation decision-making for rice using weather forecasts. Agric. Water Manag. 2021, 250, 106838. [CrossRef]

108. Sundui, B.; Ramirez Calderon, O.A.; Abdeldayem, O.M.; Lázaro-Gil, J.; Rene, E.R.; Sambuu, U. Applications of machine learning algorithms for biological wastewater treatment: Updates and perspectives. Clean Technol. Environ. Policy 2021, 23, 127–143. [CrossRef]

109. Bonny, T.; Kashkash, M.; Ahmed, F. An efficient deep reinforcement machine learning-based control reverse osmosis system for water desalination. Desalination 2022, 522, 115443. [CrossRef]

110. Zhu, M.; Wang, J.; Yang, X.; Zhang, Y.; Zhang, L.; Ren, H.; Wu, B.; Ye, L. A review of the application of machine learning in water quality evaluation. Eco-Environ. Health 2022, 1, 107–116. [CrossRef] [PubMed]

111. Croll, H.C.; Ikuma, K.; Ong, S.K.; Sarkar, S. Reinforcement learning applied to wastewater treatment process control optimization: Approaches, challenges, and path forward. Crit. Rev. Environ. Sci. Technol. 2023, 53, 1775–1794. [CrossRef]

112. Alvi, M.; Batstone, D.; Mbamba, C.K.; Keymer, P.; French, T.; Ward, A.; Dwyer, J.; Cardell-Oliver, R. Deep learning in wastewater treatment: A critical review. Water Res. 2023, 245, 120518. [CrossRef]

113. Taqvi, S.A.A.; Zabiri, H.; Tufa, L.D.; Uddin, F.; Fatima, S.A.; Maulud, A.S. A review on data-driven learning approaches for fault detection and diagnosis in chemical processes. ChemBioEng Rev. 2021, 8, 239–259. [CrossRef]

114. Wang, Y.; Pan, Z.; Yuan, X.; Yang, C.; Gui, W. A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network. ISA Trans. 2020, 96, 457–467. [CrossRef]

115. Baldea, M.; Georgiou, A.T.; Gopaluni, B.; Mercangöz, M.; Pantelides, C.C.; Sheth, K.; Zavala, V.M.; Georgakis, C. From automated to autonomous process operations. Comput. Chem. Eng. 2025, 196, 109064. [CrossRef]

116. Zhou, Z.; Li, X.; Zare, R.N. Optimizing Chemical Reactions with Deep Reinforcement Learning. Acs Cent. Sci. 2017, 3, 1337–1344. [CrossRef] [PubMed]

117. Khan, A.; Lapkin, A. Searching for optimal process routes: A reinforcement learning approach. Comput. Chem. Eng. 2020, 141, 107027. [CrossRef]

118. Park, S.; Han, H.; Kim, H.; Choi, S. Machine learning applications for chemical reactions. Chem.-Asian J. 2022, 17, e202200203. [CrossRef] [PubMed]

119. Neumann, M.; Palkovits, D.S. Reinforcement learning approaches for the optimization of the partial oxidation reaction of methane. Ind. Eng. Chem. Res. 2022, 61, 3910–3916. [CrossRef]

120. Zhang, C.; Lapkin, A.A. Reinforcement learning optimization of reaction routes on the basis of large, hybrid organic chemistry–synthetic biological, reaction network data. React. Chem. Eng. 2023, 8, 2491–2504. [CrossRef]

121. Hoque, A.; Surve, M.; Kalyanakrishnan, S.; Sunoj, R.B. Reinforcement Learning for Improving Chemical Reaction Performance. J. Am. Chem. Soc. 2024, 146, 28250–28267. [CrossRef]

122. Meuwly, M. Machine learning for chemical reactions. Chem. Rev. 2021, 121, 10218–10239. [CrossRef]

123. He, Z.; Tran, K.P.; Thomassey, S.; Zeng, X.; Xu, J.; Yi, C. A deep reinforcement learning based multi-criteria decision support system for optimizing textile chemical process. Comput. Ind. 2021, 125, 103373. [CrossRef]

124. Hubbs, C.D.; Li, C.; Sahinidis, N.V.; Grossmann, I.E.; Wassick, J.M. A deep reinforcement learning approach for chemical production scheduling. Comput. Chem. Eng. 2020, 141, 106982. [CrossRef]

125. Jitchaiyapoom, T.; Panjapornpon, C.; Bardeeniz, S.; Hussain, M.A. Production Capacity Prediction and Optimization in the Glycerin Purification Process: A Simulation-Assisted Few-Shot Learning Approach. Processes 2024, 12, 661. [CrossRef]

126. Zhu, Z.; Yang, M.; He, W.; He, R.; Zhao, Y.; Qian, F. A deep reinforcement learning approach to gasoline blending real-time optimization under uncertainty. Chin. J. Chem. Eng. 2024, 71, 183–192. [CrossRef]

127. Powell, K.M.; Machalek, D.; Quah, T. Real-time optimization using reinforcement learning. Comput. Chem. Eng. 2020, 143, 107077. [CrossRef]

128. Quah, T.; Machalek, D.; Powell, K.M. Comparing Reinforcement Learning Methods for Real-Time Optimization of a Chemical Process. Processes 2020, 8, 1497. [CrossRef]

129. Panzer, M.; Bender, B. Deep reinforcement learning in production systems: A systematic literature review. Int. J. Prod. Res. 2022, 60, 4316–4341. [CrossRef]

130. Rajasekhar, N.; Radhakrishnan, T.K.; Samsudeen, N. Exploring reinforcement learning in process control: A comprehensive survey. Int. J. Syst. Sci. 2025, 1–30

131. Garcıa, J.; Fernández, F. A comprehensive survey on safe reinforcement learning. J. Mach. Learn. Res. 2015, 16, 1437–1480.

132. Brunke, L.; Greeff, M.; Hall, A.W.; Yuan, Z.; Zhou, S.; Panerati, J.; Schoellig, A.P. Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning. arXiv 2021. [CrossRef]

133. Achiam, J.; Held, D.; Tamar, A.; Abbeel, P. Constrained policy optimization. In Proceedings of the International conference on machine learning. PMLR, Sydney, Australia, 6–11 August 2017; pp. 22–31.

134. Gu, S.; Yang, L.; Du, Y.; Chen, G.; Walter, F.; Wang, J.; Knoll, A. A Review of Safe Reinforcement Learning: Methods, Theory and Applications. arXiv 2024, arXiv:2205.10330. [CrossRef]

135. Achiam, J.; Held, D.; Tamar, A.; Abbeel, P. Constrained Policy Optimization. arXiv 2017. [CrossRef]

136. Feinberg, E.A. Constrained Discounted Markov Decision Processes and Hamiltonian Cycles. Math. Oper. Res. 2000, 25, 130–140. [CrossRef]

137. Leveson, N.G.; Stephanopoulos, G. A system-theoretic, control-inspired view and approach to process safety. AIChE J. 2013,60, 2–14. [CrossRef]

138. Wu, Z.; Christofides, P.D. Process Operational Safety and Cybersecurity; Springer: Cham, Switzerland, 2021.

139. Braniff, A.; Akundi, S.S.; Liu, Y.; Dantas, B.; Niknezhad, S.S.; Khan, F.; Pistikopoulos, E.N.; Tian, Y. Real-time process safety and systems decision-making toward safe and smart chemical manufacturing. Digit. Chem. Eng. 2025, 15, 100227. [CrossRef]

140. Savage, T.; Zhang, D.; Mowbray, M.; Río Chanona, E.A.D. Model-free safe reinforcement learning for chemical processes using Gaussian processes. IFAC-PapersOnLine 2021, 54, 504–509. [CrossRef]

141. Pan, E.; Petsagkourakis, P.; Mowbray, M.; Zhang, D.; del Rio-Chanona, A. Constrained Q-learning for batch process optimization. IFAC-PapersOnLine 2021, 54, 492–497. [CrossRef]

142. Petsagkourakis, P.; Sandoval, I.O.; Bradford, E.; Galvanin, F.; Zhang, D.; Rio-Chanona, E.A.D. Chance constrained policy optimization for process control and optimization. J. Process Control 2022, 111, 35–45. [CrossRef]

143. Kim, Y.; Oh, T.H. Model-based safe reinforcement learning for nonlinear systems under uncertainty with constraints tightening approach. Comput. Chem. Eng. 2024, 183, 108601. [CrossRef]

144. Kim, Y.; Lee, J.M. Model-based reinforcement learning for nonlinear optimal control with practical asymptotic stability guarantees. AIChE J. 2020, 66, e16544. [CrossRef]

145. Wang, Y.; Wu, Z. Control Lyapunov-barrier function-based safe reinforcement learning for nonlinear optimal control. AIChE J. 2024, 70, e18306. [CrossRef]

146. Bo, S.; Agyeman, B.T.; Yin, X.; Liu, J. Control invariant set enhanced safe reinforcement learning: Improved sampling efficiency, guaranteed stability and robustness. Comput. Chem. Eng. 2023, 179, 108413. [CrossRef]

147. Wang, Y.; Xiao, M.; Wu, Z. Safe Transfer-Reinforcement-Learning-Based Optimal Control of Nonlinear Systems. IEEE Trans Cybern. 2024, 54, 7272–7284. [CrossRef]

148. Mehta, S.; Ricardez-Sandoval, L.A. Integration of Design and Control of Dynamic Systems under Uncertainty: A New Back-Off Approach. Ind. Eng. Chem. Res. 2016, 55, 485–498. [CrossRef]

149. Rafiei, M.; Ricardez-Sandoval, L.A. Stochastic Back-Off Approach for Integration of Design and Control Under Uncertainty. Ind. Eng. Chem. Res. 2018, 57, 4351–4365. [CrossRef]

150. Bradford, E.; Imsland, L.; Zhang, D.; Del Rio Chanona, E.A. Stochastic data-driven model predictive control using gaussian processes. Comput. Chem. Eng. 2020, 139, 106844. [CrossRef]

151. Ames, A.D.; Xu, X.; Grizzle, J.W.; Tabuada, P. Control Barrier Function Based Quadratic Programs for Safety Critical Systems. IEEE Trans. Autom. Control 2017, 62, 3861–3876. [CrossRef]

152. Artstein, Z. Stabilization with relaxed controls. Nonlinear Anal. Theory Methods Appl. 1983, 7, 1163–1173. [CrossRef]

153. Wieland, P.; Allgöwer, F. Constructive Safety Using Control Barrier Functions. Ifac Proc. Vol. 2007, 40, 462–467. [CrossRef]

154. Clark, A. Control barrier functions for complete and incomplete information stochastic systems. In Proceedings of the 2019 American Control Conference (ACC), Philadelphia, PA, USA, 10–12 July 2019; pp. 2928–2935.

155. Kim, Y.; Kim, J.W. Safe model-based reinforcement learning for nonlinear optimal control with state and input constraints. AIChE J. 2022, 68, e17601. [CrossRef]

156. Blanchini, F. Set invariance in control. Automatica 1999, 35, 1747–1767. [CrossRef]

157. Towers, M.; Terry, J.K.; Kwiatkowski, A.; Balis, J.U.; De Cola, G.; Deleu, T.; Goulão, M.; Kallinteris, A.; KG, A.; Krimmel, M.; et al. Gymnasium, version v0.28.1; Zenodo: Geneva, Switzerland, 2023. [CrossRef]

158. Brockman, G.; Cheung, V.; Pettersson, L.; Schneider, J.; Schulman, J.; Tang, J.; Zaremba, W. OpenAI Gym. arXiv 2016, arXiv:1606.01540. [CrossRef]

159. Lange, R.T. gymnax: A JAX-based Reinforcement Learning Environment Library. Version 0.0 2022, 4. Available online: https://pypi.org/project/gymnax/ (accessed on 27 May 2025).

160. Freeman, C.D.; Frey, E.; Raichuk, A.; Girgin, S.; Mordatch, I.; Bachem, O. Brax–a differentiable physics engine for large scale rigid body simulation. arXiv 2021, arXiv:2106.13281.

161. Bonnet, C.; Luo, D.; Byrne, D.; Surana, S.; Abramowitz, S.; Duckworth, P.; Coyette, V.; Midgley, L.I.; Tegegn, E.; Kalloniatis, T.; et al. Jumanji: A Diverse Suite of Scalable Reinforcement Learning Environments in JAX. arXiv 2024. [CrossRef]

162. Koyamada, S.; Okano, S.; Nishimori, S.; Murata, Y.; Habara, K.; Kita, H.; Ishii, S. Pgx: Hardware-accelerated parallel game simulators for reinforcement learning. Adv. Neural Inf. Process. Syst. 2024, 36. [CrossRef]

163. Bloor, M.; Torraca, J.; Sandoval, I.O.; Ahmed, A.; White, M.; Mercangöz, M.; Tsay, C.; Chanona, E.A.D.R.; Mowbray, M. PC-Gym: Benchmark Environments For Process Control Problems. arXiv 2024. [CrossRef]

164. OpenAI.; Berner, C.; Brockman, G.; Chan, B.; Cheung, V.; D˛ebiak, P.; Dennison, C.; Farhi, D.; Fischer, Q.; Hashme, S.; et al. Dota 2 with Large Scale Deep Reinforcement Learning. arXiv 2019, arXiv:1912.06680. [CrossRef]

165. OpenAI.; Akkaya, I.; Andrychowicz, M.; Chociej, M.; Litwin, M.; McGrew, B.; Petron, A.; Paino, A.; Plappert, M.; Powell, G. Solving Rubik’s Cube with a Robot Hand. arXiv 2019, arXiv:1910.07113. [CrossRef]

166. Swan, J.; Nivel, E.; Kant, N.; Hedges, J.; Atkinson, T.; Steunebrink, B. Challenges for Reinforcement Learning. In The Road to General Intelligence; Studies in Computational Intelligence; Springer International Publishing: Cham, Switzerland, 2022; Volume 1049; pp. 33–38.

167. Cheng, Y.; Zhang, C.; Zhang, Z.; Meng, X.; Hong, S.; Li, W.; Wang, Z.; Wang, Z.; Yin, F.; Zhao, J.; et al. Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects. arXiv 2024, arXiv:2401.03428. [CrossRef]

168. Mnih, V.; Kavukcuoglu, K.; Silver, D.; Rusu, A.A.; Veness, J.; Bellemare, M.G.; Graves, A.; Riedmiller, M.; Fidjeland, A.K.; Ostrovski, G.; et al. Human-level control through deep reinforcement learning. Nature 2015, 518, 529–533. [CrossRef]

169. Schaul, T.; Quan, J.; Antonoglou, I.; Silver, D. Prioritized Experience Replay. arXiv 2016, arXiv:1511.05952. [CrossRef]

170. Tobin, J.; Fong, R.; Ray, A.; Schneider, J.; Zaremba, W.; Abbeel, P. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. arXiv 2017, arXiv:1703.06907.

171. Taylor, M.E.; Stone, P. Transfer Learning for Reinforcement Learning Domains: A Survey. J. Mach. Learn. Res. 2009, 10, 1633–1685.

172. Lazaric, A., Transfer in Reinforcement Learning: A Framework and a Survey. In Reinforcement Learning: State-of-the-Art; Wiering, M., van Otterlo, M., Eds.; Springer: Berlin/Heidelberg, Germany, 2012; pp. 143–173. [CrossRef]

173. Zhu, Z.; Lin, K.; Jain, A.K.; Zhou, J. Transfer Learning in Deep Reinforcement Learning: A Survey. IEEE Trans. Pattern Anal. Mach. Intell. 2023, 45, 13344–13362. [CrossRef] [PubMed]

174. Ada, S.E.; Ugur, E.; Akin, L.H. Generalization in Transfer Learning. arXiv 2024, arXiv:1909.01331.

175. Gao, Q.; Yang, H.; Shanbhag, S.M.; Schweidtmann, A.M. Transfer learning for process design with reinforcement learning. Comput. Aided Chem. Eng. 2023, 52, 2005–2010. [CrossRef]

176. Alam, M.F.; Shtein, M.; Barton, K.; Hoelzle, D. Reinforcement Learning Enabled Autonomous Manufacturing Using Transfer Learning and Probabilistic Reward Modeling. IEEE Control Syst. Lett. 2023, 7, 508–513. [CrossRef]

177. Mowbray, M.; Smith, R.; Del Rio-Chanona, E.A.; Zhang, D. Using process data to generate an optimal control policy via apprenticeship and reinforcement learning. AIChE J. 2021, 67, e17306.

178. Lin, R.; Chen, J.; Xie, L.; Su, H. Facilitating Reinforcement Learning for Process Control Using Transfer Learning: Overview and Perspectives. arXiv 2024, arXiv:2404.00247.

179. Zhu, Z.; Lin, K.; Dai, B.; Zhou, J. Learning Sparse Rewarded Tasks from Sub-Optimal Demonstrations. arXiv 2020, arXiv:2004.00530.

180. Oh, J.; Hessel, M.; Czarnecki, W.M.; Xu, Z.; Hasselt, H.v.; Singh, S.; Silver, D. Discovering Reinforcement Learning Algorithms. arXiv 2020, arXiv:2007.08794.

181. McClement, D.G.; Lawrence, N.P.; Backström, J.U.; Loewen, P.D.; Forbes, M.G.; Gopaluni, B.R. Meta-reinforcement learning fo the tuning of PI controllers: An offline approach. J. Process Control 2022, 118, 139–152. [CrossRef]

182. Kirsch, L.; van Steenkiste, S.; Schmidhuber, J. Improving Generalization in Meta Reinforcement Learning using Learned Objectives. arXiv 2019, arXiv:1910.04098.

183. Rakelly, K.; Zhou, A.; Quillen, D.; Finn, C.; Sergey, L. Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. arXiv 2019, arXiv:1903.08254.

184. Korkmaz, E. A Survey Analyzing Generalization in Deep Reinforcement Learning. arXiv 2024, arXiv:2401.02349v1.

185. Lockwood, O.; Si, M. A Review of Uncertainty for Deep Reinforcement Learning. In Proceedings of the Eighteenth AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, Pomona, CA, USA, 24–28 October 2022; Volume 18.

186. Zhao, X.; Hu, S.; Cho, J.H.; Chen, F. Uncertainty-based Decision Making Using Deep Reinforcement Learning. In Proceedings of the 2019 22th International Conference on Information Fusion (FUSION), Ottawa, ON, Canada, 2–5 July 2019; pp. 1–8. [CrossRef]

187. Wu, P.; Escontrela, A.; Hafner, D.; Abbeel, P.; Goldberg, K. Daydreamer: World models for physical robot learning. In Proceedings of the Conference on Robot Learning, PMLR, Atlanta, GA, USA, 6–9 November 2023; pp. 2226–2240.

188. Diehl, C.; Sievernich, T.S.; Krüger, M.; Hoffmann, F.; Bertram, T. Uncertainty-Aware Model-Based Offline Reinforcement Learning for Automated Driving. IEEE Robot. Autom. Lett. 2023, 8, 1167–1174. [CrossRef]

189. Ghavamzadeh, M.; Mannor, S.; Pineau, J.; Tamar, A. Bayesian Reinforcement Learning: A Survey. arXiv 2016, arXiv:1609.04436.

190. O’Donoghue, B. Variational bayesian reinforcement learning with regret bounds. Adv. Neural Inf. Process. Syst. 2021, 34, 28208–28221.

191. Rana, K.; Dasagi, V.; Haviland, J.; Talbot, B.; Milford, M.; Sünderhauf, N. Bayesian controller fusion: Leveraging control priors in deep reinforcement learning for robotics. Int. J. Robot. Res. 2023, 42, 123–146. [CrossRef]

192. Kang. P: Tobler. PN.: Davan, P. Bavesian reinforcement learning: A basic overview. Neurobiol. Learn. Mem. 2024. 211. 107924 [CrossRef]

193. Roy, S.S.; Everitt, R.G.; Robert, C.P.; Dutta, R. Generalized Bayesian deep reinforcement learning. arXiv 2024, arXiv:2412.11743.

194. Gan, Y.; Dai, Z.; Wu, L.; Liu, W.; Chen, L. Deep Reinforcement Learning and Dempster-Shafer Theory: A Unified Approach to Imbalanced Classification. In Proceedings of the 2023 IEEE 3rd International Conference on Computer Science, Electronic Information Engineering and Intelligent Control Technology (CEI), Wuhan, China, 15–17 December 2023; pp. 67–72.

195. Tang, Y.; Zhou, Y.; Zhou, Y.; Huang, Y.; Zhou, D. Failure Mode and Effects Analysis on the Air System of an Aero Turbofan Engine Using the Gaussian Model and Evidence Theory. Entropy 2023, 25. [CrossRef]

196. Tang, Y.; Fei, Z.; Huang, L.; Zhang, W.; Zhao, B.; Guan, H.; Huang, Y. Failure Mode and Effects Analysis Method on the Air System of an Aircraft Turbofan Engine in Multi-Criteria Open Group Decision-Making Environment. Cybern. Syst. 2025, 1–32. [CrossRef]

197. Huang, F.; Zhang, Y.; Jiang, W.; He, Y.; Deng, X. Intelligent information fusion for conflicting evidence using reinforcement learning and Dempster-Shafer theory. In Proceedings of the 2021 IEEE International Conference on Unmanned Systems (ICUS), Beijing, China, 15–17 October 2021; pp. 190–195.

198. Anyszka, W. Towards an Approximation Theory of Observable Operator Models. arXiv 2024, arXiv:2404.12070.

199. Ha, D.; Schmidhuber, J. World models. arXiv 2018, arXiv:1803.10122.

200. Matsuo, Y.; LeCun, Y.; Sahani, M.; Precup, D.; Silver, D.; Sugiyama, M.; Uchibe, E.; Morimoto, J. Deep learning, reinforcement learning, and world models. Neural Netw. 2022, 152, 267–275. [CrossRef]

201. Chen, C.; Wu, Y.F.; Yoon, J.; Ahn, S. Transdreamer: Reinforcement learning with transformer world models. arXiv 2022, arXiv:2202.09481.

202. Singh, G.; Peri, S.; Kim, J.; Kim, H.; Ahn, S. Structured world belief for reinforcement learning in pomdp. In Proceedings of the International Conference on Machine Learning. PMLR, Virtual, 18–24 July 2021; pp. 9744–9755.

203. Subramanian, J.; Sinha, A.; Seraj, R.; Mahajan, A. Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems. J. Mach. Learn. Res. 2022, 23, 1–83.

204. Hafner, D.; Pasukonis, J.; Ba, J.; Lillicrap, T. Mastering diverse domains through world models. arXiv 2023, arXiv:2301.04104.

205. Luo, F.M.; Xu, T.; Lai, H.; Chen, X.H.; Zhang, W.; Yu, Y. A survey on model-based reinforcement learning. Sci. China Inf. Sci. 2024, 67, 121101. [CrossRef]

206. Uehara, M.; Sekhari, A.; Lee, J.D.; Kallus, N.; Sun, W. Provably Efficient Reinforcement Learning in Partially Observable Dynamical Systems. Adv. Neural Inf. Process. Syst. 2022, 35, 578–592.

207. Uehara, M.; Kiyohara, H.; Bennett, A.; Chernozhukov, V.; Jiang, N.; Kallus, N.; Shi, C.; Sun, W. Future-Dependent Value-Based Off-Policy Evaluation in POMDPs. In Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems, New Orleans, LA, USA, 10–16 December 2023.

208. Wang, R.; Du, S.S.; Yang, L.; Salakhutdinov, R.R. On reward-free reinforcement learning with linear function approximation. Adv. Neural Inf. Process. Syst. 2020, 33, 17816–17826.

209. Jin, C.; Yang, Z.; Wang, Z.; Jordan, M.I. Provably efficient reinforcement learning with linear function approximation. In Proceedings of the Conference on Learning Theory, PMLR, Graz, Austria, 9–12 July 2020; pp. 2137–2143.

210. Long, J.; Han, J. Reinforcement Learning with Function Approximation: From Linear to Nonlinear. arXiv 2022, arXiv:2302.09703.

211. Zhao, H.; He, J.; Gu, Q. A nearly optimal and low-switching algorithm for reinforcement learning with general function Adv. Neural Inf. Process. Syst. 2024 37

212. Shanahan, M. Talking About Large Language Models. arXiv 2023. [CrossRef]

213. Brown, T.B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. Language Models Are Few-Shot Learners. arXiv 2020. [CrossRef]

214. Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. LLaMA: Open and Efficient Foundation Language Models. arXiv 2023. [CrossRef]

215. Bubeck, S.; Chandrasekaran, V.; Eldan, R.; Gehrke, J.; Horvitz, E.; Kamar, E.; Lee, P.; Lee, Y.T.; Li, Y.; Lundberg, S.; et al. Sparks of Artificial General Intelligence: Early experiments with GPT-4. arXiv 2023, arXiv:2303.12712.

216. Wei, J.; Tay, Y.; Bommasani, R.; Raffel, C.; Zoph, B.; Borgeaud, S.; Yogatama, D.; Bosma, M.; Zhou, D.; Metzler, D.; et al. Emergent Abilities of Large Language Models. arXiv 2022, arXiv:2206.07682.

217. Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q.V.; Zhou, D. Chain-of-thought prompting elicits reasoning in large language models. Adv. Neural Inf. Process. Syst. 2022, 35, 24824–24837.

218. Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T.; Cao, Y.; Narasimhan, K. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. Adv. Neural Inf. Process. Syst. 2023, 36, 11809–11822.

219. Besta, M.; Blach, N.; Kubicek, A.; Gerstenberger, R.; Podstawski, M.; Gianinazzi, L.; Gajda, J.; Lehmann, T.; Niewiadomski, H.; Nyczyk, P.; et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vancouver, BC, Canada, 20–27 February 2024; Volume 38, pp. 17682–17690

220. Anderson, P.; Wu, Q.; Teney, D.; Bruce, J.; Johnson, M.; Sunderhauf, N.; Reid, I.; Gould, S.; Van Den Hengel, A. Vision-and Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments. In Proceedings of the 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 18–23 June 2018; pp. 3674–3683. [CrossRef]

221. Jiang, Y.; Gu, S.; Murphy, K.; Finn, C. Language as an Abstraction for Hierarchical Deep Reinforcement Learning. arXiv 2019. [CrossRef]

222. Cao, Y.; Zhao, H.; Cheng, Y.; Shu, T.; Chen, Y.; Liu, G.; Liang, G.; Zhao, J.; Yan, J.; Li, Y. Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods. IEEE Trans. Neural Netw. Learn. Syst. 2024, 1–21. [CrossRef] [PubMed]

223. Li, H.; Yang, X.; Wang, Z.; Zhu, X.; Zhou, J.; Qiao, Y.; Wang, X.; Li, H.; Lu, L.; Dai, J. Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft. arXiv 2024. [CrossRef]

224. Chakraborty, S.; Weerakoon, K.; Poddar, P.; Elnoor, M.; Narayanan, P.; Busart, C.; Tokekar, P.; Bedi, A.S.; Manocha, D. RE-MOVE: An Adaptive Policy Design for Robotic Navigation Tasks in Dynamic Environments via Language-Based Feedback. arXiv 2023. [CrossRef]

225. Pang, J.C.; Yang, X.Y.; Yang, S.H.; Yu, Y. Natural Language-conditioned Reinforcement Learning with Inside-out Task Language Development and Translation. arXiv 2023. [CrossRef]

226. Vyas, J.; Mercangöz, M. Autonomous Industrial Control using an Agentic Framework with Large Language Models. arXiv 2024. [CrossRef]

227. Vouros, G.A. Explainable deep reinforcement learning: State of the art and challenges. Acm. Comput. Surv. 2022, 55, 1–39. [CrossRef]

228. Canese, L.; Cardarilli, G.C.; Di Nunzio, L.; Fazzolari, R.; Giardino, D.; Re, M.; Spanó, S. Multi-agent reinforcement learning: A review of challenges and applications. Appl. Sci. 2021, 11, 4948. [CrossRef]

229. Wong, A.; Bäck, T.; Kononova, A.V.; Plaat, A. Deep multiagent reinforcement learning: Challenges and directions. Artif. Intell. Rev. 2023, 56, 5023–5056. [CrossRef]

230. Dulac-Arnold, G.; Levine, N.; Mankowitz, D.J.; Li, J.; Paduraru, C.; Gowal, S.; Hester, T. Challenges of real-world reinforcement learning: Definitions, benchmarks and analysis. Mach. Learn. 2021, 110, 2419–2468. [CrossRef]

231. Hutsebaut-Buysse, M.; Mets, K.; Latré, S. Hierarchical reinforcement learning: A survey and open research challenges. Mach. Learn. Knowl. Extr. 2022, 4, 172–221. [CrossRef]

232. Srinivasan, A. Reinforcement Learning: Advancements, Limitations, and Real-world Applications. Int. J. Sci. Res. Eng. Manag. (IJSREM) 2023, 7, 8. [CrossRef]

233. Casper, S.; Davies, X.; Shi, C.; Gilbert, T.K.; Scheurer, J.; Rando, J.; Freedman, R.; Korbak, T.; Lindner, D.; Freire, P.; et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv 2023, arXiv:2307.15217.

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.