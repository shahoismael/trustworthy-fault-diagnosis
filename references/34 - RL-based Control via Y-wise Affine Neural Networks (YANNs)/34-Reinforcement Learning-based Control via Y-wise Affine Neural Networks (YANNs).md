# Reinforcement Learning-based Control via Y-wise Afine Neural Networks (YANNs)

Austin Branif<sup>a</sup>, Yuhe Tian<sup>a,∗</sup>

<sup>a</sup>Department of Chemical and Biomedical Engineering, West Virginia University, Morgantown, West Virginia, United States

## Abstract

This work presents a novel reinforcement learning (RL) algorithm based on Y-wise Afine Neural Networks (YANNs). YANNs provide an interpretable neural network which can exactly represent known piecewise afine functions of arbitrary input and output dimensions defined on any amount of polytopic subdomains. One representative application of YANNs is to reformulate explicit solutions of multi-parametric linear model predictive control. Built on this, we propose the use of YANNs to initialize RL actor and critic networks, which enables the resulting YANN-RL control algorithm to start with the confidence of linear optimal control. The YANN-actor is initialized by representing the multi-parametric control solutions obtained via ofline computation using an approximated linear system model. The YANN-critic represents the explicit form of the state-action value function for the linear system and the reward function as the objective in an optimal control problem (OCP). Additional network layers are injected to extend YANNs for nonlinear expressions, which can be trained online by directly interacting with the true complex nonlinear system. In this way, both the policy and state-value functions exactly represent a linear OCP initially and are able to eventually learn the solution of a general nonlinear OCP. Continuous policy improvement is also implemented to provide heuristic confidence that the linear OCP solution serves as an efective lower bound to the performance of RL policy. The YANN-RL algorithm is demonstrated on a clipped pendulum and a safety-critical chemical-reactive system. Our results show that

YANN-RL significantly outperforms the modern RL algorithm using deep deterministic policy gradient, especially when considering safety constraints.

Keywords: Reinforcement Learning, Model Predictive Control, Neural Networks, Machine Learning, Explicit Model Predictive Control, Multi-Parametric Programming

## 1. Introduction

Reinforcement learning (RL) has emerged as one of the most promising technologies in the modern era [1, 2]. There has been a significant surge in the research focused on this problem-solving strategy, since the seminal work showing the ability to provide human-like control for video game playing by integrating neural networks (NNs) into the RL algorithms [3]. RL has shown great promise in many areas including game playing [4, 5], robotics [6], production scheduling [7, 8], process design [9, 10], etc.

With its origin in the optimal control theory, RL has also been applied for the direct control of process systems. RL has been shown to be an efective control algorithm for bioprocessing [11], distillation columns [12], chemical reactors [13], batch processing [14], and multi-tank systems [15]. Despite these advances, RL is yet to be widely adopted for the control of chemical and energy systems. The main barriers include the inherent distrust in the exploration phase of learning and the overall lack of interpretability [16, 17, 18]. In RL, the exploration phase allows an agent to discover new actions that could lead to more optimal behavior, however this can be unsafe since it typically involves trying random and untested actions which could lead to undesirable or unsafe behavior. These issues prevent the RL-based control algorithms from practical implementations, especially in safety-critical process systems which require confidence in a controller’s ability to maintain safe and stable operations [19, 20].

To address these challenges, extensive eforts have been made in recent years. One class of strategies is to pre-train a RL policy network using data generated from other more trusted control approaches, such as Model Predictive Control (MPC) [21, 22]. The RL agent then builds on this policy network to directly compute the control actions. If it is desired to maintain the premise of model-free RL, a linear MPC can be adopted since linear system models can easily be approximated through a variety of techniques (e.g., system identification) [23]. If a reliable high-fidelity system model is available, model-based RL approaches can be used instead. Many of the model-based RL approaches can also provide some sort of confidence regarding safety, stability, or both together [24, 25, 26, 27]. Another class of strategies use RL in a more indirect way as a supervisory role to PID control [28, 29, 30]. This is a promising approach to improving PID-based control but sufers from the same limitations since each sub-level controller is constrained to a singleinput-single-output (SISO) control law [31, 32]. RL has also been used for the tuning of advanced (economic) MPC [33]. In this context, the RL agent again acts in a supervisory manner by not directly computing the control actions but instead guiding controllers with guaranteed theoretical properties to make better decisions based on system feedback data [34]. An interesting example of a combined RL and MPC approach is the AC4MPC algorithm which leverages the suggested RL-based control actions as a warm-start for an MPC problem while the RL-based value function provides a better estimated terminal cost for the problem [35]. Many other works have studied various ways to leverage the benefits of both MPC and RL simultaneously [36, 37].

For the development of safer RL algorithms, Lyapounov-based approaches have gained significant interest. In these algorithms, an approximate Lyapunov function is learned online. Control actions determined by the RL agent are generated in order to satisfy certain stability or safety conditions using the Lyapunov surrogate [38, 39]. This has also been extended to control Lyapunov barrier functions (CLBFs) and stochastic CLBFs which can simultaneously provide both safety and stability properties [40, 41]. Other forms of safe RL include: safe exploration for linear systems [42], recovering safety guarantees after an ofline training phase [43], integrating principles of linear robust MPC [44], control invariant sets (CIS) [45, 46], and Gaussian process models with chance constraints [47]. However, these approaches often require significant prior knowledge about the system, high-fidelity system model, and/or intensive computational power. Furthermore, these approaches must go through an exploration phase when learning the system model and/or training the RL agent which may be inherently unsafe to be implemented in practice [48]. To this end, a RL-based control algorithm is essential yet currently lacking which can improve the interpretability, stability, and computational eficiency while avoiding the unsafe exploration phase to be implemented with confidence in safety-critical systems.

In this work, we present an RL algorithm built on the Y-wise Afine Neural Networks (YANNs), as depicted in Fig. 1. Developed in our previous work [49], YANNs are a specialized neural network that can exactly represent known piecewise afine functions of arbitrary input and output dimensions defined on any amount of polytopic subdomains. Multi-parametric model predictive control (mp-MPC) presents an important application of YANNs, which theoretically computes the optimal control laws as piecewise afine functions of system states, outputs, setpoints, and disturbances [50]. Given this, the actor and the critic in an actor critic algorithm can be initialized to represent the explicit solutions and the objective to an optimal control problem via mp-MPC. Thus, YANNs can be utilized as a basis to provide an interpretable, eficient, and confident starting point for RL algorithms. This allows the algorithm to start with the full theoretical and rigorous guarantees of linear optimal control, and thereby skipping the exploration phase of RL entirely. These actor and critic networks can be created in such a way that they can approximate general nonlinear functions of arbitrary complexity using the techniques developed in this work. We further discuss how the algorithm can be continuously improved which gives confidence that the parameterized control policy will never be worse than the linear multi-parametric control policy found by solving the linear optimal control problem (OCP). The radical improvement in the control of safety-critical systems where safety constraint adherence is crucial is highlighted in a case study of a chemical reactive process.

![](images/efa1848e9fed5a296330197b163bb8bcc414a3c043c176420d32bdda5c694200.jpg)  
Figure 1: A schematic of the proposed RL algorithm based on YANNs. (a) Multi-parametric/explicit MPC based on approximated linear model  
(b) YANNs to exactly represent explicit control laws as piecewise linear functions (c) RL initialized with YANNs for optimal nonlinear control

The remaining sections of this paper are organized as follows: Section 2 presents a brief overview of the necessary mathematical foundations and lay out the nomenclature for this work. Section 3 reviews YANNs for exact representation of piecewise afine functions and extend the network formulation to introduce nonlinearity. Section 4 introduces YANN-based RL principles including the YANN-actor, YANN-critic, and the overall RL algorithm. Section 5 demonstrates the advantages of YANN-RL through two case studies: (i) clipped pendulum, and (ii) safety-critical chemical reactor. Section 6 gives the concluding remarks and discusses future research directions.

## 2. Theoretical Background

## 2.1. Introduction to reinforcement learning

RL is a methodology that is used to solve dynamic programming problems of various kinds. It is most commonly introduced as a way to solve Markov decision processes (MDPs) which are a special class of stochastic dynamic programs. However, in this work, we apply the principles of RL with the intentions of making as few assumptions about the true system as possible while using deterministic policies. For a more thorough introduction than what follows, the readers are referred to [51, 52, 53].

We present the nomenclature and concepts of RL from a control-theoretic perspective. In classical control theory, a cost function is typically used to determine the optimality whereas in RL a reward function is used to this purpose. This can easily be resolved by relating cost and reward as $C = - R$ , where C is the cost and R is the reward. Furthermore, we will use u to denote an action onto the system, as is typical in controls, instead of the nomenclature a which is more common in RL. In order to prevent confusion between NN inputs and states, the states of the system or environment are denoted as s which is consistent with RL. A simple schematic conceptualizing the use of this terminology is presented in Fig. 2.

![](images/199df4ed7ad65488729b03d504250d25cada4bb155418c0707511c6324ec855b.jpg)  
Figure 2: The terminology of RL-based control used in this work.

The central components of RL are a well-designed cost function $( C ( s , u ) )$ , the state value function $\left( V ( s ) \right)$ or the state-action value function $( Q ( s , u ) )$ , and the policy $( \pi ( s ) )$ ). The cost function is defined in such a way that optimizing control actions with respect to it yields the desired system behavior $( \mathrm { e . g . } )$ , quadratically penalizing the distance between the system state and the origin for regulation). The value function can be interpreted as a representation of how good it is for the system to exist in a particular state. It is also referred to as the cost-to-go function, which aims to provide the infinite time horizon summed cost (usually discounted) following an optimal system trajectory. For a deterministic system and policy, this is represented by $\operatorname { E q }$ 1 where $s _ { t }$ is the system state at time $t , t _ { 0 }$ is the initial time, $\boldsymbol { u } _ { t } ^ { \star }$ is the optimal action for state $s _ { t }$ and is found by following an optimal policy $\left( \pi ^ { \star } ( s _ { t } ) \right)$ , and $\gamma$ is a discount factor used to ensure the convergence of the function in infinite time.

$$
V (s _ {0}) = \sum_ {t = t _ {0}} ^ {\infty} \gamma^ {t} C (s _ {t}, u _ {t} ^ {\star})\tag{1}
$$

The state-action value function, sometimes referred to as the quality function, is heavily related to the state value function. It provides slightly more insight as it aims to give a description of how good it is to take a particular action while being in a certain state and then follow the optimal policy thereafter. This relationship is given by Eq. 2 where $u _ { 0 }$ is the action taken on the system at time $t = t _ { 0 }$

$$
Q (s _ {0}, u _ {0}) = C (s _ {0}, u _ {0}) + \sum_ {t = t _ {1}} ^ {\infty} \gamma^ {t} C (s _ {t}, u _ {t} ^ {\star})\tag{2}
$$

From this, Bellman’s recursive optimality principle can be used to define optimality conditions for these functions when minimizing cost, as given in Eqs. 3-4.

$$
V ^ {\star} (s _ {t}) = \min _ {u _ {t}} C (s _ {t}, u _ {t}) + \gamma V ^ {\star} (s _ {t + 1})\tag{3}
$$

$$
Q ^ {\star} (s _ {t}, u _ {t}) = C (s _ {t}, u _ {t}) + \min _ {u _ {t + 1}} \gamma Q ^ {\star} (s _ {t + 1}, u _ {t + 1})\tag{4}
$$

These optimality conditions form the basis that modern RL algorithms are developed from. Typically, a NN is used to approximate one of these functions parametrically and is updated with the goal to satisfy the optimality conditions based on feedback from the system.

## 2.2. Linear optimal control

This work will use the theoretical foundations of three types of linear optimal control: linear quadratic regulator (LQR), MPC, and mp-MPC. The OCP for LQR is defined by Eq. 5

$$
\begin{array}{l l} \min _ {u} & \sum_ {t _ {1}} ^ {\infty} \left(s _ {t} ^ {T} Q s _ {t} + u _ {t} ^ {T} R u _ {t}\right) \\ \text {s.t.} & s _ {t + 1} = A s _ {t} + B u _ {t}, \quad \forall t \in \{1, 2, \dots , \infty \} \end{array}\tag{5}
$$

where $s _ { t } \in \mathbb { R } ^ { n }$ is the system state at time step t, $u _ { t } \in \mathbb { R } ^ { m }$ is the control input at time step t, A and B are system matrices, Q and R are symmetric positive semi-definite and positive definite weighting matrices. This is a unique problem in optimal control theory since both the optimal control policy and the optimal state value function can be easily derived. Using the Bellman optimality equation (Eq. 3) along with the assumption that the value function takes the form $V ( s ) = s ^ { T } P s$ , the optimal control law is derived as a function of the state value function and the parameters in the OCP as seen in Eq. 6.

$$
u _ {t} = - (R + B ^ {T} P B) ^ {- 1} B P A s _ {t}\tag{6}
$$

Substituting $u _ { t }$ with Eq. 6 in the Bellman equation, the discrete-time algebraic Ricatti equation (DARE) can be derived as Eq. 7.

$$
P = Q + A ^ {T} P A - A ^ {T} P B (R + B ^ {T} P B) ^ {- 1} B ^ {T} P A\tag{7}
$$

This equation can be solved using various techniques (e.g., Schur method). Once a solution for the matrix P is found both the optimal control law and optimal value function can be realized.

The limitation of LQR is that these solutions are only easily found for the unconstrained system. However, in practice it is often necessary to include constraints in the decision-making process (e.g., constraints on control inputs). MPC provides an instrumental tool to handle these constraints, but it typically considers finite time. The MPC problem for linear system regulation is defined by Eq. 8.

$$
\begin{array}{l l} \min _ {u} & \sum_ {t = 0} ^ {N - 1} \left(s _ {t} ^ {T} Q _ {w} s _ {t} + u _ {t} ^ {T} R u _ {t}\right) + s _ {N} ^ {T} P s _ {N} \\ \text {s.t.} & s _ {t + 1} = A s _ {t} + B u _ {t}, \quad \forall t \in \{0, 1, 2, \dots , N - 1 \} \\ & s _ {t} \in \mathcal {S}, u _ {t} \in \mathcal {U}, \quad \forall t \in \{0, 1, 2, \dots , N - 1 \} \\ & s _ {N - 1} \in \mathcal {S} _ {f} \end{array}\tag{8}
$$

where $s _ { t } \in \mathbb { R } ^ { n }$ is the system state at time step t, $u _ { t } \in \mathbb { R } ^ { m }$ is the control input at time step $t , A$ and $B$ are system matrices, $Q _ { w }$ and $R$ are symmetric positive semi-definite and positive definite weighting matrices, $P$ is the solution to DARE (Eq. 7) to give an approximate infinite horizon cost-to-go, $N \in  { \mathbb { N } }$ is the prediction horizon, $S = \{ s \in \mathbb { R } ^ { n } : F _ { s } s \leq g _ { s } \}$ is the state path constraint polytope, $\mathcal { U } = \left. u \in \mathbb { R } ^ { m } : F _ { u } u \leq g _ { u } \right.$ is the input constraint polytope, and $S _ { f } = \{ s \in \mathbb { R } ^ { n } : F _ { f } s \leq g _ { f } \}$ is the terminal set which is a control invariant polytope used for recursive feasibility.

This problem can be solved online with rolling horizon using quadratic programming (QP) algorithms. An alternative approach, which is utilized in this work, is to equivalently reformulate this MPC problem into mp-MPC as given in Eq. 9, to solve using multi-parametric quadratic programming (mp-QP) algorithms.

$$
\begin{array}{r l} J ^ {*} (\theta) = \min _ {u} & u ^ {T} H u + u ^ {T} Z \theta + \theta^ {T} \hat {M} \theta \\ \mathrm{s.t.} & G u \leq S \theta + W \\ & C R _ {A} \theta \leq C R _ {b} \end{array}\tag{9}
$$

where $\theta$ is the parametric set comprising the state s at the current time step t = 1. u is the vector of $u _ { t } , \forall k \in [ 1 , 2 , \cdot \cdot \cdot , N ]$ with N being the prediction horizon from Eq. 8. Constraint matrices G, S, and $C R _ { A }$ , and constraint vectors W and $C R _ { b }$ are derived from the system matrices and polytopic constraints in Eq. 8. Weighting matrices H, Z, and $\hat { M }$ are derived from the system matrices and MPC weighting matrices in Eq. 8. Similar approaches to reformulate other types of MPC problems such a setpoint tracking or disturbance rejection can be used which include necessary additional parameters in the θ vector (e.g., setpoints) [50, 54].

The advantage for mp-MPC is that this problem can be solved ofline to give the optimal control law as a piecewise afine function that inherits all mathematical properties of the original formulation such as recursive feasibility and stability guarantees for the linear system. This ofline solution can also enable faster control since a simpler function needs to be evaluated instead of solving a dynamic optimization problem in real time. The domain of this function is made up of closed connected but non-overlapping convex polytopes that are each a subset of the parametric constraints in Eq. 9. This optimal control law representation is given in Eq. 10.

$$
u ^ {*} (\theta) = \left\{ \begin{array}{c} K _ {1} \theta + r _ {1}, \theta \in C R ^ {1} = \{C R _ {A} ^ {1} \theta \leq C R _ {b} ^ {1} \} \\ \vdots \\ K _ {p} \theta + r _ {p}, \theta \in C R ^ {p} = \{C R _ {A} ^ {p} \theta \leq C R _ {b} ^ {p} \} \end{array} \right.\tag{10}
$$

where $K _ { i }$ and $r _ { i }$ are the coeficient matrix and constant vector for the explicit solution defined on polytope i, and $\{ C R _ { A } ^ { i } \theta \leq C R _ { b } ^ { i } \}$ defines the $i ^ { t h }$ polytope $C R ^ { i }$ (i.e., critical region). An important property of this function is that it is continuous. That is, the solution to the optimal control problem defined on adjacent polytopic subdomains are equivalent at their boundary.

## 3. Y-wise Afine Neural Networks and Nonlinear Extensions

## 3.1. An overview of YANNs

YANNs are a specific architecture that we have developed in our prior work [49]. They are capable of exactly representing piecewise afine functions with any dimensional inputs and outputs defined on any number of polytopic subdomains. The YANN is a 5-layer neural network that exactly reformulates these known functions into a continuously exact representation across the full continuous domain space. YANNs can be leveraged to represent the explicit control solutions obtained from mp-MPC (Eq. 10). In this case, YANNs become a NN-based controller that inherits the theoretical guarantee of stability and recursive feasibility from mp-MPC for linear systems.

![](images/5187267a6c16a04f6491b2c9c7f0ab214a60f2fa10557991a999c520c5937e75.jpg)  
Figure 3: Y-wise Afine Neural Network architecture.

This features a step change from previous approaches that require additional methods to provide resemblance of these essential control-theoretic properties. A YANN is represented by Fig. 3. The first three layers determine the active subdomain for a given input, i.e. to check $\theta \in \{ C R _ { A } ^ { i } \theta \leq C R _ { b } ^ { i } \}$ The last two layers evaluate the appropriate subfunction, i.e. to compute the corresponding $u ^ { * } ( \theta ) = K _ { i } \theta + r _ { i }$ . q is the total amount of inequalities governing the polytopic subdomains in their half-space representation, p is the number of polytopic subdomains (or critical regions), m is the number of outputs for the function, <sup>1</sup> represents the solution to an indicator function defined by a polytopic subdomain, and the input X is the set of parametric variables (e.g., states at the current time step for regulation control). A formal proof of YANNs can be found in Branif and Tian [49] which explains how to determine the specific weights and biases.

YANNs thus provide an attractive tool for RL by being able to initialize policy networks with a control law that has theoretical confidence in its solutions, at least with respect to a linearly approximated model. The second advantage to using YANNs is that, by leveraging inference speed-ups for neural networks, they can evaluate these functions much faster than existing techniques. Two computational examples have been presented in [49] comparing YANNs with the conventional mp-MPC function evaluation via Python Parametric OPtimization Toolbox (PPOPT) [55]: (i) for explicit control defined by 9 critical regions, YANN evaluates seven times faster, (ii) for explicit control defined by over 2000 critical regions, YANN decreases the inference time by more than 30%.

## 3.2. Introducing nonlinearity

Despite the above advantages, the YANN is limited in its ability to learn more complex functions or to be trained from data at all. The initial layers pose a challenge by not being able to be updated via gradient-based approaches (e.g., backpropagation) since they use the binary step function as activation function. The subfunction evaluation layers (layers 4 and 5) are also limited to linear expressibility since they only evaluate the corresponding linear control law. To this end, it is necessary to add nonlinear expressibility to the YANN before it can be used in an algorithm like RL since nonlinear systems may require a more complex control law than the form of piecewise afine function. However, it is essential to retain the mathematical results from mp-MPC to better initialize the network and to deploy the NN-based controller with confidence at the beginning before further training. To enforce this, we add network layers that are initially evaluated to zero for any input. In a straightforward way, this can be accomplished by assigning all weights and biases to 0. But this is not useful as these parameters may not be able to be updated or if they are updated they would all update uniformly, therefore not allowing general nonlinear expressibility. In the proofs that follow, we show how NNs can be formulated with this desired property to retain the initial known mathematics of the YANN, while being trainable for nonlinear expressibility.

## 3.2.1. One-layer neural networks

The problem is to initialize a single layer neural network of arbitrary size such that it will provide a solution of zero for all inputs and still be trainable using modern parameter updating techniques.

Theorem 1 A single-layer neural network with any dimensional input and output that has an even amount of nodes can be initialized with randomly generated weights and biases such that the output is always a vector of zeros for any and all possible inputs if the activation function is defined as zero at an input of zero.

Proof. Consider a single-layer neural network with input $X = [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] \in$ $\mathbb { R } ^ { n }$ and output $Y \in \mathbb { R } ^ { 1 }$ . It has an activation function defined such that $\sigma ( 0 ) = 0$ . Split the nodes into two evenly sized groups. Using some random weight and bias assigning technique, generate weights and biases of the first group of nodes and define them as $W _ { h }$ and $B _ { h }$ respectively. Assign the weights and bias of the second group of nodes as $- W _ { h }$ and $- B _ { h }$ . Solving for

Y gives:

$$
\begin{array}{r l} Y & = \sigma (W X + B) \\ & = \sigma (W _ {h} X + B _ {h} - W _ {h} X - B _ {h}) \\ & = \sigma ([ 0 ]) = [ 0 ] \\ & \therefore Y = [ 0 ] \end{array}
$$

Corollary 1 A single-layer neural network with any dimensional input and output that has an odd amount of nodes can be initialized with randomly generated weights and biases such that the output is always a vector of zeros for any and all possible inputs if the activation function is defined as zero at an input of zero.

Proof. Consider a single-layer neural network with input $X = [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] \in$ $\mathbb { R } ^ { n }$ and output $Y ~ \in ~ \mathbb { R } ^ { 1 }$ . It has an activation function defined such that $\sigma ( 0 ) = 0$ . Let there be $2 h + 1$ amount of nodes in the network. Split the nodes into three groups, two evenly sized groups of size h and a single node. Using some random weight and bias assigning technique, generate weights and biases of the first group of h nodes and define them as $W _ { h }$ and $B _ { h }$ respectively. Assign the weights and bias of the second group of h nodes as $- W _ { h }$ and $- B _ { h }$ . Assign the single node to have a weight of $W _ { s i n g l e } = [ 0 ]$ and a bias of $B _ { s i n g l e } = 0$ . Solving for Y gives:

$$
\begin{array}{r l} Y & = \sigma (W X + B) \\ & = \sigma (W _ {h} X + B _ {h} - W _ {h} X - B _ {h} + [ 0 ] X + 0) \\ & = \sigma ([ 0 ]) = [ 0 ] \\ & \therefore Y = [ 0 ] \end{array}
$$

This symmetric weighting may give the impression that the network is not generally expressible. However, the symmetry is broken after the first parameter update assuming that a gradient-based update rule is used and that the activation function has a non-zero gradient at zero, $\frac { \partial \sigma } { \partial x } | _ { x = 0 } \neq 0$ . For example, assume that the network underestimates the true target and that all inputs to the network are positive. All weights and biases will increase. Namely, the positive parameters will become more positive and the negative parameters will take a step towards the positive direction. This breaks the symmetry since parameters that had the same magnitude, but opposite signs before the update will have diferent magnitudes and could have diferent signs. In this way, the NN would retain the potential expressibility of any other network of the same size and architecture.

## 3.2.2. Two-layer neural networks

The problem is to initialize a two-layer neural network of arbitrary size such that it will provide a solution of zero for all inputs and still be trainable using modern parameter updating techniques.

Theorem 2 A two-layer neural network can be initialized to yield an output vector of all zeros for any and all inputs if the first layer is developed using the technique presented in Theorem 1, the second layer uses an activation function that is defined as zero at an input of zero, and the biases of the second layer are initialized as zero.

Proof. Consider a two-layer neural network with input $X = [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] \in$ $\mathbb { R } ^ { n }$ and output $Y ^ { [ 2 ] } \ \in \ \mathbb { R } ^ { 1 }$ The first layer is initialized by following the methodology presented in Theorem 1. The activation function of the second layer is defined such that $\sigma ^ { [ 2 ] } ( 0 ) = 0$ . Let the biases in the second layer be initialized as zero, $B ^ { [ 2 ] } = 0$ . The weights of the second layer can be defined using any parameter generating technique. From Theorem 1, it is known that the output of the first layer is a vector of zeros, $Y ^ { [ 1 ] } = [ 0 ]$ , where the dimension is defined by the amount of nodes in layer two. Solving for $Y ^ { [ 2 ] }$ gives:

$$
\begin{array}{r l} & Y ^ {[ 2 ]} = \sigma^ {[ 2 ]} (W ^ {[ 2 ]} Y ^ {[ 1 ]} + B ^ {[ 2 ]}) \\ & \qquad = \sigma^ {[ 2 ]} (W ^ {[ 2 ]} [ 0 ] + [ 0 ]) \\ & \qquad = \sigma^ {[ 2 ]} ([ 0 ]) = [ 0 ] \\ & \therefore Y ^ {[ 2 ]} = [ 0 ] \end{array}
$$

Assuming that a gradient-based parameter update rule is used, the network will be as expressive as any other network of the same size and architecture. To examine how the expressibility is maintained, consider one gradient-based update. In the first update all weights in the second layer will be unchanged since the input to the second layer is a vector of all zeros and thus there is no gradient. The biases in this layer however may still be updated. Theorem 1 has discussed how the weights of the first layer will break symmetry. This will give a non-zero input to the second layer after one update. During the second update the weights of the second layer may have non-zero gradients since the first layer is now passing non-zero information. After the first update, the network weights can be fully updated. Therefore, the expressibility is not compromised.

## 3.2.3. Applying these methods

These proofs provide a mathematical basis for adding nonlinear expressibilty to YANNs for generalizing to complex functions without sacrificing the known initialization. The proofs use arbitrary parameter generating techniques to provide a general strategy. However, in practice it is best to generate these parameters with small magnitudes (e.g., a maximum of 0.01). This is to avoid any large changes when going through the training updates so that the network can provide similar outputs to the YANN. The proofs have constraints on the activation functions being used in each layer. Nevertheless, these are mild constraints. The widely used hyperbolic tangent function (tanh) meets all the necessary conditions for the first layer. Both tanh and rectified linear unit (ReLU) meet the necessary conditions for any layer other than the first layer, since the nonzero gradient at zero is no longer strictly necessary past layer one. Furthermore, the approach used in Theorem 2 may be applicable to networks of larger depths. As long as the activation function of each layer is defined as zero at zero inputs and the first layer is created according to Theorem 1, the overall network will always provide a zero output initially but still be trainable since all network parameter can be fully updated at the second update step.

## 4. YANN-based RL

## 4.1. YANN-actor

The YANN-based actor network is simple to establish since the original YANN formulation can represent known optimal control laws. To develop a YANN-initialized policy network, an mp-MPC problem (Eq. 9) needs to be solved ofline in order to find a piecewise afine explicit control law. A simplified linear system model can be used, e.g. approximated from process data. After, the YANN can be created following the steps in our previous work [49]. More nodes and/or layers can be added within the YANN according to Section 3.2, so that it can be trained to represent more complex control functions by directly interacting with the system online.

In this work, we generate the nonlinear expressibility of the YANN-actor by injecting two-layer neural networks initialized according to Theorem 2 in parallel with the computation of the linear control laws. One advantageous feature of the YANN is being able to locate a subset of nodes within the neural network that govern the control law for a certain partitioning of the state space (i.e., the critical region). To maintain this feature in YANN-actor, another layer must be added which acts as a suppression layer. This is better visualized in the graphic of the YANN-actor in Fig. 4 where $q$ is the total amount of inequalites governing the polytopic subdomains in their halfspace representation, $p$ is the number of polytopic subdomains, $m$ is the number of manipulated variables, the input X is the set of parametric variables but can be extended to other pieces of information if desired for the trainable NN component, $h ^ { [ i ] }$ is the number of nodes in the $i ^ { t h }$ layer of the trainable NN component, and $X ^ { [ 4 ] }$ is the concatenated vector of the outputs between the subdomain identification, linear control law, and trainable NN components which is used as the input to the suppression and addition layers.

![](images/aae7e948ea7fae695d6c4777d8888ecf98b9f95e7c73922a81ca639d8e0d6ecc.jpg)  
Figure 4: YANN-actor network architecture.

This structure can be interpreted in the following way. First, three vectors are computed simultaneously: (i) a vector of binaries relating to the solutions of indicator functions for the subdomains of the piecewise control law (blue box in Fig. 4), (ii) each and every piece of the control law (green box in Fig. 4), (iii) the output a two-layer NN for each piece of the control law (yellow box in Fig. 4). This information is concatenated and used to determine the solution for the active subdomain by suppressing inactive subfunction solutions to zero using ReLU bounding and then adding up the remaining values (red box in Fig. 4). This approach is adopted so that no two individual control laws interfere with each other when updating. An alternative way to conceptualize the YANN-actor is to envision that, for each polytopic subdomain that has been identified by the mp-MPC problem, the new control law is defined by a linear function plus a trainable NN. This allows each individual linear control to be uniquely updated with the updates being mutually exclusive from other control laws and also provides a degree of interpretability when examining the YANN-actor. From a mathematical standpoint, this makes sense and can allow for smaller networks to be used as opposed to one large NN that would have to represent the highly complex relationship of the true optimal control law across the full state space. Instead, each partition of the state space as identified by active constraints in the MPC problem feature their own unique control law which is consistent with optimization theory.

## 4.2. YANN-critic

The critic in an actor critic method serves as a value function, either state or state-action, which comes from optimal control theory as discussed in Section 2. To achieve the overarching goal of deploying RL with confidence, it is necessary to derive the explicit form of these functions for linear systems. The state value function for linear unconstrained systems has been derived previously when introducing LQR as it is the solution to the DARE (Eq. 7). However, the state-action value function is less commonly used in classic control theory. Given this, we derive it below for the LQR problem (Eq. 5). The final solution has also been previously reported by Bradtke [56]. Let C be the cost function, $C ( s _ { t } , u _ { t } ) = s _ { t } ^ { T } Q _ { w } s _ { t } + u _ { t } ^ { T } R u _ { t }$ , where $Q _ { w }$ and R are symmetric positive semi-definite and positive definite weighting matrices. Evaluating the state-action value function gives:

$$
Q ^ {\star} (s _ {t}, u _ {t}) = C (s _ {t}, u _ {t}) + \min _ {u _ {t + 1}} \gamma Q ^ {\star} (s _ {t + 1}, u _ {t + 1})
$$

$$
\min _ {u _ {t + 1}} \gamma Q ^ {\star} (s _ {t + 1}, u _ {t + 1}) = \gamma V ^ {\star} (s _ {t + 1}) = s _ {t + 1} ^ {T} P s _ {t + 1}
$$

$$
\begin{array}{r} Q ^ {\star} (s _ {t}, u _ {t}) = s _ {t} ^ {T} Q _ {w} s _ {t} + u _ {t} ^ {T} R u _ {t} + \gamma s _ {t + 1} ^ {T} P s _ {t + 1} \\ s _ {t + 1} = A s _ {t} + B u _ {t} \end{array}
$$

$$
\begin{array}{r l} & Q ^ {\star} (s _ {t}, u _ {t}) = s _ {t} ^ {T} Q _ {w} s _ {t} + u _ {t} ^ {T} R u _ {t} + \gamma (A s _ {t} + B u _ {t}) ^ {T} P (A s _ {t} + B u _ {t}) \\ & \qquad = s _ {t} ^ {T} (Q _ {w} + \gamma A ^ {T} P A) s _ {t} + 2 s _ {t} ^ {T} (\gamma A ^ {T} P B) u _ {t} + u _ {t} ^ {T} (R + \gamma B ^ {T} P B) u _ {t} \end{array}
$$

There are two interesting things to note about this equation. First, the DARE can be similarly derived as Section 2 by noting that $Q ^ { \star } ( s _ { t } , u _ { t } ^ { \star } ) ~ =$ $V ^ { \star } ( s _ { t } )$ if $\boldsymbol { u } _ { t } ^ { \star }$ is the optimal action. Second, this is equivalent to the objective function of the corresponding mp-MPC problem (Eq. 9) for a one-step operating horizon. This implies that for the decades of research involving mp-MPC, theorists and engineers have been using an unraveled form of the state-action value function. Furthermore, this suggests that when shifting MPC (Eq. 8) into multi-parametric form, the objective function transforms fully into a state-action value function. To the best of our knowledge, we are the first to bring this into consideration.

A schematic showing the structure of the YANN-critic is provided in Fig. 5. To achieve an exact representation of this function via an NN, the resulting coeficient matrices $( ( Q _ { w } + \gamma A ^ { T } P A ) , 2 ( \gamma A ^ { T } P B ) , ( R + \gamma B ^ { T } P B ) )$ are embedded into the weights of a linear layer using zero bias. Following this, a matrix multiplication layer is used to complete the computation of the squared and bilinear terms. A third linear layer with an all-one weight matrix and zero bias sums all of these values to yield the exact Q-value for the linear system. To add the nonlinear expressibility, a two-layer NN initialized according to Theorem 2 is added in parallel to the previously discussed layers. The YANN-critic adds both the linear system Q-value and the output of this auxiliary NN to provide a Q-value estimate that better represents the true state-value function. The advantage to building the network in this way is that the parameters of the embedded NN and the parameters defining the linear system state-action value function can be updated simultaneously. The nomenclature of YANN-critic is meant to be consistent with that of the YANN-actor, but it is important to note that they are based on diferent formulations and come from diferent principles of control theory.

![](images/8390662a225d8e66afefd4a498fe983c81293e75a8419141599d25580bf26581.jpg)  
Figure 5: YANN-critic network architecture.

## 4.3. YANN-based DDPG

In this section we present and discuss a YANN-RL algorithm based on the actor-critic algorithm deep deterministic policy gradient (DDPG). In this work, DDPG is chosen as an RL algorithm that has been widely adopted in various applications and has been used towards the control of chemical and energy systems [57, 58, 59, 60]. Additionally, it is a deterministic policy algorithm which is necessary when using a YANN-actor. The YANN-DDPG methodology is presented in Algorithm 1 which makes a few notable modifications to the original DDPG algorithm presented in Lillicrap et al. [61].

The most critical change is that the weights and biases defining the actor and the critic are not fully randomly generated. They are initialized using specific tools from control theory and with the approaches presented in Section 3. Another key adjustment is the complete and total lack of random exploration. It is a necessary component of the DDPG algorithm to add exploration noise to the computed action (e.g., Ornstein-Uhlenbeck or Gaussian) in order to search through the state space and develop a basis for the exploitation phase. The exploration is not needed in this YANN-DDPG algorithm since the policy starts with a good basis across the entire state space, and thus we can begin the algorithm already exploiting the critic. Skipping the exploration phase also allows a much safer and smoother operation since random perturbations on the system are avoided. This proposed algorithm begins with the full confidence of linear optimal control which is a significant departure from other RL algorithms.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 YANN-DDPG
1: Obtain a linearized model for the environment (e.g., Jacobian linearization, system identification, etc.)
2: Use the linear model to create an MPC problem (e.g., Eq. 8)
3: Reformulate the MPC problem into an mp-MPC problem and solve to obtain the piecewise-affine control law
4: Use the piecewise affine control law to initialize a YANN-actor, $\pi_{\theta}(s_t)$
5: Use the system model and MPC problem weighting matrices to initialize a YANN-critic, $Q_{\phi}(s_t, u_t)$
6: Initialize target networks $\pi_{\theta_{target}}(s_t)$ and $Q_{\phi_{target}}(s_t, u_t)$ with $\theta_{target} \leftarrow \theta$, $\phi_{target} \leftarrow \phi$
7: Initialize replay buffer, $\mathcal{R}$
8: for episode = 1, M do
9: Observe state $s_t$ and compute action, $u_t = \pi_{\theta}(s_t)$
10: Execute action $u_t$ and observe cost, $C_t$, and new state, $s_{t+1}$
11: Store transition ($s_t, u_t, C_t, s_{t+1}$) in $\mathcal{R}$
12: Sample a random batch of $\mathcal{N}$ transitions ($s_i, u_i, C_i, s_{i+1}$) from $\mathcal{R}$
13: Compute targets as:
$y_i = C_i + \gamma Q_{\phi_{target}}(s_{i+1}, \pi_{\theta_{target}}(s_{i+1}))$
14: Update the YANN-critic by minimizing the loss:
$\mathcal{L} = \frac{1}{\mathcal{N}} \sum_i^{\mathcal{N}} (y - Q_{\phi}(s_i, u_i))^2$
15: Update the YANN-actor using the sampled policy gradient:
$\nabla_{\theta} J = \frac{1}{\mathcal{N}} \sum_i^{\mathcal{N}} Q_{\phi}(s_i, \pi_{\theta}(s_i))$
16: Update target networks using Polyak averaging:
$\theta_{target} = \tau_{actor} \theta + (1 - \tau_{actor}) \theta_{target}$ $\phi_{target} = \tau_{critic} \phi + (1 - \tau_{critic}) \phi_{target}$
17: end for
</div>

## 4.4. Monotonic policy improvements

The seminal results of Khakade and Langford [62] give critical insight into how policies should be updated to achieve a continuous improvement in performance. The central idea is that there exists some regions around the current policy such that updating the policy to remain within this region will lead to a guaranteed improvement. This idea has been fundamental in several works such as trust region policy optimization (TRPO) [63], safe policy iteration [64], and several others [65, 66, 67, 68]. While these theoretical guarantees are highly desired, they are often too dificult to achieve in practice. The common issues among these approaches is that they either need assumptions that are challenging to verify or they are computationally intractable. For example, TRPO, which is one of the more tractable approaches of those with guarantees, requires a second-order optimization algorithm to approximate the Hessian of KL divergence constraints. This is often far too expensive to be applicable, especially for real-time systems. Another issue is that most theoretical approaches require a large amount of sampling in order to provide any confidence at all. For these reasons, most modern and practical RL algorithms implement a heuristic approach to providing safe and continuously improving policy updates. A good example is proximal policy optimization (PPO) [69] which is a direct successor of TRPO. In PPO, a small epsilon value is used to clip how much the policy updates, efectively implementing a TRPO-style algorithm without having to track through the expensive computations. One of the goals of the YANN-DDPG algorithm is to use these ideas in order to have heuristic confidence in continuously improving the policy. This is done in two ways: (i) small learning rates and small target network updates, and (ii) interpretable initializations of the actor and the critic. The small updates made to the networks in this algorithm ensure that they remain close to their representations prior to being updated. In other words, none of the networks are being updated drastically enough to push them outside of a known area of improvement, which is similar to the clipping in PPO. The interpretable initializations of both the actor and the critic provide a foundations for which this continuous improvement can occur. When either the critic or the actor is updated, they are expected to improve when making a small enough update and thus should improve over their initial representation. In this way, there is confidence that on average the control policy is never getting worse the than the previous control policy. This is not a full-proof guarantee as the algorithm still utilizes the structure of DDPG which holds very little theoretical confidence compared to that of

TRPO or PPO. Nevertheless, this is an exceptional result since linear optimal control is efectively a lower bound of performance for YANN-DDPG. This statement is empirically validated in two case studies in Section 5.

## 5. Case studies

## 5.1. Clipped pendulum

In the first case study, we consider a simple pendulum about a fulcrum point which is a widely used benchmark system for RL-based control. It is desired to move the pendulum to an upright position by applying torque on it. For simplicity we consider a constrained initial position in order to avoid the swing-up problem which is a classically dificult problem in control. The motivation of this case study is to highlight the potential benefits of YANN-DDPG in comparison to DDPG. The governing dynamics of the pendulum system are represented in Eq. 11. The system states are: (i) the angle, θ, measured from an upright position, and (ii) the angular velocity, $\frac { d \theta } { d t }$

$$
{\frac {d ^ {2} \theta}{d t ^ {2}}} = {\frac {3 g}{2 l}} \sin (\theta) + {\frac {3}{m l ^ {2}}} u\tag{11}
$$

where u is the applied torque in $N \cdot m , g = 1 0 \textstyle { \frac { m } { s ^ { 2 } } }$ is the gravitational constant, l = 1 m is the length of the pendulum, and $m = 1$ kg is the mass of the pendulum. Towards the direction of developing a YANN-actor, these equations are linearized around the origin (pendulum in an upright and unmoving state) using the Jacobian method and then discretized for a sampling rate of 0.05s which is consistent with the OpenAI Gym simulator [70]. In practice, for true model-free RL, system identification would be used instead. The resulting linear system is given in Eq. 12.

$$
A = \left[ \begin{array}{c c} 1. 0 1 8 8 & 0. 0 5 0 3 \\ 0. 7 5 4 7 & 1. 0 1 8 8 \end{array} \right], B = \left[ \begin{array}{c} 0. 0 0 3 8 \\ 0. 1 5 0 9 \end{array} \right]\tag{12}
$$

This linear system is used to create an MPC problem similar to the one shown in Eq. 8 with N = 2, Q is the two-dimensional identity matrix, $R = 0 . 0 0 1$ ， the state path and terminal constraints are $\begin{array} { r } { - 2 \leq \theta _ { t } \leq 2 , - 8 \leq \frac { d \theta _ { t } } { d t } \leq 8 } \end{array}$ , the control input constraint is $- 2 \leq u _ { t } \leq 2$ , and A and B are from Eq. 12. The constraints on the control input and on the states are consistent with the constraints of the pendulum system.

The MPC problem is reformulated into an mp-QP (Eq. 9) and solved via multi-parametric programming to give an explicit control law as a function of the states at time t = 0. The piecewise afine control law has seven subdomains partitioned over the constrained state space. This function is used to formulate a YANN-actor. The YANN-actor has an additional twolayer NN with 16 nodes in each layer using the tanh activation function for every subdomain giving a total parameter count of 15, 188 with 12, 880 of those being trainable. This is less than the DDPG actor which features a two-layer NN using 256 nodes in each layer giving 67, 073 total parameters, all of which are trainable. Furthermore the YANN-actor can be expected to evaluate faster given its sparsity over the DDPG actor which is fully dense. The YANN-critic uses the weighting and system matrices in the MPC problem along with two additional NN layers each of 64 nodes using the tanh activation function for a total of 4, 358 parameters. The DDPG critic uses the same exact structure of the DDPG actor. Both algorithms are initially tested on a set of the same random 10 episodes to benchmark their initial performance. Then each algorithm, YANN-DDPG and DDPG, is allowed to run for 50 episodes of training. After these training episodes, both algorithms are again benchmarked on the same 10 testing episodes. These results are shown in Table 1 and highlight the improvement each algorithm made.

Table 1: Total cost per test episode for the pendulum environment.

<table><tr><td>Episode #</td><td colspan="2">DDPG</td><td colspan="3">YANN-DDPG</td></tr><tr><td></td><td>Initial</td><td>Final</td><td>Initial</td><td>Final</td><td>Change</td></tr><tr><td>1</td><td>1,570.25</td><td>3.83</td><td>1.50</td><td>1.51</td><td>+0.01</td></tr><tr><td>2</td><td>1,586.54</td><td>2.89</td><td>0.55</td><td>0.55</td><td>0.00</td></tr><tr><td>3</td><td>1,597.88</td><td>4.33</td><td>1.93</td><td>1.93</td><td>0.00</td></tr><tr><td>4</td><td>1,585.44</td><td>2.61</td><td>0.15</td><td>0.15</td><td>0.00</td></tr><tr><td>5</td><td>1,597.85</td><td>3.58</td><td>1.15</td><td>1.15</td><td>0.00</td></tr><tr><td>6</td><td>1,552.79</td><td>2.75</td><td>0.35</td><td>0.35</td><td>0.00</td></tr><tr><td>7</td><td>1,585.54</td><td>2.91</td><td>0.57</td><td>0.57</td><td>0.00</td></tr><tr><td>8</td><td>1,603.19</td><td>129.03</td><td>1,354.73</td><td>127.00</td><td>-1,227.73</td></tr><tr><td>9</td><td>1,604.24</td><td>128.99</td><td>1,348.27</td><td>126.84</td><td>-1,221.43</td></tr><tr><td>10</td><td>1,585.99</td><td>3.25</td><td>0.74</td><td>0.74</td><td>+0.00</td></tr><tr><td>Average</td><td>1,586.97</td><td>28.42</td><td>270.99</td><td>26.08</td><td>-244.91</td></tr></table>

The DDPG algorithm starts of much worse than the YANN-based algorithm. The average initial test cost for DDPG is almost 1, 600 whereas the YANN-actor has an initial average cost of 271. This result highlights how the YANN-based approach can provide a significantly better starting point for RL and demonstrates the potential eficiency boosts. At some episode within the 50 training episodes, the DDPG agent performance would be similar to that of the initial YANN-actor which represents that the YANN-DDPG approach could save training time over the DDPG algorithm. Another interesting result is that the initial YANN-actor outperformed the trained DDPG agent on 8 out of the 10 test episodes while the final YANN-actor outperformed the trained DDPG agent on all test episodes. This is evidence that the YANN-based algorithm is much more eficient in the learning process since it is able to consistently outperform DDPG when both are trained for the same amount of episodes. The results confirm the expectations of continuous improvements in the policy when using small updates. The YANN-based algorithm is improved dramatically, e.g. for test episode 8 where cost drops by over 1, 200. Test episode 1 is the only episode where the YANN-actor observed worse performance after training, while cost increases marginally by 0.01. This shows that the continuous improvement is more heuristic confidence than it is a true guarantee. Nonetheless, having linear optimal control as an efective lower bound is a promising result that has been lacking in RL for control.

## 5.2. Safety-critical reactor

In the second case study, we consider a safety-critical reaction process conceptualized from a real-world process incident [71, 72]. It is a continuous stirred tank reactor (CSTR) that is represented by the equations given in Eq. 13 where modeling and parameter information is provided in the Appendix in Table A.1. The main reactions are also given below.

$$
\frac {d C _ {A}}{d t} = \frac {F _ {A , i n} - q _ {o u t}}{V} C _ {A} - k _ {1} C _ {A} C _ {B}\tag{13a}
$$

$$
\frac {d C _ {B}}{d t} = \frac {F _ {B , i n} - q _ {o u t}}{V} C _ {A} - k _ {1} C _ {A} C _ {B}\tag{13b}
$$

$$
\frac {d C _ {S}}{d t} = \frac {F _ {C , i n} - q _ {o u t}}{V} C _ {A} - k _ {2} C _ {S}\tag{13c}
$$

$$
\frac {d T}{d t} = \frac {q _ {o u t}}{V} (T _ {i n} - T) + \frac {\sum (- \Delta H _ {k} r _ {k}) - \frac {U A _ {x}}{V} (T - T _ {c})}{\rho C _ {P}}\tag{13d}
$$

Reaction 1:

$$
\begin{array}{c} \text {Methylcyclopentadiene (A) + Sodium (B)} \\ \xrightarrow {\text {Diglyme(S)}} \text {Sodium Methylcyclopentadiene (C) + Hydrogen (D)} \end{array}
$$

Reaction 2:

$$
\mathrm{Diglyme(S)} \xrightarrow {\text {Sodium(B)}} \mathrm{Hydrogen(D)} + \mathrm{Byproduct}
$$

Hydrogen gas is a product of both reactions which causes major safety concerns. Additionally, the second reaction is highly exothermic which leads to thermal runaway at high temperatures. For these reasons, it is essential to operate this reactor below temperatures of 480 K which is the temperature at which the rate of reaction 2 becomes significant. This defines the safety limits for the operation of the CSTR.

For simplicity, in this control study we consider only the regulation of this safety-critical CSTR to steady state while adhering to the safety constraint of keeping the temperature below 480 K. The dynamic equations in Eq. 13 are linearized around the steady state using the Jacobian method and then discretized to a sampling rate of 1/60 s. The resulting linearized system is given in Eq. 14.

$$
A = \left[ \begin{array}{c c c c} 0. 9 5 0 6 & 0 & 0 & 0 \\ - 0. 0 4 8 4 & 0. 9 9 4 3 & 0 & 0 \\ 0 & 0 & 0. 9 9 0 9 & 0 \\ 0. 6 9 7 0 & 0. 0 6 7 8 & 0 & 1. 0 0 3 0 \end{array} \right] B = \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ - 0. 0 0 0 7 \end{array} \right]\tag{14}
$$

This linear system is used to create an MPC problem similar to the one shown in Eq. 8 with $N = 2 , Q = 0 . 1 \times I _ { 2 } , R = 0 . 0 0 0 1$ , the state path and terminal constraints in deviation variables are $- 2 \leq \overline { { C _ { i } } } \leq 2$ for each species $i , - 7 0 \leq \overline { { T } } \leq 5$ , the control input constraint is $- 5 5 \leq u _ { t } \leq 5 5$ , and A and B are from Eq. 14. The upper bound on the temperature deviation corresponds to the safety-critical constraint for this system.

Similar to the pendulum case study, the MPC problem is reformulated into an mp-QP (Eq. 9) and solved via multi-parametric programming to give an explicit control law as a function of the states at time $t = 0$ . The piecewise afine control law again has seven subdomains partitioned over the constrained state space. This function is used to initialize a YANN-actor. The YANN-actor has an additional two-layer NN with 8 nodes in each layer using the tanh activation function for every subdomain giving a total parameter count of 5, 320 with 3, 416 of those being trainable. This is less than the DDPG actor which features a two-layer NN using 256 nodes in each layer giving 67, 329 total parameters, all of which are trainable. The YANN-critic uses the weighting and system matrices in the MPC problem along with two additional NN layers each of 64 nodes using the tanh activation function for a total of 4, 545 parameters. The DDPG critic uses the same exact structure of the DDPG actor. Any departure in the amount of parameters needed between this case study and the previous one for the NNs in DDPG or for the YANN-critic can be explained by needing more inputs (i.e., more system states). The YANN-actor has significantly less complexity in this example to showcase its ability to still perform well with limited parameters. Both algorithms are initially tested on a set of the same random 10 episodes to benchmark their initial performance. Then each algorithm is allowed to run for 50 episodes of training. The training episode stops if the safety-critical constraint is violated (T > 480 K) and a penalty of 1E5 is added to the immediate cost to encourage behavior that avoids violating safety limits. The amount of time the algorithm violates this safety condition is noted during the training process. After the training episodes, both algorithms are again benchmarked on the same 10 testing episodes. For this system the stochasticity of the DDPG algorithm can play a large role in its performance. For this reason, we have included two scenarios using the same DDPG algorithm in the results, one with a good initial performance (DDPG-G) and one with a poor initial performance (DDPG-P). These results are shown in Table 2.

The DDPG algorithm performs much worse for the CSTR system than it did for the simple pendulum example, regardless of a good random initialization (DDPG-G) with an average test cost of around 800 or a poor initialization (DDPG-P) where the agent violated the safety constraint on three of the ten test episodes. The CSTR has more states and more restrictions on the space that can be explored due to the safety constraint which may contribute to this behavior, since the DDPG algorithm relies on exploring the state space before exploiting its structure. This may be better explained by the DDPG-P trial where violating the safety constraint numerous times leads to a better final performance. Nevertheless, in both trials the algorithm violated the safety constraint at least one time (Table 2). This shows that even with an initially decent actor the algorithm may still give undesirable behavior for the real-world system.

The YANN-DDPG algorithm performs immensely better than the simple DDPG approaches. The YANN-based RL method provides an initial performance that is around an order of magnitude better than the trained DDPG-agents. This result shows that the YANN-DDPG algorithm save more than 50 episodes worth of training time over DDPG since it cannot achieve the same performance within the training episodes. Furthermore, the training process in the YANN-DDPG trial never violated the safety constraint a single time. The heuristic confidence in expecting the YANN-agent to at worst perform as well as linear optimal control can also be seen in Table 2. The YANN-based algorithm performs worse than its initialization on four of the ten episodes but only by a small margin (maximum of 0.77) whereas it improves in the remaining six episodes up to two orders of magnitude more (maximum of 14.15).

Table 2: Total cost per test episode for the CSTR environment.

<table><tr><td></td><td colspan="2">DDPG-G</td><td colspan="2">DDPG-P</td><td colspan="3">YANN-DDPG</td></tr><tr><td>Safety Violations</td><td colspan="2">1</td><td colspan="2">9</td><td colspan="3">0</td></tr><tr><td>Episode</td><td>Initial</td><td>Final</td><td>Initial</td><td>Final</td><td>Initial</td><td>Final</td><td>Change</td></tr><tr><td>1</td><td>901.95</td><td>662.14</td><td>110.99</td><td>87.04</td><td>30.67</td><td>30.84</td><td>+0.17</td></tr><tr><td>2</td><td>173.15</td><td>147.89</td><td>1,055.08</td><td>634.00</td><td>58.93</td><td>59.70</td><td>+0.77</td></tr><tr><td>3</td><td>164.74</td><td>131.13</td><td>976.04</td><td>572.71</td><td>54.77</td><td>55.40</td><td>+0.63</td></tr><tr><td>4</td><td>1,671.20</td><td>1,357.62</td><td>306.6</td><td>417.88</td><td>115.32</td><td>113.09</td><td>-2.23</td></tr><tr><td>5</td><td>193.86</td><td>184.65</td><td>&gt;1E5</td><td>749.03</td><td>75.77</td><td>76.34</td><td>+0.57</td></tr><tr><td>6</td><td>2,813.91</td><td>2,422.08</td><td>842.77</td><td>1,097.05</td><td>297.99</td><td>283.84</td><td>-14.15</td></tr><tr><td>7</td><td>517.99</td><td>334.26</td><td>154.29</td><td>31.11</td><td>10.76</td><td>10.39</td><td>-0.37</td></tr><tr><td>8</td><td>905.87</td><td>667.46</td><td>121.50</td><td>96.51</td><td>36.57</td><td>35.86</td><td>-0.71</td></tr><tr><td>9</td><td>277.34</td><td>301.08</td><td>&gt;1E5</td><td>1,032.82</td><td>110.96</td><td>109.37</td><td>-1.59</td></tr><tr><td>10</td><td>382.38</td><td>435.59</td><td>&gt;1E5</td><td>1,319.69</td><td>151.44</td><td>147.31</td><td>-4.13</td></tr><tr><td>Avg.</td><td>800.24</td><td>664.39</td><td>&gt;3E4</td><td>603.78</td><td>94.32</td><td>92.22</td><td>-2.10</td></tr></table>

## 6. Conclusions

In this work we have introduced a novel RL algorithm based on the use of YANNs, which are interpretable neural networks with desirable properties for control-theoretic applications. This algorithm represents a paradigm shift in our ability to confidently deploy RL algorithms for the optimal control of chemical and energy systems. We showed how to initialize YANN-based actors and critics to exactly represent the solution and the objective to linear optimal control problems. We discussed how one can have confidence that the control policy should only improve over its linear approximation. We also showed how to initialize neural networks to always output 0 but still be trainable. We highlighted these results in two case studies where one featured the operation of a safety-critical system where the DDPG algorithm violated safety constraints numerous times while the YANN-DDPG algorithm never once did. In the future we wish to provide more rigorous theory and algorithms that can truly guarantee continuous improvement over the linear control approximation. We will also look to integrate YANNs into other wellestablished RL algorithms and to develop a new algorithm altogether that exploits additional properties of YANNs.

## Acknowledgments

The authors acknowledge financial support from NSF RETRO Project CBET-2312457, NSF GRFP 2024370240, and Department of Chemical and Biomedical Engineering at West Virginia University.

## Appendix A. Safety-critical CSTR modeling information

Table A.1: Defining CSTR modeling parameters and values.

<table><tr><td>State variables</td><td> $C_A, C_B, C_S$ : ConcentrationsT: Temperature</td></tr><tr><td>Manipulated variable</td><td>U: Heat transfer coefficient</td></tr><tr><td>Control variable</td><td>T: Temperature</td></tr><tr><td>Modeling variables</td><td>V: Volume (4000 L) $\rho$ : Mixture density (36 mol/L) $C_p$ : Specific heat (430.91 J/mol·K) $A_x$ : Heat transfer area (5.3 m2) $T_c$ : Coolant temperature(373K) $\Delta H_k$ : (-45.6 kJ/mol, -320 kJ/mol) $k_i = A_i \exp(-\frac{E_i}{RT})$  $A_i:(A_1 = 4 \times 10^{14}, A_2 = 1 \times 10^{84})$  $E_i:(E_1 = 1.28 \times 10^5, E_2 = 8 \times 10^5 \text{ J/mol} \cdot \text{K})$ </td></tr></table>

## References

[1] O. Dogru, J. Xie, O. Prakash, R. Chiplunkar, J. Soesanto, H. Chen, K. Velswamy, F. Ibrahim, B. Huang, Reinforcement Learning in Process Industries: Review and Perspective, IEEE/CAA Journal of Automatica Sinica 11 (2) (2024) 283–300.

[2] J. Shin, T. A. Badgwell, K.-H. Liu, J. H. Lee, Reinforcement Learning – Overview of recent progress and implications for process control, Computers & Chemical Engineering 127 (2019) 282–294.

[3] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, D. Hassabis, Human-level control through deep reinforcement learning, Nature 518 (7540) (2015) 529–533.

[4] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap, F. Hui, L. Sifre, G. van den Driessche, T. Graepel, D. Hassabis, Mastering the game of Go without human knowledge, Nature 550 (7676) (2017) 354–359.

[5] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, D. Hassabis, Mastering the game of Go with deep neural networks and tree search, Nature 529 (7587) (2016) 484–489.

[6] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. M¨uller, V. Koltun, D. Scaramuzza, Champion-level drone racing using deep reinforcement learning, Nature 620 (7976) (2023) 982–987.

[7] J. Wang, J. Zhao, A virtual entity of the digital twin based on deep reinforcement learning model for dynamic scheduling process, in: F. Manenti, G. V. Reklaitis (Eds.), Computer Aided Chemical Engineering, Vol. 53 of 34 European Symposium on Computer Aided Process Engineering / 15 International Symposium on Process Systems Engineering, Elsevier, 2024, pp. 247–252.

[8] C. D. Hubbs, C. Li, N. V. Sahinidis, I. E. Grossmann, J. M. Wassick, A deep reinforcement learning approach for chemical production scheduling, Computers & Chemical Engineering 141 (2020) 106982.

[9] A. Branif, F. You, Y. Tian, Enhanced Reinforcement Learning-driven Process Design via Quantum Machine Learning, in: The 35th European Symposium on Computer Aided Process Engineering, Ghent, Belgium, 2025, pp. 1403–1408.

[10] S. Reynoso-Donzelli, L. A. Ricardez-Sandoval, An integrated reinforcement learning framework for simultaneous generation, design, and control of chemical process flowsheets, Computers & Chemical Engineering 194 (2025) 108988.

[11] P. Petsagkourakis, I. O. Sandoval, E. Bradford, D. Zhang, E. A. del Rio-Chanona, Reinforcement learning for batch bioprocess optimization, Computers & Chemical Engineering 133 (2020) 106649.

[12] S. Spielberg, A. Tulsyan, N. P. Lawrence, P. D. Loewen, R. Bhushan Gopaluni, Toward self-driving processes: A deep reinforcement learning approach to control, AIChE Journal 65 (10) (2019) e16689.

[13] R. d. R. Faria, B. D. O. Capron, M. B. de Souza Jr., A. R. Secchi, One-Layer Real-Time Optimization Using Reinforcement Learning: A Review with Guidelines, Processes 11 (1) (2023) 123.

[14] T. Joshi, S. Makker, H. Kodamana, H. Kandath, Twin actor twin delayed deep deterministic policy gradient (TATD3) learning for batch process control, Computers & Chemical Engineering 155 (2021) 107527.

[15] O. Dogru, N. Wieczorek, K. Velswamy, F. Ibrahim, B. Huang, Online reinforcement learning for a continuous space system with experimental validation, Journal of Process Control 104 (2021) 86–100.

[16] Y. Wang, X. Zhu, Z. Wu, A tutorial review of policy iteration methods in reinforcement learning for nonlinear optimal control, Digital Chemical Engineering 15 (2025) 100231.

[17] A. Branif, S. S. Akundi, Y. Liu, B. Dantas, S. S. Niknezhad, F. Khan, E. N. Pistikopoulos, Y. Tian, Real-time process safety and systems

decision-making toward safe and smart chemical manufacturing, Digital Chemical Engineering 15 (2025) 100227.

[18] R. Nian, J. Liu, B. Huang, A review On reinforcement learning: Introduction and applications in industrial process control, Computers & Chemical Engineering 139 (2020) 106886.

[19] R. d. R. Faria, B. D. O. Capron, A. R. Secchi, M. B. de Souza, Where Reinforcement Learning Meets Process Control: Review and Guidelines, Processes 10 (11) (2022) 2311.

[20] H. Yoo, H. E. Byun, D. Han, J. H. Lee, Reinforcement learning for batch process control: Review and perspectives, Annual Reviews in Control 52 (2021) 108–119.

[21] H. Hassanpour, X. Wang, B. Corbett, P. Mhaskar, A practically implementable reinforcement learning-based process controller design, AIChE Journal 70 (1) (2024) e18245.

[22] H. Hassanpour, P. Mhaskar, B. Corbett, A practically implementable reinforcement learning control approach by leveraging ofset-free model predictive control, Computers & Chemical Engineering 181 (2024) 108511.

[23] H. Hassanpour, B. Corbett, P. Mhaskar, A practical reinforcement learning control design for nonlinear systems with input and output constraints, Computers & Chemical Engineering 201 (2025) 109248.

[24] Y. Kim, T. H. Oh, Model-based safe reinforcement learning for nonlinear systems under uncertainty with constraints tightening approach, Computers & Chemical Engineering 183 (2024) 108601.

[25] Y. Kim, J. W. Kim, Safe model-based reinforcement learning for nonlinear optimal control with state and input constraints, AIChE Journal 68 (5) (2022) e17601.

[26] Y. Kim, J. M. Lee, Model-based reinforcement learning for nonlinear optimal control with practical asymptotic stability guarantees, AIChE Journal 66 (10) (2020) e16544.

[27] F. Berkenkamp, M. Turchetta, A. Schoellig, A. Krause, Safe Modelbased Reinforcement Learning with Stability Guarantees, in: Advances in Neural Information Processing Systems, Vol. 30, Curran Associates, Inc., 2017.

[28] M. Bloor, A. Ahmed, N. Kotecha, M. Mercang¨oz, C. Tsay, E. A. del R´ıo-Chanona, Control-Informed Reinforcement Learning for Chemical Processes, Industrial & Engineering Chemistry Research 64 (9) (2025) 4966–4978.

[29] M. A. Chowdhury, S. S. S. Al-Wahaibi, Q. Lu, Entropy-maximizing TD3-based reinforcement learning for adaptive PID control of dynamical systems, Computers & Chemical Engineering 178 (2023) 108393.

[30] O. Dogru, K. Velswamy, F. Ibrahim, Y. Wu, A. S. Sundaramoorthy, B. Huang, S. Xu, M. Nixon, N. Bell, Reinforcement learning approach to autonomous PID tuning, Computers & Chemical Engineering 161 (2022) 107760.

[31] D. Beahr, D. Bhattacharyya, D. A. Allan, S. E. Zitney, Development of algorithms for augmenting and replacing conventional process control using reinforcement learning, Computers & Chemical Engineering 190 (2024) 108826.

[32] N. P. Lawrence, M. G. Forbes, P. D. Loewen, D. G. McClement, J. U. Backstr¨om, R. B. Gopaluni, Deep reinforcement learning with shallow controllers: An experimental application to PID tuning, Control Engineering Practice 121 (2022) 105046.

[33] S. Gros, M. Zanon, Data-Driven Economic NMPC Using Reinforcement Learning, IEEE Transactions on Automatic Control 65 (2) (2020) 636– 648.

[34] K. Alhazmi, F. Albalawi, S. M. Sarathy, A reinforcement learning-based economic model predictive control framework for autonomous operation of chemical reactors, Chemical Engineering Journal 428 (2022) 130993.

[35] R. Reiter, A. Ghezzi, K. Baumg¨artner, J. Hofmann, R. D. McAllister, M. Diehl, AC4MPC: Actor-Critic Reinforcement Learning for Nonlinear Model Predictive Control (Jun. 2024). arXiv:2406.03995.

[36] E. Hedrick, K. Hedrick, D. Bhattacharyya, S. E. Zitney, B. Omell, Reinforcement learning for online adaptation of model predictive controllers: Application to a selective catalytic reduction unit, Computers & Chemical Engineering 160 (2022) 107727.

[37] J. W. Kim, B. J. Park, T. H. Oh, J. M. Lee, Model-based reinforcement learning and predictive control for two-stage optimal control of fed-batch bioreactor, Computers & Chemical Engineering 154 (2021) 107465.

[38] Y.-C. Chang, S. Gao, Stabilizing Neural Control Using Self-Learned Almost Lyapunov Critics, in: 2021 IEEE International Conference on Robotics and Automation (ICRA), IEEE, Xi’an, China, 2021, pp. 1803– 1809.

[39] Y. Chow, O. Nachum, E. Duenez-Guzman, M. Ghavamzadeh, A Lyapunov-based Approach to Safe Reinforcement Learning, in: Advances in Neural Information Processing Systems, Vol. 31, Curran Associates, Inc., 2018.

[40] X. Zhu, Y. Wang, Z. Wu, Reinforcement learning for optimal control of stochastic nonlinear systems, AIChE Journal 71 (7) (2025) e18840.

[41] Y. Wang, Z. Wu, Control Lyapunov-barrier function-based safe reinforcement learning for nonlinear optimal control, AIChE Journal 70 (3) (2024) e18306.

[42] Z. Marvi, B. Kiumarsi, Reinforcement Learning With Safety and Stability Guarantees During Exploration For Linear Systems, IEEE Open Journal of Control Systems 1 (2022) 322–334.

[43] B. Thananjeyan, A. Balakrishna, S. Nair, M. Luo, K. Srinivasan, M. Hwang, J. E. Gonzalez, J. Ibarz, C. Finn, K. Goldberg, Recovery RL: Safe Reinforcement Learning With Learned Recovery Zones, IEEE Robotics and Automation Letters 6 (3) (2021) 4915–4922.

[44] M. Zanon, S. Gros, Safe Reinforcement Learning Using Robust MPC, IEEE Transactions on Automatic Control 66 (8) (2021) 3638–3652.

[45] Y. Wang, M. Xiao, Z. Wu, Safe Transfer-Reinforcement-Learning-Based Optimal Control of Nonlinear Systems, IEEE Transactions on Cybernetics 54 (12) (2024) 7272–7284.

[46] S. Bo, B. T. Agyeman, X. Yin, J. Liu, Control invariant set enhanced safe reinforcement learning: Improved sampling eficiency, guaranteed stability and robustness, Computers & Chemical Engineering 179 (2023) 108413.

[47] M. Mowbray, P. Petsagkourakis, E. A. del Rio-Chanona, D. Zhang, Safe chance constrained reinforcement learning for batch process control, Computers & Chemical Engineering 157 (2022) 107630.

[48] J. Garcıa, F. Fern´andez, A comprehensive survey on safe reinforcement learning, Journal of Machine Learning Research 16 (1) (2015) 1437–1480.

[49] A. Branif, Y. Tian, YANNs: Y-wise Afine Neural Networks for Exact and Eficient Representations of Piecewise Linear Functions (May 2025). arXiv:2505.07054.

[50] E. N. Pistikopoulos, N. A. Diangelakis, R. Oberdieck, Multi-Parametric Optimization and Control, John Wiley & Sons, Ltd, 2020.

[51] R. Sutton, A. Barto, Reinforcement Learning, Second Edition: An Introduction, Adaptive Computation and Machine Learning Series, MIT Press, 2018.

[52] S. L. Brunton, J. N. Kutz, Reinforcement Learning, Cambridge University Press, 2022, p. 419–448.

[53] V. S. Devarakonda, W. Sun, X. Tang, Y. Tian, Recent Advances in Reinforcement Learning for Chemical Process Control, Processes 13 (6) (2025) 1791.

[54] Y. Tian, I. Pappas, B. Burnak, J. Katz, E. N. Pistikopoulos, Simultaneous design & control of a reactive distillation system – A parametric optimization & control approach, Chemical Engineering Science 230 (2021) 116232.

[55] D. Kenefake, E. N. Pistikopoulos, PPOPT – Multiparametric solver for explicit MPC, in: Computer Aided Chemical Engineering, Vol. 51, Elsevier, 2022, pp. 1273–1278.

[56] S. Bradtke, Reinforcement Learning Applied to Linear Quadratic Regulation, in: Advances in Neural Information Processing Systems, Vol. 5, Morgan-Kaufmann, 1992.

[57] K. M. Patel, A practical Reinforcement Learning implementation approach for continuous process control, Computers & Chemical Engineering 174 (2023) 108232.

[58] C. Panjapornpon, P. Chinchalongporn, S. Bardeeniz, R. Makkayatorn, W. Wongpunnawat, Reinforcement Learning Control with Deep Deterministic Policy Gradient Algorithm for Multivariable pH Process, Processes 10 (12) (2022) 2514.

[59] R. Siraskar, Reinforcement learning for control of valves, Machine Learning with Applications 4 (2021) 100030.

[60] M. S. F. Bangi, J. S.-I. Kwon, Deep reinforcement learning control of hydraulic fracturing, Computers & Chemical Engineering 154 (2021) 107489.

[61] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, D. Wierstra, Continuous control with deep reinforcement learning (Jul. 2019). arXiv:1509.02971.

[62] S. Kakade, J. Langford, Approximately optimal approximate reinforcement learning, in: Proceedings of the Nineteenth International Conference on Machine Learning, ICML ’02, Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 2002, p. 267–274.

[63] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, P. Abbeel, Trust Region Policy Optimization (Apr. 2017). arXiv:1502.05477.

[64] M. Pirotta, M. Restelli, A. Pecorino, D. Calandriello, Safe Policy Iteration, in: Proceedings of the 30th International Conference on Machine Learning, PMLR, 2013, pp. 307–315.

[65] B. Scherrer, M. Geist, Local Policy Search in a Convex Space and Conservative Policy Iteration as Boosted Policy Search, in: T. Calders, F. Esposito, E. H¨ullermeier, R. Meo (Eds.), Machine Learning and Knowledge Discovery in Databases, Springer, Berlin, Heidelberg, 2014, pp. 35–50.

[66] G. Dalal, K. Dvijotham, M. Vecerik, T. Hester, C. Paduraru, Y. Tassa, Safe Exploration in Continuous Action Spaces (Jan. 2018). arXiv:1801.08757.

[67] M. Papini, D. Binaghi, G. Canonaco, M. Pirotta, M. Restelli, Stochastic Variance-Reduced Policy Gradient (Jun. 2018). arXiv:1806.05618.

[68] L. Brunke, M. Greef, A. W. Hall, Z. Yuan, S. Zhou, J. Panerati, A. P. Schoellig, Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning, Annual Review of Control, Robotics, and Autonomous Systems 5 (Volume 5, 2022) (2022) 411–444.

[69] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, Proximal Policy Optimization Algorithms (Aug. 2017). arXiv:1707.06347.

[70] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, W. Zaremba, Openai gym, arXiv preprint arXiv:1606.01540 (2016).

[71] A. Branif, Y. Tian, A hierarchical multi-parametric programming approach for dynamic risk-based model predictive quality control, Control Engineering Practice 152 (2024) 106062.

[72] M. Ali, X. Cai, F. I. Khan, E. N. Pistikopoulos, Y. Tian, Dynamic riskbased process design and operational optimization via multi-parametric programming, Digital Chemical Engineering 7 (2023) 100096.