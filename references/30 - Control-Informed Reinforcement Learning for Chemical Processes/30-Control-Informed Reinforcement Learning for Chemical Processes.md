# CONTROL-INFORMED REINFORCEMENT LEARNING FOR CHEMICAL PROCESSES

Maximilian Bloor, Akhil Ahmed, Niki Kotecha,

Mehmet Mercangöz, Calvin Tsay<sup>∗</sup>, Ehecactl Antonio Del Rio Chanona<sup>∗</sup>

Sargent Centre for Process Systems Engineering, Department of Chemical Engineering

Imperial College London

London

{max.bloor22, a.del-rio-chanona, c.tsay}@imperial.ac.uk

## ABSTRACT

This work proposes a control-informed reinforcement learning (CIRL) framework that integrates proportional-integral-derivative (PID) control components into the architecture of deep reinforcement learning (RL) policies. The proposed approach augments deep RL agents with a PID controller layer, incorporating prior knowledge from control theory into the learning process. CIRL improves performance and robustness by combining the best of both worlds: the disturbance-rejection and setpoint-tracking capabilities of PID control and the nonlinear modeling capacity of deep RL. Simulation studies conducted on a continuously stirred tank reactor system demonstrate the improved performance of CIRL compared to both conventional model-free deep RL and static PID controllers. CIRL exhibits better setpoint-tracking ability, particularly when generalizing to trajectories outside the training distribution, suggesting enhanced generalization capabilities. Furthermore, the embed ded prior control knowledge within the CIRL policy improves its robustness to unobserved system disturbances. The control-informed RL framework combines the strengths of classical control and reinforcement learning to develop sample-efficient and robust deep reinforcement learning algorithms, with potential applications in complex industrial systems.

Keywords Reinforcement learning  Process control  PID Control

## 1 Introduction

In the chemical process industry, maintaining control over complex systems is crucial for achieving reliable, efficient, and high-performance operations [1]. Traditionally, process control has relied heavily on classical feedback control techniques such as proportional-integral-derivative (PID) controllers due to their simplicity, interpretability, and wellestablished tuning methods [2]. However, these tuning methods are often largely empirical, or otherwise rely on having accurate mathematical models of the open-loop system dynamics and disturbance responses, which can be challenging to derive for complex processes involving nonlinearities, delays, constraints, and changing operating conditions [3]. As a result, PID controllers often struggle to provide adequate control performance without extensive re-tuning or gain scheduling [4]. One alternative is Model Predictive Control (MPC), a successful model-based process control strategy, with the ability to optimize control actions based on the current system states and predicted future behavior while satisfying constraints. This has led to its widespread adoption in the chemical process industry [5]. Typically, MPC operates as a supervisory layer below the real-time optimization (RTO) layer, providing control setpoints to lower-level regulatory controllers, often PID controllers, which directly manipulate process variables. This hierarchical structure combines the predictive capabilities of MPC with the rapid response of traditional feedback control. However, the performance of MPC heavily relies on the accuracy of its internal process model [6]. The ongoing digitalization of the chemical process industry has opened new avenues for enhancing MPC performance through data-driven approaches. This digital transformation has enabled the integration of advanced analytics and machine learning techniques into both MPC and regulatory control layers, especially on the modeling end. For instance, artificial neural networks [7] and Gaussian processes [8] have been employed to capture complex process dynamics that may be challenging to model using purely mechanistic approaches. Furthermore, the increased availability of process data has facilitated the development of hybrid models that combine first-principles knowledge with data-driven components [9, 10]. These advancements, coupled with improvements in computational capabilities, have expanded the applicability of MPC to more complex and uncertain processes. However, challenges remain in areas such as online computational requirements for large-scale systems and the handling of uncertainties.

Recently, reinforcement learning (RL) has emerged as a promising data-driven framework for learning control policies directly from interactions with the chemical process system [11]. Deep RL methods, which utilize deep neural networks, have demonstrated success in a variety of difficult decision-making and control problems. A key advantage of modelfree RL is that it does not require accurate system models once online, instead learning control policies from experience. While model-free RL approaches can learn control policies without requiring explicit system models, they often face challenges in sample efficiency and may not fully leverage existing domain knowledge [12].

For safety reasons, RL algorithms in chemical process control are typically trained on simulation models rather than directly on physical systems. Despite this limitation, RL offers several advantages over traditional control methods. One significant benefit is the fast online inference time. This characteristic makes RL particularly suitable for systems where online computation time is critical, as the trained policy can execute control decisions rapidly in real-time applications. RL also shows promise in handling complex, nonlinear systems and adapting to process uncertainties. This feature allows RL to potentially address challenges in dynamic chemical processes more effectively than traditional control approaches. We note there is also growing interest in algorithms for safe RL, or those which can avoid (known or unknown) constraints, e.g., in physical systems [13].

It is important to acknowledge the challenges associated with implementing RL to control complex chemical processes. The offline training of RL agents often requires a large number of samples to achieve satisfactory performance, making the training process computationally intensive and time-consuming. The quality of the trained agent is highly dependent on the fidelity of the simulation model used, which may not always capture all the nuances of real-world processes, requiring online fine-tuning. Moreover, deep RL methods often treat the control problem as a black box, failing to incorporate valuable insights from control theory. This highlights the need for approaches that can balance the model-free learning capabilities of RL with the incorporation of domain expertise and efficient exploration strategies.

## 1.1 Related works

Deep reinforcement learning (deep RL), which combines deep neural networks (DNNs) with RL, has been demonstrated in various domains, including robotics, data center operations, and playing games [14, 15, 16]. This success has brought attention to RL from the process systems and control communities. The process systems engineering community has made significant progress in adapting RL to the process industries, including in distributed systems [17], constraint handling [18, 19, 20], inventory management [18, 21], batch bioprocess and control [22, 23, 24], production scheduling [25], and energy systems [26]. Early applications of RL in process control proposed model-free RL for tracking control and optimization in fed-batch bioreactors [27, 28, 29]. More recently, Mowbray et al. [30] employed a two-stage strategy using historical process data to warm-start the RL algorithm and demonstrated this on three setpoint-tracking case studies. Machalek, Quah, and Powell [31] developed an implicit hybrid machine learning model combining physics-based equations with artificial neural networks, demonstrating its application for reinforcement learning in chemical process optimization. Zhu et al. [32] developed an RL algorithm that improves scalability by reducing the size of the action space, which was demonstrated on a plantwide control problem. However, the sample efficiency of these algorithms remains a key aspect restricting their widespread industrial adoption.

To address the limitations, prior works have explored integrating reinforcement learning with existing control structures, e.g., PID controllers [33, 34, 35]. Early approaches applied model-free RL to directly tune the gains of PID controllers [36, 37] or used model-based RL techniques such as dual heuristic dynamic programming [38]. Other approaches have investigated embedding knowledge of the dynamical system using physics-informed neural networks to act as a surrogate model of the process for offline training of the RL agent [39]. Efforts have also been made to develop interpretable control structures that maintain transparency while leveraging advanced optimization techniques [40]. Lawrence et al. [41] directly parameterized the RL policy as a PID controller instead of using a deep neural network, allowing the RL agent to improve the controller’s performance while leveraging existing PID control hardware. This work demonstrates that industry can utilize actor-critic RL algorithms without the need for additional hardware or the lack of interpretability which often accompanies the use of a deep neural network. To improve this work’s training time, McClement et al. [42] used a meta-RL approach to tune PI controllers offline. The method aimed to learn a generalized RL agent on a distribution of first-order plus time delay (FOPTD) systems, resulting in an adaptive controller that can be deployed on new systems without any additional training. However, while the meta-RL approach removes the need for explicit system identification, some knowledge of the process gain and time constant magnitudes is still required to appropriately scale the meta-RL agent’s inputs and outputs when applying it to new systems.

## 1.2 Contributions

In contrast to the above methods that focus on tuning PID gain values with a fixed control structure, we propose a control-informed reinforcement learning (CIRL) framework that integrates the PID control structure with a deep neural network into the control policy architecture of an RL agent. This allows the approach to adapt to changing operating points due to the inclusion of the deep neural network. Furthermore, it aims to leverage the strengths of both PID control and deep RL: we seek to improve sample efficiency and stability using known PID structures while gaining robustness and generalizability from RL. In summary, the key contributions of this work are as follows:

1. We introduce the CIRL framework, which augments deep RL policies with an embedded PID controller layer. This enables the agent to learn adaptive PID gain tuning while preserving the stabilizing properties and interpretability of PID control, effectively acting as an automated gain scheduler.

2. We demonstrate the CIRL framework on a nonlinear continuously stirred tank reactor (CSTR) system. The CIRL agent improves setpoint tracking performance compared to both a static PID controller and a standard model-free deep RL approach, particularly when generalizing to operating regions outside the training distribution.

3. We show that by leveraging the embedded prior knowledge from the PID structure, the CIRL agent exhibits enhanced robustness to process disturbances that are not observable during training.

The remainder of this article is organized as follows. Section 2 presents the background on PID control and reinforcement learning. Section 3 describes the proposed control-informed reinforcement learning framework in detail. Section 4 discusses the simulation and experimental results. Finally, Section 5 concludes the article and outlines future research directions.

## 2 Background

## 2.1 Reinforcement Learning

The standard RL framework (Figure 1) consists of an agent that interacts with an environment. Assuming the states are fully observable, the agent receives a vector of measured states $\pmb { x } _ { t } \in \mathcal { X } \subseteq \mathbb { R } ^ { n _ { a } }$ x , and can then take some action $\pmb { u } _ { t } \in \mathcal { U } \subseteq \mathbb { R } ^ { n _ { u } }$ , which results in the environment progressing to state ${ \pmb x } _ { t + 1 }$ . Sets and represent the state and action space, respectively. For a deterministic policy π, the agent takes actions ${ \pmb u } _ { t } = \pi ( { \pmb x } _ { t } )$ , while, for a stochastic policy, the action $\mathbf { \Delta } \mathbf { u } _ { t }$ is sampled from the policy π represented by a conditional probability distribution $\mathbf { } \mathbf { } u _ { t } \sim \pi ( \cdot \mid \mathbf { } x _ { t } )$ A common assumption in RL is that the state transition given some action is defined by a density function $\mathbf { \boldsymbol { x } } _ { t + 1 } \sim p ( \cdot \mid \mathbf { \boldsymbol { x } } _ { t } , \mathbf { \boldsymbol { u } } _ { t } )$ that represents the stochastic nonlinear dynamics of the process. The reward the agent receives is defined by the function $r _ { t } = \mathcal { R } ( \pmb { x } _ { t } , \pmb { u } _ { t } )$ With a defined control policy, the policy can be implemented over a discrete time horizon T thus producing the following trajectory $\tau = ( { \boldsymbol { x } } _ { 0 } , { \boldsymbol { u } } _ { 0 } , { \boldsymbol { r } } _ { 0 } , { \boldsymbol { x } } _ { 1 } , { \boldsymbol { u } } _ { 1 } , { \boldsymbol { r } } _ { 1 } , . . . , { \boldsymbol { x } } _ { T } , { \boldsymbol { u } } _ { T } , { \boldsymbol { r } } _ { T } )$

![](images/66bf3c7bb8aa7564ed0d215e20c89d0b337249fb1d5bc64f840cc88d08ac2154.jpg)  
Figure 1: The RL framework

Formally, notice that the above state transition assumption enables modeling the underlying system as a Markov Decision Process (MDP); for further treatment of the subject the reader is referred to Sutton and Barto [43]. In reinforcement learning, particularly in our framework, the agent’s goal is to maximize the cumulative reward (return) $J ( \pi )$ over a pre-defined (often infinite) timespan, a discount factor γ is used to reflect the uncertain future and ensure computational tractability. The policy that achieves this is the optimal policy $\pi ^ { * }$ :

![](images/e8280dd0cf300ade2e219655b2975448a09b2899cdff1cf7b6d89ab0c2bd62ed.jpg)  
Figure 2: Deep policy network π<sub>θ</sub>

$$
J (\pi) = \mathbb {E} _ {\pi} \left[ \sum_ {t = 0} ^ {T} \gamma^ {t} r _ {t} (\pmb {x} _ {t}, \pi (\pmb {x} _ {t})) \right]\tag{1}
$$

$$
\pi^ {*} = \arg \max _ {\pi} J (\pi)\tag{2}
$$

The value function $V ^ { \pi } ( { \pmb x } _ { t } )$ represents the expected return starting from state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and following policy π thereafter.

$$
V ^ {\pi} (\pmb {x} _ {t}) = \mathbb {E} _ {\pi} \left[ J (\pi) | \pmb {x} _ {0} = \pmb {x} \right]\tag{3}
$$

Similarly, the action-value function, $\mathrm { o r } \ ^ { \mathrm { * } } \mathrm { Q }$ -function $\mathbf { \omega } ^ { \prime \prime } Q ^ { \pi } ( \mathbf { x } _ { t } , \mathbf { \boldsymbol { u } } _ { t } )$ represents the expected return from starting from state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and taking action $\mathbf { \Delta } \mathbf { u } _ { t } .$ , assuming that policy π is followed otherwise:

$$
Q ^ {\pi} (\boldsymbol {x} _ {t}, \boldsymbol {u} _ {t}) = \mathbb {E} _ {\pi} \left[ J (\pi) | \boldsymbol {x} _ {0} = \boldsymbol {x}, \boldsymbol {u} _ {0} = \boldsymbol {u} \right]\tag{4}
$$

Two broad classes of algorithms have emerged to solve the previously described problem: policy optimization methods and value-based methods. These two methods have been effective in deep RL, where a DNN has been used as a function approximator to mitigate the “curse of dimensionality” stemming from the discretization of both action and state spaces often necessary to solve continuous problems [44]. The use of DNNs allows the parameterization of the policy π π<sub>θ</sub> where $\boldsymbol { \theta } \in \Omega \stackrel { \cdot } { \subseteq } \mathbb { R } ^ { n _ { \theta } }$ represents the parameters of the policy and Ω is the parameter space (Figure 2).

A popular group of policy optimization methods leverages policy gradients to optimize the policy in reinforcement learning [45]. Policy gradient methods, such as Trust Region Policy Optimization (TRPO) [46] and REINFORCE [45], follow a stochastic gradient ascent strategy to update the policy parameters with a scalar learning rate α:

$$
\theta \leftarrow \theta + \alpha \nabla_ {\theta} \widehat {J} (\boldsymbol {\theta})\tag{5}
$$

These methods directly optimize the expected return $J ( \pmb \theta )$ by following the gradient of the policy parameters θ. The gradient is estimated from sampled trajectories collected by rolling out the current policy in the environment. Policy gradient methods offer several advantages, including their ability to handle continuous action spaces effectively, and the direct optimization of the policy. However, they often suffer from high variance in gradient estimates leading to these methods converging to locally optimal policies. The specific algorithms within this family have their own characteristics; for instance, TRPO [46] provides more stable updates but can be computationally expensive, and REINFORCE [45], while conceptually straightforward, often suffers from high variance in practice.

The second class of reinforcement learning methods comprises value-based algorithms, such as Deep Q-Network (DQN) [47], which rely on learning an action-value (or Q) function. This Q-function can be approximated with a deep neural network with parameters $\bar { \phi } \in \Phi \subseteq \mathbb { R } ^ { n _ { \phi } }$ resulting in $Q _ { \phi } \ ( { \mathrm { F i g u r e } } \ 3 )$ . The parameters are then updated by minimizing mean squared error against targets given by the Bellman recursion equation as follows:

![](images/328d004fdcc23ad3bda89ded1692ebd80ba98a47f4d94aaad238aa83cdde37eb.jpg)  
Figure 3: Deep Q-function Q<sub>ϕ</sub>

$$
\mathcal {L} (\phi) = \mathbb {E} _ {(\boldsymbol {x} _ {t}, \boldsymbol {u} _ {t}, r _ {t}, \boldsymbol {x} _ {t + 1})} \left[ (r _ {t} + \gamma \max _ {\boldsymbol {u} _ {t + 1}} Q _ {\phi} (\boldsymbol {x} _ {t + 1}, \boldsymbol {u} _ {t + 1}) - Q _ {\phi} (\boldsymbol {x} _ {t}, \boldsymbol {u} _ {t})) ^ {2} \right]\tag{6}
$$

The idea is to minimize the temporal difference error between the Q-value estimates and the backed-up estimates from the next state and reward, as this approximates the Bellman optimality condition.

While DQN has been successful in discrete action spaces, extending these methods to continuous action spaces presents challenges. Two notable non-actor approaches for continuous action spaces are Continuous Action Q-Learning (CAQL) [48] and Constrained Continuous Action Q-Learning (cCAQL) [49]. CAQL adapts the Q-learning framework to continuous actions by using a neural network to represent the Q-function and optimizing it with respect to actions. cCAQL improves upon CAQL by introducing constraints to the action selection process, which helps to stabilize learning and improve robustness in continuous action spaces. As an alternative to directly optimizing Q-functions, actor-critic algorithms have also been employed to extend deep Q-networks methods to continuous action-space problems by including an actor-network that approximates the action taken by maximizing the Q-function. These modern RL algorithms include TD3 [50] and Soft-Actor Critic (SAC) [51].

## 2.2 Evolutionary Strategies in Reinforcement Learning

Within policy optimization methods, in addition to algorithms that leverage policy gradients, it is also possible to use evolutionary algorithms. Both types of algorithms update the parameters to optimize the policy which takes states as inputs and outputs (optimal) control actions (Figure 2). This distinction is not unlike evolutionary and gradient-based algorithms in traditional optimization problems, i.e., evolutionary algorithms simply provide an alternative framework for learning the policy parameters. Evolutionary strategies (ES) are a class of data-driven optimization algorithms inspired by principles of biological evolution. These algorithms optimize policies by iteratively generating populations of candidate solutions, evaluating their fitness (performance), and selectively propagating the fittest individuals to subsequent generations through processes similar to mutation, recombination, and selection.

In the context of RL, ES can be used to optimize the parameters θ of a policy π<sub>θ</sub> directly, without relying on gradient information. These policies are evaluated in the environment on an episodic basis with their cumulative return as shown by a parameterized variant of Equation 1:

$$
J (\pmb {\theta}) = \mathbb {E} _ {\pmb {\theta}} \left[ \sum_ {t = 0} ^ {T} \gamma^ {t} r _ {t} (\pmb {x} _ {t}, \pi_ {\pmb {\theta}} (\pmb {x} _ {t})) \right]\tag{7}
$$

Instead of using the approximation of the Q-function (Equation 4) or using a return gradient estimate to optimise the policy parameters. ES-RL algorithms use the estimate of the J(θ) and directly update the parameters towards those policies that produce higher returns to improve performance [52]. The key advantages of ES-RL algorithms include the ability to be easily parallelized, making them computationally efficient for evaluating multiple candidate solutions simultaneously. Additionally, ES-RL algorithms are less susceptible to getting trapped in local optima compared to gradient-based methods, as they explore the parameter space more “globally” through population-based search and they do not rely on stochastic estimates of gradients, which are also computationally expensive. Furthermore, ES-RL methods can be more robust to the inherent noisiness often associated with stochastic gradient descent (SGD) methods used in policy gradient approaches. Given these advantages, there has been some research interest in the ES-RL. Salimans et al. [53] applied developed an ES-RL algorithm and evaluated it on MuJoCo and Atari environments resulting in comparable performance to policy gradient methods such as TRPO [46]. Wu, de Carvalho Servia, and Mowbray [54] used a hybrid strategy of derivative-free optimization techniques to solve an inventory management problem with improved performance over the policy gradient method Proximal Policy optimization (PPO) [55].

## 2.3 PID Controllers

The PID controller is a widely used feedback mechanism employed in industrial control systems [3]. The discrete PID controller calculates an error value $e _ { t }$ in discrete time as the difference between a desired setpoint and a measured process variable, and applies a correction based on three parameters: proportional $( K _ { P } )$ , integral time constant $( \tau _ { i } )$ and derivative time constant $( \tau _ { d } )$ . Note that other parameterizations of these degrees of freedom are possible. The proportional term applies a control action proportional to the current error, providing an immediate response to deviations from the setpoint. The integral term accumulates the error over time and applies a control action to eliminate steady-state errors. The derivative term considers the rate of change of the error and provides a dampening effect to prevent overshoot and oscillations. The discrete position form of a single PID controller is defined as

$$
u _ {t} = K _ {p} e _ {t} + \frac {K _ {p}}{\tau_ {i}} \sum_ {t = 0} ^ {t} e _ {t} + K _ {p} \tau_ {d} (e _ {t} - e _ {t - 1})\tag{8}
$$

where $e _ { t } = x _ { i , t } ^ { * } - x _ { i , t }$ , is the setpoint error of state i at timestep t, with the setpoint $\boldsymbol { x } _ { i , t } ^ { * }$ of state i at timestep t.

Tuning the PID gains refers to finding values of the parameters $K _ { p } , \tau _ { i }$ , and $\tau _ { d }$ that result in good closed-loop performance (often measured by integrated squared error, etc.) and is crucial for achieving desired control performance. As a result, many popular tuning methodologies have been developed, including the Internal Model Control [56, 57] and relay tuning [58]. The first of these methods is a model-based technique and the second excites the system, and uses the response to estimate the three PID parameters.

The time-invariant PID structure can achieve good performance on (approximately) linear systems. Historically, this condition was often sufficient, as processes are often operated around a known setpoint in an approximately linear region; however, more recent applications in control of nonlinear systems (e.g., transient, intensified, or cyclic processes) motivate more advanced control strategies. Given a nonlinear system the PID parameters will be dependent on the operating point, which motivates a gain scheduled approach. Gain scheduling involves designing multiple PID controllers for different operating regions and switching between them based on the current process conditions. In industrial applications, a common approach to gain scheduling is through the use of lookup tables, where the PID gains are pre-computed and stored for different operating conditions or setpoints [59]. More recently, data-driven, model-free approaches to gain scheduling have gained traction as they are able to design the control directly from a single set of plant input and output data without the need for system identification [60]. Despite the advancements in PID control, challenges remain in terms of the manual effort required for controller tuning, and the limited performance in highly nonlinear and time-varying systems. These challenges motivate the integration of PID control with data-driven and learning-based approaches, such as reinforcement learning, to leverage the strengths of both paradigms. While more advanced control strategies like Model Predictive Control (MPC) exist, our focus on PID control is motivated by the ability to leverage existing infrastructure and well-established systems in industrial settings, providing a practical and widely applicable solution. Furthermore, most industrial applications of MPC utilize a lower-level PID as the regulatory controller hence, highlighting the prominence of PID control in chemical processes.

![](images/5ab2bbe997238f23edfc3ac4e6bbf47e5cd790fc643da8a951b3aac5501db053.jpg)  
Figure 4: CIRL Agent

## 3 Methodology

This section presents the proposed control-informed reinforcement learning (CIRL) framework, which integrates PID control structures into the policy architecture of deep RL agents. The methodology covers the CIRL agent design, policy optimization algorithm, and implementation details.

## 3.1 Control-Informed Reinforcement Learning (CIRL) Agent

The CIRL agent consists of a deep neural network policy augmented with a PID controller layer, as illustrated in Figure 4. The base neural network takes the observed states $\mathbf { } _  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf $ as inputs and outputs the PID gain parameters $K _ { p , t } , \tau _ { i , t } ,$ and $\tau _ { d , t } \in \mathbb { R } ^ { n _ { u } }$ at each timestep t. The PID controller layer then computes the control action $\mathbf { \Delta } \mathbf { u } _ { t }$ based on the error signal ${ \pmb e } _ { t } = { \pmb x } _ { t } ^ { * } - { \pmb x } _ { t }$ and the current learned gain parameters.

The agent’s state s includes $N _ { t }$ timesteps of history for both the state and setpoint, where $N _ { t } > 2$ to fully define the velocity-form PID controller:

$$
\pmb {s} _ {t} = \left[ \pmb {x} _ {t \dots t - N _ {t}}, \pmb {x} _ {t \dots t - N _ {t}} ^ {*} \right]\tag{9}
$$

The PID layer is represented in the velocity form since, if the position form of the PID controller (Equation 8) is used and the gain changes suddenly, this can cause disturbances to the system [4]. The velocity form mitigates this issue by ensuring that the control input does not change abruptly despite sudden gain changes, and it is not necessary to reset the integral term. The $k ^ { t h }$ PID controller of the system is represented by:

$$
\Delta u _ {t} ^ {(k)} = K _ {p, t} ^ {(k)} \Delta e _ {t} ^ {(k)} + \frac {K _ {p , t} ^ {(k)}}{\tau_ {i , t} ^ {(k)}} e _ {t} ^ {(k)} \Delta t + K _ {p, t} ^ {(k)} \tau_ {d, t} ^ {(k)} \frac {\Delta^ {2} e _ {t} ^ {(k)}}{\Delta t}\tag{10}
$$

where $\Delta e _ { t } ^ { ( k ) } = e _ { t } ^ { ( k ) } - e _ { t - 1 } ^ { ( k ) } , \Delta ^ { 2 } e _ { t } ^ { ( k ) } = \Delta e _ { t } ^ { ( k ) } - 2 e _ { t - 1 } ^ { ( k ) } + e _ { t - 2 } ^ { ( k ) }$ and the superscript (k) denotes the index of the controller, where $k \in { 0 , 1 , \dots , n _ { u } } ,$ , and $n _ { u }$ is the total number of controllers in the system.

Through interacting with the environment, the CIRL agent aims to maximize the cumulative reward given by $\boldsymbol { r } _ { t } \in \mathbb { R }$ at each time step. For process control regulatory problems, various reward functions have been proposed. In general, they all involve some measure of (integrated) setpoint error, either squared or absolute, and/or a penalty for control action, similar to MPC objective functions. Adopting a similar notation to MPC, a squared error term penalizes deviations of the controlled variable from the setpoint, with larger deviations penalized more heavily:

$$
r _ {t} = - \left(\boldsymbol {e} _ {t} ^ {T} Q \boldsymbol {e} _ {t} + \boldsymbol {u} _ {t} ^ {T} R \boldsymbol {u}\right)\tag{11}
$$

where $Q \in \mathbb { R } ^ { n _ { x } \times n _ { x } }$ and $\pmb { R } \in \mathbb { R } ^ { n _ { u } \times n _ { u } }$ are weighting factors that balance the trade-off between tracking performance and control effort.

It is important to note that derivative information is not passed between the PID controller and the neural network in the proposed CIRL agent architecture, as we take an evolutionary optimization strategy. Future work may study an integrated gradient-based learning strategy. The CIRL rollout pseudocode is given in Algorithm 1

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: CIRL Rollout
Input: Policy Parameters θ, Number of simulation timesteps  $n_{s}$ , Discrete time environment f
Output: Cumulative Reward R
1  $s \leftarrow s_{0}$  // Reset observation to initial state
2  $R \leftarrow 0$  // Initialize cumulative reward
3 for  $t = 0$  to  $n_{s} - 1$  do
4  $K_{p,t}, \tau_{i,t}, \tau_{d,t} \leftarrow \pi_{\theta}(s_{t})$  // Get current PID gains from policy
5  $u_{t} \leftarrow \text{PID}(K_{p,t}, \tau_{i,t}, \tau_{d,t}, e_{t}, e_{t-1}, e_{t-2})$  // Use PID controller to output control input
6  $x_{t+1}, r_{t} \leftarrow f(u_{t}, x_{t})$  // Take one timestep in the environment
7  $s_{t+1} \leftarrow [x_{t+1}, x_{t}, x_{t+1}^{*}]$  // Update observation vector
8  $R \leftarrow R + r_{t}$ 
9 end
10 return R    // Return cumulative reward
</div>

## 3.2 Implementation of Policy Optimization

In this work, the CIRL agent’s policy is optimized using a hybrid approach based on evolutionary strategies, combining random search and particle swarm optimization (PSO) [61]. A population of candidate policy parameter vectors is initialized by sampling randomly from the allowable ranges for each parameter dimension. The parameters in this case are the weights of the neural networks. This initial random sample provides a scattered set of starting points that encourages exploration of the full parameter landscape. The random population undergoes N iterations in which the objective function value (cumulative reward obtained by the policy in the environment) is evaluated for each policy to initialize the population in a good region of the policy space. The objective function to be maximized is:

$$
J (\pmb {\theta}) = \mathbb {E} _ {\pi_ {\pmb {\theta}}} \left[ \sum_ {t = 0} ^ {T} - (\pmb {e} _ {t} ^ {T} Q \pmb {e} _ {t} + \pmb {u} _ {t} ^ {T} R \pmb {u}) \right]\tag{12}
$$

The best, or fittest, policies from this initial random sampling are carried forward as seeds to initialize the PSO phase of the algorithm. The PSO phase is then started, allowing the particles (policy parameter vectors $\pmb \theta _ { i } )$ to explore areas around the initially fit random vectors in a more structured manner. In each PSO iteration, particle velocities and positions are updated as:

$$
\pmb {v} _ {i} ^ {t + 1} = w \pmb {v} _ {i} ^ {t} + c _ {1} r _ {1} (\pmb {p} _ {i} ^ {t} - \pmb {\theta} _ {i} ^ {t}) + c _ {2} r _ {2} (\pmb {g} ^ {t} - \pmb {\theta} _ {i} ^ {t})\tag{13}
$$

$$
\pmb {\theta} _ {i} ^ {t + 1} = \pmb {\theta} _ {i} ^ {t} + \pmb {v} _ {i} ^ {t + 1}\tag{14}
$$

where $\boldsymbol { v } _ { i } ^ { t }$ and ${ \boldsymbol { \theta } } _ { i } ^ { t }$ are the velocity and policy parameter vector of particle i at iteration $t , w =$ is the inertia weight, $c _ { 1 }$ and $c _ { 2 }$ are the cognitive and social acceleration constants, $r _ { 1 }$ and $r _ { 2 }$ are random numbers in $[ 0 , 1 ] , p _ { i } ^ { t }$ is the personal best policy parameter vector of particle $i ,$ and $g ^ { t }$ is the global best policy parameter vector of the swarm. This hybrid approach leverages the global exploration capabilities of initial random sampling, while also taking advantage of the PSO’s ability to collaboratively focus its search around promising areas identified by the initial random search. The pseudocode for the policy optimization procedure is given shown by the block diagram (Figure 5). We highlight that the CIRL framework is agnostic to the policy optimization strategy, i.e., policy gradients or other policy optimization techniques can be used. Our choice of evolutionary algorithms in this work is motivated by optimization performance given the small size of the neural network, as well as robustness in training.

![](images/6342389b48c5ff8d12401172705822a4c84f856afe18a16e6577494148aeeab1.jpg)  
Figure 5: Block diagram of the policy optimization algorithm

## 4 Results and analysis

## 4.1 Computational Implementation

This section outlines the computational implementation employed in our study. We describe the state representation, neural network architecture, optimization parameters, and benchmark comparisons used to evaluate our proposed approach. Additionally, we provide information on the computational resources used. The RL state $s _ { t }$ representation used for the CIRL agent included $N _ { t } = 2$ timesteps of history, which is the minimum number of timesteps to define the PID controller layer:

$$
\pmb {s} _ {t} = \left[ \pmb {x} _ {t}, \pmb {x} _ {t - 1}, \pmb {x} _ {t - 2}, \pmb {x} _ {t} ^ {*}, \pmb {x} _ {t - 1} ^ {*}, \pmb {x} _ {t - 2} ^ {*} \right]\tag{15}
$$

The neural network architecture of the CIRL agent consists of three fully connected layers, each containing 16 neurons with ReLU activation functions, with the output being clamped to the normalised PID gain bounds. The CIRL agent is compared to a pure-RL implementation. This pure-RL agent consists solely of a deep neural network without the PID layer; we found this to require a larger network size and use three fully connected layers with 128 neurons. While other architectures incorporating previous information, such as recurrent neural networks (e.g., LSTMs, GRUs), could be employed, we opted for this simpler structure in the present study. For the PSO algorithm used for policy optimization, we define the inertia weight w as 0.6, while both the cognitive and social acceleration constants $( c _ { 1 }$ and $c _ { 2 } ,$ , respectively) are set to 1. The policy optimization algorithm is initialized with $N = 3 0$ policies, before starting the PSO loop for $T = 1 5 0$ iterations with $n _ { p } = 1 5$ particles. In this PSO loop, $n _ { e } = 3$ episodes are used for each policy evaluation, with $n _ { s } = 1 2 0$ timesteps in each rollout. All training was conducted on a 64-bit Windows laptop with an Intel i7-1355U CPU @ 3.7 GHz and an NVIDIA RTX A500 (Laptop) GPU. The CIRL agent required approximately 10 minutes of training time.

## 4.2 CSTR Case Study

To demonstrate the proposed algorithm, simulation-based experiments were carried out on a CSTR system (Figure 6) where both the volume and temperature are controlled. Though conceptually simple, this case study represents a non-trivial, multivariable system with nonlinear dynamics, capturing many challenges representative of those in real-world processes.

![](images/7af0fe8cdfdcc46596a917c3ef0e8d23c52e8cb1f4ff7e32919deb6168095fe6.jpg)  
Figure 6: CSTR Process Flow Diagram

The following generalized reactions take place in the reactor, where B is the desired component:

$$
A \underset {r _ {a}} {\rightarrow} B \underset {r _ {b}} {\rightarrow} C\tag{16}
$$

The following system of ordinary differential equations models the dynamics of the three chemical components in the reactor: $C _ { A }$ (concentration of A in mol/m<sup>3</sup>), $C _ { B }$ (concentration of B in mol/m<sup>3</sup>) and $C _ { C }$ (concentration of C in mol/m<sup>3</sup>), respectively.

$$
\frac {d C _ {A}}{d t} = \frac {F _ {i n} C _ {A , i n} - F _ {o u t} C _ {A}}{V} - r _ {a}\tag{17}
$$

$$
\frac {d C _ {B}}{d t} = r _ {b} - r _ {a} - \frac {F _ {o u t} C _ {B}}{V}\tag{18}
$$

$$
\frac {d C _ {C}}{d t} = r _ {b} - \frac {F _ {o u t} C _ {C}}{V}\tag{19}
$$

where $F _ { i n }$ is the volumetric flow of feed into the system $( \mathrm { { m ^ { 3 } / m i n } } )$ , $C _ { A , i n }$ is the feed concentration of species A (mol/m<sup>3</sup>), r is the reaction rate for reaction j (mol/m<sup>3</sup>/min), and V is the volume of the CSTR (m<sup>3</sup>). For this subsystem to be fully defined, the reaction rates are described by Arrhenius relationships for both reactions:

$$
r _ {a} = k _ {a} e ^ {\frac {E _ {a}}{R T}} C _ {A}\tag{20}
$$

$$
r _ {b} = k _ {b} e ^ {\frac {E _ {b}}{R T}} C _ {B}\tag{21}
$$

where $k _ { a } , k _ { b }$ are the Arrhenius rate constants $( \mathrm { s } ^ { - 1 } ) , E _ { A } , E _ { B }$ are the activation energies (J/mol), R is the universal gas constant (8.314 J/mol.K), and T is the temperature (K). The dynamics of the reactor temperature $T \left( \mathrm { K } \right)$ and volume V $( \mathrm { m ^ { 3 } } )$ are described by the following ordinary differential equations:

$$
\frac {d T}{d t} = \frac {F _ {i n} (T _ {f} - T)}{V} + \frac {\Delta H _ {a}}{\rho C _ {p}} r _ {A} + \frac {\Delta H _ {b}}{\rho C _ {p}} r _ {B} + \frac {U A}{V \rho C _ {p}} (T _ {c} - T)\tag{22}
$$

$$
\frac {d V}{d t} = F _ {i n} - F _ {o u t}\tag{23}
$$

where $T _ { f }$ is the inlet stream temperature (K), $\Delta H _ { A } , \Delta H _ { B }$ are the heats of reaction (J/mol), $\rho$ is the density of the solvent $( \mathrm { k g } / \mathrm { L } ) , C _ { p }$ is the heat capacity (J/kg/K), U is the overall heat transfer coefficient $\mathrm { ( J / m i n / m ^ { 2 } / K ) }$ , A is the heat transfer area $( \mathbf { m } ^ { 2 } )$ , and $T _ { c }$ is the coolant temperature (K). The parameters used in the simulation experiments are shown in Table 1. The case study was implemented as a Gym environment [62] to provide a standardized format designed for RL algorithms. Within the gym environment, the system of ODEs are integrated using SciPy’s ODEInt method.

Table 1: Parameters for the CSTR dynamic model

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td> $T_f$ </td><td>350 K</td></tr><tr><td> $C_{A,in}$ </td><td>1 mol/m3</td></tr><tr><td> $F_{out}$ </td><td>100 m3/sec</td></tr><tr><td> $\rho$ </td><td>1000 kg/m3</td></tr><tr><td> $C_p$ </td><td>0.239 J/kg-K</td></tr><tr><td>UA</td><td> $5 \times 10^{4}$  W/K</td></tr><tr><td> $\Delta H_a$ </td><td> $5 \times 10^{3}$  J/mol</td></tr><tr><td> $E_a/R$ </td><td>8750 K</td></tr><tr><td> $k_b$ </td><td> $7.2 \times 10^{10}$  s−1</td></tr><tr><td> $\Delta H_b$ </td><td> $4 \times 10^{3}$  J/mol</td></tr><tr><td> $E_b/R$ </td><td>10750 K</td></tr><tr><td> $k_b$ </td><td> $8.2 \times 10^{10}$  s−1</td></tr></table>

The three observed states of the reactor are the concentration of B $C _ { B }$ , reactor temperature $T ,$ and volume $V ,$ , which define the state vector $\pmb { x } = [ C _ { B } , T , V ]$ . We desire a policy that maps these to the action space, comprising the cooling jacket temperature $T _ { c }$ and the inlet flow rate $F _ { i n } .$ , defining the control vector $\pmb { u } = [ T _ { j } , F _ { i n } ]$ . This creates a system with two PID controllers, the first pairs $T _ { c }$ and $C _ { B }$ and the second pairs $F _ { i n }$ and V .The pairing was decided using a Relative Gain Array (RGA), which is shown in the appendix. This is additive measurement noise on all states of the CSTR. The system is simulated for 25 minutes with 120 timesteps. The bounds on the two control inputs are as follows $u ^ { L } = [ 2 9 0$ K, 99 m<sup>3</sup>/min] and $u ^ { U } = [ 4 5 0 \mathrm { K } , 1 0 5 \mathrm { m ^ { 3 } / m i n } ]$ . There are also bounds on the PID gains outputted by the DNN in the CIRL agent which are given in Table 2. The initial state is defined as $x _ { 0 } = [ 0$ mol/m<sup>3</sup>, 327 K, $1 0 2 \mathrm { m } ^ { \mathrm { \bar { 3 } } } ]$

Table 2: Bounds on PID gains

<table><tr><td></td><td> $C_b$ -loop</td><td> $V$ -loop</td></tr><tr><td> $K_p$ </td><td>[-5, 25] (K · m3/mol)</td><td>[0, 1] (s-1)</td></tr><tr><td> $τ_i$ </td><td>[0, 20] (s)</td><td>[0, 2] (s)</td></tr><tr><td> $τ_d$ </td><td>[0, 10] (s)</td><td>[0, 1] (s)</td></tr></table>

## 4.3 Training

The CIRL and pure-RL algorithms were trained on nine setpoints that span the operating space of the CSTR case study (Figure 11). The operating space is defined for $C _ { B }$ between 0.1 and 0.8 mol/m<sup>3</sup> whilst maintaining a constant volume of $\bar { 1 } 0 0 m ^ { 3 }$ . This is with the aim to learn a generalized control policy for a wide range of $C _ { b }$ setpoints. Practically, this was achieved by rollout the policy on the three sub-episodes (1-3 in Table 3) then summing them to create a single reward signal.

Table 3: Training and Test Scenarios

<table><tr><td rowspan="2">Sub-Episode</td><td colspan="2">Setpoint Schedule</td></tr><tr><td> $C_B$  [mol/m3]</td><td>V [m3]</td></tr><tr><td>1</td><td>0.1 → 0.25 → 0.4</td><td>100</td></tr><tr><td>2</td><td>0.55 → 0.65 → 0.75</td><td>100</td></tr><tr><td>3</td><td>0.7 → 0.75 → 0.8</td><td>100</td></tr><tr><td>Test</td><td>0.075 → 0.45 → 0.75</td><td>100</td></tr></table>

As mentioned above, we found that, without the PID layer, a larger DNN policy was required to reach comparable performance. Therefore, the pure-RL algorithm implemented with a larger number of neurons (128) in each full connected layer still reaches comparable training performance to CIRL (Figure 7).

![](images/d7009b2e672d11a9b5322527c0e45eac576b120f7d0785d4a67ade183e7a060c.jpg)  
Figure 7: Learning curves for both RL and CIRL policies with 16 and 128 neurons per fully connected layer

The sample efficiency of RL algorithms is one of the main concerns with their implementation. Here, we demonstrate the improved sample efficiency of CIRL compared to an RL algorithm with the PID controller removed. The CIRL agent can be seen to initialize at a higher reward than the pure-RL implementation, since it has prior knowledge of the control strategy and benefits from the inherent stabilizing properties. This leads to more efficient and faster learning compared to pure-RL approaches, as the agent can make informed decisions and requires fewer samples to learn a good policy. In the real world, this corresponds to fewer simulations/experiments before an adequate control policy is obtained. Furthermore, given the stabilizing properties of the PID layer, this results in a safer policy, which inherently maintains setpoint tracking by utilizing the setpoint error. The PID controller’s ability to continuously adjust based on the error between the desired setpoint and the current state provides a fundamental safety mechanism. This makes the overall policy more robust and less prone to dangerous deviations, especially during the early stages of learning when the neural network component might produce unreliable outputs.

The pure-RL agent, without any prior domain knowledge, needs to explore a larger number of samples, leading to slower convergence. This agent must learn the control strategy from scratch, including error correction and setpoint tracking that are inherently built into the CIRL approach. As a result, the pure-RL agent typically exhibits higher variance in its actions during the early stages of training, as it explores a wider range of potentially suboptimal strategies. The lack of a PID layer means that the neural network in the pure-RL approach is learning to output controls which is inherently a larger space than the PID-gain space. This often necessitates a larger network architecture, as seen in our implementation with 128 neurons per layer, to capture the complexity of the control task. The increased network size, while providing more expressive power, also increases the dimensionality of the parameter space that must be optimized, potentially leading to longer training times and increased computational requirements. The learning curves over 75 iterations of the policy optimization algorithm for both the CIRL and pure-RL implementations are shown in Figure 8.

![](images/d0a3f07e98c537211eb8c8b8cfc41bff9e3d1792d2abb7eb3f1955dc3ea98879.jpg)

![](images/250b68fcb3f353f00b568582b4b921b4a073fe5b38cd7c6295fd7b390be8e057.jpg)  
Figure 8: Setpoint tracking learning curves of CIRL and RL on 10 different seeds. Initial random search is omitted

## 4.4 Setpoint Tracking: Normal Operation

We then test both learned policies on a partially unseen setpoint-tracking task. Specifically, the trained policies are then tested on a setpoint schedule detailed in Table 3, which consists of three setpoints, the first setpoint is outside the training regime (shown in bold) and the other two interpolate between the training setpoints. The CIRL agent is compared to the pure-RL agent described previously in Section 4.1 and a static PID controller. The static PID controller was tuned with differential evolution strategy to find gains using the setpoints in the training regime (Table 3). The gains found for the static PID Controller are given in Table 4. Then these three controllers were simulated on the test scenario (Table 3) and shown in Figure 9.

Table 4: PID Gains for the Static PID Controller

<table><tr><td></td><td> $C_b$ -loop</td><td> $F_{in}$ -loop</td></tr><tr><td> $K_p$ </td><td>3.09 K · m $^3$ /mol</td><td>0.84 s $^{-1}$ </td></tr><tr><td> $τ_i$ </td><td>0.03 s</td><td>1.85 s</td></tr><tr><td> $τ_d$ </td><td>0.83 s</td><td>0.08 s</td></tr></table>

Figure 9: Setpoint tracking test scenario states and control inputs for CIRL, pure-RL and static PID

A conventional model-free implementation of deep RL (pure-RL in Figure 9 exhibited poor tracking when generalising to these out-of-distribution setpoints $( x _ { C _ { B } } ^ { * } = 0 . \dot { 0 } 7 5 \mathrm { ~ m o l / m ^ { 3 } ) }$ shown by the larger lower test reward (Table 5). By manipulating the proportional, integral, and derivative terms of its internal PID controller (Figure 10), the CIRL policy could adapt its control outputs to track previously unseen setpoint trajectories. This ability to adaptively tune the PID gains allowed CIRL to outperform not only the model-free RL baseline approach, but also a static PID controller tuned to the setpoints in the training data. These results highlight the key benefit of the control-informed RL approach: integrating interpretable control structures like PID into deep RL enables performance gains compared to either component in isolation.

![](images/8c39e2befdfcd96b91a5d9a1f6707a33791f335cdbdf4a7fc765bdf41a664844.jpg)  
Figure 10: Gain trajectories for the $C _ { B }$ and V loop controllers

Table 5: Final Test Reward for Pure-RL, CIRL and static PID

<table><tr><td>Method</td><td>Test Reward</td></tr><tr><td>RL</td><td>-2.08</td></tr><tr><td>CIRL</td><td>-1.33</td></tr><tr><td>Static PID</td><td>-1.77</td></tr></table>

## 4.5 Setpoint Tracking: High Operating Point

The CIRL algorithm does outperform the static controller in normal operation; however, the benefits are marginal and could potentially be attributed to an over-tuned controller. We now consider a more challenging operating scenario: if the operating point is pushed to a region of the operating space (red triangle in Figure 11) the gradient changes significantly, as can be seen at cooling temperatures above 390 K. This is due to the second reaction rate increasing and consuming species B. This also poses a problem to the PID controller and PID layer in CIRL since to maximise the concentration of species B, the proportional gain must decrease and potentially change sign to stabilize around the maximum.

![](images/83e1beea6206d4d8ab703d63d64a6933a152089f772fa5bc29a1a210be923103.jpg)  
Figure 11: Operating space at a fixed $V = 1 0 0 \mathrm { m ^ { 3 } }$ with initial and extended training setpoints.

This scenario is explored by testing on a new setpoint schedule for species B of 0.45 to 0.88 mol/m<sup>3</sup> (Figure 12). This high operating point scenario shows both the initial CIRL agent, as trained with a schedule shown in Table 3, and the static PID controller both enter a closed-loop unstable regime since their gains remain at a large positive value. To attempt to negate this problem, the CIRL agent is trained on an extended training regime which includes the maximum of the operating region. This agent with an extended training regime decreases the proportional gain (CIRL Extended in Figure 13) which stabilizes the response of the controller.

Table 6: Final Test Reward for Pure-RL, CIRL and static PID

<table><tr><td>Method</td><td>Test Reward</td></tr><tr><td>CIRL (initial)</td><td>-4.04</td></tr><tr><td>CIRL (Extended)</td><td>-2.07</td></tr><tr><td>Static PID</td><td>-6.81</td></tr></table>

![](images/da58a90b9f280dad7843bd609a43441268e7328d4da551669d18481dac6225b4.jpg)

![](images/a0f01ac98d6b44a546a44f3b11b768aa1443ecd9861e8cebc02e94936e674a35.jpg)  
Figure 12: Setpoint tracking with the static PID control, initial, and extended training CIRL

![](images/cc50d4dfa8fa20d5fed7de7218b47e2c93402fa256a13692d596c9f1a1b72a58.jpg)  
Figure 13: Gain Trajectories for the static PID control, initial, and extended training CIRL

The scenario at high operating points reveals a limitation of the initial CIRL agent, as it enters an unstable closed-loop regime similar to the static PID controller due to the significant changes in gradient at cooling temperatures above 390 K. Nevertheless, the adaptability of the deep RL component of the CIRL framework is demonstrated by extending the training regime to include the upper limits of the operating space, allowing the agent to learn and adjust its control strategy, particularly by reducing the proportional gain, to maintain stability and achieve the desired setpoint even in the presence of these challenging conditions as shown by the higher test reward in Table 6.

## 4.6 Disturbance Rejection

We now turn to evaluate the ability of the learned policies to reject disturbances. In particular, the CIRL algorithm is also tested on a scenario where there is a (unmeasured) step-change to the feed concentration of species A $( C _ { A , i n } )$ Similar to the setpoint tracking case study, the CIRL algorithm is trained on multiple disturbance sub-episodes (Table 7). Then the trained agent is tested only on interpolation within this training regime.

Table 7: Training and Test Scenarios

<table><tr><td>Sub-Episode</td><td>Disturbance $C_{A,in}$  [mol/m3]</td></tr><tr><td>1</td><td>1.5</td></tr><tr><td>2</td><td>1.6</td></tr><tr><td>3</td><td>1.9</td></tr><tr><td>Test</td><td>1.75</td></tr></table>

![](images/2276641a16daaefb27310838f3a5fa4c8c0b81304608145419328b1abad4ebf0.jpg)

![](images/b63d22b7839058113c5f1079d32f85a7791628e68195cf8b3d8a745bb7b2f942.jpg)  
Figure 14: Disturbance rejection test scenario states and control inputs for CIRL, pure-RL with nonobservable disturbance

Under this disturbance condition, which effectively changes the underlying system dynamics, CIRL demonstrates a good ability to reject the disturbance and maintain the desired setpoint tracking performance. This disturbance rejection capability stems from CIRL’s integrated PID control structure. The PID components in CIRL continuously measure and respond to the error between the setpoint and the actual system output, allowing it to adapt to and counteract unexpected disturbances in real-time, even if they were not explicitly modelled during training. Conversely, the pure-RL implementation exhibits poor setpoint tracking when faced with dynamics outside its training distribution (Table 8). Without an explicit mechanism to handle disturbances, it settles for a compromised policy, i.e., sacrificing setpoint tracking performance both before and after the test disturbance occurred. This highlights that the addition of the PID components to the CIRL provide robustness to unmodeled disturbances. Unlike the pure-RL approach that attempts to anticipate and learn responses to all possible disturbances during training, CIRL’s PID feedback mechanism allows it to adapt to unforeseen disturbances by using the measured error instead of modelling the response to the disturbance, demonstrating the fundamental advantage of closed-loop control in handling system uncertainties.

Table 8: Final Test Reward for Pure-RL, CIRL and static PID

<table><tr><td>Method</td><td>Test Reward</td></tr><tr><td>CIRL</td><td>-1.38</td></tr><tr><td>pure-RL</td><td>-1.76</td></tr></table>

## 5 Conclusion

This paper presents a control-informed reinforcement learning (CIRL) framework that integrates classical PID control structures into deep RL policies. A case study on a simulated CSTR demonstrates that CIRL outperforms both model-free deep RL and static PID controllers, particularly when tested on dynamics outside the training regime. The key advantage of CIRL lies in the embedded control structure, which allows for greater sample efficiency and generalizability. By incorporating the inductive biases of the PID controller layer, CIRL can learn effective control policies with fewer samples and adapt to novel scenarios more robustly than pure model-free RL approaches.

Future work may seek to incorporate additional existing information regarding existing PID infrastructure. For example, as a pre-processing step in the algorithm, the neural network could be initialized via offline reinforcement learning or behavioral cloning from past polices, potentially leveraging preexisting gain schedules in the plant. This initialization could potentially improve the starting point for the CIRL framework and accelerate learning. Another direction may be enabling gradient-based training of CIRL agent by investigating the end-to-end differentiability of the PID controller layer.

The proposed CIRL framework opens up exciting research directions at the intersection of control theory and machine learning. This combination seeks to benefit from the best of both worlds, merging the known disturbance-rejection and setpoint-tracking capabilities of PID control with the generalization abilities of machine learning. Further investigations into theoretical guarantees, and online adaptation schemes have the potential to enhance the sample efficiency, generalization, and real-world deployability of deep RL algorithms for control applications across various industries.

## 6 Acknowledgements

Maximilian Bloor would like to acknowledge funding provided by the Engineering & Physical Sciences Research Council, United Kingdom through grant code EP/W524323/1. Calvin Tsay acknowledges support from a BASF/Royal Academy of Engineering Senior Research Fellowship

## 7 Supplementary Information

The code and data used within this work are available at https://github.com/OptiMaL-PSE-Lab/CIRL.

## References

[1] Jodie M Simkoff et al. “Process control and energy efficiency”. In: Annual Review of Chemical and Biomolecular Engineering 11.1 (2020), pp. 423–445.

[2] Lane Desborough and Randy Miller. “Increasing Customer Value of Industrial Control Performance Monitoring—Honeywell’s Experience”. In: 2002. URL: https : / / api . semanticscholar . org / CorpusID : 14892619.

[3] Dale E Seborg et al. Process dynamics and control. John Wiley & Sons, 2016.

[4] Shuichi Yahagi and Itsuro Kajiwara. “Noniterative Data-Driven Gain-Scheduled Controller Design Based on Fictitious Reference Signal”. In: IEEE Access 11 (2023), pp. 55883–55894. DOI: 10.1109/ACCESS.2023. 3278798.

[5] Michael G Forbes et al. “Model predictive control in industry: Challenges and opportunities”. In: IFAC-PapersOnLine 48.8 (2015), pp. 531–538.

[6] Manfred Morari and Jay H. Lee. “Model predictive control: past, present and future”. In: Computers & Chemical Engineering 23.4 (1999), pp. 667–682. ISSN: 0098-1354. DOI: https://doi.org/10.1016/S0098-1354(98) 00301-9. URL: https://www.sciencedirect.com/science/article/pii/S0098135498003019.

[7] Stephen Piche et al. “Nonlinear model predictive control using neural networks”. In: IEEE Control Systems Magazine 20.3 (2000), pp. 53–62.

[8] Jus Kocijan et al. “Gaussian process model based predictive control”. In: Proceedings of the 2004 American control conference. Vol. 3. IEEE. 2004, pp. 2214–2219.

[9] Zhihao Zhang et al. “Real-time optimization and control of nonlinear processes using machine learning”. In: Mathematics 7.10 (2019), p. 890.

[10] Farshud Sorourifar et al. “A data-driven automatic tuning method for MPC under uncertainty using constrained Bayesian optimization”. In: IFAC-PapersOnLine 54.3 (2021), pp. 243–250.

[11] Rui Nian, Jinfeng Liu, and Biao Huang. “A review on reinforcement learning: Introduction and applications in industrial process control”. In: Computers & Chemical Engineering 139 (2020), p. 106886.

[12] Nathan P. Lawrence et al. “Machine learning for industrial sensing and control: A survey and practical perspective”. In: Control Engineering Practice 145 (2024), p. 105841. ISSN: 0967-0661. DOI: https://doi.org/10. 1016/j.conengprac.2024.105841. URL: https://www.sciencedirect.com/science/article/pii/ S0967066124000017.

[13] Shangding Gu et al. “A review of safe reinforcement learning: Methods, theory and applications”. In: arXiv preprint arXiv:2205.10330 (2022).

[14] Thomas A Badgwell, Jay H Lee, and Kuang-Hung Liu. “Reinforcement learning–overview of recent progress and implications for process control”. In: Computer Aided Chemical Engineering 44 (2018), pp. 71–85.

[15] Sergey Levine et al. “End-to-End Training of Deep Visuomotor Policies”. In: Journal of Machine Learning Research 17.39 (2016), pp. 1–40. URL: http://jmlr.org/papers/v17/15-522.html.

[16] David Silver et al. “Mastering the game of Go without human knowledge”. In: Nature 550 (7676 2017), pp. 354– 359. ISSN: 1476-4687. DOI: 10.1038/nature24270. URL: https://doi.org/10.1038/nature24270.

[17] Wentao Tang and Prodromos Daoutidis. “Distributed adaptive dynamic programming for data-driven optimal control”. In: Systems & Control Letters 120 (2018), pp. 36–43. ISSN: 0167-6911. DOI: https://doi.org/10. 1016/j.sysconle.2018.08.002. URL: https://www.sciencedirect.com/science/article/pii/ S0167691118301476.

[18] Radu Burtea and Calvin Tsay. “Constrained continuous-action reinforcement learning for supply chain inventory management”. In: Computers & Chemical Engineering 181 (2024), p. 108518. ISSN: 0098-1354. DOI: https: / / doi . org / 10 . 1016 / j . compchemeng . 2023 . 108518. URL: https : / / www . sciencedirect . com / science/article/pii/S0098135423003885.

[19] Panagiotis Petsagkourakis et al. “Chance constrained policy optimization for process control and optimization”. In: Journal of Process Control 111 (2022), pp. 35–45. ISSN: 0959-1524. DOI: https : / / doi . org / 10 . 1016/j.jprocont.2022.01.003. URL: https://www.sciencedirect.com/science/article/pii/ S0959152422000038.

[20] “A dynamic penalty approach to state constraint handling in deep reinforcement learning”. In: Journal of Process Control 115 (2022), pp. 157–166. ISSN: 0959-1524. DOI: https://doi.org/10.1016/j.jprocont.2022. 05.004. URL: https://www.sciencedirect.com/science/article/pii/S0959152422000816.

[21] Marwan Mousa et al. An Analysis of Multi-Agent Reinforcement Learning for Decentralized Inventory Control Systems. 2023. arXiv: 2307.11432 [cs.LG]. URL: https://arxiv.org/abs/2307.11432.

[22] P. Petsagkourakis et al. “Reinforcement learning for batch bioprocess optimization”. In: Computers & Chemical Engineering 133 (2020), p. 106649. ISSN: 0098-1354. DOI: https://doi.org/10.1016/j.compchemeng. 2019.106649. URL: https://www.sciencedirect.com/science/article/pii/S0098135419304168.

[23] Haeun Yoo et al. “Reinforcement learning for batch process control: Review and perspectives”. In: Annual Reviews in Control 52 (2021), pp. 108–119. ISSN: 1367-5788. DOI: https://doi.org/10.1016/j.arcontrol. 2021.10.006. URL: https://www.sciencedirect.com/science/article/pii/S136757882100081X.

[24] Wenbo Zhu et al. “Benchmark study of reinforcement learning in controlling and optimizing batch processes”. In: Journal of Advanced Manufacturing and Processing 4.2 (2022), e10113. DOI: https://doi.org/10.1002/ amp2.10113. URL: https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/amp2.10113.

[25] Christian D. Hubbs et al. “A deep reinforcement learning approach for chemical production scheduling”. In: Computers & Chemical Engineering 141 (2020), p. 106982. ISSN: 0098-1354. DOI: https://doi.org/10. 1016/j.compchemeng.2020.106982. URL: https://www.sciencedirect.com/science/article/ pii/S0098135420301599.

[26] Tobi Michael Alabi et al. “Automated deep reinforcement learning for real-time scheduling strategy of multienergy system integrated with post-carbon and direct-air carbon captured system”. In: Applied Energy 333 (2023), p. 120633. ISSN: 0306-2619. DOI: https://doi.org/10.1016/j.apenergy.2022.120633. URL: https://www.sciencedirect.com/science/article/pii/S0306261922018906.

[27] Niket S Kaisare, Jong Min Lee, and Jay H Lee. “Simulation based strategy for nonlinear optimal control: Application to a microbial cell reactor”. In: International Journal of Robust and Nonlinear Control: IFAC-Affiliated Journal 13.3-4 (2003), pp. 347–363.

[28] JA Wilson and EC Martinez. “Neuro-fuzzy modeling and control of a batch process involving simultaneous reaction and distillation”. In: Computers & chemical engineering 21 (1997), S1233–S1238.

[29] Catalina Valencia Peroni, Niket S Kaisare, and Jay H Lee. “Optimal control of a fed-batch bioreactor using simulation-based approximate dynamic programming”. In: IEEE Transactions on Control Systems Technology 13.5 (2005), pp. 786–790.

[30] Max Mowbray et al. “Using process data to generate an optimal control policy via apprenticeship and reinforcement learning”. In: AIChE Journal 67.9 (2021), e17306.

[31] Derek Machalek, Titus Quah, and Kody M. Powell. “A novel implicit hybrid machine learning model and its application for reinforcement learning”. In: Computers & Chemical Engineering 155 (2021), p. 107496. ISSN: 0098-1354. DOI: https : / / doi . org / 10 . 1016 / j . compchemeng . 2021 . 107496. URL: https : //www.sciencedirect.com/science/article/pii/S009813542100274X.

[32] Lingwei Zhu et al. “Scalable reinforcement learning for plant-wide control of vinyl acetate monomer process”. In: Control Engineering Practice 97 (2020), p. 104331.

[33] Athindran Ramesh Kumar and Peter J Ramadge. “DiffLoop: Tuning PID controllers by differentiating through the feedback loop”. In: 2021 55th Annual Conference on Information Sciences and Systems (CISS). IEEE. 2021, pp. 1–6.

[34] Ayub I Lakhani, Myisha A Chowdhury, and Qiugang Lu. “Stability-preserving automatic tuning of PID control with reinforcement learning”. In: arXiv preprint arXiv:2112.15187 (2021).

[35] Mostafa Sedighizadeh and Alireza Rezazadeh. “Adaptive PID controller based on reinforcement learning for wind turbine control”. In: Proceedings of world academy of science, engineering and technology. Vol. 27. Citeseer. 2008, pp. 257–262.

[36] Jay H Lee and Jong Min Lee. “Approximate dynamic programming based approach to process control and scheduling”. In: Computers & Chemical Engineering 30.10-12 (2006), pp. 1603–1618.

[37] Lena Abbasi Brujeni, Jong Min Lee, and Sirish L Shah. Dynamic tuning of PI-controllers based on model-free reinforcement learning methods. IEEE, 2010.

[38] Marcus AR Berger and JoÃo Viana da Fonseca Neto. “Neurodynamic programming approach for the PID controller adaptation”. In: IFAC Proceedings Volumes 46.11 (2013), pp. 534–539.

[39] R.R. Faria et al. “A data-driven tracking control framework using physics-informed neural networks and deep reinforcement learning for dynamical systems”. In: Engineering Applications of Artificial Intelligence 127 (2024), p. 107256. ISSN: 0952-1976. DOI: https://doi.org/10.1016/j.engappai.2023.107256. URL: https://www.sciencedirect.com/science/article/pii/S0952197623014409.

[40] Joel A Paulson, Farshud Sorourifar, and Ali Mesbah. “A tutorial on derivative-free policy learning methods for interpretable controller representations”. In: 2023 American Control Conference (ACC). IEEE. 2023, pp. 1295– 1306.

[41] Nathan P Lawrence et al. “Deep reinforcement learning with shallow controllers: An experimental application to PID tuning”. In: Control Engineering Practice 121 (2022), p. 105046.

[42] Daniel G. McClement et al. “Meta-reinforcement learning for the tuning of PI controllers: An offline approach”. In: Journal of Process Control 118 (2022), pp. 139–152. ISSN: 0959-1524. DOI: https://doi.org/10. 1016/j.jprocont.2022.08.002. URL: https://www.sciencedirect.com/science/article/pii/ S0959152422001445.

[43] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. Cambridge, MA, USA: A Bradford Book, 2018. ISBN: 0262039249.

[44] Richard S Sutton et al. “Policy gradient methods for reinforcement learning with function approximation”. In: Advances in neural information processing systems 12 (1999).

[45] Ronald J Williams. “Simple statistical gradient-following algorithms for connectionist reinforcement learning”. In: Mach. Learn. 8.3/4 (1992), pp. 229–256.

[46] John Schulman et al. “Trust region policy optimization”. In: Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37. ICML’15. Lille, France: JMLR.org, 2015, pp. 1889–1897.

[47] Volodymyr Mnih et al. Playing Atari with Deep Reinforcement Learning. 2013. arXiv: 1312.5602 [cs.LG].

[48] Moonkyung Ryu et al. CAQL: Continuous Action Q-Learning. 2020. arXiv: 1909 . 12397 [cs.LG]. URL: https://arxiv.org/abs/1909.12397.

[49] Radu Burtea and Calvin Tsay. “Constrained continuous-action reinforcement learning for supply chain inventory management”. In: Computers & Chemical Engineering 181 (2024), p. 108518. ISSN: 0098-1354. DOI: https: / / doi . org / 10 . 1016 / j . compchemeng . 2023 . 108518. URL: https : / / www . sciencedirect . com / science/article/pii/S0098135423003885.

[50] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing Function Approximation Error in Actor-Critic Methods. 2018. arXiv: 1802.09477 [cs.AI].

[51] Tuomas Haarnoja et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. 2018. arXiv: 1801.01290 [cs.LG].

[52] Daan Wierstra et al. “Natural Evolution Strategies”. In: Journal of Machine Learning Research 15.27 (2014), pp. 949–980.

[53] Tim Salimans et al. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. 2017. arXiv: 1703.03864 [stat.ML].

[54] Guoquan Wu, Miguel Ángel de Carvalho Servia, and Max Mowbray. “Distributional reinforcement learning for inventory management in multi-echelon supply chains”. In: Digital Chemical Engineering 6 (2023), p. 100073. ISSN: 2772-5081. DOI: https://doi.org/10.1016/j.dche.2022.100073.

[55] John Schulman et al. Proximal Policy Optimization Algorithms. 2017. arXiv: 1707.06347 [cs.LG].

[56] Daniel E. Rivera, Manfred Morari, and Sigurd Skogestad. “Internal model control: PID controller design”. In: Industrial & Engineering Chemistry Process Design and Development 25.1 (1986), pp. 252–265. DOI: 10.1021/i200032a041.

[57] Sigurd Skogestad. “Simple analytic rules for model reduction and PID controller tuning”. In: Journal of Process Control 13.4 (2003), pp. 291–309. ISSN: 0959-1524. DOI: https://doi.org/10.1016/S0959- 1524(02)00062-8.

[58] K.J. Åström and T. Hägglund. “Automatic tuning of simple regulators with specifications on phase and amplitude margins”. In: Automatica 20.5 (1984), pp. 645–651. ISSN: 0005-1098. DOI: https://doi.org/10.1016/0005- 1098(84)90014-1.

[59] Ertugrul Baris Ondes et al. “Model-based 2-D look-up table calibration tool development”. In: 2017 11th Asian Control Conference (ASCC). 2017, pp. 1011–1016. DOI: 10.1109/ASCC.2017.8287309.

[60] M.C. Campi, A. Lecchini, and S.M. Savaresi. “Virtual reference feedback tuning: a direct method for the design of feedback controllers”. In: Automatica 38.8 (2002), pp. 1337–1346. ISSN: 0005-1098. DOI: https: //doi.org/10.1016/S0005-1098(02)00032-8.

[61] J. Kennedy and R. Eberhart. “Particle swarm optimization”. In: Proceedings of ICNN’95 - International Conference on Neural Networks. Vol. 4. 1995, 1942–1948 vol.4. DOI: 10.1109/ICNN.1995.488968.

[62] Mark Towers et al. Gymnasium. Mar. 2023. DOI: 10.5281/zenodo.8127026. URL: https://zenodo.org/ record/8127025 (visited on 07/08/2023).

## A Policy Optimization Algorithm

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2: Policy optimization

Input: N: number of initial policies,  $n_{p}$ : number of particles,  $n_{e}$ : number of episodes per evaluation,  $n_{s}$ : number of steps per episode, T: number of iterations

Output: Optimal policy parameters  $\theta^{*}$ 

1 Initialize  $\{\theta_{i}\}_{i=1}^{N}$  randomly // Initialize population
2 for  $i \in \{1, \ldots, N\}$  do
3    $f_{i} \leftarrow 0$  // Initialize fitness
4    for  $j \in \{1, \ldots, n_{e}\}$  do
5    $R_{j} \leftarrow \text{CIRLRollout}(\theta_{i}, n_{s}, f)$  // Run Algorithm 1: CIRL Rollout
6    $f_{i} \leftarrow f_{i} + R_{j}$  // Accumulate rewards
7    end
8    $f_{i} \leftarrow f_{i}/n_{e}$  // Average fitness over episodes
9 end
10  $x \leftarrow \arg\max_{\theta_{i}} f_{i}$  // Select the best policy w.r.t. reward
11  $g \leftarrow x$  // Initialize global best
12  $p \leftarrow x$  // Initialize personal best position
13 for  $t \in \{1, \ldots, T\}$  do
14    for  $i \in \{1, \ldots, n_{p}\}$  do
15    $v_{i}^{t+1} \leftarrow Eq. (13)$  // Update particle velocity
16    $x_{i}^{t+1} \leftarrow Eq. (14)$  // Update particle position
17    $f_{i}^{t+1} \leftarrow 0$  // Initialize fitness for new position
18    for  $j \in \{1, \ldots, n_{e}\}$  do
19    $R_{j} \leftarrow \text{CIRLRollout}(x_{i}^{t+1}, n_{s}, f)$  // Run Algorithm 1: CIRL Rollout
20    $f_{i}^{t+1} \leftarrow f_{i}^{t+1} + R_{j}$  // Accumulate rewards
21    end
22    $f_{i}^{t+1} \leftarrow f_{i}^{t+1}/n_{e}$  // Average fitness over episodes
23    if  $f_{i}^{t+1} &gt; f(p_{i})$  then
24    $p_{i} \leftarrow x_{i}^{t+1}$  // Update personal best if necessary
25    end
26    if  $f_{i}^{t+1} &gt; f(g)$  then
27    $g \leftarrow x_{i}^{t+1}$  // Update global best if necessary
28    end
29    end
30 end
31 return  $\theta^{*} = g$  // Return optimal policy parameters
</div>

## B RGA Matrix

The RGA matrix used for controller pairing is displayed below using averaged gains over three repetitions:

$$
R G A = \left[ \begin{array}{c c} 0. 0 0 0 3 & 0. 9 9 9 7 \\ 0. 9 9 9 7 & 0. 0 0 0 3 \end{array} \right]\tag{24}
$$

The values in the RGA matrix suggest a strong pairing between the first controlled variable $( C _ { B }$ the concentration of B) and the second manipulated variable $( T _ { c } ,$ the cooling temperature), and between the second controlled variable (V , the volume) and the first manipulated variable $( F _ { i n }$ , the inlet flow rate).