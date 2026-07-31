# Exploring reinforcement learning in process control: a comprehensive survey

N. Rajasekhar, T.K. Radhakrishnan & N. Samsudeen

To cite this article: N. Rajasekhar, T.K. Radhakrishnan & N. Samsudeen (2025) Exploring reinforcement learning in process control: a comprehensive survey, International Journal of Systems Science, 56:14, 3528-3557, DOI: 10.1080/00207721.2025.2469821

To link to this article: https://doi.org/10.1080/00207721.2025.2469821

![](images/cd9251b4fc705188903d4bc0c550791b2c77193a5c15d880ac67d7bc161505f9.jpg)

Published online: 02 Mar 2025.

![](images/0e9e04233611b801058ea22fd64ee63c94fb3a6bd35a87d52760fdd7afb459e7.jpg)

Submit your article to this journal

![](images/5111e0e91c818b4ce5f28a71f24a4c0c33c751fe08d73fbdf47e2125b5374ed0.jpg)

Article views: 1781

![](images/e6ca43c16b5f29cba6d70c167cb8fe31d4ca095453b02a27967b6e1eb73b5a9f.jpg)

View related articles

![](images/47e8d28933bff9fd1ef49e5f0d79466508b433e656676e6b2d6184fe40da8a5d.jpg)

View Crossmark data

![](images/5cff88d2756149b73447d4532d282f99b9133faa61dbbbb856791d67086fcbb1.jpg)

Citing articles: 29 View citing articles

Check for updates

# Exploring reinforcement learning in process control: a comprehensive survey

N. Rajasekhar, T.K. Radhakrishnan and N. Samsudeen

Department of Chemical Engineering, National Institute of Technology, Tiruchirappalli, India

## ABSTRACT

Reinforcement Learning (RL) is a machine learning methodology that develops the capability to make sequential decisions in intricate issues using trial-and-error techniques. RL has become increasingly prevalent for decision-making and control tasks in diverse fields such as industrial processes, biochemical systems and energy management. This review paper presents a comprehensive examination of the development, models, algorithms and practical uses of RL, with a specific emphasis on its application in process control. The study examines the fundamental theories, methodology and applications of RL, classifying them into two categories: classical RL such as such as Markov decision processes (MDP) and deep RL viz., actor critic methods. RL is a topic of discussion in multiple process industries, such as industrial chemical process control, biochemical process control, energy systems, wastewater treatment and the oil and gas sector. Nevertheless, the paper also highlights challenges that hinder its larger acceptance, including the requirement for substantial computational resources, the complexity of simulating real-world settings and the challenge of guaranteeing the stability and resilience of RL algorithms in dynamic and unpredictable environments. RL has demonstrated significant promise, but more research is needed to fully integrate it into industrial and environmental systems in order to solve the current challenges.

ARTICLE HISTORY

Received 23 October 2024

Accepted 15 February 2025

KEYWORDS

Process control; reinforcement learning; chemical; energy; waste water treatment; biochemical process

Abbreviations: AC: Actor critic; AI: Artificial intelligence; ANN: Artificial neural networks; A3C: Asynchronous advantage actor critic; CRL : Classical Reinforcement learning; CV : Controlled variable; DDPG : Deep deterministic policy gradient; DQN: Deep Q network; DRL: Deep reinforcement learning; DP: Dynamic programming; FOMDP: Fully observable Markov decision process; GRU: Gated recurrent unit; LQR: Linear quadratic regulator; LSTM: Long short-term memory; ML: Machine learning; MV : Manipulated variable; MC: Monte Carlo; MDP: Markov decision process; MPC: Model predictive controller; MIMO: Multi input multi output; PG: Policy gradient; PID: Proportional integral derivative; PPO: Proximal policy optimisation; RL: Reinforcement learning; PPO: Proximal policy optimisation; SAC: Soft actor critic; SISO: Single input single output; TD: Temporal difference; TRPO: Trust region policy optimisation; TD3: Twin delayed deep deterministic policy gradient.

## List of symbols

C Constant $D _ { K L }$ Divergence between old and new policy $G _ { t }$ Cumulative discounted return L Loss $r _ { t }$ Instantaneous reward $U$ Action space $\nu _ { \pi }$ Value function for given policy $\nu _ { \pi } \mathrm { : }$ ∗ Optimal value function $x ^ { \prime }$ Next state $a , \beta$ Learning rate $\gamma$ Discount factor $\nabla$ Gradient θ Actor parameter

$\mu$ Sampling distribution

$\mathbb { E } _ { \pi }$ Expectation for given policy

ω Critic parameter

## 1. Introduction

Chemical processing plants have become more complex due to increased competition, tougher environmental and safety laws and fluctuating economic scenarios. The increase in the frequency of disturbances poses a challenge due to the escalating complexity of contemporary plant life (Seborg et al., 2016). Automatic control provides improved safety, environmental compliance, adherence to product quality specifications, resource utilisation eficiency and increased profitability (Coughanowr & LeBlanc, 2009).

![](images/3e45242f43b0e4c219fa002c922da01e3ceaa1562af018bdc8ffb07f5a613533.jpg)  
Schematic of AI.

Artificial intelligence (AI) is a branch of computer science that use machine learning methods to automate intelligent behavior. It aims to enable machines to perceive their environment and perform actions thereby maximise their chances of attaining preestablished goals and ultimately augmenting human intelligence. AI in recent times prompted major shifts in several industrial sectors around the globe starting from healthcare, finance, manufacturing, energy, agriculture, robotics, etc. AI’s autonomous learning capabilities, coupled with advancements in computer technology and cost reduction in data storage, have pushed it to the forefront of algorithms for various applications. The schematic of AI and its applications to the various fields is shown in Figure 1.

Machine learning (ML), an area of research within the realm of AI, encompasses the creation of statistical algorithms capable of acquiring knowledge from data and executing tasks autonomously, without the need for explicit instructions. The classification of ML is shown in Figure 1 (Right block). It comprises of supervised learning, unsupervised learning and RL (RL) (LeCun et al., 2015). In supervised learning algorithm trained on a labelled dataset, aiming to learn a mapping from input data to output labels, enabling precise predictions or judgments when faced with new, unseen data, whereas in unsupervised learning, the algorithm is trained on an unlabelled dataset with the goal of identifying structures, relationships, or patterns without the need for explicit instruction or targets for the outputs. Finally, RL where an agent learns to make decisions by interacting with an environment in a trialand-error fashion. The agent observes the current state of the environment, takes actions based on its current policy, receives feedback in terms of rewards or penalties from the environment based on its actions and then updates its policy to maximise cumulative rewards over time.

Process control is an expansive domain that has evolved over time, transitioning from initial propor tional–integral–derivative (PID) controllers (Begum et al., 2016; Ghousiya Begum et al., 2017) to more sophisticated optimum control techniques such as model predictive controllers (MPC) (Darby & Nikolaou, 2012; Pannocchia, 2015; Schwenzer et al., 2021). PID controllers are widely used in the process industry due to their simplicity of use and adaptability to changes with little process expertise. Nevertheless, many of these controllers exhibit subpar performance in the control loop because of modifications to the process conditions and parametric adjustments. PID controllers are also commonly used in multivariable control situations; nevertheless, their performance may not be suficient if the closed loop plant has a broad working region. PID controllers are suitable for basic control tasks, while advanced control schemes are more desirable for complex systems.

The general optimum control problem is articulated as a cost-minimisation issue. The linear quadratic regulator (LQR) and MPC are prevalent optimum control methodologies that utilise quadratic cost functions (Mesbah, 2016). The LQR algorithm solves optimisation issues in a single iteration, whereas the MPC algorithm solves optimisation problems continuously throughout time. MPC has greater compatibility with both linear and nonlinear system models, rendering it more versatile and adaptive. Contemporary procedures are extensively interconnected, resulting in intricate systems comprising of several sub-systems. These systems have been investigated using distributed MPC approaches (Negenborn & Maestre, 2014). Economic model predictive control (MPC) integrates economic goals with control objectives, efectively merging choices related to production levels with supervisory-level controls. Robust MPC efectively deals with discrepancies between the actual system and its model, as well as uncertainties in the model and unexpected disruptions. It employs techniques such as tube-based MPC and stochastic MPC (Mata et al., 2019).

Linear Parameter Varying (LPV) systems (Morato et al., 2020) are substantially important in process control, providing a versatile framework to manage nonlinear dynamics in constrained environments. These systems bridge linear and nonlinear control by embedding linear models that adapt to real-time variations in system parameters. This flexibility allows LPV systems to capture complex behavior and integrate seamlessly with advanced techniques like MPC (Hadian et al., 2021; Hu et al., 2019; Yang & Ding, 2020; Yu et al., 2012), enabling optimisation of control actions while adhering to operational constraints. By supporting parameter-dependent control laws, LPVbased approaches reduce conservatism and enhance robustness across diverse operating conditions. Moreover, their ability to approximate nonlinear processes through low-cost identification methods and interpolation of linear models makes them highly practical for large-scale industrial applications. LPV systems are extensively utilised in chemical process control, energy systems and aerospace, where dynamic adaptability and constraint satisfaction are crucial for performance and safety.

![](images/b813af438ef99c9faf761be7e3c0f49a041086313b351d387d85ba00f0e2d36a.jpg)  
Various attributes to design classical and model-based <sup>Figure 2.</sup>controller.

MPC formulations often require a process model, making it challenging to develop for complex systems. The design of classical and model-based controllers comprises an investigation of process dynamics, the development of a precise mathematical model, and the subsequent implementation of a control law that satisfies specific design criteria and its schematic diagram is shown in Figure 2.

Deep learning has transformed computer vision, natural language processing, finance, engineering and process industries. Optimal control issues are addressed using data-driven strategies such as Gaussian process models and historical signal trajectories. Optimal control issues include deep learning techniques, with recurrent neural networks (RNN) being often employed for simulating process dynamics. The RL-based approach and the RNN-based approach in control systems are compared in Table 1 which efectively highlights key diferences across various dimensions.

Over the past decade, RL has advanced alongside control and deep learning, with primary developments including stability in policy updates, sample eficient exploration and interactive RL. This survey paper ofers valuable intuitions into the utilisation of these methodologies in process industries. Various attributes of RL controller are shown in Figure 3.

This study specifically examines important keywords such as ‘Process control’ and ‘Reinforcement learning’. The search engine extracts scholarly literature from Web of Science, Scopus, Google, IEEE Xplore, record cited in the existing articles and its review of process sequence is shown in Figure 4. It includes peer-reviewed research articles and review articles. The survey encompasses past 5 years’ articles, as well as earlier articles for the evolution of RL section. In order to reduce redundancy, the survey’s scope does not include recent review publications that cover similar topics.

Comparison of RL and RNN based appraoches.

<table><tr><td>Aspect</td><td>RL-based approach</td><td>RNN-based approach</td></tr><tr><td>Purpose</td><td>Decision making and optimisation through interaction with the environment.</td><td>Sequential data prediction and system dynamics modelling.</td></tr><tr><td>Learning Paradigm</td><td>Model-free, trial-and-error, learns by receiving rewards.</td><td>Supervised learning, learns from labelled sequential data.</td></tr><tr><td>Model dependency</td><td>Model-free (can work without prior system knowledge).</td><td>Model-based (relies on training data to approximate system dynamics).</td></tr><tr><td>Adaptability</td><td>Highly adaptive to changing environments and unknown dynamics.</td><td>Limited adaptability; depends on retraining with new data.</td></tr><tr><td>Robustness</td><td>Robust to uncertainty, delays, and sparse rewards.</td><td>Robust in modeling time-varying dynamics but less adaptive to environmental changes.</td></tr><tr><td>Control objective</td><td>Maximises long-term rewards, handles multi-objective tasks.</td><td>Primarily focuses on accurate predictions and capturing temporal dependencies.</td></tr><tr><td>Real-time application</td><td>Suitable for online, dynamic control, especially in uncertain environments.</td><td>Suitable for system modeling and can be used in real-time with proper integration.</td></tr><tr><td>Computational cost</td><td>High, especially during training with large state-action spaces.</td><td>Moderate, but depends on the complexity of the system and data.</td></tr><tr><td>Strengths</td><td>Can handle nonlinearity, stochasticity, and complex decision-making.</td><td>Effective for sequence prediction, state estimation, and system identification.</td></tr><tr><td>Weaknesses</td><td>Requires large amounts of interaction with the environment for training.</td><td>Relies on large, high-quality labelled datasets, limited decision-making ability.</td></tr><tr><td>Complementary use</td><td>Can integrate with RNNs for state estimation or dynamics prediction in decision-making tasks.</td><td>Can enhance MPC systems by providing improved system modeling</td></tr></table>

![](images/73477bdfe8f943e9989f0e0bf4258f7e493ac6a90777ebf78ac996b36e9d03a9.jpg)  
Various attributes are required to design RL based con-<sup>Figure</sup>troller.

The subsequent part of the manuscript is organised as follows: Section 2 explains in detail on the evolution of RL. Section 3 covers basics of RL, introduction to MDP’s to diferent types of MDP’s. Deep RL explained in detail in Section 4. The recent literature review with a focus on industrial process control, biochemical process control, energy, wastewater treatment domain and oil and gas sector are explained in Section 5. Then, in Section 6 discusses challenges while implementing RL methods finally conclusions are explained in Section 7.

## 2. Evolution of RL

RL as a formal field of study didn’t emerge until the latter half of the twentieth century. The primary three domains in which the roots of RL are found are animal psychology, optimal control via dynamic programming and temporal diference learning. It’s worth noting that some concepts related to RL, such as trialand-error learning or rudimentary forms of RL, may have been informally observed or practiced in various contexts throughout over the years, particularly in the realms of psychology, animal behavior studies and early AI experiments. But, these practices are not explicitly recognised or studied as part of a structured discipline until much later. Thorndike explores the dificulties of animal cognition and behavior, with a primary focus on his renowned experiments involving puzzle boxes and cats in the year 1911. These experiments provided valuable insights into the learning mechanisms of various animal species, shedding light on how they adapt and problem-solve when faced with confined spaces. Through systematic observation and experimentation, Thorndike reveals the significance of trial and error in the learning process, ofering a deeper understanding of animal intelligence and behavior. In the year 1948, Turing introduces the p-type computing machine, consisting of neuron-like networks with two inputs for pleasure/reward and pain/punishment, aiming to discover a training method similar to a child’s learning process (Turing, 2019; Winston, 2017). Claude Shannon showcased a maze-running mouse named Theseus in 1952 through a trial-and-error procedure (Shannon, 1950). In 1954, Minsky, Farley and Clark also carried out initial computational research that focused on the examination of RL models and the development of stochastic neural analog reinforcement calculators (Clark & Farley, 1955). The utilisation of RL in engineering literature during the 1960s is notable, with Minsky’s seminal study titled ‘Steps Toward Artificial Intelligence’ exerting significant influence (Minsky, 1961). Andreae’s STeLLA system, developed in 1963, learned through trial and error in interaction with its environment. Later, it highlights teacher-led learning (Andreae, 1963). Michie developed a trial-and-error learning system for tic-tac-toe, called matchbox educable naughts and crosses engine (MENACE), which involved drawing a bead randomly from matchboxes to determine moves. These beads were then added or removed to reinforce or punish decisions (Michie, 1963). Michie and Chambers developed (Michie & Chambers, 1968) GLEE and BOXES, which are applied to balance a pole hinged to a movable cart by learning. These systems are early illustrations of RL tasks performed with partial knowledge of the environment. Klopf’s 1970s trial-and-error learning revolutionised RL, clarifying reinforcement and supervised learning confusions. Sutton and Barto developed classical conditioning models using Klopf’s animal learning theories (Klopf, 1974).

![](images/e5bfe309d131784e28cf82d5daabde6152995aa13dd2e5a4db4d6a99e9621bae.jpg)  
Review process sequence.

On the other hand, the formalisation and systematic study of RL began in the mid-twentieth century with the work of researchers like Richard Bellman and others. The concept of optimal control evolved in the late 1950s to describe the process used to develop a controller that aims to reduce the behavior of a dynamical system as time progresses. The Bellman equation developed by Richard Bellman is presently acknowledged as dynamic programming (Richard, 1956). Markov decision processes (MDPs) and the policy iteration method, pioneered by Bellman, provide a discrete stochastic framework for solving optimal control problems. In the context of stochastic optimal control problems, dynamic programming emerges as the most eficient and extensively applicable approach. Although the processing demands of this approach increase exponentially as the number of state variables increases, it has undergone significant advancements since the late 1950s. These advancements encompass expansions to partially observable MDPs, various applications, approximation techniques and asynchronous methods (Puterman, 1994).

As an ofline computational method, dynamic programming depends on accurate system models and the analytical solution of Bellman equations. The idea of dynamic programming, involving the use of precise system models and mathematical solutions to the Bellman equation, has been associated with the process of learning since the introduction of heuristic dynamic programming by Paul Werbos in the year 1977 (Webros, 1977). Researchers have established connections with RL, introducing theoretical frameworks such as neurodynamic programming and approximate dynamic programming. The objective of these approaches is to overcome the traditional limitations of dynamic programming in the context of RL.

Final classification of RL taxonomy is Temporal Difference (TD) methods. TD learning methods, unique to RL, focus on temporally successive estimates of the same quantity, playing a significant role in the RL field. TD learning or the use of secondary reinforcers, has its roots in animal learning psychology. The significance of TD concepts in artificial learning systems is acknowledged by Minsky. In 1959, Arthur Samuel implemented them in a checkersplaying programme (Samuel, 1959). In 1972, Klopf combined trial-and-error learning with temporaldiference learning, focusing on large systems and local reinforcement. He developed generalised reinforcement, where every neuron view inputs as rewards and punishments (Klopf, 1974; Klopf, 1975). Sutton and Barto refined Klopf’s ideas, focusing on animal learning theories and TD learning. They developed a psychological model of classical conditioning based on TD learning (Sutton & Barto, 1981). Watkins implemented the complete incorporation of TD and optimal control techniques to develop Q-learning (Krose, 1995). Watkins’s work sparked significant growth in RL research, particularly in machine learning and neural networks, following the 1992 success of Tesauro’s backgammon programme, TD-Gammon (Tesau & Tesau, 1995). Classic models in 1998 gave way to more recent advancements in RL algorithm i.e. the Deep-Q-Network algorithm in 2015 (Mnih et al., 2015). AlphaGo developed by Google-DeepMind, emerged victorious against Li Shishi, while Open AI triumphed in DOTA 5v5 in the year 2016 (Silver et al., 2017). The milestones of RL are shown in Figure 5.

## 3. Paradigms and algorithms of RL

RL is a rapidly evolving field in artificial intelligence research, advancing towards artificial general intelligence. Its rapid transformation has been driven by the integration of deep learning techniques, resulting in deep RL. This fusion has led to diverse applications. RL operates through a trial-and-error process, where the system interacts with its environment (Sutton & Barto,

2018). The key elements of RL are agent (A), environment (E), reward (R), state and action are defined in Table 2.

In a typical reinforcement learning (RL) algorithm, the process follows several key steps. The agent (A) interacts with the environment (E) by taking an action (U), which results in a transition from one state to another state (X). Following this action, the agent receives a reward (R) contingent upon its choice. This reward serves as feedback, allowing the agent to discern the eficacy of its action; positive rewards signify favourable outcomes, while negative ones denote fewer desirable results and it is shown in Figure 6. Subsequently, the agent adjusts its strategy accordingly: if rewarded positively, it reinforces the action, inclining towards its repetition; conversely, if the outcome is unfavourable, the agent explores alternative actions in pursuit of a more advantageous reward. Through this iterative cycle of interaction, assessment and adaptation, the agent gradually refines its decision-making process or optimal policy to achieve optimal performance within its environment.

## 3.1. Markov decision processes

The Markov decision process (MDP) ofers a mathematical framework for addressing reinforcement learning problems and is commonly applied in optimisation tasks. An MDP is characterised by states (s), actions (u), rewards (r), and state transition probabilities, as outlined in Equation (1). In the MDPs the next state depends on current state and not on the previous states. The RL environment can be modelled using MDP. There are diferent types of MDPs viz., fully observable MDP, partially observable MDP and semi-MDPs (Sutton & Barto, 2018).

$$
P (x ^ {\prime}, r | x. u) = P r \{x _ {t} = x ^ {\prime}, R _ {t} = r | X _ {t - 1} = x, U _ {t - 1} = u \}\tag{1}
$$

## 3.1.1. Fully observable MDPs

In the fully observable Markov decision process (FOMDP), all states are discretely observable in time. FOMDP is a framework in which the agent possesses full knowledge about the state of the environment at every time step ‘t’. This implies that the agent has complete knowledge and access to the present state x of the

![](images/b53228679e66e81d637f62f6f512ec1c90178d1e135cc2cb31618becf6f334dd.jpg)

Milestones in the evolution of modern RL.  
Basic RL terminology.

<table><tr><td>S. No.</td><td>Element</td><td>Description</td></tr><tr><td>1.</td><td>Agent (A)</td><td>The decision-making controller or software that learns optimal actions.</td></tr><tr><td>2.</td><td>Environment (E)</td><td>The system or plant being influenced, excluding the agent itself</td></tr><tr><td>3.</td><td>Reward (R)</td><td>A scalar value representing feedback from the environment after each action.</td></tr><tr><td>4.</td><td>State space (X)  $x \in X$ </td><td>The complete set of all possible states that the environment can occupy.</td></tr><tr><td>5.</td><td>Action space (U)  $u \in U$ </td><td>The range of actions the agent can take to influence the environment.</td></tr></table>

environment, enabling it to make well-informed decisions (Sutton & Barto, 2018). An FOMDP consists of states, actions, transition probabilities, rewards and a discount factor. In a FOMDP, the agent employs the whole state information to develop optimum policies π that maximise cumulative rewards through trial and error. The agent can traverse through time sequence $x _ { t } , u _ { t } , R _ { t + 1 } , x _ { t + 1 } , u _ { t + 1 } , R _ { t + 2 } , x _ { t + 2 } , \ldots \ldots \ldots$ and accumulate rewards and it is shown in Equation (2)

![](images/c332179ed88d016d196e864da7a2273647c705249927fe2bb07a1b9e2dc643cb.jpg)  
The agent and environment interaction in RL.

$$
\begin{array}{c} G _ {t} = R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} R _ {t + 3} \dots \dots \\ = \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1} \end{array}\tag{2}
$$

where $G _ { t }$ denotes the cumulative discounted return at time t where $\gamma$ serves as the discount factor to reflect the uncertainty of future rewards.

The RL agent aims to determine the optimal policy $\pi ^ { * }$ that maximises $G _ { t }$ cover N steps. The value function for each state in the system is described in Equation (3).

$$
\begin{array}{r l} & {\nu_ {\pi} (x) = \mathbb {E} _ {\pi} [ G _ {t} | x _ {t} = x ]} \\ & {\qquad = \mathbb {E} _ {\pi} \left[ \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1} | x _ {t} = x \right]} \\ & {\qquad = \mathbb {E} _ {\pi} [ R _ {t + 1} + \gamma G _ {t + 1} | x _ {t} = x ], \quad \forall x, \epsilon X} \end{array}\tag{3}
$$

where $\nu _ { \pi } \left( x \right)$ is value function of state x for given policy π and the action – value version of the above equation is shown in Equation (4)

$$
\begin{array}{c} q _ {\pi} (x, u) = \mathbb {E} _ {\pi} [ G _ {t} | x _ {t} = x, U _ {t} = u ] \\ q _ {\pi} (x, u) = \mathbb {E} _ {\pi} \left[ \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1} | x _ {t} = x, U _ {t} = u \right], \\ \forall   x, u \in X, U \end{array}\tag{4}
$$

Similarly, the optimal solution to a RL problem refers to the policy that generates the highest reward over trajectory and it is shown in Equation (5).

$$
\nu_ {\pi} ^ {*} (x) = m a x _ {\pi} [ G _ {t} | x _ {t} = x ]), \quad \forall x, \epsilon X\tag{5}
$$

The optima action – value (Sutton & Barto, 2018) functioned shown in Equation (6)

$$
q _ {\pi} ^ {*} (x, u) = m a x _ {\pi} q _ {\pi} (x, u), \quad \forall x, u \in X, U\tag{6}
$$

where the max represents the optimal action that is performed in the subsequent step and for the remainder of the trajectory.

## 3.1.2. Partially observable MDPs

A Partially Observable Markov decision process (POMDP) is an extension of MDPs that accounts for scenarios in which the agent lacks complete knowledge of the environment’s state. In a POMDP (Nian et al., 2020; Sutton & Barto, 2018), the agent’s decisionmaking is guided by a belief state. This belief state is a probability distribution that represents the agent’s uncertainty about the true state of the system. It is formed from observations and previous information. This framework is essential for representing and resolving decision-making problems in uncertain and dynamic environments, such as robotics, automated planning and complex games. In these scenarios, the agent must consider both the inherent unpredictability in state changes and the restricted and imprecise information it receives.

Typically, determining the exact optimal policy in a POMDP is significantly more complex compared to FOMDPs. Moreover, even if agents had complete information about the true value functions, they would still face challenges in achieving optimal behavior in POMDP systems due to the uncertainty surrounding the current state.

A Semi-MDP (SMDP) builds upon the classic MDP by permitting the duration between state transitions to be governed by any probability distribution, rather than being confined to fixed or exponential distributions (Sutton & Barto, 2018). This paradigm is especially valuable for representing decision-making issues in which actions might possess diverse durations. In a SMDP, the decision-maker adheres to a policy that selects actions based on the present state, while also considering the time of these actions. SMDPs find utility in many domains such as telecommunications, inventory control and queuing systems, where the precise timing of events significantly impacts the system’s performance.

## 3.2. Methods to solve RL problem

There are three diferent methods to solve the value and action value functions in RL namely, dynamic programming (DP), MC methods and TD methods.

DP is an algorithm that is employed to identify optimal policies $( \pi ^ { * } )$ in the presence of a perfect model. However, due to their computational demands, DP algorithms are not commonly used. Two well-known DP methods are policy iteration and value iteration.

The policy iteration seeks the optimal policy by iterating through multiple policies and retaining the one with the highest cumulative returns. It consists of policy evaluation and policy improvement. Using an iterative approach, policy evaluation predicts the value functions of policy, as illustrated in Equation (7).

$$
\begin{array}{c} \nu_ {k + 1, \pi} (x) = \mathbb {E} _ {\pi} [ R _ {t + 1} + \gamma   \nu_ {k, \pi} (x _ {k + 1}) ] \\ \nu_ {0} (x) = 0, \quad \forall x \epsilon X \end{array}\tag{7}
$$

where k: $k ^ { \mathrm { { t h } } }$ updated step, $\nu _ { k + 1 , \pi } .$ : predicted value function.

To make the policy more efective, one need to find situations where the value under the present policy $\nu _ { \pi } \left( x \right)$ , is greater than or equal to the value under any other policy, $\nu _ { \pi }$ for all states. When these kinds of situations are found, it means that the current strategy doesn’t follow the optimality principle, which makes it less than ideal. Because of this, the other policy becomes the new best one. This process will keep going until a policy is found that has a value $\nu _ { \pi ^ { * } } ( x )$ that is greater than or equal to any other policy value $\nu _ { \pi }$ for all states x in the state space X. This policy is called $\pi ^ { * }$

Value iteration (Sutton & Barto, 2018) is a simplified version of policy iteration that focuses on optimal value functions, allowing for a single-step evaluation. A perfect model of the system is required to extract the optimal policy, as it must be capable of discerning transition probabilities and selecting actions with the highest probabilities. Equation (8) illustrates the single-step assessment of policies for the action-value function and the value function.

$$
\begin{array}{r} \nu_ {k + 1} (x) = \max \mathbb {E} [ R _ {t + 1} + \gamma \nu_ {k} (x _ {t + 1}) ] \\ q _ {k + 1} (x, u) = \mathbb {E} [ [ R _ {t + 1} + \gamma \max q _ {k} (x _ {t + 1}, u _ {t + 1}) ] \end{array}\tag{8}
$$

Dynamic Programming (DP) needs a model, but the MC and TD methods don’t. They also need less computing power. Instead, agents learn from their surroundings by interacting with it, having experiences, getting rewards and evaluating the value function. MC methods (Sutton & Barto, 2018) learn from full sample returns, which means that each sampled path or ‘episode’, must end before the agent can change its value function. This process is like policy iteration in DP, but it doesn’t happen at the same time. Instead, it only changes the values of the states that are visited during each episode. MC methods work best for tasks with clear ending states that have a fixed horizon. These are called episodic tasks. By averaging the value function across a number of evaluated trajectories, these methods determine the optimal policy. However, one issue is that results could have large variances if only a few trajectories are used as samples.

TD learning, like MC methods, learns directly from sampled data (Sutton & Barto, 2018). However, it is diferent because it updates its value function before an episode ends using a method called bootstrapping, which is similar to DP value iteration. TD methods, on the other hand, estimate values at diferent times, changing the value of each visited state during simulations. TD methods have advantages for continuous tasks because they can learn quickly when they obtain rewards even before they know what the final outcomes are known Q-learning and SARSA learning are two popular TD methods featuring distinct updating rules.

## 4. Deep reinforcement learning

Classical RL (CRL) (Sutton & Barto, 2018) ofers powerful methods for learning optimal policies, but it comes with several significant drawbacks (Mnih et al., 2015). These include ineficiencies in sample usage, often requiring extensive interactions with the environment to learn efective policies, which can be impractical in real-world applications. Balancing exploration and exploitation pose another challenge, as inadequate exploration strategies can lead to suboptimal policies and prolonged learning times. CRL algorithms can exhibit high variance in learning outcomes, meaning that the quality of learned policies may vary widely across diferent training runs or environments. Designing appropriate reward functions is crucial yet challenging, as poorly defined rewards can result in unintended or undesirable agent behaviors. The curse of dimensionality limits CRL’s scalability in problems with large state and action spaces, while ensuring convergence to optimal policies remains a complex task, especially in dynamic or non-stationary environments (Singh et al., 2022). Generalisation to new tasks or environments, handling partial observability, addressing safety and ethical concerns, facilitating transfer learning, managing computational complexity and meeting real-time decision-making requirements further highlight the ongoing research challenges and limitations of CRL approaches.

Deep learning (DL) is a process not only to learn the relation among two or more variables but also the knowledge that governs the relation as well as the knowledge that makes sense of the relation (Zhang, 2018). The fundamentals of DL are a machine learning technique that employs multi-layer artificial neural networks (ANNs) (Goodfellow et al., 2016; Haykin, 2016), are examined in this section. It is a structure with multiple hidden layers that autonomously construct new features and extract abstract high-level attributes to discover the deep feature representation of data. It is derived from artificial neural networks (Haykin, 2016). There have been big steps forward in design since the early 2000s. Some examples are GANs, GRUs, VAEs, residual networks and attentionbased neural networks. These new ideas have made areas like engineering, computer vision, image processing, robotics, natural language processing and banking better. ML can be broken down into three types: unsupervised learning, supervised learning and RL. In process businesses, unsupervised learning is used for monitoring tasks like fault detection and diagnosis. DL is highly efective for managing complex industrial processes (Rajasekhar et al., 2024a). Auto-encoders, GANs, VAE and RNNs such as LSTM and GRU (Rajasekhar et al., 2024b) are some of the more complicated structures that have been studied for use in process applications. DL has been actively utilised for tasks involving supervision across process industries, especially in fault classification and soft sensing. This has been achieved through the use of architectures such as stacked auto-encoders, VAE, and LSTM. Furthermore, flexible sensors that are based on CNNs have been designed to facilitate supervised learning by extracting critical features from images. Large quantities of data are essential for the development of dependable models in both super vised and unsupervised learning techniques. Conversely, reinforcement learning (RL) entails an agent engaging with the system and receiving incentives for accomplishing objectives. In control applications and process industries, RL is efective for both traditional supervised and unsupervised learning tasks by assisting in the management of feature selection, network architecture, and adaptation to unobserved data.

Linear value function approximation requires carefully designed input features and can’t work without the right ones (Riedmiller, 1999). DNN’s function approximation ofers a richer solution, directly using states without feature specification. DNN is a popular choice for nonlinear function approximation. Some of the value approximation-based DRL approaches are discussed in the subsequent subsection.

## 4.1. Deep Q – learning

Q-learning is a classical RL algorithm of learning value function which is discussed in subsection 2.1.2. Deep Q-network (DQN), the pioneer of DRL and a variety of DQN extension methods are available in the literature. Mnih et al. (Mnih et al., 2013) describe the DQN model that DeepMind created in 2013. It is a major step forward in DRL. A type of Q-learning is used to train CNNs in this model. Later, they improved DQN’s ability to learn by adding a target network (Mnih et al., 2015). Prior to DQN, the utilisation of ANNs to approximate the value function frequently resulted in instability or non-convergence in RL applications. The target network and the experience replay (ER) mechanism are the primary methods by which DQN addressed these issues.

The ER mechanism stores agent experiences and updates network parameters using stochastic gradient descent. This breaks correlations between samples and increases data usage eficiency by randomly sampling historical data. The DQN employs two neural networks, the Q-network and the target Q-network, to enhance its stability. The Q-network updates its parameters θ by minimising loss functions as shown in Equation (9) during the training process, estimating the action-value function and the schematic diagram of DQN is shown in Figure 7.

$$
L (\theta_ {t}) = [ (r + \gamma \max Q (x ^ {\prime}, u ^ {\prime}; \theta_ {t a r}) - Q (x, u; \theta_ {t}) ] ^ {2}\tag{9}
$$

where L, θ are the loss and parameters respectively.

Calculating the derivative of the loss function concerning the weights and the resulting equation is shown in Equation (10).

$$
\begin{array}{r} \nabla_ {\theta_ {t}} L (\theta_ {t}) = [ ((r + \gamma \max Q (x ^ {\prime}, u ^ {\prime}; \theta_ {t a r}) \\ - Q (x, u; \theta_ {t})) \nabla_ {\theta_ {t}} Q (x, u; \theta_ {t}) ] \end{array}\tag{10}
$$

The target Q-network structure is identical to the $\mathrm { Q } \mathrm { - }$ network, with parameters $\theta _ { t a r }$ updated with $\theta _ { t }$ every N iteration and fixed in each time period.

DQN has grown into many diferent versions that focus on making training algorithms, neural network structure and learning processes better.

## 4.1.1. Double DQN

The Q-learning algorithm faces overestimation issues in games viz., Maze and Roulette, leading to the development of the Double Q learning algorithm (Hasselt,

![](images/099a932c1d7edc151be106af313fb0e206adcb9bc32407164414b930cf48da62.jpg)  
Schematic diagram of DQN algorithm.

2010). This approach, which integrates Q learning and deep neural networks, seeks to mitigate overestimations by partitioning the target Q value estimation into action assessment and selection. The double DQN architecture efectively mitigated overestimations and shown superior performance across multiple Atari 2600 games.

## 4.1.2. Duelling DQN

A new duelling ANN architecture (Wang et al., 2016) is introduced for improved policy evaluation, employing distinct estimators for state-dependent action advantage and state value functions, allowing the RL agent to surpass alternative approaches in the Atari 2600 environment. This duelling architecture divides abstract features into two streams, representing the scalar state value function and the action advantage function, allowing generalisation of learning across actions without altering the underlying RL algorithm.

## 4.1.3. Prioritised replay DQN

DQN and DDQN both use experience replay mechanisms to remember and reuse past experiences, but they use the same frequency and do not consider each transition’s importance. Based on TD error, Schaul et al. (Schaul et al., 2016) suggested that important transitions should be replayed more often during experience replay to improve the eficiency of learning.

## 4.1.4. Rainbow

Hessel et al. (Hessel et al., 2018) looked at six diferent types of DQN and how they could be combined.

They discovered that diferent combos worked better on Atari games and used data more efectively. They are able to combine these DQN improvements into a single learning system called Rainbow, which had the best performance available at the time.

## 4.1.5. Policy based DRL

Policy optimisation methods (Sutton & Barto, 2018) aim to identify the optimal behavior policy directly, without needing to determine value functions. This approach is particularly advantageous for problems occurring in continuous spaces. These methods can still find the best values for behavioral policy weight variables, but they can’t choose the optimal actions. The final section talks about a few significant policy optimisation methods.

In the PG methods, the policy is modelled with a parameterised function, estimating optimal policy parameters using scalar performance measures. PG methods provide better convergence guarantees than greedy approaches, as action selection probabilities change smoothly with policy parameters, unlike the greedy approach, which can drastically change with small changes. PG methods have better chances for convergence than value function-based methods. The update process for these policy parameters is detailed in Equation (11).

$$
\theta_ {K + 1} = \theta_ {K} + \alpha \nabla_ {\theta} J (\theta_ {K})\tag{11}
$$

where $\nabla _ { \boldsymbol { \theta } } J ( \boldsymbol { \theta } _ { K } ) ;$ : gradient of the performance measure and α is learning rate

PG approaches (Sutton & Barto, 2018) parameterise agent policies to be diferentiable and finite, making them stochastic for exploration. Stochastic policies are favoured for challenges such as Atari games, yielding superior outcomes compared to deterministic ones. The performance metric relies based on the chances of choosing an action and the distribution of states both influenced by policy parameters. It is hard to estimate these parameters because they are unknown and change depending on the surroundings. This means that the gradient of the performance measure can’t be found directly. By excluding the gradient of state distribution for temporal tasks, the PG theorem gives a mathematical way for establishing performance gradients based on policy parameters and it is shown in Equation (12).

$$
\nabla_ {\theta} J (\theta) \propto \sum_ {x \in X} d _ {\pi} (x) \sum_ {u \in U} Q _ {\pi} (x, u) \nabla \pi (u | x; \theta)\tag{12}
$$

here $\begin{array} { r } { d _ { \pi } ( x ) = \operatorname* { l i m } _ { k  \infty } P ( x _ { k } = x | x _ { 0 } , \pi _ { \theta } ) } \end{array}$ ; it is a stationary state distribution,

from classical REINFORCE algorithm (Willia, 1992), the gradient is shown in Equation (13)

$$
\nabla_ {\theta} J (\theta) = E _ {\pi} \left[ G _ {k} \frac {\nabla_ {\theta} \pi (u _ {k} | x _ {k} ; \theta)}{\pi (u _ {k} | x _ {k} ; \theta)} \right]\tag{13}
$$

A performance gradient is found by taking the expectation in Equation (13) and sampling it at each time step. The stochastic gradient ascent method uses this to make the REINFORCE update and it is shown in Equation (14)

$$
\theta_ {K + 1} = \theta_ {K} + \alpha G _ {k} \left[ \frac {\nabla_ {\theta} \pi (u _ {k} | x _ {k} ; \theta)}{\pi (u _ {k} | x _ {k} ; \theta)} \right]\tag{14}
$$

the updated REINFORCE parameter can be expressed in Equation (15).

$$
\theta_ {K + 1} = \theta_ {K} + \alpha G _ {k} \nabla_ {\theta} \ln \pi (u _ {k} | x _ {k}; \theta_ {k})\tag{15}
$$

## 4.2. Actor-critic methods

Any actor-critic (AC) algorithm consisting of two distinct networks namely actor and critic that exchange state information is shown in Figure 8. The actor network produces actions, while the critic network evaluates the value of the current action and updates its value function. The actor network modifies its action plan to enhance the value of the action. The critic network assesses the action strategy, improving the gradient estimate and achieving an optimal action strategy.

![](images/5a5cfbc42ea7574311f7222bd7cd06022f100e3a6f3d04dc2d16be5f4c62d18e.jpg)  
Schematic of actor-critic structure.

These features guarantee that the AC algorithm can achieve its optimal action strategy with lower variance.

Konda and Tsitsiklis (Konda & Tsitsiklis, 2003) introduced the initial actor-critic framework, Peters and Schaal (Peters & Schaal, 2008) presented the natural actor-critic method and Bhatnagar et al. (Bhatnagar et al., 2007) introduced four new RL algorithms based on the actor-critic approach.

## 4.2.1. Deep deterministic policy gradient (DDPG)

Deep deterministic policy gradient (DDPG) algorithm is a class of actor-critic RL algorithms, which combines DQN and DPG, presented by Lillicrap et al. (Lillicrap et al., 2016) in 2016. It utilises DQN as a nonlinear function approximator to estimate Q-values. DDPG combines the advantages of experience replay bufer and target networks to get accurate and robust Q-values. In the DDPG design, the target network weights do not immediately copy the critic and policy network weights. Instead, they gradually adjust to follow the learned network weights to enhance stability. The critic component of the DDPG algorithm utilises a standard DQN to calculate the Q-value estimate through loss function minimisation and the schematic of DDPG algorithm is shown in Figure 9.

## 4.2.2. Twin delayed deep deterministic policy gradient (TD3)

The DDPG approach in chemical engineering is often fragile due to hyperparameters and tuning. MC learning enhances the algorithm, promoting stability and eficiency. However, for a nonlinear semi batch polymerisation process, it is seen that DDPG has overestimation problems, afecting policy stability and convergence to local optima. An upgraded version, TD3, is proposed to address this issue (Fujimoto et al., 2018) and it is shown that TD3 algorithm has several advantages over the DDPG algorithm. One key benefit is that TD3 avoids overestimation of the target Q-value by employing two critic networks to represent diferent Q values, which helps to suppress continuous overestimation and ensures the output value is expressed as the target Q value. Additionally, TD3 delays policy updates, optimising the direction of the actor network’s parameter updates to maximise the Q value. This delay allows for gradient boosting without overestimation and helps stabilise the learning process by incorporating stabilisation techniques and policy knowledge acquisition, which can mitigate erroneous updates. Furthermore, TD3 utilises target policy smoothing, where if the Q-function approximator produces an incorrect peak, the target policy can be smoothed by using truncated normal distribution noise and constraining the target action within a specified action range. These techniques collectively contribute to the robustness and efectiveness of the TD3 algorithm in RL tasks and the schematic of TD3 algorithm is shown Figure 10.

![](images/fa23a11f31422d37ad4e39df62ccbd2048c52bbd3b5bb3be86fc007fca8ae579.jpg)  
Schematic of DDPG algorithm.

## 4.2.3. Asynchronous advantage actor critic (A3C)

The integration of DNN with online RL has been a topic of interest and the related stability issues have been addressed through various methods, including data correlation and nonstationary targets (Anschel et al., 2017). However, these methods require significant computational power and memory is limited to of-policy RL algorithms.

A3C uses multiple agents asynchronously on data samples, solving data correlation problems requiring less computation power. It outperforms previous approaches on Atari gaming platforms and reduces training time. Policy parameters and value function updated parameters are shown in Equations (16) and (17).

$$
\nabla_ {\theta} J (\theta) \approx \frac {1}{N} \sum_ {i = 0} ^ {N} \nabla_ {\theta} l o g \pi_ {\theta} (u _ {i}, x _ {i}). A (x _ {i}, u _ {i})\tag{16}
$$

where J is expected return, A is advantage function representing the advantage of acting u in state x.

$$
\nabla_ {\omega} J (\omega) = \frac {1}{N} \sum_ {i = 1} ^ {N} \nabla_ {\omega} (V _ {\omega} (x _ {i}) - Q _ {\omega} (x _ {i}, u _ {i})) ^ {2}\tag{17}
$$

ω is critic parameter.

The update rules for the actor and critic adjusting their respective parameters using gradient ascent in the actor as well as gradient descent in the critic and it is shown in Equation (19)

$$
\theta_ {t + 1} = \theta_ {t} + \alpha \nabla_ {\theta} J (\theta_ {t})\tag{18}
$$

$$
\omega_ {t} = \omega_ {t} - \beta \nabla_ {\omega} J (\omega_ {t})\tag{19}
$$

![](images/7a09b6920710e6d9670054642f15638dccb01ec6b05676eb29d53c8a7feff4f1.jpg)  
Schematic of TD3 algorithm.

## 4.2.4. Trust region policy optimisation (TRPO)

The policy optimisation algorithms include iteration, gradient and gradient-free. Gradient-free techniques such as covariance matrix adaptation and crossentropy are commonly utilised for problem-solving. Gradient-based optimisation techniques are efective for function approximations and supervised learning tasks. A new algorithm, TRPO (Schulman et al., 2015) is proposed to optimise parameterised policy π<sub>θ</sub> based on below expectations are shown in Equation (20).

$$
M a x \mathbb {E} _ {x \sim \rho \theta_ {o l d}} \left[ \frac {\pi (u | x ; \theta)}{\mu (u | x)} \right] _ {u \sim \mu} Q (x, u; \theta_ {o l d})\tag{20}
$$

subjected to,

$$
\mathbb {E} _ {x \sim \rho \theta_ {o l d}} [ D _ {K L} (\pi (\cdot | x; \theta_ {o l d}) | | \pi (\cdot | x; \theta)) ] \leq C\tag{21}
$$

where $\mu$ is sampling distribution, $D _ { K L }$ divergence between old and new policy C is a constant.

## 4.2.5. Proximal policy optimisation (PPO)

The PPO method (Schulman et al., 2017) is based on ANN function approximations which outperforms DQN vanilla policy gradient and TRPO in continuous control problems and data eficiency. The PPO algorithm suggests a new substitute goal with distorted probability ratios. It does this by updating policies over multiple periods of time using stochastic gradient ascent. It works better on continuous control tasks than other methods with diferent surrogate goals, and it’s like ACER on Atari games.

## 4.2.6. Soft actor critic (SAC)

DRL algorithms work well in area as complicated as robotic control and games. However, on-policy methods like TRPO, A3C and PPO face challenges in realworld implementation, Of-policy methods, on the other hand, work better but have problems with convergence and stability. DDPG, as proposed by Lillicrap et al. (2015), is a robust method for continuous domains but is highly susceptible to hyperparameter selection. Rawlik et al. (2012) and Fox et al. (2015) apply a maximum entropy framework to create a stable and eficient model-free DRL algorithm for continuous domain. By adding an entropy maximisation term to the traditional reward function, they improve robustness and exploration. The SAC algorithm, which leverages this maximum entropy principle, is an of-policy DRL method known for its excellent stability and sample eficiency. SAC outperforms the of-policy DDPG algorithm in complex tasks such as humanoid control, and also solves instability and complexity issues. Actor and critic updated equations are as follows in Equations (22) and (23).

$$
\theta = \theta - \lambda_ {\pi} \nabla_ {\theta} J _ {\pi} (\theta)\tag{22}
$$

$$
\omega^ {\prime} = \tau \omega - (1 - \tau) \omega^ {\prime}\tag{23}
$$

## 5. Selective RL applications in process control

RL is being utilised in the field of control because of its capacity to tackle or reduce the limitations of existing control techniques. RL has the ability to handle uncertainties that arise in processes, including inconsistencies in models, external disturbances and errors in measurements. By exposing agents to random environments and integrating many situations, RL can create a robust and dynamically adjustable optimum control method. This method is suitable for systems that have MIMO capabilities and exhibit nonlinear dynamics. RL can reduce the computational demands of MPC during real-time operations by ofloading optimisation-related calculations to precomputed ofline processes. This section examines how reinforcement learning (RL) is applied in various domains, with a particular focus on process control.

## 5.1. Industrial chemical process control

Industrial process control is the practice of monitoring, controlling and optimising various processes in manufacturing and production to ensure they operate eficiently, safely and within specified parameters. It involves the use of sensors, controllers and actuators to manage variables like temperature, pressure, flow and chemical composition, maintaining stability despite fluctuations and disturbances. With advancements in automation and digitalisation, modern process control systems increasingly incorporate technologies like AI, ML and RL, ofering greater precision, predictive capabilities and remote operation. RL is becoming vital in industrial chemical process control due to its ability to handle complex, dynamic systems that traditional methods struggle with. RL can learn optimal control strategies through experience, adapt to changing conditions and balance multiple objectives like eficiency and safety. It reduces the need for human intervention, optimises long-term outcomes and supports predictive maintenance, enhancing both process eficiency and safety.

## 5.1.1. RL act as PID controller in industrial process control

PID controllers are widely utilised in the process industry owing to their straightforwardness and capacity to adjust the parameters employing fundamental process information. Nevertheless, a significant number of these controllers demonstrate poor control loop performance as a result of fluctuations in process conditions and parametric variations. PID controllers are commonly employed in multivariable control systems, but their efectiveness can be compromised when the closed loop plant works over a large range or when parametric changes cause the system dynamics to change. RL methods are now being used to tune PID controllers. Using computer science and control theory, Dogru et al. (2022) look at how to tune PID systems in industrial processes. It suggests an RL implementation, figures out an initial step-response model, and adjusts the agent online to fit the real dynamics. The technique is put to the test on a small-scale multimodal tank system, and setpoint tracking and noise regulatory tests back to it up. The RL-TD3 method is used by Rajasekhar et al. (2023) to tune the PI controller parameters for a nonlinear three-tank hybrid system. These speeds up training and improves accuracy by removing the need to start from scratch and explore the parameters. Sammyak et al. (Mate et al., 2023) suggest using RL to tune multiple SISO PID controllers at the same time for nonlinear quadruple tank system. The trained RL AC agent manages for interactions between PID loops and learns sensitivity under diferent scenarios. This approach also allows tuning for systems with right half-plane zeros. Shuprajhaa et al. (2022) focussed on developing a modified PPO RL-based adaptive PID controller for controlling open-loop unstable processes. The RL agent, functioning as the supervisor, explores optimal gains for the PID controller to achieve desired servo and regulatory performance by incorporating adaptive modifications such as action repeat, revised reward function and early stopping criterion. The RL-PID eliminates the need for process modelling and controller tuning. Table 2 summarises important studies utilised RL in industrial process control.

## 5.1.2. RL act as a controller in industrial process control

RL shows potential as a viable alternative to conventional PI controllers in control systems. Design of

PI controllers depends on pre-established mathematical models, whereas RL-based controllers adjust to dynamic and uncertain contexts without the need of explicit modelling. The RL agents acquire the ability to reduce the discrepancy between desired setpoints and actual outputs through their interaction with the environment and the receipt of feedback through rewards. By employing RL, controllers are able to efectively manage action spaces that are both continuous and discrete, rendering them appropriate for a diverse array of control tasks. Moreover, RL’s capacity to adjust to evolving surroundings renders it a compelling option for intricate systems in which the dynamics are not completely understood or constantly altered. Yaoyao et al. (Bao et al., 2021) proposed an improved deep deterministic actor critic predictor for RL, focusing on fast and stable learning processes. The algorithm separates immediate reward from actionvalue function, providing reliable gradient information at early stages. Simulation results show it achieves more stable and faster learning than state-of-the-art DRL algorithms and ofers better performance for nonlinear processes. Yi et al. (Jiang et al., 2018) presented a model-free data-driven method for real-time control of the industrial flotation process, focusing on optimal selection of process control inputs and tracking of operational indices. The method uses interleaved learning, overcoming the challenge of establishing an accurate mathematical model and performs significantly better than standard policy and value iteration simulation experiments. This paper presents a novel approach combining inverse RL and multi-task learning for data-driven multi-mode control design. It demonstrates its efectiveness in a continuous control case using historical closed-loop data, ofering a promising solution for designing controllers for multi mode processes. Fu et al. (2020) present a deep RL (DRL) model for predicting denitrification eficiency in coal-fired power plants in China. The model, which combines a Long short-term memory (LSTM) model and the A3C, is used to control selective catalytic reduction denitrification eficiency in a domestic 1000 MW unit. The Organic Rankine Cycle (ORC) is a popular method for industrial heat recovery. However, traditional control methods struggle to adapt to varying operating conditions in smart manufacturing. Lin et al. (2024) proposed a Sim2Real transfer learning-based DRL control method for ORC superheat control, ofering a simple, feasible and user-friendly solution for energy system optimisation control. Experimental results show improved training speed and generalisation performance under multiple operating conditions. Table 3 summarises important studies that utilised RL in industrial process control.

## 5.2. Biochemical process control

Biochemical systems are integral to various industries, including pharmaceuticals, biotechnology, food production and environmental engineering. These systems involve complex interactions between biological components, such as enzymes, cells and microorganisms, to produce desired products or carry out specific processes. The inherent complexity of biochemical systems, combined with their sensitivity to environmental and operational conditions, present significant challenges in ensuring consistent performance and product quality. Process control refers to the methods and technologies used to monitor and adjust the variables within a biochemical process to maintain desired outcomes. In biochemical systems, these variables can include temperature, pH, dissolved oxygen levels, nutrient concentration and agitation speed among others. Given the delicate nature of biological reactions, maintaining these variables within optimal ranges is essential for the success of the process. Efective process control ensures that the biochemical system operates eficiently, safely and predictably which is vital for both industrial applications and research settings.

Ma et al. introduced a controller that utilises DRL with DDPG to control a non-linear semi-batch polymerisation reaction (Ma et al., 2019). This study discusses many adaptations for applying DRL to chemical process management. These adaptations include the Markov state assumption, action bounds and reward specification. The DRL controller is able to handle multiple inputs, non-linearities, significant time delays and noise tolerance. Li et al. (2023a) proposed an innovative goal-oriented MORL algorithm for solving multi-objective optimisation problems with multiple conflicting objectives in control problems. The algorithm uses adaptive thresholds, goal selection strategy and refines the reward function based on the chosen objective. Experimental results show the algorithm outperforms benchmark algorithms and is closer to the Pareto frontier of fermentation problems. Batch processes ofer flexibility, low capital and raw material costs and wider product range for valueadded products and chemicals. However, optimisation and control are challenging due to inherent nonlinearity, time-varying dynamics and batch-to-batch variations. Joshi et al. (2023) introduced stochastic actor-critic RL algorithm called Twin Actor SAC, by incorporating an ensemble of actors in a maximum entropy which enhances learning through enhanced exploration and proves efective in controlling batch transesterification. Table 4 presents a brief overview of notable studies that have utilised RL in the control of biological processes.

Literature survey of RL with respect to industrial process control.

<table><tr><td>S. No.</td><td>References</td><td>Agent</td><td>Environment</td><td>Highlights of the study</td></tr><tr><td>1</td><td>Spielberg et al. (2017)</td><td>DQN</td><td>SISO – paper machine, MIMO – distillation column</td><td>Properly formulated reward hypothesis functions can be used for industrial process control.Approach evaluated on SISO and MIMO and tested under various scenarios.</td></tr><tr><td>2</td><td>Bao et al. (2021)</td><td>MAGE-TD3</td><td>Generic nonlinear process</td><td>Separates immediate reward from action-value function for reliable gradient information.Develops an expected form of policy gradient based on the idea that the state follows the normal distribution.</td></tr><tr><td>3</td><td>Siraskar (2021)</td><td>DDPG</td><td>Thermal process control</td><td>MATLAB&#x27;s RL Toolbox is used to develop a valve controller.Introduces graded learning as a simplified, application-oriented adaptation of the more formal and algorithmic curriculum for RL.</td></tr><tr><td>4</td><td>Jiang et al. (2019)</td><td>AC algorithm</td><td>Flotation industrial process</td><td>Novel formulation for optimal process control input selection. Guarantees optimal tracking of operational indicesUses interleaved learning for real-time online computation of optimal control solution.</td></tr><tr><td>5</td><td>Jiang et al. (2018)</td><td>Data-driven OOC method based on RL Policy evaluationPolicy improvement</td><td>Flotation industrial process</td><td>New dual rate data-driven algorithm based on lifting technology and RL.Emulation experiments in hardware-in-the-loop system validate method effectiveness.</td></tr><tr><td>6</td><td>Chen et al. (2020)</td><td>AC algorithm</td><td>Goethite iron-removal process</td><td>A limited function is added to show how much the coherent system can change the subsystem cost function.A new neural network is added to the actor-critic structure. To get a better idea of part of the unknown system structure.The processes of strategy update and strategy iteration are done one after the other to find the best control strategies.</td></tr><tr><td>7</td><td>Fu and Zhang (2021)</td><td>A3C</td><td>Coal-fired powerplant</td><td>The DRL model for denitrification efficiency, a combination of LSTM and A3C algorithms, is used to predict and control SCR denitrification efficiency in coal-fired power plants.</td></tr><tr><td>8</td><td>Dogru et al. (2021b)</td><td>A2C</td><td>Hybrid tank system</td><td>RL worked well in real time, using meta-heuristic first principles model parameter optimisation, in-silico A3C/A-A2C policy learning, and online learning A2C with the best in-silico policy on the real process.The extent of exploration (EoE) is suggested as a way to measure how much the state/action area is explored.</td></tr><tr><td>9</td><td>Fujii et al. (2021)</td><td>AC-RL</td><td>MIMO industrial process</td><td>The self-tuning two-degree-of-freedom PI controller for thin film production uses RL and actor-critic algorithms, compensates for input coupling and lag and shows effectiveness compared to conventional static gain controllers.</td></tr><tr><td>10</td><td>Guan and Yamamoto (2021)</td><td>AC based PID control</td><td>Generic nonlinear process</td><td>A single radial basis function network is used to figure out the value function of critic and the control policy function of actor.Gradient descent method is used to yield updating rules based on the TD error performance index. AC Network weights and kernel function can be calculated adaptively.</td></tr><tr><td>11</td><td>Lawrence et al. (2022)</td><td>TD3</td><td>Two-Tank system</td><td>Implementation of a state-of-the-art deep RL algorithm on a real physical system using a PID controller is explored. Demonstrating its simplicity, hardware-free initialisation and deployment confidence.</td></tr><tr><td>12&#x27;</td><td>Li et al. (2024)</td><td>RL</td><td>Deep reactive ion etching process</td><td>Theoretical properties of RL-based controllers based on linear model assumptions is presented and compare their performance with traditional process controllers like exponentially weighted moving average and general harmonic rule for linear processes.</td></tr><tr><td>13</td><td>Patel (2023)</td><td>DDPG</td><td>Distillation column</td><td>A systematic method for formulating RL problems incorporating domain-specific knowledge about process constraints and objectives is presented.This method reduces dimensionality and modifications to the exploration process, enhancing safety, speed and explainability of online RL implementation.</td></tr><tr><td>14</td><td>Yifei and Lakshmi-narayanan (2023)</td><td>TD3</td><td>Distillation column process.</td><td>A unique RL agent configuration for multiloop control in a MARL control system is developed.The performance of the MARL system is weakly dependent on the reward function configuration for systems with weak to moderate loop interactions.</td></tr><tr><td>15</td><td>Deng et al. (2023)</td><td>Q-learning</td><td>steel-making industry, strip rolling manufacturing process</td><td>Ensemble Q-functions are incorporated into policy evaluation for steady performance. Behavior cloning is added to address distributional shifts.The method learns process dynamics from factory data to generate a control policy.</td></tr><tr><td>16</td><td>Rajasekhar et al. (2023)</td><td>TD3</td><td>Nonlinear three-tank hybrid (TTH) system</td><td>The TD3 is utilised to adjust the PI controller parameters for a nonlinear TTH system in a decentralised multi-agent manner.TD3 algorithm uses to tune PI controller parameters for a nonlinear TTH system, improving training speed and convergence accuracy by eliminating the need for parameter exploration from scratch.</td></tr></table>

## 5.3. RL in energy

RL holds great promise in improving sustainable energy by optimising chemical processes for the purpose of energy production, storage and conversion. Systematically testing various materials and operational situations can enhance catalyst design and process management in hydrogen production and carbon capture. RL plays a crucial role in energy storage systems by improving the charge–discharge cycles of batteries, increasing battery lifespan and eficiency and supporting sustainability. Additionally, it improves the efectiveness of electrolysis techniques used in the production of hydrogen, the capture of carbon and the conversion of waste into energy. RL has the ability to control factors such as nutrient availability, temperature and pH in order to optimise production and reduce negative efects on the environment. Additionally, it can be utilised in waste-to-energy processes, optimising energy recovery while reducing emissions and by-products. RL has the capability to integrate electrochemical processes with renewable energy sources, thus enabling chemical reactions to occur for the purpose of hydrogen production. This leads to a decrease in energy usage, reduces the amount of waste produced and improves the eficiency of energy conversion and storage technologies, thereby encouraging a transition to cleaner and more environmentally friendly energy systems. The use of RL in the energy business demonstrates its capacity to foster innovation and improve the eficiency of sustainable energy solutions, hence solving worldwide energy demands while minimising environmental impacts.

The Proton Exchange Membrane Fuel Cell (PEMFC) is a highly eficient and environmentally friendly energy generation method that directly converts hydrogen and oxygen into electricity. It requires a controller for each subsystem to ensure safe and eficient operation. Li et al. (2021) proposed a multiobjective FOPID controller to simultaneously control the oxygen excess ratio and output voltage. To enhance the controller’s robustness and control capability, they developed a large-scale deep reinforcement learning strategy known as the demonstration curriculum strategy large-scale multi-delay deep deterministic policy gradient (DCSLMD3PG). Simulation results demonstrate that this adaptive optimal FOPID algorithm achieves optimal control performance while satisfying PEMFC security constraints. Researchers worldwide have proposed several approaches and architectures to decrease energy consumption in heating, ventilation and air conditioning (HVAC) systems, which is another topic of interest in the energy sector. Traditional feedback control methods, such as on–of or PID control, may not account for external disturbances, resulting in less optimal performance. Recent advancements, like the autonomous adaptive MPC architecture and two-layer MPC methods, have efectively reduced operational costs and improved energy eficiency. Nonetheless, the efectiveness and reliability of these methods are contingent on the accuracy of the models and the robustness of online optimisation. To improve these aspects, Fu et al. (Fu & Zhang, 2021) introduced TD3-MPC, a control algorithm designed for energy consumption prediction in HVAC systems. This algorithm mitigates outdoor uncertainties and adapts to various indoor air capacity settings, enhancing prediction accuracy. Additionally, it supports ofpeak energy storage to reduce overall energy consumption costs.

The importance of energy eficiency in managing manufacturing businesses is significantly growing. The industrial sector accounts for almost 40% of worldwide energy consumption, mostly driven by manufacturing activities. At the machine level, there are two primary approaches to reduce the damage to the environment: energy-eficient scheduling (EES) and energyeficient control (EEC). The two strategies both try to solve the same problem, but they do so in diferent ways. Lofredo et al. (2023) suggested a new algorithm that would make single workstations in industrial systems that use RL techniques less harmful to the environment. The algorithm uses energy-eficient control steps that work well, showing that it is valid and useful in general and in the automotive industry. Experiments show that the RL agent quickly figures out the best policies. The approach’s potential and ease of use make it a straight and successful use in the industry. The growth of Industry 4.0 key enabling technologies has also helped this. Table 5 gives a short summary of some of the most important energy studies that has used RL.

Literature survey of RL with respect to bio chemical process control.

<table><tr><td>S. No.</td><td>Reference</td><td>Agent</td><td>Environment</td><td>Highlights of the study</td></tr><tr><td>1</td><td>Ma et al. (2019)</td><td>DDPG</td><td>polymerisation reaction system</td><td>· Idea of a Markov state, the limits of actions, and how rewards are defined is discussed.· Shows that it can handle difficult control tasks for chemical processes that have many inputs, nonlinear, have a large time delay, and can handle noise.</td></tr><tr><td>2</td><td>Dutta and Upreti (2023a)</td><td>DDPG</td><td>(1) Non-isothermal CSTR.(2) pH neutralisation process.(3) Non-isothermal process in a tubular reactor.</td><td>· Introduces a new AI strategy for process control, combining RL with a novel inverse model control approach is proposed.· The transformed inverse model provides baseline control, which is then improved by RL&#x27;s DDPG method in the TIM-RL controller.</td></tr><tr><td>3</td><td>Liu et al. (2023)</td><td>SAC</td><td>Two-dimensional iterative learning control system</td><td>· A two-dimensional iterative learning control-RL control scheme is proposed.· The DRL compensator counteracts the negative impact of model mismatch and non-repetitive nature.</td></tr><tr><td>4</td><td>Li et al. (2023a)</td><td>SAC</td><td>Fermentation process</td><td>· Utilises multi-objective RL (MORL) to tackle system complexity and preference determination.· Proposes a goal-oriented MORL algorithm for optimisation guided by adaptive thresholds and goal selection strategy.</td></tr><tr><td>5</td><td>Alhazmi and Sarathy (2023)</td><td>DDPG</td><td>Adiabatic packed-bed reactor</td><td>· Algorithm for nonintrusive, online, nonlinear parameter estimation of physical models using deep RL is proposed.· The RL problem is used to train a neural network for parameter estimation.</td></tr><tr><td>6</td><td>Joshi et al. (2023)</td><td>TASAC</td><td>Batch processes</td><td>· Twin actor soft actor-critic (TASAC) algorithm proposed for controlling continuous state and action spaces.· TASAC uses a group of actors in a maximum entropy framework to make exploration better.</td></tr><tr><td>7</td><td>Liu et al. (2024)</td><td>DQN</td><td>zinc electrowinning process (ZEP)</td><td>· An integrated optimal control method based on temporal causal network and RL is proposed. The method divides working conditions and uses a temporal causal network to estimate current efficiency.· An RL controller is established under each working condition and the optimal electrolyte temperature is placed into the controller&#x27;s reward function.</td></tr><tr><td>8</td><td>Sachio et al. (2022)</td><td>PG</td><td>1.Tank Design 2.CSTR</td><td>· A bilevel optimisation problem that addresses design and control simultaneously is proposed.· The optimal control is computed using RL and embedding it into the design problem.</td></tr><tr><td>9</td><td>Li et al. (2023a)</td><td>PPO</td><td>Fed-batch fermentation Process control</td><td>· Soft proximal policy optimisation algorithms combined with a hybrid weight-generation method are proposed for finding the Pareto front approximation of the fed-batch fermentation process.· The algorithm initially finds a single policy for the problem, then uses a hybrid weight-generation method to find Pareto optimal solutions.</td></tr><tr><td>10</td><td>Mowbray et al. (2022)</td><td>PPO</td><td>A microalgal lutein photo-production dynamic process</td><td>· Using Gaussian processes for offline simulation and posterior uncertainty prediction, the method accounts for joint chance constraints and plant-model mismatch.</td></tr><tr><td>11</td><td>Dutta and Upreti (2023b)</td><td>DDPG</td><td>Isothermal CSTR</td><td>· The novel process control strategy combines multiple neural networks (MNN) and RL to create an MNNRL controller, trained using optimal control and state data for predictive control actions.</td></tr><tr><td>12</td><td>Shuprajhaa et al. (2022)</td><td>m-PPO</td><td>CSTR</td><td>· A modified Proximal Policy Optimisation (m-PPO) RL based adaptive PID controller (RL-PID) for controlling open loop unstable processes is developed.· Adaptive modifications include action repeat, modified reward function and early stopping criterion.</td></tr><tr><td>13</td><td>Panjapornpon et al. (2022)</td><td>DDPG</td><td>pH process</td><td>· RL control with the DDPG algorithm is created to control both the pH and the level of the liquid in a mixed tank reactor.· Reward is made separately for controlling level and pH. A grid search method is used to find the best hyperparameters for RL controller models.</td></tr><tr><td>14</td><td>Rajasekhar et al. (2024)</td><td>TD3</td><td>Bioreactor</td><td>· TD3 algorithm to control of reactor temperature is categorised into unconstrained and constrained approaches.· TD3 with various reward functions tested on a nonlinear bioreactor model.</td></tr></table>

Continued.

<table><tr><td>S. No.</td><td>Reference</td><td>Agent</td><td>Environment</td><td>Highlights of the study</td></tr><tr><td>15</td><td>Gupta et al. (2024)</td><td>PPO</td><td>Bioreactor</td><td>The Twin Agent RL Framework integrates stochastic and deterministic agent&#x27;s actions in a multiagent system, actively monitoring output and using twin actor networks for action selection.</td></tr><tr><td>16</td><td>Gupta et al. (2023)</td><td>PPO</td><td>CSTR</td><td>A multi-actor proximal policy optimisation-based RL approach for controlling monoclonal antibodies (mAb) production is proposed.Multi-actor PPO outperforms other RL algorithms in terms of RMSE values and convergence performance.</td></tr><tr><td>17</td><td>Joshi et al. (2021)</td><td>TD3</td><td>Batch process control</td><td>Twin actor twin delayed deep deterministic policy gradient (TATD3) is proposed by incorporating twin actor networks in the existing TD3 algorithm.Two novel reward functions are proposed for TATD3 controller.</td></tr><tr><td>18</td><td>Nikita et al. (2021)</td><td>chromatography analysis and design toolkit (CADET) model</td><td>Chromatography process</td><td>A RL approach for cation exchange chromatography, focusing on maximisation problems and optimising process flowrates to achieve maximum yield and purity constraints is examined.</td></tr><tr><td>19</td><td>Rajasekhar et al. (2025)</td><td>DDPG, DQN</td><td>Fermented biorector</td><td>RL based control of fermented bioreactor focusing on DDPG and DQN based RL is studied.</td></tr></table>

## 5.4. RL in waste water treatment domain:

As the economy grows and people’s living circumstances improve, the release of industrial efluent and home sewage is steadily growing. To address the scarcity of urban water supplies and preserve the ecological environment, several wastewater treatment facilities have been developed in diferent countries. The activated sludge process is a commonly utilised technology in the majority of wastewater treatment plants. Although this technology is highly efective, the control problem of the wastewater treatment plant is often complex due to the frequent efects of biological, chemical and other processes. Dissolved oxygen (DO) is a key factor in wastewater treatment, afecting microorganism activity and denitrification rate. An optimal DO level is essential for eficient system functioning. Meeting strict regulations and implementing energy-saving measures are challenges faced by wastewater treatment plants, necessitating intelligent DO control for efective treatment and energy conservation. Table 6 summarises important studies utilised RL in waste water treatment plant sector.

## 5.5. RL in oil and gas sector

RL is crucial in the oil and gas industry for optimising drilling operations, enhancing production and managing assets more eficiently. It helps reduce costs by improving drilling eficiency, predicting maintenance needs and optimising supply chains. RL also supports energy eficiency and sustainability eforts such as reducing carbon emissions and improving enhanced oil recovery. Additionally, RL contributes to safety by detecting hazards and optimising crisis responses.

Open-pit ore extraction contributes to about 20% of total oil sands production. The process begins with mining, where the ore is crushed and transported for extraction. The crushed rock is mixed with chemicals and heat to make a slurry. This is then sent to the Primary Separation Vessel (PSV), which is a gravity separation vessel. There are three separate levels of slurry: the froth layer, the middling layer, and the tailings layer. The mostly bitumen-filled froth layer floats to the top to form the top layer and spills for more treatment. The thickest pieces in the tailings layer are taken out for more processing before being dumped into a tailings pond. The middle layer is made up of the rest of the makeup, which is mostly water. A stream of middlings is pumped from the middle of the tank to a second separation phase so that it can be treated even more. A very good PSV gets the most bitumen out of the water and solid bits while minimising the extra work that needs to be done on later separation steps. The best way to run a PSV helps to reach financial and environmental goals. Masliyah et al. (1984) used traditional PI controllers to solve a control problem for interface level and tailings density, but they didn’t think about bitumen recovery. Liu et al. (2015) applied an improved economic model predictive controller scheme to maximise bitumen recovery rate. Gilbert (2004) used of-line optimal input trajectories. Several things that can’t be controlled, like the grade of the ore, the rate of feed, and the distribution of particle sizes, can lead to uncertainty and disturbances. These can change the density of the tailings and the middlings layer, making it harder to recover bitumen, and changing the performance of the separation. To address these challenges, a model-free approach like

Literature survey of RL with respect to energy.

<table><tr><td>S. No.</td><td>Reference</td><td>Agent</td><td>Environment</td><td>Highlights of the study</td></tr><tr><td>1</td><td>Shi et al. (2020)</td><td>DDPG</td><td>Zinc electrowinning process</td><td rowspan="12">A model-free DDPG based controller to reduce energy consumption, overcome modelling errors and improve control periods and parameters is proposed.A hybrid approach using TD3-MPC is proposed.Sets the building&#x27;s temperature ahead of time during off-peak hours to reduce function approximation mistakesMulti-objective FOPID controller for high efficiency and control is proposed.DCSLMD3PG algorithm designed as controller tuner.Uses improved TD3 algorithm as a tuner.Algorithm adjusts controller coefficients in real time, accelerating learning speed.A model-free RL controller designed using LSTM networks for energy optimisation in buildings.The controller aims to optimise thermal comfort and energy consumption. The control scheme is implemented in MATLAB and EnergyPlus.A real-time battery energy storage control for residential houses with solar panels is studied, using a RL model to capture daily demand, electricity price and solar energy patterns.Q-learning algorithm outperforms the One-step Roll-out algorithm, with variations in electricity price affecting performance.Systematically investigates randomness, learning process, thermal comfort and energy consumption.Randomness significantly impacts initial performance and convergence speed.Model-free controller accumulates comprehensive rewards, determining convergence.A comprehensive implementation process for a novel radiant heating system in an office building, including building energy modelling, multi-objective building energy model calibration, DRL training and control deployment.A RL-based algorithm designed to optimise energy-efficient control strategies for a single workstation of identical parallel machines, targeting an ideal trade-off between system productivity and energy demand without needing comprehensive knowledge of the system dynamics.The model-based RL framework for zero energy house space heating uses short-period monitored data and rewards based on energy cost, PV self-consumption and thermal discomfort.Allows agents to interact with reduced-order thermodynamic model and uncertain environment.A supervisory control method combining DRL and PID for indirect-contact heat exchangers is proposed.Uses measurable variables as agent observations to describe heat transfer processes effectively, improving control efficiency under large disturbances.Intelligent battery energy storage control study utilises RL model for real-time battery energy storage control.Focuses on residential houses with solar panels and battery energy storage. Uniquely designed cyclic time-dependent Markov Process captures daily cyclic patterns in demand, electricity price and solar energy.</td></tr><tr><td>2</td><td>Fu and Zhang (2021)</td><td>TD3</td><td>HVAC systems</td></tr><tr><td>3</td><td>Li et al. (2021)</td><td>DCSL-MD3PG</td><td>Proton exchange membrane fuel cells</td></tr><tr><td>4</td><td>Li et al. (2021)</td><td>TD3</td><td>Proton exchange membrane fuel cell (PEMFC)</td></tr><tr><td>5</td><td>Wang et al. (2017)</td><td>DRL</td><td>HVAC system</td></tr><tr><td>6</td><td>Abedi et al. (2022)</td><td>Q-learning</td><td>Energy management system</td></tr><tr><td>7</td><td>Qin et al. (2023)</td><td>DQN</td><td>Heating, ventilation and air conditioning (HVAC) system</td></tr><tr><td>8</td><td>Zhang et al. (2019)</td><td>A2C/A3C</td><td>HVAC system</td></tr><tr><td>9</td><td>Loffredo et al. (2023)</td><td>DQN</td><td>Sustainability Manufacturing Automation system.</td></tr><tr><td>10</td><td>Li et al. (2023b)</td><td>Duelling-Double DQN</td><td>ZEH (zero energy house) space heating system</td></tr><tr><td>11</td><td>Wang et al. (2023)</td><td>DRL</td><td>Energy system control organic rankine cycle</td></tr><tr><td>12</td><td>Abedi et al. (2022)</td><td>Q-learning</td><td>Energy management system</td></tr></table>

RL is used, focusing on optimising bitumen recovery through a hierarchical structure and regulating tailings layer density to prevent sanding. Table 7 summarises important studies utilised RL in oil and gas sector.

## 6. Reinforcement learning challenges and proposed solutions

Rise of AI is becoming a strategic issue for many countries. The merging of advanced AI technologies with in-process control will be more closely intertwined.

Literature survey of RL with respect waste water treatment domain.

<table><tr><td>S. No.</td><td>Reference</td><td>Agent</td><td>Environment</td><td>Highlights of the study</td></tr><tr><td>1</td><td>Sea et al. (2021)</td><td>PPO</td><td>Wastewater treatment plants</td><td rowspan="2">A deep neural network model for wastewater inflow forecasting and a DRL agent to control pumping system signals is proposed.The DRL agents consider energy consumption and electricity price information and new features and penalty factors for pump switching.An intelligent control method for dissolved oxygen (DO) concentration in wastewater treatment processes using RL and DDPG algorithms.Proposed control method adjusts the DO concentration dynamically, aiming for energy-saving and emission reduction.</td></tr><tr><td>2</td><td>Du et al. (2023)</td><td>DDPG</td><td>Wastewater treatment process</td></tr><tr><td>3</td><td>Aponte-Rengifo et al. (2023)</td><td>Transfer RL</td><td>Wastewater treatment plant</td><td>The use of DRL and transfer learning (TL) to balance environmental impact and operating costs in wastewater treatment plants&#x27; activated sludge processes is proposed.The RL method can use the TL approach to cope with this inefficient and slow data-driven learning.</td></tr><tr><td>4</td><td>Yang et al. (2022)</td><td>Direct heuristic dynamic programming</td><td>Wastewater treatment process</td><td>A direct heuristic dynamic programming based RL control method for multivariable tracking control in WWTP, aiming to minimise tracking error and eliminate unknown disturbances.</td></tr><tr><td>5</td><td>Wang et al. (2023)</td><td>Tracking goal representation heuristic dynamic programming</td><td>Wastewater treatment process control</td><td>An intelligent control method using tracking goal representation heuristic dynamic programming, a model network, a goal network and a classical actor-critic scheme for optimal control strategy is adapted.</td></tr><tr><td>6</td><td>Lu et al. (2021)</td><td>RL-based particle swarm optimisation</td><td>Wastewater treatment</td><td>RL-based particle swarm optimisation aims to reduce energy consumption in activated sludge wastewater treatment by optimising control settings.Proposed algorithm uses valid history information to guide particle behavior, creating an elite network to predict particle velocity.</td></tr><tr><td>7</td><td>Yang et al. (2022)</td><td>Adaptive dynamic programming</td><td>wastewater treatment</td><td>A dynamic prioritised policy gradient adaptive dynamic programming method for optimal control of nonaffine nonlinear discrete-time systems is proposed.Proposed method uses a dynamic prioritised replay buffer and neural networks to track setpoints of wastewater treatment plants, alleviating disturbance effects without system modelling.</td></tr><tr><td>8</td><td>Yang et al. (2020)</td><td>SAC</td><td>Reservoirs</td><td>Two RL methods, DQN and the SAC algorithm are designed to learn and store historical experience, allowing for the development of appropriate control strategies based on reservoir characteristics.The trained models can learn and store historical experience, with an agreement between the optimal control strategy obtained by both algorithms and the global optimal strategy obtained by the exhaustive method.</td></tr></table>

Literature survey of RL with respect to oil and gas sector.

<table><tr><td>S. No.</td><td>Reference</td><td>Agent</td><td>Environment</td><td>Highlights of the study</td></tr><tr><td>1</td><td>Bangi et al. (2021)</td><td>DDPG</td><td>Hydraulic fracturing process</td><td>A model-free data-based RL controller, utilising the DDPG algorithm, actor-critic framework.Dimensionality reduction and transfer learning to optimise control policy through process interactions.</td></tr><tr><td>2</td><td>Ge et al. (2018)</td><td>TD3</td><td>approximate dynamic programming method</td><td>An optimal control method for alkali-surfactant-polymer flooding using approximate dynamic programming is proposed.Proposed method uses the net present value as a performance index, the AC algorithm, a linear approximation basis function, a temporal difference learning algorithm and a Gauss function to approximate control and value functions.</td></tr><tr><td>3</td><td>Li et al. (2019)</td><td>Approximated dynamic programming; actor-critic framework</td><td>The optimal control Model of ASP flooding</td><td>Proposed algorithm uses a linear basis function approximator, a basis function construction method and action weighting to approximate control actions.The gradient descent method updates the value function and strategy parameters, while eligibility trace accelerates convergence.</td></tr><tr><td>4</td><td>Shafi et al. (2020)</td><td>A3C</td><td>Primary Separation Vessel (PSV)</td><td>A two-level RL control structure for a Primary Separation Vessel is adapted.The structure uses an asynchronous advantage actor-critic agent to learn near-optimal control strategies, promoting stable state space exploration.</td></tr><tr><td>5</td><td>Dogru et al. (2021a)</td><td>TD3</td><td>Water based oils and separation process</td><td>RL agent for object tracking in chemical, petrochemical, metallurgical and oil industries is used.The agent uses less than 100 images and generates data without expert knowledge. It requires fewer parameters and is robust to environmental uncertainties.</td></tr></table>

Nevertheless, the study on the utilisation of RL in process control is primarily limited to laboratory experiments and is yet to be implemented in practical process control settings. There is still a long way to go between the usable and the well-used stages. The following are some challenges and proposed solutions.

## • Exploration vs. Exploitation Trade-Of

• Challenge: Striking the right balance between exploration (searching for new policies) (Kordabad et al., 2022; Ladosz et al., 2022) and exploitation (refining known good policies) is pivotal. Poor balance results in suboptimal performance or resource wastage.

• Proposed Solutions: Techniques like epsilongreedy, Bayesian methods, and intrinsic motivation are promising strategies (Chen et al., 2022; Verdier et al., 2019) to address this balance. These methods prevent agents from overexploring or converging too early to suboptimal solutions.

## • Curse of Dimensionality

• Challenge: RL’s computational burden incre ases significantly in large or continuous stateaction spaces, which limits scalability in complex applications like process control (Lin et al., 2021; Qu et al., 2020).

• Proposed Solutions: Function approximation (e.g. neural networks) (Xu et al., 2014), hierarchical RL (Pateria et al., 2021), and distributed RL (Hofman et al., 2020; Liang et al., 2018) are efective in managing this complexity by breaking tasks into manageable components or leveraging parallel computing.

## • Safety and Stability

• Challenge: Ensuring that RL agents operate safely and stably in critical applications like healthcare or autonomous systems is crucial (Kordabad et al., 2022; Ladosz et al., 2022). Unchecked behavior can lead to catastrophic consequences.

• Proposed Solutions: Approaches like Safe RL, Robust RL, and stabilising techniques (Liu & Wu, 2021; Pinto et al., 2017) (e.g. experience replay) integrate safety constraints and improve robustness during the training and operational phases.

## • Reward Function Design

• Challenge: Poorly defined reward functions (Laud, 2004; Marom & Rosman, 2018) can lead to unintended behavior, including incentive hacking.

• Proposed Solutions: Inverse RL (Arora & Doshi, 2021; Hadfield-Menell et al., 2016), multi-objective RL (Mossalam et al., 2016; Zou et al., 2021), and reward engineering ensure that agents are guided efectively and aligned with desired outcomes, reducing the risk of undesired or unethical behaviors.

## • Sample Eficiency

• Challenge: Training RL models often requires a significant number of interactions, which can be expensive or impractical in real-world process control (Yarats et al., 2021; Yu, 2018).

Proposed Solutions: Model-based RL (Polydoros & Nalpantidis, 2017; Wang et al., 2019), of-policy learning (Geist & Scherrer, 2014; Maei et al., 2010), and data augmentation techniques enhance sample eficiency, reducing reliance on extensive real-world interactions.

## • Delayed or Sparse Rewards

• Challenge: Long-term credit assignment is difficult in environments where rewards are infrequent or delayed, as in process control systems with prolonged feedback loops.

• Proposed Solutions: Techniques such as rew ard shaping, hierarchical RL, and curiositydriven learning improve learning eficiency (Devidze et al., 2022; Marthi, 2007) and enable RL agents to discover optimal policies in such settings.

## • Ethical and Bias Concerns

• Challenge: RL systems may unintentionally perpetuate biases or make unethical decisions, particularly in sensitive areas like healthcare or finance.

• Proposed Solutions: Bias detection (Smith et al., 2023), adherence to ethical guidelines, and explainability techniques can ensure RL systems are fair, accountable, and transparent.

## • Integration of RL and RNNs

• Challenge (Nonlinearity and Partial Observability): RL struggles with incomplete state information in nonlinear dynamic systems (Wang et al., 2024), limiting its efectiveness in process control.

• Proposed Solution: Integrate RNNs into RL frameworks to estimate hidden states by learning temporal patterns and dynamics (Gu et al., 2021), enabling efective control in partially observable environments.

• Challenge (Capturing Long-Term Dependencies): Process control problems often involve delayed system responses and multi-step operations (Yoo et al., 2021), which RL alone cannot eficiently handle.

• Proposed Solutions: Develop hybrid RL-RNN architectures that explicitly model long-term temporal dependencies to improve decisionmaking strategies in multi-step processes (Schmidhuber, 2015).

• Challenge (Adaptive Control in Changing Environments): RL policies often fail to adapt to evolving process dynamics caused by disturbances (Hafner & Riedmiller, 2011), equipment aging, or other changes.

• Proposed Solutions: Utilise RNNs to track and encode dynamic changes over time (Du et al., 2024), enable real-time updates to RL policies for adaptive control in non-stationary environments.

• Challenge (Generalisation Across Operating Conditions): RL policies frequently overfit to specific operating points (Whiteson et al., 2011), reducing their efectiveness across varying conditions.

• Proposed Solutions: Employ RNNs to genera lise RL policies by capturing and encoding temporal variations in operating conditions (Luo et al., 2024), improving robustness across a range of scenarios.

## 7. Conclusions

RL is being investigated in the field of process control. However, there are still notable challenges and a noticeable disparity between theoretical progress and practical implementation. Although RL is new, it is important to evaluate the advantages and limitations of using it. RL is not intended to replace model-based approaches but rather functions as a viable substitute for specific tasks. Precisely identifying the appropriate application scenarios for RL is crucial for its eficient utilisation. This review paper examines the evolution, paradigms, algorithms and real-world implementations of RL, with a specific emphasis on its significance in process control. It examines traditional RL approaches, such as MDPs and its variations, as well as more advanced techniques in DRL. Classical RL ofers systematic methods for representing decision-making situations, but DRL has enhanced its versatility and eficiency by including deep learning methods. DRL techniques, such as value function-based deep RL and policy-based deep RL, have proven to be efective in tackling complicated and high-dimensional tasks that were previously considered too challenging for conventional RL methods. The versatility and significance of RL are demonstrated in a range of industries, including industrial chemical process control, biochemical process control, the energy industry, wastewater treatment and the oil and gas sector. Nevertheless, the research also emphasises the dificulties encountered in reinforcement learning, including the need for significant computer resources, the intricacy of modelling and the necessity of maintaining stability and robustness in dynamic situations. Although facing these dificulties, RL continues to be at the forefront of intelligent control systems, providing robust tools for a diverse array of applications. Additional research is required to address present challenges and fully utilise the capabilities of RL.

## Acknowledgement

The authors thank the computer support group (CSG) of NIT-Tiruchirappalli, for providing the resources.

## Disclosure statement

No potential conflict of interest was reported by the author(s).

## Authors contributions

N. Rajasekhar: Conceptualisation, Data curation, Formal analysis, Investigation, Methodology, Validation, Visualisation, Software, Writing – original draft. T. K. Radhakrishnan: Conceptualisation, Formal analysis, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualisation, writing – original draft, Writing – review & editing. N. Samsudeen: Conceptualisation, Formal analysis, Investigation, Methodology, Project administration, Supervision, Validation, Visualisation, Writing – review & editing.

## References

Abedi, S., Yoon, S. W., & Kwon, S. (2022). Battery energy storage control using a reinforcement learning approach with cyclic time-dependent Markov Process. International Journal of Electrical Power & Energy Systems, 134 (1), 107368–107384. https://doi.org/10.1016/j.ijepes.2021. 107368

Alhazmi, K., & Sarathy, S. M. (2023). Nonintrusive parameter adaptation of chemical process models with reinforcement learning. Journal of Process Control, 123, 87–95. https://doi.org/10.1016/j.jprocont.2023.02.001

Andreae, J. H. (1963). STELLA: A scheme for a learning machine. IFAC Proceedings Volumes, 1(2), 497–502. https://doi.org/10.1016/s1474-6670(17)69682-4

Anschel, O., Baram, N., & Shimkin, N. (2017). Averaged-DQN: Variance reduction and stabilization for deep reinforcement learning. International Conference on Machine Learning (ICML).

Aponte-Rengifo, O., Francisco, M., Vilanova, R., Vega, P., & Revollar, S. (2023). Intelligent control of wastewater treatment plants based on model-free deep reinforcement learning. Processes, 11(8), 2269–2284. https://doi.org/10.3390/ pr11082269

Arora, S., & Doshi, P. (2021). A survey of inverse reinforcement learning: Challenges, methods and progress. Artificial Intelligence, 297(8), 103500. https://doi.org/10.1016/j.artint.2021. 103500

Bangi, M. S. F., & Kwon, J. S. (2021). Deep reinforcement learning control of hydraulic fracturing. Computers & Chemical Engineering, 154(11), 107489 –107500. https://doi.org/10. 1016/j.compchemeng.2021.107489

Bao, Y., Zhu, Y., & Qian, F. (2021). A deep reinforcement learning approach to improve the learning performance in process control. Industrial & Engineering Chemistry Research, 60(15), 5504–5515. https://doi.org/10.1021/acs.iecr.0c05678

Begum, K. G., Rao, A. S., & Radhakrishnan, T. K. (2016). Maximum sensitivity based analytical tuning rules for PID controllers for unstable dead time processes. Chemical Engineering Research and Design, 109, 593–606. https://doi.org/10. 1016/j.cherd.2016.03.003

Bhatnagar, S., Ghavamzadeh, M., Lee, M., & Sutton, R. S. (2007). Incremental natural actor-critic algorithms. Advances in neural Information processing systems.

Chen, N., Luo, S., Dai, J., Luo, B., & Gui, W. (2020). Optimal control of iron-removal systems based on of-policy reinforcement learning. IEEE Access, 8, 149730–149740. https://doi.org/10.1109/ACCESS.2020.3015801

Chen, X., Qu, G., Tang, Y., Low, S., & Li, N. (2022). Reinforcement learning for selective key applications in power systems: Recent advances and future challenges. IEEE Transactions on Smart Grid, 13(4), 2935–2958. https://doi.org/10. 1109/TSG.2022.3154718

Clark, W. A., & Farley, B. G. (1955). Generalization of pattern recognition in a self-organizing system. Proceedings of the Western Joint Computer Conference, AFIPS (pp. 86–91). https://doi.org/10.1145/1455292.1455309

Coughanowr, D. R., & LeBlanc, S. E. (2009). Process systems analysis and control. McGraw-Hill. 630

Darby, M. L., & Nikolaou, M. (2012). MPC: Current practice and challenges. Control Engineering Practice, 20(4), 328–342. https://doi.org/10.1016/j.conengprac.2011.12.004

Deng, J., Sierla, S., Sun, J., & Vyatkin, V. (2023). Ofline reinforcement learning for industrial process control: A case study from steel industry. Information Sciences, 632, 221–231. https://doi.org/10.1016/j.ins.2023.03.019

Devidze, R., Kamalaruban, P., & Singla, A. (2022). Exploration-Guided reward shaping for reinforcement learning under sparse rewards. Advances in neural Information processing systems.

Dogru, O., Velswamy, K., & Huang, B. (2021a). Actor–critic reinforcement learning and application in developing compu ter-vision-based interface tracking. Engineering, 7(9), 1248– 1261. https://doi.org/10.1016/j.eng.2021.04.027

Dogru, O., Velswamy, K., Ibrahim, F., Wu, Y., Sundaramoorthy, A. S., Huang, B., Xu, S., Nixon, M., & Bell, N. (2022). Reinforcement learning approach to autonomous PID tuning. Computers & Chemical Engineering, 161, 107760. https://doi.org/10.1016/j.compchemeng.2022.107760

Dogru, O., Wieczorek, N., Velswamy, K., Ibrahim, F., & Huang, B. (2021b). Online reinforcement learning for a continuous space system with experimental validation. Journal of Process Control, 104, 86–100. https://doi.org/10.1016/j.jprocont. 2021.06.004

Du, S. L., Chen, P. X., Han, H. G., & Qiao, J. F. (2023). Dissolved oxygen concentration control in wastewater treatment process based on reinforcement learning. Science China Technological Sciences, 66(9), 2549–2560. https://doi.org/10.1007 s11431-022-2403-8

Du, J., Zhang, C., Tang, T., & Qu, W. (2024). Learning-based transport control adapted to non-stationarity for real-time communication. In Proceedings of the IEEE International Workshop on Quality of Service, IWQoS; Institute of Electrical and Electronics Engineers Inc.

Dutta, D., & Upreti, S. R. (2023a). A multiple neural network and reinforcement learning-based strategy for process control. Journal of Process Control, 121, 103–118. https://doi.org/10.1016/j.jprocont.2022.12.004

Dutta, D., & Upreti, S. R. (2023b). A reinforcement learning based transformed inverse model strategy for nonlinear process control. Computers & Chemical Engineering, 178, 108386–108403. https://doi.org/10.1016/j.compchemeng. 2023.108386

Fox, R., Pakman, A., & Tishby, N. (2015). Taming the noise in reinforcement learning via soft updates. In Proceedings of the 32nd Conference on Uncertainty in Artificial Intelligence (pp. 202–211).

Fu, J., Xiao, H., Wang, H., & Zhou, J. (2020). Control strategy for denitrification eficiency of coal-fired power plant based on deep reinforcement learning. IEEE Access, 8, 65127–65136. https://doi.org/10.1109/ACCESS.2020.2985233

Fu, C., & Zhang, Y. (2021). Research and application of predictive control method based on deep reinforcement

learning for HVAC systems. IEEE Access, 9, 130845–130852. https://doi.org/10.1109/ACCESS.2021.3114161

Fujii, F., Kaneishi, A., Nii, T., Maenishi, R., & Tanaka, S. (2021). Self-Tuning two degree-of-freedom proportional–integral control system based on reinforcement learning for amultiple input multiple-output industrial process that sufers from spatial input coupling. Processes, 9(3), 487–504. https://doi. org/10.3390/pr9030487

Fujimoto, S., Van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. In Proceedings of the 35th International Conference on Machine Learning (pp. 2587–2601).

Ge, Y., Li, S., & Chang, P. (2018). An approximate dynamic programming method for the optimal control of alkaisurfactant-polymer flooding. Journal of Process Control, 64, 15–26. https://doi.org/10.1016/j.jprocont.2018.01.010

Geist, M., & Scherrer, B. (2014). Of-Policy learning with eligibility traces: A survey. Journal of Machine Learning Research, 15(1), 289–333.

Ghousiya Begum, K., Seshagiri Rao, A., & Radhakrishnan, T. K. (2017). Enhanced IMC based PID controller design for non-minimum phase (NMP) integrating processes with time delays. ISA Transactions, 68, 223–234. https://doi.org/10.1016/j.isatra.2017.03.005

Gilbert, W. A. (2004). Dynamic simulation and optimal trajectory planning for an oilsand primary separation vessel. University of Alberta.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT Press.

Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., & Re, C. (2021). Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems.

Guan, Z., & Yamamoto, T. (2021). Design of a reinforcement learning PID controller. IEEJ Transactions on Electrical and Electronic Engineering, 16(10), 1354–1360. https://doi.org/ 10.1002/tee.23430

Gupta, N., Anand, S., Joshi, T., Kumar, D., Ramteke, M., & Kodamana, H. (2023). Process control of MAb production using multi-actor proximal policy optimization. Digital Chemical Engineering, 8(3), 100108–100117. https://doi.org/ 10.1016/j.dche.2023.100108

Gupta, N., Anand, S., Kumar, D., Ramteke, M., Kandath, H., & Kodamana, H. (2024). A twin agent reinforcement learning framework by integrating deterministic and stochastic policies. Industrial & Engineering Chemistry Research, 63(24), 10692–10703. https://doi.org/10.1021/acs.iecr.4c00701

Hadfield-Menell, D., Russell, S. J., Abbeel, P., & Dragan, A. (2016). Cooperative inverse reinforcement learning, in Proceeding Advances in Neural Informatoin Processing Systems (pp. 3909–3917).

Hadian, M., Ramezani, A., & Zhang, W. (2021). Robust model predictive controller using recurrent neural networks for input–output linear parameter varying systems. Electronics, 10, 1557–1575. https://doi.org/10.3390/electronics10131 557

Hafner, R., & Riedmiller, M. (2011). Reinforcement learning in feedback control: Challenges and benchmarks from tech nical process control. Machine Learning, 84(1-2), 137–169. https://doi.org/10.1007/s10994-011-5235-x

Hasselt, H. V. (2010). Double Q-learning. In Proceedings of the Advances in Neural Information Processing Systems, 23(1), 2613–2621.

Haykin, S. S. (2016). Neural networks and learning machines. Pearson.

Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., & Silver, D. (2018). Rainbow: Combining improvements in deep reinforcement learning. In 32nd AAAI Conference on Artificial Intelligence (AAAI) (pp. 3215–3222).

Hofman, M. W., Shahriari, B., Aslanides, J., Barth-Maron, G., Momchev, N., Sinopalnikov, D., Stańczyk, P., Ramos, S., Raichuk, A., Vincent, D., Hussenot, L., Dadashi, R., Dulac-Arnold, G., Orsini, M., Jacq, A, Ferret, J., Viellard, N., Ghasemipour, S. K. S., Girgin, S., . . . Freitas, N. (2020). Acme: A research framework for distributed reinforcement learning. arXiv preprint arXiv:2006.00979.

Hu, C., Wei, X., & Ren, Y. (2019). Passive fault-tolerant control based on weighted LPV tube-MPC for air-breathing hypersonic vehicles. International Journal of Control, Automation and Systems, 17(8), 1957–1970. https://doi.org/10.1007/s12 555-018-0594-8

Jiang, Y., Fan, J., Chai, T., & Lewis, F. L. (2019). Dual-Rate operational optimal control for flotation industrial process with unknown operational model. IEEE Transactions on Industrial Electronics, 66(6), 4587–4599. https://doi.org/10.1109/ TIE.2018.2856198

Jiang, Y., Fan, J., Chai, T., Li, J., & Lewis, F. L. (2018). Data-Driven flotation industrial process operational optimal control based on reinforcement learning. IEEE Transactions on Industrial Informatics, 14(5), 1974–1989. https://doi.org/10. 1109/TII.2017.2761852

Joshi, T., Kodamana, H., Kandath, H., & Kaisare, N. (2023). TASAC: A twin-actor reinforcement learning framework with a stochastic policy with an application to batch process control. Control Engineering Practice, 134. https://doi.org 10.1016/j.conengprac.2023.105462

Joshi, T., Makker, S., Kodamana, H., & Kandath, H. (2021). Twin actor twin delayed deep deterministic policy gradient (TATD3) learning for batch process control. https://doi.org/ 10.1016/j.compchemeng.2021.107527

Klopf, A. H. (1974). Brain function and adaptive systems - a heterostatic theory (Technical Report AFCRL-72-0164). Bedford, MA: Air Force Cambridge Research Laboratories.

Klopf, A. H. (1975). A comparison of natural and artificial intelligence. ACM SIGART Bulletin, (52), 11–13.

Konda, V. R., & Tsitsiklis, J. N. (2003). On actor-critic algorithms. SIAM Journal on Control and Optimization, 42(4), 1143–1166. https://doi.org/10.1137/S0363012901385691

Kordabad, A. B., Wisniewski, R., & Gros, S. (2022). Safe reinforcement learning using wasserstein distributionally robust MPC and chance constraint. IEEE Access, 10,

130058–130067. https://doi.org/10.1109/ACCESS.2022.32 28922

Krose, B. J. A. (1995). Learning from delayed rewards. Robotics and Autonomous Systems, 15(4), 233–235. https://doi.org/10. 1016/0921-8890(95)00026-C

Ladosz, P., Weng, L., Kim, M., & Oh, H. (2022). Exploration in deep reinforcement learning: A survey. Information Fusion, 85, 1–22. https://doi.org/10.1016/j.infus.2022.03.003

Laud, A. D. (2004). Theory and application of reward shaping in reinforcement learning [Doctoral dissertation, Department of Computer Science, Univ. Illinois] Champaign, IL, USA.

Lawrence, N. P., Forbes, M. G., Loewen, P. D., McClement, D. G., Backström, J. U., & Gopaluni, R. B. (2022). Deep reinforcement learning with shallow controllers: An experimental application to PID tuning. Control Engineering Practice, 121, 105046–105060. https://doi.org/10.1016/j.conengprac. 2021.105046

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444. https://doi.org/10.1038/nature 14539

Li, Y., Du, J., & Jiang, W. (2024). Reinforcement learning for process control with application in semiconductor manufacturing. IISE Transactions, 56(6), 585–599. https://doi.org/10.1080/24725854.2023.2219290

Li, J., Geng, J., & Yu, T. (2021). Multi-Objective optimal control for proton exchange membrane fuel cell via large-scale deep reinforcement learning. Energy Reports, 7, 6422–6437. https://doi.org/10.1016/j.egyr.2021.07.067

Li, D., Gu, W., & Song, T. (2023a). Multi-Objective reinforcement learning in process control: A goal-oriented approach with adaptive thresholds. Journal of Process Control, 129, 103063–103078. https://doi.org/10.1016/j.jprocont.2023. 103063

Li, S., Han, L., Ge, Y., & Shi, Y. (2019). A new approximate dynamic programming algorithm based on an actor–critic framework for optimal control of alkali–surfactant–polymer flooding. Engineering Optimization, 51(12), 2147–2168. https://doi.org/10.1080/0305215X.2019.1570180

Li, J., Li, Y., & Yu, T. (2021). Temperature control of proton exchange membrane fuel cell based on machine learning. Frontiers in Energy Research, 9, 763099–763105. https://doi. org/10.3389/fenrg.2021.763099

Li, Y., Wang, Z., Xu, W., Gao, W., Xu, Y., & Xiao, F. (2023b). Modeling and energy dynamic control for a ZEH via hybrid model-based deep reinforcement learning. Energy, 277. https://doi.org/10.1016/j.energy.2023.127627

Liang, E., Liaw, R., Nishihara, R., Moritz, P., Fox, R., Goldberg, K., Gonzalez, J., Jordan, M., & Stoica, I. (2018). RLlib, abstractions for distributed reinforcement learning. International Conference on Machine Learning (ICML).

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015). Continuous control with deep reinforcement learning. In Proceedings of the 4th International Conference on Learning Representations ICLR (pp. 1–14).

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with

deep reinforcement learning. 4th International Conference on Learning Representations, ICLR 2016 - Conference Track Proceedings.

Lin, R., Luo, Y., Wu, X., Chen, J., Huang, B., Su, H., & Xie, L. (2024). Surrogate empowered Sim2Real transfer of deep reinforcement learning for ORC superheat control. Applied Energy, 356(6), https://doi.org/10.1016/j.apenergy.2023. 122310

Lin, Y., Qu, G., Huang, L., & Wierman, A. (2021). Multi Agent reinforcement learning in stochastic networked systems. Advances in Neural Information Processing Systems, 6(34), 7825–7837.

Liu, H., & Wu, W. (2021). Two-stage deep reinforcement learning for inverter-based volt-VAR control in active distribution networks. IEEE Transactions on Smart Grid, 12(3), 2037–2047. https://doi.org/10.1109/TSG.2020.3041620

Liu, T., Yang, C., Zhou, C., Li, Y., & Sun, B. (2024). Integrated optimal control for electrolyte temperature with temporal causal network and reinforcement learning. IEEE Transactions on Neural Networks and Learning Systems, 35(5), 5929–5941. https://doi.org/10.1109/TNNLS.2023.3278729

Liu, S., Zhang, J., & Liu, J. (2015). Economic MPC with terminal cost and application to oilsand separation. IFAC-PapersOnLine, 48(8), 20–25. https://doi.org/10.1016/j.ifacol. 2015.08.151

Liu, J., Zhou, Z., Hong, W., & Shi, J. (2023). Two-dimensional iterative learning control with deep reinforcement learning compensation for the non-repetitive uncertain batch processes. Journal of Process Control, 131(11), 103106–103117. https://doi.org/10.1016/j.jprocont.2023.103106

Lofredo, A., May, M. C., Schäfer, L., Matta, A., & Lanza, G. (2023). Reinforcement learning for energy-eficient control of parallel and identical machines. CIRP Journal of Manufacturing Science and Technology, 44, 91–103. https://doi.org/10.1016/j.cirpj.2023.05.007

Lu, L., Zheng, H., Jie, J., Zhang, M., & Dai, R. (2021). Reinforcement learning-based particle swarm optimization for sewage treatment control. Complex & Intelligent Systems, 7(5), 2199–2210. https://doi.org/10.1007/s40747-021-00395-w

Luo, F.-M., Tu, Z., Huang, Z., & Yu, Y. (2024). Eficient recurrent of-policy RL requires a context-encoder-specific learning rate. arXiv preprint arXiv:2405.15384.

Ma, Y., Zhu, W., Benton, M. G., & Romagnoli, J. (2019). Con tinuous control of a polymerization system with deep reinforcement learning. Journal of Process Control, 75, 40–47. https://doi.org/10.1016/j.jprocont.2018.11.004

Maei, H. R., Szepesvári, C., Bhatnagar, S., & Sutton, R. S. (2010). Toward of-policy learning control with function approximation. Proceedings of the 27th International Conference on Machine Learning.

Marom, O., & Rosman, B. (2018). Belief reward shaping in reinforcement learning. Proceedings of the Conference on Association for the Advancement of Artificial Intelligence.

Marthi, B. (2007). Automatic shaping and decomposition of reward functions. Proceedings of the 24th International Conference on Machine Learning, 4(1), 601–608. https://doi.org/10.1145/1273496.1273572

Masliyah, J. H., Cluett, W., Oxenford, J., & Tipman, R. (1984). Dynamic simulation of a gravity separation vessel. Proceedings of the Soc of Mining Engineers of AIME, 35(12), 145– 151.

Mata, S., Zubizarreta, A., & Pinto, C. (2019). Robust tubebased model predictive control for lateral path tracking. IEEE Transactions on Intelligent Vehicles, 4(4), 569–577. https://doi.org/10.1109/TIV.2019.2938102

Mate, S., Pal, P., Jaiswal, A., & Bhartiya, S. (2023). Simultaneous tuning of multiple PID controllers for multivariable systems using deep reinforcement learning. Digital Chemical Engineering, 9, 100131–100143. https://doi.org/10.1016/j.dche. 2023.100131

Mesbah, A. (2016). Stochastic model predictive control: An overview and perspectives for future research. IEEE Control Systems, 36(6), 30–44. https://doi.org/10.1109/MCS.2016. 2602087

Michie, D. (1963). Experiments on the mechanization of game-learning part I. Characterization of the model and its parameters. The Computer Journal, 6(3), 232–236. https://doi.org/10.1093/comjnl/6.3.232

Michie, D., & Chambers, R. A. (1968). BOXES: An experiment in adaptive control. Machine Intelligence 2(2), 137–152.

Minsky, M. (1961). Steps toward artificial intelligence. Proceedings of the IRE, 49(1), 8–30. https://doi.org/10.1109/JRPR OC.1961.287775

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with deep reinforcement learning. arXiv preprint arXiv:1312. 5602, 1–9.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236

Morato, M. M., Normey-Rico, J. E., & Sename, O. (2020). Model predictive control design for linear parameter varying systems: A survey. Annual Reviews in Control, 49, 64–80. https://doi.org/10.1016/j.arcontrol.2020.04.016

Mossalam, H., Assael, Y. M., Roijers, D. M., & Whiteson, S. (2016). Multi-objective deep reinforcement learning. arXiv preprint arXiv:1610.02707.

Mowbray, M., Petsagkourakis, P., del Rio-Chanona, E. A., & Zhang, D. (2022). Safe chance constrained reinforcement learning for batch process control. Computers & Chemical Engineering, 157, 107630–107648. https://doi.org/10.1016/ j.compchemeng.2021.107630

Negenborn, R. R., & Maestre, J. M. (2014). Distributed model predictive control: An overview and roadmap of future research opportunities. IEEE Control Systems, 34(4), 87–97. https://doi.org/10.1109/MCS.2014.2320397

Nian, R., Liu, J., & Huang, B. (2020). A review on reinforcement learning: Introduction and applications in industrial process control. Computers & Chemical Engineering, 139, 106886. https://doi.org/10.1016/j.compchemeng.2020.106886

Nikita, S., Tiwari, A., Sonawat, D., Kodamana, H., & Rathore, A. S. (2021). Reinforcement learning based optimization of process chromatography for continuous processing of biopharmaceuticals. Chemical Engineering Science, 230, 116171–1161719. https://doi.org/10.1016/j.ces.2020.116171

Panjapornpon, C., Chinchalongporn, P., Bardeeniz, S., Makka yatorn, R., & Wongpunnawat, W. (2022). Reinforcement learning control with deep deterministic policy gradient algorithm for multivariable PH process. Processes, 10(12), 116171–116179. https://doi.org/10.3390/pr10122514

Pannocchia, G. (2015). Ofset-free tracking MPC: A tutorial review and comparison of diferent formulations. European control conference (ECC).

Patel, K. M. (2023). A practical reinforcement learning implementation approach for continuous process control. Computers & Chemical Engineering, 174, 108232–108252. https://doi.org/10.1016/ j.compchemeng.2023.108232

Pateria, S., Subagdja, B., Tan, A. H., & Quek, C. (2021). Hierarchical reinforcement learning: A comprehensive survey. Acm Computing Surveys, 54(5), 1–35.

Peters, J., & Schaal, S. (2008). Natural actor-critic. Neurocomputing, 71(7-9), 1180–1190. https://doi.org/10.1016/j.neu com.2007.11.026

Pinto, L., Davidson, J., Sukthankar, R., & Gupta, A. (2017). Robust adversarial reinforcement learning. International Conference on Machine Learning, 70(1), 2817–2826.

Polydoros, A. S., & Nalpantidis, L. (2017). Survey of modelbased reinforcement learning: Applications on robotics. Journal of Intelligent and Robotic Systems: Theory and Applications, 86(2), 153–173. https://doi.org/10.1007/s10846-017- 0468

Puterman, M. L. (1994). Markov decision processes: Discrete stochastic dynamic programming. John Wiley & Sons.

Qin, H., Yu, Z., Li, T., Liu, X., & Li, L. (2023). Energy-eficient heating control for nearly zero energy residential buildings with deep reinforcement learning. Energy, 264(10), 126209–126221. https://doi.org/10.1016/j.energy.2022. 126209

Qu, G., Wierman, A., & Li, N. (2020). Scalable reinforcement learning of localized policies for multi-agent networked systems. Learning for Dynamics and Control, 120(1), 256–266.

Rajasekhar, N., Nagappan, K. K., Radhakrishnan, T. K., & Samsudeen, N. (2024a). Efective MPC strategies using deep learning methods for control of nonlinear system. International Journal of Dynamics and Control, 12(10), 3694–3707. https://doi.org/10.1007/s40435-024-01426-3

Rajasekhar, N., Nagappan, K. K., Radhakrishnan, T. K., & Samsudeen, N. (2024b). Application of recurrent neural networks for modeling and control of a quadruple-tank system. Advanced Control for Applications: Engineering and Industrial Systems, 6(2), 1–17. https://doi.org/10.1002/adc2.158

Rajasekhar, N., Radhakrishnan, T. K., & Mohamed, S. N. (2024). Reinforcement learning based temperature control of a fermentation bioreactor for ethanol production. Biotechnology and Bioengineering, 121(10), 3114–3127. https://doi. org/10.1002/bit.28784

Rajasekhar, N., Radhakrishnan, T. K., & Samsudeen, N. (2023). Decentralized multi-agent control of a three-tank hybrid system based on twin delayed deep deterministic policy gradient reinforcement learning algorithm. International Journal of Dynamics and Control, 12(4), 1098–1115. https://doi.org/10.1007/s40435-023-01227-0

Rajasekhar, N., Radhakrishnan, T. K., & Samsudeen, N. (2025). Deep deterministic policy gradient reinforcement learning based temperature control of a fermentation bioreactor for ethanol production. Journal of the Indian Chemical Society, 102(2), 101575. https://doi.org/10.1016/j.jics.2025.101 575

Rawlik, K., Toussaint, M., & Vijayakumar, S.. (2012). On stochastic optimal control and reinforcement learning by approximate inference. Proceedings of Robotics: Science and Systems VIII.

Richard, E. (1956). Bellman A problem in the sequential design of experiments. Sankhya: The Indian Journal of Statistics¯ (1933-1960), 16(3/4), 221–229.

Riedmiller, M. (1999). Concepts and facilities of a neural reinforcement learning control architecture for technical process control. Neural Computing & Applications, 8(4), 323–338. https://doi.org/10.1007/s005210050038

Sachio, S., Mowbray, M., Papathanasiou, M. M., del Rio-Chanona, E. A., & Petsagkourakis, P. (2022). Integrating process design and control using reinforcement learning. Chemical Engineering Research and Design, 183, 160–169. https://doi.org/10.1016/j.cherd.2021.10.032

Samuel, A. L. (1959). Some studies in machine learning. IBM Journal of Research and Development, 3(3), 210–229. https://doi.org/10.1147/rd.33.0210

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. In Proceedings of the 4th International Conference on Learning Representations (ICLR) (pp. 1–21).

Schmidhuber, J. (2015). On learning to think: Algorithmic information theory for novel combinations of reinforcement learning controllers AAD recurrent neural world models. arXiv preprint arXiv:1511.09249.

Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). Trust region policy optimization. International conference on achine learning (ICML).

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms, arXiv preprint arXiv: 1707.06347.

Schwenzer, M., Ay, M., Bergs, T., & Abel, D. (2021). Review on model predictive control: An engineering perspective. The International Journal of Advanced Manufacturing Technology, 117(5-6), 1327–1349. https://doi.org/10.1007/s00170- 021-07682-3

Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). Process dynamics and control. John Wiley & Sons.

Seo, G., Yoon, S., Kim, M., Mun, C., & Hwang, E. (2021). Deep reinforcement learning-based smart joint control scheme for on/of pumping systems in wastewater treatment plants. IEEE Access, 9, 95360–95371. https://doi.org/10.1109/ACC ESS.2021.3094466

Shafi, H., Velswamy, K., Ibrahim, F., & Huang, B. (2020). A hierarchical constrained reinforcement learning for optimization of bitumen recovery rate in a primary separation vessel. Computers & Chemical Engineering, 140, 106939–106945. https://doi.org/10.1016/j.compchemeng. 2020.106939

Shannon, C. E. X. X. I. I. (1950). Programming a computer for playing chess. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 41(314), 256–275. https://doi.org/10.1080/14786445008521796

Shi, X., Li, Y., Sun, B., Xu, H., Yang, C., & Zhu, H. (2020). Optimizing zinc electrowinning processes with current switching via deep deterministic policy gradient learning. Neurocomputing, 380, 190–200. https://doi.org/10.1016/j.neucom. 2019.11.022

Shuprajhaa, T., Sujit, S. K., & Srinivasan, K. (2022). Reinforcement learning based adaptive PID controller design for control of linear/nonlinear unstable processes. Applied Soft Computing, 128, 109450–109466. https://doi.org/10.1016/ j.asoc.2022.109450

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hiu, F, Sifre, L., Driesshe, G.V.D., Grapel, T., & Hassbis, D. (2017). Mastering the game of go without human knowledge. Nature, 550(7676), 354–359. https://doi.org/10.1038/nature24270

Singh, B., Kumar, R., & Singh, V. P. (2022). Reinforcement learning in robotic applications: A comprehen sive survey. Artificial Intelligence Review, 55(2), 945–990. https://doi.org/10.1007/s10462-021-09997-9

Siraskar, R. (2021). Reinforcement learning for control of valves. Machine Learning with Applications, 4, 100030. https://doi.org/10.1016/j.mlwa.2021.100030

Smith, B., Khojandi, A., & Vasudevan, R. (2023). Bias in reinforcement learning: A review in healthcare applications. ACM Computing Surveys, 56. https://doi.org/10.1145/3609 502

Spielberg, S. P. K., Gopaluni, R. B., & Loewen, P. D. (2017). Deep reinforcement learning approaches for process control. 6th International Symposium on Advanced Control of Industrial Processes (AdCONIP).

Sutton, R. S., & Barto, A. G. (1981). Toward a modern theory of adaptive networks: Expectation and prediction. Psychological Review, 88(2), 135–170. https://doi.org/10.1037/0033- 295X.88.2.135

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning an introduction. MIT Press.

Tesau, C., & Tesau, G. (1995). Temporal diference learning and TD-Gammon. Communications of the ACM, 38(3), 58–68. https://doi.org/10.1145/203330.203343

Turing, A. M. (2019). Intelligent machinery, A heretical theory. The Turing Test, 4(3), 105–110. https://doi.org/10.7551/mit press/6928.003.0014

Verdier, C. F., Babuška, R., Shyrokau, B., & Mazo, M. (2019). Near optimal control with reachability and safety guarantees. In Proceedings of the IFAC-PapersOnLine; Elsevier B.V, 52(11), 230–235. https://doi.org/10.1016/j.ifacol.2019.09.146

Wang, T., Bao, X., Clavera, I., Hoang, J., Wen, Y., Langlois, E., Zhang, S., Zhang, G., Abbeel, P., & Ba, J. (2019). Benchmarking model-based reinforcement learning. arXiv preprint arXiv:1907.02057.

Wang, X., Cai, J., Wang, R., Shu, G., Tian, H., Wang, M., & Yan, B. (2023). Deep reinforcement learning-PID based supervisor control method for indirect-contact heat transfer processes in energy systems. Engineering Applications of Artificial Intelligence, 117, 105551–105563. https://doi.org/ 10.1016/j.engappai.2022.105551

Wang, S., Duan, J., Lawrence, N. P., Loewen, P. D., Forbes, M. G., Gopaluni, R. B., & Zhang, L. (2024). Guiding reinforcement learning with incomplete system dynamics. International Conference on Intelligent Robots and Systems (IROS).

Wang, J. X., Kurth-Nelson, Z., Tirumala, D., Soyer, H., Leibo, J. Z., Munos, R., Blundell, C., Kumaran, D., & Botvinick, M. (2016). Learning to reinforcement learn. arXiv preprint arXiv:1611.05763. 2016.

Wang, D., Li, X., Hu, L., & Qiao, J. (2023). Data-driven tracking control design with reinforcement learning involving a wastewater treatment application. Engineering Applications of Artificial Intelligence, 123. https://doi.org/10.1016/j.enga ppai.2023.106242

Wang, Y., Velswamy, K., & Huang, B. (2017). A long-short term memory recurrent neural network based reinforcement learning controller for ofice heating ventilation and Air conditioning systems. Processes, 5(3), 46–64. https://doi.org/10. 3390/pr5030046

Webros, P. J. (1977). Advanced forecasting methods for global crisis warning and models of intelligence. General Systems, 22(1), 25–38.

Whiteson, S., Tanner, B., Taylor, M. E., & Stone, P. (2011). Protecting against evaluation overfitting in empirical reinforcement learning. In 2011 IEEE symposium on adaptive dynamic programming and reinforcement learning (pp. 120–127).

Willia, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3), 229–256.

Winston, P. H. (2017). On computing machinery and intelligence. Boston Studies in the Philosophy and History of Science, 324, 265–278. https://doi.org/10.1007/978-3-319-53280- 6\_11

Xu, X., Zuo, L., & Huang, Z. (2014). Reinforcement learning algorithms with function approximation: Recent advances and applications. Information Sciences, 261, 1–31. https://doi. org/10.1016/j.ins.2013.08.037

Yang, Q., Cao, W., Meng, W., & Si, J. (2022). Reinforcementlearning-based tracking control of waste water treatment

process under realistic system conditions and control performance requirements. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 52(8), 5284–5294. https://doi.org/10.1109/TSMC.2021.3122802

Yang, Y., & Ding, B. (2020). Model predictive control for LPV models with maximal stabilizable model range. Asian Journal of Control, 22(5), 1940–1950. https://doi.org/10.1002/ asjc.2070

Yang, R., Wang, D., & Qiao, J. (2022). Policy gradient adaptive critic design with dynamic prioritized experience replay for wastewater treatment process control. IEEE Transactions on Industrial Informatics, 18(5), 3150–3158. https://doi.org/10.1109/TII.2021.3106402

Yarats, D., Zhang, A., Kostrikov, I., Amos, B., Pineau, J., & Fergus, R. (2021). Improving sample eficiency in model-free reinforcement learning from images. Proceedings 35th AAAI Conference on Artificial Intelligence, 35(12), 674–681.

Yifei, Y., & Lakshminarayanan, S. (2023). Multi-agent reinforcement learning for process control: Exploring the intersection between fields of reinforcement learning, control theory, and game theory. The Canadian Journal of Chemical Engineering, 101(11), 6227–6239. https://doi.org/10.1002 cjce.24878

Yoo, H., Byun, H. E., Han, D., & Lee, J. H. (2021). Reinforcement learning for batch process control: Review and perspectives. Annual Reviews in Control, 52, 108–119. https://doi.org/10.1016/j.arcontrol.2021.10.006

Yu, Y. (2018). Towards sample eficient reinforcement learning. Twenty-Seventh International Joint Conference on Artificial Intelligence.

Yu, S., Bhm, C., Chen, H., & Allgöwer, F. (2012). Model predictive control of constrained LPV systems. International Journal of Control, 85(6), 671–683. https://doi.org/10.1080/0020 7179.2012.661878

Zhang, Z., Chong, A., Pan, Y., Zhang, C., & Lam, K. P. (2019). Whole building energy model for HVAC optimal control: A practical framework based on deep reinforcement learning. Energy and Buildings, 199, 472–490. https://doi.org/10.1016/j.enbuild.2019.07.029

Zhang, W. C., Yang, G, Lin, Y, Ji, C, & Gupta, M. M. (2018). On definition of deep learning. 2018 World Automation Congress (WAC).

Zou, F., Yen, G. G., Tang, L., & Wang, C. (2021). A reinforcement learning approach for dynamic multi-objective optimization. Information Sciences, 546, 815–834. https://doi. org/10.1016/j.ins.2020.08.101