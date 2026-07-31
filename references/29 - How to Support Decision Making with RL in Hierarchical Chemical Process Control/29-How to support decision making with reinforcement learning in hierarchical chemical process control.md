# How to support decision making with reinforcement learning in hierarchical chemical process control?

Kinga Szatm´ari <sup>a,\*</sup>, Tibor Chovan´ <sup>a</sup>, Sandor´ N´emeth <sup>a</sup>, Alex Kummer <sup>a,b</sup>

<sup>a</sup> Department of Process Engineering, University of Pannonia, Egyetem st. 10, Veszpr´em H-8200, Hungary

<sup>b</sup> HUN-REN-PE Complex Systems Monitoring Research Group, University of Pannonia, Egyetem st. 10, Veszpr´em H-8200, Hungary

## A R T I C L E I N F O

Keywords: Reinforcement learning Chemical process control Multi-agent reinforcement learning Optimal control CRISP-RL

## A B S T R A C T

In this review article, we explore the application of reinforcement learning (RL) at the different levels of hierarchical chemical process control. where reinforcement learning can improve efficiency and robustness in chemical process operations. RL algorithms are an optimal method for sequential decision making, therefore in chemical process control. where taking decisions is required continuously. RL can be a perfect fit due to its ability to handle dynamic, nonlinear, and uncertain environments. Reinforcement learning has already shown great potential in solving complex tasks, making it a promising approach for the challenges of chemical process control. We investigate the potential of reinforcement learning compared to traditional control methods. We present advanced multi-agent structures of RL, which can tackle large- scale chemical processes beyond the capabilities of a single agent. We introduce CRISP-RL (CRoss-Industry Standard Process for the development of Reinforcement Learning application), which is a paradigm that aims to deploy and maintain reinforcement learning projects, providing a methodology to handle and solve complex RL tasks and describe the current challenges and future directions for the integration of reinforcement learning into chemical process control.

## 1. Introduction

Chemical processes are usually nonlinear systems, so classic contro methods do not always work properly to change operating conditions or handle uncertainties [1]. With the rise of artificial intelligence and reinforcement learning (RL) in process control, autonomous control systems can be used in the chemical industry to solve problems that classical controls could not solve [2]. In different hierarchical control levels, reinforcement learning can be applied to different tasks to ach ieve safe operation, improve product yield, save energy, and reduce cost and time loss.

The main objective of process control is to maximize production while maintaining a desired level of product quality and safety and making the process more economical. There are a few representations of the classic control hierarchy levels, which are very similar, yet written from a different approach. In Fig. 1 the classic levels of control can be seen from what is called a process or automation engineer’s point of view. The objective of the lower levels is defined from the higher levels, but the higher levels need the lower levels to implement the control actions, so the levels are interdependent, and the good performance of the control loops helps control at the higher levels. The information goes up in the hierarchy and the control decision goes down from the higher levels to the lower levels [3].

The lowest level (Level 0) is the process itself and Level 1 is the process measurement and actuation level, where data collection and sensor validation occur. Level 2 is the safety, environment, and equipment protection, including alarm management and emergency shutdown [4]. Level 3a is the regulatory control level, where the basic process variables (temperature, concentration, pressure, liquid level and flow rate) are controlled, usually by single input-single output (SISO) control loops such as proportional-integral- derivative (PID) controllers [5], but ratio control usually to synchronize two flow control loops can also occur [6]. The obiective of the regulatory control level is to stabilize the system, ensure good tracking and maintain system performance [7], and locally control the measured variables [8]. Level 3b is the multivariable and constraint control level, where the control of a process unit is such that the actions of several control loops are coordinated and the process conditions are kept close to their optimal value, while the operating constraints are not violated. An optimization problem is solved at this level, where the model includes the production cost, the value of the product as product quality, and the process. However, the obiective function of the optimization problem depends on the market, it can be product maximization or operating cost minimization [4].

<table><tr><td colspan="2">Nomenclature</td><td>KDPP</td><td>Kernel Dynamic Policy Programming</td></tr><tr><td></td><td></td><td> $l_i$ </td><td>liquid level</td></tr><tr><td>π</td><td>policy</td><td>LCA</td><td>life cycle assessment</td></tr><tr><td>A2C</td><td>Advantage Actor Critic</td><td>LP</td><td>linear programming</td></tr><tr><td>A3C</td><td>Asynchronous Advantage Actor Critic</td><td>MADDPG</td><td>multi-agent DDPG</td></tr><tr><td>A</td><td>action space</td><td>MARL</td><td>multi-agent reinforcement learning</td></tr><tr><td>a</td><td>action</td><td>MDP</td><td>Markov Decision Process</td></tr><tr><td>AMIGO</td><td>Approximate M-constrained Integral Gain Optimisation</td><td>MILP</td><td>mixed-integer linear programming</td></tr><tr><td>ANN - PSO</td><td>artificial neural network particle swarm optimization</td><td>MIMO</td><td>multiple-input multiple-output</td></tr><tr><td>APC</td><td>Advanced Process Control</td><td>MOEA</td><td>multi-objective evolutionary algorithm</td></tr><tr><td>BEBP</td><td>blending effect based policy</td><td>MP - RLC</td><td>Model Predictive Control guided Reinforcement Learning</td></tr><tr><td> $c_i$ </td><td>constant</td><td></td><td>Control</td></tr><tr><td>CRISP - RL</td><td>CRoss-Industry Standard Process for the development of Rein- forcement Learning applications</td><td>MPC</td><td>Model Predictive Control</td></tr><tr><td></td><td></td><td>NLP</td><td>nonlinear programming</td></tr><tr><td>CSTR</td><td>continuous stirred tank reactor</td><td>P</td><td>transition probability</td></tr><tr><td>DCS</td><td>Distributed Control System</td><td>PG</td><td>Policy Gradient</td></tr><tr><td>DDP</td><td>Differential Dynamic Programming</td><td>PID</td><td>proportional integral derivative</td></tr><tr><td>DDPG - PINN</td><td>DDPG trained using physics-informed neural network</td><td>PIDO</td><td>perfect-information deterministic optimization</td></tr><tr><td>DDPG</td><td>Deep Deterministic Policy Gradient</td><td>PPO</td><td>Proximal Policy Optimization</td></tr><tr><td>DO</td><td>dissolved oxygen</td><td>r</td><td>reward</td></tr><tr><td>DPG</td><td>Deterministic Policy Gradient</td><td>RL</td><td>reinforcement learning</td></tr><tr><td>DPP</td><td>Dynamic Policy Programming</td><td>RLOps</td><td>Reinforcement Learning Operations</td></tr><tr><td>DQN</td><td>Deep Q-Learning</td><td>RTO</td><td>Real-Time Optimization</td></tr><tr><td>DRL</td><td>Deep Reinforcement Learning</td><td>s&#x27;</td><td>next state</td></tr><tr><td> $e_i$ </td><td>error from setpoint</td><td>S</td><td>state space</td></tr><tr><td> $E_{norm}$ </td><td>normalized value of energy consumption</td><td>s</td><td>state</td></tr><tr><td> $EP_{norm}$ </td><td>normalized value of eutrophication potential</td><td>SAC</td><td>Soft Actor-Critic</td></tr><tr><td>EV DO</td><td>expected-value deterministic optimization</td><td>SIMC</td><td>Simple Internal Model Control</td></tr><tr><td>FKDPP</td><td>Factorial Kernel Dynamic Policy Programming</td><td>SISO</td><td>single-input single-output</td></tr><tr><td>FP - NLP</td><td>first principles modeling with nonlinear programming</td><td>SPC</td><td>Statistical Process Control</td></tr><tr><td> $GHG_{norm}$ </td><td>normalized value of greenhouse gas emission</td><td>TD3</td><td>Twin-Delayed Deep Deterministic Policy Gradient</td></tr><tr><td>GS - MORL</td><td>Gain Scheduled PI controller using Multi-Objective RL</td><td>THRTO</td><td>time-horizon based RTO</td></tr><tr><td>H - MARL</td><td>hierarchical multi-agent reinforcement learning</td><td> $w_{EP}$ </td><td>weight of eutrophication potential,</td></tr><tr><td>HV AC</td><td>heating, ventilation, and air conditioning</td><td> $w_E$ </td><td>weight of energy consumption</td></tr><tr><td>IAE</td><td>integral absolute error</td><td> $w_{GHG}$ </td><td>weight of greenhouse gas emission</td></tr><tr><td>IMC</td><td>Internal Model Control</td><td>WWTP</td><td>wastewater treatment plant</td></tr><tr><td>ISE</td><td>integral square error</td><td>XRL</td><td>explainable reinforcement learning</td></tr></table>

Level 4 is the real-time optimization level, where the optimal conditions are determined that the multivariable and constraint control level attempts to maintain while hard constraints (equipment and material limits) are included [9]. The real-time optimization level relies on the premise that the model and disturbance transients can be neglected, provided the optimization takes place over a sufficiently long period for the process to achieve and maintain a steady state [10]. For the real-time optimization level, the yield, selectivity, and energy consumption of an entire plant are optimized, where the objective function contains the prices for energy and product [11]. Level 5 is the production planning and scheduling level, where information from the sales department is required on the quantities of the different products, the delivery deadlines, and the possible price of the products. Information from the purchasing department is also necessary about the price and availability of the materials [4]. Planning tells what product to produce and how based on economics, and scheduling tells when to produce that product based on the delivery time and to avoid storage problems [12].

In chemical process control, the design of classic controllers involves the analysis of the process dynamic and the development of a mathematical model. Maintaining the performance of the system is compli cated and ex-pensive due to drifts in process characteristics or changes in set points. Considering the limits of classical controllers, as most of them are linear and non-adaptive, reinforcement learning has started to be used in process control, as a data-based controller [13]. Instead of a design process, the operation of a closed-loop controller can be learned by interacting with the process with the reinforcement learning agent. The main difference from traditional controllers is that the RL agent learns the control law from experience, and the behavior of the reinforcement learning controller can be improved successively and can be applied to many types of processes [14].

The keywords ”reinforcement learning” and ”process control” are applied in Google Scholar to see the occurrence of RL articles with chemical process control case studies. Fig. 2 describes the keywords on a wordcloud, where the larger the word, the more times it appears in the articles. The most common keywords are reinforcement learning, learning systems, deep learning, model predictive control, controllers, and optimization. These keywords show that reinforcement learning can be applied for chemical process control as model predictive controls or for optimization problems. Many other keywords appear, such as case studies (continuous stirred tank reactor, batch process), applied RL algorithms (Q-learning, actor-critic, DDPG), or tasks (PID tuning, temperature control), showing that many articles have been written in reinforcement learning with chemical process control.

One of the keywords in Fig. 2 is the optimal control, which describes the process of finding a control policy that maximizes or minimizes a given cost function. The solution to this method can be approached through reinforcement learning [15], and traditional optimal control shares similarities with reinforcement learning in the context of process control [16]. The term control can be considered as a synonym for action, the term controller can be used as a synonym for the agent, and the term controlled system can be a synonym for the environment [17]. However, in traditional optimal control, the constraints can be clearly defined, while in reinforcement learning, they are described as soft-constraint only in the reward function. The following paragraph briefly describes reinforcement learning, furthermore a more detailed discussion is presented in Appendix A.

Reinforcement learning is a machine learning method in which an agent as a learner is not told what action to perform and has to discover the environment to know in which state which action to take to maximize its cumulative reward. The reward is the feedback, and it is delayed, as the action affects not only the immediate reward but also the later ones [15]. So, reinforcement learning is a self-learning process, and decision making is optimized based on trial and error [18]. The RL agent is in a given state s while interacting with the environment and taking an action a, receives a reward r for its action, and enters a new state s’ [19]. Reinforcement learning is based on the Markov Decision Process (MDP), where the state of the system does not depend on its previous state. A state space S, an action space A, a transition probability P : S × A × S → [0, 1], and a reward function r : $S \times A \times S \to$ R are defined, where the transition probability describes the probability of going from the current state s ∈ S to a new state s′ ∈ S if and action a ∈ A is taken [20]. The two value functions are the state value function and the state action value function. The state value function defines the expected cumulative reward if the agent in the state s follows the policy, and the state action value function defines the expected cumulative reward if the agent in the state s takes an action a based on the actual policy. The RL agent learns a policy (π(s)), which can be deterministic π : S → A or stochastic π $: S  P ( A )$ , and the policy shows what action an agent takes in the given state. The goal of the RL agent is to find and learn the optimal policy that maximizes the total cumulative reward [21]. The framework of the reinforcement learning process is described in Fig. 3.

Several algorithms can train the agent to learn the optimal policy, and the two main categories are model-free and model-based algorithms [22]. Model-free algorithms do not use the transition probability distribution and do not rely on the environmental model, as the agent learns by trial and error. In contrast, when the distribution is provided to the RL method, it is the model-based algorithm, where the model is learned by obtaining the data from the environment and optimizing the policy based on the learned model. The three types of model-free algorithms are value-based, policy-based, and the third is the combination of them as actor-critic methods, where the value function and the policy are also modeled. Value-based RL estimates the value function (Q-value) that is the expected value of the total reward for taking a specific action in a given state, and policy-based RL learns the optimal actions by directly parameterizing and optimizing the policy [23]. The two types of model-based methods are those that use a given model and those when the agent learns the model. When the model is given, the RL agent has direct access to the transition and reward functions, and in the second type, the agent first learns the model of the environment and then uses it to improve its policy. Model-free methods are easier to implement compared to model-based methods, and can be used in complex methods when building an accurate model is difficult [24].

A collection of model-based algorithms in reinforcement learning is dynamic programming that can be used to determine optimal policies when the agent has a perfect model of the environment [25]. For model-free methods, the most commonly applied value-based algorithms are Q-learning and Deep Q-network (DQN). Environments with discrete states and action values are relatively simple cases, where Q-learning can be applied, where the Q-values are updated in a Q-table for every state-action pair. If the state space is continuous but the action space is still discrete, DQN is the main algorithm applied, where the current state is passed to a neural network, and it estimates the Q-value that represents the best action to take [26]. However, to solve a complex task, continuous action space is required as in the policy- based and actor-critic algorithms. A policy-based algorithm is Reinforce, which is a Monte Carlo policy gradient algorithm, where the entire episode is required to determine the gradient and update the policy parameter [27]. The actor-critic algorithms are based on value-based and policy-based methods since the actor models the policy and the critic models the value function. One type of actor-critic method is the

![](images/6704ec9d5bd08f43a8191c6c4036b29b7c526e2bf244fff7c4fa1d7431224138.jpg)  
Fig. 1. The levels of process control.

# two term controlsystems policy optimizatior model machine earning chemica intelligenc wastewater treat deep loptima earning controls control policy adap process control lonetworksient learning algorithms -learning real-time optimizatior batch data processing reinelrcemenivelcarnin nonlin roportional control systems ladaptive control systems predictive control systems earnin systems process optimization tank reacto batch reinforcement earnin contro controllers formanc ddpg optimization deep neural

Fig. 2. The most occurred keywords in chemical process control with reinforcement learning.

![](images/203afe5475cf221928b1a466686987c00d18e03c75d22dea41cf2cf797bd2bd1.jpg)  
Fig. 3. The classical RL framework.

Deterministic Policy Gradient (DPG) algorithms that model the policy as a deterministic policy. DDPG (Deep Deterministic Policy Gradient) and TD3 (Twin Delayed Deep Deterministic) belong to these methods, where DDPG adapts DQN into continuous action space and combines it with DPG, and TD3 uses double DQN to DDPG to prevent overestimation of the value function [28]. Other actor- critic algorithms are A2C (Advantage Actor-Critic) and A3C (Asynchronous Advantage Actor-Critic), which are policy gradient algorithms. In both A2C and A3C, there are multiple agents or workers for parallel training, and each agent maintains a local policy and estimates the value function, and the agent synchronizes its parameters with the global network. The difference between A3C and A2C is that for A3C the agents work asynchronously, and for A2C there is a coordinator for synchronizing all agents [29]. SAC (Soft Actor-Critic) is an actor-critic algorithm that maximizes the expected reward while also maximizing the entropy. PPO (Proximal Policy Optimization) algorithm uses the constraint as a penalty and clips the objective, so the optimization is carried out in a predefined range [30]. Fig. 4 shows the classification of the mentioned RL algorithms [31, 32], as these algorithms are the most used in chemical process control.

The FKDPP (Factorial Kernel Dynamic Policy Programming) algorithm may be the first step in using reinforcement learning in the realworld to control a chemical plant. In the world first, in 2022, Yokogawa and JSR used FKDPP to control a chemical plant for 35 days to optimize production [33]. FKDPP is improved from Dynamic Policy Programming (DPP), as DPP does not learn the optimal value function. The main drawback of DPP is the computational complexity, so to solve this problem, a Kernel Dynamic Policy Programming (KDPP) can be used. But KDPP cannot cope with large action spaces, so this led to the use of FKDPP, which learns the action space dimension by dimension separately. The task is to control the decanter tank and the distillation column to optimize the yield and quality of the vinyl acetate monomer while maintaining stability [34]. A simulator of the plant was developed, then the evaluation and validation of the model were made, and the plant was operated with RL from January 17, 2022 to February 21, 2022. The solution with FKDPP effectively managed the challenging conditions required to maintain product quality and liquid levels in the distillation column and optimize the use of waste heat as a heat source [35].

As the test confirmed that reinforcement learning can control the distillation column, after scheduled maintenance, the test continued for a year, which showed that RL is capable of controlling a chemical plant. Stable control and operation of the liquid level of the distillation column is maintained throughout the year, also in winter and summer. The result shows that reinforcement learning can be applied to safely control a chemical plant. The high quality of the product is achieved and the emission of carbon dioxide is reduced compared to the classic control method of 40 %, and the control eliminates manual input from operators, preventing human error and decreasing workload [36].

In this work, we proposed how and where reinforcement learning can be applied at the different levels of hierarchical control of chemical processes. We grouped reinforcement learning in chemical process control articles by objectives and presented their applications and case studies. We compared the advantages of reinforcement learning methods with traditional methods and showed their possible applications. We summarized the multi-agent reinforcement learning structures and described the related case studies in chemical process control. We introduced CRISP-RL as a life cycle of a reinforcement learning project and presented its steps. To the best of our knowledge, a review article in the field of chemical process control has not been published so far, hence our contributions are as follows.

• We defined three hierarchy levels of chemical process control based on the possible objectives of reinforcement learning.

• We collected, sorted, and identified areas where reinforcement learning has potential for application at different hierarchical chemical control levels.

• We compared the performance of classic control methods with reinforcement learning control methods.

• We summarized the RL-based multi-agent control structures with centralized, decentralized, competitive, and cooperative agents in chemical process control.

• We described the CRISP-RL method, how reinforcement learning agents can be taught so that the agents can be implemented in a real system, and presented the life cycle of a reinforcement learning project.

![](images/81ff7812a20227994df88449e8247ca39d802d804071ae63fb36ba350437a82e.jpg)  
Fig. 4. Classification of RL algorithms.

![](images/cce5c27e97932d099fe569dc914dea869563fddad3197d31dfcbc1c6ea413f7a.jpg)  
Fig. 5. The three levels of chemical process control.

## 2. Chemical process control with reinforcement learning

In this work, we define three levels of control levels from the five classic levels based on the objectives of the reinforcement learning agent. We created the three levels based on the reward function, since in the processed articles, there were three different bases for the reward functions as objectives. The three levels and the tasks at the levels are presented in Section 2.1, and the objectives, applications, case studies, and algorithms with reinforcement learning at the levels are presented in Section 2.2. The comparison between traditional and RL controllers, the improvement achieved with reinforcement learning, and the research gaps for RL in chemical process control are presented in Section 2.3.

## 2.1. Introduction of the three levels of chemical process control

The three defined levels and their application are described in Fig. 5. Level 1 is the operation level, which includes the measurement and actuation level, the safety, environment, and equipment protection level, and the regulatory control level, where the Distributed Control System is applied. Level 2 is the production level, which includes the multivariable and constraint control level, advanced process control is used. Level 3 is the business level, which includes the real-time opti mization level and the planning and scheduling level, where production control is based on customer orders or management decisions. All levels have the task of ensuring the reliability and availability of the total control system through fault detection and fault tolerance, and all tasks are handled by computers [37].

The duties on Level 1 are control enforcement, system coordination and reporting, and reliability assurance. The control enforcement includes maintaining direct control of plant units and detecting and responding to emergency conditions. The system coordination and reporting of the system includes information collection and analysis of materials and energy use, forwarding the information to higher levels, and providing service to the operator's interface. Reliability assurance includes diagnostic functions and update and standby systems [37]. In most control applications, the PID (proportional-integrate-derivative) controller is the most widely used controller due to its simple structure and easy implementation and maintenance [38]. The Distributed Control System serves as the main operator interface for monitoring and controlling the plant [39].

Level 2 duties are control enforcement, plant coordination and operational data reporting, system reliability assurance, and production scheduling. The control enforcement includes responding to emergencies and locally optimizing the operation within the production schedule limits. The plant coordination and operational data reporting include collecting data on production, inventory, material and energy use, maintaining communication with lower and higher levels, making area production reports, and servicing the machine interfaces. The system reliability assurance includes performing diagnostics on themselves, but also at lower levels and updating all standby systems. The production scheduling includes establishing the production schedule and trans portation and locally optimizing the costs for its area [37].

PID controllers can be used for many tasks, but for nonlinear systems it is difficult to find a parameterization, so advanced controllers are required at this level. The model predictive control (MPC) is a modelbased controller that predicts the behavior of the future system by determining the optimal trajectory of the manipulated variable. MPC solves a constrained optimization problem, where a cost function is minimized, which is usually a tracking error between the model output and the reference [40]. Advanced process control (APC) can also be applied, which increases product yields, decreases operation costs, and improves process safety. The cost function in APC can be described as the minimization of the error of the process variable from the setpoint value and the difference of the actual manipulated variable from the initial manipulated variable [41]. Statistical process control (SPC) is another solution to address the needs of this level of the hierarchy. SPC is used as a monitoring function that can tell if the process is running satisfactorily, but this itself has no option to take actions, it only shows that the quality is beginning to decrease, and an intervention is required [42].

The duties on Level 3 are production scheduling, plant coordination and operational data reporting, and system reliability assurance. At this level, production scheduling includes the establishment of a basic production schedule, not just for a given area. It also includes modifying the production schedule in the knowledge of energy constraints and power demand, and when any interruptions happen in the lower level units. At this level, the determination of the optimum inventory level occurs. The plant coordination and operational data reporting include collecting and maintaining material use and available inventory, energy data and transfer, quality control files, production inventory files, maintaining interfaces with plant management, providing production and information to company management, sales personnel, and purchasing departments, and also providing order status information to sales personnel. The system reliability assurance includes self-check and diagnostic for itself and lower level equipment [37]. When the continuous process operates at steady-state, real-time optimization is applied that typically determines the operation point that optimizes the performance of the process. The cost function is defined as the minimization of the operating cost or the maximization of the production rate [43]. The level sets production goals and deals with capacity decisions, while supply and logistic constraints are met [4].

## 2.2. Objectives at the three levels with reinforcement learning

Chemical process control can be challenging at all levels due to nonlinearities, noise, or uncertainties in the process, which can result in long process delays, computational overhead, or economic loss. Traditional controllers due to their linear structure may be ineffective for higher-order systems, time- delay systems, nonlinear systems, and systems with uncertainties. They can suffer from additional difficulties, such as computational complexity and convergence of a solution to a local optimum, and their performance can decrease in the presence of uncertainties in quality and other process characteristics. The current traditional tuning methods are time-consuming and require system models that are difficult to obtain for complex processes. Monitoring and maintaining performance over time is also a complex and costly task due to the need for regular reidentification of the model, which is time consuming and can introduce expensive interruptions to normal operations. Developing and testing an RL-based strategy can improve the performance of process control, as it can deliver high control accuracy and robustness with reduced computational cost. However, in some cases, due to the slow learning rates and the high exploration requirements of reinforcement learning, the existing conventional process controller is required.

We searched in Google Scholar with the keywords "reinforcement learning” and ”process control” and collected the articles available on chemical process control. We grouped the articles into the three levels based on the reward functions, since the reward is also different at the three levels, such as duties. At the operation level, the reward is usually applied as the error from the setpoint. The main goal at this level is setpoint tracking, but PID tuning with RL also occurs. At the production level, the reward function is usually defined as the product concentration or yield or the delta concentration. The goals at this level are to maximize the amount of product, improve process yield and product quality, or minimize the batch process time. At the business level, the reward function is defined from cost, usually with profit as the revenue minus the inventory holding cost or the operating cost of the plant. The main goals at this level are profit maximization or operational cost

minimization.

We collected objectives, applications, and case studies at different levels and they are shown in Tables 1–3 at the operation level, the production level, and the business level, respectively. The main objectives at the operation level are setpoint tracking and adaptive control, and the most common case study that occurs is the control of the temperature, product concentration, or pH in a CSTR or batch process. At the production level, the most commonly occurring objective is maxi mizing the product concentration and improving yield and quality in a CSTR. The other most common case study is a fed-batch bioreactor, where the goal is to maximize product concentration at the end of the batch by controlling the reactor temperature or light intensity. At the business level, the two main goals are to maximize profit while ensuring product quality or to minimize operating costs. The case studies at this level are widely different from each other, CSTR, distillation column, oxygen system (oxygen generation system, storage system, consumption system), and gasoline blending process also occur.

The algorithms applied at the three levels are shown in Fig. 6, where most of the algorithms are model-free algorithms to handle complex high- dimensional environments without requiring an explicit system model. Q- learning and DQN are effective in discrete action spaces and utilize experience replay to improve sample efficiency, but struggle with continuous control. They are affected by overestimation bias [144], which can reduce learning stability and requires large memory storage to replay the experience, making computation expensive. At the business level, Q-learning and DQN are more widely used, as they are well suited for decision-making tasks that involve discrete choices, such as production scheduling.

DDPG is widely used for process control, including setpoint tracking, due to its ability to handle continuous action spaces. Benefits from off policy learning, which allows better sample efficiency by reusing past experiences. However, DDPG is prone to overestimation bias that leads to unstable learning. TD3 improves on DDPG by addressing overestimation bias using clipped Q-learning, resulting in greater stability. Despite this, TD3 requires significant computational resources and has slower training. At the production level for optimization tasks, TD3 becomes more important due to its robustness in handling nonlinearities and uncertainties. Another popular algorithm at the production level is PPO, which can handle stochastic environments more effectively than DDPG and TD3. However, it suffers from a lower sample efficiency because it is a policy-based method and requires more training data [145].

SAC is also effective in handling stochastic environments and longterm planning challenges. However, SAC requires extensive computational resources and tuning to balance exploration and exploitation effectively. It is used at the production level and is one of the dominant algorithms at the business level, where long-term optimization and adaptability are crucial. Other widely used algorithms at the business level are actor-critic methods. They require a more complex network architecture and their computational cost is also high, which may limit their use in real-time control applications. At the business level, A2C is more widely used than at the other two levels, as it is well suited to handle large-scale decision-making tasks in stochastic environments [145].

Despite its simplicity and ease of implementation, Reinforce is an algorithm that is limited in chemical process control. It suffers from high variance in gradient estimates, leading to slow and unstable learning. It requires complete trajectories before updating the policy, making it inefficient for real-time learning and adaptation [146]. At the production level, where long-term optimization is needed, it can be applied in some cases, but is still less common due to its inefficiency.

## 2.3. Improvement in chemical process control with reinforcement learning

Reinforcement learning offers a promising alternative that could provide a path to more adaptive and efficient control than traditional

Objectives, applications and case studies for process control with reinforcement learning at the operation level.

<table><tr><td>Objective</td><td>Application</td><td>Case study</td></tr><tr><td>achieving ending conditions, minimizing reaction time</td><td>control of the four key parameters (unreacted reagent, two quality parameters, safety parameter)</td><td>batch polymerization [44]</td></tr><tr><td rowspan="6">adaptive control</td><td>control of the effluent concentration and reactor&#x27;s hold-up</td><td>CSTR [45]</td></tr><tr><td>control the liquid level</td><td>mutli-modal nonlinear tank system [46]quadruple tank system [47]sef-regulatory mixing tank [48]first order system [46]</td></tr><tr><td>control of the general output</td><td>light olefin separation [49]linear second-order-plus-deadtime system [50]second order system, steam turbine system [51]</td></tr><tr><td>control of the product concentration</td><td>CSTR [52]</td></tr><tr><td>control of the reactor temperature</td><td>CSTR [50,53]</td></tr><tr><td>track a given sin signal</td><td>complex nonlinear system [54]</td></tr><tr><td rowspan="3">compare MPC and RL following an optimal trajectory</td><td>control of the process variable</td><td>SISO, MIMO transfer function [55]</td></tr><tr><td>control of the monomer and initiator flow rates</td><td>semi-batch polymerization [56,57]</td></tr><tr><td>control of the synthetic gas temperature and composition</td><td>fluidized bed biomass gasification process [58]</td></tr><tr><td rowspan="15">improving performance reach a target mean crystal size satisfy technical requirements setpoint tracking</td><td>control of the chemical plant under disturbances</td><td>vinyl acetate monomer plant [59]</td></tr><tr><td>control of the temperature</td><td>semi-batch crystallization [60,61]</td></tr><tr><td>control of the moisture content level</td><td>drying [62]</td></tr><tr><td>control of the ammonia nitrogen content</td><td>sour water treatment unit [63]</td></tr><tr><td>control of the froth-middlings interface level</td><td>primary separation vessel [64]conical tank system [65]</td></tr><tr><td>control of the liquid level</td><td>hybrid three-tank system [66] tank system [67,68]two tank system [69]</td></tr><tr><td>control of the nitrit ammonium ratio</td><td>partial nitritation and anaerobic ammonium oxidation [70]</td></tr><tr><td>control of the outlet water content</td><td>natural has dehydration system [71]distillation column, HVAC system, paper machine transfer function [72,73]</td></tr><tr><td>control of the process variable</td><td>CSTR [74-77]fed-batch bioreactor [78]</td></tr><tr><td>control of the pH</td><td>CSTR [71,74,79-81]</td></tr><tr><td>control of the pH and reactor temperature</td><td>solid oxide cell system [80]batch transesterification</td></tr><tr><td>control of the product concentration</td><td>reactor [82]batch polymerization [83]CSTR [53,84-86]semi-batch reactor [87]</td></tr><tr><td>control of the reactor temperature and liquid level</td><td>CSTR [88]</td></tr><tr><td>control of the reactor temperature and the reagent concentration</td><td>CSTR [68,89-94]CSTR fed by reactant [74,75]tubular reactor [75]zinc roasting process [95]</td></tr><tr><td>control of the roasting furnace</td><td></td></tr></table>

Table 1 (continued )

<table><tr><td>Objective</td><td>Application</td><td>Case study</td></tr><tr><td></td><td>control of the superheating</td><td>Organic Rankine Cycle [85, 96]</td></tr><tr><td></td><td>control of the temperature of the demineralized water</td><td>fluidized bed reactor [97]</td></tr><tr><td></td><td>optimal control policy</td><td>hydraulic fracturing [98]</td></tr></table>

Table 2  
Objectives, applications and case studies for process control with reinforcement learning at the production level.

<table><tr><td>Objective</td><td>Application</td><td>Case study</td></tr><tr><td>improve the product yield and quality</td><td>control of the flow, pressure and temperature</td><td>vinyl acetat monomer process [99]</td></tr><tr><td rowspan="6">maximizing the product concentration</td><td>control of the cooling jacket temperature</td><td>CSTR [100]</td></tr><tr><td>control of the heat flow rate</td><td>CSTR [101]</td></tr><tr><td>control of the inlet flow rate and the cooling jacket temperature</td><td>CSTR [102,103]</td></tr><tr><td>control of the inlet flow rate and the heat added to the reactor</td><td>CSTR [104]</td></tr><tr><td>control of the pH control of the solvent flow rate</td><td>CSTR [105] liquid-liquid extraction column [106]</td></tr><tr><td>control of the vapor density, reagent and product concentration and temperature</td><td>CSTR [107]</td></tr><tr><td rowspan="6">maximizing the product concentration at the end of the batch</td><td>control of the feed rate</td><td>batch process [108]</td></tr><tr><td>control of the heat exchanger temperature</td><td>semi-batch polymerization [109]</td></tr><tr><td>control of the inflow rate and/or the light intensity</td><td>fed-batch bioreactor [110–118]</td></tr><tr><td>control of the reactor temperature</td><td>batch reactor [119,120]</td></tr><tr><td>control of the reactor temperature and light intensity</td><td>semi-batch reactor [112] fed-batch bioreactor [121]</td></tr><tr><td>control of the reactor temperature and monomer feed rate</td><td>batch polymerization [122,123] semi-batch crystallization [120]</td></tr><tr><td>minimizing nitrogen and total aeration energy</td><td>control of the concentration of the dissolved oxygen</td><td>wastewater treatment plant [124,125]</td></tr><tr><td>minimizing the live steam consumption</td><td>control of the liquid level, density and temperature of the product liquor</td><td>multistage evaporation process [126]</td></tr><tr><td>minimizing the process time</td><td>control of the flow rate, jacket temperature, reflux and stirring speed</td><td>solvent switch case (switch the reactor solvent with the crystallization solvent) [127]</td></tr><tr><td>optimizing recovery rate</td><td>control of the middlings flow rate</td><td>primary separation vessel [64]</td></tr></table>

methods. However, RL faces its challenges, such as slow learning rates, high exploration requirements, difficulties in the implementation of real-time control in highly complex nonlinear systems, and the need for human intervention in critical situations. Despite these obstacles, RL has shown promise in improving control accuracy and robustness. Tables 4–6 show the traditional controllers, the RL controllers at the three different levels, and the task to be solved, including improvements due to the reinforcement learning method. At the operational level, improvements include faster convergence and overall better performance. The controller follows the setpoint with minimal adjustment time, no overshoot, and responds faster while maintaining robustness and lower error values. At the production level, the RL controllers have better disturbance compensation and can robustly handle constraints. At the business level, RL can achieve higher profit and less computational time.

Table 3  
Objectives and case studies for process control with reinforcement learning at the business level.

<table><tr><td>Objective</td><td>Application</td><td>Case study</td></tr><tr><td rowspan="6">maximizing the profit while ensuring product quality</td><td>build a production schedule for the full planning horizon of K days</td><td>continuous, chemical manufacturing process with a single stage and single production unit [128]</td></tr><tr><td>chemical material production scheduling control of the feed, reflux and vapor feed rate</td><td>multi-product chemical reactor [129]distillation column [130]</td></tr><tr><td>control of the feed rate optimal temperature profile and batch time</td><td>CSTR [131]batch reactor [132]</td></tr><tr><td>plant-wide control</td><td>vinyl acetate monomer process [133]</td></tr><tr><td>scheduling the oxygen system to fill the gap between production and consumption</td><td>oxygen system (oxygen generation system, storage system, the consumption system) [134]</td></tr><tr><td>scheduling the reactor on a daily basis in the face of uncertain demand and production interruptions setting the number of stages, feed stage, condenser pressure, reflux-and reboil ratio</td><td>continuous reactor stage followed by a packaging stage [135]distillation column [136]</td></tr><tr><td>minimizing the makespan of the process</td><td>complete the demands</td><td>batch plant [137]</td></tr><tr><td rowspan="4">minimizing the operating cost</td><td>control of the feed rate and the temperature setpoint</td><td>CSTR [138]</td></tr><tr><td>determine mass flow by the power load of the unit, and determine the concentration of  $SO_2$  by the sulfur content in the coal optimal blending recipe</td><td>wet flue gas desulfurization [139]</td></tr><tr><td></td><td>gasoline blending (component tanks, static mixer, storage tank) [140,141]</td></tr><tr><td>reduce energy consumption and pursue operational stability</td><td>oxygen production system with cryogenic air separation units [142]</td></tr><tr><td>satisfying the effluent requirements to keep the total nitrogen under a limit</td><td>control of the dissolved oxygen concentration and ammonia removal</td><td>wastewater treatment plant [143]</td></tr></table>

Based on the literature we can state that reinforcement learning can be applied in many ways in chemical process control as presented, and it can improve the overall efficiency of the process and reduce energy consumption. RL is also highly effective in managing complex and variable processes, where conditions change frequently. These types of processes are often nonlinear and RL can learn the best actions to take in real-time to maintain desired outcomes. RL could be applied for emergency shutdowns or could optimize start-ups and shut-downs, which is typically challenging because traditional methods struggle to handle them effectively. Optimal learning techniques can be investigated to balance exploration and exploitation, improving learning efficiency even with limited samples in complex chemical processes, and it could reduce the computational cost. RL algorithms could be modified to handle constraints, ensuring safe operation within process control systems while optimizing performance.

![](images/f1744a4ed7cdafa4f832307b9260d270caaf03aa6bf24ed0e1b2d00c34d91a89.jpg)  
(a)

![](images/8407c076b2f64b85237e153a56b671cde3039ecf442f3272bd5ada917d0e72f9.jpg)  
(b)

![](images/f2dba35af63833839033d8128b8c42e0955fa61741fdc0d077f40c480a73a0b0.jpg)  
(c)  
Fig. 6. Applied reinforcement learning algorithms at the three levels. (a) Applied algorithms at the operation level, (b) Applied algorithms at the production level, (c) Applied algorithms at the business level.

It can also be concluded that reinforcement learning could also be combined with traditional control methods that can complement each other to make systems more robust and trustable. For example, RL can handle the more unpredictable or nonlinear aspects of a process, while classical methods ensure stability during normal operations. With more agents in the processes, RL could manage the interactions between different parts of the system. For example, in large-scale operations where multiple units are connected, RL can find optimal ways to synchronize them, ensuring smoother operation and reducing inefficiencies. A robust control strategy may be developed with multiple agents, even as they work hierarchically to simulate real-world industrial processes. Hierarchical reinforcement learning could be applied in chemical process control, where the different agents operate at different levels. More agents can also be applied at one level with different structures as multi-agent reinforcement learning, and we discuss that in detail in the next section.

## 2.4. Aspect of safe reinforcement learning in chemical process control

In chemical process control for production processes, it is essential to maintain efficiency, ensure safety, and keep operations within defined parameters. Reinforcement learning can learn the optimal strategy through trial and error, adapt to changing conditions, reduce the need for human intervention, and improve both process efficiency and safety [147]. In real-world applications, it is crucial to consider human safety, environmental safety, and most importantly, control safety. To guarantee safety, a safe policy function can be provided, where the agent does not visit certain states and does not take certain actions. The design of the reward function is also a challenge for safety applications, as with a loose function, safety cannot be ensured, but with a conservative function, the reward performance would be poor [148].

Safe reinforcement learning can be described as a Constrained Markov Decision Process, in which a reward must be maximized while the agent satisfies the safety constraints. Several safe reinforcement learning methods exist, and the groups are formal method-based approaches, policy optimization-based approaches, control theory-based approaches, and Gaussian process-based approaches. Formal methodsbased approaches ensure safety without unsafe probabilities, using mathematical and logical tools. However, they can be computationally expensive and may struggle with high-dimensional or continuous envi ronments [148].

Policy optimization-based approaches optimize the policy by the cumulative cost values on the trajectories. In model-based RL methods, it can be implemented using the Chernoff function, where a parameter is used in the cost function to balance between reward performance and safety. An- other model-based safe RL method is a primal-dual optimization, where a primal-dual gradient descent is applied for the optimization under the constraint of the failure probability. In model-free reinforcement learning, one safe RL method is a Projection-based Constrained Policy Optimization that guarantees safety that maximizes reward with the Trust Region Policy Optimization, and projects the policy to the feasible region [148].

Control theory-based approaches can ensure more rigorous safety than policy optimization-based methods. Safety involves applying control strategies that keep the state of the system within a safe operating range [149]. A model-based safe RL approach can be used with the Control Lyapunov- Barrier Function, that contains the weighted sum of the control barrier functions and the control Lyapunov function. This integration ensures that both stability (Lyapunov function) and safety constraints (barrier function) are maintained during the process [150]. The control policy and the optimal value function can both be learned for the optimal control of a CSTR with applying this approach [151].

Comparison between the traditional controller and RL controller at the opera tion level.

<table><tr><td>Traditional method</td><td>RL method</td><td>Task and improvement</td></tr><tr><td rowspan="6">MPC controller</td><td>DDP, DDPG TD3, SAC</td><td>control of a semi-batch reactor to reach the setpoint earlier with more reliability [78,87,91]</td></tr><tr><td>DDPG, TD3</td><td>control of the output of transfer functions with a more robust controller [55]</td></tr><tr><td>DDPG, fuzzy Q-learning MP-RLC, PPO</td><td>control of a CSTR with practically no under or overshoot and lower MSE [74, 75,79,81,85,88,89,91]</td></tr><tr><td>SAC, TD3</td><td></td></tr><tr><td>DDPG-PINN</td><td>control of the transfer functions with computational efficiency [94]</td></tr><tr><td>DQN</td><td>control of a fluidized bed biomass gasification to minimize oscillation [58]</td></tr><tr><td rowspan="8">PID controller</td><td>A3C</td><td>control of a primary separation vessel with lower MSE and IAE value [64]</td></tr><tr><td>DDPG</td><td>control of a steam stripping process with improvement of the qualified rate of ammonia nitrogen content [63]</td></tr><tr><td>DDPG</td><td>following an optimal trajectory in a semi-batch reactor with more robust controller [56]</td></tr><tr><td>DDPG, PPO</td><td>control of a CSTR with faster response and no overshoot [71,85,88]</td></tr><tr><td>DDPG, TD3</td><td>control of the output of transfer functions with a more robust controller [55]</td></tr><tr><td>DRL</td><td>control of a fluidized bed reactor to following the setpoint with no overshoot [97]</td></tr><tr><td>PG</td><td>control of a tank system with two times smaller setpoint error value [67,68]</td></tr><tr><td>TD3</td><td>control of a solid oxide cell with limiting performance degradation [80]</td></tr><tr><td rowspan="7">PID controller tuning</td><td>DDPG</td><td>control of a light olefin separation unit using dual dividing wall columns with lower RMSE [49]</td></tr><tr><td>DDPG</td><td>control of a tank system with reduced ISE value [47]</td></tr><tr><td>DDPG</td><td>control of a CSTR with much faster convergence and stabilizing the process [45,50,52]</td></tr><tr><td>PI+policy iteration</td><td>control of a steam turbine system [51] and second order systems with better performance [50,51]</td></tr><tr><td>PI+Q-learning TD3</td><td>control of a semi-batch crystallization process with better performance [60]</td></tr><tr><td>GS-MORL (PPO)</td><td></td></tr><tr><td>TD3</td><td></td></tr></table>

Gaussian processes-based approaches estimate uncertainty and unsafe areas in contrast to policy optimization methods that pay more attention to cost values, and Lyapunov function-based techniques, which emphasize safe actions. In chemical process control, a semi-batch reactor is controlled, where maximization of the value function with respect to control actions employs the Upper Confidence Bound. With this optimization, the optimal policies can be found efficiently [152].

Despite the promising approaches in safe reinforcement learning, several challenges need to be addressed for implementation in chemical process control. One of the main obstacles is the difficulty in accurately modeling complex chemical processes with high-dimensional state spaces and nonlinear dynamics. The real-time application of safe RL in processes that require fast decision making is still limited, as many methods can be computationally expensive or require extensive simulations. Although safety constraints are typically well defined in theo retical models, translating these constraints to real-world operations, especially in dynamic and uncertain environments, could be difficult.

Table 5  
Comparison between the traditional controller and RL controller at the production level.

<table><tr><td>Traditional method</td><td>RL method</td><td>Task and improvement</td></tr><tr><td rowspan="5">MPC controller</td><td>A3C, DPG DDPG, SAC</td><td>optimizing the recovery rate in a primary separation vessel with higher average recovery rate [64]</td></tr><tr><td>actor-critic, DDPG PPO, TD3</td><td>control of a CSTR to maximize the product with better disturbance compensation and reaching steady state earlier [100–103,105,107]</td></tr><tr><td>DDPG Monte Carlo</td><td>achieve high productivity in a batch polymerization with consistent action profiles [122,123]</td></tr><tr><td>DDPG, oracle-assisted constrained Q-learning PG, PPO, SAC</td><td>control of a semi-batch reactor maximizing the product at the end of the batch with constant handling in a robust way [110, 112–114]</td></tr><tr><td>SAC, TD3</td><td>control of a semi-batch polymerization maximizing the product with the best learning speed [109]</td></tr><tr><td>PID controller</td><td>SAC, TD3</td><td>control a semi-batch polymerization to maximizing the product with the best learning speed [109]</td></tr></table>

Comparison between the traditional controller and RL controller at the business level.

<table><tr><td>Traditional method</td><td>RL method</td><td>Task and improvement</td></tr><tr><td>ANN-PSO (artificial neural network particle swarm optimization) FP-NLP (first principles modeling with nonlinear programming)</td><td>PPO</td><td>minimizing the cost in a CSTR and achieving higher profit [138]</td></tr><tr><td>BEBP (blending effect based policy), LP NLP, THRTO (time-horizon based RTO)</td><td>SAC</td><td>producing qualified gasoline at the least cost with optimal blending recipe with more profit and blending success rate [140,141]</td></tr><tr><td>EVDO (expected-value deterministic optimization) PIDO (perfect-information deterministic optimization)</td><td>A2C</td><td>maximizing the profit in a chemical production schedule with less execution time [129]</td></tr><tr><td>MILP (mixed-integer linear programming)</td><td>A2C</td><td rowspan="3">maximizing the profit in a chemical manufacturing process with better product availability level [128] minimizing the total operating cost in an oxygen production system with higher capacity [142] maximizing the real-time profit in a CSTR with less computational time [131]</td></tr><tr><td>MOEA (multi-objective evolutionary algorithm)</td><td>PPO</td></tr><tr><td>NLP</td><td>actor-critic</td></tr></table>

## 3. Reinforcement learning control structure

The goal of this section is to present the advanced control structures for reinforcement learning control, where multiple agents are used, and communicate with each other resulting in multi-agent reinforcement learning and control. Most reinforcement learning controllers act as feedback controllers with the application of one agent, but multi-agent reinforcement learning (MARL) contains multiple agents in a shared environment [153]. In the multi-agent environment from the agent’s point of view, the other agents can be considered as part of the environment [154]. Section 3.1 presents the categorization of multi-agent reinforcement learning that is based on the agents’ cooperation, as competitive, cooperative, and mixed structures. Section 3.2 presents other categorizations of MARL as decentralized and centralized structures that tell the connection between agents. In Google Scholar, we searched for ”multi-agent reinforcement learning” and ”process control” to summarize the MARL articles in chemical process control and to present the available articles and the advantages and disadvantages of MARL.

## 3.1. Cooperation of agents in chemical process control

In multi-agent reinforcement learning, the goal is to learn the optimal policy for every agent, so they can achieve the goal together. One categorization of MARL is based on the cooperation of the agents, and the three main groups are cooperative, competitive, and the mixture of both agents [155]. In cooperative settings as pure cooperation, agent aim to behave collaboratively and optimize the long-term common goal. In competitive settings such as pure competition, the agents compete against each other, and the global reward often closes to zero, as the agents improve their policy at the expense of the other agents and maximize their own goals. In mixed settings, agents share a common objective but compete with each other to reach their own goals [156]. The structure of the multi-agent reinforcement learning configuration is determined by the reward, for example, in the competitive structure, the reward of one agent is the negative of another agent. In the cooperative setting, agents share a common reward to maximize it, and in the mixed setting, both elements occur of competitive and cooperative structures [157]. A pure competitive strategy is not recommended for use in chemical process control because the win of one agent is the loss of another agent in this structure. So, the two applicable structures are the pure cooperative and the mixed structure in chemical process control. The structure of the multi-agent competitive and cooperative rein forcement learning environment is presented in Fig. 7.

So far, MARL has not been applied many times for process control, especially not in chemical process control due to the complexity of these systems, and the higher computational cost [158], since the exploration of agents is more difficult as more agents have to be coordinated to get a good result. In addition, agents learn simultaneously so that the environment does not remain stationary, and the scale of the task increases with each agent [159]. However, considering the advantages of MARL, it may be used more and more in chemical process control in the future.

The first case study shows the comparison between the cooperative and mixed structure in a distillation column with two controlled vari ables, which are the top product composition and the bottom product composition. The manipulated variables are the top reflux rate for the top product and the steam flow rate for the reboiler to the bottom product. Two RL agents are applied to control the two variables with TD3 algorithms, and the states are the error in discrete time, the discrete-time integral of the error, the discrete-time derivative of the error, and the previous control action, so the RL controllers are used as a PID controller imitation. In the pure cooperative structure, all agents have the same reward function with the same objectives as the negative sum of the IAE values for all controlling variables as described in Eq. (1). In the case of the mixed configuration, the agents are configured to minimize the IAE for their controlled variable as competition and minimize the overall IAE value as cooperation. However, an aggressive control action may influence the other control loops, so they also need to cooperate for the common goal of minimizing the overall IAE value. The results show that the two structures perform similarly, but the mixed structure performs slightly better [160].

$$
r _ {n} = - \sum_ {i = 1} ^ {n} \sum_ {k = 0} ^ {m} \left| e _ {i} (t _ {1} + k \Delta t) \right|\tag{1}
$$

So, based on the cooperation of the agents, the mixed structure can be the best, as the agents have common and also different goals. The different reward functions can help to optimize the policy of the individual agents and the common goal can help to optimize the common policy of the agents. In the case study mentioned, the RL agents do not know the behavior of the other agents and act independently from each other. This leads to other categorizations of multi-agent reinforcement learning, which are centralized and decentralized learning, where the agents can know the behavior of the other agents.

## 3.2. Connection between the agents in chemical process control

The three different structures of multi-agent reinforcement learning based on the connections between the agents are shown in Fig. $^ { 8 , }$ and they are fully centralized learning, fully decentralized learning, and centralized learning with decentralized execution [161]. Centralized learning and execution (assumptions made after learning) means that all information, for example, observations, rewards, value functions, or policies, is shared with all the agents. An agent can get the information from all the agents and tell them what action to take. Decentralized learning and execution do not have this information sharing, and only local information is applied to update the agents’ policies. The third category, centralized training with decentralized execution, combines the two previous structures [162].

In centralized learning (and distributed execution), experiences are given back to a single learner, and the policies are going under a local execution. So, independent experiences contribute to the fact that all agents have a similar policy [163]. The main disadvantage of centralized learning is that it is not always feasible due to the joint reward of the agents, since during training it has to become one reward, and the central policy has to be learned through a joint action space, which increases exponentially with the number of agents [164]. Also, the MARL algorithm has a much higher computational cost than the single agent algorithms. Because of the drawbacks, centralized learning is difficult to apply in chemical process control because it is difficultto define a common reward function that can be good for the learning of all the agents in the system.

In decentralized learning, both the execution and the learning happen locally. Each agent can adapt to the environment and can be treated independently. Against centralized learning, in decentralized learning, standard single-agent algorithms can be used since the prob lem is decomposed into decentralized single-agent problems, where agents are independent and do not rely on shared information [165]. The main disadvantage of this structure is that agents cannot use information, for example, policies, from other agents. Due to the parallel training of all agents, agents cannot make a difference between changes in the environment as the actions of the other agents, and the transition function of the environment [166]. We presented the previous case study with the distillation column [160] for the cooperative and mixed structure, but it is also a decentralized learning structure since agents do not share information.

![](images/abd48403b359a0af6b32fb1e1a3852a890a52dc274da4eed6885af7780860c29.jpg)  
Fig. 7. Competitive and cooperative structures in MARL. (a) Competitive multi-agent reinforcement learning structure, (b) Cooperative multi-agent reinforcement learning structure.

![](images/97e317d66916249b900858b486ca00ca6abe6d17a96de9756f06a54832331bca.jpg)  
(c)  
Fig. 8. Different structures for multi-agent reinforcement learning. (a) Fully centralized multi-agent reinforcement learning structure, (b) Fully decentralized multiagent reinforcement learning structure, (c) Centralized learning, decentralized execution multi-agent reinforcement learning structure.

Another case study in the decentralized multi-agent structure for chemical process control is PI controller tuning. Two algorithms are applied to solve the task, the TD3 and DDPG algorithms for a nonlinear three-tank hybrid system, where three cylindrical tanks are connected by solenoid valves that switch on and off. Two RL agents are applied, where the states are controlled variables, which are the liquid level of two tanks $( l _ { 1 } , l _ { 2 } )$ , the setpoints for the liquid levels, and the parameters of the PI controllers. The reward is defined from the states and the error from the setpoints $( e _ { i } = l _ { i , s p }$ − l ) for both agents, as described in Eq. (2), where c is a constant. In the results, the performance of the controller is compared to the classic controller tuning methods (IMC, SIMC. AMIGO) and the RL-DDPG controller, where the RL-TD3 controller has less overshoot and less integral squared error. The advantages of tuning with RL over the traditional PI controller tuning are operational flexibility, failure tolerance, and simplified design [167].

$$
\begin{array}{c} \boldsymbol {r} _ {1} = - c _ {1} | \boldsymbol {e} _ {1} | ^ {2} \\ \boldsymbol {r} _ {2} = - c _ {2} | \boldsymbol {e} _ {2} | ^ {2} \end{array}\tag{2}
$$

The mix of centralized learning and decentralized learning structures is centralized learning with decentralized execution, where the advantages of both structures are combined. Centralized learning helps to improve sampling efficiency and speed up learning, and decentralized execution allows using independent policies during evaluation. What is difficult in centralized learning is that the policy is required to be known but is affected by all agents, so information is needed on all actions of the agents. A well-suited RL algorithm is the actor-critic algorithm for this problem since it considers the actions of the agents, so the only task is to extend the algorithm to all the actions of all agents. This means an independent actor and a centralized critic, and agents can have different policies, so this structure can be applied in both cooperative and competitive settings [163,168].

For centralized learning with decentralized execution, the case study is a wastewater treatment plant (WWTP). The task is optimizing dissolved oxygen (DO) and chemical dosage due to complex control and high nonlinearity of the plant. Two agents are applied in the work with the MADDPG (multiagent DDPG) algorithm, where Gaussian noise is applied for exploration. The DO is controlled by one agent and the dosage is controlled by the other agent. The algorithm has a decentral ized actor and a centralized critic, so the actor network has its own observations and actions. and the critic network has the entire observation. The states are influents, inflow rate, time, and current DO and dosage. The reward function is designed from the cost and the life cycle assessment (LCA), and normalization is applied to balance the evaluation. Eq. (3) describes the reward function, where $E _ { n o r m } ,$ w<sub>E</sub>, $E P _ { n o r m } ,$ w<sub>EP</sub>, and GHG , w are the normalized values and weights of energy consumption, eutrophication potential, and greenhouse gas emission, respectively. The results show that the optimization with MADDPG has lower environmental impacts and energy and cost can be reduced [169].

$$
r = w _ {E} E _ {\text {norm}} + w _ {E P} E P _ {\text {norm}} + w _ {G H G} G H G _ {\text {norm}}\tag{3}
$$

The advantage of multi-agent reinforcement learning based on the presented case studies is that the computational cost of MARL can be reduced by parallel computation with a decentralized structure. The multiple agents can share their experiences, exchange information towards communication, teach other agents, or imitate the skilled agents. If an agent fails to complete its task, the other agent can help it out, so overall MARL is robust [170,171]. In process control, more agents can be applied at the given level, as presented for the quality control of the distillation column, the liquid level control of the three-tank hybrid system, and the control of the dissolved oxygen and chemical dosage of the wastewater treatment plant. Later, MARL can be extended to be applied in large state and action spaces with more adaptive and robust control. Besides, MARL cannot only be at a given level, but similarly to hierarchical process control, where the information is not only transmitted within levels but also between different levels of the hierarchy, and when reinforcement learning agents are applied at the hierarchy levels, hierarchical multi-agent reinforcement learning (H-MARL) comes into play.

H-MARL can solve coordination issues, can handle the increasing number of parameters with the increasing number of agents, and partial observability in MARL, since higher-level agents can coordinate the actions of lower-level agents, just as in hierarchical process control. A task hierarchy can be defined by decomposing the overall task into smaller subtasks that can be solved by an agent. H-MARL has been applied recently in traffic management [172,173] or energy management [174], but we see a potential of H-MARL in hierarchical chemical process control, since scalability and adaptability to the non-stationary environment can be improved. However, currently, even multi-agent reinforcement learning is in its infancy in chemical process control, so it has to overcome its challenges to be applicable in more complex ways.

## 4. Training RL agents for chemical process control

When you have the best possible model to teach reinforcement learning for chemical processes, you can use that model in an online environment for management at some level, or you can use it to complement RL to estimate future states. The reinforcement learning algorithms are ideally applicable to problems where multiple decisions are required or in environments that do not reset after a decision is taken. Using these models, users can achieve greater efficiency, resilience, and precision of control in complex systems. The four main phases of an RL project are problem definition, environment and Markov Decision Process (MDP) refinement, modeling and implementation, and evaluation and deployment, which are described in Fig. 9. Section 4.1 presents the life cycle of a reinforcement learning project, where feelings during the project are also described. Section 4.2 presents the environment and MDP refinement, here the refinement steps are described. Section 4.3 presents the modeling and implementation phase, where offline learning, imitation learning, and transfer learning are presented, and Section 4.4 presents the evaluation and deployment phases.

## 4.1. Life cycle of an RL project

At the beginning of the project, you may start with many positive feelings, but they can change during the project. Feelings during the life cycle of a reinforcement learning project can be described in Fig. 10. The project usually starts with optimism, since it is the beginning and the task seems exciting, but stress and depression begin to bloom when obstacles appear. Now, the realization comes that the task will not be as easy as one might first think, and fear comes into play when there is still no result. The simplification of the real data can lead to surprises and happiness when the project is working, but then it starts again from the beginning. In general, the two main tips for an RL project can be to start simple and to have a small development cycle. Starting simple helps to focus on the most important aspects of the project, reduce stress during the task, and make work efficient. Long development cycles result in a greater chance of a non-working code and more waiting for feedback, which reduces enthusiasm and interest in the project, therefore, reduced cycle time also helps to improve efficiency [163].

![](images/0da79eceb3e9bd0ee0a8909db8a60c0141e56edc8162db7a7283b2c3f3440005.jpg)  
Fig. 9. The four main phases of an RL project.

![](images/a1c69bbeb717119a07e66557f239ff2ffc377a149f5f0568d5b02690da08c266.jpg)  
Fig. 10. Feelings during a reinforcement learning project.

CRoss-Industry Standard Process model for the development of Reinforcement Learning applications is called CRISP-RL, and the life cycle is presented in Fig. 11. In the life cycle, after the definition of the project, the development of the RL agents will initially be performed offline in a simulation environment due to the high data demand. For simulator development, data of sufficient quality must be collected. Once the offline reinforcement learning agent has been deployed in the real world, it is fine-tuned and taught online.

In a reinforcement learning project, the problem has high- and lowlevel aspects, where changing the high-level aspects influences the lowlevel aspects. The high-level aspects are the refinement of the Markov decision process model, the definition of interactions, and how the agent learns. The low-level aspects are the individual components of the problem, like the en- vironment, the state, the action, and the reward. The environment can be a chemical process, the state can be temperature, concentration, integral error, or flow rate, and the action can be changing the control valve position. The reward function is a scalar and it must be designed as the objective function. After the definition of the problem, the data must be gathered from the environment and then prepared for training purposes.

![](images/1ff78e99ed6ebb6098cc66e9f160b45ae8a97817ea5b8fa26360a40cf45d8173.jpg)  
Fig. 11. Life cycle of a reinforcement learning process.

The data collection from the industrial plant requires costs and time, and even the project can be delayed until the right amount of data is collected. The collected data requires quality verification, which includes the description and requirements of the data. The data description ensures quality checks by analyzing and visualizing the data to understand the process. The data requirement defines the data conditions, such as certain feature ranges, the format of the data, and the limit of the missing values. This verification ensures that all data meets the requirements [175].

The next step is data preparation, which builds on data under standing to create a data set for modeling. The data set must capture sufficient diversity and quality to reflect the complexity of the envi ronment. When varied stateaction pairs are missing, the agent may fail to learn robust policies. Since RL relies on interactions between the agent and the environment, the selected data sets often represent the trajectories of states and actions. Standardization can be applied to the data, so the features are defined on the same scale, reducing bias and improving the performance of the neural network [175].

Process simulation offers a controlled and efficient environment for the training of the agents, in which environment modeling is safe, repeatable, and scalable, compared to real-world training. For training in complex environments, the agent needs a lot of interactions to learn the optimal policy, which would also take a long time in the real-world. Simulator development can be implemented in different process simulator software or programming environments. A well-designed simulator accurately models the real environment, capturing constraints and uncertainties.

An offline reinforcement learning is a data-driven approach to reinforcement learning. In this case, the agent has no ability to interact with the environment, only learns from the collected static data set from a simulator or real-world interactions. The train set (state, action, new state, reward trajectories) is collected once with a potential unknown policy, and it does not change during training [176]. The main challenge in offline reinforcement learning is distributional shift, since the data come from a fixed distribution, but the learned policy creates a different one, overestimation can often occur. To solve this problem, methods such as importance sampling, conservative Q-learning, or behavior cloning can be used to reduce variance and prevent the policy from selecting out-of-distribution actions [177]. Evaluation can be performed in the test set and deployment involves testing the learned policy in a simulator prior to real-world implementation. However, due to the offline dataset, the agent has poor performance and cannot be deployed directly [178].

It is desirable to further fine-tune the pre-trained RL agents through online interactions with the environment. During online reinforcement learning, the agent interacts with the environment, collects new trajectories, and can improve its performance. However, the environments are often non- stationary, so the conditions, objectives, or constraints may change over time. Without continual learning, an RL agent may struggle to adapt to new tasks or forget previously learned behaviors. Continual learning enables an agent to acquire knowledge incrementally and adapt efficiently to new challenges, making it essential for long-term autonomy in complex and evolving environments [179].

Performance metrics should be monitored during the agent’s continuous interactions with the environment for evaluation, which can include cumulative rewards or learning stability. During deployment, it is important to ensure that the RL agent is capable of effectively handling non-stationary conditions. The agent can be tested on real-time tasks, where it can be observed whether it successfully adapts to new tasks. This can involve monitoring the agent’s ability to avoid unexpected events by periodically retraining it on past data.

The chosen reinforcement learning algorithm is usually an algorithm that can handle a continuous state and action spaces and an off-policy method that needs less data to train than an on-policy method. Offpolicy methods improve a different policy from the one that is used to generate the data, while on-policy methods improve the policy that is used to make decisions [15]. However, off-policy learning currently has more potential, but on-policy learning can be better later, as it is more stable and predictable.

## 4.2. Environment and MDP refinement

The next step of an RL project after the definition of the problem and the preparation of data is to develop a process simulator. With this, the environment can made understandable and the project is efficiently managed. This gives a deeper understanding of the problem and allows improvements to be explained and developed more effectively. The most popular environment interface in RL is OpenAI’s Gym environment, which is a Python framework. From an industrial point of view, the interface serves as an excellent tool for modeling the problem. Simula tion can represent a real-life problem, since it is cheaper, easier, and faster to operate as it reduces the feedback cycle and increases productivity, but it can miss some important information from real life. However, simulations can be implemented to validate that the idea is technically viable. When the simulation results are acceptable with offline reinforcement learning, a real-life implementation can be performed. The main challenges in real-world implementation involve operational factors such as scaling, monitoring, or feasibility. For example, real-world applications have safety concerns that simulations do not have, so the application of safe reinforcement learning is vital. Additional common issues include partial observability, which is missing in simulations with complete knowledge, and high levels of stochasticity. The refinement process is shown in Fig. 12, where the iteration of analysis, design, implementation, and evaluation is part of the process. In every iteration after the evaluation, it can be decided if the result is good enough or if it is not good enough, then an analysis starts again to make improvements. Since improvements can always be made, it is important not to get lost in the loop. The Pareto distribution also applies here, as 80 % of the solution will take 20 % of the time and vice versa [163].

## 4.3. Modeling and implementation

Most RL algorithms interact with the environment and create observations, which are stored in the replay buffer. In many frameworks, it is assumed that the RL agent learns online, where the agent learns at the same time when it interacts with the real or simulated environment. Online learning requires a high number of interactions to learn an optimal policy, where exploration can be improved by increasing the sample efficiency. But because of sampling delays due to stochastic approximation and stability issues, exploration always comes with a price [163]. In process control, the trial and error method only increases the computational cost that can lead to serious safety incidents [180]. To solve this problem, offline or batch learning is introduced, which has no further interaction with the environment. During offline learning, data are generated online with a random policy and stored in a buffer. A policy is trained based on those data, and then this new policy can be deployed in new scenarios, and RL learns from data already collected without interacting further with the environment. In the online case, interaction with the environment can be costly or time consuming, so data collected in advance can help to speed up RL agent learning and increase efficiency [181]. The framework of offline reinforcement learning is described in Fig. 13.

![](images/967b032c250fdede5f2597bd937c4e78b9a210bf2e78d9e6e597e07c9514add9.jpg)  
Fig. 12. The process of refinement.

Offline reinforcement learning is similar to learning by example, which is called imitation learning. Imitation learning is about mimicking the behavior of an expert (human or agent) behavior who can perform well in the given task, and it shows the agent which action to take at the given observation. The simplest form is behavioral cloning, where the agent clones the behavior of the experts as supervised learning for the state action pairs. The other form is inverse reinforcement learning, where the reward function is unknown and is updated when the agent’s learned policy is compared to the expert’s policy [182]. An imitation learning is used for the cooling crystallization of paracetamol, where experts are the PID and MPC strategies, and the agent mimics their performance to achieve an effective trajectory tracking and the final crystal size [183]. For fed-batch chemical process control, four offline reinforcement learning methods are compared, such as Conservative Q-Learning, Batch Constrained Q-Learning, Implicit Q-Learning, and Behavior Cloning [184]. The results show that imitation learning has a high potential in chemical process control, as the algorithms perform well.

Offline reinforcement learning increases sample efficiency, and a model can be trained many times. This is a kind of supervised learning method with labeled transitions and rewards, but MDP is applied for data generation, so RL algorithms are the best way to find the optimal policy for MDPs. But in this type of learning, exploration is lost due to the sampled batch. The disadvantage of learning from offline data is that the collected data may not be representative of the state-action space, and the agent does not have the opportunity to self-correct its decision through interactions. However, learning from offline data can be useful in transfer learning, where a model is trained to offline data, and the same model is transferred to a similar problem to improve its performance, where fine-tuning is already made to online data [163]. So in transfer learning, some training parameter is kept constant to reduce the number of episodes and apply stored knowledge from a different but similar and related problem. The control of a photo-production system model is described as case studies with deterministic and stochastic differential equations and a nonsmooth model. The result shows that a near- optimal policy can be achieved for a stochastic system even without knowing its true dynamics [113].

![](images/4327e438f865ddfdde6690d55475b7c5fc541ddfdc0e11cb99b4e569468f95a6.jpg)  
Fig. 13. The framework of offline reinforcement learning.

During the implementation phase, the code needs to be handled, and GitHub is a suitable solution for this since it created an archive in the Artic for storing projects. Frameworks can be used to create an RL application, and open-source frameworks are available from a public source code, but the disadvantage, despite their popularity, is that they do not always reflect real value and quality. A good RL framework is based on good abstractions as it is easy to swap policy models, replay buffers, or exploration techniques. A simple codebase is also helpful, as the navigation through the code is easier and more understandable, and observable, as the framework includes monitoring. For complex models, the computational requirements increase and it becomes infeasible to train the agent, so scaling is important in a code to reduce the complexity of the model. With multiple agents in a shared environment, synchronization is required to avoid independent copies of experiences. A project is the best if it has minimal dependencies on other software and documentation is made with examples [163].

## 4.4. Evaluation and deployment

Evaluation can be implemented by the policy performance measure, which can be described by two metrics, policy performance and learning performance. The performance of the policy tells how well the policy solves the problem, and the learning performance tells how quickly an agent can be trained to achieve the optimal policy. Quick online learning is important to reduce the impact of suboptimal decisions and minimize the cost of missed opportunities during the learning process. Policy performance is usually represented by the discounted reward, and learning performance is represented by time. Time is determined differently depending on the task, for simulations it can be the number of episodes, or computational time can be used to improve the efficiency of the algorithm. The total difference between the optimal policy and the agent’s policy is defined as regret [163]. It represents the area between the optimal episodic reward and the actual reward [185], as presented in Fig. 14. To the best of our knowledge, regret is not included in chemical process control yet.

Another evaluation metric can be explainability, which tells the agent why the given decision was made and increases trust in reinforcement learning. In our previous article, we have developed an explainable reinforcement learning (XRL) agent in chemical process safety by decision tree and Shapley value [186]. Reinforcement learning for industrial control implementation needs to be safe and explainable to confidently use the system. By integrating known process domain knowledge directly into the formulation of the RL problem, for example, process constraints, the agent does not need to learn these constraints. This method guarantees that actions are consistent with process limitations, ensuring that implementation is safe and explainable [187].

Deployment is the phase when the application is prepared for oper ation. The projects in this stage are viable and valuable, and they need to be reliable. An RL agent can be deployed in three steps to be used by people, that are during development, hardening, and production. It emphasizes that in the early stages of development, rapid and flexible experimentation is crucial because the goal is to demonstrate the project’s viability and value. Optimization, such as scalability or repeatability, becomes important later. At this stage, the biggest cost is caused by slow feedback loops. Then the process goes through the hardening phase, when the robustness is improved and the policy is maintained. Priorities change when integrating an RL solution into a product or service. Initially, value is maximized through quick experimentation and research, but over time, factors such as reliability, efficiency, and robustness become more critical. As a result, different best practices are needed at various stages of the process, and the three important aspects are reliability, scalability, and flexibility [163].

![](images/a5c0d397197f3b9465dd59f2038ff2282e02d89f43a7e15d0919cbd31abf4a12.jpg)  
Fig. 14. Regret in reinforcement learning.

## 5. Conclusion and future directions

To summarize, several opportunities exist in chemical process control with reinforcement learning. These directions can make RL tech nology more suitable for industrial applications to improve the resilience of the system and handle challenges such as nonlinearity, uncertainties, and safety concerns in chemical processes. We present the applied reinforcement learning controllers for the three defined hierarchical control levels and compare their applicability with traditional controllers. We introduce the objectives at the different hierarchical levels, which are defined in the reward function, and show the reinforcement learning algorithms used. The multi-agent structures and case studies are presented in chemical process control, where the competitive structure cannot be a good idea, unlike the cooperative and mixed structures. We summarize the difference between centralized and decentralized structures and present their advantages and disadvantages. Finally, we described CRISP-RL as a framework for efficiently solving a reinforcement learning task.

Advantages: Reinforcement learning can help to improve the performance of the controllers in chemical process control. It can reach the setpoint earlier technically without under or overshoots, minimize the oscillation, and reduce the MSE and IAE values. RL can have the best learning speed, and less computational time, and robustly handle constraints to maximize the product concentration, minimize the batch process time, maximize the profit, or minimize the total operating cost. In summary, reinforcement learning controllers can be a robust solution in chemical industrial applications.

Limitations: In chemical process control, reinforcement learning faces several limitations at the different hierarchical levels. One of the most important challenges is ensuring safety and stability, which makes it difficult to apply in complex nonlinear systems. RL is inefficient on poor samples, it requires a large amount of data to learn the policy effectively. Since real-time data collection is often slow and impractical, simulations lead to model errors between the trained policy and real world performance. Another major issue is the lack of explainability, as RL decisions are not interpretable, making it difficult to troubleshoot or validate learned control policies.

Future directions: Future research in reinforcement learning for chemical process control will likely focus on addressing its current limitations at different hierarchical levels. One direction is to develop safe RL techniques that ensure stability and robustness in real-world applications. Another direction could focus on further enhancing the robustness of the Bayesian reinforcement learning algorithm exploring advanced uncertainty quantification techniques that can be robust to uncertainty. Improving sampling efficiency through model-based RL and transfer learning could reduce dependence on large datasets and enable faster learning in real-world applications. The application of multi-agent RL can also help to improve the performance of the RL controllers, and a multi-agent system using Bayesian inference for un certainty quantification can also be developed for chemical process control. It is crucial that these methods not only be tested in simulation, but also be validated in laboratory and industrial processes to ensure their effectiveness.

In general, a reinforcement learning agent presents promising potential in chemical process control to improve efficiency and accuracy. Agents can adapt to nonlinear and dynamic environments, optimize complex processes, and continually improve performance over time, offering significant advantages over traditional methods. The capability of learning from interactions makes reinforcement learning a powerful tool for the control challenges in chemical processes. However, further research and development are required to overcome industrial imple mentation challenges and realize the full potential benefits in real industrial applications.

## CRediT authorship contribution statement

Kinga Szatmari:´ Writing – review & editing, Writing – original draft, Visualization, Methodology, Investigation, Formal analysis, Conceptualization. Tibor Chovan:´ Writing – review & editing, Conceptualization. Sandor´ Nemeth:´ Writing – review & editing, Supervision, Methodology, Conceptualization. Alex Kummer: Writing – review & editing, Supervision, Methodology, Conceptualization.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgement

This work has been prepared with the professional support of the Cooperative Doctoral Program of the University Research Scholarship Program of the Ministry of Culture and Innovation financed from the National Research, Development and Innovation Fund.

This work has been supported by the 2024-2.1.1-EKOP<sup>¨</sup> University Research Scholarship Programme of the Ministry for Culture and Inno vation from the Source of the National Research, Development and Innovation Fund. This work has been implemented by the TKP2021- NVA-10 project with the support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development and Innovation Fund, financed under the 2021 Thematic Excellence Programme funding scheme.

This work has been implemented by the TKP2021-NVA-10 project with the support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development and Innovation Fund, financed under the 2021 Thematic Excellence Programme funding scheme.

## Appendix A. Basics of reinforcement learning

Reinforcement learning is one type of machine learning method, in addition to supervised and unsupervised learning. In supervised learning, the goal is to learn from labeled data, and in unsupervised learning, the goal is to detect patterns in the data. In contrast to them, during reinforcemen learning, an autonomous agent makes decisions to maximize its cumulative reward through trial and error. Reinforcement learning deals with sequential decision making, where a Markoy Decision Process (MDP) is a classical formalization. The current action affects not only the immediate reward, but subsequent states, and consequently future rewards [188].

The main concepts of reinforcement learning are agent, environment, state, action, reward, and policy. The agent is the learner and decision maker, and it interacts with the environment, so everything outside the agent is the environment. During the interaction, the agent decides as select actions, and as a result of the decision the environment presents a new situation to the agent. For decisions, the agent receives rewards from the environment as numerical values, and the agent wants to maximize the reward over time. The policy is a mapping function from states to probabilities of selecting each possible action. If the agent follows the policy π at time $t ,$ then $\pi ( s | a )$ is the probability that the agent takes a given action in a given state [189].

The interaction between the agent and the environment is a sequence of discrete time steps $t = 0 , 1 , 2 , 3 , \ldots$ In each time step, the agent gets a representation of the environment as state $\boldsymbol { S } _ { t } \in { \cal S } ,$ selects an action $A _ { t } \in A ,$ and one time step later, the agent gets a reward $R _ { t } \in R \in \mathbb { R }$ for its action, and goes to a new state $S _ { t + 1 }$ . Equation A.1 shows the sequence or trajectory that represents the MDP and the agent together [15].

$$
S _ {0}, A _ {0}, R _ {1}, S _ {1}, A _ {1}, R _ {2}, S _ {2}, A _ {2}, R _ {3}, \dots\tag{A.1}
$$

The set of states, actions, and rewards (S, A, R) have a finite number of elements in a finite MDP. So, the discrete probability distributions of the variables $R _ { t }$ and $S _ { t }$ depend on the previous state and action. The probability of $\boldsymbol { s ^ { \prime } } \in \boldsymbol { S }$ and $r \in S$ occurring at time t is defined in Equation A.2, where the function $p$ is the dynamics of MDP, and $s ^ { \prime } , s \in S , r \in R ,$ , and $a \in A ( s )$ [15].

$$
p (s ^ {\prime}, r | s, a) \equiv P r \{S _ {t} = s ^ {\prime}, R _ {t} = r | S _ {t - 1} = s, A _ {t - 1} = a \}\tag{A.2}
$$

The dynamic function $p : S \times R \times S \times A \to [ 0 ,$ , 1] is a deterministic function with four arguments. The function p describes a probability distribution for every s and $a ,$ as shown in Equation A.3 [15].

$$
\sum_ {s ^ {\prime} \in S} \sum_ {r \in R} p (s ^ {\prime}, r | s, a) = 1, f o r a l l s \in S, a \in A (s)\tag{A.3}
$$

The state must have information about all past agent-environment interactions that make a difference for the future, so it has the Markov property. When the four-argument dynamics function is known, the state-transition probability can be computed, which is written in Equation A.4, and it is a three-argument function $p : S \times S \times A  [ 0 ,$ , 1] [15].

$$
p (s ^ {\prime} | s, a) \equiv P r \{S _ {t} = s ^ {\prime} | S _ {t - 1} = s, A _ {t - 1} = a \} = \sum_ {r \in R} p (s ^ {\prime}, r | s, a)\tag{A.4}
$$

The expected reward for state-action pairs can also be determined as described in Equation $\mathsf { A } . 5 ,$ which is a two-argument function $r \colon S \times A \to$ R [15].

$$
r (s, a) \equiv \mathbb {E} [ R _ {t} | S _ {t - 1} = s, A _ {t - 1} = a ] = \sum_ {r \in R} r \sum_ {s ^ {\prime} \in S} p (s ^ {\prime}, r | s, a)\tag{A.5}
$$

The agent’s goal is to maximize the cumulative reward and not the immediate reward. The cumulative reward is called the return $\left( G _ { t } \right)$ , and can be defined for episodic tasks as shown in Equation $\mathsf { A } . 6 ,$ where T is for the terminal state, since an episodic task has an end state [189].

$$
G _ {t} = R _ {t + 1} + R _ {t + 2} + R _ {t + 3} + \dots + R _ {T}\tag{A.6}
$$

However, for process control tasks, the agent-environment interaction does not break down into identifiable episodes. It is called continuing task, which has no terminal state, so the return would go to infinity. So in this case the return is defined as the discounted sum of future rewards in Equation $\mathbf { A . 7 , }$ where $\gamma$ is a discount rate that tells the importance of future rewards compared to immediate rewards [189].

$$
G _ {t} = R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} R _ {t + 3} + \dots = \sum_ {k = 0} ^ {\infty} \gamma R _ {t + k + 1}\tag{A.7}
$$

Almost all reinforcement learning algorithms estimate value functions, which can be the state value function and the action value function. The state value function $( \nu _ { \pi } ( s ) )$ estimates the expected return of the agent in a given state s following a policy π as presented in Equation A.8. The action value function $( q _ { \pi } ( s , a ) )$ estimates the expected return of the agent in a given state s taking an action a following a policy π, which is shown in Equation A.9 [190].

$$
\nu_ {\pi} (s) = \mathbb {E} _ {\pi} [ G _ {t} | S _ {t} = s ] = \mathbb {E} _ {\pi} \left[ \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1} | S _ {t} = s \right]\tag{A.8}
$$

$$
q _ {\pi} (s, a) = \mathbb {E} _ {\pi} \left[ G _ {t} \mid S _ {t} = s, A _ {t} = a \right] = \mathbb {E} _ {\pi} \left[ \sum_ {k = 0} ^ {\infty} \gamma^ {k} R _ {t + k + 1} \mid S _ {t} = s, A _ {t} = a \right]\tag{A.9}
$$

Solving a reinforcement learning task means finding an optimal policy. One policy is better than the other policy if the expected return is greater with the policy. The optimal policy is $\pi ^ { * } { } _ { ! }$ , and the optimal state-value functiontion is described in Equation A.10, and the optimal action-value function is described in Equation A.11 [191].

$$
v _ {*} (s) = \max _ {\pi} v _ {\pi} (s) \tag {A.10}
$$

$$
q _ {*} (s, a) = \max _ {\pi} q _ {\pi} (s, a) \tag {A.11}
$$

Optimal policies have an optimal action value function, which is written in Equation A.12. These equations are the Bellman optimality equation, and solving one of the equations means finding an optimal policy, and so solving the reinforcement learning task [15].

$$
q _ {*} (s, a) = \mathbb {E} [ R _ {t + 1} + \gamma \nu_ {*} (S _ {t + 1}) | S _ {t} = s, A _ {t} = a ]\tag{A.12}
$$

In chemical process control tasks with reinforcement learning, there are many states and actions, and it would be too slow to learn the value of the states individually. $s _ { 0 } ,$ function approximation techniques occur most commonly, that can estimate the action value function, which is described in Equation A.13 [192].

$$
\widehat {q} (s, a, \boldsymbol {w}) \approx q _ {\pi} (s, a)\tag{A.13}
$$

The goal is to minimize the mean squared error between the approximate action value function and the true action value function that is given in

Equation A.14 [192].

$$
J (\boldsymbol {w}) = \mathbb {E} _ {\pi} \left[ \left(q _ {\pi} (s, a) - \widehat {q} (s, a, \boldsymbol {w})\right) ^ {2} \right]
$$

## Data availability

No data was used for the research described in the article.

## References

[1] M. Bloor, A. Ahmed, N. Kotecha, M. Mercangoz,¨ C. Tsay, E.A.D.R. Chanona, Control-informed reinforcement learning for chemical processes, arXiv preprin arXiv:2408.13566 (2024).

[2] R. Nian, J. Liu, B. Huang, A review on reinforcement learning: introduction and applications in industrial process control, Comput. Chem. Eng. 139 (2020) 106886.

[3] J. Hahn, T.F. Edgar, Process control. Kirk-Othmer Encyclopedia of Chemical

[4] M. Hovd, Advanced Chemical Process Control: Putting Theory Into Practice, John Wiley & Sons, 2023.

[5] R. Arroba, K. Rocha, M. Herrera, P. Leica, O. Camacho, Pid and sliding mode control for a reactor-separator-recycler system: a regulatory controllers comparison. in: Proceedings of the Systems and Information Sciences: ICCIS 2020, Springer, 2021, pp. 366–377.

[6] T. H¨agglund, J.L. Guzm´an, Development of basic process control structures, IFAC PapersOnLine 51 (4) (2018) 775–780.

[7] H. Komari Alaei, A. Yazdizadeh, Robust flow controller design and analysis for a chemical process, Trans. Inst. Meas. Control 36 (6) (2014) 723–733.

[8] M.S. Govatsmark, S. Skogestad, G. Sobocan, P. Glavic, Application of a plantwide control design procedure to a distillation column with heat pump., https://skoge. folk.ntnu.no/publications/2003/govatsmark\_escape13\_heatpump/escape13\_ no327.pdf [Accessed: 2025. 04. 10.].

[9] D.F. Mendoza, J.E.A. Graciano, F. dos Santos Liporace, G.A.C. Le Roux, Assessing the reliability of different real-time optimization methodologies. Can. J. Chem. Eng, 94 (3) (2016) 485–497.

[10] V. Adetola, M. Guay, Integration of real-time optimization and model predictive control, J. Process Control 20 (2) (2010) 125–133.

[11] D. Müller, B. Dercks, E. Nabati, M. Blazek, T. Eifert, J. Schallenberg, U. Piechottka, K. Dadhe, Real-time optimization in the chemical processing industry, Chem. Ing. Tech. 89 (11) (2017) 1464–1470.

[12] M.L. Darby, M. Nikolaou, J. Jones, D. Nicholson, Rto: an overview and assessment of current practice, J. Process Control 21 (6) (2011) 874–884.

[13] S. Spielberg, A. Tulsyan, N.P. Lawrence, P.D. Loewen, R.B. Gopaluni, Deep reinforcement learning for process control: a primer for beginners, arXiv preprint arXiv:2004.05490 (2020)

[14] S. Spielberg, R. Gopaluni, P. Loewen, Deep reinforcement learning approaches for process control, in: Proceedings of the 6th International Symposium on Advanced Control of Industrial Processes (AdCONIP), IEEE, 2017, pp. 201–206.

[15] R.S. Sutton, A.G. Barto, Reinforcement Learning: An Introduction, MIT Press, 2018.

[16] X. Zhu, Y. Wang, Z. Wu, Reinforcement learning for optimal control of stochastic nonlinear systems. AIChE J. (2025) e18840.

[17] T. Lattimore, C. Szepesvari,´ Bandit Algorithms, Cambridge University Press, 2020.

[18] T. Kegyes, Z. Süle, J. Abonyi, The applicability of reinforcement learning methods in the development of industry 4.0 applications, Complexity 2021 (1) (2021) 7179374.

[19] H. Tan, X. Hong, Z. Liao, J. Sun, Y. Yang, J. Wang, Y. Yang, Combining reinforcement learning with mathematical programming: an approach for optimal design of heat exchanger networks, Chin. J. Chem, Eng, 69 (2024) 63–71

[20] R.J. Williams, Reinforcement learning and markov decision processes (2007).

[21] R. Burtea, C. Tsay, Constrained continuous-action reinforcement learning for supply chain inventory management, Comput. Chem. Eng. 181 (2024) 108518.

[22] A. Seyyedabbasi, R. Aliyev, F. Kiani, M.U. Gulle, H. Basyildiz, M.A. Shah, Hybrid algorithms based on combining reinforcement learning and metaheuristic methods to solve global optimization problems. Knowl. Based Syst. 223 (2021) 107044.

[23] G. Wu, D. Zhang, Z. Miao, W. Bao, J. Cao, How to design reinforcement learning methods for the edge: An integrated approach toward intelligent decision making, Electronics 13 (7) (2024) 1281.

[24] N. Akalin. A. Loutfi, Reinforcement learning approaches in social robotics. Sensors 21 (4) (2021) 1292.

[25] A M Andrew, Reinforcement learning: an introduction Kvbernetes 27 (9) (1998 1093-1096.

[26] K.U. Ahn, C.S. Park, Application of deep q-networks for model-free optimal control balancing between different hvac systems, Sci. Technol. Built Environ. 26 (1) (2020) 61–74.

[27] H. Dong, H. Dong, Z. Ding, S. Zhang, T. Chang, Deep Reinforcement Learning, Springer, 2020.

(A.14)

[28] S. Dankwa, W. Zheng, Twin-delayed ddpg: a deep reinforcement learning technique to model a continuous movement of an intelligent robot agent, in: Proceedings of the 3rd International Conference on Vision, Image and Signa Processing, 2019, pp. 1–5.

[29] D. Mehta, State-of-the-art reinforcement learning algorithms, Int. J. Eng. Res. Technol. 8 (2020) 717–722.

[30] F. AlMahamid, K. Grolinger, Reinforcement learning algorithms: an overview and classification, in: Proceedings of the JEEE Canadian Conference on Electrical and Computer Engineering (CCECE), IEEE, 2021, pp. 1–7.

[31] M.A.M. Khan, M.R.J. Khan, A. Tooshil, N. Sikder, M.P. Mahmud, A.Z. Kouzani, A. A. Nahid. A systematic review on reinforcement learning-based robotics within the last decade, JEEE Access 8 (2020) 176598–176623

[32] R.K. Tan, Y. Liu, L. Xie, Reinforcement learning for systems pharmacologyoriented and personalized drug design, Expert Opin. Drug Discov. 17 (8) (2022) 849–863.

[33] J.C. Yokogawa Electric Corporation, ”a world’s first” is now routine – eneos materials successfully achieves autonomous control at a plant using ai, https:/ www.yokogawa.com/library/resources/references/successstory-eneos-mater ials/ [Accessed: 2024. 11. 26.] (2022).

[34] Y. Cui, L. Zhu, M. Fujisaki, H. Kanokogi, T. Matsubara, Factorial kernel dynamic policy programming for vinyl acetate monomer plant model control, in: Proceedings of the IEEE 14th International Conference on Automation Science and Engineering (CASE), IEEE, 2018, pp. 304–309.

[35] J.C. Yokogawa Electric Corporation, In a world first, yokogawa and jsr use ai to autonomously control a chemical plant for 35 consecutive days, https://www.vok ogawa.com/hu/news/press-releases/2022/2022-03-22/[Accessed: 2024. 11. 26.] (2022).

[36] Y. E. C. ENEOS Materials Corporation, In a world first, yokogawa’s autonomous control ai is officially adopted for use at an eneos materials chemical plant, https ://www.yokogawa.com/hu/news/press-releases/2023/2023-03-30/[Accessed: 2025. 01. 13.] (2023).

[37] B.G. Liptak, M.J. Piovoso, F.G. Shinskey, H. Eren, G.K. Totherow, J.E. Jamison D. Morgan, H.I. Hertanu, E.M. Marszal, J. Berge, et al., Instrument Engineers Handbook, Volume Two: Process Control and Optimization, CRC Press, 2018.

[38] R.P. Borase, D. Maghade, S. Sondkar, S. Pawar, A review of pid control, tuning methods and applications, Int. J. Dyn. Control 9 (2021) 818–827.

[39] M. Tka´ˇcik, J. Jadlovský, S. Jadlovska,´ A. Jadlovska,´ T. Tka´ˇcik, Modeling and analysis of distributed control systems: proposal of a methodology, Processes 12 (1) (2023) 5.

[40] M. Schwenzer, M. Ay, T. Bergs, D. Abel, Review on model predictive control: an engineering perspective, Int. J. Ady, Manuf, Technol, 117 (5) (2021) 1327–1349.

[41] J. Oravec, M. Bakoˇsova,´ P. Artzova,´ Advanced process control design for a distillation column using unisim design, in: Proceedings of the 21st International Conference on Process Control (PC), IEEE, 2017, pp. 303–308.

[42] J.C.D. Toledo, F.L. Lizarelli, M.B. Santana, Success factors in the implementation of statistical process control: action research in a chemical plant. Production 27 (2017) e20162208.

[43] B. Chachuat, B. Srinivasan, D. Bonvin, Adaptation strategies for real-time optimization, Comput. Chem. Eng. 33 (10) (2009) 1557–1567.

[44] W. Zhu, R. Rendall, I. Castillo, Z. Wang, L.H. Chiang, P. Hayot, J.A. Romagnoli, Control of a polyol process using reinforcement learning, IFAC-PapersOnLine 54

[45] H. Wang, L.A. Ricardez-Sandoval. A deep reinforcement learning-based pic tuning strategy for nonlinear mimo systems with time-varying uncertainty, IFAC PapersOnLine 58 (14) (2024) 887–892.

[46] O. Dogru, K. Velswamy, F. Ibrahim, Y. Wu, A.S. Sundaramoorthy, B. Huang, S. Xu, M. Nixon, N. Bell. Reinforcement learning approach to autonomous pid tuning, Comput. Chem. Eng. 161 (2022) 107760.

[47] S. Mate, P. Pal, A. Jaiswal, S. Bhartiya, Simultaneous tuning of multiple pid controllers for multivariable systems using deep reinforcement learning, Digit Chem Eng 9 (2023) 100131

[48] E.H. Bras, T.M. Louw, S.M. Bradshaw, Safe, visualizable reinforcement learning for process control with a warm-started actor network based on pi-control, J. Process Control 144 (2024) 103340.

[49] A. Syauqi, H. Kim, H. Lim, Optimizing olefin purification: an artificia intelligencebased process-conscious Pi controller tuning for double dividing wall column distillation, Chem. Eng, J. 500 (2024) 156645

[50] M.A. Chowdhury, S.S. Al-Wahaibi, Q. Lu, Entropy-maximizing td3-based reinforcement learning for adaptive PID control of dynamical systems, Comput. Chem. Eng. 178 (2023) 108393.

[51] K.P. Detroja, et al., Gain scheduled pi controller design using multi-objective

[52] H. Shah, M. Gopal, Reinforcement learning framework for adaptive control of nonlinear chemical processes, Asia-Pac. J. Chem. Eng. 6 (1) (2011) 138–146.

[53] B.J. Pandian. M.M. Noel. Tracking control of a continuous stirred tank reactor using direct and tuned reinforcement learning based controllers. Chem. Prod. Process Model, 13 (3) (2018) 20170040

[54] X.S. Wang, Y.H. Cheng, S. Wei, A proposal of adaptive pid controller based on reinforcement learning, J. China Univ. Min. Technol. 17 (1) (2007) 40–44.

[55] Y. Bao, Y. Zhu, F. Qian, A deep reinforcement learning approach to improve the learning performance in process control, Ind. Eng. Chem. Res. 60 (15) (2021) 5504–5515.

[56] Y. Ma, W. Zhu, M.G. Benton, J. Romagnoli, Continuous control of a polymerization system with deep reinforcement learning, J. Process Control 75 (2019) 40–47.

[57] N. Ballard, K. Farajzadehahary, S. Hamzehlou, U. Mori, J.M. Asua, Reinforcement learning for the optimization and online control of emulsion polymerization reactors: particle morphology, Comput. Chem. Eng. 187 (2024) 108739.

[58] I.K. Faridi, E. Tsotsas, A. Kharaghani, Advancing process control in fluidized bed biomass gasification using model-based deep reinforcement learning, Processes 12 (2) (2024) 254.

[59] N. Bougie, T. Onishi, Y. Tsuruoka, Data-efficient reinforcement learning from controller guidance with integrated self-supervision for process control, IFAC PapersOnLine 55 (7) (2022) 863–868.

[60] V. Manee, R. Baratti, J.A. Romagnoli, Learning to navigate a crystallization model with deep reinforcement learning, Chem. Eng. Res. Des. 178 (2022) 111–123.

[61] V. Manee, R. Baratti, J.A. Romagnoli, Optimal strategies to control particle size and variance in antisolvent crystallization operations using deep rl, Chem. Eng. Trans. 86 (2021) 943–948.

[62] S. Bi, B. Zhang, L. Mu, X. Ding, J. Wang, Optimization of tobacco drying process control based on reinforcement learning, Dry. Technol. 38 (10) (2020) 1291–1299.

[63] H. Wang, Y. Guo, L. Li, S. Li, Development of ai-based process controller of sour water treatment unit using deep reinforcement learning, J. Taiwan Inst. Chem. Eng. 157 (2024) 105407.

[64] H. Shafi, K. Velswamy, F. Ibrahim, B. Huang, A hierarchical constrained reinforcement learning for optimization of bitumen recovery rate in a primar separation vessel, Comput. Chem. Eng. 140 (2020) 106939.

[65] P. Ramanathan, K.K. Mangla, S. Satpathy, Smart controller for conical tank system using reinforcement learning algorithm, Measurement 116 (2018) 422–428.

[66] O. Dogru, N. Wieczorek, K. Velswamy, F. Ibrahim, B. Huang, Online reinforcement learning for a continuous space system with experimental validation, J. Process Control 104 (2021) 86–100

[67] S. Sachio, E. Antonio, P. Petsagkourakis, Simultaneous process design and control optimization using reinforcement learning, IFAC-PapersOnLine 54 (3) (2021) 510–515.

[68] S. Sachio, M. Mowbray, M.M. Papathanasiou, E.A. del Rio-Chanona, P. Petsagkourakis. Integrating process design and control using reinforcement learning, Chem. Eng. Res. Des. 183 (2022) 160–169.

[69] N.P. Lawrence, M.G. Forbes, P.D. Loewen, D.G. McClement, J.U. Backstrom,¨ R. B. Gopaluni. Deep reinforcement learning with shallow controllers: an experimental application to pid tuning, Control Eng. Pract. 121 (2022) 105046.

[70] S. Heo, T. Oh, T. Woo, S. Kim, Y. Choi, M. Park, J. Kim, C. Yoo, Real-scale demonstration of digital twins-based aeration control policy optimization in partial nitritation/anammox process: policy iterative dynamic programming approach, Desalination 593 (2025) 118235.

[71] J. Zhang, S. Fan, Z. Feng, L. Dong, Y. Dai, Supervised integrated deep deterministic policy gradient model for enhanced control of chemical processes, Chem. Eng. Sci. 301 (2025) 120762.

[72] S. Spielberg, A. Tulsvan, N.P. Lawrence. P.D. Loewen, R. Bhushan Gopaluni. Toward self-driving processes: a deep reinforcement learning approach to control. AIChE J. 65 (10) (2019) e16689.

[73] S.S.P. Kumar, B. Gopaluni, P. Loewen, Process control using deep reinforcement learning (2017).

[74] D. Dutta, S.R. Upreti. A multiple neural network and reinforcement learning. based strategy for process control, J. Process Control 121 (2023) 103–118

[75] D. Dutta, S.R. Upreti, A reinforcement learning-based transformed inverse model strategy for nonlinear process control, Comput. Chem. Eng. 178 (2023) 108386.

[76] S. Syafiie, F. Tadeo, E. Martinez, Model-free learning control of neutralization processes using reinforcement learning, Eng. Appl. Artif. Intell. 20 (6) (2007) 767–782.

[77] D.A. Goulart, R.D. Pereira, Autonomous ph control by reinforcement learning for electroplating industry wastewater, Comput. Chem. Eng. 140 (2020) 106909.

[78] J.W. Kim, B.J. Park, T.H. Oh, J.M. Lee, Model-based reinforcement learning and predictive control for two-stage optimal control of fed-batch bioreactor, Comput. Chem, Eng, 154 (2021) 107465

[79] H. Shah, M. Gopal, Model-free predictive control of nonlinear processes based on reinforcement learning, IFAC-PapersOnLine 49 (1) (2016) 89–94.

[80] D. Beahr, D. Bhattacharyya, D.A. Allan, S.E. Zitney, Development of algorithms for augmenting and replacing conventional process control using reinforcement learning, Comput. Chem. Eng. 190 (2024) 108826.

[81] V. Rajpoot, S. Munusamy, T. Joshi, D. Patil, V. Pinnamaraju, Comparison of reinforcement learning and model predictive control for a nonlinear continuous process, IFAC-PapersOnLine 57 (2024) 304–308

[82] T. Joshi, H. Kodamana. H. Kandath, N. Kaisare, Tasac: a twin-actor reinforcement learning framework with a stochastic policy with an application to batch process control, Control Eng. Pract. 134 (2023) 105462.

[83] V. Singh, H. Kodamana, Reinforcement learning based control of batch polymerisation processes, IFAC-PapersOnLine 53 (1) (2020) 667–672.

[84] K. Alhazmi, S.M. Sarathy. Continuous control of complex chemical reaction network with reinforcement learning, in: Proceedings of the European Control Conference (ECC), IEEE, 2020, pp. 1066–1068

[85] R. Lin, J. Chen, L. Xie, H. Su, Accelerating reinforcement learning with case-based model-assisted experience augmentation for process control, Neural Netw. 158 (2023) 197–215.

[86] J. Hoskins, D. Himmelblau, Process control via artificial neural networks and reinforcement learning, Comput. Chem. Eng. 16 (4) (1992) 241–251.

[87] A.<sup>´</sup> Sass, A. Kummer, J. Abonyi, Multi-agent reinforcement learning-based exploration of optimal operation strategies of semi-batch reactors, Comput Chem. Eng. 162 (2022) 107819.

[88] M. Yu, B. Li, S. Zhao, N. Roy, B. Zhang, PPO-based resilient control framework for safer operation of exothermic CSTR. Process Saf. Environ. Prot. 193 (2025) 558–576.

[89] H. Xie, X. Xu, Y. Li, W. Hong, J. Shi, Model predictive control guided reinforcement learning control scheme, in: Proceedings of the International Joint Conference on Neural Networks (IJCNN), IEEE, 2020, pp. 1–8.

[90] E.M.L. Luz, W. Caarls, Comparison of reinforcement learning techniques for controlling a CSTR process, Braz. J. Chem. Eng. (2023) 1–12.

[91] T.H. Oh, Quantitative comparison of reinforcement learning and data-driven model predictive control for chemical and biological processes, Comput. Chem Eng. 181 (2024) 108558.

[92] M. Mowbray, R. Smith, E.A. Del Rio-Chanona, D. Zhang, Using process data to generate an optimal control policy via apprenticeship and reinforcement learning, AIChE J. 67 (9) (2021) e17306.

[93] Y. Wang, Z. Wu, Physics-informed reinforcement learning for optimal control of nonlinear systems, AIChE J. 70 (10) (2024) e18542.

[94] R.R. Faria, B. Capron, A.R. Secchi, M.B. de Souza Jr, A data-driven tracking control framework using physics-informed neural networks and deep reinforcement learning for dynamical systems, Eng. Appl. Artif. Intell. 127 (2024) 107256.

[95] H. Liang, C. Yang, M. Lv, X. Zhang, Z. Feng, Y. Li, B. Sun, Zinc roasting temperature field control with CFD model and reinforcement learning, Adv. Eng. Inform, 59 (2024) 102332

[96] R. Lin, Y. Luo, X. Wu, J. Chen, B. Huang, H. Su, L. Xie, Surrogate empowered Sim2Real transfer of deep reinforcement learning for orc superheat control, Appl. Energy 356 (2024) 122310.

[97] X. Hong, Z. Shou, W. Chen, Z. Liao, J. Sun, Y. Yang, J. Wang, Y. Yang, A reinforcement learning-based temperature control of fluidized bed reactor in gas-phase polyethylene process, Comput. Chem. Eng. 183 (2024) 108588.

[98] M.S.F. Bangi, J.S.I. Kwon, Deep reinforcement learning control of hydraulic fracturing, Comput. Chem. Eng. 154 (2021) 107489.

[99] L. Zhu, Y. Cui, G. Takami, H. Kanokogi, T. Matsubara, Scalable reinforcement learning for plant-wide control of vinyl acetate monomer process, Control Eng. Pract, 97 (2020) 104331.

[1oo] K. Alhazmi, S.M. Sarathy, Direct learning of improved control policies from historical plant data, Comput. Chem. Eng. 185 (2024) 108662.

[1011 D Machalek T Ouah. K M Powell Dynamic economic optimization of a continuously stirred tank reactor using reinforcement learning, in: Proceedings of the American Control Conference (ACC), IEEE, 2020, pp. 2955–2960.

[102] P. de Azevedo Delou. L. Ferreira Bernardino. B.D.O. Capron. A. Resende Secchi A comparison between process control strategies: reinforcement learning with rbfs and nmpc coupled with ekf, Braz. J. Chem. Eng. (2023) 1–14.

[103] G. Cassol, G. Campos, D. Thomaz, B. Capron, A. Secchi, Reinforcement learning applied to process control: a van der vusse reactor case study, Comput. Aided Chem. Eng. 44 (2018) 553–558.

[104] D. Machalek, T. Ouah. K.M. Powell. A novel implicit hybrid machine learning model and its application for reinforcement learning, Comput, Chem. Eng. 155 (2021) 107496.

[105] H. Hassanpour, P. Mhaskar, B. Corbett, A practically implementable reinforcement learning control approach by leveraging offset-free mode predictive control, Comput. Chem. Eng. 181 (2024) 108511.

[106] S. Hwangbo, G. Sin, Design of control framework based on deep reinforcement learning and monte-carlo sampling in downstream separation, Comput. Chem. Eng 140 (2020).106910

[107] Z. Zhang, S. Li, Enhanced reinforcement learning in two-layer economic model predictive control for operation optimization in dynamic environment, Chem. Eng. Res. Des. 196 (2023) 133–143.

[108] R. Jia, M. Zhang, J. Zheng, D. He, F. Chu, K. Li, Offline constrained reinforcement learning for batch-to-batch optimization of cobalt oxalate synthesis process, Chem, Eng, Res, Des, 209 (2024) 334–345

[109] D. Brandner, S. Lucia, Optimizing operation recipes with reinforcement learning for safe and interpretable control of chemical processes, https://ml4cce-ecml.com /papers/200.pdf [Accessed: 2025. 04. 10.].

[110] M. Mowbray, P. Petsagkourakis, E.A. del Rio-Chanona, D. Zhang, Safe chance constrained reinforcement learning for batch process control, Comput. Chem. Eng. 157 (2022) 107630.

[111] D. Li, F. Zhu, X. Wang, Q. Jin, Multi-objective reinforcement learning for fed batch fermentation process control., J. Process Control 115 (2022) 89–99

[112] E. Pan, P. Petsagkourakis, M. Mowbray, D. Zhang, E.A. del Rio-Chanona, Constrained model-free reinforcement learning for process optimization, Comput. Chem, Eng, 154 (2021) 107462.

[113] P. Petsagkourakis, I.O. Sandoval, E. Bradford, D. Zhang, E.A. del Rio-Chanona, Reinforcement learning for batch bioprocess optimization, Comput, Chem. Eng 133 (2020).106649

[114] H. Li. T. Oiu. E. You, Ai-based optimal control of fed-batch biopharmaceutical process leveraging deep reinforcement learning, Chem. Eng. Sci. 292 (2024) 119990.

[115] D. Li, W. Gu, T. Song, Multi-objective reinforcement learning in process control: a goal-oriented approach with adaptive thresholds, J. Process Control 129 (2023) 103063.

[116] P. Petsagkourakis, I.O. Sandoval, E. Bradford, D. Zhang, E.A. del Rio-Chanona, Constrained reinforcement learning for dynamic optimization under uncertainty, JFAC-PapersOnLine 53 (2) (2020) 11264–11270

[117] P. Petsagkourakis, I.O. Sandoval, E. Bradford, D. Zhang, E.A. del Rio-Chanona, Reinforcement learning for batch-to-batch bioprocess optimisation. Comput. Aided Chem. Eng. 46 (2019) 919–924.

[118] W.Y. Chai, M.K. Tan, K.T.K. Teo, H.J. Tham, Optimization of fed-batch baker’s yeast fermentation using deep reinforcement learning, Process Integr. Optim. Sustain. 8 (2) (2024) 395–411.

[119] P. Zhang, J. Zhang, Y. Long, B. Hu, An improved reinforcement learning control strategy for batch processes, in: Proceedings of the 24th International Conference on Methods and Models in Automation and Robotics (MMAR), IEEE, 2019, pp. 360–365.

[120] H.E. Byun, B. Kim, J.H. Lee, Multi-step lookahead bayesian optimization with active learning using reinforcement learning and its application to data-driven batchto-batch optimization, Comput. Chem. Eng. 167 (2022) 107987.

[121] Y. Ma, D.A. Norena-Caro,˜ A.J. Adams, T.B. Brentzel, J.A. Romagnoli, M. G. Benton, Machine-learning-based simulation and fed-batch control of cyanobacterialphycocyanin production in plectonema by artificial neural network and deep reinforcement learning, Comput. Chem. Eng. 142 (2020) 107016.

[122] H. Yoo, B. Kim, J.H. Lee, A phase segmentation approach for applying reinforcement learning to batch polymerization process control, in: Proceedings of the IFAC 2020, International Federation of Automatic Control, 2020.

[123] H. Yoo, B. Kim, J.W. Kim, J.H. Lee, Reinforcement learning based optimal control of batch processes using monte-carlo deep deterministic policy gradient with phase segmentation, Comput. Chem. Eng. 144 (2021) 107133.

[124] H.C. Croll, K. Ikuma, S.K. Ong, S. Sarkar, Unified control of diverse actions in a wastewater treatment activated sludge system using reinforcement learning for multi-obiective optimization. Water Res. 263 (2024) 122179

[125] Z. Klawikowska, M. Grochowski, Optimizing Control of Wastewater Treatment Plant With Reinforcement learning: Technical evaluation of Twin-Delayed Deep Deterministic Policy Gradient Agent. JEEE Access. 2024

[126] Y. Yao, J. Ding, C. Zhao, Y. Wang, T. Chai, Data-driven constrained reinforcement learning for optimal control of a multistage evaporation process, Control Eng. Pract, 129 (2022) 105345.

[127] F. Elmaz, U. Di Caprio, M. Wu, Y. Wouters, G. Van Der Vorst, N. Vandervoort, A. Anwar, M.E. Leblebici, P. Hellinckx, S. Mercelis, Reinforcement learning-based approach for optimizing solvent-switch processes, Comput. Chem. Eng. 176 (2023) 108310.

[130] K.M. Patel, A practical reinforcement learning implementation approach for continuous process control. Comput, Chem. Eng, 174 (2023) 108232.

[129] C.Y. Lee, Y.T. Huang, P.J. Chen, Robust-optimization-guiding deep reinforcement learning for chemical material production scheduling. Comput. Chem. Eng. 187 (2024) 108745.

[128] C.D. Hubbs, C. Li, N.V. Sahinidis, I.E. Grossmann, J.M. Wassick, A deep reinforcement learning approach for chemical production scheduling. Comput Chem Eng 141 (2020).106982

[131] K.M. Powell, D. Machalek, T. Quah, Real-time optimization using reinforcement learning, Comput, Chem. Eng. 143 (2020) 107077.

[132] S. Kannan, U. Diwekar, An efficient reinforcement learning approach to optimal control with application to biodiesel production. Comput. Chem. Eng, 174 (2023) 108258.

[133] L. Zhu, G. Takami, M. Kawahara, H. Kanokogi, T. Matsubara, Alleviating parameter-tuning burden in reinforcement learning for large-scale process control, Comput. Chem. Eng. 158 (2022) 107658.

[134] L. Li, X. Yang, S. Yang, X. Xu, Optimization of oxygen system scheduling in hybrid action space based on deep reinforcement learning, Comput. Chem. Eng. 171 (2023) 108168.

[135] C.D. Hubbs, A. Kelloway, J.M. Wassick, N.V. Sahinidis, I.E. Grossmann, An industrial application of deep reinforcement learning for chemical production scheduling, in: Proceedings of the Machine Learning for Engineering Modeling Simulation, and Design Workshop at Neural Information Processing Systems, Vancouver, Canada 12. 2020.

[136] S.C. van Kalmthout, L.I. Midgley, M.B. Franke, Synthesis of separation processes with reinforcement learning, arXiv preprint arXiv:2211.04327 (2022).

[137] D. Rangel-Martinez, L.A. Ricardez-Sandoval, A recurrent reinforcement learning strategy for optimal scheduling of partially observable job-shop and flow-shor batch chemical plants under uncertainty, Comput. Chem. Eng. (2024) 108748.

[138] T. Quah, D. Machalek, K.M. Powell, Comparing reinforcement learning methods for real-time optimization of a chemical process, Processes 8 (11) (2020) 1497.

[139] Z. Shao, F. Si, D. Kudenko, P. Wang, X. Tong, Predictive scheduling of wet flue gas desulfurization system based on reinforcement learning, Comput. Chem. Eng. 141 (2020) 107000.

[140] M. Huang, R. He, X. Dai, W. Du, F. Qian, Reinforcement learning based gasoline blending optimization: achieving more efficient nonlinear online blending of fuels, Chem. Eng, Sci, 300 (2024) 120574.

[141] Z. Zhu, M. Yang, W. He, R. He, Y. Zhao, F. Qian, A deep reinforcement learning approach to gasoline blending real-time optimization under uncertainty, Chin. J. Chem. Eng. (2024).

[142] G. Che, Y. Zhang, L. Tang, S. Zhao, A deep reinforcement learning based multiobiective optimization for the scheduling of oxvgen production system in integrated iron and steel plants. Appl. Energy 345 (2023) 121332.

[143] F. Hern´andez-del Olmo, E. Gaudioso, R. Dormido, N. Duro, Tackling the start-up of a reinforcement learning agent for the control of wastewater treatment plants, Knowl. Based Syst. 144 (2018) 9–15.

[144] Q. Lan, Y. Pan, A. Fyshe, M. White, Maxmin q-learning: controlling the estimation bias of q-learning, arXiv preprint arXiv:2002.06487 (2020).

[145] H. Ali, H. Majeed, I. Usman, K.A. Almejalli, Reducing entropy overestimation in soft actor critic using dual policy network, Wirel. Commun. Mob. Comput. 2021 (1) (2021) 9920591.

[146] Z. Ahmed, N. Le Roux, M. Norouzi, D. Schuurmans, Understanding the impact of entropy on policy optimization, in: Proceedings of the International Conference on Machine Learning, PMLR, 2019, pp. 151–160.

[147] N. Rajasekhar, T. Radhakrishnan, N. Samsudeen, Exploring reinforcemen learning in process control: a comprehensive survey, Int. J. Syst. Sci. (2025) 1–30.

[148] S. Gu, L. Yang, Y. Du, G. Chen, F. Walter, J. Wang, A. Knoll, A review of safe reinforcement learning: methods, theory and applications, arXiv preprint arXiv: 2205.10330 (2022).

[149] Y. Wang, M. Xiao, Z. Wu, Safe transfer-reinforcement-learning-based optima control of nonlinear systems, IEEE Trans. Cybern. (2024).

[150] Y. Wang, Z. Wu, Control lyapunov-barrier function-based safe reinforcement learning for nonlinear optimal control, AIChE J. 70 (3) (2024) e18306.

[151] Y. Wang, Z. Wu, Machine learning model-based optimal tracking control of nonlinear affine systems with safety constraints, Int. J. Robust Nonlinear Control 35 (2) (2025) 511–535.

[152] T. Savage, D. Zhang, M. Mowbray, E.A.D.R. Chanona, Model-free safe reinforcement learning for chemical processes using gaussian processes, IFAC PapersOnLine 54 (3) (2021) 504–509.

[153] Z. Ning, L. Xie, A survey on multi-agent reinforcement learning and it application, J. Autom. Intell. (2024).

[154] L. Canese, G.C. Cardarilli, L. Di Nunzio, R. Fazzolari, D. Giardino, M. Re, S. Spano,\` Multi-agent reinforcement learning: a review of challenges and applications Appl. Sci. 11 (11) (2021) 4948.

[155] A. Oroojlooy, D. Hajinezhad, A review of cooperative multi-agent deep reinforcement learning, Appl, Intell, 53 (11) (2023) 13677–13722

[156] P.J. oen, K. Tuyls, L. Panait, S. Luke, J.A. La Poutre, An overview of cooperative and competitive multiagent learning, in: Proceedings of the Learning and Adaption in Multi-Agent Systems: First International Workshop. LAMAS 2005 Utrecht, The Netherlands, Springer, 2006, pp. 1–46. July 25, 2005, Revised Selected Papers.

[157] S. Gronauer, K. Diepold, Multi-agent deep reinforcement learning: a survey, Artif. Intell. Rev. 55 (2) (2022) 895–943.

[158] Q. Meng, P.D. Anandan, C.D. Rielly, B. Benyahia, Multi-agent reinforcement learning and rl-based adaptive pid control of crystallization processes, in: Computer Aided Chemical Engineering, 52. Elsevier. 2023, pp. 1667–1672

[159] R.R. Kumar, P. Varakantham, On solving cooperative marl problems with a few good experiences, arXiv preprint arXiv:2001.07993 (2020).

[160] Y. Yifei, S. Lakshminaravanan, Multi-agent reinforcement learning for proces control: exploring the intersection between fields of reinforcement learning, control theory, and game theory, Can. J. Chem. Eng, 101 (11) (2023) 6227–6239

[161] V. Gabler, D. Wollherr, Decentralized multi-agent reinforcement learning based on best-response policies, Front, Robot. AI 11 (2024) 1229026.

[162] S.V. Albrecht, F. Christianos, L. Schafer,¨ Multi-Agent Reinforcement Learning Foundations and Modern Approaches, MIT Press, 2024. URL, https://www.mar -book.com.

[163] P. Winder, Reinforcement Learning, O'Reilly Media. 2020

[164] P. Atrazhey, P. Musilek, It's all about reward: contrasting joint rewards and individual reward in centralized learning decentralized execution algorithms, Systems 11 (4) (2023) 180.

[165] A. Tampuu, T. Matiisen, D. Kodelja, I. Kuzovkin, K. Korjus, J. Aru, J. Aru, R. Vicente. Multiagent cooperation and competition with deep reinforcement learning, PLoS ONE 12 (4) (2017) e0172395.

[166] C. Zhu, M. Dastani, S. Wang, A survey of multi-agent deep reinforcement learning with communication, Auton. Agent Multi-Agent Syst. 38 (1) (2024) 4.

[167] N. Rajasekhar, T. Radhakrishnan, N. Samsudeen, Decentralized multi-agent control of a three-tank hybrid system based on twin delayed deep deterministic policy gradient reinforcement learning algorithm, Int. J. Dyn. Control 12 (4) (2024) 1098–1115.

[168] C. Amato, An introduction to centralized training for decentralized execution in cooperative multi-agent reinforcement learning, arXiv preprint arXiv:2409.03052 (2024).

[169] K. Chen, H. Wang, B. Valverde-P´erez, S. Zhai, L. Vezzaro, A. Wang, Optimal control towards sustainable wastewater treatment plants based on multi-agen reinforcement learning. Chemosphere 279 (2021) 130498

[170] L. Busoniu, R. Babuska, B. De Schutter, A comprehensive survey of multiagent reinforcement learning, IEEE Trans. Syst. Man Cybern. Part C (Appl. Rev.) 38 (2) (2008)156–172

[171] L. Bușoniu, R. Babuˇska, B. De Schutter, Multi-agent reinforcement learning: an overview. Innovations in multi-agent systems and applications-1, 2010, pp. 183–221.

[172] A.J. Singh. A. Kumar. H.C. Lau, Hierarchical multiagent reinforcement learning for maritime traffic management (2020).

[173] C. Spatharis, A. Bastas, T. Kravaris, K. Blekas, G.A. Vouros, J.M. Cordero Hierarchical multiagent reinforcement learning schemes for air traffic management, Neural Comput. Appl. (2023) 1–13.

[174] I. Jendoubi, F. Bouffard, Multi-agent hierarchical reinforcement learning for energy management. Appl. Energy 332 (2023) 120500.

[175] S. Studer, T.B. Bui, C. Drescher, A. Hanuschkin, L. Winkler, S. Peters, K.R. Müller, Towards CRISP-ML(Q): a machine learning process model with quality assurance methodology, Mach. Learn. Knowl. Extr. 3 (2) (2021) 392–413.

[176] S. Levine, A. Kumar, G. Tucker, J. Fu, Offline reinforcement learning: tutorial, review, and perspectives on open problems 5 (2020).

[177] A. Kumar, A. Zhou, G. Tucker, S. Levine, Conservative q-learning for offline reinforcement learning, Adv. Neural Inf. Process. Syst. 33 (2020) 1179–1191.

[178] Z. Li, K. Xu, L. Liu, L. Li, D. Ye, P. Zhao, Deploying offline reinforcement learning with human feedback, arXiv preprint arXiv:2303.07046 (2023).

[179] D. Abel, A. Barreto, B. Van Roy, D. Precup, H.P. van Hasselt, S. Singh, A definition of continual reinforcement learning, Ady. Neural Inf. Process. Syst. 36 (2023) 50377–50407.

[180] H. Liang, J. Xie, B. Huang, Y. Li, B. Sun, C. Yang, A novel sim2real reinforcement learning algorithm for process control, Reliab. Eng. Syst. Saf. 254 (2025) 110639.

[181] H. Zhong, Z. Wang, Y. Hao, Offline reinforcement learning based feeding strategy of ethylene cracking furnace, Comput. Chem. Eng. 192 (2025) 108864.

[182] S. Dey, T. Marzullo, X. Zhang, G. Henze, Reinforcement learning building control approach harnessing imitation learning, Energy AI 14 (2023) 100255.

[183] P.D. Anandan, C.D. Rielly, B. Benyahia, Optimal control policies of a crystallization process using inverse reinforcement learning, Comput. Aided Chem. Eng. 51 (2022) 1093–1098.

[184] O. Sobhani, F. Elmaz, M. Robeyn, J. Van den Hauwe, S.P. Gerdposhteh, B. Carius, K. Mets, S. Mercelis, A comparative study of data-driven offline reinforcement

learning for fed-batch process control, in: Computer Aided Chemical Engineering, 53, Elsevier, 2024, pp. 3157–3162.

[185] Y. Wu, J. Izawa, The regret motivated reinforcement learning, in: Proceedings of the 32nd International Symposium on Micro-NanoMechatronics and Human Science. JEEE. 2021, pp. 1–5.

[186] K. Szatm´ari, G. Horvath,´ S. N´emeth, W. Bai, A. Kummer, Resilience-based explainable reinforcement learning in chemical process safety, Comput. Chem Eng. 191 (2024) 108849.

[187] K.M. Patel, Safe, fast and explainable online reinforcement learning for continuous process control, in: Proceedings of the IEEE International Symposium on Advanced Control of Industrial Processes (AdCONIP), IEEE, 2022, pp. 54–60.

[188] V. François-Lave, P. Henderson, R. Islam, M.G. Bellemare, J. Pineau, et al., An introduction to deep reinforcement learning. Found. Trends® Mach. Learn. 11 (3–4) (2018) 219–354

[189] M. Ghasemi, A.H. Moosavi, I. Sorkhoh, A. Agrawal, F. Alzhouri, D. Ebrahimi, An introduction to reinforcement learning: fundamental concepts and practical applications, arXiv preprint arXiv:2408.07712 (2024).

[191] K. Arulkumaran, M.P. Deisenroth, M. Brundage, A.A. Bharath, Deep reinforcement learning: a brief survey, IEEE Signal Process. Mag. 34 (6) (2017) 26–38.

[190] C. Szepesv´ari. Reinforcement learning algorithms for MDPS, 2009.

[192] D. Silver, Lecture 1: Introduction to Reinforcement Learning, 1, Google DeepMind, 2015, p. 10.