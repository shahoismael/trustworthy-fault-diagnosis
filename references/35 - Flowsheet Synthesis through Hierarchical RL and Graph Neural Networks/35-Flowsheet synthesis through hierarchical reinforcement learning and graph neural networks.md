# Flowsheet synthesis through hierarchical reinforcement learning and graph neural networks

Laura Stops<sup>1,∗</sup>, Roel Leenhouts<sup>1,∗</sup>, Qinghe Gao<sup>1</sup>, Artur M. Schweidtmann<sup>1,∗∗</sup>

∗ contributed equally

<sup>1</sup> Delft University of Technology,

Department of Chemical Engineering,

Van der Maasweg 9,

Delft 2629 HZ,

The Netherlands

Abstract: Process synthesis experiences a disruptive transformation accelerated by digitization and artificial intelligence. We propose a reinforcement learning algorithm for chemical process design based on a state-of-the-art actor-critic logic. Our proposed algorithm represents chemical processes as graphs and uses graph convolutional neural networks to learn from process graphs. In particular, the graph neural networks are implemented within the agent architecture to process the states and make decisions. Moreover, we implement a hierarchical and hybrid decision-making process to generate flowsheets, where unit operations are placed iteratively as discrete decisions and corresponding design variables are selected as continuous decisions. We demonstrate the potential of our method to design economically viable flowsheets in an illustrative case study comprising equilibrium reactions, azeotropic separation, and recycles. The results show quick learning in discrete, continuous, and hybrid action spaces. Due to the flexible architecture of the proposed reinforcement learning agent, the method is predestined to include large action-state spaces and an interface to process simulators in future research.

Topical heading: Process systems engineering

Keywords: artificial intelligence, reinforcement learning, graph convolutional neural networks, process synthesis, graph generation

## 1 Introduction

The chemical industry is approaching a disruptive transformation towards a more sustainable and circular future <sup>1–3</sup>. As a major contributor to global emissions, tremendous changes are required and the chemical industry needs to face a paradigm shift <sup>1</sup>. This also requires rethinking regarding the conceptualization of novel processes<sup>2,4</sup>. Simultaneously, innovations are pushed by new possibilities due to emerging digital technologies. Digitization and in particular artificial intelligence (AI) ofer new possibilities for process design and therefore have the potential contribute to the transformation of chemical engineering <sup>1,3,5</sup>.

In the last decade, reinforcement learning (RL) has demonstrated its potential to solve complex decision-making problems, e.g., by showing human-like or even superhuman performance in a large variety of game applications <sup>6–8</sup>. RL is a subcategory of machine learning (ML) where an agent learns to interact with an environment based on trial-and-error <sup>9</sup>. Especially since 2016, when DeepMind’s AlphaGo<sup>10</sup> succeeded against a world-class player in the game Go, RL has attracted great attention. In recent developments, RL applications have proven to successfully compete with top-tier human players in even real-time strategy video games like StarCraft II <sup>11</sup> and Dota 2 <sup>12</sup>.

The accomplishments of RL in gaming have initiated significant developments in other research fields, including chemistry and chemical engineering. In process systems engineering, RL has been mainly applied to scheduling <sup>13,14</sup> and process control <sup>15–19</sup>. After first appearances of RL for process control in the early 1990s<sup>15</sup>, the development was pushed with the rise of deep RL in continuous control in games <sup>20</sup> and physical tasks <sup>21</sup>. Spielberg et al. <sup>16</sup> first transferred deep RL to chemical process control. In recent works, the satisfaction of joint chance constraints <sup>17</sup> and the integration of process control into process design tasks <sup>18,19</sup> via RL were considered.

In contrast to continuous process control tasks, RL in molecule design is characterized by discrete decisions, such as adding or removing atoms. Several methods use RL for the design of molecules with de sired properties <sup>22–26</sup>. First applications generate simplified molecular-input line-entry system (SMILES) strings using RL agents with pre-trained neural networks <sup>23,26</sup>. Zhou et al. <sup>24</sup> introduced a method solely based on RL, thereby ensuring chemical validity. Recently, RL based molecule design has been further enhanced in terms of exploration strategies <sup>27</sup> or by combining RL with orientation simulations <sup>28</sup>. In another approach, You et al. <sup>22</sup> introduced a graph convolutional policy network (GCPN) that represents molecules as graphs. It allows using graph neural networks (GNNs) to approximate the policy of the RL agent and to learn directly on the molecular graph. Using GNNs on molecule graphs to predict molecule properties <sup>29–32</sup> has also shown promising results besides RL. For example, Schweidtmann et al.<sup>29</sup> achieved competitive results for fuel property prediction by concatenating the output of a GNN into a molecule fingerprint and further passing it trough a multi-layer perceptron (MLP).

Graph representation and RL are also applied in other engineering fields. For example, Ororbia and Warn <sup>33</sup> represent design configurations of planar trusses as graphs in an RL optimization task.

Recently, important first steps have been made towards using RL to synthesize novel process flow sheets <sup>34–39</sup>. Midgley <sup>34</sup> introduced the ”Distillation Gym”, an environment in which distillation trains for non-azeotropic mixtures are generated by a soft-actor-critic RL agent and simulated in the open source process simulator COCO. The agent first decides whether to add a new distillation column to the intermediate flowsheet and subsequently selects continuous operating conditions. In an alternative approach to generate process flowsheets, Khan and Lapkin <sup>35</sup> presented a value-based agent that chooses the next action by assessing its value, based on previous experience. The agent operates within a hybrid action space, i.e., it makes discrete and continuous decisions. In a recent publication, Khan and Lapkin <sup>40</sup> introduced a hierarchical RL approach to process design, capable of designing more advanced process flowsheets, also including recycles. A higher level agent constructs process sections by choosing sub objectives of the process, such as maximizing the yield. Then, a lower level agent operates within these sections and chooses unit types and discretized parametric control variables that define unit conditions. Due to the discretization, the agent operates only in a discrete action space. As another approach to synthesize flowsheets with RL, G¨ottl et al. <sup>36</sup> developed a turn-based two-player-game environment called ”SynGameZero”. The interpretation of flowsheeting as a two-player game allowed them to reuse an established tree search RL algorithm from DeepMind <sup>8</sup>. Recently, G¨ottl et al. <sup>37</sup> enhanced their work by allowing for recycles and utilizing convolutional neural networks (CNNs) for processing large flowsheet matrices. Additionally, the company Intemic <sup>38</sup> has recently developed a ”flowsheet copilot” that generates flowsheets iteratively, embededded in a 1-player-game. Intemic ofers a web front-end in which raw materials and desired products can be specified. Then, a RL agents selects unit operations as discrete decisions using the economic value of the resulting process as objective. Furthermore, Plathottam et al. <sup>39</sup> introduced a RL agent that optimizes a solvent extraction process by selecting discrete and continuous

design variables within predefined flowsheets.

One major gap in the previous literature on RL for process synthesis is the state representation of flowsheets. We believe that a meaningful information representation is key to enable breakthroughs of AI in chemical engineering<sup>5</sup>. Previous works represent flowsheet in matrices comprising thermodynamic stream data, design specifications, and topological information <sup>37</sup>. However, we know from computer science research that passing such matrices through CNNs is limited as they can only operate on fixed grid topologies, thereby exploiting spatial but not geometrical features <sup>41</sup>. In contrast, graph convolutional neural networks (GCNs) handle diferently sized and ordered neighborhoods <sup>42</sup> with the topology becoming a part of the network’s input <sup>43</sup>. Since flowsheets are naturally represented as graphs with varying size and order of neighborhoods, GCNs can take their topological information into account. Another gap in the literature concerns the combination of multiple unit operation types, recycle streams and a larger, hybrid action space. While previous works proposed these promising techniques in individual contributions<sup>34–40</sup>, they have not yet been combined to a unified framework.

In this contribution, we represent flowsheets as graphs consisting of unit operations as nodes and streams as edges (c.f.<sup>44,45</sup>). The developed agent architecture features a flowsheet fingerprint, which is learned by processing flowsheet graphs in GNNs. Thereby, proximal policy optimization (PPO) <sup>46</sup> is deployed with modifications to learn directly on graphs and to allow for hierarchical decisions. In addition, we combine a hybrid action space, hierarchical actor-critic RL, and graph generation in a unified framework.

## 2 Reinforcement learning for process synthesis

In this section, we introduce the methodology and the architecture of the proposed method. To apply RL to process synthesis, the problem is first formulated as a Markov decision process (MDP) which is defined by the tuple $M = \{ S , A , T , R \}$ . A MDP consists of states $s \in S$ , actions $a \in A .$ , a transition model $T : S \times A \to S$ , and a reward function $R ^ { 9 }$ . In the considered problem, states are represented by flowsheets graphs, while actions comprise discrete and continuous decisions. More specifically, the discrete decisions consist of selecting a new unit operation as well as the location where it is added to the intermediate flowsheet. The continuous decisions are to define one or several specific continuous design variables per unit operation. For the environment, we implemented simple functions in Python to simulate the considered flowsheet. Finally, a reward is calculated and returned to the agent.

While most RL methods can be divided into value-based and policy-based approaches, actor-critic RL takes advantage of both concepts <sup>9</sup>. In contrast to value-based RL methods that cannot be easily adapted to continuous actions <sup>21,47</sup>, actor-critic approaches can learn policies for both, discrete and continuous action spaces and are thus also suitable for hybrid tasks <sup>48</sup>. Subsequently, several recent state-of-the-art policy optimization methods propose an actor-critic setup <sup>21,46–50</sup>. As shown in Figure 1, actor-critic agents consist of a critic that estimates the value function and an actor that decides for actions by approximating the policy<sup>9</sup>.

The RL framework presented in this work is derived from the actor-critic PPO algorithm by OpenAI<sup>46</sup>. In PPO, the objective function is clipped to prevent a collapse of the agent’s performance during training. To favor exploration, an entropy term <sup>51</sup> is added to the loss function. Additionally, the generalized estimation of the advantage $\hat { A } ^ { 5 2 }$ is used for updating the networks.

## 2.1 State representation

The main feature of the proposed method is the representation of the states by directed flowsheet graphs. This characteristic allows us to process the states in GNNs, thereby taking topological information into account.

Figure 2 demonstrates the graph representation of flowsheets. Feeds, products, and unit operations are represented by nodes, storing the type of unit operation and design variables. The edges include thermodynamic information about process streams, like temperature, molar flow, and molar fractions.

Intermediate flowsheets feature nodes of the type “undefined“. Whenever a new unit operation is added to the flowsheet, the resulting open streams are considered as such “undefined“ nodes. In subsequent steps, they represent possible locations for placing new unit operations. Consequently, adding a new unit operation practically means replacing an “undefined“ node with a defined one.

![](images/aa9344cdfc8dcd1bd0bd410c4df02a714bb94c4b295797df2e0f94f59f2b6db7.jpg)  
Figure 1: Agent-environment interaction in an actor-critic policy optimization approach for flowsheet synthesis. The agent approximates the policy and makes decisions. Meanwhile, the critic estimates the value of the environment’s state using the flowsheet graph, which is used to evaluate the agent’s decisions. Here, actor and critic both deploy graph convolutional neural networks.

![](images/41ecd5bc607ca7f39fecd41d94c50f34ebf0c5a244750bae98e6503f60fb2011.jpg)  
Figure 2: Example of a flowsheet displayed as a graph. Unit operations, feeds, and products are represented as nodes, whereas streams are represented as edges.

## 2.2 Agent

At the heart of the proposed RL method stands a hierarchical, hybrid actor-critic agent composed of multiple GNNs and MLPs. Its characteristics are introduced hereinafter.

## 2.2.1 Hierarchical, hybrid action space

The architecture of the agent is decisively afected by the considered hierarchical and hybrid action space. The decision-making process is illustrated in Figure 3. Every action consists of three levels of decisions: (i) select a location, (ii) add a new unit operation, and (iii) define a continuous design variable.

![](images/5fe250011f51d3ae9c8e1e7bb6e6d32451d6cd1f811ae913e14afc7e4a4940e4.jpg)  
Figure 3: Hierarchical decision levels of the agent, starting from an intermediate flowsheet. In the first level, the agent selects a location where the flowsheet will be extended. Possible locations are open streams, represented by “undefined“ nodes. In the presented flowsheet, both streams leaving the column can be chosen. Then, the agent selects a unit operation. Thereby, the options are to add a heat exchanger, a reactor, a column, a recycle or to sell the stream as a product. Finally, a continuous design variable is selected for each unit operation. This third decision depends on which unit operation was selected previously.

In the first level, the agent decides for an open stream and thus for the location of the next flowsheet expansion. As discussed in Section 2.1, open streams are identified by “undefined“ nodes. In the second level, the agent decides which type of unit operation will be added. Thereby, the agent can choose to add a distillation column, a heat exchanger, or a reactor. Furthermore, it can decide to add a recycle by introducing a splitter and a mixer into the flowsheet. As a fifth option, the agent can declare the considered stream as a product. If a unit operation is added, the third level decision is to specify the design variables of the corresponding unit operation. Although it is possible to set multiple design variables in this step, we chose to only set one variable for simplification reasons. Thus, one characteristic variable for each unit operation is defined in this step while all other variables are fixed. For the current implementation of the agent, the recycle stream is always inserted into the feed stream. Whereas the first two levels are discrete decisions, the third level decisions are continuous. This combination of discrete and continuous decisions is referred to as hybrid action space.

## 2.2.2 Using GNNs to generate flowsheet fingerprints

In RL, every iteration of the agent-environment-interaction starts with the observation of the envi ronment’s state s, as shown in Figure 1. In other approaches <sup>34,36,37,40</sup>, states or rather flowsheets are represented by vectors or matrices and, e.g., passed through CNNs for the observation step <sup>37</sup>. Instead, in the herein presented approach, states are represented by flowsheet graphs (cf. Section 2.1). To observe and process the therein stored information, the flowsheet graphs are passed through GCNs and encoded into a vector format called flowsheet fingerprint. The advantage of using graphs and GCNs is that it allows operating in variable neighborhoods with diferent numbers and ordering of nodes, thereby taking spatial and spectral information into account <sup>41–43</sup>. Thus, we believe that graphs and GCNs are better suited for representing and processing the branched connectivity of flowsheets than passing matrices through CNNs.

For this step, we transfer the method introduced by Schweidtmann et al. <sup>29</sup>, who apply GNNs to generate molecule fingerprints, to flowsheets. The approach utilizes the message passing neural network

(MPNN) proposed by Gilmer et al. <sup>30</sup>.

The overall scheme to process a flowsheet graph is displayed in Figure 4 and consists of a message passing and a readout phase. First, the flowsheet graph is processed through an MPNN, using a GCN with several layers to exchange messages and update node embeddings. Afterward, a pooling function generates a vector format, the flowsheet fingerprint, in the readout phase. After several steps of message passing, sum-pooling is deployed for the subsequent readout phase. Thereby, the node embeddings of the last layer are concatenated into a vector format, the flowsheet fingerprint.

![](images/06fa7528d5aea47f42935912ab57675d92029ce264bf03c37f24f6bd134a697c.jpg)  
Figure 4: Flowsheet fingerprint generation derived from Schweidtmann et al. <sup>29</sup>. The flowsheet graph is processed through an MPNN, using GCNs to perform message passing and update node embeddings. In the readout step, a pooling function is applied, resulting in a vector format, the flowsheet fingerprint.

For every step in the message passing phase, first the node and edge features of the neighborhood of each node in the flowsheet graph are processed. Therefore, GCNs are utilized to exchange and update information in the message passing phase. The functionality of a graph convolutional layer is illustrated in Figure 5, following Schweidtmann et al. <sup>29</sup>. The figure visualizes the procedure to update the node embeddings of the blue node. Therefore, the information stored in the yellow neighboring nodes and the corresponding edges is processed and combined to a message through the message function M. Then, the considered node is updated through the message in the update function U. In each layer of a GCN, this procedure is conducted for every node of the graph.

## 2.2.3 Hierarchical agent architecture

For the architecture of the agent, a structure suggested by Fan et al. <sup>48</sup> for hierarchical and hybrid action spaces is used. Thereby, individual MLPs are applied for each level of decisions and one MLP is applied as a critic to evaluate the decisions.

The architecture of the actor-critic approach is illustrated in Figure 6. In the “fingerprint generation” step, the state represented by a flowsheet graph is processed to a flowsheet fingerprint through a GCN (cf. Section 2.2.2). Additionally, the updated graph resulting from the message passing phase of the fingerprint generation is passed to the “actor” step. Therein, the updated graph is further processed by an additional GCN. This represents the first level of the actor which is to select an open stream to further extend the flowsheet. Thereby, the method takes advantage of the graph representation in which open streams end in “undefined“ nodes. In the GCN of the first level decision, the number of node features is reduced to one (cf. related literature on node classification tasks <sup>42</sup>). Furthermore, all nodes which do not correspond to open streams are filtered out. The remaining node feature of each nodes in the last GCN layer represents its probability to be chosen as the location for adding a new unit. Then, the ID of the selected node is concatenated with the previously computed flowsheet fingerprint before it is passed on to the second and third level actors as input.

The second level actor consists of a MLP that returns probabilities for each unit operation to be chosen. For each type of unit operation, an individual MLP is set up as the actor for the third level decision. Thereby, the third level MLPs take the concatenated vector including the flowsheet fingerprint and the ID of the selected location as an input. They return two outputs which are interpreted as parameters, α and $\beta ,$ describing a beta distribution $B \left( \stackrel { \cdot } { \alpha } , \beta \right) ^ { 5 3 }$ . Based on this distribution, a continuous decision regarding the respective design variable is made

![](images/5e40c8586bb14c5e57a801f4103971eff4d0b9884a272da3ca90cab1c0a6470f.jpg)  
Figure 5: Update of the node embeddings during the message passing phase in a graph convolutional layer. The considered node is marked in blue and its neighbors in yellow. First, the information stored in the neighboring nodes and the respective edges is processed and combined through a message function M. Then, a message is generated to update the information embedded in the considered node through the update function U. The approach and its illustration follow a method proposed by Schweidtmann et al. <sup>29</sup>.

![](images/00a70f0b1ca9ec436d26237b0d476247b0bf74f0e18749deb6cc9313153b38f6.jpg)  
Figure 6: Architecture of the deployed actor-critic agent. First, a GNN is used to process the graph representation of the flowsheet into a flowsheet fingerprint. While the critic estimates the value of the fingerprint in one linear MLP, the actor takes three levels of decisions. The first decision is to choose a location for expanding the flowsheet. Practically, this means selecting the ID of a node representing an open stream. The selected node ID is combined with the flowsheet fingerprint and passed through an MLP for the second level decision of choosing a type of unit operation. Finally, a continuous design variable of the unit is chosen. Thereby, a diferent MLP is used for each unit type.

The critic that estimates the value of the original state is displayed in the upper half of Figure 6. Therefore, the flowsheet fingerprint is passed through another MLP. This value is an estimation of how much reward is expected to be received by the agent until the end of an episode when starting at the considered state and further following the current policy <sup>9</sup>. In our approach, we utilize the value to compute the generalized advantage estimation A<sup>ˆ</sup> introduced by Schulman et al. <sup>52</sup>. It tells whether an action performed better or worse than expected and is used to calculate losses of the actor’s networks. By comparing the value to the actual rewards, an additional loss is computed for the critic

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Pseudocode of the agent-environment interaction.

done = False

while not done do

    observe state s

    actions a, probs p, value v = AGENT(s)

    new state s', reward r, done = ENV(a)

    store transition (s, a, p, r, done) in memory

end while

function AGENT(state s)

    for level=1,2,3 do

    probs  $p_{level}$  = actor(s)

    action  $a_{level}$  = sample( $p_{level}$ )

    end for

    value v = critic(s)

    return a, p, v

end function

function ENV(actions a)

    next state  $s'$  = SimulateFlowsheet(a)

    if no more open streams then

    done = True

    reward r = NetCashFlow( $s'$ )

    if reward r &lt; 0 then

    reward r = reward r / 10

    end if

    else

    reward r = 0 €

    end if

    return  $s'$ , r, done

end function
</div>

## 2.3 Agent-environment interaction

The interaction between the environment and the hierarchical actor-critic agent is further clarified in Algorithm 1. After the environment is initialized with a feed, the flowsheet is generated in an iterative scheme. The agent first observes the current state s of the environment and chooses actions a for all three hierarchical decision levels by sampling. The agent returns the probabilities and the selected actions as well as the value v of the state.

In the next step, the actions are applied to the environment. Therefore, the next state $s ^ { \prime }$ is computed by simulating the extended flowsheet. Additionally, the environment checks whether any open stream is left in the flowsheet, indicating that the episode is still to be completed. Since the weights of the agent’s networks are randomly initialized, early training episodes can result in very large flowsheets. Thus, the total number of units is limited to 25 as additional guidance. If a flowsheet exceeds this number, all open streams are declared as products.

Additionally, the environment calculates the reward that depends on whether the flowsheet is com pleted or not. If the net cash flow is positive, the reward equals the net cash flow. If the net cash flow is negative, the reward equals the net cash flow divided by a factor 10. This procedure is implemented in order to encourage exploration of the agent. For the intermediate steps during the synthesis, process rewards of zero are given to the agent. After each iteration, the transition is stored in a batch and later used for batch learning.

## 2.4 Training

The presented method, including the flowsheet simulations, is implemented in Python 3.9. The training procedure is adapted from PPO by OpenAI <sup>46</sup>. It consists of multiple epochs of minibatch updates, whereby the minibatches result from sampling on the transition tuples stored in the memory. The agent’s networks are thereby updated by gradient descent, using a loss function derived from summing up and weighting all losses of the individual actors, their entropies, and the loss of the critic.

## 3 Case study

The proposed method is demonstrated in an illustrative case study considering the production of methyl acetate (MeOAc), a low-boiling liquid often used as a solvent <sup>54</sup>. In an industrial setting, MeOAc is primarily produced in reactive columns by esterification of acetic acid $\mathrm { ( H O A c ) ^ { 5 5 , 5 6 } }$ . For illustration, we consider only simplified flowsheets that use separate units for reaction and separation.

## 3.1 Process simulation

For computing new states and rewards, the flowsheets generated by the agent are simulated in Python. Therefore, we implemented a model for each type of unit operation that can be selected in the second level decision. In our case study, the agent can decide to place reactors, distillation columns, and heat exchangers. Furthermore, the agent can add recycles or sell open streams as products.

Reactor. The reactor is modeled as a plug flow reactor (PFR), in which the reversible equilibrium reaction shown in Equation 1 takes place.

$$
\mathrm{HOAc} + \mathrm{MeOH} \rightleftharpoons \mathrm{MeOAc} + \mathrm{H} _ {2} \mathrm{O}\tag{1}
$$

MeOAc and its by-product water $\mathrm { ( H _ { 2 } O ) }$ are produced by esterification of HOAc with methanol (MeOH) under the presence of a strong acid. To calculate the composition of the process stream leaving the PFR, we formulated a boundary value problem, depending on the reaction rate, and manually implemented a fourth-order Runge-Kutta method with fixed step-size as solver. Thereby, the reactor is modeled isothermal, based on the temperature of the inflowing stream. The reaction kinetics are based on Xu and Chuang<sup>57</sup>.

The length of the PFR is specified by the agent as the continuous third level decision within the range of 0.05 m to 20 m. Thereby, the relation of the cross-sectional area A of the PFR to the molar flow N<sup>˙</sup> passing through it is fixed to $A / \dot { N } = 0 . 1 \mathrm { m ^ { 2 } s m o l ^ { - 1 } }$ . Notably, the length of the reactor significantly influences the conversion in the PFR. In addition, the equilibrium of the considered reaction depends on the temperature of the process stream which thus afects the reaction rate and the conversion in the PFR. Thereby, the temperature of the process stream can be influenced by heat exchangers upstream of the reactor.

Heat exchanger. In the heat exchanger, heat is transferred between the process stream and a water stream. The continuous third level decision specifies the inlet temperature of the water and thus also whether the process stream is cooled or heated. To avoid evaporation of the process stream, the inlet water temperature is chosen within the range of $5 ~ ^ { \circ } \mathrm { C }$ to $5 3 . 8 ^ { \circ } \mathrm { C }$ , where the upper limit corresponds to the lowest possible boiling point of the considered quarternary system. The heat exchanger model computes the heat duty, the required heat transfer area, and the outlet temperature of the process stream. The model is based on a countercurrent flow, shell and tube heat exchanger <sup>58</sup>. A typical heat transfer coeficient of 568 W $\mathrm { K ^ { - 1 } m ^ { 2 } }$ is used <sup>59</sup>. Additionally, we assume that the process stream always approaches the water stream temperature within 5 K in the heat exchanger.

Distillation column. The distillation column is deployed to separate the quarternary system $\mathrm { M e O A c } ,$ MeOH, HOAc, and $_ \mathrm { H _ { 2 } O }$ . The vapor-liquid equilibrium of the system is displayed in Figure 7. It contains two binary minimum azeotropes between MeOAc and $\mathrm { H _ { 2 } O } .$ , and respectively between MeOAc and MeOH. As shown in Figure 7, the azeotropes split up the separation task into two distillation regimes. To simplify the problem, we follow the assumption made by G¨ottl et al. <sup>37</sup> that the distillation boundary can be approximated by the simplex spanned between both azeotropes and the fourth component, HOAc.

We implemented a shortcut column model using the $\infty / \infty$ analysis<sup>60–62</sup>. The only remaining degree of freedom in the $\infty / \infty$ model is the distillate to feed ratio $D / F .$ It is set by the agent in the continuous third level decision within a range of 0.05 to 0.95.

![](images/493498e976d86e851c0c5b030a8e98a98422441ba96e75dc8330a7a3ce584ec2.jpg)  
Figure 7: Vapor-liquid-equilibrium in the quarternary system consisting of MeOAc, HOAc, $_ \mathrm { H _ { 2 } O }$ and MeOH at 1 bar. The gray surface markes the distillation boundary spanned by the two azeotropic points and the fourth component HOAc, spliting the diagram into two distillation regimes.

Recycle. The agent can also select to recycle an open process stream back to the feed stream. Thereby, the ratio of the considered stream that will be recycled is selected by the agent in the third level decision. The recycle is modelled by adding a splitting unit and a mixing unit to the flowsheet. First, the considered stream is split up in a recycle stream and a purge stream. The latter one ends in a new ”undefined” node. To simulate the recycle, a tear stream is initialized. Then, the Wegstein method <sup>63</sup> is used to solve the recycle stream flow rate iteratively. When the Wegstein method is converged, the tear stream is closed and the recycle stream is fed into the feed stream by the mixing unit. This method is based on the implementation of flexsolve <sup>64</sup>.

## 3.2 Reward

The reward assesses the economic viability of the generated process, following Seider et al. <sup>59</sup> for calculating annualized cost and Smith <sup>58</sup> for estimating unit capital costs. After completing a flowsheet by specifying all open streams as products, the agent receives a final reward. This final reward r represents an approximate net cash flow of the process within one year. If this net cash flow is negative, it is reduced by a factor 10 to encourage exploration of the agent. The economic value of incomplete flowsheets is more dificult to estimate because it may depend on future actions. Thus, a reward of zero is given after every single action since the actual value of an action can only be assessed when an episode is complete. As shown in Equation 2, the final reward includes costs for units and feeds as well as revenue for sold products.

$$
r = \sum P _ {\mathrm{products}} - \sum C _ {\mathrm{feed}} - \sum (U + 0. 1 5 * I) _ {\mathrm{units}}\tag{2}
$$

The values of the products are estimated by an s-shaped price function P , depending on the purity of the considered streams. The pure component price C is used to compute the cost of the raw material stream. The annualized cost is computed by adding the annual utility costs U and the total capital investment I multiplied by a factor $0 . 1 5 ^ { 5 9 }$ . Furthermore, the reward is used to teach the agent to make feasible decisions. Whenever infeasible actions are selected that cause the simulation to fail, e.g., if the reactor simulation fails due to bad initial values in the solver, the episode is interrupted immediately and a negative reward of −10 Mio e is given. When the agent decides to not add units at all and just sell the feed streams, the same penalty is given to prevent the agent from falling into this trivial local optimum.

Notably, the considered case study is meant to facilitate illustration and the considered parameter values for prices are only approximations.

## 4 Results & discussion

In this section, we present and analyze the learning behavior of the developed agent. For investigating all single parts of the agent, the training procedure was first conducted in a discrete action space, consisting of the first and second hierarchical decision levels. Afterward, the same procedure was conducted in a continuous action space which only includes the third decision level. Finally, all decision levels are combined to the hybrid action space. In all runs, the environment was initialized with a feed consisting of an equimolar binary mixture of MeOH and HOAc. The feed’s molar flow rate was set to $\mathrm { 1 0 0 m o l s ^ { - 1 } }$ and its temperature to $2 7 ^ { \circ } \mathrm { C }$

Table 1: Fixed continuous design variables for each unit type during the training in a discrete action space. This selection replaces the third level decision.

<table><tr><td>Unit operation</td><td>Design variable</td><td>Symbol</td><td>Unit</td><td>Fixed value</td></tr><tr><td>Heat exchanger</td><td>Water inlet temperature</td><td> $T_{\text{water}}^{in}$ </td><td>°C</td><td>32</td></tr><tr><td>Reactor</td><td>Reactor length</td><td>l</td><td>m</td><td>10</td></tr><tr><td>Column</td><td>Distillate to feed ratio</td><td>D/F</td><td>-</td><td>0.5</td></tr><tr><td>Recycle</td><td>Recycling ratio</td><td>-</td><td>-</td><td>0.9</td></tr></table>

![](images/c44c8b3aad5c070a49c714af81273a28b6e2154cc6e252ffd0cdf17a437fb004.jpg)  
Figure 8: Learning curve of the agent in a discrete action space over 10 000 episodes. It shows the scores of the generated flowsheets, averaged over 50 episodes. The score of each episode corresponds to the reward which is the estimated net cash flow. An episode is a sequence of actions to generate a flowsheet, starting with a feed.

The proposed learning process and the agent architecture include several hyperparameters that are listed in the appendix in Table 4. The selected hyperparameters are based on literature <sup>29,30,46,65</sup>.

## 4.1 Flowsheet generation in a discrete action space

To investigate the agent’s behavior in a discrete action space, the third level actor was deactivated and only the first and second level decisions were conducted. Thus, in each step, the agent selected a location for a new unit operation as well as its type. Thereby, fixed values for the unit’s continuous design variables were used. They are displayed in Table 1. Throughout the presented case study, constant pressure of 1 bar was assumed. The agent was trained in 10 000 episodes with the procedure described in 2.3.

Figure 8 shows the learning curve of the agent in the discrete action space. The displayed scores correspond to the reward which is the estimated net cash flow of the final process. Thus, they are a measure of the economic viability of the final process.

During the first 2000 episodes, the learning curve rises almost exponentially. In this early training stage, the agent produces predominantly long flowsheets and often reaches the maximum allowed number of unit operations. However, throughout the training the agent learns that shorter flowsheets are economically more valuable. Soon, the agent mainly produces flowsheets with a positive score, meaning that the final process is economically viable. Afterward, the learning curve still rises but only in minor scales. One reason for the marginal improvements could be that the agent mainly exploits its experience at this time while still finding slightly better flowsheets through exploration.

![](images/bd6d739b1b27554c9053ffb9611c90e109d57d65125470de67b7acb3582c00d2.jpg)  
Figure 9: Best Flowsheet generated by the agent in a discrete action space after training for 10 000 episodes. In a reactor (R1), MeOAc and its side product $_ \mathrm { H _ { 2 } O }$ are produced from the feed (F1). Then, the resulting quarternary mixture is split up in two columns (C1 and C2). Parts of the third product stream (P3) are recycled and mixed with the feed stream.

![](images/97dfeda5a049e4fc2695ab83e301e66d676c6e197acb6655cc6dc85a0af11a6c.jpg)  
Figure 10: Fixed flowsheet structure during the training in a continuous action space. It consists of a heat exchanger (HEX1), a reactor (R1) and a column (C1). The bottom product (P2) is split up and partially recycled.

The best flowsheet the agent generated throughout training is displayed in Figure 9. The depicted process first uses a reactor (R1) to produce MeOAc and its side product $_ \mathrm { H _ { 2 } O }$ from the feed (F1). Then, the resulting quarternary mixture is split up in two distillation columns. The distillate (P1) of the first column (C1) is enriched with MeOAc but also includes MeOH and $_ \mathrm { H _ { 2 } O }$ . The bottom product of the first column is further split up in a second column (C2) to produce a mixture of $_ \mathrm { H _ { 2 } O }$ and MeOH in the distillate (P2) and pure MeOH in the third product stream (P3). 90 % of the latter product is recycled and mixed with the feed stream. During the training, the agent learned, for example, that heat exchangers do not add value to the flowsheet.

## 4.2 Flowsheet generation in a continuous action space

The third level actor was investigated by deactivating the first and second level actors and thus only including continuous decisions. Therefore, the sequence of unit operations in the flowsheet was fixed, as shown in Figure 10, and only the continuous design variables defining each unit were selected by the agent. Within this structure, the agent was trained for 10 000 episodes. Similar to the findings in the discrete action space, the agent learns quickly at the beginning of the training. After the steep increase, the policy starts to converge and is almost constant after 10 000 episodes. The resulting learning curve of the continuous agent is displayed in Figure 11, showing the scores of the final flowsheets averaged over 50 episodes.

Table 2 lists the continuous design variables of the best flowsheet the agent observed throughout the training. In the heat exchanger (HEX1), the feed is slightly heated before entering the reactor. With a length of 5.24 m, the reactor (R1) is relatively short compared to the allowed length range of 0.05 m to 20 m. A shorter reactor means a lower conversion but also lower costs. The column (C1) is characterized by the distillate to feed ratio $D / F$ of 0.59. As a result, MeOAc is enriched in the distillate which also contains MeOH and $_ \mathrm { H _ { 2 } O }$ . The bottom product is a mixture of MeOH and HOAc. In the investigated flowsheet shown in Figure 10, the bottom product is partially recycled to the feed. Remarkably, the recycled ratio is set to zero in the depicted best flowsheet. These results show that a recycle does not make economic sense for the illustrative flowsheet used for this study.

![](images/f16f4111daea0b7afd8c716390ac46caa3c8df13a6fe551e44c1cfdfc17ed432.jpg)  
Figure 11: Learning curve of the agent in a continuous action space over 10 000 episodes. Analogously to Figure 8, it shows the scores of the generated flowsheets, averaged over 50 episodes.

Table 2: Continuous design variables selected by the continuous agent in the best flowsheet observed during 10 000 episodes of training.

<table><tr><td>Unit operation</td><td>Design variable</td><td>Symbol</td><td>Unit</td><td>Best run</td></tr><tr><td>Heat exchanger (H1)</td><td>Water inlet temperature</td><td> $T_{\text{water}}^{in}$ </td><td>°C</td><td>39.7</td></tr><tr><td>Reactor (R1)</td><td>Reactor length</td><td>l</td><td>m</td><td>5.24</td></tr><tr><td>Column (C1)</td><td>Distillate to feed ratio</td><td>D/F</td><td>-</td><td>0.59</td></tr><tr><td>Recycle</td><td>Recycled ratio</td><td>-</td><td>-</td><td>0</td></tr></table>

## 4.3 Flowsheet generation in a hybrid action space

After the previous sections have shown that all three actors are able to learn separately, they are combined hereinafter. Therefore, the hybrid agent, combining all previously described elements, is trained in 10 000 episodes.

The resulting learning curve is displayed in Figure 12, showing the scores of the flowsheets generated during the training, averaged over 50 episodes. Despite the complexity of the hybrid problem, the agent is learning fast and quickly produces flowsheets with a positive value after approximately 1000 episodes. The best flowsheet the agent observed during training is shown in Figure 13. The continuous design variables the agent selected for this best flowsheet are shown in Table 3.

The feed (F1) is fed directly into a reactor (R1) where MeOAc and $_ \mathrm { H _ { 2 } O }$ are produced from esterification of HOAc with MeOH. With a length of 18.4 m, the reactor is significantly larger compared to the best flowsheet generated with the continuous agent in Section 4.2 which results in a higher conversion but also higher costs. In the next step, the resulting quarternary mixture is heated in a heat exchanger (HEX1) and split up in a column (C1). In the distillate of the column (P1), MeOAc is enriched but it also includes MeOH and residues of $_ \mathrm { H _ { 2 } O }$ . The bottom product of the column (P2) contains HOAc and

![](images/268afc9377925f89fc06f61f67aaa3eb9ce15263e028400b824bb68b7017fafb.jpg)  
Figure 12: Learning curve of the agent in a hybrid action space over 10 000 episodes. Analogously to Figure 8 and Figure 11, it shows the scores of the generated flowsheets, averaged over 50 episodes.

![](images/1e3465c097f11bb042ee6286cef117d60b7932a5088bb28f33c5e94dc50d0574.jpg)  
Figure 13: Best flowsheet generated by the agent in a hybrid action space within 10 000 training episodes. First, MeOAc and its side product $_ \mathrm { H _ { 2 } O }$ are produced from the feed (F1) in a reactor (R1). Then, the resulting quarternary mixture is heated up in a heat exchanger (HEX1) and split up in a column (C1). Before entering the column, 24% of the stream are split up and recycled. The first product (P1) is enriched with MeOAc but also includes MeOH and residues of $_ \mathrm { H _ { 2 } O }$ . The second product (P2) is a mixture of HOAc and MeOH.

Table 3: Continuous design variable selected by the hybrid agent in the best flowsheet observed during 10 000 episodes of training.

<table><tr><td>Unit operation</td><td>Design variable</td><td>Symbol</td><td>Unit</td><td>Best run</td></tr><tr><td>Reactor (R1)</td><td>Reactor length</td><td> $l$ </td><td>m</td><td>18.4</td></tr><tr><td>Heat exchanger (HEX1)</td><td>Water inlet temperature</td><td> $T_{\text{water}}^{in}$ </td><td>°C</td><td>36.2</td></tr><tr><td>Column (C1)</td><td>Distillate to feed ratio</td><td> $D/F$ </td><td>-</td><td>0.55</td></tr><tr><td>Recycle</td><td>Recycled ratio</td><td>-</td><td>-</td><td>0.24</td></tr></table>

MeOH. The sequence of unit operations difers from the best flowsheet generated by the discrete agent in Section 4.1, were no heat exchanger and two columns were used. Here, the desired product MeOAc is completely in the distillate and the bottom product consists of less valuable chemicals. Thus, the agent learnt that the second column does not add economic value. Before entering the column, 24 % of the process stream are recycled to the feed. In contrast to the flowsheet investigated in the continuous action space in Section 4.2, the recycle does add value to the flowsheet since it increases the total conversion in the reactor.

## 4.4 Discussion

Overall, the learning curves shown in the previous sections indicate that all parts of the agent learn quickly. It is assumed, however, that the policy does not always converge towards the global optimum for the considered task since the hyperparameters have not been optimized for this first fundamental study. In future works, it is advised to conduct an extensive hyperparameter study to investigate their influence on the learning behavior.

Compared to other approaches, the main contribution of the presented method is the representation of flowsheets as graphs and combining GNNs with RL. GNNs have already shown promising performance in various deep learning tasks <sup>42</sup>. One of their key advantage is that they are able to process the topological information of the graphs<sup>43</sup>. Since the structural information about flowsheets is automatically captured in the graph format, GNNs can take advantage of this structure. Deriving fingerprints from graphs with GNNs has already shown promising results in the molecule field <sup>29,66,67</sup>. Here, we transfer the methodology to the flowsheet domain. During the implementation and analysis of the training procedure, the graph presentation of the flowsheets has proven to be handy. The graphs generated by the agent can be visualized easily and thus immediately give an insight into the process and its meaningfulness. An additional advantage of the approach is its flexibility. Through its hierarchical structure, the diferent components of the agent can be easily decoupled and new parts can be added. By using a separate MLP for each unit operation in the third level decision, the number of the continuous decisions can vary for the diferent unit operations. In the presented work, only one continuous decision is made for each unit operation but the agent architecture allows including more decisions within this step. By allowing for more unit operations and setting more design variables, the action space and thus the complexity of the problem should be increased for future investigations.

Furthermore, the reward function will require additional attention. Giving rewards is not straightfor ward in the considered problem since it is hard to assess the value of an intermediate flowsheet. Still, it is crucial for the performance of the RL algorithm. In the presented work, the reward function is only an estimation of economic assessments that neglects multiple cost factors in real processes. However, for future developments, investigating ways of reward shaping <sup>68</sup> will be an interesting aspect that can stabilize the training process especially when the size of the considered problem gets larger.

## 5 Conclusion

We propose the first RL agent that learns from flowsheet graphs using GNNs to synthesize new processes. The deployed RL agent is hierarchical and hybrid meaning it takes multiple dependent discrete and continuous decisions within one step. In the proposed methodology, the agent first selects a location in an existing flowsheet and a unit operation to extend the flowsheet at the selected position. Both selections are discrete. Then, it takes a continuous decision by selecting a design variable that defines the unit operation. Naturally, each sub-decision strongly depends on the previous one. Thereby, flowsheets are represented as graphs which allows us to utilize GNNs within the RL structure. As a result, our methodology generates economical valuable flowsheets only based on experience of the RL agent.

In an illustrative case study considering the production of methyl acetate, the approach shows steep and mostly stable learning in discrete, continuous, and hybrid action spaces. This work is a fundamental study that demonstrates that graph-based RL is able to create meaningful flowsheets. Thus, it encourages to incorporate AI in chemical process design.

A further advantage of the presented approach is that the proposed architecture is a good foundation for further developments like enhancing the state-action space. Thus, the selected structure of the agent is predestined for increasing the complexity and solving more advanced problems in the future. A subsequent step following this paper should be to implement an interface to an advanced process simulator. This will tremendously increase the complexity of the problem but also allow for easier extension of the action space and more rigorous simulations. As the process simulator will need to deal with random combinations of unit operations, guaranteeing convergence will become a major challenge and including constraints is advisable.

Acknowledgements This work is supported by the TU Delft AI Labs Programme.

Abbreviations

AI artificial intelligence  
ANN artificial neural network  
CNN convolutional neural network  
GCN graph convolutional network  
GCPN graph convolutional policy network  
GNN graph neural network $\mathbf{H}_2\mathbf{O}$ water  
HOAc acetic acid  
MDP Markov decision process  
MeOAc methyl acetate  
MeOH methanol  
MINLP mixed integer non-linear programming  
ML machine learning  
MLP multi-layer perceptron  
MPNN message passing neural network  
PFR plug flow reactor  
PPO proximal policy optimization  
RL reinforcement learning  
RNN recurrent neural network  
SMILES simplified molecular-input line-entry system

## References

1. Fantke P, Cinquemani C, Yaseneva P, et al. Transition to sustainable chemistry through digitalization. Chem. 2021; 7(11): 2866–2882. doi: 10.1016/j.chempr.2021.09.012

2. Meramo-Hurtado SI, Gonz´alez-Delgado AD. Process Synthesis, Analysis, and Optimization Method-<sup>´</sup> ologies toward Chemical Process Sustainability. Industrial & Engineering Chemistry Research. 2021; 60(11): 4193–4217. doi: 10.1021/acs.iecr.0c05456

3. Mohan S, Katakojwala R. The circular chemistry conceptual framework: A way forward to sustainability in industry 4.0. Current Opinion in Green and Sustainable Chemistry. 2021; 28: 100434. doi: 10.1016/j.cogsc.2020.100434

4. Martinez-Hernandez E. Trends in sustainable process design—from molecular to global scales. Current Opinion in Chemical Engineering. 2017; 17: 35–41. doi: 10.1016/j.coche.2017.05.005

5. Schweidtmann AM, Esche E, Fischer A, et al. Machine Learning in Chemical Engineering: A Perspective. Chemie Ingenieur Technik. 2021; 93(12): 2029–2039. doi: 10.1002/cite.202100083

6. Mnih V, Kavukcuoglu K, Silver D, et al. Playing Atari with Deep Reinforcement Learning. arXiv preprint arXiv:1312.5602. 2013.

7. Kempka M, Wydmuch M, Runc G, Toczek J, Jaskowski W. ViZDoom: A Doom-based AI research platform for visual reinforcement learning. In: IEEE. ; 20.09.2016 - 23.09.2016: 1–8

8. Silver D, Hubert T, Schrittwieser J, et al. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science. 2018; 362(6419): 1140–1144. doi: 10.1126/science.aar6404

9. Sutton RS, Barto A. Reinforcement learning, second edition: An introduction (2nd edition). Cambridge, Massachusetts and London, England: The MIT Press . 2018.

10. Silver D, Huang A, Maddison CJ, et al. Mastering the game of Go with deep neural networks and tree search. Nature. 2016; 529(7587): 484–489. doi: 10.1038/nature16961

11. Vinyals O, Babuschkin I, Czarnecki WM, et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature. 2019; 575(7782): 350–354. doi: 10.1038/s41586-019-1724-z

12. OpenAI , Berner C, Brockman G, et al. Dota 2 with Large Scale Deep Reinforcement Learning. arXiv preprint arXiv:1912.06680. 2019.

13. Lee YH, Lee S. Deep reinforcement learning based scheduling within production plan in semiconductor fabrication. Expert Systems with Applications. 2022; 191: 116222. doi: 10.1016/j.eswa.2021.116222

14. Hubbs CD, Li C, Sahinidis NV, Grossmann IE, Wassick JM. A deep reinforcement learning approach for chemical production scheduling. Computers & Chemical Engineering. 2020; 141: 106982. doi: 10.1016/j.compchemeng.2020.106982

15. Hoskins JC, Himmelblau DM. Process control via artificial neural networks and reinforcement learn ing. Computers & Chemical Engineering. 1992; 16(4): 241–251. doi: 10.1016/0098-1354(92)80045-b

16. Spielberg SPK, Gopaluni RB, Loewen PD. Deep reinforcement learning approaches for process control. In: IEEE. ; 2017

17. Mowbray M, Petsagkourakis P, del Rio-Chanona EA, Zhang D. Safe chance constrained reinforcement learning for batch process control. Computers & Chemical Engineering. 2022; 157: 107630. doi: 10.1016/j.compchemeng.2021.107630

18. Sachio S, del Rio-Chanona EA, Petsagkourakis P. Simultaneous Process Design and Control Optimization using Reinforcement Learning. IFAC-PapersOnLine. 2021; 54(3): 510–515. doi: 10.1016/j.ifacol.2021.08.293

19. Sachio S, Mowbray M, Papathanasiou MM, del Rio-Chanona EA, Petsagkourakis P. Integrating process design and control using reinforcement learning. Chemical Engineering Research and Design. 2022; 183: 160–169. doi: 10.1016/j.cherd.2021.10.032

20. Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature. 2015; 518(7540): 529–533. doi: 10.1038/nature14236

21. Lillicrap TP, Hunt JJ, Pritzel A, et al. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971. 2015.

22. You J, Liu B, Ying Z, Pande V, Leskovec J. Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation. In: Bengio S, Wallach H, Larochelle H, Grauman K, Cesa-Bianchi N, Garnett R. , eds. Advances in Neural Information Processing Systems. 31. Curran Associates, Inc. ; 2018.

23. Popova M, Isayev O, Tropsha A. Deep reinforcement learning for de novo drug design. Science advances. 2018; 4(7): eaap7885. doi: 10.1126/sciadv.aap7885

24. Zhou Z, Kearnes S, Li L, Zare RN, Riley P. Optimization of Molecules via Deep Reinforcement Learning. Scientific reports. 2019; 9(1): 10752. doi: 10.1038/s41598-019-47148-x

25. Cao ND, Kipf T. MolGAN: An implicit generative model for small molecular graphs. arXiv preprint arXiv:1805.11973. 2018.

26. Olivecrona M, Blaschke T, Engkvist O, Chen H. Molecular de-novo design through deep reinforcement learning. Journal of cheminformatics. 2017; 9(1): 48. doi: 10.1186/s13321-017-0235-x

27. Pereira T, Abbasi M, Ribeiro B, Arrais JP. Diversity oriented Deep Reinforcement Learning for targeted molecule generation. Journal of cheminformatics. 2021; 13(1): 21. doi: 10.1186/s13321-021- 00498-z

28. Jeon W, Kim D. Autonomous molecule generation using reinforcement learning and docking to develop potential novel inhibitors. Scientific reports. 2020; 10(1): 22104. doi: 10.1038/s41598-020- 78537-2

29. Schweidtmann AM, Rittig JG, K¨onig A, Grohe M, Mitsos A, Dahmen M. Graph Neural Networks for Prediction of Fuel Ignition Quality. Energy & Fuels. 2020; 34(9): 11395–11407. doi: 10.1021/acs.energyfuels.0c01533

30. Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. Neural message passing for quantum chemistry. In: PMLR. ; 2017: 1263–1272.

31. Coley CW, Barzilay R, Green WH, Jaakkola TS, Jensen KF. Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction. Journal of chemical information and modeling. 2017; 57(8): 1757–1772. doi: 10.1021/acs.jcim.6b00601

32. Yang K, Swanson K, Jin W, et al. Analyzing Learned Molecular Representations for Property Prediction. Journal of chemical information and modeling. 2019; 59(8): 3370–3388. doi: 10.1021/acs.jcim.9b00237

33. Ororbia ME, Warn GP. Design Synthesis Through a Markov Decision Process and Reinforcement Learning Framework. Journal of Computing and Information Science in Engineering. 2022; 22(2). doi: 10.1115/1.4051598

34. Midgley LI. Deep Reinforcement Learning for Process Synthesis. arXiv preprint arXiv:2009.13265. 2020.

35. Khan A, Lapkin A. Searching for optimal process routes: A reinforcement learning approach. Computers & Chemical Engineering. 2020; 141: 107027. doi: 10.1016/j.compchemeng.2020.107027

36. G¨ottl Q, Grimm DG, Burger J. Automated synthesis of steady-state continuous processes using reinforcement learning. Frontiers of Chemical Science and Engineering. 2022; 16(2): 288–302. doi: 10.1007/s11705-021-2055-9

37. G¨ottl Q, T¨onges Y, Grimm DG, Burger J. Automated Flowsheet Synthesis Using Hierarchical Reinforcement Learning: Proof of Concept. Chemie Ingenieur Technik. 2021; 93(12): 2010–2018. doi: 10.1002/cite.202100086

38. Intemic . Flowsheet Copilot. https://intemic.com; 2022.

39. Plathottam SJ, Richey B, Curry G, Cresko J, Iloeje CO. Solvent extraction process design using deep reinforcement learning. Journal of Advanced Manufacturing and Processing. 2021; 3(2). doi: 10.1002/amp2.10079

40. Khan AA, Lapkin AA. Designing the process designer: Hierarchical reinforcement learning for optimisation-based process design. Chemical Engineering and Processing - Process Intensification. 2022: 108885. doi: 10.1016/j.cep.2022.108885

41. Bruna J, Zaremba W, Szlam A, LeCun Y. Spectral Networks and Locally Connected Networks on Graphs. arXiv preprint arXiv:1312.6203. 2013.

42. Zhou J, Cui G, Hu S, et al. Graph neural networks: A review of methods and applications. AI Open. 2020; 1: 57–81. doi: 10.1016/j.aiopen.2021.01.001

43. Bronstein MM, Bruna J, Cohen T, Veliˇcković P. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv preprint arXiv:2104.13478. 2021.

44. Zhang T, Sahinidis NV, Siirola JJ. Pattern recognition in chemical process flowsheets. AIChE Journal. 2019; 65(2): 592–603. doi: 10.1002/aic.16443

45. Friedler F, Aviso KB, Bertok B, Foo DCY, Tan RR. Prospects and challenges for chemical process synthesis with P-graph. Current Opinion in Chemical Engineering. 2019; 26: 58–64. doi: 10.1016/j.coche.2019.08.007

46. Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347. 2017.

47. Mnih V, Badia AP, Mirza M, et al. Asynchronous Methods for Deep Reinforcement Learning. In: Balcan MF, Weinberger KQ. , eds. Proceedings of The 33rd International Conference on Machine Learning. 48 of Proceedings of Machine Learning Research. PMLR. ; 2016: 1928–1937.

48. Fan Z, Su R, Zhang W, Yu Y. Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space. arXiv preprint arXiv:1903.01344. 2019.

49. Fujimoto S, van Hoof H, Meger D. Addressing Function Approximation Error in Actor-Critic Methods. In: Dy J, Krause A. , eds. Proceedings of the 35th International Conference on Machine Learning. 80 of Proceedings of Machine Learning Research. PMLR. ; 2018: 1587–1596.

50. Haarnoja T, Zhou A, Hartikainen K, et al. Soft Actor-Critic Algorithms and Applications. arXiv preprint arXiv:1812.05905. 2018.

51. Williams RJ, Peng J. Function Optimization using Connectionist Reinforcement Learning Algorithms. Connection Science. 1991; 3(3): 241–268. doi: 10.1080/09540099108946587

52. Schulman J, Moritz P, Levine S, Jordan M, Abbeel P. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv preprint arXiv:1506.02438. 2015.

53. Gupta AK, Nadarajah S. Handbook of beta distribution and its applications. CRC Press . 2004.

54. National Center for Biotechnology Information . PubChem Compound Summary for CID 6584, Methyl acetate. https://pubchem.ncbi.nlm.nih.gov/compound/Methyl-acetate; . Accessed Jan. 8, 2022.

55. Huss RS, Chen F, Malone MF, Doherty MF. Reactive distillation for methyl acetate production. Computers & Chemical Engineering. 2003; 27(12): 1855–1866. doi: 10.1016/s0098-1354(03)00156-x

56. Agreda VH, Partin LR. Reactive distillation process for the production of methyl acetate.; 1982.

57. Xu ZP, Chuang KT. Kinetics of acetic acid esterification over ion exchange catalysts. The Canadian Journal of Chemical Engineering. 1996; 74(4): 493–500. doi: 10.1002/cjce.5450740409

58. Smith R. Chemical process: Design and integration (2nd edition). Chichester, West Sussex, United Kingdom: Wiley . 2016.

59. Seider WD, Lewin DR, Seader JD, Widagdo S, Gani R, Ng KM. Product and process design principles: Synthesis, analysis, and evaluation (4th edition). Hoboken, NJ: Wiley . 2017.

60. Bekiaris N, Meski GA, Radu CM, Morari M. Multiple steady states in homogeneous azeotropic distillation. Industrial & Engineering Chemistry Research. 1993; 32(9): 2023–2038. doi: 10.1021/ie00021a026

61. Ryll O, Blagov S, Hasse H. ∞/∞-Analysis of homogeneous distillation processes. Chemical Engineering Science. 2012; 84: 315–332. doi: 10.1016/j.ces.2012.08.018

62. Burger J, Hasse H. Multi-objective optimization using reduced models in conceptual design of a fuel additive production process. Chemical Engineering Science. 2013; 99: 118–126. doi: 10.1016/j.ces.2013.05.049

63. Wegstein JH. Accelerating convergence of iterative processes. Communications of the ACM. 1958; 1(6): 9–13.

64. Cortes-Pena Y. flexsolve: Flexible function solvers. https://github.com/yoelcortes/flexsolve; 2019.

65. AurelianTactics . PPO Hyperparameters and Ranges. https://medium.com/aureliantactics/ppohyperparameters-and-ranges-6fc2d29bccbe; 2018.

66. Duvenaud DK, Maclaurin D, Iparraguirre J, et al. Convolutional Networks on Graphs for Learning Molecular Fingerprints. In: Cortes C, Lawrence N, Lee D, Sugiyama M, Garnett R. , eds. Advances in Neural Information Processing Systems. 28. Curran Associates, Inc. ; 2015.

67. Kearnes S, McCloskey K, Berndl M, Pande V, Riley P. Molecular graph convolutions: moving beyond fingerprints. Journal of computer-aided molecular design. 2016; 30(8): 595–608. doi: 10.1007/s10822- 016-9938-8

68. Ng AY, Harada D, Russell S. Policy invariance under reward transformations: Theory and application to reward shaping. In: Morgan Kaufmann; 1999: 278–287.

## Appendix

Table 4: Hyperparameters for the architecture and training procedure of the actor-critic agent.

<table><tr><td>Parameter</td><td></td><td>Value</td></tr><tr><td>Learning rate</td><td> $\alpha$ </td><td>0.0002</td></tr><tr><td>Policy clipping factor</td><td> $\epsilon$ </td><td>0.3</td></tr><tr><td>Discount factor</td><td> $\gamma$ </td><td>1.0</td></tr><tr><td> $\lambda$ -return factor</td><td> $\lambda$ </td><td>0.95</td></tr><tr><td>Batch size</td><td> $n_{\text{B}}$ </td><td>60</td></tr><tr><td>Mini batch size</td><td> $n_{\text{MB}}$ </td><td>30</td></tr><tr><td>Number of epochs</td><td> $n_{\text{E}}$ </td><td>4</td></tr><tr><td>Weight for loss of level 1 actor</td><td> $c_0$ </td><td>0.1</td></tr><tr><td>Weight for loss of level 2 actor</td><td> $c_1$ </td><td>1.0</td></tr><tr><td>Weight for loss of level 3 actor</td><td> $c_2$ </td><td>0.5</td></tr><tr><td>Weight for loss of critic</td><td> $c_3$ </td><td>0.2</td></tr><tr><td>Weight for entropy of level 1 actor</td><td> $d_1$ </td><td>0.001</td></tr><tr><td>Weight for entropy of level 2 actor</td><td> $d_2$ </td><td>0.3</td></tr><tr><td>Weight for entropy of level 3 actor</td><td> $d_3$ </td><td>0.5</td></tr><tr><td>Weight for entropy of level 3 actor</td><td> $d_3$ </td><td>0.001</td></tr><tr><td>Hidden layers edge processing for fingerprint</td><td>-</td><td>10</td></tr><tr><td>Message passing steps for fingerprint</td><td>-</td><td>6</td></tr><tr><td>Hidden layers dimension level 1 actor</td><td>-</td><td>12</td></tr><tr><td>Hidden layers dimension level 2 actor</td><td>-</td><td>256</td></tr><tr><td>Hidden layers dimension level 3 actor</td><td>-</td><td>256</td></tr><tr><td>Hidden layers dimension critic</td><td>-</td><td>256</td></tr><tr><td>Feature size flowsheet fingerprint</td><td>-</td><td>50</td></tr></table>