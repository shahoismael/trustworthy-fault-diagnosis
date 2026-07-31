# Hybrid Neural Ordinary Diferential Equations for Data-Eficient Polymerization Modeling with Incomplete Kinetics

Marah Almanasreh<sup>1</sup>, Alexander Mitsos<sup>2,1,3</sup>, and Eike Cramer<sup>4,\*</sup>

<sup>1</sup>RWTH Aachen University, Process Systems Engineering (AVT.SVT), 52074 Aachen, Germany <sup>2</sup>JARA-CSD, 52425 Jülich, Germany

<sup>3</sup>Energy Systems Engineering (ICE-1), Forschungszentrum Jülich, 52425 Jülich, Germany <sup>4</sup>Department of Chemical Engineering, Sargent Centre for Process Systems Engineering, University College London, Torrington Place, London WC1E 7JE, United Kingdom <sup>\*</sup>Corresponding author: e.cramer@ucl.ac.uk

## Abstract

![](images/538f1c407a76d70c9a240c27d37b0b6687992760fb163136e9c9b5252ce769ea.jpg)

Accurate prediction of polymerization dynamics is essential for process design, control, and optimization. Yet, purely mechanistic models require labor-intensive parameterization of partially characterized kinetics, while purely data-driven models demand large, diverse datasets that are costly to obtain, particularly in early-design stages. We propose a hybrid Neural Ordinary Diferential Equation (NODE) framework for data-eficient modeling of free-radical polymerization. Using batch polymerization of methyl methacrylate (MMA) as a case study, the mechanistic mass balances are retained explicitly, and only the partially-characterized efective radical concentration governing monomer consumption is learned from data through a neural network surrogate, while established reactions such as initiator decomposition, propagation, and termination remain physically modeled. The hybrid NODE is evaluated against a discrete-time feedforward neural network and a purely data-driven NODE under sparse data conditions, with models trained on as few as ten measurements under both regular and irregular sampling. The hybrid NODE consistently achieves lower prediction errors and more physically consistent extrapolations than both purely data-driven baselines. In a generalization scenario with noisy data and unseen operating conditions, the hybrid NODE achieves an RMSE of 0.013, compared to 0.31 for the data-driven NODE and 0.68 for the discrete-time model, demonstrating that learning only a closure term rather than the full dynamics is suficient for reliable prediction under limited data availability.

## 1 Introduction

Polymerization processes play a crucial role in producing a wide range of materials, from commodity plastics to high-performance polymers for advanced technologies. Accurate modeling and control of these processes are essential for optimizing production eficiency, enhancing product quality, and reduc ing manufacturing costs and environmental impacts [1, 2]. Among polymerization techniques, radica polymerization is particularly versatile and economically viable, which has led to its widespread in dustrial adoption [3, 4]. However, the inherent complexity of radical polymerization kinetics involves numerous concurrent reaction pathways whose rates and radical population dynamics are sensitive to process conditions such as temperature and initiator concentration [3, 5], posing significant challenges for accurate dynamic modeling. These sensitivities directly afect polymer quality, reactor safety, and process economics, making reliable dynamic predictions indispensable for industrial design, optimization, and control [6, 7]. Accordingly, this work focuses on radical polymerization as a representative and industrially relevant polymerization technique.

Mechanistic models based on first-principles chemical kinetics have long served as the foundation of polymer reaction engineering [8]. These models provide physical interpretability and mechanistic insight into polymerization pathways. Radical polymerization is typically described as following the steps of initiation, propagation, and termination, sequentially adding monomers to growing chains. Both deterministic and stochastic formulations have been used [6]. Deterministic modeling is commonly used in reactor-scale analysis and process optimization to describe macroscopic behaviors such as conversion and molecular weight [9]. However, these models typically rely on simplifying assumptions and lack detailed molecular resolution, particularly with respect to chain length–dependent reactivity and sequence efects [9]. Stochastic models are typically used to describe molecular-level polymerization behavior, such as chain-length distributions and chain topology, whereas deterministic models are more commonly applied to reactor-scale process analysis and optimization [10]. The limitations of both approaches have motivated growing interest in machine learning (ML)-based modeling approaches [11, 12, 13].

The body of ML literature in polymer science has expanded rapidly in recent years. In the following, we focus on ML approaches relevant to dynamic polymerization process modeling under limited data availability.

Most ML contributions in the literature focus on predicting static kinetic quantities, such as prop agation rate coeficients or comonomer reactivity ratios, directly from molecular structure, often using SMILES-based descriptors [14], molecular fingerprints [15], or graph-based representations [16, 17]. Recent work has also explored ML-based approaches for polymerization process modeling, including the prediction of time-evolving quantities such as molecular weight distributions and other reactor-scale properties [18, 19]. Related data-driven eforts extend beyond kinetics, for example, to polymer-solvent compatibility, where hybrid symbolic-continuous ML models have been used to jointly learn solvation states and quantitative solubility behavior from experimental data [20]. Ge et al. [12] review the growing role of ML in polymer research, including applications relevant to polymerization, and highlight its potential to accelerate property prediction and materials discovery. However, the authors also empha size several key limitations of purely data-driven approaches, such as the scarcity and inconsistency of polymerization data, the dominance of small-data regimes, dificulties in representing polymerization mechanisms and reaction dynamics, limited extrapolation beyond training conditions, and the lack of physical interpretability. As a result, Ge et al. [12] argue that ML models that neglect polymer theory and reaction kinetics risk learning correlations rather than causative behavior. The authors therefore advocate theory-guided ML, where first-principles knowledge is integrated with data-driven models to improve robustness, interpretability, and generalization. These observations, together with the chal lenges of modeling polymerization dynamics under sparse and noisy data conditions, further motivate the hybrid modeling approach pursued in this work

The integration of mechanistic and data-driven models has been studied for decades under the framework of hybrid modeling [21, 22, 23]. In recent years, renewed interest in these approaches has emerged in the context of modern ML methods and diferentiable dynamical systems [24, 25]. Multiple strategies for physics-informed and theory-guided ML approaches have been proposed [26, 27, 28, 29, 30]. One approach incorporates theoretical knowledge directly into the modeling framework, as shown by Audus et al. [31], where combining polymer scaling theory with simulation data improved predictions even when the theoretical model was approximate. Related studies relevant to polymerization and processing demonstrate that physically meaningful parameters, such as interaction parameters or kinetic descriptors, can act as informative metafeatures, enabling accurate predictions under limited data availability [28]. Beyond such representations, theory can also be embedded into model architectures and training proce dures; for example, physics-guided neural networks with physically motivated activation functions and constraints have been shown to significantly reduce prediction errors in polymer processing and thermal response problems [32]. Collectively, these studies highlight that theory-guided ML provides a promising framework for polymerization modeling.

For the batch free-radical polymerization (FRP) systems considered in this work, the process dynam ics are described by systems of ordinary diferential equations derived from reaction kinetics. Building on this structure, we formulate a hybrid Neural Ordinary Diferential Equation (NODE) model for FRP dynamics under limited data availability. NODEs extend residual neural networks to the continuous time setting by parameterizing the system dynamics through ordinary diferential equations [33, 34] Recent work in chemical engineering has demonstrated that NODE-based models provide an efective framework for reaction systems under partial mechanistic knowledge and sparse data [35]. NODEs also inherit favorable properties such as parameter eficiency and robustness to perturbations [36]. Their continuous-time structure provides a natural framework for combining mechanistic balance equations with learned dynamic contributions within the same dynamical system. While NODE-based approaches have attracted growing interest in chemical engineering and scientific machine learning, their application to hybrid polymerization modeling remains comparatively limited [12], motivating the approach adopted in this work, where we build on this framework by retaining the known polymerization kinetics explicitly and introducing neural networks only for the missing or approximated dynamic contributions.

In this work, FRP serves both as the application focus and as a representative example of a re action system with partially-characterized kinetics. The proposed hybrid NODE formulation follows a structured modeling strategy in which mechanistic relations that are well established, broadly valid, and readily parameterized are retained explicitly, while dificult-to-model or uncertain kinetic contributions are learned from data. In the present polymerization system, the mechanistic mass balances and known kinetic dependencies are preserved, whereas the efective radical contribution governing monomer con sumption is represented through a learned closure term. The neural-network component is therefore not used to replace the mechanistic model but rather to complement it where the kinetics are dificult to specify accurately within the reduced mechanistic formulation. This formulation preserves the physical structure of the system while reducing the learning problem compared to a fully data-driven approach and reducing the amount of data required for training, since only the unresolved dynamic contribution is learned from data.

By combining mechanistic modeling with data-driven learning in a structure-consistent continuoustime formulation, this work proposes a hybrid NODE approach for FRP systems. The resulting model achieves improved predictive accuracy and generalization across operating conditions, particularly in sparse-data regimes where only limited measurements are available.

The remainder of this work is organized as follows. First, the mechanistic model used in this work for the polymerization process and its state-space formulation are presented in Section 2. Next, Section 3 discusses the ML formulations considered for dynamic process modeling, followed by the proposed hybrid Neural ODE formulation in Section 4. Section 5 then presents the model implementation, training procedure, and evaluation scenarios. Finally, Section 6 presents and discusses the results, and Section 7 concludes the paper with a summary and outlook.

## 2 Free-Radical Polymerization Process Modeling

We consider a homogeneous batch FRP reactor operated under constant-volume conditions. This section summarizes the standard mechanistic model used as the underlying mechanistic structure throughout this work. The model serves two purposes. First, it is used to generate synthetic data for the present case study, which serves as a proof of concept for the proposed approach; in a real-world implementation the data would come from experiments. Second, it provides a partial mechanistic description of the polymerization dynamics that is incorporated into the hybrid NODE formulation, and it is presented here for completeness and to make the subsequent hybrid NODE formulation self-contained. First, the key aspects of FRP kinetics and the corresponding material balances are summarized. The resulting species balances are subsequently combined with the kinetic rate expressions and approximated using a finite-dimensional chain-length discretization, yielding a continuous-time state-space representation of the polymerization dynamics.

## 2.1 Free-Radical Polymerization Kinetics and Modeling

Modeling polymerization kinetics requires combining well-established mechanistic reaction pathways with kinetic contributions that are often dificult to specify or measure [37]. This subsection focuses on the kinetic aspects of the FRP process, with limited emphasis on detailed chemical considerations. The model formulation follows the reaction network modeling workflow described in [38], i.e., first defining the reaction network, then deriving the species balances using stoichiometric relations, and finally imposing constitutive kinetic rate expressions. Despite the availability of general-purpose mechanistic models for FRP, their application to specific polymerization systems remains challenging due to the complexity of the kinetic rate, including propagation, termination, and chain-transfer reactions, and the need for substantial kinetic characterization and parameter estimation. The mechanistic model introduced here, therefore, serves as the underlying mechanistic structure for the subsequent hybrid formulation.

A key feature of FRP is the presence of an adjustable radical source, typically provided by a thermolabile initiator that decomposes upon heating to generate free radicals, i.e., highly reactive species with an unpaired electron. During the polymerization of monomers, each monomer addition preserves the radical structure at the chain end, allowing the growing chain to remain active and continue reacting with additional monomer units. The reaction system contains the following species:

$$
\mathcal {S} = \{I, M, C T A, S \} \cup \{R _ {n} ^ {\bullet}, D _ {n} \} _ {n \geq 1}
$$

where, I denotes the initiator, M the monomer, CT A the chain transfer agent, and S the solvent. The species $R _ { n } ^ { \bullet }$ represents the living radical of chain length n, i.e., a polymer chain carrying an active radical center capable of further propagation, while $D _ { n }$ denotes the dead polymer chain of length $n , \mathrm { i . e . }$ , a chain that no longer contains an active radical center as a result of termination reactions. The conventiona kinetic description of FRP comprises initiation, propagation, and termination, with additional chain transfer reactions included to account for transfer to chain transfer agent, monomer, and solvent [39, 40, 41]. In the following, $n , m \in  { \mathbb { N } } _ { \geq 1 }$ denote polymer chain lengths, $f \in ( 0 , 1 ]$ denotes the initiato eficiency, i.e., the fraction of decomposed initiator radicals that successfully initiate polymer chains, and $k _ { d } , k _ { p } , k _ { t r } , k _ { t r m } , k _ { t r s } , k _ { t d } .$ and $k _ { t c }$ the kinetic rate constants associated with initiator decomposition, propagation, chain transfer, and termination, respectively. The corresponding reaction scheme is given by:

$$
\begin{array}{c c c c} \text {Initiator decomposition:} & I & \xrightarrow {f k _ {d}} & 2 R _ {1} ^ {\bullet} \\ \text {Propagation:} & R _ {n} ^ {\bullet} + M & \xrightarrow {k _ {p}} & R _ {n + 1} ^ {\bullet} \\ \text {Chain transfer to chain transfer agent:} & R _ {n} ^ {\bullet} + C T A & \xrightarrow {k _ {t r}} & D _ {n} + R _ {1} ^ {\bullet} \\ \text {Chain transfer to monomer:} & R _ {n} ^ {\bullet} + M & \xrightarrow {k _ {t r m}} & D _ {n} + R _ {1} ^ {\bullet} \\ \text {Chain transfer to solvent:} & R _ {n} ^ {\bullet} + S & \xrightarrow {k _ {t r s}} & D _ {n} + R _ {1} ^ {\bullet} \\ \text {Termination by disproportionation:} & R _ {n} ^ {\bullet} + R _ {m} ^ {\bullet} & \xrightarrow {k _ {t d}} & D _ {n} + D _ {m} \\ \text {Termination by combination:} & R _ {n} ^ {\bullet} + R _ {m} ^ {\bullet} & \xrightarrow {k _ {t c}} & D _ {n + m} \end{array}
$$

For simplicity, the intermediate radical species generated during chain-transfer reactions to monomer (M ), chain transfer agent (CT A), and solvent (S) are not modeled explicitly; instead, each transfer event is represented by the formation of an efective propagating radical species R<sup>•</sup> [42]. For the considered homogeneous batch reactor with constant volume V , the material balances are written as:

$$
\left. \frac {d c _ {i}}{d t} \right| _ {t} = R _ {i} (t)
$$

where $c _ { i }$ denotes the concentration of species i and $R _ { i }$ the net rate of formation. The species production rates are expressed as

$$
R _ {i} (t) = \sum_ {j \in \mathcal {R}} \nu_ {i, j} r _ {j} (t),
$$

where $r _ { j } ( t )$ denotes the reaction rate associated with reaction $j \in \mathcal R$ and $\nu _ { i , j }$ the corresponding stoichiometric coeficient, where R denotes the set of reactions listed in the reaction stoichiometry scheme above. The reaction rates are modeled as

$$
r _ {j} (t) = k _ {j} (T) f _ {j} (\mathbf {c} (t)),
$$

where $f _ { j } ( \mathbf { c } ( t ) )$ denotes the concentration-dependent kinetic term, which, under the assumption of mass action kinetics, is given by the product of reagent concentrations raised to powers corresponding to the reaction stoichiometry. The temperature dependence is modeled using Arrhenius form:

$$
k _ {j} (T) = k _ {0, j} \exp \left(- \frac {E _ {j}}{R T}\right).
$$

For compactness, the temperature dependence of the rate constants is suppressed in the following, i.e., $k _ { j } \equiv k _ { j } ( T )$ ). The elementary reaction rates corresponding to the kinetic scheme are defined as

$$
r _ {d} (t) = f k _ {d} c _ {I} (t),
$$

$$
r _ {p, n} (t) = k _ {p} c _ {M} (t) c _ {R _ {n} ^ {\bullet}} (t),
$$

$$
n \geq 1,
$$

$$
r _ {t r, n} (t) = k _ {t r} c _ {C T A} (t) c _ {R _ {n} ^ {\bullet}} (t),
$$

$$
n \geq 1,
$$

$$
r _ {t r m, n} (t) = k _ {t r m} c _ {M} (t) c _ {R _ {n} ^ {\bullet}} (t),
$$

$$
n \geq 1,
$$

$$
r _ {t r s, n} (t) = k _ {t r s} c _ {S} (t) c _ {R _ {n} ^ {\bullet}} (t),
$$

$$
n \geq 1,
$$

$$
r _ {t d, n, m} (t) = k _ {t d} c _ {R _ {n} ^ {\bullet}} (t) c _ {R _ {m} ^ {\bullet}} (t),
$$

$$
n, m \geq 1,
$$

$$
r _ {t c, n, m} (t) = k _ {t c} c _ {R _ {n} ^ {\bullet}} (t) c _ {R _ {m} ^ {\bullet}} (t),
$$

$$
n, m \geq 1.
$$

For the molecular species, the balances read

$$
\left. \frac {d c _ {I}}{d t} \right| _ {t} = - \frac {r _ {d} (t)}{f},
$$

$$
\left. \frac {d c _ {M}}{d t} \right| _ {t} = - \sum_ {n \geq 1} r _ {p, n} (t) - \sum_ {n \geq 1} r _ {t r m, n} (t),
$$

$$
\left. \frac {d c _ {C T A}}{d t} \right| _ {t} = - \sum_ {n \geq 1} r _ {t r, n} (t),
$$

$$
\left. \frac {d c _ {S}}{d t} \right| _ {t} = - \sum_ {n \geq 1} r _ {t r s, n} (t).
$$

where $f$ is the initiator eficiency and $\boldsymbol { r } _ { d } ( t ) = f \boldsymbol { k } _ { d } c _ { I } ( t )$ denotes the efective rate of polymerizing radical generation. While only a fraction $f$ of the generated primary radicals become active in polymerization, the initiator decomposes at rate $k _ { d } c _ { I } ( t )$ . Consequently, the initiator consumption is governed by the decomposition rate. For the living radical of chain length $n = 1$ , the balance is

$$
\begin{array}{l} \frac {d c _ {R _ {1} ^ {\bullet}}}{d t} \bigg | _ {t} = 2 r _ {d} (t) - r _ {p, 1} (t) - r _ {t r, 1} (t) - r _ {t r m, 1} (t) - r _ {t r s, 1} (t) \\ \qquad + \sum_ {n \geq 1} r _ {t r, n} (t) + \sum_ {n \geq 1} r _ {t r m, n} (t) + \sum_ {n \geq 1} r _ {t r s, n} (t) \\ \qquad - \sum_ {m \geq 1} r _ {t d, 1, m} (t) - \sum_ {m \geq 1} r _ {t c, 1, m} (t). \end{array}
$$

For living radicals of chain length $n \geq 2 .$ , the balances are

$$
\begin{array}{c} \frac {d c _ {R _ {n} ^ {\bullet}}}{d t} \bigg | _ {t} = r _ {p, n - 1} (t) - r _ {p, n} (t) - r _ {t r, n} (t) - r _ {t r m, n} (t) - r _ {t r s, n} (t) \\ - \sum_ {m \geq 1} r _ {t d, n, m} (t) - \sum_ {m \geq 1} r _ {t c, n, m} (t), \qquad n \geq 2. \end{array}
$$

For dead polymer chains, the balances are

$$
\begin{array}{l} \left. \frac {d c _ {D _ {n}}}{d t} \right| _ {t} = r _ {t r, n} (t) + r _ {t r m, n} (t) + r _ {t r s, n} (t) \\ \qquad + \sum_ {m \geq 1} r _ {t d, n, m} (t) + \frac {1}{2} \sum_ {m = 1} ^ {n - 1} r _ {t c, m, n - m} (t), \qquad n \geq 1. \end{array}
$$

The balances above define the formal infinite-dimensional population balance model. In principle, this leads to a system of population balances over the chain length, where the concentrations of living and dead polymer species are defined for all chain lengths n. However, directly resolving these balances is computationally demanding, as propagation continuously shifts the distribution and termination by combination generates products at chain length n + m. To obtain a numerically tractable formulation, we adopt a pivot-based discretization of the chain-length domain, following the approach of Butté et al. [43, 44]. In the implemented reference model, the infinite-dimensional balances are approximated on a finite pivot grid, and the pairwise termination interactions are closed through the total radica concentration. In this method, the chain-length distribution is represented on a finite set of pivot points, and the evolution of the distribution is approximated by projecting reaction products onto these pivots. This allows the computation of chain-length distributions while maintaining a manageable number of state variables.

In the implemented mechanistic reference model, the infinite-dimensional chain-length distributions are discretized on a finite pivot grid following the method of Butté et al. [43, 44]. After discretization, the same kinetic rate expressions are evaluated for the finite set of pivot species $i = 1 , \ldots , N _ { p } ,$ with the total radical concentration approximated by $\begin{array} { r } { c _ { R } \bullet \left( t \right) = \sum _ { i } c _ { R _ { i } ^ { } } ( t ) } \end{array}$ . The termination rates are then expressed in terms of this total radical concentration, resulting in a closure of the pairwise radical interactions [45]. The corresponding species balances read

$$
\left. \frac {d c _ {I}}{d t} \right| _ {t} = - k _ {d} c _ {I} (t),
$$

$$
\left. \frac {d c _ {M}}{d t} \right| _ {t} = - k _ {p} c _ {M} (t) c _ {R ^ {\bullet}} (t) - k _ {t r m} c _ {M} (t) c _ {R ^ {\bullet}} (t),
$$

$$
\left. \frac {d c _ {C T A}}{d t} \right| _ {t} = - k _ {t r} c _ {C T A} (t) c _ {R ^ {\bullet}} (t),
$$

$$
\left. \frac {d c _ {S}}{d t} \right| _ {t} = - k _ {t r s} c _ {S} (t) c _ {R ^ {\bullet}} (t).
$$

For the living radicals on the pivot grid, the balances are given by

$$
\begin{array}{r l} \frac {d c _ {R _ {i} ^ {\bullet}}}{d t} \bigg | _ {t} = & \delta_ {i 1}   2 f k _ {d}   c _ {I} (t) + \mathcal {P} _ {i} (c _ {R ^ {\bullet}} (t), c _ {M} (t)) \\ & - (k _ {t r}   c _ {C T A} (t) + k _ {t r m}   c _ {M} (t) + k _ {t r s}   c _ {S} (t))   c _ {R _ {i} ^ {\bullet}} (t) \\ & - (k _ {t d} + k _ {t c})   c _ {R _ {i} ^ {\bullet}} (t)   c _ {R ^ {\bullet}} (t), \qquad i = 1, \ldots , N _ {p} \end{array}
$$

For the dead polymer distribution, the balances read

$$
\begin{array}{r l} & {\left. \frac {d c _ {D _ {i}}}{d t} \right| _ {t} = (k _ {t r} c _ {C T A} (t) + k _ {t r m} c _ {M} (t) + k _ {t r s} c _ {S} (t)) c _ {R _ {i} ^ {\bullet}} (t)} \\ & {\qquad + k _ {t d} c _ {R _ {i} ^ {\bullet}} (t) c _ {R ^ {\bullet}} (t) + \mathcal {C} _ {i} (c _ {R ^ {\bullet}} (t)), \qquad i = 1, \ldots , N _ {p}} \end{array}
$$

The operator $\mathcal { P } _ { i } ( \cdot )$ represents the propagation-induced shift of radicals between neighboring pivots, and $\mathcal { C } _ { i } ( \cdot )$ denotes the projection of combination products onto the pivot grid. Both operators follow the pivot-based discretization procedure introduced by Butté et al. [43, 44]. The resulting mechanistic model provides a structured description of the polymerization dynamics. In the following, this model is reformulated in a continuous-time state-space representation to facilitate its integration with the hybrid NODE formulation introduced later.

## 2.2 State-Space Modeling in Continuous Time

The full radical and polymer chain-length distributions are discretized using the pivot method introduced in Section 2.1. Following the implementation used for data generation and repeated model evaluations during training, the chain-length distributions are approximated using 16 pivots, providing a practi cal compromise between numerical resolution and computational efort. The resulting system states, therefore, include the molecular species concentrations together with the discretized living radical and dead-polymer populations. To formulate the mechanistic reference model in state-space form, we collect all dynamic variables into the state vector

$$
\mathbf {x} _ {\mathrm{mech}} (t) = \left[ \begin{array}{c c c c c c} c _ {I} (t) & c _ {M} (t) & c _ {C T A} (t) & c _ {S} (t) & \mathbf {c} _ {R ^ {\bullet}} (t) ^ {\top} & \mathbf {c} _ {D} (t) ^ {\top} \end{array} \right] ^ {\top}
$$

with

$$
\mathbf {c} _ {R ^ {\bullet}} (t) = \left[ \begin{array}{c c c} c _ {R _ {1} ^ {\bullet}} (t) & \dots & c _ {R _ {1 6} ^ {\bullet}} (t) \end{array} \right] ^ {\top}, \qquad \mathbf {c} _ {D} (t) = \left[ \begin{array}{c c c} c _ {D _ {1}} (t) & \dots & c _ {D _ {1 6}} (t) \end{array} \right] ^ {\top}.
$$

The reactor is assumed to operate under constant-volume conditions. Consequently, V is a constant parameter rather than a time-dependent state, and all states are expressed in terms of concentrations. The initial conditions of the system are given by

$$
\mathbf {x} _ {\mathrm{mech}} (0) = \left[ \begin{array}{c c c c c c c} c _ {I, 0} & c _ {M, 0} & c _ {C T A, 0} & c _ {S, 0} & 0 & \dots & 0 \end{array} \right] ^ {\top}
$$

since initially, no radicals or polymer chains are present.

The kinetic parameters of the mechanistic model are collected in the parameter vector

$$
\mathbf {p} _ {\text { mech }} = \left[ \begin{array}{c c c c c c c c} k _ {d} & k _ {p} & k _ {t r} & k _ {t r m} & k _ {t r s} & k _ {t d} & k _ {t c} & f \end{array} \right] ^ {\top}
$$

where the parameters $k _ { j }$ denote efective, temperature-dependent rate constants, while $f \in ( 0 , 1 ]$ denotes the initiator eficiency. The operating conditions are represented by the input vector

$$
\mathbf {u} _ {\mathrm{mech}} (t) = \left[ T \right],
$$

where $T$ denotes the reactor temperature, which afects the reaction rates through Arrhenius-type dependencies.

The system dynamics can thus be written in continuous-time state-space form as

$$
\dot {\mathbf {x}} _ {\mathrm{mech}} (t) = \mathbf {f} _ {\mathrm{mech}} (\mathbf {x} _ {\mathrm{mech}} (t), \mathbf {u} _ {\mathrm{mech}} (t), \mathbf {p} _ {\mathrm{mech}}),
$$

where $\mathbf { f } _ { \mathrm { m e c h } } ( \cdot )$ represents the mechanistic reaction kinetics defined by the species balances.

The model output corresponds to the monomer conversion,

$$
y _ {\mathrm{mech}} (t) = 1 - \frac {c _ {M} (t)}{c _ {M , 0}}.
$$

This formulation provides the structural foundation for the hybrid modeling approach introduced in the following section.

## 3 Machine Learning for Dynamic Process Modeling

This section introduces the ML formulations considered in this work for modeling dynamic process behavior from time-series data. The formulations are presented to establish the continuous- and discretetime learning frameworks used later for comparison with the proposed hybrid NODE approach and to provide the notation and modeling structure required for the subsequent hybrid formulations. We first define the general learning problem for nonlinear dynamical systems and the associated training objective. We then present two model classes considered throughout this work: discrete-time neural transition models and continuous-time Neural Ordinary Diferential Equations (NODEs).

## 3.1 Learning Problem and Training Objective

We consider a general non-autonomous nonlinear dynamic system in continuous time [46]:

$$
\dot {\mathbf {x}} (t) = \mathbf {f} (\mathbf {x} (t), \mathbf {u} (t)), \quad \mathbf {x} (t _ {0}) = \mathbf {x} _ {0},\tag{1}
$$

where $\mathbf { x } ( t ) \in \mathbb { R } ^ { n _ { x } }$ denotes the state vector, ${ \bf u } ( t ) \in \mathbb { R } ^ { n _ { u } }$ the input vector, and f the nonlinear statetransition function.

In general, the full state vector is not directly observable. Instead, measurements are obtained through

$$
\mathbf {y} (t) = \mathbf {h} (\mathbf {x} (t)) + \varepsilon (t), \quad \varepsilon (t) \sim \mathcal {N} (0, \Sigma),\tag{2}
$$

where $\mathbf { y } ( t ) \in \mathbb { R } ^ { n _ { y } }$ denotes the system outputs and $ { \varepsilon } ( t ) \in \mathbb { R } ^ { n _ { y } }$ represents measurement noise.

At discrete sampling times $t _ { i } .$ the measured outputs are denoted by

$$
\mathbf {y} _ {i} := \mathbf {y} (t _ {i}) + \varepsilon_ {i}.
$$

Given time-series data

$$
\mathcal {D} = \{\mathbf {u} (t _ {i}), \mathbf {y} _ {i} \} _ {i = 1} ^ {N},
$$

the learning problem consists of estimating the trainable parameter vector of the neural network θ such that the model predictions match the observed measurements. This is formulated as the following training objective:

$$
\min _ {\theta} \frac {1}{N} \sum_ {i = 1} ^ {N} \| \hat {\mathbf {y}} _ {\theta} (t _ {i}) - \mathbf {y} _ {i} \| _ {2} ^ {2} + \lambda_ {\text { reg }} \mathcal {R} (\theta),\tag{3}
$$

where $\hat { \bf y } _ { \pmb { \theta } } ( t _ { i } )$ denotes the model prediction at time $t _ { i } .$ , and ${ \mathcal { R } } ( \theta )$ is a regularization term, $\mathrm { e . g . }$ ., an $\ell _ { 2 }$ penalty on the network weights, weighted by $\lambda _ { \mathrm { r e g } } \geq 0$

Depending on the model class, we obtain predictions either by recursive discrete-time updates or by solving a continuous-time initial-value problem.

## 3.2 Discrete-Time Neural Modeling

A general discrete-time approximation of a nonlinear dynamical system can be written as

$$
\mathbf {x} _ {k + 1} = \mathbf {F} (\mathbf {x} _ {k}, \mathbf {u} _ {k}),\tag{4}
$$

where $\mathbf { x } _ { k } \ \in \ \mathbb { R } ^ { n _ { x } }$ denotes the system state at time step k, $\mathbf { u } _ { k } \in \mathbb { R } ^ { n _ { u } }$ the corresponding input, and $\mathbf { F } : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } \to \mathbb { R } ^ { n _ { x } }$ the discrete-time state-transition mapping [47]. When $\mathbf { F }$ is represented by a neural network with parameters $\theta ,$ this yields:

$$
\mathbf {x} _ {k + 1} = \mathbf {F} _ {\boldsymbol {\theta}} ^ {\mathrm{disc}} \big (\mathbf {x} _ {k}, \mathbf {u} _ {k}, \Delta t _ {k} \big),\tag{5}
$$

where θ denotes the trainable parameters of the neural network. Here, $\Delta t _ { k }$ is included to account for variable time steps, enabling learning from irregularly sampled data. This formulation is purely data-driven and treats the dynamics as a static regression problem that maps $\left( { \bf x } _ { k } , { \bf u } _ { k } , \Delta t _ { k } \right)$ to $\mathbf { x } _ { k + 1 }$ at each time step. While simple to implement, it does not explicitly enforce an underlying continuoustime structure, and its performance depends strongly on the chosen time discretization and the density of available training data, typically requiring suficiently dense time-series measurements to accurately capture fast dynamics and avoid instability in recursive predictions [47].

## 3.3 Neural Ordinary Diferential Equations

Neural Ordinary Diferential Equations (NODEs) parameterize the continuous-time dynamics directly [48]:

$$
\dot {\mathbf {x}} (t) = \mathbf {f} _ {\boldsymbol {\theta}} ^ {\mathrm{NODE}} \big (\mathbf {x} (t), \mathbf {u} (t) \big),\tag{6}
$$

where $\mathbf { x } ( t ) \in \mathbb { R } ^ { n _ { \mathrm { { a } } } }$ x denotes the system state, $\mathbf { u } ( t ) \in \mathbb { R } ^ { n _ { u } }$ the input vector, and $\mathbf { f } _ { \theta } ^ { \mathrm { N O D E } } : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } \to \mathbb { R } ^ { n _ { x } }$ is a neural network with trainable parameters θ that approximates the continuous-time state-transition function.

We obtain the state trajectory by solving the corresponding initial-value problem with initial condition $\mathbf { x } ( t _ { 0 } ) = \mathbf { x } _ { 0 } \colon$

$$
\mathbf {x} (t) = \mathbf {x} _ {0} + \int_ {t _ {0}} ^ {t} \mathbf {f} _ {\pmb {\theta}} ^ {\mathrm{NODE}} \big (\mathbf {x} (\tau), \mathbf {u} (\tau) \big) d \tau ,\tag{7}
$$

We train the model parameters by minimizing the objective in Equation (3), where we obtain predicted trajectories by solving the corresponding initial-value problem. Gradients are computed via backpropagation through the ODE solver, typically using adjoint sensitivity methods. For further details on NODE training and adjoint-based methods, we refer to [33].

This formulation treats the neural network as a parameterization of the continuous-time statetransition function and provides a flexible framework for learning dynamics directly from time-series data. It forms the foundation for the hybrid modeling approach introduced in the following section. In contrast to discrete-time models, NODEs directly model the time derivative of the state and re construct trajectories through integration using a diferentiable ODE solver. As a result, the model enforces a continuous-time dynamic structure and can naturally accommodate irregularly sampled mea surements [33].

## 4 Hybrid Neural Ordinary Diferential Equations

As discussed in the Introduction, purely data-driven models for polymerization dynamics are dificult to train reliably under limited and noisy data conditions [7]. At the same time, the overall structure of FRP dynamics is characterized through well-established reaction mechanisms and mass balances, as presented in Section 2, while several constitutive kinetic contributions remain uncertain or dificult to model accurately. This mechanistic structure can therefore be incorporated directly into the mode formulation to reduce the amount of information that must be inferred from data alone. In the following, we first introduce the general hybrid NODE framework and then subsequently specialize it to the FRP system considered in this work.

## 4.1 Hybrid NODE Formulation

Our approach is to retain the known mechanistic structure of the polymerization system and introduce data-driven learning only for the unresolved kinetic contribution required to complete the reduced mech anistic model. In particular, the mechanistic mass balances and established kinetic relations are retained explicitly, while the efective radical concentration governing monomer consumption is approximated through a neural-network surrogate. The neural component, therefore, does not replace the full mech anistic model but instead complements it by learning only the part of the dynamics that is dificult to characterize accurately within the reduced mechanistic formulation. Unlike conventional hybrid correc tion approaches, the proposed formulation does not learn additive corrections to an existing mechanistic model but instead learns the unresolved closure relation required to complete an otherwise structurally consistent mechanistic description. Here, the term “closure” refers to the representation of an unresolved quantity required to complete the mechanistic description of the reduced system dynamics.

To implement this approach, we employ a hybrid Neural Ordinary Diferential Equation (NODE) formulation. This choice is motivated by the fact that polymerization processes are naturally described by continuous-time ordinary diferential equations derived from reaction kinetics, as shown in Section 2. NODEs provide a compatible framework in which the system dynamics are learned directly in continuous time, allowing mechanistic and learned components to be embedded within the same system of diferential equations.

Within this framework, the system dynamics are written as

$$
\dot {\mathbf {x}} (t) = \mathbf {f} _ {\mathrm{phys}} \left(\mathbf {x} (t), \mathbf {u} (t), \mathbf {f} _ {\boldsymbol {\theta}} ^ {\mathrm{hyb}} (\mathbf {x} (t), \mathbf {u} (t))\right),\tag{8}
$$

where $\mathbf { f } _ { \mathrm { p h y s } }$ represents the mechanistic balance equations and $\mathbf { f } _ { \theta } ^ { \mathrm { h y b } }$ approximates the unresolved constitutive contribution required to close the system. The continuous-time formulation preserves consistency with the underlying reaction kinetics while embedding the learned closure relation directly within the governing diferential equations.

In the following, this formulation is specialized to the FRP system considered in this work.

## 4.2 Hybrid NODE Formulation for Free-Radical Polymerization Dynamics

We now specify the general hybrid NODE formulation introduced in Section 4.1 to the FRP system described in Section 2. Based on the reaction mechanism, monomer is consumed primarily through propagation and, to a lesser extent, through chain transfer to monomer [37]. The monomer balance can therefore be written as:

$$
\left. \frac {d c _ {M}}{d t} \right| _ {t} = - (k _ {p} + k _ {t r m}) c _ {M} (t) c _ {R ^ {\bullet}} (t),\tag{9}
$$

$$
\left. \frac {d c _ {M}}{d t} \right| _ {t} \approx - k _ {p}   c _ {M} (t)   c _ {R ^ {\bullet}} (t),\tag{10}
$$

where $\begin{array} { r } { c _ { R } \bullet \left( t \right) = \sum _ { i } c _ { R _ { i } ^ { } } ( t ) } \end{array}$ denotes the total radical concentration.

Following the hybrid modeling strategy, we retain this balance structure and focus on closing the term that is dificult to describe accurately, namely the total radical concentration. This quantity results from the coupled efects of initiation, propagation, transfer, and termination reactions, and its evaluation in the full mechanistic model depends on the discretized radical population balance. We reformulate the monomer balance using the logarithmic state [37]

$$
z (t) = \log \left(\frac {c _ {M} (t)}{c _ {M , 0}}\right).\tag{11}
$$

In particular, the total radical concentration is represented by a neural surrogate,

$$
c _ {R ^ {\bullet}} (t) \approx \widehat {R} _ {\theta} \big (z (t), \mathbf {u} \big),\tag{12}
$$

where $\widehat { R } _ { \theta }$ denotes a learned approximation that closes the monomer balance, $z ( t )$ denotes the logarithmic state, and u represents the operating conditions.

The resulting hybrid monomer balance becomes

$$
\left. \frac {d c _ {M}}{d t} \right| _ {t} = - k _ {p} c _ {M} (t) \widehat {R} _ {\theta} \big (z (t), \mathbf {u} \big),\tag{13}
$$

Applying the chain rule, we obtain

$$
\dot {z} (t) = \frac {1}{c _ {M} (t)} \left. \frac {d c _ {M}}{d t} \right| _ {t}.\tag{14}
$$

Substituting the hybrid monomer balance yields

$$
\dot {z} (t) = - k _ {p} \widehat {R} _ {\theta} \big (z (t), \mathbf {u} \big), \qquad y (t) = 1 - \exp \big (z (t) \big),\tag{15}
$$

The neural surrogate $\widehat { R } _ { \theta }$ is embedded directly within the NODE formulation and trained end-to-end through the governing diferential equations.

In summary, the proposed formulation combines mechanistic balance equations with a learned closure relation for the radical concentration, yielding a structure-consistent hybrid NODE model with the flexibility required to capture complex reactions such as FRP dynamics under limited data availability.

## 5 Model Implementation and Training

This section describes the implementation of the mechanistic, discrete-time, NODE, and hybrid NODE models introduced in Sections 2, 3, and 4, together with the data generation procedure, evaluation scenarios, and training methodology used to assess their performance for FRP dynamics.

## 5.1 Reference Trajectories and Data Generation

We generate all training and evaluation data using the mechanistic simulator introduced in Section 2 for the FRP of methyl methacrylate (MMA), using the pivot discretization approach of Butté and Morbidelli [43, 44]. The simulator provides time-resolved concentration trajectories for the molecular species, from which the monomer conversion is computed as [49]

$$
X (t) = 1 - \frac {c _ {M} (t)}{c _ {M , 0}}.\tag{16}
$$

Each trajectory consists of 25 normalized time points over a fixed batch duration of three hours. To evaluate model performance under limited-data conditions, we use only the first 10 time points for training, while the remaining 15 points constitute the evaluation segment. We apply this split consistently across all scenarios.

## 5.2 Evaluation Scenarios

The evaluation scenarios examine the three model formulations under progressively more challenging data conditions. In all scenarios, we keep the model structures and loss formulations consistent across models. Only the data sampling pattern and recipe variability change.

Scenario 1: Limited data with regular sampling. In the first scenario, we consider a single trajectory on a regular time grid with 25 normalized time points. We train all models using only the first

10 time points and evaluate predictions over the full 25-point horizon. This setting isolates the ability of each model to reconstruct a smooth conversion trajectory from sparse but regularly sampled data.

Scenario 2: Limited data with irregular sampling. In the second scenario, we consider the same limited-data setting, but we replace the regular grid with irregular time points. We keep the number of training points fixed at 10 and evaluate over the full trajectory. By changing only the sampling pattern, we assess how sensitive each model is to nonuniform measurement intervals while keeping the data quantity and training procedure unchanged. This experiment evaluates robustness to irregular measurement schedules, which are common in practical polymerization experiments [50].

Scenario 3: Generalization under noisy data. In the third scenario, we evaluate generalization to unseen operating conditions in the presence of measurement noise. We generate full conversion trajec tories using the mechanistic simulator and add Gaussian noise with standard deviation $\sigma = 0 . 0 2$ for the conversion observations. We generate training trajectories by randomly sampling temperature, initial monomer concentration, initiator concentration, chain-transfer agent concentration, and solvent concen tration over representative ranges. In this scenario, the models are trained on a limited set of complete noisy trajectories and then predict the conversion dynamics for completely unseen recipes. This setting tests the ability of each formulation to preserve physically consistent behavior when both the operating conditions and the measurement realizations difer from the training data.

## 5.3 Training Procedure

We implement all models in Python using PyTorch [51] for automatic diferentiation and parameter optimization, and torchdiffeq [52] for diferentiable ODE integration. We generate all reference trajectories using the mechanistic simulator described in Section 2, and the resulting system of diferential equations is integrated using the LSODA solver from SciPy [53].

We evaluate the three model formulations introduced in Sections 3 and 4: the discrete-time FNN model, the continuous-time NODE, and the hybrid NODE. In all cases, we optimize the model parameters by minimizing the mean-squared prediction error defined in Equation (3). All trainable parameters are optimized using the AdamW optimizer [54].

To promote physically consistent behavior, we include soft penalty terms to discourage nonphysical predictions as they enforce nondecreasing conversion and bounded outputs. We implement these penalties using ReLU-based functions that penalize decreases in conversion or violations of the admissible range $0 \leq X \leq 1$

For the discrete-time FNN model, we learn one-step transitions of the form $X _ { k + 1 }$ from $\left( X _ { k } , u _ { k } , \Delta t _ { k } \right)$ and generate full trajectories through recursive application. For both regular and irregular sampling scenarios, we train the model for 1500 epochs on the first ten time points. In the unseen-recipe generalization scenario, we train it for 1000 epochs on full trajectories consisting of 25 time points.

For the continuous-time NODE, we parameterize the right-hand side of the ODE as described in Section 3 and train it end-to-end through the ODE solver. For both regular and irregular sampling scenarios, we integrate the system using the dopri5 solver and train it for 1500 epochs. In the unseen recipe generalization scenario, we integrate the system using an explicit rk4 scheme and train it for 1000 epochs on full trajectories.

For the hybrid NODE, we retain the mechanistic initiator and monomer mass balances and represent the unresolved radical contribution governing the monomer consumption rate through the learned closure relation described in Section 4.2. We train the hybrid model with the same number of epochs as the data-driven models in each scenario (1500 epochs in the regular and irregular cases, 1000 epochs in the generalization case). We integrate the hybrid system using dopri5 in the single-trajectory scenarios and rk4 in the generalization scenario.

Across all three evaluation scenarios, we keep the model structures, optimization settings, and training protocols constant within each comparison. Between the first two scenarios, only the measurement sampling pattern is modified.

## 6 Results and Discussion

We evaluate the three model formulations under the scenarios defined in Section 5. In all scenarios, each trajectory contains 25 time points; training uses the first 10 points (up to t ≈ 0.3 in normalized time), and evaluation uses the remaining 15 points. Model performance is evaluated by comparing predicted and reference conversion trajectories over the full time horizon.

## 6.1 Regular Sampling: Single-Trajectory Reconstruction

We first examine a single polymerization trajectory sampled on an equidistant time grid. Figure 1 shows the predicted conversion trajectories of the three models together with the mechanistic reference.

![](images/836e6962615631449964fd68d6cd4c7421a7397c5fa6ebceaa573cb117bf56f9.jpg)  
Figure 1: Regular sampling, single-trajectory results. Model predictions compared to the mechanistic reference trajectory, highlighting the reconstruction and extrapolation behavior beyond the training region. The vertical dashed line indicates the end of the training region and the beginning of the extrapolation region.

All models follow the reference closely within the training window, where the predictions are directly constrained by the available data. Beyond the training cutof (indicated by the vertical dashed line), clear diferences in extrapolation behavior become apparent.

The discrete-time FNN model formulated according to the discrete-time state-space model introduced in Section 3, learns a purely data-driven transition mapping between successive time steps. During evalu ation, the model is recursively applied to generate the full trajectory. Although this formulation enforces a stepwise temporal structure, it remains entirely unconstrained by mechanistic relations. As shown in Figure 1, the discrete-time FNN model exhibits a pronounced overprediction after the training window and continues to diverge at intermediate and late times. Recursive application of the learned transition mapping amplifies local prediction errors, and the model fails to reproduce the correct curvature and late-stage slowdown of the conversion profile. In this limited-data regime, the discrete-time data-driven formulation does not provide reliable extrapolation.

The fully data-driven NODE produces a smooth trajectory and clearly improves upon the discrete time FNN model. However, it displays a steeper slope than the mechanistic reference already within the training region. Although it captures the overall trajectory trend more consistently than the discrete time FNN formulation, it does not accurately reproduce the curvature and late-time slowdown of the mechanistic reference trajectory. The learned continuous-time vector field therefore overestimates the reaction rate within the training window, and this bias accumulates during forward integration. As a result, the model exhibits overprediction at intermediate and late times. While the deviation is less severe than that of the discrete-time FNN model, it remains clearly visible in the trajectory.

In contrast, the hybrid NODE closely follows the reference trajectory over the entire time horizon. After the training cutof, it maintains a slope consistent with the mechanistic reference and continues to track the conversion trend at late times, indicating improved extrapolation capability compared to both purely data-driven models. The improved agreement with the mechanistic reference across both the training and extrapolation regions can be attributed to the hybrid formulation introduced in Equa tion (8). Rather than learning the full system dynamics from data, the hybrid NODE only learns the unresolved efective radical contribution governing monomer consumption, while the remaining mass balances and kinetic dependencies are retained from the mechanistic model. As a result, the learning task is significantly simplified. In the limited-data regime considered here, the available measurements are insuficient to reliably identify the full dynamics, as required by the purely data-driven models, but are suficient to identify the lower-dimensional learned contribution embedded within the mechanistic balances. This leads to improved generalization and physically consistent extrapolation behavior.

In conclusion, the results of Figure 1 support the central premise of the proposed hybrid NODE formulation that restricting the learning task to the unresolved kinetic contribution improves the extrapolation capabilities of the model and reduces the amount of data required for accurate prediction even when trained on as few as 10 data points.

## 6.2 Irregular Sampling: Single-Trajectory Reconstruction

We repeated the single-trajectory experiment under irregular sampling while keeping the number of measurements, the training-evaluation split, and all model settings unchanged. Only the measurement schedule was modified such that the time points were no longer regularly spaced. Consequently, the early time dynamics were observed at irregular intervals, altering how information about the reaction kinetics was represented in the training data. To isolate the efect of irregular sampling, all model architectures, optimization settings, loss formulations, and training epochs were kept identical to the regular-sampling scenario.

![](images/3016b963a395c5b37933b5e85c3f7af4c6d9324c5290bd2028f53352202ec2a7.jpg)  
Figure 2: Irregular sampling, single-trajectory results. Model predictions are compared to the mechanis tic reference trajectory, highlighting the reconstruction and extrapolation behavior beyond the training region. The vertical dashed line indicates the end of the training region and the beginning of the extrap olation region.

Figure 2 shows that all models capture the general early-time trend, although deviations are already visible within the training region. In particular, the fully data-driven NODE exhibits a consistently steeper slope than the mechanistic reference, indicating a bias in the learned dynamics. Beyond the training window, the diferences in extrapolation behavior become more pronounced.

Under irregular sampling, after the cutof, the discrete-time FNN systematically overpredicts con version and predicts complete conversion earlier than the mechanistic reference. Recursive trajectory propagation amplifies local prediction errors, and the varying time increments further increase the sensitivity of the learned transition mapping, resulting in a clear late-time deviation. The fully data-driven NODE produces smoother trajectories but maintains its slope bias. Nevertheless, it exhibits a slightly steeper slope than the mechanistic reference already within the training window, and this bias is carried into the extrapolation region, resulting in a consistent upward deviation over the full horizon. Meanwhile, the hybrid NODE remains closest to the reference trajectory over the entire time horizon. While a slight upward deviation is observed at later times, the model preserves the overall curvature despite the irregular distribution of training points, and no systematic drift is observed after the training cutof. The improved agreement of the hybrid model with the mechanistic reference is consistent with the formula tion introduced in Equation (8). Since only the closure term is learned while the mechanistic balances are retained, the model does not need to infer the full system dynamics from irregularly sampled data.

Under irregular sampling, the early-time dynamics are represented at nonuniform time intervals, which afects how information about the reaction rate is captured. As a result, purely data-driven models learn inconsistent dynamics. The fully data-driven NODE exhibits a biased slope due to the limited and irregularly sampled training data. For the discrete-time FNN, the transition mapping in Equation (5) depends on the time increment $\Delta t _ { k }$ . While this enables the model to handle irregular sampling, it also increases the variability of the learned mapping, as the model must learn transitions across a wider range of time scales. In the sparse-data regime, this leads to increased sensitivity to loca errors, which are subsequently amplified after the training cutof.

The improved agreement of the hybrid model with the mechanistic reference seen in Figure 2 can be attributed to two complementary aspects of the formulation. The continuous-time NODE representation in Equation (15) avoids the explicit dependence on the time increments and is therefore less sensitive to irregular sampling. In addition, embedding the neural component within the mechanistic balance restricts the learning task to the unresolved efective radical contribution, which reduces the amount of dynamic information that must be inferred from the limited data. Together, these properties lead to more stable and physically consistent predictions compared to both purely data-driven models.

In summary, irregular sampling increased the dificulty of learning the early-time dynamics, yet the hybrid NODE continued to exhibit the most accurate and physically consistent extrapolation behavior, whereas the purely data-driven models exhibited systematic deviations. These results further support the advantage of embedding mechanistic structure directly within the continuous-time learning formulation.

## 6.3 Generalization Across Unseen Recipes Under Noisy Measurements

In this scenario, we evaluate generalization across unseen recipes under noisy measurements. We generate full conversion trajectories using the mechanistic simulator and perturb the conversion trajectories with additive Gaussian noise $( \sigma = 0 . 0 2 )$ . We train all models on the noisy conversion observations and evaluate them against the corresponding noise-free mechanistic trajectories to isolate generalization error from measurement noise.

For each trajectory, we sample the initial monomer concentration $M _ { 0 }$ , initiator concentration $I _ { 0 } .$ , chain transfer agent concentration $C T A _ { 0 }$ , and solvent concentration $S _ { 0 }$ independently from continuous uniform ranges: $M _ { 0 } \in [ 3 . 0 , 5 . 0 ]$ $I _ { 0 } \in [ 0 . 0 5 , 0 . 1 2 ]$ , CT $\mathrm { \Delta } A _ { 0 } ~ \in ~ [ 0 . 0 0 , 0 . 0 6 ]$ , and $S _ { 0 } \in [ 3 . 0 , 5 . 0 ]$ . We select training temperatures from the discrete set $\{ 5 5 , 6 0 \} ^ { \circ } \mathrm { C }$ and evaluate performance at an unseen temperature of $7 0 ^ { \circ } \mathrm { C }$ . The test trajectories, therefore, difer from the training data both in temperature and in the specific combinations of initial conditions, resulting in previously unseen operating conditions. Table 1 lists the representative training and test recipes used for visualization.

We use the same three model formulations introduced previously without modifying the model archi tectures. We keep the training procedure and the number of training epochs consistent across models, ensuring that performance diferences arise from model formulation rather than tuning. Although the models are trained on complete trajectories, the overall dataset remains small, consisting of only two training trajectories with 25 time points each. In Figure 3, we present one representative training tra jectory and one representative test trajectory.

Table 1: Representative training and test recipes for the unseen-operating-condition generalization sce nario.

<table><tr><td></td><td>Training trajectory</td><td>Test trajectory</td></tr><tr><td> $T$  [°C]</td><td>60.0</td><td>70.0</td></tr><tr><td> $M_0$ </td><td>4.3647</td><td>3.3422</td></tr><tr><td> $I_0$ </td><td>0.05377</td><td>0.11313</td></tr><tr><td> $CTA_0$ </td><td>0.01322</td><td>0.00851</td></tr><tr><td> $S_0$ </td><td>3.3687</td><td>4.2225</td></tr></table>

Figure 3 summarizes the results for the unseen-condition generalization scenario under noisy training data. Subfigure (a) shows a representative training trajectory at $6 0 ^ { \circ } \mathrm { C }$ , including the noisy observations used for training and the corresponding model predictions. Subfigure (b) presents a representative test trajectory at the unseen temperature of $7 0 ^ { \circ } \mathrm { C } .$ , where all models are evaluated against the mechanistic reference. Together, these subfigures illustrate how well each model fits the noisy training data and how well the learned dynamics generalize to an unseen operating condition.

![](images/282608d9441e2eb980b863e128cfd501b14042bd669d9629e4de7739ba8c33f7.jpg)

(a) Training trajectory (with noisy observations).  
![](images/d53a37feed8c799c2d5a9341c56e06016358666ae5e2425d8837eb8e3eaae9ab.jpg)  
(b) Test trajectory at unseen temperature (evaluated against the mechanistic reference).  
Figure 3: Generalization under noisy data. Model predictions for (a) the training trajectory with noisy observations and (b) the test trajectory at an unseen temperature, highlighting fitting behavior on the noisy training trajectory and generalization to an unseen operating condition.

In Figure 3a, all models follow the noisy observations within the training trajectory, which is expected since the models are evaluated on the training dataset. The diferences become pronounced in the test trajectory in Figure 3b at the unseen temperature of $7 0 ^ { \circ } \mathrm { C }$ . The discrete-time FNN fails to generalize to unseen conditions. Although it fits the noisy training trajectory, it predicts a significantly slower increase in conversion and remains far below the mechanistic reference throughout the batch. This failure to generalize is expected, as the second recipe lies entirely outside the training data, and the discrete-time FNN model, as a purely data-driven transition model, has no mechanism to infer how the dynamics should change under diferent temperature and initial conditions. Instead, the discrete-time FNN model efectively learns an averaged transition behavior representative of the training conditions and applies it unchanged to the unseen recipe. This limitation leads to severe underestimation of the reaction rate and large prediction errors over the full time horizon.

The continuous-time NODE improves upon the discrete-time FNN formulation but still substantially underestimates the reaction rate at 70 <sup>◦</sup>C. Although the predicted conversion increases more rapidly than for the discrete-time FNN model, the learned dynamics fail to reproduce the accelerated kinetics observed in the mechanistic reference trajectory. Because the NODE must infer the full dynamic vector field from a limited and noisy dataset, the learned dynamics do not extrapolate reliably when both temperature and initial conditions change simultaneously.

In contrast, the hybrid NODE remains in close agreement with the mechanistic reference across the full trajectory. It accurately reproduces the rapid rise in conversion and follows the reference curve without noticeable systematic deviation. This improved agreement arises from the hybrid formulation, in which the mechanistic balances preserve the temperature dependence of the reaction while the neural component learns only the unresolved efective radical contribution governing monomer consumption. As a result, the model generalizes efectively to unseen conditions even when trained on limited and noisy data.

To quantify predictive accuracy, we report the root-mean-square error (RMSE) between the predicted conversion and the mechanistic reference over the full batch horizon. The results in Table 2 clearly support the qualitative observations in Figure 3. The discrete-time FNN exhibits the largest error, reflecting its inability to generalize beyond the training distribution. The fully data-driven NODE reduces the deviation but still shows substantial error, indicating that learning the full system dynamics directly from limited and noisy data remains challenging for reliable extrapolation.

Table 2: Test performance (RMSE) on the unseen recipe.

<table><tr><td>Model</td><td>RMSE on test trajectory</td></tr><tr><td>FNN</td><td>0.68</td></tr><tr><td>NODE</td><td>0.31</td></tr><tr><td>Hybrid NODE</td><td>0.013</td></tr></table>

The hybrid NODE achieves a substantially lower error compared to the fully data-driven models, demonstrating accurate prediction of the conversion dynamics under unseen conditions. This result highlights the advantage of the hybrid formulation in addressing one of the central challenges in ML for dynamic systems, namely, learning from limited data. By embedding the neural component within the mechanistic balance equations, the model leverages mechanistic knowledge to constrain the system dynamics while learning only the unresolved efective radical contribution. As a result, the model does not need to infer the full dynamics from data alone but instead complements the available mechanistic structure.

This combination enables the hybrid model to generalize to unseen operating conditions, including diferent temperatures, while maintaining physically consistent behavior. The results, therefore, show that integrating prior knowledge within the learning framework is not only beneficial for extrapolation but also essential for achieving reliable predictions in data-scarce regimes.

## 7 Conclusion and Outlook

This work demonstrates that a hybrid NODE formulation, in which mechanistic polymerization balances are combined with a learned representation of the unresolved efective radical contribution governing monomer consumption, can accurately predict FRP dynamics under limited-data conditions. Across all evaluation scenarios, the hybrid NODE achieved more accurate and physically consistent extrapolation behavior than the purely data-driven formulations. These results indicate that embedding mechanistic structure directly within the continuous-time learning formulation substantially reduces the amount of dynamic information that must be inferred from data alone. This formulation directly addresses a central challenge in polymerization modeling, namely that key kinetic contributions are only partially known and available data are limited, making it dificult to reliably model the full system dynamics using either purely mechanistic or purely data-driven approaches. In addition, classical mechanistic modeling of FRP typically requires extensive kinetic characterization and parameter estimation, which can become challenging under limited data availability. By combining both, the hybrid formulation leverages existing physical knowledge while using data to learn the remaining unknown contributions.

In the generalization across unseen recipes under a noisy measurements scenario, the hybrid model achieves an RMSE more than an order of magnitude lower than both the discrete-time FNN model and the continuous-time NODE. These results show that learning only the unresolved constitutive contribution governing monomer consumption, rather than the full system dynamics, is suficient to achieve accurate predictions even with limited, irregular, and noisy data. In contrast to conventional hybrid correction approaches, the proposed formulation does not learn additive corrections to mechanistic trajectories but instead learns the unresolved constitutive contribution required to complete the reduced mechanistic description.

The results also highlight limitations of the purely data-driven formulations considered in this work. Both the discrete-time FNN model and the continuous-time NODE exhibited systematic deviations when evaluated on unseen operating conditions, particularly when extrapolation in both temperature and initial conditions was required. Since these formulations attempt to infer the full system dynamics directly from limited and noisy data, the learned representations remained sensitive to changes in temperature and initial conditions.

These findings align with the growing body of literature on theory-guided and physics-informed ML [24, 25, 35]. In particular, the superior performance of the hybrid NODE under sparse, noisy, and extrapolative conditions directly supports the observations of Ge et al. [12], who highlight the limitations of purely data-driven approaches in polymer systems and advocate for hybrid frameworks that embed mechanistic knowledge. The results presented here provide concrete evidence, in the context of FRP, that such structure-consistent hybridization improves both predictive accuracy and generalization.

At the same time, the present study considers a batch polymerization system and evaluates the methodology using simulator-generated data. The mechanistic simulator, therefore, provides a con trolled benchmark for systematically analyzing the influence of sparse, irregular, and noisy data on the diferent learning formulations. Practical polymerization systems introduce additional challenges, includ ing experimental noise, model mismatch, transport efects, and unobserved disturbances, which were not considered in the present study.

Beyond predictive accuracy, the hybrid NODE provides a practical way to extend classical kinetic models. In polymerization systems, several kinetic contributions are often simplified or treated as con stants for tractability, despite their known dependence on conversion, chain length, or transport efects. Within the hybrid formulation, such contributions do not need to be specified explicitly. Instead, they can be learned from data through a NODE component embedded within the mechanistic balances, ensur ing that the resulting dynamics remain physically consistent while incorporating efects that are dificult to model analytically.

Future work should investigate the application of the proposed hybrid NODE formulation to experimental polymerization datasets, including systems with transport limitations, temperature dynamics and more complex reaction mechanisms. In addition, extending the framework toward simultaneous prediction of conversion and polymer property distributions, such as molecular-weight distributions, represents an important next step toward practical polymerization process modeling and control. More broadly, the results suggest that hybrid continuous-time learning formulations provide a promising frame work for reaction systems in which partial mechanistic knowledge is available but dificult-to-characterize kinetic contributions remain unresolved.

## Acknowledgments

This project (GA number 101072732) has received funding from the HORIZON-MSCA-2021-DN-01 call of the research and innovation programme of Horizon Europe 2021 under the Marie Skłodowska-Curie actions. The authors acknowledge the support of the Werner Siemens Foundation in the frame of the WSS Research Centre “catalaix”.

We also gratefully acknowledge Prof. Nicholas Ballard for valuable discussions and guidance on the mechanistic modeling of polymerization processes, as well as Jannik Lüthje for helpful insights related to the systematic formulation of reaction models involving equilibrium and kinetically-limited steps.

## References

[1] M. Ohshima, M. Tanigaki, Quality control of polymer production processes, Journal of Process Control 10 (2) (2000) 135–148. doi:https://doi.org/10.1016/S0959-1524(99)00042-6.

[2] M. Karuppusamy, R. Thirumalaisamy, S. Palanisamy, S. Nagamalai, E. E. S. Massoud, N. Ayrilmis, A review of machine learning applications in polymer composites: advancements, challenges, and future prospects, Journal of Materials Chemistry A 13 (22) (2025) 16290–16308. doi:10.1039/ D5TA00982K.

[3] J. M. Asua, Emulsion polymerization: From fundamental mechanisms to process developments, Journal of Polymer Science Part A: Polymer Chemistry 42 (5) (2004) 1025–1041. doi:https: //doi.org/10.1002/pola.11096.

[4] R. Guerrero-Santos, E. Saldívar-Guerra, I. Zapata-González, J. Bonilla-Cruz, E. Vivaldo-Lima, Freeradical polymerization, Polymer Science, Engineering, and Sustainability 1 (2025) 65–95. doi: https://doi.org/10.1002/9781119820123.ch3.

[5] K. Farajzadehahary, S. Hamzehlou, N. Ballard, Adding machine learning to the polymer reaction engineering toolbox, Progress in Polymer Science 170 (2025) 102029. doi:https://doi.org/10. 1016/j.progpolymsci.2025.102029.

[6] Y. Fang, H. Gao, Kinetic modeling for radical polymerization and depolymerization, Current Opinion in Chemical Engineering 49 (2025) 101152. doi:https://doi.org/10.1016/j.coche.2025. 101152.

[7] N. E. Jackson, B. M. Savoie, Ten problems in polymer reactivity prediction, Macromolecules 58 (4) (2025) 1737–1754.

[8] B. B. Noble, M. L. Coote, First principles modelling of free-radical polymerisation kinetics, Interna tional Reviews in Physical Chemistry 32 (3) (2013) 467–513. doi:10.1080/0144235X.2013.797277.

[9] G. R. Jones, H. S. Wang, K. Parkatzidis, R. Whitfield, N. P. Truong, A. Anastasaki, Reversed controlled polymerization (rcp): depolymerization from well-defined polymers to monomers, Journal of the American Chemical Society 145 (18) (2023) 9898–9915. doi:10.1021/jacs.3c00589.

[10] N. Wulkow, R. Telgmann, K.-D. Hungenberg, C. Schütte, M. Wulkow, Deterministic and stochastic parameter estimation for polymer reaction kinetics i: theory and simple examples, Macromolecular Theory and Simulations 30 (6) (2021) 2100017. doi:https://doi.org/10.1002/mats.202100017.

[11] A. M. Schweidtmann, A. D. Clayton, N. Holmes, E. Bradford, R. A. Bourne, A. A. Lapkin, Machine learning meets continuous flow chemistry: Automated optimization towards the pareto front of multiple objectives, Reaction Chemistry & Engineering 5 (5) (2020) 1075–1085. doi:https://doi. org/10.1016/j.cej.2018.07.031.

[12] W. Ge, R. De Silva, Y. Fan, S. A. Sisson, M. H. Stenzel, Machine learning in polymer research, Advanced Materials 37 (11) (2025) 2413695. doi:https://doi.org/10.1002/adma.202413695.

[13] M. von Stosch, R. Oliveira, J. Peres, S. Feyo de Azevedo, Hybrid semi-parametric modeling in process systems engineering: Past, present and future, Computers & Chemical Engineering 60 (2014) 86–101. doi:https://doi.org/10.1016/j.compchemeng.2013.08.008.

[14] Y. Wang, Y. Fang, H. Zhou, H. Gao, A machine learning model for predicting the propagation rate coeficient in free-radical polymerization, Molecules 29 (19) (2024) 4694. doi:10.3390/ molecules29194694.

[15] Y. Shi, J. Wang, Q. Wang, Q. Jia, F. Yan, Z.-H. Luo, Y.-N. Zhou, Supervised machine learning algorithms for predicting rate constants of ozone reaction with micropollutants, Industrial & Engineering Chemistry Research 61 (24) (2022) 8359–8367. doi:10.1021/acs.iecr.1c04697.

[16] D. Li, Y. Ru, J. Liu, Gatboost: Mining graph attention networks-based important substructures of polymers for a better property prediction, Materials Today Communications 38 (2024) 107577. doi:https://doi.org/10.1016/j.mtcomm.2023.107577.

[17] E. Inae, Y. Liu, Y. Zhu, J. Xu, G. Liu, R. Zhang, T. Luo, M. Jiang, Modeling Polymers with Neural Networks, American Chemical Society, 2025.

[18] D. Mora-Mariano, A. Flores-Tlacuahuac, I. Zapata-González, E. Saldívar-Guerra, Data-driven deep learning prediction of full molecular weight distribution in polymerization processes, The Canadian Journal of Chemical Engineering 103 (8) (2025) 3713–3725. arXiv:https://onlinelibrary. wiley.com/doi/pdf/10.1002/cjce.25635, doi:https://doi.org/10.1002/cjce.25635.

[19] A. Bardooli, Y. Dong, C. Georgakis, Data-driven process modeling and optimization aided by material and energy balances: The case of a batch polymerization process, IFAC-PapersOnLine 54 (3) (2021) 1–6, 16th IFAC Symposium on Advanced Control of Chemical Processes ADCHEM 2021. doi:https://doi.org/10.1016/j.ifacol.2021.08.209.

[20] Z. J. Liew, Z. Elkhaiary, A. A. Lapkin, Parameter eficient multi-model vision assistant for polymer solvation behaviour inference, npj Computational Materials 11 (1) (2025) 161.

[21] G. Mogk, T. Mrziglod, A. Schuppert, Application of hybrid models in chemical industry, in: J. Grievink, J. van Schijndel (Eds.), European Symposium on Computer Aided Process Engineering 12, Vol. 10 of Computer Aided Chemical Engineering, Elsevier, 2002, pp. 931–936. doi:https: //doi.org/10.1016/S1570-7946(02)80183-3.

[22] O. Kahrs, W. Marquardt, Incremental identification of hybrid process models, Computers & Chem ical Engineering 32 (4) (2008) 694–705, festschrift devoted to Rex Reklaitis on his 65th Birthday. doi:https://doi.org/10.1016/j.compchemeng.2007.02.014.

[23] D. C. Psichogios, L. H. Ungar, A hybrid neural network-first principles approach to process mod eling, AIChE Journal 38 (10) (1992) 1499–1511. arXiv:https://aiche.onlinelibrary.wiley. com/doi/pdf/10.1002/aic.690381003, doi:https://doi.org/10.1002/aic.690381003.

[24] K. Merkelbach, A. M. Schweidtmann, Y. Müller, P. Schwoebel, A. Mhamdi, A. Mitsos, A. Schuppert, T. Mrziglod, S. Schneckener, Hybridml: Open source platform for hybrid modeling, Computers & Chemical Engineering 160 (2022) 107736. doi:https://doi.org/10.1016/j.compchemeng.2022. 107736.

[25] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nature Reviews Physics 3 (6) (2021) 422–440. doi:https://doi.org/10.1038/ s42254-021-00314-5.

[26] D. J. Audus, J. J. de Pablo, Polymer informatics: opportunities and challenges, ACS macro letters 6 (10) (2017) 1078–1082. doi:10.1021/acsmacrolett.7b00228.

[27] D. Lin, H.-Y. Yu, Deep learning and inverse discovery of polymer self-consistent field theory inspired by physics-informed neural networks, Physical Review E 106 (1) (2022) 014503. doi:https://doi. org/10.1103/PhysRevE.106.014503.

[28] J. G. Ethier, D. J. Audus, D. C. Ryan, R. A. Vaia, Integrating theory with machine learning for predicting polymer solution phase behavior, Giant 15 (2023) 100171. doi:https://doi.org/10. 1016/j.giant.2023.100171.

[29] Y. Li, Y. Lin, Z. Mai, Q. Deng, Z. Ju, B. Cao, W. Li, Theory-guided machine learning for strength prediction of polymers at diferent temperatures and strain rates with physical information, Materials Today Communications 48 (2025) 113519. doi:https://doi.org/10.1016/j.mtcomm.2025. 113519.

[30] M. Velioglu, S. Zhai, S. Rupprecht, A. Mitsos, A. Jupke, M. Dahmen, Physics-informed neural networks for dynamic process operations with limited physical knowledge and data, Computers & Chemical Engineering 192 (2025) 108899. doi:https://doi.org/10.1016/j.compchemeng.2024. 108899.

[31] D. J. Audus, A. McDannald, B. DeCost, Leveraging theory for enhanced machine learning, ACS Macro Letters 11 (9) (2022) 1117–1122, pMID: 36018715. doi:10.1021/acsmacrolett.2c00369.

[32] N. Zobeiry, A. Poursartip, Theory-guided machine learning for process simulation of advanced composites, arXiv preprint arXiv:2103.16010 (2021). doi:https://doi.org/10.48550/arXiv.2103. 16010.

[33] R. T. Chen, Y. Rubanova, J. Bettencourt, D. K. Duvenaud, Neural ordinary diferential equations, Advances in neural information processing systems 31 (2018).

[34] S. Massaroli, M. Poli, J. Park, A. Yamashita, H. Asama, Dissecting neural odes, in: H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, H. Lin (Eds.), Advances in Neural Information Processing Systems, Vol. 33, Curran Associates, Inc., 2020, pp. 3952–3963.

[35] F. Sorourifar, Y. Peng, I. Castillo, L. Bui, J. Venegas, J. A. Paulson, Physics-enhanced neural ordinary diferential equations: Application to industrial chemical reaction systems, Industrial & Engineering Chemistry Research 62 (38) (2023) 15563–15577.

[36] H. Yan, J. Du, V. Y. F. Tan, J. Feng, On robustness of neural ordinary diferential equations (2022). arXiv:1910.05513. URL https://arxiv.org/abs/1910.05513

[37] G. T. Russell, The kinetics of free-radical polymerization: Fundamental aspects, Australian Journal of Chemistry 55 (7) (2002) 399–414. arXiv:https://connectsci.au/ch/article-pdf/55/7/399/ 104682/ch02114.pdf, doi:10.1071/CH02114.

[38] O. Walz, C. Marks, J. Viell, A. Mitsos, Systematic approach for modeling reaction networks involving equilibrium and kinetically-limited reaction steps, Computers & Chemical Engineering 98 (2017) 143–153. doi:https://doi.org/10.1016/j.compchemeng.2016.12.014.

[39] K. Matyjaszewski, T. P. Davis, et al., Handbook of radical polymerization, Vol. 922, Wiley Online Library, 2002.

[40] R. Guerrero-Santos, E. Saldívar-Guerra, I. Zapata-González, J. Bonilla-Cruz, E. Vivaldo-Lima, Free-Radical Polymerization, John Wiley & Sons, Ltd, 2025, Ch. 3, pp. 65–95. arXiv: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119820123.ch3, doi:https://doi. org/10.1002/9781119820123.ch3.

[41] F. Ehlers, J. Barth, P. Vana, Kinetics and thermodynamics of radical polymerization, in: Fundamentals of Controlled/Living Radical Polymerization, The Royal Society of Chemistry, 2013, pp. 1–59. doi:10.1039/9781849737425-00001.

[42] G. Moad, D. H. Solomon, The chemistry of radical polymerization, Elsevier, 2005.

[43] A. Butté, G. Storti, M. Morbidelli, Evaluation of the chain length distribution in free-radical poly merization, 1. bulk polymerization, Macromolecular theory and simulations 11 (1) (2002) 22–36.

[44] A. Butté, G. Storti, M. Morbidelli, Evaluation of the chain length distribution in free-radical polymerization, 2. emulsion polymerization, Macromolecular theory and simulations 11 (1) (2002) 37–52.

[45] B. Sanderse, P. Stinis, R. Maulik, S. E. Ahmed, Scientific machine learning for closure models in multiscale problems: a review (2024). arXiv:2403.02913. URL https://arxiv.org/abs/2403.02913

[46] P. E. Kloeden, M. Rasmussen, Nonautonomous dynamical systems, no. 176, American Mathematical Soc., 2011.

[47] X. Wang, E. K. Blum, Discrete-time versus continuous-time models of neural networks, Journal of Computer and System Sciences 45 (1) (1992) 1–19. doi:https://doi.org/10.1016/ 0022-0000(92)90038-K.

[48] S. Massaroli, M. Poli, J. Park, A. Yamashita, H. Asama, Dissecting neural odes, Advances in neural information processing systems 33 (2020) 3952–3963.

[49] J. D. Tan, B. Ramalingam, S. L. Wong, J. Cheng, Y.-F. Lim, V. Chellappan, S. A. Khan, J. Kumar, K. Hippalgaonkar, Machine learning predicts conversion and molecular weight distributions in computer controlled polymerization, Journal of Chemical Information and Modeling (2022)

[50] J. R. Richards, J. P. Congalidis, Measurement and control of polymerization reactors, Computers & Chemical Engineering 30 (10) (2006) 1447–1463, papers form Chemical Process Control VII. doi:https://doi.org/10.1016/j.compchemeng.2006.05.021.

[51] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, Pytorch: An imperative style, high-performance deep learning library, Advances in neural information processing systems 32 (2019).

[52] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. K. Duvenaud, Neural ordinary diferential equa tions, Advances in neural information processing systems 31 (2018).

[53] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, et al., Scipy 1.0: fundamental algorithms for scientific computing in python, Nature methods 17 (3) (2020) 261–272. doi:https://doi.org/10.1038/ s41592-019-0686-2.

[54] I. Loshchilov, F. Hutter, Decoupled weight decay regularization (2019). arXiv:1711.05101. URL https://arxiv.org/abs/1711.05101