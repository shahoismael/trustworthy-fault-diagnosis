Original Article

# Hybrid neural networks for improved chemical process modeling: Bridging data-driven insights with physical consistency

![](images/ddd646bb70db87c9e40f074c297bc6df45da010556fc887fc2c73a6e20d87cd3.jpg)

Jana Mousa , St´ephane Negny <sup>\*</sup> , Rachid Ouaret

Laboratoire de G´enie Chimique, Universit´e de Toulouse, CNRS, INPT, UPS, Toulouse, France

## A R T I C L E I N F O

Keywords: Neural network Nonlinear data reconciliation Karush-Kuhn-Tucker Unit operations

## A B S T R A C T

The increasing reliance on neural networks (NN) in chemical process modeling highlights their capability for accurate predictions, yet their standalone application often struggles to adhere to fundamental physical laws such as equilibrium constraints and mass balance. Addressing this limitation, hybrid methods that integrate data driven insights with physical consistency have gained prominence. This study systematically explores the inte gration of NNs with nonlinear data reconciliation (NDR) across multiple testing dimensions, including a Gibbs reactor, data robustness evaluations, and reactor-distillation system integration. Hybrid methodologies such as NN + NDR, NN + KKT (Karush-Kuhn-Tucker), and KKT + PINN (Physics-Informed Neural Networks with KKT conditions) are comparatively assessed. The proposed NN + NDR framework demonstrates superior performance in minimizing errors and enforcing physical laws, with minimal computational overhead. This work emphasizes the scalability, robustness, and transformative potential of modular hybrid strategies in advancing reliable, physically consistent chemical process modeling.

## 1. Introduction

The integration of artificial intelligence, particularly neural net works (NNs), has significantly advanced process systems engineering by enabling the modeling of complex, nonlinear phenomena such as reac tion kinetics, process optimization, and fault diagnosis (Cavalcanti et al., 2021; Kiˇs and Klauˇco, 2019). Neural networks offer flexibility and computational efficiency by learning directly from data, bypassing the need for detailed mechanistic models. However, despite their predictive strengths, standalone NNs often struggle to ensure physical fidelity. But some chemical process applications, where strict adherence to physical laws and principles (such as mass and energy conservation) is critical, unconstrained NNs may produce scientifically implausible results. Moreover, the black-box nature of NNs limits interpretability, compli cating their acceptance in safety-critical industries (Schweidtmann et al., 2024), while their heavy reliance on extensive, high-quality datasets poses additional challenges when data are scarce or noisy.

To address these limitations, recent research has focused on embedding physical constraints within neural network models. Incor porating conservation laws and relationships improves model general ization, enhances reliability, and reduces dependency on large datasets (Chen et al., 2024).

Hybrid modeling approaches, which blend data-driven learning with first-principles knowledge, have emerged as promising solutions to bridge the gap between empirical flexibility and physical rigor (Ghosh et al., 2019). Examples include physics-informed neural networks (PINNs), which incorporate governing differential equations into the training process (Cuomo et al., 2022), and Karush-Kuhn-Tucker (KKT)-based methods that impose equilibrium constraints during opti mization (Lˆe, 2020).

Despite these advances, current hybrid models often rely on soft constraint enforcement, because the way the constraints of physical laws are introduced allows deviations from a strict adherence to these con straints, that compromise model fidelity. The strict application of physical laws, also named hard constraints models, remains limited, especially in complex nonlinear systems, creating a significant gap be tween data-driven prediction and guaranteed physical validity. There is thus a critical need for hybrid frameworks capable of systematically embedding both soft and hard constraints without sacrificing compu tational efficiency or scalability.

Building on prior work that combined NNs with linear data recon ciliation (Mousa et al., 2025), this study proposes a novel hybrid framework by improving the previous one: the integration of neural networks with nonlinear data reconciliation (NN + NDR). Nonlinear data reconciliation (NDR) is a robust technique for adjusting raw process data to satisfy nonlinear physical constraints. In this approach, the outputs of a trained neural network are post-processed through NDR, ensuring strict compliance with physical laws without interfering with the NN’s training. This modular framework enhances predictive accu racy while enforcing physical consistency, addressing key limitations of existing hybrid models. This approach could be described as predictor-corrector, with the NN providing estimate of the output vari ables and NDR correcting them so that they comply strictly with the physical laws we aim to impose.

The proposed NN + NDR methodology is validated using a Gibbs reactor case study characterized by nonlinear equilibrium constraints. Its performance is benchmarked against two different hybrid approaches based on the improvement of established ones: (i) NN integrated with KKT conditions (NN + KKT) and (ii) KKT-constrained physics-informed neural networks (KKT + PINN). The comparative analysis evaluates each method’s ability to minimize predictive errors while enforcing physical constraints. Additionally, the robustness of NN + NDR is tested under noisy data conditions, data scarcity, and scalability challenges using a reactor–distillation integration system.

The results demonstrate that NN + NDR achieves superior enforce ment of physical laws with high predictive reliability, positioning it as a powerful tool for hybrid modeling in process systems engineering. The subsequent sections discuss related literature, detail the proposed hybrid methodologies, present case study results, and analyze the implications for future applications and research avenues.

## 2. Literature review and context of hybrid modeling

In process system engineering, hybrid models refer to a wide variety of models. In the context of this research, they refer to the coupling between data driven models and first principles models. Even within this category of models, there are several possible hybridizations, among which we will focus on models which integrate data driven machine learning with physic constraints.

These hybrid modeling frameworks have emerged as powerful tools to overcome the limitations of standalone neural networks in complex process systems. Traditional NNs excel in learning nonlinear mappings from data but often fail to guarantee adherence to fundamental physical principles, such as mass and energy conservation (Giovannelli et al., 2023). This shortcoming limits their reliability in engineering applications, particularly under noisy, sparse, or extrapolative conditions (Nakamura-Zimmerer et al., 2022; Qiu et al., 2023).

Meanwhile, first-principles models, grounded in mechanistic laws, offer high accuracy and interpretability but often require extensive domain knowledge and suffer from scalability challenges in high dimensional, nonlinear systems. Each approach, empirical and mecha nistic, carries distinct advantageous and limited characteristics, as

illustrated in Fig. 1.

Hybrid models aim to fuse the strengths of both paradigms: leveraging the flexible, pattern-recognizing capabilities of machine learning while embedding the robustness and interpretability of physicsbased modeling. This synergy enables hybrid frameworks to produce reliable, scalable, and physically consistent predictions across a wide array of real-world systems. Their growing importance has led to the emergence of frameworks such as Physics-Informed Neural Networks, which incorporate governing equations into the NN loss function (Cuomo et al., 2022; Raissi et al., 2019), and KKT-based hybrid models, which embed constraints through constrained optimization techniques (Chen et al., 2024; Femine, 2024). While these developments have advanced the field significantly, critical limitations remain.

In the case of PINNs, although they elegantly embed physical laws as soft penalties during training, they often struggle with ensuring strict constraint satisfaction, particularly in stiff or highly nonlinear systems (Escapil-Inchausp´e and Ruz, 2023). Their reliance on balancing data fidelity and physical consistency introduces trade-offs that require extensive hyperparameter tuning, making them sensitive to training settings (Krishnapriyan et al., 2021). Moreover, as dimensionality and problem complexity increase, PINNs face growing computational bur dens, limiting their scalability for large-scale industrial applications (Cai et al., 2021).

Similarly, KKT-based hybrid models aim to enforce hard physical constraints by formulating governing relationships as optimization constraints using the Karush-Kuhn-Tucker conditions. Although effec tive at constraint enforcement, these methods are inherently dependent on gradient-based solvers and are therefore prone to convergence issues in non-convex or highly complex optimization landscapes (Moghadda and Tohidi, 2020). Furthermore, embedding KKT conditions directly into the NN training process adds significant computational overhead, impacting scalability and hindering their applicability to real-time or large-scale systems (Femine, 2024). Accurate initialization and careful constraint design become critical to avoiding suboptimal solutions and ensuring robustness (Arvind et al., 2024).

Parallel research in process industries has emphasized the role of Data Reconciliation (DR) for ensuring data consistency with physical laws. Data reconciliation is a process that aims to correct measured or observed data by adjusting it slightly so that it remains consistent with each other, respects theoretical models or physical constraints, and minimizes errors by assuming that the data contains random measure ment noise. Nonlinear Data Reconciliation (NDR), in particular, ad dresses systems governed by nonlinear constraints by adjusting data to satisfy mass balances, energy balances, or equilibrium relations (Cencic, 2016; Kelly, 2004; Tian et al., 2006). Data reconciliation is often used a a standalone approach, very few studies seek to combine NN and NDR, which proves that this field of investigation remains to be explored and the novelty of the proposed approach.

![](images/3f1b15df9e1b7d1b2bc89bc35379aca0f521d1744fb0d2c42f4e46b858219920.jpg)  
Fig. 1. Venn Diagram of hybrid models.

However, most traditional DR techniques, including dynamic data reconciliation frameworks, focus primarily on correcting measured process data rather than refining model predictions. For instance, (Bai et al., 2005) introduced the use of auto-associative neural networks (AANNs) combined with dynamic data reconciliation to attenuate measurement noise and reconstruct missing sensor information. Importantly, they do not perform nonlinear data reconciliation, nor do they enforce nonlinear physical constraints post-prediction. While effective for measurement correction, this method was restricted to simple or semi-linear systems and did not tackle the correction of pre dictive outputs from standalone NNs or hybrid models. Moreover, the physical constraints were embedded into the training phase rather than explicitly enforced post-prediction, limiting the flexibility and adapt ability of the approach to unseen system conditions or model uncertainties.

Recent work by (Sharma et al., 2022) explored the use of data reconciliation frameworks to improve the operational reliability of hybrid renewable energy systems, emphasizing the need to correct on line measurement inconsistencies to maintain system stability under dynamic conditions. Their approach effectively improved data consis tency for sensor-based monitoring but remained limited to measurement corrections rather than post-prediction model refinement. Like (Bai et al., 2005), their reconciliation method is applied to measurements, not predictions, and there is no use of nonlinear optimization techniques to enforce hard physical constraints such as equilibrium relations. The framework is tightly integrated with physical system monitoring and does not generalize to post-prediction reconciliation of NN outputs. Importantly, their reconciliation strategies were inherently tied to the online operation of physical assets and were not generalized for sys tematic application on machine learning model outputs, thus restricting flexibility for broader hybrid modeling applications.

Critically, these and other existing hybrid frameworks do not apply nonlinear reconciliation directly to the outputs of data-driven models after training. Instead, most approaches either focus on smoothing sensor measurements or softening physical constraints within the learning objective itself, tolerating constraint violations rather than strictly enforcing them (Raissi et al., 2019).

In contrast, the hybrid framework proposed in this study distin guishes itself by modularly applying Nonlinear Data Reconciliation (NDR) directly on neural network outputs. This decoupled, rigorous strategy ensures strict constraint satisfaction after prediction, preserving model flexibility while achieving full physical consistency without network retraining or architectural modifications. While prior works have demonstrated the benefits of reconciliation techniques in process modeling, particularly in correcting measured or noisy inputs obtained with a real sensor, this study extends their role by applying NDR at the post-prediction stage of neural networks, considering the latter as soft sensors. This allows for systematic enforcement of hard, nonlinear, physical constraints while maintaining the adaptability of neural net works. By integrating NDR in a modular fashion across diverse valida tion settings, this approach contributes a scalable and generalizable methodology for physically consistent hybrid modeling in process sys tems engineering.

To clarify how the proposed NN + NDR framework distinguishe itself from previous works, we summarize below a comparative analysis of relevant studies that have attempted to combine neural networks with reconciliation strategies. As shown in Table 1, these existing methods either focus on correcting raw measurements (not model outputs), use linear or semi-linear reconciliation techniques, or embed physical con sistency within the training phase rather than enforcing it explicitly post-prediction. None of the referenced approaches employ nonlinear data reconciliation on NN outputs, which is the key innovation of the present study.

## 3. Proposed hybrid methodologies for constraint enforcement

In complex systems, achieving accurate predictions while adhering to physical constraints is a significant challenge. In this study, we pro pose different hybrid approaches firstly by combining neural networks with nonlinear data reconciliation, followed by integrating optimization techniques based on the Karush-Kuhn-Tucker conditions, and then by embedding KKT-based constraints within physics-informed neural net works. These methodologies aim to address the limitations of standalone data-driven models, ensuring predictions that are both accurate and physically consistent.

## 3.1. Neural network x nonlinear data reconciliation

Nonlinear Data Reconciliation (NDR) is a statistical optimization technique that adjusts measured process data to satisfy known nonlinear physical constraints (Kelly, 2004). While NDR has traditionally been used to refine measurement data, in the proposed framework it is innovatively applied after a neural network’s predictions, reconciling model outputs with physical laws without modifying the network ar chitecture or training phase.

This merge between NN and NDR is grounded in the sense that the neural network acts as a soft sensor, generating initial predictions that approximate output variables with an error. Nonlinear data reconcilia tion then refines these predictions. However, it’s important to monitor the computational time of the system to assess its feasibility.

This integration transforms the NN from a purely data-driven model into a physics-informed hybrid model. By reconciling the NN’s outputs, NN + NDR not only would aim to improve predictive accuracy but also to ensure that the predictions adhere to domain knowledge. It also ad dresses the main limitation of existing hybrid approaches, which typi cally embed physical knowledge into the loss function or training structure, often resulting in increased training complexity and constraint violations.

## 3.1.1. Methodology workflow

## Step 1. Neural Network Predictions

The process begins with the design and training of a neural network to predict key system variables based on a given data set. The NN ar chitecture is chosen to accurately capture the system’s nonlinear behavior, incorporating an appropriate number of layers, nodes, and activation functions. Additionally, dropout regularization layers are introduced to mitigate overfitting and improve generalization by preventing the network from relying too heavily on specific neurons and by targeting the uncertainty distribution. Once trained, the NN is used as a soft sensor to generate predictions for the system’s variables. To ac count for uncertainties in the NN’s predictions, multiple runs are per formed with varying initializations (for example different random seeds), producing an ensemble of predictions.

Comparison of hybrid methods combining neural networks and data reconciliation.

<table><tr><td>Study</td><td>Type of Reconciliation</td><td>Applied To</td><td>Constraint Handling</td><td>Model Interaction</td><td>Observed Points</td></tr><tr><td>Bai et al. (2005)</td><td>Dynamic Data Reconciliation</td><td>Sensor measurements</td><td>Implicit via AANN autoencoder</td><td>Embedded in AANN training</td><td>No nonlinear constraint enforcement; not applied post-prediction</td></tr><tr><td>Sharma et al. (2022)</td><td>Steady-state Data Reconciliation</td><td>Renewable system sensors</td><td>Implicit, linear constraints only</td><td>Real-time online measurement correction</td><td>Not used on ML predictions; limited to operational signal reliability</td></tr><tr><td>This Work (NN + NDR)</td><td>Nonlinear Data Reconciliation</td><td>Neural network predictions</td><td>Explicit hard constraint enforcement</td><td>Post-prediction (separate stage)</td><td>Generalizable; avoids retraining; supports physical feasibility</td></tr><tr><td>Most soft-constraint PINNs</td><td>No formal reconciliation</td><td>Loss function (training)</td><td>Soft constraint (penalty term)</td><td>Loss-term blending</td><td>Physical feasibility not guaranteed; sensitive to loss balancing</td></tr></table>

## Step 2. Data Segmentation and Statistical Analysis of NN Predictions

The ensemble of the network’s predictions is segmented into smaller subsets to ensure that the reconciliation process accounts for localized system behaviors and variations. Then, the mean vector $\left( X _ { a n } \right)$ and the covariance matrix (V) for each subset are computed (Eqs. (1) & (2)). These statistical metrics provide a robust representation of the predicted values and their associated uncertainties:

$$
X _ {a n} = \frac {1}{N} \sum_ {i = 1} ^ {N} x _ {i}\tag{1}
$$

$$
V = \frac {1}{N - 1} \sum_ {i = 1} ^ {N} \left(\boldsymbol {x} _ {i} - X _ {a n}\right) \left(\boldsymbol {x} _ {i} - X _ {a n}\right) ^ {T}\tag{2}
$$

Where N is the number of predictions, and x represents the pre dicted values from the i th initialization. These statistics provide the foundation for NDR, quantifying the variability and central tendency of the neural network’s predictions.

## Step 3. Formulation of the Objective Function

The core of the reconciliation process revolves around minimizing a constrained objective function $\left( \mathrm { E q . } \left( 3 \right) \right)$ that resembles the discrepancies between the reconciled variables x and the inaccurate predictions:

$$
y (x) = \frac {1}{2} (x - X _ {a n}) ^ {T} V ^ {- 1} (x - X _ {a n}) + p (x)\tag{3}
$$

$$
p (x) = \sum \log \left(1 + | x - X _ {a n} |\right)\tag{4}
$$

Where the first term represents the weighted squares error that measures and minimizes the differences between the reconciled values (x) and the NN-derived mean values $\left( X _ { a n } \right)$ . While the second term $\mathbf { \rho } ( \mathbf { p } ( \mathbf { x } ) )$ is a penalty function that introduces a logarithmic barrier to constraint violations, ensuring a smoother and more stable optimization process. The logarithmic growth of this term allows the system to prioritize reducing large constraint violations while remaining less sensitive to minor errors, thereby preventing overcorrection and ensuring a balanced reconciliation that maintains physical consistency without unnecessary rigidity. This choice allows for smooth and gradual penal ization of constraints violation. Indeed, unlike other function such as quadratic penalties, the log function grows slowly for large deviations resulting in reduce major inconstancies without over penalizing toler able differences.

The variance-covariance matrix (V) represents the statistical un certainties of the measurements, with the variances on the diagonal (they indicate the reliability of the measurements: the lower the vari ance, the better the stability). The covariances (off the diagonal) capture the correlation between the measurement errors.

We have already discussed the relative importance of the V matrix in the case of a linear problem (Mousa et al., 2025), and we should do the same for the non-linear case. In the case of non-linear constraints, the V matrix plays a crucial role. Non-linear constraints imply complex shapes for the solution space, so the $\boldsymbol { \nabla } ^ { 1 }$ matrix distorts this search space by giving more or less weight to certain dimensions. In this way, the complex structure of the solution space amplifies the effects of incorrect weighting. Therefore, a poor estimate of V can bias the solution, cause numerical instabilities (poor conditioning of V) or lead to unrealistic reconciliation (especially if the constraints are highly non-linear). This poor estimation can produce reconciled data that is less reliable than the measurements, and mask errors in the models or constraints. A good reconciliation depends as much on correct constraints as on a realistic estimate of the uncertainties via V.

## Step 4. Definition of Nonlinear Constraints

The constraints (z(x)) are derived from the fundamental physical o operational laws governing the system, expressed as nonlinear equality or inequality functions. These constraints ensure that the reconciled values comply with system requirements, such as conservation princi ples, equilibrium relationships, or other governing dynamics. An example of such a constraint could be expressed as:

$$
z (x) = \prod_ {i} x _ {i} ^ {a _ {i}} - \prod_ {j} x _ {j} ^ {b _ {j}} = 0\tag{5}
$$

Here, $x _ { i }$ and $x _ { j }$ are the reconciled variables, while $a _ { i }$ and $b _ { j }$ are co efficients that define the relationship between the variables.

For example, this form can represent a generalized thermodynamic equilibrium constraint such as:

$$
K _ {e q} = \frac {\prod_ {i} c _ {i} ^ {\nu_ {i}}}{\prod_ {j} c _ {j} ^ {\nu_ {j}}}\tag{6}
$$

Finally, the optimization model is

$$
\begin{array}{l} \text { Min } \left(y (x) = \frac {1}{2} (x - X _ {a n}) ^ {T} V ^ {- 1} (x - X _ {a n}) + p (x)\right) \\ \text { s.t. } z (x) = 0 \end{array}\tag{7}
$$

The optimization problem, as formulated, amounts to finding the statistically closest estimate to the measured data while remaining compatible with the constraints. So, according to Eq. (7), if a measure ment is highly uncertain (large variance) it will count less in minimi zation. and if two measurements are correlated their deviation will be penalized in a combined way (adjustment of one affects the other). Also, it is important to note that the convexity of the constraints plays a sig nificant role in the resolution process (a non-convex problem may have several local minima). The quality of initial data is also crucial to pre vent the algorithm from converging to an unrealistic solution.

## Step 5. Optimization and Reconciliation

After computing the variables’ mean $\mathbf { ( X _ { a n } ) }$ and variance (V) values, then defining the physical constraints (z(x)) to be enforced, followed by introducing the objective function (y(x)) to be minimized, it’s time to start the reconciliation process. This step employs an advanced numer ical optimization algorithm, such as Sequential Least Squares Quadratic Programming (SLSQP), to solve the constrained optimization problem. The algorithm iteratively adjusts the reconciled variables (x) to mini mize the objective function while strictly satisfying the nonlinear con straints defined by z(x) which are incorporated directly into the optimization as explicit equality constraints. Within the reconciliation procedure, they are passed to the optimizer as constraint functions that must be satisfied at every iteration. These constraints are then evaluated and enforced through a dedicated constraint-handling mechanism of the solver, which actively projects intermediate solutions back into the feasible region defined by $\mathbf { z } ( \mathbf { x } ) = 0$ . This ensures final solutions remain physically admissible throughout the optimization. By implementing the constraints directly within the solver’s constraint interface, the method ensures that the reconciled outputs are both statistically consistent with the neural network’s predictions and strictly compliant with the phys ical laws governing the system.

While NN + NDR method (Fig. 2) stands as the main contribution of this study, the subsequent proposed methodologies will be employed to compare its performance and further benchmark its effectiveness against other constraint-enforcing techniques.

## 3.2. Neural network x Karush-Kuhn-Tucker

The Karush-Kuhn-Tucker conditions are a set of necessary conditions

![](images/bcad0c5c001b6f74c10f3a992a0c5af4439c3b3e4f686ae88921fe8276cb2977.jpg)  
Solving the optimization problem to obtain reconciled predictions

$$
y (x) = \frac {1}{2} (x - X _ {a n}) ^ {T} V ^ {- 1} (x - X _ {a n}) + p (x)
$$

$$
p (x) = \sum l o g (1 + | x - X _ {a n} |)
$$

$$
z (x) = \prod_ {i} x _ {i} ^ {a _ {i}} - \prod_ {j} x _ {j} ^ {b _ {j}} = 0
$$

Fig. 2. Flowsheet of NN + NDR.

for a solution to be optimal in a constrained optimization problem, particularly in cases where the objective function and the constraints are nonlinear. These conditions are fundamental in optimization theory, especially in the context of problems involving inequality and equality constraints (Kuhn and Tucker, 1951). Traditional KKT methods address constrained optimization problems by introducing Lagrange multipliers, which quantify the importance of satisfying constraints while opti mizing an objective function. This integration of theoretical rigor with neural network methodologies creates a powerful framework for tack ling complex, constraint-driven modeling challenges.

For this approach, a similar philosophy as that of (Chen et al., 2024) is adopted, where the neural network is trained to predict the output variables. But a new modification is implemented with an additional mechanism to integrate the constraints into the network’s predictions. Rather than explicitly formulating a Lagrangian function, a correction factor is introduced that dynamically adjusts the predicted outputs to ensure compliance with the required physical constraint. This correction mechanism mirrors the role of Lagrange multipliers in traditional KKT, acting as a dynamic adjustment to align the model’s outputs with imposed physical laws.

The methodology employs a projection-based correction mechanism to enforce the physical constraint, implemented as follows:

## 1. Prediction of Outputs:

The neural network is trained to predict output variables, denoted as ${ \widehat { \boldsymbol { y } } } ,$ representing the predicted values for each system variable. These initial predictions, however, may not inherently satisfy the desired physical constraint.

## 2. Computation of the Required Physical Constraint:

Based on the predicted outputs, the physical constraint that needs to be enforced is calculated. To keep consistency with the constraint taken as an example in NN + NDR, the equilibrium constant can be considered as the physical constraint in this section as well, and it is determined from the predicted variables as in:

$$
K _ {e q. p r e d i c t e d} = \frac {\prod_ {i \in N _ {n u m}} (\widehat {y} _ {i}) ^ {p _ {i}}}{\prod_ {j \in N _ {d e n}} (\widehat {y} _ {j}) ^ {q _ {j}}}\tag{8}
$$

Where:

$N _ { n u m }$ and $N _ { d e n }$ are the sets of indices corresponding to the numerator and denominator of the equilibrium equation.

• $p _ { i }$ and $q _ { j }$ are the respective exponents for each $\widehat { y } _ { i }$ and $\widehat { y } _ { j }$ in the equation.

## 3. Reference Value for the Physical Constraint:

The target or actual value of the physical constraint, such as the equilibrium constant $( K _ { e q , ~ a c t u a l } )$ , is obtained from reference data, experimental observations, or validated simulations.

## 4. Computation of the Scaling Factor:

A scaling factor, $\mathfrak { Q } ,$ is determined to adjust the predicted outputs such that the physical constraint is satisfied. This factor is calculated as:

$$
\alpha = \left(\frac {K _ {e q , a c t u a l}}{K _ {e q , p r e d i c t e d}}\right) ^ {\frac {1}{n}}\tag{9}
$$

Where n is the sum of the exponents in the constraint equation:

$$
n = \sum_ {i \in N _ {\text { num }}} p _ {i} + \sum_ {j \in N _ {\text { den }}} q _ {j}\tag{10}
$$

## 5. Applying the Correction:

The scaling factor is then applied to the predicted outputs to produce the final corrected outputs:

$$
y _ {\text { corrected }} = \alpha . \widehat {y}\tag{11}
$$

This correction process happens in the form of projecting the pre dicted output to the desired, physically consistent, output; and it is in tegrated directly inside the neural network architecture as a custom constraint correction layer. By functioning as a specific layer within the network, the correction mechanism adjusts the predictions after they are generated, ensuring they satisfy the equilibrium constant conditions. This setup (Fig. 3) allows seamless integration into the training process, making the correction step differentiable.

To conclude the NN + KKT method modifies the neural network’s training objective to incorporate the constraints directly into the opti mization process. A KKT-inspired constraint layer dynamically projects unconstrained NN outputs onto a feasible solution space satisfying the equilibrium relation, correcting deviations during training while pre serving flexibility in the learning process.

## 3.3. Karush-Kuhn-Tucker x physics-informed neural network

To further evaluate the coupling of KKT principles with neural net works against our proposed methodology of merging nonlinear data reconciliation with neural networks, the architecture of the neural network was modified to develop a physics-informed neural network. This variation, referred to as KKT + PINN, retains the core mechanism of KKT-style constraint enforcement while introducing a custom loss function that blends both data-driven and physics-based components. The goal of this approach is to combine the predictive generalization capability of PINNs with the rigorous constraint enforcement introduced in the NN + KKT model. While standard PINNs embed physical laws as soft penalties in the loss function, the KKT + PINN model seeks to strengthen this enforcement by adopting a structure inspired by con strained optimization principles.

This integration overthrows the standard PINNs that impose physical constraints as soft penalties in the loss function due to the fact that the change in the network is not only merely in the loss function, but also in having a KKT-inspired custom constraint layer to ensure strict constraint-compliance. Therefore, we are combining the flexibility of PINNs with the rigorous constraint enforcement of KKT. The KKT + PINN framework we implemented is structurally identical to the NN + KKT approach we described earlier in the manuscript. The key difference is simply in the type of neural network used:

• In NN + KKT, we use a standard neural network trained with MSE loss.

• In KKT + PINN, we switch to a PINN architecture, where the loss function is composed of two terms: a data-driven loss and a physicsbased constraint loss.

The loss function in KKT + PINN combines traditional data-driven loss terms with the necessary constraints imposed by the KKT condi tions, which helps the network learn both the empirical relationships from the data and the underlying physics governing the system. Spe cifically, the KKT + PINN loss function includes several key components:

## 1. Data Loss Term (Empirical Loss):

The first component of the loss function is the traditional empirical loss term, which measures how well the model fits the observed data. This is typically a mean squared error (MSE) between the predicted outputs and the actual observed data:

$$
L _ {d a t a} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\widehat {y _ {i}} - y _ {i}\right) ^ {2}\tag{12}
$$

Where:

• ŷ is the predicted output

• y<sub>i</sub> is the true observed value

• N is the number of data points

## 2. Physical Constraint Term (PINN Loss):

The second term incorporates the physical constraint directly into the loss function. For instance, in a system governed by an equilibrium condition, this term might be designed to penalize the network if its predictions deviate from the expected physical law. Using the equilib rium constant as an example, the constraint can be formulated as:

$$
L _ {c o n s t r a i n t} = \left| K _ {e q. p r e d i c t e d} - K _ {e q. a c t u a l} \right| ^ {2}\tag{13}
$$

Where:

$K _ { e q , p r e d i c t e d }$ is the equilibrium constant calculated from the network’s predicted outputs.

$K _ { e q , a c t u a l }$ is the reference or known equilibrium constant based on experimental or simulated data.

• This term ensures that the neural network predictions not only fit the data but also respect the physical law governing the system.

Finally, the total loss function to minimize can be written as:

$$
L _ {t o t a l} = L _ {d a t a} + \omega L _ {c o n s t r a i n t}\tag{14}
$$

With ω is a ponderation factor between the two terms of the loss function.

The subsequent section will focus on the application of these methods to a specific case study, providing a critical evaluation of their performances, specifically our proposed novel approach, against each

![](images/64308b271152081888b3f09d295445bb7c2a1157324da5d692463a667ccecb9e.jpg)  
Fig. 3. Integrating neural networks with KKT inspired constraints for optimized predictions.

other.

## 4. Case studies and results

To comprehensively evaluate the proposed hybrid method (NN × NDR) and the two other improved methods, a multi-stage validation strategy was developed. Two case studies will be presented to demon strate its capabilities at unit level and plant level. Rather than relying on a single case study, the method’s performance was assessed across:

• A baseline Gibbs reactor system under equilibrium constraints,

• Robustness tests under noisy data and data scarcity conditions,

• An extended application to an integrated reactor–distillation pro cess, testing scalability to multi-unit systems.

This layered validation approach rigorously tests the method’s ac curacy, constraint compliance, robustness, and adaptability across increasingly realistic process engineering challenges.

## 4.1. Case study I: gibbs reactor system benchmark

We examine a Gibbs reactor performing the incomplete hydrogena tion of toluene, producing methane, benzene, and biphenyl as reaction products, Fig. 4. Pure reactants are fed at a steady state in a fixed volume reactor. The variables retained for this study are the mass flow rates. The temperature and the pressure can also vary to create the data set, and are considered as input variables pour the NN. The system’s behavior is governed by the equilibrium constant, a fundamental physical constraint ensuring thermodynamic consistency.

$$
K _ {e q}. \left(x _ {2} ^ {2}. x _ {5} ^ {3}\right) = \left(x _ {3} ^ {3}. x _ {4}. x _ {6}\right)\tag{15}
$$

Where $x _ { 2 } , x _ { 3 } , x _ { 4 } , x _ { 5 } , x _ { 6 }$ are the reconciled concentrations of methane, benzene, biphenyl, toluene, and hydrogen, respectively.

The neural network architecture employed in this study is consistent throughout all the applications and the scope of this article and it con sists of four layers: an input layer, two hidden layers, and an output layer. Each hidden layer comprises 128 neurons with the rectified linear unit (ReLU) activation function to introduce non-linearity. Also, to ac count for uncertainty in the model’s predictions, dropout layers of 20 % probability were incorporated. By randomly deactivating neurons dur ing training, dropout prevents overfitting and improves generalization. This neural network was obtained after testing different network to pologies and activation functions, and the best configuration was selected. To maintain the same basis for comparison, NN, NN + NDR, NN + KKT and KKT + PINN use the same topology for each case study.

The input features include key reactor operating conditions, such as the feed flow rates of toluene and hydrogen, along with temperature and pressure. The network is trained to predict the reactor’s outlet flow rates, corresponding to methane, benzene, and biphenyl. A dataset of 2000 points corresponding to feasible simulations (data corresponding to unconverge or numerical issues were withdrawn), divided into 80 % for training and 20 % for testing, is obtained by simulations of the reactor via Fives ProSim software and is used to build and evaluate the model. The choice of a dataset consisting solely of simulation results is motivated by the desire to avoid the problem of data quality. Indeed, by removing potential biases due to the data set, we can solely concentrate on the method’s capabilities.

With prior knowledge that the neural network generates initial predictions of the outlet flow rates, based on the input condition, that are purely data-driven and do not inherently satisfy the equilibrium constraint, the following parts will discuss the employment of the aforementioned hybrid approaches to test their performance in the equilibrium constraint enforcement.

## 4.1.1. NN × NDR: baseline constraint enforcement

In the reactor context, this approach ensures that predictions align with thermodynamic laws, specifically the equilibrium constant (Keq). By embedding this constraint into the reconciliation process, the NN + NDR model refines the predictions to ensure they comply with the equilibrium law governing the chemical system. In addition to the constraints on the equilibrium constant, we need to ensure that the reconciled data remains compatible with the mass balances. Conse quently, the constraints associated with the mass balances are also included in the model. The test $\mathrm { R } ^ { 2 }$ (coefficient for determination) values were exceptionally high, ranging from 0.9994 to 0.9998 across all output variables. This indicates that the NN predictions explain virtually all the variance in the test set and confirms that the model performs very well in the interpolation regime of the simulator-generated data.

However, it is important to note that high R² alone does not imply that physical constraints are satisfied. In fact, even with near-perfect R², the raw NN predictions violate the equilibrium constraint significantly.

Fig. 4 compares the prediction errors for Keq between the standalone neural network and the NN + NDR hybrid framework. The error asso ciated with the standalone NN is both substantial and oscillatory, fluc tuating around the true equilibrium constant. These fluctuations directly highlight the NN’s inability to inherently satisfy thermodynamic con straints, as its training objective focuses solely on minimizing statistical error without regard for physical laws. Without explicit consideration of physical constraints, the NN is free to generate predictions that may statistically minimize error but deviate from thermodynamic consis tency. Quantitatively, the standalone NN achieves a mean absolute error (MAE) of 258,972 with a large standard deviation of 387,542, con firming substantial equilibrium violations.

In contrast, the NN + NDR model demonstrates markedly improved performance. By applying nonlinear reconciliation after prediction, NN + NDR systematically corrects outputs to satisfy the thermodynamic law, achieving a near-zero MAE and fully eliminating equilibrium vio lations, as shown in Fig. 5, effectively bridging the gap between datadriven accuracy and physical validity. This post-processing correction transforms purely data-driven NN outputs into physically meaningful results without requiring changes to the neural network training process itself.

To address concerns regarding potential data overlap or leakage that might artificially reduce prediction error or explain constraint satisfac tion, we analyzed the statistical distributions of the training and test sets. The histograms and kernel density plots for all input features are shown in Fig. 6. These plots demonstrate that while both datasets originate from the same simulation environment, the test set is

![](images/4b58438d2b370097c81b182a0381edf8facf42f545a0573599232238d3504bb0.jpg)  
Fig. 4. Reactor description

![](images/5778e62d6a1cc831e163f2a934c6a1f490fde136cfea805a3fe5f91ee9f42a6f.jpg)  
Fig. 5. Scatter plot of the prediction errors of Keq between NN and NN + NDR against true desired values.

![](images/3ce516a1a7bc8a6e32f46dbc0d57b1b5e2cef0d420569e2662c0a4050e2a0dc8.jpg)

![](images/c7bebc214811f681d616c2ab2845da1714695381ec31548946fc05f40e770581.jpg)

![](images/ac164e6f6fd4aa59745a527f00fc5ab8a02e48925927147038f2b554e48bb77a.jpg)

![](images/f8959d525858c7d2f05fd2cecdacb06a91422c221c855e837077dfdbd629dda1.jpg)  
Fig. 6. Histogram and kernel density comparison of training and test sets for all input features.

composed of statistically distinct samples that span the same input domain without duplication. Therefore, the observed results, particu larly the zero-constraint error achieved through NN + NDR, are not a product of memorization, but rather a consequence of effective recon ciliation applied to valid generalization data.

The superiority of the NN + NDR framework lies in its dual opti mization strategy, balancing statistical accuracy with rigorous enforce ment of physical constraints. Additionally, the approach offers outstanding computational efficiency: ten independent NN initializations and reconciliations require approximately 10 min total, with the reconciliation step itself being computationally negligible. It is important to explain the choice of 10 independent initializations and reconciliations. The purpose of this step is to calculate the V matrix, the importance of which was discussed in Section 3.1. A sensitivity analysis was carried out on the number of initializations required to evaluate V with sufficient accuracy for the optimization. After testing 5, 10, 15, 20 and 25 independent initializations, it was found that with 10 initiali zations the matrix V had been evaluated with very good accuracy, and that beyond that the calculation time increased without any impact on the quality of the solutions.

Thus, this first benchmark confirms that NN + NDR not only refines model predictions but also fully restores thermodynamic consistency, providing a significant advantage over unconstrained NNs. An impor tant clarification is warranted regarding the "zero error" reported for the NN + NDR method. This refers specifically to the constraint residuals, such as the deviation from thermodynamic equilibrium, not to the neural network’s raw prediction error. The NN, trained without physical constraints, does produce significant deviations from physical laws (as shown in Fig. 5). These are corrected post-prediction by the NDR step, which solves a nonlinear constrained optimization problem to enforce physical feasibility. The optimization (using SLSQP) consistently ach ieves constraint satisfaction within machine-level precision (≤ 1e− 10), justifying the use of the term “zero error” in this context. This process does not imply perfect predictions, but rather perfect enforcement of the physical constraint. Therefore, comparing this method to alternative frameworks can further validate its efficacy and demonstrate its appli cability to a broader range of complex systems.

## 4.1.2. NN + KKT benchmark

As the second validation stage, the performance of the NN + NDR framework was benchmarked against another prominent hybrid meth odology: the NN + KKT approach.

The effectiveness of NN + KKT was evaluated by comparing equi librium constraint adherence against the unconstrained neural network, as illustrated in Fig. 7.

In contrast, NN + KKT significantly reduced prediction errors, achieving a MAE of 8876 and a standard deviation of 18,264, repre senting a 96 % reduction in mean absolute error. This confirms that the constraint correction mechanism meaningfully constrains the network’ outputs, ensuring predictions stay within a thermodynamically consis tent range, albeit with minor residual violations. However, despite these improvements, NN + KKT is not entirely free from errors. Residual de viations persist due to the optimization trade-offs inherent in balancing data-driven loss minimization with strict constraint enforcement. Moreover, because constraint satisfaction is integrated into the training process, convergence difficulties can still lead to small inaccuracies, particularly in complex or highly nonlinear systems. Thus, although NN + KKT enforces constraints more strictly than soft-penalty methods, it still does not guarantee zero-constraint error.

Importantly, computational performance was also evaluated. The NN + KKT approach introduced moderate computational overhead compared to an unconstrained NN: training and correction required approximately 5 min, compared to 1 min for standalone NN and about 10 min for NN + NDR. While relatively efficient, NN + KKT’s runtime can increase further in multi-reaction systems or under more complex constraints.

By comparison, NN + NDR fully eliminates equilibrium violations by decoupling prediction from constraint enforcement. The two-step workflow; first predicting, then reconciling, achieves strict physical adherence without embedding constraint balancing into the training optimization, making it more robust and generalizable across data regimes.

Thus, while NN + KKT offers a computationally attractive improvement over unconstrained networks, it falls short of the full physical fidelity achieved by NN + NDR. The choice between these methods ultimately depends on application priorities: faster training with approximate constraint adherence (NN + KKT) versus slightly higher computational cost with complete thermodynamic consistency (NN + NDR).

To further enhance constraint enforcement while maintaining pre dictive flexibility, the next section extends the benchmarking to a hybrid Physics-Informed Neural Network combined with KKT corrections (KKT + PINN framework).

## 4.2. KKT + PINN benchmark

As the third validation benchmark, we extend the hybrid modeling evaluation by incorporating Physics-Informed Neural Networks into the KKT framework, forming the KKT + PINN hybrid model. This approach aims to embed physical constraints more intrinsically into the network’s training process. In the KKT + PINN framework, the neural network architecture is adapted into a PINN structure, with a customized loss function composed of a data driven term, and a residual term. Addi tionally, a KKT-inspired constraint projection layer ensures that physical laws are respected during network updates, reinforcing constraint adherence throughout training.

![](images/a42535eac2f9b6b6ff2617c485d97fe49b4c52cde60c2de120c8781358dac55b.jpg)  
Fig. 7. Boxplot of sample-wise absolute errors in Keq predictions across methods

The results, presented in Fig. 8, demonstrate that the KKT + PINN method achieves a significant improvement in enforcing Keq over both the unconstrained NN and the NN + KKT framework. Specifically, KKT + PINN achieves a mean absolute error (MAE) of 5765 between pre dicted Keq and true Keq (with a small standard deviation 7986.4 compared to the two previous methods), compared to 8876 for NN + KKT, confirming a notable enhancement in constraint enforcement while maintaining predictive accuracy.

This performance improvement stems from the dual-layer constraint strategy: KKT-based correction during optimization combined with continuous penalty enforcement via the residual term in the loss func tion. Consequently, predictions remain closer to physically valid solu tions throughout training.

However, despite its advantages, the KKT + PINN approach does not surpass the performance of NN + NDR. While KKT + PINN significantly improves constraint adherence, it is fundamentally limited by the tradeoff inherent in loss function design: the model must simultaneously minimize prediction error and constraint violation, leading to small residual errors that cannot be entirely eliminated.

Although the equilibrium constant is calculated precisely, it is stil essential to examine the value of the flow rates for each compound. As data reconciliation uses degrees of freedom to adjust the value of vari ables, it is possible that the enforcement of the equilibrium constant is obtained to the detriment of consistent and realistic values for flow rates. However, the addition of material balances in the equations of the model on which the reconciliation is based is also intended to limit this disadvantage. Fig. 7 shows a comparison of the biphenyl flow rates obtained during the simulation (true value) with the flow rates estimated by the different methods previously explained in part 3. Biphenyl is selected as the representative compound because all the other species show equivalent predictive behaviors. In order to keep the figure legible, 10 representative points of the flow variation interval for this component were selected during the test phase. The NN predictions of biphenyl flowrate display noticeable fluctuations around the true value because the constraints are not imposed in this purely data driven technique, reflecting its inconstancies in relation to our objective. These discrepancies also explain the differences in the prediction of the equi librium constant. Conversely, the predictions of the NN + NDR method provide points that are confused with the true values for all the obser vation points. The hybrid approach not only enforces strict thermody namic equilibrium but also preserves structural realism compounds output flowrates, ensuring that constraint satisfaction does not compromise predictive integrity. NN + KKT predictions closely follow the same patter, reflecting that the constrains are enforced during training and testing without interfering with the core prediction mech anism, hence preserving accuracy. The method gives good results because the NN predictions are very satisfactory, which makes the projection/correction mechanism all the more effective.

However, the KKT + PINN method exhibits the greatest deviation, with predictions furthest from the true value. Although it improves equilibrium constraint satisfaction (as shown in Fig. 8), embedding the constraint directly in the loss function introduces a trade-off: the network must balance fitting the data with satisfying the constraint. This often leads to reduced predictive accuracy, particularly for local flow rate variations. This explains why Fig. 8 ranks KKT + PINN higher on constraint satisfaction, while Fig. 9 ranks it lower on predictive accu racy. The methods are being evaluated on two distinct criteria, and the observed differences are a result of this trade-off.

The computational performance of the hybrid methods was also compared:

Comparison of Mean Absolute Errors Across Methods  
![](images/c8f445ba33454bdd62427655d8bae12802f01bb191b520531991d7b1b6070694.jpg)  
Fig. 8. Mean absolute error (MAE) and standard deviation of Keq predictions across methods.

![](images/82b7a28d9ec0ab742dfaa791aeaf25a8cdb35c8f08ad43092e3986d85909a0d6.jpg)  
Fig. 9. Comparison of Biphenyl flowrate (kg/h) predictions.

• NN alone required \~1 min, with no constraint enforcement.

• NN + KKT reduced errors significantly with a runtime of \~5 min.

• KKT + PINN introduced additional computational complexity, requiring \~10 min, due to the physics-informed loss optimization and KKT corrections.

• NN + NDR, despite 10 independent neural network initializations (\~1 min each), maintained a total computational burden of \~10 min, while achieving complete constraint satisfaction.

While KKT + PINN provides a valuable intermediate solution, of fering improved constraint adherence with moderate computational cost, NN + NDR remains the most robust and physically consistent hybrid approach among all methods evaluated.

These two methods, NN + KKT and NN + NDR, collectively validate the effectiveness and superiority of the proposed reconciliation-based framework, beyond a single system or training strategy. Whole KKT + PINN improves nonlinear constraints enforcement, it can fail to capture variables not included in the constraints.

## 5. Extended validation: robustness to noise and data scarcity conditions

In addition to comparing multiple hybrid frameworks under normal operating conditions, this study further expands the evaluation by testing the proposed NN + NDR approach across varied real-world challenges: (i) noisy measurements, (ii) severe data reduction, and (iii) limited input parameters. This multi-scenario validation ensures the generalizability and robustness of the method beyond a single case study setting.

## 5.1. Evaluating model performance with noise introduction

Real-world chemical process data are frequently affected by sensor inaccuracies and measurement noise. To assess the robustness of NN + NDR under such conditions, Gaussian noise was injected into the data set. The previous data were standardized prior to noise addition to ensure that variability was introduced in a realistic yet controlled manner.

The results from this testing demonstrated that, despite the presence of noise in the dataset introduced to the neural network, the integration of the network with nonlinear data reconciliation maintained high ac curacy in adhering to the physical constraints, effectively minimizing deviation from the true equilibrium values.

As shown in Fig. 10, the violin plot provides a visual comparison between the distribution of errors in the normal and noisy datasets, illustrating the spread and density of the predictions in each case. The similar shape and range of both distributions indicate that the model’s performance remained stable even under noisy conditions, with no significant deviation in the reconciled predictions. This confirms that the NN + NDR framework is highly resilient to noise and maintains strict adherence to physical laws even under degraded measurement quality.

To analyze the results further in depth, and for the same reasons as for Figs. 9 and 11 depicts the biphenyl flowrate under noisy conditions. We can draw the same conclusion as in Fig. 11 concerning the com parison of both methods. Furthermore, the noisy data does not change the quality of the prediction for the NN + NDR method, which dem onstrates its robustness. This feature is critical and valuable for deployment in industrial environments, where sensor noise is inevitable and reliability under imperfect data conditions is essential.

## 5.2. Evaluating model performance with sparse data inputs

Beyond noise robustness, we further tested the methodology under severe data scarcity scenarios. Training data were deliberately reduced by up to 90 %, simulating real-world cases where collecting extensive datasets may be infeasible due to operational constraints or cost. Indeed, using less data in machine learning is also a challenge as they are very expensive to produce and maintain, both on their economic and envi ronmental impacts. This reduction allowed us to assess the method’s ability to maintain reliability and stability in enforcing physical con straints, such as adherence to equilibrium constant conditions, under data-sparse conditions. To ensure consistency, biphenyl was selected again as a representative chemical compound, as its flowrate exhibited trends similar to other components, allowing focused yet meaningful analysis.

Despite the substantial reduction in available data, the proposed NN + NDR methodology demonstrated remarkable robustness, retaining its effectiveness in upholding the physical constraints. The results indicated that even with a 90 % decrease in input data, the method was able to maintain accurate constraint enforcement, reinforcing its adaptability and suitability for applications where data scarcity is a limiting factor. Notably, this evaluation yielded patterns similar to those observed in the noisy data testing in Fig. 8, with the model’s ability to maintain equi librium conditions remaining consistent.

![](images/389dbe90be8c1262216e9c2ae2e5a48a52141d6fc8f042611ebeff87fc45e8ef.jpg)  
Fig. 10. A violin plot that visualizes the similarity of distribution of the output between normal and noisy data.

![](images/2f8fbad6ee5b80c2bf4d5cd49c372dcda2af90dbedf3445f716b86f38b91198f.jpg)  
Fig. 11. Comparison of Biphenyl flowrate (kg/h) predictions under noisy data.

However, the flowrate predictions for the chemical components made by the neural network were understandably impacted by the data reduction. This variability, driven by reduced information, is particu larly important for users operating in sensitive systems where prediction fidelity matters.

The NDR framework must adjust the flowrates of components to satisfy the equilibrium constant constraint, even as input data becomes increasingly sparse. This dual effect was assessed in Fig. 12 by exam ining both the predictive accuracy of the NN and the adaptability of the NDR in reconciling the flowrates.

At moderate reduction levels (10 %–30 %), the NN alone appears to show slightly better alignment with the true values. This can be attrib uted to two factors: a possible reduction in overfitting due to less noisy data, and the fact that NN predictions are not corrected or constrained, allowing them to stay numerically close to the target in this specific range.

However, as reduction reaches and exceeds 50 %, NN predictions begin to degrade significantly, showing erratic deviations. This poor generalization leads to increasingly inaccurate outputs and makes the NDR’s reconciliation step more challenging. Since NDR relies on the NN output as an initial guess, it adjusts the predictions to satisfy the physical constraint, even if doing so causes further deviation from the true value. This behavior is especially noticeable at 50 % reduction, where the NN initialization is furthest from feasible space.

It’s important to clarify that NDR’s objective is not to improve the raw prediction accuracy, but to enforce strict thermodynamic consis tency. This is why, even when NN + NDR predictions visually appear less accurate in some cases, they are physically valid, unlike the un constrained NN predictions. In the NN + NDR approach, physical con sistency is treated as a global system-level constraint, not a pointwise property of individual flowrates. The equilibrium constant $\mathrm { K _ { e q } }$ is gov erned by a nonlinear function of all five species $\left( \operatorname { E q . } \right.$ (15)). To satisfy this equation, the NDR block reconciles the outputs of the NN by adjusting all flowrates jointly (degree of freedom for the reconciliation). This reconciliation ensures that the entire set of flowrates lies on the physi cally valid constraint manifold, even if that leads to deviations in indi vidual flowrates (such as biphenyl). In contrast, the pure NN model minimizes pointwise prediction error and does not “see” or enforce the global equilibrium constraint. Therefore, it may produce individual values that are numerically closer to the true value, but this proximity does not guarantee that the entire set of predicted flowrates satisfies the physical relationship defined by $\mathrm { K _ { e q } } .$ To avoid the differences on the flowrates of the different compounds a constraint must be added both on the equilibrium constraint and on the flowrates (reduced degree of freedom for the data reconciliation).

Moreover, as data becomes increasingly limited, the reconciliation requires more iterations to converge due to less reliable NN outputs. Despite this, the optimization always converged and maintained constraint adherence.

## 5.3. Evaluating model performance under input variable reduction and data scarcity

Recognizing that feature quality is as critical as quantity, we further tested how removing key input variables affects the hybrid framework. Starting with the removal of pressure as an input feature of the NN, followed by incremental dataset reduction, we monitored these com bined effects on biphenyl flowrate predictions (Fig. 13).

Compared to pure data reduction alone, removing less informative inputs first led to noticeably smoother predictions. Even at the initial point without data reduction, we can see that the prediction with NN + NDR is no longer exactly the same as the true value, as was the case previously (Fig. 12). This suggests that careful feature selection can mitigate overfitting, forcing the NN to generalize better on fewer inputs. Nevertheless, as both features and samples were progressively removed, the predictive accuracy inevitably deteriorated, though NDR continued to enforce equilibrium constraints successfully. It is important to note that while NN predictions may appear closer to the true flowrate in some cases, NDR ensures that the predictions remain physically consistent with the equilibrium constraint, even if this results in some deviation from the observed value. The NDR step is not designed to minimize flowrate prediction error directly. Its objective, as previously mentioned, is to reconcile the NN’s output so that it strictly satisfies the thermodynamic equilibrium constraint, in this case, based on Keq. As a result, it may adjust the NN’s predictions away from the exact true value in order to enforce physical feasibility.

![](images/a03da4f0a222fd10aa2a703fd7fc38cb4db2f27cbfb8d4f48c4146541dbf923d.jpg)  
Fig. 12. Variation in NN and NN+NDR performances under reduced data (Biphenyl flowrate in kg/h).

![](images/f7052d048964f9459b638a28aa55086c61d7a1632caa2335b6edb2056381fe79.jpg)  
Fig. 13. Combined effect of parameter reduction and data scarcity on flowrate (kg/h) prediction.

NDR followed a trend similar to that of the NN predictions. As its primary objective is to reconcile the flowrates in a manner that adheres to the equilibrium constant constraint, its behavior is inherently tied to the NN predictions, which serve as its input. Thus, the reconciliation process mirrored the NN’s trend, showing improved stability during moderate reductions but eventually diverging at higher levels of data scarcity. Therefore, even in extremely sparse and reduced-feature sce narios, NDR effectively acted as a corrective safety net, ensuring physically meaningful outputs despite predictive uncertainty.

We computed and compared the absolute error and RMSE of Keq between the NN and NN + NDR models across various data reduction levels. The results show that: (i) the NN-only model exhibits a significant $\mathrm { K _ { e q } }$ error under data reduction conditions, indicating growing deviation from physical equilibrium, (ii) the NN+NDR model maintains a nearzero Keq error, confirming that it preserves thermodynamic consis tency even under severe data reduction.

In summary, these extensive robustness evaluations; across noise perturbations, data scarcity, and input feature reduction, demonstrate that the NN + NDR framework is not limited to a single case study but represents a systematically validated, practical, and scalable solution. The hybrid model’s ability to uphold physical consistency indepen dently of raw prediction strength marks a significant advantage over conventional neural network approaches. Such comprehensive analysis positions the proposed methodology as a strong candidate for real world, industrial deployment across diverse process systems facing imperfect data conditions.

![](images/507ad0a4b4a7c0881980e45aeef7732417c78119e02b67822416e95aa275f377.jpg)  
Fig. 14. Process flowsheet.

## 6. Case study II: integrated multi-unit system: reactor–distillation modeling

To further validate the robustness, scalability, and broad applica bility of the proposed methodology beyond isolated reactor testing, a more complex integrated process comprising a Gibbs reactor, a sepa rator, a heat exchanger, and a distillation column was analyzed (Fig. 14). This step represents an additional, distinct case study, demonstrating that the work extends well beyond a single model or unit. In this process, the feed stream consists primarily of toluene and hydrogen, which react in the Gibbs reactor to form methane, benzene, and biphenyl. The component splitter removes the lighter species (hydrogen, methane, and benzene), leaving only toluene and biphenyl to enter the distillation column. The cooler/heater adjusts the temper ature of this stream prior to separation. In the distillation column, toluene and biphenyl are partially separated based on their volatilities, with each component distributed between the top and bottom product streams according to operating conditions.

For clarity in method evaluation, the focus was placed on the direct interaction between the reactor and the distillation column, since the intermediate units mainly serve technical purposes and do not introduce additional constraint complexities. By selecting this simplified yet industrially relevant sub-process, we ensured a stringent and practical validation of constraint enforcement across interconnected system components.

The first stage involved applying NDR to the Gibbs reactor’s outlet flowrates predicted by the NN, similarly to the earlier reactor case (over 1210 independent data points), ensuring that the equilibrium constant constraint was enforced post-prediction. By reconciling the output before passing them downstream, the data fed into the distillation model was guaranteed to be thermodynamically consistent, as shown in Fig. 15 where NN + NDR achieves zero residual within solver tolerance (≤ 1e− 10), shown as 0.00. For clarity, the optimization is a deterministic nonlinear reconciliation problem, and in this case, it is solvable to machine-level precision (typically below 1e-10). Therefore, the constraint error is effectively zero and rounded as such in the figure. This outcome is consistent with the small-scale, algebraic nature of the constraint and the solver’s ability to converge tightly.

Initially, the reconciled reactor outputs were directly passed to a pretrained neural network for the distillation column. However, this led to poor predictive performance, as the neural network, trained on a sepa rate dataset, struggled to extrapolate to the new operating regime introduced by the reconciled reactor conditions. This highlighted the key limitation of neural networks that is their inability to generalize effectively beyond their training domain, particularly in small datasets. To overcome this and ensure consistency in both data and prediction quality, ProSim was used to simulate new, aligned output data for the distillation column based on the reconciled reactor conditions. This simulation-based retraining ensured that the distillation column’s NN operated within a domain it could reliably model. Following this, the validated, reconciled reactor outputs were introduced as inputs to a second NN trained to model the behavior of the distillation column. The inputs to the column’s NN included reconciled reactor outputs, reflux ratio, distillate flowrate, and column efficiency, while the outputs were the flowrates of toluene and biphenyl at the top and bottom sections of the column.

However, a critical challenge emerged: due to the prior reconcilia tion step and resulting data availability constraints, only 242 data points were available for the distillation column model, with just 49 points used for testing. This substantial data reduction led to a noticeable deterioration in the raw NN predictive performance, manifesting as scattered and inconsistent flowrate predictions for the column outputs.

![](images/ea41d4c0a1c986223587c2611a91c17277d0dbc205a3c600001eb7a410bba53a.jpg)  
Fig. 15. Comparison of MSE and RMSE for Keq predictions of NN vs NN + NDR.

To address this, Linear Data Reconciliation (LDR) was applied again post-prediction, this time focusing on enforcing mass balance across the distillation column streams (Mousa et al., 2025). Unlike the nonlinear reconciliation used for the reactor, the LDR here specifically enforced algebraic constraints (mass balances) without retraining the NN. The LDR application successfully eliminated constraint violations, achieving zero mass balance constraint error, as illustrated in Fig. 16 where NN + LDR bar is not visible due to zero residual value (fully reconciled result). However, it is important to note that although LDR corrected constraint violations, it could not fully compensate for inaccuracies originating from the degraded NN predictions caused by data scarcity.

This dependency highlights an important insight: while reconcilia tion methods are powerful for enforcing physical validity, their success is partially reliant on the quality of the NN’s original outputs.

Thus, the observed deviations in flowrate predictions after recon ciliation can be interpreted in two ways, depending on user objectives:

• If the primary goal is strict physical consistency (mass balance enforcement), then the method remains highly effective even under limited data.

• If high predictive accuracy is also a priority, then maintaining robust NN training is essential, particularly under data-sparse conditions.

This two-step case study (reactor followed by distillation) demon strates not only the effectiveness but also the versatility and modularity of the proposed NN + (N/L)DR approach across multi-unit process systems. It further establishes that while reconciliation ensures physical law adherence at each process unit, initial data-driven model accuracy remains a cornerstone for overall system-wide prediction reliability.

In conclusion, this integrated process study reinforces the potential of the methodology for real-world, large-scale process systems, espe cially when multiple interconnected constraints must be satisfied sequentially. Although additional computational overhead may arise from repeated reconciliations across units, the results clearly demonstrate that the methodology can maintain high physical fidelity even in challenging, data-limited scenarios, highlighting the compre hensive and multi-faceted nature of the work conducted.

## 7. Conclusion

To conclude this paper, we have introduced a set of hybrid frame works aimed at addressing the challenges of achieving accurate pre dictions for data driven method while ensuring adherence to fundamental physical and operational constraints. These methodologies integrate artificial neural network with constraint-enforcing principles, leveraging the synergy between predictive capabilities and the preser vation of system integrity. By embedding domain knowledge into modeling frameworks, they provide a pathway for reconciling the flex ibility of machine learning with the rigor of physical laws.

The main contribution of this study is the introduction of a novel hybrid methodology integrating neural networks with nonlinear data reconciliation (NDR) to simultaneously achieve predictive accuracy and strict physical constraint adherence in chemical process modeling. Un like traditional hybrid approaches that embed constraints within network training, the proposed NN + NDR framework decouples pre diction and constraint enforcement into two distinct stages: first generating flexible, purely data-driven predictions, and then applying nonlinear reconciliation to strictly enforce physical laws. This separa tion allows NN + NDR to eliminate constraint violations entirely, achieving perfect physical consistency. This modular structure, that acts as a predictor-corrector method, significantly enhances flexibility, scalability, and computational efficiency, providing a practical alter native to more complex physics-informed architectures. To reach this conclusion, the proposed method was compared with two recent ap proaches which we have modified (second contribution) in an attempt to make them more effective in terms of strict compliance with physics constraints.

The effectiveness of the proposed framework was validated across multiple structured evaluations, ranging from a Gibbs reactor case, robustness assessments under noise and data scarcity, to a full reactor distillation integrated process. These extensive studies demonstrate that the methodology was not limited to a single case study, but sys tematically tested under diverse and challenging conditions.

Error Distribution  
![](images/4e1dfbdb29b88fd771e912e69d552309e593b97e6a0f007a2c0fef8cc9cac890.jpg)  
Fig. 16. Error distribution of mass balance equality of NN vs NN + LDR.

Across these applications, the NN + NDR framework proved capable of strictly enforcing nonlinear physical laws such as equilibrium con straints and mass balances while maintaining strong predictive perfor mance. Importantly, it achieved these outcomes with minimal computational overhead including multiple neural network initializa tions and the reconciliation step, thus confirming its suitability for scalable, industrial-scale systems.

The results highlighted the major strengths of the approach: by explicitly correcting predictions through reconciliation rather than embedding constraints into the learning process, the method maintains physical rigor without sacrificing the flexibility of data-driven modeling. At the same time, the analysis revealed the critical dependence of reconciliation effectiveness on the quality of initial neural network predictions, particularly under conditions of extreme data scarcity.

In conclusion, this work bridges empirical machine learning with first-principles rigor in a computationally efficient and modular way, offering a promising pathway for advancing modern process systems engineering. Future research will aim to extend the NN + NDR meth odology to dynamic, time-dependent systems and explore its integration with economic optimization objectives, enabling simultaneous enforcement of physical feasibility and financial performance in com plex industrial processes. Such developments would further enhance the methodology’s role in process design, optimization, and decision making.

## CRediT authorship contribution statement

Jana Mousa: Writing – original draft, Validation, Software, Meth odology, Formal analysis, Conceptualization. Stephane ´ Negny: Writing – review & editing, Validation, Supervision, Project administration, Methodology, Funding acquisition, Formal analysis, Conceptualization. Rachid Ouaret: Software, Methodology, Formal analysis, Conceptualization.

## Declaration of competing interest

No conflicts of interest to report to the board.

## References

Arvind. S.. Pomaie. R.. Bhat. R.V.. 2024. Karush-Kuhn-Tucker Condition-Trained Neural Networks (KKT Nets). doi:10.48550/arXiv.2410.15973.

Bai, S., McLean, D.D., Thibault, J., 2005. Enhancing controller performance via dynamic data reconciliation. Can. J. Chem. Eng. 83, 515–526. https://doi.org/10.1002 cice.5450830315.

Cai, S., Mao, Z., Wang, Z., Yin, M., Karniadakis, G.E., 2021. Physics-informed neura networks (PINNs) for fluid mechanics: a review. doi:10.48550/arXiv.2105.09506.

Cavalcanti, F.M., Kozonoe, C.E., Pacheco, K.A., Alves, R.M., de, B., Cavalcanti, F.M., Kozonoe, C.E., Pacheco, K.A., Alves, R.M., de, B., 2021. Application of Artificial neural networks to chemical and process engineering. Deep Learning Applications. IntechOpen. https://doi.org/10.5772/intechopen.96641.

Cencic, O., 2016. Nonlinear data reconciliation in material flow analysis with software STAN. Sustain. Environ. Res. 26. https://doi.org/10.1016/j.serj.2016.06.002.

Chen. H.. Flores, G.E.C.. Li. C.. 2024. Physics-informed neural networks with hard linear equality constraints. doi:10.48550/arXiv.2402.07251

Cuomo, S., Cola, V.S.di, Giampaolo, F., Rozza, G., Raissi, M., Piccialli, F., 2022. Scientific machine learning through physics-informed neural networks: where we are and what's next. doi:10.48550/arXiv.2201.05624.

Escapil-Inchausp´e, P., Ruz, G.A., 2023. Hyper-parameter tuning of physics-informed neural networks: application to Helmholtz problems. doi:10.48550/arXiv.2205 .06704.

Femine, C.D., 2024. KKT-Informed Neural Network. doi:10.48550/arXiv.2409.09087

Ghosh, D., Hermonat, E., Mhaskar, P., Snowling, S., Goel, R., 2019. Hybrid modeling approach integrating first-principles models with subspace identification. Ind. Eng. Chem. Res. 58, 13533–13543. https://doi.org/10.1021/acs.iecr.9b00900.

Giovannelli, T., Sohab, O., Vicente, L.N., 2023. The limitation of neural nets for approximation and optimization. doi:10.48550/arXiv.2311.12253.

Kelly, J.D., 2004. Techniques for solving industrial nonlinear data reconciliation problems. Comput. Chem. Eng. 28, 2837–2843. https://doi.org/10.1016/j. compchemeng.2004.06.009.

Kiˇs, K., Klauˇco, M., 2019. Neural network based explicit MPC for chemical reactor control. doi:10.48550/arXiv.1912.04684.

Krishnapriyan, A.S., Gholami, A., Zhe, S., Kirby, R.M., Mahoney, M.W., 2021. Characterizing possible failure modes in physics-informed neural networks doi:10.48550/arXiv.2109.01050.

Kuhn, H.W., Tucker, A.W., 1951. Nonlinear Programming. In: Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability. University of California Press, pp. 481–493.

Lˆe, T., 2020. Karush-Kuhn-Tucker optimality conditions and duality for multiobjective semi-infinite programming with equilibrium constraints. Yugosl. J. Oper. Res. 31, 24. https://doi.org/10.2298/YJOR200117024L. –24.

Moghaddas, M., Tohidi, G., 2020. An efficient neurodynamic model to solve nonconvex nonlinear optimization problems and its applications. Expert. Syst. 37, e12498. https://doi.org/10.1111/exsy.12498

Mousa, J., Negny, S., Ouaret, R., Pretoro, A.D., Montastruc, L., 2025. Incorporating physical constraints inside neural networks to improve their accuracy and physica reliability for chemical engineering unit operations modeling. Comput. Chem. Eng. 199. 109156. https://doi.org/10.1016/i.compchemeng.2025.109156.

Nakamura-Zimmerer, T., Gong, Q., Kang, W., 2022. Neural Network Optimal Feedback Control with Guaranteed Local Stability JEFE Open J Control Syst 1. 210–222 https://doi org/10.1109/OJCSYS 2022.3205863

Qiu, S., Cui, X., Ping, Z., Shan, N., Li, Z., Bao, X., Xu, X., 2023. Deep learning techniques in intelligent fault diagnosis and prognosis for industrial systems: a review. Sensors 23, 1305. https://doi.org/10.3390/s23031305.

Raissi, M., Perdikaris, P., Karniadakis, G.E., 2019. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. Phys. 378, 686–707. https://doi org/10.1016/j.jcp.2018.10.045.

Schweidtmann, A.M., Zhang, D., von Stosch, M., 2024. A review and perspective on hybrid modeling methodologies. Digit. Chem. Eng. 10, 100136. https://doi.org/ 10.1016/j.dche.2023.100136.

Sharma, R., Agrawal, D., Kodamana, H., 2022. Data reconciliation frameworks for dynamic operation of hybrid renewable energy systems. ISA Trans. 128, 424–436. https://doi.org/10.1016/j.isatra.2021.12.006.

Tian, X., Xia, B., Yu, Z., Yang, S.-H., 2006. Non-linear dynamic data reconciliation for industrial processes. In: 2006 IEEE International Conference on Systems, Man and Cybernetics. Presented at the 2006 IEEE International Conference on Systems, Man and Cybernetics, pp. 5291–5296. https://doi.org/10.1109/ICSMC.2006.385149.