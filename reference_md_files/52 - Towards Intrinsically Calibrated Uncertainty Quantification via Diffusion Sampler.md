# Towards Intrinsically Calibrated Uncertainty Quantification in Industrial Data-Driven Models via Diffusion Sampler

Yiran Ma , Jerome Le Ny , Senior Member, IEEE, Zhichao Chen , and Zhihuan Song

Abstract—In modern process industries, data-driven models are important tools for real-time monitoring when key performance indicators are difficult to measure directly. While accurate predictions are essential, reliable uncertainty quantification (UQ) is equally critical for safety, reliability, and decision-making, but remains a major challenge in current data-driven approaches. In this work, we introduce a diffusion-based posterior sampling framework that inherently produces well-calibrated predictive uncertainty via faithful posterior sampling, eliminating the need for post-hoc calibration. In extensive evaluations on synthetic distributions, the Raman-based phenylacetic acid soft sensor benchmark, and a real ammonia synthesis case study, our method achieves practical improvements over existing UQ techniques in both uncertainty calibration and predictive accuracy. These results highlight diffusion samplers as a principled and scalable paradigm for advancing uncertainty-aware modeling in industrial applications.

Index Terms—Industrial data-driven modeling, uncertainty quantification, Bayesian inference, diffusion sampler, stochastic optimal control, Schrodinger bridge, neural networks¨

## I. INTRODUCTION

Data-driven models have become essential tools in modern process industries, enabling the indirect estimation of key performance indicators that are difficult to physically measure in real time [1]. However, in practice, such models often suffer from a lack of trust from industrial practitioners, which substantially limits their deployment in safety-critical and decision-driven scenarios. A major reason is that most existing data-driven models provide only point predictions without reliable measures of confidence, making it difficult to assess the risk associated with their outputs and to support robust operational decision making [2].

Formally, let $x \in \mathbb { R } ^ { d }$ denote the vector of easy-to-measure variables and $y ~ \in ~ \mathbb { R }$ the target performance indicator. In most existing data-driven modeling practices, given a historical dataset $\mathcal { D } = ( x _ { i } , y _ { i } ) i = 1 ^ { N }$ , a deterministic mapping $\hat { y } = f _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) , \ \boldsymbol { \theta } \in \Theta .$ , is learned by minimizing the prediction error (e.g., in a mean-square sense). However, such point estimators provide no explicit mechanism to account for either data noise (e.g., measurement errors and process disturbances) or model uncertainty arising from limited data and model misspecification. Consequently, they often yield overconfident yet incorrect predictions, which is particularly dangerous in safety-critical and optimization-driven process control.

Therefore, beyond providing point predictions, it is crucial to assess their reliability. To meet this demand, this work focuses on uncertainty quantification, aiming to deliver not only accurate predictions of $y$ but also a calibrated predictive distribution $p ( \boldsymbol { y } \mid \boldsymbol { x } , \mathcal { D } )$ . Such a distribution allows for constructing statistically meaningful credible intervals. For example, a nominal 95% credible interval should empirically contain the true value approximately 95% of the time. This capability is particularly valuable for industrial tasks such as reliable decision-making [3], risk-sensitive control [4], and optimization [5]. Moreover, it underpins data-efficient learning by identifying areas of uncertainty in the model’s knowledge [6].

Bayesian approaches naturally address this need because, when faced with multiple plausible and competing explanations for observed data, they take all possibilities into account rather than prematurely committing to any single explanation [7], [8]. This helps in understanding what a data-driven model does not know, which is important for safety-critical scenarios. In practice, Bayesian models predicts by marginalization $\begin{array} { r } { p ( y | x ) = \int p ( y | x , \theta ) p ( \theta | \mathcal { D } ) d \theta } \end{array}$ , thus requiring a proper likelihood function $p ( \boldsymbol { y } | \boldsymbol { x } , \boldsymbol { \theta } )$ and the complete representation of the posterior $p ( \theta | \mathcal { D } )$ . Although conjugate Bayesian models underpinned many early successes in industrial data-driven modeling [9], typically by adopting linear and Gaussian likelihood, conjugate exponential-family likelihoods, or low-dimensional parameterizations that admit closed-form posteriors, these successes rely on strong structural assumptions. In the nonlinear, heteroscedastic noise and high-dimensional regimes of modern complex industrial practice, more complex models such as neural networks (NNs) are used [10], where closed-form posteriors are no longer available; explicit Bayesian inference becomes either inapplicable or computationally prohibitive. Therefore, the problem of predictive uncertainty quantification (UQ) for industrial models has not been comprehensively addressed in the literature.

Furthermore, due to the intractability of exact Bayesian inference in modern nonlinear and high-dimensional industrial process models, most existing industrial UQ methods resort to classical approximate inference schemes or heuristic uncertainty estimators, thus requiring post-hoc calibration to compensate for systematic bias in uncertainty estimates. In industrial scenarios, however, obtaining ground-truth data often relies on time-consuming laboratory analysis; consequently, reserving a portion of this already scarce data or collecting a new independent set for post-hoc calibration [2], [11], [12] is practically expensive, time-consuming, and sometimes even infeasible. Consequently, the central challenge in designing industrial UQ frameworks lies in achieving intrinsically calibrated predictive uncertainty. Namely, uncertainty estimates that are reliable by construction and do not depend on additional post-hoc calibration or extra ground-truth data.

Fundamentally, the need for post-hoc calibration of existing methods reflects a deeper issue: the underlying approximate inference or sampling procedures fail to faithfully represent the true posterior distribution, leading to systematic undercoverage or mis-calibration. In complex industrial models, the posterior is often high-dimensional and non-Gaussian, making faithful posterior approximation particularly challenging. Existing methods typically fail to provide reliable posterior coverage for three fundamentally different reasons. First, methods based on restrictive parameterizations (e.g., Monte Carlo dropout [13]–[15] and Mean-field variational inference (MFVI) [16]) impose unimodal or factorized posterior families that cannot represent complex structures. Second, samplingbased methods such as SG-MCMC [17] often suffer from prohibitively slow mixing in high-dimensional landscapes. Third, particle-based methods such as SVGD [2] rely on finite-particle kernelized approximations that degenerate in high dimensions, leading to particle collapse or insufficient mode coverage [2], [11], [12]. As a result, these methods produce systematically under-covered posteriors, resulting in miscalibrated uncertainty estimates in practice and often necessitating post-hoc calibration on an additional independent dataset. Therefore, the key technical difficulty lies in ensuring faithful coverage of the entire posterior support, rather than approximating only the dominant mode(s).

To achieve a more faithful approximation of the posterior distribution, we adopt the Schrodinger bridge (SB) perspective¨ [18]. This theoretically grounded framework casts posterior inference as a stochastic transport problem: finding a pathspace distribution whose terminal marginal matches the target posterior while minimizing the KL divergence from a reference diffusion process. However, the main difficulty lies in reformulating this strict transport problem into a tractable, unconstrained objective to enable efficient end-to-end training via gradient descent. While earlier approaches relied on iterative proportional fitting [19], recent work shows that the SB problem admits a stochastic optimal control (SOC) interpretation, offering a principled and well-posed objective for inference.

Therefore, recent advancements have increasingly focused on learning-based methods leveraging diffusion processes [20]– [22], which are named diffusion samplers. These diffusion samplers are particularly compelling due to their scalability in high-dimensional settings. Recently, adjoint matching [23], [24] has further improved the efficiency of training diffusion samplers, making them more practical for complex models. While these methods yield principled samplers from complex posterior or energy distributions that can be used to compute uncertainty measures, explicit UQ applications of diffusion samplers are still relatively limited; most existing work focuses on sampling methodology and theoretical analysis rather than dedicated UQ case studies.

In this work, we therefore build upon the SB formulation to construct a robust diffusion sampler for industrial datadriven modeling, aiming to provide calibrated uncertainty by capturing complex structure of posterior distributions inherent in such practical industrial models.

To summarize, the contributions of this paper are as follows:

1) We introduce the diffusion sampler for posterior sampling, providing a principled alternative to conventional industrial data-driven modeling and establishing a theoretically grounded framework for UQ.

2) We demonstrate the practical value of the framework on synthetic benchmarks and real-world industrial modeling tasks, showing improved uncertainty quality without the need for post-hoc calibration.

3) Additional analyses show that the method exhibits robustness to hyperparameters and smooth optimization dynamics, highlighting diffusion samplers as a promising direction for uncertainty-aware industrial models.

## II. PRELIMINARIES

## A. Uncertainty Quantification in Machine Learning

Machine learning models, in particular deep neural networks, are often regarded as black-box predictors, producing outputs without revealing the confidence or reliability of their decisions. In many real-world scenarios, such as the industrial applications we study, however, relying solely on point predictions is insufficient, as overconfident yet incorrect predictions can lead to costly or unsafe outcomes. This has motivated a growing demand for UQ, which aims to complement point predictions with principled measures of uncertainty.

The central objective in UQ is calibration, which formalizes the statistical consistency between predicted probabilities and empirical frequencies. A well-calibrated model satisfies Definition 1.

Definition 1 (Calibration). Given an input x, consider a probabilistic model that provides a predictive distribution $p ( y \mid x )$ over the target y. Let $\mathcal { C } _ { 1 - \alpha } ( x )$ denote the $( 1 - \alpha ) \cdot$ credible set induced by $p ( y \mid x )$ , representing the region that contains true y with a nominal coverage level of $1 - \alpha$ The model is perfectly calibrated if, for any coverage level $1 - \alpha \in ( 0 , 1 )$ , the following holds:

$$
\operatorname * {P r} \left(y \in \mathcal {C} _ {1 - \alpha} (x)\right) = 1 - \alpha .\tag{1}
$$

Intuitively, (1) means that the probability of observations falling inside the empirical $( 1 - \alpha )$ )-credible interval matches the expected coverage 1−α. Based on this understanding, calibration for regression tasks can be defined, see Definition 2.

Definition 2 (Calibrated regression [25]). A regression model is well-calibrated if, for any $1 - \alpha \in ( 0 , 1 )$ ,

$$
\operatorname * {P r} \bigl (F _ {x} ^ {- 1} (\alpha / 2) \leq y \leq F _ {x} ^ {- 1} (1 - \alpha / 2) \mid x \bigr) = 1 - \alpha\tag{2}
$$

for every $x ,$ where $F _ { x } ^ { - 1 }$ denotes the inverse CDF of predictive distribution of y given x.

## B. Bayesian Learning

Bayesian learning is known as a natural and comprehensive framework for UQ, as it incorporates the representation of both aleatoric (data) and epistemic (model) uncertainty. Unlike point estimation methods that rely on a single optimum, i.e., maximum a posteriori (MAP) or maximum likelihood estimation (MLE), Bayesian approaches integrate over all plausible parameter configurations, yielding calibrated predictive uncertainty and improved robustness.

Mathematically, given labeled training dataset D = $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , and the probabilistic model $p ( \boldsymbol { y } \mid \boldsymbol { x } , \boldsymbol { \theta } )$ parameterized by $\theta \in \Theta$ , the Bayesian predictive distribution is

$$
p (y \mid x, \mathcal {D}) = \int_ {\Theta} p (y \mid x, \theta) p (\theta \mid \mathcal {D}) \mathrm{d} \theta .\tag{3}
$$

In practice, this integral is empirically estimated by sampling θ from the posterior distribution $p ( \theta \mid \mathcal { D } )$ . The point predictions and the confidence intervals of each prediction are acquired by computing the quantiles of the Bayesian predictive distribution $p ( \boldsymbol { y } \mid \boldsymbol { x } , \mathcal { D } )$ . Neural networks that predict under this paradigm are called Bayesian neural networks (BNNs) [7], [8], [26].

Definition 3 (Point Predictions and Credible Intervals for Regression). 1) The point prediction is the expected value of y under the predictive distribution,

$$
\hat {y} (x) = \mathbb {E} _ {p (y | x, \mathcal {D})} [ y ] = \int y p (y \mid x, \mathcal {D}) \mathrm{d} y.\tag{4}
$$

2) Let $F _ { x } ^ { - 1 }$ denote the inverse CDF of $p ( y \mid x , \mathcal { D } )$ , which is available in numerical form, a $( 1 - \alpha )$ credible interval for y is obtained by taking the lower $( \alpha / 2 )$ and upper $( 1 - \alpha / 2 )$ quantiles:

$$
\mathcal {C} _ {1 - \alpha} (x) = \left[ F _ {x} ^ {- 1} (\frac {\alpha}{2}), F _ {x} ^ {- 1} (1 - \frac {\alpha}{2}) \right],\tag{5}
$$

Under this Bayesian framework, it is also natural to consider the decomposition of the aleatoric and epistemic uncertainty to guarantee the mathematical rigor of UQ [27]. Specifically, the aleatoric uncertainty is depicted by the probabilistic model (likelihood) $p ( \boldsymbol { y } \mid \boldsymbol { x } , \boldsymbol { \theta } )$ itself, while the epistemic uncertainty is from the posterior $p ( \theta \mid \mathcal { D } )$

Posterior inference lies at the core of Bayesian learning. Specifically, given a dataset $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { \bar { N } }$ , a likelihood function $p ( \boldsymbol { y } \mid \boldsymbol { x } , \boldsymbol { \theta } )$ , and a prior distribution $p ( \theta )$ , the objective is to infer the posterior distribution over model parameters:

$$
p (\theta \mid \mathcal {D}) = \frac {p (\mathcal {D} \mid \theta) p (\theta)}{p (\mathcal {D})},
$$

where $\begin{array}{c} \begin{array} { l } { { p ( \mathcal { D } ) } } \end{array} { } = { } \int p ( \mathcal { D } \quad | \quad \theta ) p ( \theta ) d \theta  \end{array}$ is the evidence. By maintaining a full posterior distribution rather than a single point estimate, Bayesian learning naturally captures epistemic uncertainty, which can then be propagated to the predictive distribution $p ( y \mid x )$ . Such uncertainty quantification serves as the foundation for the label-efficient strategies developed in this work.

In practice, posterior inference implements this process computationally by approximating the intractable posterior distribution, as the marginal likelihood $p ( \mathcal { D } )$ cannot be computed analytically. Typically, the posterior $p ( \theta \mid \mathcal { D } )$ is approximated by minimizing the KL divergence between a proposed distribution $q ( \theta )$ and the true posterior distribution

$$
\underset {q} {\arg \min} \mathbb {D} _ {\mathrm{KL}} (q (\theta) \| p (\theta \mid \mathcal {D})).\tag{6}
$$

The proposed distribution $q ( \theta )$ denotes the variational approximation in variational inference, and corresponds to the law of particles in sampling-based approaches.

C. Connections Between Sampling and Finite-Horizon Stochastic Optimal Control

Bayesian inference can be performed through sampling, where samples drawn from $p ( \theta \mid \mathcal { D } )$ are used to approximate expectations under the posterior distribution. Compared to explicit variational methods, sampling-based approaches make fewer assumptions on the posterior and can achieve higherfidelity approximations of complex posterior distributions [28]. However, sampling from complex and unnormalized highdimensional posteriors remains a challenging problem in practice. Recent developments in optimal transport and stochastic control [29] have inspired a unifying view of sampling as a Schrodinger bridge problem (Definition 4), naturally formu-¨ lated as a finite-horizon optimal control problem.

Definition 4 (Schrodinger bridge problem [18])¨ . Let <sup>S</sup> be a reference path measure on $C ( [ 0 , 1 ] , \mathbb { R } ^ { d } )$ (e.g., Wiener measure), and let $\pi _ { 0 } , \ \pi _ { 1 }$ be probability measures on $\mathbb { R } ^ { d } .$ . For any path measure $\mathbb { Q }$ on this space, denote by $\mathbb { Q } _ { t }$ its marginal distribution at time t. Define the set of admissible path measures:

$$
\mathcal {D} (\pi_ {0}, \pi_ {1}) := \left\{\mathbb {Q} \ll \mathbb {S} \mid \mathbb {Q} _ {0} = \pi_ {0}, \mathbb {Q} _ {1} = \pi_ {1} \right\}.
$$

Then the Schrodinger bridge problem seeks¨

$$
\mathbb {Q} ^ {*} = \underset {\mathbb {Q} \in \mathcal {D} (\pi_ {0}, \pi_ {1})} {\arg \min} \mathbb {D} _ {\mathrm{KL}} (\mathbb {Q} \| \mathbb {S}),\tag{7}
$$

where $\mathbb { D } _ { \mathrm { K L } }$ denotes the Kullback–Leibler $( K L )$ divergence (or relative entropy) on path space.

Definition 4 states the modern formulation of the Schrodinger bridge problem, which seeks the stochastic evolu-¨ tion <sup>Q</sup> that transports $\pi _ { 0 }$ to $\pi _ { 1 }$ while minimizing the relative entropy with respect to the reference diffusion <sup>S</sup>. Next, we explain how the posterior inference problem can be reformulated as such a Schrodinger bridge problem, where the key¨ step relies on the data processing inequality (Lemma 1).

Lemma 1 (Data processing inequality (See Appendix A in [18])). Let <sup>Q</sup>, <sup>S</sup> be two probability measures on path space, with terminal distributions $\mu _ { 1 } = \mathbb { Q } _ { 1 } , \pi _ { 1 } = \mathbb { S } _ { 1 }$ . Then

$$
\mathbb {D} _ {\mathrm{KL}} (\mu_ {1} \parallel \pi_ {1}) \leq \mathbb {D} _ {\mathrm{KL}} (\mathbb {Q} \parallel \mathbb {S}),
$$

where the equality holds if and only if

$$
\mathbb {Q} (\cdot \mid \theta_ {1}) = \mathbb {S} (\cdot \mid \theta_ {1}) \quad f o r \mu_ {1} \text {-almost every} \theta_ {1}.
$$

By the data processing inequality, the KL divergence between marginals $\mathbb { D } _ { \mathrm { K L } } ( \mu _ { 1 } \parallel \pi _ { 1 } )$ is upper-bounded by the pathspace KL divergence $\mathbb { D } _ { \mathrm { K L } } ( \mathbb { Q } \parallel \mathbb { S } )$ . Revisit the typical optimization problem 6 of posterior inference, and let $\mu _ { 1 } = q ( \theta )$ and $\pi _ { 1 } = p ( \theta \mid \mathcal { D } )$ . The objective of 6 can be relaxed into a pathspace KL minimization problem as formulated in (7). This relaxation enables a more tractable and smooth optimization via drift control, as elaborated in the following part.

Proposition 1 (Stochastic optimal control formulation of Schrodinger bridge problem [18])¨ . Let the reference process $\mathbb { S } ~ = ~ \mathbb { Q } ^ { \bar { f } , \pi _ { 0 } }$ be an Ito diffusion with driftˆ f and constant diffusion coefficient γ:

$$
\mathrm{d} \theta_ {t} = f (t, \theta_ {t}) \mathrm{d} t + \sqrt {\gamma} \mathrm{d} B _ {t}, \quad \theta_ {0} \sim \pi_ {0},
$$

and the controlled process $\mathbb { Q } ^ { f + u , \pi _ { 0 } }$ with additional drift control u:

$$
\mathrm{d} \theta_ {t} = f (t, \theta_ {t}) \mathrm{d} t + u (t, \theta_ {t}) \mathrm{d} t + \sqrt {\gamma} \mathrm{d} B _ {t}, \quad \theta_ {0} \sim \pi_ {0}.
$$

The Schrodinger bridge problem with marginals¨ $\theta _ { 0 } \sim \pi _ { 0 }$ and $\theta _ { 1 } \sim \pi _ { 1 }$ is equivalent to the finite-horizon stochastic optimal control problem

$$
\begin{array}{l} \min _ {u} \mathbb {E} \bigg [ \int_ {0} ^ {1} \frac {1}{2 \gamma} \| u (t, \theta_ {t}) \| ^ {2}   \mathrm{d} t \bigg ], \\ s. t. \quad \mathrm{d} \theta_ {t} = f (t, \theta_ {t})   \mathrm{d} t + u (t, \theta_ {t})   \mathrm{d} t + \sqrt {\gamma}   \mathrm{d} B _ {t}, \\ \theta_ {0} \sim \pi_ {0}, \theta_ {1} \sim \pi_ {1}. \end{array}\tag{8}
$$

The solution of (8) defines a controlled SDE whose simulation yields terminal states distributed according to the target posterior. This observation motivates the construction of diffusion samplers, which approximate the optimal control and generate samples by simulating the controlled SDE and collecting the terminal states.

However, (8) enforces exact matching of the target distribution within a finite time horizon, which can require large control magnitudes when the target is complex, as is often the case for posteriors in industrial models, leading to an ill-conditioned optimization problem. Therefore, we relax this hard constraint into a soft terminal penalty, yielding the following equivalent formulation.

Proposition 2 (Stochastic optimal control cost with soft terminal penalty [21], [30], [31]). Under the conditions $f ( t , { \boldsymbol { \theta } } _ { t } ) \equiv 0$ and the initial distribution being the Dirac measure at the origin $\begin{array} { r l r } { \pi _ { 0 } } & { { } = } & { \delta _ { 0 } , } \end{array}$ (8) admits, by Girsanov’s theorem and Ito’s formula, the same solution as the following optimizationˆ

problem:

$$
\begin{array}{l} u ^ {*} = \underset {u} {\arg \min} \mathbb {E} \left[ \int_ {0} ^ {1} \frac {1}{2 \gamma} \| u (t, \theta_ {t}) \| ^ {2} \mathrm{d} t - \log \frac {\pi_ {1} (\theta_ {1})}{\mathcal {N} (\theta_ {1} | 0 , \gamma \mathbf {I} _ {d})} \right], \\ s. t. \quad \mathrm{d} \theta_ {t} = u (t, \theta_ {t})   \mathrm{d} t + \sqrt {\gamma}   \mathrm{d} B _ {t}, \\ \theta_ {0} \sim \delta_ {0}. \end{array}\tag{9}
$$

Proposition 2 converts the hard terminal constraint to a soft terminal penalty, which facilitates a principled trade-off between staying close to the reference prior (running cost) and matching the target posterior distribution (terminal cost). Moreover, this transition yields a smoother, unconstrained differentiable objective [30] that enables end-to-end training via gradient descent. In practice, the drift control $u ( t , \theta _ { t } )$ is parameterized by a neural network and optimized via gradient descent on (9). Diffusion-based samplers, such as the Path Integral Sampler (PIS) [21] and the Neural Schrodinger–F ¨ ollmer¨ Sampler (NSFS) [31], build upon this relaxed formulation. Both methods learn to approximate an optimal control policy with neural networks, but differ in their theoretical derivations, parameterization schemes, and application domains.

## III. UNCERTAINTY QUANTIFICATION IN INDUSTRIAL MODELS VIA DIFFUSION SAMPLER

This section elaborates on how to incorporate diffusion samplers into industrial data-driven modeling to equip models with reliable and well-calibrated predictive uncertainty. Given that the diffusion sampler lies at the core of modeling epistemic uncertainty in our framework, we refer to our method as Diffusion-based Uncertainty Quantification (DiffUQ).

Fig. 1 illustrates the overall workflow of our proposed method. Specifically, our method includes 3 steps: 1) construct a probabilistic regression model $p ( \boldsymbol { y } \mid \boldsymbol { x } , \boldsymbol { \theta } )$ as the base model, where x denotes the process variables, y is the target variable to be predicted, and θ represents the model’s parameter vector; 2) draw n samples $\{ \theta _ { i } \} _ { i = 1 } ^ { n }$ from the posterior $p ( \theta \mid \mathcal { D } )$ given a dataset D; 3) predict by n forward passes and aggregation.

Epistemic uncertainty arises from the presence of multiple plausible parameter configurations, which are captured by the complex and typically unnormalized posterior distribution $p ( \theta \mid \mathcal { D } )$ after observing the dataset D.

## A. Probabilistic Regression Model

The probabilistic model captures the conditional distribution of y given x and θ, thereby providing a framework to predict the target variable and represent aleatoric uncertainty. A proper probabilistic model $p ( \boldsymbol { y } \mid \boldsymbol { x } , \boldsymbol { \theta } )$ is critical for both accurate prediction and aleatoric uncertainty representation. Considering that industrial processes may have heteroscedastic data noise [32], industrial models require explicit estimation of input-dependent noise variances. Consequently, two distinct neural networks (or, more generally, parametric functions) are employed to parameterize the probabilistic model, providing the mean and precision, respectively.

$$
p (y | x, \theta) = \mathcal {N} (\mathrm{NN} _ {\theta^ {\mu}} (x), e ^ {- \mathrm{NN} _ {\theta^ {\tau}} (x)}),\tag{10}
$$

![](images/a6727843526bd9416727de89fb350350855e89eb1d58a202f404cafa309c3ae4.jpg)  
Step 1: Designing a Probabilistic Soft Sensor

![](images/9730fa0ccde3157bf4ee8b1a41caef0ac4014b26b4d577ca31b8dd6baf4bf43a.jpg)  
Step 2: Sampling $\{ \theta _ { i } \} _ { i = 1 } ^ { n }$ from Posterior via Di usion Sampler

![](images/607ccac5af21f600f96d926e764d25a9d60ebea7ba216d2a3214a478e5247e2d.jpg)  
Step 3: Prediction ( Forward Passesn and Aggregation)

Fig. 1: Overview of the DiffUQ Framework

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Training a diffusion sampler
Require: Dataset $\mathcal{D} = \{x_i, y_i\}_{y=1}^N$, probabilistic model $p(\cdot|x,\theta)$, prior $p(\theta) = \mathcal{N}(0,I)$.
Ensure: $u_\phi(t,\theta)$ parameterized by $\phi$
1: Define: Augmented SDE drift $f_\phi(t,[\theta_t,c_t]) = [u_\phi(t,\theta_t), \frac{1}{2}\|u_\phi(t,\theta_t)\|^2]$, diffusion $g(t,[\theta_t,c_t]) = [\gamma,0]$, $\theta_0 = \mathbf{0}$, $c_0 = 0$; discretization step $\Delta t = 1/T$
2: for i = 1 to max_iter do
3: SDE simulation
$(\theta_1,c_1) \leftarrow (\theta_0,c_0) + \Sigma_{t=0}^T f(t,[\theta_t,c_t]) \Delta t + \Sigma_{t=0}^T g(t,[\theta_t,c_t]) \Delta W_t$ (11)
4: Sample mini-batch $\mathcal{S} \subset \mathcal{D}$ and run gradient descent step
$\phi \leftarrow \phi - \eta \nabla_\phi[c_1 - \frac{|\mathcal{D}|}{|\mathcal{S}|} \log p(\mathcal{S}|\theta_1)p(\theta_1) + \log \mathcal{N}(\theta_1|0,\gamma\mathbf{I}_d)]$
5: end for
</div>

where $\theta = [ \theta ^ { \mu } , \theta ^ { \tau } ]$ . Specifically, $\theta ^ { \mu }$ and $\theta ^ { \tau }$ parameterize the mean network $\mathrm { N N } _ { \theta ^ { \mu } }$ (point prediction) and the precision network $\mathrm { N N } _ { \theta ^ { \tau } }$ (aleatoric uncertainty), respectively. This design enables learning input-dependent noise variance instead of using a fixed one. Note that despite the notation ‘’NN’, these components can be any differentiable parametric functions.

## B. Posterior Sampling via Diffusion Sampler

1) Training: With the probabilistic regression model $p ( y \mid$ $x , \theta )$ defined in section III-A and a standard Gaussian distribution as an uninformative prior, we utilize a diffusion sampler to obtain samples from the resulting parameter posterior given D. Algorithm 1 shows the complete training procedure of the training process. Consider a reference Wiener process $\mathrm { d } \theta = \gamma \mathrm { d } W _ { t }$ . Instead of directly working with θ, the goal is to learn a optimal control $u _ { \phi } ( t , \theta )$ parameterized by $\phi ,$ such that the terminal states induced by the controlled diffusion process $\mathrm { d } \theta ~ = ~ u _ { \phi } ( t , \theta ) \mathrm { d } t + \gamma \mathrm { d } W _ { t }$ distribute according to the posterior $p ( \theta | \mathcal { D } )$ . This goal can be accomplished through gradient descent on a single objective (9). An augmented state $c _ { t }$ is used to integrate the running cost term $\| u \| ^ { 2 }$ along the path:

$$
c _ {1} = \int_ {0} ^ {1} \frac {1}{2 \gamma} \| u _ {\phi} (t, \theta_ {t}) \| ^ {2} \mathrm{d} t
$$

thus significantly reducing peak memory usage during training. The drift neural network $u _ { \phi } ( t , \theta )$ is parameterized<sup>1</sup> as an 8-layer MLP of width 32, using GELU activations for all experiments in our work (see Section A for details on network capacity). Layer normalization without learnable affine parameters is also adopted, as it can stabilize training. Batch normalization is strictly forbidden as it introduces dependence between samples within a batch and compromises the uniqueness of the solution [33].

Equation (11) shows that the SDE is simulated using the Euler–Maruyama (EM) discretization scheme, where $\Delta W _ { t } \sim$ $\mathcal { N } ( 0 , \Delta t )$ . Theoretically, this discretization has a provable error bound [20], [31] for finite-horizon stochastic optimal control problems. This discrete SDE simulation is implemented in an efficient manner using torchsde [34], [35], which integrates n trajectories in parallel. We use $\Delta t = 0 . 0 4$ for the training process and $\Delta t = 0 . 0 1$ , which will be introduced next.

2) Sampling: After training, n i.i.d. samples $\{ \theta _ { i } \} _ { i = 1 } ^ { n }$ are generated via simulating n trajectories and collecting the their terminal states. Each sample requires only $1 / \Delta t$ forward passes of the control network, and all trajectories are computed in parallel via torchsde. Hence, the per-sample computational cost becomes marginal and can be regarded as amortized into the training phase.

In our framework, multiple forward predictions are indeed used to estimate predictive statistics. However, the key distinction is that the diversity of posterior samples is not primarily governed by the number of samples drawn at inference time, but by the learned diffusion dynamics. During training, the controlled SDE learns a global transport that maps the reference distribution to the posterior manifold, effectively encoding multi-modal structure in the drift function. As a result, even a small number of trajectories can traverse different modes, since exploration is embedded in the dynamics itself rather than achieved through independent Monte Carlo exploration. Increasing the number of samples mainly reduces Monte Carlo estimation variance, but does not fundamentally alter posterior coverage.

## C. Posterior Prediction

Once posterior samples are obtained, the posterior predictive distribution $p ( y \mid x )$ is evaluated via the posterior predictive equation (3), where the intractable integral is approximated empirically using Monte Carlo estimation over the posterior samples

$$
p (y \mid x, \mathcal {D}) = \frac {1}{n} \sum_ {i = 1} ^ {n} p (y \mid x, \theta_ {i}),\tag{12}
$$

where $\theta _ { i } ~ \sim ~ p ( \theta ~ \mid ~ { \mathcal { D } } ) , i ~ = ~ 1 , . . . , n$ . Equation (12) implies that each prediction of our method requires n forward passes, which, in our implementation, is parallelized using torch.vmap. This process is referred to as aggregation and is also known as an ensemble of base models in some literature. Finally, in accordance with definition 3, uncertaintyaware prediction is performed.

## IV. EXPERIMENTS

In this section, we first introduce the evaluation setup (Section IV-A), including the metrics and baseline methods. We then demonstrate the advantage of diffusion-based sampling in capturing complex posterior distributions of industrial models through two illustrative toy examples (Section IV-B). Next, the effectiveness of DiffUQ is validated on a simple linear Raman-based soft sensor from the penicillin fermentation process (Section IV-C). Finally, experiments on a neural network–based industrial process model for the real-world ammonia synthesis process further highlight the capability of DiffUQ to deliver reliable uncertainty quantification in higherdimensional parameter spaces (Section IV-D).

## A. Evaluation Setup

1) Metrics: Consider a test set $\begin{array} { r c l } { \mathcal D _ { \mathrm { t e s t } } } & { = } & { \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { M } } \end{array}$ of size M . A series of evaluation metrics is employed to assess both the point prediction accuracy and the predictive uncertainty quality. To jointly evaluate these two aspects, the mean negative log-likelihood (NLL) is adopted to measure the quality of probabilistic predictions:

$$
\mathrm{NLL} = - \frac {1}{M} \sum_ {i = 1} ^ {M} \log p (y _ {i} \mid x _ {i}, \mathcal {D}),\tag{13}
$$

where $p ( y _ { i } \mid x _ { i } , \mathcal { D } )$ denotes the posterior predictive distribution given by the uncertainty-aware model. A lower NLL indicates that the model produces confident and accurate predictions by jointly accounting for accuracy and uncertainty calibration.

To further evaluate the calibration of predictive uncertainty, the expected calibration error (ECE) and maximum calibration error (MCE) [25], [36] are computed based on the empirical coverage of credible intervals. Given the posterior predictive distribution (12), the empirical coverage level $\hat { p }$ of the nominal coverage level $p$ is defined as $\begin{array} { r } { \hat { p } = \frac { \bar { 1 } } { M } \sum _ { j = 1 } ^ { M } \mathrm { \bar { I } } \{ y _ { i } \in \mathcal { C } _ { p } ( x _ { i } ) \} } \end{array}$ where $ { \mathcal { C } } _ { p } ( x _ { i } )$ is the credible interval defined in (5). The expected calibration error is then given by

$$
\mathrm{ECE} = \frac {1}{B} \sum_ {k = 0} ^ {B} | \hat {p} _ {k} - p _ {k} |,\tag{14}
$$

while the maximum calibration error is defined as

$$
\mathrm{MCE} = \max _ {j} (| \hat {p} _ {k} - p _ {k} |),\tag{15}
$$

where $p _ { k } = k / B , k = 1 , . . . , B$ , and B denotes the number of different nominal coverage levels evaluated. ECE evaluates the average alignment between predicted confidence and empirical accuracy, reflecting overall reliability. MCE captures the worstcase calibration error, which is crucial for preventing dangerous overconfidence in safety-critical operations. Together, they validate the trustworthiness of the predictive distribution.

For point prediction accuracy, denoting $\hat { y } _ { i }$ the point prediction (4) given $x _ { i } ,$ , the mean squared error (MSE) and mean absolute error (MAE) are used:

$$
\mathrm{MSE} = \frac {1}{M} \sum_ {i = 1} ^ {M} (y _ {i} - \hat {y} _ {i}) ^ {2}, \quad \mathrm{MAE} = \frac {1}{M} \sum_ {i = 1} ^ {M} | y _ {i} - \hat {y} _ {i} |.\tag{16}
$$

Finally, the coefficient of determination $( R ^ { 2 } )$ evaluates the overall goodness of fit:

$$
R ^ {2} = 1 - \frac {\sum_ {i = 1} ^ {M} (y _ {i} - \hat {y} _ {i}) ^ {2}}{\sum_ {i = 1} ^ {M} (y _ {i} - \bar {y}) ^ {2}}, \quad \text { where } \bar {y} = \frac {1}{M} \sum_ {i} y _ {i}.\tag{17}
$$

2) Baseline Methods: We compare our approach with several representative uncertainty quantification methods that cover the main methodological categories commonly adopted in industrial data-driven modeling. Specifically, we include: (i) MC Dropout [13], [14], which approximates Bayesian inference by injecting stochasticity at test time; (ii) deep ensembles (DE) [37], which capture model uncertainty through diversity across independently trained networks; (iii) meanfield variational inference (MFVI) [16], a classical variational Bayesian approach with factorized posterior assumptions; (iv) stochastic gradient Langevin dynamics (SGLD), as a representative stochastic-gradient MCMC method [38]; and (v) Stein variational gradient descent (SVGD) [2], representing deterministic particle-based Bayesian inference.

These methods are selected because they represent widely used and conceptually distinct approaches to uncertainty quantification, spanning variational inference, ensemble-based, and particle-based variational inference. In addition, we report results obtained using maximum a posteriori (MAP) estimation. Although MAP yields only a point estimate of model parameters and therefore does not account for model uncertainty, data uncertainty is still captured through the probabilistic output model, making MAP a meaningful baseline for assessing the effect of explicitly modeling parameter uncertainty.

In order to enable a comparison of the intrinsic calibration accuracy of uncertainty estimates, all methods are evaluated without post-hoc calibration.

## B. Sampling from Toy Posterior Distributions

This section presents synthetic examples that demonstrate the diffusion sampler’s capacity to explore complex or even illposed posterior distributions. In particular, the non-Gaussian and multimodal ‘smiley-face’ distribution and an ill-posed funnel distribution featuring a narrow neck and wide base are examined, reflecting typical challenges that may arise in posteriors of industrial models.

MFVI restricts the variational approximation to a meanfield Gaussian family, and Fig. 2a, 2e illustrate the limitations of this assumption when representing complex target distributions. Gradient flow–based sampling methods, such as SGLD (Fig. 2b, 2f) and SVGD (Fig. 2c, 2g), both fail to fully recover the structural characteristics of the target distribution. Fig. 2d, 2h demonstrates that the diffusion sampler can generate highquality samples covering all modes of such challenging distributions, outperforming the compared methods.

![](images/46ff87210699eaf52f43c812cf226fb2812ea18b9340776f87043aaf74549d19.jpg)

![](images/6394fb94794546e9f814224579a7847745141468d112482a2dccdca93670d499.jpg)  
(d) Diffusion Sampler  
(h) Diffusion Sampler  
Fig. 2: Comparison on the smiley-face (left) and funnel (right) distributions. (a,e) show mean-field variational contours; other panels show samples from different sampling methods.

## C. Penicillin Fermentation System Simulation

In this section, we consider an industrial-scale simulation (IndPenSim) [39], shown on Fig. 3, a 100,000-litre penicillin fermentation process. The dataset includes 100 batches of process and Raman spectroscopy measurements, which is the largest available resource for advanced data analytics in this domain. The process involves a Raman-based phenylacetic acid (PAA) soft sensor task, where Raman spectra serve as input variables to predict the PAA concentration. Since the Raman-based PAA soft sensor exhibits a strong linearity, it serves as an appropriate benchmark to investigate the performance of our UQ method in the context of simple linear models. Hence, this setting provides a simple case that is wellsuited for comparing the effectiveness of different methods.

![](images/a55572066d7b7f4b45a6af67ab903f93af64f984b29955256d9d45a4243580c0.jpg)  
Fig. 3: Summary of the first principle-based mathematical simulator of industrial-scale penicillin simulation [39]

1) Task Description: In this case study, the purpose is to develop a Raman-based soft sensor for the online prediction of PAA concentration during penicillin fermentation [39]. Out of the 100 batches, we select batches 1–60 for training and batches 61–90 for testing. The first 60 batches correspond to open-loop or manually operated processes, while batches 61–90 are subject to closed-loop control based on a simple linear soft sensor of PAA, which reflects the intended application of PAA prediction—supporting closed-loop process control. This split, therefore, ensures that the evaluation directly aligns with the original motivation of the task.

Raman spectra are generated in IndPenSim every 12 minutes across the wavenumber range $2 5 0 { - } 2 2 5 0 \ \mathrm { c m ^ { - 1 } }$ , yielding a large high-dimensional dataset, which is preprocessed following [39]. Specifically, the spectral regions $1 \bar { 5 } 4 0 { - } 1 5 8 0 \ \mathrm { c m ^ { - 1 } }$ and $1 9 5 0 { - } 2 0 5 0 ~ \mathrm { c m ^ { - 1 } }$ were identified and selected as informative for PAA concentration. The selected spectral regions were preprocessed using a Savitzky–Golay smoothing filter (15- point window), followed by taking the first derivative. These processed spectra served as the secondary variables x, while offline PAA measurements were interpolated via cubic splines to align with the spectral acquisition times.

Conventional Raman-based PAA soft sensors, however, provide only point predictions without accounting for predictive uncertainty. Considering prediction of PAA concentrations is essential for operator decision-making and closed-loop control, incorporating well-calibrated uncertainty estimates has the potential to enhance downstream task performance and improve overall process robustness and safety.

2) Probabilistic Model Structure: The probabilistic regression model for Raman-based PAA soft sensor follows (10). Considering the classical Raman-based PAA soft sensor is linear [39], $\mathrm { N N } _ { \theta ^ { \mu } }$ is specified as a linear model. $\mathrm { N N } _ { \theta ^ { \tau } }$ is specified as an MLP with 1 hidden layer of width 4 to capture data noise.

TABLE I: Results of Raman-based PAA soft sensor.

<table><tr><td rowspan="2">Method</td><td colspan="3">Uncertainty Quality</td><td colspan="3">Accuracy</td></tr><tr><td>NLL</td><td>ECE</td><td>MCE</td><td>MSE</td><td>MAE</td><td> $R^2$ </td></tr><tr><td>MAP</td><td>-2.6339 ± 0.2131</td><td>0.1331 ± 0.0896</td><td>0.2267 ± 0.1564</td><td>524.7643 ± 86.7733</td><td>18.1418 ± 1.4578</td><td>0.9020 ± 0.0162</td></tr><tr><td>DE</td><td>-2.8330 ± 0.0085</td><td>0.0654 ± 0.0044</td><td>0.1058 ± 0.0068</td><td>493.4771 ± 6.1154</td><td>17.6894 ± 0.0907</td><td>0.9078 ± 0.0011</td></tr><tr><td>MC Dropout</td><td>-2.3574 ± 0.4713</td><td>0.1421 ± 0.1027</td><td>0.2537 ± 0.1963</td><td>1414.1550 ± 1809.8815</td><td>26.3602 ± 16.5030</td><td>0.7358 ± 0.3381</td></tr><tr><td>MFVI</td><td>-2.8593 ± 0.0238</td><td>0.0493 ± 0.0071</td><td>0.0856 ± 0.0111</td><td>486.7107 ± 26.2902</td><td>17.5233 ± 0.4535</td><td>0.9091 ± 0.0049</td></tr><tr><td>SGLD</td><td>-2.4741 ± 0.0271</td><td>0.2036 ± 0.0080</td><td>0.3630 ± 0.0128</td><td>541.3067 ± 7.5517</td><td>18.5968 ± 0.1380</td><td>0.8989 ± 0.0014</td></tr><tr><td>SVGD</td><td>-2.3324 ± 0.0128</td><td>0.1111 ± 0.0215</td><td>0.2046 ± 0.0482</td><td>1001.8524 ± 83.1500</td><td>25.6316 ± 1.1111</td><td>0.8128 ± 0.0155</td></tr><tr><td>DiffUQ</td><td>-2.8741 ± 0.0021</td><td>0.0429 ± 0.0009</td><td>0.0716 ± 0.0015</td><td>477.8259 ± 2.1541</td><td>17.3031 ± 0.0379</td><td>0.9107 ± 0.0004</td></tr></table>

$n = 6 4$ posterior samples are used for all Bayesian methods, except MAP. Values are reported as mean ± standard deviation over five runs. Bold indicates the best result.

3) Results: Table I compares the performance of DiffUQ with other uncertainty quantification methods on the Ramanbased PAA soft sensor task. We report both uncertainty quality metrics (NLL, ECE, MCE) and accuracy-oriented metrics $( R ^ { 2 }$ , MSE, MAE). DiffUQ consistently outperforms alternative methods: it achieves the lowest NLL, ECE, and MCE, indicating well-calibrated predictive distributions, while simultaneously attaining the best $R ^ { 2 }$ , MSE, and MAE, reflecting superior point prediction quality. Although MFVI assumes a factorized Gaussian variational family, this structural bias appears mild in this relatively simple task; in contrast, samplingbased methods may suffer from finite-sample effects and mixing inefficiencies, leading to inferior empirical performance.

The case of the Raman-based PAA soft sensor highlights that our DiffUQ, when applied to a linear model, yields highly competitive performance.

D. Real-World Case Study: High-Low Transformer Unit of Ammonia Synthesis Process

![](images/876467d25242e8cb07f7736e5f333f5dc784fd3cf4dd73e4a672b6b23c8d63f3.jpg)  
Fig. 4: High-Low Transformer unit from an ammonia synthesis process.

This section uses the residual carbon monoxide (CO) concentration prediction task from the high-low transformer (HLT) unit of an ammonia synthesis process as a real-world case study to demonstrate the value of DiffUQ. Unlike using

Raman spectra as secondary variables x, predicting residual CO concentrations depends on dynamic process variables and involves nonlinear relationships. Thus, this process model requires a more complex neural network architecture.

1) Task Description: The ammonia synthesis process serves as a representative industrial benchmark: it is operated at large scale under extreme conditions, involves strongly coupled and nonlinear unit operations, and faces safety-critical and economically vital constraints. The key process variables (e.g., residual CO concentration of HLT unit) are difficult to measure online, motivating the use of data-driven models.

The dataset employed in this case study originates from the high-low transformer (HLT) unit of a full-scale ammonia synthesis plant, as illustrated in Fig. 4. CO is a severe poison for ammonia synthesis catalysts, and excessive CO also leads to unnecessary $\mathrm { H _ { 2 } }$ consumption during downstream methanation. Controlling the CO concentration at the outlet of the HLT unit is crucial to protect the catalyst and maintain the efficiency of the process. However, this key quality variable is not directly available via online measurement but is instead obtained offline through gas chromatography, hindering the timely intervention.

The objective is to develop a process model capable of estimating the residual CO concentration in real time, using the available process measurements, thereby eliminating the reliance on delayed laboratory assay results. Moreover, credible prediction intervals are as operationally important as the predicted residual CO value itself, as they guide riskaware operational decision making about timely intervention and enhance process safety. The collected data contains 26 continuously monitored operational measurements, including flow rates, inlet gas compositions, temperatures, pressures, and liquid levels, as well as one key variable measured offline: the residual CO concentration at the unit outlet. To account for temporal dependencies within the process, the dataset is split into training and testing subsets based on the sampling order, with the initial 80% of continuous samples for training and the final 20% for evaluation.

2) Probabilistic Model Structure: The probabilistic regression model in this case study follows (10), where $\mathrm { N N } _ { \theta _ { 1 } }$ is a 3-layer MLP with a hidden layer of 32 neurons, and $\mathrm { N N } _ { \theta _ { 2 } }$ is a 4-layer MLP with 2 hidden layers of width 4 and 2.

3) Main Results: Table II reports the uncertainty quality and accuracy of our method against the reference methods.

TABLE II: Uncertainty qualities and accuracy of the HLT residual CO concentration model based on our method compared with commonly used UQ approaches in industrial models.

<table><tr><td rowspan="2">Method</td><td rowspan="2">n</td><td colspan="3">Uncertainty Quality</td><td colspan="3">Accuracy</td></tr><tr><td>NLL</td><td>ECE</td><td>MCE</td><td>MSE</td><td>MAE</td><td> $R^2$ </td></tr><tr><td>MAP</td><td>1</td><td>-0.2729 ± 0.0841</td><td>0.0321 ± 0.0269</td><td>0.0649 ± 0.0469</td><td>0.0330 ± 0.0034</td><td>0.1440 ± 0.0079</td><td>0.9430 ± 0.0059</td></tr><tr><td rowspan="3">DE</td><td>4</td><td>-0.3363 ± 0.0216</td><td>0.0210 ± 0.0104</td><td>0.0404 ± 0.0156</td><td>0.0298 ± 0.0015</td><td>0.1365 ± 0.0040</td><td>0.9472 ± 0.0026</td></tr><tr><td>64</td><td>-0.3399 ± 0.0204</td><td>0.0227 ± 0.0101</td><td>0.0450 ± 0.0210</td><td>0.0295 ± 0.0015</td><td>0.1363 ± 0.0037</td><td>0.9489 ± 0.0036</td></tr><tr><td>128</td><td>-0.3290 ± 0.0387</td><td>0.0229 ± 0.0171</td><td>0.0480 ± 0.0303</td><td>0.0301 ± 0.0028</td><td>0.1376 ± 0.0063</td><td>0.9479 ± 0.0039</td></tr><tr><td rowspan="3">MC Dropout</td><td>4</td><td>-0.0038 ± 0.0100</td><td>0.0758 ± 0.0017</td><td>0.1364 ± 0.0049</td><td>0.0602 ± 0.0017</td><td>0.1920 ± 0.0021</td><td>0.8959 ± 0.0029</td></tr><tr><td>64</td><td>-0.0296 ± 0.0019</td><td>0.1059 ± 0.0010</td><td>0.1847 ± 0.0038</td><td>0.0499 ± 0.0002</td><td>0.1769 ± 0.0005</td><td>0.9138 ± 0.0004</td></tr><tr><td>128</td><td>-0.0291 ± 0.0004</td><td>0.1090 ± 0.0005</td><td>0.1881 ± 0.0019</td><td>0.0498 ± 0.0001</td><td>0.1766 ± 0.0002</td><td>0.9138 ± 0.0001</td></tr><tr><td rowspan="3">MFVI</td><td>4</td><td>-0.0592 ± 0.1246</td><td>0.0910 ± 0.0201</td><td>0.1614 ± 0.0398</td><td>0.0379 ± 0.0044</td><td>0.1557 ± 0.0103</td><td>0.9344 ± 0.0075</td></tr><tr><td>64</td><td>-0.0611 ± 0.1247</td><td>0.0906 ± 0.0203</td><td>0.1608 ± 0.0400</td><td>0.0379 ± 0.0043</td><td>0.1556 ± 0.0102</td><td>0.9345 ± 0.0075</td></tr><tr><td>128</td><td>-0.0614 ± 0.1249</td><td>0.0905 ± 0.0203</td><td>0.1609 ± 0.0398</td><td>0.0379 ± 0.0043</td><td>0.1556 ± 0.0102</td><td>0.9345 ± 0.0075</td></tr><tr><td rowspan="3">SGLD</td><td>4</td><td>-0.3666 ± 0.0094</td><td>0.0186 ± 0.0053</td><td>0.0376 ± 0.0084</td><td>0.0280 ± 0.0006</td><td>0.1325 ± 0.0015</td><td>0.9515 ± 0.0010</td></tr><tr><td>64</td><td>-0.3734 ± 0.0032</td><td>0.0199 ± 0.0009</td><td>0.0412 ± 0.0012</td><td>0.0276 ± 0.0002</td><td>0.1315 ± 0.0005</td><td>0.9522 ± 0.0003</td></tr><tr><td>128</td><td>-0.3714 ± 0.0038</td><td>0.0195 ± 0.0011</td><td>0.0401 ± 0.0007</td><td>0.0278 ± 0.0002</td><td>0.1319 ± 0.0006</td><td>0.9520 ± 0.0004</td></tr><tr><td rowspan="3">SVG D</td><td>4</td><td>-0.3516 ± 0.0157</td><td>0.0063 ± 0.0035</td><td>0.0132 ± 0.0058</td><td>0.0290 ± 0.0009</td><td>0.1347 ± 0.0024</td><td>0.9498 ± 0.0016</td></tr><tr><td>64</td><td>-0.3697 ± 0.0092</td><td>0.0134 ± 0.0027</td><td>0.0272 ± 0.0057</td><td>0.0279 ± 0.0005</td><td>0.1321 ± 0.0014</td><td>0.9517 ± 0.0009</td></tr><tr><td>128</td><td>-0.3683 ± 0.0114</td><td>0.0125 ± 0.0040</td><td>0.0249 ± 0.0068</td><td>0.0280 ± 0.0007</td><td>0.1323 ± 0.0017</td><td>0.9515 ± 0.0011</td></tr><tr><td rowspan="3">DiffUQ</td><td>4</td><td>-0.4262 ± 0.0073</td><td>0.0088 ± 0.0028</td><td>0.0237 ± 0.0034</td><td>0.0250 ± 0.0003</td><td>0.1234 ± 0.0010</td><td>0.9568 ± 0.0006</td></tr><tr><td>64</td><td>-0.4242 ± 0.0064</td><td>0.0081 ± 0.0011</td><td>0.0233 ± 0.0010</td><td>0.0251 ± 0.0003</td><td>0.1237 ± 0.0009</td><td>0.9566 ± 0.0005</td></tr><tr><td>128</td><td>-0.4266 ± 0.0055</td><td>0.0086 ± 0.0022</td><td>0.0238 ± 0.0021</td><td>0.0250 ± 0.0003</td><td>0.1234 ± 0.0007</td><td>0.9568 ± 0.0004</td></tr></table>

Values are reported as ‘mean ± standard deviation’ over five runs. Bold indicates the best result, and underline indicates the second best.

Except for MAP, which only finds a single optimal parameter vector, all approximate inference methods are evaluated with different sampling sizes of 4, 64, and 128.

In comparison with commonly used approximate inference methods, DiffUQ achieves consistently superior uncertainty metrics across different sample sizes. An exception is observed for SVGD at $n \ = \ 4 .$ , where calibration is slightly better. However, this is accompanied by reduced predictive accuracy $( R ^ { 2 } / \mathrm { M S E } / \mathrm { M A E } )$ and a worse NLL. These results suggest that the improved calibration of SVGD in this setting may be associated with a trade-off in predictive precision. In contrast, DiffUQ maintains a more balanced performance across both accuracy and calibration metrics, as reflected by its consistently lower NLL even in the small-sample regime.

rather than sample quantity. The generated samples are highly informative, reflecting the amortized nature of the proposed diffusion-based inference procedure. Specifically, extensive SDE simulations during training enable thorough exploration of the posterior support, with the resulting structural information, such as multimodality and posterior geometry, encoded into the learned drift network. As a consequence, inferencetime samples are not obtained through unguided random exploration, but are instead explicitly guided toward highprobability regions of the posterior. This separation between posterior exploration and inference-time sampling allows DiffUQ to achieve accurate uncertainty estimation with a small number of forward predictions, even in the presence of complex posterior structures.

In terms of predictive accuracy, our method also consistently outperforms the baselines. Furthermore, under limited sample budgets, our approach maintains stable performance, whereas MFVI, SGLD, and SVGD all suffer from increased NLL. Notably, MFVI and MC dropout perform even worse than MAP, underscoring that restrictive structural assumptions on the posterior distribution (e.g., factorized Gaussian of MFVI) can be detrimental to the uncertainty-aware modeling.

In addition, the results obtained by our method exhibit overall smaller standard deviations compared with the baselines, which highlights the improved stability and robustness of our approach.

4) Hyperparameter Sensitivity Analysis: This part is to demonstrate the robustness of DiffUQ.

Number of Samples n: Results with different sample sizes are reported in Table II. Even with as few as $ { n _ { \mathrm { ~  ~ } } } =  { \mathrm { ~  ~ } } 4$ posterior samples, DiffUQ maintains comparable performance, exhibiting only a slight increase in variance. This behavior should be interpreted from the perspective of sample efficiency

Discretization Step Size ∆t: To assess the sensitivity of our method to hyperparameters, we first examined the effect of the discretization. Specifically, we varied the step size of EM discretization during the training phase, selecting $\Delta t \in \{ 0 . 0 2 5 , 0 . 0 5 , 0 . 0 7 5 , 0 . 1 \}$ . The results in Fig. 5a indicate that predictive accuracy is only slightly reduced when $\Delta t = 0 . 1$ , while calibration metrics remain largely unaffected. This demonstrates that our method is robust to the choice of discretization step size. From a theoretical perspective, this behavior is expected. The optimization problem is defined over continuous-time trajectories, while ∆t only affects the numerical approximation of the SDE. Within a reasonable discretization regime, varying ∆t does not change the target objective but merely adjusts the resolution of the simulated paths.

Diffusion Coefficient γ: We further conducted a sensitivity analysis with respect to the diffusion coefficient by varying its value across three orders of magnitude. The results in Fig. 5b consistently indicate that this hyperparameter has no meaningful influence on predictive accuracy or uncertainty quantification. Only when the diffusion coefficient was increased to $1 0 ^ { - 3 }$ did we observe a minor degradation in calibration, while the other metrics remained unaffected. This robustness suggests that the method is relatively insensitive to the precise choice of the diffusion coefficient. γ in our framework defines the volatility of the reference process. As requested by the theoretical analysis, these properties are not coincidental but stem from the stochastic control formulation. The optimization objective (9) drives a neural network to learn a drift function u(t, θ) that adaptively compensates for the noise level to reach the target posterior. This creates a self-regulating mechanism where the learned control effort balances the diffusion intensity, ensuring consistent posterior sampling across orders of magnitude of γ.

![](images/9a48e54933c67ced3f58652912b22d948e5edbc68821993b0e817b9c27c1ed28.jpg)

![](images/bdcd102f2a0b01b38cc6a31956965b7a3ee014176e52fb62116228601df192a2.jpg)

![](images/9a0c50c7c9e72d794b182f881202e1fa67832f1738210acbf2585755dd1815e4.jpg)

![](images/6fb87a642f383d9e2d174eb4a4496c7faf53f23706504a1d13e60ae7319e34dd.jpg)

![](images/dd906acd90df4cd1bcb73e83154bf49622f940acd1cd508b2dc54207150f5fdf.jpg)  
(a) Effect of different ∆t during training phase.

![](images/5faea4ee1eaddabdb663b19d13608eeb25455aee2ff25ae0a71f5faca8a4a1c4.jpg)

![](images/2b4f13c59eac1c7b4980068f29e1d3140d46e40e5f3e3c23d8d1bf8c839d98c0.jpg)

![](images/b3fc5af8597b062293b711af3236db45aa76d327059e8cac04e7f055c59a6854.jpg)

![](images/7cfb1650c4bb536dfc10d8062efaea94985b2b7c3dfd5e8e52f06bb7bd9b3a2b.jpg)

![](images/aa8880343a2d4da7aaab95ce3197a593b645fcc88273e490ac16a47d727d1ba3.jpg)

![](images/830f50dacdfe3701b347387b2f93cad16ef95d93767fc04f555636a7d6f1ce9e.jpg)  
(b) Effect of different diffusion coefficient γ.

![](images/0f329fc05edbd606e734e497527c35dd5f83013903beaaaabf6ff89c64a393b0.jpg)  
Fig. 5: Sensitivity Analysis. The shaded area indicates ±1 standard deviation over 5 independent runs.

5) Training Dynamics: To provide a clearer understanding of DiffUQ’s optimization behavior, we present the progression of its composite loss over multiple training runs (Fig. 6). Accoring to (9), the total loss compromises two competing terms: the running cost <sup>E</sup> $\begin{array} { r } { \left| \int _ { 0 } ^ { 1 } \frac { 1 } { 2 \gamma } \| u ( t , \dot { \theta } _ { t } ) \| ^ { 2 } \mathrm { d } t \right| } \end{array}$ , and the terminal cost $\begin{array} { r } { \mathbb { E } \left\lceil - \log \frac { \pi _ { 1 } ( \theta _ { 1 } ) } { \mathcal { N } ( \theta _ { 1 } | 0 , \gamma { \bf I } _ { d } ) } \right\rceil } \end{array}$

Overall, the optimization process demonstrates stable and reproducible behavior across runs, with a gradual and monotonic balance established between the terminal and control objectives. As shown in Fig. 6b, varying γ mainly changes the relative scale of the running and terminal costs, while the overall convergence behavior remains smooth and consistent across runs. Similarly, for $\Delta t \in \lbrace 0 . 0 2 5 , 0 . 0 5 , 0 . 0 7 5 , 0 . 1 \rbrace$ , the optimization trajectories (Fig. 6a) exhibit stable and largely overlapping convergence patterns, with no noticeable oscillation or instability. These results indicate that DiffUQ maintains smooth and robust optimization dynamics across a broad range of discretization and diffusion hyperparameters.

The smooth optimization dynamics can be explained by the entropic regularization intrinsically induced by the diffusion process [18], [29], [30]. Mathematically, the diffusion term acts as a smoothing operator on the probability density, convexifying the optimization landscape and preventing sharp minima. Furthermore, by relaxing the hard boundary constraints into a soft terminal penalty (Proposition 2), the framework avoids the numerical instabilities typical of constrained transport problems, resulting in a stable and monotonic convergence trajectory.

In addition, we notice that the total loss appears to stabilize after a few thousand steps; however, its decomposition indicates that training continues to evolve, revealing a trade-off relationship. This behavior suggests that convergence should be evaluated by jointly monitoring both objectives, which together delineate an empirical Pareto front characterizing the balance between accuracy and regularization.

![](images/0f3b098309b59b9e50b5ce0c55e61d490c54de9dde1a2d763e641bb676768a3d.jpg)  
(a) Variation of ∆t

![](images/e1889275c58612843e1941e496ab5970709a7574e738cc66b03d185c1744584c.jpg)  
(b) Variation of γ  
Fig. 6: Training Curves

## V. CONCLUSION

This work introduced DiffUQ, a new approach for uncertainty quantification in industrial models based on a diffusion sampler. By leveraging elegant connections between sampling and optimal control, the method avoids strong assumptions on the posterior form that compromise fidelity, and can achieve robust representation of complex posterior landscapes through sufficient exploration enabled by SDE simulation during training. These properties ensure reliable posterior sampling that directly translates into well-calibrated predictive uncertainty. Experiments on toy distributions, a Raman-based PAA benchmark, and a process modeling task from a real-world ammonia synthesis imply that these theoretical advantages consistently yield improvements in both calibration and accuracy over existing baselines, without relying on post-hoc adjustments.

In addition, DiffUQ also exhibits robustness to hyperparameters and shows well-behaved training dynamics, which are particularly valuable in practical industrial applications where ease of deployment is of primary concern. Overall, this work highlights the potential of diffusion samplers as a principled and scalable family of methods for advancing uncertainty quantification in complex industrial data-driven modeling. Future research may further extend this framework by developing more efficient training strategies and exploring alternative diffusion formulations. Specifically, investigating other advanced solvers within the diffusion family, such as Denoising Diffusion Samplers (DDS) [22] or Adjoint Sampling (AS) [23], could offer further potential to optimize the training dynamics and sampling efficiency for specific industrial scenarios.

## APPENDIX

## A. On Drift Network Capacity

![](images/430aa60eeb74608f2492561c1d294adae709fd276c9ecacac7866b1dbf296b14.jpg)  
(a) NLL vs. Width

![](images/b1e0f217ce366e4718aa729d45f1353ee4bbf14b364730762aec27c9cd1b8fdb.jpg)

![](images/2c2f3bee805bd457274c40fd5ad02d98e060f1b9b45d10c0248d18f53710fd63.jpg)

(b) ECE vs. Width  
![](images/64b46d7da45f2acacf63d093bdd4f00380adb00dd07c50f225ca6a39963cb15d.jpg)  
(c) NLL vs. Depth  
(d) ECE vs. Depth  
Fig. 7: Impact of Drift Network Size

1) Drift Network Capacity and Discretization Error: From a theoretical perspective, existing results, e.g., Theorem 2 in [21], ensure that a sufficiently expressive neural drift can approximate the target distribution in KL to arbitrary accuracy under mild regularity assumptions. However, these are existence results and do not account for finite-step numerical discretization. In practice, the learned drift is simulated using a fixed-step Euler–Maruyama scheme. Under global Lipschitz and linear growth conditions, its strong error satisfies

$$
\mathbb {E} \Big [ \sup _ {s \leq T} \| X _ {s} - \hat {X} _ {s} \| ^ {2} \Big ] \leq C (L, T) h,
$$

where the constant $C ( L , T )$ depends on the regularity of the drift via a Gronwall argument (e.g., Theorem 10.2.2 in [40]).¨ While the convergence order is governed by the step size h, the prefactor is sensitive to the drift’s Jacobian magnitude. Increasing network capacity expands the hypothesis space and can lead to drifts with larger Jacobians, making fixedstep linear integration less accurate and effectively enlarging $C ( L , T )$

As a result, under a fixed discretization budget, larger networks do not necessarily improve sampling accuracy. Instead, there exists a practically relevant capacity regime in which the drift is sufficiently expressive, while further increasing network size mainly amplifies discretization error rather than improving posterior approximation.

This observation is supported by the ablation results on the CO soft sensor case study (Fig. 7). Empirically, we observe a non-monotonic dependence on network capacity: small networks suffer from insufficient expressivity, whereas overly large networks offer no further benefits and can even degrade both likelihood and calibration under a fixed step size, likely due to amplified discretization effects.

2) Other drift network parameterization: Exploring more suitable drift architectures and parameterizations remains an important direction for future work. [21] augment the time variable t with Fourier features or sinusoidal embeddings to enhance temporal representation. However, our preliminary experiments did not show clear improvements.

## REFERENCES

[1] P. Kadlec, B. Gabrys, and S. Strandt, “Data-driven soft sensors in the process industry,” Computers & chemical engineering, vol. 33, no. 4, pp. 795–814, 2009.

[2] Y. Ma, Z. Chen, Z. Yang, X. Zhang, and Z. Song, “Heat equation Stein variational ensemble: Rethinking and advancing uncertainty-aware soft sensor modeling,” IEEE Transactions on Industrial Informatics, 2024.

[3] M. J. Kochenderfer, Decision making under uncertainty: theory and application. MIT press, 2015.

[4] L. P. Hansen and T. J. Sargent, “Robust control and model uncertainty,” American Economic Review, vol. 91, no. 2, pp. 60–66, 2001.

[5] N. V. Sahinidis, “Optimization under uncertainty: state-of-the-art and opportunities,” Computers & chemical engineering, vol. 28, no. 6-7, pp. 971–983, 2004.

[6] Y. Gal, R. Islam, and Z. Ghahramani, “Deep Bayesian active learning with image data,” in International conference on machine learning. PMLR, 2017, pp. 1183–1192.

[7] D. J. MacKay, “A practical Bayesian framework for backpropagation networks,” Neural computation, vol. 4, no. 3, pp. 448–472, 1992.

[8] R. M. Neal, “Bayesian leaning for neural networks,” 1996.

[9] Z. Ge, “Process data analytics via probabilistic latent variable models: A tutorial review,” Industrial & Engineering Chemistry Research, vol. 57, no. 38, pp. 12 646–12 661, 2018.

[10] Q. Sun and Z. Ge, “A survey on deep learning for data-driven soft sensors,” IEEE Transactions on Industrial Informatics, vol. 17, no. 9, pp. 5853–5866, 2021.

[11] C. Liu, J. Zhuo, P. Cheng, R. Zhang, and J. Zhu, “Understanding and accelerating particle-based variational inference,” in International Conference on Machine Learning. PMLR, 2019, pp. 4082–4092.

[12] F. D’Angelo and V. Fortuin, “Annealed Stein variational gradient descent,” in The Third Symposium on Advances in Approximate Bayesian Inference, 2021.

[13] J. Yang, Y. Peng, J. Xie, and P. Wang, “Remaining useful life prediction method for bearings based on lstm with uncertainty quantification,” Sensors, vol. 22, no. 12, p. 4549, 2022.

[14] L. Cao, H. Zhang, Z. Meng, and X. Wang, “A parallel GRU with dualstage attention mechanism model integrating uncertainty quantification for probabilistic rul prediction of wind turbine bearings,” Reliability Engineering & System Safety, vol. 235, p. 109197, 2023.

[15] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in international conference on machine learning. PMLR, 2016, pp. 1050–1059.

[16] M. Lee, J. Bae, and S. B. Kim, “Uncertainty-aware soft sensor using Bayesian recurrent neural networks,” Advanced Engineering Informatics, vol. 50, p. 101434, 2021.

[17] Y. Zhang, O. D. Akyildiz, T. Damoulas, and S. Sabanis, “Nonasymptotic<sup>¨</sup> estimates for stochastic gradient Langevin dynamics under local conditions in nonconvex optimization,” Applied Mathematics & Optimization, vol. 87, no. 2, p. 25, 2023.

[18] C. Leonard, “A survey of the Schr´ odinger problem and some of its con-¨ nections with optimal transport,” Discrete and Continuous Dynamical Systems, vol. 34, no. 4, pp. 1533–1574, 2013.

[19] S. Kullback, “Probability densities with given marginals,” The Annals of Mathematical Statistics, vol. 39, no. 4, pp. 1236–1243, 1968.

[20] J. Huang, Y. Jiao, L. Kang, X. Liao, J. Liu, and Y. Liu, “Schrodinger-¨ Follmer sampler,”¨ IEEE Transactions on Information Theory, vol. 71, no. 2, pp. 1283–1299, 2025.

[21] Q. Zhang and Y. Chen, “Path integral sampler: A stochastic control approach for sampling,” in The Tenth International Conference on Learning Representations, ICLR, 2022.

[22] F. Vargas, W. S. Grathwohl, and A. Doucet, “Denoising diffusion samplers,” in The Eleventh International Conference on Learning Representations, ICLR, 2023.

[23] A. J. Havens, B. K. Miller, B. Yan, C. Domingo-Enrich, A. Sriram, D. S. Levine, B. M. Wood, B. Hu, B. Amos, B. Karrer et al., “Adjoint sampling: Highly scalable diffusion samplers via adjoint matching,” in International Conference on Machine Learning. PMLR, 2025, pp. 22 204–22 237.

[24] G.-H. Liu, J. Choi, Y. Chen, B. K. Miller, and R. T. Chen, “Adjoint Schrodinger bridge sampler,” in ¨ The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[25] V. Kuleshov, N. Fenner, and S. Ermon, “Accurate uncertainties for deep learning using calibrated regression,” in International conference on machine learning. PMLR, 2018, pp. 2796–2804.

[26] A. Kendall and Y. Gal, “What uncertainties do we need in Bayesian deep learning for computer vision?” Advances in neural information processing systems, vol. 30, 2017.

[27] E. Hullermeier and W. Waegeman, “Aleatoric and epistemic uncertainty¨ in machine learning: An introduction to concepts and methods,” Machine learning, vol. 110, no. 3, pp. 457–506, 2021.

[28] A. Foong, D. Burt, Y. Li, and R. Turner, “On the expressiveness of approximate inference in Bayesian neural networks,” Advances in Neural Information Processing Systems, vol. 33, pp. 15 897–15 908, 2020.

[29] Y. Chen, T. T. Georgiou, and M. Pavon, “On the relation between optimal transport and Schrodinger bridges: A stochastic control viewpoint,”¨ Journal of Optimization Theory and Applications, vol. 169, no. 2, pp. 671–691, 2016.

[30] B. Tzen and M. Raginsky, “Theoretical guarantees for sampling and inference in generative models with latent diffusions,” in Conference on Learning Theory. PMLR, 2019, pp. 3084–3114.

[31] F. Vargas, A. Ovsianas, D. Fernandes, M. Girolami, N. D. Lawrence, and N. Nusken, “Bayesian learning via neural Schr¨ odinger–F¨ ollmer flows,”¨ Statistics and Computing, vol. 33, no. 1, p. 3, 2023.

[32] S. Kay, H. Kay, M. Mowbray, A. Lane, C. Mendoza, P. Martin, and D. Zhang, “Integrating autoencoder and heteroscedastic noise neural networks for the batch process soft-sensor design,” Industrial & Engineering Chemistry Research, vol. 61, no. 36, pp. 13 559–13 569, 2022.

[33] W. Xu, R. T. Chen, X. Li, and D. Duvenaud, “Infinitely deep Bayesian neural networks with stochastic differential equations,” in International Conference on Artificial Intelligence and Statistics. PMLR, 2022, pp. 721–738.

[34] X. Li, T.-K. L. Wong, R. T. Q. Chen, and D. Duvenaud, “Scalable gradients for stochastic differential equations,” International Conference on Artificial Intelligence and Statistics, 2020.

[35] P. Kidger, J. Foster, X. Li, H. Oberhauser, and T. Lyons, “Neural SDEs as infinite-dimensional GANs,” International Conference on Machine Learning, 2021.

[36] M. P. Naeini, G. Cooper, and M. Hauskrecht, “Obtaining well calibrated probabilities using Bayesian binning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 29, no. 1, 2015.

[37] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” Advances in neural information processing systems, vol. 30, 2017.

[38] M. Welling and Y. W. Teh, “Bayesian learning via stochastic gradient Langevin dynamics,” in Proceedings of the 28th international conference on machine learning (ICML-11), 2011, pp. 681–688.

[39] S. Goldrick, C. A. Duran-Villalobos, K. Jankauskas, D. Lovett, S. S. Farid, and B. Lennox, “Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process,” Computers & Chemical Engineering, vol. 130, p. 106471, 2019.

[40] P. E. Kloeden and E. Platen, Numerical Solution of Stochastic Differential Equations, ser. Applications of Mathematics: Stochastic Modelling and Applied Probability. Berlin, Heidelberg: Springer-Verlag, 1992, vol. 23.