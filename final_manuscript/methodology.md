# 3. Materials and Methods

## 3.1 Datasets
Five open-access datasets are used, chosen to span simulated versus real operation and continuous versus batch processes (Table 1).

- **Tennessee Eastman Process, extended (primary).** A simulated closed-loop chemical plant with twenty programmed faults plus normal operation, 52 measured and manipulated variables, and independent training and testing simulation campaigns (Downs and Vogel, 1993; Rieth et al., 2017). We use the canonical protocol: models are trained on the *training* simulations and evaluated on the separate *testing* simulations, in which each fault is introduced at sample 160.
- **PRONTO multiphase-flow facility (real plant).** An industrial-scale air–water–oil pilot rig with seeded fault conditions (air blockage, air leakage, diverted flow) and real process noise (Stief et al., 2019). Seventeen continuous process variables are used; the *slugging* regime is excluded as an operating condition rather than a seeded fault.
- **IndPenSim.** A simulated industrial-scale fed-batch penicillin fermentation providing fault-flagged operation (Goldrick et al., 2015); used for a sample-level detection check.
- **Debutanizer and Sulphur Recovery Unit (SRU).** A real refinery soft-sensor benchmark (Fortuna et al., 2007); used for quality-variable regression.
- **Steel Plates Faults.** A seven-class metallurgical fault set used as a cross-domain generalization probe.

## 3.2 Windowing and preprocessing
Each multivariate record is segmented into overlapping windows of length $W$ and stride $S$:

$$
\mathbf{X}_t = [\,\mathbf{x}_t,\, \mathbf{x}_{t+1},\, \dots,\, \mathbf{x}_{t+W-1}\,]^{\top} \in \mathbb{R}^{W \times d},
$$

with $W=50$, $S=25$, and $d=52$ for Tennessee Eastman. For faulty testing runs, samples before the fault onset are discarded so that a window's label reflects the active fault. Training and validation windows are drawn from the training simulations and split **by simulation run** using grouped shuffling, so that no run contributes windows to more than one partition and adjacent windows cannot leak across the split. Features are standardized with statistics estimated on the training partition only,

$$
\tilde{x}^{(j)} = \frac{x^{(j)} - \mu_j}{\sigma_j + \varepsilon}, \qquad \varepsilon = 10^{-8},
$$

and the same $\mu_j,\sigma_j$ are applied unchanged to validation and test.

## 3.3 Backbone network
The backbone is a one-dimensional convolutional network operating on a window $\mathbf{X}\in\mathbb{R}^{W\times d}$ transposed to $d$ channels of length $W$. Two convolutional blocks are followed by global average pooling, dropout, and a linear head:

$$
\mathbf{h} = \operatorname{GAP}\!\big(\phi(\mathbf{W}_2 * \phi(\operatorname{BN}(\mathbf{W}_1 * \mathbf{X}^{\top})))\big) \in \mathbb{R}^{128},
\qquad
\mathbf{z} = \mathbf{W}_o \mathbf{h} + \mathbf{b}_o,
$$

where $\phi(\cdot)=\max(0,\cdot)$ is the ReLU activation, $\operatorname{BN}$ is batch normalization, $*$ denotes 1-D convolution with kernel size 3, $\mathbf{h}$ is the penultimate feature vector, and $\mathbf{z}\in\mathbb{R}^{C}$ are the class logits. Class posteriors follow from the softmax,

$$
p_c = \frac{\exp(z_c)}{\sum_{k=1}^{C}\exp(z_k)}.
$$

The identical architecture is used for Tennessee Eastman diagnosis and for PRONTO detection; only the input channel count and output width change.

## 3.4 Training objective: focal loss for incipient faults
Because incipient faults are weak and, under deployment conditions, rare, we train with the focal loss (Lin et al., 2017), which down-weights easy examples and concentrates learning on hard ones:

$$
\mathcal{L}_{\text{focal}} = -\frac{1}{N}\sum_{i=1}^{N} \alpha_{y_i}\,(1-p_{i,y_i})^{\gamma}\,\log p_{i,y_i},
\qquad \gamma = 2,
$$

where $p_{i,y_i}$ is the predicted probability of the true class of sample $i$. When class weights are used, they follow a square-root-balanced scheme that lifts minority faults without collapsing the majority (normal) class:

$$
\alpha_c = \sqrt{\frac{N}{C\,\max(n_c,1)}},
$$

with $n_c$ the number of training windows of class $c$. Plain inverse-frequency weighting ($\alpha_c \propto 1/n_c$) was found to inflate the false-alarm rate and is reported in the ablation for contrast.

## 3.5 Calibrated detection: FDR, FAR and delay at a chosen operating point
For deployment, a diagnosis label is insufficient; an operator needs a detection rule with a known false-alarm budget. We define the fault score of a window as the probability mass not assigned to the normal class,

$$
s = 1 - p_{\text{normal}},
$$

and choose the decision threshold $\tau$ on the *validation* normal windows so that the empirical false-alarm rate equals a target $\beta$:

$$
\tau = \operatorname{Quantile}_{1-\beta}\big(\{\,s_v : y_v = \text{normal}\,\}\big), \qquad \beta = 0.05 .
$$

At test time a window is declared faulty when $s \ge \tau$. The reported metrics are the fault-detection rate, the false-alarm rate, and the mean detection delay,

$$
\text{FDR} = \frac{|\{\,s_i \ge \tau,\; y_i \in \mathcal{F}\,\}|}{|\{\,y_i \in \mathcal{F}\,\}|},
\qquad
\text{FAR} = \frac{|\{\,s_i \ge \tau,\; y_i = \text{normal}\,\}|}{|\{\,y_i = \text{normal}\,\}|},
$$

$$
\Delta_{\text{delay}} = \frac{3\,\text{min}}{|\mathcal{R}|}\sum_{r\in\mathcal{R}} \big(t^{\star}_r - t_{\text{onset}}\big),
$$

where $\mathcal{F}$ is the set of known faults, $t^{\star}_r$ is the first sample in run $r$ whose window crosses $\tau$, $t_{\text{onset}}=160$, and each Tennessee Eastman sample spans three minutes.

## 3.6 Open-set recognition: feature-space Mahalanobis score
Unknown faults are held out of training entirely (faults 16, 17, 18) and appear only at test time. Following Lee et al. (2018), we fit a class-conditional Gaussian in the penultimate feature space using the known-class training features: per-class means $\boldsymbol{\mu}_c$ and a shared covariance $\boldsymbol{\Sigma}$. A window's novelty score is the Mahalanobis distance to the nearest class centroid,

$$
m(\mathbf{h}) = \min_{c}\; (\mathbf{h}-\boldsymbol{\mu}_c)^{\top}\,\boldsymbol{\Sigma}^{-1}\,(\mathbf{h}-\boldsymbol{\mu}_c),
$$

and open-set performance is the area under the ROC curve of $m(\cdot)$ separating held-out unknown faults from known classes. For comparison we also evaluate three softmax-derived scores computed from the same model: maximum softmax probability $-\max_c p_c$, predictive entropy $-\sum_c p_c \log p_c$, and the energy score $-\log\sum_c \exp(z_c)$ (Liu et al., 2020).

## 3.7 Interpretability: input-gradient attribution
Attribution uses the gradient of the predicted-class logit with respect to the input window (Simonyan et al., 2014), averaged over samples and time to yield a per-variable importance:

$$
a_j = \frac{1}{N\,W}\sum_{i=1}^{N}\sum_{t=1}^{W}\left|\frac{\partial\, z_{\hat{y}_i}}{\partial\, X^{(i)}_{t,j}}\right|.
$$

Attributed variables are compared against the known process topology so that flagged sensors can be checked against the physical fault mechanism.

## 3.8 Evaluation protocol
Classification is reported with macro-averaged F1 and per-class recall, with emphasis on the incipient faults 3, 9 and 15. Detection uses FDR, FAR and delay at the calibrated operating point of §3.5. Open-set uses AUROC against held-out unknown faults. A controlled ablation isolates the contribution of the focal objective and of class weighting on the same canonical test set. The unified backbone is retrained on real PRONTO data as a within-domain detector, and the auxiliary datasets are evaluated over five random seeds with mean and standard deviation reported. Soft-sensor regression uses time-lagged inputs, reflecting the transport delay of the units, and reports RMSE and the coefficient of determination $R^2$. All deep training uses the Adam optimizer; code and preprocessing pipelines are released.
