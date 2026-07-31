Article

# A Gradient-Penalized Conditional TimeGAN Combined with Multi-Scale Importance-Aware Network for Fault Diagnosis Under Imbalanced Data

Ranyang Deng <sup>1,2</sup> , Dongning Chen <sup>1,2,</sup>\* , Chengyu Yao <sup>3</sup> , Dongbo Hu <sup>1,4</sup> , Qinggui Xian <sup>1,2</sup> and Sheng Zhang <sup>1,2</sup>

1 School of Mechanical Engineering, Yanshan University, Qinhuangdao 066004, China; rydeng@stumail.ysu.edu.cn (R.D.); 18629448551@163.com (Q.X.)

2 Hebei Provincial Key Laboratory of Heavy Machinery Fluid Power Transmission and Control, Yanshan University, Qinhuangdao 066004, China

3 Hebei Key Laboratory of Industrial Computer Control Engineering, Yanshan University, Qinhuangdao 066004, China; chyyao@ysu.edu.cn

4 Citic Heavy Industries Co., Ltd., Luoyang 471000, China

Correspondence: dnchen@ysu.edu.cn

![](images/3e9d2751d5cdddbacb7e420277e6dc5ac6f2455be1f54cd46c2dcb6c4522ed44.jpg)

Academic Editor: Andrea Cataldo

Received: 19 September 2025 Revised: 30 October 2025 Accepted: 5 November 2025 Published: 7 November 2025

## Abstract

Citation: Deng, R.; Chen, D.; Yao, C.; Hu, D.; Xian, Q.; Zhang, S. A Gradient-Penalized Conditional TimeGAN Combined with Multi-Scale Importance-Aware Network for Fault Diagnosis Under Imbalanced Data. Sensors 2025, 25, 6825. https:// doi.org/10.3390/s25226825

Copyright: © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/).

In real-world industrial settings, obtaining class-balanced fault data is often difficult. Im balanced data across categories can degrade diagnostic accuracy. Time-series Generative Adversarial Network (TimeGAN) is an effective tool for addressing one-dimensional data imbalance; however, when dealing with multiple fault categories, it faces issues such as unstable training processes and uncontrollable generation states. To address this issue, from the perspective of data augmentation and classification, a gradient-penalized Condi tional Time-series Generative Adversarial Network with a Multi-Scale Importance-aware Network (CTGAN-MSIN) is proposed in this paper. Firstly, a gradient-penalized Condi tional Time-Series Generative Adversarial Network (CTGAN) is designed to alleviate data imbalance by controllably generating high-quality fault samples. Secondly, a Multi-scale Importance-aware Network (MSIN) is constructed for fault classification. The MSIN consists of the Multi-scale Depthwise Separable Residual (MDSR) and Scale Enhanced Local Attention (SELA): the MDSR network can efficiently extract multi-scale features, while the SELA network is capable of screening out the most discriminative scale features from them. Finally, the proposed method is validated using the HUST bearing dataset and the axial piston pump dataset. The results show that under the data imbalance ratio of 15:1, the CTGAN-MSIN achieves diagnostic accuracies of 98.75% and 96.50%, respectively, on the two datasets and outperforms the comparison methods under different imbalance ratios.

Keywords: rotating machinery; fault diagnosis; data augmentation; time-series generative adversarial network

## 1. Introduction

As a core element of sophisticated industrial systems, the sustained and reliable functioning of rotating machinery is vital for maintaining both operational safety and productivity [1,2]. Typical rotating machinery, such as bearings and piston pumps, is often subjected to various pressures and loads, making it susceptible to failures like wear and fatigue, which may lead to shutdowns or even safety accidents [3,4]. In real-world industrial scenarios, rotating machinery is characterized by short operational durations under fault conditions and difficulties in obtaining fault samples, resulting in a scarcity of fault data. An imbalance in the data arises from the disproportionate representation of normal samples compared to fault samples [5]. Under data imbalance conditions, diagnostic models are often biased toward the majority class, overlooking the rare one and thus diminishing diagnosis accuracy [6]. Therefore, investigating rotating-machinery fault diagnosis amid data imbalance is of pronounced practical value

The application of data augmentation techniques can counteract data imbalance problems, leading to improved performance in fault diagnosis systems [7]. Common data augmentation techniques mainly fall into two categories: methods based on data sampling and those relying on data generation [8]. The data sampling-based methods include oversampling [9], undersampling [10], and interpolation-based techniques [11]. By undersampling the majority classes and oversampling the minority classes, the class distribution can be balanced. However, such methods suffer from limitations such as limited representational capacity and insufficient data diversity. Data generation-based methods include approaches such as Variational Autoencoder (VAE) [12] and Generative Adversarial Network (GAN) [13]. However, data generated by VAE often exhibits limited diversity and authenticity, along with the vanishing gradient problem in long sequences [14]. GAN employs an adversarial training mechanism where the generator and discriminator undergo dynamic game-theoretic optimization. This framework effectively captures the underlying data distribution characteristics, enabling the generation of synthetic data that maintains high statistical consistency with real data. GAN has now become the primary method for data augmentation.

Lately, a series of enhanced GAN-based techniques has continually surfaced, such as Auxiliary Classifier GAN (ACGAN) [15], Deep Convolutional GAN (DCGAN) [16], Conditional GAN (CGAN) [17], Wasserstein GAN (WGAN) [18], and so on. Additionally, some scholars have explored innovative approaches that combine GANs with other network architectures. For example, Chen et al. [19] proposed integrating a VAE with an attention mechanism as the generator of the GAN, thereby enhancing its attention on fault-relevant features. Li et al. [20] proposed an augmentation strategy combining an adaptive diffusion model with GAN, which enhances the diversity of fault samples. Yoon et al. [21] proposed Time-series Generative Adversarial Network (TimeGAN), a novel generative framework specifically designed for synthesizing realistic one-dimensional timeseries data, as opposed to generating images directly. This model synergistically combines the strengths of GAN with supervised learning, enabling the generation of high-fidelity temporal data while preserving the original time-dependent structural characteristics. Shi et al. [22] proposed an augmentation strategy for imbalanced datasets based on a Wasser stein Temporal Generative Adversarial Network with Gradient Penalty (WTGAN-GP), and validated its performance in addressing class imbalance using real-world air compressor fault data from industrial plants. Sim et al. [23] employed TimeGAN to address data imbalance issues in real-world industrial datasets and developed a CNN-LSTM hybrid network, successfully forecasting equipment that remains useful across diverse load scenarios. However, despite demonstrating advantages in time-series data generation, TimeGAN still has certain limitations: (1) Due to the dynamic adversarial training mechanism, it is prone to mode collapse or convergence difficulties. (2) As a general time-series generation framework, the TimeGAN method still faces critical challenges such as training instability and uncontrollable generation outcomes.

The design of feature extraction and classification networks is a critical component in fault diagnosis. Currently, most imbalanced fault diagnosis studies primarily focus on data augmentation techniques, while subsequent fault diagnosis tasks often rely solely on some basic benchmark methods. For instance, Ye et al. [24] developed a feature fusion deep convolutional generative adversarial network and subsequently utilized convolutional stacks and fully connected layers to achieve fault classification in aircraft engines. Wang [17] and Chen et al. [19] employed variants of GAN-based networks for data augmentation and then implemented fault classification using deep convolutional neural networks and Softmax. Deep Learning (DL) algorithms have gained broad traction in recent years for their capabilities in automatic feature extraction, multi-level feature learning, and strong generalization performance [25]. Common DL approaches include Convolutional Neural Network (CNN) [26], Recurrent Neural Network (RNN) [27], and Transformer Net work [28]. Compared with CNN, RNN and Transformer models exhibit higher architectural complexity and require substantially more time for network training. Due to the inherent multi-timescale characteristics in rotating machinery signals, the Multi-scale Convolutional Neural Network (MSCNN) has been widely adopted for analysis [29]. As the number of network layers increases and the scale of parameters expands, MSCNN-based models tend to suffer from gradient vanishing or explosion due to the excessive number of nonlinear layers. When gradient vanishing occurs, parameter updates essentially cease, and even in cases where extreme gradient problems are avoided, parameter optimization becomes progressively more difficult with increasing network depth [30]. Moreover, deeper net works inevitably bring greater complexity and longer training times. Therefore, MSCNN still faces some challenges in practical applications.

Residual Network (ResNet) offers an effective remedy for gradient vanishing and explosion in deep neural networks. Liu et al. [31] proposed a multiscale kernel-based residual CNN, incorporating residual learning modules with multiscale kernels to accomplish motor fault diagnosis. Xu et al. [32] developed a residual network architecture integrated with a multi-scale attention mechanism for diagnosing faults in variable displacement pumps. However, existing multi-scale feature extraction methods focus on expanding the broader receptive field, while neglecting the varying differences in features across different scales.

To address the aforementioned challenges, a gradient-penalized Conditional Timeseries Generative Adversarial Network (CTGAN) with a Multi-scale Importance-aware Network (MSIN) is proposed in this paper, collectively named CTGAN-MSIN. This paper’s principal contributions are summarized below:

(1) A time-series data augmentation method named the CTGAN network is proposed in this paper. It introduces Wasserstein distance with gradient penalty as the loss function, effectively mitigating the mode collapse issue commonly encountered in traditional TimeGAN network training. By injecting category labels into both the generator and discriminator, this approach effectively overcomes the limitations of uncontrollable outputs in original TimeGAN. It enables the controlled generation of multi-source fault samples, providing a precise data augmentation solution for fault diagnosis.

(2) The MDSR block is proposed to efficiently capture multi-scale feature representations. This module employs multiple branches of depthwise separable convolutions, which significantly reduce computational complexity while accurately capturing multiscale features. Additionally, MDSR incorporates a residual connection mechanism that effectively preserves the integrity of original features while mitigating gradient vanishing.

(3) To address the issue of low diagnostic accuracy caused by differences in the importance of features across different scales, a Scale Enhanced Local Attention (SELA) module is proposed. The SELA module enhances the expressive capability of features across positional and scale dimensions, enabling the model to precisely focus on the most discriminative scale features. This has significantly improved both recognition accuracy and robustness.

(4) This paper proposes the CTGAN-MSIN network from the perspectives of data augmentation and classification. CTGAN resolves the issues of unstable training and uncontrollable generation outcomes in TimeGAN, effectively accomplishing data augmentation tasks. The MSIN integrates both MDSR and SELA modules to extract comprehensive and highly discriminative features. This model effectively addresses the issue of low diagnostic accuracy in classical MSCNN, which arises from its neglect of the important differences among features at different scales. Practicality and effectiveness of the CTGAN-MSIN are validated using both the public bearing dataset and the self-collected axial piston pump dataset.

The structure of this paper is outlined below: Section 2 offers a concise overview of related theories; Section 3 elaborates on the CTGAN-MSIN methodology, while Section 4 verifies its superiority through two case studies; and Section 5 summarizes the conclusions.

## 2. Preliminaries

## 2.1. Time-Series Generative Adversarial Network

The Time-series Generative Adversarial Network (TimeGAN) is a temporal generative adversarial network proposed by Yoon et al. in 2019 [21]. By integrating adversarial training with supervised learning, TimeGAN effectively generates high-fidelity synthetic data that preserves temporal dependency structures, thus addressing the limitations of traditional GANs in time-series data modeling. As illustrated in Figure 1, the TimeGAN is built upon four essential modules: an embedding, a recovery, a sequence generator, and a sequence discriminator. The core of TimeGAN lies in the collaborative training that combines an auto-encoding component (the embedding module and the reconstruction module) with an adversarial component (sequence generation and sequence discrimination).

![](images/ad991dcd4d92a7889f2929156b40cadaf2339de9ae142cc048dbd0fba79af913.jpg)  
Figure 1. The structure of TimeGAN.

Static characteristics S and temporal characteristics Ω are two characteristics of timeseries data. s and ω are random variables belonging to S and Ω, respectively. Embedding and recovery networks mediate the feature-latent space transformation, allowing an adversarial network of temporal dynamics in condensed representations. Denote the feature space as $H _ { S }$ and its associated latent space as $H _ { \Omega }$ . Define the embedding network as e: $\begin{array} { r } { S \times \prod _ { t } \Omega  H _ { S } \times \prod _ { t } H _ { \Omega } } \end{array}$ . The e module transforms both static and temporal features into latent codes $h _ { S }$ and $h _ { t } ,$ respectively. Implement the embedding function e with a recurrent neural network (RNN).

$$
h _ {S} = e _ {S} (s), h _ {t} = e _ {\Omega} (h _ {S}, h _ {t - 1}, \omega_ {t})\tag{1}
$$

where $e _ { S }$ and $e _ { \Omega }$ represent the embedding network for static and temporal features, respectively.

The recovery network is denoted as r, which corresponds to the embedding network. The expression for r is $\begin{array} { r } { H _ { S } \times \prod _ { t } H _ { \Omega } \to S \times \prod _ { t } \Omega } \end{array}$ . The r decodes latent codes back into static features s˜ and temporal features $\tilde { \omega } _ { 1 : T } = r ( h _ { S } , h _ { 1 : T } )$ . A feedforward neural network is employed to realize the recovery function r.

$$
\tilde {s} = r _ {S} (h _ {S}), \tilde {\omega} _ {1: T} = r _ {\Omega} (h _ {t})\tag{2}
$$

where r : $H _ { S } {  } S$ and r<sub>Ω</sub>: $H _ { \Omega } {  } \Omega$ correspond to functions that project latent representations back to original feature spaces, with $r _ { S }$ for static embeddings and $r _ { \Omega }$ for temporal embeddings.

Rather than the direct generation of sequences, the generator of TimeGAN first operates within the embedded latent space. Vectors are randomly sampled from known domains $Z _ { S }$ and $Z _ { \Omega }$ as inputs to generate $H _ { S }$ and $H _ { \Omega } ,$ , with the generation process defined by function g: $Z _ { S } \times \prod _ { t } Z _ { \Omega }  H _ { S } \times \prod _ { t } H _ { \Omega }$ . Latent codes $\hat { h } _ { S } , \hat { h } _ { 1 : T } = g ( z _ { s } , z _ { 1 : T } )$ are generated by transforming static and temporal random vectors through the generation function $g .$ The generation function g is implemented with the RNN

$$
\hat {h} _ {S} = g _ {S} (z _ {s}), \hat {h} _ {t} = g _ {\Omega} \Big (\hat {h} _ {S}, \hat {h} _ {t - 1}, z _ {t} \Big)\tag{3}
$$

where g<sub>S</sub>: $Z _ { S } {  } H _ { S }$ and g<sub>Ω</sub>: $H _ { S } \times H _ { \Omega } \times Z _ { \Omega }  H _ { \Omega }$ denote the generator networks for static and temporal features, respectively. The generation of $z _ { s }$ allows for sampling from an arbitrary distribution, while z is governed by a stochastic process.

Define the discriminator function as d: $\begin{array} { r } { H _ { S } \times \prod _ { t } H _ { \Omega } \to [ 0 , 1 ] \times \prod _ { t } [ 0 , 1 ] } \end{array}$ . The discriminator returns the classifier $\tilde { y } _ { S } , \tilde { y } _ { 1 : T } = d ( h _ { S } , h _ { 1 : T } )$ after receiving both static and temporal latent codes. A bidirectional RNN capped by a feedforward output layer serves as the discriminator d.

$$
\tilde {y} _ {S} = d _ {S} \big (\tilde {h} _ {S} \big), \tilde {y} _ {t} = d _ {\Omega} \Big (\stackrel {\leftarrow} {u} _ {t}, \stackrel {\rightarrow} {u} _ {t} \Big)\tag{4}
$$

where $\vec { u } _ { t } = \vec { c } _ { \Omega } \Big ( \tilde { h } _ { S } , \tilde { h } _ { t } , \vec { u } _ { t - 1 } \Big )$ denotes the forward hidden state sequence, whereas $\stackrel {  } { c } _ { \Omega } ( \tilde { h } _ { S } , \tilde { h } _ { t } , \stackrel {  } { u } _ { t + 1 } )$ denotes the backward counterpart; ${ \vec { c } } _ { \Omega }$ and are the hidden states from the preceding and succeeding time steps, respectively; and $d _ { S }$ and $d _ { \Omega }$ serve as the output layer classification functions for static and temporal features.

During training, the TimeGAN framework employs three separate loss functions for network optimization: reconstruction loss $( L _ { R } )$ , unsupervised loss $( L _ { U } )$ , and supervised loss $( L _ { S } )$ . These loss functions are formulated mathematically below

$$
\begin{array}{r l} & L _ {R} = E _ {s, \omega_ {1: T \sim P}} [ \| s - \tilde {s} \| _ {2} + \sum_ {t} \| \omega_ {t} - \tilde {\omega} _ {t} \| _ {2} ] \\ & L _ {U} = E _ {s, \omega_ {1: T \sim P}} [ l o g y _ {S} + \sum_ {t} l o g y _ {t} ] + E _ {s, \omega_ {1: T \sim \tilde {P}}} [ l o g (1 - \hat {y} _ {S}) + \sum_ {t} l o g (1 - \hat {y} _ {t}) ] \\ & L _ {S} = E _ {s, \omega_ {1: T \sim P}} [ \sum_ {t} \| h _ {t} - g _ {\Omega} (h _ {S}, h _ {t - 1}, z _ {t}) \| _ {2} ] \end{array}\tag{5}
$$

## 2.2. Depthwise Separable Convolutional Network

The Depthwise Separable Convolution (DSC) was proposed by Chollet in 2017, and its structure is a combination of Depthwise Convolution (DC) and Pointwise Convolution (PC) [33]. Compared to standard convolution, it significantly improves the computational efficiency of the model [34]. DC performs independent convolutions on each input channel, generating output feature maps with a quantity consistent with that of the input channels. PC employs $1 \times 1$ convolution kernels for both cross-channel feature fusion and flexible output channel dimension adjustment. Figure 2 illustrates the architectural comparison between standard convolution and DSC (DC and PC). Unlike standard convolution, which simultaneously processes all input channels, DSC significantly reduces computational complexity. The parameter counts for standard convolution and DSC are calculated as follows

$$
P _ {\mathrm{Conv}} = C \times M \times K\tag{6}
$$

$$
P _ {\mathrm{DSConv}} = C \times K + C \times M\tag{7}
$$

where $P _ { \mathrm { C o n v } }$ represents the standard convolution, and $P _ { \mathrm { D S C o n v } }$ represents the DSC. Let C, M, and K represent the input channel count, output channel count, and kernel size, respectively.

![](images/c0f6588524cd87a79d7bf6241bf410c07887e4067a84363d39cb8a146e754e5a.jpg)  
Figure 2. Structural diagrams of standard convolution and DSC.

## 2.3. Efficient Local Attention Module

The attention mechanism is a technique that mimics human cognitive processes, enabling the model to dynamically learn the weights of each input region and highlight only the most salient features. The Efficient Local Attention (ELA) module can accurately focus on key regions, significantly enhancing feature representation capabilities while maintaining its lightweight characteristics [35].

The calculation process for ELA is as follows:

For input X of dimensions $C \times H \times W ,$ , horizontal and vertical averaging pooling is performed on each channel using pooling kernels of dimensions (H, 1) and (1, W), respectively. Here, C represents the number of channels, while H and W denote the height and width of the image. The pooling calculations for the c-th channel at height h and width w are given by Equations (8) and (9), respectively.

$$
z _ {c} ^ {h} (h) = \frac {1}{W} \sum_ {0 \leq i <   W} x _ {c} (h, i)\tag{8}
$$

$$
z _ {c} ^ {w} (w) = \frac {1}{H} \sum_ {0 \leq j <   H} x _ {c} (j, w)\tag{9}
$$

where $z _ { c } ^ { h } ( h )$ and $z _ { c } ^ { w } ( w )$ represent the sequence signal outputs at height h and width w of the c-th channel, respectively, so one-dimensional convolution is used to capture positional details along both horizontal and vertical directions.

To enhance the model’s generalization capability, the Batch Normalization (BN) layer is replaced with a Group Normalization (GN) layer, yielding attention representations in the horizontal and vertical directions

$$
y ^ {h} = \sigma (G _ {n} (F _ {h} (z ^ {h})))\tag{10}
$$

$$
y ^ {w} = \sigma (G _ {n} (F _ {w} (z ^ {w})))\tag{11}
$$

where $\sigma$ represents a nonlinear function, $G _ { n }$ represents the GN layer, and $F _ { h }$ and $F _ { w }$ denote the horizontal and vertical one-dimensional convolutions, respectively. $y ^ { h }$ and $y ^ { w }$ represent horizontal and vertical position attention, respectively. The final output $Y$ of the ELA module is obtained by computing $y ^ { h }$ and $y ^ { w }$ through the Softmax function, expressed as

$$
Y = x _ {c} \times y ^ {h} \times y ^ {w}\tag{12}
$$

## 3. Proposed Method

## 3.1. Gradient-Penalized Conditional Time-Series Generative Adversarial Network

TimeGAN employs the Jensen–Shannon (JS) divergence as the metric to quantify the dissimilarity between real and generated sequences. To measure the discrepancy between real and generated data, when the two distributions have no overlap, the JS divergence converges to the constant log 2, causing gradient vanishing during training [22]. Meanwhile, TimeGAN suffers from the issue of uncontrollable generated samples. To address these issues, this paper proposes an improved model termed the gradient-penalized Conditional TimeGAN (CTGAN).

Firstly, in the CTGAN, the Wasserstein distance is introduced as a new metric. Its advantage lies in providing stable gradient information, even under substantial distributional divergence between generated and real data, thereby effectively mitigating the vanishing gradient problem. Additionally, a gradient penalty mechanism is replaced by weight clipping, preventing drastic weight fluctuations and thereby reducing issues of vanishing or exploding gradients. This results in a smoother training process. Therefore, the unsupervised loss function $\hat { L } _ { U }$ of CTGAN is

$$
\hat {L} _ {U} = - E _ {s, \omega_ {1: T \sim P}} \left[ y _ {S} + \sum_ {t} y _ {t} \right] + E _ {s, \omega_ {1: T \sim \tilde {P}}} \left[ y _ {S} + \sum_ {t} y _ {t} \right] + \lambda E _ {\tilde {s}, \tilde {\omega} _ {1: T}} \left[ \left\| \nabla_ {\tilde {s}, \tilde {\omega} _ {1: T}} (\hat {y} _ {s} + \sum_ {t} \hat {y} _ {t}) \right\| _ {2} - 1 \right] ^ {2}\tag{13}
$$

$$
\tilde {s}, \tilde {\omega} _ {1: T} = \varepsilon (s + \omega_ {t}) + (1 - \varepsilon) (y _ {s} + \sum_ {t} y _ {t}) \sim \tilde {p} _ {s, \omega_ {1: T}}\tag{14}
$$

where ε denotes a randomly generated value within the interval (0,1), λ represents the weight of the gradient penalty, and $\begin{array} { r l } { \big [ \big \| \nabla _ { \tilde { s } , \tilde { \omega } _ { 1 : T } } \big ( \hat { y } _ { s } + \sum _ { t } \hat { y } _ { t } \big ) \big \| _ { 2 } - 1 \big ] ^ { 2 } } & { { } } \end{array}$ represents the gradient penalty term.

Secondly, the class-conditional labeling mechanism is introduced into CTGAN to generate class-specific fault samples. Then the embedding network is improved to

$$
h _ {S} = e _ {S} (s, c l a s s \_ l a b e l), h _ {t} = e _ {\Omega} (h _ {S}, h _ {t - 1}, \omega_ {t}, c l a s s \_ l a b e l)\tag{15}
$$

The generator and discriminator are improved through the following modifications

$$
\hat {h} _ {S} = g _ {S} (z _ {s}, c l a s s \_ l a b e l), \hat {h} _ {t} = g _ {\Omega} \Big (\hat {h} _ {S}, \hat {h} _ {t - 1}, z _ {t}, c l a s s \_ l a b e l \Big)\tag{16}
$$

$$
\tilde {y} _ {S} = d _ {S} \big (\tilde {h} _ {S}, c l a s s \_ l a b e l \big), \tilde {y} _ {t} = d _ {\Omega} \Big (\stackrel {\leftarrow} {u} _ {t}, \stackrel {\rightarrow} {u} _ {t}, c l a s s \_ l a b e l \Big)\tag{17}
$$

The supervision loss function is improved to

$$
\hat {L} _ {S} = E _ {s, \omega_ {1: T \sim P}} \left[ \sum_ {t} \| h _ {t} - g _ {\Omega} (h _ {S}, h _ {t - 1}, z _ {t}, c l a s s \_ l a b e l) \| \right. _ {2}\tag{18}
$$

## 3.2. Multi-Scale Importance-Aware Network

In scenarios with imbalanced data, the design of feature extraction and classification networks is a critical step in fault diagnosis. Multi-scale CNN (MSCNN) has emerged as a predominant approach in this field due to its unique capability to concurrently extract multi-temporal-scale features, which aligns perfectly with the multi-scale characteristics of rotating machinery signals. However, features at different scales have varying degrees of importance. Concatenating and fusing features from different scales may result in the loss of critical scale-specific feature information and consequently reduce diagnostic accuracy.

To tackle this challenge, this study innovatively proposes a Multi-scale Importanceaware Network (MSIN) framework for fault classification. MSIN employs a multilevel feature extraction mechanism, which can effectively capture fault feature representations at different scales. Furthermore, a Scale Enhanced Local Attention (SELA) module is introduced that utilizes an adaptive weight allocation mechanism, enabling the model to direct its attention with high precision to the most discriminative scale features. This design not only achieves dynamic assessment of the importance of features at different scales but also significantly enhances feature amplification and classification performance.

## (a) Multi-scale depthwise separable residual block

Drawing inspiration from the Inception module [36] and residual design of ResNet, the Multi-scale Depthwise Separable Residual (MDSR) module is proposed. By leveraging DSC and residual connection techniques, the MDSR effectively mitigates the issues of gradient vanishing and explosion while efficiently guiding the classification model to extract rich fault features. A schematic diagram of the MDSR is presented in Figure 3.

![](images/0ee6e71fb12bfc48158f0f5168a168ed456b480017f87938e2d93e5321bef749.jpg)  
Figure 3. Structural diagram of MDSR.

The input signal S to the MDSR module is a one-dimensional signal with dimensions (C, L), where C and L represent the number of channels and the signal length, respectively. The MDSR operation process is as follows.

Firstly, ${ \mathsf { a } } 9 \times 1$ large-kernel DSC module is designed to initially capture broader-scale fault features, whose mathematical expression is

$$
F = \sigma (\mathrm{LN} (\mathrm{DSConv1d} (S)))\tag{19}
$$

where F denotes the obtained feature. To accelerate network convergence and mitigate overfitting, Layer Normalization (LN) is adopted. The activation function utilized is the smoother GELU.

Subsequently, we constructed a multi-branch parallel structure comprising three independent processing branches. Each branch follows the following processing workflow: first, different-sized DC is applied (with kernel sizes of $3 \times 1 , 5 \times 1 ,$ and $7 \times 1$ for each branch), followed by LN and GELU activation function processing.

Then, PC $( 1 \times 1$ convolutions) is performed, and the GELU activation function is applied again. The expression for this process is

$$
B _ {i} = \sigma (\mathrm{PConv1d} (\sigma (\mathrm{LN} (\mathrm{DConv1d} (F))))\tag{20}
$$

where $B _ { i }$ represents the characteristics of different branches, i = 1, 2, 3. The shape of $B _ { i }$ is $( C / r , L )$ , where r is the reduction ratio. Channel parameters can be reduced by decreasing r.

Finally, the output features of the three branches are concatenated along the scale to form a two-dimensional multi-scale feature map $I _ { m } { \in } R ^ { C / r \times L \times B } .$ , where B denotes the number of multi-scale kernels. To preserve critical information from the original input and mitigate gradient vanishing in deep architectures, residual connections are incorporated. This operation concatenates features $F$ and $I _ { m }$ to produce the output feature y. The expressions for the multi-scale feature map $I _ { m }$ and the final output feature y are given by Equations (21) and (22), respectively,

$$
I _ {\mathrm{m}} = [ B _ {1}, B _ {2}, B _ {3} ]\tag{21}
$$

$$
y = [ I _ {\mathrm{m}}, F ]\tag{22}
$$

In the above equation, $y { \in } R ^ { C / r \times N \times L }$ , where $N = B + 1$

## (b) Scale enhanced local attention module

Considering that features at different scales have varying degrees of importance, a Scale Enhanced Local Attention (SELA) module is proposed. This module draws inspiration from ELA for generating precise positional attention maps through average pooling along the height and width directions. For the multi-scale features y generated by MDSR, it performs average pooling along both the length and scale dimensions, thereby gen erating attention maps containing distinct scale and positional information. The SELA module effectively enhances the expressive power of features across different positions and scales, precisely guiding the model to focus on the most discriminative scale features. Figure 4 illustrates the specific operational flow of the SELA module and the corresponding dimensional changes.

![](images/e23840d22db165876138b1f2d2283d8859984de9e39b11f12c4feff8fdb392a7.jpg)  
Figure 4. The proposed SELA module.

Input the one-dimensional vibration signal into the MDSR module to obtain the multiscale feature y as input for SELA. Perform average pooling on each channel using pooling kernels of sizes $( N , 1 )$ and (1, L). The pooling computation process for the c-th channel at scale n is as follows

$$
z _ {c} ^ {n} (n) = \frac {1}{L} \sum_ {0 \leq i <   L} y _ {c} (n, i)\tag{23}
$$

The pooling calculation process for the c-th channel at length l is as follows

$$
z _ {c} ^ {l} (l) = \frac {1}{N} \sum_ {0 \leq j <   N} y _ {c} (j, l)\tag{24}
$$

One-dimensional convolutions are employed to enhance scale and positional information, followed by GN to augment the feature information. The resulting horizontal attention $g ^ { b }$ and vertical attention $g ^ { l }$ expressions are as follows

$$
g ^ {b} = \sigma (G _ {n} (F _ {b} (z ^ {b})))\tag{25}
$$

$$
g ^ {l} = \sigma (G _ {n} (F _ {l} (z ^ {l})))\tag{26}
$$

In the above description, $F _ { b }$ and $F _ { l }$ denote one-dimensional convolution operations. Applying Equation (27) yields the SELA output O

$$
O = y _ {c} \times g ^ {b} \times g ^ {l}\tag{27}
$$

## (c) Multi-scale importance-aware network

A novel architecture, the Multi-scale Importance-aware Network (MSIN), is proposed in this subsection. The framework of MSIN is illustrated in Figure 5, and its diagnostic framework consists of the MDSR module and the SELA module. The MDSR efficiently extracts features at different scales, while the SELA module performs adaptive weighting on the multi-scale features extracted by the MDSR to emphasize discriminative key scale features. The classifier compresses feature dimensions using Global Average Pooling (GAP), then maps them through a Fully Connected (FC) layer, and finally uses the Softmax function to output the probability distribution of fault categories. The entire process of CTGAN-MSIN is illustrated in Figure 6.

![](images/fc9e62f1902b85fff424d16a825e5576ac00c1d05ea22bb5404832dc6dde4be2.jpg)  
Figure 5. Framework diagram of the MSIN.

![](images/cf4baa8bb31f73cd09067ed4221ab7004c075a11b504ec06755474f5c05365a7.jpg)  
Figure 6. The process of CTGAN-MSIN.

## 4. Case Studies

Two cases of rotating machinery are used to demonstrate the data augmentation and fault diagnosis capabilities of the CTGAN-MSIN, namely the HUST Bearing public dataset [37] and the laboratory axial piston pump self-collected dataset.

## 4.1. Case 1: HUST Bearing Dataset

## 4.1.1. Description of HUST Bearing Test

The HUST bearing failure test rig integrates a speed control, a motor, a shaft, and three acceleration sensors. The test collected vibration signals in three directions, with four different speed conditions set: 3900 rpm, 4200 rpm, 4500 rpm, and 4800 rpm. Data were acquired at a sampling frequency of 25.6 kHz for a duration of 10.2 s. Table 1 presents the specific fault types of the bearing and the corresponding data-splitting details adopted in this work. To simulate data imbalance scenarios, we constructed 150 samples for the normal state and 10 samples for each fault state. Each sample contains at least two rotation cycles, thus comprising 1024 data points.

Table 1. HUST bearing fault dataset description.

<table><tr><td>Fault Type</td><td>Operating Condition</td><td>Label</td><td>Sample Number</td></tr><tr><td>Health</td><td>3900 rpm</td><td>H</td><td>150</td></tr><tr><td>Inner ring fault</td><td>3900 rpm</td><td>IR</td><td>10</td></tr><tr><td>Outer ring fault</td><td>3900 rpm</td><td>OR</td><td>10</td></tr><tr><td>Rolling element fault</td><td>3900 rpm</td><td>RF</td><td>10</td></tr></table>

## 4.1.2. Data Generation and Evaluation in the HUST Bearing Dataset

This study utilizes the proposed CTGAN model to augment fault state data. In the CTGAN architecture, GRU is employed as the recurrent neural network layer. Dropout layers are incorporated into the network to mitigate overfitting and improve generalization ability. Table 2 details the structures and parameter settings of the embedding, recovery, generator, and discriminator networks.

Table 2. The structure and parameters of the CTGAN.

<table><tr><td>Network</td><td>Structure</td><td colspan="2">Parameter</td></tr><tr><td rowspan="5">Embedder</td><td>GRU</td><td>Hidden_dim = 24</td><td rowspan="5"></td></tr><tr><td>Dropout</td><td>Dropout rate = 0.2</td></tr><tr><td>GRU</td><td>Hidden_dim = 24</td></tr><tr><td>Dense</td><td>Units = 24</td></tr><tr><td>Layer Norm</td><td> $\beta = 0, \gamma = 1$ </td></tr><tr><td rowspan="6">Recovery</td><td>GRU</td><td>Hidden_dim = 24</td><td>Optimizer = Adam</td></tr><tr><td>Dropout</td><td>Dropout rate = 0.2</td><td rowspan="3">Initial learning rate = 0.0001</td></tr><tr><td>GRU</td><td>Hidden_dim = 24</td></tr><tr><td>Dense</td><td>Units = 24</td></tr><tr><td>Layer Norm</td><td> $\beta = 0, \gamma = 1$ </td><td>Decay steps = 100</td></tr><tr><td>Dense1</td><td>Units = 3</td><td rowspan="4">Decay rate = 0.96 Epochs = 2000</td></tr><tr><td rowspan="3">Generator</td><td>GRU</td><td>Hidden_dim = 24</td></tr><tr><td>Dropout</td><td>Dropout rate = 0.2</td></tr><tr><td>Dense1</td><td>Units = 3</td></tr><tr><td rowspan="5">Discriminator</td><td>GRU</td><td>Hidden_dim = 24</td><td rowspan="5"></td></tr><tr><td>Dropout</td><td>Dropout rate = 0.2</td></tr><tr><td>Dense</td><td>Units = 24</td></tr><tr><td>Layer Norm</td><td> $\beta = 0, \gamma = 1$ </td></tr><tr><td>Dense2</td><td>Units = 1</td></tr></table>

During the model training phase, the CTGAN utilizes an overlapping sampling strategy (with a compensation coefficient of 400) to improve the diversity of generated data. The CTGAN successfully expands the sample size for each fault state from an initial 10 samples to 150 samples, achieving a balanced distribution with normal samples.

To make the resemblance between the raw signals and generated signals more visually apparent, Figure 7 displays the time-domain waveforms under different fault conditions of raw signals and generated signals. Figure 7 demonstrates that compared to the raw signals, the generated signals exhibit slight differences in peak and trough amplitudes; they fully retain the glitch features of the raw signals. This ensures both class consistency and inherent variability in the generated samples.

![](images/f5a865e219dc5052015a5f8edbf75572ecd677135d2879edeb867ab7ecc51e1a.jpg)  
Figure 7. Time domain plots of the original fault signal and the generated fault signal of HUST Bearing.

## 4.1.3. Ablation Experiment in the HUST Bearing Dataset

For fault diagnosis under imbalanced data, the CTGAN-MSIN method is proposed in this paper. To evaluate the contribution of each module, ablation experiments were performed on the CTGAN (a data augmentation model), MSIN (a classification module), and CTGAN-MSIN (overall model).

## (a) Ablation experiments for CTGAN

Table 3 summarizes the results of the ablation test conducted on the CTGAN. Accord ing to Table 3, TimeGAN, which employs JS divergence and weight clipping, achieves a diagnostic accuracy of 92%. When JS divergence and weight clipping in TimeGAN are replaced with Wasserstein distance and gradient penalty, respectively, the accuracy of the Gradient-penalized Time-series Generative Adversarial Network (GPTGAN) increases to 95.50%, demonstrating superior optimization performance. Additionally, the Conditional Time-series Generative Adversarial Network (CTGAN), which incorporates label information into the generator and discriminator of TimeGAN, shows improved accuracy compared to TimeGAN, confirming the effectiveness of label information in diagnostic tasks. Notably, the proposed CTGAN demonstrates obvious superiority in terms of classification accuracy, and this outstanding performance fully validates the synergistic advantages of combining the condition label mechanism with the Wasserstein distance with gradient penalty strategy in enhancing fault classification performance.

## (b) Ablation experiments for MSIN

The expanded dataset is fed into different classification models in the ablation experiments for performance comparison. Table 4 shows the ablation test results for MSIN. Table 4 shows that the 1D-CNN attained the poorest performance, achieving an accuracy standing at only 89.33%. After introducing the SELA module into the 1D-CNN model, the resultant CNN-ELA model exhibited a 3.93% increase in accuracy over the 1D-CNN. The MDSR model is used only for fault diagnosis with the MDSR proposed in Section 3.2. In summary, the proposed MSIN method achieves a superior diagnostic accuracy of 98.75%, outperforming all comparative approaches. Ablation test results confirm the important role of the MDSR and SELA modules in the MSIN model.

Table 3. The ablation experiment results of the CTGAN in the HUST bearing dataset.

<table><tr><td>Model</td><td>TimeGAN</td><td>Condition Label</td><td>Wasserstein Distance with Gradient Penalty</td><td>Accuracy (%)</td></tr><tr><td>TimeGAN</td><td>√</td><td>×</td><td>×</td><td>92.00</td></tr><tr><td>CTGAN (without GP)</td><td>√</td><td>√</td><td>×</td><td>96.86</td></tr><tr><td>GPTGAN</td><td>√</td><td>×</td><td>√</td><td>95.50</td></tr><tr><td>CTGAN</td><td>√</td><td>√</td><td>√</td><td>98.75</td></tr></table>

Table 4. The ablation experiment results of the MSIN in the HUST bearing dataset.

<table><tr><td>Model</td><td>1D-CNN</td><td>MDSR</td><td>ELA</td><td>SELA</td><td>Accuracy (%)</td></tr><tr><td>MDSR</td><td>×</td><td>√</td><td>×</td><td>×</td><td>94.73</td></tr><tr><td>CNN-ELA</td><td>√</td><td>×</td><td>√</td><td>×</td><td>93.26</td></tr><tr><td>1D-CNN</td><td>√</td><td>×</td><td>×</td><td>×</td><td>89.33</td></tr><tr><td>MSIN</td><td>×</td><td>√</td><td>×</td><td>√</td><td>98.75</td></tr></table>

(c) Ablation experiments for CTGAN-MSIN

Table 5 presents the ablation experiment results for the CTGAN-MSIN approach. The MSIN-only model refers to fault diagnosis using MSIN directly, while the CTGAN-CNN model employs CNN for fault diagnosis based on CTGAN data augmentation. Table 5 indicates that the proposed complete method achieves the highest accuracy. Diagnostic performance significantly deteriorates when either the CTGAN data augmentation module or the MSIN module is omitted. This result further validates the effectiveness of the overall method and the complementary nature of its components.

Table 5. The ablation experiment results of the CTGAN-MSIN in the HUST bearing dataset.

<table><tr><td>Model</td><td>CTGAN</td><td>MSIN</td><td>CNN</td><td>Accuracy (%)</td></tr><tr><td>MSIN-only</td><td>×</td><td>√</td><td>×</td><td>85.03</td></tr><tr><td>CTGAN-CNN</td><td>√</td><td>×</td><td>√</td><td>89.33</td></tr><tr><td>CTGAN-MSIN</td><td>√</td><td>√</td><td>×</td><td>98.75</td></tr></table>

## 4.1.4. Diagnostic Results Analysis in the HUST Bearing Dataset

Table 2 details the specific structure and hyperparameter configuration of the CTGAN data augmentation network employed in this paper. In the classification task after data augmentation, the 150 samples for each state are split 70% for training and 30% for validation in the ensuing classification task. Additionally, the test set is composed of 100 original samples selected from each state. The hyperparameters of MSIN are listed in Table 6.

Table 6. The hyperparameters of MSIN.

<table><tr><td>Main Parameters</td><td>Values</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>0.0001</td></tr><tr><td>Training epochs</td><td>100</td></tr></table>

The programming code for this study is written in Python 3.9 and executed on a computer equipped with an Intel i5 CPU (manufacturer: Intel Corporation, Santa Clara, United States) and NVIDIA GeForce RTX 4070 SUPER GPU (manufacturer: NVIDIA Corporation, Santa Clara, United States).

## (a) Comparison of data augmentation algorithms with other methods

To highlight the superiority of the proposed approach, CTGAN is benchmarked against the Conditional Variational Autoencoder (CVAE) [12] and TimeVAE [38], the oversampling method SMOTE [39], and four GAN-based models: GAN [40], WGAN [18], DCGAN [16], and COT-GAN [41]. A CNN fed with the original signals is also included as a baseline. Table 7 summarizes the comparative results, presenting the average values and standard deviations obtained from ten runs. Figure 8 shows the distribution plots of real and generated data for the inner ring (IR) fault across different models.

Table 7. Comparative diagnostic results of data augmentation methods for the HUST bearing.

<table><tr><td>Method</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-Score (%)</td><td>Time (s)</td></tr><tr><td>CNN</td><td>67.75 ± 2.82</td><td>73.56 ± 3.69</td><td>67.75 ± 2.82</td><td>65.61 ± 3.75</td><td>——</td></tr><tr><td>CVAE</td><td>88.75 ± 8.56</td><td>90.93 ± 6.77</td><td>88.75 ± 8.56</td><td>87.04 ± 10.99</td><td>0.0215</td></tr><tr><td>TimeVAE</td><td>89.86 ± 4.59</td><td>91.63 ± 2.40</td><td>89.86 ± 4.59</td><td>89.07 ± 5.83</td><td>0.0070</td></tr><tr><td>SMOTE</td><td>92.45 ± 3.32</td><td>92.82 ± 3.37</td><td>92.45 ± 3.32</td><td>92.35 ± 3.37</td><td>——</td></tr><tr><td>GAN</td><td>91.39 ± 5.35</td><td>93.14 ± 3.45</td><td>91.39 ± 5.35</td><td>90.10 ± 6.55</td><td>0.3119</td></tr><tr><td>WGAN</td><td>94.25 ± 3.89</td><td>95.30 ± 3.1</td><td>94.25 ± 3.89</td><td>94.19 ± 4.10</td><td>3.0125</td></tr><tr><td>DCGAN</td><td>96.75 ± 1.45</td><td>96.83 ± 2.12</td><td>96.75 ± 1.45</td><td>96.72 ± 2.16</td><td>0.6826</td></tr><tr><td>COT-GAN</td><td>92.08 ± 4.91</td><td>92.74 ± 5.45</td><td>92.08 ± 4.91</td><td>92.70 ± 3.73</td><td>5.4786</td></tr><tr><td>CTGAN</td><td>98.75 ± 1.19</td><td>98.66 ± 1.30</td><td>98.75 ± 1.19</td><td>98.48 ± 1.46</td><td>5.8146</td></tr></table>

As shown in Table 7, CTGAN achieves the best diagnostic performance with the smallest standard deviation. Apart from the baseline CNN method, CVAE yields the least favorable results. Additionally, Table 7 lists the training time for each method in one epoch. Since CNN does not involve data augmentation and SMOTE does not require designing training epochs, their training times were not calculated. From the perspective of training time, VAE-based methods (including CVAE and TimeVAE) have the shortest running time, but their accuracy and model stability are relatively poor. Among GAN-based methods, the benchmark GAN method also has the shortest running time, yet its accuracy is the lowest. Considering the three indicators of training time, accuracy, and standard deviation comprehensively, the method proposed in this paper performs more balanced and achieves relatively ideal results.

Figure 8 displays a notable divergence of the generated data distribution from the original, indicating lower-quality generated samples. The oversampling method SMOTE attains a diagnostic accuracy of 92.45% under imbalanced data conditions; however, Figure 8 shows that the generated samples of SMOTE are concentrated in a specific region, reflecting limited diversity in the generated data. Among the various GAN models, DCGAN performs the best, yet its data distribution plot still exhibits limited diversity in generated samples. Comparative analysis in Figure 8 demonstrates that the data generated by the proposed CTGAN method most closely aligns with the real data in both diversity and distribution, confirming its superior sample generation quality.

![](images/1f1fdaafde87e64e01951ee2e85dc4b56428ed1df1d6fe3123ef84981de3b2c3.jpg)  
(a) $\mathrm { C V A E }$

![](images/2e70f656fcca399fe6b3d9bb981c954ac61185dc29483c1f18a74a7b5d73eb3c.jpg)  
(b) TimeVAE

![](images/a1e19b108965b2dd31027ed8bfdd18ca302bb5df1848dc41c7c19e8ff1b11fd1.jpg)  
(c) SMOTE

![](images/f040a0af82897470b65fec94ec9fb60338bf024697638192f3b4dbe233dd01d5.jpg)  
(d) GAN

![](images/5b331e4c603871394921f26477d141558902c54a21a1d0cdc5e50e87fc7d40db.jpg)  
(e) WGAN

![](images/5edef8eab74a301a6b29ad0eb8f5b5f7dbb4f9c0ed4ebf95e4394bbd8f3f6a3b.jpg)  
(f) $\mathsf { D C G A N }$

![](images/00392bcda46d8c515abb7b44773120f0fd8822be4e1dbad74645a44a87b2c6ca.jpg)  
(g) $\mathrm { C O T - G A N }$

![](images/be663018228c09a5de171b711585249cbc1ff714f28763262eaabb5973e4a67b.jpg)  
(h) CTGAN  
Figure 8. Distribution diagrams of real and generated samples in different models for HUST Bearing.

Figure 9 presents the diagnostic performance results under varying numbers of fault samples. In the experiments, the size of training samples for fault state was configured as 1, 3, 6, 10, and 15, respectively, corresponding to imbalance ratios of 150:1, 50:1, 25:1, 15:1, and 10:1, respectively. As shown in Figure 9, the diagnostic accuracy of all methods shows progressive improvement with an increasing number of faults training samples, consistently outperforming the baseline CNN by a considerable margin. The proposed CTGAN achieves the best performance across all imbalance ratios. Notably, when the data imbalance ratios are 25:1, 15:1, and 10:1, the diagnostic accuracy of CTGAN exceeds 90%. In contrast, the accuracy of the comparative methods SMOTE, GAN, WGAN, DCGAN, and COT-GAN only exceeds 90% when the imbalance ratio is reduced to 15:1.

![](images/40c8580e42823624416fb6f152c428e7faf41cd82c064ae52f4a5987ad43d1eb.jpg)  
Figure 9. Diagnostic results under different numbers of fault samples for HUST Bearing.

## (b) Comparison of classification models with different methods

The effectiveness of the classification model is evaluated by comparing MSIN against multiple advanced methods, under two scenarios: with CTGAN-based data augmentation and without CTGAN-based data augmentation. The comparative methods include ResNet-18 [42], CNN-LSTM [43], and DPCNN [44], while a standard CNN is used as the benchmark model to maintain experimental objectivity.

Figure 10 presents the diagnostic results of different classification models. As can be seen from Figure 10, the MSIN model attains optimal classification results among all compared methods. Even without CTGAN-based data augmentation, MSIN attains an accuracy of 85.03%, which still significantly outperforms other methods. This result demonstrates that MSIN excels in handling fault diagnosis under imbalanced data conditions compared to other approaches. After applying CTGAN, the accuracy of all methods improves significantly. The combination of the highest accuracy along with the lowest standard deviation highlights the overall advantages of our proposed classification model of MSIN.

![](images/7a2e5cc077e48c0203b455e1e3671f8354c98b41d1ed5fd296cf9e2dc440afff.jpg)  
Figure 10. Diagnostic results of different classification models for HUST bearing.

The confusion matrix and feature visualization results of the MSIN are provided in Figures 11 and 12. The feature visualization depicted in Figure 12 demonstrates that MSIN effectively avoids mode mixing, indicating its strong robustness in feature extraction.

![](images/01756567c9a11a3ead8c1d22aad489a7993dfdc12ca84065a97cf1b86314f50a.jpg)  
Figure 11. Confusion matrix in HUST bearing.

![](images/170a553382181d57bd97cd3b9f5bfc775cf3785fcbe304c77dfd3bb2589033d5.jpg)  
Figure 12. t-SNE Visualization in HUST bearing.

## 4.2. Case 2: Axial Piston Pump Dataset

## 4.2.1. Description of Axial Piston Pump Test

A structural schematic diagram of the axial piston pump is depicted in Figure 13, showing components such as the shaft, swash plate, slipper, piston, cylinder block, etc. The friction pairs are essential to the axial piston pump, as they decisively influence the pump’s ability to operate properly. The main friction pairs include: (1) piston and cylinder block hole, (2) slipper and swash plate, (3) valve plate and the cylinder block’s valve plate surface, (4) slipper’s spherical socket and piston head. During long-term operation, these friction pairs inevitably experience varying degrees of wear. Therefore, common piston pump failures include piston wear, slipper wear, valve plate wear, loose slipper, and so on.

![](images/5551d0f557ac48b603dfe65dc9bd85323e035cb2e42396ff839f293db2799e80.jpg)  
Figure 13. Schematic diagram of the axial piston pump.

Figure 14 shows the test bench for the axial piston pump of our laboratory. The experiments employed an axial piston pump (Model: P08-B3-F-R-01) with the following key specifications: 7 pistons, maximum displacement of 8 cm<sup>3</sup>/rev, and maximum pressure of 21 MPa. Based on the aforementioned study on the working mechanism of the piston pump, seven types of faults have been identified, with specific fault information listed in Table 8. Three acceleration sensors, one pressure sensor, and one flow sensor were installed during the test to monitor the operating condition of the pump. The experiment was conducted at a working pressure of 15 MPa while multi-source signals were synchronously recorded at 30 kHz. Fault diagnosis under imbalance conditions uses three vibration signals from the piston pump. To simulate data imbalance scenarios, we set the number of samples for the normal state at 150, while each fault state was assigned only 10 samples. Different faulty components of the piston pump are identified in Figure 15.

![](images/3dc32c70e11006f68514ed710d9d20a4e0102a37ff91ad190048fef8f71f7812.jpg)  
Figure 14. Fault diagnosis test rig of axial piston pump.

![](images/6e21006c490a8626a8f06dd85dc3bd103b24256df2d0f2b0cfc7c3145266e824.jpg)

![](images/6581c320731fa20b70b121612097e0d03a58961602aae8c0111518b2489a6bb4.jpg)

![](images/126f3ac15a4b531829efada1830ce624e6f3b874c59767cf4df0d6b4615406cf.jpg)

![](images/6655ef2e03863575cf4654c5124cdfda27401104a1be833dbac2fe2d399891db.jpg)

![](images/87c3140f66b1bfc01e36eb2a354ce5bc6dbea944f8b93fa1a0ce7049a89b5e79.jpg)  
IR

![](images/127e7fb07a5b6e6dc4dd8d10b987cb803c27a080b0517dc790dab11d0b4ed2ba.jpg)  
OR

![](images/85ce3a6aaa51e608c2b1d6d45913b99714ab060c7eb30e7ae28c7452f27af31c.jpg)  
Figure 15. Faults in the axial piston pump.  
RF

Table 8. Types of axial piston pump failures.

<table><tr><td>Fault Type</td><td>Label</td><td>Sample Number</td></tr><tr><td>Health</td><td>H</td><td>150</td></tr><tr><td>Wear of piston</td><td>WP</td><td>10</td></tr><tr><td>Wear of slipper</td><td>WS</td><td>10</td></tr><tr><td>Loose slipper</td><td>LS</td><td>10</td></tr><tr><td>Wear of valve plate</td><td>VP</td><td>10</td></tr><tr><td>Inner ring fault</td><td>IR</td><td>10</td></tr><tr><td>Outer ring fault</td><td>OR</td><td>10</td></tr><tr><td>Rolling element fault</td><td>RF</td><td>10</td></tr></table>

## 4.2.2. Data Generation and Evaluation in the Axial Piston Pump Dataset

In this section, CTGAN is used to perform data augmentation for each fault state, and the network architecture and hyperparameter configuration are shown in Table 2. Figure 16 presents a visualization comparing the raw fault signal data with generated data from the axial piston pump. Figure 16 shows that the generated signals are highly similar to the raw signals, although they are not identical.

![](images/99cb15fcebdab8760ec9bc3c16557db90a6722d002fd30a03b733a562f65d074.jpg)  
Figure 16. Time domain diagram of raw fault signals and generated fault signals of axial piston pump.

## 4.2.3. Ablation Experiment in the Axial Piston Pump Dataset

This section conducts confusion experiments on the axial piston pump dataset to evaluate the contribution of each module to fault diagnosis. Tables 9–11 present the ablation test results for CTGAN, MSIN, and the model CTGAN-MSIN on this dataset, respectively. The ablation test in Tables 9 and 10 demonstrates that both the CTGAN and MSIN modules designed in this paper exhibit significant effectiveness. From Table 11, it can be observed that the diagnostic accuracy shows a significant decline in the CTGAN-MSIN model when either the CTGAN or the MSIN module is absent.

Table 9. The ablation experiment results of the CTGAN in the axial piston pump dataset.

<table><tr><td>Model</td><td>TimeGAN</td><td>Condition Label</td><td>Wasserstein Distance with Gradient Penalty</td><td>Accuracy (%)</td></tr><tr><td>TimeGAN</td><td>√</td><td>×</td><td>×</td><td>90.86</td></tr><tr><td>CTGAN (without GP)</td><td>√</td><td>√</td><td>×</td><td>94.50</td></tr><tr><td>GPTGAN</td><td>√</td><td>×</td><td>√</td><td>93.25</td></tr><tr><td>CTGAN</td><td>√</td><td>√</td><td>√</td><td>96.50</td></tr></table>

Table 10. The ablation experiment results of the MSIN in the axial piston pump dataset.

<table><tr><td>Model</td><td>1D-CNN</td><td>MDSR</td><td>ELA</td><td>SELA</td><td>Accuracy (%)</td></tr><tr><td>MDSR</td><td>×</td><td>√</td><td>×</td><td>×</td><td>91.50</td></tr><tr><td>CNN-ELA</td><td>√</td><td>×</td><td>√</td><td>√</td><td>88.68</td></tr><tr><td>1D-CNN</td><td>√</td><td>×</td><td>×</td><td>×</td><td>76.75</td></tr><tr><td>MSIN</td><td>×</td><td>√</td><td>×</td><td>√</td><td>96.50</td></tr></table>

Table 11. The ablation experiment results of the CTGAN-MSIN in the axial piston pump dataset.

<table><tr><td>Model</td><td>CTGAN</td><td>MSIN</td><td>CNN</td><td>Accuracy (%)</td></tr><tr><td>MSIN-only</td><td>×</td><td>√</td><td>×</td><td>81.75</td></tr><tr><td>CTGAN-CNN</td><td>√</td><td>×</td><td>√</td><td>76.75</td></tr><tr><td>CTGAN-MSIN</td><td>√</td><td>√</td><td>×</td><td>96.50</td></tr></table>

## 4.2.4. Diagnostic Results Analysis in the Axial Piston Pump Dataset

## (a) Comparison of data augmentation algorithms with other methods

The CTGAN is compared with other advanced methods using experimental data from an axial piston pump. The architecture and hyperparameters of CTGAN are shown in Table 2, and the hyperparameter configuration of MSIN is shown in Table 6. Table 12 presents the comparative results. Taking the IR fault as an example, Figure 17 displays the distribution plots of real versus generated fault data from different models. Consistent with the conclusions in Section 4.1.4(a), the proposed CTGAN demonstrates superior diagnostic performance with the smallest standard deviation under data-imbalanced scenarios. Among all compared methods, the traditional GAN performs the least satisfactorily, except for the baseline CNN method. The GAN’s distribution plot in Figure 17 reveals clustered generated samples, indicating limited diversity in generated data. DCGAN achieves the best performance among conventional methods, yet its accuracy reaches only 91.75% with a relatively larger standard deviation compared to CTGAN. As shown in Figure 17, the generated data of CTGAN maintains a consistent distribution with raw data, while other methods exhibit issues of generated sample clustering and distribution inconsistency.

Table 12. Comparative diagnostic results of data augmentation methods for axial piston pump.

<table><tr><td>Method</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-Score (%)</td><td>Time (s)</td></tr><tr><td>CNN</td><td>50.69 ± 4.16</td><td>61.23 ± 2.78</td><td>50.69 ± 4.16</td><td>48.06 ± 4.12</td><td>— —</td></tr><tr><td>CVAE</td><td>73.47 ± 5.83</td><td>75.94 ± 7.99</td><td>73.47 ± 5.83</td><td>70.93 ± 5.61</td><td>0.0378</td></tr><tr><td>TimeVAE</td><td>90.63 ± 4.78</td><td>91.02 ± 5.48</td><td>90.63 ± 4.78</td><td>90.76 ± 4.20</td><td>0.0191</td></tr><tr><td>SMOTE</td><td>89.69 ± 5.46</td><td>91.44 ± 5.08</td><td>89.69 ± 5.46</td><td>89.42 ± 5.72</td><td>— —</td></tr><tr><td>GAN</td><td>67.00 ± 7.75</td><td>61.90 ± 9.42</td><td>67.00 ± 7.75</td><td>61.37 ± 10.23</td><td>0.3777</td></tr><tr><td>WGAN</td><td>88.75 ± 6.10</td><td>92.59 ± 5.78</td><td>88.75 ± 6.10</td><td>85.78 ± 6.12</td><td>3.5187</td></tr><tr><td>DCGAN</td><td>91.75 ± 4.83</td><td>93.19 ± 2.12</td><td>91.75 ± 4.83</td><td>91.33 ± 5.12</td><td>0.8125</td></tr><tr><td>COT-GAN</td><td>87.75 ± 3.53</td><td>88.40 ± 4.60</td><td>87.75 ± 3.53</td><td>88.37 ± 2.80</td><td>6.0504</td></tr><tr><td>CTGAN</td><td>96.50 ± 1.22</td><td>96.84 ± 0.82</td><td>96.50 ± 1.22</td><td>96.37 ± 1.59</td><td>6.3361</td></tr></table>

![](images/936fbbe7d55280cc7fb39399a6bedf8bbd11aa582143a474fab3de7e8d21b924.jpg)  
(a) CVAE

![](images/5d072e115232ebc43753d6613b544b29242c04a3014027368098e75341f4ebbf.jpg)

![](images/3a6ccf63835cc5599e0ab76451886c46066776c1228f9f59c541d0daa92d5eca.jpg)

![](images/a7a86a6cfd4273b05e5057fd281f79325aba8e9acd12df23c42d35464290ee37.jpg)  
(d) GAN

(b) TimeVAE  
(c) SMOTE  
![](images/025e40b10af5c03ea6717b01c77b4e2e98ef147a265622fd391decbf3c25a2c5.jpg)  
(e) WGAN

![](images/261c24eda826f3262798611fceeff674bc2ef9fca7888c30cf2f00a6ede8c6ce.jpg)  
(f) DCGAN

![](images/9541898144f9ac466616c7659f83214a21819bf41e8816bfad65f7051344c02d.jpg)  
(g) COT-GAN

![](images/1c2c972ca3e55a0154ec2ed19202cafa195ff9f56f7806f7b4450ee87032328f.jpg)  
(h) CTGAN  
Figure 17. Distribution diagrams of real and generated samples in different models for the axial piston pump.

Diagnostic results of the piston pump under varying numbers of fault samples are presented in Figure 18. As shown in Figure 18, the accuracy of all methods improves with a decrease in the class imbalance ratio. The proposed CTGAN method achieves optimal diagnostic performance under imbalanced data conditions, maintaining accuracy above 90% when the imbalance ratio is below 25:1.

![](images/b087779e74571755ea8010e6597c4cc16059709c682c245b1b34042090480895.jpg)  
Figure 18. Diagnostic results under different numbers of fault samples for axial piston pump.

## (b) Comparison of classification models with different methods

Figure 19 presents a comparative analysis of different classification models in axial piston pump fault diagnosis, revealing the following key findings: First, when employing CTGAN for data augmentation, all classification models demonstrated significantly better diagnostic performance compared to scenarios without data augmentation. Second, the proposed MSIN model consistently achieved optimal performance regardless of whether CTGAN data augmentation was applied, attaining not only the highest diagnostic accuracy but also maintaining the smallest standard deviation. These experimental results compre hensively verify MSIN’s superior performance and strong robustness in fault diagnosis tasks. Figures 20 and 21 present the confusion matrix and feature visualization diagrams, respectively, of the proposed CTGAN-MSIN method.

![](images/bef17128cebf7bc821c2e0776fb353ddbd8890719d296e117238a04a828b881d.jpg)  
Figure 19. Diagnostic results of different classification models for axial piston pump.

![](images/04b7dbddba4cdedc43d6d1644249e730993927fcea79075f6838764c1f4767bd.jpg)  
Figure 20. Confusion matrix in axial piston pump.

![](images/b3a333351a9c43e2fb78ed1060b51feac93d153ab78cd1cc262112b69513c21b.jpg)  
Figure 21. t-SNE Visualization in axial piston pump.

## 5. Conclusions

This paper proposes a CTGAN-MSIN to address fault diagnosis under imbalanced data in rotating machinery. CTGAN is employed to expand the imbalanced data, while MSIN is used for fault diagnosis after data augmentation. On the one hand, the CTGAN introduces the Wasserstein distance and gradient penalty as loss functions to construct a more stable adversarial training framework. Meanwhile, CTGAN achieves controllable generation of fault type samples by adding label conditions, ultimately outputting diverse and high-quality multi-source fault data. On the other hand, MSIN extracts fault features at different scales and incorporates a SELA module to address the issue of the varying importance of features at different scales. This enables the precise selection of more discriminative scale features, thereby ensuring the robustness of the diagnostic performance. Tests of the HUST bearing and the self-collected axial piston pump dataset demonstrate the superior performance of CTGAN-MSIN. Through comparative experiments on data augmentation methods and classification models, the proposed method achieved optimal accuracy, demonstrating the significant advantages of CTGAN in data augmentation and the strong capability of MSIN in fault diagnosis. Moreover, CTGAN-MSIN exhibits excellent classification performance under different data imbalance ratios. Future efforts will focus on utilizing algorithms to fine-tune the hyperparameters of the CTGAN to enhance its data generation quality.

However, CTGAN still possesses significant potential for optimization. On the one hand, the current model parameters are all manually configured; subsequent fine-tuning of its hyperparameters using optimization algorithms could further enhance the quality of generated data. On the other hand, the training time of CTGAN is relatively long, and future work could explore lightweight designs for generative adversarial networks to reduce computational costs.

Author Contributions: R.D.: Conceptualization, Investigation, Methodology, Software, Writing— original draft. D.C.: Funding acquisition, Project administration, Supervision, Writing—review & editing. C.Y.: Funding acquisition, Supervision, Validation. D.H.: Writing—review & editing, Investigation. Q.X.: Supervision, Validation. S.Z.: Supervision, Writing—review & editing. All authors have read and agreed to the published version of the manuscript.

Funding: The project is supported by the National Natural Science Foundation of China (Grant No.51975508), Provincial Key Laboratory Performance Subsidy Project (22567612H).

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: Data is unavailable due to privacy.

Conflicts of Interest: Author Dongbo Hu was employed by the company Citic Heavy Industries Co., Ltd. The remaining authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

## References

1. Ma, J.; Huang, J.; Liu, S.; Luo, J.; Jing, L. A self-attention Legendre graph convolution network for rotating machinery fault diagnosis. Sensors 2024, 24, 5475. [CrossRef] [PubMed]

2. Luo, S.; Zhang, D.; Wu, J.; Wang, Y.; Zhou, Q.; Hu, J. A limited annotated sample fault diagnosis algorithm based on nonlinear coupling self-attention mechanism. Eng. Fail. Anal. 2025, 174, 109474. [CrossRef]

3. Wang, G.; Liu, D.; Xiang, J.; Cui, L. Attention guided partial domain adaptation for interpretable transfer diagnosis of rotating machinery. Adv. Eng. Inf. 2024, 62, 102708. [CrossRef]

4. Zhuo, S.; Bai, X.; Han, J.; Ma, J.; Sun, B.; Li, C.; Zhan, L. A novel rolling bearing fault diagnosis method based on the NEITD ADTL-JS algorithm. Sensors 2024, 25, 873. [CrossRef]

5. Chen, W.; Yang, K.; Yu, Z.; Shi, Y.; Chen, C. A survey on imbalanced learning: Latest research, applications and future directions. Artif. Intell. Rev. 2024, 57, 137. [CrossRef]

6. Alimoradia, M.; Sadeghi, R.; Daliri, A.; Zabihimayvan, M. Statistic deviation mode balancer (SDMB): A novel sampling algorithm for imbalanced data. Neurocomputing 2025, 624, 129484. [CrossRef]

7. Chen, Y.; Zhao, Z.; Liu, J.; Tan, S.; Liu, C. Application of generative AI-based data augmentation technique in transformer winding deformation fault diagnosis. Eng. Fail. Anal. 2024, 159, 108115. [CrossRef]

8. Zhang, Z.; Gao, H.; Sun, W.; Song, W.; Li, Q. Multivariate time series generation based on dual-channel Transformer conditional GAN for industrial remaining useful life prediction. Knowl.-Based Syst. 2025, 308, 112749. [CrossRef]

9. Chen, L.; Wan, S.; Dou, L. Improving diagnostic performance of high-voltage circuit breakers on imbalanced data using an oversampling method. IEEE Trans. Power Deliv. 2022, 37, 2704–2716. [CrossRef]

10. Huang, Y.; Liu, D.; Lee, S.; Hsu, C.; Liu, Y. A boosting resampling method for regression based on a conditional variational autoencoder. Inf. Sci. 2022, 590, 90–105. [CrossRef]

11. Wan, D.: Lu, R.: Xu, T.: Shen, S.: Lang, X.: Ren, Z. Random Interpolation Resize: A free image data augmentation method for object detection in industry. Expert Syst. Appl. 2023, 228, 120355. [CrossRef]

12. Vega-Bayo, M.; Perez-Aracil, J.; Prieto-Godino, L.; Salcedo-Sanz, S. Improving the prediction of extreme wind speed events with generative data augmentation techniques. Renew. Energy 2024, 221, 119769. [CrossRef]

13. Shen, Z.; Kong, X.; Cheng, L.; Wang, R.; Zhu, Y. Fault diagnosis of the rolling bearing by a multi-task deep learning method based on a classifier generative adversarial network. Sensors 2024, 24, 1290. [CrossRef] [PubMed]

14. Li, P.; Pei, Y.; Li, J. A comprehensive survey on design and application of autoencoder in deep learning. Appl. Soft Comput. 2023, 138, 110176. [CrossRef]

15. Odena, A.; Olah, C.; Shlens, J. Conditional image synthesis with auxiliary classifier GANs. In Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, 6–11 August 2017; Volume 70, pp. 2642–2651.

16. Yang, Z.; Han, Y.; Zhang, C.; Xu, Z.; Tang, S. Research on transformer transverse fault diagnosis based on optimized LightGBM model. Measurement 2025, 244, 116499. [CrossRef]

17. Wang, R.; Jia, X.; Liu, Z.; Dong, E.; Li, S.; Cheng, Z. Conditional generative adversarial network based data augmentation for fault diagnosis of diesel engines applied with infrared thermography and deep convolutional neural network. Eksploat. Niezawodn.–Maint. Reliab. 2023, 26, 175291. [CrossRef]

18. Li, Y.; Zou, W.; Jiang, L. Fault diagnosis of rotating machinery based on combination of Wasserstein generative adversarial networks and long short term memory fully convolutional network. Measurement 2022, 191, 110826. [CrossRef]

19. Chen, G.; Sheng, B.; Fu, G.; Chen, X.; Zhao, G. A GAN-based method for diagnosing bodywork spot welding defects in response to small sample condition. Appl. Soft Comput. 2024, 157, 111544. [CrossRef]

20. Li, X.; Wu, X.; Wang, T.; Xie, Y.; Chu, F. Fault diagnosis method for imbalanced data based on adaptive diffusion models and generative adversarial networks. Eng. Appl. Artif. Intell. 2025, 147, 110410. [CrossRef]

21. Yoon, J.; Jarrett, D.; van der Schaar, M. Time-series generative adversarial networks. In Proceedings of the 33rd Conference on Neural Information Processing Systems (NeurIPS), Vancouver, BC, Canada, 8–14 December 2019; Volume 32, pp. 1–9.

22. Shi, Y.; Li, J.; Li, H.; Yang, B. An imbalanced data augmentation and assessment method for industrial process fault classification with application in air compressors. IEEE Trans. Instrum. Meas. 2023, 72, 3521510. [CrossRef]

23. Sim, Y.; Lee, C.; Hwang, J.; Kwon, G.; Chang, S. AI-based remaining useful life prediction for transmission systems: Integrating operating conditions with TimeGAN and CNN-LSTM networks. Electr. Power Syst. Res. 2025, 238, 111151. [CrossRef]

24. Ye, L.; Zhang, K.; Jiang, B. Synergistic feature fusion with deep convolutional GAN for fault diagnosis in imbalanced rotating machinery. IEEE Trans. Ind. Inform. 2024, 21, 1901–1910. [CrossRef]

25. Wang, C.; Yang, J.; Jie, H.; Tao, Z.; Zhao, Z. A lightweight progressive joint transfer ensemble network inspired by the Markov process for imbalanced mechanical fault diagnosis. Mech. Syst. Sig. Process. 2025, 224, 111994. [CrossRef]

26. Guo, H.; Ping, D.; Wang, L.; Zhang, W.; Wu, J.; Ma, X.; Xu, Q.; Lu, Z. Fault diagnosis method of rolling bearing based on 1d multi-channel improved convolutional neural network in noisy environment. Sensors 2025, 25, 2286. [CrossRef]

27. Choi, J.; Lee, S. RNN-based integrated system for real-time sensor fault detection and fault-informed accident diagnosis in nuclear power plant accidents. Nucl. Eng. Technol. 2023, 55, 814–826. [CrossRef]

28. Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. An image is worth 16 × 16 words: Transformers for image recognition at scale, Internation Conference on Learning Representations (ICLR). arXiv 2021, arXiv:2010.11929.

29. Liu, X.; Lu, J.; Li, Z. Multiscale fusion attention convolutional neural network for fault diagnosis of aero-engine rolling bearing. IEEE Sens. J. 2023, 23, 19918–19934. [CrossRef]

30. Zhang, S.; Liu, Z.; Chen, Y.; Jin, Y.; Bai, G. Selective kernel convolution deep residual network based on channel-spatial attention mechanism and feature fusion for mechanical fault diagnosis. ISA Trans. 2023, 133, 369–383. [CrossRef]

31. Liu, R.; Wang, F.; Yang, B.; Qin, S. Multiscale kernel based residual convolutional neural network for motor fault diagnosis under nonstationary conditions. IEEE Trans. Ind. Inform. 2020, 16, 3797–3806. [CrossRef]

32. Xu, X.; Zhang, J.; Huang, W.; Yu, B.; Lyu, F.; Zhang, X.; Xu, B. The loose slipper fault diagnosis of variable-displacement pumps under time-varying operating conditions. Reliab. Eng. Syst. Saf. 2024, 252, 110448. [CrossRef]

33. Chollet, F. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 21–26 July 2017; pp. 1800–1807. [CrossRef]

34. Zhang, X.; Zhang, X.; Liang, W.; He, F. Research on rolling bearing fault diagnosis based on parallel depthwise separable ResNet neural network with attention mechanism. Expert Syst. Appl. 2025, 286, 128105. [CrossRef]

35. Xu, W.; Wan, Y. ELA: Efficient local attention for deep convolutional neural networks. arXiv 2024, arXiv:2403.01123. [CrossRef]

36. Szegedy, C.; Vanhoucke, V.; Ioffe, S.; Shlens, J.; Wojna, Z. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 27–30 June 2016; pp. 2818–2826. [CrossRef]

37. Zhao, C.; Zio, E.; Shen, W. Domain generalization for cross-domain fault diagnosis: An application-oriented perspective and a benchmark study. Reliab. Eng. Syst. Saf. 2024, 245, 109964. [CrossRef]

38. Desai, A.; Freeman, C.; Beaver, I.; Wang, Z. Timevae: A variational auto-encoder for multivariate time series generation. arXiv 2021, arXiv:2111.08095. [CrossRef]

39. Chawla, N.; Bowyer, K.; Hall, L.; Kegelmeyer, W. SMOTE: Synthetic minority over-sampling technique. J. Artif. Intell. Res. 2002, 16, 321–357. [CrossRef]

40. Goodfellow. I: Xu. B.: Farley. D.: Ozair. S.: Courville. A.: Bengio. Y. Generative adversarial nets. In Advances in Neural Information Processing Systems 27 (NIPS 2014), Proceedings of the Annual Conference on Neural Information Processing Systems 2014, Montreal, QC, Canada, 8–13 December 2014; Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N., Weinberger, K.Q., Eds.; Curran Associates, Inc.: Red Hook, NY, USA, 2014.

41. Xu, T.; Li, W.; Munn, M.; Acciaio, B. COT-GAN: Generating sequential data via causal optimal transport. In Proceedings of the Advances in Neural Information Processing Systems, Vancouver, BC, Canada, 6–12 December 2020; Volume 33, pp. 8798–8809.

42. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 27–30 June 2016; pp. 770–778. [CrossRef]

43. Huang, T.; Zhang, Q.; Tang, X.; Zhao, S.; Lu, X. A novel fault diagnosis method based on CNN and LSTM and its application in fault diagnosis for complex systems. Artif. Intell. Rev. 2022, 55, 1289–1315. [CrossRef]

44. Qin, N.; You, Y.; Huang, D.; Jia, X.; Zhang, Y.; Du, J.; Wang, T. AttGAN-DPCNN: An extremely imbalanced fault diagnosis method for complex signals from multiple sensors. IEEE Sens. J. 2024, 24, 38270–38285. [CrossRef]

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.