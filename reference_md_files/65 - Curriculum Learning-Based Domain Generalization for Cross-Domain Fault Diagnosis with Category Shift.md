# Curriculum learning-based domain generalization for cross-domain fault diagnosis with category shift

![](images/839f96348316b9a5be8e60ade2cef875d79b3f89b7007cc53ab6f6128fe94c66.jpg)

Yu Wang <sup>a,b,c,d</sup>, Jie Gao <sup>a,b,d,\*</sup>, Wei Wang <sup>a,b,d</sup>, Xu Yang <sup>a,b,d</sup>, Jinsong Du <sup>a,b,d</sup>

<sup>a</sup> Shenyang Institute of Automation, Chinese Academy of Sciences, Shenyang 110016, China

<sup>b</sup> Institutes for Robotics and Intelligent Manufacturing, Chinese Academy of Sciences, Shenyang 110169, China

<sup>c</sup> University of Chinese Academy of Sciences, Beijing 100049, China

<sup>d</sup> Key Laboratory on Intelligent Detection and Equipment Technology of Liaoning Province, Shenyang 110179, China

## A R T I C L E I N F O

Communicated by Olga Fink

Keywords: Domain generalization Category shift Curriculum learning Fault diagnosis

## A B S T R A C T

Intelligent fault diagnosis has witnessed significant advancements in the preceding years. Domain generalization-based methods can effectively alleviate the domain shift problem and be employ for fault diagnosis in unknown domains. Apart from the problem of domain shift, another chal lenge arises from the incomplete label space of each source domain due to the difficulty of data acquisition. Category shift can have a significant impact on the subsequent application of intel ligent algorithms. To confront this more challenging and practical problem, we begin by formulating the setting of domain generalization with category shift. This paper proposes a Curriculum Learning-based Domain Generalization method (CLDG) to tackle with the intricate problem. The basic network consists of a feature extractor, a mixup-based reciprocal point learning classifier for tackling the category shift between the source and target domains, and a conditional domain discriminator for addressing the domain shift. In addition, we construct a curriculum learning strategy that uses the knowledge of categories with high observation degree to assist in extracting domain invariant features of lower ones, dealing with the category shift between the source domains and improving the generalization ability of the categorical infor mation. Extensive experimental results on two datasets provide evidence for the effectiveness and superiority of the proposed algorithm in classifying known and missing classes in each source domain, as well as identifying unobserved failure modes in unknown target domains.

## 1. Introduction

Along with the enormous development of modern industry, equipment is undergoing a transformation towards increased complexity, flexibility and automaticity. This heightened sophistication brings about higher requirements for reliability and system safety. The fault diagnosis techniques have proven to provide a viable and effective guarantee for these requirements. As the rise of industrial big data, data-based intelligent fault diagnosis (IFD) has emerged as a pivotal part and widely concerned topic in the yield of prognostics and health management (PHM) for machine in recent years [1,2].

IFD methods enable the detection of fault categories by exploiting the inherent relationships between monitoring signals and fault patterns [3]. Over the past few years, IFD methods has gradually evolved towards more general scenarios. It started with Machine

Learning-based (ML) methods that required manual extraction of expert knowledge [4], then progressed to Deep Learning-based (DL) methods capable of autonomously extracting features but still reliance on independent and identically distributed (i.i.d.) assumption [5,6], and further advanced to Transfer Learning-based (TL) methods that allow for domain shift [7,8]. Considering the diversity in working conditions and equipment types, a crucial issue for the practical application of IFD lies in enhancing its generalization capability across potential target domains by leveraging information from several accessible source domains [9]. As a feasible strategy, Domain Generalization-based (DG) approaches naturally emerge as an attractive research spot in IFD. Two mainstream DG strategies are the data augmentation and domain-invariant (DI) representation learning strategies. The former tries to tackle DG issues by elevating data diversity through the augmentation of source domain data [10,11]. Shi et al. [12] enhance the model’s generalization performance through multisource augmentation and sample adaptive screening and weighting (SASW) strategies. The latter is the common strategy in field of TL to address the domain shift [13,14]. Jia et al. [15] developed an IFD strategy with causal view, they employed the maximizing predicted entropy to separate causal factors from non-causal ones, and learned the stable and sufficient distinguishing fault representations.

Numerous existing literatures consider IFD tasks in a close-set setup in which the domains exactly share their label space [16–18]. Nevertheless, extensive researchers suggest that this setup, merely addressing the domain shift, is simplistic and unrealistic [19] Hence, they redirect their focus towards the more comprehensive problem involving inconsistent label spaces across domains, namely, the category shift problem. This challenge arises from the impracticality of gathering comprehensive fault data for each accessible domain. Within another subset of TL known as domain adaptation (DA), the category shift problem can be categorized into partial domain adaptation (PDA), open set domain adaptation (OSDA), and universal domain adaptation (UDA). For instance. Li et al. [20] introduced a PDA algorithm effectively minimizes the transfer of source outlier knowledge to the target domain, this algorithm combines multi-representation structure intraclass compact and double-aligned domain adaptation. She et al. [21] presented a so lution to OSDA by devising an integrated criterion with three soft thresholds, allowing for the discrimination of unknown classes in the target domain. For the UDA problem. Ma et al. [22] proposed a Meta Bi-classifier Gradient Discrepancy (MBGD). this approach employed a coupled divergence criterion and a constrained residual bi-classifier alleviate the impact of private classes in the source and target domains, respectively. The studies mentioned above are limited to single-source domain transfer. It is evident that re searches with multi-source domains setting will also encounter challenge of category shift, as outlined in Deep Cocktail Network [23] To illustrate, Zhao et al. [24] expanded the concepts of OSDA to the DG field, proposed an adaptive open set domain generalization network with as outlier detection module to learn optimal decision boundaries for individual class representation spaces to classify known fault modes and recognize unknown fault modes. However, the category shift with multi-source domains setting is not limited to the source and target domain, it may also emerge among various source domains. With respect to this, Feng et al. [25] designed a novel IFD approach termed GlocalNet with the capability to align features at domain, class, and samples level. According to the above analysis, this paper proposes a novel setting of DG with category shift, as illustrated in Fig. 1. In this setting, the categories of each source domain may be incomplete, and the target domain includes unobserved failure modes.

As depicted in Fig. 1(b), the proposed setting confronts challenges arising not solely from domain shift but also from category shift, which encompasses both source-category-shift and target-category-shift. In our paper, a new curriculum learning-based domain generalization method is proposed to address these challenges. Our basic network architecture consists of a feature extractor, a mixupbased reciprocal point learning classifier and a conditional domain discriminator. To handle the target-category shift, the mixup-based reciprocal point learning classifier indirectly models the unknown space by constructing reciprocal points of all known classes achieving the separation between known and unknown classes. To enhance the informativeness of the unknown space and further reduce the intra-class distance of known classes, and increase the inter-class distance, the mixup-based method is used for multi-source data augmentation. Besides, combining the constructed unknown space, this paper employed the conditional domain discriminator to extract domain-invariant features for both known and unknown classes, thereby mitigating the impact of domain shift and enhancing the overall generalization capacity of the model. Meanwhile, the training phase of the DG is affected by source-category-shift, wherein some categories only appear in specific source domains and may exist as private classes within certain domain. Consequently, the model encounters challenges in acquiring adequate category information, resulting in a subsequent decline in fault diagnosis accuracy.

![](images/ffe5dd80a9767fbc84ce176fceec672e70fae19ebba4dcb812a0d4530a5d1d38.jpg)  
Fig. 1. Illustration for two domain generalization settings. (a) Conventional Domain Generalization. (b) Domain Generalization with Category Shift. The shape of the symbol represents the fault categories, and the color of the symbol symbolizes the domains. The gray symbols denote the missing categories of each source domain, and the hexagonal symbols represent the unknown categories. It can be seen that category shift in DG encom passes both source-category-shift and target-category-shift.

In tackling this issue, we implement a curriculum learning strategy that initially trains the model using categories with high category observation degree, i.e., those prevalent in the majority of source domains. The trained model then guides the domain-invariant feature extraction for the categories with low category observation degree, diminishing the influence of spurious features and up holding the accuracy of fault diagnosis. The highlights and contributions of this paper can be summarized as follows.

1) A novel network architecture is proposed to tackle the domain shift and target-category shift in cross-domain fault diagnosis. Specifically, conditional domain discriminator and mixup-based reciprocal point learning classifier alleviate the above problem, respectively.

2) A novel curriculum learning strategy is introduced to address the source-category shift in cross-domain fault diagnosis.

3) The effectiveness of this method is validated by extensive experiments built on two datasets. The proposed method is expected to help realize online fault diagnosis.

The remainder of this paper starts with the related works in Section 2. The proposed method is introduced in Section $^ { 3 , }$ and experimentally validated and discussion in Section 4 and 5. Finally, Section 6 concludes the whole paper.

## 2. Related work

## 2.1. Descriptions of adversarial training

Adversarial Training (AdvT) strategy emerges as the prevalent technique in ${ \mathrm { T L } } ,$ effectively capturing domain-invariant features across diverse domains. Our paper also employes this strategy to overcome the domain shift problem. AdvT constructs a min–max deep learning framework by introducing the domain discriminator D into a foundational model composed of a feature extractor F and classifier C. Within this framework, the features obtained through F should be adept at confusing $D ,$ while providing desirable discriminative property for C. Meanwhile, considering that the representations and predictions of samples with similar failure modes will be closer. The method proposed by [26] is employed in this paper to condition D on category predictions with discriminative information, utilizing a multilinear map $x \otimes y .$ . This map captures multiplicative interactions between feature representations and classifier predictions. Therefore, the loss of the domain discriminator utilized is denoted by:

$$
\mathcal {L} _ {d} = \frac {1}{n _ {j}} \sum_ {i = 1} ^ {n _ {j}} \left[ \mathcal {L} (\mathbf {D} (\mathbf {F} (\mathbf {x} _ {\mathrm{i}}; \theta_ {F}) \otimes \mathbf {C} (\mathbf {z} _ {i}; \theta_ {C}); \theta_ {D}), d _ {i}) \right]\tag{1}
$$

where $\theta _ { F } , \theta _ { C } , \theta _ { D }$ is the learnable parameters of the feature extractor, the classifier, and the domain discriminator, respectively. $x _ { i }$ is the sample, $\mathbf { z } _ { i }$ and $d _ { i }$ is the feature and domain label of the sample. n is the sample number. $\mathcal { L }$ is the cross-entropy function.

Thus, the overall optimization objective can be summarized as follows:

$$
\begin{array}{l} \max _ {F, C} - \mathcal {L} _ {c} + \lambda_ {1} \mathcal {L} _ {d} \\ \min _ {D} \mathcal {L} _ {d} \end{array}\tag{2}
$$

where $\mathcal { L } _ { \mathrm { c } }$ is the classification loss. $\lambda _ { 1 }$ is a hyperparameter.

## 2.2. Introduction of reciprocal point learning

Reciprocal Point Learning (RPL) [27] emerges as a viable strategy to mitigate the open space risk arising from unknown classes by modeling the extra-space using the known categories. For observable categories, each category is associated with a reciprocal point that serves as a latent prototype representation for features that do not belong to the given category. Enhancing inter-class separability could be achieved by decreasing the similarity between sample features and their corresponding reciprocal points. To quantify this similarity, the distance between the representation of sample x and the reciprocal point $\mathcal { P } ^ { \mathbf { k } }$ is calculated using both Euclidean distance $\mathbf { d } _ { \mathbf { e } }$ and dot product $\mathbf { d } _ { \mathbf { d } } ,$ , as expressed below:

$$
\begin{array}{l} \mathbf {d} \big (\mathbf {F} (\mathbf {x}), \mathcal {P} ^ {\mathbf {k}} \big) = \mathbf {d} _ {\mathbf {e}} \big (\mathbf {F} (\mathbf {x}), \mathcal {P} ^ {\mathbf {k}} \big) - \mathbf {d} _ {\mathbf {d}} \big (\mathbf {F} (\mathbf {x}), \mathcal {P} ^ {\mathbf {k}} \big) \\ = \frac {1}{\mathbf {m}} \big \| \mathbf {F} (\mathbf {x}) - \mathcal {P} ^ {\mathbf {k}} \big \| _ {2} ^ {2} - \mathbf {F} (\mathbf {x}) \cdot \mathcal {P} ^ {\mathbf {k}} \end{array}\tag{3}
$$

where m is the dimension of the representation and reciprocal point $\mathcal { P } ^ { \mathbf { k } }$ .

By incorporating the distance metric and the Softmax function, the final classification probability of sample x can be expressed as follows:

$$
\mathbf {p} (\mathbf {y} = \mathbf {k} | \mathbf {x}, \mathbf {F}, \mathbf {P}) = \frac {e ^ {\mathbf {d} \left(\mathbf {F} (\mathbf {x}) , \mathcal {P} ^ {\mathbf {k}}\right) / \tau}}{\sum_ {\mathbf {i} = 1} ^ {\mathbf {N}} e ^ {\mathbf {d} \left(\mathbf {F} (\mathbf {x}) , \mathcal {P} ^ {\mathbf {i}}\right) / \tau}}\tag{4}
$$

where τ is the temperature value that determining the hardness of the distance-probability conversion. As previously indicated, the embedding features ought to be far from their associated reciprocal points, which could be used for supervised classification. Thus, the classification loss in this paper is:

$$
\mathcal {L} _ {\mathbf {c}} (\mathbf {x}, \theta , \mathcal {P}) = - \mathbf {l o g p} (\mathbf {y} = \mathbf {k} | \mathbf {x}, \mathbf {F}, \mathcal {P})\tag{5}
$$

The aforementioned equations solely optimize the distance between the known categories and the prototypes of their corre sponding unknown spaces, while disregarding constraints on the unknown spaces themselves. This absence of constraints potentially introduces the unpredictable overlap between the known and the unknown categories. Owing of the limited accessibility of the un known categories, the complementary know categories is utilized to impose an indirect constraint on the unknown space, as shown below:

$$
\mathcal {L} _ {\mathbf {0}} = \max \left(\left(\mathbf {d} _ {\mathrm{e}} (\mathbf {F} (\mathbf {x}), \mathscr {P} ^ {\mathbf {k}}) - \mathbf {R}\right), 0\right)\tag{6}
$$

where R is learnable margin. And a boundary centered on the reciprocal point $\mathcal { P } ^ { \mathbf { k } }$ with R as the margin is used to determine whether a sample belongs to the corresponding category k or not. By Eq. (6), the unknown samples can be closer to the reciprocal point compared to the corresponding known categories. And in the testing phase, we consider samples that fall within the range of all reciprocal points as unknown category samples.

## 3. Methodology

## 3.1. Problem Definition

In this paper, we proposed a DG algorithm to tackle the cross-domain fault diagnosis problem under the following premises.

• In the training phase, only the labeled samples in multi-source domains are accessible.

• The categories for each source domain may be incomplete, but considering the real industry, samples labeled as normal should be available for all source domains.

![](images/88104346cee6bda8d05c0c8df09d95a5fbe8b7b6b6eef078d70bd04f8b6a73a4.jpg)  
Fig. 2. Basic Network Architecture.

• In the testing phase, the unseen target domain may contain the unknown categories.

Based on this, the fault diagnosis tasks contain K source domains $\mathcal { D } ^ { \bf S } = \{ \mathcal { D } _ { \bf i } ^ { { \bf S } } \} _ { \bf i = 1 } ^ { { \bf K } }$ and an unseen target domain ${ \mathcal { D } } ^ { \mathbf { T } }$ . We defined the ith source domain as $\mathcal { D } _ { \bf i } ^ { \bf S } = \{ \mathcal { X } _ { \bf i } ^ { \bf S } , ~ \mathcal { Y } _ { \bf i } ^ { \bf S } \}$ , where $\mathcal { X } _ { i } ^ { S } = \left\{ \boldsymbol { x } _ { i } ^ { j } \right\} _ { j = 1 } ^ { N _ { S _ { i } } }$ represents the set of $\mathbf { N _ { S _ { i } } }$ samples from source $i ,$ and the $\mathcal { Y } _ { \mathrm { ~ i ~ } } ^ { \mathrm { s } }$ denotes the corresponding label space. The unseen target domain is denoted as $\mathcal { D } ^ { \mathbf { T } } = \{ \mathcal { X } ^ { \mathbf { T } } , ~ \mathcal { Y } ^ { \mathbf { T } } \}$ as well. We represent the union of label spaces for all source domain as $\mathcal { Y } ^ { \bf S } = \cup _ { \bf i = 1 } ^ { \bf K } \mathcal { Y } _ { \bf i } ^ { \bf S }$ , and the entire label space of all fault types including the unknown ones as $\mathcal { Y } .$ . In conventiona domain generalization setting, all domains share the same label spaces $\{ \mathcal { Y } _ { \mathfrak { i } } ^ { \mathbf { s } } \} _ { \mathfrak { i } = 1 } ^ { \mathbf { K } } = \mathcal { Y } ^ { \mathbf { s } } = \mathcal { Y } ^ { \mathbf { r } } { \subset } \mathcal { Y }$ , and this setting can be seen as a specific case of our setting.

Definition 1. (Domain Generalization with Category Shift, DGCS). The category shift in DG task can be divided into two parts, the sourcecategory-shift and the target-category-shift. In source-category-shift, the label space of the i-th source domain $\mathcal { D } _ { \mathbf { i } } ^ { \mathbf { s } }$ is a subset of the union of labe spaces for all source domain $\mathcal { Y } _ { \mathrm { ~ i ~ } } ^ { \mathbf { S } } \subseteq \mathcal { Y } ^ { \mathbf { S } }$ , and the label space of the missing categories denotes as $\overline { { \mathcal { Y } } } _ { \bf i } ^ { \bf S } = \mathcal { Y } ^ { \bf S } - \mathcal { Y } _ { \bf i } ^ { \bf S }$ . In the target-category-shift, the label space of unobserved categories for source domains is represented as $\mathcal { U } ^ { \mathbf { S } } = \mathcal { Y } - \mathcal { Y } ^ { \mathbf { S } }$ . Besides, the label space of the target domain can be denoted as $\mathcal { Y } ^ { \mathbf { r } } = \mathcal { Y } ^ { \mathbf { s } } \cup \mathcal { U } ^ { \mathbf { r } }$ , where $\mathcal { U } ^ { \mathbf { T } } \subseteq \mathcal { U } ^ { \mathbf { s } }$ represents the new categories in unseen target domain

Definition 2. (Category Observation Degree). The Category Observation Degree ${ \mathcal { O } } _ { \mathbf { C } }$ represents is the occur rate of category C in the multisource domains $\mathcal { O } _ { \mathbf { C } } = \mathbf { N } _ { \mathbf { C } } / \mathbf { K } _ { : }$ ,where $\mathbf { N } _ { \mathbf { C } }$ is the number of source domains contained category C.

Given the strong likelihood of encountering the proposed setting in practical, it is essential to extract sufficient categorical in formation from source domains with incomplete classes. To cope with the domain shift and category shift in this setting, we propose a curriculum learning-based domain generalization method. This method aims to ensure the detection accuracy of known categories in source domains while facilitating the classification of the missing categories within each source domain as well as the unknown categories in the target domain, thus enhancing the generalizability of the IFD models.

## 3.2. Our methods

## 3.2.1. Basic network architecture

The proposed method comprises three essential components: a feature extractor, a reciprocal point learning-based classifier, and a domain discriminator. The overall architecture can be seen in Fig. 2. During forward propagation, the weight-sharing feature extractor F is denoted as a mapping from data to high-dimensional representations $z = F ( x )$ , the classifier C outputs category predictions $g =$ $C ( F ( x )$ ). As described in Section 2.1, the results of the multilinear map on the representations z and predictions g are fed into the domain discrimination to ensure the extraction of domain-invariant feature. Furthermore, this paper utilizes the reciprocal point learning mentioned in Section 2.2 to construct a classifier. Through the combination of Eqs. (5) and (6) for model optimization, the two learnable parameters $\mathcal { P } ^ { \mathbf { k } }$ and R are optimized, and an improvement is realized in the segregating the known categories from the potential unknown spaces, thus mitigating the target-category shift problem. The optimization objective for the samples in known categories is:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{c}} + \lambda_ {2} \mathcal {L} _ {o} - \lambda_ {1} \mathcal {L} _ {d}\tag{7}
$$

where $\lambda _ { 2 }$ is a hyperparameter.

![](images/fabf18dfb181b2cf148e065110317bb04ecec7b05c14121f0997427d14b39fec.jpg)  
(a)

![](images/db320ef89e525214f92c1481ce69f4edcc6e2b6688241e11ca287b05ea6ad61f.jpg)  
(b)  
Fig. 3. Mixup Reciprocal Point Learning. (a) Reciprocal Point Learning; (b) Mixup Strategy;

## 3.2.2. Mixup-based RPL

To further alleviate the overlap between known and unknown samples, and to increase the informativeness of the constructed global unknown space, the mixup strategy is introduced into RPL to generate novel samples for unknown space, and improve the mode generalization ability, as shown in Fig. 3. Given two input samples and their labels, namely $( \bf x _ { i } , \psi _ { j _ { i } } , \psi _ { d _ { i } } )$ and $( \bf x _ { j } , \psi _ { j } , \psi _ { j } )$ , the mixup strategy can be formulated as:

$$
\widetilde {x} = \mathcal {M} (x _ {i}, x _ {j}, \mu) = \mu x _ {i} + (1 - \mu) x _ {j}\tag{8}
$$

$$
\widetilde {y} = \mathcal {M} \left(y _ {i}, \mathbf {y} _ {j}, \mu\right) = \mu y _ {i} + (1 - \mu) y _ {j}\tag{9}
$$

$$
\widetilde {d} = \mathscr {M} (d _ {i}, d _ {j}, \mu) = \mu d _ {i} + (1 - \mu) d _ {j}\tag{10}
$$

where $\mathcal { M }$ is the convex combination function; μ is a mixing factor observed from a beta distribution Beta $( \alpha , \alpha )$ for $\alpha > 0 .$

When two samples are in the same category, the generated samples are considered as the sample from known categories to learn more universal diagnostic knowledge and thus improve the model’s generalizability. In contrast, the generated samples are regarded as unknown samples, and the features of these samples should be close to all reciprocal points. In addition, using the condition adversarial learning strategy, we enhance the capability of the model to learn domain invariant features of unknown classes. For these unknown sample, they are employed to optimize the proposed methods using the below objective function:

$$
\mathcal {L} _ {\mathbf {u}} = - \frac {1}{\mathbf {n}} \sum_ {\mathrm{i} = 1} ^ {\mathrm{n}} \left[ \left(- \frac {1}{\mathbf {N}} \sum_ {\mathrm{k} = 1} ^ {\mathrm{N}} \mathbf {S} \left(\widetilde {\mathbf {x}} _ {\mathrm{i}}, \mathcal {P} ^ {\mathrm{k}}\right) \cdot \log \left(\mathbf {S} \left(\widetilde {\mathbf {x}} _ {\mathrm{i}}, \mathcal {P} ^ {\mathrm{k}}\right)\right)\right) + \lambda_ {1} \mathcal {L} \left(\mathbf {D} \left(\mathbf {F} \left(\widetilde {\mathbf {x}} _ {\mathrm{i}}\right) \otimes \mathbf {C} \left(\widetilde {\mathbf {z}} _ {i}\right)\right), \widetilde {d} _ {i}\right) \right]\tag{11}
$$

where $\bf S ( \bf x _ { i } , \delta \bf P ^ { k } )$ = softmax $\left( \mathbf { d } _ { \mathbf { e } } \left( \mathbf { F } ( \mathbf { x _ { i } } ) , \ { \mathcal { P } } ^ { \mathbf { k } } \right) \right)$ .

## 3.2.3. Curriculum learning phase

The above sections only address the domain shift and target-category shift problems, but still do not consider the source-category shift. For categories with small ${ \mathcal { O } } _ { \mathbf { C } } ,$ the categorical information learned is still restricted, and with the potential of extracting spurious environmental features, which drastically affects the diagnostic performance of the model. The proposed method regards the cognitive process as a process of continuously obtaining known information from the unknown. On this basis, the training process is divided into multiple phases. When the training phase N-1 is completed, we can obtain a mixup-based RPL classifier that discriminates the observed categories in the previous phases. In addition, with Eq. (11), the model can identify unknown samples and extract their domaininvariant features. Therefore, in the phase N, directing the samples of novel categories into the unknown space constructed by the

![](images/196c217a3a725294e786ae6d8c38a06227ad3b5a7840b3cc9f7267234ae541ec.jpg)  
Fig. 4. The Proposed Curriculum Learning Strategy.

(a)

classifier ensures the distinction between the old and the new samples and guarantees the domain invariance of the extracted features for these categories, especially those with small ${ \mathcal { O } } _ { \mathbf { C } } $ , thus mitigating the effect of source-category shift. In this paper, we can use the ${ \mathcal { O } } _ { \mathbf { C } }$ to rank the known categories, thereby designing the curriculum learning phases, and use the model trained by samples with large ${ \mathcal { O } } _ { \mathbf { C } }$ to guide the extraction of domain invariant features for samples with small ${ \mathcal { O } } _ { \mathbf { C } } ,$ improving the overall generalization of the model. However, considering the small number of fault categories in this paper, only two-phase training is considered, and the comparison results of different training strategies are added in the experimental part. In this paper, apart from the category labeled as normal, all observable categories are organized in descending order according to ${ \mathcal { O } } _ { \mathbf { C } }$ . Following this arrangement, the training samples are partitioned into two groups, each containing a roughly equivalent number of categories. The samples categorized as “normal” should be assigned to the first group. Besides, for the categories with the same value of ${ \mathcal { O } } _ { \mathbf { C } } ,$ it is important to ensure that the first group can cover all source domains as comprehensively as possible.

For the training phase N, the training samples contain samples from novel categories and part of old class samples, and the pa rameters of the model from the previous phase N-1 are frozen. The samples from novel categories are first fed into the feature extractor of this stage ${ \bf F } _ { n e w } ,$ , and then the previous stage’s classifier $\mathcal { P } _ { o l d } ^ { \mathbf { k } }$ is used to constrain the features so that they fall into the unknown space of the old model, thus reducing the influence of environmental features and ensuring categorical information, as shown in Fig. 4. Therefore, apart from optimizing the initial phase model through Eqs. (7) and (11), the new phase model should be constrained using the previous one with the knowledge distillation strategy. The optimization objective of knowledge distillation across phases is:

$$
\mathcal {L} _ {k d} = - \frac {\eta}{n} \sum_ {i = 1} ^ {n} \left[ - \frac {1}{N} \sum_ {k = 1} ^ {N} \sigma \left(\mathbf {d} _ {\mathbf {e}} \left(\mathbf {F} _ {\text { new }} \left(\mathbf {x} _ {\mathbf {i}, \text { new }}\right), \mathcal {P} _ {\text { old }} ^ {\mathbf {k}}\right)\right) \cdot \log \left(\sigma \left(\mathbf {d} _ {\mathbf {e}} \left(\mathbf {F} _ {\text { new }} \left(\mathbf {x} _ {\mathbf {i}, \text { new }}\right), \mathcal {P} _ {\text { old }} ^ {\mathbf {k}}\right)\right)\right) \right]\tag{12}
$$

where σ is Softmax function, ${ \bf { X } } _ { \bf { i } , \ n e w }$ is the sample of novel categories in current phase. η is a hyperparameter of knowledge distillation

## 4. Experimental study

In this section, extensive experiments are conducted under the DGCS setting on two datasets to verify the performance of the proposed approach, several DG-related and category shift-related algorithms are as compared methods. All experiments are conducted on Windows10 operating system with i7-10870H CPU and RTX 2070 Super GPU on CUDA 10.1.

## 4.1. Data descriptions

1) KAT Dataset: The first rolling bearing dataset is provided by KAt-DataCenter of the Paderborn University [28]. The bearing vibration data were collected from a modular test rig, as shown in Fig. 5(a). The bearing type used in this dataset is 6203, and a variety of vibration signals with different health conditions were collected under four working conditions with a sampling frequency of 64 kHz.We refer to the bearing code in [29] to construct the experimental dataset. The detailed information is listed in Tables 1–2.

2) DIRG Dataset: This dataset is released by Politecnico di Torino [30]. Bearings are specifically manufactured for high-speed aeronautical applications. The test rig is shown in Fig. 5(b). The bearings in position B1 are tested with various dimensions of indentation. The fault damages are introduced to the inner ring (I) and the roller (R) and details are given in Table 3. In our study, we selected six working conditions to construct the experimental dataset.

![](images/6e21cfb1cdd09c21b8b305957e5dcf5668c46775fd32c2367e461d3474ce00d1.jpg)  
Fig. 5. The Test rigs of the Datasets. (a) KAT Dataset. (b) DIRG Dataset.

Table 1  
Categories of KAT Datasets. (OR: Outer Ring, IR: Inner Ring, S: Single Damage, D: Distributed Damage).

<table><tr><td>Bearing Digit Code</td><td>Damage Position</td><td>Damage Level</td><td>Characteristic of damage</td><td>Category Code</td></tr><tr><td>K002</td><td>-</td><td>-</td><td>-</td><td>H</td></tr><tr><td>K15</td><td>OR</td><td>1</td><td>S</td><td>OR1</td></tr><tr><td>K16</td><td>OR</td><td>2</td><td>S</td><td>OR2</td></tr><tr><td>KI16</td><td>IR</td><td>3</td><td>S</td><td>IR1</td></tr><tr><td>KI18</td><td>IR</td><td>2</td><td>S</td><td>IR2</td></tr><tr><td>KI21</td><td>IR</td><td>1</td><td>S</td><td>IR3</td></tr><tr><td>KB23</td><td>IR + OR</td><td>2</td><td>S</td><td>Mix1</td></tr><tr><td>KB24</td><td>IR + OR</td><td>3</td><td>D</td><td>Mix2</td></tr><tr><td>KB27</td><td>IR + OR</td><td>1</td><td>D</td><td>Mix3</td></tr></table>

Table 2  
Working conditions of KAT Datasets.

<table><tr><td>Name of Setting</td><td>Rotational Speed/rpm</td><td>Load Torque/Nm</td><td>Radial Force/N</td><td>Working Condition Code</td></tr><tr><td>N15_M07_F10</td><td>1500</td><td>0.7</td><td>1000</td><td>0</td></tr><tr><td>N09_M07_F10</td><td>900</td><td>0.7</td><td>1000</td><td>1</td></tr><tr><td>N15_M01_F10</td><td>1500</td><td>0.1</td><td>1000</td><td>2</td></tr><tr><td>N15_M07_F04</td><td>1500</td><td>0.7</td><td>400</td><td>3</td></tr></table>

Table 3  
Categories (a) and working conditions (b) of DIRG Datasets.

<table><tr><td colspan="4">(a)</td><td colspan="3">(b)</td></tr><tr><td>Bearing Digit Code</td><td>Damage Position</td><td>Dimension/μm</td><td>Category Code</td><td>Working Condition Code</td><td>Nominal Load/N</td><td>Nominal Speed/Hz</td></tr><tr><td>C0A</td><td>-</td><td>-</td><td>N</td><td>0</td><td>0</td><td>100</td></tr><tr><td>C1A</td><td>I</td><td>450</td><td>F1</td><td>1</td><td>1800</td><td>200</td></tr><tr><td>C2A</td><td>I</td><td>250</td><td>F2</td><td>2</td><td>1400</td><td>300</td></tr><tr><td>C3A</td><td>I</td><td>150</td><td>F3</td><td>3</td><td>1000</td><td>400</td></tr><tr><td>C4A</td><td>R</td><td>450</td><td>F4</td><td>4</td><td>0</td><td>500</td></tr><tr><td>C5A</td><td>R</td><td>250</td><td>F5</td><td>5</td><td>1400</td><td>400</td></tr><tr><td>C6A</td><td>R</td><td>150</td><td>F6</td><td></td><td></td><td></td></tr></table>

## 4.2. Case construction

Based on the proposed DGCS setting, we designed different experimental cases for the two datasets, as listed in Table 4 and $^ { 5 , }$ respectively. In the cases, the domain shift and two kinds of category shifts, that is, source-category-shift and target-category-shift are considered For instance the case $C _ { 0 1 }$ in Table 4 is a traditional domain generalization task. The source domains contain KAT data under the working condition codes $\{ 0 , 1 , 2 \}$ , and the target domain is the data under code 3. All source and target domains share the same label space Λ. For the case $C _ { 0 5 }$ in Table 4, the label spaces of the three source domains are {H, OR1}, {H, OR, IR1}, and {H, OR IR}, respectively. The novel label space brought by the target domain is $\mathcal { U } ^ { \mathbf { T } } =$ ={Mix}. The bold symbols are the concatenation of labels at the same fault location, i.e., OR={OR1, OR2}, IR={IR1, IR2, IR3}, Mix={Mix1, Mix2, Mix3}. The label settings for the last case in each dataset $( C _ { 0 5 }$ and $C _ { 1 6 } )$ are presented in a stepped form, in which the label space of a source domain could be a subset of another source domain. For the rest of the experimental cases, as the case number increases, the overlap of the label space between the source domains becomes less and less, and the difficulty of fault diagnosis increases. In particular, for the penultimate case $( C _ { 0 4 }$ and $C _ { 1 5 } )$ , the categories in each source domain do not appear in another source domain, except for the data labeled as normal, as mentioned in the problem setup.

Table 4  
The fault diagnosis cases on the KAT Dataset.

<table><tr><td>Cases</td><td> $\mathcal{D}^{\mathrm{S}}$ </td><td> $\mathcal{Y}^{\mathrm{S}}$ </td><td> $\mathcal{D}^{\mathrm{T}}$ </td><td> $\mathcal{Y}^{\mathrm{T}}$ </td><td> $\mathcal{U}^{\mathrm{T}}$ </td></tr><tr><td> $C_{01}$ </td><td> $\text{KAT}_{\{0, 1, 2\}}$ </td><td> $\Lambda$ </td><td> $\text{KAT}_{\{3\}}$ </td><td> $\Lambda$ </td><td>-</td></tr><tr><td> $C_{02}$ </td><td> $\text{KAT}_{\{0, 1, 2\}}$ </td><td> $\{H, OR1, IR, Mix1, Mix2\}$  $\{H, OR, IR2, IR3, Mix2\}$  $\{H, OR, IR1, IR3, Mix1, Mix2\}$ </td><td> $\text{KAT}_{\{3\}}$ </td><td> $\Lambda$ </td><td> $\{Mix3\}$ </td></tr><tr><td> $C_{03}$ </td><td> $\text{KAT}_{\{1, 2, 3\}}$ </td><td> $\{H, IR1, IR2, Mix1, Mix2\}$  $\{H, OR, Mix1, Mix2\}$  $\{H, OR, IR1, IR2\}$ </td><td> $\text{KAT}_{\{0\}}$ </td><td> $\Lambda$ </td><td> $\{IR3, Mix3\}$ </td></tr><tr><td> $C_{04}$ </td><td> $\text{KAT}_{\{0, 2, 3\}}$ </td><td> $\{H, Mix1, Mix2\} + \{H, IR\} + \{H, OR\}$ </td><td> $\text{KAT}_{\{1\}}$ </td><td> $\Lambda$ </td><td> $\{Mix3\}$ </td></tr><tr><td> $C_{05}$ </td><td> $\text{KAT}_{\{0, 1, 2\}}$ </td><td> $\{H, OR1\} + \{H, OR, IR1\} + \{H, OR, IR\}$ </td><td> $\text{KAT}_{\{3\}}$ </td><td> $\Lambda$ </td><td> $\{Mix\}$ </td></tr></table>

Table 5  
The fault diagnosis cases on the DIRG Dataset.

<table><tr><td>Cases</td><td> $\mathcal{D}^{\text{S}}$ </td><td> $\mathcal{Y}^{\text{S}}$ </td><td> $\mathcal{D}^{\text{T}}$ </td><td> $\mathcal{Y}^{\text{T}}$ </td><td> $\mathcal{U}^{\text{T}}$ </td></tr><tr><td> $C_{11}$ </td><td>DIRG $_{0, 1, 2, 3, 4}$ </td><td> $\Lambda$ </td><td>DIRG $_{5}$ </td><td> $\Lambda$ </td><td>-</td></tr><tr><td> $C_{12}$ </td><td>DIRG $_{1, 2, 3, 4, 5}$ </td><td> $\{N, F2, F3, F4, F5\} + \{N, F1, F3, F4, F5\} + \{N, F1, F2, F4, F5\} + \{N, F1, F2, F3, F5\} + \{N, F1, F2, F3, F4\}$ </td><td>DIRG $_{0}$ </td><td> $\Lambda$ </td><td> $\{F6\}$ </td></tr><tr><td> $C_{13}$ </td><td>DIRG $_{0, 2, 3, 4, 5}$ </td><td> $\{N, F1, F2, F3\} + \{N, F1, F4, F5\} + \{N, F1, F2, F5\} + \{N, F3, F4, F5\} + \{N, F1, F2, F5\}$ </td><td>DIRG $_{1}$ </td><td> $\Lambda$ </td><td> $\{F6\}$ </td></tr><tr><td> $C_{14}$ </td><td>DIRG $_{0, 1, 3, 4, 5}$ </td><td> $\{N, F1, F2\} + \{N, F3, F4\} + \{N, F1, F4, F5\} + \{N, F2, F4\} + \{N, F1, F4\}$ </td><td>DIRG $_{2}$ </td><td> $\Lambda$ </td><td> $\{F6\}$ </td></tr><tr><td> $C_{15}$ </td><td>DIRG $_{0, 1, 2, 4, 5}$ </td><td> $\{N, F1\} + \{N, F2\} + \{N, F3\} + \{N, F4\} + \{N, F5\}$ </td><td>DIRG $_{3}$ </td><td> $\Lambda$ </td><td> $\{F6\}$ </td></tr><tr><td> $C_{16}$ </td><td>DIRG $_{0, 1, 2, 3, 5}$ </td><td> $\{N\} + \{N, F1\} + \{N, F1, F2\} + \{N, F1, F2, F3\} + \{N, F1, F2, F3, F4\}$ </td><td>DIRG $_{4}$ </td><td> $\Lambda$ </td><td> $\{F5, F6\}$ </td></tr></table>

## 4.3. Experimental settings

## 1) Compared Methods:

To convincingly demonstrate the superiority of the proposed method, we select several state-of-the-art techniques in the fields of domain generalization and category shift as comparison algorithms. Table 6 concisely outlines these methods in three essential aspects. Additionally, a detailed description of each method is presented below.

1) ERM: It is based on CNN trained on multi-source domains without TL strategy. The network structure of this method is consistent with our method.

2) MMD: It’s a DG method, which employs MMD for DG. The structure is in accordance with our method.

3) Multiadversarial Domain Adaptation (MADA) [31]: This method adopts a multi-adversarial DA strategy, exploiting labeled in formation to guide the discriminative learning. Fine-grained distribution alignment is achieved by multiple discriminators for multiple categories. The structure of generator and classifier follows our network.

4) Learning and Removing Domain-specific features for Generalization (LRDG) [32]: LRDG firstly design a classifier to learn the domain-specific features for each source domain. By developing an encoder-decoder network and another classifier to map the input sample to a new embedding space where the learned domain-specific features are removed, DG is achieved.

5) Adaptive Open Set Domain Generalization Network (AOSDGN) [24]: It’s a DG method, the triplet loss is used to enhance the interclass separability and alleviate the domain shift. A class-wise decision boundary mechanism is introduced to distinguish known and unknown class samples.

6) Deep Cocktail Network (DCTN) [23]: DCTN is a method to address the MDA with category shifts. It uses a multi-way adversaria learning pipeline to minimize the domain discrepancy between the target and each of the multiple in order to learn domain-invariant features. The derived source-specific perplexity scores measure how similar each target feature appears as a feature from one of source domains. The multi-source category classifiers are integrated with the perplexity scores to categorize target samples

7) GlocalNet [25]: GlocalNet is a MDA technique consisting of a feature extractor and three classifiers. It employs the accumulative higher order multisource moment (AHMM) to mitigate domain shift, while tackling category shift through the Wasserstein discrepancy of classifiers (WDC) and multisource distilling (MSD). Furthermore, an adaptive classifier discrepancy weighting strategy is proposed for the target domain.

## Table 6

Description of Compared Methods. (TL: Transfer Learning, DG: Domain Generalization, MDA: Multi-domain Adaptation).

<table><tr><td>Methods</td><td>TL Strategy</td><td>Source-category-shift</td><td>Target-category-shift</td></tr><tr><td>ERM</td><td>Non-TL</td><td>×</td><td>×</td></tr><tr><td>MMD</td><td>DG</td><td>×</td><td>×</td></tr><tr><td>MADA</td><td>MDA</td><td>×</td><td>×</td></tr><tr><td>LRDG</td><td>DG</td><td>×</td><td>×</td></tr><tr><td>AOSDGN</td><td>DG</td><td>×</td><td>√</td></tr><tr><td>DCTN</td><td>MDA</td><td>√</td><td>√</td></tr><tr><td>GlocalNet</td><td>MDA</td><td>√</td><td>√</td></tr><tr><td>MTCCS</td><td>Non-TL, Multi-Task</td><td>√</td><td>×</td></tr><tr><td>OURS</td><td>DG</td><td>√</td><td>√</td></tr></table>

8) MTCCS [33]: MTCCS is introduced to deal with the problem setting of multi-task classification with category shifts. This al gorithm constructs the association graph with nodes representing tasks, classes and instances, and encode the relationships among the nodes in the edges to guide their mutual knowledge transfer. Meanwhile, the utilize of assignment entropy maximization avoids spurious correlations between task and class nodes in the graph

## 2) Implementation Details.

All methods are implemented via Python 3.8 and the DL framework PyTorch 1.7.0. The proposed method shares a common structural framework with other methods, including feature extractor, classifier, and domain discriminator. For the training of all algorithm, Adam solver is taken as the optimizer. The reciprocal points are initialized by a random normal distribution and each margin is initialized to one. The learning rate is set as lr $= \mathbf { I r } _ { 0 } / ( 1 + 1 0 \times \mathbf { p } ) ^ { 0 . 7 5 }$ (where ${ \bf { l r } } _ { 0 }$ is the initial learning rate,p is the ratio of the current epoch to all epochs). The hyperparameters of the comparison algorithm are maintained in the original settings of the relevant references. The hyperparameters of our experiments are presented in Table 7.

## 3) Evaluation Metrics.

The evaluation of classifying known categories and identifying unknown categories is based on the accuracy (ACC) and the area under the receiver operating characteristic curve (AUROC). In addition, a metric for open set recognition known as the open set classification rate (OSCR) is also introduced in this article. Similar to AUROC, a larger OSCR implies a more robust recognition capability for the algorithm. The OSCR is based on correct classification rate (CCR) and the false positive rate (FPR). CCR is the fraction of the samples where the correct category k has maximum probability and has a probability greater than τ. FPR is the fraction of the samples from unknown category space that are classified as any known category k with a probability greater than τ. The formulas for these two are as follows:

$$
C C R (\tau) = \frac {\left| \left\{x \mid x \in \left(\mathcal {Y} ^ {\mathrm{s}}\right) \wedge \operatorname{argmax} _ {k} P (k \mid x) = \widehat {k} \wedge P (\widehat {k} \mid x) \geq \tau \right\} \right|}{\left| \mathcal {Y} ^ {\mathrm{s}} \right|}\tag{13}
$$

$$
F P R (\tau) = \frac {\left| \left\{x \mid x \in \left(\mathcal {U} ^ {\mathrm{T}}\right) \wedge \max _ {k} P (k \mid x) \geq \tau \right\} \right|}{\left| \mathcal {U} ^ {\mathrm{T}} \right|}\tag{14}
$$

## 4.4. Experiments

In this section, the fault diagnosis results on the two datasets can be seen in Tables 8 and 9. The $A C C _ { M } , A C C _ { T } , A U R O C ,$ and OSCR on the left side of table are the accuracy of the missing label space on the source domains, the accuracy of the known label space on the target domains, the AUROC and OSCR of target domain, respectively. Since the MTCCS method requires the corresponding domain labels in the phase, it is not used for the comparison of the target domain accuracy.

## 4.4.1. Results on KAT Dataset

The experimental results for KAT datasets are presented in Table $^ { 8 , }$ and the average results are shown in Fig. 6. Consequently, we can draw the subsequent conclusions:

1) The performance of all algorithms gradually decreases with increasing experimental difficulty. As shared categories diminish among source domains, the accuracy of all algorithms in known categories remains consistent or shows slight fluctuations. In addition, the trend of the $A C C _ { M } , A U R O C ,$ and OSCR shows that the algorithm's ability to recognize the missing categories of the source domains and the novel categories of target domain decreases significantly. This decline could be attributed to algorithms extracting algorithms extracting instead of effectively capturing domain-invariant features.

2) TL-based algorithms demonstrate superior performance compared to the ERM benchmark approach without TL strategy, exhibiting enhanced adaptability to both domain and category shifts across all tasks. Furthermore, although traditional DG algorithms address domain shift proficiently, they struggle to effectively gather categorical information when faced with category shift. Conversely, the strategies that considers category shift effectively alleviate both challenges and accomplishes more effective fault diagnosis.

3) Displaying outstanding performance in all tasks, the proposed algorithm attains 90.60 % and 88.85 % for the missing categories in the source domains and the known categories in the target domain. which represents a substantial improvement of 27.22 % and 25.06 %, respectively, compared to ERM. Additionally, the algorithm shows comparable accuracy for missing categories in the source domains when compared to the MTCCS algorithm. Furthermore, in the target domain. the algorithm demonstrates excellent performance with scores of 88.85 %, 94.92 %, and 80.08 % for the average of $A C C _ { T } , A U R O C ,$ , and OSCR, respectively, surpassing the

## Table 7

Implementation details of the proposed method. $\mathbf { \nabla } . \mathbf { c ^ { s } }$ is the category number of responding source domain.

<table><tr><td>Hyperparameter</td><td>Setting</td><td>Hyperparameter</td><td>Setting</td></tr><tr><td>Batch size</td><td>64</td><td> $\tau$ </td><td>0.1</td></tr><tr><td>Max epoch</td><td>200</td><td> $\lambda_1$ </td><td>0.1</td></tr><tr><td>Initial learning rate</td><td>0.001</td><td> $\lambda_2$ </td><td>0.1</td></tr><tr><td>Each source domain sample number  $n^s$ </td><td> $500×c^s$ </td><td> $\mu$ </td><td>0.2</td></tr><tr><td>Sample Length</td><td>1024</td><td> $\eta$ </td><td>5</td></tr></table>

Table 8  
The experiment results of different algorithms on the KAT dataset.

<table><tr><td>Cases</td><td></td><td>ERM</td><td>MMD</td><td>MADA</td><td>LRDG</td><td>AOSDGN</td><td>DCTN</td><td>GlocalNet</td><td>MTCCS</td><td>Ours</td></tr><tr><td> $C_{01}$ </td><td> $ACC_T$ </td><td>76.55</td><td>92.11</td><td>93.54</td><td>95.62</td><td>95.67</td><td>96.15</td><td>97.33</td><td>-</td><td>96.42</td></tr><tr><td rowspan="4"> $C_{02}$ </td><td> $ACC_M$ </td><td>70.38</td><td>76.19</td><td>76.12</td><td>89.25</td><td>93.42</td><td>93.11</td><td>94.47</td><td>94.73</td><td>94.53</td></tr><tr><td> $ACC_T$ </td><td>71.42</td><td>78.83</td><td>78.01</td><td>84.48</td><td>86.74</td><td>87.42</td><td>91.66</td><td>-</td><td>92.29</td></tr><tr><td>AUROC</td><td>80.80</td><td>83.15</td><td>81.43</td><td>90.47</td><td>92.28</td><td>93.01</td><td>95.19</td><td>-</td><td>95.39</td></tr><tr><td>OSCR</td><td>68.83</td><td>70.69</td><td>69.08</td><td>77.14</td><td>79.65</td><td>80.49</td><td>83.23</td><td>-</td><td>85.84</td></tr><tr><td rowspan="4"> $C_{03}$ </td><td> $ACC_M$ </td><td>65.68</td><td>73.21</td><td>74.66</td><td>90.20</td><td>89.74</td><td>90.17</td><td>88.75</td><td>90.14</td><td>91.62</td></tr><tr><td> $ACC_T$ </td><td>57.13</td><td>64.19</td><td>67.67</td><td>83.57</td><td>82.83</td><td>86.16</td><td>84.09</td><td>-</td><td>88.36</td></tr><tr><td>AUROC</td><td>63.33</td><td>71.04</td><td>73.48</td><td>93.00</td><td>91.55</td><td>95.03</td><td>92.13</td><td>-</td><td>96.47</td></tr><tr><td>OSCR</td><td>50.01</td><td>58.13</td><td>61.14</td><td>74.50</td><td>71.30</td><td>79.82</td><td>77.36</td><td>-</td><td>82.95</td></tr><tr><td rowspan="4"> $C_{04}$ </td><td> $ACC_M$ </td><td>53.09</td><td>64.31</td><td>67.10</td><td>73.15</td><td>78.37</td><td>80.22</td><td>82.52</td><td>84.23</td><td>83.67</td></tr><tr><td> $ACC_T$ </td><td>54.30</td><td>59.87</td><td>63.27</td><td>74.05</td><td>76.45</td><td>76.21</td><td>78.03</td><td>-</td><td>82.37</td></tr><tr><td>AUROC</td><td>55.23</td><td>61.30</td><td>70.45</td><td>78.33</td><td>81.64</td><td>80.11</td><td>82.35</td><td>-</td><td>91.05</td></tr><tr><td>OSCR</td><td>42.47</td><td>51.09</td><td>55.43</td><td>60.21</td><td>66.60</td><td>65.06</td><td>67.70</td><td>-</td><td>71.16</td></tr><tr><td rowspan="4"> $C_{05}$ </td><td> $ACC_M$ </td><td>64.38</td><td>72.05</td><td>71.88</td><td>85.40</td><td>87.20</td><td>89.66</td><td>88.73</td><td>90.82</td><td>92.60</td></tr><tr><td> $ACC_T$ </td><td>59.54</td><td>63.33</td><td>61.11</td><td>80.80</td><td>81.25</td><td>82.20</td><td>84.81</td><td>-</td><td>84.79</td></tr><tr><td>AUROC</td><td>60.23</td><td>72.03</td><td>70.87</td><td>85.26</td><td>86.53</td><td>87.64</td><td>90.75</td><td>-</td><td>91.35</td></tr><tr><td>OSCR</td><td>52.81</td><td>58.63</td><td>54.32</td><td>74.20</td><td>75.01</td><td>75.55</td><td>76.67</td><td>-</td><td>77.47</td></tr><tr><td rowspan="4">Average</td><td> $ACC_M$ </td><td>63.38</td><td>71.44</td><td>72.44</td><td>84.50</td><td>87.18</td><td>88.29</td><td>88.62</td><td>89.98</td><td>90.60</td></tr><tr><td> $ACC_T$ </td><td>63.79</td><td>71.67</td><td>72.72</td><td>83.70</td><td>84.59</td><td>85.63</td><td>87.18</td><td>-</td><td>88.85</td></tr><tr><td>AUROC</td><td>66.92</td><td>74.32</td><td>74.82</td><td>90.43</td><td>90.48</td><td>92.68</td><td>92.55</td><td>-</td><td>94.92</td></tr><tr><td>OSCR</td><td>58.24</td><td>62.78</td><td>61.98</td><td>72.17</td><td>75.23</td><td>75.39</td><td>77.71</td><td>-</td><td>80.08</td></tr></table>

Table 9  
The experiment results of different algorithms on the DIRG dataset.

<table><tr><td>Cases</td><td></td><td>ERM</td><td>MMD</td><td>MADA</td><td>LRDG</td><td>AOSDGN</td><td>DCTN</td><td>GlocalNet</td><td>MTCCS</td><td>Ours</td></tr><tr><td> $C_{11}$ </td><td> $ACC_T$ </td><td>61.86</td><td>80.73</td><td>84.20</td><td>94.66</td><td>92.21</td><td>94.13</td><td>93.55</td><td>-</td><td>95.04</td></tr><tr><td rowspan="4"> $C_{12}$ </td><td> $ACC_M$ </td><td>72.89</td><td>81.48</td><td>83.56</td><td>90.14</td><td>88.65</td><td>91.48</td><td>91.27</td><td>92.84</td><td>92.08</td></tr><tr><td> $ACC_T$ </td><td>64.39</td><td>78.38</td><td>79.03</td><td>87.74</td><td>82.99</td><td>89.45</td><td>89.53</td><td>-</td><td>91.28</td></tr><tr><td>AUROC</td><td>68.61</td><td>79.75</td><td>82.62</td><td>92.44</td><td>90.15</td><td>94.17</td><td>94.32</td><td>-</td><td>95.47</td></tr><tr><td>OSCR</td><td>58.34</td><td>68.55</td><td>69.87</td><td>80.12</td><td>75.67</td><td>82.21</td><td>82.09</td><td>-</td><td>83.69</td></tr><tr><td rowspan="4"> $C_{13}$ </td><td> $ACC_M$ </td><td>68.58</td><td>81.70</td><td>80.52</td><td>85.07</td><td>84.77</td><td>86.57</td><td>87.14</td><td>88.19</td><td>87.37</td></tr><tr><td> $ACC_T$ </td><td>63.74</td><td>76.45</td><td>75.60</td><td>82.12</td><td>83.55</td><td>83.22</td><td>84.65</td><td>-</td><td>83.23</td></tr><tr><td>AUROC</td><td>70.31</td><td>81.39</td><td>81.02</td><td>90.20</td><td>91.09</td><td>91.22</td><td>92.33</td><td>-</td><td>92.58</td></tr><tr><td>OSCR</td><td>55.26</td><td>68.67</td><td>65.94</td><td>73.13</td><td>79.81</td><td>77.81</td><td>80.18</td><td>-</td><td>81.11</td></tr><tr><td rowspan="4"> $C_{14}$ </td><td> $ACC_M$ </td><td>63.45</td><td>75.33</td><td>76.81</td><td>81.43</td><td>82.78</td><td>82.33</td><td>84.52</td><td>85.05</td><td>83.81</td></tr><tr><td> $ACC_T$ </td><td>56.07</td><td>69.31</td><td>67.66</td><td>78.15</td><td>80.01</td><td>80.36</td><td>81.77</td><td>-</td><td>81.64</td></tr><tr><td>AUROC</td><td>61.43</td><td>72.05</td><td>71.98</td><td>82.62</td><td>85.82</td><td>85.95</td><td>87.00</td><td>-</td><td>86.77</td></tr><tr><td>OSCR</td><td>53.85</td><td>60.33</td><td>58.58</td><td>71.63</td><td>74.78</td><td>73.85</td><td>75.10</td><td>-</td><td>75.62</td></tr><tr><td rowspan="4"> $C_{15}$ </td><td> $ACC_M$ </td><td>55.30</td><td>68.98</td><td>63.25</td><td>76.35</td><td>78.79</td><td>81.90</td><td>80.12</td><td>81.95</td><td>81.48</td></tr><tr><td> $ACC_T$ </td><td>45.52</td><td>59.27</td><td>57.63</td><td>73.54</td><td>74.88</td><td>75.10</td><td>74.86</td><td>-</td><td>76.34</td></tr><tr><td>AUROC</td><td>57.62</td><td>64.91</td><td>63.32</td><td>78.48</td><td>80.02</td><td>83.63</td><td>80.44</td><td>-</td><td>84.84</td></tr><tr><td>OSCR</td><td>40.09</td><td>51.21</td><td>48.01</td><td>67.78</td><td>68.45</td><td>70.09</td><td>69.05</td><td>-</td><td>71.06</td></tr><tr><td rowspan="4"> $C_{16}$ </td><td> $ACC_M$ </td><td>65.91</td><td>80.58</td><td>78.63</td><td>82.71</td><td>83.41</td><td>85.33</td><td>84.50</td><td>84.87</td><td>85.02</td></tr><tr><td> $ACC_T$ </td><td>60.68</td><td>75.75</td><td>74.39</td><td>78.88</td><td>78.92</td><td>82.75</td><td>82.45</td><td>-</td><td>82.84</td></tr><tr><td>AUROC</td><td>68.47</td><td>79.63</td><td>77.67</td><td>80.64</td><td>79.48</td><td>89.35</td><td>90.02</td><td>-</td><td>90.39</td></tr><tr><td>OSCR</td><td>57.82</td><td>67.27</td><td>66.18</td><td>70.01</td><td>73.93</td><td>72.32</td><td>74.31</td><td>-</td><td>75.44</td></tr><tr><td rowspan="4">Average</td><td> $ACC_M$ </td><td>65.23</td><td>77.61</td><td>76.55</td><td>93.14</td><td>83.68</td><td>85.52</td><td>85.51</td><td>86.58</td><td>85.95</td></tr><tr><td> $ACC_T$ </td><td>58.71</td><td>73.32</td><td>73.08</td><td>82.51</td><td>82.09</td><td>84.16</td><td>84.63</td><td>-</td><td>85.06</td></tr><tr><td>AUROC</td><td>65.29</td><td>75.55</td><td>75.32</td><td>84.88</td><td>85.31</td><td>88.86</td><td>88.82</td><td>-</td><td>90.01</td></tr><tr><td>OSCR</td><td>53.07</td><td>63.20</td><td>61.71</td><td>72.53</td><td>74.52</td><td>75.25</td><td>76.15</td><td>-</td><td>77.38</td></tr></table>

![](images/347496332a5017e953db94a3fb01b2e1b68ec82bddc3fedf10de579ee0546566.jpg)  
(a) ACCT

![](images/a99a38e87af1d32db95976df7293570310223b63a2425fec1bd5a93838bda465.jpg)  
(b) AUROC  
Fig. 6. The fault diagnosis result on target domain in KAT Dataset.

suboptimal algorithm by 1.67 %, 2.24 %, and 2.37 %. These results validate the proposed algorithm’s superior generalization per formance, efficiency leveraging known categorical information from multi-source domains and showcasing robust classification ca pabilities for both known and unknown classes

In order to further analyze algorithm performance, we compared feature visualization results of each algorithm in case $C _ { 0 3 } .$ , The ERM algorithm demonstrates a lack of inter-domain alignment, inadequate inter-class separation, and a notable mixture of known and unknown classes in the target domain. The results depicted in Fig. 7(d and e) demonstrate the improved inter-class separation achieved by the LRDG and AOSDGN algorithms with DG strategy. Nonetheless, some confusion remains for the missing categories in each source domain, and the LRDG algorithm without consideration of the target-category shift struggles to correctly identify unknown classes in the target domain. While GlocalNet and DCTN tackle the category shift problem, resulting in better performance in known classes clustering and identification of unknown classes, it is notable that misclassifications persist in both algorithms. In contrast. by examining Fig. 7(h), it becomes apparent that the proposed algorithm excels in feature alignment, effectively decreases inter-class distance, increases inter-class distance, and achieves superior recognition outcomes for both known and unknow classes. It shows that the proposed method effectively addresses domain and category shift problems, and confirming its strong generalization. In addition, we further analyze the diagnostic ability of the proposed algorithm for each category in all domains using the confusion matrix, as shown in Fig. 8. The proposed algorithm ensures a high level of accuracy on known classes across all source domains. Although there is a slight decline in accuracy on missing classes in each source domain, it still maintains precise identification. Meanwhile, in the target domain, the algorithm shows the ability to distinguish between the known and unknown categories.

## 4.4.2. Results on DIRG Dataset

The experimental results in the dataset DIRG are shown in Table 9 and Fig. 9. Across most fault diagnosis tasks, the proposed algorithm outperforms alternative methods, providing evidence of its effectiveness in achieving superior fault diagnosis through the utilization of incomplete multi-source do-mains. In tasks $C _ { 1 3 }$ and $C _ { 1 4 } ,$ , the proposed algorithm exhibits a minor decrement in accuracy compared to GlocalNet. However, it is noteworthy that these MDA algorithms require involving the target domain in the training process. Considering the practical constraints of unavailability of the target domain in advance for training in engineering applications, it becomes necessary and feasible to investigate the proposed method.

Fig. 10 displays the feature visualization results corresponding to task $C _ { 1 6 } .$ . Traditional domain generalization algorithms exhibit significant overlap between categories, making it difficult to accurately distinguish both known and unknown categories. The DCTN and GlocalNet demonstrate the ability to differentiate between them, although occasional misclassifications occur. The distributions of unknown categories and known categories F2, F3 and F4 display greater proximity, with all algorithms, exhibiting varying degrees of overlap. Based on the confusion matrix in Fig. 11, the algorithm’s effectiveness in addressing category shift issues can be further validated. In general, the proposed algorithm successfully segregates samples with different health states into separate clusters, effectively distinguishing between known and unknown categories, thus confirming its algorithmic superiority.

## 5. Discussion

## 5.1. Ablation study

In the following table, we conduct comparative experiments for different variants of the proposed model using the DIRG dataset as the baseline dataset to verify the impact of each component of the model on the overall effect.

![](images/84a5c3e3d7455ef8072f347fcde0a1756377c06b8dd6764a8a8592150d29fe77.jpg)  
Fig. 7. The feature visualization result of case $C _ { 0 3 }$ in KAT Dataset.

## 5.1.1. Effect of curriculum learning strategy

Table 10 presents a comparison of experimental results for different curriculum learning strategies, which is a key concept in the proposed methods. The compared methods consist of: A. the proposed method utilizing two-stage training, B. single-stage training without the curriculum learning strategy, and C. two-stage training with randomized categories at each stage. D. multi-stage training, each stage fixedly uses the categories with the same ${ \mathcal { O } } _ { \mathbf { C } }$ value. From the comparison of the experimental results, it can be seen that the proposed algorithm curriculum learning has the capability to make the categories with low ${ \mathcal { O } } _ { \mathbf { C } }$ utilize the knowledge of the mode trained by high ${ \mathcal { O } } _ { \mathbf { C } }$ samples, thus optimizing the domain-invariant feature extraction of the incomplete categories and improve the generalization ability of the model.

![](images/5f2cabbd09c7c9196b06354ce922854dc351beb0f1dcc81e6d72514163152e54.jpg)  
(a) Source domain 1

![](images/70187e043a3bae4a26e6a7995bf4f295e67acc54c597e3e0d8fb3f00d169f13a.jpg)  
(b) Source domain 2

![](images/affd7d180efc2fc11d58ec416c422fa0cf54126d51d7b1dfb61aabd953c75f5d.jpg)  
(c) Source domain 3

![](images/429d01156e6da716366c1b03fc62db58a2a3f53c7ed258e52859f06096dd4d1b.jpg)  
(d) Target domain 0

Fig. 8. The confusion matrix of case $C _ { 0 3 }$ in KAT Dataset.  
![](images/975952397b86e83f18d1d3b1d556fc514b4f4ebb78686346c5cf671e89f1c777.jpg)  
(a) $A C C _ { T }$

![](images/32698c5e8efd1b14433b5a712f54e5b1b13862da9f1cc1a065c1a978086054b2.jpg)  
(b) AUROC  
Fig. 9. The fault diagnosis result on target domain in DIRG Dataset.

![](images/99f051b285139f9dd321707c388fdbe9a949bc8d6bb364df8ca21f14c5b0a9ca.jpg)  
Fig. 10. The feature visualization result of case $C _ { 1 6 }$ in DIRG Dataset.

## 5.1.2. Effect of mixup strategy, reciprocal point learning and conditional domain discriminator

The analysis presented in Table 10 reveals the positive effects of both Mixup and RPL on enhancing the model’s overall perfor mance. (A vs E for Mixup, E vs F for RPL in Table 10). Notably, Mixup significantly improves the informativeness of the unknown space and facilitates the classification of known and unknown categories, as observed in the case $C _ { 1 5 } .$ Additionally, RPL is essential for addressing the target-category shift problem and effectively distinguishing between known and unknown classes. Table 10 (A vs G) further demonstrates that the employed CDD meets the requirements for domain generalization, promoting alignment among various source domains and ensuring the model’s generalization ability.

![](images/31b163b52d2ef4c86f5cc57678bfdf69db29ae8abf1b710e2775989505532a42.jpg)  
(a) Source domain 0

![](images/f5db78fb3b7a12369756df9ad6fa63f297a0408d8398b702c1a23923ad9f9ddd.jpg)  
(b) Source domain 1

![](images/1318bf0423061d0cda5671458bf2a6e4017fc6cec32077cbb53837af11919972.jpg)  
(c) Source domain 2

![](images/41d4ab566c5634b391a7849fd0a5981334e5fdf739fabfe90d522fff3094dbb2.jpg)  
(d) Source domain 3

![](images/d9084975e1e530aa4b71060d9769f39d5a48526fa93c185d5b9022540fff2d82.jpg)  
(e) Source domain 5

![](images/4de594d9e0fdfedb48d9158f280c6c1f988327b911ce8ffb2a3675f33aa92321.jpg)  
(f) Target domain 4  
Fig. 11. The feature visualization result of case $C _ { 1 6 }$ in DIRG Dataset.

The result of ablation study on the DIRG Dataset. (CLS: Curriculum Learning Strategy, RPL: Reciprocal Point Learning, CDD: Conditional Domain Discriminator, A: AUROC, O: OSCR).

<table><tr><td rowspan="3">Methods</td><td colspan="4">Components</td><td colspan="8">Cases</td></tr><tr><td rowspan="2">Mixup</td><td rowspan="2">RPL</td><td rowspan="2">CDD</td><td rowspan="2">CLS</td><td colspan="2"> $C_{12}$ </td><td colspan="2"> $C_{13}$ </td><td colspan="2"> $C_{14}$ </td><td colspan="2"> $C_{16}$ </td></tr><tr><td>A</td><td>O</td><td>A</td><td>O</td><td>A</td><td>O</td><td>A</td><td>O</td></tr><tr><td>A. Ours</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>95.47</td><td>83.69</td><td>92.58</td><td>81.11</td><td>86.77</td><td>75.62</td><td>90.39</td><td>75.44</td></tr><tr><td>B. CLS</td><td>✓</td><td>✓</td><td>✓</td><td>Without</td><td>94.21</td><td>81.23</td><td>91.66</td><td>79.01</td><td>84.91</td><td>73.05</td><td>88.50</td><td>72.20</td></tr><tr><td>C. CLS</td><td>✓</td><td>✓</td><td>✓</td><td>Random</td><td>91.14</td><td>78.74</td><td>89.04</td><td>74.16</td><td>81.61</td><td>71.92</td><td>84.23</td><td>67.58</td></tr><tr><td>D. CLS</td><td>✓</td><td>✓</td><td>✓</td><td>Fixed</td><td>94.40</td><td>80.54</td><td>90.03</td><td>79.02</td><td>83.74</td><td>72.43</td><td>88.10</td><td>71.28</td></tr><tr><td>E. Mixup</td><td>×</td><td>✓</td><td>✓</td><td>✓</td><td>93.80</td><td>79.86</td><td>88.58</td><td>78.04</td><td>83.71</td><td>71.15</td><td>86.34</td><td>70.67</td></tr><tr><td>F. RPL</td><td>×</td><td>×</td><td>✓</td><td>✓</td><td>90.47</td><td>77.15</td><td>87.09</td><td>74.76</td><td>79.63</td><td>67.40</td><td>82.73</td><td>67.05</td></tr><tr><td>G. CDD</td><td>✓</td><td>✓</td><td>×</td><td>✓</td><td>89.22</td><td>78.79</td><td>88.45</td><td>75.38</td><td>82.03</td><td>71.48</td><td>84.31</td><td>69.54</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td></td><td>68.61</td><td>58.34</td><td>70.31</td><td>55.26</td><td>61.43</td><td>53.85</td><td>68.47</td><td>57.82</td></tr></table>

## 5.2. Parameter analysis

The training of the proposed algorithm involves several learnable parameters. To study their impact on the model performance, a parameter sensitivity analysis is performed for the cases of the DIRG dataset. $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are the weight parameters of CDD and Mixup-

RPL, respectively. And η is the parameter for the knowledge distillation. Except for the analysis parameter, the remaining parameters are fixed to the values used in this paper (Section 4.3 Table 7). The results are shown in Fig. 12. When $\lambda _ { 1 }$ is in [0.1, 0.2], the fault diagnosis has the best accuracy. Too large or too smal $\lambda _ { 2 }$ will both affect the model performance, and it is more appropriate to choose between [0.01,1]. When the parameter η is set to either very small or large values, the model’s accuracy experiences a significant reduction. If η is too small, it can lead to catastrophic forgetting, while if it is too large, the model may fail to acquire new knowledge. Therefore, it is more appropriate to choose between [2,10].

## 5.3. Future work

1) The proposed curriculum learning strategy lacks sufficient intelligence, and the diagnosis result is unknown in the case of more fault categories. Meanwhile, the accuracy of the algorithm can be further improved.

2) The proposed method falls short in tackling fault diagnosis in scenarios where multi-source domains do not have a share class. In our future work, we will focus on developing an adaptive curriculum learning strategy to enhance algorithm performance. Additionally, we aim to optimize the algorithm by leveraging multi-source domains shared and non-shared categorical information. This optimization aims to eliminate spurious environmental feature, enabling the development of a fault diagnosis model with enhanced generalization ability and improved practical applicability.

## 6. Conclusion

This paper mainly explores the problem of domain generalization fault diagnosis with category shift. We mainly divide this problem into three aspects to be solved, domain shift, source-category shift and target-category shift. For target-category shift, we propose a mixup-based reciprocal point learning classifier. Meanwhile, in order to eliminate the domain shift, we introduce a con ditional domain discriminator. Furthermore, we address the challenge of extracting categorical information caused by source-category shift through a curriculum learning strategy. By combining these three methods, we develop an IFD model with high accuracy and generalization. We validate the effectiveness of our algorithm through extensive experiments on two datasets. We analyze the fault diagnosis performance under varving degrees of shared categories among multi-source domains. The experimental results demonstrate the superior performance of our algorithm compared to state-of -the-art domain generalization IFD algorithms, as it effectively identifies known and unknown classes and exhibits strong generalization capability. However, the current curriculum learning strategy remains suboptimal, and the algorithm’s performance significantly degrades when dealing with scenarios where most cat egories in each source domain are private categories. To address these issues, we plan to investigate adaptive curriculum learning strategy and enhance the model’s generalization ability by employing strategies such as disentangled representation learning in future studies.

## CRediT authorship contribution statement

Yu Wang: Investigation, Methodology, Software, Visualization, Writing – original draft. Jie Gao: Funding acquisition, Software, Supervision, Writing – original draft. Wei Wang: Conceptualization, Funding acquisition, Project administration, Software. Xu Yang: Funding acquisition, Supervision, Validation, Visualization. Jinsong Du: Methodology, Project administration, Supervision, Validation.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

![](images/59dedf4f762dafdcf8419a1eb372cd7cefccde0c815e06a279bf15926d17f0f5.jpg)  
(a) $\lambda _ { _ { 1 } }$

![](images/6dc312642615d62455c3aa644f64146c5145a7b310596fcdad8de73fbd4f5e23.jpg)  
(b) $\lambda _ { _ 2 }$

![](images/b8ba31204fe4468e2f07dd6595f4012528991b655d6b997a875ff12105db1003.jpg)  
Fig. 12. The results of Parameter analysis.  
(c) η

## Data availability

No data was used for the research described in the article.

## References

[1] G. Liu, W. Shen, L. Gao, A. Kusiak, Predictive modeling with an adaptive unsupervised broad transfer algorithm, IEEE Trans. Instrum. Meas. 70 (2021) 1–12, https://doi.org/10.1109/TIM.2021.3088496.

[2] S. Luo, X. Huang, Y. Wang, et al., Transfer learning based on improved stacked autoencoder for bearing fault diagnosis, Knowl.-Based Syst. 256 (2022) 109846, https://doi.org/10.1016/j.knosys.2022.109846.

[3] X. Guo, L. Chen, C. Shen, Hierarchical adaptive deep convolution neural network and its application to bearing fault diagnosis, Measurement 93 (2016) 490–502, https://doi.org/10.1016/j.measurement.2016.07.054.

[4] B. Cui, Y. Weng, N. Zhang, A feature extraction and machine learning framework for bearing fault diagnosis, Renew. Energy 191 (2022) 987–997, https://doi. org/10.1016/j.renene.2022.04.061.

[5] L. Eren, T. Ince, S. Kiranyaz, A generic intelligent bearing fault diagnosis system using compact adaptive 1D CNN classifier, Journal of Signal Processing Systems 91 (2019) 179–189, https://doi.org/10.1007/s11265-018-1378-3.

[6] T. Li, Z. Zhao, C. Sun, Multireceptive field graph convolutional networks for machine fault diagnosis, IEEE Trans. Ind. Electron. 68 (12) (2020) 12739–12749, https://doi.org/10.1109/TIE.2020.3040669.

[7] S. Schwendemann, Z. Amjad, A. Sikora, Bearing fault diagnosis with intermediate domain based layered maximum mean discrepancy: A new transfer learning approach, Eng. Appl. Artif. Intel. 105 (2021) 104415, https://doi.org/10.1016/j.engappai.2021.104415.

[8] Y. Liu, Y. Wang, T.W.S. Chow, et al., Deep adversarial subdomain adaptation network for intelligent fault diagnosis, IEEE Trans. Ind. Inf. 18 (9) (2022) 6038–6046, https://doi.org/10.1109/TII.2022.3141783.

[9] Q. Li, L. Chen, L. Kong, Cross-domain augmentation diagnosis: An adversarial domain-augmented generalization method for fault diagnosis under unseen working conditions, Reliab. Eng. Syst. Saf. 234 (2023) 109171, https://doi.org/10.1016/j.ress.2023.109171.

[10]. Y. Shu. Z. Cao. C. Wang, Open domain generalization with domain-augmented meta-learning, in: Proceed-Ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021, pp. 9624–9633. https://doi.org/10.1109/CVPR46437.2021.00950.

[11] T. Gokhale, R. Anirudh, J.J. Thiagarajan, Improving diversity with adversarially learned transformations for domain generalization, in: Proceedings of the IEEE/ CVF Winter Conference on Applications of Computer Vision, 2023, pp. 434–443, https://doi.org/10.1109/WACV56688.2023.00051.

[12] Y. Shi. A. Deng, M. Deng, Domain augmentation generalization network for real-time fault diagnosis un-der unseen working conditions, Reliab. Eng, Syst, Saf 235 (2023) 109188, https://doi.org/10.1016/j.ress.2023.109188.

[13] M.H. Bui, T. Tran, A. Tran, Exploiting domain-specific features to enhance domain generalization, Ady. Neural Inf. Proces. Syst. 34 (2021) 21189–21201

[14] Y.F. Zhang, Z. Zhang, D. Li, Learning domain invariant representations for generalizable person re-identification, IEEE Trans. Image Process. 32 (2022) 509–523, https://doi.org/10.1109/TIP.2022.3229621.

[15] S. Jia, Y. Li, X. Wang, Deep causal factorization network: A novel domain generalization method for cross-machine bearing fault diagnosis, Mech. Syst. Sig. Process, 192 (2023) 110228. https://doi.org/10.1016/i,ymssp.2023.110228.

[16] S. Yang, X. Kong, Q. Wang, Deep multiple auto-encoder with attention mechanism network: A dynamic domain adaptation method for rotary machine fault diagnosis under different working conditions, Knowl.-Based Syst. 249 (2022) 108639, https://doi.org/10.1016/j.knosys.2022.108639.

[17] J. Tian, D. Han, M. Li, A multi-source information transfer learning method with subdomain adaptation for cross-domain fault diagnosis, Knowl.-Based Syst. 243 (2022) 108466, https://doi.org/10.1016/j.knosys.2022.108466.

[18] C. Zhao, W. Shen, A domain generalization network combing invariance and specificity towards real-time intelligent fault diagnosis, Mech. Syst. Sig. Process. 173 (2022) 108990, https://doi.org/10.1016/j.ymssp.2022.108990.

[19] C. Zhao, W. Shen, Dual adversarial network for cross-domain open set fault diagnosis, Reliab. Eng, Syst. Saf. 221 (2022) 108358, https://doi.org/10.1016/j ress.2022.108358.

[20] W. Li, Z. Shang, M. Gao, Intelligent fault diagnosis of partial deep transfer based on multi-representation structural intraclass compact and double-aligned domain adaptation, Mech. Syst. Sig. Process. 197 (2023) 110412, https://doi.org/10.1016/j.isatra.2023.06.035.

[21] B. She, W. Liang, F. Qin, et al., Known classes aware and emerging unknown classes rejection based on adversarial training for open set fault diagnosis, ISA Trans. 141 (2023) 455–469, https://doi.org/10.1016/j.isatra.2023.06.035.

[22]. Y. Ma, J. Yang, L. Li. Meta Bi-classifier Gradient Discrepancy for noisy and universal domain adaptation in intelligent fault diagnosis, Knowl.-Based Syst, 110735 (2023), https://doi.org/10.1016/j.knosys.2023.110735.

[23] Xu R, Chen Z, Zuo W, Yan J, Lin L. Deep Cocktail Network: Multi-source Unsupervised Domain Adaptation with Category Shift. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3964-3973. DOI: 10.1109/CVPR.2018.00417.

[24] C. Zhao, W. Shen, Adaptive open set domain generalization network: Learning to diagnose unknown faults under unknown working conditions, Reliab. Eng. Syst. Saf. 226 (2022) 108672, https://doi.org/10.1016/j.ress.2022.108672.

[25] Y. Feng, J. Chen, S. He, et al., Globally localized multisource domain adaptation for cross-domain fault diagnosis with category shift, IEEE Trans. Neura Networks Learn. Syst. 34 (2021) 3082–3096. https://doi,org/10.1109/TNNLS.2021.3111732

[26] M. Long, Z. Cao, J. Wang, Conditional adversarial domain adaptation, Adv. Neural Inf. Proces. Syst. 31 (2018).

[27] G. Chen, P. Peng, X. Wang, Adversarial reciprocal points learning for open set recognition, JEEE Trans, Pattern Anal, Mach, Intell, 44 (11) (2021) 8065–8081. https://doi.org/10.1109/TPAMI.2021.3106743.

[28] C. Lessmeier, J.K. Kimotho. D. Zimmer, et al., Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification, PHM Society European Conference 3 (1) (2016). https://doi,org/10.36001/phme.2016. v3j1.1577.

[29] T. Zhang, J. Jiao, J. Lin, Uncertainty-based contrastive prototype-matching network towards cross-domain fault diagnosis with small data, Knowl.-Based Syst. 254 (2022) 109651, https://doi.org/10.1016/j.knosys.2022.109651.

[30] A.P. Daga, A. Fasana, S. Marchesiello, et al., The Politecnico di Torino rolling bearing test rig: Description and analysis of open access data, Mech. Syst. Sig. Process, 120 (2019) 252–273. https://doi.org/10.1016/i,ymssp.2018.10.010.

[31] Pei Z, Cao Z, Long M. Multi-adversarial domain adaptation. Proceedings of the AAAI conference on artificial intelligence. 2018, 32(1)

[32] Ding Y, Wang L, Liang B. Domain Generalization by Learning and Removing Domain-specific Features. arXiv preprint arXiv:2212.07101, 2022.

[33] J. Shen, Z. Xiao, X. Zhen, et al., Association graph learning for multi-task classification with category shifts, Adv. Neural Inf. Proces. Syst. 35 (2022) 4503–4516.