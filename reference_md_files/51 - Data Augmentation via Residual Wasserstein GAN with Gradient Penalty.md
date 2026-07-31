# Data Augmentation and Fault Diagnosis for Imbalanced Industrial Process Data Based on Residual Wasserstein Generative Adversarial Network With Gradient Penalty

Ying Tian<sup>1</sup> | Jian Shen<sup>1</sup> | Ao Wang<sup>1</sup> | Zeqiu Li<sup>2</sup> | Xiuhui Huang<sup>2</sup>

<sup>1</sup>School of Optical-Electrical and Computer Engineering, University of Shanghai for Science and Technology, Shanghai, China | <sup>2</sup>School of Energy and Power Engineering, University of Shanghai for Science and Technology, Shanghai, China

Correspondence: Ying Tian (tianying@usst.edu.cn)

Received: 26 June 2024 | Revised: 25 September 2024 | Accepted: 18 October 2024

Funding: This work was sponsored by National Natural Science Foundation of China (61903251, 22308217, 62003215)

Keywords: fault diagnosis | generative adversarial network | gradient penalty | imbalanced dataset | Wasserstein distanc

## ABSTRACT

In practical industrial applications, equipment usually operates normally and failures are relatively rare, resulting in serious imbalances in the collected data. This imbalance leads to issues such as overfitting, instability, and poor robustness, significantly reducing the accuracy and stability of fault diagnosis system. To address these challenges, this research proposes a method for imbalanced data augmentation and industrial process fault diagnosis based on improved Generative Adversarial Network (GAN). The method adopts Wasserstein distance with gradient penalty and integrates residual connections into the architecture of the generator. This innovation not only helps improve gradient transfer in the generator, but also significantly enhances the data generation capabilities of the generative model through improving the stability of training. Limited industrial process data is used by a generative model to produce synthetic samples with high similarity and diversity. These high-quality samples improve fault diagnosis by enriching the imbalanced dataset. Experimental results on two industrial datasets confirm the method's effectiveness in enhancing fault diagnosis performance with limited data

## 1 | Introduction

Fault diagnosis of industrial equipment is of vital importance. Equipment failures can lead to safety accidents and economic losses. Therefore, performing fault diagnosis on equipment can effectively reduce the adverse consequences of operational errors or failures. With the development of industrial modernization, more monitoring devices are being deployed in factories to monitor the different states of equipment during operation. By incorporating advanced Deep Learning (DL) techniques into fault diagnosis systems and utilizing the information obtained from monitoring devices, it is possible to improve fault diagnosis strategies [1–4]. In many emerging industrial scenarios, advanced methodologies have been developed, such as using a novel multi-scale fused feature and gated recurrent unit for bearing health management [5], employing Digital Twin Enabled Domain Adversarial Graph Networks for bearing fault diagnosis [6], utilizing digital twindriven methods for intelligent assessment of gear surface degradation [7], and applying Physics-Informed Residual Network (PIResNet) for fault diagnosis of rolling bearings [8], among others. Recent innovations have addressed specific challenges in fault diagnosis: for instance, techniques have been proposed to enhance Remaining Useful Life (RUL) prediction even when target data is incomplete [9], and novel methods have been developed to predict the state of charge in lithium-ion batteries used in electric aircraft, leveraging advanced Swin Transformer models [10]. Convolutional Neural Networks (CNNs), as exemplary deep learning algorithms known for their robust learning capabilities and effective extraction of nonlinear features, have found extensive applications in the domain of fault diagnosis. Given the swift advancements in Deep Learning (DL), significant progress has been achieved in formulating fault diagnosis models employing sophisticated deep neural networks.

While there has been progress in utilizing DL models for industrial process fault diagnosis, they encounter diverse challenges that require attention and resolution. For example, achieving ideal results with DL methods requires sufficient high-quality data to support them, which means a large number of highquality samples of both fault and normal states are needed [11]. However, the actual situation is quite the opposite of this requirement. Firstly, in practical industrial operations, due to the safety hazards and economic losses associated with equipment failures, the equipment mostly operates in normal states, with only a very small amount of time spent in fault states. Therefore, the number of samples in real engineering scenarios obtained from monitoring devices is much larger for normal states than for fault states. Secondly, although some methods achieve data balance through experimental simulation data, but acquiring fault data through laboratory simulation experiments requires significant human, material, and financial resources. Moreover, the fault data obtained through simulation often lacks the complex environmental information present in real-world situations, which is a limitation for subsequent data processing and application. Additionally, some industrial processes involve large-scale equipment that is difficult to simulate with high fidelity in a laboratory setting.

This phenomenon is called imbalance between the number of samples in normal and fault states, which is one of the main challenges in data-driven industrial process fault diagnosis [12, 13]. If imbalance fault data is directly used for training intelligent diagnostic models, the models may focus on the abundant normal samples and overlook the fewer but critical fault samples. As a result, well-trained fault diagnostic models can easily misclassify fault samples as normal ones. This outcome is ineffective and can even lead to more severe consequences if the faults occurring during the operation of industrial equipment cannot be accurately identified. Therefore, it is of great importance and significance to find effective methods to address the imbalance issue.

For a long time, addressing the issue of data imbalance in industrial processes has been a research focus for scholars. For example, in the field of Machine Vision (MV), some researchers apply simple geometric transformations, such as random cropping, flipping, rotation, masking, deformation, and scaling [14, 15], to insufficient fault samples for data augmentation. However, this approach clearly fails to effectively address the issue of data imbalance in industrial processes. Additionally, some researchers employ resampling techniques on the dataset, aiming to balance the sample quantities of different classes by increasing or decreasing the number of certain categories. Resampling techniques are mainly divided into Random Under-Sampling (RUS) and Random Over-Sampling (ROS). For example, Synthetic Minority Over-sampling Technique (SMOTE) [16] is a common oversampling method. Th basic idea of SMOTE is to interpolate the samples of the minority class, generating new synthetic samples to increase the sample quantity of that class in the dataset. Frequently employed in classification scenarios, this method aims to tackle imbalanced datasets characterized by a scarcity of samples in the minority class. Although resampling techniques improve the classification performance of imbalanced data, there are still many problems such as the changing distribution of training data. Moreover, ROS leads to increased training time due to the increase in training data sets, and there is a risk of overfitting. RUS eliminates part of the data, reduces the amount of information used to train the model, and causes a certain amount of information waste. Furthermore, some models balance the imbalanced dataset by adjusting the weights of each class during training, such that the model pays more attention to the imbalanced and more challenging classes during training, achieving the desired effect. But how to set an appropriate cost matrix is a difficult problem to be solved at present. In summary, neither employing simple geometric transformations to augment the dataset nor using techniques such as Random Under-Sampling (RUS) and Random Over-Sampling (ROS) effectively addresses the issue of data imbalance in industrial processes.

The issue of data imbalance in industrial processes is particularly significant because it directly impacts the reliability and accuracy of fault detection systems. Industrial processes often generate a large volume of normal operation data compared to fault data, leading to an imbalanced dataset that can hinder the perfor mance of machine learning models. Traditional approaches like geometric transformations and resampling techniques have their limitations, as they either do not address the root causes of data imbalance effectively or introduce additional challenges such as increased training time and potential overfitting. The emergence of Generative Adversarial Networks (GANs) [17] provides another effective solution to address the data imbalance problem. GAN comprises a generator and a discriminator, engaging in adversarial learning to produce novel samples resembling the training data by transforming random noise. The generated samples obtained through training can be added to the original imbalanced dataset, thereby improving or eliminating the imbalance in the dataset and achieving better classification performance. However, although GAN as generative models have significant advantages in generating high-quality samples, it also has some drawbacks. Firstly, because GANs aim to achieve a Nash equilibrium between the generator and discriminator throughout the entire training process, this task has proven to be challenging in practical appli cations. Moreover, the training process of GANs is often unstable and may suffer from issues such as vanishing or exploding gradients. Another challenge lies in the fact that, despite GAN's ability to consistently generate samples, the produced samples frequently exhibit limited diversity, with small differences between them. Such minimal variations in samples do not effectively increase the diversity of the data. This limitation becomes more evident when applying GAN to generate image samples with small differences between faults in industrial processes. Such minor fault variations are insufficient to effectively address the issue of data imbalance in industrial processes. Since the function of the gener ator in GAN is to generate new samples according to the distribution of the original data based on the random noise distribution, the structural design and training of the generator directly affec whether the model can generate high-quality samples.

The residual network in DL alleviates the problem of gradient vanishing in the deep network through “shortcut connections”, which can effectively improve the learning ability of complex features of deep network models. Therefore, this research applies the residual connection to the generator structure in the GAN to establish a Residual GAN (RGAN), which can effectively improve the network's ability to generate samples. At the same time, in order to ensure that the GAN network can be trained ef fectively, the introduction of the Wasserstein distance technique is implemented, addressing challenges such as mode collapse and gradient vanishing throughout the training process. The gradient penalty technique is also added to constrain the gradient of the discriminator, which further improves the stability of training and the quality of generated images. Therefore, the Residual Wasserstein GAN with Gradient Penalty (RWGAN-GP) model is proposed in this research, it can take advantage of the residual connection to better transfer the gradient and feature information, so that the generator can learn the advanced feature representation of the data more effectively. At the same time, by introducing the Wasserstein distance and gradient penalty technology to improve the training stability and image quality of the GAN, it can speed up the convergence of the network.

The main contributions of this research are summarized as follows:

1. The Residual Wasserstein GAN with Gradient Penalty (RWGAN-GP) model is proposed to generate high-quality samples for addressing data imbalance in fault diagnosis.

2. Aiming at the imbalanced data problem in the actual industrial process, a fault diagnosis framework for industrial process imbalanced data is proposed.

3. Model training is performed through limited samples to achieve in-depth verification of the model, making the verification results closer to actual scenarios and fully establishing the effectiveness of the proposed model.

This research is structured as follows: In Section 2, we introduce related work. In Section  3, we describe in detail the proposed framework for imbalanced fault diagnosis of industrial processes. In Section 4, we describe the two different datasets and the related experimental results. In Section  5, we summarize our work and give an outlook.

## 2 | Related Work

## 2.1 | Imbalanced Data Processing Method Based on Traditional Machine Learning

In the past research on imbalanced data problems, traditional Machine Learning (ML) methods have been widely used by many scholars, and some progress has been made. These methods aim to change the training data set to improve the situation of data imbalance, or increase the focus on minority samples by modifying the decision-making process of the classification model, which can reduce the model's deviation from the minority class in the imbalanced data set. Collectively, these techniques can be categorized as data-based and algorithm-based approaches.

The data-based processing method is to improve the imbalanced data set through some mechanisms in order to obtain a balanced data distribution. For example, under-sampling is a resampling technique widely used for imbalanced data. Bin et al. [18] applied the random under-sampling technique to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, combined with other measures, it achieved the accuracy of breast cancer classification up to 99.42%. However, some samples of the main category are deleted during the under-sampling process, resulting in the loss of information, which may lead to underfitting problems. Xie et  al. [19] proposed a data undersampling method that uses a series of density peaks to gradually extract instances from most imbalanced data classes, generates a sampling sequence according to the importance of the instance for classification, and extracts important instances from the sequence step by step, to automatically determine the optimal under-sampling size for most classes, alleviating the information loss problem to some extent. Oversampling, on the other hand, balances the dataset by increasing the number of samples in that minority class. As a typical oversampling technique, SMOTE is widely used in imbalanced data problems. However, when SMOTE generates synthetic samples, it may overemphasize some specific patterns or features, causing the model to overfit on these features. In response to this problem, Pan et  al. proposed the Adaptive-SMOTE method, by adaptively selecting the Inner and Danger data sets from the minority class, and compiling a new minority class based on the selected data, thereby preventing the expansion of the category boundary and strengthening the distribution characteristics of the original data. Soltanzadeh et al. [20] proposed Range-Controlled SMOTE (RCSMOTE) to generate synthetic samples with an accurately calculated safe range, where the range is calculated according to the characteristics of the input data to avoid the overfitting problem.

Unlike data sampling methods, algorithm-based workarounds do not change the distribution of the training data, but modify the minority class weights, or shift the decision threshold in a way that reduces bias towards negative classes. Sun et al. [21] introduced the cost item into the AdaBoost learning framework and explored different cost-sensitive boosting algorithms. Frumosu et  al. [22] improved the cost-sensitive classification strategy to address the data imbalance problem in manufacturing engineering. Although these methods have made some achievements, the performance of cost-sensitive learning is very dependent on the definition of cost. If the cost is not well defined, it may lead to poor model performance and even produce unstable results.

## 2.2 | Imbalanced Data Processing Method Based on DL

The rapid development of DL provides another possibility to solve the problem of data imbalance. Anand et al. [23] found very early that if the imbalanced data is fed into the neural network, the gradient component length of the minority class is much smaller than the gradient component length of the majority class during backpropagation. In other words, the update weight of the model is mainly dominated by the samples of the majority class, while the influence of the minority class is very weak. This leads to an increase in the network's classification error rate for the minority class. But in the field of industrial process fault diagnosis, these misdiagnosed minority class samples are crucial. To solve this problem, a large number of data-based or algorithm-based methods have been proposed. For example, one study proposed a Digital Twin-assisted Multi-scale Residual Self-Attention Feature Fusion Network (MRFFN), which significantly improved the accuracy of hypersonic vehicle fault diagnosis [24]. Another study introduced a Dynamic Normalization Supervised Contrastive Network (DNSCN) and a multi-scale compound attention mechanism, effectively addressing the sample imbalance issue in gearbox fault diagnosis [25]. Furthermore, to tackle the challenges of information utilization and feature extraction in multi-sensor data fusion, a lightweight convolutional double regularization contrastive self-attention network was developed, achieving outstanding performance in small-sample fault diagnosis of aerospace bearings [26]. These studies demonstrate that advanced deep learning techniques and innovative model optimization strategies can significantly enhance the performance of fault diagnosis systems [27]. For the WHOI-Plankton imbalanced dataset  [28], Lee et  al. [29] used RUS technology for the majority class during network pre-training, and then fine-tuned by retraining the original data. The experimental results show that the classification performance of the network has been significantly improved. Buda et  al. [30] systematically studied the impact of imbalanced data on the classification performance of convolutional neural networks, and compared the different effects of oversampling, under-sampling, and two-stage training. Although the development of DL has provided some methods to solve the problem of imbalanced data, scholars still need to continuously explore new methods.

The emergence of GAN has provided a solution for generating diverse images and handling class-imbalanced datasets [31–33]. GAN consists of a generator and a discriminator. The generator maps input random data to the distribution of real image classes and generates similar images to deceive the discriminator. The discriminator's main task is to distinguish between real images and generated images from the generator. Through adversarial learning between the two networks, the performance of both the generator and the discriminator is continuously improved. Ideally, GAN reaches a Nash equilibrium where both the generator and the discriminator reach an optimal state. In this ideal state, the generated images from the generator are considered approximations of samples from the real distribution. In this sense, GAN can be used to learn the distribution of the training dataset and generate additional sample images for data augmentation.

Yang et  al. [34] utilized GAN to generate new fault samples, addressing the issue of imbalanced original datasets. The results showed that GAN-based data augmentation significantly improved the accuracy of fault classification models. Radford [35] proposed Deep Convolutional Generative Adversarial Networks (DCGAN), which improved GAN by using deep convolutional neural networks, enabling the generator to generate higher-resolution images. Maayan et al. [36] used DCGAN to synthesize limited liver lesion CT images for data augmentation, leading to a significant enhancement in classifier specificity.

However, the original GAN suffers from training instability, gradient vanishing, and mode collapse as it requires reaching a Nash equilibrium during training. To address these issues, Arjovsky et  al. [37] introduced Wasserstein GAN (WGAN), which replaces the Jensen-Shannon (J-S) divergence with Wasserstein distance. By optimizing the objective function without changing the network structure of GAN, WGAN effectively tackles the problems of gradient vanishing and mode collapse during training. However, WGAN uses weight clipping to enforce the Lipschitz constraint on the discriminator, resulting in slow gradient changes during training and poor training performance. To overcome this problem, Gulrajani et al. [38] replaced weight clipping with Gradient Penalty (GP), penalizing the norm of the gradient of the discriminator with respect to its input. This significantly improves the performance and stability of WGAN, allowing its application in various GAN architectures. Li et al. [39] proposed an Auxiliary-Class Wasserstein GAN with Gradient Penalty (ACWGAN-GP) to perform data augmentation on insufficient rotation machinery fault data, generating highquality image samples for classification. Dai et al. [40] applied a balanced GAN with gradient penalty to defect image detection and classification in spot welding, improving the performance of industrial detection and classification by augmenting the minority class images.

In addition, to address the limitation of GANs in generating labeled images, Mirza et  al. [41] proposed Conditional GAN (CGAN), which incorporates class labels as conditional inputs to both the generator and the discriminator, enabling the generation of images with specific class labels. Yin et  al. [42] designed a gradient penalty-based conditional WGAN using One-Dimensional Convolutional Neural Networks (CWGAN-GP1DCNN) to augment different categories of bearing fault data.

Although different improvements have been made for different defects of GAN, and certain results have been achieved, but due to the unreasonable design of the network structure, the generator may not be able to perfectly simulate the real data distribution. If the generator cannot fully capture the features and details of the real data, the generated image may be quite different from the real sample, such as image blur, distortion, inaccurate color, unclear structure and other problems. Therefore, in order to improve the ability of the GAN network to generate samples, the network structure of the generator can be designed and improved.

## 3 | Method

## 3.1 | GAN, WGAN, and Gradient Penalty

## 3.1.1 | GAN

GAN is a DL architecture for generative models, proposed by Goodfellow et  al. in 2014. The fundamental principle involves adversarial training between two neural networks—namely, the generator and the discriminator—in an attempt to reach a Nash equilibrium through iterative training processes, so as to obtain a generator and a discriminator with good performance.

![](images/19551a064d4e56ea0d501f49907e78ca46753d6ec3bdec6b917795a0f90634fe.jpg)  
FIGURE 1 | GAN network structure diagram.

Its structure is shown in Figure 1. The loss function of GAN is as follows:

$$
\underset {G} {\min} \underset {D} {\max} J (D, G) = E _ {x \sim P _ {d a t a}} [ \log D (x) ] + E _ {z \sim P _ {z}} [ \log (1 - D (G (z))) ]\tag{1}
$$

Among them, x comes from the real data $P _ { d a t a } ,$ z comes from the prior distribution $P _ { z }$ .The probability that the output $D ( \cdot )$ comes from $P _ { d a t a } . \ E _ { x { \sim } d a t a }$ Denotes the expected value of x in the real data $P _ { d a t a }$ distribution and $E _ { z \sim P _ { \tau } }$ denote the expected value of z sampled from the noise z. The training process of D and G can be regarded as a binary minimax problem. The goal of the generator is to minimize this objective function, and the goal of the discriminator is to maximize the objective function. In practical applications, the discriminator's output (represented as D(x)) assume distinct values as it evaluates whether the input data originates from the real distribution or is generated by the generator. If the data comes from the real distribution, the discriminator output is 1, and the result after taking the logarithm is zero; if the data comes from the generator's output, the discriminator output is 0, causing log(1 − D(G(z))) to be zero. However, since the goal of the generator is to fool the discriminator into making its output real data, D(G(z)) outputs greater than 0, making log(1 − D(G(z))) negative. Such a design aims to confuse the discriminator to the greatest extent, making it indistinguishable between real samples and generated samples, thereby improving the performance of the generative model.

## 3.1.2 | WGAN

Although GAN can achieve data generation, but it also has many problems to be solved. First of all, as an ideal state, the Nash equilibrium between the generator and the discriminator is basically difficult to achieve in actual training. But it is difficult for the generator and discriminator in GAN to find a common indicator to indicate that the training can end. In other words, non-convergence and unstable training have become the primary problems of GAN. Secondly, at the initial stage of training, since the generator has not obtained enough distribution characteristics of real data, the generated simple samples are easily recognized by the discriminator, causing the gradient vanishing of the generator, making it difficult to train the generator network to achieve the desired effect. In conventional GAN frameworks, the Jensen-Shannon (J-S) divergence serves as a metric for quantifying the dissimilarity between the distribution of generated samples and that of real samples. However, when the probability that the two data distributions intersect during training is zero, the J-S divergence will be a constant, which will cause the gradient to be zero, that is, the gradient vanishing.

To address this issue, Arjovsky et al. introduced the Wasserstein distance to assess the dissimilarity between the distribution of generated samples and the authentic sample distribution, presenting a novel GAN model known as WGAN. WGAN offers significant advantages over traditional GANs in terms of training stability and convergence. By introducing the Wasserstein distance as its objective function, WGAN addresses common issues in traditional GAN training, such as gradient vanishing and mode collapse. The Wasserstein distance provides more stable gradient information, leading to a smoother training process for both the generator and the discriminator, thus enhancing overall training stability and effectiveness. Additionally, WGAN reduces the adversarial nature between the generator and the discriminator, contributing to a more reliable training procedure.

The Wasserstein distance, which is a method of measuring the difference between two probability distributions, can remain continuous even when the probability of the intersection of the two distributions is zero, so it can circumvent issues associated with gradient vanishing during the training process. The formal expression of the Wasserstein distance is as follows:

$$
W (p, q) = \inf _ {\gamma \sim \prod (p, q)} E _ {x, y \sim \gamma} \left[ \| x - y \| \right]\tag{2}
$$

where $p$ and q are used to represent two probability distributions, $\prod ( p , q )$ represents the set of all possible joint distributions combining distributions p and q. For each possible joint distribution γ, we can sample $( x , y ) \sim \gamma$ from it to get a sample x and y, and calculate the distance x − y of the pair of samples, so we can calculate the expected value $E _ { x , y \sim \gamma } \left[ \| x - y \| \right]$ of the distance between samples under the joint distribution 𝛾. The lower bound that can be taken on this expected value in all possible joint distributions is the Wasserstein distance.

In other words, the Wasserstein distance measures the minimum cost from one distribution to another, and represents the distance between distributions. Moreover, it is a derivable convex function. All these advantages make Wasserstein distance more suitable for computing gradients.

Based on the idea of Wasserstein distance, the loss function of WGAN is designed as follows:

$$
\min _ {G} \max _ {D \in \Omega} E _ {x \sim P _ {d a t a}} [ D (x) ] - E _ {z \sim P _ {z}} [ D (G (z)) ]\tag{3}
$$

where, Ω represents a collection of 1-Lipschitz functions whose weights lie in the compact space [ − 𝜔, 𝜔]. If the discriminator has been trained to an optimal state, then the minimization value function of the generator parameters will be equivalent to minimizing the Wasserstein distance between the generated distribution and the true distribution.

Furthermore, compared with GAN, the logarithmic operation is canceled in the objective function of WGAN, which avoids the problem that when the output of the discriminator is close to 0 or 1, the gradient of the logarithmic function will become very small, resulting in an unstable training process. And in WGAN, the task of the discriminator is converted from the binary classification of judging true and false to the regression task of approximately fitting the Wasserstein distance, so the sigmoid operation of the last layer of the discriminator can be canceled, and the mode collapse problem can be avoided to a certain extent.

The introduction of Wasserstein distance into GAN does not change the network structure of GAN, but only by improving the loss function, it can well alleviate the problem that the training is not easy to converge. In contrast to traditional GANs, the training dynamics of WGAN are characterized by increased stability and swifter convergence; nevertheless, challenges persist in generating high-quality samples consistently.

## 3.1.3 | Gradient Penalty

Although WGAN can effectively improve the stability and training effect of GAN, it also has some disadvantages. First, WGAN may not be good enough for some complex generative tasks, such as generating high-resolution images or videos, because it still suffers from training instability and mode collapse. Second, WGAN requires more training time to achieve good results. Third, in WGAN, to calculate the Wasserstein distance, it is necessary to ensure that the discriminator function is Lipschitz continuous. Hence, in WGAN, the application of weight clipping is employed to restrict the weight values within a specified range, contributing to enhanced training stability. But this also creates some problems. Weight clipping confines the neural network's weights within a predetermined range, thereby restricting the model's expressive capacity. This limitation can impede the model's ability to comprehensively capture the underlying data distribution. And, choosing the right amount of trim often requires a lot of experimentation and adjustment.

Therefore, Gulrajani et al. proposed an improved WGAN, using gradient penalty to instead of weight clipping to further improve training stability and generation effect. Gradient penalty can ensure that the discriminator satisfies Lipschitz continuity constraints while reducing training time and computational cost, that is, WGAN-GP. The loss function and WGAN-GP objective are as follows:

$$
\min _ {G} \max _ {D \in \Omega} E _ {x \sim P _ {d a t a}} [ D (x) ] - E _ {z \sim P _ {z}} [ D (G (z)) ] + \lambda \Big (\left\| \nabla_ {\hat {x}} D (\hat {x}) \right\| _ {2} - 1 \Big) ^ {2} (4)
$$

Among them, $\widehat { x } \gets \varepsilon x + ( 1 - \varepsilon ) G ( z ) .$ . The gradient penalty term promotes gradient smoothness by minimizing the difference between the 2-norm and 1-norm of the discriminator gradient. λ is the weight of the gradient penalty, which is used to control the intensity of the gradient penalty. By adding the gradient penalty term, the possibility of gradient explosion or gradient vanishing problems during the training process is reduced, making it easier to converge. Moreover, gradient penalty does not need to manually set the weight clipping range, which reduces the adjustment work of hyperparameters and makes the model easier to train.

In short, opting for gradient penalty over weight clipping enables a more precise estimation of gradients, facilitates improved utilization of gradient information during training, and enhances the overall stability and reliability of network training.

## 3.2 | Residual WGAN-GP

Although WGAN-GP can improve the stability of training, in the case of complex data distribution or extremely large or extremely small penalty coefficients, the model may suffer from training oscillation or divergence problems that lead to training instability. In some deep models, vanishing gradient or exploding gradient problems still occur.

Residual Network (ResNet) is a deep convolutional neural network structure proposed by Kaiming He et  al. in 2015 [43]. It solves the gradient vanishing and gradient explosion problems in deep neural networks by introducing residual blocks. The residual block structure is shown in Figure  2. In the residual block, the input data directly skips some of the layers, and the output is adjusted through a residual mapping. The residual block can be described by the formula as:

$$
\mathcal {F} (x) = R e L U (C o n v (R e L U (C o n v (x)))) y = \mathcal {F} (x) + x\tag{5}
$$

where, x denotes the input of the Residual Block, ReLU( <sup>⋅</sup> ) signifies the activation function, and Conv(<sup>⋅</sup>) denotes the convolution layer. In instances where the input and output dimensions of the module coincide, the skip connection ⨁ is utilized to elementwise add the two feature matrices.

Therefore, combined with the problem that GAN is prone to vanishing generator gradients during training, this research introduces the residual block structure into the generator structure, and uses the residual connection to help the transfer of feature information in the generator, so that the generator can learn the representation of data features better. Therefore, the introduction of residual connections can improve the problem of generator gradient vanishing during the training process and improve the stability of training.

The structural diagrams of the RWGAN-GP generator and discriminator proposed in this research are shown in Figure 3 and Figure 4.

The generator's input comprises a 100-dimensional random noise vector, adhering to a Gaussian distribution, and then, this 100-dimensional noise vector is converted into a feature map with different channels by applying the transposed convolution operation multiple times. The 100-dimensional noise vector is first expanded into a 1,024-channel feature map through transposed convolution, and then continues to perform multiple transposed convolution operations. The number of channels is (1,024, 512, 256, 128, 64, 32). The output channel of the last transposed convolution needs to be determined according to the specific task, and can be set to 1 or 3 as needed. The final number of channels can be selected based on specific

circumstances. After each transposed convolutional layer, a batch normalization layer is added to speed up the training process and improve the stability of the model. Following the batch normalization layer, the introduction of the ReLU activation function is employed to incorporate nonlinear characteristics. This facilitates the generator in learning intricate image features. It should be noted that this research introduces residual connections after the fifth and sixth transposed convolution operations to help improve the performance and stability of the generator. After the last transposed convolutional layer, instead of using a batch normalization layer, the Tanh function is used directly. The Tanh function maps the output value of the generator to [ − 1, 1] to ensure that the generated image pixel value range is within a suitable interval. The final output is a generated image of 256\*256\*3.

![](images/df6b6affb847ad9793e25a911ab2d7aad8cc63cbd161da0b9b842fd69c1cafcc.jpg)  
FIGURE 2 | Residual block structure diagram.

The input to the discriminator is an image with a size of $2 2 4 ^ { * } 2 2 4 ^ { * } 3 ,$ and then a convolution operation is applied to the input image, incrementally raising the number of channels to 1,024, with a simultaneous reduction in the image's width and height. After each convolutional layer, an instance normalization layer is added to speed up the training process and improve the stability of the model. In DL, normalization technology has become one of the important means to improve the stability and performance of neural network training. Instance normalization is a method of normalizing each sample (instance) independently instead of normalizing the entire batch. Its definition is as follows:

$$
\mathrm{InstanceNorm} (x) = \frac {x - \mu}{\sigma}\tag{6}
$$

Among them, 𝜇 represents the mean of the sample, and 𝜎 represents the standard deviation of the sample. After the instance normalization layer, the LeakyReLU activation function is introduced to introduce nonlinear characteristics and enhance the expressive ability of the model. The introduction of LeakyReLU solves the problem of complete inactivation of the standard ReLU function in the negative area. It allows small gradients of negative input values to pass through, thereby maintaining a certain activity in the negative area. Its definition is as follows:

$$
f (x) = \left\{ \begin{array}{l l} x & \text {if} x > 0 \\ \alpha x & \text {if} x \leq 0 \end{array} \right.\tag{7}
$$

![](images/885b94f9c580e432b334843545eb1be723090e8417e6f32bc808ddfb89345885.jpg)  
FIGURE 3 | The structure of generator.

![](images/530f4815a1930db701c779c9d3839e36c70b43b0c708a8262ea965a421316493.jpg)  
FIGURE 4 | The structure of discriminator.

where 𝛼 is a small positive number, usually close to zero, to control the gradient in the negative region. This prevents the Leaky ReLU function from being completely deactivated when activating neurons, helping to alleviate the vanishing gradient problem. After the last convolutional layer, no instance normalization layer is used. The output layer translates image features into a scalar that signifies the probability of the image being real. And the network does not perform sigmoid activation function processing on the output at the end, because the goal of the discriminator is to output a probability value, rather than map the feature to a value within a fixed range.

## 3.3 | Fault Diagnosis Framework for Imbalanced Data Based on RWGAN-GP

This paper presents an industrial process imbalance data fault diagnosis framework based on data enhancement using RWGAN-GP. It is divided into two stages: offline training and online diagnosis. Its overall framework is shown in Figure 5.

## 3.3.1 | Offline Training Phase

In the offline phase, the RWGAN-GP model is used to deal with the data imbalance problem. First, a separate RWGAN-GP model is trained for each fault category using the imbalanced dataset. Before training the model, the original training data set needs to be preprocessed. When training the model, it is necessary to correctly select the hyperparameters of the model and train a RWGAN-GP model that converges and has good performance. After obtaining the model, different models are used to generate different amounts of enhanced data for different fault categories. The specific number of generated data is determined based on the difference between the existing number of different fault categories and the number of samples required for the equilibrium state. Afterwards, in order to test the authenticity and reliability of the enhanced samples generated by different models, the generated samples and the original samples were analyzed for similarity under different indicators. Ultimately, the generated augmented data is employed to maintain the balance of the training dataset and mitigate the deleterious effects of data imbalance on online fault diagnosis.

## 3.3.2 | Online Diagnostic Phase

In the online diagnosis phase, the augmented data generated using the RWGAN-GP model is mixed with the original samples to create a balanced dataset. Then, these data are sent to the VGG-16 classification network for deep feature extraction and fault classification to achieve fault diagnosis functions.

## 4 | Experiment

In order to verify the proposed method, this research conducts experiments on two datasets, the PRONTO benchmark dataset and the NEU surface defect dataset. For each dataset, the RWGAN-GP model undergoes training to generate novel samples. Subsequently, the generated samples undergo a quality assessment to determine their suitability for fault diagnosis. Following the evaluation, the generated samples are incrementally added into the original dataset for fault diagnosis. This simulates a multi-type imbalanced fault diagnosis scenario in engineering practice for the two datasets.

## 4.1 | PRONTO Benchmark Dataset

## 4.1.1 | Data Introduction

The PRONTO benchmark dataset comes from an industrialscale, fully automated, high-pressure, multiphase flow facility at Cranfield University's Process Systems Engineering Laboratory. A flow diagram of this multiphase flow facility is shown in Figure 6. The main research focus of the facility is multiphase flow consisting of water, air and oil. The third phase, oil, is not used in this case, only the state of water and air is concerned. During the experiments, air and water flows were the inputs to the facility, and their flow rates were controlled to achieve different operating states. After the input water flow is mixed with air in the mixing zone, it is led to the 2″ vertical riser through the horizontal pipe, and there is an S-shaped connection along the middle of the riser. After reaching the top of the standpipe, the mixed flow is separated in the separator. The two separators are in turn flow, the water flows back to the water storage tank, and the air is discharged into the atmosphere after separation. At the same time, two transparent parts are installed at the bottom and top of the riser for observing the flow state. In Figure  6, the piping of the test facility, the mixing and separation process of water and air, and the acquisition locations of the different process variables can be seen.

![](images/6f914bb3d8cdd41034d4e5186165b3fdd864c2a0fdb5be812218a100fa42339a.jpg)  
FIGURE 5 | Fault diagnosis framework of industrial process imbalanced data based on RWGAN-GP.

![](images/3ccdf9cb0dc28d4e77a26f119ff143d49408361aa6b2c3a16b400b55f656fb85.jpg)  
FIGURE 6 | Schematic diagram of the overall flow of the multiphase flow device.

The collected datasets describe the facility, tests and induced failures under different operating conditions, which are suitable for algorithm development and verification of fault detection, fault identification, fault classification, fault severity assessment, and fault evolution monitoring. The types of data collected include structured process data, i.e. sensor data, unstructured video surveillance data and text data. Two transparent parts are installed at the bottom and top of the 2″ vertical riser to observe the flow state. The video surveillance data is collected from this transparent pipe. The data set contains a normal state and three fault states, respectively normal, slug, air leakage and diverted flow. It should be noted that, because Slugging is an intermittent fault, in this experiment, only normal, diverted flow, and air leakage are selected as the research objects.

Table  1 displays the correlation between water flow and air flow during the normal operational state of the multiphase flow equipment. Tables  2 and 3 display the associations between water flow and air flow during the conditions of air leakage and diverted flow, respectively.

In order to study the image data imbalance problem in industrial processes, this study is based on the video dataset, which first obtains the original image dataset by extracting different frames of the video. Subsequently, we constructed an imbalanced data set based on the random sampling principle to adapt to the classification needs of different faults. An example image is shown in Figure 7. The sample images show discernible variations in the flow conditions within the pipeline under different operating conditions. Table  4 shows the number of samples in different states. In this way, the sample size distribution of the data set is more in line with the actual situation in the industrial process.

## 4.1.2 | Data Preprocessing

During the experiment, the water flow information in the pipeline is crucial for equipment status evaluation. However, as can be seen from Figure 7, the original images contain various unrelated elements, such as brackets for pipeline fixation, cables for data transmission and power, and varying lighting conditions at different times. These factors pose challenges to key information analysis. Hence, preprocessing the original images is of utmost significance. In this article, the approach we adopt is to first crop the obtained image data before inputting the original sample into the RWGAN-GP model, leaving only the water flow part related to the operating status of the equipment to reduce the interference of irrelevant information on the experiment. At the same time, by reducing the irrelevant feature information contained in the sample, the goal of the generated model can be fixed on the main features, improving the quality of the images generated by the model. The processed image is shown in Figure 8.

## 4.1.3 | Model Training and Hyperparameter Selection

When training the model, there are some hyperparameters that need to be set manually. Considering the experimental results, we set some as follows (see table 5):

TABLE 2 | Flow conditions for air leakage.

<table><tr><td>Air flow rate (sm3/h)</td><td>Water flow rate (kg/s)</td><td>Valve opening (°)</td></tr><tr><td>150</td><td>0.5</td><td>0,10,20,30,40,90</td></tr><tr><td>120</td><td>0.1</td><td>0,5,10,15</td></tr><tr><td>150</td><td>0.5</td><td>0,5,10,15,20,25</td></tr></table>

TABLE 3 | Flow conditions for diverted flow.

<table><tr><td>Air flow rate (sm3/h)</td><td>Water flow rate (kg/s)</td><td>Valve opening (°)</td></tr><tr><td>120</td><td>0.1</td><td>5,10,15,20,30,40,50,60</td></tr><tr><td>150</td><td>0.5</td><td>10,20,30,40,45,50,60</td></tr></table>

TABLE 1 | Flow conditions for normal.

<table><tr><td></td><td colspan="5">Water flow rate (kg/s)</td></tr><tr><td>Air rate (sm3/h)</td><td>0.1</td><td>0.5</td><td>1</td><td>2</td><td>3.5</td></tr><tr><td>20</td><td>—</td><td>—</td><td>—</td><td>—</td><td>Normal</td></tr><tr><td>50</td><td>—</td><td>—</td><td>—</td><td>Normal</td><td>Normal</td></tr><tr><td>100</td><td>Normal</td><td>Normal</td><td>Normal</td><td>Normal</td><td>Normal</td></tr><tr><td>200</td><td>Normal</td><td>Normal</td><td>Normal</td><td>—</td><td>—</td></tr></table>

![](images/94e3365cd5d8a8d8e0029a4ed66ca645fc23c601ef01a5b71e3d5100dfb91bbb.jpg)

![](images/71430f23f766d4152a17a70081d5e7fc86f3097082588182c78711f4882b8476.jpg)  
FIGURE 7 | Example of a partial frame image from a video dataset.

TABLE 4 | Dataset settings.

<table><tr><td>Kind</td><td>Normal</td><td>Diverted flow</td><td>Air leakage</td></tr><tr><td>Train</td><td>100</td><td>50</td><td>30</td></tr><tr><td>Test</td><td>50</td><td>50</td><td>50</td></tr></table>

The number of critic iterations is an important hyperparameter, which controls the relative number of updates of the discriminator and generator in each round of training. Setting a different number of critic iterations during training will lead to different training results, as well as differences in the stability of the model during training and the quality of the generated samples. A common practice is to train the discriminator multiple times per generator per iteration. This setting helps to improve the robustness and stability of the discriminator, allowing it to better distinguish between real samples and generated samples. However, too many critic iterations will cause the generator to be unable to obtain effective gradient information in time, thus affecting the training effect of the generator. Therefore, a balance needs to be struck between the number of iterations of the discriminator and the number of iterations of the generator, and adjusted according to the characteristics of the specific task and data set.

In order to find the optimal number of critic iterations, this article selected different critic iterations for training, and analyzed the different impacts of different critic iterations on the convergence, stability and generated sample quality of the RWGAN-GP network through the training results, the results are shown in Figure 9.

It can be seen from Figure  9, when critic iterations increase, the loss of the generator in the model will increase rapidly, while the loss of the discriminator is relatively stable without large fluctuations. This phenomenon occurs due to that when the number of iterations of the discriminator increases, the discriminator can update parameters faster than the generator and can more accurately identify the generated samples, ultimately reaching a higher performance level. The generator provides stronger negative feedback because the discriminator can better distinguish the generated samples from the real samples. So the generator tries to generate more realistic samples by adjusting the parameters, but due to the improvement of the performance of the discriminator, it is difficult for the generator to meet the requirements of the discriminator, resulting in a gradual increase in loss. Therefore, choosing an appropriate number of critic iterations is crucial to the stable training of the model. In this case, based on the training results of models with different parameters, the number of critic iterations selected was 2. In order to obtain a model with excellent sample generation ability and convergence, the epoch during training was set to 2000.

![](images/053551e9620b641487d22a3db2d042c1fa3969ac6e9d12262cba330363d4a3d8.jpg)

## 4.1.4 | Image Generation Quality and Image Similarity Calculation

In this section, we utilize the trained RWGAN-GP model to generate enhanced samples for each class of fault. The preprocessed image is used as the input of the model for training, and the training state of the model is reflected by the value of the loss function of the generator and the discriminator, such as Figure 10.

It can be seen from the figure that, initially, the loss values for both the generator and discriminator exhibit considerable fluctuations. However, with the progression of iterations, these loss values gradually stabilize, oscillating within a confined range. This observation suggests that the model has undergone effective training, rendering it suitable for sample generation.

Figure 11 is a sample example of diverted flow classes generated by different GAN models. In the order of a, b, c, d, e, and f, the images are generated by the RWGAN-GP, GAN, DCGAN, Self-Attention Generative Adversarial Network [44] (SAGAN) Multi-Scale Gradient Generative Adversarial Network [45] (MSG-GAN), and WGAN-GP models, respectively. And the generated images shown belong to the diverted flow category.

![](images/5a85d8de9df8234235d66aa7991b32b7a62a5a4482f1fddf9303ac443e75faaa.jpg)  
FIGURE 8 | Example of preprocessed image.

![](images/0f7ebd23a1edbcda3d1785266eb5942883ee1c51bc2968002da30549ad9f61d5.jpg)

![](images/24c263cf74c3eaf1ddf611ee4df55141b3266c7282071b99716fe0738c7a8cd4.jpg)

TABLE 5 | Hyperparameter selection.

<table><tr><td rowspan="2">Gradient penalty factor</td><td rowspan="2">Batch size</td><td colspan="3">Adam optimizer</td></tr><tr><td>Learning rate</td><td>Beat1</td><td>Beat2</td></tr><tr><td>10</td><td>10</td><td>0.0001</td><td>0.5</td><td>0.9</td></tr></table>

It can be seen that the images generated by different models have different characteristics. The original GAN generated image is darker, the DCGAN generated image has a large fluctuation in color, and the WGAN-GP generated image has a clear boundary of the transparent pipeline. In general, the images generated by the RWGAN-GP model typically convey the most realistic and intuitively comprehensible impressions.

For the image samples generated by the GAN network, the similarity and difference between the generated image and the original fault image can be visually compared by observing the generation. In addition, the quality of generated fault samples can be quantitatively evaluated by calculating the image similarity index between the generated image and the original image. In order to verify the effectiveness and applicability of the images generated by the RWGAN-GP model, this research selects commonly used image similarity indicators for quantitative analysis. The indicators are detailed as follows. For the convenience of description, A and B are respectively used to represent two different images.

(1) Cosine Similarity serves as a metric for assessing the likeness between two images. It quantifies the similarity of two image vectors by calculating the cosine of the angle formed between them. The cosine similarity values range between −1 and 1, with proximity to 1 indicating a higher degree of similarity between the two images. The calculation formula is:

$$
S i m i l a r i t y (A, B) = \frac {A \cdot B}{\| A \| \times \| B \|} = \frac {\sum_ {i = 1} ^ {n} \left(A _ {i} \times B _ {i}\right)}{\sqrt {\sum_ {i = 1} ^ {n} A _ {i} ^ {2}} \times \sqrt {\sum_ {i = 1} ^ {n} B _ {i} ^ {2}}}\tag{8}
$$

(2) The Mean Squared Error (MSE) serves as a prevalent index for image quality assessment, employed to quantify the disparity between two images. It computes the square of the difference between corresponding pixels in the two images and subsequently calculates the average. The smaller the value of MSE, the more similar the two images are. The calculation formula is:

$$
M S E = \frac {1}{m \cdot n} \sum_ {i = 0} ^ {m - 1} \sum_ {j = 0} ^ {n - 1} \left[ A (i, j) - B (i, j) \right] ^ {2}\tag{9}
$$

(3) The Structural Similarity Index (SSIM) is a metric employed for quantifying the structural similarity between two images. It holistically incorporates considerations for image brightness, contrast, and structural information, assessing image similarity by comparing these features. SSIM values fall within the range of 0 to 1, with proximity to 1 signifying a heightened resemblance between the two images. The calculation formula is:

![](images/f42f0829500397aa6ac1cf49bdda6e97857f40794a73b2759a20456d705e53be.jpg)  
Loss

![](images/19bb6fc24dc5ac39660274ef2f8a5b42deefadcfcf97e01c7b4b6780e31f88a5.jpg)  
Loss

![](images/1c91ceb48fd3d15a4759b262ca870dda2785cbed4e2562da5759c3c325a837f8.jpg)  
(c)

![](images/3102c0454ef73edce6994698aea66511029cd4e766d12e18ec99c3e0c2708659.jpg)  
FIGURE 9 | Model loss graph under different number of critic iterations. (a) critic iterations = 2. (b) critic iterations = 3. (c) critic iterations = 5. (d) critic iterations = 7.

![](images/9ed9a5f33e18e1cd00a480ba16240bb40ec60ed69a10b4b7fa295741fe1dbff0.jpg)  
FIGURE 10 | RWGAN-GP model training loss for PRONTO benchmark dataset.

$$
S S I M (A, B) = \frac {\left(2 \mu_ {x} \mu_ {y} + C _ {1}\right) \left(2 \delta_ {x y} + C _ {2}\right)}{\left(\mu_ {x} ^ {2} \mu_ {y} ^ {2} + C _ {1}\right) \left(\delta_ {x} ^ {2} \delta_ {y} ^ {2} + C _ {2}\right)}\tag{10}
$$

Among them, $\mu _ { x }$ and $\mu _ { y }$ represent the average value of A and B respectively, $\delta _ { x } ^ { 2 }$ and $\delta _ { y } ^ { 2 }$ represent the variance of A and B respectively, and $\delta _ { x y }$ represents the covariance of A and $\mathrm { B } ; C _ { 1 }$ and $C _ { 2 }$ are constants to prevent the denominator from being 0.

(4) The Peak Signal-to-Noise Ratio (PSNR) serves as a metric for quantifying the signal-to-noise ratio between two images. It assesses the distortion level of an image by evaluating the disparity between the original image and the reconstructed or compressed image. A higher PSNR value signifies enhanced image quality. The calculation formula is:

![](images/5add9ca0a46f030109247090493ae0f41b33a9663029653e7493ee430f931eab.jpg)  
FIGURE 11 | Different GAN models generate images. (a) RWGAN-GP. (b) GAN. (c) DCGAN. (d) SAGAN. (e) MSG-GAN. (f) WGAN-GP.

$$
P S N R = 2 0 l o g _ {1 0} \left(\frac {M A X _ {i}}{\sqrt {M S E}}\right)\tag{10}
$$

These indicators can be used to measure the similarity or difference between images. The difference is that cosine similarity is a similarity index based on vector angles, while MSE, SSIM, and PSNR are indexes based on pixel differences. And MSE pays more attention to the difference between image pixels, while SSIM and PSNR comprehensively consider the brightness, contrast and structural information of the image. Therefore, SSIM and PSNR can better reflect the subjective perception of human eyes on image quality than MSE. These indicators have different characteristics and scope of application in measuring image similarity and quality, and appropriate indicators should be selected to evaluate images according to specific application scenarios.

In this research, for different similarity indicators, the same cal culation strategy is used for calculation. First, calculate the similarity between different generated images and all original images, sum and average the similarity results, and then calculate and average the similarity results of all generated images. Figure 12 is an example of calculating cosine similarity between a generated image and an original image. The specific similarity indicators between the generated image and the original image are shown in Table  6. The ‘self’ represents the average similarity between each original image and the original data set, which can be regarded as the benchmark for comparison. That is, the closer the similarity value between the image generated by different models and the original image is, it proves that the image generated by the model is of higher quality and more reliable.

As can be seen from Table  6, there is generally a high degree of similarity between the generated images of the three models and the original images. However, the similarity between the images generated by the RWGAN-GP model and the original images is closer to the benchmark, indicating that compared with the GAN, WGAN-GP, SAGAN, MSG-GAN models, the RWGAN-GP model generates higher quality images.

In addition, in order to verify the training effect of the model as the number of iterations increases, this research compares the similarity between the generated images of different training batches and the original images, as shown in Table 7. Among them, 200, 300, 500, 1,000, and 2000 respectively represent the epochs of RWGAN-GP model training.

As can be seen from Table 7, as the epoch of the model increases, the similarity between the model-generated image and the original image becomes closer and closer to the baseline value. This shows that with the training, the quality of generated images is constantly improving, and it also shows that the generation ability of the network is constantly increasing.

Additionally, we investigated the impact of different convolutional kernel sizes on model performance in both the Generator and Discriminator. As shown in Table 8, when the kernel size is 4, the generated images exhibit higher Cosine Similarity, SSIM, and PSNR, and lower MSE compared to other kernel sizes. This indicates that a kernel size of 4 results in higher-quality images produced by the model.

Based on the aforementioned analysis, it is evident that the RWGAN-GP model is proficient in generating novel samples exhibiting a high degree of similarity to the original samples.

## 4.1.5 | Imbalance Fault Diagnosis

The commonly used model evaluation indicators for classification problems are accuracy and error rates. However, these two metrics alone are not sufficient when dealing with imbalanced industrial process data. Majority class samples tend to guide the classifier, while minority class samples tend to be misclassified. For example, if a given dataset has only 1% of the minority class examples, the classifier can achieve 99% accuracy by simply labeling all examples as negative. Of course, this model brings no benefits. Therefore, using the accuracy rate alone may ignore the important information provided by other indicators, and it is necessary to find additional evaluation indicators to meet the needs of classification research on imbalanced datasets. These indicators are described below.

Precision, recall and F1-score are commonly used indicators when dealing with imbalanced industrial process fault classification problems. The accuracy rate signifies the proportion of correctly classified samples by the classifier to the total number of samples predicted as a specific class among those identified as such. Precision measures how well a classifier classifies minority classes. The recall rate signifies the ratio of correctly classified samples by the classifier to the total number of samples belonging to a specific class within the samples of a particular category. Recall assesses the classifier's capability to identify minority classes. F1-score, as the weighted harmonic mean of precision and recall, comprehensively considers the classifier's classification performance across all categories. In imbalanced datasets, F1-score is often more useful than accuracy because it balances the classification performance across all classes.

![](images/c393e88b49d31cf7f9dba18f08596f728b409061d2ba150473555d3841f1a016.jpg)

![](images/372e89a9cd0434a88527ee749411bb591196ae4ebf7c76829b340c34b6b58871.jpg)  
FIGURE 12 | Example of calculating the similarity between the generated image and the original image.

TABLE 6 | Similarity of different metrics.

<table><tr><td></td><td>Self</td><td>GAN</td><td>WGAN-GP</td><td>SAGAN</td><td>MSG-GAN</td><td>RWGAN-GP</td></tr><tr><td>Cosine similarity</td><td>0.9726</td><td>0.9782</td><td>0.9705</td><td>0.9553</td><td>0.9665</td><td>0.9732</td></tr><tr><td>MSE</td><td>91.8780</td><td>99.7074</td><td>102.5098</td><td>99.4196</td><td>100.9490</td><td>94.9303</td></tr><tr><td>SSIM</td><td>0.5917</td><td>0.4973</td><td>0.5850</td><td>0.5493</td><td>0.4458</td><td>0.5908</td></tr><tr><td>PSNR</td><td>—</td><td>17.0541</td><td>15.7480</td><td>14.5292</td><td>14.4616</td><td>16.2543</td></tr></table>

TABLE 7 | Similarity of different metrics.

<table><tr><td></td><td>Self</td><td>RWGAN-GP200</td><td>RWGAN-GP300</td><td>RWGAN-GP500</td><td>RWGAN-GP1000</td><td>RWGAN-GP2000</td></tr><tr><td>Cosine similarity</td><td>0.9726</td><td>0.9589</td><td>0.9639</td><td>0.9671</td><td>0.9651</td><td>0.9732</td></tr><tr><td>MSE</td><td>91.8780</td><td>102.5638</td><td>100.7959</td><td>100.1498</td><td>98.8902</td><td>94.9303</td></tr><tr><td>SSIM</td><td>0.5917</td><td>0.5049</td><td>0.5211</td><td>0.5299</td><td>0.5392</td><td>0.5908</td></tr><tr><td>PSNR</td><td></td><td>13.8864</td><td>14.7868</td><td>15.1870</td><td>14.9799</td><td>16.2543</td></tr></table>

TABLE 8 | Impact of different convolutional kernel sizes on model performance.

<table><tr><td></td><td>RWGAN-GP, kernel size = 4</td><td>RWGAN-GP, kernel size = 3</td><td>RWGAN-GP, kernel size = 5</td></tr><tr><td>Cosine similarity</td><td>0.9732</td><td>0.9685</td><td>0.9706</td></tr><tr><td>MSE</td><td>94.9303</td><td>102.2167</td><td>94.9982</td></tr><tr><td>SSIM</td><td>0.5908</td><td>0.4090</td><td>0.3801</td></tr><tr><td>PSNR</td><td>16.2543</td><td>14.5253</td><td>15.3054</td></tr></table>

The four performance evaluation indexes can be calculated by the following formulas:

$$
\left\{ \begin{array}{c} \text {accuracy} = \frac {T P + T N}{T P + T N + F P + F N} \\ p r e c i s i o n = \frac {T P}{T P + F P} \\ r e c a l l = \frac {T P}{T P + F N} \\ F 1 = 2 \cdot \frac {p r e c i s i o n \times r e c a l l}{p r e c i s i o n + r e c a l l} \end{array} \right.\tag{11}
$$

## DIAGNOSTIC ACCURACY

![](images/b4c893250a96d0f53375c0d643690a942c898470644b3aa170af84be4aa961a2.jpg)

FIGURE 13 | Fault diagnosis results of different data augmentation methods.

Among them, TP, TN, FP, and FN represent the sample sizes for judging positive cases as positive cases, negative cases as negative cases, negative cases as positive cases, and positive cases as negative cases, respectively.

To enhance the efficacy of imbalanced fault diagnosis, we augment the imbalanced data set with the generated fault samples so that the number of fault samples matches the number of

normal samples. Then, we use different methods to train and classify the data sets before and after enhancement, and compare the experimental results to analyze the improvement and positive effect of the new images generated by the generative model on imbalance fault diagnosis. At the same time, we also use other GAN models to generate enhanced data to compare the effectiveness of the RWGAN-GP model proposed in this article in improving the data imbalance problem.

![](images/a6065ef434598d60521bcbecc741f35889ea570fd8e3bb4b8bd968df2ff93ab9.jpg)  
(a)

![](images/e0ff83df7f2a1f7e8ba8947eb5bd24a58b2f9b218ae548830fdc68f9bc4e588e.jpg)  
(c)

![](images/a82aac2d0e63e057990ec40cdd4d8f6cb9b4a86a85418c727264fdbc3f7aea1d.jpg)  
(e)

![](images/e0c5aa9406f3b918d5f343ea2191339e52cfd63291d23edf695081c7b3e22de7.jpg)

![](images/6364c7b8faa3f3673957e0c09cde2c9cbe16061d993c22fbe139e4c2c47ff574.jpg)  
(d)

![](images/8fc43a21a1b860c5df31171f3c99e8ab0f67321f20aab725c213a0cfcfe3d132.jpg)  
(f)  
FIGURE 14 | VGG network classification confusion matrix of different data enhancement methods. (a) imbalanced data. (b) GAN. (c) WGAN-GP. (d) SAGAN. (e) MSG-GAN. (f) RWGAN-GP.

patches

In the experiments, we used DL methods (VGG-16 network) and traditional ML methods (SVM) for fault classification. When using SVM for classification, we perform PCA dimensionality reduction on the fault samples that need to be classified. In Figure 13, we show the accuracy of two different fault data set classification methods before and after

TABLE 9 | Fault diagnosis performance indicators.

<table><tr><td>Dataset</td><td>Recall(%)</td><td>Specificity</td><td>F1 score</td><td>G-Mean</td><td>Precision</td></tr><tr><td>Imbalanced</td><td>0.920</td><td>0.960</td><td>0.460</td><td>0.940</td><td>0.920</td></tr><tr><td>+GAN</td><td>0.973</td><td>0.987</td><td>0.487</td><td>0.980</td><td>0.975</td></tr><tr><td>+WGAN-GP</td><td>0.953</td><td>0.977</td><td>0.476</td><td>0.964</td><td>0.959</td></tr><tr><td>+SAGAN</td><td>0.973</td><td>0.987</td><td>0.487</td><td>0.980</td><td>0.975</td></tr><tr><td>+MSG-GAN</td><td>0.953</td><td>0.977</td><td>0.476</td><td>0.964</td><td>0.959</td></tr><tr><td>+RWGAN-GP</td><td>0.993</td><td>0.997</td><td>0.497</td><td>0.995</td><td>0.993</td></tr></table>

![](images/5e0ca716b9a2c8c95adefdb2a938ed301a44d3b224ea9dbd4b2522ee30601a7e.jpg)  
rolled-in scale  
crazing  
pitted surface  
inclusion  
scratches

FIGURE 15 | Sample images of six typical surface defects.

enhancement. The results show that after the imbalanced data set is enhanced, both methods have achieved varying degrees of improvement in fault classification. When using imbalanced data for fault classification, the diagnostic accuracy of SVM is 0.673, while VGG-16 is 0.920. By using the GAN model for data augmentation, the accuracy of the two methods increased by 1.4% and 5.3% respectively. After using the WGAN-GP model for data augmentation, the accuracy increased by 2.0% and 3.3% respectively. And after using the SAGAN model and MSG-GAN model for data augmentation, there is still a notable improvement in accuracy. It is worth noting that the effect of data augmentation using the RWGAN model exceeds that of other GAN models, improving the accuracy of the two methods by 4.0% and 7.1% respectively. This shows that the RWGAN model has a significant improvement effect in imbalance fault diagnosis.

In order to present the fault diagnosis results more clearly, we use the confusion matrix diagram in Figure 14 to show the performance of the VGG neural network in fault data classification. When the data is imbalanced, the classifier tends to misclassify minority class samples, but after data augmentation, the classification effect is significantly improved. This shows that with the augmentation of minority class samples, the diagnostic model demonstrates an enhanced ability to discern fault information. Especially in the data set enhanced by the RWGAN-GP model, only one fault sample was misclassified, and the fault diagnosis accuracy reached 99.3%.

Moreover, to provide a more comprehensive illustration the effect of fault diagnosis, Table  8 lists various indicators of the classification network after applying different GANs for data enhancement. The ‘+’ indicates that different data augmentation methods are used.

Table  9 shows that under the imbalanced data set, the fault diagnosis performance is poor, but after image enhancement through different models, the performance is improved. In particular, after adding the images generated by the RWGAN-GP model, the fault diagnosis performance is the best. Therefore, it can be concluded that using generative models for data augmentation is an effective strategy to deal with the problem of imbalanced fault diagnosis. In particular, the improved WGAN-GP model plays a very active role in imbalance fault diagnosis.

TABLE 10 | Dataset settings.

<table><tr><td>Kind</td><td>Cr</td><td>In</td><td>Pa</td><td>Ps</td><td>Rs</td><td>Sc</td></tr><tr><td>Train</td><td>180</td><td>120</td><td>60</td><td>30</td><td>20</td><td>10</td></tr><tr><td>Test</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td></tr></table>

The RWGAN-GP model demonstrates exceptional performance in addressing data imbalance issues, achieving a recall rate of 0.993, which significantly surpasses other comparative methods. This outstanding performance is attributed to the incorporation of residual mechanisms and gradient penalty, which substantially enhance the quality of generated samples, making them more representative of real data. Compared to traditional GAN, WGAN-GP, SAGAN, and MSG-GAN, RWGAN-GP not only excels in sample generation but also shows superior efficacy in practical industrial fault diagnosis. The model's effectiveness and stability in tackling data imbalance problems in industrial processes are further validated through training and verification with limited samples.

## 4.2 | NEU Surface Defect Database

## 4.2.1 | Data Introduction

This data set collects six typical surface defects of hot-rolled steel strip, namely rolling scale (RS), plaque (Pa), crack (Cr), pitting surface (PS), inclusion (In) and scratch (Sc). The database includes 1800 grayscale image samples: 300 each of 6 different types of typical surface defects.

Figure  15 illustrates sample images featuring six representative surface defects, each originally at a resolution of 200 × 200 pixels. The visual inspection reveals substantial variations in the appearance of defects within the same class. For instance, scratches (located in the last column) may manifest as horizontal, vertical, or oblique scratches. Concurrently, interclass defects, such as rolled-in scale, cracked, and pitted surfaces, exhibit similarities. Furthermore, the grayscale of intraclass defect images varies due to the impact of illumination and material alterations. In summary, the NEU surface defect database includes two difficulties, that is, defects within classes have large appearance differences, while defects between classes have similar aspects, and defect images are affected by illumination and material changes.

![](images/44c822819f5dfdd14119621cf9548f17141be8ef1ff2182ba9165aeccc075cec.jpg)  
FIGURE 16 | RWGAN-GP model training loss for NEU surface defect database.

TABLE 11 | Hyperparameter selection.

<table><tr><td rowspan="2">Gradient penalty factor</td><td rowspan="2">Batch size</td><td colspan="3">Adam optimizer</td><td rowspan="2">Critic iterations</td></tr><tr><td>Learning rate</td><td>Beat1</td><td>Beat2</td></tr><tr><td>10</td><td>10</td><td>0.0001</td><td>0.5</td><td>0.9</td><td>3</td></tr></table>

![](images/3f00d049845b0276ed6f70a755812f0022904bb4e2acae953076db4e5144fd50.jpg)  
FIGURE 17 | Examples of different samples generated by the RWGAN-GP model during training. (a) In. (b) Pa. (c) Ps. (d) Rs. (e) Sc.

![](images/d836f0ca1886dd57a93ef41428d06253cf64f9bc6c410fd912bc734f748b416e.jpg)

![](images/fda91082751c636921f61243cf2054022a1fb1c7f4192c7896458e83480cf0fe.jpg)  
FIGURE 18 | Example of calculating the similarity between the generated image and the original image.

The experimental data sets are set according to different imbalance ratios, and the data distribution of different data sets is shown in Table 10.

## 4.2.2 | Model Training and Hyperparameter Selection

Similar to the previous case, some hyperparameters need to be set during model training. Critic iterations is a key parameter, and its different settings have a significant impact on model training. Therefore, we conducted experimental training on multiple sets of critic iterations parameters to determine the most appropriate configuration.

In this case, the hyperparameter settings are shown in Table 11:

## 4.2.3 | Image Generation Quality

Similar to the previous case, first, we use the preprocessed images to train the RWGAN-GP model to generate new images for each category. After sufficient evaluation, we use the trained model for image generation. The training status of the model is reflected by the loss function values of the generator and discriminator, as shown in Figure 16. It can be observed from the loss graph that the loss values of the generator and discriminator fluctuate greatly at the beginning, but as the number of iterations increases, the loss values of both gradually stabilize and fluctuate within a smaller range. This indicates that the model is effectively trained and can be used for sample generation.

In Figure  17, sample examples generated by the RWGAN-GP model during the training process are shown. Comparing with Figure 15, it can be clearly observed that the RWGAN-GP model can accurately capture the key features of the original data, the generated images are highly similar to the original images, and it shows high-quality generation capabilities when generating images of different categories.

In order to verify the rationality and effectiveness of the generated image, we calculated the similarity between the generated image and the original image (see Figure 18). In addition, in order to avoid the bias brought by a single metric, we also use multiple similarity measures to evaluate the similarity between the generated image and the original image, and the results are recorded in Table 12. Through the example figures and tables, it becomes evident that the generated samples exhibit a noteworthy resemblance to the original samples. Different similarity measures indicate a high degree of similarity between the two samples, thus verifying the rationality and effectiveness of the image generated by the RWGAN-GP model.

## 4.2.4 | Imbalance Fault Diagnosis

In this section, we use VGG-16 network and SVM to diagnose fault images. In order to improve the effect of imbalance fault diagnosis, we added images generated by different models to the imbalanced dataset, compared the performance of the classification models before and after data enhancement, and compared the results of different models. Specifically, we compared the results of adding images generated by different generative models. Through these analyses, we verified the feasibility and excellence of the RWGAN-GP model on imbalance fault diagnosis problems. Figure 18 shows the diagnostic performance of these two methods on different fault data sets.

It can be seen from Figure  19 that when the data is in an imbalanced state, the correct rate of fault diagnosis under the two methods is at the lowest, and the correct rate under the VGG classification method is only 0.540. After using different GAN models for data enhancement, the effect of fault diagnosis has been significantly improved. Among them, the result of data enhancement using the RWGAN-GP model is the most obvious, increasing the correct rate from 0.540 to 0.673, an increase of 13.3%. The experimental results not only prove the positive effect of data enhancement on the problem of imbalanced fault diagnosis, but also show that it is very useful to introduce the improvement of the residual connection to the structure of the generator in GAN.

In order to visually present the diagnosis results, we generated a confusion matrix using the VGG classification network based

## DIAGNOSTIC ACCURACY

![](images/a4b0a77342d55c5c97af8ca4fe3f0e0723a2038017f88928e81cc70b1fc948d0.jpg)  
FIGURE 19 | Fault diagnosis results of different data augmentation methods.

TABLE 12 | Different similarity index values.

<table><tr><td></td><td>Cosine similarity</td><td>MSE</td><td>PSNR</td><td>SSIM</td></tr><tr><td>RWGAN-GP</td><td>0.8541</td><td>105.0434</td><td>8.7833</td><td>0.1053</td></tr></table>

![](images/c7b9807748c174747b3770ad89b17a69fe97868ecbddec8440e4819f348240c0.jpg)

![](images/5dfc42e89d239e5f353cf4e8da5f3fe34e504a558a178ea2b1a9624a2a300194.jpg)

![](images/cf3a40402d8ce4b32278e32620958cd8614b11c6e9d030ed7fbd82db22ff04fe.jpg)  
(e)

![](images/ce7aac497f81802cf26f1bfbc190dab3c1b88468af4944f4db8b56e4cac7a01b.jpg)

![](images/369ea002c74ae926b2a44358e7ec422e4898736e3b3b72403cb794a4998d84b8.jpg)

![](images/3aa9e34bf0f74d07e5a8c59ff0fb3e4e81a43637face7c9e5afe6832e8769102.jpg)  
(f)  
| VGG network classification confusion matrix of different data enhancement methods. (a) imbalanced data. (b) GAN. (c) WGAN-GP. (d) SAGAN. (e) MSG-GAN. (f) RWGAN-GP.

on RWGAN-GP model data augmentation (see Figure 20). It can be seen from the figure that in the imbalanced data set, the diagnostic model has errors in the diagnosis of fault categories with a small number of samples. For example, in the Rs and Sc categories, only 1 and 8 samples are correctly classified respectively. However, when data augmentation is performed on imbalanced data, the classification accuracy of minority samples improves significantly. Following the application of the RWGAN-GP model for data augmentation, the count of accurately classified samples within the Rs category increased to 19, and within the Sc category, it increased to 10. The overall effect on improving fault diagnosis is the most obvious, which shows that the fault Significant improvement in diagnosis. It is worth noting that data augmentation using GAN models achieved excellent results in fault diagnosis, even surpassing the WGAN-GP model, although slightly lower than the RWGAN-GP model. We believe this is because different data sets may exhibit normal differences. For some complex image data sets, more complex models may be needed, and in the NEU surface defect data set used in this section, the image size is small and the image information is relatively simple, so the simpler GAN model can perform better than WGAN-GP. This further emphasizes the positive improvement that we introduce residual connections in the generator structure, making the model more adaptable and with more stable performance.

RWGAN-GP demonstrates significant advantages in addressing class imbalance issues, primarily due to its incorporation of resid ual mechanisms and gradient penalty. The residual mechanism, which introduces skip connections in the generator, alleviates the vanishing gradient problem commonly encountered in traditional GANs, thus substantially improving the quality of generated samples. Additionally, the application of gradient penalty enhances training stability by effectively mitigating issues such as gradient explosion or vanishing gradients, resulting in higherquality and more diverse samples. In contrast, traditional GANs and their variants, such as WGAN, SAGAN, and MSG-GAN, exhibit limitations in terms of sample quality, diversity, and training stability. These issues lead to lower-quality generated samples and a reduced effectiveness in mitigating class imbalance. Therefore, RWGAN-GP excels in experimental settings, effectively filling gaps in the dataset and improving classifier performance.

Based on the preceding analysis, it is evident that the proposed RWGAN-GP approach proves effective in addressing the issue of data imbalance in the context of NEU surface defect fault diagnosis.

## 5 | Conclusion

In order to solve the problem of poor fault diagnosis results due to the large number of normal state samples and the small number of fault state samples in actual industrial processes, this research proposes an industrial process imbalance fault image diagnosis framework based on RWGAN-GP data enhancement strategy. First, in order to solve the fault diagnosis difficulties caused by the common data imbalance problem in industrial processes, the RWGAN-GP model is proposed. This model introduces Wasserstein distance and gradient penalty into the original GAN, which improves the unstable problems that traditional GAN easily occurs during training, such as mode collapse and training oscillation of the generator and discriminator. In addition, the residual connection is added to the generator structure, which is conducive to better gradient propagation of the generator during training and alleviates the problem of generator gradient vanishing that easily occurs during model training. Then use the samples generated by the RWGAN-GP model to perform data enhancement on the data set to change the imbalance of the data set. Next the deep neural network is used to extract the characteristic information of the balanced data set and perform fault diagnosis. Finally, the proposed imbalance fault diagnosis framework is applied to the PRONTO benchmark dataset and the NEU surface defect database. Experiments show that the diagnosis framework has good performance for the data imbalance problem in real industrial processes. In the future, more industrial process datasets will be used to evaluate the proposed framework and facilitate its application in industrial practice. In future work, efforts can be directed towards evaluating the framework on a wider range of industrial process datasets, particularly across various domains and types, to validate its generalizability and adaptability. Additionally, improvements can be made to enhance the model's robustness in the presence of noise and anomalous data, ensuring stable performance across diverse industrial environments.

## Conflicts of Interest

The authors declare no conflicts of interest.

## Data Availability Statement

The data that support the findings of this study are available on request from the corresponding author. The data are not publicly available due to privacy or ethical restrictions.

## References

1. J. Zhang, K. Zhang, Y. An, H. Luo, and S. Yin, “An Integrated Multitasking Intelligent Bearing Fault Diagnosis Scheme Based on Representation Learning Under Imbalanced Sample Condition,” IEEE Transactions on Neural Networks and Learning Systems 35, no. 5 (2024): 6231–6242.

2. L. A. Al-Haddad and A. A. Jaber, “An Intelligent Fault Diagnosis Approach for Multirotor UAVs Based on Deep Neural Network of Multi-Resolution Transform Features,” Drones 7, no. 2 (2023): Art. no.82.

3. S. N. Tang, S. Q. Yuan, and Y. Zhu, “Deep Learning-Based Intelligent Fault Diagnosis Methods Toward Rotating Machinery,” IEEE Access 8 (2020):9335-9346.

4. T. T. Zhou, T. Han, and E. L. Droguett, “Towards Trustworthy Machine Fault Diagnosis: A Probabilistic Bayesian Deep Learning Framework,” Reliability Engineering & System Safety 224 (2022): Art. no. 108525.

5. Q. Ni, J. C. Ji, K. Feng, Y. Zhang, D. Lin, and J. Zheng, “Data-Driven Bearing Health Management Using a Novel Multi-Scale Fused Feature and Gated Recurrent Unit,” Reliability Engineering & System Safety 242 (2024): 109753 /02/01/2024.

6. K. Feng, Y. Xu, Y. Wang, et al., “Digital Twin Enabled Domain Adversarial Graph Networks for Bearing Fault Diagnosis,” IEEE Transactions on Industrial Cyber-Physical Systems 1 (2023): 113–122.

7. K. Feng, J. C. Ji, Y. Zhang, Q. Ni, Z. Liu, and M. Beer, “Digital Twin-Driven Intelligent Assessment of Gear Surface Degradation,” Mechanical Systems and Signal Processing 186 (2023): 109896 /03/01/2023.

8. Q. Ni, J. C. Ji, B. Halkon, K. Feng, and A. K. Nandi, “Physics-Informed Residual Network (PIResNet) for Rolling Element Bearing Fault Diagnostics,” Mechanical Systems and Signal Processing 200 (2023): 110544 /10/01/2023.

9. X. Li, W. Zhang, X. Li, and H. Hao, “Partial Domain Adaptation in Remaining Useful Life Prediction With Incomplete Target Data,” IEEE/ ASME Transactions on Mechatronics 29, no. 3 (2024): 1903–1913.

10. W. Zhang, H. Hao, and Y. Zhang, “State of Charge Prediction of Lithium-Ion Batteries for Electric Aircraft With Swin Transformer,” IEEE/CAA Journal of Automatica Sinica (2024): 1–3, https://doi.org/10. 1109/JAS.2023.124020.

11. Z. W. Gao, C. Cecati, and S. X. Ding, “A Survey of Fault Diagnosis and Fault-Tolerant Techniques-Part I: Fault Diagnosis With Model-Based and Signal-Based Approaches,” IEEE Transactions on Industrial Electronics 62, no. 6 (2015): 3757–3767.

12. Z. J. Ren, T. T. Lin, K. Feng, Y. S. Zhu, Z. Liu, and K. Yan, “A Systematic Review on Imbalanced Learning Methods in Intelligent Fault Diagnosis,” IEEE Transactions on Instrumentation and Measurement 72 (2023): Art. no. 3508535.

13. T. C. Zhang, et  al., “Intelligent Fault Diagnosis of Machines With Small & Imbalanced Data: A State-Of-The-Art Review and Possible Ex tensions,” ISA Transactions 119 (2022): 152–171.

14. A. Krizhevsky, I. Sutskever, and G. Hinton, “ImageNet Classification With Deep Convolutional Neural Networks,” Advances in Neural Information Processing Systems 25, no. 2 (2012): 1097–1105.

15. K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” Computer Science (2014), https://doi.org/10.48550/arXiv.1409.1556.

16. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic Minority Over-Sampling Technique,” AI Access Foundation no. 1 (2002): 321–357.

17. I. Goodfellow, J. Pouget-Abadie, M. Mirza, et al. “Generative Adversarial Networks,” Communications of the ACM 63, no. 11 (2020): 139-144.

18. A. Z. S. B. Habib, K. T. Islam, M. M. H. Pranto, and M. Nooruddin, “Breast Cancer Classification Using Ensemble Hard Voting With Random Under-Sampling,” in 11th International Conference on Electrical and Computer Engineering (ICECE), Buet, Electr Network, vol. 2020 (NEW YORK: Ieee, 2020), 379–382.

19. X. Y. Xie, H. W. Liu, S. Z. Zeng, L. B. Lin, and W. Li, “A Novel Progressively Undersampling Method Based on the Density Peaks Sequence for Imbalanced Data, (In English),” Knowledge-Based Systems 213 (2021): Art. no. 106689.

20. P. Soltanzadeh and M. Hashemzadeh, “RCSMOTE: Range-Controlled Synthetic Minority Over-Sampling Technique for Handling the Class Imbalance Problem (In English),” Information Sciences 542 (2021): 92–111.

21. Y. Sun, M. S. Kamel, A. K. C. Wong, and Y. Wang, “Cost-Sensitive Boosting for Classification of Imbalanced Data,” Pattern Recognition 40, no. 12 (2007): 3358–3378.

22. F. D. Frumosu, A. R. Khan, H. Schioler, M. Kulahci, M. Zaki, and P. Westermann-Rasmussen, “Cost-Sensitive Learning Classification Strategy for Predicting Product Failures (In English),” Expert Systems With Applications 161 (2020): Art. no. 113653.

23. R. Anand, K. G. Mehrotra, C. K. Mohan, and S. Ranka, "An Improved Algorithm for Neural Network Classification of Imbalanced Training Sets,” IEEE Transactions on Neural Networks 4, no. 6 (1993): 962–969.

24. Y. Dong, H. Jiang, Z. Wu, Q. Yang, and Y. Liu, “Digital Twin-Assisted Multiscale Residual-Self-Attention Feature Fusion Network for Hypersonic Flight Vehicle Fault Diagnosis,” Reliability Engineering & System Safety 235 (2023): 109253 /07/01/2023.

25. Y. Dong, H. Jiang, W. Jiang, and L. Xie, “Dynamic Normalization Supervised Contrastive Network With Multiscale Compound Attention Mechanism for Gearbox Imbalanced Fault Diagnosis,” Engineering Applications of Artificial Intelligence 133 (2024): 108098 /07/01/2024.

26. Y. Dong, H. Jiang, M. Mu, and X. Wang, “Multi-Sensor Data Fusion-Enabled Lightweight Convolutional Double Regularization Contrast Transformer for Aerospace Bearing Small Samples Fault Diagnosis,” Advanced Engineering Informatics 62 (2024): 102573 /10/01/2024.

27. X. Wang, H. Jiang, Z. Wu, and Q. Yang, “Adaptive Variational Autoencoding Generative Adversarial Networks for Rolling Bearing Fault Diagnosis,” Advanced Engineering Informatics 56 (2023): 102027 /04/01/2023.

28. E. C. Orenstein, O. Beijbom, E. E. Peacock, and H. M. Sosik, “WHOI-Plankton- A Large Scale Fine Grained Visual Recognition Benchmark Dataset for Plankton Classification,” Computer Science (2015), https:// doi.org/10.48550/arXiv.1510.00745.

29. H. Lee, M. Park, and J. Kim, “Plankton Classification on Imbalanced Large Scale Database via Convolutional Neural Networks With Transfer Learning,” in 2016 IEEE International Conference on Image Processing (ICIP) (IEEE, 2016), 3713–3717.

30. B. Mateusz, M. Atsuto, and M. A. Mazurowski, “A Systematic Study of the Class Imbalance Problem in Convolutional Neural Networks,” Neural Networks 106 (2017): 249–259.

31. T. Y. Pan, J. L. Chen, T. C. Zhang, S. Liu, S. L. He, and H. X. Lv, “Generative Adversarial Network in Mechanical Fault Diagnosis Under Small Sample: A Systematic Review on Applications and Future Perspectives,” ISA Transactions 128 (2022): 1–10.

32. W. Li, X. Zhong, H. D. Shao, B. P. Cai, and X. K. Yang, “Multi-Mode Data Augmentation and Fault Diagnosis of Rotating Machinery Using Modified ACGAN Designed With new Framework,” Advanced Engineering Informatics 52 (2022): Art. no. 101552.

33. W. Luo, W. Yang, J. He, et  al., “Fault Diagnosis Method Based on two-Stage GAN for Data Imbalance," IEEE Sensors Journal 22 (2022): 21961-21973.

34. G. Yang, Y. Zhong, L. Yang, H. Tao, J. Y. Li, and R. X. Du, “Fault Diagnosis of Harmonic Drive With Imbalanced Data Using Generative Adversarial Network (In English),” IEEE Transactions on Instrumentation and Measurement

35. A. Radford, L. Metz, and S. Chintala, “Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks,” Computer Science (2015). https://doi.org/10.48550/arXiv.1511.06434.

36. M. Frid-Adar, I. Diamant, E. Klang, M. Amitai, J. Goldberger, and H. Greenspan, “GAN-Based Synthetic Medical Image Augmentation for Increased CNN Performance in Liver Lesion Classification,” Neurocomputing 321 (2018): 321–331.

37. M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein GAN,” p. arXiv:1701.07875. Accessed on: January 01, 2017. https://doi.org/10. 48550/arXiv.1701.07875, https://ui.adsabs.harvard.edu/abs/2017arXiv1 70107875A.

38. I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, “Improved Training of Wasserstein GANs,” Advances in Neural Information Processing Systems (2017): 30.

39. Z. Li, T. Zheng, Y. Wang, Z. Cao, Z. Guo, and H. Fu, “A Novel Method for Imbalanced Fault Diagnosis of Rotating Machinery Based on Generative Adversarial Networks,” IEEE Transactions on Instrumentation and Measurement 70 (2020): 1–17.

40. W. Dai, D. Li, D. Tang, H. Wang, and Y. Peng, “Deep Learning Approach for Defective Spot Welds Classification Using Small and Class-Imbalanced Datasets,” Neurocomputing 477 (2022): 46–60.

41. M. Mirza and S. Osindero, “Conditional Generative Adversarial Nets,” Computer Science (2014): 2672–2680, https://doi.org/10.48550/ arXiv.1411.1784.

42. H. Yin, Y. Gao, C. Liu, and S. Liu, “Fault Diagnosis Method Based on CWGAN-GP-1DCNN,” in 2021 IEEE 24th International Conference on Computational Science and Engineering (CSE) (IEEE, 2021), 20–26.

43. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (IEEE, 2016), 770–778.

44. H. Zhang, I. J. Goodfellow, D. N. Metaxas, and A. J. A. Odena, "Self-Attention Generative Adversarial Networks,” in International Conference on Machine Learning (PMLR, 2019), 7354–7363.

45. A. Karnewar and O. Wang, “MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (IEEE, 2020), 7799–7808.