# A Survey on Deep Learning for Data-Driven Soft Sensors

Qingqiang Sun and Zhiqiang Ge , Senior Member, IEEE

Abstract—Soft sensors are widely constructed in process industry to realize process monitoring, quality prediction, and many other important applications. With the development of hardware and software, industrial processes have embraced new characteristics, which lead to the poor performance of traditional soft sensor modeling methods. Deep learning, as a kind of data-driven approach, shows its great potential in many fields, as well as in soft sensing scenarios. After a period of development, especially in the last five years, many new issues have emerged that need to be investigated. Therefore, in this article, the necessity and significance of deep learning for soft sensor applications are demonstrated first by analyzing the merits of deep learning and the trends of industrial processes. Next, mainstream deep learning models, tricks, and frameworks/toolkits are summarized and discussed to help designers propel the developing progress of soft sensors. Then, existing works are reviewed and analyzed to discuss the demands and problems occurred in practical applications. Finally, outlook and conclusions are given.

Index Terms—Data-driven modeling, deep learning (DL), industrial big data, neural networks (NNs), soft sensor.

## I. INTRODUCTION

N <sup>OWADAYS,</sup> <sup>the</sup> <sup>process</sup> <sup>industry</sup> <sup>is</sup> <sup>becoming</sup> <sup>more</sup> <sup>and</sup>more complicated, due to the development of information more complicated, due to the development of information technologies and the increase of customer demands. As a result, the cost and difficulty of direct measurement and analysis of key quality variables are increasing [1]–[3]. However, in order to monitor the operation status of systems, realize the smooth control of processes and improve the quality of products, those key variables or quality indices have to be obtained as fast and accurately as possible. Therefore, soft sensing technique, which is a kind of mathematical model with easy-to-measured auxiliary variables as input and hard-to-measured variables as output, has been developed to estimate or predict important variables expediently during the past decades [4].

There are three main types of approaches to establish soft sensing models, namely mechanism-based, knowledge-based, and data-driven methods [5]. The first two kinds of approaches can work well if detailed and accurate mechanism of process is known or a wealth of experience and knowledge about process is available. However, the increasing complexity of industrial process makes these preconditions difficult to meet. As a result, data-driven modeling has become the mainstream soft sensing modeling methods [6], [7].

Conventional data-driven soft sensor modeling methods mainly include a wide variety of statistical inference techniques and machine learning techniques, such as principal component regression, which combines principal component analysis with a regression model, partial least squares (PLS) regression, support vector machine (SVM), and artificial neural network (ANN) [8]–[12]. In last two decades, with technical breakthroughs on some key issues, networks with enough number of hidden layers or with complex enough structures are available, which are known as deep learning (DL) techniques [13], [14]. Due to DL techniques, computational models that are composed of multiple processing layers are allowed to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state of the art in speech recognition, object detection, and many other domains, such as drug discovery and genomics [15].

In recent years, there has been a proliferation of research that applies DL approaches to soft sensors. From conventional artificial intelligence field to soft sensing field, many differences exist objectively. There are many questions need to be investigated and discussed (including but not limited to the following issues): Is it necessary and suitable to use DL techniques in soft sensing scenario? What DL models can be utilized for practical application? How to apply them to solving problems in real processes? What are the potential research points for the future? Therefore, the motivation of this article is to answer these questions as reasonably as possible.

The rest of this article is organized as follows. Section II discusses the distinct merit of DL and demonstrates its necessity for soft sensor modeling. Section III provides an overview of several typical DL models and core training techniques. Then, the state of the art of soft sensor applications using DL approaches is investigated in Section IV. Discussions and outlook are given in Section V. Finally, Section VI concludes this article.

![](images/fc1591692d9f362ac2d57ab9e33a1e58fd51481cd3b1c6d1fab3b69a64155f17.jpg)  
Fig. 1. Structure of a network with single hidden layer.

## II. SIGNIFICANCE OF DL FOR SOFT SENSOR

Detailed review about conventional methods can be found in existing work, such as [7], [16], etc. Although those methods already have many applications, they may suffer from some drawbacks, such as heavy workload brought by handcrafted feature engineering or inefficiency when dealing with large amount of data, etc. To demonstrate the significance of DL for soft sensor modeling, the distinct merits of DL and the trends or characteristics of industrial processes should be discussed.

## A. Merits of DL Techniques

To begin with, the structure of a simple network with single hidden layer is shown in Fig. 1. There are three layers, namely an input layer, a hidden layer, and an output layer. Input layer contains variables $x _ { 1 } , \cdots , x _ { m }$ and a constant node “1.” The hidden layer has many nodes, and each node has an activation function ϕ. The feature in each node is extracted through affine transformation and activation function transformation from original input layer, which are defined as the following formula:

$$
\begin{array}{l} H _ {i} = \varphi \left(M _ {i} \left(x _ {1}, \dots , x _ {m}\right)\right) \\ = \varphi \left(\left(\sum_ {k = 1} ^ {m} w _ {i k} ^ {0} * x _ {k}\right) + b _ {i} ^ {0}\right). \end{array}\tag{1}
$$

Then, the final output is the combination of those composite functions

$$
y (x) = \sum_ {k = 1} ^ {n} w _ {k} ^ {1} H _ {k} (x).\tag{2}
$$

The weight and bias parameters $( w _ { i j } ^ { 0 } , b _ { i } ^ { 0 } )$ need to be learned by minimizing the loss function, which is defined according to specific task and target. This process is called as “training” or “learning.”

![](images/3948875a2d97ad2564aaabc8f92924698c68612b5634999c6b0eca801114d64b.jpg)  
Fig. 2. Comparison of four kinds of theories.

According to universal approximation theory, if there are enough nodes in the hidden layer, the function represented by the network shown in Fig. 1 can approximate any continuous function [17]–[19]. Furthermore, using multiple layers of neurons to represent some functions is much simpler.

Since Hinton et al. proposed a faster learning algorithm, which was applied to deep belief network (DBN), the maximum depth of network can be tens of layers [13]. Later on, He et al. [20] proposed the deep residual network, which solved the performance degradation problem caused by increasing network depth. From then on, the depth of neural network (NN) can reach a level of hundreds of layers. However, “deep” in DL theory is not absolutely defined. In speech recognition domain, four layers of network can be considered as “deep,” whereas in image recognition, networks with more than 20 layers are common.

DL has its own advantage compared with conventional soft sensor modeling methods. Here, we classify them into four categories at a greater granularity: rule-based system, classical machine learning, shallow representation learning, and deep learning. The differences between them are shown in Fig. 2, in which the green blocks indicate components that are able to learn information from data [21].

Rule-based system, also known as production system or expert system, is the simplest form of artificial intelligence. Rules are coded into the programs as the representation of knowledge, which tell the system what to do or what to conclude in different situations [22]–[24]. In this way, the performance of rule-based system depends almost entirely on expert knowledge, which is hard to obtain and hard to update, especially in complicated cases. A rule-based system could be considered as having “fixed” intelligence, in contrast, a machine learning system is more adaptive and closer to human intelligence. Instead of outputting a result directly from a fixed set of rules wrote by human, classical machine learning first extracts features from raw input data and then the final output is obtained by mapping the features. However, the forms of features are still handcrafted based on knowledge and experience, which is called as feature engineering [25], [26]. In order to extract features that better represent the underlying problem, the process of feature engineering is usually complicated, including feature selection, feature construction, and feature extraction. Because the upper bound of the performance of conventional machine learning is mainly determined by data and features, the effect of those approaches relies heavily on the ability of the engineer to extract good features. Therefore, representation learning approaches were proposed so as to automatically learn the implicit useful representations or features from raw data [27]. In this way, data representation is often trained in conjunction with subsequent predictive tasks. Representation learning does not rely on expert experience, but requires a large training dataset. Compared with shallow representation learning, DL is a kind of deep representation learning, which tries to learn more hierarchical and more abstract representations using deep networks. As an end-to-end approach, what DL needs is enough and quality data, rather than complicated feature engineering.

![](images/a2e92a86e9391d547f559376a06fb84b3718d9c11b652a492559ea28f9a3049d.jpg)  
Fig. 3. Scale drives algorithm performance.

However, is DL always better than conventional machine learning or is deep representation learning always better than shallower one? The key factor is the amount of data that are available for modeling, especially labeled ones [28]. Visually, the performance of algorithm is plotted as the function of the amount of data used for a task in Fig. 3.

Improvements in data availability and computational scale have been two of the biggest drivers of recent progress in machine learning, which means large enough training sets are available and large enough NNs are trainable. As for traditional learning algorithms, such as SVM or logistic regression, the performance improves for a while as more data are added. However, even as more data are accumulated after that, usually the performance of those algorithms goes into plateaus. This means their learning curves flattens out and the algorithms stop improving even as more data are given since they do not know what to do with huge amounts of data. Nevertheless, if a small NN, which contains only a small number of hidden units/layers/parameters, is trained on the same supervised learning task, slightly better performance might be attainable. Analogously, if larger and larger NNs are trained, even better performance can be obtained. Besides, it is notable that in the regime of small training sets, the relative ordering of the algorithms is actually not very well defined. In this case, the performance of the model depends mainly on the skill of feature engineering and other algorithm details, so it is quite possible that traditional algorithms could do better. By the way, even if only small amount of data is available, the transferable character of DL algorithms can also ensure the performance of modeling since the underlying networks are relatively general as long as the data distribution is as consistent as possible [29], [30]. In contrast, in big data regions where there are very large training sets, it can be more consistently seen that large NN dominates the other approaches. Thus, the relatively more reliable way to improve the performance of an algorithm today is to train a bigger network and get more data.

In conclusion, the merits of DL techniques compared with traditional algorithms mainly lie in learning representation without the requirement of knowledge or experience and taking full advantage of huge amount of data for performance improvements.

## B. Trends of Industrial Process

The industrial processes are more and more complicated and ever changing. The ever-increasing demands for profits and environmental factors have added the complexity of industrial processes. For example, the demands of different product grades lead to many chemical processes working with multiple conditions [31], [32]. Besides, the complicated process mechanisms also increase the difficulty of process modeling, such as penicillin fermentation process, in which the microorganisms have to experience multiple growth phases [33]. Due to such causes, process industry may possess many characteristics, such as nonlinear, multimodal properties, etc. Therefore, it is increasingly difficult to construct monitoring or predictive models for those complex processes. In addition, changes in process characteristics or operating conditions are almost ubiquitous [34]. In chemical processes, for instance, equipment characteristics are changed due to catalyst deactivation, scale adhesion, preventive maintenance, and others. The changes of loads and feedstocks also result in process variations and deteriorate the performance of process modeling, such as in the pharmaceutical industry [35]. Therefore, soft sensors have to be updated as the process characteristics change, but manual and frequent construction of them should be avoided due to their heavy workload, especially in feature engineering. This trend and corresponding issues are shown in the left part of Fig. 4.

Looking at the development of the process industry in recent years, industrial big data is another trend that cannot be ignored [36], [37]. More and more process monitoring sensors are installed to measure real time process status (e.g., temperature, flow rate, pressure, etc.) and a lot of data storage devices (e.g., distributed control system) are utilized in plants and factories [38]. All of these developments make it possible to obtain large amount of data for process modeling. At the same time, the data form also evolves a lot [39]. For instance, from univariate to multivariate, to high-dimensional [40]–[42]; from homogeneous data to heterogeneous datasets [43], [44]; from static to dynamic [45], [46]. Therefore, enough and various data are available, which need to be utilized efficiently to train monitoring or predictive models. This trend is shown in the right part of Fig. 4.

![](images/c28e9509c2427c6371065dae02cff62eb9e7c95c3584da11ce99d6cef9cdf4c6.jpg)  
Fig. 4. DL matches the trends of industrial processes.

In a nutshell, based on sufficient literature research and to our best knowledge, two main trends in the development of industrial processes are concluded: they are more and more complicated and ever changing and a huge amount of process data are generated and stored. Under such a circumstance, the characteristics of DL technique, discussed in Section II, exactly match these two trends well. First of all, DL can avoid complicated feature engineering and learn abstract representation automatically (see Fig. 2). Second, DL can make full use of large amounts of data to effectively improve modeling performance (see Fig. 3). These are why DL techniques are of great significance and are going to be more and more significant for soft sensor applications.

## III. DL MODELS AND GENERAL TRICKS

In this section, typical models and general tricks in DL field are reviewed and summarized, including autoencoder (AE), restricted Boltzmann machine (RBM), convolutional NN (CNN), and recurrent NN (RNN).

## A. Autoencoder

An AE is actually a system that attempts to reproduce its original input. To achieve this goal, AE must capture the most important information that can represent the input data [47], [48]. Therefore, the code dimension is constrained to be less than the input dimension, which is also called as undercomplete AE.

Technically, the full encoding and decoding process can be represented as the following formula:

$$
\boldsymbol {h} = \operatorname{encode} (\boldsymbol {x}) = f _ {\mathrm{e}} \left(\boldsymbol {W} _ {\mathrm{e}} \cdot \boldsymbol {x} + \boldsymbol {b} _ {\mathrm{e}}\right)\tag{3}
$$

![](images/b788f9d309202f1a16c422f1be4f8db64e9f30eb5e2e0e7c0b58df4f4326c0e6.jpg)  
Fig. 5. Learning strategy of SAE.

$$
\tilde {\boldsymbol {x}} = \operatorname{decode} (\boldsymbol {h}) = f _ {\mathrm{d}} \left(\boldsymbol {W} _ {\mathrm{d}} \cdot \boldsymbol {h} + \boldsymbol {b} _ {\mathrm{d}}\right)\tag{4}
$$

where x is the original input vector, h is the feature vector after encoding, x˜ is the vector of reconstructed input, $\{ W _ { \mathrm { e } } , b _ { \mathrm { e } } \}$ and $\{ W _ { \mathrm { d } } , b _ { \mathrm { d } } \}$ are weights and biases of encoder and decoder, respectively, and $f _ { \mathrm { e } } ( \cdot )$ and $f _ { \mathrm { d } } ( . )$ are corresponding nonlinear activation function, such as sigmoid, Tanh, ReLU, etc.

Besides, AEs can be stacked to construct deeper network, namely stacked AE (SAE). The learning strategy of SAE is represented as Fig. 5. The whole process is actually a process of unsupervised layerwise training. SAE possesses more encoding layers so that it can extract more abstract representations. Besides, AE has many extensions, such as denoising AE (DAE) [49], sparse AE [50], [51], contractive AE, and etc. [52].

## B. Restricted Boltzmann Machine

RBM is an undirected probability graph model with one visible layer and one hidden layer. There is no connection between neurons in the same layer, which is the meaning of “restricted.” The goal of RBM is to make the output of the visible layer as close to the original input as possible so that the hidden layers are regarded as different representations of the visible layer. The joint probability distribution and conditional distribution are related to an energy function, and detail derivation process can be found in [21]. RBMs can be trained by approximate maximum likelihood stochastic gradient descent, often involving a Monte Carlo Markov chain to obtain those model samples. A much more complete tutorial and other tips or tricks can be seen in [53] and [54].

RBM has various extensions, among which are DBN and deep Boltzmann machine (DBM). DBN is a hybrid graphical model involving both directed and undirected connections. Except that the top two layers are undirected (pure RBM), the connections of all the other layers are directed (Bayesian network). DBN has multiple hidden layers and hidden units in adjacent layers are connected. All of the local conditional probability distributions in DBN are copied directly from that in its constituent RBMs. DBN is layerwise pretrained by a fast and greedy algorithm and then is fine-tuned using contractive wake–sleep algorithm [13]. While a DBM is an undirected graphical model with several layers, and it is constructed to learn high-level representations of the input [55]. Generally speaking, DBM is more robust than DBN, but the cost is greater computational complexity since DBM needs to be jointly trained.

![](images/2038b149d235a54fd3ec13c8b6fef9981563462ffc6996a90530e2803107b442.jpg)  
Fig. 6. 2-D convolution case.

## C. Convolution NN

CNN is a specialized kind of NN for processing data that have a gridlike topology, such as time series data (1-D grid taking samples at regular time intervals) and image data (2-D grid of pixels). It is notable that the “convolution” here actually refers to cross-correlation function, which is the same as convolution but without flipping the kernel

$$
S (i, j) = (I * K) (i, j) = \sum_ {m} \sum_ {n} I (i + m, j + n) K (m, n)\tag{5}
$$

where I and K denote 2-D input and 2-D kernel function, respectively, the symbol “∗” denotes convolution operation and i and j are the indexes in these two dimensions.

The detailed computation process of a 2-D convolution case can be seen in Fig. 6. From the example, the merits of such convolution operation can be concluded, which are as follows.

1) Sparse interactions: The size of kernel is much smaller than that of input so the interaction between the input and output is a kind of sparse connectivity, which saves a lot of time complexity compared with common fully connected networks.

2) Parameters sharing: Different from the entries of weight matrix of traditional networks, which are used only one time when computing the output of a layer, every element of a kernel is used at every position of the input so the storage requirements for parameters are reduced significantly.

3) Equivariant representations: Due to the characteristic of parameter sharing, the result of convolution operation before which the input is shifted is the same as that of shifting the output of convolution of the input.

It is because of these three features that CNN is particularly suited to processing gridlike data [56].

Generally, after the convolution, there is a pooling operation to further adjust the output. The pooling function uses the overall statistical characteristics of the adjacent outputs at a certain location to replace the network output at that location, and no parameters need to be learned. For instance, the max-pooling operation uses the maximum output to represent the corresponding rectangular region [57]. Other common pooling functions, such as the average of a rectangular neighborhood, the $L ^ { 2 }$ norm of a rectangular neighborhood, or a weighted average based on the distance from the central pixel, are also widely used to compress parameter space. CNN also has a lot of variants, such as AlexNet [56], LeNet [58], VggNet [59], etc.

![](images/15e78ea21ab0cec5ac305543027c5788624e7de0edb67e61d1528f1c7238c51c.jpg)  
Fig. 7. Typical components of RNN (x is the input data in sequence form, h is the hidden layer, o is the output layer, y is the target label, and L is the loss. U, V , and W are corresponding weight matrixes.).

## D. Recurrent NN

RNN is developed for processing sequential data. The basic architecture and loss computation graph of RNN are shown in Fig. 7. The left network can be unfolded over time sequence to get the right form. Every time step has an input, a hidden unit, and an output. Besides, recurrent connections exist between hidden units.

Given a specific status $h ^ { ( 0 ) }$ , RNN can propagate forward. Suppose the activation of hidden layer is tanh(-) and the output layer is fed into a softmax function to generate normalized probabilities yˆ, the corresponding layers from t = 1 to $t = \tau$ can be updated according to the following formulas:

$$
\boldsymbol {a} ^ {(t)} = \boldsymbol {b} + \boldsymbol {W h} ^ {(t - 1)} + \boldsymbol {U x} ^ {(t)}\tag{6}
$$

$$
\boldsymbol {h} ^ {(t)} = \tanh \left(\boldsymbol {a} ^ {(t)}\right)\tag{7}
$$

$$
\pmb {o} ^ {(t)} = \pmb {c} + V \pmb {h} ^ {(t)}\tag{8}
$$

$$
\hat {\pmb {y}} ^ {(t)} = \mathrm{softmax} \left(\pmb {o} ^ {(t)}\right)\tag{9}
$$

where b and c denote the bias vectors.

The total loss is just the sum of the losses over all the time steps. For example, if $L ^ { ( t ) } \mathrm { i } s$ s computed as the negative log-likelihood of $\mathbf { \boldsymbol { y } } ^ { ( t ) }$ given $\pmb { x } ^ { ( 1 ) } , \cdots , \pmb { x } ^ { ( \hat { t } ) }$ , then

$$
\begin{array}{l} L \left(\left\{\boldsymbol {x} ^ {(1)}, \dots , \boldsymbol {x} ^ {(\tau)} \right\}, \left\{\boldsymbol {y} ^ {(1)}, \dots , \boldsymbol {y} ^ {(\tau)} \right\}\right) \\ = \sum_ {t} L ^ {(t)} = - \sum_ {t} \log p _ {\text { model }} \left(\boldsymbol {y} ^ {(t)}   \Big |   \left\{\boldsymbol {x} ^ {(1)}, \dots , \boldsymbol {x} ^ {(t)} \right\}\right) \end{array}\tag{10}
$$

where $p _ { \mathrm { m o d e l } } ( \pmb { y } ^ { ( t ) } | \{ \pmb { x } ^ { ( 1 ) } , \cdots , \ b { x } ^ { ( t ) } \} )$ is given by reading the entry for $\mathbf { \boldsymbol { y } } ^ { ( t ) }$ from the model’s output vector $\hat { \pmb y } ^ { ( t ) }$ . The parameters are updated using backpropagation through time (BPTT) [21], [60].

TABLE I  
SUMMARY OF FOUR MAIN TYPES OF DL MODELS

<table><tr><td>Model</td><td>Characteristics</td><td>Merits</td><td>Demerits</td><td>Main Applications for Soft Sensor modeling</td></tr><tr><td>AE</td><td>Unsupervised; Common data; Learn feature representations of the input automatically.</td><td>Effective dimension reduction; Denoising; Low computational complexity.</td><td>The output layer has no practical use after training, and high hidden dimension may lead to self-replication of the input.</td><td>Semi-supervised modeling, missing data problem, et al.</td></tr><tr><td>RBM</td><td>Unsupervised; Common data; Probabilistic generative model.</td><td>Robust to ambiguous data; Dimension reduction; Feature extraction; Collaborative filtering.</td><td>High computational complexity caused by joint parameter optimization.</td><td>Strong correlation problem, ensemble learning, et al.</td></tr><tr><td>CNN</td><td>Supervised; Grid-like data; local feature extractor.</td><td>Sparse interactions; Parameters sharing; Equivariant representations.</td><td>The contradiction between the dependence on the depth of network and the slow parameter updating of deeper network.</td><td>Local dynamic modeling, frequency domain processing, et al.</td></tr><tr><td>RNN</td><td>Supervised; Sequence data; Update parameters by BPTT.</td><td>Learn the relationship between different time steps.</td><td>The challenge of long-term dependence.</td><td>Dynamic modeling, et al.</td></tr></table>

The basic problem of RNN is that gradients propagated over many stages tend to either vanish or explode, which is called as the challenge of long-term dependencies [61], [62]. Therefore, long short-term memory (LSTM) and other gated RNNs, such as gated recurrent units (GRUs), are proposed, which use several gate units to control the memory and forgetting behaviors of the hidden state [63]–[66].

The summary of four main commonly used DL techniques is listed in Table I.

## E. General Tricks for Developing DL Models

Although DL has huge potential, it could be very challenging to train deep models with satisfactory generalization performance efficiently. The reasons mainly lie in the overfitting and gradient vanish problems caused by deep structure. To overcome or mitigate these issues, several tricks should be helpful when training deep models.

1) Regularization: Regularization is an effective tool to overcome high-variance problem, namely overfitting. A direct way is to regularize the cost function with a parameter norm penalty, such as L<sup>2</sup> regularization. When minimizing the cost function, the parameters are also constrained to not be too large [67].

2) Dataset Augmentation: Getting more data for training machine learning models is the best way to improve their generalization performance. Although, it may be not easy to collect large amount of data from real scenarios, creating new fake data is meaningful for some specific tasks, such as object recognition [68] and speech recognition [69]. Introducing noise into the input layer can also be regarded as a kind of data augmentation [70], [71].

3) Early Stopping: The cost of training process usually runs down first and then may increase when too much further learning is conducted, which denotes the occur of overfitting. To avoid this problem, each time a better validation error is achieved, the parameter setting should be saved so that returning to the point with best performance after all training steps is realizable [72]. Therefore, the early stop strategy can prevent overlearning of parameters.

4) Sparse Representations: Another kind of parameter penalty is to constrain activation unit, which will indirectly impose a penalty on the complexity of parameters. Similar with common regularization, the penalty term based on the activation state of hidden units is added into the cost function. To obtain a relatively smaller cost, the probability of neuronal activation should be as small as possible [73]. Other approaches, such as KL divergence penalties or imposing a hard constraint on activation values, are also applied [74], [75].

5) Dropout: Dropout is a kind of ensemblelike strategy [76]. The basic principle is to remove the nonoutput units (e.g., multiply the output by zero) from base network to form several subnetworks. Every input unit and hidden unit is included according to a sampling probability so that the randomness and diversity of submodels can be guaranteed. The ensemble weights are often obtained according to the probability p(y|x)of submodels [77]. Another significant advantage is that there are few restrictions on the applicable model or training process. However, it does not work well if there is only a small amount of data [76].

6) Batch Normalization: Batch normalization is a method of adaptive reparameterization that aims to better train extremely deep network [78]. When training, the parameters of hidden layers in deep networks will consistently change, which leads to the internal covariate shift problem. Generally, the global distribution gradually approaches the upper and lower limits of the value interval of the nonlinear function. Thus, the gradients are easy to vanish when conducting backpropagation. With batch normalization, the mean and the variance of each unit are standardized so as to stabilize learning, but the relationships between units and the nonlinear statistics of a single unit are allowed to change.

## F. Frameworks for Developing DL Algorithms

To better realize the development of DL algorithm, several open-source frameworks are available, which may consist of state-of-the-art algorithms or well-designed underlying network elements, such as TensorFlow [79], Caffe [80], Theano [81], CNTK [82], Keras [83], Pytorch [84], and etc. The comparison of these platforms is shown in Table II.

## IV. DL APPLICATIONS FOR SOFT SENSOR MODELING

A successful development of DL algorithms is actually a highly iterative process, which can be summarized as Fig. 8.

TABLE II COMPARISON OF MAINSTREAM PLATFORMS

<table><tr><td>Platform</td><td>Characteristics</td></tr><tr><td>TensorFlow</td><td>The most popular deep learning framework at present with powerful communities. However, the interface design is too arcane and the system design is too complex.</td></tr><tr><td>Caffe</td><td>Easy to use, concise source code, superior performance and fast prototyping. However, it is difficult to extend and configure.</td></tr><tr><td>Theano</td><td>It has a strong academic atmosphere, but there are big defects in the engineering design. Now it has stopped the development.</td></tr><tr><td>CNTK</td><td>The performance is outstanding, good at the relevant research on speech, but the community is not active.</td></tr><tr><td>Keras</td><td>More like a deep learning interface. The most easily to get started but not flexible enough.</td></tr><tr><td>Pytorch</td><td>Concise, fast, easy to use, active community.</td></tr></table>

![](images/b4506a14bed47392ec25e80f4fc24ca633cba88f4c16169bf63ec2d2378613b7.jpg)  
Fig. 8. Iterative process for developing DL algorithms.

For soft sensing applications, the first step is to find the demands or problems existing in real industrial processes (such as semisupervised learning, dynamic modeling, missing data, etc.) and try to come up with a new idea worth trying. The next thing that needs to be done is to code it up with open-source frameworks or toolkits. After that, the data are collected and fed into the program to obtain a result that tells the designer how well this particular algorithm or configuration works. Based on the outcome, the designer should refine the ideas and change the strategies to find a better NN. Then, the process is repeated and the scheme is improved iteratively until the ideal effect is achieved.

To help the readers know about state-of-the-art progress and better develop high-performance soft sensors, the soft sensing applications based on DL techniques are reviewed here. The existing work is introduced and discussed, and the factors, such as motivation, strategy, and effectiveness, are mainly highlighted. The following contents are expanded according to the mainstream model to which each work belongs.

## A. AE-Based Applications

AE and its variants are widely used to construct soft sensors for semisupervised learning and dealing with missing data in industrial processes. Also, excellent performance can be achieved by combining with traditional machine learning algorithms.

Since AE is an unsupervised-learning model, it is often modified to a semisupervised or supervised form so as to complete the predictive tasks. For example, a semisupervised probabilistic latent variable regression model was developed using variational AE (VAE) in [85]. A common way is to introduce the supervision from label variables into the procedures of encoding and decoding. In [86], a variablewise weighted SAE was proposed to introduce the linear Pearson coefficient between the inputs of each hidden layer and quality labels when pretraining so as to extract features in a semisupervised way. Furthermore, techniques based on nonlinear relationships, such as mutual information [87], were adopted to better extract feature representations. However, both linear and nonlinear relationships are artificially specified and may be inadequate or unsuitable. Thus, a relatively more intelligent and automatic way is to add the predictive loss of quality labels into the pretraining cost [88]. Besides, other strategies also can be adopted to build the connections between hidden layers and label values. Sun and Ge used gated units to measure the contribution of the features in different hidden layers and better control the information flows between hidden layers and the output layer [89]. Moreover, focusing on semisupervised scenarios when there are only a small number of labeled samples and an excess of unlabeled samples, a kind of double ensemble learning approach was proposed that takes both data diversity and structural diversity into account [90].

Missing data is one of the most commonly encountered problems while designing industrial soft sensors. As a variant of AE, VAE performs well in learning data distribution and dealing with missing data problem. For example, a generative model named VA-WGAN was proposed based on VAE and Wasserstein GAN, and it can generate the same distributions of real data from industrial processes, which is hard to achieve by conventional regression models [91]. In [92], VAE was employed to extract the distribution of each feature variable for a just-in-time modeling approach, and the effectiveness of it was verified through a numerical example and an industrial process. Moreover, the authors enriched the theory by proposing an output-relevant VAE for just-in-time soft sensor application, which aims to deal with missing data [93]. Different with the former, two kinds of VAEs were used in a new soft sensor framework, which also focuses on the missing data [94]. The first one named supervised deep VAE was designed to obtain the distribution of latent features, which was used as a prior of the second one known as the modified unsupervised deep VAE. Then, the framework was constructed by combining the encoder of the first one with the decoder of the second one, which works well under the missing data situation.

In some cases, AEs could work better by combining it with other methods or improving its learning strategy. For example, Yao and Ge implemented a deep network of AEs for unsupervised feature extraction and then utilized extreme learning machine (ELM) for regression task [95]. Wang and Liu [96] adopted the limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm to optimize the weights parameters learned by SAE, and then the features extracted were fed into support vector regression (SVR) model for estimating the rotor deformation of air preheaters. Instead of using pure data-driven models (DDMS), Wang and Liu combined a knowledge-based model (KDM) named the lab model with a DDM namely SAE, and the experimental results verified that the hybrid method is prior than using only KDM or DDM [97]. Using an improved gradient descent algorithm, Yan et al. [98] proposed a DAE-based method, which was demonstrated to be effective compared with conventional approaches, such as shallow learning methods. Besides, to adaptively model time-varying processes, a just-in-time fine-tuning framework was proposed for SAE-based soft sensor construction [99].

## B. RBM-Based Applications

Nonlinearity is a widely existing characteristic in industrial processes. Aiming at this, RBM and its variants, especially DBN, are generally used as unsupervised nonlinear feature extractors in industrial process modeling.

Predictors can take advantage of features learned by RBM or DBN, and SVR and BPNN are two common kinds of predictors. For example, to address the problem of high nonlinearity and strong correlation among multivariables in the process of coal-fired boiler, a novel deep structure using continuous RBM and SVR algorithms was proposed [100]. A related work was proposed by Lian et al. [101], which uses DBN and SVR with the improved particle swarm optimization to complete the task of rotor thermal deformation prediction. In [102], a soft sensor model based on the DBN and BPNN was proposed to predict the 4-carboxy-benzaldchydc concentration in the purified terephthalic acid (PTA) industrial production process. Faced with the complexity and nonlinearity of nonlinear system modeling, an improved BPNN based on RBM was proposed in [103]. In this article, the structure of BPNN is optimized by utilizing sensitivity analysis and mutual information theories and the initialization of parameters is done by RBM. While in [104], DBN was used to learn hierarchical features for a BPNN, which was constructed for modeling the relationships between extracted features and mill level in a ball mill production process. In addition to SVR and BPNN, ELM can also work as a predictor based on the features extracted by DBN. The idea was realized in the measurement of nutrient solution composition for soilless culture [105].

To overcome the data-rich-but-information-poor problem, RBMs can be utilized for ensemble learning. For instance, Zheng et al. [106] proposed a soft sensing framework that integrates the ensemble strategy, DBN, and correntropy kernel regression into a unified soft sensing framework. Similarly, an ensemble deep kernel learning model was proposed in industrial polymerization process, which adopts DBN for unsupervised information extraction [107]. In the other case, lack of the labeled sample also leads to poor information, which can be settled by semisupervised learning using DBN, such as the work proposed in [108]. In [109], focusing on labeled data scarcity, computational complexity reduction, and unsupervised feature exploitation, a DBN-based soft sensor is designed.

RBMs have some other interesting applications as well. Graziani and Xibilia [110] designed a soft sensor based on DBN for a plant process to estimate an unknown measurement delay rather than quality variables. Another DBN-based model was applied to process flame images, rather than common structural data, in industrial combustion processes for oxygen content prediction [111]. Zhu and Zhang [112] investigated the selection of DBN structure for the soft sensor application in an industrial polymerization process. By comparing with feedforward NNs, the DBN-based method can give more accurate predictions of the polymer melt index.

## C. CNN-Based Applications

CNNs are mainly utilized for processing gridlike data, especially image data. Besides, they can also be developed to capture local dynamic characteristics of industrial process data or process signals in frequency domain.

By processing image data, CNN can be used to construct soft sensors. For example, Horn et al. [113] use CNN to extract features in froth flotation sensing, which shows a good feature extraction speed and predictive performance. However, images are still seldom utilized for soft sensor construction compared to common data forms.

As for dynamic problems, Yuan et al. [114] proposed multichannel CNN for soft sensing application in the industrial debutanizer column and hydrocracking process, which can learn dynamics and various local correlations of different variable combinations. Besides, Wang et al. [115] used two CNN-based soft sensor models to deal with abundant process data for the purpose of staying low complexity and embracing the process dynamics at the same time. In [116], a soft sensor was proposed using the CNN, which predicts the measurements at next time step by extracting time-dependent correlations from a moving window.

In frequency domain, CNNs can acquire high invariance to signal translation, scaling, and distortion. In [117], a pair of convolution layer and max-pooling layer was utilized at the lowest part of network to extract high-level abstraction from the vibration spectral features of the mill bearing. Then, ELM learns a mapping from the extracted features to the mill level. In the field of aerospace engineering, a virtual sensor model with partial vibration measurements using a CNN was proposed for estimating the structural response, which is important for structural health monitoring and damage detection but physical sensors are limited in the corresponding operational conditions [118].

## D. RNN-Based Applications

RNNs are widely used for dynamic modeling, and various variants, such as LSTM, are also applied in real cases.

RNN-based soft sensors were developed to estimate variables with strong dynamic characteristic, such as the curing of epoxy/graphite fiber composites [119], the contact area that tires of a car are making with the ground [120], the indoor air quality in the subway [121], the melt-flow-length in the injection molding process [122], the biomass concentrations [123], and the product concentration of reactive distillation columns [124].

Apart from methods based on ordinary RNN, LSTM is also a popular model in soft sensing applications, which can be deeper and more powerful since long-term dependence is weakened. For example, an LSTM-based soft sensor model was proposed to cope with strong nonlinearity and dynamics of the process in [125]. Besides, Yuan et al. [126] proposed a supervised LSTM network, which used both the input and quality variables to learn dynamic hidden states, and the method was proved to be effective on a penicillin fermentation process and an industrial debutanizer column. Besides, an LSTM network was used to predict the content of nitrogen-derived components in wastewater treatment plants [127].

There are other variants that are designed for specific industrial applications. As an example, a two-stream network structure was designed, which adopts batch normalization and dropout tricks, to learn diverse features of the various process data [128]. In [129], another type of RNN called time delayed NN (TDNN) was implemented for inferential state estimation for an ideal reactive distillation column. Besides, the echo state network as a kind of RNN was also used for soft sensing application in the high-density polyethylene production process and PTA production process [130]. By taking advantage of singular value decomposition, the collinearity and overfitting problems were solved. Recently, an ensemble semisupervised model, which combining SAE with bidirectional LSTM, was proposed in [131]. The new method can not only extract and utilize the temporal behavior in labeled and unlabeled data but also take the time dependence hidden in quality metric self into consideration. Also, GRU-based method is proposed for automatic deep extraction of robust dynamic features in [132], and achieves good performance in a debutanizer distillation process.

## E. Other DL-Based Applications

In addition to applications based on aforementioned mainstream models, some other deep models are also used to solve soft sensing problems. Some typical applications are discussed as the following and the others will not be analyzed in detail here.

1) Semisupervised Modeling: In [133], a semisupervised framework was constructed by integrating manifold embedding into a deep NN (DNN), in which manifold embedding exploited the local neighbor relationship among industrial data and improved the utilization efficiency of unlabeled data in DNN. Besides, a just-in-time semisupervised soft sensor based on ELM was proposed to online estimate the Mooney viscosity with multiple recipes in [134].

2) Dynamic Modeling: Except CNNs and RNNs, there are some other NNs that are used for dynamic modeling. Graziani and Xibilia [135] proposed a dynamic DNN-based soft sensor to estimate the research octane number for a reformer unit in a refinery and nonlinear finite input response models were investigated. Wang et al. [136] proposed a dynamic network called NARX-DNN, which can interpret the quality prediction error of validation data from different aspects and automatically determine the most appropriate delay of historical data. Besides, a dynamic strategy is adopted to improve the dynamic capture performance of the ELM, which is combined with PLS in [137].

3) Data Generation: Due to the harsh environment of the industrial process, directly collecting data may be difficult.

Therefore, a generative adversarial networks based method was proposed for data generation in [138].

4) Elimination of Redundancy: In [139], a double least absolute shrinkage and selection operator algorithm was integrated into a multilayer perceptron network to solve two redundancy problems: the input variable redundancy and the model structure redundancy.

5) Inference and Approximation: Due to the strong learning ability, DNNs can be used for intelligent control purposes. For example, a soft sensor based on Levenberg–Marquart and adaptive linear network was designed and applied in inferential control of a multicomponent distillation process [140]. In addition, the adaptive fuzzy means algorithm was utilized to evolve a radial basis function NN, which aimed at the approximation of an unknown system [141].

## F. Summary of the Existing Applications

The purposes of developing DL-based novel soft sensors include feature extraction, solving missing value issues, dynamic characteristics capture, semisupervised modeling, etc. (see Table I). It is worth noting that only existing applications in soft sensor field are discussed in detail, which does not mean that what has not yet appeared in the field of soft sensor is not possible. For example, although VAE is the mainstream method to deal with missing value problems for soft sensor application using DL, methods based on RBM and GAN are also feasible in other fields [142], [143]. To design feasible models, different strategies were adopted, such as optimizing network structure, improving the training algorithm, and integrating different algorithms.

From the applications discussed in aforementioned sections, some points can be further summarized. First, the statistics on soft sensor applications using DL methods can be seen in Fig. 9, which is based on a total of 57 references discussed and cited in Section IV. From Fig. 9(a), the trend is clear that there are more and more algorithms based on DL theory during recent years, which is a reflection of the increasing demand for DL models in real industrial process modeling. Moreover, compared with three other main theories, CNN-based methods are applied less. This is because gridlike data, such as images, are more used for classification rather than regression tasks. Besides, although AE looks simpler than other main models, it is easier to be developed and extended, so it is also of great potential.

As shown in Fig. 9(b), soft sensors based on DL theory were constructed in many scenarios, including chemical industry, power industry, machinery manufacturing, aerospace engineering, etc. Among them, chemical industry applications account for the largest proportion at about 66.7%. The effectiveness of most of the work reviewed in this survey is verified by doing numerical simulation experiments (e.g., [93], [114], etc.), or by using public available benchmark datasets (e.g., [137]), or by modeling the datasets from real-world processes (e.g., [91], [92], [93], [108], [114], [121], etc.). The most common case is the third type, which can reflect the characteristics of real processes as much as possible. For example, in chemical industry field, actual run data are collected from processes, such as debutanizer process [94], polymerization processes [107], hydrocracking process [114], to name a few. However, more detailed and specific factors need to be considered when applying those soft sensors to real scenarios.

![](images/1016c18825fec1808ccc9e605b43b0fba00fb137b8f2cfda5a2099fb80a87622.jpg)  
(a)

![](images/5eaf8cecf998ed4a48e725db30dfa79a7a7fa7a479f4f9caac7a0be3270186de.jpg)  
(b)  
Fig. 9. Statistics on existing relevant work. (a) Publications in different years. (b) Applications on different fields.

## V. DISCUSSIONS AND OUTLOOK

Although DL has made great progress in many fields, there is still a lot of work to do to better apply the advanced methods in the soft sensor domain, especially to meet the demands in practical industrial processes. Data and structure are the two most important issues required to be considered all the time. Around these two topics, some hot research directions should be paid more attention in the future.

## A. Lack of Labeled Sample

Although the data are easy to obtain under the trend of big data, the annotation cost is still very expensive. Therefore, we always hope that using fewer labeled samples can train a model with good generalization ability. Traditional solution of this problem is using semisupervised learning methods, whereas the more and more serious imbalance problem between unlabeled and labeled data makes it less satisfactory. Self-supervised learning (SSL) is another feasible solution, which is a kind of unsupervised strategy [144]. Different with transfer learning [32], [33], the useful feature representations are learned from a pretext task designed from the unlabeled input data (not from other similar datasets). Contrastive way is one of the most popular types of SSL, and has made some great achievements in speech, images, text, and reinforcement learning fields [146]. However, a lot of investigation and exploration work remains to be done for its soft sensing application.

## B. Hyperparameter Optimization

For a long time, how to optimize hyperparameters and structures of networks is a difficult issue for researchers and engineers [104], [112], [139]. Most of such work require manual trial. To avoid heavy workload and great randomness, meta-learning was proposed and investigated, which is also called as “learn to learn” [146]. The motivation is to offer machine with humanlike learning ability. Instead of learning a single function for a specific task, meta-learning learns a function to output functions for several subtasks. At the same time, many subtasks are required for meta-learning, and each subtask has its own training set and test set. After effective training, machine can possess the ability to optimize hyperparameters, including selecting network structures by itself. This is attractive for multimodal and changing processes.

## C. Model Reliability

DL methods learn features in an end-to-end way, which increases the difficulty for engineers or designers to understand what and how they learned. Besides, the dependence of the learning process on data increases the inaccuracy caused by poor data quality. Both of these two factors pose a threat on the reliability of DL models. To improve the model reliability, model visualization [147], [148] and combining DL models with experience or knowledge [149] are two feasible ways. Model visualization helps researchers to understand what has been learned, and introducing experience or knowledge helps to reduce inaccuracy brought by just relying on data. Nevertheless, these two points need more investigations for practical industrial application.

## D. Distributed Parallel Modeling

With the trend of industrial big data discussed in Section II, how to efficiently model the process from large amount of data is an important and urgent issue. A feasible solution is to transform original DL models into the distributed and parallel modeling. By splitting a large dataset into several small distributed blocks, data processing can be carried out simultaneously, which is conducive to large-scale data modeling [150], [151]. So far, however, there is still a long distance to go.

## VI. CONCLUSION

DL techniques have shown their great potential in many fields, as well as in soft sensor. In order to summarize the past, analyze the present, and look into the future, in this article, we made the following contributions to the application of DL theory in the field of soft sensor.

1) The merits of DL compared with traditional algorithms and the trends of the industrial processes were discussed in detail to demonstrate the necessity and significance of DL algorithms for soft sensor modeling.

2) Main DL models, tricks, and frameworks/toolkits were discussed and summarized to help readers better develop DL-based soft sensors.

3) Practical application scenarios were analyzed by reviewing and discussing existing work or publications.

4) Possible research hot points for future work were investigated shortly.

It is our hope for this article to serve as a taxonomy and also a tutorial of advances elucidated from a multitude of works on DL-based soft sensors, and to provide the community with a picture of the roadmap and matters for future endeavors.

## REFERENCES

[1] B. Huang and R. Kadali, Dynamic Modeling, Predictive Control and Performance Monitoring. London, U.K.: Springer, 2008.

[2] X. Wang, B. Huang, and T. Chen, “Multirate minimum variance control design and control performance assessment: A data-driven subspace approach,” IEEE Trans. Control Syst. Technol., vol. 15, no. 1, pp. 65–74, Jan. 2007.

[3] Z. Chen, S. X. Ding, T. Peng, C. Yang, and W. Gui, “Fault detection for non-Gaussian processes using generalized canonical correlation analysis and randomized algorithms,” IEEE Trans. Ind. Electron., vol. 65, no. 2, pp. 1559–1567, Feb. 2018.

[4] Y. Jiang, S. Yin, J. Dong, and O. Kaynak, “A review on soft sensors for monitoring, control and optimization of industrial processes,” IEEE Sensors J., to be published, doi: 10.1109/JSEN.2020.3033153.

[5] V. Venkatasubramanian, R. Rengaswamy, and S. N. Kavuri, “A review of process fault detection and diagnosis: Part II: Qualitative models and search strategies,” Comput. Chem. Eng., vol. 27, no. 3, pp. 313–326, 2003.

[6] P. Kadlec, B. Gabrys, and S. Strandt, “Data-driven soft sensors in the process industry,” Comput. Chem. Eng., vol. 33, pp. 795–814, 2009.

[7] M. Kano and M. Ogawa, “The state of the art in chemical process control in Japan: Good practice and questionnaire survey,” J. Process Control, vol. 20, pp. 969–982, 2010.

[8] K. Pearson, “LIII. On lines and planes of closest fit to systems of points in space,” Philos. Mag., vol. 2, no. 11, pp. 559–572, 1901.

[9] H. Wold, “Estimation of principal components and related models by iterative least squares,” J. Multivariate Anal., vol. 1, pp. 391–420, 1966.

[10] Q. Jiang, X. Yan, H. Yi, and F. Gao, “Data-driven batch-end quality modeling and monitoring based on optimized sparse partial least squares,” IEEE Trans. Ind. Electron., vol. 67, no. 5, pp. 4098–4107, May 2020.

[11] W. Yan, H. Shao, and X. Wang, “Soft sensing modeling based on support vector machine and Bayesian model selection,” Comput. Chem. Eng., vol. 28, pp. 1489–1498, 2004.

[12] K. Desai, Y. Badhe, S. S. Tambe, and B. D. Kulkarni, “Soft-sensor development for fed-batch bioreactors using support vector regression,” Biochem. Eng. J., vol. 27, pp. 225–239, 2006.

[13] G. Hinton, S. Osindero, and Y.-W. Teh, “A fast learning algorithm for deep belief nets,” Neural Comput., vol. 18, no. 7, pp. 1527–1554, 2006.

[14] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” J. Mach. Learn. Res., vol. 9, pp. 249–256, 2010.

[15] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, pp. 436–444, 2015.

[16] F. A. A. Souza, R. Araújo, and J. Mendes, “Review of soft sensor methods for regression applications,” Chemometrics Intell. Lab. Syst., vol. 152, pp. 69–79, 2016.

[17] K. Hornik et al. “Multilayer feedforward networks are universal approximations,” Neural Netw., vol. 2, pp. 359–366, 1989.

[18] G. Cybenko, “Approximation by superpositions of a sigmoidal function,” Math. Control Signals Syst., vol. 2, pp. 303–314, 1989.

[19] K. Hornik, “Approximation capabilities of multilayer feedforward networks,” Neural Netw., vol. 4, pp. 251–257, 1991.

[20] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.

[21] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, Cambridge, MA, USA: MIT Press, 2016.

[22] C. Grosan and A. Abraham, “Rule-based expert systems,” Intell. Syst., vol. 17, pp. 149–185, 2011.

[23] A. Lig˛eza, Logical Foundations for Rule-Based Systems, 2nd ed., Heidelberg, Germany: Springer, 2006.

[24] J. Durkin, Expert Systems: Design and Development. New York, NY, USA: Prentice-Hall, 1994.

[25] C. R. Turner, A. Fuggetta, L. Lavazza, and A. L. Wolf, “A conceptual basis for feature engineering,” J. Syst. Softw., vol. 49, no. 1, pp. 3–15, 1999.

[26] F. Nargesian, H. Samulowitz, U. Khurana, E. B. Khalil, and D. Turaga, “Learning feature engineering for classification,” in Proc. 26th Int. Joint Conf. Artif. Intell., 2017, pp. 2529–2535.

[27] Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 8, pp. 1798–1828, Aug. 2013.

[28] A. Ng, “Machine learning yearning,” 2017. [Online]. Available: http: //www.mlyearning.org/(96)

[29] S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE Trans. Knowl. Data Eng., vol. 22, no. 10, pp. 1345–1359, Oct. 2010.

[30] Y. Bengio, “Deep learning of representations for unsupervised and transfer learning,” in Proc. ICML Workshop Unsupervised Transfer Learn., 2012, pp. 17–36.

[31] W. Shao, Z. Song, and L. Yao, “Soft sensor development for multimode processes based on semisupervised Gaussian mixture models,” IFAC-PapersOnLine, vol. 51, no. 18, pp. 614–619, 2018.

[32] F. A. A. Souza and R. Araújo, “Mixture of partial least squares experts and application in prediction settings with multiple operating modes,” Chemometrics Intell. Lab. Syst., vol. 130, no. 15, pp. 192–202, 2014.

[33] H. Jin, X. Chen, L. Wang, K. Yang, and L. Wu, “Dual learning-based online ensemble regression approach for adaptive soft sensor modeling of non-linear time-varying processes,” Chemometrics Intell. Lab. Syst., vol. 151, pp. 228–244, 2016.

[34] M. Kano and K. Fujiwara, “Virtual sensing technology in process industries: Trends and challenges revealed by recent industrial applications,” J. Chem. Eng. Jpn., vol. 46, 2012, Art. no. 12we167, doi: 10.1252/jcej.12we167.

[35] L. X. Yu, “Pharmaceutical quality by design: Product and process development, understanding, and control,” Pharmaceut. Res., vol. 25, pp. 781–791, 2008.

[36] S. J. Qin, “Process data analytics in the era of big data,” AIChE J., vol. 60, no. 9, pp. 3092–3100, 2014.

[37] N. Stojanovic, M. Dinic, and L. Stojanovic, “Big data process analytics for continuous process improvement in manufacturing,” in Proc. IEEE Int. Conf. Big Data, 2015, pp. 1398–1407.

[38] L. Yao and Z. Ge, “Big data quality prediction in the process industry: A distributed parallel modeling framework,” J. Process Control, vol. 68, pp. 1–13, 2018.

[39] M. S. Reis and G. Gins, “Industrial process monitoring in the big data/Industry 4.0 era: From detection, to diagnosis, to prognosis,” Processes, vol. 5, no. 3, 2017, Art. no. 35.

[40] S. W. Roberts, “Control charts tests based on geometric moving averages,” Technometrics, vol. 1, pp. 239–250, 1959.

[41] C. A. Lowry, W. H. Woodall, C. W. Champ, and C. E. Rigdon, “A multivariate exponentially weighted moving average control chart,” Technometrics, vol. 34, pp. 46–53, 1992.

[42] T. Kourti and J. F. MacGregor, “Multivariate SPC methods for process and product monitoring,” J. Qual. Technol., vol. 28, pp. 409–428, 1996.

[43] M. S. Reis and P. M. Saraiva, “Prediction of profiles in the process industries,” Ind. Eng. Chem. Res., vol. 51, pp. 4254–4266, 2012.

[44] C. Duchesne, J. J. Liu, and J. F. MacGregor, “Multivariate image analysis in the process industries: A review,” Chemometrics Intell. Lab. Syst., vol. 117, pp. 116–128, 2012.

[45] D. C. Montgomery and C. M. Mastrangelo, “Some statistical process control methods for autocorrelated data,” J. Qual. Technol., vol. 23, pp. 179–193, 1991.

[46] T. J. Rato and M. S. Reis, “Advantage of using decorrelated residuals in dynamic principal component analysis for monitoring largescale systems,” Ind. Eng. Chem. Res., vol. 52, pp. 13685–13698, 2013.

[47] G. E. Hinton and J. L. McClelland, “Learning representations by recirculation,” in Proc. Int. Conf. Neural Inf. Process. Syst., 1988, pp. 358–366.

[48] D. E. Rumelhar, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” Nature, vol. 323, no. 6088, pp. 533–536, 1986.

[49] H. Larochelle, I. Lajoie, Y. Bengio, and P. A. Manzagol, “Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion,” J. Mach. Learn. Res., vol. 11, no. 12, pp. 3371–3408, 2010.

[50] B. Schölkopf, J. Platt, and T. Hofmann, “Efficient learning of sparse representations with an energy-based model,” in Proc. Adv. Neural Inf. Process. Syst., 2006, pp. 1137–1144.

[51] M. A. Ranzato, Y. L. Boureau, and Y. Lecun, “Sparse feature learning for deep belief networks,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2007, pp. 1185–1192.

[52] A. Hassanzadeh, A. Kaarna, and T. Kauranne, “Unsupervised multimanifold classification of hyperspectral remote sensing images with contractive autoencoder,” in Proc. Scandinavian Conf. Image Anal., 2017, pp. 169–180.

[53] Y. Bengio, “Learning deep architectures for AI,” Found. Trends Mach. Learn., vol. 2, no. 1, pp. 1–127, 2009.

[54] G. E. Hinton, “A practical guide to training restricted Boltzmann machines,” in Neural Networks: Tricks of the Trade. Berlin, Heidelberg: Springer, 2012, pp. 599–619.

[55] G. E. Hinton and R. R. Salakhutdinov, “Deep Boltzmann machines,” J. Mach. Learn. Res., vol. 5, no. 2, pp. 1967–2006, 2009.

[56] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Proc. Adv. Neural Inf. Process. Syst., 2012, pp. 1097–1105.

[57] Y. Zhou and R. Chellappa, “Computation of optical flow using a neural network,” in Proc. IEEE Int. Conf. Neural Netw., 1988, pp. 71–78.

[58] Y. LeCun et al., “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998.

[59] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Adv. Neural Inf. Process. Syst., vol. 2012, no. 25, pp. 1097–1105, 2012.

[60] P. J. Werbos, “Backpropagation through time: What it does and how to do it,” Proc. IEEE, vol. 78, no. 10, pp. 1550–1560, Oct. 1990.

[61] Y. Bengio, P. Simard, and P. Frasconi, “Learning long-term dependencies with gradient descent is difficult,” IEEE Trans. Neural Netw., vol. 5, no. 2, pp. 157–166, Mar. 1994.

[62] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” in Proc. Int. Conf. Mach. Learn., 2013, pp. 1310–1318.

[63] F. A. Gers, J. Schmidhuber, and F. Cummins, “Learning to forget: Continual prediction with LSTM,” Neural Comput., vol. 12, no. 10, pp. 2451–2471, 2000.

[64] R. Pascanu, C. Gulcehre, K. Cho, and Y. Bengio, “How to construct deep recurrent neural networks,” 2013, arXiv:1312.6026.

[65] K. Cho, B. V. Merriënboer, C. Gulcehre, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in Proc. Conf. Empirical Methods Nat. Lang. Process., 2014, pp. 1724–1734.

[66] G. Chrupala, A. Kadar, and A. Alishahi, “Learning language through pictures,” 2015, arXiv:1506.03694.

[67] F. Girosi, M. Jones, and T. Poggio, “Regularization theory and neural networks architectures,” Neural Comput., vol. 7, no. 2, pp. 219–269, 1995.

[68] D. M. Montserrat, Q. Lin, J. Allebach, and E. J. Delp, “Training object detection and recognition CNN models using data augmentation,” Electron. Imag., vol. 2017, no. 10, pp. 27–36, 2017.

[69] N. Jaitly and G. E. Hinton, “Vocal tract length perturbation (VTLP) improves speech recognition,” in Proc. ICML Workshop Deep Learn. Audio, Speech, Lang. Process., 2013.

[70] P. Vincent et al., “Extracting and composing robust features with denoising autoencoders,” in Proc. 25th Int. Conf. Mach. Learn., 2008, pp. 1096–1103.

[71] B. Poole, J. Sohl-Dickstein, and S. Ganguli, “Analyzing noise in autoencoders and deep networks,” 2014, arXiv:1406.1831.

[72] R. Caruana, S. Lawrence, and C. L. Giles, “Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping,” in Proc. Adv. Neural Inf. Process. Syst., 2001, pp. 381–387.

[73] Z. Zhang, Y. Xu, J. Yang, X. Li, and D. Zhang, “A survey of sparse representation: Algorithms and applications,” IEEE Access, vol. 3, pp. 4910–4530, 2015.

[74] H. Larochelle and Y. Bengio, “Classification using discriminative restricted Boltzmann machines,” in Proc. 25th Int. Conf. Mach. Learn., 2008, pp. 536–543.

[75] Y. Pati, R. Rezaiifar, and P. Krishnaprasad, “Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition,” in Proc. 27th Annu. Asilomar Conf. Signals, Syst., Comput., 1993, pp. 40–44.

[76] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, pp. 1929–1958, 2014.

[77] G. E. Hinton et al., “Improving neural networks by preventing coadaptation of feature detectors,” 2012, arXiv:1207.0580.

[78] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proc. Int. Conf. Mach. Learn., 2015, pp. 448–456.

[79] M. Abadi et al., “TensorFlow: A system for large-scale machine learning,” in Proc. 12th Symp. Oper. Syst. Des. Implementation, 2016, pp. 265–283.

[80] Y. Jia et al., “Caffe: Convolutional architecture for fast feature embedding,” in Proc. 22nd ACM Int. Conf. Multimedia, 2014, pp. 675–678.

[81] F. Bastien et al., “Theano: New features and speed improvements,” 2012, arXiv:1211.5590.

[82] F. Seide and A. Agarwal, “CNTK: Microsoft’s open-source deep-learning toolkit,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining, 2016, pp. 2135–2135.

[83] A. Gulli and S. Pal, Deep Learning With Keras. Birmingham, U.K.: Packt Publishing Ltd., 2017.

[84] A. Paszke et al. “PyTorch: An imperative style, high-performance deep learning library,” in Proc. Adv. Neural Inf. Process. Syst., 2019, pp. 8026–8037.

[85] B. Shen, L. Yao, and Z. Ge, “Nonlinear probabilistic latent variable regression models for soft sensor application: From shallow to deep structure,” Control Eng. Pract., vol. 94, 2020, Art. no. 104198.

[86] X. Yuan, B. Huang, Y. Wang, C. Yang, and W. Gui, “Deep learning-based feature representation and its application for soft sensor modeling with variable-wise weighted SAE,” IEEE Trans. Ind. Informat., vol. 14, no. 7, pp. 3235–3243, Jul. 2018.

[87] X. Yan, J. Wang, and Q. Jiang, “Deep relevant representation learning for soft sensing,” Inf. Sci., vol. 514, pp. 263–274, 2020.

[88] X. Yuan, J. Zhou, B. Huang, Y. Wang, C. Yang, and W. Gui, “Hierarchical quality-relevant feature representation for soft sensor modeling: A novel deep learning strategy,” IEEE Trans. Ind. Informat., vol. 16, no. 6, pp. 3721–3730, Jun. 2019.

[89] Q. Sun and Z. Ge, “Gated stacked target-related autoencoder: A nove deep feature extraction and layerwise ensemble method for industria soft sensor application,” IEEE Trans. Cybern., to be published, doi: 10.1109/TCYB.2020.3010331.

[90] Q. Sun and Z. Ge, “Deep learning for industrial KPI prediction: When ensemble learning meets semi-supervised data,” IEEE Trans. Ind. Informat., vol. 17, no. 1, pp. 260–269, Jan. 2021.

[91] X. Wang and H. Liu, “Data supplement for a soft sensor using a new generative model based on a variational autoencoder and Wasserstein GAN,” J. Process Control, vol. 85, pp. 91–99, 2020.

[92] F. Guo, R. Xie, and B. Huang, “A deep learning just-in-time modeling approach for soft sensor based on variational autoencoder,” Chemometrics Intell. Lab. Syst., vol. 197, 2020, Art. no. 103922.

[93] F. Guo, W. Bai, and B. Huang, “Output-relevant variational autoencoder for just-in-time soft sensor modeling with missing data,” J. Process Control, vol. 92, pp. 90–97, 2020.

[94] R. Xie, N. M. Jan, K. Hao, L. Chen, and B. Huang, “Supervised variational autoencoders for soft sensor modeling with missing data,” IEEE Trans. Ind. Informat., vol. 16, no. 4, pp. 2820–2828, Apr. 2019.

[95] L. Yao and Z. Ge, “Deep learning of semisupervised process data with hierarchical extreme learning machine and soft sensor application,” IEEE Trans. Ind. Electron., vol. 65, no. 2, pp. 1490–1498, Feb. 2017.

[96] X. Wang and H. Liu, “Soft sensor based on stacked auto-encoder deep neural network for air preheater rotor deformation prediction,” Adv. Eng Informat., vol. 36, pp. 112–119, 2018.

[97] X. Wang and H. Liu, “A knowledge- and data-driven soft sensor based on deep learning for predicting the deformation of an air preheater rotor,” IEEE Access, vol. 7, pp. 159651–159660, 2019.

[98] W. Yan, D. Tang, and Y. Lin, “A data-driven soft sensor modeling method based on deep learning and its application,” IEEE Trans. Ind. Electron., vol. 64, no. 5, pp. 4237–4245, May 2017.

[99] Y. Wu, D. Liu, X. Yuan, and Y. Wang, “A just-in-time fine-tuning framework for deep learning of SAE in adaptive data-driven modeling of time-varying industrial processes,” IEEE Sensors J., vol. 21, no. 3, pp. 3497–3505, Feb. 2021.

[100] W. Fan et al., “Integration of continuous restricted Boltzmann machine and SVR in NOx emissions prediction of a tangential firing boiler,” Chemometrics Intell. Lab. Syst., vol. 195, 2019, Art. no. 103870.

[101] P. Lian et al., “Soft sensor based on DBN-IPSO-SVR approach for rotor thermal deformation prediction of rotary air-preheater,” Measurement, vol. 165, 2020, Art. no. 108109.

[102] R. Liu, Z. Rong, B. Jiang, Z. Pang, and C. Tang, “Soft sensor of 4- CBA concentration using deep belief networks with continuous restricted Boltzmann machine,” in Proc. 5th IEEE Int. Conf. Cloud Comput. Intell. Syst., Nanjing, China, 2018, pp. 421–424.

[103] J. Qiao and L. Wang, “Nonlinear system modeling and application based on restricted Boltzmann machine and improved BP neural network,” Appl. Intell., vol. 51, pp. 37–50, 2020.

[104] M. Lu, Y. Kang, X. Han, and G. Yan, “Soft sensor modeling of mill level based on deep belief network,” in Proc. 26th Chin. Control Decis. Conf., 2014, pp. 189–193.

[105] X. Wang, W. Hu, K. Li, L. Song, and L. Song, “Modeling of soft sensor based on DBN-ELM and its application in measurement of nutrient solution composition for soilless culture,” in Proc. IEEE Int. Conf. Saf. Produce Informatization, Chongqing, China, 2018, pp. 93–97.

[106] S. Zheng et al., “Robust soft sensor with deep kernel learning for quality prediction in rubber mixing processes,” Sensors, vol. 20, no. 3, 2020, Art. no. 695.

[107] Y. Liu et al., “Ensemble deep kernel learning with application to quality prediction in industrial polymerization processes,” Chemometrics Intell. Lab. Syst., vol. 174, pp. 15–21, 2018.

[108] C. Shang et al., “Data-driven soft sensor development based on deep learning technique,” J. Process Control, vol. 24, no. 3, pp. 223–233, 2014.

[109] S. Graziani and M. G. Xibilia, “Deep learning for soft sensor design,” in Proc. Develop. Anal. Deep Learn. Architectures, 2020, pp. 31–59.

[110] S. Graziani and M. G. Xibilia, “Design of a soft sensor for an industrial plant with unknown delay by using deep learning,” in Proc. IEEE Int. Instrum. Meas. Technol. Conf., Auckland, New Zealand, 2019, pp. 1–6.

[111] Y. Liu, Y. Fan, and J. Chen, “Flame images for oxygen content prediction of combustion systems using DBN,” Energy Fuels, vol. 31, no. 8, pp. 8776–8783, 2017.

[112] C. H. Zhu and J. Zhang, “Developing soft sensors for polymer melt index in an industrial polymerization process using deep belief networks,” Int. J. Autom. Comput., vol. 17, no. 1, pp. 44–54, 2020.

[113] Z. C. Horn et al., “Performance of convolutional neural networks for feature extraction in froth flotation sensing,” IFAC-PapersOnLine, vol. 50, no. 2, pp. 13–18, 2017.

[114] X. Yuan et al., “Soft sensor model for dynamic processes based on multichannel convolutional neural network,” Chemometrics Intell. Lab. Syst., vol. 203, 2020, Art. no. 104050.

[115] K. Wang et al., “Dynamic soft sensor development based on convolutional neural networks,” Ind. Eng. Chem. Res., vol. 58, no. 26, pp. 11521–11531, 2019.

[116] W. Zhu et al., “Deep learning based soft sensor and its application on a pyrolysis reactor for compositions predictions of gas phase components,” Comput. Aided Chem. Eng., vol. 44, pp. 2245–2250, 2018.

[117] J. Wei, L. Guo, X. Xu, and G. Yan, “Soft sensor modeling of mill level based on convolutional neural network,” in Proc. 27th Chin. Control Decis. Conf., 2015, pp. 4738–4743.

[118] S. Sun et al., “A data-driven response virtual sensor technique with partial vibration measurements using convolutional neural network,” Sensors, vol. 17, no. 12, 2017, Art. no. 2888.

[119] H. B. Su, L. T. Fan, and J. R. Schlup, “Monitoring the process of curing of epoxy/graphite fiber composites with a recurrent neural network as a soft sensor,” Eng. Appl. Artif. Intell., vol. 11, no. 2, pp. 293–306, 1998.

[120] C. A. Duchanoy et al., “A novel recurrent neural network soft sensor via a differential evolution training algorithm for the tire contact patch,” Neurocomputing, vol. 235, pp. 71–82, 2017.

[121] J. Loy-Benitez, S. K. Heo, and C. K. Yoo, “Soft sensor validation for monitoring and resilient control of sequential subway indoor air quality through memory-gated recurrent neural networks-based autoencoders,” Control Eng. Pract., vol. 97, 2020, Art. no. 104330.

[122] X. Chen, F. Gao, and G. Chen, “A soft-sensor development for melt-flowlength measurement during injection mold filling,” Mater. Sci. Eng.: A, vol. 384, no. 1/2, pp. 245–254, 2004.

[123] L. Z. Chen et al., “Soft sensors for on-line biomass measurements,” Bioprocess Biosyst. Eng., vol. 26, no. 3, pp. 191–195, 2004.

[124] G. Kataria and K. Singh, “Recurrent neural network based soft sensor for monitoring and controlling a reactive distillation column,” Chem. Product Process Model., vol. 13, no. 3, 2017, doi: 10.1515/cppm-2017-0044.

[125] W. Ke, D. Huang, F. Yang, and Y. Jiang, “Soft sensor development and applications based on LSTM in deep neural networks,” in Proc. IEEE Symp. Ser. Comput. Intell., 2017, pp. 1–6.

[126] X. Yuan, L. Li, and Y. Wang, “Nonlinear dynamic soft sensor modeling with supervised long short-term memory network,” IEEE Trans. Ind. Informat., vol. 16, no. 5, pp. 3168–3176, May 2020.

[127] I. Pisa et al., “ANN-based soft sensor to predict effluent violations in wastewater treatment plants,” Sensors, vol. 19, no. 6, 2019, Art. no. 1280.

[128] R. Xie, K. Hao, B. Huang, L. Chen, and X. Cai, “Data-driven modeling based on two-stream λ gated recurrent unit network with soft sensor application,” IEEE Trans. Ind. Electron., vol. 67, no. 8, pp. 7034–7043, Aug. 2020.

[129] S. R. V. Raghavan, T. K. Radhakrishnan, and K. Srinivasan, “Soft sensor based composition estimation and controller design for an ideal reactive distillation column,” ISA Trans., vol. 50, no. 1, pp. 61–70, 2011.

[130] Y. L. He et al., “Novel soft sensor development using echo state network integrated with singular value decomposition: Application to complex chemical processes,” Chemometrics Intell. Lab. Syst., vol. 200, 2020, Art. no. 103981.

[131] X. Yin et al., “Ensemble deep learning based semi-supervised soft sensor modeling method and its application on quality prediction for coal preparation process,” Adv. Eng. Informat., vol. 46, 2020, Art. no. 101136.

[132] X. Zhang and Z. Ge, “Automatic deep extraction of robust dynamic features for industrial big data modeling and soft sensor application,” IEEE Trans. Ind. Informat., vol. 16, no. 7, pp. 4456–4467, Jul. 2020.

[133] W. Yan et al., “Soft sensor modeling method based on semisupervised deep learning and its application to wastewater treatment plant,” Ind. Eng. Chem. Res., vol. 59, no. 10, pp. 4589–4601, 2020.

[134] W. Zheng et al., “Just-in-time semi-supervised soft sensor for quality prediction in industrial rubber mixers,” Chemometrics Intell. Lab. Syst., vol. 180, pp. 36–41, 2018.

[135] S. Graziani and M. G. Xibilia, “Deep structures for a reformer unit soft sensor,” in Proc. IEEE 16th Int. Conf. Ind. Informat., 2018, pp. 927–932.

[136] K. Wang, C. Shang, F. Yang, Y. Jiang, and D. Huang, “Automatic hyper-parameter tuning for soft sensor modeling based on dynamic deep neural network,” in Proc. IEEE Int. Conf. Syst., Man, Cybern., 2017, pp. 989–994.

[137] Y. He, Y. Xu, and Q. Zhu, “Soft-sensing model development using PLSRbased dynamic extreme learning machine with an enhanced hidden layer,” Chemometrics Intell. Lab. Syst., vol. 154, pp. 101–111, 2016.

[138] X. Wang, “Data preprocessing for soft sensor using generative adversarial networks,” in Proc. 15th Int. Conf. Control, Autom., Robot. Vis., 2018, pp. 1355–1360.

[139] Y. Fan, B. Tao, Y. Zheng, and S. Jang, “A data-driven soft sensor based on multilayer perceptron neural network with a double LASSO approach,” IEEE Trans. Instrum. Meas., vol. 69, no. 7, pp. 3972–3979, Jul. 2020.

[140] A. Rani, V. Singh, and J. R. P. Gupta, “Development of soft sensor for neural network based control of distillation column,” ISA Trans., vol. 52, no. 3, pp. 438–449, 2013.

[141] A. Alexandridis, “Evolving RBF neural networks for adaptive soft-sensor design,” Int. J. Neural Syst., vol. 23, no. 6, 2013, Art. no. 1350029.

[142] M. D. Zeiler et al., “Modeling pigeon behavior using a conditional restricted Boltzmann machine,” in Proc. Eur. Symp. Artif. Neural Netw., 2009.

[143] Y. Luo et al., “Multivariate time series imputation with generative adversarial networks,” in Proc. Adv. Neural Inf. Process. Syst., 2018, pp. 1603–1614.

[144] L. Jing and Y. Tian, “Self-supervised visual feature learning with deep neural networks: A survey,” in Proc. IEEE Trans. Pattern Anal. Mach. Intell., 2020, doi: 10.1109/TPAMI.2020.2992393.

[145] A. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” 2018, arXiv:1807.03748.

[146] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in Proc. Int. Conf. Mach. Learn., 2017, pp. 1126–1135.

[147] L. Maaten and G. Hinton, “Visualizing data using t-SNE,” J. Mach. Learn. Res., vol. 9, pp. 2579–2605, Nov. 2008.

[148] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in Proc. Eur. Conf. Comput. Vis., 2014, pp. 818–833.

[149] S. Kabir et al., “An integrated approach of belief rule base and deep learning to predict air pollution,” Sensors, vol. 20, no. 7, 2020, Art. no. 1956.

[150] Q. Jiang, S. Yan, H. Cheng, and X. Yan, “Local-global modeling and distributed computing framework for nonlinear plant-wide process monitoring with industrial big data,” IEEE Trans. Neural Netw. Learn. Syst., to be published, doi: 10.1109/TNNLS.2020.2985223.

[151] Z. Yang and Z. Ge, “Monitoring and prediction of big process data with deep latent variable models and parallel computing,” J. Process Control, vol. 92, pp. 19–34, 2020.

![](images/ba5efc068499024da090d9667c920720222414aa26a59efd2aba477ffe45cbcc.jpg)

Qingqiang Sun received the B.Eng. degree in electrical engineering and automation from the School of Aerospace Engineering, Xiamen University, Xiamen, China, in 2017, and the M.Eng. degree in automation from the Department of Control Science and Engineering, Zhejiang University, Hangzhou, China, in 2020.

His research interests include data-driven modeling, deep learning, and soft sensor.

![](images/1dfb8be75a355970a2dcb3ca1ec2682eb3b170eb47aaab346c17a83f5d2a37c9.jpg)

Zhiqiang Ge (Senior Member, IEEE) received the B.Eng. and Ph.D. degrees in automation from the Department of Control Science and Engineering, Zhejiang University, Hangzhou, China, in 2004 and 2009, respectively.

He was a Research Associate with the Department of Chemical and Biomolecular Engineering, Hong Kong University of Science and Technology, Hong Kong, from 2010 to 2011 and a Visiting Professor with the Department of Chemical and Materials Engineering, University

of Alberta, Edmonton, AB, Canada, in 2013. He is currently a Full Professor with the College of Control Science and Engineering, Zhejiang University. His research interests include industrial big data, process monitoring, soft sensor, data-driven modeling, machine intelligence, and knowledge automation.

Dr. Ge was an Alexander von Humboldt Research Fellow with the University of Duisburg-Essen, Duisburg, Germany, from 2014 to 2017, and also a JSPS Invitation Fellow with Kyoto University, Kyoto, Japan, in 2018.