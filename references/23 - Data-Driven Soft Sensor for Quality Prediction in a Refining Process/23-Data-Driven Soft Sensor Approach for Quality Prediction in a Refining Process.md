# Data-Driven Soft Sensor Approach for Quality Prediction in a Refining Process

David Wang, Member, IEEE, Jun Liu, and Rajagopalan Srinivasan

Abstract—In the petrochemical industry, the product quality reflects the commercial and operational performance of a manufacturing process. However, real-time measurement of product quality is generally difficult. Online prediction of quality using readily available, frequent process measurements would be beneficial in terms of operation and quality control. In this paper, a novel soft sensor technology based on partial least squares (PLS) regression is developed and applied to a refining process for quality prediction. The modeling process is described, with emphasis on data preprocessing, multivariate-outlier detection and variables selection. Enhancement of PLS strategy is also discussed for taking into account the dynamics in the process data. The proposed approach is applied to data from a refining process and the performance of the resulting soft sensor is evaluated by comparison with laboratory data and analyzer measurements.

Index Terms—Outliers, partial least squares, quality prediction, refining process, soft sensor.

## I. INTRODUCTION

through a modern control system. Advanced process control (APC) is widely used to improve efficiency, optimize the process operation, and produce products with specified quality. In refinery, both lab and online analyzer measurements are available for measuring product quality. However, the analyzer measurements are slow (total time delay of about an hour—in comparison to the process time constant of few minutes), infrequent (one sample every half an hour) and sometimes unreliable; the more accurate lab measurement of the same variable is even slower (time delay of hours) and less frequent (about one or two samples per day). Thus, a direct sensing-based control strategy is not tenable.

Reliable online prediction of quality would be extremely beneficial in this environment; the real-time information about the effect of operational maneuvers on the product quality would allow control room personnel to make timely adjustment to the process to keep the control variables within limits. Such prediction is based on the ability to correlate the effect of process variables on the product quality. First principles-based physicochemical models are difficult to develop for large-scale reallife processes due to their inherent complexity. This motivates data-driven models. The widespread use of distributed control system (DCS) has allowed refiners to increase the number of measurements and actuators on the process. The integration of process historians with DCS provides a wealth of process data that can be exploited to extract information.

A soft sensor is an inferential model based on software technique to estimate the value of a process variable [1]. This is in contrast to a physical sensor that directly measures the value of the process variable. What makes soft sensors useful is the ability to infer in real time a measurement otherwise available only after significant delays as is associated with analyzers and lab tests. Early work on soft sensor is based on the Kalman filter, where the process model is assumed available [2]. In the case of complex systems where the process mechanism is not well-understood, empirical models developed by system identification techniques are used to derive the correlation among variables. Among the techniques, neural network (NN) technology and multiple linear regression have been widely employed to develop such data-driven models.

NNs are a powerful tool for nonlinear system identification. They have been applied in a variety of applications such as process control, signal processing, pattern recognition, and process monitoring. Recently, it has been reported that neural net-based models have been used in refinery industry for quality prediction [3]–[5]. Many advantages can be identified from their applications to inferential modeling. They can easily learn the nonlinear relationship among the variables and approximate it to the desired degree. However, NNs-based black-box model do not reveal a transparent relationship between the process and quality variables. Therefore, the inverse models for control purposes cannot be easily obtained.

Multiple linear regression can also be used for inferential model development. The PLSs method, in particular, is considered a robust alternative for the analysis of correlated data [6], [7]. It provides a transparent correlation model and a unique solution for a given set of training data. In this paper, a data-driven soft sensor approach based on PLS is presented for quality prediction and control in refinery.

This paper is organized as follows. In Section II, the PLS regression method is introduced in brief. Then, data preprocessing and multi-outliers detection are described in Section III. In Section IV, variable selection in PLS regression is discussed, the autocorrelation in process variables is considered and the enhancement of PLS to account for dynamics is presented. In Section V, the proposed approach is implemented on data collected from a refining plant and the performance of the resulting model is reported. Finally, conclusions are given in Section VI.

## II. PARTIAL LEAST SQUARES REGRESSION

## A. Multiple Linear Regression (MLR)

MLR [8] is one of the widely used regression methods, which relates predictor variable with response variable in a linear polynomial way

$$
y = b _ {1} x _ {1} + b _ {2} x _ {2} + \dots + b _ {m} x _ {m} + f.\tag{1}
$$

Vector can contain spectra, chromatograms, electrochemical data, collections of univariate process data such as temperature, pressure, viscosity, density, flow rate, etc., $b _ { k } ( k = 1 , \ldots , m )$ $\boldsymbol { b } = [ b _ { 1 } , b _ { 2 } , \dots , b _ { m } ]$ are regression coefficients, $f$ is a residual. In all practical situations, calibration measurements must be made for finding the regression coefficients, giving a most compact equation

$$
Y = X b + F.\tag{2}
$$

is a column vector of $i = 1 , \ldots , n$ measured response. When there are many responses, the matrix $Y$ and the matrix contains these as $q = 1 , \ldots , q _ { c }$ column vectors.

There are many ways to determine the coefficient , but per haps the most obvious is using ordinary least squares

$$
\hat {b} = \left(X ^ {T} X\right) ^ {- 1} X ^ {T} y.\tag{3}
$$

Unfortunately, this approach often fails in practice if the data are not well designed. It cannot handle more variables than objects, it makes the calibration problem ill-conditioned due to the high degree of correlation between collinear variables, and it projects the noise present in into the calculation of the coefficient . The ill-conditioned problem can be dealt with by various modifications of MLR, such as principal components regression (PCR) and ridge regression (RR). PCR focuses more on predictors’ variance than prediction, while RR is concerned with the regression coefficients themselves. The PLS regression technique explains variations in both and simultaneously and also maximizes the and covariance.

## B. Partial Least Squares (PLS) Regression

Several ways exist to calculate PLS model parameters; however, the most intuitive method is known as NIPALS, non-iterative PLS [9]. NIPALS calculates new latent variables (LVs), known as scores and the loading and an additional set of vectors known as weight, . The addition of weights is required to maintain the orthogonal scores.

It can be shown that PLS coefficient can be obtained by

$$
\hat {b} = W (P ^ {T} W) ^ {- 1} (T ^ {T} T) ^ {- 1} T ^ {T} y.\tag{4}
$$

The scores and loadings calculated in PLS are not the same as those calculated in principal components analysis (PCA) and principal components regression. They can be thought of, however, as PCA scores and loadings that have been rotated to be more relevant for predicting . As in PCR, the PLS model converges to the MLR solution if all latent variables are included.

## C. Goodness of Fit of PLS Model and its Prediction Power

To investigate the fitting and predicting ability of a PLS model, it is useful to introduce the following measures, which tell one about the average deviation of the model from the data. The root-mean-square error of calibration indicates the fit of the model to the calibration data [10]. It is defined as

$$
\mathrm{RMSEC} = \sqrt {\frac {\sum_ {i = 1} ^ {n} (\hat {y} _ {i} - y _ {i}) ^ {2}}{n}}\tag{5}
$$

where the $\hat { y } _ { i }$ are the values of the predicted variable when all samples are included in the model formation and is the number of calibration samples. RMSEC measures how well the model fits the data.

This is contrast to the root-mean-square error of cross validation (RMSECV) which is a measure of the model’s ability to predict new samples [9]. The RMSECV is defined as in (5), except the $\hat { y } _ { i }$ are predictions for samples not included in the model formation. RMSECV is related to the prediction error for the number of LVs included in the model, i.e.,

$$
\mathrm{RMSECV} _ {k} = \sqrt {\frac {P R E S S _ {k}}{n}}\tag{6}
$$

where $\mathrm { P R E S } _ { k }$ is the sum of squares prediction error for the model which includes factors. The exact value of RMSECV depends not only on but also on how the test sets were formed. It is common to calculate PRESS, and then RMSECV, using leave-one-out cross-validation [11], i.e., where each sample is left out of the model formulation and predicted once. Further details on PLS can be seen in the cited literature.

## III. DATA PREPROCESSING AND OUTLIER DETECTION

The quality of the predictor data is very important in PLS regression models. Suitable pretreatment of is sometimes crucial for the resulting model; often it may mean the difference between success and failure. Industrial data offers unique challenges due to its quality—while an ideal dataset would contain process information as much as possible, little noise and no outliers, the contrary is, in fact, most common in industrial situations. Missing data points are also very common in industrial practice and with no readily apparent pattern.

These data quality issues need to be carefully addressed prior to the modeling step. We have followed the following steps to improve data quality. (1) All process data were checked visually at first. Anything that appeared suspicious from a process point-of-view was double-checked and carefully considered to determine if they should be removed. (2) In the refinery process, missing data points not only exist in predictor data but also in the predicted quality data. For instance, in the current work, one of the quality streams is measured in the analytical lab and at irregular intervals (the average is one measurement every eight hours). This resulted in a large number of constant (missing) recordings of the predicted quality. To overcome this, the time of new quality measurements were carefully noted and only the corresponding process variable values were selected for data construction and validation. The predictor data were then obtained only from high-quality samples. If missing elements exist in a small block of the predictor dataset, interpolated values based on neighboring observations may be inserted. In the case of a large block corrupted with missing data, the entire block was not included in the training set.

Outliers, which can be simply regarded as the data points that are not consistent with the bulk of data, are common in industrial data set [12]. A soft sensor derived by PLS or other methods may deteriorate significantly even with a single outlier. Therefore, outlier detection constitutes an essential step in soft sensor development.

In the univariate approach, outliers are detected based on visualization together with the six-sigma rule. $x _ { i }$ is labeled as an outlier based on a six-sigma threshold if

$$
\left| x _ {i} - \bar {x} \right| > 3 \sigma\tag{7}
$$

where is the mean of data sequence and is the standard deviation, based on the normal distribution assumption of data. Unfortunately, this procedure fails in both theory and industrial practice because the outliers tend to inflate the variance estimation, resulting in too few outliers being detected. Even though robust outliers-resistant estimates such as median absolute deviation have been developed [13], they are applicable only to univariate data.

Since variables in a refinery process are usually not independent on one another, univariate outlier detection may not be effective and can result in masking and swamping effects [14], where outliers are incorrectly identified as normal samples, or normal samples classified as outliers. We have developed a multivariate analysis scheme based on PLS regression for abnormal sample detection and elimination [15]. The multivariate outlier detection scheme is summarized next.

Samples can be classified based on their effect on the multivariate model developed during multilinear regression. High leverage points are those that are outliers with respect to the independent variables [13]. Influential points are those that when deleted cause large changes in the parameter estimates. Although an influential point will typically have high leverage, a high leverage point is not necessarily an influential point. The leverage in PLS is typically defined as the diagonal of the hat matrix

$$
H = T W \left(P ^ {T} W\right) ^ {- 1} \left(T ^ {T} T\right) ^ {- 1} T ^ {T}.\tag{8}
$$

It defines the influence that a given sample will have on a model and is related to the $T ^ { 2 }$ value. On the other hand, once the coefficients in PLS model are available, the prediction of the predicted variables based on the model can be obtained. The residuals are the difference between the known values and predicted ones, and are an indication of the fit of the -value of a sample. The residual versus leverage plot thus allows identification of unusual, harmful samples. Generally, the decision to classify a sample as an outlier is made based not only on the statistics but also on process knowledge and the experience of the user.

## IV. VARIABLES SELECTION AND DYNAMIC PLS

## A. Variables Selection

In industrial processes, many variables describing process conditions are usually included in the predictor matrix . Even through including all available variables may appear to be a prudent choice in order to extract maximum possible information content, but some variables may have no effect on . Some variables may contain higher noise and thus deteriorate the accuracy and precision of the regression model. Excluding the less relevant and noisy variables often improves predictive power of the model. Further models with small number of variables are easily maintainable in the industrial setting. It is therefore useful to identify a subset of the variables that allows sufficient prediction accuracy and precision.

Process knowledge can be used to identify the subset of variables to be included in the model [16]. Alternatively, given a predictor data and a predicted variable one can choose a random subset of variables from and, through the use of PLS and cross-validation discussed in Section II, determine the RM-SECV obtained when using only that subset of variables in the model. One can use this approach iteratively to locate the variable subset or subsets that give the lowest RMSECV.

Following a strategy similar to a genetic algorithm, the first step in this process is to generate a large number of random selections of variables and calculate the RMSECV for each of given subsets. Each subset of variables is called an individual and a string of Boolean flags indicating which variables are used by that individual is considered the gene for that individual. A pool of individuals is called a population. The RMSECV results for all of the individuals can be examined as a function of the number of included variables. The second step is selection. The individuals with fitness lower than the median fitness (i.e., higher RMSECV) are discarded. The remaining individuals use variables which, for one reason or another, provide a lower RM-SECV—a better fit to the data. At this point, the population has been shrunk to half its original size. To replace the discarded individuals, the retained individuals are allowed to reproduce.

Reproducing can be done by cross-over—splitting the genes from two random individuals at some random point and swapping their parts, resulting in two new individuals. After adding the new individuals to the population, every gene is given a chance to undergo random mutation. This allows a finite chance of adding or removing the use of variables that might be over or underrepresented in the population. Finally, after all the individuals have been paired and bred, the population has returned to the original size and the process can continue again at the fitness evaluation step. This process will finish when one of two conditions are met: (a) after a finite number of iterations, or (b) after some percentage of the individuals in the population have identical variables subsets. Individuals using noisy or less relevant variables will tend to be discarded and, the variables used by those individuals will become less represented in the overall gene population. Likewise, less noisy and more relevant variables will become more and more represented. Depending on the number of variables and the rate of mutation, many of the individuals will eventually contain the same genes.

## B. Dynamic PLS

The previously discussed PLS regression implicitly assumes that the observations at one time instant are independent of past observations, i.e., the process is statistically stationary. This assumption is not valid in chemical processes. The product quality at an instant can result from the cumulative effects of past process conditions as well as the current ones. This suggests that a method taking into account the serial correlation in the data, as well as spatial correlation among the variables, is needed. An autocorrelation chart can be used to check if a significant autocorrelation exists in the process. To take into account the serial correlation, an extension of PLS can be implemented by augmenting each observation vector with previous observations and stacking the predictor matrix in the following way:

$$
X \left(h\right) = \left( \begin{array}{c c c c} x _ {t} ^ {T} & x _ {t - 1} ^ {T} & \dots & x _ {t - h} ^ {T} \\ x _ {t - 1} ^ {T} & x _ {t - 2} ^ {T} & \dots & x _ {t - h - 1} ^ {T} \\ \vdots & \vdots & \ddots & \vdots \\ x _ {t + h - n} ^ {T} & x _ {t + h - n - 1} ^ {T} & \dots & x _ {t - n} ^ {T} \end{array} \right)\tag{9}
$$

where $x _ { t } ^ { T }$ is the m-dimensional observation vector in the training set at time instant . Using the predictor in PLS regression results in a dynamic PLS model [17]. Including such time lags in the predictor data can provide information on the dynamic process conditions to the model. Hence, dynamic PLS is expected to perform better than static PLS for modeling and prediction of serially correlated data.

## V. CASE STUDY

## A. Process Description

The proposed data driven soft sensor approach was applied to a data set from an industrial refinery process. Crude is preheated to the desired temperature and is desalted to remove salt and impurities before it enters the crude tower. In the crude tower—crude distillation unit (CDU), fractionation occurs on the trays, separating the crude oil into desired fractions ac cording to their boiling points, from lighter to heavier. Wet gas and unstabilized naphtha pass overhead, and the low pressure wet gas is sent to the gas compression section. Light kerosene, heavy kerosene, light diesel, and heavy diesel are trapped out of the crude tower and then steam stripped to remove light hydrocarbons. The light kerosene and heavy kerosene are then combined and after cooling sent for sulphur removal. Some undesulphrized heavy kerosene is used as cutter stock for atmospheric residue. The light diesel is sent to the Diesel Hydrofiner. The heavy diesel is further processed (Fig. 1). The process described above is controlled through a DCS and equipped with a Plant Information Management System, which records and stores process data.

The ASTM 90% distillation temperature (D90) of the distillate streams is the quality index of the products. In the real plant an online analyzer is available for measurement. Also, infre quent measurements of the quality from lab are also available. In this work, the quality variable is estimated by the developed soft sensor in order to provide operators real-time information on actual operation performance. This allows control room personnel to make timely adjustment in process to keep the control variables within control limits.

## B. Soft Sensor Development and Prediction Results

This section presents the integrated application of the proposed modeling techniques to this industrial problem. Only one of the quality predictions using the proposed approach is reported here. It is to be noted that all the results presented in this work contain masked process variable values for reasons of confidentiality, therefore, the quality values and the values of process variables shown on the plots are transformed values and not the original values obtained from the plant. Fig. 2 shows the quality data from lab measurement and analyzer. It can be seen from the figure that even after filtering the analyzer data, there are still large discrepancies from the lab measurements. The aim here is to reduce the discrepancy by using a prediction model, i.e., the soft sensor. Sixty-three days data of 35 process condition variables and one quality variable are collected from the process historian. These were pretreated according to the strategy described in Section III. Five of 35 process variables are discarded based on the visual analysis. For variable selection, a set of 88 random individuals were generated and used as the initial population. The most relevant process variables were selected based on the strategy described in Section IV, with five generations which was adequate for convergence (less than 3% difference between average and best fitness). A PLS model was developed for each individual in the final population with the objective of predicting the lab measurement. Fig. 3 shows the fitness versus variable usage plot for the training data. Several trends in variable usage can be identified from the figure. If inclusion of a given variable tends to improve RMSECV, the inclusion marks for this variable will tend to appear towards the bottom of the figure. Conversely, a variable which tends to degrade the RMSECV will appear towards the top of the figure. If a variable does not show up at all, it is likely to be not useful. Conversely, a variable is probably more useful if it is used in all models. As indicated in the figure, 14 process variables are selected for the predictor. Fig. 4 displays the 151 data pairs for modelling, where the first part is the process stream quality data from lab measurement and the second part presents the snapshots of 14 process variables which are most significant to .

![](images/ef67d45f801d385debd86bac90a89b038e6c2b74b1cbbda9d609b670a447de60.jpg)  
Fig. 1. Crude distillation unit.

![](images/f648c55050e7a0dc44cd98098ceeac026f36d5fd8638896e285063d24d228cd2.jpg)  
Fig. 2. Quality data   measured from the Lab and produced by analyzer.

![](images/b18fb3663185a5143bcffb9e19fbf5adc2745a15ce3305920eb013f64d94ca10.jpg)  
Fig. 3. Fitness versus variables plot of training data.

![](images/9b4cd294db1f59a7aff257ef2415db15727aaa420e383625ddcdf9d44cf43838.jpg)

![](images/a3a147dc0c631f5f0dde8be5e4d9c172412a7bc525d789610eabcfa94c51cbf0.jpg)  
Fig. 4. Quality data   from Lab and predictor data  .

![](images/bcc873763ec2dff349b0ce8c516585659c178383dab08e9c7b15e5e58721d31d.jpg)  
Fig. 5. Studentized residuals versus leverage.

![](images/dd9fafdc8d4796b29885a1c3a3c0bdf70e7fac9b6bebe8484e9ccc6794f428b9.jpg)

![](images/932dbf749f997bd9df5d8035156c75e8898097927cff07da1327257c9320ce15.jpg)  
Fig. 6. Prediction results: (a) predicted values versus lab measurements and (b) prediction residuals distribution.

The outliers in the training data set can be detected by investigating the plot in Fig. 5, where each sample is represented as a single point in a scatter plot of studentized, that is autoscaled, residuals versus the leverages. The lower-most sample—Sample 89—has errors with standard deviations of greater than 3, which are quite large. This suggests that the -values associated with the samples may be in error. Samples 69 and 99 both have very high leverages and/or high residuals. This suggests that these samples are a “bad” influence on the model. Samples like these were carefully detected and removed from the training data set.

The lagged process data were included in the predictor data set to consider the process dynamics in order to improve the precision of the model. The lagged values were determined by correlation analysis and trial-and-error methods.

PLS was applied to the training data and leave-one-out crossvalidation was applied. The resulting model was also evaluated on a separate data set containing 63 samples. Fig. 6 shows the comparison between predicted and lab-measured quality along with their residual analysis. It can be seen that the residuals of the proposed prediction has narrower distribution than that of the analyzer. This means that the soft sensor has better perfor mance than the analyzer deployed in the plant. The performance criterion used to compare the accuracy of the prediction was

TABLE I  
AVERAGE ERROR (LAB MEASUREMENT AS BENCHMARK).

<table><tr><td></td><td>average % error $\overline{E}$ </td><td>Standard deviation of  $\overline{E}$ </td></tr><tr><td>Soft sensor</td><td>5.943e-3</td><td>4.4e-3</td></tr><tr><td>Analyzer</td><td>9.099e-3</td><td>7.5e-3</td></tr></table>

![](images/8c5d9b8b04946a26b2f3e6ea3982b75b67c126fd5bf146842f45250e228fda63.jpg)  
Fig. 7. Prediction results: Predicted values versus lab measurements.

$$
\bar {\varepsilon} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {\sqrt {\left(y _ {\mathrm{pred}} ^ {i} - y _ {\mathrm{meas}} ^ {i}\right) ^ {2}}}{y _ {\mathrm{meas}} ^ {i}}\tag{10}
$$

where is the number of testing samples and , are predicted and measured values, respectively. The average absolute error is smaller for the soft sensor than for the analyzer as tabulated in Table I.

The performance of prediction of the soft sensor can also be visualized by plotting the measured result against the predicted ones. When the predicted values match the measured ones, all points would lie on a diagonal. It can be seen from Fig. 7 that the soft sensor gives a good prediction of the product quality. Although some points are relatively far from the diagonal line, the prediction of soft sensor is closer to the lab values than that of the analyzer.

Next, we present a comparison between the PLS model and a NN-based soft sensor. In this instance, both models were developed to predict the quality as measured by the analyzer. Comparisons between the NN and PLS models are shown in Figs. 8 and 9, and Table II. It can be seen from these that similar degrees of precision is obtained by the two models. Given this comparative performance, and the black-box nature of NN models, as mentioned in Section I, the proposed PLS-based model is considered suitable for online deployment.

## VI. CONCLUSION

Soft sensors provide timely, beneficial information that would help process operations. While the basic multivariate statistical techniques underlying such model development are by now well known, their deployment in the industrial setting still remains a challenge because of data quality related issues. Industrial processes are often nonlinear, therefore, NN-based models are also a good candidate for soft sensor development. A key benefit of the statistical approaches is the transparency and uniqueness of the PLS model and the resulting ability to associate a physical significance to the underlying correlations which makes them intuitive to operators and suitable for simple inverse models that can also be used for control purposes. The precision of the proposed soft sensor can be improved by combining the linear predictor backbone with nonlinear terms. The authors intend to pursue this in the future.

![](images/c8a4a6bd3cdde84008fb967a13c0d59732277b8aaddcf92c8429d72f9ef5955c.jpg)

![](images/4613993503982d2ee66e26a66f4f61456262683390da381f3e1fd717605e6a4b.jpg)  
Fig. 8. Prediction results: (a) predictions of soft sensor versus filtered analyzer and (b) prediction residuals distributions.

![](images/e72f899b08c770ee44ba6d104ef9ba61eec34e85bf9cc74f0b73e2ce6b96e929.jpg)  
Fig. 9. Prediction results: predicted values versus filters analyzer values.

TABLE II  
AVERAGE PERCENTAGE ERROR COMPARISON WITH ANALYZER MEASUREMENT AS BENCHMARK (R: CORRELATION COEFFICIENT BETWEEN MEASUREMENT AND PREDICTION)

<table><tr><td></td><td>Average error  $\overline{E}$ </td><td>Standard deviation of  $\overline{E}$ </td><td>r</td></tr><tr><td>PLS model</td><td>4.542e-3</td><td>4.0e-3</td><td>0.8126</td></tr><tr><td>NN model</td><td>3.957e-3</td><td>4.0e-3</td><td>0.8494</td></tr></table>

## ACKNOWLEDGMENT

The authors would like to thank L. H. Wee, P. N. S. Guru, and K. M. Leong from the Singapore Refining Company for their strong support and invaluable assistance throughout this study.

## REFERENCES

[1] L. Fortuna, S. Graziani, A. Rizzo, and M. G. Xibilia, Soft Sensors for Monitoring and Control of Industrial Processes. New York: Springer, 2007.

[2] D Simon, Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approachesa. New York: Wiley, 2006.

[3] G. Martin, G. Barber, Z. Friedman, and E. Bullerdiek, “Refining and petrochemical property predictions for distillation, fractionation and crude switch,” in Proc. 2000 Comput. Conf., Chicago, IL, Nov. 13–15, 2000, 13 pages.

[4] N. Bonavita and T. Matsko, “Neural network technology applied to refinery inferential analyzer problems,” Hydrocarbon Engineering, Dec. 1999 [Online]. Available: http://library.abb.com/global/scot/ scot267.nsf/veritydisplay/1e9fce2471d1a93385256f9b005b9d1a/ \$File/NN\_for\_Refineries.pdf, Last accessed date: 21 July 2009

[5] J. Liu, R. Srinivasan, and P. N. S. Guru, “Practical challenges in development data-driven soft sensors for quality prediction,” ESCAPE-18, pp. 961–966.

[6] H. Wold, “Partial least squares,” in Encyclopedia of Statistical Sciences. Wiley, New York: , 1985, vol. 6.

[7] S. de Jong, B. M. Wise, and N. L. Ricker, “Canonical partial least squares and continuum power regression,” J. Chemometrics, vol. 15, no. 2, pp. 85–100, 2001.

[8] A. Hoskuldsson, Prediction Methods in Science and Technology. Copenhagen, Denmark: Thor Publishing, 1996, vol. 1.

[9] A. Hoskuldsson, “PLS regression methods,” J. Chemomatrics, vol. 2, pp. 211–228, 1988.

[10] L. Ljung, System Identification—Theory for the User, 2nd ed. Upper Saddle River, N.J: Prentice-Hall, 1999.

[11] M. Stone, “An Asymptotic Equivalence of Choice of Model by Cross-Validation and Akaike’s Criterion,” J. R. Stat. Soc., B, vol. 38, pp. 44–47, 1977.

[12] D. Wang and J. A. Romagnoli, “Robust multi-scale principal components analysis with applications to process monitoring,” J. Process Control, vol. 15, no. 8, pp. 869–882, 2005.

[13] P. J. Huber, Robust Statistics. New York: Wiley, 1981.

[14] J.-T. Chiang, “The algorithm for multiple outliers detection against masking and swamping effects,” Int. J. Contemp. Math. Sci., vol. 3, no. 17, pp. 839–859, 2008.

[15] D. Wang and R. Srinivasan, “Eliminating the effect of multivariate outliers in PLS-based models for inferring process quality,” in Proc. 19th Eur. Symp. Comput. Aided Process Eng., ESCAPE19, Cracow, Poland, Jun. 2009, pp. 14–17.

[16] R. Srinivasan and M. Qian, “State-specific key variables for monitoring multi-state processes,” Chem. Eng. Res. Design, vol. 85, no. A12, pp. 1630–1644, 2007.

[17] E. L. Russell, L. H. Chiang, and R. D. Braatz, Data-Driven Techniques for Fault Detection and Diagnosis in Chemical Processes. New York: Springer, 2000.

![](images/d36e2f4a0ed86596f0e9bc47d24b576308204ce090aafdcb783b9c9c6f00c354.jpg)  
David Wang (M’08) is a Senior Research Fellow at the Institute of Chemical and Engineering Sciences, Singapore. He is working on improving process efficiency, optimality, and profitability by applying advanced process control, where the work is focused on the theoretical and methodological development of process modeling, control, monitoring, optimization, and data analysis.

![](images/a847338c3b52e02a7ab844f99258aacda9aa30aca6d250895c7109a51463427f.jpg)

Jun Liu received the B.Eng., M.Eng., and Ph.D. degrees from Southeast University, China, in 1991, 1994, and 1997, respectively.

He is a Senior Research Fellow at the Institute of Chemical and Engineering Sciences, Singapore. His current research interests include process control and fuel cells.

![](images/4c9776ce3ee3231e3ee0c67154c378baf0c41ee604d96d474e4d50adcda8fb23.jpg)

Rajagopalan Srinivasan is an Associate Professor at the Department of Chemical and Biomolecular Engineering, National University of Singapore. He is concurrently a Principal Scientist at the Institute of Chemical and Engineering Sciences, where he leads the Process Systems and Control Team. His research program is targeted towards developing artificial intelligence and systems engineering approaches for process design and control and enterprise optimization.