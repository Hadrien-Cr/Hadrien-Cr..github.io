---
title: "Random Forests 🌳"
date: 2024-04-20
permalink: /posts/2024/03/RandomForests/
excerpt: "In this post, I showcase one of my favorite ML models, explain the gist of the algorithm, share my implementation and compare its performances with benchmark references."
---

# <img src="/images/RandomForests/RF1.jpeg" width="900" height="200">
# <img src="/images/DIR.png" width="300" height="70" style="font-size: 15px;">


In this post, I showcase one of my favorite ML models, explain the gist of the algorithm, share my implementation and compare its performances with benchmark references.

* * *
## 📰 Quick presentation of the model
* * *
The Random Forest model is a versatile and robust machine learning algorithm that utilizes a multitude of decision trees.

* * *

A decision tree is a simple model, quite optimistic, that hopes to classify a sample, only by asking this sample binary questions: For instance, one can categorize species into families by asking binary questions, thus one can build an efficient decision tree for this classification task.

<img src="/images/RandomForests/RF2.png" alt="partial decision tree for classifying species" width="400" height="250" class="jop-noMdConv">

partial decision tree for classifying species

The major flaw of the decision tree is that overfitting is not controlled: To make the best decision tree with a certain training set, it has to ask the most questions possible for each sample. However, this hyper-specificity can lead to poor generalization when applied to discovering an new species: If an alien has 2 legs and hair, it doesnt mean it is a mammal !

The only way to control the overfitting of a decision tree is to limit his depth or the features available, but it means that his classifying power is limited.

* * *

The Random Forest is a model that uses multiple decision trees **to strike a balance between overfitting and modeling effectiveness**. Moreover, Random Forest extends the application of decision trees to various tasks, including regression.

* * *

## 🔬Details on the Random Forest Classifier

The algorithm comes from the following publication: *[Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001).](https://doi.org/10.1023/A:1010933404324)*

The algorithm applies a Bagging (bootstrap aggregation (Breiman 1994)) of decision trees.

This is the general form of bagging:

1.  Repeatedly draw, **with replacement,** $n \leq N$ samples from the $N$ samples. (boostraping)
2.  For each set of samples, estimate a statistic (for example an estimated class)
3.  The aggreagted estimate is the mean of the individual estimates.

The Random Forest model operates by aggregating predictions from multiple individual decision trees, each decision tree is trained on a bootstrapped sample of the original data.

The random forest algorithm is an extension of the bagging method as it utilizes both bagging and feature randomness, also known as feature bagging or “[the random subspace method](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)”. It consist of keeping only a fraction of the features for each decision tree.

### **Algorithm:**

1 - Repeatedly draw, **with replacement,** $n≤ N$ samples. Then determine the best Decision tree Classifier corresponding

To do so, here is the procedure:

- The decision tree is trained using the $m$ attributes and the $n$ samples. The root node represent the origin of the process, where all samples are asked the same binary question o (the split).
- For each node, search among $m$ randomly drawn attributes for the best split of the data to maximize the **reduction in impurity.**

![example of a trained decison tree able to predict if a passenger of the titanic is likely to survive](/images/RandomForests/RF4.png)

example of a trained decison tree able to predict if a passenger of the titanic is likely to survive

*For binary trees, the reduction in impurity is computed with the formula:*

$$
\Delta Gini = Gini( \text{parent node})-  \alpha Gini( \text{left node}) -(1-\alpha) Gini( \text{right node}) \\ (\alpha \text{ is the share of the samples going to the left node)}
$$

*The [Gini Index](https://en.wiktionary.org/wiki/Gini_coefficient#English) is a metrics of the impurity of a node*

*For example if the parent node has samples of 2 equiprobable classses*

(ie $Gini(\text{parent node}) = 0.5$ ) and if the split separate perfectly the 2 classes

(ie $Gini(\text{left node}) = Gini(\text{right node}=0)$), **then the reduction in impurity is (0.5).**

*But if the split does not separate the 2 classes at all*

*(ie $Gini(\text{left node}) = Gini(\text{right node})= Gini(\text{parent node})=0.5$)**, then the reduction in impurity is 0***

**Note that the computation of the impurity can change with the criterion chosen.**

- The criterion to stop splitting a node (making it a leaf node) is either hitting **maximum depth** of the decision tree, or **running short of samples** on the node, or the $m$ features drawn are not able to split the samples (the samples are identical with respect to the features).

2 - The process is repeated for the specified number of trees.

3 - For predicting the class of a test sample, the sample goes through all the trees, and the prediction is made by aggregating the votes to a majority vote. The votes can also be interpreted as a probibility: if 80 % of trees vote “class A” and the others voter “class B”, the probability of class.

<img src="/images/RandomForests/RF6.png" alt="partial decision tree for classifying species" width="500" height="350" >

summary of the algorithm

### **Parameters:**

1.  **n_estimators:** number of decision trees in the forest.
2.  **max_depth**: The maximum depth allowed for each decision tree. This parameter controls the complexity of individual trees and helps prevent overfitting.
3.  **min_samples_split**: The minimum number of samples required to split a node. It prevents further splitting of nodes if the number of samples falls below this threshold, helping to control tree growth.
4.  **min_samples_leaf**: The minimum number of samples required to be at a leaf node. It ensures that each leaf node represents a certain amount of data, preventing overly specific splits.
5.  **m (max_features)**: The number of features to consider when looking for the best split. A smaller value introduces more randomness and diversity among the trees.

## 🔬 Details on the Random Forest Regressor

For regression tasks, where the target is a continuous value, each decision tree is tasked to output its predictions as a numerical value, and the output prediction is the mean (or the median) of all the predictions from all the trees. Note that each decision tree as only a finite of possible outputs, limited by the number of leaf nodes.

## 📝 Advantages and Limitations of the random forests

### **Advantages:**

- **Reduced overfitting**: By aggregating predictions from multiple trees, Random Forest reduces the risk of overfitting compared to individual decision trees.
- **Robustness**: Random Forest is less sensitive to outliers and noise in the data due to the averaging effect of multiple trees.
- **Feature importance**: It provides a measure of feature importance, indicating which features are most influential in making predictions.

### **Limitations:**

- **Computational complexity**: Training and prediction can be computationally expensive, especially with a large number of trees and features. It does not exploit the GPU computer power
- **Interpretability**: While Random Forest can provide insights into feature importance, the individual decision trees may be difficult to interpret, especially in complex forests.

Overall, Random Forest is a versatile and powerful algorithm suitable for various machine learning tasks, offering a good balance between model complexity and performance.

## 💻 Other variations of the model

The random forest has inspired a lot of research because it is the pioneer of the so-called tree-based estimators. Here are the most famous examples:

- **Extremely Randomized Trees (Extra-Trees):** Another variation of the random forest is the Extra-Trees method. It goes one step further in injecting randomness into the model by also using random thresholds for each feature rather than searching for the best possible thresholds (like a regular decision tree).
- **Gradient Boosted Decision Trees (GBDT):** Unlike Random Forests, which builds each tree independently, GBDT builds one tree at a time, where each new tree helps to correct errors made by previously trained trees.
- **Adaptive Boosting (AdaBoost):** This is a boosting algorithm that works by fitting a sequence of weak learners on repeatedly modified versions of the data. It starts by predicting the original data set and then every subsequent iteration modifies the weights of the incorrectly classified instances such that it ensures the accuracy of prediction in the next iteration.
- **Histogram-Based Gradient Boosting** is a variant of gradient boosting machines and it is much faster than the traditional Gradient Boosting. This is because instead of making one decision tree at a time, this method sorts the data before making a decision tree, and then, reuses the sorted data to make further trees. It results in a significant decrease in computational time.The histogram-based method also reduces memory usage. Instead of using the original data, it works on integer-valued bins that approximate the original continuous feature values.
- **XGBoost:** It stands for Extreme Gradient Boosting. It is a very efficient implementation of gradient boosting framework by Chen & Guestrin (2016) that includes both a linear model solver and tree learning algorithms. It is known for its performance and computational speed, and recognized for its effectiveness in large datasets. It also exploit the computing power of GPUs.

* * *

## 👨‍💻My implementation [(Source code)](https://github.com/Hadrien-Cr/Discover-Implement-Repeat/tree/main/Models/Random_Forest)

- Step 1: Build an Algorithm to train a single decision tree
- Step 2: Combine decisions trees into a random Forest

Step 1 required to know a suitable way to encode graph and a clever way to compute the best split.

For simplification, i did not implement an option to partial fit, so the fit can only be done once.(Otherwise it requires to code a solution to update the search for the best split. Maybe a solution could be to store the training samples, and when we want to make an new fit with other training samples, erase all the knowledge of the tree and fit it with all the training samples.)

I initialy misinterpreted the algorithm: instead a randomly drawing $m$ features **for each node** of the decision tree, I thought the draw had to be done only once, which makes each decision less powerful.

The first test I conducted was the task of classifying these 1000 of samples of 3 “[blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs)” (the set being half for training and half for testing).

<img src="/images/RandomForests/RF8.png"  width="500" height="350" >

Illustration of the classification task

The models are the Random Forest Classifier from sklearn, my custom Random Forest Classifier, and a single custom Decision Tree Classifier (parameters are max_features=1, n_estimators=100, max_depth=2, min_samples_split=2, min_samples_leaf=1)

<img src="/images/RandomForests/RF10.png"  width="500" height="250" >

Accuracy of each model on the task (averaged on 10 experiments)

* * *

## **📋 Benchmark**

To make a more formal benchmark, let’s try to reproduce one of the results from the publication [Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001)](https://doi.org/10.1023/A:1010933404324). I used [the following dataset](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) of breast cancer diagnosis (the same one used in the publication)

<img src="/images/RandomForests/RF11.png"  width="500" height="100" >

Here are the results presented in the article:

<img src="/images/RandomForests/RF13.png"  width="520" height="100" >


Test errors in percentage.

The test set is 10% of the original dataset, drawn randomly. There is no limit of depth of the trees, so i put $depth = log_2(N) +1$$=11$ (more leaves possible than samples). The number of trees is 100. max_features is set to 1 (for the colunm “Forest-RI single input). max_sample_split and min_samples_leaf are not mentionned so they are set to their minimimu values.

Here are the results of he Random Forest Classifier from sklearn, my custom Random Forest Classifier, and a single custom Decision Tree Classifier compared to the benchmark.

<img src="/images/RandomForests/RF12.png"  width="500" height="250" >


The experiment is repeated 50 times on different seeds.

* * *

The implementation is succesful !

* * *