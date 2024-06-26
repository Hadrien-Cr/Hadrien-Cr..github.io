---
title: "Bayesian Optimization 📉"
date: 2024-04-14
permalink: /posts/2024/04/BayesianOptimization/
excerpt: "In this post, I present the Bayesian Optimization algorithm, explain why it is so useful and share my implementation of it."
---
# <img src="/images/BayesianOpt/BayesianOpt1.jpeg" width="900" height="200">

# <img src="/images/DIR.png" width="300" height="70" style="font-size: 15px;">

In this post, I present the Bayesian Optimization algorithm, explain why it is so useful and share my implementation of it.

* * *

## 📰 Quick Presentation of the Tool

Bayesian Optimization is an optimization algorithm that works by iteratively approximating the objective function using a set of candidate functions that often resemble multimodal Gaussians. 

It requires very few assumptions on the objective function and is designed to achieve high efficiency for objective functions with a low-dimensional domain but large evaluation complexity.

* * *

## 🔬 Details on the Tool

### Step by Step

At the beginning of the algorithm, points to evaluate are drawn randomly within the domain of the objective function \( f \), starting with an arbitrary prior.

At each step, given the known values of the function, a Gaussian Process model is fitted, representing a distribution guess of the possible values that \( f(x) \) can take at points where \( f \) has not yet been evaluated.

# <img src="/images/BayesianOpt/BayesianOpt2.png" width="700" height="300">

The shape of the distribution is controlled by the kernel of the Gaussian Process model. This kernel can either allow high variations or enforce the estimate to be smoother to avoid overfitting a noisy objective function.

Given these distributions, the zone where to sample \( f \) next is determined. Both the lack of information about a zone and the performance of recent points are considered. Each point in the domain is granted an “acquisition score,” given by an arbitrary method that should strike a balance between exploration (visiting the least explored zones) and exploitation (visiting zones where the estimate of \( f \) is maximized).

Here is the pseudo code:

# <img src="/images/BayesianOpt/BayesianOpt4.png" width="500" height="400">

# <img src="/images/BayesianOpt/BayesianOpt3.png" width="500" height="370">

### What Acquisition Functions to Choose?

An acquisition function should only take the current distribution estimates of the function's values as input. A simple approach is to value the zones of the domain where the estimate \( \mu(x) \) is high, but the score should also be high if the estimate is uncertain.

Here is a list of common acquisition functions:

- **UCB (Upper Confidence Bound):** The most optimistic.
  
  \[
  \text{Score: } UCB(x) = \mu(x) + k \sigma(x)
  \]

  Where:
  \[
  \mu(x) \text{ is the mean prediction from the model at point } x,
  \]
  \[
  \sigma(x) \text{ is the standard deviation of the prediction at point } x,
  \]
  \[
  k \text{ is a trade-off parameter that balances exploration and exploitation.}
  \]

- **LCB (Lower Confidence Bound):** Similar, but pessimistic.
  
  \[
  \text{Score: } LCB(x) = \mu(x) - k \sigma(x)
  \]

- **PI (Probability of Improvement):** Measures the probability that the function value at a given point is better than the current best-known function value plus a margin of \( k \).

  \[
  \text{Score: } PI(x) = \Phi\left(\frac{f(x_{best}) + k - \mu(x)}{\sigma(x)}\right)
  \]

  Where:
  \[
  \Phi \text{ is the normal cumulative distribution function.}
  \]

- **EI (Expected Improvement):** Measures the expected improvement at a given point \( x \).

  \[
  \text{Score: } EI(x) = (\mu(x) - f(x_{best}) - k) \Phi\left(\frac{\mu(x) - f(x_{best}) - k}{\sigma(x)}\right)
  \]

* * *

## 👨‍💻 My Implementation [(Source Code)](https://github.com/Hadrien-Cr/Discover-Implement-Repeat/tree/main/Optimization/BAYESIAN_OPT)

I implemented the algorithm presented in the previous section.

For the Gaussian Process model, I started with the [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) from scikit-learn, then drew inspiration from Chapter 2 of [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/RW.pdf) to build my custom version.

Here is the pseudo code on how to fit and predict with a Gaussian Process Regressor:

My custom random regressor ended up being slower than the one from scikit-learn, especially during the prediction step.

* * *

## 👀 Visualizing the Search

### 1-Dimensional Domain

Search in a 1-dimensional domain, with kernel \( RBF(0.2) \), \( k = 0.1 \), acquisition function EI:

![1D Search Visualization](/images/BayesianOpt/BayesianOpt3.gif)

The upper and lower bounds form a 99% confidence interval on the surrogate function.

### 2-Dimensional Domain

Search in a 2-dimensional domain, with kernel \( RBF(0.5) \), \( k = 0.1 \), acquisition function EI:

![2D Search Visualization](/images/BayesianOpt/BayesianOpt4.gif)

* * *
