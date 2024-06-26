---
title: "Bandit Algorithms 🎰"
date: 2024-05-10
permalink: /posts/2024/05/BanditAlg/
excerpt: "In this post, I present a class of reinforcement learning algorithms, explain the most important variations, and implement them."
---

# <img src="/images/BanditAlg/Bandit0.jpeg" width="900" height="200">
# <img src="/images/DIR.png" width="300" height="70" style="font-size: 15px;">



In this post, I present a class of reinforcement learning algorithms, explain the most important variations, and implement them.



Main source: Chapter 2 of [Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html)



## 📰 Quick Overview of Bandit Algorithms

Bandit algorithms belong to a class of reinforcement learning algorithms used to solve a problem known as the **bandit problem**. In this scenario:

An agent faces a set of \( k \) different actions. After choosing an action, the agent receives a reward from a stationary probability distribution associated with that action.

The goal for the agent is to maximize the total reward accumulated over a series of steps (e.g., 1000 actions).

![Playing with 4 different models of slot machines available is a 4-armed bandit problem](/images/BanditAlg/Bandit2.png){: width="550" height="350"}

For the agent, some actions yield higher expected rewards (good choices) while others yield lower expected rewards (bad choices). The challenge lies in balancing exploration (trying out different actions) and exploitation (repeating actions believed to be good).

Bandit problems are fundamental in reinforcement learning as many real-world problems can be modeled as repeated bandit problems.

## 🔬Details on Bandit Algorithms



### Definitions:

<p>Let \( A_t \) denote the action selected at step \( t \),</p>
<p>Let \( R_t \) denote the corresponding reward,</p>
<p>\( q_*(a) = \mathbb{E}[R_t \mid A_t=a] \) represents the expected reward for action \( a \), also known as the *value* of \( a \) (which remains constant due to the stationary assumption).</p>

<img src="/images/BanditAlg/Bandit3.png" width="500" height="350">

<p>\( Q_t(a) \) is the estimated value of action \( a \) at step \( t \), which should converge to \( q_*(a) \) over time.A **greedy move** selects the action that maximizes \( Q_t(a) \).An **exploratory move** selects an action that is not greedy.</p>

### Epsilon-Greedy Action Selection

<p>This method is an action-value approach where the agent uses its estimates to decide on actions. It involves choosing a greedy move with probability \( \varepsilon \) and a random exploratory move with probability \( \varepsilon \).
The update rule for \( Q_t(a) \) after selecting action \( a \) for the \( n \)-th time, with reward \( R_n \), is:</p>

$$
Q_{n+1} = Q_n + \frac{1}{n} (R_n - Q_n)
$$

<p>This rule allows \( Q_t(a) \) to converge towards \( q_*(a) \) as more steps are taken. The parameter \( \varepsilon \) controls the balance between exploration and exploitation.</p>

### Upper-Confidence-Bound (UCB) Action Selection

<p>This method also uses an action-value approach with the same update rule as \( \varepsilon \)-greedy. However, it uses a different action selection criterion:</p>

$$
A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]
$$

<p>Here, \( c \) is a parameter that balances exploration and exploitation. Actions with higher uncertainty (less explored) are prioritized.</p>

### Gradient Bandit Algorithm

<p>Unlike the previous methods, this approach is not based on action values. Instead, it learns preferences for each action relative to others. The probability of selecting action \( a \) follows a softmax distribution based on preferences \( H_t(a) \):</p>

$$
P(A_t = a) = \frac{e^{H_t(a)}}{\sum_{b \in \mathcal{A}} e^{H_t(b)}}
$$

<p>Preferences \( H_t(a) \) are updated based on the reward received \( R_t \) and a baseline \( \bar{R}_t \):</p>

$$
H_{t+1}(A_t) \leftarrow H_t(A_t) + \alpha (R_t - \bar{R}_t) (1 - P_t(A_t))
$$

### Tips and Tricks

- **Optimistic Initial Values:** Starting with high initial estimates encourages exploration, which is crucial for methods like \( \varepsilon \)-greedy.

## 👨‍💻 My Implementation ([source code](https://github.com/Hadrien-Cr/Discover-Implement-Repeat/tree/main/Reinforcement_Learning/Bandit_Algorithms))



<p>I implemented an \( \epsilon \)-greedy agent, a UCB agent, and a gradient bandit algorithm agent. Each agent is encapsulated in a class and includes a method to play a given bandit problem, updating its history of actions and rewards.</p>

I also developed a k-armed Bandit Problem class, although it currently only supports stationary reward distributions.

## **📋 Benchmark**



<p>The benchmark used in [Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html) involves a 10-armed bandit problem where \( q_*(1), ..., q_*(10) \) are drawn from a normal distribution with mean 0 and variance 1. The reward for each action follows a normal distribution with mean \( q_*(a) \) and variance 1.</p>

Results are averaged over 2000 independent runs of this test bed to reduce variance.

![Example of a 10-armed bandit problem](/images/BanditAlg/Bandit4.png){: width="550" height="350"}

To evaluate a model, the average reward over the first 1000 steps of learning is calculated and averaged across the 2000 bandit problems.

![Parameter study of bandit algorithms](/images/BanditAlg/Bandit5.png){: width="550" height="300"}

The experiments exhibit high variance, necessitating the 2000 runs to achieve reliable results. Conducting these runs took approximately 1 hour on my CPU.

![Results of my experiments](/images/BanditAlg/Bandit6.png){: width="400" height="350"}