---
title: "Bandit Algorithms 🎰"
date: 2024-05-10
permalink: /posts/2024/05/BanditAlg/
excerpt: "In this post, I present a class of reinforcement learning algorithms, explain what are the most imprtant variations, and implement them.
."
---


# <img src="/images/BanditAlg/Bandit0.jpeg" width="900" height="200">
# <img src="/images/DIR.png" width="300" height="70" style="font-size: 15px;">

---

In this post, I present a class of reinforcement learning algorithms, explain what are the most imprtant variations, and implement them.

---

Main source: Chapter 2 of  [Reinforcement Learning: An Introduction Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html)

---

## 📰 Quick presentation of the tool

Bandit algorithms is a class of reinforcement learning algorithms that teaches a machine to play a game called a **bandit problem.**

Consider the following problem:

An agent has $$ k $$ different actions. After taking an action, the agent receive a reward. The rewards are chosen from a stationary probability distribution based on the action selected.

The objective of the agent is to maximize the total reward over all the period (for example 1000 choices of action made, called steps)

This is called the stationnary $$ k $$-armed bandit problem, because the agent has $$ k $$ choices and because the probability distributions are stationary.

![Playing with 4 different models of slot machines available is a 4-armed bandit problem](/images/BanditAlg/Bandit2.png)

Playing with 4 different models of slot machines available is a 4-armed bandit problem

For the agent, there are good choices (great expected reward), and bad choices (low expected reward), but he does not anything about it, and it will have to exploit the best actions (repeat the actions that he thinks are good) and to try the least explored actions (to make his estimates accurate).

Bandit problems are quite important in reinforcement learning because many problems can be formulated as repeated bandit problems.

## 🔬Details on the bandit algorithms

---

Some definitions:

- Let’s denote $$ A_t $$ the action selected at the step $$ t $$,
- $$ R_t $$ the corresponding reward,
- $$ q_*(a) = \mathbb{E}[R_t|A_t=a]$$ the expected reward of the action $$ a $$, also called *the value* of $$ a $$ (it does not change between the steps because of the stationnarity assumption).


<img src="/images/BanditAlg/Bandit3.png" width="500" height="350">

- $$ Q_t(a) $$  the estimate value of $$a$$ (estimated by the agent at step $$t$$). $$ Q_t(a) $$  should converge to $$ q_*(a) $$. 
- A greedy move consists of choosing the action that maximizes $$ Q_t(a) $$ .
- An exploratory move is a move that is not greedy.

### $$\varepsilon$$-greedy action selection

It is an action-value method, which means, that it uses **the estimates** to make action selection decisions. 

Its consists of picking a greedy move with probability $$ \varepsilon $$, picking a exploratory move with probability $$ \varepsilon $$, and using the rule  $$ Q_t(a) = \dfrac{\sum R_i \mathbb{1}_{A_i=a }}{N(a)}  +Q_1 $$;

 $$ Q_1 $$ being a initializing value common for all actions and $$ N(a) $$ is the number of times that the action $$ a $$  was selected. If the action has never been selected, use the initializing value instead.

With enough many steps, $$ N(a) $$ will get large enough for all actions $$ a $$ such that all estimates will be accurate, and the greedy move will be the ground best move. The lower $$ \varepsilon $$ is, the greedier the agent is, and the slower he learns the distributions. With higher values of $$ \varepsilon $$  is, the agent is less exploitative and more explorative.

The update rule can be implement as follows: after selecting the action $$ a $$ for the $$ n $$-th time, denoting $$ R_n $$  the reward obtained, $$ Q_n $$  the last estimate of $$ a $$  and $$ Q_{n+1} $$  its update:

$$
\begin{equation} 
Q_{n+1} = Q_n+\frac1n ( R_n -Q_n ) \end{equation}
$$

This corresponds to a more general update rule:

$$
\textit{NewEstimate} \leftarrow \textit{OldEstimate} + \textit{Step Size}\times(\textit{Target} -\textit{OldEstimate})
$$

1. Modifying the update rule for tracking a non-stationary problem: The choice of the step size $$ \alpha = \frac1n $$  gives the same weight to every reward occurence. The agent has infinite memory and trust the past experiments as much as the recent ones. 
    
    It works for stationary distributions, but if the distributions are changing over time, the agent would be better-off having a short term memory: if the best value drop off, we want the agent to adapt quickly. 
    
    The solution is changing the step size: With a constant step size of $$\alpha,$$ the weights of the past experiments are decaying geometrically with the rate $$(1-\alpha)$$ . This also applies to the initial guess of the value. 
    

### Upper-Confidence-Bound Action Selection (UCB Method)

This method is also a value-action method.

This method uses the same update rule $$ (1) $$ as the $$ \varepsilon $$-greedy action selection, but use a different action selection criteria, to discriminate between the exploratory moves; and choose a more clever exploration frequency.

The moves that should be picked are the ones that perform the best and have not been explored enough. Therefore, UCB uses the following criteria:

$$ A_t= \argmax_a  \left[ Q_t(a) +c\sqrt{\frac{\ln t}{N(a)}}  \right], \text{c being a fixed parameter} $$

The square root term is a term that assess how few explorations have been made on the specific action. 

If the action $$ a $$ has been picked $$N(a)$$  times, the interval $$ \left[Q_t(a) -c\sqrt{\frac{\ln t}{N(a)}} ; Q_t(a) +c\sqrt{\frac{\ln t}{N(a)}} \right] $$ is a interval of confidence for $$ q_*(a) $$ (for a certain level of confidence that lowers with time). 

The bigger $$c$$ is, the more confidence is required for discarding a bad move from being further explored, which means that exploration is encouraged.

Note that to complete the criterion, you have to set the unexplored actions to maximizing action, so that the first moves are explored

Overall, this action selection criterion encourages exploration by selecting actions that are potentially suboptimal but have not been explored much yet, thus providing a balance between exploitation and exploration.

### Gradient Bandit Algorithm

This method is **not** a value-action method. Instead of estimatimng the value of actions, the agent learns a preference for each action over another. 

The preference of selecting an action *a* over the other actions is $$ H_t(a) $$ but this does not have any interpretation, only the relative preferences do.

At each step, the probability of selecting any action is given according to a *soft-max distribution:*

$$ P(A_t=a) =\dfrac{e^{H_t(a)} }{\sum_{b\in \mathbb{A}} e^{H_t(b)}} $$

After selecting action $$a$$, the preferences are updated depending on the reward received and a **baseline** $$\bar{R}_t$$ (which can be running average of the of the last rewards):

$$
\begin{aligned}H_{t+1}\left(A_t\right) & \doteq H_t\left(A_t\right)+\alpha\left(R_t-\bar{R}_t\right)\left(1-\pi_t\left(A_t\right)\right), & & \text { and } \\H_{t+1}(a) & \doteq H_t(b)-\alpha\left(R_t-\bar{R}_t\right) \pi_t(b), & & \text { for all } b \neq A_t\end{aligned}
$$

- The difference $$ \left(R_t-\bar{R}_t\right) $$ represents how much the choice of $$ a $$  was relevant.
- $$ \alpha $$  is the learning rate of the algorithm
- For the action selected, the less it is prefered the more it is updated, and the more it is preferd the less it is updated.
- It can be shown that this algorithm is the algorithm of the stochastic gradient descent, where the objective function is $$ \mathbb{E}_\theta(R_t) $$ and the set of parameters is $$ \theta=(H_t(a_0),...H_t(a_k)) $$

### Tricks

- Optimistic initial values: Setting high default estimates allows to enhance the exploration. It is quite important for the $$\varepsilon$$   greedy action selection.

## 👨‍💻My implementation ([source code](https://github.com/Hadrien-Cr/Discover-Implement-Repeat/tree/main/Reinforcement_Learning/Bandit_Algorithms))

---

I implemented an $$ \epsilon $$ -greedy Agent, a UCB Agent, and a Agent that uses Gradient Bandit Algorithm.

For each of them, I created a class, and they all have in common a function play(problem): given a problem, and a history of the steps played, the agent picks a move, receives a reward and updates its history.

I also created a class k-armed-Bandit Problem. I did not implemented repeated Bandit problems. The class only implements the normal distribution for the reward distributtion, but other modes could be added.

## **📋 Benchmark**

---

The test used in the book [Reinforcement Learning: An Introduction Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html) is the following:

- A 10-armed bandit problem can be constructed as follows: $$ q_*(1),...,q_*(10) $$ are drawn randomly according to a normal gaussian distribution with mean 0 and variance 1, and the actual reward of choosing $$a$$ follows a normal distribution of mean $$ q_*(a) $$ and variance 1.
- The construction of such a test bed is repeated 2000 times independently and the results are averaged over these 2000 experiments.

<img src="/images/BanditAlg/Bandit4.png" width="550" height="350">

example of a problem

To evaluate a model on a draw of a 10-armed bandit problem, the metrics chosen is the average reward obtained over the first 1000 steps of learning, and this results is averaged on the 2000 bandit problems drawn.

<img src="/images/BanditAlg/Bandit5.png" width="550" height="350">

A parameter study of the various bandit algorithms presented in this chapter.

The experiment have very high variance, so making the 2000 runs is unavoidable in order to retrieve the same results. Making the 2000 runs was quite long (~1h with my budget CPU).

<img src="/images/BanditAlg/Bandit6.png" width="500" height="420">

results of my experiments