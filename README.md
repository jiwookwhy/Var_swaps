# Variance Swap Pricing with RL Agents

## Introduction

A variance swap is a deriative with payoff equal to the realized variance of an underlying asset up to a pre-defined maturity date. This project trains a reinforcement learning (RL) agent to fairly price variance swaps of any maturity given European-style option prices as the underlying asset. The RL agent is trained in the OpenAI gymnasium which provides a standardized API for training reinforcement learning agents. 

The training routine has the agent learn to play a game where the observed environment consists of a strip of European option prices, strike price and option type (call or put). We know from the work of [Demeterfi et al. (1999)](https://doi.org/10.2307/2677652) that the replicating portfolio can be priced using a portfolio of European-style options. Specifically, this portfolio statically replicates a multiple of a log contract (a European-style payoff equal to the natural algorithm of the underlying at expiration).  

The trained agent acts to select up to 15 options from the observed environment to include in a replicating portfolio whose price is as close to the model-free variance as possible. The agent can be trained using several contemporary deep learning models: Deep Q-learning, PPO (Proximate Policy Optimization), and Asynchronous Actor-Critic (A2C). This project aims to serve as a proof of concept for the robust replication of more exotic payoffs from vanilla options using RL agents. 

The pricing theory section provides an overview of the replication-based pricing approach for variance swaps and the RL environment section describes the learning/optimization problem implemented here. 

---

## Pricing Theory

For expositional purposes, I derive the variance swap replicating portfolio in the context of a simple univariate diffusion process. Let $S_t$ be the price of the underlying asset, $B_t$ denote a Brownian motion, and $T$ be the maturity of the swap. The dynamics of $S_t$ are assumed to be:

$$
\frac{d S_t}{S_t}=\mu \, d t+\sigma \, d B_t
$$

By Ito's Lemma we have:
$$
 d\left(\log S_t\right)=\left(\mu-\frac{\sigma^2}{2}\right) \, d t+\sigma \, d Z_t  
$$

Then the difference of the two stochastic processes gives a drift term purely in terms of the diffusion coefficent $\sigma$:
$$ 
\frac{d S_t}{S_t}-d\left(\log S_t\right)=\frac{\sigma^2}{2} \, d t
$$

Integrating we obtain the realized variance over the life of the swap:
$$
\frac{1}{T} \int_0^T \sigma^2 \, d t=\frac{2}{T}\left(\int_0^T \frac{d S_t}{S_t}-\ln \left(\frac{S_T}{S_0}\right)\right)
$$

We see from the right hand side that a variance swap can be perfectly hedged using a short position in two log contracts and a dynamic trading strategy holding $1/S_t$ units of the underlying. Since the dynamic trading strategy has zero cost, the variance swap is priced by the short position in two log contracts. Using the [Carr & Madan (1998)](https://doi.org/10.1111/j.1540-6261.1998.tb03270.x) formula we can obtain the replication for the log contract position. Applying the Carr-Madan formula, we obtain the fair price of variance:
$$
\mathbb{E}^\mathbb{Q}\left(\frac{1}{T} \int_0^T \sigma^2) = -2\mathbb{E}^\mathbb{Q}\left(\log(S_T)\right)=\frac{1}{T} \int_0^T \sigma^2 \, d t=\frac{2}{T}\left(\int_0^T \frac{d S_t}{S_t}-\ln \left(\frac{S_T}{S_0}\right)\right)
$$

Finally, we can discretizing the integral to obtain a tradable portfolio. This is computation is nearly idnentical to the Chicago Board of Exchange's computation of the VIX Index. 
$$
\text{Var Swap Price} = 2 e^{rf \times T} \left[ \sum\limits_{i: K_i < F_t} \dfrac{P(K)}{K_i^2} \dfrac{K_{i+1} - K_{i-1}}{2} + \sum\limits_{i: K_i \geq F_t} \dfrac{C(K_i)}{K_i^2} \dfrac{K_{i+1} - K_{i-1}}{2} \right]
$$

Where:
- \( F_t \) is the forward price of the underlying asset,
- \( P(K) \) is the price of a put option with strike \( K \),
- \( C(K) \) is the price of a call option with strike \( K \),
- \( rf \) is the risk-free rate,
- \( T \) is the time to maturity.

---

## RL Environment

The model is trained using a RL framework where the agent’s goal is to minimize the discounted sum of its rewards.  The optimization objective is formulated as:

\[
\min_{\theta} \sum_{i} \left( y_i - f_{\theta}(x_i) \right)^2
\]

where:
- \( x_i \) represents the input features derived from option data,
- \( y_i \) is the target variance swap rate,
- \( \theta \) denotes the model parameters.

This approach allows the agent to learn an optimal pricing strategy through continuous interaction with market data. For more on reinforcement learning fundamentals, refer to [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html).
