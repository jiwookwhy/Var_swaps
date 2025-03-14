## Exotics pricing with RL Agents
This project aims to develop a training framework for reinforcement learning (RL) agents to price deriatives with exotic payoffs given option prices on the underlying. I apply this framework to the pricing of variance swaps as a proof-of-concept, as it requires a relatively complex replicating portfolio to price in a model-free fashion. 

## Variance Swaps

A variance swap is a deriative with payoff equal to the realized variance of an underlying asset up to a pre-defined maturity date. This project trains a RL agent to fairly price variance swaps of any maturity given European-style option prices as the underlying asset. The RL agent is trained in the OpenAI gymnasium which provides a standardized API for training reinforcement learning agents. 

The training routine has the agent learn to play a game where the observed environment consists of a strip of European option prices, strike price and option type (call or put). We know from the work of [Demeterfi et al. (1999)](https://doi.org/10.2307/2677652) that the replicating portfolio can be priced using a portfolio of European-style options. Specifically, this portfolio statically replicates a multiple of a log contract (a European-style payoff equal to the natural algorithm of the underlying at expiration).  

The trained agent acts to select up to 15 options from the observed environment to include in a replicating portfolio whose price is as close to the model-free variance as possible. The agent can be trained using several contemporary deep learning models: Deep Q-learning, PPO (Proximate Policy Optimization), and Asynchronous Actor-Critic (A2C). This project aims to serve as a proof of concept for the robust replication of more exotic payoffs from vanilla options using RL agents. 

The pricing theory section provides an overview of the replication-based pricing approach for variance swaps and the RL environment section describes the learning/optimization problem implemented here. 

---

## Pricing Theory

For expositional purposes, I derive the variance swap replicating portfolio in the context of a simple univariate diffusion process. Let $S_t$ be the price of the underlying asset, $B_t$ denote a Brownian motion, and $T$ be the maturity of the swap. The dynamics of $S_t$ are assumed to be:

$$
\frac{d S_t}{S_t}=\mu  d t+\sigma  d B_t
$$

By Ito's Lemma we have:
```math
d\left(\ln S_t\right)=\left(\mu-\frac{\sigma^2}{2}\right)  d t+\sigma  d B_t
```

Then the difference of the two stochastic processes gives a drift term purely in terms of the diffusion coefficent $\sigma$:
```math
d\left(\ln S_t\right) = \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma  d B_t
```

Integrating we obtain the realized variance over the life of the swap:
```math
\frac{1}{T} \int_0^T \sigma^2 d t=\frac{2}{T}\left(\int_0^T \frac{d S_t}{S_t}-\ln \left(\frac{S_T}{S_0}\right)\right)
```

We see from the right hand side that a variance swap can be perfectly hedged using a short position in two log contracts and a dynamic trading strategy holding $1/S_t$ units of the underlying. Since the dynamic trading strategy has zero cost, the variance swap is priced by the short position in two log contracts. We obtain the fair price of variance as follows:
```math
\mathbb{E}^\mathbb{Q}\left(\frac{1}{T} \int_0^T \sigma^2\right) = -2\mathbb{E}^\mathbb{Q}\left(\log(S_T)\right)=\frac{1}{T} \int_0^T \sigma^2 \, d t=\frac{2}{T}\left(\int_0^T \frac{d S_t}{S_t}-\ln \left(\frac{S_T}{S_0}\right)\right)
```

Finally, we can discretizing the integral to obtain a tradable portfolio. This is computation is nearly idnentical to the Chicago Board of Exchange's computation of the VIX Index.  Using the [Carr & Madan (1998)](https://doi.org/10.1111/j.1540-6261.1998.tb03270.x) formula we can obtain the replication for the log contract position. Applying the Carr-Madan formula and discretizing, we obtain the tradable replicating portfolio of the log contract position:
```math
\text{Var Swap Price} = 2 e^{rf \times T} \left[ \sum\limits_{i: K_i < F_t} \dfrac{P(K)}{K_i^2} \dfrac{K_{i+1} - K_{i-1}}{2} + \sum\limits_{i: K_i \geq F_t} \dfrac{C(K_i)}{K_i^2} \dfrac{K_{i+1} - K_{i-1}}{2} \right]
```
Notation:
- $F_t$ denotes the $T$-forward price of the underlying asset.
- $P(K)$ and $C(K)$ denote the OTM put and call option prices with strike $K$ and maturity $T$.
- $rf$ denotes the risk-free rate over the life of the swap.
- $\mathbb{Q}$ denotes the risk-nuetral measure (pricing measure) and $\mathbb{E}^\mathbb{Q}(\cdot)$ denotes the expectation taken over this measure.

---

## RL Environment

The model is trained using a RL framework where the agent’s goal is to minimize the discounted sum of its rewards. On each turn $N$, the agent chooses strike to include from the availible OTM strikes that remain unincluded and chooses a weight for that contract. At the end of each turn $N$, a reward $R_N$ is given to the agent based on the replication portfolio consisting of OTM contracts with the selected strikes, $\lbrace K_{i_j}\rbrace_{j=1}^N$, and weights, $\lbrace w_j \rbrace_{j=1}^N$. The reward that turn is equal to the pricing error of the replicating portfolio at the end of that turn.

```math
R_N = 2 e^{rf \times T} \left[ \sum\limits_{i_j: K_{i_j} < F_t} \dfrac{P(K)}{K_i^2} \dfrac{K_{i+1} - K_{i-1}}{2} + \sum\limits_{i: K_i \geq F_t} \dfrac{C(K_i)}{K_i^2} \dfrac{K_{i+1} - K_{i-1}}{2} -\mathbb{E}^\mathbb{Q}\left(-2\log(S_t)\right)\right]
```

Then after the pre-defined number of turns (15 in this use case), we may compute the aggregate reward with discount factor $\gamma \in (0,1)$ which serves as the objective function to maximize in this learning problem:
```math
\sum\limits_{j = 1}^{N_{max}} \gamma^j R_j
```
where $N_{max}$ is the number of turns to be played in total. This approach allows the agent to learn an optimal static replication for the variance swap through continuous self-play.
