# RL-Text_Flappy_Bird

Individual assignment for the **3MD3220: Reinforcement Learning** course at CentraleSupélec (March 2026).

## Overview

This repository implements and compares two tabular reinforcement learning agents on the [Text Flappy Bird](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym) environment:

- **On-policy first-visit Monte Carlo control** (Sutton & Barto, Alg. 5.4)
- **Sarsa(λ) with replacing eligibility traces** (Sutton & Barto, Sec. 12.7)

The continuous `(dx, dy)` observation is discretized into a 20×20 grid, yielding a tabular Q-function of 800 entries. After 6,000 training episodes, Sarsa(λ) reaches ~492 pipes passed in greedy evaluation versus ~95 for Monte Carlo.

## Repository structure

```
RL_Assignment_TFB.ipynb   # Main notebook (agents, training, plots)
README.md
```

## Setup

```bash
# Install the environment from CentraleSupélec GitLab
pip install git+https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym.git

# Other dependencies
pip install numpy matplotlib gymnasium
```

The notebook can also be run directly in Google Colab — the installation cell handles everything automatically.

## Results

| Agent | Avg. score (greedy eval) | Reward |
|---|---|---|
| Monte Carlo | 95 ± 96 | 960 ± 955 |
| Sarsa(λ) | 492 ± 350 | 4934 ± 3497 |

Evaluation over 200 greedy episodes, `max_steps=10,000`.

Sarsa(λ) converges ~5× faster to a higher asymptotic performance, owing to online bootstrapped updates and eligibility traces that propagate credit efficiently across long episodes.

## Notebook contents

1. Installation & imports
2. Environment wrapper and state discretization
3. Monte Carlo agent
4. Sarsa(λ) agent with replacing traces
5. Training loops
6. Learning curves
7. State-value function visualization
8. Greedy policy visualization
9. Greedy evaluation
10. Parameter sweeps (ε-decay, λ, α)
11. Cross-configuration generalization
12. Discussion (original Flappy Bird gym transferability)

## Reference

Sutton, R.S., Barto, A.G. — *Reinforcement Learning: An Introduction*, 2nd edn. MIT Press (2018)
