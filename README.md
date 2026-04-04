# RL-Text_Flappy_Bird

Individual assignment for the **3MD3220: Reinforcement Learning** course at CentraleSupélec (March 2026).

## Overview

This repository implements and compares two tabular reinforcement learning agents on the [Text Flappy Bird](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym) environment (`TextFlappyBird-v0`), a text-based variant of the classic Flappy Bird game developed at CentraleSupélec.

Two environment versions are available in the gym:
- `TextFlappyBird-v0` — returns a compact 2D observation `(dx, dy)`: the distance of the player from the centre of the closest pipe gap. Used in this work.
- `TextFlappyBird-screen-v0` — returns the full rendered character screen, incompatible with tabular methods due to the curse of dimensionality.

For reference, the original [flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym) exposes either a raw RGB pixel observation or a continuous feature vector. Both are too high-dimensional for tabular RL; extending to this setting would require function approximation (e.g. DQN).

## Repository structure

```
RL_Assignment_TFB.ipynb   # Main notebook (agents, training, plots)
Final_Report.pdf          # Report (Springer LNCS format)
Instructions.pdf          # Assignment instructions
README.md
.gitignore
```

## Report

The full PDF report is available [here](https://github.com/comegenet/RL-Text_Flappy_Bird/blob/main/Final_Report.pdf).

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
