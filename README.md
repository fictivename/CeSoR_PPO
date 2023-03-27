# CeSoR-PPO

[RoML](https://github.com/fictivename/RoML-varibad) (Robust Meta Reinforcement Learning) is a meta reinforcement learning method. Instead of the average return of the agent, it optimizes a risk measure, which assigns more weight to high-risk tasks. Specifically, it optimizes the Conditional Value at Risk (CVaR), i.e., the average over the worst alpha quantile of the tasks.

As a reference for comparison with RoML, **this repository implements [CeSoR](https://arxiv.org/abs/2205.05138) (Cross-entropy Soft-Risk)**. CeSoR is a meta-algorithm for risk-averse reinforcement learning, which also optimizes a risk-measure of the return. It runs on top of any reinforcement learning baseline; in this repository, it is implemented on top of PPO.

The code relies on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), and in particular the implementation of PPO. Meta reinforcement learning environments are added under `environments/`, and CeSoR is implemented in `stable_baselines3/common/on_policy_algorithm.py` and `stable_baselines3/ppo/ppo.py`.
