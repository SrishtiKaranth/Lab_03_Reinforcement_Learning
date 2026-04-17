# Lab 03 – Reinforcement Learning: Super Mario Bros

This project explores training reinforcement learning agents to play Super Mario Bros using raw pixel input and reward-based learning. The agent learns entirely through interaction with the environment, without any prior knowledge, rules, or human demonstrations.

## Overview

We implement and compare three reinforcement learning algorithms:

* DDQN (Double Deep Q-Network) – Value-based method improving stability over DQN
* PPO (Proximal Policy Optimization) – Policy-gradient method with stable updates
* A3C (Asynchronous Advantage Actor-Critic) – Parallelized actor-critic approach

The objective is to evaluate how different reinforcement learning approaches perform in a complex, high-dimensional environment.

## Branches

| Branch | Description                                                            |
| ------ | ---------------------------------------------------------------------- |
| `ddqn` | DDQN implementation, training checkpoints, logs, and evaluation videos |
| `ppo`  | PPO implementation                                                     |
| `a3c`  | A3C implementation                                                     |

## Problem Statement

Can an AI agent learn to play Super Mario Bros purely from pixels and rewards, without any handcrafted features or prior knowledge?

Key challenges include:

* Delayed rewards (credit assignment problem)
* High-dimensional state space (raw image input)
* Exploration vs exploitation trade-off

## Methods

* DDQN: Uses separate target and online networks to reduce overestimation and improve stability
* PPO: Uses a clipped objective function to ensure stable and controlled policy updates
* A3C: Trains multiple agents in parallel to update a shared global network

## Results Summary

* PPO achieved the best overall performance and stability
* DDQN served as a strong baseline but struggled with local optima
* A3C demonstrated fast learning but exhibited unstable reward patterns

## Key Takeaways

* Policy-based methods such as PPO perform better in complex environments
* Agents can learn meaningful behaviors directly from reward signals
* Model architecture and training strategy significantly impact performance

## Future Work

* Hyperparameter tuning for PPO (learning rate, clipping range)
* Improving training stability and sample efficiency
* Extending experiments to more complex environments

