# -*- coding: utf-8 -*-
"""
Train a Mario-playing RL Agent with PPO (Proximal Policy Optimization)
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["DISPLAY"] = ":0"

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from collections import deque
import datetime
from PIL import Image

import gymnasium as gym
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


######################################################################
# Wrappers  (unchanged from original)
######################################################################

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        return obs, info


class GrayScaleAndResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        img = Image.fromarray(gray).resize((84, 84), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


class CustomFrameStack:
    def __init__(self, env, num_stack=4):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(num_stack, 84, 84), dtype=np.float32
        )
        self.action_space = env.action_space

    def _get_obs(self):
        return np.stack(list(self.frames), axis=0).astype(np.float32)

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs, dtype=np.float32)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = np.array(obs, dtype=np.float32)
        self.frames.append(obs)
        return self._get_obs(), reward, done, trunc, info


######################################################################
# Build environment
######################################################################

def make_env():
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        render_mode='rgb_array',
        apply_api_compatibility=True
    )
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleAndResize(env)
    env = CustomFrameStack(env, num_stack=4)
    return env


env = make_env()

state, _ = env.reset()
print(f"State shape: {state.shape}, dtype: {state.dtype}")
assert state.shape == (4, 84, 84), f"Wrong state shape: {state.shape}"
print("Environment OK!")


######################################################################
# Actor-Critic Network
#
# Shared CNN backbone → two heads:
#   policy_head  → logits over actions  (actor)
#   value_head   → scalar state value   (critic)
######################################################################

class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()

        # Shared convolutional feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_head = nn.Linear(512, num_actions)

        # Value head (critic)
        self.value_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller gain on policy and value output layers
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None):
        """Sample an action and return (action, log_prob, entropy, value)."""
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, x):
        _, value = self(x)
        return value


######################################################################
# PPO Agent
######################################################################

class MarioPPO:
    # ---------- Hyper-parameters ----------
    ROLLOUT_STEPS   = 512       # steps collected before each update
    PPO_EPOCHS      = 4         # gradient passes over one rollout
    MINIBATCH_SIZE  = 64        # mini-batch size inside each epoch
    GAMMA           = 0.9       # discount factor
    GAE_LAMBDA      = 0.95      # GAE smoothing parameter
    CLIP_EPS        = 0.2       # PPO clip range
    VALUE_COEF      = 0.5       # value loss weight
    ENTROPY_COEF    = 0.01      # entropy bonus weight
    MAX_GRAD_NORM   = 0.5       # gradient clipping
    LR              = 2.5e-4
    SAVE_EVERY_EP   = 500       # save checkpoint every N episodes
    # ---------------------------------------

    def __init__(self, state_dim, action_dim, save_dir):
        self.action_dim = action_dim
        self.save_dir   = save_dir
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = ActorCritic(state_dim[0], action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.LR, eps=1e-5)

        self.curr_step    = 0
        self.episode_count = 0

        # Rollout buffer (pre-allocated for efficiency)
        self._init_rollout_buffer(state_dim)

    def _init_rollout_buffer(self, state_dim):
        n = self.ROLLOUT_STEPS
        self.buf_states   = np.zeros((n, *state_dim), dtype=np.float32)
        self.buf_actions  = np.zeros(n, dtype=np.int64)
        self.buf_rewards  = np.zeros(n, dtype=np.float32)
        self.buf_dones    = np.zeros(n, dtype=np.float32)
        self.buf_logprobs = np.zeros(n, dtype=np.float32)
        self.buf_values   = np.zeros(n, dtype=np.float32)
        self.buf_ptr      = 0  # write pointer

    @torch.no_grad()
    def act(self, state):
        """Select an action; returns (action, log_prob, value) as Python scalars."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, _, value = self.net.get_action_and_value(state_t)
        self.curr_step += 1
        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, reward, done, log_prob, value):
        """Write one transition into the rollout buffer."""
        idx = self.buf_ptr
        self.buf_states[idx]   = state
        self.buf_actions[idx]  = action
        self.buf_rewards[idx]  = reward
        self.buf_dones[idx]    = float(done)
        self.buf_logprobs[idx] = log_prob
        self.buf_values[idx]   = value
        self.buf_ptr += 1

    def buffer_full(self):
        return self.buf_ptr >= self.ROLLOUT_STEPS

    # ------------------------------------------------------------------
    # GAE + Returns
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_gae(self, last_state, last_done):
        """
        Compute Generalized Advantage Estimates and discounted returns
        for the current rollout buffer.
        """
        n = self.ROLLOUT_STEPS
        advantages = np.zeros(n, dtype=np.float32)
        last_state_t = torch.tensor(
            last_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        last_value = self.net.get_value(last_state_t).item()

        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.buf_values[t + 1]
            next_done  = last_done  if t == n - 1 else self.buf_dones[t + 1]

            delta = (
                self.buf_rewards[t]
                + self.GAMMA * next_value * (1.0 - next_done)
                - self.buf_values[t]
            )
            gae = delta + self.GAMMA * self.GAE_LAMBDA * (1.0 - next_done) * gae
            advantages[t] = gae

        returns = advantages + self.buf_values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------
    def learn(self, last_state, last_done):
        """
        Run PPO_EPOCHS passes over the current rollout buffer.
        Returns mean (policy_loss, value_loss, entropy).
        """
        advantages, returns = self._compute_gae(last_state, last_done)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert buffer to tensors (stay on CPU, move mini-batches to device)
        b_states   = torch.tensor(self.buf_states,   dtype=torch.float32)
        b_actions  = torch.tensor(self.buf_actions,  dtype=torch.long)
        b_logprobs = torch.tensor(self.buf_logprobs, dtype=torch.float32)
        b_values   = torch.tensor(self.buf_values,   dtype=torch.float32)
        b_advantages = torch.tensor(advantages,      dtype=torch.float32)
        b_returns  = torch.tensor(returns,           dtype=torch.float32)

        n = self.ROLLOUT_STEPS
        indices = np.arange(n)

        total_pg_loss, total_v_loss, total_ent = 0.0, 0.0, 0.0
        num_updates = 0

        for _ in range(self.PPO_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, n, self.MINIBATCH_SIZE):
                mb_idx = indices[start : start + self.MINIBATCH_SIZE]

                mb_states     = b_states[mb_idx].to(self.device)
                mb_actions    = b_actions[mb_idx].to(self.device)
                mb_old_logprobs = b_logprobs[mb_idx].to(self.device)
                mb_advantages = b_advantages[mb_idx].to(self.device)
                mb_returns    = b_returns[mb_idx].to(self.device)
                mb_old_values = b_values[mb_idx].to(self.device)

                # Forward pass
                _, new_logprob, entropy, new_value = self.net.get_action_and_value(
                    mb_states, mb_actions
                )

                # Policy (actor) loss — clipped surrogate objective
                log_ratio  = new_logprob - mb_old_logprobs
                ratio      = log_ratio.exp()
                pg_loss1   = -mb_advantages * ratio
                pg_loss2   = -mb_advantages * ratio.clamp(1 - self.CLIP_EPS, 1 + self.CLIP_EPS)
                pg_loss    = torch.max(pg_loss1, pg_loss2).mean()

                # Value (critic) loss — clipped to stabilize training
                v_clipped  = mb_old_values + (new_value - mb_old_values).clamp(
                    -self.CLIP_EPS, self.CLIP_EPS
                )
                v_loss     = torch.max(
                    (new_value - mb_returns).pow(2),
                    (v_clipped  - mb_returns).pow(2)
                ).mean()

                # Entropy bonus (encourages exploration)
                entropy_loss = entropy.mean()

                loss = (
                    pg_loss
                    + self.VALUE_COEF * v_loss
                    - self.ENTROPY_COEF * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.MAX_GRAD_NORM)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss  += v_loss.item()
                total_ent     += entropy_loss.item()
                num_updates   += 1

        # Reset buffer write pointer for next rollout
        self.buf_ptr = 0

        return (
            total_pg_loss / num_updates,
            total_v_loss  / num_updates,
            total_ent     / num_updates,
        )

    def save(self):
        path = self.save_dir / f"mario_ppo_ep{self.episode_count}.chkpt"
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.curr_step,
                "episode": self.episode_count,
            },
            path,
        )
        print(f"Checkpoint saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.curr_step     = ckpt["step"]
        self.episode_count = ckpt["episode"]
        print(f"Loaded checkpoint from {path} (step={self.curr_step})")


######################################################################
# Logger
######################################################################

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanPGLoss':>15}"
                f"{'MeanVLoss':>15}{'MeanEntropy':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot    = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot    = save_dir / "length_plot.jpg"
        self.ep_pg_losses_plot  = save_dir / "pg_loss_plot.jpg"
        self.ep_v_losses_plot   = save_dir / "v_loss_plot.jpg"
        self.ep_entropies_plot  = save_dir / "entropy_plot.jpg"

        self.ep_rewards    = []
        self.ep_lengths    = []
        self.ep_pg_losses  = []
        self.ep_v_losses   = []
        self.ep_entropies  = []

        # Moving averages
        self.moving_avg_ep_rewards   = []
        self.moving_avg_ep_lengths   = []
        self.moving_avg_ep_pg_losses = []
        self.moving_avg_ep_v_losses  = []
        self.moving_avg_ep_entropies = []

        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

    def log_update(self, pg_loss, v_loss, entropy):
        """Called once per PPO update (not every step)."""
        self.pg_losses.append(pg_loss)
        self.v_losses.append(v_loss)
        self.entropies.append(entropy)

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_pg_losses.append(np.mean(self.pg_losses)  if self.pg_losses  else 0.0)
        self.ep_v_losses.append( np.mean(self.v_losses)   if self.v_losses   else 0.0)
        self.ep_entropies.append(np.mean(self.entropies)  if self.entropies  else 0.0)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.pg_losses  = []
        self.v_losses   = []
        self.entropies  = []

    def record(self, episode, step):
        mean_reward   = np.round(np.mean(self.ep_rewards[-100:]),   3)
        mean_length   = np.round(np.mean(self.ep_lengths[-100:]),   3)
        mean_pg_loss  = np.round(np.mean(self.ep_pg_losses[-100:]), 5)
        mean_v_loss   = np.round(np.mean(self.ep_v_losses[-100:]),  5)
        mean_entropy  = np.round(np.mean(self.ep_entropies[-100:]), 5)

        self.moving_avg_ep_rewards.append(mean_reward)
        self.moving_avg_ep_lengths.append(mean_length)
        self.moving_avg_ep_pg_losses.append(mean_pg_loss)
        self.moving_avg_ep_v_losses.append(mean_v_loss)
        self.moving_avg_ep_entropies.append(mean_entropy)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - Step {step} - "
            f"Mean Reward {mean_reward} - Mean Length {mean_length} - "
            f"PG Loss {mean_pg_loss} - V Loss {mean_v_loss} - "
            f"Entropy {mean_entropy} - Time Delta {time_delta} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{mean_reward:15.3f}"
                f"{mean_length:15.3f}{mean_pg_loss:15.5f}"
                f"{mean_v_loss:15.5f}{mean_entropy:15.5f}"
                f"{time_delta:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric, attr in [
            ("ep_rewards",   "moving_avg_ep_rewards"),
            ("ep_lengths",   "moving_avg_ep_lengths"),
            ("ep_pg_losses", "moving_avg_ep_pg_losses"),
            ("ep_v_losses",  "moving_avg_ep_v_losses"),
            ("ep_entropies", "moving_avg_ep_entropies"),
        ]:
            plt.clf()
            plt.plot(getattr(self, attr), label=attr)
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))


######################################################################
# Training Loop
######################################################################

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario  = MarioPPO(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

episodes = 10000

# PPO is on-policy, so we interleave rollout collection with updates.
# The outer loop is still over episodes, but updates fire whenever the
# rollout buffer fills — which may happen mid-episode.

state, _ = env.reset()

for e in range(episodes):

    state, _ = env.reset()

    while True:
        # ---- Collect one step ----
        action, log_prob, value = mario.act(state)
        next_state, reward, done, trunc, info = env.step(action)

        mario.store(state, action, reward, done, log_prob, value)
        logger.log_step(reward)

        state = next_state

        # ---- Update when rollout buffer is full ----
        if mario.buffer_full():
            pg_loss, v_loss, entropy = mario.learn(state, done)
            logger.log_update(pg_loss, v_loss, entropy)

        if done or info.get("flag_get", False):
            break

    mario.episode_count += 1
    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, step=mario.curr_step)

    if (e % mario.SAVE_EVERY_EP == 0) or (e == episodes - 1):
        mario.save()