# -*- coding: utf-8 -*-
"""
Train a Mario-playing RL Agent
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["DISPLAY"] = ":0"

import torch
from torch import nn
import numpy as np
from pathlib import Path
from collections import deque
import datetime
from PIL import Image

import gymnasium as gym
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


######################################################################
# Wrappers
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
    """Convert RGB to grayscale, resize to 84x84, normalize to [0,1]"""
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
        # obs is (H, W, 3) uint8
        gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        img = Image.fromarray(gray).resize((84, 84), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0  # (84, 84)


class CustomFrameStack:
    """Stack n frames into shape (n, 84, 84)"""
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
        assert obs.shape == (84, 84), f"Expected (84,84) got {obs.shape}"
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = np.array(obs, dtype=np.float32)
        assert obs.shape == (84, 84), f"Expected (84,84) got {obs.shape}"
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

# Sanity check
state, _ = env.reset()
print(f"State shape: {state.shape}, dtype: {state.dtype}")
assert state.shape == (4, 84, 84), f"Wrong state shape: {state.shape}"
next_state, reward, done, trunc, info = env.step(0)
print(f"Next state shape: {next_state.shape}")
print(f"Reward: {reward}, Done: {done}")
print("Environment OK!")


######################################################################
# Neural Network
######################################################################

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.online = self.__build_cnn(c, output_dim)
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


######################################################################
# Agent
######################################################################

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5

        # FIX: reduced buffer size from 100000 to 10000 to avoid OOM
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(10000, device=torch.device("cpu"))
        )
        self.batch_size = 32
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # FIX: reduced burnin from 10000 to 1000 to match smaller buffer
        self.burnin = 1000
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_t = torch.tensor(
                np.array(state), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action_values = self.net(state_t, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        self.memory.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "action": torch.tensor([action]),
            "reward": torch.tensor([reward]),
            "done": torch.tensor([done]),
        }, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(k) for k in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        return self.net(state, model="online")[np.arange(0, self.batch_size), action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss


######################################################################
# Logger
######################################################################

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        ep_avg_loss = (
            np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            if self.curr_ep_loss_length > 0 else 0
        )
        ep_avg_q = (
            np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
            if self.curr_ep_loss_length > 0 else 0
        )
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - Step {step} - Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))


######################################################################
# Training Loop
######################################################################

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

episodes = 10000

for e in range(episodes):

    state, _ = env.reset()

    while True:
        action = mario.act(state)
        next_state, reward, done, trunc, info = env.step(action)
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)
        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)