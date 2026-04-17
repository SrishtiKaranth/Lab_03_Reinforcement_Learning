import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import numpy as np
from collections import deque
from PIL import Image

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace


# =========================
# ENV WRAPPERS
# =========================

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        done = False
        info = {}

        for _ in range(self.skip):
            out = self.env.step(action)

            # gym (4 or 5 return formats)
            if len(out) == 4:
                obs, reward, done, info = out
                trunc = False
            else:
                obs, reward, done, trunc, info = out
                done = done or trunc

            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
        img = Image.fromarray(gray.astype(np.uint8)).resize((84, 84))
        return np.array(img, dtype=np.float32) / 255.0


class FrameStack:
    def __init__(self, env, k=4):
        self.env = env
        self.k = k
        self.frames = deque(maxlen=k)
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(0, 1, (k, 84, 84), dtype=np.float32)

    def reset(self):
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out

        for _ in range(self.k):
            self.frames.append(obs)

        return np.stack(self.frames)

    def step(self, action):
        out = self.env.step(action)

        if len(out) == 4:
            obs, reward, done, info = out
        else:
            obs, reward, done, trunc, info = out
            done = done or trunc

        self.frames.append(obs)
        return np.stack(self.frames), reward, done, info


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, 4)
    env = GrayScaleResize(env)
    env = FrameStack(env, 4)
    return env


# =========================
# MODEL
# =========================

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape

        self.shared = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# =========================
# WORKER
# =========================

def worker(global_net, optimizer, worker_id, max_episodes):
    env = make_env()
    local_net = ActorCritic((4, 84, 84), env.action_space.n)
    local_net.load_state_dict(global_net.state_dict())

    gamma = 0.99

    for episode in range(max_episodes):

        state = env.reset()

        log_probs = []
        values = []
        rewards = []

        done = False

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            logits, value = local_net(state_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            action = dist.sample()

            next_state, reward, done, info = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            state = next_state

        # returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values).squeeze()

        advantage = returns - values

        log_probs = torch.stack(log_probs)

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()

        # async update
        for gp, lp in zip(global_net.parameters(), local_net.parameters()):
            gp._grad = lp.grad

        optimizer.step()

        local_net.load_state_dict(global_net.state_dict())

        if worker_id == 0 and episode % 10 == 0:
            print(f"[Worker {worker_id}] Episode {episode} Reward {sum(rewards)}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)

    env = make_env()
    global_net = ActorCritic((4, 84, 84), env.action_space.n)
    global_net.share_memory()

    optimizer = optim.Adam(global_net.parameters(), lr=1e-4)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    max_episodes = 1000

    processes = []

    for i in range(num_workers):
        p = mp.Process(target=worker, args=(global_net, optimizer, i, max_episodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
