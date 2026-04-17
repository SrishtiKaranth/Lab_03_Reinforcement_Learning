# visualize_mario_ppo.py
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["DISPLAY"] = ":0"

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image
import gymnasium as gym
from gymnasium.spaces import Box
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Wrappers ───────────────────────────────────────────────────────────────────

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
        self.observation_space = Box(low=0.0, high=1.0, shape=(84, 84), dtype=np.float32)

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
        self.observation_space = Box(low=0.0, high=1.0, shape=(num_stack, 84, 84), dtype=np.float32)
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

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()


# ── Actor-Critic Network (matches training code exactly) ──────────────────────

class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()

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
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head  = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits   = self.policy_head(features)
        value    = self.value_head(features).squeeze(-1)
        return logits, value


# ── Environment ────────────────────────────────────────────────────────────────

def make_env():
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        apply_api_compatibility=True
    )
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleAndResize(env)
    env = CustomFrameStack(env, num_stack=4)
    return env


# ── Paths ──────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("/scratch/project_2016769/super_mario_duy/checkpoints/2026-04-12T22-37-51")
OUTPUT_DIR     = Path("/scratch/project_2016769/super_mario_duy/videos_ppo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Discover all PPO checkpoints in the directory and pick up to 3 evenly spaced
# ones so you get an early / mid / late training comparison automatically.
all_chkpts = sorted(CHECKPOINT_DIR.glob("mario_ppo_ep*.chkpt"))
if not all_chkpts:
    raise FileNotFoundError(f"No PPO checkpoints found in {CHECKPOINT_DIR}")

# Pick (up to) 3 evenly-spaced checkpoints
n = len(all_chkpts)
if n <= 3:
    selected = all_chkpts
else:
    indices  = [0, n // 2, n - 1]
    selected = [all_chkpts[i] for i in indices]

CHECKPOINTS = {p.stem: p for p in selected}
print("Checkpoints selected for visualisation:")
for name, path in CHECKPOINTS.items():
    print(f"  {name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_net(chkpt_path):
    chkpt = torch.load(chkpt_path, map_location=device)
    net   = ActorCritic(input_channels=4, num_actions=2).float().to(device)
    net.load_state_dict(chkpt["model"])
    net.eval()
    print(
        f"  Loaded: {chkpt_path.name}"
        f"  (episode={chkpt.get('episode', '?')}"
        f"  step={chkpt.get('step', '?')})"
    )
    return net


def get_base_env(env):
    """Walk wrapper stack all the way down to the raw nes_py env."""
    e = env
    while hasattr(e, 'env'):
        e = e.env
    return e


def run_episode(net, checkpoint_name, greedy=True):
    """
    Roll out one episode with the given PPO net.

    greedy=True  → argmax over logits (deterministic, best for visualisation)
    greedy=False → sample from the policy distribution
    """
    print(f"  Running episode (greedy={greedy})...")
    env = make_env()
    state, _ = env.reset()

    base_env = get_base_env(env)
    print(f"  Base env type: {type(base_env).__name__}")

    frames       = []
    total_reward = 0.0
    steps        = 0

    def capture_frame():
        # Method 1: nes_py exposes .screen directly
        try:
            screen = base_env.screen
            if screen is not None and isinstance(screen, np.ndarray) and screen.ndim == 3:
                return screen.copy()
        except AttributeError:
            pass
        # Method 2: old-style render('rgb_array') on base env
        try:
            frame = base_env.render('rgb_array')
            if frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
                return frame
        except Exception:
            pass
        # Method 3: render on JoypadSpace wrapper
        try:
            frame = env.env.env.env.render('rgb_array')
            if frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
                return frame
        except Exception:
            pass
        return None

    # Capture initial frame after reset
    f = capture_frame()
    if f is not None:
        frames.append(f)

    while True:
        with torch.no_grad():
            state_t = torch.tensor(
                np.array(state), dtype=torch.float32, device=device
            ).unsqueeze(0)

            logits, _ = net(state_t)               # PPO: get raw logits

            if greedy:
                action = torch.argmax(logits, dim=1).item()
            else:
                action = Categorical(logits=logits).sample().item()

        state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps        += 1

        f = capture_frame()
        if f is not None:
            frames.append(f)

        if done or info.get("flag_get", False):
            if info.get("flag_get", False):
                print("  🎉 Flag reached!")
            break

    env.close()

    # Sanity check — are frames actually different?
    if len(frames) > 10:
        diff = np.abs(
            frames[0].astype(float) - frames[10].astype(float)
        ).mean()
        print(f"  Frame diff check (frame 0 vs 10): {diff:.2f}  (0.0 = static = broken)")
    else:
        print(f"  ⚠️  Very few frames captured: {len(frames)}")

    print(f"  Steps: {steps}  |  Total reward: {total_reward:.1f}  |  Frames: {len(frames)}")
    return frames, total_reward


def save_gif(frames, path, fps=30):
    if not frames:
        print("  ⚠️  No frames to save")
        return
    processed = []
    for f in frames:
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)
        if f.ndim == 3 and f.shape[2] == 3:
            processed.append(Image.fromarray(f, mode='RGB'))
        else:
            processed.append(Image.fromarray(f))

    processed[0].save(
        path,
        save_all=True,
        append_images=processed[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"  GIF saved → {path}  ({len(processed)} frames @ {fps}fps)")


def save_comparison_plot(rewards):
    plt.figure(figsize=(7, 4))
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    labels = list(rewards.keys())
    values = list(rewards.values())
    plt.bar(labels, values, color=colors[:len(rewards)])
    plt.ylabel("Total Episode Reward")
    plt.title("PPO Mario — Reward by Checkpoint")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    out = OUTPUT_DIR / "checkpoint_comparison.jpg"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Comparison plot → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

rewards = {}

for name, chkpt_path in CHECKPOINTS.items():
    print(f"\n=== {name} ===")
    if not chkpt_path.exists():
        print(f"  ⚠️  Not found: {chkpt_path}, skipping.")
        continue
    net = load_net(chkpt_path)
    frames, reward = run_episode(net, name, greedy=True)
    rewards[name]  = reward
    save_gif(frames, OUTPUT_DIR / f"{name}_demo.gif", fps=30)

print("\n=== Reward Summary ===")
for name, r in rewards.items():
    print(f"  {name}: {r:.1f}")

if rewards:
    save_comparison_plot(rewards)

print("\nDone! Fetch your videos locally with:")
print(f"  scp -r <user>@mahti.csc.fi:{OUTPUT_DIR}/ ./mario_videos_ppo/")