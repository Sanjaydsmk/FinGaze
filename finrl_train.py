import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import os

from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3


def _to_scalar(action) -> float:
    return float(np.squeeze(action))

# ==================================
# DOWNLOAD DATA
# ==================================
print("Downloading data...")

df = yf.download("AAPL", period="2y", progress=False)

if df.empty:
    print("❌ Data download failed.")
    exit()

df = df[['Close']]
df.dropna(inplace=True)

# Use returns instead of price
df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

returns = df['Return'].values

# ==================================
# CUSTOM ENVIRONMENT
# ==================================
class TradingEnv(gym.Env):

    def __init__(self, returns):
        super().__init__()

        self.returns = returns
        self.current_step = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return np.array([self.returns[self.current_step]], dtype=np.float32), {}

    def step(self, action):
        # No look-ahead: action decided at t is rewarded on return at t+1.
        action_value = _to_scalar(action)
        next_step = self.current_step + 1
        reward = action_value * self.returns[next_step]

        self.current_step = next_step
        done = self.current_step >= len(self.returns) - 1

        obs = np.array([self.returns[self.current_step]], dtype=np.float32)

        return obs, reward, done, False, {}

# ==================================
# TRAIN MODELS
# ==================================
env = TradingEnv(returns)

models = {
    "ppo_model": PPO("MlpPolicy", env, verbose=1),
    "a2c_model": A2C("MlpPolicy", env, verbose=1),
    "ddpg_model": DDPG("MlpPolicy", env, verbose=1),
    "sac_model": SAC("MlpPolicy", env, verbose=1),
    "td3_model": TD3("MlpPolicy", env, verbose=1),
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...")
    model.learn(total_timesteps=20000)
    model.save(f"models/{name}")

print("✅ ALL MODELS TRAINED SUCCESSFULLY")
