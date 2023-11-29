import gym
import numpy as np
from stable_baselines3 import PPO

# Initialize TORCS environment
env = gym.make('Torcs-v0')

# Initialize PPO model
model = PPO("CnnPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the trained model
obs = env.reset(relaunch=True)
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
    if done:
        obs = env.reset(relaunch=True)
