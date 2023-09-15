from torcs_env import TORCSEnv  # Make sure torcs_env.py is in the same directory or is accessible
from stable_baselines3 import PPO
import numpy as np

def main():
    # Initialize custom TORCS environment
    env = TORCSEnv()

    # Initialize PPO model
    model = PPO("CnnPolicy", env, verbose=1)  # Using CNN policy for image-based observation

    n_epochs = 100
    n_steps = 100

    for epoch in range(n_epochs):
        obs = env.reset()
        rewards = []
        actions = []

        for step in range(n_steps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            actions.append(action)

        avg_reward = np.mean(rewards)
        print(f"Epoch {epoch + 1}/{n_epochs} - Average Reward: {avg_reward}")

if __name__ == "__main__":
    main()
