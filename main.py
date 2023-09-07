from stable_baselines3 import PPO
from torcs_env import TorcsEnv  # Make sure to import your custom environment
from advanced_torcs_env import AdvancedTorcsEnv
# Initialize the environment and the model
# env = TorcsEnv()
env=AdvancedTorcsEnv()
model = PPO("CnnPolicy", env, verbose=1, n_steps=128)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_torcs")

# Test the trained agent
obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()
