import gym
import gym_torcs
import os
import subprocess

# Change the working directory to the directory containing 'wtorcs.exe'
os.chdir(r"C:\Program Files (x86)\torcs")

# Start the TORCS process (assuming no additional arguments are needed)
subprocess.Popen("wtorcs.exe", shell=False)

# Create the Gym environment
env = gym.make('Torcs-v0')

# Initialize the environment
observation = env.reset(relaunch=True)  # Relaunch TORCS at the beginning

# Run the agent-environment loop
for _ in range(1000):
    action = env.action_space.sample()  # Your agent's action
    observation, reward, done, info = env.step(action)
    if done:
        break

# Close the environment
env.close()
