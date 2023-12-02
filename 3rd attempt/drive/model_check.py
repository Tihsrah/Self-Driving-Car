import os
from stable_baselines3 import PPO
from RL_model_train8 import TorcsEnv  # Import your TorcsEnv class from its script

def run_model(model_path):
    # Load the trained model
    model = PPO.load(model_path)

    # Create the environment
    env = TorcsEnv()
    obs = env.reset()

    # Run the model in the environment
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

        # Check for the end of the episode
        if done:
            obs = env.reset()

try:
    # Provide the path to your saved model
    saved_model_path = 'train\best_model_34000.zip'
    if os.path.exists(saved_model_path):
        run_model(saved_model_path)
    else:
        print("Saved model not found.")
except KeyboardInterrupt:
    print("Execution stopped manually.")
