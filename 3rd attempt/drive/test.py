import os
from stable_baselines3 import PPO
from RL_model_train6 import TorcsEnv
# Function to find the latest model
def find_latest_model(directory):
    saved_models = [f for f in os.listdir(directory) if f.startswith("best_model_")]
    if not saved_models:
        return None
    latest_model = max(saved_models, key=lambda x: int(x.split('_')[-1].replace('.zip', '')))
    return os.path.join(directory, latest_model)

# Directory where your models are saved
CHECKPOINT_DIR = './train/'

# Load the latest model
latest_model_path = find_latest_model(CHECKPOINT_DIR)
if latest_model_path:
    print(f"Loading model from {latest_model_path}")
    model = PPO.load(latest_model_path, env=TorcsEnv())
else:
    print("No saved model found. Please check your model directory.")
    exit()

# Initialize the environment
env = TorcsEnv()

# Run the model in the environment
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    # Add any visualization or logging code here if needed

# Make sure to close any resources used by your environment
env.close()
