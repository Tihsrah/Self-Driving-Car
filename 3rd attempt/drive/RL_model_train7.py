import os
import time
import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import torch
from pynput import keyboard
from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Global variable to store the human input
human_action = None

def on_press(key):
    global human_action
    if key == keyboard.Key.up:    # Up arrow key
        human_action = 0          # Assuming 0 is the action index for 'up'
    elif key == keyboard.Key.left:  # Left arrow key
        human_action = 1          # Assuming 1 is the action index for 'left'
    elif key == keyboard.Key.right: # Right arrow key
        human_action = 2          # Assuming 2 is the action index for 'right'

# Set up the listener for keyboard input
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.set_device(torch.cuda.current_device())
    print(f"Using CUDA device {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
else:
    print("Using CPU")

def find_latest_model(directory):
    saved_models = [f for f in os.listdir(directory) if f.startswith("best_model_")]
    if not saved_models:
        return None
    latest_model = max(saved_models, key=lambda x: int(x.split('_')[-1].replace('.zip', '')))
    return os.path.join(directory, latest_model)

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.rate_of_learning = 0  # Initialize rate_of_learning
        self.previous_mean_reward = 0  # Initialize previous_mean_reward

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    # def _on_step(self):
    #     if self.n_calls % self.check_freq == 0:
    #         model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
    #         self.model.save(model_path)

    #         # Calculate the rate of learning
    #         current_mean_reward = np.mean(self.model.ep_info_buffer)
    #         self.rate_of_learning = current_mean_reward - self.previous_mean_reward
    #         self.previous_mean_reward = current_mean_reward
    #         print(f"Rate of learning after {self.n_calls} steps: {self.rate_of_learning}")
            
    #     return True

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            # Extract rewards from each episode info and compute the mean
            rewards = [episode_info['r'] for episode_info in self.model.ep_info_buffer]
            if rewards:  # Check if the list is not empty
                current_mean_reward = np.mean(rewards)
            else:
                current_mean_reward = 0

            self.rate_of_learning = current_mean_reward - self.previous_mean_reward
            self.previous_mean_reward = current_mean_reward
            print(f"Rate of learning after {self.n_calls} steps: {self.rate_of_learning}")
            
        return True

class TorcsEnv(Env):
    def __init__(self):
        super(TorcsEnv, self).__init__()
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=(580, 795, 1), dtype=np.uint8)
        self.start_time = time.time()
        self.cumulative_reward = 0
        self.step_counter = 0 

    def reset(self):
        # Perform the reset steps
        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('enter')  # Assume this selects "Abandon Race"
        time.sleep(1)
        pyautogui.press('enter')  # Assume this selects "New Race"
        time.sleep(4)  # Wait for 4 seconds before starting a new race
        
        self.cumulative_reward = 0
        self.start_time = time.time()
        self.state, self.reward = self.get_state_reward()
        print(f"Total steps in the last episode: {self.step_counter}")
        self.step_counter = 0
        return self.state

    def step(self, action):
        self.step_counter += 1
        self.take_action(action)
        self.state, self.reward = self.get_state_reward()
        elapsed_time = time.time() - self.start_time
        self.cumulative_reward += self.reward
        done = False
        if elapsed_time >= 30:
            done = True
            print(f"Total reward for the episode: {self.cumulative_reward}")
            self.reset()

        return self.state, self.reward, done, {}

    def take_action(self, action):
        action_map = ['up', 'left', 'right']
        print(action,action_map[action])
        pyautogui.keyDown(action_map[action])
        # time.sleep(1)
        pyautogui.keyUp(action_map[action])

    def get_state_reward(self):
        img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
        lane_img, lines = self.lane_detection(img)
        # print(f"Reward: {self.get_reward(lines)}")

        cv2.imshow('Lane Detection', lane_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        state = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(state, axis=-1), self.get_reward(lines)

    def lane_detection(self, image):
        polygon = np.array([[20, 470], [320, 200], [480, 200], [900, 470]])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        stencil = np.zeros_like(gray)
        cv2.fillConvexPoly(stencil, polygon, 1)
        masked = cv2.bitwise_and(canny, canny, mask=stencil.astype(np.uint8))
        ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
        line_image = np.zeros_like(image)
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        except TypeError:
            pass
        combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        return combined_image, lines

    # def get_reward(self, lines):
    #     if lines is None:
    #         return -1.0
    #     num_lines = len(lines)
    #     return 1.0 / (num_lines + 1)

    def get_reward(self, lines):
        if lines is None:
            return -100  # You can set this to any other negative number too
        num_lines = len(lines)
        reward = 1.0 / (num_lines + 1)
        
        # Multiply reward by 100 to make it an integer
        reward *= 100
        
        # Apply a threshold for negative reward
        threshold = 8  # You can adjust this value based on your requirements
        if reward < threshold:
            return -1*num_lines  # Or any other negative value
        
        return int(reward)

# CHECKPOINT_DIR = './train/'

# # Adjust the saving frequency to every 100 steps
# callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR)

# env = TorcsEnv()

# model = PPO("MlpPolicy", env, verbose=1)

# try:
#     model.learn(total_timesteps=1000, callback=callback)
# except KeyboardInterrupt:
#     print("Training was interrupted manually.")
# finally:
#     print("Saving the final model...")
#     model.save("ppo_torcs_final")

# print("Training and saving completed.")
# Set up the environment and model
CHECKPOINT_DIR = './train/'
latest_model_path = find_latest_model(CHECKPOINT_DIR)
env = TorcsEnv()

if latest_model_path:
    print(f"Loading latest model from {latest_model_path}...")
    model = PPO.load(latest_model_path, env=env)
else:
    print("No saved model found. Training from scratch.")
    model = PPO("MlpPolicy", env, verbose=1)

callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR)

# Training loop
try:
    for _ in range(1000):  # Replace 1000 with your desired number of timesteps
        obs = env.reset()
        done = False
        while not done:
            if human_action is not None:
                print("human action :",human_action)
                action = human_action
                human_action = None  # Reset human action
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, rewards, done, info = env.step(action)

except KeyboardInterrupt:
    print("Training was interrupted manually.")
finally:
    print("Saving the final model...")
    model.save("ppo_torcs_final")

print("Training and saving completed.")