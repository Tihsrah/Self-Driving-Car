import os
import time
import numpy as np
import pyautogui
import torch
from PIL import ImageGrab
import cv2
import pytesseract
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gym import Env
from gym.spaces import Box, Discrete
import csv
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter
# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
    def __init__(self, check_freq, save_path, csv_file="training_data.csv", verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.csv_file = csv_file
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["episode", "mean_reward", "episode_length", "duration"])
        self.writer = SummaryWriter(log_dir=save_path)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            rewards = [episode_info['r'] for episode_info in self.model.ep_info_buffer]
            if rewards:
                mean_reward = np.mean(rewards)
                self.episode_rewards.append(mean_reward)
                episode_length = len(rewards)
                self.episode_lengths.append(episode_length)
                duration = time.time() - self.start_time

                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([len(self.episode_rewards), mean_reward, episode_length, duration])

            # Log to TensorBoard
            self.writer.add_scalar('Reward/Mean', mean_reward, self.n_calls)
            self.writer.add_scalar('Episode Length', episode_length, self.n_calls)
            self.writer.add_scalar('Training/Duration', duration, self.n_calls)

            print(f"Saved model and data at step {self.n_calls}")
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
        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(4)
        
        self.cumulative_reward = 0
        self.start_time = time.time()
        self.state, self.reward = self.get_state_reward(0)
        print(f"Total steps in the last episode: {self.step_counter}")
        self.step_counter = 0
        return self.state

    def check_any_word_on_screen(self, bbox, phrase):
        screen_capture = ImageGrab.grab(bbox=bbox)
        text = pytesseract.image_to_string(screen_capture).lower()
        words = phrase.lower().split()
        matched_words = [word for word in words if word in text]
        return matched_words

    # def step(self, action):
    #     self.step_counter += 1
    #     self.take_action(action)
    #     self.state, self.reward = self.get_state_reward(action)

    #     bbox = (230, 80, 600, 150)
    #     phrase = "Hit Wall laptime invalidated"
    #     matched_words = self.check_any_word_on_screen(bbox, phrase)
    #     if matched_words:
    #         print(f"Crash detected: {', '.join(matched_words)}. Applying penalty.")
    #         self.reward -= 1000

    #     elapsed_time = time.time() - self.start_time
    #     self.cumulative_reward += self.reward

    #     done = False
    #     if elapsed_time >= 60:
    #         done = True
    #         print(f"Total reward for the episode: {self.cumulative_reward}")
    #         self.reset()

    #     return self.state, self.reward, done, {}
    def step(self, action):
        # Increment step counter
        self.step_counter += 1

        # Perform the action
        self.take_action(action)

        # Update state and reward from the environment
        self.state, self.reward = self.get_state_reward(action)

        # Update cumulative reward
        elapsed_time = time.time() - self.start_time
        self.cumulative_reward += self.reward

        # Check if the episode is done (e.g., based on elapsed time)
        done = False
        if elapsed_time >= 180:  # Example condition for ending an episode
            done = True
            print(f"Total reward for the episode: {self.cumulative_reward}")
            return self.reset(), 1000, done, {}

        # OCR check for specific words (collision detection)
        bbox = (230, 80, 600, 150)  # Bounding box for screen capture
        phrase = "Hit Wall laptime invalidated"  # Example phrase indicating a collision
        matched_words = self.check_any_word_on_screen(bbox, phrase)

        # If any word is detected, reset the environment and apply penalty
        if matched_words:
            print(f"Detected word(s): {', '.join(matched_words)}. Resetting environment and applying penalty.")
            self.cumulative_reward -= 1000  # Apply penalty to the cumulative reward
            return self.reset(), -1000, True, {}

        # Return the current state, reward, done flag, and additional info
        return self.state, self.reward, done, {}

    def take_action(self, action):
        action_map = ['up', 'left', 'right']
        print("action and mapped",action,action_map[action])
        pyautogui.keyDown(action_map[action])
        pyautogui.keyUp(action_map[action])

    def get_state_reward(self, action):
        img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
        lane_img, lines = self.lane_detection(img)
        cv2.imshow('Lane Detection', lane_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        state = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
        reward = self.get_reward(lines, action)
        print(f"Get State Reward: {reward}")
        return np.expand_dims(state, axis=-1), reward

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

    def get_reward(self, lines, action):
        if lines is None:
            return -100
        num_lines = len(lines)
        print("num lines : ",num_lines)
        base_reward = 1.0 / (num_lines + 1) * 10

        if action == 0:  # 'up' action
            reward = base_reward + 10
        elif action==1:
            reward = base_reward + 7
        else:
            reward = base_reward + 5
        
        if num_lines>20:
            print("reward from lines : ", -1*int(reward))
            return -1*int(reward)
        print("reward from lines : ", reward)
        return int(reward)

# Create a log directory for TensorBoard
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Configure the logger
logger = configure(log_dir, ["stdout", "tensorboard"])

# Training script
CHECKPOINT_DIR = './train/'
latest_model_path = find_latest_model(CHECKPOINT_DIR)
if latest_model_path:
    print(f"Loading latest model from {latest_model_path}...")
    model = PPO.load(latest_model_path, env=TorcsEnv(), tensorboard_log=log_dir)
else:
    print("No saved model found. Training from scratch.")
    model = PPO("MlpPolicy", TorcsEnv(), verbose=1, tensorboard_log=log_dir)

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
env = TorcsEnv()

try:
    model.learn(total_timesteps=1000000, callback=callback)
except KeyboardInterrupt:
    print("Training was interrupted manually.")
finally:
    print("Saving the final model...")
    model.save("ppo_torcs_final")
    callback.writer.close()
    print("TensorBoard writer closed.")

print("Training and saving completed.")