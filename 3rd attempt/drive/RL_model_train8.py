
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from gym import Env
from gym.spaces import Box, Discrete
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import torch
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.set_device(torch.cuda.current_device())  # Corrected this line
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
    def check_any_word_on_screen(self,bbox, phrase):
        # Capture a specific area of the screen
        screen_capture = ImageGrab.grab(bbox=bbox)

        # Use Tesseract to do OCR on the captured image
        text = pytesseract.image_to_string(screen_capture).lower()

        # Split the phrase into words and convert to lower case
        words = phrase.lower().split()

        # Initialize an empty list to store matched words
        matched_words = []

        # Check each word and add to the matched_words list if found in the text
        for word in words:
            if word in text:
                matched_words.append(word)

        # Return matched words
        return matched_words
    def step(self, action):
        # Increment step counter
        self.step_counter += 1

        # Perform the action
        self.take_action(action)

        # Update state and reward from the environment
        self.state, self.reward = self.get_state_reward()

        # Update cumulative reward
        elapsed_time = time.time() - self.start_time
        self.cumulative_reward += self.reward

        # Check if the episode is done (based on your condition, e.g., elapsed time)
        done = False
        if elapsed_time >= 60:  # Example condition for ending an episode
            done = True
            print(f"Total reward for the episode: {self.cumulative_reward}")
            self.reset()

        # OCR check for specific words
        bbox = (230, 80, 600, 150)  # Bounding box for screen capture
        phrase = "Hit Wall laptime invalidated"
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

CHECKPOINT_DIR = './train/'

# Check for the latest saved model in the directory
latest_model_path = find_latest_model(CHECKPOINT_DIR)

if latest_model_path:
    print(f"Loading latest model from {latest_model_path}...")
    model = PPO.load(latest_model_path, env=TorcsEnv())
else:
    print("No saved model found. Training from scratch.")
    model = PPO("MlpPolicy", TorcsEnv(), verbose=1)

# Your callback and environment setup
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
env = TorcsEnv()

# Continue or start new training
try:
    model.learn(total_timesteps=1000000, callback=callback)
except KeyboardInterrupt:
    print("Training was interrupted manually.")
finally:
    print("Saving the final model...")
    model.save("ppo_torcs_final")

print("Training and saving completed.")
