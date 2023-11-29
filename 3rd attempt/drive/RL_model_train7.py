import os
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from gym import Env
from gym.spaces import Box, Discrete
import cv2
from PIL import ImageGrab
import pyautogui

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

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
            print(f"Model saved at {model_path}")
        return True

class TorcsEnv(Env):
    def __init__(self):
        super(TorcsEnv, self).__init__()
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=255, shape=(580, 795, 1), dtype=np.uint8)
        self.start_time = time.time()

    def reset(self):
        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(4)
        self.start_time = time.time()
        self.state, self.reward = self.get_state_reward()
        return self.state

    def step(self, action):
        self.take_action(action)
        self.state, self.reward = self.get_state_reward()
        print(f"Real-time Reward: {self.reward}")
        elapsed_time = time.time() - self.start_time
        done = False
        if elapsed_time >= 30:
            done = True
            self.reset()
        return self.state, self.reward, done, {}

    def take_action(self, action):
        action_map = ['up', 'down', 'left', 'right']
        pyautogui.keyDown(action_map[action])
        time.sleep(1)
        pyautogui.keyUp(action_map[action])

    def get_state_reward(self):
        img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
        left_fit, right_fit = self.lane_detection(img)
        state = np.zeros((580, 795, 1))  # Placeholder, you may update as required
        reward = self.get_reward(left_fit, right_fit)
        print(f"Real-time Reward: {reward}")
        return np.expand_dims(state, axis=-1), reward

    def lane_detection(self, image):
        polygon = np.array([[20, 470], [320, 200], [480, 200], [900, 470]])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        stencil = np.zeros_like(gray)
        cv2.fillConvexPoly(stencil, polygon, 1)
        masked = cv2.bitwise_and(canny, canny, mask=stencil.astype(np.uint8))
        lines = cv2.HoughLinesP(masked, 1, np.pi/180, 30, minLineLength=50, maxLineGap=200)
        
        left_lane_inds = []
        right_lane_inds = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                    if slope < 0:
                        left_lane_inds.append([x1, y1])
                        left_lane_inds.append([x2, y2])
                    else:
                        right_lane_inds.append([x1, y1])
                        right_lane_inds.append([x2, y2])

        left_fit = np.polyfit(np.array(left_lane_inds)[:, 1], np.array(left_lane_inds)[:, 0], 2) if left_lane_inds else None
        right_fit = np.polyfit(np.array(right_lane_inds)[:, 1], np.array(right_lane_inds)[:, 0], 2) if right_lane_inds else None

        return left_fit, right_fit

    def get_reward(self, left_fit, right_fit):
        if left_fit is None and right_fit is None:
            return -100
        reward = 0
        if left_fit is not None:
            reward += 50
        if right_fit is not None:
            reward += 50
        return reward


CHECKPOINT_DIR = './train'
TENSORBOARD_LOG_DIR = './tensorboard/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

latest_model = find_latest_model(CHECKPOINT_DIR)
if latest_model:
    model = PPO.load(latest_model, env=TorcsEnv(), tensorboard_log=TENSORBOARD_LOG_DIR)
    print(f"Resuming training from {latest_model}")
else:
    model = PPO("CnnPolicy", TorcsEnv(), verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR)
    print("Training from scratch.")

callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR)

try:
    model.learn(total_timesteps=1000, callback=callback, tb_log_name="PPO_run")
except KeyboardInterrupt:
    print("Training was interrupted manually.")

print("Training and saving completed.")
