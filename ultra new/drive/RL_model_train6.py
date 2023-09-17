# import cv2
# import numpy as np
# from PIL import ImageGrab
# import pyautogui
# import time
# import gym
# from gym import spaces
# from stable_baselines3 import PPO

# class TorcsEnv(gym.Env):
#     def __init__(self):
#         super(TorcsEnv, self).__init__()
#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Box(low=0, high=255, shape=(580, 795, 1), dtype=np.uint8)
#         self.start_time = time.time()

#     def reset(self):
#         self.start_time = time.time()
#         self.state, self.reward = self.get_state_reward()
#         return self.state

#     def step(self, action):
#         self.take_action(action)
#         self.state, self.reward = self.get_state_reward()
#         elapsed_time = time.time() - self.start_time

#         done = False
#         if elapsed_time >= 30:
#             self.reset()
#             done = True

#         return self.state, self.reward, done, {}

#     def take_action(self, action):
#         action_map = ['up', 'down', 'left', 'right']
#         pyautogui.press(action_map[action])

#     def get_state_reward(self):
#         img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
#         lane_img, lines = self.lane_detection(img)
#         print(f"Reward: {self.get_reward(lines)}")  # Printing the reward

#         # Display the result
#         cv2.imshow('Lane Detection', lane_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
            
#         state = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
#         return np.expand_dims(state, axis=-1), self.get_reward(lines)

#     def lane_detection(self, image):
#         polygon = np.array([[20, 470], [320, 200], [480, 200], [900, 470]])
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         canny = cv2.Canny(blur, 50, 150)
#         stencil = np.zeros_like(gray)
#         cv2.fillConvexPoly(stencil, polygon, 1)
#         masked = cv2.bitwise_and(canny, canny, mask=stencil.astype(np.uint8))
#         ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
#         lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
#         line_image = np.zeros_like(image)
#         try:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
#         except TypeError:
#             pass
#         combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
#         return combined_image, lines

#     def get_reward(self, lines):
#         if lines is None:
#             return -1.0
#         num_lines = len(lines)
#         return 1.0 / (num_lines + 1)

# env = TorcsEnv()
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)
# model.save("ppo_torcs")

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from gym import Env
from gym.spaces import Box, Discrete
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import os

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
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

class TorcsEnv(Env):
    def __init__(self):
        super(TorcsEnv, self).__init__()
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=255, shape=(580, 795, 1), dtype=np.uint8)
        self.start_time = time.time()

    def reset(self):
        # Perform the reset steps
        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('enter')  # Assume this selects "Abandon Race"
        time.sleep(1)
        pyautogui.press('enter')  # Assume this selects "New Race"
        time.sleep(4)  # Wait for 4 seconds before starting a new race
        
        self.start_time = time.time()
        self.state, self.reward = self.get_state_reward()
        return self.state

    def step(self, action):
        self.take_action(action)
        self.state, self.reward = self.get_state_reward()
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
        lane_img, lines = self.lane_detection(img)
        print(f"Reward: {self.get_reward(lines)}")

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

    def get_reward(self, lines):
        if lines is None:
            return -1.0
        num_lines = len(lines)
        return 1.0 / (num_lines + 1)

CHECKPOINT_DIR = './train/'

callback = TrainAndLoggingCallback(check_freq=1, save_path=CHECKPOINT_DIR)

env = TorcsEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10, callback=callback)
model.save("ppo_torcs")

