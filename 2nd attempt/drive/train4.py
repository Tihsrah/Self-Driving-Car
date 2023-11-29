import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import gym
from gym import spaces
from stable_baselines3 import PPO

class TorcsEnv(gym.Env):
    def __init__(self):
        super(TorcsEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(620, 795, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(580, 795, 3), dtype=np.uint8)

        self.state = None
        self.reward = 0
        self.start_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 30:
            self.start_time = time.time()
            pyautogui.press('esc')
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(4)
        self.state, self.reward = self.get_state_reward()
        return self.state

    def step(self, action):
        self.take_action(action)
        self.state, self.reward = self.get_state_reward()
        done = False
        info = {}
        return self.state, self.reward, done, info

    def take_action(self, action):
        print(f"Taking action: {['Turn Left', 'Turn Right', 'Accelerate', 'Decelerate'][action]}")

    def get_state_reward(self):
        img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
        state, lines = self.lane_detection(img)
        reward = self.get_reward(lines)
        return state, reward

    def lane_detection(self, image):
        polygon = np.array([[20, 470], [320, 200], [480, 200], [900, 470]])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        stencil = np.zeros_like(gray)
        cv2.fillConvexPoly(stencil, polygon, 1)
        masked = cv2.bitwise_and(canny, canny, mask=stencil)
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


env = TorcsEnv()
model = PPO("MlpPolicy", env, verbose=1)

n_epochs = 1000  # Number of training epochs
model.learn(total_timesteps=n_epochs)

model.save("ppo_torcs")
