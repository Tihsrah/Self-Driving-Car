import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from PIL import ImageGrab
import cv2

class TorcsEnv(gym.Env):
    def __init__(self):
        super(TorcsEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(580, 795, 3), dtype=np.uint8)
        self.state = None
        self.reward = 0
        self.prev_x = None  # Initialize previous x-coordinate

    def reset(self):
        self.state, self.reward = get_state_reward()
        return self.state

    def step(self, action):
        take_action(action)
        self.state, self.reward = get_state_reward()
        done = False
        info = {}
        return self.state, self.reward, done, info

def take_action(action):
    print(f"Taking action: {['Turn Left', 'Turn Right', 'Accelerate', 'Decelerate'][action]}")

polygon = np.array([[20, 470], [320, 200], [480, 200], [900, 470]])

def get_reward(lines):
    if lines is None:
        return -1.0
    num_lines = len(lines)
    return 1.0 / (num_lines + 1)

def lane_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined_image

def get_state_reward():
    img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
    lane_img = lane_detection(img)
    cv2.imshow('Lane Detection', lane_img)  # Show real-time edge detection
    cv2.waitKey(1)
    stencil = np.zeros_like(img[:,:,0])
    cv2.fillConvexPoly(stencil, polygon, 1)
    masked = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)
    ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
    reward = get_reward(lines)

    # Forward movement reward logic
    current_x = 0  # Replace with the actual x-coordinate from TORCS
    if env.prev_x is not None and current_x > env.prev_x:
        reward += 0.5
    env.prev_x = current_x
    
    return lane_img, reward

env = TorcsEnv()
model = PPO("MlpPolicy", env, verbose=1)

n_epochs = 100
n_steps = 100

for epoch in range(n_epochs):
    obs = env.reset()
    rewards = []
    actions = []

    for step in range(n_steps):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)

    avg_reward = np.mean(rewards)
    print(f"Epoch {epoch + 1}/{n_epochs} - Average Reward: {avg_reward}")
