import gym
import numpy as np
from gym import spaces
from PIL import ImageGrab
import pyautogui
import cv2
import time
import pygetwindow as gw

# Load templates
car_template = cv2.imread('car_template.png', 0)  # Grayscale
road_template = cv2.imread('road_template_2.png', 0)

class AdvancedTorcsEnv(gym.Env):
    def __init__(self):
        super(AdvancedTorcsEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(649, 820, 1), dtype=np.uint8)
        self.current_frame = None  # Add this to keep track of the current frame
        self.title = "TORCS"

    def calculate_reward(self, action, car_pos):
        road_color = 30  # Assuming the road color is 128 in grayscale
        tolerance = 10
        car_width, car_height = car_template.shape[::-1]
        car_left, car_top = car_pos
        car_right, car_bottom = car_left + car_width, car_top + car_height
        region = self.current_frame[car_top:car_bottom, car_left:car_right]
        matching_pixels = np.sum(np.abs(region - road_color) < tolerance)
        total_pixels = region.size
        ratio = matching_pixels / total_pixels

        reward = 0
        if ratio > 0.7:  # More than 70% of the pixels match the road color
            reward += 1
            if action == 0:
                reward += 1  # Reward for moving forward on the road
        else:
            reward -= 1  # Penalty for being off the road

        if action == 1:  # Penalty for moving backward
            reward -= 1

        return reward
    
    def detect_objects(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Only convert to grayscale, no resizing
        
        car_result = cv2.matchTemplate(gray_frame, car_template, cv2.TM_CCOEFF_NORMED)
        road_result = cv2.matchTemplate(gray_frame, road_template, cv2.TM_CCOEFF_NORMED)
        
        _, _, _, car_position = cv2.minMaxLoc(car_result)
        _, _, _, road_position = cv2.minMaxLoc(road_result)
        
        return car_position, road_position


    def reset(self):
        # Capture the initial frame without resizing
        window = gw.getWindowsWithTitle(self.title)[0]
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=-1)  # Keep single channel
        return frame

    def step(self, action):
        if action == 0:
            self.move_forward()
        elif action == 1:
            self.move_backward()
        elif action == 2:
            self.turn_left()
        elif action == 3:
            self.turn_right()

        # Update current_frame attribute after capturing
        window = gw.getWindowsWithTitle(self.title)[0]
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
        full_frame = np.array(screenshot)
        self.current_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)  # Add this line

        # Detect car and road positions on the original full-sized frame
        car_pos, road_pos = self.detect_objects(full_frame)

        # Calculate reward based on the new function
        reward = self.calculate_reward(action, car_pos)

        # Create the observation space frame without resizing
        frame = self.current_frame  # Use the updated current_frame
        frame = np.expand_dims(frame, axis=-1)  # Keep single channel

        done = False
        return frame, reward, done, {}

    def press_key(self, key):
        pyautogui.keyDown(key)
        time.sleep(0.1)
        pyautogui.keyUp(key)

    def move_forward(self):
        self.press_key('up')

    def move_backward(self):
        self.press_key('down')

    def turn_left(self):
        self.press_key('left')

    def turn_right(self):
        self.press_key('right')

# Placeholder function to check if the car is within the road boundaries.
