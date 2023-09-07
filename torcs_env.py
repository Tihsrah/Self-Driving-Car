import gym
import numpy as np
from gym import spaces
from PIL import ImageGrab  # <-- New import
import pyautogui
import cv2
import time
import pygetwindow as gw

# Load templates
car_template = cv2.imread('car_template.png', 0)  # Grayscale
road_template = cv2.imread('road_template_2.png', 0)

class TorcsEnv(gym.Env):
    def __init__(self):
        super(TorcsEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 actions: forward, backward, left, right
        # self.observation_space = spaces.Box(low=0, high=255, shape=(160, 120, 1), dtype=np.uint8)
        # Update observation space to original game size
        self.observation_space = spaces.Box(low=0, high=255, shape=(649, 820, 1), dtype=np.uint8)


        # Initialize game window title
        self.title = "TORCS"

    def detect_objects(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Only convert to grayscale, no resizing
        
        car_result = cv2.matchTemplate(gray_frame, car_template, cv2.TM_CCOEFF_NORMED)
        road_result = cv2.matchTemplate(gray_frame, road_template, cv2.TM_CCOEFF_NORMED)
        
        _, _, _, car_position = cv2.minMaxLoc(car_result)
        _, _, _, road_position = cv2.minMaxLoc(road_result)
        
        return car_position, road_position


    def calculate_reward(self, car_pos, road_pos):
        # Placeholder logic to calculate reward based on car and road positions
        # This should be replaced by more accurate collision detection between car and road bounding boxes
        if car_pos and road_pos:
            return 1 if car_is_within_road(car_pos, road_pos) else -1
        else:
            return 0

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

        # Capture the full-sized frame for object detection
        window = gw.getWindowsWithTitle(self.title)[0]
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
        full_frame = np.array(screenshot)

        # Detect car and road positions on the original full-sized frame
        car_pos, road_pos = self.detect_objects(full_frame)

        # Calculate reward
        reward = self.calculate_reward(car_pos, road_pos)

        # Create the observation space frame without resizing
        frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
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
# You'll need to implement this based on your specific setup.
def car_is_within_road(car_pos, road_pos):
    # Actual bounding box sizes for car and road based on templates
    # car_width, car_height = 374, 264
    # road_width, road_height = 725, 274
    car_width, car_height = car_template.shape[::-1]
    road_width, road_height = road_template.shape[::-1]

    # Calculate the bounding box corners for car and road
    car_left, car_top = car_pos
    car_right, car_bottom = car_left + car_width, car_top + car_height

    road_left, road_top = road_pos
    road_right, road_bottom = road_left + road_width, road_top + road_height

    # Check if car's bounding box is within road's bounding box
    if (car_left >= road_left and car_right <= road_right and
        car_top >= road_top and car_bottom <= road_bottom):
        return True
    else:
        return False
