import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
# action from python
import pyautogui
import time

# Functions to simulate keypress events for game control
def press_key(key):
    pyautogui.keyDown(key)
    time.sleep(0.1)
    pyautogui.keyUp(key)

def move_forward():
    press_key('up')

def move_backward():
    press_key('down')

def turn_left():
    press_key('left')

def turn_right():
    press_key('right')

# Function to capture frames from a specific window given its title
def capture_window(title):
    try:
        window = gw.getWindowsWithTitle(title)[0]
        if window is not None:
            while True:
                screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow(title, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    except IndexError:
        print(f"Window with title '{title}' not found.")


if __name__ == "__main__":
    title = "TORCS"  # Replace this with the exact title of your TORCS window
    
    # Run the frame capturing
    capture_window(title)
    
    # Simulate some game controls
    time.sleep(5)  # Wait for 5 seconds before starting the actions
    
    print("Moving forward")
    move_forward()
    time.sleep(5)  # Move forward for 5 seconds
    
    print("Turning left")
    turn_left()
    time.sleep(2)  # Turn left for 2 seconds