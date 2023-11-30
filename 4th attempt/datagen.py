import numpy as np
from PIL import ImageGrab
from pynput import keyboard
import cv2
import time
import os
import pickle
# Initialize variables
keys_state = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
data = []
frame_count = 0
save_interval = 500  # Save after these many frames
running = True

# Ensure the data folder exists
if not os.path.exists('data'):
    os.makedirs('data')

def on_press(key):
    global running
    if key == keyboard.Key.up:
        keys_state['up'] = 1
    elif key == keyboard.Key.down:
        keys_state['down'] = 1
    elif key == keyboard.Key.left:
        keys_state['left'] = 1
    elif key == keyboard.Key.right:
        keys_state['right'] = 1
    elif key == keyboard.Key.esc:
        running = False  # Stop running when Esc key is pressed

def on_release(key):
    if key == keyboard.Key.up:
        keys_state['up'] = 0
    elif key == keyboard.Key.down:
        keys_state['down'] = 0
    elif key == keyboard.Key.left:
        keys_state['left'] = 0
    elif key == keyboard.Key.right:
        keys_state['right'] = 0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def capture_and_save():
    global frame_count, data
    bbox_region = (5, 40, 800, 620)
    screen = ImageGrab.grab(bbox=bbox_region)

    # Convert to grayscale and resize
    screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (480, 270))

    frame_count += 1

    # Create the one-hot encoded key state list
    key_state_list = [keys_state['up'], keys_state['down'], keys_state['left'], keys_state['right']]
    data.append([screen, key_state_list])
    print(frame_count, len(data))
    if frame_count == save_interval:
        # Save using pickle
        with open(f'data/data_{int(time.time())}.pkl', 'wb') as f:
            pickle.dump(data, f)

        frame_count = 0
        data = []  # Reset the data array

try:
    while running:
        capture_and_save()
        time.sleep(0.1)  # Adjust the frequency of captures
except KeyboardInterrupt:
    pass
finally:
    listener.stop()
