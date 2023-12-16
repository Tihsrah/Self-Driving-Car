# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Load the model
# model = load_model('RLmodel.h5')
# # Load the data
# data = pd.read_pickle('cleaned_data.pkl')
# X = np.array(data[0].tolist())
# y = np.array(data[1].tolist())

# # Load the image you want to predict
# img = image.load_img('img.png', grayscale=True, target_size=(X.shape[1], X.shape[2]))

# # Preprocess the image for the model
# img = image.img_to_array(img)
# img = img / 255.0  # Normalize the image like you did during training
# img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch

# # Make the prediction
# probabilities = model.predict(img)
# # classes = np.argmax(probabilities, axis=-1)

# print(probabilities)

import numpy as np
import cv2
from PIL import ImageGrab
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyautogui
import time
# Load the model
model = load_model('RLmodel.h5')


def take_action(action_index):
    action_map = ['up', 'down', 'left', 'right']  # Updated action map
    action = action_map[action_index]
    print("Action:", action)
    pyautogui.keyDown(action)
    time.sleep(0.1)
    pyautogui.keyUp(action)

def real_time_prediction():
    while True:
        # Capture the screen
        bbox_region = (5, 40, 800, 620)
        screen = ImageGrab.grab(bbox=bbox_region)

        # Convert to grayscale and resize
        screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (480, 270))  # Resize to the model's expected input size

        # Preprocess the image for the model
        img = np.expand_dims(screen, axis=-1)  # Add channel dimension
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make the prediction
        probabilities = model.predict(img)

        # Determine the action with the highest probability
        action_index = np.argmax(probabilities)

        # Take the action
        # print(action_index)
        take_action(action_index)

        # Break the loop with a specific key (e.g., 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Call the real-time prediction function
real_time_prediction()
