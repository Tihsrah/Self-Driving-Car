import os
import time
import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
from tensorflow.keras.models import load_model

import tensorflow as tf
print(tf.__version__)

# Load the trained model
model_path = 'Model.h5'  # Replace with your model's filename
model = load_model(model_path)

# def preprocess_image(image):
#     # Resize the image to match the input format of the model
#     image = cv2.resize(image, (270, 480))  # Resize to 270x480

#     # Convert to grayscale if the model expects 1 channel
#     if image.shape[-1] != 1:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Normalize pixel values
#     image = image / 255.0  # Normalize pixel values

#     # Reshape to add a channel dimension if necessary
#     if len(image.shape) == 2:  # Only for grayscale images
#         image = np.expand_dims(image, axis=-1)

#     return image

def preprocess_image(image):
    # Resize the image to match the input format of the model
    image = cv2.resize(image, (480, 270))  # Resize to 480x270

    # If the image is grayscale (1 channel), convert it to RGB (3 channels)
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Normalize pixel values
    image = image / 255.0  # Normalize pixel values

    return image

def main():
    while True:
        # Capture the screen
        # screen = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
        # screen = preprocess_image(screen)

        # # Predict the output
        # prediction = model.predict(screen.reshape(1, 480, 270, 3))  # Adjust shape if needed
        # print("Prediction:", prediction)

        screen = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
        screen = preprocess_image(screen)

        # Predict the output
        prediction = model.predict(screen.reshape(1, 480, 270, 3))  # Ensure this shape matches your model's input shape


        # Here you can map the prediction to corresponding keyboard actions
        # Example:
        # if np.argmax(prediction) == 0:
        #     pyautogui.press('up')
        # elif np.argmax(prediction) == 1:
        #     pyautogui.press('down')
        # ... and so on for other predictions

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
