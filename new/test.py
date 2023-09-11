from PIL import ImageGrab
import numpy as np
import cv2

while True:
    img = np.array(ImageGrab.grab(bbox=(40, 40, 800, 680)))
    cv2.imshow('Test Capture', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
