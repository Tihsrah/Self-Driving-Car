from pynput.keyboard import Key, Controller
import time
import random
import pyautogui
# keyboard = Controller()
keys = ["down", "left", "right"]
time.sleep(4)

while True:
    pyautogui.keyDown("up")
    time.sleep(1)
    pyautogui.keyUp("up")

    key2 = random.choice(keys)
    pyautogui.keyDown(key2)
    time.sleep(1)
    pyautogui.keyUp(key2)