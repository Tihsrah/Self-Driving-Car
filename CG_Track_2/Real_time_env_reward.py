import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time


# Specify coordinates of the polygon for lane masking
polygon = np.array([[20, 470], [320, 200], [480, 200], [900, 470]])
# polygon = np.array([ [200, 70], [600, 100]])


# Reward function based on the number of lines detected
def get_reward(lines):
    if lines is None:
        return -1.0
    num_lines = len(lines)
    return 1.0 / (num_lines + 1)

def lane_detection(image, polygon):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    canny = cv2.Canny(blur, 50, 150)
    
    # Create a zero array for stencil
    stencil = np.zeros_like(gray)
    
    # Fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)
    
    # Apply the polygon as a mask on the frame
    masked = cv2.bitwise_and(canny, canny, mask=stencil)
    
    # Apply image thresholding
    ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
    
    # Create an image with the same dimensions as the original
    line_image = np.zeros_like(image)
    
    # Draw lines on the image
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    except TypeError:
        pass
    
    # Combine original image with lane lines
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
    return combined_image, lines



# Initialize the timer
start_time = time.time()

# Main loop for capturing the screen
# Main loop for capturing the screen
while True:

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Check if 30 seconds have passed
    if elapsed_time >= 30:
        # Reset the timer
        start_time = time.time()
        
        # Perform the reset steps
        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('enter')  # Assume this selects "Abandon Race"
        time.sleep(1)
        pyautogui.press('enter')  # Assume this selects "New Race"
        time.sleep(4)  # Wait for 4 seconds

    # Step 1: Capture the screen using fixed coordinates
    img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))
    
    # Step 1.5: Check for the "Hit wall, laptime invalidated" message
    # check_for_hit_wall()  # Call the function here
    
    # Step 2: Call the lane_detection function
    lane_img, lines = lane_detection(img, polygon)
    
    # Draw dots on the vertices of the polygon for visualization
    for vertex in polygon:
        cv2.circle(lane_img, tuple(vertex), 10, (0, 255, 0), -1)
    
    # Step 3: Get the reward based on the number of lines detected
    reward = get_reward(lines)
    print(f"Reward: {reward}")  # For now, we'll print the reward. This will be passed to the RL model later.
    
    # Display the result
    cv2.imshow('Lane Detection', lane_img)
    
    # Close the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break