import cv2
import numpy as np
from PIL import ImageGrab

# Specify coordinates of the polygon for lane masking
polygon = np.array([[0, 470], [300, 200], [500, 200], [850, 470]])

def average_slope_intercept(lines):
    left_lines    = []
    left_weights  = []
    right_lines   = []
    right_weights = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # Skip vertical lines
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            
            if slope < 0:  # Left lane
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:  # Right lane
                right_lines.append((slope, intercept))
                right_weights.append(length)
    
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    
    return left_lane, right_lane  # (slope, intercept)


# Function to do lane detection on an image
def lane_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    canny = cv2.Canny(blur, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # Create an image with the same dimensions as the original
    line_image = np.zeros_like(image)

    # Draw lines on the image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    
    # Combine original image with lane lines
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined_image

# Main loop for capturing the screen
while True:
    # Capture the screen using fixed coordinates
    img = np.array(ImageGrab.grab(bbox=(5, 40, 800, 620)))

    # Create a zero array for stencil
    stencil = np.zeros_like(img[:,:,0])

    # Fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)

    # Draw dots on the vertices of the polygon
    for vertex in polygon:
        cv2.circle(img, tuple(vertex), 10, (0, 255, 0), -1)

    # Apply the polygon as a mask on the frame
    masked = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)

    # Apply image thresholding
    ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)

    # Apply Hough Line Transformation
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
    dmy = img.copy()

    if lines is not None:
        # Average and extrapolate lines
        left_lane, right_lane = average_slope_intercept(lines)
        
        y1 = img.shape[0]  # Bottom of the image
        y2 = int(y1 * 0.6)  # Slightly lower than the middle
        
        # Inside the while loop, where you calculate x1 and x2 based on the slope and intercept
        if left_lane is not None:
            slope, intercept = left_lane
            if abs(slope) > 0.1:  # Only consider slopes that are not close to zero to avoid division by zero
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                cv2.line(dmy, (x1, int(y1)), (x2, int(y2)), (255, 0, 0), 10)

        if right_lane is not None:
            slope, intercept = right_lane
            if abs(slope) > 0.1:  # Same here
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                cv2.line(dmy, (x1, int(y1)), (x2, int(y2)), (0, 255, 0), 10)


    # Combine the image with the dots and the lines
    combined = cv2.addWeighted(dmy, 0.8, img, 0.2, 0)
  
    # Display the result
    cv2.imshow('Lane Detection', combined)

    # Close the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break