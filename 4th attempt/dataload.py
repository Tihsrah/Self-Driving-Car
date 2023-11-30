import pickle
import cv2
import matplotlib.pyplot as plt

# Replace this with your actual filename
filename = 'data\data_1701360732.pkl'

# Load the data
with open(filename, 'rb') as f:
    data = pickle.load(f)

# Display images and keypresses
for i, (image, keypress) in enumerate(data):
    print(f"Keypress for image {i}: {keypress}")

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Image {i} with Keypress {keypress}")
    plt.show()

    # Uncomment this to use cv2 for displaying the image
    # cv2.imshow(f"Image {i}", image)
    # cv2.waitKey(0)

    # Add a break or limit the number of images to display
    if i == 500:  # Change this number to display more or fewer images
        break

# Make sure to close all windows if using cv2
# cv2.destroyAllWindows()
