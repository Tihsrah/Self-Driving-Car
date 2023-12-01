from PIL import ImageGrab
import pytesseract
import time
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Function to capture screen, extract text, and check for any word from the phrase
def check_any_word_on_screen(bbox, phrase):
    # Capture a specific area of the screen
    screen_capture = ImageGrab.grab(bbox=bbox)

    # Use Tesseract to do OCR on the captured image
    text = pytesseract.image_to_string(screen_capture).lower()

    # Split the phrase into words and convert to lower case
    words = phrase.lower().split()

    # Initialize an empty list to store matched words
    matched_words = []

    # Check each word and add to the matched_words list if found in the text
    for word in words:
        if word in text:
            matched_words.append(word)

    # Return matched words
    return matched_words

# Bounding box coordinates (left_x, top_y, right_x, bottom_y)
bbox = (230, 80, 600, 150)
phrase = "Hit Wall laptime invalidated"

try:
    while True:
        matched_words = check_any_word_on_screen(bbox, phrase)
        if matched_words:
            print(f"Matched word(s): {', '.join(matched_words)}")
        else:
            print("None of the words from the phrase found in screen capture.")
        time.sleep(1)  # Add a short delay
except KeyboardInterrupt:
    print("Script stopped by user.")
