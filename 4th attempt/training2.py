# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
# from sklearn.model_selection import train_test_split

# # Load the data
# data = pd.read_pickle('cleaned_data.pkl')

# # Assuming your image data is in a column named 'image_column' and labels in 'expected_output'
# X = np.array(data[0].tolist())
# y = np.array(data[1].tolist())

# # Normalize the image data
# X = X / 255.0

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Reshape data for the CNN
# X_train = X_train.reshape(-1, X.shape[1], X.shape[2], 1)
# X_test = X_test.reshape(-1, X.shape[1], X.shape[2], 1)

# # Define the CNN model
# model = Sequential()
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(50, activation='relu'))
# model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
# # Save the model
# MODEL_NAME="RLmodel"
# model.save(MODEL_NAME + '.h5')

# print("Model trained and saved as " + MODEL_NAME + ".h5")



import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Check for GPU availability
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Load the data
data = pd.read_pickle('cleaned_data.pkl')

# Assuming your image data is in a column named 'image_column' and labels in 'expected_output'
X = np.array(data[0].tolist())
y = np.array(data[1].tolist())

# Normalize the image data
X = X / 255.0

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for the CNN
X_train = X_train.reshape(-1, X.shape[1], X.shape[2], 1)
X_test = X_test.reshape(-1, X.shape[1], X.shape[2], 1)

# Define the CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Save the model
MODEL_NAME = "RLmodel"
model.save(MODEL_NAME + '.h5')

print("Model trained and saved as " + MODEL_NAME + ".h5")
