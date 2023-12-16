import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your cleaned data from a .pkl file
data = pd.read_pickle('cleaned_data.pkl')
print(data.head())
# Split the data into features and labels
X = np.array(data[0].tolist())  # Adjust shape as needed
Y = np.array(data[1].tolist())
print(X.shape)
print(len(Y))
# Split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Model parameters
WIDTH = 480
HEIGHT =  270
LR = 1e-3
EPOCHS = 30
MODEL_NAME = 'self-driving-car-model'

# Define your model architecture here
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 9 outputs corresponding to your labels
])

model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(LR), metrics=['accuracy'])

# Training the model
model.fit(X_train, Y_train, batch_size=32, epochs=EPOCHS, validation_data=(X_test, Y_test))

# Save the model
model.save(MODEL_NAME + '.h5')

print("Model trained and saved as " + MODEL_NAME + ".h5")
