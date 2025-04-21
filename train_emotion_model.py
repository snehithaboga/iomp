import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv("fer2013.csv")

# Extract pixels and emotions
pixels = data['pixels'].tolist()
emotions = pd.get_dummies(data['emotion']).values

X = np.array([np.fromstring(pix, sep=' ') for pix in pixels])
X = X.reshape(-1, 48, 48, 1)
X = X / 255.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, emotions, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

# Save the trained model
model.save("emotion_model.h5")

print("✅ Model training complete and saved as emotion_model.h5")

