import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Flatten,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical


X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

NUM_CLASSES = 3

y_cat = to_categorical(y, NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential([
    TimeDistributed(
        Conv2D(16, (3, 3), activation="relu"),
        input_shape=(20, 50, 100, 3)
    ),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(
        Conv2D(32, (3, 3), activation="relu")
    ),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(Flatten()),

    LSTM(64),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=4
)

os.makedirs("models", exist_ok=True)

model.save("models/lip_reading_model.keras")

print("[✓] Model kaydedildi: models/lip_reading_model.keras")