import os
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Flatten,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical


DATA_DIR = "data/external_processed"
MODEL_DIR = "models"

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

with open(os.path.join(DATA_DIR, "labels.json"), "r", encoding="utf-8") as f:
    labels_map = json.load(f)

NUM_CLASSES = len(labels_map)

y_cat = to_categorical(y, NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential([
    Input(shape=(20, 50, 100, 3)),

    TimeDistributed(Conv2D(16, (3, 3), activation="relu")),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(Conv2D(32, (3, 3), activation="relu")),
    TimeDistributed(MaxPooling2D((2, 2))),

    TimeDistributed(Flatten()),

    LSTM(128),

    Dense(128, activation="relu"),
    Dropout(0.4),

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
    epochs=20,
    batch_size=8
)

os.makedirs(MODEL_DIR, exist_ok=True)

model.save(os.path.join(MODEL_DIR, "turkish_lip_reading_external.keras"))

print("[✓] Model kaydedildi: models/turkish_lip_reading_external.keras")