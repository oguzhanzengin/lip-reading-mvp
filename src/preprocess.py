import os
import cv2
import numpy as np


DATA_DIR = "data/raw"

LABELS = {
    "evet": 0,
    "hayir": 1,
    "tamam": 2
}

SEQUENCE_LENGTH = 20


X = []
y = []


for label_name, label_index in LABELS.items():

    label_path = os.path.join(DATA_DIR, label_name)

    if not os.path.exists(label_path):
        continue

    sequences = os.listdir(label_path)

    for seq in sequences:

        seq_path = os.path.join(label_path, seq)

        frames = []

        frame_files = sorted(
            os.listdir(seq_path),
            key=lambda x: int(x.split(".")[0])
        )

        for frame_file in frame_files:

            frame_path = os.path.join(
                seq_path,
                frame_file
            )

            img = cv2.imread(frame_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img / 255.0

            frames.append(img)

        if len(frames) == SEQUENCE_LENGTH:

            X.append(frames)
            y.append(label_index)


X = np.array(X, dtype=np.float32)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

print("[✓] Dataset kaydedildi.")